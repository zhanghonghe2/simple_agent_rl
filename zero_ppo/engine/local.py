#!/usr/bin/env python3
"""The in-process engine: one vLLM instance on the cards this process owns.

Weights arrive by rebuilding from a snapshot, because this vLLM's transfer channel
wants an HTTP client or a Ray handle. Rebuilding costs tens of seconds a step and
needs no distributed state.

    python3 -m zero_ppo.engine.local --self-test     # no vLLM needed
"""
from __future__ import annotations

import argparse
import gc
import os
import time

STOP = ("<|im_end|>", "<|endoftext|>")


class LocalEngine:
    backend_name = "local"

    def __init__(self, model_path: str, *, dtype: str = "bfloat16",
                 tensor_parallel: int = 1, gpu_util: float = 0.3,
                 max_model_len: int = 4096, prefix_cache: bool = True,
                 tmp_root: str = "runs/zero_ppo"):
        from vllm import LLM
        self.args = dict(dtype=dtype, tensor_parallel=tensor_parallel,
                         gpu_util=gpu_util, max_model_len=max_model_len,
                         prefix_cache=prefix_cache)
        self.tmp_root = tmp_root
        self.llm = self._build(model_path)
        self.tokenizer = None
        self._awake = True

    # ---------------------------------------------------------------- internals
    def _build(self, path: str):
        from vllm import LLM
        from vllm.distributed.weight_transfer.base import WeightTransferConfig
        return LLM(model=path, dtype=self.args["dtype"],
                   # CUDA IPC, not NCCL: NCCL needs each rank on its own device, and
                   # here the trainer shares the cards with the engine.
                   weight_transfer_config=WeightTransferConfig(backend="ipc"),
                   tensor_parallel_size=self.args["tensor_parallel"],
                   gpu_memory_utilization=self.args["gpu_util"],
                   max_model_len=self.args["max_model_len"],
                   enable_prefix_caching=bool(self.args["prefix_cache"]),
                   enable_sleep_mode=True,
                   disable_custom_all_reduce=True)

    # -------------------------------------------------------------------- verbs
    def generate(self, prompts, temperature=0.7, max_tokens=1024):
        from vllm import SamplingParams
        sps = [SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              stop=list(STOP)) for _ in prompts]
        outs = self.llm.generate(list(prompts), sps)
        return [{"text": o.outputs[0].text,
                 "tokens": list(o.outputs[0].token_ids)} for o in outs]

    # Forwarded to the vLLM object, which is where the transfer channel lives.
    def init_weight_transfer_engine(self, request):
        return self.llm.init_weight_transfer_engine(request)

    def start_weight_update(self):
        return self.llm.start_weight_update()

    def update_weights(self, request):
        return self.llm.update_weights(request)

    def finish_weight_update(self):
        return self.llm.finish_weight_update()

    def reload(self, model, tok, step: int):
        """Save the updated weights and swap the engine for a new one."""
        snap = os.path.join(self.tmp_root, f"snap_step{step}")
        model.save_pretrained(snap)
        tok.save_pretrained(snap)      # without it the rebuilt engine renders nothing
        old, self.llm = self.llm, None
        # The engine runs in its own process; dropping the handle does not stop it,
        # and its KV blocks stay allocated until it exits.
        shut = getattr(old, "shutdown", None)
        if shut is not None:
            try:
                shut()
            except Exception as e:
                print(f"  ! shutdown failed: {type(e).__name__}: {e}", flush=True)
        del old
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        time.sleep(3)      # give the child processes a moment to release the cards
        self.llm = self._build(snap)
        self._awake = True
        return snap

    def sleep(self, level: int = 1):
        if self._awake and hasattr(self.llm, "sleep"):
            self.llm.sleep(level=level)
            self._awake = False

    def wake(self, tags=None):
        """Wake the engine, optionally only some memory tags.

        For a weight sync the official colocated pattern is: sleep -> wake only
        ``weights`` -> transfer -> wake ``kv_cache`` (+ ``scheduling``). Waking the
        KV cache before the IPC write touches cumem-managed memory that is still
        freed, which segfaults in cuMemcpyDtoDAsync. A full wake (tags=None) is for
        inference, not for the transfer.
        """
        if hasattr(self.llm, "wake_up"):
            if tags is not None:
                self.llm.wake_up(tags=list(tags))
            else:
                self.llm.wake_up()
                self._awake = True
        elif hasattr(self.llm, "wake"):      # older vLLM: no tag support
            self.llm.wake()
            self._awake = True

    def pause(self):
        """Stop scheduling around a weight sync so no request is mid-flight."""
        fn = getattr(self.llm, "pause_generation", None)
        if fn is not None:
            fn(mode="wait")

    def resume(self):
        fn = getattr(self.llm, "resume_generation", None)
        if fn is not None:
            fn()


def _self_test() -> None:
    from zero_ppo.engine.base import Engine, check
    e = LocalEngine.__new__(LocalEngine)        # no vLLM: check the verb surface
    e._awake = True
    e.llm = None
    missing = [m for m in ("generate", "reload", "sleep", "wake", "pause", "resume",
                           "start_weight_update", "update_weights",
                           "finish_weight_update", "init_weight_transfer_engine")
               if not hasattr(e, m)]
    assert not missing, missing
    e.sleep()                                   # no llm: must be a no-op, not a crash
    e.wake()
    e.pause()
    e.resume()
    assert e._awake is True
    print("  engine.local self-test OK")
    print("    five verbs plus reload/pause/resume; all safe without an engine")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()
