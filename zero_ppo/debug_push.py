#!/usr/bin/env python3
"""Isolate the weight push: engine up, one persistent sync, nothing else.

A hang inside the training loop tells us nothing about where it hangs, so this
script does the push alone and dumps every thread's stack after a timeout - the
dump names the line that never returned.

The Ray path runs in a single process (the driver pattern, world=1): one trainer
worker colocated with a TP=1 engine, so there is no cross-process handle shipping
to confound the diagnosis. The HTTP path talks to an already-running server.

    python3 -m zero_ppo.debug_push --model /data/... --engine ray --timeout 180
    python3 -m zero_ppo.debug_push --model /data/... --engine http --vllm-url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", default="ray", choices=["local", "ray", "http"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--vllm-util", type=float, default=0.30, dest="vllm_util")
    ap.add_argument("--max-model-len", type=int, default=4096, dest="max_model_len")
    ap.add_argument("--prefix-cache", type=int, default=1, dest="prefix_cache")
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8000", dest="vllm_url")
    ap.add_argument("--out", default="runs/zp_debug")
    ap.add_argument("--timeout", type=float, default=180.0)
    a = ap.parse_args()

    # The dump fires once, after the timeout, whether or not anything is stuck.
    faulthandler.dump_traceback_later(a.timeout, exit=True)
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    import torch
    from transformers import AutoModelForCausalLM
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[a.dtype]

    if a.engine == "http":
        # Talk to a running server; the trainer model lives on card 0.
        torch.cuda.set_device(0)
        hf = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
                                                 torch_dtype=dtype).to("cuda:0")
        from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
        from zero_ppo.engine.http import HttpEngine
        from zero_ppo.engine.weight_sync import build
        eng = HttpEngine(a.vllm_url)
        assert eng.wait_ready(timeout=120), "server not ready"
        print("  [http] server ready, building transfer engine ...", flush=True)
        sync = build(hf, rank=0, world=1, client=HTTPVLLMWeightSyncClient(a.vllm_url))
        eng.pause()
        t0 = time.time()
        sync.send()
        print(f"  [http] send_weights returned in {time.time() - t0:.1f}s", flush=True)
        eng.resume()
        out = eng.generate(["def f(x):\n    return "], 0.0, 8)
        print(f"  [http] sample after push: {out[0]['text'][:40]!r}", flush=True)
        sync.close()
        faulthandler.cancel_dump_traceback_later()
        return 0

    # ray / local: one trainer worker colocated with a TP=1 engine, single process.
    torch.cuda.set_device(0)
    hf = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True,
                                             torch_dtype=dtype).to("cuda:0")
    print("  model loaded on cuda:0", flush=True)

    if a.engine == "ray":
        from zero_ppo.engine.ray_actor import RayEngine
        t0 = time.time()
        engine = RayEngine(a.model, dtype=a.dtype, tensor_parallel=1,
                           gpu_util=a.vllm_util, max_model_len=a.max_model_len,
                           prefix_cache=bool(a.prefix_cache), tmp_root=a.out,
                           devices=[0])
        print(f"  engine up in {time.time() - t0:.1f}s", flush=True)
        client = None
        handle = engine.handle
    else:
        from zero_ppo.engine.local import LocalEngine
        engine = LocalEngine(a.model, dtype=a.dtype, tensor_parallel=1,
                             gpu_util=a.vllm_util, max_model_len=a.max_model_len,
                             prefix_cache=bool(a.prefix_cache), tmp_root=a.out)
        client = None
        handle = None

    from zero_ppo.engine.weight_sync import build
    # local engine has no Ray handle; drive its verbs through a thin client adapter
    if a.engine == "local":
        client = _LocalClient(engine)
    sync = build(hf, rank=0, world=1, client=client, handle=handle)
    engine.pause()
    t0 = time.time()
    print("  pushing ...", flush=True)
    sync.send()
    print(f"  push returned in {time.time() - t0:.1f}s", flush=True)
    engine.resume()

    out = engine.generate(["def f(x):\n    return "], 0.0, 8)
    print(f"  sample after push: {out[0]['text'][:40]!r}", flush=True)
    sync.close()
    faulthandler.cancel_dump_traceback_later()
    return 0


class _LocalClient:
    """Adapt an in-process engine to the VLLMWeightSyncClient verb surface."""

    def __init__(self, engine):
        self.engine = engine

    def init_weight_transfer_engine(self, init_info):
        return self.engine.init_weight_transfer_engine(init_info)

    def start_weight_update(self):
        return self.engine.start_weight_update()

    def update_weights(self, request):
        return self.engine.update_weights(request)

    def finish_weight_update(self, weight_version=None):
        return self.engine.finish_weight_update()

    def shutdown_weight_transfer_engine(self):
        fn = getattr(self.engine, "shutdown_weight_transfer_engine", None)
        if fn is not None:
            fn()


if __name__ == "__main__":
    raise SystemExit(main())