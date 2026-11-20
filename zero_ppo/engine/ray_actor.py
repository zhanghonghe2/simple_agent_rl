#!/usr/bin/env python3
"""The engine as a Ray actor: same five verbs, different process.

The class is defined at module level on purpose. Ray reconstructs an actor class in
another process from its module and qualified name, so a class created inside a
function cannot be restored there - the receiving rank aborts while unpickling the
handle. The decorator is applied lazily so this module still imports without ray.

Weights still travel by CUDA IPC through the trainer engine; the handle is kept
because that is what the vLLM client needs.

    python3 -m zero_ppo.engine.ray_actor --self-test     # no Ray needed
"""
from __future__ import annotations

import argparse

#: decorator applied on first use, see `_remote_cls`
_ACTOR = None


class _EngineActor:
    """One in-process engine behind a Ray actor, exposed with the vLLM verbs."""

    def __init__(self, model_path: str, **kw):
        from zero_ppo.engine.local import LocalEngine
        self.inner = LocalEngine(model_path, **kw)

    # ---- inference ---------------------------------------------------------
    def generate(self, prompts, temperature, max_tokens):
        return self.inner.generate(prompts, temperature, max_tokens)

    def sleep(self, level=1):
        self.inner.sleep(level=level)

    def wake(self, tags=None):
        self.inner.wake(tags=tags)

    def pause(self):
        self.inner.pause()

    def resume(self):
        self.inner.resume()

    # ---- weight update: the client calls these by name on the handle --------
    def init_weight_transfer_engine(self, request):
        return self.inner.init_weight_transfer_engine(request)

    def start_weight_update(self):
        return self.inner.start_weight_update()

    def update_weights(self, request):
        return self.inner.update_weights(request)

    def finish_weight_update(self, **kw):
        return self.inner.finish_weight_update()

    # ---- snapshot reload (the non-transfer path) ---------------------------
    def reload(self, snapshot: str, step: int):
        self.inner.llm = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        self.inner.llm = self.inner._build(snapshot)
        self.inner._awake = True
        return snapshot


def _remote_cls():
    global _ACTOR
    if _ACTOR is None:
        import ray
        _ACTOR = ray.remote(num_gpus=0)(_EngineActor)   # devices are pinned by env
    return _ACTOR


class RayEngine:
    """Client-side handle implementing the same verbs as LocalEngine."""

    backend_name = "ray"

    def __init__(self, model_path: str, *, dtype: str = "bfloat16",
                 tensor_parallel: int = 1, gpu_util: float = 0.3,
                 max_model_len: int = 4096, prefix_cache: bool = True,
                 tmp_root: str = "runs/zero_ppo", address: str | None = None,
                 devices: list | None = None):
        import ray
        if not ray.is_initialized():
            ray.init(address=address, num_gpus=0, ignore_reinit_error=True)
        # Colocation: Ray books no GPU for the engine; it pins its cards through
        # CUDA_VISIBLE_DEVICES injected into the actor process.
        opts = {}
        if devices is not None:
            opts["runtime_env"] = {"env_vars": {
                "CUDA_VISIBLE_DEVICES": ",".join(str(d) for d in devices)}}
        self.handle = _remote_cls().options(**opts).remote(
            model_path, dtype=dtype, tensor_parallel=tensor_parallel,
            gpu_util=gpu_util, max_model_len=max_model_len,
            prefix_cache=prefix_cache, tmp_root=tmp_root)
        self.tmp_root = tmp_root
        self.world = tensor_parallel

    def generate(self, prompts, temperature=0.7, max_tokens=1024):
        import ray
        return ray.get(self.handle.generate.remote(prompts, temperature, max_tokens))

    def sleep(self, level=1):
        import ray
        ray.get(self.handle.sleep.remote(level=level))

    def wake(self, tags=None):
        import ray
        ray.get(self.handle.wake.remote(tags=tags))

    def pause(self):
        import ray
        ray.get(self.handle.pause.remote())

    def resume(self):
        import ray
        ray.get(self.handle.resume.remote())

    def init_weight_transfer_engine(self, request):
        import ray
        return ray.get(self.handle.init_weight_transfer_engine.remote(request))

    def start_weight_update(self):
        import ray
        return ray.get(self.handle.start_weight_update.remote())

    def update_weights(self, request):
        import ray
        return ray.get(self.handle.update_weights.remote(request))

    def finish_weight_update(self, **kw):
        import ray
        return ray.get(self.handle.finish_weight_update.remote(**kw))

    def push_weights(self, model, rank: int = 0):
        from zero_ppo.engine.weight_sync import push
        return push(None, model, handle=self.handle, rank=rank, world=self.world)

    def reload(self, model, tok, step: int):
        """Snapshot fallback, only used when the transfer is unavailable."""
        import os
        import ray
        snap = os.path.join(self.tmp_root, f"snap_step{step}")
        model.save_pretrained(snap)
        tok.save_pretrained(snap)
        ray.get(self.handle.reload.remote(snap, step))
        return snap


class _TrainActor:
    """One card's training worker behind a Ray actor. Module level so Ray can
    rebuild it in another process. The driver pins it to a card with num_gpus=1."""

    def __init__(self, args, rank: int, world: int):
        from zero_ppo.train_actor import TrainWorker
        self.worker = TrainWorker(args, rank, world)

    def init_sync(self, handle=None, client=None, packed: bool = False):
        self.worker.init_sync(handle=handle, client=client, packed=packed)

    def compute_values(self, ids, mask):
        return self.worker.compute_values(ids, mask)

    def train_step(self, ids, mask, adv, returns=None):
        return self.worker.train_step(ids, mask, adv, returns=returns)

    def do_sync(self):
        self.worker.do_sync()

    def save(self, path):
        return self.worker.save(path)

    def free_memory(self):
        """Release cached blocks so the engine can re-allocate its KV cache.

        Called by the driver around the inline eval: the trainer's allocator keeps
        freed tensors in its cache, and that cache is what starves the wake.
        """
        return self.worker.free_memory()

    def trainable(self):
        return self.worker.trainable()


_TRAIN_ACTOR = None


def train_actor_cls():
    global _TRAIN_ACTOR
    if _TRAIN_ACTOR is None:
        import ray
        _TRAIN_ACTOR = ray.remote(num_gpus=1)(_TrainActor)
    return _TRAIN_ACTOR


def _self_test() -> None:
    verbs = ("generate", "sleep", "wake", "pause", "resume", "reload", "push_weights",
             "init_weight_transfer_engine", "start_weight_update",
             "update_weights", "finish_weight_update")
    missing = [m for m in verbs if not hasattr(RayEngine, m)]
    assert not missing, missing
    actor_verbs = [m for m in ("generate", "sleep", "wake", "pause", "resume", "reload",
                               "init_weight_transfer_engine",
                               "start_weight_update", "update_weights",
                               "finish_weight_update")
                   if not hasattr(_EngineActor, m)]
    assert not actor_verbs, actor_verbs
    train_verbs = [m for m in ("init_sync", "train_step", "do_sync", "trainable",
                               "compute_values", "save")
                   if not hasattr(_TrainActor, m)]
    assert not train_verbs, train_verbs
    print("  engine.ray self-test OK")
    print(f"    client verbs: {len(verbs)}, actor verbs mirror the vLLM object")
    print("    engine + train actor classes are module level for cross-process rebuild")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()
