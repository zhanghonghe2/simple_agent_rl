#!/usr/bin/env python3
"""Hand updated weights to the inference engine, the vLLM-native way.

The trainer-side transfer engine is **stateful and built once**: `trainer_init`
opens the channel before the training loop, and every step afterwards is a single
`send_weights()`, which drives start -> update -> finish on the inference side and
moves the tensors over the configured backend. Re-initialising (or shutting down)
the channel every step is what desynchronised the engine's state machine and hung
the workers; that pattern is gone.

Only rank 0 is the sender. The other ranks still build an engine and call
`send_weights()`, because the IPC handle all-gather is a collective over the
trainer process group - skipping it on any rank deadlocks the rest. For IPC that
group must be exactly the colocated trainer ranks, and every rank must own every
parameter (ZeRO-2 keeps a full copy per rank, which is why stage 3 is rejected).

    python3 -m zero_ppo.engine.weight_sync --self-test     # payload building only
    python3 -m zero_ppo.engine.weight_sync --describe      # what this install offers
"""
from __future__ import annotations

import argparse


def _trainer_side():
    """Return the trainer-side transfer classes this vLLM install exposes."""
    out = {}
    try:
        import vllm.distributed.weight_transfer as wt
        for name in ("WeightTransferTrainerFactory", "TrainerWeightTransferEngine",
                     "ModuleSource", "WeightSource"):
            obj = getattr(wt, name, None)
            if obj is not None:
                out[name] = obj
    except Exception as e:                                   # pragma: no cover
        out["error"] = e
    return out


def describe() -> None:
    """Print every signature this install offers on the trainer side."""
    import inspect
    print("  --- clients ---")
    try:
        import vllm.distributed.weight_transfer.clients as cl
        for name in ("VLLMWeightSyncClient", "HTTPVLLMWeightSyncClient",
                     "RayVLLMWeightSyncClient"):
            obj = getattr(cl, name, None)
            if obj is None:
                continue
            try:
                print(f"  {name}{inspect.signature(obj)}")
            except Exception:
                print(f"  {name}: <no signature>")
    except Exception as e:
        print(f"  clients unavailable: {type(e).__name__}: {e}")

    print("  --- trainer engine methods ---")
    from vllm.distributed.weight_transfer import TrainerWeightTransferEngine as T
    for m in ("trainer_init", "send_weights"):
        fn = getattr(T, m, None)
        if fn is not None:
            try:
                print(f"  {m}{inspect.signature(fn)}")
            except Exception:
                print(f"  {m}: <no signature>")

    print("  --- classes ---")
    found = _trainer_side()
    if "error" in found:
        print(f"  weight_transfer unavailable: {found['error']}")
        return
    for name, obj in found.items():
        try:
            print(f"  {name}{inspect.signature(obj)}")
            methods = [m for m in dir(obj)
                       if not m.startswith("_") and ("weight" in m or "send" in m
                                                     or "create" in m or "init" in m)]
            if methods:
                print(f"    methods: {methods}")
        except Exception:
            print(f"  {name}: <no signature>")


def payload(name_dtype_shape: list[tuple[str, str, list[int]]]) -> dict:
    """The metadata dictionary the update phase expects.

    Built from three parallel lists rather than tensor objects, because the tensors
    travel over the transfer channel and only these descriptions go through the RPC.
    """
    return {
        "update_info": {
            "names": [n for n, _, _ in name_dtype_shape],
            "dtype_names": [d for _, d, _ in name_dtype_shape],
            "shapes": [list(s) for _, _, s in name_dtype_shape],
        }
    }


def describe_model(model) -> list[tuple[str, str, list[int]]]:
    import torch
    out = []
    for n, t in model.state_dict().items():
        if not torch.is_floating_point(t):
            continue
        out.append((n, str(t.dtype).replace("torch.", ""), list(t.shape)))
    return out


def _client(handle, client):
    """Resolve the control-plane client: an explicit one, or a Ray handle."""
    if client is not None:
        return client
    if handle is None:
        raise SystemExit("  weight sync needs either a client or the engine's handle")
    from vllm.distributed.weight_transfer.clients import RayVLLMWeightSyncClient
    return RayVLLMWeightSyncClient(handle)


def _init_info(rank: int, packed: bool, packed_buffer_size_bytes):
    from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo
    kw = dict(rank=rank, packed=packed)
    if packed and packed_buffer_size_bytes is not None:
        # older revisions may not accept the buffer-size field; fall back cleanly
        try:
            return IPCTrainerInitInfo(packed_buffer_size_bytes=packed_buffer_size_bytes,
                                      **kw)
        except TypeError:
            pass
    return IPCTrainerInitInfo(**kw)


class WeightSync:
    """Persistent trainer-side transfer engine: build once, `send()` per step.

    Holding the engine across steps is the whole point - the inference side keeps
    its channel and wire params, so a sync is one call, not a renegotiation.
    """

    def __init__(self, side, rank: int):
        self._side = side
        self.rank = rank
        self.sent = 0

    def send(self) -> None:
        """One weight sync. Called on every rank; only rank 0 puts bytes on wire."""
        self._side.send_weights()
        self.sent += 1

    def close(self) -> None:
        closer = getattr(self._side, "shutdown", None)
        if closer is not None:
            closer()


def build(model, *, rank: int, world: int = 1, client=None, handle=None,
          packed: bool = False, packed_buffer_size_bytes=None) -> WeightSync:
    """Open the transfer channel once. `model` must be the HF model (checkpoint
    names), not the Actor wrapper - vLLM loads by HF parameter name."""
    import vllm.distributed.weight_transfer as wt

    client = _client(handle, client)
    init_info = _init_info(rank, packed, packed_buffer_size_bytes)
    source = wt.ModuleSource(model)
    factory = wt.WeightTransferTrainerFactory()
    side = factory.trainer_init(init_info, client=client, source=source)
    return WeightSync(side, rank)


def push(engine, model, handle=None, rank: int = 0, world: int = 1,
         client=None, packed: bool = False) -> str:
    """One-shot build + send + close, for isolating a single transfer in a debug
    run. The training loop uses `build` once and `WeightSync.send` per step."""
    sync = build(model, rank=rank, world=world, client=client, handle=handle,
                 packed=packed)
    try:
        sync.send()
    finally:
        sync.close()
    return f"ipc,rank={rank}"


def _engine_init_class():
    """The engine side's description: same rendezvous, different field names."""
    import vllm.distributed.weight_transfer.nccl_engine as m
    for n, o in vars(m).items():
        if n == "NCCLWeightTransferInitInfo":
            return o
    raise RuntimeError("no NCCLWeightTransferInitInfo in this vllm")


def _trainer_init_class():
    import vllm.distributed.weight_transfer.nccl_engine as m
    for n, o in vars(m).items():
        if n.startswith("NCCL") and n.endswith("TrainerInitInfo"):
            return o
    raise RuntimeError("no NCCLTrainerInitInfo in this vllm")


def _self_test() -> None:
    meta = [("a.weight", "bfloat16", [4, 8]), ("b.weight", "bfloat16", [8])]
    p = payload(meta)
    ui = p["update_info"]
    assert ui["names"] == ["a.weight", "b.weight"]
    assert ui["dtype_names"] == ["bfloat16", "bfloat16"]
    assert ui["shapes"] == [[4, 8], [8]]
    assert set(ui) == {"names", "dtype_names", "shapes"}

    # the persistent engine counts sends and only closes once
    class _Side:
        def __init__(self):
            self.calls = []

        def send_weights(self):
            self.calls.append("send")

        def shutdown(self):
            self.calls.append("shutdown")

    side = _Side()
    sync = WeightSync(side, rank=0)
    sync.send()
    sync.send()
    sync.close()
    assert side.calls == ["send", "send", "shutdown"], side.calls
    assert sync.sent == 2
    print("  weight_sync self-test OK")
    print("    metadata payload built as three parallel lists, no tensors")
    print("    persistent engine: two sends then one shutdown, in order")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    ap.add_argument("--describe", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    elif a.describe:
        describe()
    else:
        ap.print_help()