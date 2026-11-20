#!/usr/bin/env python3
"""DeepSpeed configuration, ZeRO-2 only.

Stage 2 shards gradients and optimiser state but keeps a full copy of the weights
on every rank. That is the whole reason weights can be pushed into the inference
engine without a cross-rank gather, so stage 3 is rejected rather than quietly
accepted - a stage-3 run would need a gather we do not implement, and it would
fail in a way that looks like the model is simply not learning.

    python3 -m zero_ppo.engine.ds_config --self-test
"""
from __future__ import annotations

import argparse


def zero2(micro_batch: int, accum: int, *, lr: float, clip: float = 1.0,
          offload_optimizer: bool = False, dtype: str = "bf16") -> dict:
    if micro_batch < 1 or accum < 1:
        raise ValueError("micro_batch and accum must be >= 1")
    cfg = {
        "train_micro_batch_size_per_gpu": int(micro_batch),
        "gradient_accumulation_steps": int(accum),
        "gradient_clipping": float(clip),
        # Volta has no bfloat16; fp16 with dynamic loss scaling is the fallback.
        # Accept both "bf16" and "bfloat16" (the driver passes the latter); a bare
        # == "bf16" check silently fell through to fp16 and mismatched the engine.
        ("bf16" if dtype in ("bf16", "bfloat16") else "fp16"): {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": float(lr), "betas": [0.9, 0.95], "eps": 1e-8,
                       "weight_decay": 0.0},
        },
        "steps_per_print": 10,
    }
    if offload_optimizer:
        cfg["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu", "pin_memory": True}
    _check(cfg)
    return cfg


def _check(cfg: dict) -> None:
    z = cfg.get("zero_optimization", {})
    stage = z.get("stage")
    if stage != 2:
        raise ValueError(f"only ZeRO-2 is supported, got stage={stage!r}")
    for bad in ("stage3_gather_16bit_weights_on_model_save",
                "stage3_prefetch_bucket_size", "stage3_param_persistence_threshold"):
        if bad in z:
            raise ValueError(f"stage-3 key {bad!r} present in a ZeRO-2 config")


def _self_test() -> None:
    c = zero2(2, 8, lr=5e-6)
    assert c["zero_optimization"]["stage"] == 2
    assert c["optimizer"]["params"]["lr"] == 5e-6
    assert c["train_micro_batch_size_per_gpu"] == 2
    assert c["gradient_accumulation_steps"] == 8
    assert "offload_optimizer" not in c["zero_optimization"]
    assert "bf16" in c and "fp16" not in c
    assert "fp16" in zero2(1, 1, lr=1e-6, dtype="fp16")
    assert "offload_optimizer" in zero2(1, 1, lr=1e-6,
                                        offload_optimizer=True)["zero_optimization"]

    for bad in ({"zero_optimization": {"stage": 3}},):
        try:
            _check(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("stage 3 was accepted")
    for args in ((0, 1), (1, 0)):
        try:
            zero2(*args, lr=1e-6)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad batch args {args} accepted")
    print("  ds_config self-test OK")
    print("    stage 2 enforced, stage-3 keys rejected, lr/batch carried through")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()
