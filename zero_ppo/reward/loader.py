#!/usr/bin/env python3
"""Load a reward from a file path.

    python3 -m zero_ppo.reward.loader --self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def load(path: str):
    """Import a file and return its Reward instance.

    The file must expose either `REWARD` (an instance) or a `Reward` subclass.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"reward file not found: {path}")
    name = "zero_ppo_reward_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if hasattr(mod, "REWARD"):
        return mod.REWARD
    for obj in vars(mod).values():
        if isinstance(obj, type) and obj.__name__.lower().endswith("reward"):
            return obj()
    raise AttributeError(f"{path} exposes neither REWARD nor a *Reward class")


def _self_test() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "agent_reward_test.py")
    r = load(path)
    assert hasattr(r, "__call__")
    a = load(path)
    b = load(path)
    assert a is not b, "each load should give a fresh instance"
    try:
        load("/nonexistent/x.py")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing file accepted")
    print("  reward loader OK")
    print(f"    loaded {os.path.basename(path)} -> {type(r).__name__} (name={r.name})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()
