#!/usr/bin/env python3
"""Colocation schedule: one wake, one sync and one sleep per optimisation step.

The engine and the trainer share the cards, so the inference engine has to give up
its cache before the backward pass and take it back before the next rollout. The
failure mode is not an error: doing it once per chunk means sixteen wake/sleep
cycles per step and the step time triples, while every number still looks correct.
This object exists so the count is a test, not a hope.

    python3 -m zero_ppo.engine.colocate --self-test
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager


class Colocation:
    """`inference()` for rollout, `training()` for the update, one each per step."""

    def __init__(self, engines):
        self.engines = list(engines)
        self.wakes = 0
        self.sleeps = 0
        self.syncs = 0
        self._awake = False

    def _each(self, method: str, counter: str) -> None:
        for e in self.engines:
            fn = getattr(e, method, None)
            if fn is not None:
                fn()
        setattr(self, counter, getattr(self, counter) + 1)

    @contextmanager
    def inference(self):
        """Engine awake for the whole rollout; closed once, not per chunk."""
        if not self._awake:
            self._each("wake", "wakes")
            self._awake = True
        yield
        # deliberately no sleep here: the caller decides when the rollout is done

    def sync_weights(self, state_dict) -> None:
        for e in self.engines:
            fn = getattr(e, "load_weights", None)
            if fn is not None:
                fn(state_dict)
        self.syncs += 1

    def training(self) -> None:
        """Hand the memory back exactly once, before the update."""
        if self._awake:
            self._each("sleep", "sleeps")
            self._awake = False

    def check(self, steps: int) -> None:
        if not (self.wakes == self.sleeps == steps):
            raise AssertionError(
                f"colocation cycle broken: wakes={self.wakes} sleeps={self.sleeps} "
                f"steps={steps} - a sleep inside the rollout loop looks like this")


# --------------------------------------------------------------------- self test
class _FakeEngine:
    def __init__(self):
        self.log = []
        self.cache = None

    def wake(self):
        self.log.append("wake")

    def sleep(self):
        self.log.append("sleep")

    def load_weights(self, sd):
        self.cache = sd
        self.log.append("load")


def _self_test() -> None:
    eng = _FakeEngine()
    col = Colocation([eng])
    for step in range(3):
        with col.inference():                 # one wake per step
            for _chunk in range(8):           # many chunks, still one wake
                pass
            col.sync_weights({"w": step})
        col.training()                        # one sleep per step
    col.check(3)
    assert eng.log.count("wake") == 3 and eng.log.count("sleep") == 3
    assert eng.log.count("load") == 3 and eng.cache == {"w": 2}
    assert eng.log == ["wake", "load", "sleep"] * 3, eng.log

    # training twice without a rollout must not sleep twice
    col.training()
    assert eng.log.count("sleep") == 3

    col.wakes, col.sleeps = 3, 16
    try:
        col.check(3)
    except AssertionError:
        pass
    else:
        raise AssertionError("a broken cycle was accepted")
    print("  colocate self-test OK")
    print("    3 steps -> wake/load/sleep exactly once each, order preserved")
    print("    a sleep-per-chunk pattern is rejected by check()")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()
