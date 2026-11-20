#!/usr/bin/env python3
"""The engine interface: the whole surface a rollout backend has to provide.

Deliberately five verbs. Everything the training loop needs from inference is
"generate for these prompts", "hand over the new weights", and "give the memory
back or take it again". Keeping it this small is what makes a second backend
possible - an in-process engine and a Ray actor differ in everything except this.

    python3 -m zero_ppo.engine.base --self-test
"""
from __future__ import annotations

import argparse
from typing import Protocol, runtime_checkable


@runtime_checkable
class Engine(Protocol):
    def generate(self, prompts: list[str], temperature: float,
                 max_tokens: int) -> list[dict]:
        """-> [{"text": str, "tokens": list[int]}, ...] in prompt order."""

    def start_weight_update(self) -> None: ...
    def update_weights(self, request) -> None: ...
    def finish_weight_update(self) -> None: ...

    def sleep(self) -> None:
        """Release the cache so the backward pass has room."""

    def wake(self) -> None:
        """Take it back before the next rollout."""


def check(engine) -> list[str]:
    """Which of the five verbs are missing. Empty means the backend is usable."""
    missing = [m for m in ("generate", "sleep", "wake") if not hasattr(engine, m)]
    upd = [m for m in ("start_weight_update", "update_weights") if not hasattr(engine, m)]
    return missing + upd


# --------------------------------------------------------------------- self test
class _Good:
    def generate(self, prompts, temperature, max_tokens):
        return [{"text": "", "tokens": []} for _ in prompts]

    def start_weight_update(self): ...
    def update_weights(self, request): ...
    def finish_weight_update(self): ...
    def sleep(self): ...
    def wake(self): ...


class _Bad:
    def generate(self, prompts, temperature, max_tokens):
        return []


def _self_test() -> None:
    assert check(_Good()) == [], check(_Good())
    assert isinstance(_Good(), Engine), "a complete backend must satisfy the protocol"
    assert not isinstance(_Bad(), Engine), "an incomplete backend must not"
    bad = check(_Bad())
    assert "sleep" in bad and "update_weights" in bad, bad
    print("  engine.base self-test OK")
    print("    five verbs: generate, start/update/finish weight, sleep, wake")
    print(f"    an incomplete backend is reported as missing {len(bad)} of them")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()
