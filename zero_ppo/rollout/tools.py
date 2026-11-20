#!/usr/bin/env python3
"""Tool execution, as an injectable callable.

The runner is the only place untrusted text turns into a subprocess. Keeping it an
argument means a remote sandbox can be dropped in later without touching the
rollout loop, and it means the loop can be tested with a fake that returns canned
outputs.

    runner(code: str) -> (stdout or None, error string)
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

#: patterns that must never reach a subprocess
DENY = re.compile(r"\b(import\s+os|import\s+subprocess|open\s*\(|shutil|socket|"
                  r"__import__|eval\s*\(|exec\s*\()")


def local_runner(timeout: float = 5.0, deny: re.Pattern | None = DENY):
    """Run code in a child interpreter, returning the last lines of stdout."""
    def run(code: str):
        if deny is not None and deny.search(code):
            return None, "denied"
        from zero_ppo.util.sandbox import run_code
        return run_code(code, timeout=timeout)
    return run


def fake_runner(table: dict[str, tuple]):
    """Test double: maps a substring of the code to a canned result."""
    calls = []

    def run(code: str):
        calls.append(code)
        for key, val in table.items():
            if key in code:
                return val
        return None, "no match"
    run.calls = calls
    return run
