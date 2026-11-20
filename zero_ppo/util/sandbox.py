#!/usr/bin/env python3
"""Run a program in a child interpreter, with a deny list and a timeout.

Kept inside the package so the boundary between "model output" and "our process"
lives in one file that can be swapped for a remote sandbox later.

    python3 -m zero_ppo.util.sandbox --self-test
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

DENY = re.compile(r"\b(import\s+os|import\s+subprocess|import\s+socket|shutil|"
                  r"__import__|eval\s*\(|exec\s*\()")


def run_code(code: str, timeout: float = 5.0, max_lines: int = 12,
             deny: re.Pattern | None = DENY):
    """-> (output or None, error). Output keeps head and tail lines, because the
    model needs to see both what it printed first and what it printed last."""
    if deny is not None and deny.search(code):
        return None, "denied"
    try:
        r = subprocess.run([sys.executable, "-I", "-c", code],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    out = r.stdout.strip()
    if not out:
        errs = [l for l in (r.stderr or "").strip().splitlines() if l.strip()]
        return None, (errs[-1][:120] if errs else "no output")
    lines = [l for l in out.splitlines() if l.strip()]
    if len(lines) > max_lines:
        lines = lines[:4] + ["..."] + lines[-(max_lines - 4):]
    return "\n".join(lines)[-400:], ""


def _self_test() -> None:
    out, err = run_code("print(6*7)")
    assert out == "42" and err == ""
    out, err = run_code("print(1/0)")
    assert out is None and "ZeroDivisionError" in err
    out, err = run_code("x = 1")
    assert out is None and err == "no output"
    out, err = run_code("import os")
    assert out is None and err == "denied", err
    many = "\n".join(f"print({i})" for i in range(20))
    out, _ = run_code(many)
    assert out.splitlines()[0] == "0" and out.splitlines()[-1] == "19"
    assert "..." in out
    print("  sandbox self-test OK")
    print("    runs, reports errors, rejects denial, keeps head and tail of long output")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()
