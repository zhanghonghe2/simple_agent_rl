#!/usr/bin/env python3
"""Decide whether a boxed answer matches the gold value.

Mirrors the order used by the earlier pipeline: a symbolic verifier if it is
installed, then a sympy comparison, then a normalised string comparison. Kept
inside the package so nothing has to reach across to the old one.

The answer is extracted with a brace-balanced scan, because a value like
\\boxed{\\sqrt{51}} or \\boxed{\\frac{1}{2}} contains braces and a naive pattern
misses it - which silently scored correct answers as wrong.

    python3 -m zero_ppo.util.judge --self-test
"""
from __future__ import annotations

import argparse
import re

_FRAC = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")


def extract_answer(text: str | None) -> str | None:
    """The content of the last \\boxed{...}, with nested braces respected."""
    if not text:
        return None
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth, buf = 1, []
    while i < len(text) and depth:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(c)
        i += 1
    return "".join(buf).strip() or None


def _norm(s: str) -> str:
    s = s.strip().replace("$", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    for _ in range(6):
        s2 = _FRAC.sub(r"((\1)/(\2))", s)
        if s2 == s:
            break
        s = s2
    s = s.replace("\\sqrt", "sqrt").replace("\\pi", "pi").replace("\\cdot", "*")
    # sqrt{4} is not a call sympy can parse; sqrt(4) is. Without this, \sqrt{4}
    # against 2 fell through to a string compare and was scored wrong.
    for _ in range(4):
        s2 = re.sub(r"\bsqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
        if s2 == s:
            break
        s = s2
    # Removing \left/\right above leaves the padding spaces behind, so
    # "\left( 3, x \right)" became "( 3, x )" and never matched "(3, x)" - a
    # correct coordinate answer was scored wrong under BOTH rules. Space around
    # structural punctuation carries no meaning, so drop it.
    s = re.sub(r"\s*([(),\[\];])\s*", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def _sympy_eq(a: str, b: str) -> bool | None:
    try:
        import sympy as sp
    except Exception:
        return None
    try:
        pa = sp.sympify(_norm(a))
        pb = sp.sympify(_norm(b))
        return bool(sp.simplify(pa - pb) == 0)
    except Exception:
        return None


def equal(pred: str | None, gold: str | None, tol: float = 1e-5) -> bool:
    if pred is None or gold is None:
        return False
    try:                                        # numeric first, tolerance aware
        return abs(float(_norm(pred)) - float(_norm(gold))) <= tol
    except Exception:
        pass
    try:
        from math_verify import parse, verify
        return bool(verify(parse(pred), parse(gold)))
    except Exception:
        pass
    r = _sympy_eq(pred, gold)
    if r is not None:
        return r
    return _norm(pred) == _norm(gold)


def _self_test() -> None:
    assert extract_answer("so \\boxed{5}") == "5"
    assert extract_answer("\\boxed{\\sqrt{51}}") == "\\sqrt{51}"
    assert extract_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_answer("no box here") is None

    cases = [
        ("5", "5", True),
        ("0.5", "\\frac{1}{2}", True),
        ("\\sqrt{51}", "\\sqrt{51}", True),
        ("\\sqrt{52}", "\\sqrt{51}", False),
        ("(0,3)", "(0,3)", True),
        ("(0,3)", "(0,5)", False),
        ("p - q", "p - q", True),
        ("[0,3)", "[0,3)", True),
        ("[0,3)", "[0,4)", False),
        # \left/\right and its padding must not make a correct answer wrong
        ("(3, \\frac{\\pi}{2})", "\\left( 3, \\frac{\\pi}{2} \\right)", True),
        ("(3,\\frac{\\pi}{2})", "\\left( 3, \\frac{\\pi}{2} \\right)", True),
        ("[0,3)", "\\left[0, 3\\right)", True),
        ("(1,2)", "\\left( 1, 2 \\right)", True),
        ("(1,3)", "\\left( 1, 2 \\right)", False),
        ("2", "\\sqrt{4}", True),
    ]
    for pred, gold, want in cases:
        got = equal(pred, gold)
        assert got == want, f"{pred!r} vs {gold!r}: got {got}, want {want}"
    assert equal(None, "5") is False

    print("  judge self-test OK")
    print(f"    brace-balanced extraction; {len(cases)} equal/unequal cases")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()
