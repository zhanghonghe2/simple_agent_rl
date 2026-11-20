#!/usr/bin/env python3
"""Greedy held-out evaluation DURING training, on the colocated engine.

Why this exists: RL runs on this project peak early and then collapse (entropy
falls, all-wrong groups give exactly zero gradient, and the policy locks). The
best checkpoint is therefore usually somewhere in the middle, and `final` may
already be past the collapse. Saving every N steps keeps the weights but does not
tell you WHICH one is best; this measures it and keeps the winner.

Two hard constraints, both about memory:

  * NO second engine. Spawning a vLLM instance for evaluation would put a second
    copy of the weights plus its own KV cache on cards that already hold the
    policy, the optimiser, the reference model and the training engine. The
    existing engine is reused through `eval._run_shard`, which takes `generate`
    as an argument - so the prompts, tool loop and context guard are byte-identical
    to the offline evaluator, with no second implementation to drift.
  * Sleep/wake around the call. The trainer's activations must be released before
    the KV cache is re-allocated, and the cache must be given back before the next
    backward. The caller owns that ordering (see driver); this module only asks
    for the engine to be awake and always leaves it asleep.

Greedy only (temperature 0, one sample per problem): deterministic, so scores are
comparable across steps without sampling noise, and it costs one pass instead of
n. The loose verdict uses the local equivalence rule by default; `--eval-judge-api`
adds the same concurrent API arbitration the offline evaluator uses.

    python3 -m zero_ppo.inline_eval --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def greedy_generate(engine):
    """Adapt the colocated engine to the signature `eval._run_shard` expects.

    _run_shard always calls generate(prompts, temp, n, max_tokens, seeds=[...]).
    The engine takes (prompts, temperature, max_tokens) and returns one completion
    list per prompt. Greedy means n == 1 and temperature 0, where a seed cannot
    change the output, so dropping `seeds` is safe. `n != 1` is refused rather than
    silently returning one sample - that would fabricate a pass@k.
    """
    def gen(prompts, temperature, n, max_tokens, seeds=None):
        if int(n) != 1:
            raise ValueError("inline eval is greedy only: n must be 1, got %r" % (n,))
        prompts = list(prompts)
        outs = engine.generate(prompts, float(temperature), int(max_tokens))
        # Shape mismatch that crashed the first inline eval: the colocated engine
        # returns a FLAT list, one {text, tokens} dict per prompt, while
        # _run_shard indexes got[0]["text"] - it expects a list of CANDIDATES per
        # prompt (the offline evaluator's generate returns n candidates). Wrap
        # each dict into its own single-candidate list.
        if len(outs) != len(prompts):
            raise ValueError(f"engine returned {len(outs)} results for "
                             f"{len(prompts)} prompts")
        wrapped = []
        for o in outs:
            if isinstance(o, dict):
                wrapped.append([o])
            elif isinstance(o, (list, tuple)):
                wrapped.append(list(o))      # already nested: leave as is
            else:
                raise TypeError(f"unexpected completion type {type(o).__name__}")
        return wrapped
    return gen


def score_greedy(results, rule: str = "strict") -> dict:
    """Accuracy plus the behaviour counters that reveal a collapsing policy."""
    rows = [r for r in results
            if any(s["mode"] == "greedy" for s in r["samples"])]
    n = len(rows)
    if not n:
        return {"n": 0, "acc": 0.0, "boxed": 0.0, "turns": 0.0, "tool_ok": 0.0}
    hit = boxed = turns = ok = 0
    cut = 0
    for r in rows:
        for s in r["samples"]:
            if s["mode"] != "greedy":
                continue
            hit += bool(s.get(rule))
            boxed += bool(s.get("texts") and "\\boxed{" in (s["texts"][-1] or ""))
            turns += int(s.get("turns", 0))
            ok += int(s.get("tool_ok", 0))
            cut += bool(s.get("cut"))
    return {"n": n, "acc": round(hit / n, 4), "boxed": round(boxed / n, 4),
            "turns": round(turns / n, 3), "tool_ok": round(ok / n, 3),
            "cut": round(cut / n, 4)}


def run_eval(engine, tok, fmt, run_code, problems, *, max_model_len=4096,
             max_new=1024, max_turns=6, batch=64, min_new=32, seed=0,
             judge=None) -> dict:
    """One greedy pass over `problems` on the ALREADY-AWAKE engine.

    The caller must have woken the engine (weights + kv_cache) and released the
    trainer's memory first. This function does not sleep or wake: doing it here
    would hide the ordering that keeps the colocated setup from segfaulting.
    """
    from zero_ppo.eval import _run_shard, arbitrate

    sa = types.SimpleNamespace(
        greedy=True, samples=0, temperature=0.0, top_p=1.0,
        max_turns=int(max_turns), max_new=int(max_new), batch=int(batch),
        max_model_len=int(max_model_len), min_new=int(min_new), seed=int(seed))
    results, calls = _run_shard(problems, sa, greedy_generate(engine),
                                tok, fmt, run_code)
    out = score_greedy(results, "strict")
    loose = score_greedy(results, "loose")
    # Loose is a superset of strict by definition. A run nonetheless reported
    # strict 63.32% > loose 61.79%: math_verify's verify() can return False (not
    # raise) on inputs it parses but cannot decide, short-circuiting the string
    # fallback inside util.judge.equal. loose_equal now unions with strict; this
    # per-sample union is the belt-and-braces so the reported numbers can never
    # contradict each other, and the disagreement count makes any recurrence
    # visible instead of silent.
    odd = 0
    for r in results:
        for s in r["samples"]:
            if s.get("mode") != "greedy":
                continue
            if s.get("strict") and not s.get("loose"):
                s["loose"] = True
                odd += 1
    out["acc_loose"] = max(loose["acc"], out["acc"])
    if odd:
        out["acc_loose"] = score_greedy(results, "loose")["acc"]
    out["rule_conflicts"] = odd
    out["calls"] = calls["n"]
    if judge is not None and getattr(judge, "available", False):
        stats = arbitrate(results, judge)
        out["acc_loose"] = score_greedy(results, "loose")["acc"]
        out["judge"] = stats
    return out


class BestTracker:
    """Remember the best score and where its weights live.

    Ties keep the EARLIER step: with equal accuracy the less-drifted policy is the
    safer artifact, and later steps are closer to the collapse.

    The default metric is `acc_loose`, i.e. value-equivalent correctness, because
    that is what the RL reward optimises (reward_result judges by value, ignoring
    LaTeX form). Selecting on strict string equality would pick a checkpoint for a
    different objective than the one being trained.
    """

    def __init__(self, path: str | None = None, metric: str = "acc_loose"):
        self.metric = metric
        self.best = None
        self.best_step = None
        self.best_path = None
        self.path = path
        self.history = []

    def offer(self, step: int, score: dict, ckpt: str | None = None) -> bool:
        v = float(score.get(self.metric, 0.0))
        self.history.append({"step": step, self.metric: v,
                             "boxed": score.get("boxed"),
                             "turns": score.get("turns"), "ckpt": ckpt})
        improved = self.best is None or v > self.best
        if improved:
            self.best, self.best_step, self.best_path = v, step, ckpt
        if self.path:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"metric": self.metric, "best": self.best,
                           "best_step": self.best_step,
                           "best_ckpt": self.best_path,
                           "history": self.history}, f,
                          ensure_ascii=False, indent=2)
        return improved


def _self_test() -> None:
    import torch  # noqa: F401  (kept for parity with the other self-tests)

    # --- generate adapter: greedy passes through, n>1 is refused -------------
    # The stub mirrors the REAL engine: engine/local.py returns a FLAT list of
    # {text, tokens}, one per prompt. An earlier stub returned nested lists and
    # therefore hid the shape bug that crashed the first inline eval.
    class _Eng:
        def __init__(self):
            self.seen = []

        def generate(self, prompts, temperature, max_tokens):
            self.seen.append((list(prompts), temperature, max_tokens))
            return [{"text": "\\boxed{7}", "tokens": [1, 2]} for _ in prompts]

    e = _Eng()
    g = greedy_generate(e)
    out = g(["a", "b"], 0.0, 1, 128, seeds=[3, 4])
    assert e.seen == [(["a", "b"], 0.0, 128)], e.seen
    # must be nested now, so _run_shard's got[0]["text"] works
    assert len(out) == 2 and isinstance(out[0], list), out
    assert out[0][0]["text"] == "\\boxed{7}", out
    try:
        g(["a"], 0.7, 4, 128)
    except ValueError:
        pass
    else:
        raise AssertionError("n>1 must be refused, not silently downgraded")

    # an already-nested engine must pass through unchanged
    class _Nested:
        def generate(self, prompts, temperature, max_tokens):
            return [[{"text": "x", "tokens": []}] for _ in prompts]

    assert greedy_generate(_Nested())(["a"], 0.0, 1, 8)[0][0]["text"] == "x"

    # a length mismatch is an error, not a silent zip truncation
    class _Short:
        def generate(self, prompts, temperature, max_tokens):
            return [{"text": "x", "tokens": []}]

    try:
        greedy_generate(_Short())(["a", "b"], 0.0, 1, 8)
    except ValueError:
        pass
    else:
        raise AssertionError("a short result list must raise")

    # --- scoring ------------------------------------------------------------
    def row(strict, loose, boxed=True, turns=2, ok=1, cut=""):
        text = "reasoning \\boxed{7}" if boxed else "no answer"
        return {"samples": [{"mode": "greedy", "strict": strict, "loose": loose,
                             "texts": [text], "turns": turns, "tool_ok": ok,
                             "cut": cut}]}

    s = score_greedy([row(True, True), row(False, True), row(False, False),
                      row(False, False, boxed=False, turns=6, cut="context")])
    assert s["n"] == 4 and abs(s["acc"] - 0.25) < 1e-9, s
    assert abs(s["boxed"] - 0.75) < 1e-9, s
    assert abs(s["turns"] - 3.0) < 1e-9, s          # (2+2+2+6)/4
    assert abs(s["cut"] - 0.25) < 1e-9, s
    assert score_greedy([])["n"] == 0

    # sample-mode rows must not leak into a greedy score
    mixed = {"samples": [{"mode": "greedy", "strict": True, "loose": True,
                          "texts": ["\\boxed{7}"], "turns": 1, "tool_ok": 0},
                         {"mode": "sample", "strict": False, "loose": False,
                          "texts": ["x"], "turns": 9, "tool_ok": 0}]}
    assert score_greedy([mixed])["acc"] == 1.0
    assert score_greedy([mixed])["turns"] == 1.0

    # --- best tracker: selects on LOOSE, the objective RL actually optimises --
    t = BestTracker()
    assert t.metric == "acc_loose", t.metric
    assert t.offer(20, {"acc_loose": 0.60}, "c20") is True
    assert t.offer(40, {"acc_loose": 0.65}, "c40") is True
    assert t.offer(60, {"acc_loose": 0.65}, "c60") is False   # tie -> keep step 40
    assert t.offer(80, {"acc_loose": 0.40}, "c80") is False   # collapse
    assert t.best_step == 40 and t.best_path == "c40" and t.best == 0.65
    assert len(t.history) == 4

    # the two rules can disagree: strict must NOT drive the choice
    t3 = BestTracker()
    t3.offer(20, {"acc": 0.70, "acc_loose": 0.72}, "a")
    t3.offer(40, {"acc": 0.60, "acc_loose": 0.80}, "b")   # strict drops, loose wins
    assert t3.best_step == 40 and t3.best == 0.80, (t3.best_step, t3.best)
    # explicit strict selection still available for ablation
    t4 = BestTracker(metric="acc")
    t4.offer(20, {"acc": 0.70, "acc_loose": 0.72}, "a")
    t4.offer(40, {"acc": 0.60, "acc_loose": 0.80}, "b")
    assert t4.best_step == 20 and t4.best == 0.70

    # a collapse after a peak must not erase the peak
    t2 = BestTracker()
    t2.offer(20, {"acc_loose": 0.70}, "a")
    for st in (40, 60, 80):
        t2.offer(st, {"acc_loose": 0.0}, f"b{st}")
    assert t2.best_step == 20 and t2.best == 0.70

    # --- end to end on the real scheduler, with a stub engine ---------------
    from zero_ppo.util.prompts import ToolFormat

    class _Tok:
        chat_template = "yes"
        pad_token_id = 0
        eos_token_id = 0

        def apply_chat_template(self, messages, add_generation_prompt=True,
                                tokenize=False):
            s = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
            return s + ("<|assistant|>" if add_generation_prompt else "")

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text)))}

    tk = _Tok()
    fmt = ToolFormat(tk)
    probs = [{"conversations": [{"from": "human", "value": "What is 2+2?"}],
              "system": "s", "gold": "4"},
             {"conversations": [{"from": "human", "value": "What is 3+3?"}],
              "system": "s", "gold": "6"}]

    class _Ans:
        def generate(self, prompts, temperature, max_tokens):
            assert temperature == 0.0, "inline eval must be greedy"
            # flat, exactly like engine/local.py
            return [{"text": "\\boxed{4}", "tokens": [1]} for _ in prompts]

    def no_code(src):
        return ("", None)

    got = run_eval(_Ans(), tk, fmt, no_code, probs, max_model_len=512,
                   max_new=64, max_turns=2, batch=8)
    assert got["n"] == 2, got
    assert abs(got["acc"] - 0.5) < 1e-9, got      # 4 matches gold 4, not gold 6
    assert got["calls"] >= 1

    # Regression: a run reported strict 63.32% > loose 61.79%. Loose is a superset
    # by definition, so that is a bug, not noise. math_verify's verify() can return
    # False without raising, short-circuiting the string fallback in judge.equal;
    # it only happens where math_verify is installed, so stub that behaviour here.
    import zero_ppo.util.judge as _J
    _orig_equal = _J.equal

    def _false_on_text(p, g, tol=1e-5):
        if p is not None and g is not None and "\\text" in str(g):
            return False                      # parsed, undecided, no exception
        return _orig_equal(p, g)

    _J.equal = _false_on_text
    try:
        from zero_ppo.eval import loose_equal, strict_equal
        t = "\\text{No solution}"
        assert strict_equal(t, t) is True
        assert loose_equal(t, t) is True, "loose must contain strict"
    finally:
        _J.equal = _orig_equal

    # summary-level guard: a per-sample conflict is unioned and counted
    conflict = [{"samples": [{"mode": "greedy", "strict": True, "loose": False,
                              "texts": ["\\boxed{7}"], "turns": 1,
                              "tool_ok": 0, "cut": ""}]}]
    assert score_greedy(conflict, "strict")["acc"] == 1.0
    assert score_greedy(conflict, "loose")["acc"] == 0.0   # raw scorer is faithful

    print("  inline_eval self-test OK")
    print("    loose contains strict even when math_verify returns False")
    print("    reuses the colocated engine; n>1 refused (no fabricated pass@k)")
    print("    greedy-only scoring with boxed/turns/cut counters")
    print("    best tracker: selects on acc_loose (the RL objective), ties keep "
          "the earlier step, a collapse keeps the peak")
    print("    end-to-end through eval._run_shard with a stub engine")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()