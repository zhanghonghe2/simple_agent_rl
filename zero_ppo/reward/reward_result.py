#!/usr/bin/env python3
"""Outcome reward for GRPO: 1.0 if the final answer is numerically correct, else 0.

Loose (value-equivalent) judgement, decided by the API. Formatting is explicitly
NOT part of the task: 0.5 earns full credit against \\frac{1}{2}, 2 against
\\sqrt{4}, and \\left(1,2\\right) against (1,2). Only the value matters.

The API is the primary judge, not a tie-breaker: every extracted answer is asked,
so the verdict does not depend on which forms the local rules happen to handle.
`eval.LLMJudge` is reused rather than reimplemented - it sends ONLY the two answer
strings (never the problem, the gold-bearing trajectory, or any reasoning), asks
concurrently, dedupes identical pairs, and retries transient failures.

If a pair fails every retry it is absent from the API result, and the sample falls
back to the local equivalence rule (`util.judge.equal`) instead of being scored 0.
Scoring a network failure as wrong would inject false negatives into the training
signal and look exactly like a policy that cannot solve the problem. Failures are
counted and printed, never silent.

An answer with no \\boxed{} scores 0 without an API call: there is nothing to
compare, and the box is how the pipeline locates the final answer.

Outcome only, on the last generated token: GRPO reduces a trajectory to one scalar
and broadcasts it, so per-turn credit cannot be expressed here (that needs GAE and
the PPO path). Tool success is irrelevant - a crashed program whose boxed value is
right still scores 1.

GRPO note: a group whose members all score the same gets ZERO advantage, so use a
group size large enough to catch a correct sample (GROUP=8 or more).

    DEEPSEEK_API_KEY=sk-... python3 -m zero_ppo.reward.reward_result --self-test
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from zero_ppo.reward.base import Reward, Sample, final_answer_text
from zero_ppo.util.judge import equal, extract_answer


class ResultReward(Reward):
    """1.0 on the last generated token iff the boxed value equals gold."""

    name = "reward_result"

    def __init__(self, w_final: float = 1.0, model: str | None = None,
                 base_url: str | None = None, api_key_env: str = "DEEPSEEK_API_KEY",
                 workers: int | None = None, retries: int = 2,
                 timeout: float = 30.0):
        self.w_final = float(w_final)
        from zero_ppo.eval import LLMJudge
        self.judge = LLMJudge(
            model=model or os.environ.get("JUDGE_MODEL", "deepseek-chat"),
            base_url=base_url, api_key_env=api_key_env, timeout=timeout,
            workers=int(workers if workers is not None
                        else os.environ.get("JUDGE_WORKERS", 16)),
            retries=retries)
        self._last = {"n": 0, "correct": 0, "boxed": 0, "api": 0,
                      "fallback": 0, "errors": 0}

    def _answers(self, samples: list[Sample]) -> list[str | None]:
        out = []
        for s in samples:
            text = final_answer_text(s.texts)
            out.append(extract_answer(text) if text is not None else None)
        return out

    def __call__(self, samples: list[Sample], seq_len: int) -> torch.Tensor:
        R = torch.zeros(len(samples), seq_len, dtype=torch.float32)
        ans = self._answers(samples)

        # One concurrent pass over every distinct (answer, gold) pair. Pairs that
        # fail all retries are simply missing from the returned map.
        pairs = [(a, s.gold) for a, s in zip(ans, samples)
                 if a is not None and s.gold]
        verdicts = self.judge.equivalent_many(pairs) if pairs else {}

        n_api = n_fb = 0
        for i, (s, a) in enumerate(zip(samples, ans)):
            if a is None or not s.gold:
                continue                      # no box / no gold: nothing to judge
            v = verdicts.get((a, s.gold))
            if v is None:
                # API unavailable or failed: keep the local rule, do not score 0
                v = bool(equal(a, s.gold))
                n_fb += 1
            else:
                n_api += 1
            R[i, s.done] += self.w_final * float(bool(v))

        self._last = {"n": len(samples),
                      "correct": int((R.sum(dim=1) > 0).sum()),
                      "boxed": sum(1 for a in ans if a is not None),
                      "api": n_api, "fallback": n_fb,
                      "errors": self.judge.errors}
        return R

    def summary(self, samples, rewards) -> dict:
        d = dict(self._last)
        d["correct_rate"] = round(d["correct"] / d["n"], 4) if d["n"] else 0.0
        d["judge_available"] = self.judge.available
        return d


REWARD = ResultReward()


# --------------------------------------------------------------------- self test
def _self_test() -> None:
    B = "\\" + "boxed{"

    def s(val=None, gold="7", T=12, raw=None):
        texts = [raw if raw is not None else (B + val + "}" if val is not None
                                              else "no answer")]
        return Sample(gold=gold, texts=texts, seg_end=[T - 1], seg_ok=[None],
                      done=T - 1, length=T), T

    # A stub judge: no network in the self-test. Records what it was asked.
    class _Stub:
        def __init__(self, verdicts=None, available=True):
            self.verdicts = verdicts or {}
            self.available = available
            self.errors = 0
            self.asked = []
            self.calls = 0

        def equivalent_many(self, pairs):
            self.asked = list(pairs)
            return {k: v for k, v in self.verdicts.items() if k in set(pairs)}

    def mk(verdicts=None, available=True):
        r = ResultReward()
        r.judge = _Stub(verdicts, available)
        return r

    T = 12
    # API says YES -> 1, API says NO -> 0, regardless of local rule
    r = mk({("0.5", "\\frac{1}{2}"): True, ("9", "7"): False})
    a, _ = s("0.5", "\\frac{1}{2}")
    b, _ = s("9", "7")
    R = r([a, b], T)
    assert [float(x) for x in R.sum(dim=1)] == [1.0, 0.0]
    assert r.summary([a, b], R)["api"] == 2

    # API overrides the local rule in BOTH directions
    r = mk({("7", "7"): False})                # identical strings, API says NO
    c, _ = s("7", "7")
    assert float(r([c], T).sum()) == 0.0, "API must be the primary judge"
    r = mk({("weird", "7"): True})             # local rule would say NO
    d, _ = s("weird", "7")
    assert float(r([d], T).sum()) == 1.0

    # API failure -> local rule, NOT zero (a network blip must not fake a wrong answer)
    r = mk({}, available=False)
    e, _ = s("0.5", "\\frac{1}{2}")            # locally equivalent -> 1
    f, _ = s("9", "7")                         # locally wrong -> 0
    R = r([e, f], T)
    assert [float(x) for x in R.sum(dim=1)] == [1.0, 0.0]
    summ = r.summary([e, f], R)
    assert summ["fallback"] == 2 and summ["api"] == 0, summ

    # no box: 0 and NO api call
    r = mk({})
    g, _ = s(raw="the answer is 7")
    assert float(r([g], T).sum()) == 0.0
    assert r.judge.asked == [], "an unboxed answer must not be sent"

    # tools are irrelevant: crashed program, right value -> 1
    r = mk({("7", "7"): True})
    h, _ = s("7")
    h.seg_ok = [False]
    assert float(r([h], T).sum()) == 1.0

    # all credit on the final token only
    r = mk({("7", "7"): True})
    k, _ = s("7")
    Rk = r([k], T)
    assert Rk[0].nonzero().flatten().tolist() == [k.done]

    # dedup: identical pairs are asked once (LLMJudge does this; verify the wiring
    # sends one pair per sample and relies on the judge to dedupe)
    r = mk({("7", "7"): True})
    m1, _ = s("7")
    m2, _ = s("7")
    R = r([m1, m2], T)
    assert [float(x) for x in R.sum(dim=1)] == [1.0, 1.0]

    # value-equivalence is the intent: these differ in form but not in value
    from zero_ppo.util.judge import equal as local_equal
    for pred, gold in [("0.5", "\\frac{1}{2}"), ("2", "\\sqrt{4}"),
                       ("\\left(1,2\\right)", "(1,2)"), ("7.0", "7")]:
        assert local_equal(pred, gold) is True, (pred, gold)

    # GRPO wiring: mixed group gives +-1, all-equal group gives 0
    from zero_ppo.algo.advantages import group_relative
    mask = torch.zeros(4, T, dtype=torch.bool)
    mask[:, 3:] = True
    r = mk({("7", "7"): True, ("9", "7"): False})
    s1, _ = s("7")
    s2, _ = s("7")
    s3, _ = s("9")
    s4, _ = s("9")
    Rg = r([s1, s2, s3, s4], T)
    adv = group_relative(Rg, mask, 4)
    got = [round(float(adv[i][mask[0]].mean()), 3) for i in range(4)]
    assert got == [1.0, 1.0, -1.0, -1.0], got
    r = mk({("9", "7"): False})
    Ra = r([s3, s4, s3, s4], T)
    assert float(group_relative(Ra, mask, 4).abs().max()) == 0.0, \
        "an all-wrong group must produce no gradient"

    print("  reward_result self-test OK")
    print("    API is the primary judge (overrides the local rule both ways)")
    print("    value-equivalent forms score 1: 0.5 == \\frac{1}{2}, 2 == \\sqrt{4}")
    print("    API failure falls back to the local rule, never to 0")
    print("    no box -> 0 with no API call; tools and process are not scored")
    print("    credit only on the final token; GRPO mixed group +-1, all-equal 0")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()