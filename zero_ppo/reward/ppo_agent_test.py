#!/usr/bin/env python3
"""Process reward for PPO: per-turn tool credit spread across the turn's tokens.

This is the reward side of the PPO path. Everything about what a reward MEANS lives
here; `algo/advantages.py` only propagates whatever this file returns.

    outcome   `w_final` at the last generated token when the boxed answer is right.
    process   `w_proc` split over the tool calls, and each call's share spread
              evenly over the tokens OF THAT TURN. A turn that called the tool and
              whose program ran earns its share; a turn that only reasoned earns
              none.
    no tool   the process budget is paid at the final position when the answer is
              right, so reasoning alone and a working program are worth the same.

Why spread instead of a pulse at the turn's end: GAE propagates credit backwards
by (gamma*lambda)^k, so with lambda=0.95 the effective reach is 1/(1-0.95) = 20
tokens. A turn is often 50-150 tokens, so an end-of-turn pulse only trains its
tail - measured, the first token of a 100-token turn received 5.9e-3 of the credit
that spreading gives it (a 33x difference). Spreading makes the whole turn's
reasoning accountable. This mirrors the finding in ppo_llm/GAE_PSEUDOCODE.md.

The program only has to RUN, not be relevant to the problem: judging the maths of
a program would need a model, and this reward deliberately needs none.

    python3 -m zero_ppo.reward.ppo_agent_test --self-test
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
from zero_ppo.util.judge import extract_answer


_warned = {"v": False}


def _judge(text: str | None, gold: str) -> bool:
    """Never fail silently: a judge that errors would score every sample zero and
    look exactly like a model that learned nothing."""
    if not text or not gold:
        return False
    try:
        from zero_ppo.util.judge import equal
        return bool(equal(extract_answer(text), gold))
    except Exception as e:
        if not _warned["v"]:
            _warned["v"] = True
            print(f"  ! judge unavailable: {type(e).__name__}: {e}", flush=True)
        return False


class PPOReward(Reward):
    name = "ppo_agent_test"

    def __init__(self, w_final: float = 1.0, w_proc: float = 1.0,
                 spread: bool = True):
        self.w_final = float(w_final)
        self.w_proc = float(w_proc)
        self.spread = bool(spread)

    def _span_of(self, s: Sample, k: int) -> tuple[int, int]:
        """The token range of turn k: from the previous turn's end to this one."""
        end = s.seg_end[k]
        start = (s.seg_end[k - 1] + 1) if k > 0 else 0
        if s.mask is not None:
            # never pay for a prompt or a tool-output token: only generated ones
            idx = s.mask.nonzero().flatten()
            if idx.numel():
                start = max(start, int(idx[0]))
        return start, end

    def __call__(self, samples: list[Sample], seq_len: int) -> torch.Tensor:
        R = torch.zeros(len(samples), seq_len, dtype=torch.float32)
        for i, s in enumerate(samples):
            correct = _judge(final_answer_text(s.texts), s.gold)
            calls = [(k, ok) for k, ok in enumerate(s.seg_ok) if ok is not None]

            if not calls:
                # no tool used: the process budget rides on the outcome
                R[i, s.done] += self.w_final * float(correct)
                if correct:
                    R[i, s.done] += self.w_proc
                continue

            per = self.w_proc / len(calls)
            for k, ok in calls:
                if not ok:
                    continue                      # a failed run earns nothing
                a, b = self._span_of(s, k)
                if self.spread and b >= a:
                    R[i, a:b + 1] += per / (b - a + 1)
                else:
                    R[i, b] += per                # pulse, for comparison runs
            R[i, s.done] += self.w_final * float(correct)
        return R

    def summary(self, samples, rewards) -> dict:
        return {"n": len(samples)}


REWARD = PPOReward()


# --------------------------------------------------------------------- self test
def _self_test() -> None:
    BOX = "\\" + "boxed{7}"
    WRONG = "\\" + "boxed{9}"

    def s(texts, oks, gold="7", ends=None, T=40):
        seg_end = ends if ends is not None else [9 + 10 * k for k in range(len(oks))]
        done = seg_end[-1] if seg_end else T - 1
        mask = torch.zeros(T, dtype=torch.bool)
        mask[0:done + 1] = True                  # everything up to done is generated
        return Sample(gold=gold, texts=texts, seg_end=seg_end, seg_ok=oks,
                      done=done, length=T, mask=mask), T

    r = PPOReward()
    cases = []

    # no tool, right -> outcome + full process at the end
    sm, T = s(["reasoning " + BOX], [])
    R = r([sm], T)
    cases.append(("no tool + right", float(R.sum()), 2.0))
    assert R[0, sm.done].item() == 2.0

    # no tool, wrong -> nothing
    sm, T = s(["reasoning " + WRONG], [])
    cases.append(("no tool + wrong", float(r([sm], T).sum()), 0.0))

    # three calls, two ran, right -> 2/3 process + 1 outcome
    sm, T = s(["c1", "c2", BOX], [True, False, True])
    R = r([sm], T)
    cases.append(("3 calls, 2 ok, right", float(R.sum()), 1.0 + 2.0 / 3.0))

    # the share is SPREAD over the turn, not piled on its last token
    sm, T = s(["c1", BOX], [True, None], ends=[19, 29])
    R = r([sm], T)
    turn = R[0, 0:20]
    assert torch.allclose(turn, torch.full_like(turn, 1.0 / 20)), turn.tolist()
    assert abs(float(turn.sum()) - 1.0) < 1e-6
    cases.append(("spread over 20 tokens", float(turn[0]), 1.0 / 20))

    # pulse mode keeps the old shape, for ablations
    rp = PPOReward(spread=False)
    R2 = rp([sm], T)
    assert R2[0, 19].item() == 1.0 and float(R2[0, 0:19].sum()) == 0.0
    cases.append(("pulse at turn end", float(R2[0, 19]), 1.0))

    # a failed run earns nothing but the outcome still counts
    sm, T = s(["c1", BOX], [False, None])
    cases.append(("1 call failed, right", float(r([sm], T).sum()), 1.0))

    # tool used, wrong answer -> the process share still counts
    sm, T = s(["c1", WRONG], [True, None])
    cases.append(("1 call ok, wrong", float(r([sm], T).sum()), 1.0))

    for label, got, want in cases:
        assert abs(got - want) < 1e-6, f"{label}: got {got}, want {want}"

    # no reward may land outside the generated mask
    sm, T = s(["c1", BOX], [True, None], ends=[19, 29])
    sm.mask[0:5] = False                        # first 5 tokens are prompt
    R = r([sm], T)
    assert float(R[0, 0:5].sum()) == 0.0, "reward leaked onto prompt tokens"

    # the total is conserved when spreading: same budget, different placement
    a = float(PPOReward(spread=True)([sm], T).sum())
    b = float(PPOReward(spread=False)([sm], T).sum())
    assert abs(a - b) < 1e-6, (a, b)

    print("  ppo_agent_test self-test OK")
    for label, got, _ in cases:
        print(f"    {label:24} -> {got:.4f}")
    print("    credit spread over each turn; total conserved; mask respected")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()