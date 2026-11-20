#!/usr/bin/env python3
"""Outcome-only reward for GRPO: 1.0 on the last generated token if the answer is right.

GRPO collapses a trajectory to a single scalar and broadcasts it, so it cannot
express per-turn credit - a process share at a turn's end and the same share at the
final token produce identical advantages. Process reward therefore belongs to the
PPO path (`ppo_agent_test.py` + `--algo ppo`), where GAE keeps the per-token
structure. This file answers one question: was the final boxed answer correct?

No judge model is needed: the comparison is symbolic equality against `gold`.

    python3 -m zero_ppo.reward.agent_reward_test --self-test
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


class AgentReward(Reward):
    """Outcome-only reward, for GRPO. Everything lands on the last generated token.

    GRPO reduces a trajectory to one scalar and broadcasts it to every token, so
    per-turn credit cannot be expressed: a process share at a turn's end would be
    indistinguishable from the same share at the final token. Scoring the process
    needs the per-token structure that GAE keeps - use `ppo_agent_test.py` with
    `--algo ppo` for that. Here the only question is whether the answer is right.
    """

    name = "agent_reward_test"

    def __init__(self, w_final: float = 1.0):
        self.w_final = float(w_final)

    def __call__(self, samples: list[Sample], seq_len: int) -> torch.Tensor:
        R = torch.zeros(len(samples), seq_len, dtype=torch.float32)
        for i, s in enumerate(samples):
            correct = _judge(final_answer_text(s.texts), s.gold)
            R[i, s.done] += self.w_final * float(correct)
        return R

    def summary(self, samples, rewards) -> dict:
        return {"n": len(samples)}


REWARD = AgentReward()


# --------------------------------------------------------------------- self test
def _self_test() -> None:
    BOX = "\\" + "boxed{7}"
    WRONG = "\\" + "boxed{9}"

    def s(texts, oks, gold="7", T=12):
        seg_end = [3 + 3 * k for k in range(len(oks))]
        done = seg_end[-1] if seg_end else T - 1
        return Sample(gold=gold, texts=texts, seg_end=seg_end, seg_ok=oks,
                      done=done, length=T), T

    r = AgentReward()
    cases = []

    # outcome only: right is 1.0 wherever the tools went, wrong is 0.0
    for label, texts, oks, want in [
        ("no tool + right", ["reasoning " + BOX], [], 1.0),
        ("no tool + wrong", ["reasoning " + WRONG], [], 0.0),
        ("3 calls, 2 ok, right", ["c1", "c2", BOX], [True, False, True], 1.0),
        ("3 calls, 0 ok, right", ["c1", "c2", BOX], [False, False, False], 1.0),
        ("1 call ok, wrong", ["c1", WRONG], [True, None], 0.0),
    ]:
        sm, T = s(texts, oks)
        cases.append((label, float(r([sm], T).sum()), want))

    for label, got, want in cases:
        assert abs(got - want) < 1e-6, f"{label}: got {got}, want {want}"

    # ALL credit sits on the last generated token, nothing at a turn's end
    sm, T = s(["c1", "c2", BOX], [True, True, None])
    R = r([sm], T)
    nz = R[0].nonzero().flatten().tolist()
    assert nz == [sm.done], f"credit must be only at the final token, got {nz}"
    assert R[0, sm.seg_end[0]].item() == 0.0, "GRPO reward must not score a turn"
    assert float(R[0].sum()) == 1.0

    print("  agent_reward_test self-test OK")
    for label, got, _ in cases:
        print(f"    {label:22} -> {got:.3f}")
    print("    all credit on the final token; turns are not scored (that is PPO's job)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()
