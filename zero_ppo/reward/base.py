#!/usr/bin/env python3
"""The contract every reward file implements.

A reward sees finished trajectories and returns a (batch, seq) tensor. It never
touches padding, offsets or the model: the sample carries positions that are
already expressed in the padded tensor, so a reward file stays a dozen lines of
scoring logic and nothing else.

A segment is one assistant turn. `seg_ok` is True or False for a turn that called
the tool, and None for a turn that only reasoned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Sample:
    gold: str
    texts: list[str] = field(default_factory=list)      # assistant turns, in order
    seg_end: list[int] = field(default_factory=list)    # positions in the padded tensor
    seg_ok: list[bool | None] = field(default_factory=list)
    done: int = 0                                       # last generated position
    length: int = 0
    mask: torch.Tensor | None = None
    # What a per-segment judge needs: the problem itself, whether each turn was
    # reasoning ("nl") or a program ("code"), and what that program printed.
    # Optional so the outcome-only rewards keep working unchanged.
    question: str = ""
    seg_kinds: list[str] = field(default_factory=list)
    seg_outs: list[str | None] = field(default_factory=list)

    def seg_start(self, k: int) -> int:
        """First token of turn k in the padded tensor (previous turn's end + 1)."""
        if k <= 0:
            if self.mask is not None:
                idx = self.mask.nonzero().flatten()
                if idx.numel():
                    return int(idx[0])
            return 0
        return self.seg_end[k - 1] + 1

    def gen_positions(self, k: int):
        """The GENERATED token positions of turn k, as a 1-D index tensor.

        A turn's raw span runs into the next tool reply, which the model did not
        produce and the trainer masks out. Spreading credit over the raw span
        therefore loses part of it - and loses more on longer conversations, which
        quietly penalises the multi-turn behaviour the reward is meant to encourage.
        Returning only masked positions keeps the whole share on policy tokens.
        """
        import torch
        a, b = self.seg_start(k), self.seg_end[k]
        if b < a:
            return torch.tensor([max(0, b)], dtype=torch.long)
        idx = torch.arange(a, b + 1, dtype=torch.long)
        if self.mask is not None:
            idx = idx[self.mask[a:b + 1]]
        if idx.numel() == 0:
            return torch.tensor([max(0, b)], dtype=torch.long)
        return idx


class Reward:
    """Subclass and implement __call__. `name` is what the logs print."""

    name: str = "reward"

    def __call__(self, samples: list[Sample], seq_len: int) -> torch.Tensor:
        raise NotImplementedError

    def summary(self, samples: list[Sample], rewards: torch.Tensor) -> dict:
        return {}


def final_answer_text(texts: list[str]) -> str | None:
    """The last assistant turn that actually boxed an answer."""
    for t in reversed(texts or []):
        if "\\boxed{" in t:
            return t
    return None
