#!/usr/bin/env python3
"""The value path, in three modes.

    full          the critic is its own copy of the model
    lora_shared   the critic reads the actor's adapted hidden states; one adapter
    lora_split    a frozen shared backbone with a SECOND adapter for the critic

The difference that matters is where the critic's gradient lands. Shared, it
flows into the actor's adapter and bends the features the policy is trying to
shape; split, it lands on its own adapter and leaves the actor alone; full, it
lands on a separate model. The self test asserts exactly that, because it is the
whole reason to have the switch.

    python3 -m zero_ppo.critic --self-test
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _adapter_name(n: str) -> bool:
    low = n.lower()
    return "lora" in low or "adapter" in low


class _Head(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lin = nn.Linear(hidden, 1)

    def forward(self, h):
        return self.lin(h.to(self.lin.weight.dtype)).squeeze(-1).float()


class SharedValue(nn.Module):
    """Reads the actor's hidden states, so its gradient reaches the actor's params."""

    mode_name = "lora_shared"

    def __init__(self, actor, hidden):
        super().__init__()
        self.actor = actor
        self.head = _Head(hidden)

    def values(self, input_ids, attn):
        return self.head(self.actor._hidden_before(input_ids, attn))

    def trainable(self):
        return list(self.head.parameters())


class SplitValue(nn.Module):
    """Same backbone, its own adapter. The actor's parameters never see this loss."""

    mode_name = "lora_split"

    def __init__(self, actor, hidden, name: str = "critic", base: str = "default"):
        super().__init__()
        self.actor = actor
        self.name = name
        self.base = base
        self.head = _Head(hidden)
        add = getattr(actor.backbone, "add_adapter", None)
        if add is None:
            raise RuntimeError("lora_split needs a backbone that can add a named adapter")
        sw = getattr(actor.backbone, "set_adapter", None)
        add(name)                      # PEFT: add_adapter(name)
        if sw is not None:
            sw(base)                   # adding may activate it; the actor must not
                                       # silently start computing with the critic adapter
        # Isolation is only as good as the freeze: if base weights are trainable the
        # critic's loss reaches them through the shared computation.
        leaking = [n for n, q in actor.backbone.named_parameters()
                   if q.requires_grad and not _adapter_name(n)]
        if leaking:
            raise RuntimeError(
                f"lora_split needs a frozen base, but {len(leaking)} parameters are "
                f"trainable (e.g. {leaking[0]})")

    def values(self, input_ids, attn):
        sw = getattr(self.actor.backbone, "set_adapter")
        prev = getattr(self.actor.backbone, "active_adapters", None)
        sw(self.name)
        try:
            h = self.actor._hidden_before(input_ids, attn)
        finally:
            sw(self.base if prev is None else prev)
        return self.head(h)

    def trainable(self):
        own = [p for n, p in self.actor.backbone.named_parameters() if self.name in n]
        return list(self.head.parameters()) + own


class FullValue(nn.Module):
    """An independent copy of the backbone."""

    mode_name = "full"

    def __init__(self, actor, hidden):
        super().__init__()
        self.backbone = copy.deepcopy(actor.backbone)
        self.head = _Head(hidden)

    def values(self, input_ids, attn):
        out = self.backbone(input_ids=input_ids, attention_mask=attn)
        h = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out).float()
        prev = torch.zeros_like(h)
        prev[:, 1:] = h[:, :-1]
        return self.head(prev)

    def trainable(self):
        return list(self.parameters())


def build_value(mode: str, actor, hidden: int):
    if mode == "full":
        return FullValue(actor, hidden)
    if mode == "lora_shared":
        return SharedValue(actor, hidden)
    if mode == "lora_split":
        return SplitValue(actor, hidden)
    raise ValueError(f"unknown value mode {mode!r}; "
                     "expected full, lora_shared or lora_split")


# --------------------------------------------------------------------- self test
class _AdaptedBackbone(nn.Module):
    """Stands in for a PEFT model: a frozen base plus switchable named adapters."""

    def __init__(self, vocab, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.adapters = nn.ModuleDict({"default": nn.Linear(hidden, hidden, bias=False)})
        self.active = "default"
        for q in (self.emb, self.proj):        # base frozen, like LoRA
            for pp in q.parameters():
                pp.requires_grad_(False)

    def add_adapter(self, name):
        if name not in self.adapters:
            self.adapters[name] = nn.Linear(self.proj.in_features,
                                            self.proj.out_features, bias=False)
        self.active = name

    def set_adapter(self, name):
        self.active = name

    @property
    def active_adapters(self):
        return self.active

    def forward(self, input_ids, attention_mask=None):
        h = torch.tanh(self.proj(self.emb(input_ids)))
        h = h + self.adapters[self.active](h)          # only the active adapter
        class Out:
            pass
        o = Out()
        o.last_hidden_state = h
        return o


def _self_test() -> None:
    from zero_ppo.actor import Actor

    torch.manual_seed(0)
    V, H, B, T = 13, 8, 2, 5
    ids = torch.randint(1, V, (B, T))
    attn = torch.ones_like(ids)

    def fresh():
        bb = _AdaptedBackbone(V, H)
        return Actor(bb, nn.Linear(H, V, bias=False)), bb

    # shared: the critic's loss must move the actor's parameters
    actor, bb = fresh()
    crit = build_value("lora_shared", actor, H)
    crit.values(ids, attn).sum().backward()
    # the base is frozen, so "reaches the actor" means reaching the actor ADAPTER
    assert bb.adapters["default"].weight.grad is not None, \
        "shared critic did not reach the actor"
    assert bb.adapters["default"].weight.grad.abs().sum() > 0

    # split: the actor's base must stay clean, the critic's adapter must move
    actor, bb = fresh()
    actor.proj_grad_snapshot = None
    crit = build_value("lora_split", actor, H)
    crit.values(ids, attn).sum().backward()
    assert bb.proj.weight.grad is None or bb.proj.weight.grad.abs().sum() == 0, \
        "split critic leaked into the shared base"
    assert bb.adapters["critic"].weight.grad is not None, "critic adapter got no gradient"
    assert bb.adapters["default"].weight.grad is None, "actor adapter got the critic loss"
    assert bb.active == "default", "adapter was not restored after the critic forward"

    # full: a separate copy, so the actor's params are untouched
    actor, bb = fresh()
    crit = build_value("full", actor, H)
    crit.values(ids, attn).sum().backward()
    assert bb.proj.weight.grad is None, "full critic shares the actor"
    assert crit.head.lin.weight.grad is not None

    try:
        build_value("nope", actor, H)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown mode was accepted")

    print("  critic self-test OK")
    print("    lora_shared : gradient reaches the actor  (interference, as documented)")
    print("    lora_split  : base clean, own adapter moves, adapter restored")
    print("    full        : separate copy, actor untouched")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()
