#!/usr/bin/env python3
"""The actor: one copy of the policy on one GPU, with an optional value head.

Deliberately free of orchestration. It does not know about DeepSpeed, about the
inference engine, or about rewards; the engine wraps it, syncs its weights and
sleeps the engine around it. The only interface that matters downstream is
`token_logprobs`, because that is what every loss and every ratio consumes.

Two rules this file exists to enforce, both learned the hard way:

  * log-probabilities are gathered only at the masked positions, so the
    vocabulary-sized tensor is (n_masked, V) instead of (batch, seq, V). The two
    are exactly equal, because a linear layer acts on rows independently.
  * the logit at position t comes from the hidden state *before* token t was
    consumed, so the hidden states are shifted right by one. Getting this wrong
    shifts every advantage by one token and raises nothing.

    python3 -m zero_ppo.actor --self-test
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Actor(nn.Module):
    """Policy on a device. Trainable parameters are exposed separately so the
    engine can hand exactly those to the optimiser."""

    def __init__(self, backbone: nn.Module, lm_head: nn.Module, pad_id: int = 0,
                 value_head: bool = False, hidden: int | None = None,
                 freeze_backbone: bool = False, amp_dtype: str | None = None):
        super().__init__()
        self.backbone = backbone
        self.lm_head = lm_head
        self.pad_id = pad_id
        self.value_head = (nn.Linear(hidden, 1) if value_head else None)
        # fp16 for Volta, bf16 for Ampere and later, None for full fp32. The
        # hidden states and logits are cast back to fp32 regardless, because the
        # losses downstream assume fp32.
        self.amp_dtype = amp_dtype
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------ utils
    def _hidden_before(self, input_ids, attn):
        if self.amp_dtype is None:
            out = self.backbone(input_ids=input_ids, attention_mask=attn)
        else:
            dt = {"float16": torch.float16, "bfloat16": torch.bfloat16}[self.amp_dtype]
            with torch.autocast(device_type=input_ids.device.type, dtype=dt):
                out = self.backbone(input_ids=input_ids, attention_mask=attn)
        h = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out).float()
        prev = torch.zeros_like(h)
        prev[:, 1:] = h[:, :-1]          # position t sees tokens 0..t-1
        return prev

    def token_logprobs(self, input_ids, attn, mask=None, no_grad: bool = True):
        """(B,T) log-probability of the token actually at each position.

        With `mask`, only those positions are computed; the result is zero
        everywhere else, so callers must treat unmasked entries as invalid rather
        than as probability 1.
        """
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            prev = self._hidden_before(input_ids, attn)
            # Hidden states come back in fp32 for the losses, but the head holds the
            # model's own dtype; the projection needs them to agree.
            wd = self.lm_head.weight.dtype
            if mask is None:
                logits = self.lm_head(prev.to(wd))                 # (B,T,V)
                return -torch.nn.functional.cross_entropy(
                    logits.transpose(1, 2), input_ids, reduction="none").float()
            sel = prev[mask].to(wd)                                # (N,H)
            logits = self.lm_head(sel)                             # (N,V)  <- memory
            tgt = input_ids[mask]                                  # (N,)
            # cross entropy is fused: log_softmax would materialise a second (N,V)
            # float tensor and its gradient, which is tens of gigabytes here.
            picked = -torch.nn.functional.cross_entropy(
                logits, tgt, reduction="none").float()
            out = torch.zeros(input_ids.shape, dtype=torch.float32,
                              device=input_ids.device)
            out[mask] = picked
            return out

    def values(self, input_ids, attn):
        if self.value_head is None:
            return None
        prev = self._hidden_before(input_ids, attn)
        wd = self.value_head.weight.dtype
        return self.value_head(prev.to(wd)).squeeze(-1).float()

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def export_weights(self, dtype=torch.bfloat16):
        """Full weights, not sharded: with ZeRO-2 every rank holds a complete copy,
        which is what makes pushing them into the inference engine cheap."""
        sd = self.state_dict()
        return {k: v.detach().to(dtype).cpu() for k, v in sd.items()}


# --------------------------------------------------------------------- self test
class _TinyBackbone(nn.Module):
    def __init__(self, vocab: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, input_ids, attention_mask=None):
        class Out:
            pass
        o = Out()
        o.last_hidden_state = torch.tanh(self.proj(self.emb(input_ids)))
        return o


def _self_test() -> None:
    torch.manual_seed(0)
    V, H, B, T = 17, 8, 2, 6
    actor = Actor(_TinyBackbone(V, H), nn.Linear(H, V, bias=False),
                  value_head=True, hidden=H)
    ids = torch.randint(1, V, (B, T))
    attn = torch.ones_like(ids)
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, 2:5] = True
    mask[1, 3:4] = True

    full = actor.token_logprobs(ids, attn, mask=None)
    part = actor.token_logprobs(ids, attn, mask=mask)
    assert torch.allclose(full[mask], part[mask], atol=1e-6), "masked gather differs"
    assert part[~mask].abs().max().item() == 0.0, "unmasked entries must be zero"

    # the shift really is a shift: logp at t must depend on token t-1
    one = ids.clone()
    one[0, 3] = (one[0, 3] + 1) % V          # change token 3
    moved = actor.token_logprobs(one, attn, mask=mask)
    assert not torch.allclose(moved[0, 4], part[0, 4]), "logit at t+1 did not move"

    # a half precision head must work too: the real one is loaded in the model dtype
    half = Actor(_TinyBackbone(V, H), nn.Linear(H, V, bias=False).to(torch.bfloat16))
    lp_half = half.token_logprobs(ids, attn, mask=mask)
    assert torch.isfinite(lp_half[mask]).all()

    vals = actor.values(ids, attn)
    assert vals.shape == (B, T) and torch.isfinite(vals).all()
    w = actor.export_weights()
    assert all(t.dtype == torch.bfloat16 for t in w.values())
    n_train = sum(p.numel() for p in actor.trainable())
    for amp in (None, "bfloat16"):
        a2 = Actor(_TinyBackbone(V, H), nn.Linear(H, V, bias=False), amp_dtype=amp)
        lp2 = a2.token_logprobs(ids, attn, mask=mask)
        assert lp2.shape == full.shape and torch.isfinite(lp2[mask]).all()

    print("  actor self-test OK")
    print(f"    masked-gather == full-gather on {int(mask.sum())} positions")
    print(f"    zeros outside mask, shift verified by perturbing token t")
    print(f"    trainable params {n_train}, export dtype bf16, amp None/bf16 both ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()
