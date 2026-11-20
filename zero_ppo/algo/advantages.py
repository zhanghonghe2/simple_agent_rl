#!/usr/bin/env python3
"""Advantage estimators: group-relative (GRPO) and generalised (PPO).

Both are pure functions over rewards, values and the mask, so both are tested here
rather than on a GPU. Nothing here knows what a reward MEANS: where credit sits,
how a tool call is scored and what counts as a correct answer all live in the
reward file. This module only turns a (batch, seq) reward tensor into advantages.

The two estimators consume the reward differently, which is why they pair with
different reward files:

  * `group_relative` reduces a trajectory to one scalar and broadcasts it. Every
    token gets the same advantage, so credit placement inside the trajectory is
    irrelevant - a GRPO reward file puts everything on the last token.
  * `gae` keeps the per-token structure and propagates credit backwards with
    (gamma*lambda)^k. Placement matters here, so a PPO reward file may score
    individual turns.

    python3 -m zero_ppo.algo.advantages --self-test
"""
from __future__ import annotations

import argparse

import torch


def group_relative(rewards: torch.Tensor, mask: torch.Tensor,
                   group_size: int, eps: float = 1e-4) -> torch.Tensor:
    """One scalar per trajectory: (R - group mean) / group std, then broadcast.

    R is the trajectory's TOTAL reward over the generated positions. A reward file
    may put process credit at each tool turn's end and the outcome at the last
    token, so reading only the final position silently dropped the process part -
    and zeroed the advantage whenever outcomes tied but process scores differed.
    Summing over the mask also matches what `statistics` reports as the reward.

    A group whose members all scored the same is zeroed rather than divided by a
    near-zero spread, which would otherwise turn rounding into a huge advantage.
    """
    B, T = rewards.shape
    if B % group_size:
        raise ValueError(f"batch {B} is not a multiple of group {group_size}")
    per_row = (rewards * mask).sum(dim=1)                               # (B,)
    g = per_row.view(-1, group_size)
    mu = g.mean(dim=1, keepdim=True)
    sd = g.std(dim=1, keepdim=True, unbiased=False)
    a = torch.where(sd > eps, (g - mu) / sd, torch.zeros_like(g))
    adv = a.reshape(-1, 1).expand(B, T).contiguous()
    return torch.where(mask, adv, torch.zeros_like(adv))


def whiten(adv: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8,
           center: bool = False, signal_ref: float = 0.5) -> torch.Tensor:
    """Scale advantages to unit spread, then restore weak-signal attenuation.

    Scaling is needed because otherwise the step size is whatever the reward
    scale happens to be: raising the cap from 2.0 to 3.0 grew the update by half.

    Centering is NOT safe here, which is why it is off by default. On a batch
    where every trajectory failed, the critic still expects a positive return, so
    delta = gamma*V_next - V is negative everywhere: the advantages are uniformly
    negative and mean "push all of these actions down". Subtracting the mean turns
    that into half positive / half negative, promoting whichever positions the
    critic happened to mispredict - a signal unrelated to being correct. Measured
    on a failed batch: 400 negative advantages became 240 positive after centering.
    Dividing by the spread keeps every sign, so the failure still reads as failure.

    But dividing by std ALSO erases signal strength, and that collapsed two runs.
    A GAE batch's std is dominated by the credit-assignment POSITION structure (the
    last token carries the outcome, earlier tokens decay by (gamma*lambda)^k), not
    by how much real signal exists: even a perfect critic leaves std ~ 0.2, and std
    scales linearly with the reward - so a std floor would never fire. Measured on
    the overfit run: an all-wrong batch with the critic expecting 0.3 had raw |A|
    0.054 (correctly weak) but whitened |A| 0.737, the same magnitude as a strong
    batch: a 13.6x amplification of what is mostly critic error. The policy then
    trains on that noise at full strength until reward hits 0.

    Fix: multiply by min(1, mean|A_raw| / signal_ref). Strong batches (raw |A| at or
    above signal_ref) are untouched, so scale-invariance still holds where it
    matters; weak batches keep a proportionally weak gradient instead of being
    promoted to full strength. The factor is non-negative, so no sign ever flips.
    signal_ref=0 disables the attenuation (previous behaviour, for ablations).

    `center=True` restores textbook whitening for ablations. Statistics use masked
    positions only, so padding cannot shift them.
    """
    m = mask.bool()
    sel = adv[m]
    if sel.numel() < 2:
        return torch.where(m, adv, torch.zeros_like(adv))
    sd = sel.std(unbiased=False)
    if not torch.isfinite(sd) or sd < eps:
        # No spread: nothing to normalise. Keep the level (and its sign) intact -
        # a uniformly negative batch must stay uniformly negative. No attenuation
        # needed: without division the magnitude already tracks the raw signal.
        out = adv - sel.mean() if center else adv
        return torch.where(m, out, torch.zeros_like(adv))
    out = (adv - sel.mean()) / (sd + eps) if center else adv / (sd + eps)
    if signal_ref > 0:
        strength = sel.abs().mean()
        out = out * torch.clamp(strength / signal_ref, max=1.0)
    return torch.where(m, out, torch.zeros_like(adv))


def returns(adv: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """The critic's regression target: G = A + V.

    GAE gives A = G_lambda - V, so adding V back recovers the lambda-return the
    critic should predict. Computed from the pre-update values, which is why the
    caller must detach it before the critic's own backward.
    """
    return adv + values


def gae(rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor,
        gamma: float = 1.0, lam: float = 0.95) -> torch.Tensor:
    """Per-token advantages, walking backwards over each row.

    `mask` marks the positions the model generated; anything else contributes no
    advantage. The value at a masked position is the critic's estimate there.
    """
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    running = torch.zeros(B, device=rewards.device)
    for t in range(T - 1, -1, -1):
        m = mask[:, t]
        v_t = values[:, t]
        v_next = values[:, t + 1] if t + 1 < T else torch.zeros_like(v_t)
        delta = rewards[:, t] + gamma * v_next - v_t
        running = delta + gamma * lam * running
        adv[:, t] = torch.where(m, running, torch.zeros_like(running))
        running = torch.where(m, running, torch.zeros_like(running))
    return adv


def _self_test() -> None:
    # GRPO: a mixed group of four becomes +-1, an all-equal group becomes zero
    mask = torch.zeros(8, 4, dtype=torch.bool)
    mask[:, 2:] = True
    r = torch.zeros(8, 4)
    r[0:4, 3] = torch.tensor([1.0, 0.0, 0.0, 1.0])
    r[4:8, 3] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    a = group_relative(r, mask, 4)
    got = a[0:4, 2].tolist()
    assert all(abs(x - y) < 1e-5 for x, y in zip(got, [1.0, -1.0, -1.0, 1.0])), got
    assert a[4:8].abs().max().item() == 0.0, "an all-equal group must not produce advantage"
    assert a[~mask].abs().max().item() == 0.0, "advantage outside the mask"

    # PPO: one reward at the end, no critic, gamma=lambda=1 -> flat advantage
    m2 = torch.zeros(1, 4, dtype=torch.bool)
    m2[0, 1:] = True
    rew = torch.zeros(1, 4)
    rew[0, 3] = 1.0
    vals = torch.zeros(1, 4)
    g = gae(rew, vals, m2, gamma=1.0, lam=1.0)
    assert all(abs(x - 1.0) < 1e-6 for x in g[0, 1:].tolist()), g.tolist()
    assert g[0, 0].item() == 0.0

    # with lambda < 1 the far tokens are discounted, monotonically
    g2 = gae(rew, vals, m2, gamma=1.0, lam=0.9)
    assert 0 < g2[0, 1].item() < g2[0, 2].item() < g2[0, 3].item()

    # a perfect critic predicts the return FROM each position, which with gamma=1 is
    # one everywhere after the reward, not the reward itself (non-zero only at the end)
    perfect = torch.zeros(1, 4)
    perfect[0, 1:] = 1.0
    g3 = gae(rew, perfect, m2, gamma=1.0, lam=1.0)
    assert g3.abs().max().item() < 1e-6, g3.tolist()

    try:
        group_relative(r[:5], mask[:5], 4)
    except ValueError:
        pass
    else:
        raise AssertionError("a ragged batch was accepted")

    # process credit sits at a tool turn's end, not at the last token. Reading only
    # the final position dropped it, and tied outcomes then gave zero advantage.
    mp = torch.zeros(2, 8, dtype=torch.bool)
    mp[:, 2:] = True
    rp = torch.zeros(2, 8)
    rp[0, 3] = 1.0                      # used the tool and it ran
    rp[:, 7] = 1.0                      # both answers correct
    ap = group_relative(rp, mp, 2)
    assert ap[0, 2].item() > 0 > ap[1, 2].item(), \
        "process reward must separate two trajectories whose outcomes tie"
    # the scalar is the trajectory total, and padding never contributes
    rp2 = torch.zeros(2, 8)
    rp2[0, 3], rp2[0, 7] = 0.5, 1.0
    rp2[1, 3], rp2[1, 7] = 0.5, 0.0
    rp2[:, 0] = 99.0                    # outside the mask: must be ignored
    ap2 = group_relative(rp2, mp, 2)
    assert abs(ap2[0, 2].item() - 1.0) < 1e-5, ap2[0, 2].item()
    assert abs(ap2[1, 2].item() + 1.0) < 1e-5, ap2[1, 2].item()

    # scaling decouples the step size from the reward scale IN THE STRONG RANGE
    # (raw |A| >= signal_ref): there the attenuation factor is exactly 1. Below
    # signal_ref weak batches are deliberately attenuated - see the next block.
    mw = torch.zeros(4, 12, dtype=torch.bool)
    mw[:, 3:] = True
    means = []
    for scale in (2.0, 3.0, 10.0):
        rw = torch.zeros(4, 12)
        rw[:2, -1] = scale
        aw = gae(rw, torch.zeros(4, 12), mw, gamma=1.0, lam=0.95)
        assert float(aw[mw].abs().mean()) >= 0.5, "fixture must be in the strong range"
        w = whiten(aw, mw)
        means.append(float(w[mw].abs().mean()))
        assert float(w[~mw].abs().max()) == 0.0, "no advantage outside the mask"
        # signs must survive: scaling may not flip a single position
        assert bool(((aw[mw] > 0) == (w[mw] > 0)).all()), "sign flipped"
    assert max(means) - min(means) < 1e-5, f"scale leaked into the step: {means}"

    # THE regression for the overfit collapse: an all-wrong batch whose critic
    # still expects a small positive return has a WEAK raw advantage (mostly
    # critic error). Plain whitening promoted it to the same |A| as a strong
    # batch (0.054 -> 0.737, 13.6x), and the policy trained on that noise until
    # reward hit 0. The attenuation must keep weak signals weak, signs intact,
    # and be exactly identity for strong signals and when disabled.
    vf2 = torch.full((4, 12), 0.05)
    af2 = gae(torch.zeros(4, 12), vf2, mw, gamma=1.0, lam=0.95)
    raw_mean = float(af2[mw].abs().mean())
    assert raw_mean < 0.5, "fixture must be in the weak range"
    w_plain = whiten(af2, mw, signal_ref=0.0)
    w_att = whiten(af2, mw)
    assert float(w_att[mw].abs().mean()) < float(w_plain[mw].abs().mean()), \
        "weak batch must be attenuated, not promoted"
    assert abs(float(w_att[mw].abs().mean())
               - raw_mean / float(af2[mw].std(unbiased=False)) * raw_mean / 0.5) < 1e-5
    assert bool((w_att[mw] < 0).all()), "attenuation must not flip signs"
    # strong batch: attenuation factor is exactly 1
    rs = torch.zeros(4, 12)
    rs[:2, -1] = 3.0
    as_ = gae(rs, torch.zeros(4, 12), mw, gamma=1.0, lam=0.95)
    assert float(as_[mw].abs().mean()) >= 0.5
    assert torch.equal(whiten(as_, mw), whiten(as_, mw, signal_ref=0.0))
    # disabled: exact previous behaviour on the weak batch too
    assert torch.equal(w_plain, whiten(af2, mw, signal_ref=0.0))

    # THE regression this guards: a failed batch has uniformly negative
    # advantages ("push all of these down"). Centering would promote half of them.
    vf = torch.full((4, 12), 0.3)
    af = gae(torch.zeros(4, 12), vf, mw, gamma=1.0, lam=0.95)
    assert bool((af[mw] < 0).all()), "fixture should be uniformly negative"
    wf = whiten(af, mw)
    assert bool((wf[mw] < 0).all()), "scaling must keep a failed batch negative"
    wc = whiten(af, mw, center=True)
    assert bool((wc[mw] > 0).any()), "centering does promote some - that is the bug"
    assert abs(float(wc[mw].mean())) < 1e-5

    # degenerate inputs stay as they are; no spread means nothing to normalise
    assert float(whiten(torch.zeros(4, 12), mw).abs().max()) == 0.0
    const_neg = torch.full((4, 12), -2.0)
    assert bool((whiten(const_neg, mw)[mw] < 0).all()), "constant sign preserved"

    print("  advantages self-test OK")
    print("    grpo: mixed group -> +-1, all-equal -> 0, outside mask -> 0")
    print("    grpo: total reward over the mask, so process credit counts")
    print("    whiten: scales only, keeps signs; a failed batch stays all-negative")
    print("    ppo : flat advantage with gamma=lambda=1, decays with lambda<1, "
          "zero with a perfect critic")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()
