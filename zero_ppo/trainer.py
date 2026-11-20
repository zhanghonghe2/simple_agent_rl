#!/usr/bin/env python3
"""The algorithm trunk: advantages in, one optimiser step out.

It knows nothing about engines, Ray, ranks or vLLM. Everything here is a pure
function of tensors, which is why it can be tested on a laptop.

    python3 -m zero_ppo.trainer --self-test
"""
from __future__ import annotations

import argparse
import math

import torch

from zero_ppo.algo.advantages import gae, group_relative


def lr_at(step: int, total: int, base_lr: float, *, warmup_frac: float = 0.1,
          min_frac: float = 0.0) -> float:
    """Learning rate for a 0-indexed `step`: linear warmup, then cosine decay.

    The first `warmup_frac` of `total` steps ramp linearly from ~0 to `base_lr`;
    the rest decay on a cosine down to `min_frac * base_lr`. Pure arithmetic, so
    the schedule can be checked on a laptop without an optimiser.
    """
    if total <= 0 or base_lr <= 0:
        return base_lr
    warmup = max(1, int(total * warmup_frac)) if warmup_frac > 0 else 0
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if total <= warmup:
        return base_lr
    progress = (step - warmup) / (total - warmup)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_frac + (1.0 - min_frac) * cos)


def advantages(algo: str, rewards, mask, group: int, *, values=None,
               gamma: float = 1.0, lam: float = 0.95):
    if algo == "grpo":
        return group_relative(rewards, mask, group)
    if algo == "ppo":
        if values is None:
            raise ValueError("PPO needs values from the critic")
        return gae(rewards, values, mask, gamma=gamma, lam=lam)
    raise ValueError(f"unknown algo {algo!r}")


def clipped_loss(logp, old_logp, adv, mask, eps: float = 0.1):
    """The surrogate, averaged over generated tokens only.

    `old_logp` must be the policy that PRODUCED the batch, held fixed across the
    inner epochs. Passing the current logp makes the ratio identically one, which
    silently turns PPO into a plain policy gradient and pins clip_frac at zero.
    """
    ratio = torch.exp(logp - old_logp)
    un = ratio * adv
    cl = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
    per = -torch.min(un, cl)
    return (per * mask).sum() / mask.sum().clamp(min=1)


def kl_penalty(logp, ref_logp, mask, max_ratio_log: float = 5.0,
               max_kl: float = 10.0) -> torch.Tensor:
    """Mean KL(policy || reference) over generated tokens, k3 estimator, bounded.

    The clip bounds the ratio to the policy that GENERATED the batch, which is
    re-captured every rollout - so it says nothing about drift from the SFT model
    across many steps. Without an anchor the policy walked off-distribution and
    collapsed into repeated multilingual gibberish while its weights stayed
    finite. This term pulls it back toward the frozen reference.

    k3 = exp(d) - d - 1 with d = ref_logp - logp is non-negative, so the penalty
    never rewards moving away. But it is EXPONENTIAL: a handful of tokens whose
    probability collapsed to 1e-4 of the reference gives d ~ 9 and k3 ~ 2e4.
    Observed in a real run: one step reported KL 34.05 against a median of 0.086,
    so the penalty term outweighed the policy loss ~34x and that single update
    dominated the step.

    Neither guard may kill the gradient, because a saturated penalty removes the
    restoring force exactly where the policy has drifted furthest. Both obvious
    choices fail that test:

      * `d.clamp(max=c)` has EXACTLY zero slope past c (PyTorch semantics), so any
        token with d > c stops contributing to the pull altogether;
      * `tanh` underflows to zero slope in float32 once its argument exceeds ~9.

    So d is compressed logarithmically past the cap, c + log1p(d - c), whose slope
    1 / (1 + d - c) stays strictly positive for every finite d; and the mean is
    bounded by the rational squash max_kl * k / (max_kl + k), slope
    max_kl^2 / (max_kl + k)^2 > 0 likewise. Both are ~identity in the normal range,
    so ordinary steps are unchanged while more drift always means more pull.
    """
    m = mask.bool()
    if m.sum() == 0:
        return logp.sum() * 0.0
    d = (ref_logp - logp)[m]
    c = max_ratio_log
    d = torch.where(d > c, c + torch.log1p(d - c),
                    torch.where(d < -c, -c - torch.log1p(-c - d), d))
    k = (torch.exp(d) - d - 1.0).mean()
    return max_kl * k / (max_kl + k)


def ppo_diagnostics(logp, old_logp, mask, eps: float = 0.1) -> dict:
    """How far the policy moved from the one that generated the batch.

    `approx_kl` is the k3 estimator (ratio - 1 - log ratio): non-negative and lower
    variance than the naive mean log-difference. `clip_frac` is the share of tokens
    whose ratio left the trust region. Both are only meaningful with inner epochs >
    1; on the first pass they are zero by construction.
    """
    with torch.no_grad():
        logratio = (logp - old_logp)[mask]
        if logratio.numel() == 0:
            return {"approx_kl": 0.0, "clip_frac": 0.0, "kl_saturated": 0.0}
        # same exponential guard as kl_penalty: a few collapsed tokens would
        # otherwise report an astronomically large KL and hide the real trend
        sat = (logratio.abs() > 5.0).float().mean()
        lr_safe = logratio.clamp(min=-5.0, max=5.0)
        ratio = torch.exp(lr_safe)
        kl = (ratio - 1.0 - lr_safe).mean()
        # clip_frac uses the true ratio: saturation must not mask a real excursion
        true_ratio = torch.exp(logratio.clamp(min=-50.0, max=50.0))
        clipped = ((true_ratio < 1.0 - eps) | (true_ratio > 1.0 + eps)).float().mean()
        return {"approx_kl": float(kl), "clip_frac": float(clipped),
                "kl_saturated": float(sat)}


def statistics(rewards, adv, mask) -> dict:
    return {
        "reward": float(rewards[mask].sum()),
        "adv_abs": float(adv[mask].abs().mean()),
        "adv_std": float(adv[mask].std()),
    }


def _self_test() -> None:
    torch.manual_seed(0)
    T = 6
    mask = torch.zeros(4, T, dtype=torch.bool)
    mask[:, 3:] = True
    rew = torch.zeros(4, T)
    rew[:, 5] = torch.tensor([1.0, 0.0, 0.0, 1.0])
    a = advantages("grpo", rew, mask, 4)
    assert abs(a[0, 3].item() - 1.0) < 1e-5 and abs(a[1, 3].item() + 1.0) < 1e-5, a.tolist()

    # identical log probabilities -> ratio one -> loss is minus the mean advantage
    lp = torch.randn(4, T) * 0.1
    loss = clipped_loss(lp, lp.clone(), a, mask)
    want = -a[mask].mean()
    assert abs(loss.item() - want.item()) < 1e-6, (loss.item(), want.item())

    # a PPO request without a critic must be refused, not silently graded with GRPO
    try:
        advantages("ppo", rew, mask, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("PPO without values was accepted")

    st = statistics(rew, a, mask)
    assert abs(st["reward"] - 2.0) < 1e-6 and st["adv_abs"] > 0

    # lr schedule: 10% linear warmup then cosine decay to zero
    base, total = 1e-3, 100
    assert abs(lr_at(0, total, base) - base * 0.1) < 1e-9      # first warmup step
    assert abs(lr_at(9, total, base) - base) < 1e-9            # warmup ends at base
    assert abs(lr_at(10, total, base) - base) < 1e-6           # cosine starts at base
    assert abs(lr_at(total - 1, total, base)) < base * 0.01    # decays toward zero
    mid = lr_at(55, total, base)
    assert 0 < mid < base                                      # monotone decay region
    assert lr_at(20, total, base) > lr_at(60, total, base)
    assert lr_at(0, 0, base) == base                           # degenerate total
    assert abs(lr_at(50, 100, base, warmup_frac=0.0) - base * 0.5) < 1e-6  # no warmup

    # inner epochs: a moved policy must give ratio != 1, a live clip and KL > 0
    old = torch.zeros(4, T)
    moved = old.clone()
    moved[:, 3:] = 0.5                       # ratio = e^0.5 ~ 1.65, outside 1+-0.2
    d = ppo_diagnostics(moved, old, mask, eps=0.2)
    assert d["clip_frac"] > 0.99, d
    assert d["approx_kl"] > 0.1, d
    same = ppo_diagnostics(old, old, mask, eps=0.2)
    assert same["approx_kl"] == 0.0 and same["clip_frac"] == 0.0, same
    # the clip must actually bound the objective for a positive advantage
    pos = torch.ones(4, T)
    l_clip = clipped_loss(moved, old, pos, mask, eps=0.2)
    l_uncl = clipped_loss(moved, old, pos, mask, eps=1e9)
    assert l_clip > l_uncl, (l_clip.item(), l_uncl.item())   # -min() picks the clip
    # k3 is non-negative even when the policy moves the other way
    down = ppo_diagnostics(-moved, old, mask, eps=0.2)
    assert down["approx_kl"] > 0.0, down

    # KL to the frozen reference: non-negative, zero at identity, grows with drift
    mk = torch.zeros(2, 8, dtype=torch.bool)
    mk[:, 2:] = True
    ref = torch.zeros(2, 8)
    assert abs(float(kl_penalty(ref, ref, mk))) < 1e-9
    prev = 0.0
    for d in (0.1, 0.3, 0.6, 1.0):
        k = float(kl_penalty(ref + d, ref, mk))
        assert k >= 0.0 and k > prev, (d, k, prev)
        prev = k
    assert float(kl_penalty(ref - 0.5, ref, mk)) > 0.0   # both directions cost
    empty = torch.zeros(2, 8, dtype=torch.bool)
    assert float(kl_penalty(ref, ref, empty)) == 0.0

    # Regression: a real run logged KL 34.05 against a median of 0.086, letting the
    # penalty outweigh the policy loss ~34x on that one step. The bound must hold
    # AND keep a restoring gradient - a hard clamp or tanh both go flat out there,
    # which would silence the pull exactly when drift is worst.
    for d in (6.0, 11.0, 30.0, 100.0, 1e4):
        for sign in (-1.0, 1.0):
            lp = (ref + sign * d).detach().clone().requires_grad_(True)
            k = kl_penalty(lp, ref, mk)
            v = float(k.detach())
            assert v == v and v < 10.0, (d, sign, v)
            k.backward()
            g = float(lp.grad[mk].mean())
            # sign>0: logp above ref, gradient must push it down (positive)
            assert g * sign > 0.0, (d, sign, g)
    # healthy range stays readable: median observed KL ~0.086 must not be distorted
    raw = float((torch.exp(torch.full((12,), 0.3)) - 0.3 - 1.0).mean())
    got = float(kl_penalty(ref - 0.3, ref, mk))
    assert abs(got - raw) / raw < 0.03, (raw, got)

    print("  trainer self-test OK")
    print("    kl_penalty: non-negative, zero at identity, grows with drift")
    print("    kl_penalty: bounded under extreme drift, gradient never goes flat")
    print("    grpo advantage +-1 on a mixed group")
    print("    equal log probs -> loss equals minus mean advantage (ratio is one)")
    print("    ppo without a critic is refused")
    print("    lr schedule: 10% warmup to base, cosine decay toward zero")
    print("    inner epochs: clip engages, approx_kl >= 0, both zero on the first pass")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()
