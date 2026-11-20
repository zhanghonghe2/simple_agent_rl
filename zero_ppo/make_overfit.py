#!/usr/bin/env python3
"""Build a tiny fixed pool to test whether RL can overfit a known-solvable set.

The diagnostic question: SFT memorises any 8 problems by construction (the gold
trajectory is fed directly as the target). RL has a hard precondition - the policy
must occasionally sample a CORRECT trajectory, because reward only scores what was
generated. A problem with pass rate 0 gives nothing to reinforce, and repeatedly
pushing its wrong answers down is exactly the collapse mechanism seen in run 5.

So this picks problems the policy ALREADY sometimes solves (pass rate in a band
around 0.5, never 0 and never 1). If RL cannot drive even this narrow, solvable set
to 100%, the algorithm/reward/implementation is suspect. If it can overfit but fails
to transfer, the problem is generalisation, not learning.

It reads the select_pool report (uid + measured pass rate) and the candidate pool
(uid + question + gold), joins them, and writes a ShareGPT pool identical in shape
to the training data. Run it on the server where those files live:

Simplest path - sample straight from an already-filtered pool (every item there is
in band, which is all this experiment needs):

    python3 -m zero_ppo.make_overfit \
        --pool agent_data_math/rl_pool_v3.jsonl \
        --n 32 --out agent_data_math/overfit32.jsonl

Or select by measured pass rate, if the report carries the `qkey` field:

    python3 -m zero_ppo.make_overfit \
        --report runs/probe.jsonl \
        --candidates agent_data_math/rl_candidates.jsonl \
        --sft agent_data_math/sft_tir_v4.jsonl \
        --n 32 --lo 0.3 --hi 0.7 \
        --out agent_data_math/overfit32.jsonl

Then train with enough steps to cover the pool many times:

    MODEL=/path/to/your_sft POOL=agent_data_math/overfit32.jsonl \
    GPUS=2 ALGO=grpo CRITIC=none REWARD=zero_ppo/reward/reward_result.py \
    BATCH=32 GROUP=8 MINI_BATCH=4 PPO_EPOCHS=1 STEPS=60 KL_COEF=0.02 \
    bash zero_ppo/train.sh

With 32 problems and batch 8, one full pass is 4 steps. "All correct" means the
last 4 steps sum to 32/32, NOT a single 8/8 step (that is only a quarter of the set).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero_ppo.select_pool import (build_record, core_question, first_user,
                                  load_jsonl)


def pick(probed: list[dict], n: int, lo: float, hi: float) -> list[dict]:
    """In-band items, most informative (closest to 0.5) first, capped at n."""
    band = [p for p in probed if lo <= p["rate"] <= hi]
    band.sort(key=lambda p: abs(p["rate"] - 0.5))
    return band[:n]


def _key(rec: dict) -> str:
    return core_question(first_user(rec))


def join_and_build(picked: list[dict], cands: list[dict], system: str,
                   turn: str) -> tuple[list[dict], list[str]]:
    """Match report rows to problems by QUESTION TEXT, not uid.

    Every uid in this corpus is the empty string, so a uid join silently matches
    nothing (it produced an empty pool). The normalised question is unique across
    all 3247 candidates, so it is a safe key. `qkey` is present in reports written
    after that fix; older reports are matched by falling back to the pool below.
    """
    by_key = {}
    for c in cands:
        k = _key(c)
        if k:
            by_key.setdefault(k, c)
    out, missing = [], []
    for p in picked:
        k = p.get("qkey") or ""
        rec = by_key.get(k)
        if rec is None:
            missing.append(k or str(p.get("uid", "")) or "<no key>")
            continue
        out.append(build_record(rec, system, turn))
    return out, missing


def sample_pool(pool: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Take n problems straight from an already-filtered training pool.

    This is the path that works with the v3 artefacts. That report carries no
    question text and every uid is empty, so report and pool share NO usable join
    key - re-attaching pass rates is impossible without another sampling pass.

    It is also unnecessary: every item in the v3 pool is already in band (hard items
    0.1-0.9, easy items 0.5-1.0 with zero all-correct admitted), so none has pass
    rate 0 or 1. Any subset therefore satisfies the overfit precondition "sometimes
    solved, not yet always solved". Shuffling with a fixed seed keeps it reproducible
    and avoids taking only the first subject/level block.
    """
    import random
    rows = list(pool)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="agent_data_math/rl_pool_v3.jsonl",
                    help="an already-filtered training pool; take n items from it "
                         "directly (the default path). Every pooled item is already "
                         "in band, which is all this experiment needs. Pass "
                         "--report instead to select by measured pass rate")
    ap.add_argument("--report", default=None,
                    help="select_pool per-problem report; needs the qkey field "
                         "(written after the uid fix) to be joinable. Overrides "
                         "--pool when given")
    ap.add_argument("--candidates", default="agent_data_math/rl_candidates.jsonl")
    ap.add_argument("--sft", default="agent_data_math/sft_tir_v4.jsonl",
                    help="only its system text is used, copied verbatim")
    ap.add_argument("--out", default="agent_data_math/overfit32.jsonl")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--lo", type=float, default=0.3)
    ap.add_argument("--hi", type=float, default=0.7)
    ap.add_argument("--turn", default="Solve this problem. Call the Python tool for "
                                    "any numerical computation.\n\nProblem:\n{q}")
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    return ap.parse_args()


def main() -> int:
    args = build_args()
    if args.self_test:
        _self_test()
        return 0

    rates: list[float] = []
    # --report wins when given explicitly; otherwise sample the (default) pool.
    if args.pool and not args.report:
        pool = load_jsonl(args.pool)
        if not pool:
            raise SystemExit(f"no pool at {args.pool}")
        records = sample_pool(pool, args.n)
        if len(records) < args.n:
            print(f"  ! 池里只有 {len(pool)} 题 (需 {args.n})", flush=True)
        print(f"  从已筛池 {args.pool} 直接抽 {len(records)} 题", flush=True)
        print(f"  (池内全部带内, 无通过率0或1的题, 满足过拟合前提)", flush=True)
    else:
        probed = load_jsonl(args.report)
        if not probed:
            raise SystemExit(f"no report at {args.report}; "
                             "or pass --pool to sample an existing pool directly")
        if not any(p.get("qkey") for p in probed):
            raise SystemExit(
                f"{args.report} has no qkey field (written before the uid fix) and "
                "all uids are empty, so it cannot be joined to the candidates.\n"
                "  Use --pool agent_data_math/rl_pool_v3.jsonl instead: every item "
                "there is already in band, which is all this experiment needs.")
        cands = load_jsonl(args.candidates)
        if not cands:
            raise SystemExit(f"no candidates at {args.candidates}")
        sft = load_jsonl(args.sft)
        if not sft or not sft[0].get("system"):
            raise SystemExit(f"the SFT corpus at {args.sft} carries no system text")
        picked = pick(probed, args.n, args.lo, args.hi)
        if len(picked) < args.n:
            print(f"  ! 带内只有 {len(picked)} 题 (需 {args.n})；"
                  f"可放宽 --lo/--hi", flush=True)
        records, missing = join_and_build(picked, cands, sft[0]["system"], args.turn)
        if missing:
            print(f"  ! {len(missing)} 题在候选里找不到: {missing[:3]}", flush=True)
        rates = [p["rate"] for p in picked]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 62)
    print(f"  过拟合池 {len(records)} 题 -> {args.out}")
    if rates:
        print(f"    通过率: 均值 {sum(rates)/len(rates):.3f} · "
              f"范围 [{min(rates):.3f}, {max(rates):.3f}]")
    print(f"    难度={sorted(Counter(r.get('level', 0) for r in records).items())}")
    print(f"    学科={Counter(r.get('subject', '') for r in records).most_common()}")
    print(f"  判据: batch=8 时 4 步轮完一遍, '全对' = 最近4步合计 {len(records)}/{len(records)}")
    print("=" * 62)
    return 0


def _self_test() -> None:
    probed = [{"uid": f"u{i}", "rate": r} for i, r in enumerate(
        [0.0, 0.1, 0.3, 0.5, 0.5, 0.7, 0.9, 1.0])]
    got = pick(probed, 3, 0.3, 0.7)
    # closest to 0.5 first; the two 0.5s lead, then 0.3 or 0.7
    assert [p["rate"] for p in got[:2]] == [0.5, 0.5], got
    assert len(got) == 3 and all(0.3 <= p["rate"] <= 0.7 for p in got)
    assert all(p["rate"] not in (0.0, 1.0) for p in got)
    # band excludes the unsolvable (0) and the already-perfect (1)
    assert pick(probed, 99, 0.3, 0.7).__len__() == 4

    # join by QUESTION, because every uid in this corpus is the empty string
    cands = [{"uid": "", "question": "What is 2+2?", "gold": "4",
              "subject": "Algebra", "level": 1},
             {"uid": "", "question": "What is 3+3?", "gold": "6",
              "subject": "Algebra", "level": 2}]
    picked = [{"uid": "", "qkey": core_question("What is 2+2?"), "rate": 0.5},
              {"uid": "", "qkey": core_question("What is 3+3?"), "rate": 0.5},
              {"uid": "", "qkey": "ghost", "rate": 0.5}]
    recs, missing = join_and_build(picked, cands, "protocol",
                                   "Solve.\n\nProblem:\n{q}")
    assert len(recs) == 2 and missing == ["ghost"], (recs, missing)
    assert recs[0]["system"] == "protocol" and recs[0]["gold"] == "4"
    assert recs[0]["conversations"][0]["value"].count("Solve.") == 1
    # an empty uid must never be used as a key: it would match everything
    assert join_and_build([{"uid": "", "rate": 0.5}], cands, "s", "{q}")[0] == []

    # sampling an existing pool: reproducible, and not just the first block
    pool = [{"conversations": [{"from": "human", "value": f"q{i}"}],
             "system": "s", "gold": str(i), "level": i % 5} for i in range(50)]
    a = sample_pool(pool, 32, seed=0)
    b = sample_pool(pool, 32, seed=0)
    assert len(a) == 32 and a == b
    assert len({r["gold"] for r in a}) == 32          # no duplicates
    assert {r["level"] for r in a} == {0, 1, 2, 3, 4}  # spans levels, not a prefix
    assert sample_pool(pool, 99) == pool or len(sample_pool(pool, 99)) == 50

    print("  make_overfit self-test OK")
    print("    band excludes pass-rate 0 and 1, picks most-informative first")
    print("    joins by question text (uid is empty corpus-wide)")
    print("    pool sampling is reproducible and spans difficulty levels")


if __name__ == "__main__":
    raise SystemExit(main())