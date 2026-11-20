#!/usr/bin/env python3
"""Pick RL problems the policy can SOMETIMES solve, from problems it never saw.

Two filters, in this order:

  1. drop figure-dependent problems. A text-only policy cannot read `[asy]` code
     or "as shown below", so those items are unsolvable noise: they pay nothing
     and drag the advantage estimate around.
  2. sample the CURRENT policy n times per problem and keep the ones whose pass
     rate lands in [lo, hi]. All-correct items give no gradient (nothing to
     improve); all-wrong items give no signal either (nothing to reinforce).
     The learning happens in between.

Why not reuse the SFT pool: those problems were memorised during SFT, so RL on
them rewards recall rather than reasoning, and the gain does not transfer to a
held-out test set. This script also refuses any problem already in the SFT data
or in the evaluation set.

The previous pool was built as "2 samples, at least one correct" out of the SFT
data: 4821 -> 162 items, all seen before, all already solvable. Measured result:
three RL runs failed to beat SFT.

Output is the ShareGPT shape the trainer and evaluator already speak, with the
system text copied verbatim from the SFT corpus so the rendering stays identical.

    python3 -m zero_ppo.select_pool --self-test
    python3 -m zero_ppo.select_pool --dry-run          # prefilter only, no GPU
    python3 -m zero_ppo.select_pool --model /path/to/your_sft \
        --candidates agent_data_math/rl_candidates.jsonl \
        --sft agent_data_math/sft_tir_v4.jsonl \
        --eval agent_data_math/math500.jsonl \
        --samples 8 --lo 0.1 --hi 0.9 --min-level 2 --easy-frac 0.3 \
        --report runs/probe.jsonl --out agent_data_math/rl_pool_v3.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ASY = re.compile(r"\[asy\]", re.I)
REF = re.compile(r"(shown|given|depicted|illustrated|seen)\s+(in|below)|"
                 r"(figure|diagram|graph|picture)\s+(below|above|shown)|"
                 r"in\s+the\s+(figure|diagram|graph|picture)|"
                 r"the\s+(figure|diagram|graph|picture)\s+(shows|below|above)|"
                 r"as\s+shown|refer\s+to\s+the", re.I)


def needs_figure(question: str) -> bool:
    """True when the problem points at a picture the model cannot see."""
    q = question or ""
    return bool(ASY.search(q) or REF.search(q))


def _norm(s: str) -> str:
    s = re.sub(r"\s+", "", s or "")
    for t in ("\\left", "\\right", "\\!", "\\,", "\\;"):
        s = s.replace(t, "")
    return s.lower()


def core_question(text: str) -> str:
    """The problem statement without the instruction prefix, for dedup."""
    return _norm((text or "").split("Problem:")[-1])


def first_user(record: dict) -> str:
    if record.get("conversations"):
        return record["conversations"][0].get("value", "")
    return record.get("raw_question") or record.get("question", "")


def load_jsonl(path: str) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def prefilter(candidates: list[dict], exclude: set[str],
              min_level: int = 0, max_level: int = 99) -> tuple[list[dict], dict]:
    """Remove figure problems, reused problems, and uninformative difficulty tiers.

    A measured probe showed level-1 items are solved on all 8 samples (27/33) and
    yield nothing in the band, so probing them only burns rollouts. `min_level`
    skips those tiers; it is a cost optimisation, not a claim about difficulty.
    """
    kept, stats = [], Counter()
    seen = set()
    for r in candidates:
        q = first_user(r)
        k = core_question(q)
        if not k:
            stats["empty"] += 1
            continue
        if k in seen:
            stats["duplicate"] += 1
            continue
        if k in exclude:
            stats["already_used"] += 1
            continue
        if needs_figure(q):
            stats["figure"] += 1
            continue
        lv = r.get("level", 0) or 0
        if not (min_level <= lv <= max_level):
            stats["level_out"] += 1
            continue
        seen.add(k)
        kept.append(r)
    stats["kept"] = len(kept)
    return kept, dict(stats)


def in_band(rate: float, lo: float, hi: float) -> bool:
    return lo <= rate <= hi


def split_quotas(probed: list[dict], args) -> tuple[list[dict], list[dict], dict]:
    """Hard items in the informative band, plus a quota of easy items.

    A run trained only on level 3-5 items improved on the training pool (correct
    55.2% -> 58.1%) while dropping on held-out data (pass@8 loose -6.77), and its
    boxed rate fell 90.4% -> 84.4% with turns rising 2.26 -> 2.77. The evaluation
    set is 26.9% level<=2, so a policy that only ever practises long chains on hard
    problems over-thinks the easy ones and stops emitting an answer.

    Easy items cannot use the same band: a probe found level 1 solved on all 8
    samples, so hi=0.9 admits none of them. They get their own upper bound
    `easy_hi` (default 1.0). Preference order within the easy pool:

      1. in-band easy items (rate < easy_hi): a mix of right and wrong gives a real
         advantage signal. Measured on a synthetic batch, correct-and-short scored
         +1.92 while wrong-and-long scored -0.65, i.e. it directly reinforces the
         short solution.
      2. all-correct easy items only to fill a shortfall. Once the critic fits them
         their advantage decays to ~0 (measured |A| 1.11 -> 0.12), so they carry
         almost no policy gradient and act mainly through the KL anchor holding the
         SFT behaviour in place. Useful, but weaker - hence second choice.

    Ordering by |rate - 0.5| still puts the most informative items first inside
    each group.
    """
    easy_lv = args.easy_max_level
    hard = [p for p in probed
            if (p["rec"].get("level", 0) or 0) > easy_lv
            and in_band(p["rate"], args.lo, args.hi)]
    easy_all = [p for p in probed
                if (p["rec"].get("level", 0) or 0) <= easy_lv
                and args.easy_lo <= p["rate"] <= args.easy_hi]
    easy_band = [p for p in easy_all if p["rate"] < 1.0]
    easy_solved = [p for p in easy_all if p["rate"] == 1.0]
    for g in (hard, easy_band, easy_solved):
        g.sort(key=lambda p: abs(p["rate"] - 0.5))

    n_hard = len(hard)
    if args.target:
        n_hard = min(n_hard, args.target)
    hard = hard[:n_hard]
    want = int(round(n_hard * args.easy_frac / max(1e-9, 1.0 - args.easy_frac)))
    easy = (easy_band + easy_solved)[:want]
    info = {"hard": len(hard), "easy": len(easy), "easy_want": want,
            "easy_band": sum(1 for p in easy if p["rate"] < 1.0),
            "easy_solved": sum(1 for p in easy if p["rate"] == 1.0),
            "easy_pool": len(easy_all)}
    return hard, easy, info


def build_record(rec: dict, system: str, turn_template: str) -> dict:
    """The ShareGPT shape: system + one human turn, plus gold and bookkeeping."""
    q = first_user(rec)
    body = q.split("Problem:")[-1].strip() if "Problem:" in q else q
    user = turn_template.format(q=body)
    return {"conversations": [{"from": "human", "value": user}],
            "system": system, "gold": str(rec.get("gold", "")),
            "uid": rec.get("uid", ""), "subject": rec.get("subject", ""),
            "level": rec.get("level", 0)}


# ------------------------------------------------------------------ gpu sampling
def detect_gpus() -> list[int]:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        return [int(x) for x in vis.split(",") if x.strip().isdigit()]
    try:
        import torch
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def _worker(gpu: int, shard: list[dict], args, system: str, queue) -> None:
    """One card: roll out each problem n times, report how many were correct."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from zero_ppo.eval import _run_shard, resolve_max_len
        from zero_ppo.rollout.tools import local_runner
        from zero_ppo.util.prompts import ToolFormat

        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        fmt = ToolFormat(tok)
        win, note = resolve_max_len(args.max_model_len, args.model)
        if note:
            print(f"  [gpu{gpu}] {note}", flush=True)
        llm = LLM(model=args.model, dtype=args.dtype, tensor_parallel_size=1,
                  gpu_memory_utilization=args.gpu_util, max_model_len=win,
                  max_num_seqs=args.max_num_seqs, seed=args.seed,
                  enable_prefix_caching=True, disable_custom_all_reduce=True)
        stop = ("<|im_end|>", "<|endoftext|>")

        def generate(prompts, temperature, n, max_tokens, seeds=None):
            top = 1.0 if temperature == 0 else args.top_p

            def mk(sd):
                return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                      n=n, stop=list(stop), seed=sd, top_p=top)
            params = mk(args.seed) if seeds is None else [mk(s) for s in seeds]
            outs = llm.generate(list(prompts), params)
            return [[{"text": c.text, "tokens": list(c.token_ids)}
                     for c in o.outputs] for o in outs]

        # reuse the evaluator's scheduler: same chat rendering, same tool loop,
        # same context guard - a selection pass must measure the real task
        import types
        sa = types.SimpleNamespace(
            greedy=False, samples=args.samples, temperature=args.temperature,
            top_p=args.top_p, max_turns=args.max_turns, max_new=args.max_new,
            batch=args.batch, max_model_len=win, min_new=32, seed=args.seed)
        prepared = [build_record(r, system, args.turn) for r in shard]
        results, calls = _run_shard(prepared, sa, generate, tok, fmt, local_runner())
        out = []
        for rec, res in zip(shard, results):
            ss = [s for s in res["samples"] if s["mode"] == "sample"]
            c = sum(1 for s in ss if s["loose"])
            out.append({"rec": rec, "n": len(ss), "correct": c,
                        "rate": (c / len(ss)) if ss else 0.0,
                        "turns": (sum(s["turns"] for s in ss) / len(ss)) if ss else 0.0,
                        "tool_ok": (sum(s["tool_ok"] for s in ss) / len(ss)) if ss else 0.0})
        queue.put(("ok", gpu, out))
    except Exception as e:
        import traceback
        queue.put(("err", gpu, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="the SFT policy to probe")
    ap.add_argument("--candidates", default="agent_data_math/rl_candidates.jsonl")
    ap.add_argument("--sft", default="agent_data_math/sft_tir_v4.jsonl")
    ap.add_argument("--eval", default="agent_data_math/math500.jsonl")
    ap.add_argument("--out", default="agent_data_math/rl_pool_v3.jsonl")
    ap.add_argument("--report", default=None, help="per-problem pass rates (jsonl)")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=0, help="probe only the first N")
    ap.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                    help="keep source order (it is sorted by subject/level, so a "
                         "--limit probe would be biased toward easy problems)")
    ap.add_argument("--stratify", type=int, default=0,
                    help="probe N problems spread evenly across difficulty levels")
    ap.add_argument("--min-level", type=int, default=0, dest="min_level",
                    help="skip easier tiers; a probe showed level 1 yields no "
                         "in-band items, so probing it only costs rollouts")
    ap.add_argument("--max-level", type=int, default=99, dest="max_level")
    ap.add_argument("--target", type=int, default=0, help="stop after N selected")
    ap.add_argument("--easy-frac", type=float, default=0.0, dest="easy_frac",
                    help="share of easy items in the final pool, e.g. 0.3. The "
                         "evaluation set is 26.9%% level<=2; training only on hard "
                         "items made the policy over-think easy ones. 0 disables")
    ap.add_argument("--easy-max-level", type=int, default=2, dest="easy_max_level",
                    help="levels counted as easy for the quota")
    ap.add_argument("--easy-lo", type=float, default=0.5, dest="easy_lo",
                    help="lower pass-rate bound for easy items")
    ap.add_argument("--easy-hi", type=float, default=1.0, dest="easy_hi",
                    help="upper bound for easy items; must exceed --hi or level 1 "
                         "(all-correct) can never be admitted")
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=1024, dest="max_new")
    ap.add_argument("--max-turns", type=int, default=6, dest="max_turns")
    ap.add_argument("--max-model-len", type=int, default=4096, dest="max_model_len")
    ap.add_argument("--gpu-util", type=float, default=0.85, dest="gpu_util")
    ap.add_argument("--max-num-seqs", type=int, default=256, dest="max_num_seqs")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--turn", default="Solve this problem. Call the Python tool for "
                                    "any numerical computation.\n\nProblem:\n{q}")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="only apply the figure/overlap filter, no GPU needed")
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    return ap.parse_args()


def main() -> int:
    args = build_args()
    if args.self_test:
        _self_test()
        return 0

    cands = load_jsonl(args.candidates)
    if not cands:
        raise SystemExit(f"no candidates at {args.candidates}")
    sft = load_jsonl(args.sft)
    if not sft:
        raise SystemExit(f"SFT corpus not found at {args.sft}; "
                         "its system text is required")
    system = sft[0].get("system", "")
    if not system:
        raise SystemExit("the SFT corpus carries no system text")
    exclude = {core_question(first_user(r)) for r in sft}
    exclude |= {core_question(first_user(r)) for r in load_jsonl(args.eval)}
    exclude.discard("")

    # An easy quota needs easy items to survive the prefilter: --min-level 2 would
    # otherwise drop every level-1 problem before it is ever sampled, and the quota
    # would silently come up short.
    min_lv = args.min_level
    if args.easy_frac > 0:
        if args.easy_hi <= args.hi:
            raise SystemExit(f"--easy-hi ({args.easy_hi}) must exceed --hi "
                             f"({args.hi}), otherwise all-correct easy items are "
                             "excluded and level 1 yields nothing")
        if min_lv > 1:
            print(f"  简单题配额已启用: 预筛难度下限 {min_lv} -> 1", flush=True)
            min_lv = 1
    kept, stats = prefilter(cands, exclude, min_lv, args.max_level)
    print(f"  候选 {len(cands)} 题 · 预筛: {stats}")
    print(f"  预筛后 {len(kept)} 题进入采样", flush=True)
    # The source file is ordered by subject and level, so taking the first N would
    # probe only easy Algebra and report a misleading selection rate. Shuffle with
    # a fixed seed: a representative probe, still reproducible.
    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(kept)
    if args.stratify:
        # equal quota per difficulty level, so a small probe covers the range
        by_lv = {}
        for r in kept:
            by_lv.setdefault(r.get("level", 0), []).append(r)
        per = max(1, args.stratify // max(1, len(by_lv)))
        mixed = []
        for lv in sorted(by_lv):
            mixed.extend(by_lv[lv][:per])
        kept = mixed
        print(f"  分层探测: 每个难度取 {per} 题, 共 {len(kept)} 题", flush=True)
    elif args.limit:
        kept = kept[:args.limit]
    if args.dry_run:
        print(f"  dry-run: 学科={Counter(r.get('subject') for r in kept).most_common()}")
        print(f"  dry-run: 难度={sorted(Counter(r.get('level') for r in kept).items())}")
        return 0
    if not args.model:
        raise SystemExit("--model is required (the SFT policy to probe)")

    gpus = ([int(x) for x in args.gpus.split(",")] if args.gpus else detect_gpus())
    if not gpus:
        raise SystemExit("no CUDA device visible; use --dry-run for the filter only")

    import multiprocessing as mp
    shards = [kept[i::len(gpus)] for i in range(len(gpus))]
    print(f"  采样: {len(kept)} 题 x {args.samples} 次 · {len(gpus)} 卡 {gpus} · "
          f"保留通过率 [{args.lo}, {args.hi}]", flush=True)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for gpu, shard in zip(gpus, shards):
        if not shard:
            continue
        p = ctx.Process(target=_worker, args=(gpu, shard, args, system, queue))
        p.start()
        procs.append(p)
    probed, failed = [], []
    for _ in procs:
        kind, gpu, payload = queue.get()
        if kind == "ok":
            probed.extend(payload)
            print(f"  [gpu{gpu}] 完成 {len(payload)} 题", flush=True)
        else:
            failed.append(gpu)
            print(f"  [gpu{gpu}] 失败:\n{payload}", flush=True)
    for p in procs:
        p.join()
    if not probed:
        raise SystemExit("all workers failed")

    if args.easy_frac > 0:
        hard, easy, qi = split_quotas(probed, args)
        selected = hard + easy
        print(f"  配额: 难题 {qi['hard']} + 简单题 {qi['easy']}/{qi['easy_want']} "
              f"(带内 {qi['easy_band']}, 全对 {qi['easy_solved']}; "
              f"可选简单题 {qi['easy_pool']})", flush=True)
        if qi["easy"] < qi["easy_want"]:
            print(f"  ! 简单题不足: 需 {qi['easy_want']} 实得 {qi['easy']}；"
                  f"可放宽 --easy-lo 或增大探测量", flush=True)
    else:
        selected = [p for p in probed if in_band(p["rate"], args.lo, args.hi)]
        selected.sort(key=lambda p: abs(p["rate"] - 0.5))  # most informative first
        if args.target:
            selected = selected[:args.target]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in selected:
            f.write(json.dumps(build_record(p["rec"], system, args.turn),
                               ensure_ascii=False) + "\n")
    # mark what actually entered the pool: with an easy quota, membership is no
    # longer the same predicate as the hard band
    picked_ids = {id(p) for p in selected}
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            for p in probed:
                f.write(json.dumps({"uid": p["rec"].get("uid", ""),
                                    # uid is empty throughout this corpus, so the
                                    # normalised question is the only usable join key
                                    "qkey": core_question(first_user(p["rec"])),
                                    "subject": p["rec"].get("subject", ""),
                                    "level": p["rec"].get("level", 0),
                                    "n": p["n"], "correct": p["correct"],
                                    "rate": p["rate"], "turns": p["turns"],
                                    "tool_ok": p["tool_ok"],
                                    "selected": id(p) in picked_ids},
                                   ensure_ascii=False) + "\n")

    rates = [p["rate"] for p in probed]
    buckets = Counter()
    for r in rates:
        buckets["0.0 (全错)" if r == 0 else "1.0 (全对)" if r == 1.0
                else f"{args.lo}~{args.hi} (选中)" if in_band(r, args.lo, args.hi)
                else "其他"] += 1
    print("\n" + "=" * 62)
    print(f"  探测 {len(probed)} 题 · 平均通过率 {sum(rates)/max(len(rates),1):.3f}")
    for k, v in buckets.most_common():
        print(f"    {k:22} {v:5d}  ({100*v/len(probed):.1f}%)")
    # per-level breakdown: tells you whether to widen the band or probe harder items
    by_lv = {}
    for p in probed:
        by_lv.setdefault(p["rec"].get("level", 0), []).append(p)
    if len(by_lv) > 1:
        # two separate columns: "in band" is the probe verdict, "picked" is what
        # actually entered the pool. An easy quota makes them differ, and folding
        # them into one column hides how many usable items were left on the table.
        print(f"  {'难度':<6}{'题数':>6}{'平均通过率':>12}{'全对':>8}{'全错':>8}"
              f"{'带内':>8}{'入池':>8}")
        for lv in sorted(by_lv):
            g = by_lv[lv]
            rr = [p["rate"] for p in g]
            print(f"  {lv:<6}{len(g):>6}{sum(rr)/len(rr):>12.3f}"
                  f"{sum(1 for x in rr if x==1.0):>8}"
                  f"{sum(1 for x in rr if x==0.0):>8}"
                  f"{sum(1 for p in g if in_band(p['rate'],args.lo,args.hi)):>8}"
                  f"{sum(1 for p in g if id(p) in picked_ids):>8}")
    print(f"  选中 {len(selected)} 题 -> {args.out}")
    if selected:
        print(f"    学科={Counter(p['rec'].get('subject') for p in selected).most_common()}")
        print(f"    平均通过率={sum(p['rate'] for p in selected)/len(selected):.3f} · "
              f"平均轮数={sum(p['turns'] for p in selected)/len(selected):.2f} · "
              f"每轨迹成功工具调用={sum(p['tool_ok'] for p in selected)/len(selected):.2f}")
        slv = Counter(p["rec"].get("level", 0) for p in selected)
        n_easy = sum(n for lv, n in slv.items() if (lv or 0) <= args.easy_max_level)
        print(f"    难度={sorted(slv.items())} · "
              f"level<={args.easy_max_level} 占 {100*n_easy/len(selected):.1f}% "
              f"(评测集 26.9%)")
    if failed:
        print(f"  ! 失败的卡: {failed}（结果不完整）")
    print("=" * 62)
    if args.report:
        print(f"  逐题通过率 -> {args.report}")
    return 0


# --------------------------------------------------------------------- self test
def _self_test() -> None:
    assert needs_figure("In the diagram below, [asy]size(2cm);[/asy] find x.")
    assert needs_figure("The triangle is shown below. Find its area.")
    assert needs_figure("As shown, the circles are tangent.")
    assert not needs_figure("What is 47*89 - 1234?")
    assert not needs_figure("Find the sum of the roots of x^2-5x+6=0.")
    # a graph of a function is still a figure
    assert needs_figure("The complete graph of y=f(x) is shown in red below.")

    assert core_question("Solve this problem.\n\nProblem:\nWhat is 2+2?") == \
           core_question("What is 2+2?")
    assert core_question("\\left( 3, 4 \\right)") == core_question("(3,4)")

    assert in_band(0.5, 0.2, 0.8)
    assert in_band(0.2, 0.2, 0.8) and in_band(0.8, 0.2, 0.8)
    assert not in_band(0.0, 0.2, 0.8)      # nothing to reinforce
    assert not in_band(1.0, 0.2, 0.8)      # nothing to improve
    assert not in_band(0.125, 0.2, 0.8)

    cands = [{"question": "What is 2+2?", "gold": "4", "uid": "a"},
             {"question": "As shown below, find x.", "gold": "1", "uid": "fig"},
             {"question": "What is 3+3?", "gold": "6", "uid": "used"},
             {"question": "What is 2+2?", "gold": "4", "uid": "dup"},
             {"question": "[asy]draw((0,0)--(1,1));[/asy] area?", "gold": "2", "uid": "asy"}]
    excl = {core_question("What is 3+3?")}
    kept, st = prefilter(cands, excl)
    assert [r["uid"] for r in kept] == ["a"], kept
    assert st["figure"] == 2 and st["already_used"] == 1 and st["duplicate"] == 1, st

    # difficulty tiers: skip the ones a probe showed to be uninformative
    lv = [{"question": f"q{i}", "gold": str(i), "uid": f"u{i}", "level": i}
          for i in (1, 2, 3, 5)]
    k2, s2 = prefilter(lv, set(), min_level=2)
    assert [r["level"] for r in k2] == [2, 3, 5], k2
    assert s2["level_out"] == 1, s2
    k3, _ = prefilter(lv, set(), min_level=3, max_level=4)
    assert [r["level"] for r in k3] == [3], k3
    k4, s4 = prefilter(lv, set())          # default keeps everything
    assert len(k4) == 4 and "level_out" not in s4

    SYS = "protocol"
    TURN = "Solve this problem.\n\nProblem:\n{q}"
    rec = build_record({"question": "What is 2+2?", "gold": "4", "uid": "a",
                        "subject": "Algebra", "level": 1}, SYS, TURN)
    assert rec["system"] == SYS and rec["gold"] == "4"
    assert [c["from"] for c in rec["conversations"]] == ["human"]
    assert rec["conversations"][0]["value"] == TURN.format(q="What is 2+2?")
    assert rec["subject"] == "Algebra" and rec["level"] == 1
    # already-folded input must not double the instruction
    rec2 = build_record({"question": TURN.format(q="What is 2+2?"), "gold": "4"},
                        SYS, TURN)
    assert rec2["conversations"][0]["value"].count("Solve this problem") == 1

    # the record must render exactly like the training/eval pipeline
    from zero_ppo.util.prompts import ToolFormat, opening_messages

    class _Tok:
        chat_template = "yes"

        def apply_chat_template(self, messages, add_generation_prompt=True,
                                tokenize=False):
            s = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
            return s + ("<|assistant|>" if add_generation_prompt else "")

    f = ToolFormat(_Tok())
    p = f.prompt(rec)
    assert p.startswith("<|system|>protocol") and p.endswith("<|assistant|>")
    assert [m["role"] for m in opening_messages(rec)] == ["system", "user"]

    # easy quota: hard items keep the informative band, easy items get their own
    # upper bound because level 1 is all-correct and would otherwise never qualify
    import types

    def mk(uid, lv, rate):
        return {"rec": {"uid": uid, "level": lv, "question": f"q{uid}",
                        "gold": "1", "subject": "Algebra"},
                "rate": rate, "n": 8, "correct": int(rate * 8),
                "turns": 2.0, "tool_ok": 1.0}

    probed = ([mk(f"h{i}", 4, 0.5) for i in range(10)]
              + [mk(f"e{i}", 1, 1.0) for i in range(6)]
              + [mk(f"b{i}", 2, 0.75) for i in range(4)]
              + [mk(f"x{i}", 1, 0.0) for i in range(5)])   # all-wrong: excluded
    qa = types.SimpleNamespace(lo=0.1, hi=0.9, easy_frac=0.3, easy_max_level=2,
                               easy_lo=0.5, easy_hi=1.0, target=0)
    hard, easy, qi = split_quotas(probed, qa)
    assert len(hard) == 10 and qi["easy_want"] == 4, qi
    assert len(easy) == 4, qi
    # in-band easy items must be preferred over all-correct ones
    assert qi["easy_band"] == 4 and qi["easy_solved"] == 0, qi
    assert all(p["rate"] > 0 for p in easy), "all-wrong easy items must not enter"

    # shortfall is filled with all-correct items, and reported honestly
    probed2 = [mk(f"h{i}", 4, 0.5) for i in range(10)] + [mk("s", 1, 1.0)]
    h2, e2, q2 = split_quotas(probed2, qa)
    assert len(e2) == 1 and q2["easy_solved"] == 1 and q2["easy"] < q2["easy_want"]

    # quota disabled -> old behaviour exactly
    qa0 = types.SimpleNamespace(lo=0.1, hi=0.9, easy_frac=0.0, easy_max_level=2,
                                easy_lo=0.5, easy_hi=1.0, target=0)
    h0, e0, _ = split_quotas(probed, qa0)
    assert len(h0) == 10 and e0 == []

    # the prefilter must not drop easy items before they are sampled
    lvq = [{"question": f"problem number {i}", "gold": "1", "uid": f"u{i}",
            "level": lv} for i, lv in enumerate((1, 1, 2, 4))]
    kq, _ = prefilter(lvq, set(), min_level=1)
    assert sum(1 for r in kq if r["level"] <= 2) == 3, kq
    kq2, _ = prefilter(lvq, set(), min_level=2)   # what the old default did
    assert sum(1 for r in kq2 if r["level"] <= 2) == 1, kq2

    print("  select_pool self-test OK")
    print("    figure filter, dedup/overlap exclusion, pass-rate band")
    print("    easy quota: in-band preferred, all-correct only as fallback")
    print("    ShareGPT output renders identically to the training pipeline")


if __name__ == "__main__":
    raise SystemExit(main())