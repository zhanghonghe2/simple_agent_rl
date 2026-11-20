#!/usr/bin/env python3
"""Evaluate a trained policy on MATH-500 with the same protocol it was trained on.

The rollout is the TRAINING rollout (`rollout.multiturn` + `util.prompts`), so the
model sees the identical chat rendering and the same `<output>` tool turns. An eval
that rendered prompts differently would measure a distribution the policy never
trained on.

Two decoding modes, two scoring rules:

  greedy     temperature 0, one sample per problem -> accuracy
  pass@k     temperature T, n samples per problem -> pass@1 .. pass@k, using the
             unbiased estimator 1 - C(n-c, k)/C(n, k) rather than "any of k",
             which is biased when n > k.

  strict     the extracted answer must equal the gold string after normalisation
  loose      numeric / symbolic equivalence (sympy, math_verify), optionally
             arbitrated by an LLM API for the cases rules cannot settle

 GPUs are detected automatically and the problems are sharded across them, one
 vLLM instance per card (TP=1 each) - for a 1.5B model that beats one TP=N engine
 because there is no cross-card sync per token.

    python3 -m zero_ppo.eval --self-test
    python3 -m zero_ppo.eval --model runs/zero_ppo/checkpoints/final \
        --pool agent_data_math/math500.jsonl --samples 8 --out runs/eval
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero_ppo.rollout.multiturn import MAX_TURNS


# --------------------------------------------------------------------- scoring
def strict_equal(pred: str | None, gold: str | None) -> bool:
    """Exact match after whitespace/LaTeX-decoration normalisation, no algebra."""
    if pred is None or gold is None:
        return False
    from zero_ppo.util.judge import _norm
    return _norm(pred) == _norm(gold)


def loose_equal(pred: str | None, gold: str | None) -> bool:
    """Numeric or symbolic equivalence: 0.5 == \\frac{1}{2}, sqrt(4) == 2.

    Loose must be a SUPERSET of strict - an exact match is trivially equivalent -
    but a run reported greedy strict 63.32% against loose 61.79%, i.e. the wider
    rule scored LOWER. Cause: `util.judge.equal` tries math_verify before its
    string fallback, and `verify(parse(p), parse(g))` can RETURN False rather than
    raise on inputs it parses but cannot decide (text answers, sets, tuples). That
    False short-circuits the `_norm(pred) == _norm(gold)` fallback at the end, so a
    character-identical answer is rejected. Only reproduces where math_verify is
    installed, which is why local self-tests never caught it.

    Taking the union with strict makes the containment hold by construction,
    regardless of which optional parser is present or how it behaves.
    """
    from zero_ppo.util.judge import equal
    return bool(equal(pred, gold)) or strict_equal(pred, gold)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples of which c are correct.

    1 - C(n-c, k) / C(n, k). Computed as a product to avoid huge factorials, and
    it is 1.0 as soon as n-c < k (every k-subset must contain a correct one).
    """
    if n < k or k <= 0:
        raise ValueError(f"pass@{k} needs n >= k, got n={n}")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


class LLMJudge:
    """Optional arbiter for answers the rules cannot settle.

    Off unless --judge-api is given. Only the two answer strings are sent, never
    the problem or the trajectory, and a failure is reported rather than silently
    counted as wrong.
    """

    def __init__(self, model: str, base_url: str | None = None,
                 api_key_env: str = "DEEPSEEK_API_KEY", timeout: float = 30.0,
                 workers: int = 16, retries: int = 2):
        self.model = model
        self.base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://api.deepseek.com/v1")
        self.api_key = os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.workers = max(1, int(workers))
        self.retries = max(0, int(retries))
        self.calls = 0
        self.errors = 0
        self.cache: dict[tuple[str, str], bool] = {}

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _ask(self, pred: str, gold: str) -> bool | None:
        """One request, with bounded retries on transient failures."""
        import time
        prompt = (
            "Are these two mathematics answers equivalent in value? "
            "Ignore formatting, units and LaTeX style. "
            "Reply with exactly YES or NO.\n\n"
            f"Answer A: {pred}\nAnswer B: {gold}\n")
        last = None
        for attempt in range(self.retries + 1):
            try:
                import requests
                r = requests.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json={"model": self.model, "temperature": 0,
                          "max_tokens": 8,
                          "messages": [{"role": "user", "content": prompt}]},
                    timeout=self.timeout)
                r.raise_for_status()
                txt = r.json()["choices"][0]["message"]["content"].strip().upper()
                return txt.startswith("Y")
            except Exception as e:
                last = e
                if attempt < self.retries:
                    time.sleep(min(2.0 ** attempt, 4.0))
        raise last if last else RuntimeError("judge request failed")

    def equivalent(self, pred: str, gold: str) -> bool | None:
        """Single pair. Prefer `equivalent_many` - this blocks on the network."""
        key = (pred, gold)
        if key in self.cache:
            return self.cache[key]
        if not self.available:
            return None
        try:
            out = self._ask(pred, gold)
        except Exception as e:
            self.errors += 1
            if self.errors == 1:
                print(f"  ! judge api failed: {type(e).__name__}: {e}", flush=True)
            return None
        self.calls += 1
        self.cache[key] = out
        return out

    def equivalent_many(self, pairs) -> dict:
        """Resolve many (pred, gold) pairs CONCURRENTLY.

        The API is the slow part, so calls overlap in a thread pool instead of
        being awaited one by one. Identical pairs are asked once. Pairs that fail
        every retry are simply absent from the result, so the caller keeps its
        rule-based verdict rather than recording a wrong answer.
        """
        uniq = []
        seen = set()
        for p, g in pairs:
            k = (p, g)
            if k not in seen and k not in self.cache:
                seen.add(k)
                uniq.append(k)
        if not uniq or not self.available:
            return dict(self.cache)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        done = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futs = {pool.submit(self._ask, p, g): (p, g) for p, g in uniq}
            for fut in as_completed(futs):
                key = futs[fut]
                try:
                    self.cache[key] = fut.result()
                    self.calls += 1
                except Exception as e:
                    self.errors += 1
                    if self.errors == 1:
                        print(f"  ! judge api failed: {type(e).__name__}: {e}",
                              flush=True)
                done += 1
                if done % 50 == 0 or done == len(uniq):
                    print(f"  judge: {done}/{len(uniq)} "
                          f"(失败 {self.errors})", flush=True)
        return dict(self.cache)


def score_one(pred_text: str, gold: str, judge: LLMJudge | None = None) -> dict:
    """Both verdicts for one completion, plus what was extracted.

    With `judge` given this blocks on the API, so the evaluation loop calls it
    WITHOUT a judge and arbitrates the leftovers in one concurrent batch
    afterwards - a network wait must not stall the GPU.
    """
    from zero_ppo.util.judge import extract_answer
    ans = extract_answer(pred_text)
    strict = strict_equal(ans, gold)
    loose = loose_equal(ans, gold)
    arbitrated = False
    if judge is not None and ans is not None and not loose:
        verdict = judge.equivalent(ans, gold)
        if verdict is not None:
            loose = verdict
            arbitrated = True
    return {"answer": ans, "strict": strict, "loose": loose,
            "arbitrated": arbitrated, "boxed": ans is not None}


def arbitrate(results: list[dict], judge: LLMJudge) -> dict:
    """Resolve every rule-unsettled answer in one concurrent pass.

    Only pairs the rules rejected are sent, deduplicated across the whole run, so
    a repeated wrong answer costs one call rather than one per sample.
    """
    pairs = []
    for r in results:
        gold = r["gold"]
        for s in r["samples"]:
            if s["answer"] is not None and not s["loose"]:
                pairs.append((s["answer"], gold))
    if not pairs:
        return {"asked": 0, "calls": 0, "errors": 0, "flipped": 0}
    uniq = len(set(pairs))
    print(f"  judge: {uniq} 个待判答案对（{len(pairs)} 次出现），"
          f"并发 {judge.workers}", flush=True)
    verdicts = judge.equivalent_many(pairs)
    flipped = 0
    for r in results:
        gold = r["gold"]
        for s in r["samples"]:
            if s["answer"] is None or s["loose"]:
                continue
            v = verdicts.get((s["answer"], gold))
            if v is not None:
                s["loose"] = bool(v)
                s["arbitrated"] = True
                if v:
                    flipped += 1
    return {"asked": uniq, "calls": judge.calls, "errors": judge.errors,
            "flipped": flipped}


# ------------------------------------------------------------------- gpu / run
def detect_gpus() -> list[int]:
    """Visible CUDA devices, honouring CUDA_VISIBLE_DEVICES; [] when none."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        return [int(x) for x in vis.split(",") if x.strip().isdigit()]
    try:
        import torch
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def model_max_len(model_path: str) -> int | None:
    """The context window the checkpoint actually declares.

    Qwen2.5-Math ships max_position_embeddings=4096 while the Instruct variants
    ship 32768, so a window copied from a sibling model is rejected by vLLM (and
    forcing it past RoPE's trained range yields NaNs). Read the config instead of
    trusting the flag.
    """
    try:
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return None
    for key in ("max_position_embeddings", "seq_length", "max_seq_len",
                "n_positions"):
        v = cfg.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    tc = cfg.get("text_config") or {}
    v = tc.get("max_position_embeddings")
    return int(v) if isinstance(v, (int, float)) and v > 0 else None


def resolve_max_len(requested: int, model_path: str) -> tuple[int, str]:
    """Clamp the requested window to what the model supports, and say so."""
    real = model_max_len(model_path)
    if real is None:
        return int(requested), ""
    if requested > real:
        return real, (f"请求 {requested} 超过模型上限 {real}，已自动降为 {real} "
                      f"(config.json 的 max_position_embeddings)")
    return int(requested), ""


def _worker(gpu: int, shard: list[dict], args, queue) -> None:
    """One card: build a TP=1 engine, roll out its shard, score, report."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    try:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from zero_ppo.rollout.tools import local_runner
        from zero_ppo.util.prompts import ToolFormat

        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        fmt = ToolFormat(tok)
        llm = LLM(model=args.model, dtype=args.dtype, tensor_parallel_size=1,
                  gpu_memory_utilization=args.gpu_util,
                  max_model_len=args.max_model_len,
                  max_num_seqs=args.max_num_seqs,
                  seed=args.seed,
                  enable_prefix_caching=True, disable_custom_all_reduce=True)
        run_code = local_runner()
        stop = ("<|im_end|>", "<|endoftext|>")

        def generate(prompts, temperature, n, max_tokens, seeds=None):
            # Each SAMPLE needs its own seed. One shared seed makes every sample
            # of a problem identical, which collapses pass@k onto pass@1 and
            # silently destroys the measurement. Per-sample seeds keep the samples
            # independent while staying reproducible across checkpoints.
            base = list(prompts)
            top = 1.0 if temperature == 0 else args.top_p

            def mk(sd):
                return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                      n=n, stop=list(stop), seed=sd, top_p=top)

            # vLLM accepts one SamplingParams per prompt, so independence does not
            # cost the batch: all requests still go in a single generate call.
            params = mk(args.seed) if seeds is None else [mk(sd) for sd in seeds]
            outs = llm.generate(base, params)
            return [[{"text": c.text, "tokens": list(c.token_ids)}
                     for c in o.outputs] for o in outs]

        judge = None
        if args.judge_api:
            judge = LLMJudge(args.judge_model, args.judge_base_url, args.judge_key_env,
                             timeout=args.judge_timeout, workers=args.judge_workers,
                             retries=args.judge_retries)
            if not judge.available:
                print(f"  [gpu{gpu}] ! {args.judge_key_env} not set; "
                      "LLM arbitration disabled", flush=True)

        # One wavefront per turn, exactly like the training rollout: every live
        # trajectory goes into the same `generate` call so vLLM batches them.
        # Feeding one prompt at a time pinned the engine at batch size 1.
        print(f"  [gpu{gpu}] {len(shard)} 题 · batch={args.batch} · "
              f"max_num_seqs={args.max_num_seqs}", flush=True)
        results, calls = _run_shard(shard, args, generate, tok, fmt, run_code)
        print(f"  [gpu{gpu}] generate 调用 {calls['n']} 次 · "
              f"单次最多 {max(calls['sizes']) if calls['sizes'] else 0} 条 prompt",
              flush=True)

        stats = None
        if judge is not None and judge.available:
            print(f"  [gpu{gpu}] 推理完成，开始并发判分", flush=True)
            stats = arbitrate(results, judge)
        queue.put(("ok", gpu, results, stats))
    except Exception as e:
        import traceback
        queue.put(("err", gpu, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


# ---------------------------------------------------------------------- report
def summarise(results: list[dict], ks: list[int]) -> dict:
    """Greedy accuracy and pass@k, per scoring rule, overall and by subject."""
    def block(rows):
        out = {}
        greedy = [r for r in rows
                  if any(s["mode"] == "greedy" for s in r["samples"])]
        if greedy:
            for rule in ("strict", "loose"):
                hit = sum(1 for r in greedy
                          for s in r["samples"] if s["mode"] == "greedy" and s[rule])
                out[f"greedy_{rule}"] = hit / len(greedy)
        sampled = [r for r in rows if any(s["mode"] == "sample" for s in r["samples"])]
        if sampled:
            n = max(sum(1 for s in r["samples"] if s["mode"] == "sample")
                    for r in sampled)
            for rule in ("strict", "loose"):
                for k in ks:
                    if k > n:
                        continue
                    vals = []
                    for r in sampled:
                        ss = [s for s in r["samples"] if s["mode"] == "sample"]
                        c = sum(1 for s in ss if s[rule])
                        vals.append(pass_at_k(len(ss), c, k))
                    out[f"pass@{k}_{rule}"] = sum(vals) / len(vals)
            out["n_samples"] = n
        out["problems"] = len(rows)
        boxed = [s for r in rows for s in r["samples"]]
        if boxed:
            out["boxed_rate"] = sum(1 for s in boxed if s["boxed"]) / len(boxed)
            out["avg_turns"] = sum(s["turns"] for s in boxed) / len(boxed)
            out["tool_ok_per_traj"] = sum(s["tool_ok"] for s in boxed) / len(boxed)
            # how much the context window interfered: a low score caused by
            # truncation is a harness limit, not a capability measurement
            cuts = [s.get("cut", "") for s in boxed]
            out["cut_context"] = sum(1 for c in cuts if c == "context") / len(boxed)
            out["cut_budget"] = sum(1 for c in cuts if c == "budget") / len(boxed)
            out["cut_rejected"] = sum(1 for c in cuts if c == "rejected") / len(boxed)
        return out

    report = {"overall": block(results)}
    by_sub = defaultdict(list)
    for r in results:
        by_sub[r.get("subject", "?")].append(r)
    report["by_subject"] = {k: block(v) for k, v in sorted(by_sub.items())}
    return report


def print_report(rep: dict) -> None:
    o = rep["overall"]
    print("\n" + "=" * 68)
    print(f"  MATH-500 评测 · {o['problems']} 题")
    print("=" * 68)
    if "greedy_strict" in o:
        print(f"  贪心解码   严格 {o['greedy_strict']*100:6.2f}%   "
              f"宽口径 {o['greedy_loose']*100:6.2f}%")
    if "n_samples" in o:
        print(f"  采样 n={o['n_samples']}")
        ks = sorted({int(k.split("@")[1].split("_")[0])
                     for k in o if k.startswith("pass@")})
        for k in ks:
            s = o.get(f"pass@{k}_strict")
            l = o.get(f"pass@{k}_loose")
            if s is not None:
                print(f"    pass@{k:<2}  严格 {s*100:6.2f}%   宽口径 {l*100:6.2f}%")
    print(f"  出框率 {o.get('boxed_rate',0)*100:.1f}% · "
          f"平均轮数 {o.get('avg_turns',0):.2f} · "
          f"平均成功工具调用 {o.get('tool_ok_per_traj',0):.2f}")
    cc, cb, cr = (o.get("cut_context", 0), o.get("cut_budget", 0),
                  o.get("cut_rejected", 0))
    if cc or cb or cr:
        print(f"  长度受限: 上下文耗尽 {cc*100:.1f}% · 生成被压缩 {cb*100:.1f}% · "
              f"被拒 {cr*100:.1f}%  (偏高说明 --max-model-len 需加大)")
    print("-" * 68)
    print(f"  {'学科':<24}{'题数':>5}{'贪心严格':>10}{'pass@1宽':>10}")
    for sub, b in rep["by_subject"].items():
        g = b.get("greedy_strict")
        p1 = b.get("pass@1_loose")
        gs = f"{g*100:.1f}%" if g is not None else "-"
        ps = f"{p1*100:.1f}%" if p1 is not None else "-"
        print(f"  {sub:<24}{b['problems']:>5}{gs:>10}{ps:>10}")
    print("=" * 68)


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="trained checkpoint directory")
    ap.add_argument("--pool", default="agent_data_math/math500.jsonl")
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--gpus", default=None, help="comma list; default = all visible")
    ap.add_argument("--limit", type=int, default=0, help="only the first N problems")
    ap.add_argument("--samples", type=int, default=8, help="n for pass@k (0 = skip)")
    ap.add_argument("--no-greedy", dest="greedy", action="store_false")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed; keep it identical across checkpoints so "
                         "pass@k differences are the model, not sampling noise")
    ap.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    ap.add_argument("--max-new", type=int, default=1024, dest="max_new")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS, dest="max_turns",
                    help=f"assistant turns per trajectory (default {MAX_TURNS})")
    ap.add_argument("--max-model-len", type=int, default=4096, dest="max_model_len")
    ap.add_argument("--min-new", type=int, default=32, dest="min_new",
                    help="stop a trajectory when fewer than this many tokens of "
                         "context remain, instead of overrunning the window")
    ap.add_argument("--gpu-util", type=float, default=0.85, dest="gpu_util")
    ap.add_argument("--batch", type=int, default=256,
                    help="prompts per generate call (wavefront chunk); vLLM "
                         "schedules them concurrently, so bigger is faster")
    ap.add_argument("--max-num-seqs", type=int, default=256, dest="max_num_seqs",
                    help="vLLM concurrent sequence cap")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--ks", default="1,2,4,8", help="which pass@k to report")
    ap.add_argument("--judge-api", action="store_true", dest="judge_api",
                    help="arbitrate unresolved answers with an LLM API")
    ap.add_argument("--judge-model", default="deepseek-chat", dest="judge_model")
    ap.add_argument("--judge-base-url", default=None, dest="judge_base_url")
    ap.add_argument("--judge-key-env", default="DEEPSEEK_API_KEY", dest="judge_key_env")
    ap.add_argument("--judge-workers", type=int, default=16, dest="judge_workers",
                    help="concurrent judge API calls per GPU worker")
    ap.add_argument("--judge-retries", type=int, default=2, dest="judge_retries",
                    help="retries per judge call on a transient failure")
    ap.add_argument("--judge-timeout", type=float, default=30.0, dest="judge_timeout")
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    return ap.parse_args()


def main() -> int:
    args = build_args()
    if args.self_test:
        _self_test()
        return 0
    if not args.model:
        raise SystemExit("--model is required (a trained checkpoint directory)")

    import multiprocessing as mp
    # Check the judge credential first: it is a configuration mistake the user can
    # fix immediately, and it must not surface only after the engines have loaded.
    if args.judge_api and not os.environ.get(args.judge_key_env, ""):
        raise SystemExit(
            f"--judge-api 需要环境变量 {args.judge_key_env}，当前未设置。\n"
            f"  设置后重跑，或去掉 --judge-api 用纯规则判分（但三个模型必须一致）。")
    rows = [json.loads(l) for l in open(args.pool, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    gpus = ([int(x) for x in args.gpus.split(",")] if args.gpus else detect_gpus())
    if not gpus:
        raise SystemExit("no CUDA device visible; run on a GPU machine "
                         "or pass --gpus")
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    if args.samples:
        ks = [k for k in ks if k <= args.samples]

    # Clamp before spawning so every worker builds the engine with a window the
    # checkpoint actually supports; vLLM rejects an over-long request outright.
    args.max_model_len, note = resolve_max_len(args.max_model_len, args.model)
    if note:
        print(f"  ! {note}", flush=True)
    if args.max_new >= args.max_model_len:
        args.max_new = max(64, args.max_model_len // 2)
        print(f"  ! --max-new 超过窗口，已降为 {args.max_new}", flush=True)

    shards = [rows[i::len(gpus)] for i in range(len(gpus))]
    print(f"  eval: {len(rows)} 题 · {len(gpus)} 卡 {gpus} · "
          f"greedy={args.greedy} · n={args.samples} · T={args.temperature}", flush=True)
    print(f"  model: {args.model} · 上下文窗口 {args.max_model_len} · "
          f"每轮最多 {args.max_new} token · 最多 {args.max_turns} 轮", flush=True)
    if args.judge_api:
        print(f"  宽口径判分: 规则 + {args.judge_model} 仲裁 · "
              f"并发 {args.judge_workers} · 答案串将发送至外部 API", flush=True)
    else:
        print("  宽口径判分: 仅规则（未启用 API 仲裁）", flush=True)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for gpu, shard in zip(gpus, shards):
        if not shard:
            continue
        p = ctx.Process(target=_worker, args=(gpu, shard, args, queue))
        p.start()
        procs.append(p)

    results, failed, judge_stats = [], [], []
    for _ in procs:
        msg = queue.get()
        kind, gpu, payload = msg[0], msg[1], msg[2]
        if kind == "ok":
            results.extend(payload)
            if len(msg) > 3 and msg[3]:
                judge_stats.append(msg[3])
            print(f"  [gpu{gpu}] 完成 {len(payload)} 题", flush=True)
        else:
            failed.append((gpu, payload))
            print(f"  [gpu{gpu}] 失败:\n{payload}", flush=True)
    for p in procs:
        p.join()
    if failed and not results:
        raise SystemExit("all workers failed")

    rep = summarise(results, ks)
    rep["config"] = {"model": args.model, "pool": args.pool, "gpus": gpus,
                     "samples": args.samples, "temperature": args.temperature,
                     "seed": args.seed, "top_p": args.top_p,
                     "max_model_len": args.max_model_len, "max_new": args.max_new,
                     "greedy": args.greedy, "max_turns": args.max_turns,
                     "judge_api": bool(args.judge_api),
                     "judge_model": args.judge_model if args.judge_api else None,
                     "judge_workers": args.judge_workers,
                     "judge_stats": judge_stats,
                     # what actually happened, not just what was asked for: the
                     # comparison checks this so a silent fallback cannot pass
                     "judge_used": bool(judge_stats),
                     "judge_arbitrated": sum(
                         1 for r in results for s in r["samples"] if s.get("arbitrated")),
                     "judge_flipped": sum(s.get("flipped", 0) for s in judge_stats),
                     "failed_gpus": [g for g, _ in failed]}
    os.makedirs(args.out, exist_ok=True)
    rp = os.path.join(args.out, "report.json")
    dp = os.path.join(args.out, "details.jsonl")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(dp, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print_report(rep)
    print(f"  报告 -> {rp}\n  逐题明细 -> {dp}")
    return 0


# --------------------------------------------------------------------- self test
def _self_test() -> None:
    # strict vs loose: formatting differs, value identical
    assert strict_equal("5", "5")
    assert not strict_equal("0.5", "\\frac{1}{2}")
    assert loose_equal("0.5", "\\frac{1}{2}")
    assert loose_equal("2", "\\sqrt{4}")
    assert not loose_equal("3", "5")
    assert not strict_equal(None, "5")

    v = score_one("thus \\boxed{0.5}", "\\frac{1}{2}")
    assert v["strict"] is False and v["loose"] is True and v["boxed"]
    v2 = score_one("no box", "5")
    assert not v2["boxed"] and not v2["strict"] and not v2["loose"]

    # pass@k estimator: known closed forms
    assert pass_at_k(8, 0, 1) == 0.0
    assert pass_at_k(8, 8, 1) == 1.0
    assert abs(pass_at_k(8, 1, 1) - 1 / 8) < 1e-12
    assert abs(pass_at_k(8, 1, 2) - 2 / 8) < 1e-12       # 1 - C(7,2)/C(8,2)
    assert pass_at_k(8, 7, 2) == 1.0                      # n-c=1 < k=2
    assert abs(pass_at_k(4, 2, 2) - (1 - 1 / 6)) < 1e-12  # C(2,2)/C(4,2)=1/6
    for k in (1, 2, 4, 8):
        assert 0.0 <= pass_at_k(8, 3, k) <= 1.0
    assert pass_at_k(8, 3, 1) < pass_at_k(8, 3, 4)        # monotone in k
    try:
        pass_at_k(2, 1, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("k > n was accepted")

    # report: one problem right in greedy, and 1/2 correct samples
    res = [{"uid": "a", "subject": "Algebra", "level": 1, "gold": "5",
            "samples": [{"mode": "greedy", "strict": True, "loose": True,
                         "boxed": True, "turns": 1, "tool_ok": 0},
                        {"mode": "sample", "strict": True, "loose": True,
                         "boxed": True, "turns": 2, "tool_ok": 1},
                        {"mode": "sample", "strict": False, "loose": False,
                         "boxed": True, "turns": 2, "tool_ok": 1}]},
           {"uid": "b", "subject": "Geometry", "level": 2, "gold": "7",
            "samples": [{"mode": "greedy", "strict": False, "loose": False,
                         "boxed": True, "turns": 1, "tool_ok": 0},
                        {"mode": "sample", "strict": False, "loose": False,
                         "boxed": True, "turns": 1, "tool_ok": 0},
                        {"mode": "sample", "strict": False, "loose": False,
                         "boxed": True, "turns": 1, "tool_ok": 0}]}]
    rep = summarise(res, [1, 2])
    o = rep["overall"]
    assert abs(o["greedy_strict"] - 0.5) < 1e-9
    assert abs(o["pass@1_strict"] - 0.25) < 1e-9        # (0.5 + 0)/2
    assert abs(o["pass@2_strict"] - 0.5) < 1e-9         # (1.0 + 0)/2
    assert o["problems"] == 2 and "Algebra" in rep["by_subject"]
    assert abs(o["avg_turns"] - (1 + 2 + 2 + 1 + 1 + 1) / 6) < 1e-9

    # the judge stays off without a key, and never invents a verdict
    j = LLMJudge("m", api_key_env="__NO_SUCH_KEY__")
    assert not j.available and j.equivalent("1", "1") is None
    assert j.equivalent_many([("1", "1")]) == {}

    # concurrent batch: dedup, real overlap, and failures keep the rule verdict
    import threading
    import time as _time

    class _J(LLMJudge):
        def __init__(self):
            super().__init__("m", api_key_env="__NO_SUCH_KEY__", workers=4)
            self.api_key = "fake"
            self.seen = []
            self.live = 0
            self.peak = 0
            self.lock = threading.Lock()

        def _ask(self, pred, gold):
            with self.lock:
                self.seen.append((pred, gold))
                self.live += 1
                self.peak = max(self.peak, self.live)
            _time.sleep(0.02)
            with self.lock:
                self.live -= 1
            if pred == "boom":
                raise RuntimeError("api down")
            return pred == gold

    jj = _J()
    pairs = [("5", "5"), ("5", "5"), ("6", "5"), ("boom", "5")]
    got = jj.equivalent_many(pairs)
    assert len(jj.seen) == 3, jj.seen            # the duplicate was asked once
    assert jj.peak > 1, jj.peak                  # calls really overlapped
    assert got[("5", "5")] is True and got[("6", "5")] is False
    assert ("boom", "5") not in got              # failure -> no verdict recorded
    assert jj.errors == 1

    # arbitrate flips only the unsettled ones, and never touches strict
    res2 = [{"gold": "5", "samples": [
        {"answer": "5", "loose": True, "strict": True},         # rules already ok
        {"answer": "five", "loose": False, "strict": False},    # judge says yes
        {"answer": "six", "loose": False, "strict": False},     # judge says no
        {"answer": "five", "loose": False, "strict": False},    # duplicate pair
        {"answer": None, "loose": False, "strict": False}]}]    # no box: not sent

    class _J2(_J):
        def _ask(self, pred, gold):
            return pred == "five"

    st = arbitrate(res2, _J2())
    ss = res2[0]["samples"]
    assert ss[0]["loose"] is True and not ss[0].get("arbitrated")
    assert ss[1]["loose"] is True and ss[1]["arbitrated"]
    assert ss[2]["loose"] is False and ss[2]["arbitrated"]
    assert ss[3]["loose"] is True and ss[3]["arbitrated"]      # same pair, same verdict
    assert ss[4]["loose"] is False and not ss[4].get("arbitrated")
    assert st["asked"] == 2 and st["flipped"] == 2, st         # deduped to 2 calls
    assert all(s["strict"] is False for s in ss[1:]), "strict must not be touched"

    # the window is clamped to what the checkpoint declares, not to a sibling's
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"max_position_embeddings": 4096}, f)
        assert model_max_len(d) == 4096
        got, note = resolve_max_len(8192, d)
        assert got == 4096 and note, (got, note)      # Math variant: clamped
        got2, note2 = resolve_max_len(2048, d)
        assert got2 == 2048 and not note2             # smaller request kept
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"max_position_embeddings": 32768}, f)
        got3, note3 = resolve_max_len(8192, d)
        assert got3 == 8192 and not note3             # Instruct variant: allowed
    assert model_max_len("/no/such/dir") is None
    got4, note4 = resolve_max_len(8192, "/no/such/dir")
    assert got4 == 8192 and not note4                 # unknown: leave it alone

    assert detect_gpus.__doc__
    print("  eval self-test OK")
    print("    strict vs loose split; unbiased pass@k against closed forms")
    print("    report aggregation, per-subject split, judge off without a key")
    print("    judge batch: deduped, concurrent, failures keep the rule verdict")
    print("    context window clamped to the checkpoint's own config")


def _run_shard(shard, args, generate, tok, fmt, run_code):
    """The wavefront scheduler, factored out so it can be tested without a GPU.

    Returns the same `results` structure the worker builds, and the number of
    `generate` calls made - the whole point of batching is that this number is
    turns, not trajectories.
    """
    from zero_ppo.rollout.multiturn import CODE, BOX
    from zero_ppo.util.prompts import opening_messages, render

    calls = {"n": 0, "sizes": [], "truncated": 0, "dropped": 0}

    def gen(prompts, temp, n, max_tokens, seeds=None):
        calls["n"] += 1
        calls["sizes"].append(len(prompts))
        if seeds is None:
            return generate(prompts, temp, n, max_tokens)
        return generate(prompts, temp, n, max_tokens, seeds=seeds)

    # The context window is shared by the WHOLE conversation, not one turn: six
    # turns of 1024 new tokens each overrun a 4096 window and vLLM rejects the
    # request outright, which used to abort the entire GPU shard. Each turn now
    # gets only the room actually left, and a trajectory with no room stops
    # instead of raising.
    limit = int(getattr(args, "max_model_len", 4096))
    reserve = int(getattr(args, "min_new", 32))

    def ntok(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    results = []
    modes = ([("greedy", 0.0, 1)] if args.greedy else []) + \
            ([("sample", args.temperature, args.samples)] if args.samples else [])
    for mi, (mode, temp, n) in enumerate(modes):
        trajs = []
        for pi, rec in enumerate(shard):
            for si in range(n):
                msgs = opening_messages(rec)
                convo = render(tok, msgs, add_generation_prompt=True)
                # Distinct per (problem, sample) so the n samples really differ,
                # yet identical across checkpoints so the comparison is paired.
                # Greedy needs no seed variation (temperature 0 is deterministic).
                sd = (int(getattr(args, "seed", 0)) * 1000003 + pi * 1009 + si
                      if mode == "sample" else int(getattr(args, "seed", 0)))
                trajs.append({"pi": pi, "si": si, "msgs": msgs, "convo": convo,
                              "ntok": ntok(convo), "texts": [], "turns": 0,
                              "ok": 0, "live": True, "cut": "", "seed": sd})
        for _turn in range(args.max_turns):
            live = [t for t in trajs if t["live"]]
            if not live:
                break
            # budget per trajectory; those that cannot fit even a short reply end
            ready = []
            for t in live:
                room = limit - t["ntok"]
                if room < reserve:
                    t["live"] = False
                    t["cut"] = "context"
                    calls["dropped"] += 1
                    continue
                t["budget"] = min(int(args.max_new), room)
                ready.append(t)
            if not ready:
                break
            step = max(1, int(args.batch))
            for s in range(0, len(ready), step):
                part = ready[s:s + step]
                # one call per distinct budget keeps each request within its own
                # remaining window (vLLM takes a single max_tokens per call)
                by_budget = {}
                for t in part:
                    by_budget.setdefault(t["budget"], []).append(t)
                for budget, group in by_budget.items():
                    try:
                        outs = gen([t["convo"] for t in group], temp, 1, budget,
                                   seeds=[t["seed"] for t in group])
                    except Exception as e:
                        # A rejected request must not discard the whole shard.
                        # Fall back to one-at-a-time so only the offending
                        # trajectories are lost, and say why.
                        msg = str(e)
                        if "context length" in msg or "input_tokens" in msg:
                            calls["rejected"] = calls.get("rejected", 0) + len(group)
                            print(f"  ! {len(group)} 条超长被拒，逐条重试: "
                                  f"{msg.splitlines()[0][:100]}", flush=True)
                            outs = []
                            for t in group:
                                try:
                                    outs.append(gen([t["convo"]], temp, 1, budget,
                                                    seeds=[t["seed"]])[0])
                                except Exception:
                                    t["live"] = False
                                    t["cut"] = "rejected"
                                    outs.append([{"text": "", "tokens": []}])
                        else:
                            raise
                    for t, got in zip(group, outs):
                        text = got[0]["text"]
                        t["texts"].append(text)
                        t["turns"] += 1
                        t["ntok"] += len(got[0].get("tokens") or []) or ntok(text)
                        if budget < int(args.max_new):
                            t["cut"] = "budget"
                            calls["truncated"] += 1
                        blocks = CODE.findall(text)
                        if not blocks:
                            t["live"] = False    # reasoned without the tool
                            continue
                        # Execute FIRST, then check for an answer - the training
                        # rollout does it in this order, so a turn carrying both
                        # code and \boxed{} still counts as a successful tool call.
                        out, err = run_code(blocks[-1].strip())
                        if out is not None:
                            t["ok"] += 1
                        if BOX.search(text):
                            t["live"] = False    # answered: no extra tool turn
                            continue
                        msg = ("<output>" + (out if out is not None else
                               "ERROR: " + str(err).splitlines()[0][:80]) + "</output>")
                        t["msgs"], tail = fmt.extend(t["msgs"], text, msg)
                        t["convo"] = t["convo"] + tail
                        t["ntok"] += ntok(tail)
        if mi == 0:
            for rec in shard:
                results.append({"uid": rec.get("uid", ""),
                                "subject": rec.get("subject", ""),
                                "level": rec.get("level", 0),
                                "gold": str(rec.get("gold", "")), "samples": []})
        for t in trajs:
            v = score_one(t["texts"][-1] if t["texts"] else "",
                          str(shard[t["pi"]].get("gold", "")))
            v.update({"mode": mode, "sample": t["si"], "turns": t["turns"],
                      "tool_ok": t["ok"], "texts": t["texts"], "cut": t["cut"]})
            results[t["pi"]]["samples"].append(v)
    return results, calls


def _self_test_batch() -> None:
    """Batched scheduling must give the same verdicts as one-at-a-time."""
    import types
    from zero_ppo.util.prompts import ToolFormat
    from zero_ppo.rollout.tools import fake_runner

    class _Tok:
        chat_template = "yes"

        def apply_chat_template(self, messages, add_generation_prompt=True,
                                tokenize=False):
            s = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
            return s + ("<|assistant|>" if add_generation_prompt else "")

        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [1] * len(s)}      # char-level, for budget maths

    tok = _Tok()
    fmt = ToolFormat(tok)
    FENCE = "```python\nprint(7)\n```"
    BOXED = "\\" + "boxed{7}"
    shard = [{"uid": f"u{i}", "subject": "Algebra", "level": 1,
              "gold": "7", "system": "sys", "question": f"problem {i}"}
             for i in range(5)]

    def make_gen():
        # every trajectory: tool call, then the boxed answer
        state = {}

        def generate(prompts, temp, n, max_tokens, seeds=None):
            out = []
            for p in prompts:
                k = p.count("<|assistant|>")
                text = FENCE if k <= 1 else BOXED
                out.append([{"text": text, "tokens": [1]}])
            return out
        return generate

    runner = fake_runner({"print": ("7", "")})
    base = dict(greedy=True, samples=3, temperature=0.7, max_turns=4,
                max_new=64, batch=256)
    big, cb = _run_shard(shard, types.SimpleNamespace(**base), make_gen(),
                         tok, fmt, runner)
    small, cs = _run_shard(shard, types.SimpleNamespace(**{**base, "batch": 1}),
                           make_gen(), tok, fmt, runner)

    # identical verdicts, ordering and per-sample metadata
    assert len(big) == len(small) == 5
    for a, b in zip(big, small):
        assert a["uid"] == b["uid"] and len(a["samples"]) == len(b["samples"]) == 4
        for x, y in zip(a["samples"], b["samples"]):
            for k in ("mode", "sample", "strict", "loose", "turns", "tool_ok", "texts"):
                assert x[k] == y[k], (k, x[k], y[k])
    # batching really happened: far fewer calls, and they carried many prompts
    assert cb["n"] < cs["n"], (cb["n"], cs["n"])
    assert max(cb["sizes"]) > 1, cb["sizes"]
    assert all(s["strict"] and s["loose"] for r in big for s in r["samples"])

    # a turn carrying BOTH code and the answer must still execute the code, as the
    # training rollout does (execute first, then notice the box)
    calls_made = []

    def both(prompts, temp, n, max_tokens, seeds=None):
        return [[{"text": FENCE + " so " + BOXED, "tokens": [1]}] for _ in prompts]

    def spy(code):
        calls_made.append(code)
        return ("7", "")

    one, _ = _run_shard(shard[:1], types.SimpleNamespace(**{**base, "samples": 0}),
                        both, tok, fmt, spy)
    s0 = one[0]["samples"][0]
    assert len(calls_made) == 1, calls_made      # the program really ran
    assert s0["tool_ok"] == 1 and s0["turns"] == 1, s0
    assert s0["strict"] and s0["loose"]

    # a failing program is reported back and the model may retry
    seq = ["```python\nprint(1/0)\n```", FENCE, BOXED]

    def retry(prompts, temp, n, max_tokens, seeds=None):
        out = []
        for p in prompts:
            k = p.count("<|assistant|>") - 1
            out.append([{"text": seq[min(k, len(seq) - 1)], "tokens": [1]}])
        return out

    rr, _ = _run_shard(shard[:1], types.SimpleNamespace(**{**base, "samples": 0}),
                       retry, tok, fmt, fake_runner({"1/0": (None, "ZeroDivisionError"),
                                                     "print": ("7", "")}))
    s1 = rr[0]["samples"][0]
    assert s1["turns"] == 3 and s1["tool_ok"] == 1, s1
    assert s1["strict"] and s1["loose"]

    # never answering: capped at max_turns, scored wrong, no infinite loop
    def loop(prompts, temp, n, max_tokens, seeds=None):
        return [[{"text": FENCE, "tokens": [1]}] for _ in prompts]

    rl, _ = _run_shard(shard[:1], types.SimpleNamespace(**{**base, "samples": 0}),
                       loop, tok, fmt, spy)
    s2 = rl[0]["samples"][0]
    assert s2["turns"] == base["max_turns"] and not s2["boxed"], s2

    # the shared cap is 6 turns, and it binds even if a caller asks for more
    from zero_ppo.rollout.multiturn import MAX_TURNS
    assert MAX_TURNS == 6, MAX_TURNS
    import argparse as _ap
    for mod in ("zero_ppo.eval", "zero_ppo.driver", "zero_ppo.main"):
        import importlib
        m = importlib.import_module(mod)
        src = open(m.__file__, encoding="utf-8").read()
        assert "default=MAX_TURNS" in src, f"{mod} does not use the shared cap"
    capped, _ = _run_shard(shard[:1],
                           types.SimpleNamespace(**{**base, "samples": 0,
                                                    "max_turns": MAX_TURNS}),
                           loop, tok, fmt, spy)
    sc = capped[0]["samples"][0]
    assert sc["turns"] == 6 and not sc["boxed"], sc
    # a trajectory that answers on turn 6 is still scored, not discarded
    late = ["```python\nprint(7)\n```"] * 5 + [BOXED]

    def late_answer(prompts, temp, n, max_tokens, seeds=None):
        out = []
        for p in prompts:
            k = p.count("<|assistant|>") - 1
            out.append([{"text": late[min(k, len(late) - 1)], "tokens": [1]}])
        return out

    rl6, _ = _run_shard(shard[:1],
                        types.SimpleNamespace(**{**base, "samples": 0,
                                                 "max_turns": MAX_TURNS}),
                        late_answer, tok, fmt, spy)
    s6 = rl6[0]["samples"][0]
    assert s6["turns"] == 6 and s6["boxed"] and s6["strict"], s6

    # context budget: a long conversation must be clipped, never raise
    class _LenTok(_Tok):
        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [1] * len(s)}      # 1 char = 1 token

    lt = _LenTok()
    lfmt = ToolFormat(lt)
    long_shard = [{"uid": "L", "subject": "Algebra", "level": 1, "gold": "7",
                   "system": "s" * 200, "question": "q" * 200}]
    seen_budgets = []

    def greedy_long(prompts, temp, n, max_tokens, seeds=None):
        seen_budgets.append(max_tokens)
        return [[{"text": FENCE, "tokens": [1] * 60}] for _ in prompts]

    largs = types.SimpleNamespace(greedy=True, samples=0, temperature=0.0,
                                  max_turns=6, max_new=1024, batch=256,
                                  max_model_len=600, min_new=32)
    lr_, lc = _run_shard(long_shard, largs, greedy_long, lt, lfmt, spy)
    sl = lr_[0]["samples"][0]
    assert all(b <= 600 for b in seen_budgets), seen_budgets
    assert any(b < 1024 for b in seen_budgets), seen_budgets   # clipping happened
    assert sl["turns"] < 6 and sl["cut"], sl                   # stopped early
    assert lc["dropped"] or lc["truncated"], lc
    # every request stayed inside the window: prompt + max_tokens <= limit
    print(f"    context guard: budgets={seen_budgets[:4]}... "
          f"turns={sl['turns']} cut={sl['cut']!r}")

    # a rejected request loses only its own trajectory, not the shard
    def reject(prompts, temp, n, max_tokens, seeds=None):
        if len(prompts) > 1:
            raise RuntimeError("This model's maximum context length is 4096 tokens "
                               "(parameter=input_tokens, value=4097)")
        return [[{"text": BOXED, "tokens": [1]}] for _ in prompts]

    rj, jc = _run_shard(shard[:2], types.SimpleNamespace(**{**base, "samples": 0}),
                        reject, tok, fmt, spy)
    assert len(rj) == 2, rj                     # both problems still reported
    assert jc.get("rejected", 0) >= 1, jc
    assert any(s["loose"] for r in rj for s in r["samples"])

    # per-sample seeds: one shared seed made all n samples identical and collapsed
    # pass@k onto pass@1, silently destroying the measurement
    seen_seeds = []

    def seed_spy(prompts, temp, n, max_tokens, seeds=None):
        seen_seeds.append(list(seeds) if seeds is not None else None)
        return [[{"text": BOXED, "tokens": [1]}] for _ in prompts]

    _run_shard(shard[:2], types.SimpleNamespace(**{**base, "samples": 4,
                                                 "greedy": False, "seed": 7}),
               seed_spy, tok, fmt, spy)
    flat = [s for call in seen_seeds if call for s in call]
    assert flat and None not in seen_seeds, seen_seeds
    assert len(set(flat)) == len(flat), f"seeds must differ per sample: {flat}"
    # reproducible: the same args give the same seeds
    seen_seeds.clear()
    _run_shard(shard[:2], types.SimpleNamespace(**{**base, "samples": 4,
                                                 "greedy": False, "seed": 7}),
               seed_spy, tok, fmt, spy)
    flat2 = [s for call in seen_seeds if call for s in call]
    assert flat == flat2, "seeds must be reproducible across runs"
    # a different --seed shifts them
    seen_seeds.clear()
    _run_shard(shard[:2], types.SimpleNamespace(**{**base, "samples": 4,
                                                 "greedy": False, "seed": 8}),
               seed_spy, tok, fmt, spy)
    flat3 = [s for call in seen_seeds if call for s in call]
    assert set(flat3).isdisjoint(set(flat)), "a new seed must change the samples"

    print("  eval batching self-test OK")
    print(f"    same verdicts; generate calls {cs['n']} (batch=1) -> {cb['n']} "
          f"(batch=256), max prompts/call {max(cb['sizes'])}")
    print(f"    per-sample seeds distinct ({len(set(flat))} unique) and reproducible")
    print("    code+answer in one turn still runs the code; retry verified")
    print(f"    turn cap {MAX_TURNS} shared by train/eval; a turn-{MAX_TURNS} answer still scores")
    print("    context budget clips generation; a rejected prompt cannot kill the shard")


if __name__ == "__main__":
    a = build_args()
    if a.self_test:
        _self_test()
        _self_test_batch()
        raise SystemExit(0)
    raise SystemExit(main())