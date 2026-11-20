#!/usr/bin/env python3
"""The Ray driver: one cluster, one process group, colocated train + inference.

This replaces the torchrun + HTTP-service orchestration. The shape is verl's:

    driver            ray.init() once, owns the schedule
      TrainActor x N  num_gpus=1 each, Actor + DeepSpeed ZeRO-2, one torch group
      EngineActor x1  num_gpus=0 (devices pinned by env), vLLM TP=N, five verbs

Per step: wake -> rollout (rank 0) -> broadcast rows -> reward + advantage ->
sleep -> backward/step on every rank -> pause -> send_weights on every rank ->
resume. The weight channel is opened once before the loop; each sync is a single
`send_weights`, which is what keeps the engine's state machine in step.

Colocation note: Ray must not book the cards twice. The trainer actors take one
GPU each; the engine actor takes none and pins its devices through
CUDA_VISIBLE_DEVICES, so on N cards the cluster needs N GPUs, not 2N.

    python3 -m zero_ppo.driver --self-test          # schedule logic, no GPU/Ray
    python3 -m zero_ppo.driver --model ... --pool ... --iters 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero_ppo.rollout.multiturn import MAX_TURNS


def step_problems(problems, step: int, batch: int, group: int):
    """The batch for one step, expanded to group samples per problem."""
    take = problems[(step * batch) % max(1, len(problems)):][:batch]
    expanded = [q for q in take for _ in range(group)]
    golds = [str(q.get("gold", "")) for q in expanded]
    return expanded, golds


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--algo", default="ppo", choices=["ppo", "grpo"])
    ap.add_argument("--reward", default="zero_ppo/reward/ppo_agent_test.py",
                    help="ppo_agent_test.py for PPO (process reward); "
                         "agent_reward_test.py for GRPO (outcome only)")
    ap.add_argument("--critic-mode", default="full", dest="critic_mode",
                    choices=["none", "full", "lora_shared", "lora_split"])
    ap.add_argument("--critic-lr", type=float, default=1e-5, dest="critic_lr",
                    help="critic learning rate (0 = reuse --lr)")
    ap.add_argument("--vf-coef", type=float, default=0.5, dest="vf_coef",
                    help="value loss weight when actor and critic share an optimiser")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95,
                    help="GAE lambda; credit reaches ~1/(1-gamma*lam) tokens")
    ap.add_argument("--ppo-epochs", type=int, default=4, dest="ppo_epochs",
                    help="optimiser steps per rollout batch (PPO inner loop); "
                         "1 makes the ratio identically one, i.e. plain policy gradient")
    ap.add_argument("--clip-eps", type=float, default=0.2, dest="clip_eps",
                    help="PPO trust region on the importance ratio")
    ap.add_argument("--kl-coef", type=float, default=0.02, dest="kl_coef",
                    help="penalty on KL(policy || frozen SFT reference); 0 disables "
                         "the reference model and saves its memory")
    ap.add_argument("--mini-batch", type=int, default=0, dest="mini_batch",
                    help="trajectories per forward pass; the batch is split and "
                         "gradients accumulated, so the update is unchanged but "
                         "activation memory drops (0 = no split)")
    ap.add_argument("--no-whiten", dest="whiten", action="store_false",
                    help="keep raw GAE advantages (their scale then follows the "
                         "reward scale, which is what destabilised the last run)")
    ap.add_argument("--signal-ref", type=float, default=0.5, dest="signal_ref",
                    help="weak-signal attenuation reference for whitening: batches "
                         "whose raw mean|A| is below this keep a proportionally "
                         "weaker gradient instead of being promoted to full "
                         "strength (0 disables, restoring plain whitening)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--shuffle", type=int, default=0,
                    help="shuffle the training pool once at startup (fixed seed). "
                         "The pool is ordered hard-then-easy, so unshuffled "
                         "sequential batches make reward oscillate with the "
                         "curriculum instead of showing learning")
    ap.add_argument("--shuffle-seed", type=int, default=0, dest="shuffle_seed")
    ap.add_argument("--luffy", type=int, default=0,
                    help="per group of GROUP samples, this many trajectories come "
                         "from the external API model instead of the policy "
                         "(LUFFY-style off-policy guidance). They run the same "
                         "tool loop and sandbox; the update on them is "
                         "advantage-weighted imitation. 0 disables")
    ap.add_argument("--luffy-model", default=None, dest="luffy_model",
                    help="external model name (default env LUFFY_MODEL or deepseek-chat)")
    ap.add_argument("--luffy-base-url", default=None, dest="luffy_base_url")
    ap.add_argument("--luffy-key-env", default="DEEPSEEK_API_KEY", dest="luffy_key_env")
    ap.add_argument("--luffy-max-tokens", type=int, default=1024, dest="luffy_max_tokens")
    ap.add_argument("--luffy-workers", type=int, default=8, dest="luffy_workers")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--target-kl", type=float, default=0.02, dest="target_kl",
                    help="stop the inner epochs once approx_kl exceeds 1.5x this. "
                         "Measures drift from the policy that GENERATED the batch "
                         "(resets每rollout), not from the frozen SFT reference. "
                         "The clip bounds single tokens; this bounds the sum. 0 disables")
    ap.add_argument("--warmup-frac", type=float, default=0.1, dest="warmup_frac",
                    help="fraction of steps for linear lr warmup, then cosine decay")
    ap.add_argument("--min-lr-frac", type=float, default=0.0, dest="min_lr_frac",
                    help="floor the cosine at this fraction of the base lr, as "
                         "open-r1's cosine_with_min_lr / min_lr_rate does (0.1 "
                         "there). 0 keeps the previous decay-to-zero")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--world", type=int, default=2, help="cards = trainer ranks = TP")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=1024, dest="max_new")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS, dest="max_turns",
                    help=f"assistant turns per trajectory (default {MAX_TURNS})")
    ap.add_argument("--mt-tokens", type=int, default=1800, dest="mt_tokens")
    ap.add_argument("--rollout-chunk", type=int, default=8, dest="rollout_chunk")
    ap.add_argument("--max-model-len", type=int, default=4096, dest="max_model_len")
    ap.add_argument("--vllm-util", type=float, default=0.30, dest="vllm_util")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--prefix-cache", type=int, default=1, dest="prefix_cache")
    ap.add_argument("--tuning", default="full", choices=["full", "lora"])
    ap.add_argument("--gradient-checkpoint", type=int, default=1, dest="gradient_checkpoint")
    ap.add_argument("--lora-r", type=int, default=16, dest="lora_r")
    ap.add_argument("--lora-alpha", type=int, default=32, dest="lora_alpha")
    ap.add_argument("--lora-targets", default="all", dest="lora_targets",
                    choices=["attn", "all"])
    ap.add_argument("--deepspeed", type=int, default=1)
    ap.add_argument("--packed", type=int, default=0, help="chunked IPC transfer")
    ap.add_argument("--out", default="runs/zero_ppo")
    ap.add_argument("--save-dir", default=None, dest="save_dir",
                    help="where to write the trained HF model (default: <out>/final)")
    ap.add_argument("--eval-every", type=int, default=0, dest="eval_every",
                    help="run a greedy held-out eval every N steps on the "
                         "COLOCATED engine (no second vLLM instance) and keep the "
                         "best checkpoint. 0 disables")
    ap.add_argument("--eval-pool", default="agent_data_math/math500.jsonl",
                    dest="eval_pool", help="held-out set for the inline eval")
    ap.add_argument("--eval-limit", type=int, default=0, dest="eval_limit",
                    help="evaluate only the first N problems (0 = all)")
    ap.add_argument("--eval-batch", type=int, default=64, dest="eval_batch")
    ap.add_argument("--eval-max-new", type=int, default=1024, dest="eval_max_new")
    ap.add_argument("--eval-judge-api", action="store_true", dest="eval_judge_api",
                    help="also arbitrate the loose verdict via the API "
                         "(sends answer strings only)")
    ap.add_argument("--save-every", type=int, default=0, dest="save_every",
                    help="also save a checkpoint every N steps (0 = final only)")
    ap.add_argument("--master-port", type=int, default=29500, dest="master_port")
    ap.add_argument("--no-system", dest="use_system", action="store_false",
                    help="send the bare turn, without the system prompt")
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    return ap.parse_args()


def run(args) -> int:
    import ray
    import torch
    from zero_ppo.engine.ray_actor import RayEngine, train_actor_cls
    from zero_ppo.reward.loader import load
    from zero_ppo.rollout.multiturn import rollout_batch
    from zero_ppo.rollout.tools import local_runner
    from zero_ppo.train_actor import assemble
    from zero_ppo.trainer import advantages, statistics
    from zero_ppo.algo.advantages import returns as value_returns, whiten
    from zero_ppo.util.prompts import ToolFormat, question_of
    from transformers import AutoTokenizer

    if not args.model or not args.pool:
        raise SystemExit("--model and --pool are required to train")

    # Fight fragmentation WITHOUT expandable_segments: that backs weights with
    # cuMemMap memory whose CUDA IPC needs pidfd_getfd (denied without ptrace
    # rights, pytorch#165685). garbage_collection_threshold makes the caching
    # allocator reclaim cached blocks proactively, and max_split_size_mb prevents
    # large blocks from being split into unusable fragments - both keep plain
    # cudaMalloc memory, so IPC weight transfer still works.
    alloc_conf = "garbage_collection_threshold:0.8,max_split_size_mb:512"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf

    world = args.world
    ray.init(num_gpus=world, ignore_reinit_error=True)
    master_addr = ray.util.get_node_ip_address()

    # One torch process group over exactly the colocated trainer ranks - the IPC
    # handle all-gather runs over this default group, so its membership is load
    # bearing. Env vars set in the driver do NOT reach actor subprocesses, so each
    # actor gets its rendezvous through runtime_env. Ray assigns one card per actor
    # (CUDA_VISIBLE_DEVICES is a single id), hence the worker addresses cuda:0.
    cls = train_actor_cls()
    actors = []
    for rank in range(world):
        env = {"MASTER_ADDR": master_addr, "MASTER_PORT": str(args.master_port),
               "RANK": str(rank), "WORLD_SIZE": str(world), "LOCAL_RANK": "0",
               "PYTORCH_CUDA_ALLOC_CONF": alloc_conf}
        actors.append(cls.options(runtime_env={"env_vars": env}).remote(args, rank, world))

    # The engine pins all cards itself; Ray books none for it (colocation). Built
    # after the trainer actors so its CUDA_VISIBLE_DEVICES covers the whole node.
    engine = RayEngine(args.model, dtype=args.dtype, tensor_parallel=world,
                       gpu_util=args.vllm_util, max_model_len=args.max_model_len,
                       prefix_cache=bool(args.prefix_cache), tmp_root=args.out,
                       devices=list(range(world)))

    # Open the weight channel once, on every rank, before the loop.
    handle = engine.handle
    ray.get([a.init_sync.remote(handle=handle, packed=bool(args.packed)) for a in actors])

    reward = load(args.reward)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    run_code = local_runner()
    encode = lambda t: list(tok(t, add_special_tokens=False)["input_ids"])
    fmt = ToolFormat(tok)          # closes assistant turns and opens the tool reply
    problems = [json.loads(l) for l in open(args.pool, encoding="utf-8") if l.strip()]
    if args.shuffle:
        import random
        random.Random(args.shuffle_seed).shuffle(problems)
        print(f"  训练池已洗牌 (seed={args.shuffle_seed})", flush=True)

    # LUFFY-style guidance: k of each group's samples come from an external model.
    api_gen, luffy_k = None, int(args.luffy or 0)
    if luffy_k:
        if not (0 < luffy_k < args.group):
            raise SystemExit(f"--luffy must be in (0, GROUP={args.group}), got {luffy_k}")
        from zero_ppo.rollout.api_gen import ApiGenerator
        api_gen = ApiGenerator(model=args.luffy_model, base_url=args.luffy_base_url,
                               api_key_env=args.luffy_key_env,
                               temperature=args.temperature,
                               max_tokens=args.luffy_max_tokens,
                               workers=args.luffy_workers)
        if not api_gen.available:
            raise SystemExit(f"--luffy needs the API key in ${args.luffy_key_env}; "
                             "refusing to run with silent empty traces")
        print(f"  LUFFY: 每组 {args.group} 条中 {luffy_k} 条来自 {api_gen.model}",
              flush=True)
    save_dir = args.save_dir or os.path.join(args.out, "checkpoints")
    print(f"  driver: {len(problems)} problems | {args.algo} | "
          f"{args.batch} x {args.group} | world={world} | save -> {save_dir}", flush=True)

    # A structured per-step record, flushed immediately. Terminal scrollback is
    # lost on a dropped SSH session, and grepping human-readable lines is the
    # wrong tool for "when did the reward collapse"; this file survives both.
    os.makedirs(args.out, exist_ok=True)
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    metrics = open(metrics_path, "a", encoding="utf-8")
    metrics.write(json.dumps({"event": "run_start", "model": args.model,
                              "pool": args.pool, "algo": args.algo,
                              "reward": args.reward, "batch": args.batch,
                              "group": args.group, "iters": args.iters,
                              "lr": args.lr, "ppo_epochs": args.ppo_epochs,
                              "clip_eps": args.clip_eps, "lam": args.lam,
                              "critic_lr": getattr(args, "critic_lr", None),
                              "problems": len(problems),
                              "shuffle": bool(args.shuffle),
                              "shuffle_seed": args.shuffle_seed if args.shuffle else None,
                              "luffy": luffy_k,
                              "luffy_model": api_gen.model if api_gen else None,
                              "ts": time.strftime("%Y-%m-%d %H:%M:%S")},
                             ensure_ascii=False) + "\n")
    metrics.flush()
    print(f"  metrics -> {metrics_path}", flush=True)

    # Inline held-out eval. RL here peaks early then collapses, so `final` is not
    # necessarily the best artifact - measure and keep the winner.
    eval_problems, tracker, eval_judge = [], None, None
    if args.eval_every:
        from zero_ppo.inline_eval import BestTracker
        if not os.path.exists(args.eval_pool):
            raise SystemExit(f"--eval-pool not found: {args.eval_pool}")
        eval_problems = [json.loads(l) for l in
                         open(args.eval_pool, encoding="utf-8") if l.strip()]
        if args.eval_limit:
            eval_problems = eval_problems[:args.eval_limit]
        tracker = BestTracker(os.path.join(args.out, "best_eval.json"))
        if args.eval_judge_api:
            from zero_ppo.eval import LLMJudge
            eval_judge = LLMJudge(model=os.environ.get("JUDGE_MODEL", "deepseek-chat"))
            if not eval_judge.available:
                raise SystemExit("--eval-judge-api needs DEEPSEEK_API_KEY; "
                                 "refusing to silently fall back to rules")
        print(f"  内联评测: 每 {args.eval_every} 步 · {len(eval_problems)} 题 · "
              f"贪心 · 复用共卡引擎 · 含第0步基线", flush=True)

    def do_eval(label: int, sync: bool = False) -> dict:
        """One greedy eval on the colocated engine, with the memory dance.

        Ordering is the whole game: sleep fully, release every rank's cached
        blocks, wake WEIGHTS then KV cache (waking the cache before the weights
        are in place touches cumem memory that is still freed -> segfault), score,
        then always sleep and release again so the next backward starts from the
        same budget it would have had without the eval.

        `sync` pushes the trained weights first. Needed whenever the regular
        per-step sync did not run: before step 0 there is nothing to push, and on
        the last step the sync block is skipped, so the engine would otherwise
        still hold the previous step's policy.
        """
        from zero_ppo.inline_eval import run_eval
        t_ev = time.time()
        engine.sleep()
        ray.get([a.free_memory.remote() for a in actors])
        engine.wake(tags=["weights"])
        if sync:
            ray.get([a.do_sync.remote() for a in actors])
        engine.wake(tags=["kv_cache"])
        try:
            ev = run_eval(engine, tok, fmt, run_code, eval_problems,
                          max_model_len=args.max_model_len,
                          max_new=args.eval_max_new,
                          max_turns=args.max_turns, batch=args.eval_batch,
                          judge=eval_judge)
        finally:
            engine.sleep()
            ray.get([a.free_memory.remote() for a in actors])
        ck = os.path.join(save_dir, "base" if label == 0 else f"step_{label}")
        improved = tracker.offer(label, ev, ck)
        if improved:
            ray.get(actors[0].save.remote(os.path.join(save_dir, "best")))
        ev.update({"event": "eval", "step": label,
                   "improved": improved, "best": tracker.best,
                   "best_step": tracker.best_step,
                   "sec": round(time.time() - t_ev, 1),
                   "ts": time.strftime("%H:%M:%S")})
        metrics.write(json.dumps(ev, ensure_ascii=False) + "\n")
        metrics.flush()
        tag = "基线" if label == 0 else f"步{label}"
        print(f"  评测 {tag}: 贪心宽 {ev['acc_loose']*100:.2f}% · "
              f"严格 {ev['acc']*100:.2f}% · 出框 {ev['boxed']*100:.1f}% · "
              f"轮数 {ev['turns']:.2f}"
              + (" · 新最佳 (保存 best)" if improved else
                 f" · 最佳仍是步{tracker.best_step} ({tracker.best*100:.2f}%)"),
              flush=True)
        return ev

    # Baseline BEFORE any update. Two reasons: it is the reference the RL gain is
    # measured against (rl - sft, not rl - base), and it exercises the whole eval
    # path at minute zero instead of failing 20 steps in.
    if tracker is not None:
        do_eval(0, sync=False)

    for step in range(args.iters):
        t_step = time.time()
        expanded, golds = step_problems(problems, step, args.batch, args.group)

        engine.wake()
        # step_problems repeats each problem GROUP times consecutively, so the
        # last luffy_k slots of every group are the off-policy members
        api_mask = ([j % args.group >= args.group - luffy_k
                     for j in range(len(expanded))] if luffy_k else None)
        rows = rollout_batch(expanded,
                             lambda ps: engine.generate(ps, args.temperature, args.max_new),
                             run_code,
                             lambda pr: (fmt.prompt(pr) if args.use_system
                                          else question_of(pr)), encode,
                             max_turns=args.max_turns, max_tokens=args.mt_tokens,
                             chunk=args.rollout_chunk,
                             fmt=fmt if args.use_system else None,
                             generate_api=api_gen, api_mask=api_mask)
        n_api = sum(1 for r in rows if r.get("source") == "api")

        ids, mask, samples = assemble(rows, tok, pad_id, lambda i: golds[i],
                                      question_of=lambda i: question_of(expanded[i]))

        # Sleep FIRST: a reward that calls a judge API would otherwise hold the
        # engine awake (and the KV cache allocated) while waiting on the network.
        engine.sleep()                       # give the cache back for the backward
        rewards = reward(samples, ids.shape[1])
        rsum = reward.summary(samples, rewards) if hasattr(reward, "summary") else {}
        api_correct = 0
        if n_api:
            per_row = (rewards * mask).sum(dim=1)
            api_correct = sum(1 for r, v in zip(rows, per_row)
                              if r.get("source") == "api" and float(v) > 0)
            print(f"  LUFFY: api轨迹 {n_api} 条 · 答对 {api_correct} · "
                  f"api调用 {api_gen.calls} · 失败 {api_gen.errors}", flush=True)
        if rsum.get("judged") or rsum.get("fallback"):
            print(f"  判分: 答对 {rsum.get('correct',0)}/{rsum.get('n',0)} · "
                  f"段判定 {rsum.get('judged',0)} · 回退 {rsum.get('fallback',0)} · "
                  f"通过段 {rsum.get('turns_passed',0)}", flush=True)

        # PPO: the critic's pre-update values feed GAE, and A + V_old is the
        # critic's regression target. GRPO needs no critic.
        rets = None
        if args.algo == "ppo":
            values = ray.get(actors[0].compute_values.remote(ids, mask))
            adv = advantages("ppo", rewards, mask, args.group, values=values,
                             gamma=args.gamma, lam=args.lam)
            # The critic target uses the RAW advantage: A + V_old recovers the
            # lambda-return. Whitening first would corrupt the regression target.
            rets = value_returns(adv, values)
            if args.whiten:
                adv = whiten(adv, mask, signal_ref=args.signal_ref)
        else:
            adv = advantages("grpo", rewards, mask, args.group)

        results = ray.get([a.train_step.remote(ids, mask, adv, rets) for a in actors])
        losses = [r[0] for r in results]
        cur_lr = results[0][1]
        vloss = results[0][2] if len(results[0]) > 2 else 0.0
        approx_kl = results[0][3] if len(results[0]) > 3 else 0.0
        clip_frac = results[0][4] if len(results[0]) > 4 else 0.0
        ref_kl = results[0][5] if len(results[0]) > 5 else 0.0
        stopped_at = results[0][6] if len(results[0]) > 6 else 0

        if step + 1 < args.iters:
            # Official colocated IPC pattern (rlhf_ipc_fsdp_ep.py): the engine is
            # already asleep from the backward. Wake ONLY the weights, transfer,
            # then wake the KV cache. Waking the KV cache before the IPC write
            # touches cumem memory that is still freed -> cuMemcpyDtoDAsync segfault.
            engine.wake(tags=["weights"])
            ray.get([a.do_sync.remote() for a in actors])
            engine.wake(tags=["kv_cache"])

        st = statistics(rewards, adv, mask)
        extra = (f" · vloss {vloss:.4f} · kl {approx_kl:.4f} · clip {clip_frac:.3f}"
                 + (f" · refkl {ref_kl:.4f}" if args.kl_coef > 0 else "")
                 + (f" · 早停@{stopped_at}/{args.ppo_epochs}" if stopped_at else "")
                 if args.algo == "ppo" else "")
        print(f"  步 {step + 1}/{args.iters}: {len(rows)} 条 · "
              f"reward 合计 {st['reward']:.2f} · |A| {st['adv_abs']:.3f} · "
              f"loss {losses[0]:.4f}{extra} · lr {cur_lr:.2e}", flush=True)

        # one JSON line per step, flushed now: survives a dropped SSH session
        metrics.write(json.dumps({
            "event": "step", "step": step + 1, "iters": args.iters,
            "n_rows": len(rows),
            "n_api": n_api, "api_correct": api_correct,
            "reward": round(st["reward"], 4),
            "adv_abs": round(st["adv_abs"], 5),
            "adv_std": round(st["adv_std"], 5),
            "loss": round(losses[0], 5),
            "vloss": round(vloss, 5),
            "approx_kl": round(approx_kl, 6),
            "ref_kl": round(ref_kl, 6),
            "clip_frac": round(clip_frac, 5),
            "stopped_at": stopped_at,
            "lr": cur_lr,
            "correct": rsum.get("correct"), "n_samples": rsum.get("n"),
            "judged": rsum.get("judged"), "fallback": rsum.get("fallback"),
            "turns_passed": rsum.get("turns_passed"),
            "api_calls": rsum.get("api_calls"), "api_errors": rsum.get("api_errors"),
            "zero_reward": bool(st["reward"] == 0.0),
            "sec": round(time.time() - t_step, 1),
            "ts": time.strftime("%H:%M:%S"),
        }, ensure_ascii=False) + "\n")
        metrics.flush()

        if args.save_every and (step + 1) % args.save_every == 0:
            ckpt = os.path.join(save_dir, f"step_{step + 1}")
            ray.get(actors[0].save.remote(ckpt))
            print(f"  checkpoint -> {ckpt}", flush=True)

        if tracker is not None and (step + 1) % args.eval_every == 0:
            do_eval(step + 1, sync=(step + 1 >= args.iters))

    # Final trained policy as a standalone HF checkpoint (rank 0 holds the full
    # ZeRO-2 weights). Loadable with from_pretrained for inference or resuming.
    final = os.path.join(save_dir, "final")
    ray.get(actors[0].save.remote(final))
    print(f"  trained model saved -> {final}", flush=True)

    if tracker is not None and tracker.best is not None:
        best_dir = os.path.join(save_dir, "best")
        print(f"  最佳评测点(宽口径): 步 {tracker.best_step} · "
              f"{tracker.best*100:.2f}% -> {best_dir}", flush=True)
        print(f"  逐次评测记录 -> {os.path.join(args.out, 'best_eval.json')}",
              flush=True)
    metrics.write(json.dumps({"event": "run_end", "saved": final,
                              "best_step": tracker.best_step if tracker else None,
                              "best_metric": tracker.metric if tracker else None,
                              "best_acc": tracker.best if tracker else None,
                              "best_ckpt": (os.path.join(save_dir, "best")
                                            if tracker and tracker.best is not None
                                            else None),
                              "ts": time.strftime("%Y-%m-%d %H:%M:%S")},
                             ensure_ascii=False) + "\n")
    metrics.close()
    ray.shutdown()
    return 0


def _self_test() -> None:
    problems = [{"question": f"q{i}", "gold": i} for i in range(5)]
    expanded, golds = step_problems(problems, 0, 2, 3)
    assert len(expanded) == 6 and golds == ["0", "0", "0", "1", "1", "1"]
    expanded, golds = step_problems(problems, 1, 2, 3)
    assert golds == ["2", "2", "2", "3", "3", "3"]
    # wrap-around when the pool is smaller than the step offset
    expanded, golds = step_problems(problems, 3, 2, 2)
    assert golds == ["1", "1", "2", "2"], golds
    print("  driver self-test OK")
    print("    batch expansion, group replication and pool wrap-around verified")


if __name__ == "__main__":
    a = build_args()
    if a.self_test:
        _self_test()
        raise SystemExit(0)
    raise SystemExit(run(a))