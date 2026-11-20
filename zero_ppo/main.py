#!/usr/bin/env python3
"""Entry point: PPO or GRPO over multi-turn tool-use rollouts.

Only orchestration lives here: parse, build an engine, load the model, loop. The
algorithm is in `trainer.py`, the backends are in `engine/`, and the reward is a
plugin chosen by path.

    python3 -m zero_ppo.main --dry-run --reward zero_ppo/reward/agent_reward_test.py
    python3 -m zero_ppo.main --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero_ppo.rollout.multiturn import MAX_TURNS


@dataclass
class Plan:
    algo: str
    batch: int
    group: int
    critic_mode: str
    reward_path: str
    reward_name: str
    lr: float
    iters: int
    tuning: str = "full"        # last, so the fields without defaults stay first

    @property
    def needs_critic(self) -> bool:
        return self.algo == "ppo"


def build_plan(args) -> Plan:
    from zero_ppo.reward.loader import load

    if args.algo not in ("ppo", "grpo"):
        raise SystemExit(f"unknown algo {args.algo!r}")
    if args.batch < 1:
        raise SystemExit("--batch must be >= 1")
    if args.group < 1:
        raise SystemExit("--group must be >= 1")

    if args.algo == "grpo":
        if args.group < 2:
            raise SystemExit("GRPO needs --group >= 2: a single sample has no group "
                             "mean to compare against")
        if args.batch % args.group:
            raise SystemExit(f"--batch {args.batch} is not a multiple of "
                             f"--group {args.group}")
        if args.critic_mode != "none":
            raise SystemExit("GRPO takes its baseline from the group mean; a critic "
                             "here is dead weight, pass --critic-mode none")
        critic_mode = "none"
    else:
        critic_mode = args.critic_mode
        if critic_mode == "none":
            raise SystemExit("PPO needs a critic; pass --critic-mode "
                             "full, lora_shared or lora_split")

    if not os.path.exists(args.reward):
        raise SystemExit(f"reward file not found: {args.reward}")
    reward = load(args.reward)

    return Plan(algo=args.algo, batch=args.batch, group=args.group,
                critic_mode=critic_mode, reward_path=args.reward,
                reward_name=getattr(reward, "name", type(reward).__name__),
                lr=args.lr, iters=args.iters,
                tuning=getattr(args, "tuning", "full"))


def print_plan(p: Plan, engine_name: str) -> None:
    rows = p.batch * p.group          # problems x samples
    print("  resolved plan")
    print(f"    algorithm      {p.algo}")
    print(f"    engine         {engine_name}")
    print(f"    batch / group  {p.batch} problems x {p.group} = {rows} trajectories")
    print(f"    critic         {p.critic_mode}"
          f"{'  (group mean instead)' if p.algo == 'grpo' else ''}")
    print(f"    reward         {p.reward_name}  <- {os.path.basename(p.reward_path)}")
    print(f"    lr / iters     {p.lr} / {p.iters}")
    print(f"    tuning         {p.tuning}")
    print(f"    advantage      "
          f"{'group-relative' if p.algo == 'grpo' else 'gae against the critic'}")


def _tune(hf, args, keep_cache: bool = True):
    """Full fine-tuning trains the base; LoRA freezes it and trains an adapter."""
    if args.tuning == "full":
        for q in hf.parameters():
            q.requires_grad_(True)
        return hf

    from peft import LoraConfig, TaskType, get_peft_model
    targets = (["q_proj", "k_proj", "v_proj", "o_proj"] if args.lora_targets == "attn"
               else ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"])
    hf = get_peft_model(hf, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0.0, target_modules=targets, bias="none"))
    hf.print_trainable_parameters()
    return hf


def _engine(args, rank: int, world: int):
    """One backend, chosen by name. Only rank zero holds it."""
    if rank != 0:
        return None
    if args.engine == "http":
        from zero_ppo.engine.http import HttpEngine
        e = HttpEngine(args.vllm_url)
        if not e.wait_ready(timeout=args.serve_wait):
            raise SystemExit(f"  vLLM server not ready at {args.vllm_url}")
        print(f"  server ready at {args.vllm_url}", flush=True)
        return e
    if args.engine == "ray":
        from zero_ppo.engine.ray_actor import RayEngine
        return RayEngine(args.model, dtype=args.dtype, tensor_parallel=world,
                         gpu_util=args.vllm_util, max_model_len=args.max_model_len,
                         prefix_cache=bool(args.prefix_cache),
                         tmp_root=args.out)
    from zero_ppo.engine.local import LocalEngine
    return LocalEngine(args.model, dtype=args.dtype, tensor_parallel=world,
                       gpu_util=args.vllm_util, max_model_len=args.max_model_len,
                       prefix_cache=bool(args.prefix_cache), tmp_root=args.out)


def _assemble(rows, tok, pad_id, gold_of):
    """Rows -> padded tensors plus the Sample objects a reward file consumes."""
    import torch
    from zero_ppo.reward.base import Sample

    B = len(rows)
    T = max(len(r["ids"]) for r in rows)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.bool)
    samples = []
    for i, r in enumerate(rows):
        n = len(r["ids"])
        off = T - n                       # left padding
        ids[i, off:] = torch.tensor(r["ids"], dtype=torch.long)
        for a, b in r["spans"]:
            mask[i, off + a:off + b + 1] = True
        nz = mask[i].nonzero().flatten()
        done = int(nz[-1]) if nz.numel() else T - 1
        samples.append(Sample(gold=gold_of(i), texts=r["texts"],
                              seg_end=[off + e for e in r["seg_end"]],
                              seg_ok=r["seg_ok"], done=done, length=n,
                              mask=mask[i]))
    return ids, mask, samples


def train(p: Plan, args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from zero_ppo.actor import Actor
    from zero_ppo.rollout.multiturn import rollout_batch
    from zero_ppo.rollout.tools import local_runner
    from zero_ppo.reward.loader import load
    from zero_ppo.trainer import advantages, clipped_loss, statistics
    from zero_ppo.util.prompts import ToolFormat, question_of

    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    reward = load(p.reward_path)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # The engine reserves its share before the weights are loaded, otherwise it
    # computes the memory left after the model plus the allocator's cache and gets
    # a negative number of KV blocks.
    engine = _engine(args, rank, world)

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch_dtype)
    hf = _tune(hf, args, keep_cache=False)
    if args.gradient_checkpoint:
        hf.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        hf.config.use_cache = False
    backbone = hf.base_model.model if args.tuning == "lora" else hf.model
    actor = Actor(backbone, hf.lm_head, pad_id=pad_id, value_head=False,
                  hidden=hf.config.hidden_size, amp_dtype=args.dtype)
    actor = actor.to(args.train_device)
    if args.tuning == "lora":
        for name, q in actor.backbone.named_parameters():
            if "lora" not in name.lower():
                q.requires_grad_(False)

    run_code = local_runner()
    encode = lambda t: list(tok(t, add_special_tokens=False)["input_ids"])
    fmt = ToolFormat(tok)          # closes assistant turns and opens the tool reply
    problems = [json.loads(l) for l in open(args.pool, encoding="utf-8") if l.strip()]
    print(f"  rank {rank}/{world} · pool {len(problems)} problems | algo {p.algo} | "
          f"{p.batch} x {p.group}", flush=True)

    opt = None
    ds_engine = None
    for step in range(p.iters):
        take = problems[(step * p.batch) % max(1, len(problems)):][:p.batch]
        expanded = [q for q in take for _ in range(p.group)]
        golds = [str(q.get("gold", "")) for q in expanded]

        if engine is not None:
            engine.wake()
        rows = (rollout_batch(expanded,
                              lambda ps, _e=engine: _e.generate(
                                  ps, args.temperature, args.max_new),
                              run_code,
                              lambda pr: (fmt.prompt(pr) if args.use_system
                                          else question_of(pr)), encode,
                              max_turns=args.max_turns, max_tokens=args.mt_tokens,
                              chunk=args.rollout_chunk,
                              fmt=fmt if args.use_system else None) if engine is not None else None)
        if world > 1:
            box = [rows]
            torch.distributed.broadcast_object_list(box, src=0)
            rows = box[0]

        ids, mask, samples = _assemble(rows, tok, pad_id, lambda i: golds[i])
        rewards = reward(samples, ids.shape[1]).to(ids.device)

        # Checked here, not at load time: a debug run should show that the rollout,
        # the tool and the reward all work before anything is refused.
        if not actor.trainable():
            print(f"  步 {step + 1}: rollout 与 reward 都通了 ✓"
                  f"（reward 合计 {float(rewards.sum()):.2f}）", flush=True)
            raise SystemExit("  no trainable parameters")

        if engine is not None:
            engine.sleep()          # give the cache back for the backward
        if p.algo == "ppo":
            # This fallback path builds no critic, so GAE has no values to use.
            # Say so instead of failing deep inside the estimator.
            raise SystemExit("  PPO needs a critic: use `python3 -m zero_ppo.driver "
                             "--algo ppo` (or train.sh). This local fallback is "
                             "outcome-only GRPO.")
        adv = advantages(p.algo, rewards, mask, p.group)
        if opt is None:
            if args.deepspeed:
                import deepspeed
                from zero_ppo.engine.ds_config import zero2
                cfg = zero2(micro_batch=int(p.batch * p.group), accum=1, lr=p.lr,
                            dtype=args.dtype)
                actor, opt, _, _ = deepspeed.initialize(model=actor, config=cfg)
                ds_engine = actor
            else:
                opt = torch.optim.AdamW(actor.trainable(), lr=p.lr,
                                        betas=(0.9, 0.95))

        actor.train()
        ids_d = ids.to(args.train_device)
        gen_d = mask.to(args.train_device)
        attn_d = (ids_d != pad_id)
        logp = actor.token_logprobs(ids_d, attn_d, mask=gen_d, no_grad=False)
        loss = clipped_loss(logp, logp.detach(), adv.to(logp.device), gen_d)
        if ds_engine is not None:
            ds_engine.backward(loss)
            ds_engine.step()
        else:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.trainable(), 1.0)
            opt.step()

        if step + 1 < p.iters:
            if args.engine == "http":
                from vllm.distributed.weight_transfer.clients import (
                    HTTPVLLMWeightSyncClient)
                from zero_ppo.engine.weight_sync import push
                push(None, hf, rank=rank, world=world,
                     client=HTTPVLLMWeightSyncClient(args.vllm_url))
            elif args.engine == "ray":
                from zero_ppo.engine.weight_sync import push
                h = engine.handle if rank == 0 else None
                if world > 1:
                    # one cluster, or a handle from one is useless in the other
                    import ray
                    addr = ray.get_runtime_context().gcs_address if rank == 0 else None
                    box = [addr]
                    torch.distributed.broadcast_object_list(box, src=0)
                    if rank != 0 and not ray.is_initialized():
                        ray.init(address=box[0], num_gpus=0,
                                 ignore_reinit_error=True)
                    box = [h]
                    torch.distributed.broadcast_object_list(box, src=0)
                    h = box[0]
                push(None, hf, handle=h, rank=rank, world=world)
            elif engine is not None:
                engine.reload(hf, tok, step + 1)
        st = statistics(rewards, adv, mask)
        print(f"  [r{rank}] 步 {step + 1}/{p.iters}: {len(rows)} 条 · "
              f"reward 合计 {st['reward']:.2f} · |A| {st['adv_abs']:.3f} · "
              f"loss {float(loss.detach()):.4f}", flush=True)
    return 0


def _self_test() -> None:
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward",
                        "agent_reward_test.py")

    def args(**kw):
        base = dict(algo="grpo", batch=8, group=4, critic_mode="none",
                    reward=here, lr=5e-6, iters=10)
        base.update(kw)
        return argparse.Namespace(**base)

    p = build_plan(args())
    assert p.algo == "grpo" and p.needs_critic is False and p.group == 4
    p2 = build_plan(args(algo="ppo", critic_mode="lora_split"))
    assert p2.needs_critic is True and p2.critic_mode == "lora_split"

    rejects = [
        dict(algo="grpo", group=1),                 # no group
        dict(algo="grpo", batch=6, group=4),        # does not divide
        dict(algo="grpo", critic_mode="lora_split"),  # critic on grpo
        dict(algo="ppo", critic_mode="none"),       # ppo without critic
        dict(algo="nope"),                          # unknown
        dict(batch=0),                              # empty batch
        dict(reward="/nonexistent/r.py"),           # missing reward
    ]
    for kw in rejects:
        try:
            build_plan(args(**kw))
        except SystemExit:
            continue
        raise AssertionError(f"config was accepted: {kw}")

    print("  main self-test OK")
    print(f"    grpo and ppo both resolved; {len(rejects)} bad configs rejected")
    print("    rejections cover: no group, ragged batch, critic on grpo, "
          "ppo without critic, unknown algo, empty batch, missing reward")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--algo", default="grpo", choices=["ppo", "grpo"])
    ap.add_argument("--reward", default="zero_ppo/reward/agent_reward_test.py")
    ap.add_argument("--critic-mode", default="none", dest="critic_mode",
                    choices=["none", "full", "lora_shared", "lora_split"])
    ap.add_argument("--batch", type=int, default=8, help="problems per step")
    ap.add_argument("--group", type=int, default=4, help="samples per problem")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--pool", default=None,
                    help="jsonl of problems, each with question and gold "
                         "(required unless --dry-run or --self-test)")
    ap.add_argument("--train-device", default="cuda:1", dest="train_device")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=1024, dest="max_new")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS, dest="max_turns",
                    help=f"assistant turns per trajectory (default {MAX_TURNS})")
    ap.add_argument("--mt-tokens", type=int, default=1800, dest="mt_tokens")
    ap.add_argument("--rollout-chunk", type=int, default=8, dest="rollout_chunk")
    ap.add_argument("--max-model-len", type=int, default=4096, dest="max_model_len")
    ap.add_argument("--vllm-util", type=float, default=0.55, dest="vllm_util")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16"])
    ap.add_argument("--prefix-cache", type=int, default=1, dest="prefix_cache")
    ap.add_argument("--tuning", default="full", choices=["full", "lora"],
                    help="full trains the base; lora freezes it and trains an adapter")
    ap.add_argument("--gradient-checkpoint", type=int, default=1,
                    dest="gradient_checkpoint")
    ap.add_argument("--lora-r", type=int, default=16, dest="lora_r")
    ap.add_argument("--lora-alpha", type=int, default=32, dest="lora_alpha")
    ap.add_argument("--lora-targets", default="all", dest="lora_targets",
                    choices=["attn", "all"],
                    help="attn = q,k,v,o only; all = plus the MLP")
    ap.add_argument("--out", default="runs/zero_ppo")
    ap.add_argument("--engine", default="local",
                    choices=["local", "ray", "http"])
    ap.add_argument("--vllm-url", default="http://127.0.0.1:8000",
                    dest="vllm_url")
    ap.add_argument("--serve-wait", type=float, default=600.0,
                    dest="serve_wait")
    ap.add_argument("--deepspeed", type=int, default=0,
                    help="1 to wrap the actor in DeepSpeed ZeRO-2")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    ap.add_argument("--no-system", dest="use_system", action="store_false")
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()

    if a.self_test:
        _self_test()
        return 0

    # Under torchrun each rank gets one card. Pinning CUDA_VISIBLE_DEVICES before
    # anything imports torch or vllm means "cuda:0" is this rank's card, so the
    # engine and the actor land on the same one - which is what colocation needs.
    local = os.environ.get("LOCAL_RANK")
    if local is not None:
        # Every rank sees both cards and addresses its own by index. Hiding the
        # other card makes torch build a process group for cuda:{LOCAL_RANK} on a
        # machine that now reports one accelerator, which is an immediate error.
        a.train_device = f"cuda:{local}"
    p = build_plan(a)
    print_plan(p, a.engine)
    if a.dry_run:
        return 0
    if not a.pool:
        raise SystemExit("--pool is required to train (a path to a jsonl of "
                         "problems with question and gold)")
    if not a.model:
        raise SystemExit("--model is required to train")
    return train(p, a)


if __name__ == "__main__":
    raise SystemExit(main())
