#!/usr/bin/env python3
"""The training worker: one policy + DeepSpeed ZeRO-2 on one card.

This is the body of the old `main.py` loop, lifted into a Ray actor so the driver
can colocate it with the inference engine. It owns the HF model, the Actor wrapper,
the optimiser and (for PPO) the critic; the driver owns the rollout, the reward and
the schedule. Nothing here knows about Ray beyond being constructible as an actor.

Weight sync is the vLLM-native shape: the transfer channel is opened **once**
(`init_sync`) and every step is a single `sync()` (`send_weights`). The engine is
paused around the transfer by the driver. Only rank 0 talks to the engine; the
other ranks still join `send_weights` because the IPC handle all-gather is a
collective over this trainer group.

    python3 -m zero_ppo.train_actor --self-test     # tensor assembly only, no GPU
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def assemble(rows, tok, pad_id, gold_of, question_of=None):
    """Rows -> padded tensors plus the Sample objects a reward file consumes.

    Left padding, so every trajectory ends at the right edge; the mask marks only
    generated tokens. `question_of(i)` supplies the problem text for rewards that
    judge each turn against the question; without it those fields stay empty and
    outcome-only rewards are unaffected.
    """
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
                              mask=mask[i],
                              question=(question_of(i) if question_of else ""),
                              seg_kinds=list(r.get("seg_kinds") or []),
                              seg_outs=list(r.get("seg_outs") or [])))
    return ids, mask, samples


def tune(hf, args):
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


class TrainWorker:
    """One card's training state. Constructed inside a Ray actor by the driver."""

    def __init__(self, args, rank: int, world: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from zero_ppo.actor import Actor

        self.args = args
        self.rank = rank
        self.world = world
        # Under Ray each actor sees a single card (CUDA_VISIBLE_DEVICES is one id),
        # so the local index is 0 even though the global rank differs.
        local = int(os.environ.get("LOCAL_RANK", str(rank)))
        self.device = f"cuda:{local}"
        torch.cuda.set_device(local)
        if world > 1 and not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")

        self.tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        self.pad_id = (self.tok.pad_token_id
                       if self.tok.pad_token_id is not None else self.tok.eos_token_id)

        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
        hf = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch_dtype)
        hf = tune(hf, args)
        if args.gradient_checkpoint:
            hf.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            hf.config.use_cache = False
        self.hf = hf
        backbone = hf.base_model.model if args.tuning == "lora" else hf.model
        self.actor = Actor(backbone, hf.lm_head, pad_id=self.pad_id, value_head=False,
                           hidden=hf.config.hidden_size, amp_dtype=args.dtype)
        self.actor = self.actor.to(self.device)
        if args.tuning == "lora":
            for name, q in self.actor.backbone.named_parameters():
                if "lora" not in name.lower():
                    q.requires_grad_(False)

        self.critic = None
        self.critic_opt = None
        self.ds_critic = None
        self.vf_coef = float(getattr(args, "vf_coef", 0.5))
        if args.algo == "ppo":
            if args.critic_mode == "none":
                raise SystemExit("PPO needs a critic; pass --critic-mode full "
                                 "(or lora_shared / lora_split with --tuning lora)")
            from zero_ppo.critic import build_value
            self.critic = build_value(args.critic_mode, self.actor, hf.config.hidden_size)
            self.critic = self.critic.to(self.device)

        # A frozen copy of the starting policy. The clip only bounds movement
        # relative to the policy that generated the current batch, which is
        # re-captured every rollout, so nothing stopped cumulative drift away
        # from SFT - the run collapsed into degenerate text with finite weights.
        # This reference anchors it. Inference only: no grads, no optimiser state.
        self.ref = None
        self.kl_coef = float(getattr(args, "kl_coef", 0.0))
        if args.algo == "ppo" and self.kl_coef > 0:
            ref = AutoModelForCausalLM.from_pretrained(
                args.model, local_files_only=True, torch_dtype=torch_dtype)
            ref_backbone = ref.model
            self.ref = Actor(ref_backbone, ref.lm_head, pad_id=self.pad_id,
                             value_head=False, hidden=ref.config.hidden_size,
                             amp_dtype=args.dtype).to(self.device)
            self.ref.eval()
            for q in self.ref.parameters():
                q.requires_grad_(False)
            self._ref_hf = ref          # keep alive: the Actor holds its modules

        # Early stop on the inner loop, as SB3 does (target_kl): the clip bounds
        # each token's ratio, but many tokens each drifting a little can still add
        # up to a large policy move. This measures that move against the policy
        # that GENERATED the batch (old_logp), not against the frozen reference.
        # 0 disables it, keeping the previous behaviour.
        self.target_kl = float(getattr(args, "target_kl", 0.0) or 0.0)

        self.opt = None
        self.ds_engine = None
        self.sync = None
        self.step_count = 0

    # ------------------------------------------------------------------ setup
    def _split(self) -> tuple[int, int]:
        """(micro_batch, accum_steps) for one optimiser update.

        `mini_batch` counts TRAJECTORIES per forward; 0 means no splitting. The
        product must equal the rollout batch so the effective batch - and hence
        the gradient - is identical to the unsplit run.
        """
        rows = max(1, int(self.args.batch * self.args.group))
        mb = int(getattr(self.args, "mini_batch", 0) or 0)
        if mb <= 0 or mb >= rows:
            return rows, 1
        # round up so every row is covered; DeepSpeed scales the loss by accum
        accum = (rows + mb - 1) // mb
        return mb, accum

    def _ensure_opt(self):
        import torch
        if self.opt is not None:
            return
        params = self.actor.trainable()
        if self.critic is not None:
            params = params + self.critic.trainable()
        if self.args.deepspeed:
            import deepspeed
            from zero_ppo.engine.ds_config import zero2
            # Split the rollout batch into micro-batches and accumulate: activation
            # memory scales with the micro-batch, not the full batch, which is what
            # makes room for the frozen reference model. The optimiser still sees
            # the whole batch's gradient, so the update is unchanged.
            micro, accum = self._split()
            cfg = zero2(micro_batch=micro, accum=accum, lr=self.args.lr,
                        dtype=self.args.dtype)
            self.actor, self.opt, _, _ = deepspeed.initialize(model=self.actor, config=cfg)
            self.ds_engine = self.actor
            if self.critic is not None:
                # The critic is a full backbone copy, so leaving it on plain AdamW
                # kept its grads and fp32 Adam states replicated on every card -
                # more memory than the sharded policy. Its own ZeRO-2 engine shards
                # both, and gives it fp32 master weights so a 1e-5 update is not
                # rounded away in bf16.
                vlr = float(getattr(self.args, "critic_lr", 0.0) or self.args.lr)
                vcfg = zero2(micro_batch=micro, accum=accum, lr=vlr,
                             dtype=self.args.dtype)
                self.critic, self.critic_opt, _, _ = deepspeed.initialize(
                    model=self.critic, config=vcfg)
                self.ds_critic = self.critic
        else:
            self.opt = torch.optim.AdamW(params, lr=self.args.lr, betas=(0.9, 0.95))

    def init_sync(self, handle=None, client=None, packed: bool = False):
        """Open the weight-transfer channel once, before the loop."""
        from zero_ppo.engine.weight_sync import build
        self.sync = build(self.hf, rank=self.rank, world=self.world,
                          handle=handle, client=client, packed=packed)

    def _set_lr(self):
        """Apply the warmup + cosine schedule to the optimiser(s) for this update.

        DeepSpeed has no lr scheduler configured, so its step keeps whatever lr we
        set here; the plain AdamW path reads param_groups. `step_count` counts
        OPTIMISER UPDATES, so the schedule spans iters * ppo_epochs - otherwise the
        cosine would reach zero after iters/epochs rollout steps.
        """
        from zero_ppo.trainer import lr_at
        epochs = max(1, int(getattr(self.args, "ppo_epochs", 1)))
        total = int(self.args.iters) * epochs
        warm = getattr(self.args, "warmup_frac", 0.1)
        # Floor the cosine at min_lr * base_lr, matching open-r1's
        # cosine_with_min_lr / min_lr_rate. Default 0 keeps the old decay-to-zero.
        minf = float(getattr(self.args, "min_lr_frac", 0.0) or 0.0)
        lr = lr_at(self.step_count, total, self.args.lr,
                   warmup_frac=warm, min_frac=minf)
        self.current_lr = lr
        if self.ds_engine is not None:
            for group in self.ds_engine.optimizer.param_groups:
                group["lr"] = lr
        elif self.opt is not None:
            for group in self.opt.param_groups:
                group["lr"] = lr
        if self.critic_opt is not None:
            base = float(getattr(self.args, "critic_lr", 0.0) or self.args.lr)
            vlr = lr_at(self.step_count, total, base,
                        warmup_frac=warm, min_frac=minf)
            opt = (self.ds_critic.optimizer if self.ds_critic is not None
                   else self.critic_opt)
            for group in opt.param_groups:
                group["lr"] = vlr

    # ------------------------------------------------------------------- step
    def compute_values(self, ids, mask):
        """V(s) at every position, from the critic BEFORE this step's update.

        GAE needs these, so the driver asks for them before computing advantages.
        Returns None when there is no critic (the GRPO path).
        """
        import torch
        if self.critic is None:
            return None
        # After deepspeed.initialize the critic is an engine; forward it to the
        # wrapped module's `values` so the value head is used, not the LM head.
        self.critic.eval()
        with torch.no_grad():
            ids_d = ids.to(self.device)
            attn_d = (ids_d != self.pad_id)
            v = self._values(ids_d, attn_d).float().cpu()
        self.free_memory()
        return v

    def _values(self, ids_d, attn_d):
        """`values` through either the raw module or its DeepSpeed engine."""
        fn = getattr(self.critic, "values", None)
        if fn is None:
            fn = self.critic.module.values
        return fn(ids_d, attn_d)

    def _logprobs(self, ids_d, attn_d, gen_d):
        """`token_logprobs` through the engine: DeepSpeed does not proxy custom
        methods, so reach the wrapped module. The engine still owns the backward."""
        fn = getattr(self.actor, "token_logprobs", None)
        if fn is None:
            fn = self.actor.module.token_logprobs
        return fn(ids_d, attn_d, mask=gen_d, no_grad=False)

    def train_step(self, ids, mask, adv, returns=None):
        """`ppo_epochs` optimiser steps over ONE rollout batch (PPO's inner loop).

        The batch is generated once, then reused: `old_logp` is captured from the
        policy that produced it and held fixed, so the importance ratio actually
        moves and the clip does its job. Sampling is the expensive part, so reuse
        buys gradient steps without more rollouts - unlike `group`, which only
        repeats the same problem to give GRPO a baseline.

        `returns` is the critic's target (A + V_old), fixed for the whole batch.
        The critic loss is separate so value gradients cannot leak into the
        policy's objective, and each graph is backwarded exactly once.
        """
        import torch
        from zero_ppo.trainer import clipped_loss, ppo_diagnostics, kl_penalty

        self._ensure_opt()
        self.actor.train()
        ids_d = ids.to(self.device)
        gen_d = mask.to(self.device)
        attn_d = (ids_d != self.pad_id)
        adv_d = adv.to(self.device)
        tgt = returns.to(self.device) if returns is not None else None
        if self.critic is not None and tgt is None:
            raise ValueError("PPO needs `returns` as the critic target")

        epochs = max(1, int(getattr(self.args, "ppo_epochs", 1)))
        eps = float(getattr(self.args, "clip_eps", 0.1))
        micro, accum = self._split()
        B = ids_d.shape[0]
        chunks = [(s, min(s + micro, B)) for s in range(0, B, micro)]

        # The reference policy for the whole batch: the one that generated it.
        # Computed in MICRO-BATCH chunks, not in one shot: the lm_head projection
        # materialises (generated_tokens, vocab) and vocab is 151936, so a 64-row
        # batch needs ~11GiB for that tensor alone. Doing it outside the chunk loop
        # made MINI_BATCH useless against this peak and OOMed a 44GiB card. The
        # result is identical - each row's logprobs depend only on that row.
        with torch.no_grad():
            old_logp = torch.zeros_like(ids_d, dtype=torch.float32)
            ref_logp = None
            if self.ref is not None and self.kl_coef > 0:
                ref_logp = torch.zeros_like(ids_d, dtype=torch.float32)
            for a, b in chunks:
                old_logp[a:b] = self._logprobs(
                    ids_d[a:b], attn_d[a:b], gen_d[a:b]).detach()
                # The KL anchor is the FROZEN starting policy, not old_logp: the
                # clip already handles the latter, and it resets every rollout.
                if ref_logp is not None:
                    ref_logp[a:b] = self.ref.token_logprobs(
                        ids_d[a:b], attn_d[a:b], mask=gen_d[a:b],
                        no_grad=True).detach()
                self.free_memory()

        out = vout = kout = 0.0
        diag = {"approx_kl": 0.0, "clip_frac": 0.0}
        stopped_at = 0                        # epochs actually run (0 = all)
        last_approx = 0.0                     # previous epoch's mean approx_kl
        for ep in range(epochs):
            # Checked at the EPOCH boundary, never mid-chunk: DeepSpeed only
            # reduces and steps on the last micro-batch, so breaking inside the
            # accumulation window would leave gradients half-applied and its
            # micro-step counter out of sync with the next call.
            if self.target_kl > 0 and ep > 0 and last_approx > 1.5 * self.target_kl:
                stopped_at = ep
                break
            self._set_lr()                    # the schedule advances per update
            ep_loss = ep_v = ep_kl = 0.0
            ep_approx = 0.0
            for ci, (a, b) in enumerate(chunks):
                last = (ci == len(chunks) - 1)
                ids_c, attn_c, gen_c = ids_d[a:b], attn_d[a:b], gen_d[a:b]
                adv_c, old_c = adv_d[a:b], old_logp[a:b]

                logp = self._logprobs(ids_c, attn_c, gen_c)
                loss = clipped_loss(logp, old_c, adv_c, gen_c, eps=eps)
                d = ppo_diagnostics(logp, old_c, gen_c, eps=eps)
                diag["approx_kl"] += d["approx_kl"] / len(chunks)
                diag["clip_frac"] += d["clip_frac"] / len(chunks)
                # per-epoch value for the early-stop test: `diag` accumulates
                # across epochs, so testing it would inflate the threshold with
                # every pass and never fire
                ep_approx += d["approx_kl"] / len(chunks)

                kl_val = 0.0
                if ref_logp is not None:
                    kl = kl_penalty(logp, ref_logp[a:b], gen_c)
                    loss = loss + self.kl_coef * kl
                    kl_val = float(kl.detach())
                    ep_kl += kl_val / len(chunks)

                vloss = None
                if self.critic is not None:
                    self.critic.train()
                    values = self._values(ids_c, attn_c)
                    tc = tgt[a:b]
                    vloss = ((values - tc) ** 2 * gen_c).sum() / gen_c.sum().clamp(min=1)

                if self.ds_engine is not None:
                    # DeepSpeed divides by accum and only reduces/steps on the
                    # last micro-batch, so the update equals the unsplit one.
                    self.ds_engine.backward(loss)
                    self.ds_engine.step()
                    if vloss is not None:
                        self.ds_critic.backward(vloss)
                        self.ds_critic.step()
                else:
                    total = loss if vloss is None else loss + self.vf_coef * vloss
                    (total / accum).backward()
                    if last:
                        torch.nn.utils.clip_grad_norm_(
                            self.actor.trainable()
                            + (self.critic.trainable() if self.critic else []), 1.0)
                        self.opt.step()
                        self.opt.zero_grad()
                    del total

                ep_loss += float(loss.detach()) / len(chunks)
                if vloss is not None:
                    ep_v += float(vloss.detach()) / len(chunks)
                    del values, vloss, tc
                del logp, loss, ids_c, attn_c, gen_c, adv_c, old_c
                self.free_memory()            # release each micro-batch's graph

            self.step_count += 1              # one update per epoch, not per chunk
            out, vout, kout = ep_loss, ep_v, ep_kl
            last_approx = ep_approx

        del old_logp, ids_d, gen_d, attn_d, adv_d, tgt
        if ref_logp is not None:
            del ref_logp
        self.free_memory()
        return (out, self.current_lr, vout, diag["approx_kl"], diag["clip_frac"], kout,
                stopped_at)

    def free_memory(self):
        """Return cached (unused) GPU memory to the driver so the engine can wake."""
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    def do_sync(self):
        """Push the updated weights into the engine. Called on every rank."""
        if self.sync is None:
            raise RuntimeError("init_sync must run before the first sync")
        self.sync.send()
        # The IPC-shared copies are referenced until the send barrier completes;
        # release them before the engine wakes its KV cache.
        self.free_memory()

    def save(self, path: str) -> str:
        """Write the trained policy as a standalone HF checkpoint (rank 0 only).

        ZeRO-2 keeps a full weight copy on every rank, so rank 0's model is the
        complete policy - no gather needed. LoRA is merged into the base first so
        the result loads with a plain `from_pretrained`, no adapter required.
        """
        os.makedirs(path, exist_ok=True)
        model = self.hf
        if self.args.tuning == "lora":
            model = model.merge_and_unload()
        model.save_pretrained(path, safe_serialization=True)
        self.tok.save_pretrained(path)
        return path

    def trainable(self) -> bool:
        mod = getattr(self.actor, "module", self.actor)   # unwrap DeepSpeed
        return len([p for p in mod.parameters() if p.requires_grad]) > 0


# --------------------------------------------------------------------- self test
def _self_test() -> None:
    import torch
    from zero_ppo.reward.base import Sample

    class _Tok:
        pad_token_id = 0
        eos_token_id = 0

    rows = [
        {"ids": [1, 2, 3, 4], "spans": [(2, 3)], "seg_end": [3],
         "seg_ok": [True], "texts": ["a"]},
        {"ids": [5, 6], "spans": [(0, 1)], "seg_end": [1],
         "seg_ok": [None], "texts": ["b"]},
    ]
    ids, mask, samples = assemble(rows, _Tok(), 0, lambda i: str(i))
    assert ids.shape == (2, 4)
    assert ids[0].tolist() == [1, 2, 3, 4]
    assert ids[1].tolist() == [0, 0, 5, 6], ids[1].tolist()   # left padded
    assert mask[0].tolist() == [False, False, True, True]
    assert mask[1].tolist() == [False, False, True, True]
    assert samples[1].seg_end == [3]                          # offset into padded tensor
    assert isinstance(samples[0], Sample) and samples[0].gold == "0"
    # Early stop reads the PREVIOUS epoch's mean, not the running total: `diag`
    # accumulates across epochs, so testing it would raise the bar every pass and
    # the guard would never fire. Simulated here without a model.
    def _sim(target, per_epoch, epochs=4):
        last, ran, stopped = 0.0, 0, 0
        for ep in range(epochs):
            if target > 0 and ep > 0 and last > 1.5 * target:
                stopped = ep
                break
            ran += 1
            last = per_epoch[ep]
        return ran, stopped

    assert _sim(0.0, [0.0, 0.5, 0.9, 1.2]) == (4, 0), "disabled must run all epochs"
    assert _sim(0.02, [0.0, 0.005, 0.010, 0.015]) == (4, 0), "healthy drift runs on"
    assert _sim(0.02, [0.0, 0.050, 0.060, 0.070]) == (2, 2), "must stop after epoch 1"
    # epoch 0 is always allowed: old_logp is that policy, so its kl is 0 by construction
    assert _sim(1e-9, [0.0, 0.0, 0.0, 0.0]) == (4, 0)

    print("  train_actor self-test OK")
    print("    left padding, mask on generated tokens, segment offsets shifted")
    print("    target_kl early stop: per-epoch test, epoch 0 exempt, 0 disables")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()