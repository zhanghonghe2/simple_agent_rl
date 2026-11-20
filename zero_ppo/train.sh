#!/usr/bin/env bash
# One command for a training run, on Ray: one cluster, colocated train + inference.
#
# The driver does ray.init() once, starts one TrainActor per card (DeepSpeed ZeRO-2)
# and one EngineActor (vLLM TP=N), and drives the loop. No torchrun, no separate
# server, no handle shipping between independent clusters.
#
#   bash zero_ppo/train.sh                       # defaults below
#   STEPS=40 BATCH=8 GROUP=4 bash zero_ppo/train.sh
set -euo pipefail

# START FROM THE SFT MODEL, not the raw Math-Instruct: RL's gain is measured
# against SFT, so training from the base would have to relearn the tool protocol.
# Pass MODEL=/path/to/sft explicitly - there is no safe default for it.
MODEL=${MODEL:-}
# The ShareGPT pool: system + first human turn, figure-dependent items removed.
# rl_pool_v3 = 457 problems (320 hard + 137 easy), selected by pass rate from
# problems never used in SFT or evaluation. The older rl_train.jsonl (162 items)
# is kept for reference only: every one of its problems was already in the SFT
# corpus, so RL on it rewards recall rather than reasoning.
POOL=${POOL:-agent_data_math/rl_pool_v3.jsonl}
OUT=${OUT:-runs/zero_ppo}
GPUS=${GPUS:-2}
STEPS=${STEPS:-40}
BATCH=${BATCH:-4}
# PPO takes its baseline from the critic, so repeating a problem buys nothing:
# group only multiplies the trajectories per step (batch x group) and the memory.
# GRPO is the one that needs group >= 2.
GROUP=${GROUP:-1}
# Shuffle the pool once at startup (fixed seed). The pool is ordered hard-then-easy;
# unshuffled sequential batches make reward oscillate with the curriculum.
SHUFFLE=${SHUFFLE:-0}
SHUFFLE_SEED=${SHUFFLE_SEED:-0}
# LUFFY-style guidance: of each GROUP samples, LUFFY come from the external API
# model (same tool loop + sandbox), the rest from the policy. 0 disables.
LUFFY=${LUFFY:-0}
LUFFY_MODEL=${LUFFY_MODEL:-deepseek-chat}
LUFFY_MAX_TOKENS=${LUFFY_MAX_TOKENS:-1024}
LUFFY_WORKERS=${LUFFY_WORKERS:-8}
LR=${LR:-5e-6}
WARMUP_FRAC=${WARMUP_FRAC:-0.1}
# Floor the cosine at MIN_LR_FRAC * LR. open-r1 uses cosine_with_min_lr with
# min_lr_rate 0.1; 0 keeps our previous decay-to-zero behaviour.
MIN_LR_FRAC=${MIN_LR_FRAC:-0.0}
# PPO carries the process reward (per-turn credit, GAE propagates it).
# GRPO is outcome-only: ALGO=grpo CRITIC=none REWARD=zero_ppo/reward/agent_reward_test.py
ALGO=${ALGO:-ppo}
CRITIC=${CRITIC:-full}
# Outcome-gated process reward: 2.0 for a correct answer, up to 1.0 split over the
# turns, each turn judged by an LLM. No answer -> no process credit at all, which
# closes the "keep calling tools, never answer" loophole.
REWARD=${REWARD:-zero_ppo/reward/ppo_agent_judge.py}
JUDGE_WORKERS=${JUDGE_WORKERS:-16}
JUDGE_MODEL=${JUDGE_MODEL:-deepseek-chat}
CRITIC_LR=${CRITIC_LR:-1e-5}
VF_COEF=${VF_COEF:-0.5}
GAMMA=${GAMMA:-1.0}
LAM=${LAM:-0.95}
# reuse each rollout batch this many times (PPO inner loop). Sampling is the costly
# part; 1 would make the importance ratio identically one and disable the clip.
PPO_EPOCHS=${PPO_EPOCHS:-4}
CLIP_EPS=${CLIP_EPS:-0.2}
# Anchor the policy to the frozen SFT model. The clip only bounds movement within
# one rollout batch; without this the run drifted off-distribution and collapsed
# into degenerate text while its weights stayed finite. 0 disables the reference.
KL_COEF=${KL_COEF:-0.02}
# Stop the inner epochs once approx_kl (drift from the policy that GENERATED this
# batch) exceeds 1.5x this. The clip bounds each token's ratio; many small drifts
# can still sum to a large move, and this catches that. 0 disables.
TARGET_KL=${TARGET_KL:-0.02}
# Trajectories per forward pass. Splitting + accumulating keeps the update
# identical but cuts activation memory, which pays for the reference model.
MINI_BATCH=${MINI_BATCH:-2}
# Weak-signal attenuation for whitening: batches whose raw mean|A| is below this
# keep a proportionally weaker gradient. Without it, an all-wrong batch (raw |A|
# ~0.05, mostly critic error) was whitened to the same strength as a real signal
# and the policy collapsed. 0 disables (plain whitening, for ablations).
SIGNAL_REF=${SIGNAL_REF:-0.5}
VLLM_UTIL=${VLLM_UTIL:-0.30}
SAVE_DIR=${SAVE_DIR:-$OUT/checkpoints}
SAVE_EVERY=${SAVE_EVERY:-0}
# Greedy held-out eval every N steps, on the COLOCATED engine (no second vLLM).
# Keeps the best checkpoint as <SAVE_DIR>/best. 0 disables.
EVAL_EVERY=${EVAL_EVERY:-0}
EVAL_POOL=${EVAL_POOL:-agent_data_math/math500.jsonl}
EVAL_LIMIT=${EVAL_LIMIT:-0}
EVAL_BATCH=${EVAL_BATCH:-64}
EVAL_MAX_NEW=${EVAL_MAX_NEW:-1024}

# stale workers from an earlier run hold the cards and make the next attempt hang
pkill -9 -f "VLLM::Worker" 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 1

if [[ -z "$MODEL" ]]; then
    echo "  ERROR: 必须指定 MODEL=<你的 SFT 模型路径>" >&2
    echo "  RL 的起点是 SFT 后的模型；用原始 Math-Instruct 会让 RL 先重新学工具协议。" >&2
    echo "  例: MODEL=/path/to/your_sft BATCH=4 GROUP=1 bash zero_ppo/train.sh" >&2
    exit 1
fi
if [[ ! -f "$POOL" ]]; then
    echo "  ERROR: 训练池不存在: $POOL" >&2
    exit 1
fi

# Always keep a log: terminal scrollback is lost the moment SSH drops, and the
# metrics.jsonl written by the driver is structured but this file keeps stdout and
# stderr (vLLM INFO, tracebacks) too. LOG=0 disables it if you really want only TTY.
STAMP=$(date +%m%d_%H%M)
LOG_DIR=${LOG_DIR:-$OUT/logs}
if [[ "${LOG:-1}" != "0" ]]; then
    mkdir -p "$LOG_DIR"
    OUT_LOG="$LOG_DIR/train.$STAMP.out.log"
    ERR_LOG="$LOG_DIR/train.$STAMP.err.log"
else
    OUT_LOG=""
fi

echo "  training on Ray ($GPUS cards, $STEPS steps, $BATCH x $GROUP, algo=$ALGO)"
echo "  model=$MODEL"
echo "  pool=$POOL · ppo_epochs=$PPO_EPOCHS · clip_eps=$CLIP_EPS · lam=$LAM"
echo "  kl_coef=$KL_COEF (锚定SFT) · mini_batch=$MINI_BATCH (梯度累积) · 优势白化=on · signal_ref=$SIGNAL_REF (弱信号衰减)"
echo "  target_kl=$TARGET_KL (内层早停, 超1.5倍即止; 0=关闭)"
echo "  更新总次数 = $STEPS x $PPO_EPOCHS (学习率调度按此铺开)"
if [[ "$SHUFFLE" != "0" ]]; then
    echo "  训练池洗牌: seed=$SHUFFLE_SEED"
fi
if [[ "$LUFFY" != "0" ]]; then
    echo "  LUFFY: 每组 $GROUP 条中 $LUFFY 条来自 $LUFFY_MODEL (同一工具循环+沙箱)"
fi
if [[ "$EVAL_EVERY" != "0" ]]; then
    echo "  内联评测: 每${EVAL_EVERY}步 · $EVAL_POOL · 贪心 · 最佳存 $SAVE_DIR/best"
fi
echo "  metrics=$OUT/metrics.jsonl"
[[ -n "$OUT_LOG" ]] && echo "  stdout -> $OUT_LOG · stderr -> $ERR_LOG"
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here. Expandable
# segments back weights with cuMemMap virtual memory; sharing that over CUDA IPC
# needs the pidfd_getfd syscall, which the kernel denies without ptrace rights
# ("pidfd_getfd: Operation not permitted"). Plain cudaMalloc memory uses classic
# IPC handles and works. See pytorch/pytorch#165685.
# PYTHONUNBUFFERED so the log file fills in real time, not at the next flush.
RUN_CMD=(python3 -m zero_ppo.driver
    --model "$MODEL" --pool "$POOL"
    --algo "$ALGO" --critic-mode "$CRITIC"
    --batch "$BATCH" --group "$GROUP"
    --shuffle "$SHUFFLE" --shuffle-seed "$SHUFFLE_SEED"
    --luffy "$LUFFY" --luffy-model "$LUFFY_MODEL"
    --luffy-max-tokens "$LUFFY_MAX_TOKENS" --luffy-workers "$LUFFY_WORKERS"
    --reward "$REWARD"
    --world "$GPUS"
    --vllm-util "$VLLM_UTIL" --max-model-len 4096
    --dtype bfloat16 --prefix-cache 1
    --tuning full --deepspeed 1 --gradient-checkpoint 1
    --out "$OUT" --iters "$STEPS" --lr "$LR" --warmup-frac "$WARMUP_FRAC"
    --min-lr-frac "$MIN_LR_FRAC"
    --critic-lr "$CRITIC_LR" --vf-coef "$VF_COEF"
    --gamma "$GAMMA" --lam "$LAM"
    --ppo-epochs "$PPO_EPOCHS" --clip-eps "$CLIP_EPS"
    --kl-coef "$KL_COEF" --mini-batch "$MINI_BATCH" --target-kl "$TARGET_KL"
    --signal-ref "$SIGNAL_REF"
    --save-dir "$SAVE_DIR" --save-every "$SAVE_EVERY"
    --eval-every "$EVAL_EVERY" --eval-pool "$EVAL_POOL"
    --eval-limit "$EVAL_LIMIT" --eval-batch "$EVAL_BATCH"
    --eval-max-new "$EVAL_MAX_NEW")

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export JUDGE_WORKERS JUDGE_MODEL
export PYTHONUNBUFFERED=1 RAY_DEDUP_LOGS=0
if [[ -n "$OUT_LOG" ]]; then
    # tee keeps stdout on screen AND in the file; stderr goes to its own file
    "${RUN_CMD[@]}" > >(tee -a "$OUT_LOG") 2> >(tee -a "$ERR_LOG" >&2)
else
    "${RUN_CMD[@]}"
fi