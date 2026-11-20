# simple_agent_rl

从零实现的 **多轮工具调用 RL 训练框架**：让语言模型写 Python 代码、真实执行、
读取输出后继续推理，用 PPO 或 GRPO 优化最终答案的正确性。

面向小模型（1.5B / 3B）在数学推理上的强化学习实验。参考 [verl](https://github.com/volcengine/verl)
的分层思路，但代码量刻意保持很小，每个模块都能在 CPU 上单独自测。

**核心设计目标：加一个实验 = 加一个奖励文件。** 换算法、换组大小、换模型、换推理后端
都只是一个参数。

---

## 特性

- **两种算法，一个开关**：`ALGO=ppo`（critic + GAE，支持过程奖励）/ `ALGO=grpo`（组内相对优势，仅结果奖励）
- **真实工具执行**：模型写代码 → 子进程沙箱执行 → `<output>` 回填 → 继续推理，最多 6 轮
- **奖励即文件**：`reward/` 下每个文件是一个方案，按路径加载，各自带 `--self-test`
- **训练推理共卡**：一个 Ray 集群内，每卡一个 DeepSpeed ZeRO-2 训练 actor + 一个 vLLM 引擎（TP=N），
  通过 CUDA IPC 同步权重，睡眠/唤醒交替释放显存
- **外部教师引导（LUFFY 简化版）**：每组采样中指定若干条由外部 API 模型生成，
  走**同一套工具循环和沙箱**，解决"模型完全不会的题没有正信号"的死锁
- **训练中评测**：每 N 步在共卡引擎上跑贪心评测，自动保存最佳 checkpoint
- **29 个 CPU 自测**，不需要 GPU 就能验证算法层、奖励、提示词渲染、沙箱等逻辑

---

## 安装

需要 NVIDIA GPU（开发验证环境为 2×L40S 48GB）。

```bash
pip install torch transformers vllm ray deepspeed peft requests sympy
# 可选：更准的数学判等
pip install math_verify
```

> 版本未固定。vLLM 的权重传输接口（`vllm.distributed.weight_transfer`）在不同版本间
> 有差异，建议使用与 verl 相近的较新版本。

把仓库放在工作目录下，以包的方式运行（`python3 -m simple_agent_rl.driver`），
或保留目录名 `zero_ppo` 并使用下文的命令原文。

---

## 快速开始

### 1. 数据

数据放在仓库同级的 `agent_data_math/` 下，ShareGPT 风格 jsonl，每行一题：

```json
{
  "system": "You are a math expert with access to a Python interpreter...",
  "conversations": [{"from": "human", "value": "Problem:\n..."}],
  "gold": "42",
  "level": 3,
  "subject": "Algebra"
}
```

`gold` 是标准答案，只用于判分，**不会进入提示词**。`level` / `subject` 用于分桶统计。

随仓库提供的文件：

| 文件 | 条数 | 用途 |
|---|---|---|
| `agent_data_math/rl_pool_v3.jsonl` | 457 | **默认训练池**：320 难题 + 137 简单题，按通过率筛选 |
| `agent_data_math/math500.jsonl` | 458 | **默认评测集**（MATH-500 去掉依赖图形的题） |
| `agent_data_math/rl_candidates.jsonl` | 3247 | 选题候选：未被 SFT 用过、且不在评测集中 |
| `agent_data_math/sft_tir_v4.jsonl` | 4821 | SFT 语料；此处只用于**排除已用题**和复制 system 文本 |
| `agent_data_math/overfit32.jsonl` | 32 | 过拟合诊断用的小池 |
| `agent_data_math/rl_pool_v2.jsonl` | 661 | 上一版训练池（保留供对比） |
| `agent_data_math/rl_train.jsonl` | 162 | 早期池，**全部题目都在 SFT 语料里**，不建议再用 |
| `agent_data_math/rl_pool_probe*.jsonl` | 7 / 28 | 早期小规模探测的产物 |

训练池与评测集互不重叠，也都排除了 SFT 用过的题，因此评测衡量的是泛化而非记忆。

`rl_candidates.jsonl` 由 MATH 训练集（7496 题）扣除 SFT 已用的 4247 题得到，
原集不在本仓库内，可自行从 [MATH](https://github.com/hendrycks/math) 获取后
用 `select_pool.py` 重新生成候选与训练池。

### 2. 训练

`MODEL` 必须显式指定，且应当是**已经 SFT 过的模型**——从原始 base 起步会让 RL
先重新学习工具协议，浪费预算。

```bash
export DEEPSEEK_API_KEY=sk-...        # 判分与教师引导用

MODEL=/path/to/your_sft_model \
POOL=agent_data_math/rl_pool_v3.jsonl \
GPUS=2 ALGO=grpo CRITIC=none \
REWARD=zero_ppo/reward/reward_result.py \
BATCH=16 GROUP=8 LUFFY=4 MINI_BATCH=4 PPO_EPOCHS=4 STEPS=40 \
LR=5e-6 KL_COEF=0.02 SHUFFLE=1 EVAL_EVERY=10 \
bash zero_ppo/train.sh
```

`POOL` 和 `EVAL_POOL` 已有默认值（分别是 `rl_pool_v3.jsonl` 和 `math500.jsonl`），
上面显式写出只是为了清楚。最简形式：

```bash
MODEL=/path/to/your_sft_model ALGO=grpo CRITIC=none \
REWARD=zero_ppo/reward/reward_result.py GROUP=8 bash zero_ppo/train.sh
```

PPO（带过程奖励）：

```bash
MODEL=/path/to/your_sft_model \
POOL=agent_data_math/rl_pool_v3.jsonl \
ALGO=ppo CRITIC=full REWARD=zero_ppo/reward/ppo_agent_judge.py \
BATCH=8 GROUP=1 bash zero_ppo/train.sh
```

### 3. 评测

```bash
python3 -m zero_ppo.eval \
  --model runs/zero_ppo/checkpoints/best \
  --pool agent_data_math/math500.jsonl \
  --samples 8 --ks 1,2,4,8 --seed 0 --judge-api \
  --out runs/eval_rl

# 起点模型也测一遍，作为参照
python3 -m zero_ppo.eval \
  --model /path/to/your_sft_model \
  --pool agent_data_math/math500.jsonl \
  --samples 8 --seed 0 --judge-api \
  --out runs/eval_sft

# 并列对比（自动校验 seed / samples / 判分口径是否一致）
python3 -m zero_ppo.compare sft=runs/eval_sft rl=runs/eval_rl
```

跨 checkpoint 比较时 **`--seed` 和 `--samples` 必须一致**，否则 pass@k 的差异里
混着采样噪声；`--judge-api` 也要两边都开或都不开，否则宽口径不可比。

---

## 加一个奖励实验

这是框架的主要用途。复制一个现有文件，改打分逻辑，跑自测，用路径加载：

```bash
cp zero_ppo/reward/reward_result.py zero_ppo/reward/my_reward.py
# 编辑打分逻辑
python3 -m zero_ppo.reward.my_reward --self-test
REWARD=zero_ppo/reward/my_reward.py bash zero_ppo/train.sh
```

契约很小（`reward/base.py`）：

```python
class Reward:
    name = "my_reward"

    def __call__(self, samples: list[Sample], seq_len: int) -> torch.Tensor:
        R = torch.zeros(len(samples), seq_len)
        for i, s in enumerate(samples):
            R[i, s.done] += 1.0 if correct(s) else 0.0
        return R

    def summary(self, samples, rewards) -> dict:   # 可选，进日志
        return {...}
```

`Sample` 已经把位置换算成 padded 张量里的下标，奖励文件不需要关心 padding、
偏移或模型：

| 字段 | 含义 |
|---|---|
| `gold` | 标准答案 |
| `texts` | 各轮 assistant 文本，按顺序 |
| `seg_end[k]` | 第 k 轮最后一个 token 的位置 |
| `seg_ok[k]` | 该轮工具是否执行成功；纯推理轮为 `None` |
| `seg_kinds[k]` | `"code"` 或 `"nl"` |
| `seg_outs[k]` | 该轮程序的真实输出 |
| `done` | 最后一个生成位置 |
| `gen_positions(k)` | 第 k 轮**被 mask 的**生成位置（工具回填的 token 不算） |

文件末尾需要暴露一个 `REWARD = MyReward()` 实例。

### 内置奖励

| 文件 | 口径 | 适用 |
|---|---|---|
| `reward_result.py` | 结果分：API 判数值等价，对 1 错 0，无 `\boxed{}` 记 0 | GRPO |
| `ppo_agent_judge.py` | 结果门控 + 过程分：答对给 2.0，另 1.0 按通过轮数分摊到各轮段末 | PPO |
| `agent_reward_test.py` | 结果 + 本地规则过程分，无需 API | 冒烟测试 |

**结果门控**很重要：过程分只在最终答案正确时才发放，否则"一直调工具、永不作答"
会成为最优策略——这个漏洞在早期实验中真实发生过。

---

## 外部教师引导（LUFFY 简化版）

纯 RL 有个硬约束：**模型必须偶尔采样到正确轨迹才有正信号**。全组答错时 GRPO 的
组内方差为 0，优势恒 0，该题永远学不动。

`LUFFY=k` 让每组 `GROUP` 条采样中有 `k` 条来自外部 API 模型：

```bash
GROUP=8 LUFFY=4 LUFFY_MODEL=deepseek-chat bash zero_ppo/train.sh
```

教师轨迹走**完全相同的工具循环、沙箱和对话模板**，因此落在训练分布上；
标准答案永不外发。数学语义上这是**优势加权的模仿**（重要性比率从 1 起步），
不是 [LUFFY 原文](https://arxiv.org/abs/2504.14945) 的 shaped importance sampling。

需要 `DEEPSEEK_API_KEY`（或自定义 `--luffy-key-env`）。缺 key 会直接报错，
不会静默产出空轨迹。

---

## 训练数据筛选

`select_pool.py` 两阶段选题：先做无 GPU 的预筛（去重、排除 SFT/评测集、
剔除依赖图形的题），再用当前策略采样估计通过率，保留信息量最大的中间带。

默认输入 `rl_candidates.jsonl`，默认输出 `rl_pool_v3.jsonl`，
排除集自动取 `sft_tir_v4.jsonl` 与 `math500.jsonl`。

```bash
# 阶段一：只看预筛统计，不采样（无需 GPU）
python3 -m zero_ppo.select_pool --dry-run

# 阶段二：按通过率选题
python3 -m zero_ppo.select_pool \
  --model /path/to/your_sft_model \
  --candidates agent_data_math/rl_candidates.jsonl \
  --sft agent_data_math/sft_tir_v4.jsonl \
  --eval agent_data_math/math500.jsonl \
  --samples 8 --lo 0.1 --hi 0.9 --min-level 2 --easy-frac 0.3 \
  --report runs/probe.jsonl \
  --out agent_data_math/rl_pool_v3.jsonl
```

过拟合诊断用的小池（从已筛池直接抽，无需再采样）：

```bash
python3 -m zero_ppo.make_overfit \
  --pool agent_data_math/rl_pool_v3.jsonl \
  --n 32 --out agent_data_math/overfit32.jsonl
```

- 全对（通过率 1）和全错（0）的题都不提供梯度，默认排除
- `--easy-frac 0.3` 混入 30% 简单题：只训难题会让模型对简单题过度思考
- 启用教师引导后，"模型全错"的题可以重新纳入，但筛选标准应改为**教师通过率 > 0**

---

## 项目结构

```
zero_ppo/
├── train.sh                一条命令启动训练（全部配置走环境变量）
├── driver.py               编排：ray.init → 起 actor → 训练循环 → 评测 → 存档
├── train_actor.py          每卡一个训练 worker：前向、反向、优化器、权重推送
├── trainer.py              纯张量层：裁剪损失、KL 惩罚、诊断（CPU 可测）
├── actor.py                策略模型：只在 mask 处 gather，fused cross_entropy
├── critic.py               价值模型三种模式：full / lora_shared / lora_split
├── algo/advantages.py      GAE（PPO）/ 组内相对优势（GRPO）/ 白化
├── engine/
│   ├── base.py             引擎协议：generate / sleep / wake / 权重同步
│   ├── ray_actor.py        Ray 后端（当前主路径）
│   ├── local.py            进程内 vLLM
│   ├── http.py             HTTP 后端（历史实现，不推荐）
│   ├── colocate.py         共卡显存编排与状态检查
│   └── weight_sync.py      CUDA IPC 权重推送
├── rollout/
│   ├── multiturn.py        多轮 wavefront：一轮一次 generate，支持按行注入教师
│   ├── api_gen.py          外部模型轨迹生成器
│   └── tools.py            工具执行入口
├── reward/                 奖励方案（见上文）
├── inline_eval.py          训练中的贪心评测 + 最佳点追踪
├── eval.py                 离线评测：贪心 + pass@k，分卡分片
├── compare.py              多份报告并列对比 + 口径一致性校验
├── select_pool.py          训练池筛选
├── make_overfit.py         小规模过拟合实验的选题工具
└── util/
    ├── sandbox.py          子进程执行 + 拒绝名单 + 超时
    ├── judge.py            答案提取与判等（数值 / sympy / 可选 math_verify）
    └── prompts.py          对话模板渲染，与 SFT 阶段逐字节一致
```

---

## 自测

不需要 GPU：

```bash
for m in zero_ppo.algo.advantages zero_ppo.trainer zero_ppo.actor zero_ppo.critic \
         zero_ppo.reward.reward_result zero_ppo.reward.ppo_agent_judge \
         zero_ppo.rollout.multiturn zero_ppo.rollout.api_gen \
         zero_ppo.util.judge zero_ppo.util.prompts zero_ppo.util.sandbox \
         zero_ppo.inline_eval zero_ppo.select_pool zero_ppo.eval; do
    python3 -m $m --self-test || echo "FAILED: $m"
done
```

共 29 个模块带自测，覆盖优势计算、奖励打分、提示词渲染、沙箱、教师注入、
最佳点追踪等。

---

## 主要配置

全部通过环境变量传给 `train.sh`，完整列表见脚本头部注释。

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL` | 必填 | 起点模型，应为 SFT 后的权重 |
| `POOL` | `agent_data_math/rl_pool_v3.jsonl` | 训练池（457 题） |
| `EVAL_POOL` | `agent_data_math/math500.jsonl` | 内联评测集（458 题） |
| `ALGO` / `CRITIC` | `ppo` / `full` | GRPO 用 `ALGO=grpo CRITIC=none` |
| `REWARD` | `ppo_agent_judge.py` | 奖励文件路径 |
| `BATCH` / `GROUP` | 4 / 1 | 每步题数 / 每题采样数；GRPO 需 `GROUP≥2` |
| `PPO_EPOCHS` | 4 | 每批数据复用次数；设 1 会让重要性比率恒为 1、clip 失效 |
| `MINI_BATCH` | 2 | 每次前向的轨迹数，梯度累积保持更新等价 |
| `LR` / `MIN_LR_FRAC` | 5e-6 / 0 | 余弦调度的峰值与保底比例 |
| `KL_COEF` | 0.02 | 对冻结参考模型的 KL 锚 |
| `TARGET_KL` | 0.02 | 内层早停阈值（超过 1.5 倍即止） |
| `SIGNAL_REF` | 0.5 | 优势白化的弱信号衰减参考尺度 |
| `SHUFFLE` | 0 | 启动时按固定种子洗一次训练池 |
| `LUFFY` | 0 | 每组中来自外部教师的轨迹数 |
| `EVAL_EVERY` | 0 | 每 N 步内联评测，最佳存 `checkpoints/best` |
| `SAVE_EVERY` | 0 | 每 N 步存一次 checkpoint |

输出：`runs/zero_ppo/metrics.jsonl`（每步一行 JSON）、
`logs/train.<stamp>.{out,err}.log`、`best_eval.json`、`checkpoints/`。

---

## 实践笔记

这些是实验中真实踩过的坑，代码注释里有更详细的说明。

**测量**

- 小样本上 1 个百分点只有几道题，**3~4 点以下的差异不足以确证**；同配置重复运行
  本身就有约 ±2 点抖动，贪心也不例外
- 训练 reward 与能力不是一回事：分布变尖时贪心可能上升而采样期望下降
- 判分口径必须对齐。`math_verify` 对可解析但判不定的输入会返回 `False` 而非抛异常，
  曾导致"宽口径低于严格口径"，现已强制宽口径包含严格口径

**优化**

- 优势白化**只缩放、不减均值**：全错批次的优势本来一致为负（"整体压低"），
  减均值会把它变成一半正一半负，等于在 critic 的预测误差上做强化
- 除以标准差会抹掉信号强度，因此加了弱信号衰减（`SIGNAL_REF`）
- KL 惩罚用 k3 估计并做软饱和：指数项在少数 token 概率崩塌时会爆炸，
  而硬截断会让梯度在最需要拉回的地方归零

**工程**

- 共卡必须用 **IPC** 权重传输，NCCL 会报 `Duplicate GPU`
- 唤醒顺序是 `sleep → 释放显存 → wake(weights) → wake(kv_cache)`，反了会段错误
- 不要设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：它用 cuMemMap 虚拟内存，
  跨进程共享需要 `pidfd_getfd`，内核默认拒绝
- `old_logp` 必须按 micro-batch 分块计算：词表 15 万，整批 logits 会吃掉十几 GiB

---

## 安全提示

`util/sandbox.py` 用**子进程 + 拒绝名单 + 超时**执行模型生成的代码。
拒绝名单只是正则匹配，**不是安全边界**——它可以被绕过。

在不可信输入或共享机器上使用前，请替换为真正的隔离方案（容器、gVisor、
远程沙箱等）。该模块被刻意独立出来，就是为了方便替换。

外部教师引导会把**题目文本和模型自己的历史轮次**发送到第三方 API。
标准答案不会外发，但题目内容会离开本机，请自行评估数据合规性。

---

## 已知限制

- LoRA 模式下 actor/critic 共享适配器会互相污染，建议 `lora_split` 或全参微调
- 优化器状态未纳入 checkpoint，续训不是精确恢复
- HTTP 编排后端为历史实现，权重同步在该路径上不可靠，不建议使用
- 训练侧采样 `top_p` 用引擎默认值，离线评测默认 0.95，两者口径略有差异

---

## 致谢

分层设计参考 [verl](https://github.com/volcengine/verl)；
教师引导思路来自 [LUFFY](https://arxiv.org/abs/2504.14945)；
超参与调度对齐 [open-r1](https://github.com/huggingface/open-r1) 的公开配方。