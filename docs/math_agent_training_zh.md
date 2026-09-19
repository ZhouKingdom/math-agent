# Math Agent 完整训练指南

本文给出 Math Agent 在单机 8 卡上的完整 SFT 与标准 GRPO baseline 训练方式。两条训练链相互独立：

- SFT：`Qwen3-8B-Base` + ReTool SFT；
- RL：`Qwen3-8B` + 去重后的 DAPO-Math-17k + 在线无状态 Python 工具。

截至本文写入时，模型转换、数据处理、SFT 单步更新、RL 多轮工具调用与单步更新均已通过；两份完整训练脚本已通过固定镜像下的完整数据预检和命令展开检查，但尚未跑完整个 3 epoch SFT 或 3,000 update RL。不要把“完整脚本可启动”写成“完整训练结果已验证”。

## 1. 脚本与默认实验

| 阶段 | 脚本 | 默认输入模型 | 数据 | 默认预算 |
|---|---|---|---|---|
| SFT | `scripts/run_sft_full.sh` | `Qwen3-8B-Base_torch_dist` | 1,992 条 ReTool | 3 epoch，747 updates |
| RL baseline | `scripts/run_rl_grpo.sh` | `Qwen3-8B_torch_dist` | 17,243 条 DAPO-Math | 3,000 updates |

Slime 将转换后的 `release` checkpoint 视为 iteration 0，并从 rollout id 1 开始训练；同时，`--num-rollout` 是循环的开区间上界。脚本接收的是更直观的 `TRAIN_UPDATES`，传给 Slime 时自动加一。例如 RL 的 3,000 次更新会传入 `--num-rollout 3001`，最后保存 iteration 3000。

两个脚本默认启用本地 TensorBoard，默认不启用 W&B。它们都会：

- 校验固定模型、Megatron checkpoint 和数据是否存在；
- 对处理后数据执行预期行数检查；
- 在输出目录非空时拒绝覆盖；
- 仅在 `RESUME=1` 时从同一训练目录恢复模型、优化器和数据游标；
- 在 GPU 已被占用或已有 Ray 集群时拒绝启动；
- 在退出时只停止脚本自己启动的 Ray 集群；
- 把控制台日志写入 `logs/train/`。

## 2. 目标服务器基础预检

先在目标服务器的**宿主机**完成以下检查。这里的 `/path/to/math-agent` 应替换为目标服务器上的真实仓库路径，例如 `/home/<user>/math-agent`：

```bash
cd /path/to/math-agent
MATH_SHARED_ROOT=/path/to/shared-storage/math-agent

test -f data/raw/retool_sft/train_2000.parquet
test -f data/raw/dapo_math_17k/data/dapo-math-17k.parquet

nvidia-smi
test "$(docker image inspect math-agent/slime-runtime:20260903-cu129 \
  --format '{{.Id}}')" = \
  sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c
test "$(docker image inspect python:3.11-slim \
  --format '{{.Id}}')" = \
  sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76
mkdir -p "${MATH_SHARED_ROOT}/models" "${MATH_SHARED_ROOT}/checkpoints"
df -h "${MATH_SHARED_ROOT}"
```

这里有意不检查 `Qwen3-8B-Base_torch_dist` 和 `Qwen3-8B_torch_dist`：模型不通过移动硬盘迁移，而是在目标服务器按下一节下载和转换。

默认 checkpoint 包含优化器状态，当前 8B 单个 checkpoint 实测约为 107 GB。考虑 Slime 当前按 `(rollout_id + 1) % save_interval` 判断周期保存且总会保存最终 step，SFT 默认会留下约 3 个 checkpoint（约 321 GB），RL 默认会留下约 7 个 checkpoint（约 749 GB）。还应为临时写入、日志和 Docker 留出额外空间。Slime 当前不会替脚本自动清理旧 checkpoint。

## 3. 在目标服务器下载并转换模型

以下命令在目标服务器的**宿主机**执行。下载过程复用已迁移的 Slime 镜像，因此宿主机不需要另配 Python 或 Hugging Face CLI。两套 Hugging Face 权重固定到下面验证过的 revision，并直接写入目标服务器的共享存储：

```bash
cd /path/to/math-agent
MATH_SHARED_ROOT=/path/to/shared-storage/math-agent
MODEL_ROOT="${MATH_SHARED_ROOT}/models"
RUNTIME_IMAGE=math-agent/slime-runtime:20260903-cu129
mkdir -p "${MODEL_ROOT}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "${MODEL_ROOT}:${MODEL_ROOT}" \
  -e HOME=/tmp \
  -e HF_HOME="${MODEL_ROOT}/.cache/huggingface" \
  -e MODEL_ROOT="${MODEL_ROOT}" \
  --entrypoint bash \
  "${RUNTIME_IMAGE}" \
  -lc '
    set -euo pipefail
    hf download Qwen/Qwen3-8B-Base \
      --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 \
      --local-dir "${MODEL_ROOT}/Qwen3-8B-Base"
    hf download Qwen/Qwen3-8B \
      --revision b968826d9c46dd6066d109eabc6255188de91218 \
      --local-dir "${MODEL_ROOT}/Qwen3-8B"
  '
```

下载完成并获得一张独占 GPU 后，在同一宿主机运行仓库内的转换脚本。脚本默认在 GPU 0 上顺序转换 Base 和 Instruct 两套模型；若检测到该卡已有计算进程会拒绝启动：

```bash
MATH_SHARED_ROOT=/path/to/shared-storage/math-agent
MODEL_ROOT="${MATH_SHARED_ROOT}/models"
RUNTIME_IMAGE=math-agent/slime-runtime:20260903-cu129

MATH_SHARED_ROOT="${MATH_SHARED_ROOT}" \
MODEL_ROOT="${MODEL_ROOT}" \
RUNTIME_IMAGE="${RUNTIME_IMAGE}" \
CONVERT_GPU=0 \
bash scripts/convert_qwen3_8b_to_torch_dist.sh all

test -f "${MODEL_ROOT}/Qwen3-8B-Base_torch_dist/latest_checkpointed_iteration.txt"
test -f "${MODEL_ROOT}/Qwen3-8B_torch_dist/latest_checkpointed_iteration.txt"
```

可以用 `base` 或 `instruct` 只转换一套；转换日志写入宿主机仓库的 `logs/conversion/`。这一步仅生成训练所需的 Megatron `torch_dist` checkpoint，不会开始 SFT 或 RL。

## 4. 进入固定 Slime 环境

训练脚本应在固定的 Slime 容器中运行。RL 需要创建同级 Python 沙箱，因此必须挂载 Docker CLI 和 socket。为了让 SFT/RL 使用同一条进入命令，可以统一执行：

下面整段命令在**宿主机**执行：

```bash
cd /path/to/math-agent
MATH_AGENT_ROOT="$(pwd -P)"
MATH_SHARED_ROOT=/path/to/shared-storage/math-agent
RUNTIME_IMAGE=math-agent/slime-runtime:20260903-cu129

docker run -it \
  --name math-agent-train \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${MATH_AGENT_ROOT}:/workspace/math-agent" \
  -v "${MATH_SHARED_ROOT}:${MATH_SHARED_ROOT}" \
  -v /usr/bin/docker:/usr/local/bin/docker:ro \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w /workspace/math-agent \
  -e SLIME_ROOT=/workspace/math-agent/third_party/slime \
  -e MEGATRON_ROOT=/root/Megatron-LM \
  -e MATH_SHARED_ROOT="${MATH_SHARED_ROOT}" \
  -e MODEL_ROOT="${MATH_SHARED_ROOT}/models" \
  "${RUNTIME_IMAGE}" \
  bash
```

这里不使用 `--rm`，并将容器固定命名为 `math-agent-train`，便于退出后重新进入和保留调试现场。第一次创建容器后，不要重复执行上面的 `docker run`；后续根据容器状态使用下面的命令。

如果当前没有训练任务，输入 `exit` 或按 `Ctrl-D` 会退出 Bash 并停止容器。之后可以在宿主机重新启动并进入同一个容器：

```bash
docker start -ai math-agent-train
```

如果训练正在前台运行，不要输入 `exit` 或按 `Ctrl-C`。依次按 `Ctrl-P`、`Ctrl-Q` 可以在不停止容器和训练进程的情况下脱离当前终端。之后可用下面的命令重新连接原终端：

```bash
docker attach math-agent-train
```

也可以从另一个宿主机终端打开新的容器 shell；这种方式不会干扰原终端中正在运行的训练：

```bash
docker exec -it math-agent-train bash
```

从这个额外 shell 执行 `exit` 只会关闭该 shell，不会停止容器的主进程。无论是否保留容器，训练日志和 checkpoint 都必须写入上面已经挂载的宿主机目录。

进入容器后的路径映射如下：

| 内容 | 宿主机路径 | 容器内路径 | 说明 |
|---|---|---|---|
| Git 仓库 | `${MATH_AGENT_ROOT}`，例如 `/home/<user>/math-agent` | `/workspace/math-agent` | 容器工作目录；代码、处理后数据和日志通过 bind mount 回写宿主机仓库 |
| 原始/处理后数据 | `${MATH_AGENT_ROOT}/data` | `/workspace/math-agent/data` | 脚本中的 `data/...` 是相对于容器工作目录的路径 |
| 训练脚本 | `${MATH_AGENT_ROOT}/scripts` | `/workspace/math-agent/scripts` | 第 5 节起的 `scripts/...` 均在容器内执行 |
| 日志 | `${MATH_AGENT_ROOT}/logs` | `/workspace/math-agent/logs` | Git 忽略，但会保留在宿主机仓库目录中 |
| 模型与 checkpoint | `${MATH_SHARED_ROOT}` | `${MATH_SHARED_ROOT}` | 使用相同源/目标绝对路径的 identity mount |
| Megatron | 不要求宿主机存在 | `/root/Megatron-LM` | 来自固定 Slime 镜像 |

`-v "${MATH_SHARED_ROOT}:${MATH_SHARED_ROOT}"` 故意让共享存储在宿主机和容器内保持同一个绝对路径。因此脚本默认出现 `/data/dhsun/math-agent/models` 或 `/data/dhsun/math-agent/checkpoints` 是合理的：它们是容器内可见路径，同时对应宿主机同名目录。在另一台服务器上，如果共享盘位于 `/mnt/training/math-agent`，应在宿主机先设置 `MATH_SHARED_ROOT=/mnt/training/math-agent`；上面的 identity mount 和容器环境变量会一起切换，不要只改挂载的一侧。

容器启动后可以验证：

```bash
# 以下命令在容器内执行。
pwd
test "$(pwd -P)" = /workspace/math-agent
test -f data/raw/retool_sft/train_2000.parquet
test -d "${MODEL_ROOT}/Qwen3-8B-Base_torch_dist"
```

除非代码块前明确写着“宿主机”，本文第 5 节及之后的所有训练、恢复、日志检查命令都在容器内的 `/workspace/math-agent` 执行。

挂载 Docker socket 等价于向训练容器授予较高的宿主机权限，只能用于可信代码。SFT 本身不需要 Docker CLI 和 socket 这两个挂载。

## 5. 完整 SFT

### 5.1 默认配置

- 模型：Qwen3-8B-Base；
- 数据：1,992 条严格转换后的 ReTool 轨迹；
- epoch：3；
- rollout/global batch：8/8；
- optimizer updates：`1992 / 8 * 3 = 747`；
- 学习率：`1e-5`，cosine decay，最低 `1e-6`，warmup fraction `0.1`；
- 单卡动态 token 上限：16,384；
- tensor parallel：2；
- Slime 保存间隔参数：250，约每个 epoch 一次并保存最终 step；
- loss mask：Qwen3 多轮 mask，仅 assistant 推理、工具调用和最终答案参与损失。

完整数据可由脚本首次启动时自动生成，也可以先单独生成：

```bash
python -m math_agent.data retool \
  --input data/raw/retool_sft/train_2000.parquet \
  --output data/processed/retool_sft_train.jsonl
```

先只检查命令，不启动 Ray/GPU 训练：

```bash
PYTHON_BIN=python DRY_RUN=1 bash scripts/run_sft_full.sh
```

普通 dry-run 不会执行 Slime 的深度参数校验，因为 Megatron 的 validator 自身会初始化 CUDA context。只有已经获得独占 GPU allocation 时，才使用 `VALIDATE_ARGS_ON_DRY_RUN=1`。小 batch 的 smoke 配置此前已通过该路径；新设定的 64/128 prompt 正式 RL batch 当前只完成了 CPU-safe command expansion，得到下一次八卡 allocation 后应先补做深度校验和长度 pilot。

正式启动：

```bash
PYTHON_BIN=python bash scripts/run_sft_full.sh
```

默认输出：

```text
${MATH_SHARED_ROOT}/checkpoints/qwen3-8b-base-retool-sft
```

在当前服务器未覆盖 `MATH_SHARED_ROOT` 时，它展开为
`/data/dhsun/math-agent/checkpoints/qwen3-8b-base-retool-sft`。

### 5.2 中断恢复

恢复时必须保持数据、batch、目标更新数、学习率计划和 `SAVE_DIR` 不变：

```bash
PYTHON_BIN=python RESUME=1 bash scripts/run_sft_full.sh
```

若要创建另一组实验，修改 `RUN_NAME`，不要复用旧输出目录：

```bash
PYTHON_BIN=python \
RUN_NAME=qwen3-8b-base-retool-sft-lr5e6 \
LR=5e-6 \
bash scripts/run_sft_full.sh
```

## 6. 完整标准 GRPO baseline

### 6.1 固定的 baseline 边界

默认 RL 配置为：

- actor 初始化：Qwen3-8B；
- reference：训练期间固定的原始 Qwen3-8B；
- 数据：17,243 条去重且移除冲突标签的 DAPO-Math；
- 每轮 64 个 prompt，每个 prompt 8 条采样，共 512 条 trajectory；
- global batch 512、每轮恰好一次 optimizer update；
- 3,000 次 on-policy update；
- GRPO group normalization；
- 对称 PPO clipping：`0.2/0.2`；
- low-variance KL loss：`0.001`；
- entropy coefficient：0；
- 最多 4 次无状态 Python 工具调用；
- 不启用 dynamic sampling、partial rollout、TIS、process reward、length reward 或工具使用奖励。

这组配置是后续“优于 GRPO”方法的基线。更改采样数、更新数、token 预算或 verifier 后，必须同步记录 trajectory 数、生成 token、GPU-hour 和环境调用量，不能只对齐 optimizer step。

默认 3,000 updates 对应 192,000 次 prompt exposure 和 1,536,000 条 trajectory。这个预算远大于早期 `4 prompts/update` 的 smoke 草案；后续方法必须使用相同的 prompt exposure、trajectory 和 token/environment budget。如果要把每轮 prompt 提高到 128，应同时设置 `GLOBAL_BATCH_SIZE=1024`，并根据目标总采样预算重新决定 `TRAIN_UPDATES`：

```bash
ROLLOUT_BATCH_SIZE=128 \
N_SAMPLES_PER_PROMPT=8 \
GLOBAL_BATCH_SIZE=1024 \
ACK_RL_LENGTH_BUDGET=1 \
bash scripts/run_rl_grpo.sh
```

### 6.2 先做长度 pilot

8,192 response token 的 smoke run 实测截断率为 0.625。因此正式脚本把默认 response budget 提高到 12,288，同时保持 context 和单卡训练 token 上限为 16,384；这个新长度尚未完成多步实测。

脚本会拒绝直接执行超过 5 次更新的任务，除非显式设置 `ACK_RL_LENGTH_BUDGET=1`。默认每步已有 512 条 trajectory，因此先运行 2-update pilot 即可收集 1,024 条轨迹：

```bash
PYTHON_BIN=python \
RUN_NAME=qwen3-8b-dapo-grpo-length-pilot \
TRAIN_UPDATES=2 \
SAVE_INTERVAL=2 \
bash scripts/run_rl_grpo.sh
```

检查日志中的关键指标：

```bash
rg "rollout/truncated_ratio|rollout/raw_reward|zero_std|train/grad_norm|train/train_rollout_logprob_abs_diff" \
  logs/train/qwen3-8b-dapo-grpo-length-pilot-*.log
```

建议至少满足：截断率显著低于已知的 0.625，最好不高于 0.2；reward 不是全常数；存在非零梯度；没有持续上升的 train/rollout log-prob mismatch。若不满足，应先调整 response/context budget、最大工具轮数或提示词，再冻结正式 baseline。

### 6.3 启动 3,000-update baseline

确认长度 pilot 后再启动独立的正式实验：

```bash
PYTHON_BIN=python \
ACK_RL_LENGTH_BUDGET=1 \
bash scripts/run_rl_grpo.sh
```

默认输出：

```text
${MATH_SHARED_ROOT}/checkpoints/qwen3-8b-dapo-grpo
```

在当前服务器未覆盖 `MATH_SHARED_ROOT` 时，它展开为
`/data/dhsun/math-agent/checkpoints/qwen3-8b-dapo-grpo`。

恢复命令：

```bash
PYTHON_BIN=python \
ACK_RL_LENGTH_BUDGET=1 \
RESUME=1 \
bash scripts/run_rl_grpo.sh
```

`REF_CHECKPOINT` 在恢复时仍应指向原始 Qwen3-8B，而 `--load` 会自动切换到 `SAVE_DIR` 中的 actor checkpoint。不要把训练后的 actor 同时用作 reference，否则不再是同一个 baseline。

## 7. 监控

控制台日志默认位于：

```text
logs/train/<RUN_NAME>-<UTC timestamp>.log
```

TensorBoard 数据默认位于：

```text
logs/train/tensorboard/<RUN_NAME>/
```

启动查看：

```bash
tensorboard --logdir logs/train/tensorboard --host 127.0.0.1 --port 6006
```

需要 W&B 时显式开启，不要把 API key 写进脚本或 Git：

```bash
export WANDB_API_KEY=...
USE_WANDB=1 \
WANDB_PROJECT=math-agent \
WANDB_GROUP=qwen3-8b-dapo-grpo \
ACK_RL_LENGTH_BUDGET=1 \
bash scripts/run_rl_grpo.sh
```

RL 至少监控：

- `rollout/raw_reward` 和每个 group 的 zero-std 数量；
- `rollout/truncated_ratio`、response length、工具调用错误；
- `train/grad_norm`、`train/pg_clipfrac`、`train/ppo_kl`；
- `train/train_rollout_logprob_abs_diff`；
- rollout、训练、checkpoint 保存耗时；
- Python 沙箱创建失败、超时和残留容器。

SFT 至少监控 loss、grad norm、学习率、有效 token 数和 checkpoint 保存结果。

## 8. 常用覆盖参数

| 变量 | SFT 默认 | RL 默认 | 含义 |
|---|---:|---:|---|
| `RUN_NAME` | `qwen3-8b-base-retool-sft` | `qwen3-8b-dapo-grpo` | 实验名和默认 checkpoint 目录名 |
| `TRAIN_UPDATES` | 根据数据和 epoch 计算 | 3000 | 目标 optimizer update 数 |
| `SAVE_INTERVAL` | 250（自动计算） | 500 | 传给 Slime 的 checkpoint 间隔参数 |
| `RESUME` | 0 | 0 | 设为 1 后从 `SAVE_DIR` 恢复 |
| `ROLLOUT_BATCH_SIZE` | 8 | 64 | 每轮 prompt 数；RL 可显式提高到 128 |
| `GLOBAL_BATCH_SIZE` | 8 | 512 | 每个 optimizer step 的样本数；128 prompt 时应为 1024 |
| `MAX_TOKENS_PER_GPU` | 16384 | 16384 | 动态训练 token 上限 |
| `ROLLOUT_MAX_RESPONSE_LEN` | 不适用 | 12288 | RL 总响应预算，包含多轮工具轨迹 |
| `ROLLOUT_MAX_CONTEXT_LEN` | 不适用 | 16384 | RL prompt + response 上限 |
| `FORCE_PREPROCESS` | 0 | 0 | 设为 1 后重新生成处理数据 |
| `USE_TENSORBOARD` | 1 | 1 | 本地 TensorBoard |
| `USE_WANDB` | 0 | 0 | W&B 记录 |
| `DRY_RUN` | 0 | 0 | 只预处理、校验并打印命令 |
| `VALIDATE_ARGS_ON_DRY_RUN` | 0 | 0 | 设为 1 后在自有 GPU allocation 中执行 Slime 深度参数校验 |

修改 `ROLLOUT_BATCH_SIZE` 或 `N_SAMPLES_PER_PROMPT` 时，RL 脚本要求：

```text
GLOBAL_BATCH_SIZE = ROLLOUT_BATCH_SIZE × N_SAMPLES_PER_PROMPT
```

这样每轮 rollout 仍只训练一次，保持 on-policy baseline 的数据消费关系。

## 9. 当前不包含的内容

- SFT checkpoint 初始化 RL 的串联实验；
- AIME、AMC 等冻结评测集的训练期调参；
- 优于 GRPO 的算法改动；
- 多机训练；
- 自动删除旧 checkpoint；
- 完整训练后的效果结论。

这些内容应在标准 GRPO baseline 的预算、checkpoint 和评测协议冻结后单独增加，不能混入当前基线脚本。
