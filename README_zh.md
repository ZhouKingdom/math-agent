# Math Agent

[English](README.md) | 简体中文

上传和同步本项目请参阅[GitHub 上传指南](docs/github_upload_zh.md)。

本仓库是一个基于 slime、面向数学、搜索和代码工具智能体可复现训练的 monorepo。当前首先实现的是 Math Agent，其初始验证阶段有意相互独立：

1. SFT 冒烟测试：使用 `Qwen/Qwen3-8B-Base` 在转换后的 ReTool 轨迹上训练。
2. RL 冒烟测试：使用 `Qwen/Qwen3-8B` 在去重后的 DAPO-Math-17k 上运行标准 GRPO。
3. 只有在上述两条链路都通过后，才评估 SFT-to-RL 初始化以及预期优于 GRPO 的方法。

目前这两个冒烟测试**还不是**一条连续的训练链。这样可以区分 SFT 格式、多轮在线工具交互和 GRPO 优化各自的问题。

## 固定版本输入

- slime：`THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e`
- ReTool SFT：Hugging Face revision `74943ce52f389f16926702302fcff6255875cbb2`，文件 SHA-256 `152dd6fa574d1a095064e89230b550521c9fa6b00b22e816f0f6490b0c5ab72a`
- DAPO-Math-17k：Hugging Face revision `65877096c24ffa7abc4e4fa5edb95cf3413a5674`，文件 SHA-256 `534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941`
- Qwen3-8B-Base：revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`
- Qwen3-8B：revision `b968826d9c46dd6066d109eabc6255188de91218`
- slime CUDA 12 运行环境：`slimerl/slime@sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c`

原始数据集位于 `data/raw/`。模型权重存放在 `${MODEL_ROOT}` 指向的共享存储中（默认：`/data/dhsun/math-agent/models`）；生成的 JSONL、本地模型链接、checkpoint 和日志均由 Git 忽略。

## 数据契约

ReTool 转换会移除数据集自带的指令包装，并将每一组严格匹配的 `<code>...</code>` 与 `<interpreter>...</interpreter>` 转换为 Qwen/Hermes 消息，工具函数统一命名为 `code_interpreter`。Assistant 的推理和工具调用参与训练，工具返回的观察结果不参与训练。标签格式错误或顺序异常的样本会被拒绝，而不是通过启发式规则修复。审计结果为 1,992 条可用样本和 8 条拒绝样本。原有 `<answer>` 包装会被规范化为严格的最终行：

```text
Answer: \boxed{answer}
```

DAPO 转换会扫描全部 1,791,700 条物理记录，仅在去除题目文本首尾空白后进行去重；如果同一道题对应多个不同的真实答案，则整道题都会被丢弃。内部 TeX 和空白不会被规范化，因为它们可能影响数学含义。审计结果为 17,255 道唯一题目、12 道冲突题目被丢弃，最终得到 17,243 条可用样本。因此，原始数据中 100 倍的行级复制不会改变采样概率。

需要时可直接生成训练数据：

```bash
cd /path/to/math-agent
python -m math_agent.data retool \
  --input data/raw/retool_sft/train_2000.parquet \
  --output data/processed/retool_sft.jsonl

python -m math_agent.data dapo \
  --input data/raw/dapo_math_17k/data/dapo-math-17k.parquet \
  --output data/processed/dapo_math_rl.jsonl
```

两个命令都会输出机器可读的审计报告，其中包括被拒绝的 ReTool 样本或标签冲突的 DAPO 题目。

## Python 工具边界

在线 Python 工具是无状态的。每次调用都会启动一个新容器，并满足以下约束：

- 禁用网络；
- 根文件系统只读，临时文件系统容量受限；
- 使用非 root 用户，移除全部 Linux capabilities，并启用 `no-new-privileges`；
- 限制 CPU、内存、PID、执行时间、输入大小和合并输出大小；
- 在 stdout 返回模型前转义聊天和工具协议分隔符。

默认镜像固定为 `python@sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76`。RL 进程需要访问 Docker CLI，并且只能创建这些临时沙箱容器；训练任务本身应运行在独占的资源分配或容器中。

## 运行环境

当前运行环境使用上面固定的 slime CUDA 12 镜像，其中包含 Python 3.12、PyTorch 2.11.0+cu129、SGLang 0.5.15.post1、Ray 2.58.0，以及 Megatron commit `1dcf0dafa884ad52ffb243625717a3471643e087`。镜像内置的 slime checkout 早于本仓库固定的版本，因此以下命令会挂载并执行本仓库的 `third_party/slime`，同时复用镜像中的原生依赖。

启动可见全部 GPU 的交互式环境：

```bash
cd /path/to/math-agent
MATH_AGENT_ROOT="$(pwd -P)"
MATH_SHARED_ROOT="${MATH_SHARED_ROOT:-/data/dhsun/math-agent}"

docker run --rm -it \
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
  slimerl/slime@sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c \
  bash
```

共享存储挂载会让模型权重在宿主机与容器内保持相同的绝对路径。只有 RL 的同级 Python 沙箱需要挂载 Docker socket 和 CLI；SFT 与 checkpoint 转换可以省略这两个挂载。冒烟测试脚本在可见 GPU 已存在计算进程时会拒绝运行，因此不会终止或复用无关的训练任务。

## 离线迁移

如需通过移动硬盘将固定 Docker 镜像和数据迁移到另一台服务器，并从 Git 拉取代码，请参阅[环境与数据迁移指南](docs/offline_migration_zh.md)。

## 模型准备

模型不包含在迁移包中，应在目标服务器下载并转换。固定 revision、共享存储路径、容器化下载命令、Megatron 转换脚本及结果检查统一见[Math Agent 完整训练指南的模型准备章节](docs/math_agent_training_zh.md#3-在目标服务器下载并转换模型)。

## 冒烟测试

请在具备 8 张独占 GPU 的 slime 兼容环境中运行以下命令。如果脚本检测到已有 Ray 集群，或可见 GPU 上已有计算进程，它会拒绝复用或停止这些资源。

```bash
cd /path/to/math-agent
PYTHON_BIN=python bash scripts/run_sft_smoke.sh
PYTHON_BIN=python bash scripts/run_rl_smoke.sh
```

SFT 冒烟测试默认在 64 条 ReTool 样本上执行一次更新，使用 Qwen3 多轮 loss mask，并将单卡 token 上限设为 16,384。完整转换数据的最大渲染长度为 15,789 tokens，其中 21 条样本超过 8,192 tokens。RL 冒烟测试默认执行一次 on-policy 更新：4 个 prompt × 每个 prompt 8 个样本，响应上限为 8,192 tokens、总上下文上限为 16,384 tokens，使用对称 PPO clipping（`0.2/0.2`）、显式 low-variance KL loss（`0.001`），每轮 rollout 执行一个 learner step。冒烟测试默认有意不启用 dynamic sampling、partial rollout、TIS、speculative decoding、process reward、length reward 或工具使用奖励。

RL 数据与 rollout 共用同一协议：Qwen/Hermes `<tool_call>` JSON、唯一的 `code_interpreter` 函数，以及作为终止动作的 `Answer: \boxed{...}`。模型生成 token 的 loss mask 为 1，环境 observation 的 loss mask 为 0。

## 完整训练

完整 SFT 和标准 GRPO baseline 分别使用以下脚本：

以下命令在上面的固定 Slime 容器内执行；此时工作目录是 `/workspace/math-agent`，而模型与 checkpoint 共享目录通过 identity mount 保持宿主机和容器内绝对路径一致。

```bash
PYTHON_BIN=python bash scripts/run_sft_full.sh

# 先完成文档规定的长度 pilot，再启动长程 RL。
PYTHON_BIN=python ACK_RL_LENGTH_BUDGET=1 bash scripts/run_rl_grpo.sh
```

默认 SFT 使用全部 1,992 条有效 ReTool 轨迹训练 3 epoch，共 747 次更新。默认 RL 使用全部 17,243 条去重 DAPO-Math prompt，每次更新采样 64 个 prompt × 8 条 response，并执行 3,000 次标准 GRPO 更新。运行方式、恢复语义、总采样预算、长度 pilot、监控指标和参数覆盖见[Math Agent 完整训练指南](docs/math_agent_training_zh.md)。目前完整脚本已通过数据预检和命令展开检查，但完整时长训练尚未执行。

## CPU 验证

```bash
cd /path/to/math-agent
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
python -X pycache_prefix=/tmp/math-agent-pycache -m compileall -q math_agent tests
bash -n scripts/run_sft_smoke.sh scripts/run_rl_smoke.sh \
  scripts/run_sft_full.sh scripts/run_rl_grpo.sh

MATH_AGENT_RUN_DOCKER_TESTS=1 \
  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_python_tool.py
```

单元测试不会在宿主机上执行不受信任的代码，也不需要 GPU。实时沙箱集成测试还要求本地存在已经配置好的 Docker 镜像。
