# Math Agent 环境与数据迁移指南

[返回中文 README](../README_zh.md) | [English README](../README.md)

本文说明如何把 Math Agent 的 Docker 运行环境和原始数据制作成一个可复制的离线包，并在该目录已经传到正式训练服务器后完成恢复。代码默认不放入离线包：源机器只在 manifest 中记录已经测试并 push 的 Git commit，目标服务器从 GitHub 拉取该版本。模型权重也不放入离线包；环境和数据恢复完成后，统一按照[Math Agent 完整训练指南](math_agent_training_zh.md#3-在目标服务器下载并转换模型)下载并转换模型。

## 迁移边界

离线包包含：

- `math-agent/slime-runtime:20260903-cu129`：Slime、PyTorch、SGLang、Ray 和 Megatron 运行环境；
- `python:3.11-slim`：RL 在线 Python 工具沙箱；
- 经过测试的 Math Agent commit 和 Docker 镜像 ID 记录；
- ReTool 和 DAPO-Math 原始数据。

默认流程中，离线包**不包含**代码、Hugging Face 模型或 Megatron `torch_dist` checkpoint。正式训练服务器必须能够访问 GitHub 和 Hugging Face（或配置等价的镜像站），当前账号还需拥有 Math Agent 仓库的 SSH 权限；完成模型下载后还需要一张独占 GPU 做转换。若目标机可能无法访问 GitHub，可额外生成 Git bundle 作为可选备份。

离线包不能替代正式机器上的：

- NVIDIA 驱动；
- Docker Engine；
- NVIDIA Container Toolkit；
- 足够的本地 Docker 数据空间；
- 下载模型所需的网络、代理或 Hugging Face 凭据；
- 可供训练独占使用的 GPU。

不要直接复制 `/var/lib/docker`，也不要使用 `docker export` 导出镜像。跨机器迁移镜像应使用 `docker save` 和 `docker load`。

## 当前固定版本

| 组件 | 固定标识 |
|---|---|
| Math Agent 训练镜像 | `math-agent/slime-runtime:20260903-cu129` |
| 训练镜像 ID | `sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c` |
| Python 沙箱镜像 | `python:3.11-slim` |
| Python 沙箱镜像 ID | `sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76` |
| Slime commit | `4c193f1f37509cca70f0e88807a9305b70f63f4e` |

镜像在 Docker 中展开后约占 67 GB，原始数据约占 291 MB；离线包所在存储主要按 Docker 归档和原始数据的实际大小预留即可。可选 Git bundle 体积较小。正式服务器的共享存储还需容纳两套 Hugging Face 权重和两套转换后的 `torch_dist` checkpoint，当前合计约 64 GB；训练 checkpoint 的空间需求另见[完整训练指南](math_agent_training_zh.md#2-目标服务器基础预检)。

如果使用移动硬盘或其他可移动介质中转，推荐使用 ext4。FAT32 存在单文件 4 GB 限制，不能用于镜像归档；exFAT 可以存放大文件，但不会保留完整的 Unix 权限语义。

## 一、在源机器制作离线包

### 1. 设置路径并检查状态

从 Math Agent 仓库内执行：

```bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
TRANSFER_ROOT=/path/to/output/math-agent-transfer

mkdir -p \
  "${TRANSFER_ROOT}/images" \
  "${TRANSFER_ROOT}/data/raw" \
  "${TRANSFER_ROOT}/manifest"

git -C "${REPO_ROOT}" status --short
git -C "${REPO_ROOT}" submodule status
df -h "${TRANSFER_ROOT}"
```

打包前，主仓库和 Slime submodule 的工作树都应为空。完整训练脚本和文档必须先提交并 push；manifest 只能记录目标服务器可以从远端取得的 commit。

### 2. 导出两套 Docker 镜像

推荐使用 Zstandard 压缩。源机器和目标机器都需要 `zstd`：

```bash
set -euo pipefail

docker image inspect math-agent/slime-runtime:20260903-cu129 >/dev/null
docker image inspect python:3.11-slim >/dev/null

docker save \
  math-agent/slime-runtime:20260903-cu129 \
  python:3.11-slim \
  | zstd -T0 -3 -o "${TRANSFER_ROOT}/images/docker-images.tar.zst"
```

如果目标机器没有 `zstd`，可以改为不压缩的 tar，但离线包需要更多空间：

```bash
docker save \
  math-agent/slime-runtime:20260903-cu129 \
  python:3.11-slim \
  -o "${TRANSFER_ROOT}/images/docker-images.tar"
```

只需保留 `docker-images.tar.zst` 或 `docker-images.tar` 中的一种。

### 3. 记录已经提交并 push 的代码版本

```bash
set -euo pipefail

test -z "$(git -C "${REPO_ROOT}" status --porcelain)"
test -z "$(git -C "${REPO_ROOT}/third_party/slime" status --porcelain)"

# 确认当前主仓库 commit 已经存在于团队远端 main。
git -C "${REPO_ROOT}" fetch origin main
test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = \
  "$(git -C "${REPO_ROOT}" rev-parse origin/main)"

git -C "${REPO_ROOT}" rev-parse HEAD \
  > "${TRANSFER_ROOT}/manifest/math-agent.commit"

{
  printf '%s ' 'math-agent/slime-runtime:20260903-cu129'
  docker image inspect math-agent/slime-runtime:20260903-cu129 \
    --format '{{.Id}}'
  printf '%s ' 'python:3.11-slim'
  docker image inspect python:3.11-slim --format '{{.Id}}'
} > "${TRANSFER_ROOT}/manifest/image-ids.txt"
```

如果目标服务器可能无法访问 GitHub，再额外生成 Git bundle；正常联网迁移不需要这一步：

```bash
mkdir -p "${TRANSFER_ROOT}/git"

git -C "${REPO_ROOT}" bundle create \
  "${TRANSFER_ROOT}/git/math-agent.bundle" --all
git -C "${REPO_ROOT}/third_party/slime" bundle create \
  "${TRANSFER_ROOT}/git/slime.bundle" --all

git bundle verify "${TRANSFER_ROOT}/git/math-agent.bundle"
git bundle verify "${TRANSFER_ROOT}/git/slime.bundle"
```

### 4. 复制原始数据

```bash
set -euo pipefail

rsync -aH --info=progress2 \
  "${REPO_ROOT}/data/raw/" \
  "${TRANSFER_ROOT}/data/raw/"
```

模型目录不要复制到离线包；恢复完成后转入[完整训练指南的模型准备步骤](math_agent_training_zh.md#3-在目标服务器下载并转换模型)。

### 5. 生成完整性校验文件

该步骤会顺序读取全部镜像、数据文件和可选 Git bundle，耗时取决于离线包所在存储的读取速度：

```bash
set -euo pipefail

(
  cd "${TRANSFER_ROOT}"
  find . -type f \
    ! -path './manifest/SHA256SUMS' \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "${TRANSFER_ROOT}/manifest/SHA256SUMS"

sync
```

写入结束后先等待 `sync` 完成，再复制整个 `math-agent-transfer/` 目录到目标服务器。

离线包应具有以下结构：

```text
math-agent-transfer/
├── images/
│   └── docker-images.tar.zst
├── data/
│   └── raw/
│       ├── retool_sft/
│       └── dapo_math_17k/
└── manifest/
    ├── SHA256SUMS
    ├── math-agent.commit
    └── image-ids.txt
```

若生成了断网备用包，顶层还会有可选的 `git/math-agent.bundle` 和 `git/slime.bundle`。

## 二、在正式机器恢复

### 1. 检查宿主机基础能力

```bash
set -euo pipefail

nvidia-smi
docker version
docker info

DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
printf 'Docker data-root: %s\n' "${DOCKER_ROOT}"
df -h "${DOCKER_ROOT}"
```

正式机器的 NVIDIA 驱动必须支持镜像所用的 CUDA 12.9。Docker 必须已经配置 NVIDIA Container Toolkit。`DOCKER_ROOT` 通常是 `/var/lib/docker`，但也可能由管理员改到其他数据盘；必须以 `docker info` 的输出为准。镜像导入后由 Docker daemon 自动写入这个目录，不要手工创建、复制或修改其中的文件。

如果当前账号执行 `docker info` 报权限错误，需要由管理员把账号加入允许访问 Docker daemon 的用户组，或在本节所有 `docker` 命令前统一加 `sudo`。

### 2. 校验已经传到服务器的离线包

先通过 `rsync`、`scp` 或其他文件传输方式，把完整的 `math-agent-transfer/` 目录复制到目标服务器。`TRANSFER_ROOT` 应指向目标服务器上这个目录的实际位置，而不是 Docker 数据目录或它的上级目录：

```bash
set -euo pipefail

TRANSFER_ROOT=/path/on/target-server/math-agent-transfer

test -d "${TRANSFER_ROOT}"
test -f "${TRANSFER_ROOT}/manifest/SHA256SUMS"
test -f "${TRANSFER_ROOT}/manifest/math-agent.commit"
test -f "${TRANSFER_ROOT}/manifest/image-ids.txt"
test -d "${TRANSFER_ROOT}/data/raw/retool_sft"
test -d "${TRANSFER_ROOT}/data/raw/dapo_math_17k"
test -f "${TRANSFER_ROOT}/images/docker-images.tar.zst" || \
  test -f "${TRANSFER_ROOT}/images/docker-images.tar"

(
  cd "${TRANSFER_ROOT}"
  sha256sum -c manifest/SHA256SUMS
)
```

任何一项校验失败都应停止恢复，重新复制对应文件。

### 3. 加载 Docker 镜像

不需要把镜像归档复制到 `/var/lib/docker` 或代码仓库。直接从目标服务器上的 `TRANSFER_ROOT` 加载即可；`docker load` 会解包并把镜像层写入上一节查到的 `DOCKER_ROOT`。

Zstandard 归档：

```bash
set -euo pipefail

zstd -dc "${TRANSFER_ROOT}/images/docker-images.tar.zst" \
  | docker load
```

未压缩归档：

```bash
docker load -i "${TRANSFER_ROOT}/images/docker-images.tar"
```

加载可能持续较长时间。命令成功返回后，训练不再从归档文件读取镜像；目标服务器上的归档只用于迁移和备份。

确认镜像 ID：

```bash
test "$(docker image inspect \
  math-agent/slime-runtime:20260903-cu129 \
  --format '{{.Id}}')" = \
  'sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c'

test "$(docker image inspect \
  python:3.11-slim \
  --format '{{.Id}}')" = \
  'sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76'

docker image ls math-agent/slime-runtime:20260903-cu129
docker image ls python:3.11-slim
```

### 4. 恢复 Git 仓库和 submodule

默认从 GitHub 拉取 manifest 固定的版本。这里要求目标服务器能够访问 GitHub，且当前账号已配置 Math Agent 仓库的 SSH 权限：

```bash
set -euo pipefail

WORK_ROOT=/path/to/workspace
REPO_ROOT="${WORK_ROOT}/math-agent"
EXPECTED_COMMIT="$(cat "${TRANSFER_ROOT}/manifest/math-agent.commit")"

mkdir -p "${WORK_ROOT}"
git clone --recurse-submodules \
  git@github.com:YoungAstronaut/math-agent.git \
  "${REPO_ROOT}"

git -C "${REPO_ROOT}" checkout "${EXPECTED_COMMIT}"
git -C "${REPO_ROOT}" submodule update --init --recursive

test "$(git -C "${REPO_ROOT}" rev-parse HEAD)" = \
  "${EXPECTED_COMMIT}"
test "$(git -C "${REPO_ROOT}/third_party/slime" rev-parse HEAD)" = \
  "$(git -C "${REPO_ROOT}" rev-parse HEAD:third_party/slime)"

git -C "${REPO_ROOT}" submodule status
```

如果 GitHub 临时不可用且离线包中包含可选 bundle，才改用以下离线恢复方式：

```bash
set -euo pipefail

WORK_ROOT=/path/to/workspace
REPO_ROOT="${WORK_ROOT}/math-agent"
EXPECTED_COMMIT="$(cat "${TRANSFER_ROOT}/manifest/math-agent.commit")"

git clone "${TRANSFER_ROOT}/git/math-agent.bundle" "${REPO_ROOT}"
git -C "${REPO_ROOT}" checkout "${EXPECTED_COMMIT}"

git -C "${REPO_ROOT}" submodule init
git -C "${REPO_ROOT}" config submodule.third_party/slime.url \
  "${TRANSFER_ROOT}/git/slime.bundle"
git -C "${REPO_ROOT}" -c protocol.file.allow=always \
  submodule update --init --checkout
git -C "${REPO_ROOT}" submodule sync
```

### 5. 恢复原始数据并设置共享存储

选择正式机器上的共享存储目录：

```bash
set -euo pipefail

TARGET_SHARED_ROOT=/path/to/shared-storage/math-agent
MODEL_ROOT="${TARGET_SHARED_ROOT}/models"

mkdir -p "${MODEL_ROOT}" "${REPO_ROOT}/data/raw"

rsync -aH --info=progress2 \
  "${TRANSFER_ROOT}/data/raw/" \
  "${REPO_ROOT}/data/raw/"

export MATH_SHARED_ROOT="${TARGET_SHARED_ROOT}"
export MODEL_ROOT
```

如果目标存储路径与源机器不同，只需在启动容器和训练脚本时继续传入 `MATH_SHARED_ROOT` 与 `MODEL_ROOT`。

### 6. 验证 GPU 容器环境

```bash
docker run --rm \
  --gpus all \
  --entrypoint bash \
  math-agent/slime-runtime:20260903-cu129 \
  -lc 'nvidia-smi && python - <<\"PY\"
import torch
import ray
import sglang
import transformers
import megatron.core

print(\"torch\", torch.__version__)
print(\"cuda\", torch.cuda.is_available(), torch.cuda.device_count())
print(\"ray\", ray.__version__)
print(\"sglang\", sglang.__version__)
print(\"transformers\", transformers.__version__)
print(\"megatron\", megatron.core.__version__)
PY'
```

如果 `torch.cuda.is_available()` 不是 `True`，不要继续转换或训练，应先修复 NVIDIA Driver、Docker 或 NVIDIA Container Toolkit。

### 7. 验证代码和 Python 工具沙箱

```bash
docker run --rm \
  -v "${REPO_ROOT}:/workspace/math-agent" \
  -v /usr/bin/docker:/usr/local/bin/docker:ro \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w /workspace/math-agent \
  -e MATH_AGENT_RUN_DOCKER_TESTS=1 \
  --entrypoint bash \
  math-agent/slime-runtime:20260903-cu129 \
  -lc 'PYTHONPATH=/workspace/math-agent:/workspace/math-agent/third_party/slime:/root/Megatron-LM \
       pytest -q'
```

该检查应通过普通单元测试和真实的无网络 Python 沙箱测试。挂载 Docker socket 等价于向容器授予较高的宿主机权限，只应在可信的训练容器中使用。

### 8. 转入模型准备和训练流程

至此，Docker 镜像、Git 代码和原始数据已经在正式机器恢复完成。下一步统一执行[完整训练指南第 3 节](math_agent_training_zh.md#3-在目标服务器下载并转换模型)，在正式机器下载固定 revision 的 Hugging Face 模型，并用仓库脚本生成 Megatron `torch_dist` checkpoint。

模型下载和转换命令只在完整训练指南中维护，本文不再复制，避免两份文档的 revision、镜像名或路径约定发生漂移。模型转换需要一张独占 GPU；不要在其他训练占用 GPU 时强制执行。

## 三、训练前最终验收

运行训练前应满足：

- `git status --short` 没有非预期修改；
- `git submodule status` 指向固定的 Slime commit；
- 两套 Docker 镜像 ID 与本文一致；
- 两套 HF 模型已在正式机器下载，两套 `torch_dist` checkpoint 已在正式机器转换完成；
- ReTool 与 DAPO-Math 原始数据存在；
- 8 张 GPU 可由本次任务独占；
- 当前容器内没有需要复用或停止的 Ray 集群；
- CPU/容器测试与 Python 沙箱测试通过。

先运行一次 SFT smoke update，确认 loss、loss mask 和 checkpoint 保存正常；再运行一次 RL smoke update，确认 rollout、Python 工具调用、reward 和 learner update 正常。具体命令见[中文 README 的冒烟测试章节](../README_zh.md#冒烟测试)。

smoke 验收通过后，按照[Math Agent 完整训练指南](math_agent_training_zh.md)先执行完整 SFT，再完成 RL response-length pilot。只有长度、reward 方差、梯度和 train/rollout log-prob mismatch 达到文档中的 gate 后，才启动 3,000-update GRPO baseline。

当前脚本会在 GPU 已被占用或 checkpoint 缺失时拒绝启动。不要通过 `ALLOW_BUSY_GPUS=1` 绕过共享服务器上的保护检查。

## 常见故障

| 现象 | 优先检查 |
|---|---|
| `could not select device driver` 或 `--gpus` 不可用 | NVIDIA Container Toolkit 是否安装并重启 Docker |
| 转换报 `No usable accelerator was detected` | `docker run` 是否包含 `--gpus device=0` |
| 容器内找不到模型 | `MATH_SHARED_ROOT` 是否挂载，`MODEL_ROOT` 是否指向容器内可见路径 |
| RL 找不到 `docker` 或无法连接 daemon | 是否挂载 Docker CLI 和 `/var/run/docker.sock` |
| GitHub clone 报权限错误 | 当前账号的 SSH key 是否已加入 GitHub，并拥有 Math Agent 仓库权限 |
| `git submodule status` 前出现 `-` | 是否执行了 `git submodule update --init --recursive`，以及目标机能否访问公开的 THUDM/slime |
| `docker load` 空间不足 | 检查 Docker data-root 所在文件系统，而不只是离线包所在目录的空间 |
| GPU 可见但 CUDA 初始化 OOM | GPU 是否已有计算进程；应申请独占资源后重试 |

完成上述验收后，正式训练不再依赖 `math-agent-transfer/`。镜像已经位于 Docker daemon 的 `DOCKER_ROOT` 中，原始数据也已经复制到仓库；服务器上的离线包目录可以继续保留作灾备，也可以在确认备份完整后另行清理。
