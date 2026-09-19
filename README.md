# Math Agent

[English](README.md) | [简体中文](README_zh.md)

上传和同步本项目请参阅[GitHub 上传指南](docs/github_upload_zh.md)。

Reproducible training recipes for a tool-using math agent built on [slime](https://github.com/THUDM/slime). The repository currently focuses on a strict, inspectable Math Agent baseline: the model can call a sandboxed Python interpreter, receive its observation, and finish with a canonical boxed answer.

## What is included

- **Data preparation**: fail-closed conversion and audit reports for ReTool SFT and DAPO-Math RL data.
- **Tool boundary**: stateless, networkless Python execution in a resource-limited Docker container.
- **Training entry points**: CPU-safe preflight plus SFT and plain-GRPO smoke/full scripts for Qwen3-8B.
- **Reproducibility docs**: pinned revisions, container digests, model conversion, offline migration, and recovery steps.

The initial validation gates are deliberately separated:

1. SFT smoke test: `Qwen/Qwen3-8B-Base` on converted ReTool trajectories.
2. RL smoke test: `Qwen/Qwen3-8B` on deduplicated DAPO-Math-17k with plain GRPO.
3. Only after both contracts pass: evaluate SFT-to-RL initialization and methods intended to improve over GRPO.

The two smoke tests do **not** yet form one training chain. This keeps failures in SFT formatting, online tool interaction, and GRPO optimization distinguishable.

## Quick start

The CPU checks require Python 3.11+ and the development extra. GPU training additionally requires the pinned Slime runtime, eight exclusively allocated GPUs, Docker access for the Python sandbox, and the input datasets/models described below.

```bash
git clone --recurse-submodules https://github.com/YoungAstronaut/math-agent.git
cd math-agent
python -m pip install -e '.[dev]'

PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
bash -n scripts/*.sh
```

For a smoke run, first enter the documented container environment and make sure the Qwen3 checkpoints and raw datasets are available:

```bash
PYTHON_BIN=python bash scripts/run_sft_smoke.sh
PYTHON_BIN=python bash scripts/run_rl_smoke.sh
```

The smoke scripts intentionally refuse to reuse busy GPUs, active Ray clusters, incomplete checkpoints, or an occupied output directory.

## Project layout

```text
math-agent/
├── math_agent/       # Data contracts, rollout protocol, reward, and Python tool
├── scripts/          # Model conversion plus SFT/RL smoke and full runs
├── tests/            # CPU contracts and opt-in Docker sandbox integration test
├── docs/             # Chinese training and offline migration guides
└── third_party/slime # Pinned training framework submodule
```

## Pinned inputs

- slime: `THUDM/slime@4c193f1f37509cca70f0e88807a9305b70f63f4e`
- ReTool SFT: Hugging Face revision `74943ce52f389f16926702302fcff6255875cbb2`, file SHA-256 `152dd6fa574d1a095064e89230b550521c9fa6b00b22e816f0f6490b0c5ab72a`
- DAPO-Math-17k: Hugging Face revision `65877096c24ffa7abc4e4fa5edb95cf3413a5674`, file SHA-256 `534375d6bb8630d22ab46a56e11f2ffec1d288d8f7d04099bc82d68948705941`
- Qwen3-8B-Base: revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`
- Qwen3-8B: revision `b968826d9c46dd6066d109eabc6255188de91218`
- slime CUDA 12 runtime: `slimerl/slime@sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c`

The raw datasets are already present under `data/raw/`. Model weights live in shared storage under `${MODEL_ROOT}` (default: `/data/dhsun/math-agent/models`); generated JSONL files, local model links, checkpoints, and logs are ignored by Git.

## Data contracts

ReTool conversion removes the dataset-specific instruction wrapper and turns every strict `<code>...</code>` plus `<interpreter>...</interpreter>` pair into Qwen/Hermes messages for one function named `code_interpreter`. Assistant reasoning and calls are trainable; tool observations are not. Rows with malformed or out-of-order tags are rejected rather than repaired heuristically. The audited result is 1,992 usable rows and 8 rejected rows. The old `<answer>` wrapper is normalized to an exact final line:

```text
Answer: \boxed{answer}
```

DAPO conversion scans all 1,791,700 physical rows, deduplicates by extracted question text after trimming only its outer whitespace, and drops every question for which more than one ground-truth answer exists. Internal TeX/whitespace is intentionally not normalized because it can be mathematically significant. The audited result is 17,255 unique questions, 12 conflicting questions dropped, and 17,243 usable rows. The original 100-fold row replication therefore does not alter sampling probability.

Generate data directly when needed:

```bash
cd /path/to/math-agent
python -m math_agent.data retool \
  --input data/raw/retool_sft/train_2000.parquet \
  --output data/processed/retool_sft.jsonl

python -m math_agent.data dapo \
  --input data/raw/dapo_math_17k/data/dapo-math-17k.parquet \
  --output data/processed/dapo_math_rl.jsonl
```

Both commands print a machine-readable audit report, including rejected ReTool rows or conflicting DAPO labels.

## Python tool boundary

Online Python is stateless. Each call starts a fresh container with:

- no network;
- a read-only root filesystem and bounded temporary filesystem;
- a non-root user, all Linux capabilities dropped, and `no-new-privileges`;
- CPU, memory, PID, wall-time, input-size, and combined-output limits;
- escaped chat/tool delimiters in stdout before it returns to the model.

The default image is pinned as `python@sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76`. The RL process needs Docker CLI access and permission to create only these ephemeral sandbox containers; run the training job itself in a dedicated allocation/container.

## Runtime environment

The configured runtime is the pinned slime CUDA 12 image above. It contains Python 3.12, PyTorch 2.11.0+cu129, SGLang 0.5.15.post1, Ray 2.58.0, and Megatron commit `1dcf0dafa884ad52ffb243625717a3471643e087`. The image's bundled slime checkout is older than this repository's pinned source, so the commands mount and execute `third_party/slime` from this repository while reusing the image's native dependencies.

Start an interactive environment with all GPUs visible:

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

The shared-storage mount exposes model weights at the same absolute path on the host and in the container. The Docker socket and CLI mounts are needed only for RL's sibling Python sandboxes. They can be omitted for SFT and checkpoint conversion. The smoke scripts refuse to run when visible GPUs are occupied, so this container does not terminate or reuse an unrelated training job.

## Offline migration

To move the pinned Docker images and data with a portable drive, then fetch the code from Git, follow the [Chinese environment and data migration guide](docs/offline_migration_zh.md).

## Model preparation

Models are not included in the transfer bundle; download and convert them on the target server. The pinned revisions, shared-storage paths, containerized download commands, Megatron conversion script, and result checks are maintained in the [Math Agent model-preparation section](docs/math_agent_training_zh.md#3-在目标服务器下载并转换模型).

## Smoke runs

Run these inside a slime-compatible environment with eight exclusively allocated GPUs. Each script refuses to stop/reuse an existing Ray cluster or run on GPUs that already expose compute processes.

```bash
cd /path/to/math-agent
PYTHON_BIN=python bash scripts/run_sft_smoke.sh
PYTHON_BIN=python bash scripts/run_rl_smoke.sh
```

The SFT smoke default is one update over 64 ReTool examples with Qwen3 multi-turn loss masking and a 16,384-token per-GPU ceiling. The full converted set has a maximum rendered length of 15,789 tokens; 21 examples exceed 8,192 tokens. The RL smoke default is one on-policy update: 4 prompts × 8 samples, an 8,192-token response budget within a 16,384-token context, symmetric PPO clipping (`0.2/0.2`), an explicit low-variance KL loss (`0.001`), and one learner step per rollout. The smoke run intentionally enables none of dynamic sampling, partial rollout, TIS, speculative decoding, process reward, length reward, or tool-use bonus.

The RL data and rollout share one protocol: Qwen/Hermes `<tool_call>` JSON, exactly one `code_interpreter` function, and `Answer: \boxed{...}` as the terminal action. Model-generated tokens have loss mask 1; environment observations have loss mask 0.

## Full training

The full SFT and plain-GRPO baseline entry points are:

Run the following commands inside the pinned Slime container described above. Its working directory is `/workspace/math-agent`; the model/checkpoint shared root uses an identity bind mount and therefore has the same absolute path on the host and in the container.

```bash
PYTHON_BIN=python bash scripts/run_sft_full.sh

# Complete the documented response-length pilot before a long RL run.
PYTHON_BIN=python ACK_RL_LENGTH_BUDGET=1 bash scripts/run_rl_grpo.sh
```

SFT defaults to all 1,992 accepted ReTool trajectories for three epochs (747 updates). RL defaults to all 17,243 deduplicated DAPO-Math prompts, 64 prompts × 8 responses per update, and 3,000 plain-GRPO updates. See the [Chinese Math Agent training guide](docs/math_agent_training_zh.md) for launch, resume, total sampling budget, length-pilot, monitoring, and override details. The full scripts have passed complete-data preflight and command expansion; the full-duration runs have not yet been executed.

## CPU verification

```bash
cd /path/to/math-agent
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
python -X pycache_prefix=/tmp/math-agent-pycache -m compileall -q math_agent tests
bash -n scripts/run_sft_smoke.sh scripts/run_rl_smoke.sh \
  scripts/run_sft_full.sh scripts/run_rl_grpo.sh

MATH_AGENT_RUN_DOCKER_TESTS=1 \
  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_python_tool.py
```

The unit suite does not execute untrusted code on the host and does not require GPUs. A live sandbox integration check additionally requires the configured Docker image to be present.
