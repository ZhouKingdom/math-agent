#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
SLIME_ROOT=${SLIME_ROOT:-"${PROJECT_ROOT}/third_party/slime"}
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
PYTHON_BIN=${PYTHON_BIN:-python3}
MATH_SHARED_ROOT=${MATH_SHARED_ROOT:-/data/dhsun/math-agent}
MODEL_ROOT=${MODEL_ROOT:-"${MATH_SHARED_ROOT}/models"}

RAW_DATA=${RAW_DATA:-"${PROJECT_ROOT}/data/raw/retool_sft/train_2000.parquet"}
SFT_DATA=${SFT_DATA:-"${PROJECT_ROOT}/data/processed/retool_sft_smoke.jsonl"}
SFT_MAX_ROWS=${SFT_MAX_ROWS:-64}
NUM_ROLLOUT=${NUM_ROLLOUT:-2}
HF_CHECKPOINT=${HF_CHECKPOINT:-"${MODEL_ROOT}/Qwen3-8B-Base"}
MEGATRON_CHECKPOINT=${MEGATRON_CHECKPOINT:-"${MODEL_ROOT}/Qwen3-8B-Base_torch_dist"}
SAVE_DIR=${SAVE_DIR:-"${PROJECT_ROOT}/checkpoints/qwen3-8b-base-retool-sft-smoke"}
NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}

for path in "${SLIME_ROOT}/train_async.py" "${SLIME_ROOT}/scripts/models/qwen3-8B.sh" "${RAW_DATA}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done
for path in "${MEGATRON_ROOT}" "${HF_CHECKPOINT}" "${MEGATRON_CHECKPOINT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
done
for command in ray nvidia-smi "${PYTHON_BIN}"; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing command: ${command}" >&2
    exit 1
  fi
done

if [[ ${ALLOW_BUSY_GPUS:-0} != 1 ]] && nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
  echo "Visible GPUs already have compute processes; refusing to interfere. Set ALLOW_BUSY_GPUS=1 only in an isolated allocation." >&2
  exit 1
fi
if ray status >/dev/null 2>&1; then
  echo "A Ray cluster is already active; refusing to reuse or stop it." >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${SLIME_ROOT}:${MEGATRON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export no_proxy="127.0.0.1,${MASTER_ADDR}${no_proxy:+,${no_proxy}}"

"${PYTHON_BIN}" -m math_agent.data retool \
  --input "${RAW_DATA}" \
  --output "${SFT_DATA}" \
  --max-rows "${SFT_MAX_ROWS}"

# shellcheck source=/dev/null
source "${SLIME_ROOT}/scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --load "${MEGATRON_CHECKPOINT}"
  --save "${SAVE_DIR}"
  --save-interval 1
)
SFT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data "${SFT_DATA}"
  --input-key messages
  --tool-key tools
  --rollout-shuffle
  # A release checkpoint resumes at rollout_id=1; the upper bound is exclusive.
  # Therefore num_rollout=2 executes exactly one learner update.
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}"
  --global-batch-size "${GLOBAL_BATCH_SIZE:-8}"
  --loss-type sft_loss
  --loss-mask-type qwen3
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)
PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-2}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
)
OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR:-1e-5}"
  --lr-decay-style cosine
  --min-lr "${MIN_LR:-1e-6}"
  --lr-warmup-fraction 0.1
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
)
MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host 127.0.0.1 \
  --dashboard-port "${RAY_DASHBOARD_PORT}"
cleanup() {
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

RUNTIME_ENV_JSON=$(printf '{"env_vars":{"PYTHONPATH":"%s","CUDA_DEVICE_MAX_CONNECTIONS":"1","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}}' "${PYTHONPATH}")

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${PYTHON_BIN}" "${SLIME_ROOT}/train_async.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MISC_ARGS[@]}"
