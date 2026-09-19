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
SFT_DATA=${SFT_DATA:-"${PROJECT_ROOT}/data/processed/retool_sft_train.jsonl"}
EXPECTED_TRAIN_ROWS=${EXPECTED_TRAIN_ROWS:-1992}
FORCE_PREPROCESS=${FORCE_PREPROCESS:-0}

HF_CHECKPOINT=${HF_CHECKPOINT:-"${MODEL_ROOT}/Qwen3-8B-Base"}
INITIAL_CHECKPOINT=${INITIAL_CHECKPOINT:-"${MODEL_ROOT}/Qwen3-8B-Base_torch_dist"}
RUN_NAME=${RUN_NAME:-qwen3-8b-base-retool-sft}
SAVE_DIR=${SAVE_DIR:-"${MATH_SHARED_ROOT}/checkpoints/${RUN_NAME}"}
RESUME=${RESUME:-0}

NUM_GPUS=${NUM_GPUS:-8}
NUM_EPOCHS=${NUM_EPOCHS:-3}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
SAVE_INTERVAL=${SAVE_INTERVAL:-auto}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
SEED=${SEED:-1234}
ROLLOUT_SEED=${ROLLOUT_SEED:-42}

USE_TENSORBOARD=${USE_TENSORBOARD:-1}
USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-math-agent}
WANDB_GROUP=${WANDB_GROUP:-"${RUN_NAME}"}
WANDB_MODE=${WANDB_MODE:-online}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/train"}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"${LOG_DIR}/tensorboard/${RUN_NAME}"}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
DRY_RUN=${DRY_RUN:-0}
# Megatron's argument validator initializes a CUDA context. Keep it opt-in so
# an ordinary dry-run remains safe on a login node or while GPUs belong to
# another job.
VALIDATE_ARGS_ON_DRY_RUN=${VALIDATE_ARGS_ON_DRY_RUN:-0}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_positive_integer() {
  local name=$1
  local value=$2
  if [[ ! ${value} =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer, got: ${value}" >&2
    exit 1
  fi
}

checkpoint_is_loadable() {
  local checkpoint=$1
  local tracker="${checkpoint}/latest_checkpointed_iteration.txt"
  [[ -f ${tracker} ]] || return 1
  local step
  step=$(tr -d '[:space:]' < "${tracker}")
  local payload
  if [[ ${step} == release ]]; then
    payload="${checkpoint}/release"
  elif [[ ${step} =~ ^[0-9]+$ ]]; then
    printf -v payload '%s/iter_%07d' "${checkpoint}" "$((10#${step}))"
  else
    return 1
  fi
  [[ -f "${payload}/.metadata" ]] &&
    [[ -n $(find "${payload}" -maxdepth 1 -type f -name '*.distcp' -print -quit 2>/dev/null) ]]
}

for path in "${SLIME_ROOT}/train_async.py" "${SLIME_ROOT}/scripts/models/qwen3-8B.sh" "${RAW_DATA}"; do
  if [[ ! -e ${path} ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done
for path in "${MEGATRON_ROOT}" "${HF_CHECKPOINT}" "${INITIAL_CHECKPOINT}"; do
  if [[ ! -d ${path} ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
done
if ! checkpoint_is_loadable "${INITIAL_CHECKPOINT}"; then
  echo "Initial checkpoint is incomplete: ${INITIAL_CHECKPOINT}" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Missing Python executable: ${PYTHON_BIN}" >&2
  exit 1
fi

require_positive_integer NUM_GPUS "${NUM_GPUS}"
require_positive_integer NUM_EPOCHS "${NUM_EPOCHS}"
require_positive_integer ROLLOUT_BATCH_SIZE "${ROLLOUT_BATCH_SIZE}"
require_positive_integer GLOBAL_BATCH_SIZE "${GLOBAL_BATCH_SIZE}"
require_positive_integer MAX_TOKENS_PER_GPU "${MAX_TOKENS_PER_GPU}"
require_positive_integer TENSOR_MODEL_PARALLEL_SIZE "${TENSOR_MODEL_PARALLEL_SIZE}"
require_positive_integer SEED "${SEED}"
require_positive_integer ROLLOUT_SEED "${ROLLOUT_SEED}"
if [[ ${GLOBAL_BATCH_SIZE} -ne ${ROLLOUT_BATCH_SIZE} ]]; then
  echo "SFT requires GLOBAL_BATCH_SIZE == ROLLOUT_BATCH_SIZE for one optimizer step per rollout." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
RUN_TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_LOG=${RUN_LOG:-"${LOG_DIR}/${RUN_NAME}-${RUN_TIMESTAMP}.log"}
exec > >(tee -a "${RUN_LOG}") 2>&1

if [[ ! -s ${SFT_DATA} ]] || is_true "${FORCE_PREPROCESS}"; then
  echo "Preparing the complete ReTool SFT dataset: ${SFT_DATA}"
  "${PYTHON_BIN}" -m math_agent.data retool --input "${RAW_DATA}" --output "${SFT_DATA}"
else
  echo "Reusing processed SFT data: ${SFT_DATA}"
fi

DATA_ROWS=$(wc -l < "${SFT_DATA}" | tr -d '[:space:]')
require_positive_integer DATA_ROWS "${DATA_ROWS}"
if [[ ${EXPECTED_TRAIN_ROWS} != 0 && ${DATA_ROWS} -ne ${EXPECTED_TRAIN_ROWS} ]]; then
  echo "Expected ${EXPECTED_TRAIN_ROWS} SFT rows, found ${DATA_ROWS}." >&2
  echo "Set EXPECTED_TRAIN_ROWS=0 only after auditing a deliberate dataset change." >&2
  exit 1
fi
UPDATES_PER_EPOCH=$((DATA_ROWS / ROLLOUT_BATCH_SIZE))
if [[ ${UPDATES_PER_EPOCH} -eq 0 ]]; then
  echo "ROLLOUT_BATCH_SIZE exceeds the SFT dataset size." >&2
  exit 1
fi
if (( DATA_ROWS % ROLLOUT_BATCH_SIZE != 0 )); then
  echo "Warning: $((DATA_ROWS % ROLLOUT_BATCH_SIZE)) trailing rows per epoch are not counted in the update budget."
fi
TRAIN_UPDATES=${TRAIN_UPDATES:-$((UPDATES_PER_EPOCH * NUM_EPOCHS))}
require_positive_integer TRAIN_UPDATES "${TRAIN_UPDATES}"
SLIME_NUM_ROLLOUT=$((TRAIN_UPDATES + 1))
if [[ ${SAVE_INTERVAL} == auto ]]; then
  # Slime checks (rollout_id + 1) % save_interval while a release checkpoint
  # starts at rollout_id=1. Adding one avoids a checkpoint immediately before
  # every intended epoch boundary and an extra adjacent final checkpoint.
  SAVE_INTERVAL=$((UPDATES_PER_EPOCH + 1))
fi
require_positive_integer SAVE_INTERVAL "${SAVE_INTERVAL}"

LOAD_CHECKPOINT=${INITIAL_CHECKPOINT}
if is_true "${RESUME}"; then
  if ! checkpoint_is_loadable "${SAVE_DIR}"; then
    echo "RESUME=1 but no complete checkpoint exists at ${SAVE_DIR}" >&2
    exit 1
  fi
  LOAD_CHECKPOINT=${SAVE_DIR}
  LOADED_STEP=$(tr -d '[:space:]' < "${SAVE_DIR}/latest_checkpointed_iteration.txt")
  if [[ ! ${LOADED_STEP} =~ ^[0-9]+$ ]]; then
    echo "Saved step is not numeric: ${LOADED_STEP}" >&2
    exit 1
  fi
  if (( 10#${LOADED_STEP} >= TRAIN_UPDATES )); then
    echo "Saved step ${LOADED_STEP} already reaches TRAIN_UPDATES=${TRAIN_UPDATES}." >&2
    exit 1
  fi
elif [[ -d ${SAVE_DIR} ]] && find "${SAVE_DIR}" -mindepth 1 -print -quit | grep -q .; then
  echo "SAVE_DIR is non-empty; refusing to overwrite it: ${SAVE_DIR}" >&2
  echo "Use RESUME=1 for the same run or choose a new RUN_NAME/SAVE_DIR." >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${SLIME_ROOT}:${MEGATRON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export no_proxy="127.0.0.1,${MASTER_ADDR}${no_proxy:+,${no_proxy}}"
export TENSORBOARD_DIR

# shellcheck source=/dev/null
source "${SLIME_ROOT}/scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --load "${LOAD_CHECKPOINT}"
  --save "${SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL}"
)
SFT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data "${SFT_DATA}"
  --input-key messages
  --tool-key tools
  --rollout-shuffle
  --rollout-seed "${ROLLOUT_SEED}"
  --num-rollout "${SLIME_NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --num-steps-per-rollout 1
  --loss-type sft_loss
  --loss-mask-type qwen3
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
)
PERF_ARGS=(
  --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)
OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style cosine
  --lr-decay-iters "${TRAIN_UPDATES}"
  --min-lr "${MIN_LR}"
  --lr-warmup-fraction 0.1
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
  --seed "${SEED}"
)
MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)
TRACKING_ARGS=()
if is_true "${USE_TENSORBOARD}"; then
  mkdir -p "${TENSORBOARD_DIR}"
  TRACKING_ARGS+=(--use-tensorboard --tb-project-name math-agent --tb-experiment-name "${RUN_NAME}")
fi
if is_true "${USE_WANDB}"; then
  TRACKING_ARGS+=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-mode "${WANDB_MODE}"
    --disable-wandb-random-suffix
  )
fi

TRAIN_CMD=(
  "${PYTHON_BIN}" "${SLIME_ROOT}/train_async.py"
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${NUM_GPUS}"
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${SFT_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${MISC_ARGS[@]}"
  "${TRACKING_ARGS[@]}"
)

echo "Run name: ${RUN_NAME}"
echo "Data rows: ${DATA_ROWS}; epochs: ${NUM_EPOCHS}; target updates: ${TRAIN_UPDATES}"
echo "Load: ${LOAD_CHECKPOINT}"
echo "Save: ${SAVE_DIR}; interval: ${SAVE_INTERVAL}"
echo "Log: ${RUN_LOG}"

if is_true "${DRY_RUN}"; then
  printf 'Dry-run training command:'
  printf ' %q' "${TRAIN_CMD[@]}"
  printf '\n'
  if is_true "${VALIDATE_ARGS_ON_DRY_RUN}"; then
    "${PYTHON_BIN}" -c \
      'from slime.utils.arguments import parse_args; args = parse_args(); print(f"Slime arguments valid: num_rollout={args.num_rollout}")' \
      "${TRAIN_CMD[@]:2}"
  fi
  exit 0
fi

for command in ray nvidia-smi; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing command: ${command}" >&2
    exit 1
  fi
done
VISIBLE_GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]')
if [[ ${VISIBLE_GPU_COUNT} -lt ${NUM_GPUS} ]]; then
  echo "Need ${NUM_GPUS} visible GPUs, found ${VISIBLE_GPU_COUNT}." >&2
  exit 1
fi
if [[ ${ALLOW_BUSY_GPUS:-0} != 1 ]] &&
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
  echo "Visible GPUs already have compute processes; refusing to interfere." >&2
  exit 1
fi
if ray status >/dev/null 2>&1; then
  echo "A Ray cluster is already active; refusing to reuse or stop it." >&2
  exit 1
fi

mkdir -p "$(dirname -- "${SAVE_DIR}")"
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

RUNTIME_ENV_JSON=$(printf '{"env_vars":{"PYTHONPATH":"%s","CUDA_DEVICE_MAX_CONNECTIONS":"1","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True","TENSORBOARD_DIR":"%s"}}' \
  "${PYTHONPATH}" "${TENSORBOARD_DIR}")

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${TRAIN_CMD[@]}"
