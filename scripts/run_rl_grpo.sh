#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
SLIME_ROOT=${SLIME_ROOT:-"${PROJECT_ROOT}/third_party/slime"}
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
PYTHON_BIN=${PYTHON_BIN:-python3}
MATH_SHARED_ROOT=${MATH_SHARED_ROOT:-/data/dhsun/math-agent}
MODEL_ROOT=${MODEL_ROOT:-"${MATH_SHARED_ROOT}/models"}

RAW_DATA=${RAW_DATA:-"${PROJECT_ROOT}/data/raw/dapo_math_17k/data/dapo-math-17k.parquet"}
RL_DATA=${RL_DATA:-"${PROJECT_ROOT}/data/processed/dapo_math_rl_train.jsonl"}
EXPECTED_TRAIN_ROWS=${EXPECTED_TRAIN_ROWS:-17243}
FORCE_PREPROCESS=${FORCE_PREPROCESS:-0}

HF_CHECKPOINT=${HF_CHECKPOINT:-"${MODEL_ROOT}/Qwen3-8B"}
INITIAL_CHECKPOINT=${INITIAL_CHECKPOINT:-"${MODEL_ROOT}/Qwen3-8B_torch_dist"}
REF_CHECKPOINT=${REF_CHECKPOINT:-"${MODEL_ROOT}/Qwen3-8B_torch_dist"}
RUN_NAME=${RUN_NAME:-qwen3-8b-dapo-grpo}
SAVE_DIR=${SAVE_DIR:-"${MATH_SHARED_ROOT}/checkpoints/${RUN_NAME}"}
RESUME=${RESUME:-0}

NUM_GPUS=${NUM_GPUS:-8}
TRAIN_UPDATES=${TRAIN_UPDATES:-3000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-64}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-12288}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-16384}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.7}
SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY:-32}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
LR=${LR:-1e-6}
SEED=${SEED:-1234}
ROLLOUT_SEED=${ROLLOUT_SEED:-42}

MATH_AGENT_TOOL_IMAGE=${MATH_AGENT_TOOL_IMAGE:-python@sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76}
MATH_AGENT_TOOL_CONCURRENCY=${MATH_AGENT_TOOL_CONCURRENCY:-8}
MATH_AGENT_TOOL_TIMEOUT=${MATH_AGENT_TOOL_TIMEOUT:-10}
MATH_AGENT_MAX_TOOL_CALLS=${MATH_AGENT_MAX_TOOL_CALLS:-4}

USE_TENSORBOARD=${USE_TENSORBOARD:-1}
USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-math-agent}
WANDB_GROUP=${WANDB_GROUP:-"${RUN_NAME}"}
WANDB_MODE=${WANDB_MODE:-online}
LOG_DIR=${LOG_DIR:-"${PROJECT_ROOT}/logs/train"}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-"${LOG_DIR}/tensorboard/${RUN_NAME}"}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
ACK_RL_LENGTH_BUDGET=${ACK_RL_LENGTH_BUDGET:-0}
PILOT_MAX_UPDATES=${PILOT_MAX_UPDATES:-5}
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

for path in "${SLIME_ROOT}/train.py" "${SLIME_ROOT}/scripts/models/qwen3-8B.sh" "${RAW_DATA}"; do
  if [[ ! -e ${path} ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done
for path in "${MEGATRON_ROOT}" "${HF_CHECKPOINT}" "${INITIAL_CHECKPOINT}" "${REF_CHECKPOINT}"; do
  if [[ ! -d ${path} ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
done
for checkpoint in "${INITIAL_CHECKPOINT}" "${REF_CHECKPOINT}"; do
  if ! checkpoint_is_loadable "${checkpoint}"; then
    echo "Checkpoint is incomplete: ${checkpoint}" >&2
    exit 1
  fi
done
for command in "${PYTHON_BIN}" docker; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing command: ${command}" >&2
    exit 1
  fi
done
if ! docker image inspect "${MATH_AGENT_TOOL_IMAGE}" >/dev/null 2>&1; then
  echo "Sandbox image is not local: ${MATH_AGENT_TOOL_IMAGE}" >&2
  exit 1
fi

for integer_setting in \
  "NUM_GPUS:${NUM_GPUS}" \
  "TRAIN_UPDATES:${TRAIN_UPDATES}" \
  "SAVE_INTERVAL:${SAVE_INTERVAL}" \
  "ROLLOUT_BATCH_SIZE:${ROLLOUT_BATCH_SIZE}" \
  "N_SAMPLES_PER_PROMPT:${N_SAMPLES_PER_PROMPT}" \
  "GLOBAL_BATCH_SIZE:${GLOBAL_BATCH_SIZE}" \
  "ROLLOUT_MAX_RESPONSE_LEN:${ROLLOUT_MAX_RESPONSE_LEN}" \
  "ROLLOUT_MAX_CONTEXT_LEN:${ROLLOUT_MAX_CONTEXT_LEN}" \
  "MAX_TOKENS_PER_GPU:${MAX_TOKENS_PER_GPU}" \
  "TENSOR_MODEL_PARALLEL_SIZE:${TENSOR_MODEL_PARALLEL_SIZE}" \
  "ROLLOUT_NUM_GPUS_PER_ENGINE:${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
  "SGLANG_SERVER_CONCURRENCY:${SGLANG_SERVER_CONCURRENCY}" \
  "MATH_AGENT_TOOL_CONCURRENCY:${MATH_AGENT_TOOL_CONCURRENCY}" \
  "MATH_AGENT_TOOL_TIMEOUT:${MATH_AGENT_TOOL_TIMEOUT}" \
  "MATH_AGENT_MAX_TOOL_CALLS:${MATH_AGENT_MAX_TOOL_CALLS}" \
  "SEED:${SEED}" \
  "ROLLOUT_SEED:${ROLLOUT_SEED}" \
  "PILOT_MAX_UPDATES:${PILOT_MAX_UPDATES}"; do
  require_positive_integer "${integer_setting%%:*}" "${integer_setting#*:}"
done
EXPECTED_GLOBAL_BATCH=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
if [[ ${GLOBAL_BATCH_SIZE} -ne ${EXPECTED_GLOBAL_BATCH} ]]; then
  echo "GLOBAL_BATCH_SIZE must equal ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT (${EXPECTED_GLOBAL_BATCH})." >&2
  exit 1
fi
if [[ ${ROLLOUT_MAX_RESPONSE_LEN} -gt ${ROLLOUT_MAX_CONTEXT_LEN} ]]; then
  echo "ROLLOUT_MAX_RESPONSE_LEN cannot exceed ROLLOUT_MAX_CONTEXT_LEN." >&2
  exit 1
fi
if [[ ${MAX_TOKENS_PER_GPU} -lt ${ROLLOUT_MAX_CONTEXT_LEN} ]]; then
  echo "MAX_TOKENS_PER_GPU must cover ROLLOUT_MAX_CONTEXT_LEN for the validated TP=2 configuration." >&2
  exit 1
fi
if [[ ${TRAIN_UPDATES} -gt ${PILOT_MAX_UPDATES} ]] && ! is_true "${ACK_RL_LENGTH_BUDGET}"; then
  echo "Refusing a long RL run before the response-length pilot is accepted." >&2
  echo "The 8,192-token smoke run truncated 62.5% of trajectories." >&2
  echo "Run <= ${PILOT_MAX_UPDATES} updates first, then set ACK_RL_LENGTH_BUDGET=1 after reviewing truncation." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
RUN_TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_LOG=${RUN_LOG:-"${LOG_DIR}/${RUN_NAME}-${RUN_TIMESTAMP}.log"}
exec > >(tee -a "${RUN_LOG}") 2>&1

if [[ ! -s ${RL_DATA} ]] || is_true "${FORCE_PREPROCESS}"; then
  echo "Preparing the complete deduplicated DAPO-Math dataset: ${RL_DATA}"
  "${PYTHON_BIN}" -m math_agent.data dapo --input "${RAW_DATA}" --output "${RL_DATA}"
else
  echo "Reusing processed RL data: ${RL_DATA}"
fi

DATA_ROWS=$(wc -l < "${RL_DATA}" | tr -d '[:space:]')
require_positive_integer DATA_ROWS "${DATA_ROWS}"
if [[ ${EXPECTED_TRAIN_ROWS} != 0 && ${DATA_ROWS} -ne ${EXPECTED_TRAIN_ROWS} ]]; then
  echo "Expected ${EXPECTED_TRAIN_ROWS} RL rows, found ${DATA_ROWS}." >&2
  echo "Set EXPECTED_TRAIN_ROWS=0 only after auditing a deliberate dataset change." >&2
  exit 1
fi
SLIME_NUM_ROLLOUT=$((TRAIN_UPDATES + 1))

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
export MATH_AGENT_TOOL_IMAGE MATH_AGENT_TOOL_CONCURRENCY MATH_AGENT_TOOL_TIMEOUT MATH_AGENT_MAX_TOOL_CALLS
export TENSORBOARD_DIR

# shellcheck source=/dev/null
source "${SLIME_ROOT}/scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_CHECKPOINT}"
  --load "${LOAD_CHECKPOINT}"
  --save "${SAVE_DIR}"
  --save-interval "${SAVE_INTERVAL}"
)
ROLLOUT_ARGS=(
  --prompt-data "${RL_DATA}"
  --input-key prompt
  --label-key label
  --tool-key tools
  --rollout-shuffle
  --rollout-seed "${ROLLOUT_SEED}"
  --reward-key score
  --num-rollout "${SLIME_NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --num-steps-per-rollout 1
  --balance-data
)
GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef "${KL_LOSS_COEF}"
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.2
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
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --seed "${SEED}"
)
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
)
MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)
CUSTOM_ARGS=(
  --custom-generate-function-path math_agent.rollout.generate
  --custom-rm-path math_agent.reward.reward_func
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
  "${PYTHON_BIN}" "${SLIME_ROOT}/train.py"
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${NUM_GPUS}"
  --colocate
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${GRPO_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
  "${CUSTOM_ARGS[@]}"
  "${TRACKING_ARGS[@]}"
)

echo "Run name: ${RUN_NAME}"
echo "Data rows: ${DATA_ROWS}; target updates: ${TRAIN_UPDATES}"
echo "Per update: ${ROLLOUT_BATCH_SIZE} prompts x ${N_SAMPLES_PER_PROMPT} samples = ${GLOBAL_BATCH_SIZE} trajectories"
echo "Response/context budget: ${ROLLOUT_MAX_RESPONSE_LEN}/${ROLLOUT_MAX_CONTEXT_LEN}"
echo "Load: ${LOAD_CHECKPOINT}; fixed reference: ${REF_CHECKPOINT}"
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

RUNTIME_ENV_JSON=$(printf '{"env_vars":{"PYTHONPATH":"%s","CUDA_DEVICE_MAX_CONNECTIONS":"1","PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True","MATH_AGENT_TOOL_IMAGE":"%s","MATH_AGENT_TOOL_CONCURRENCY":"%s","MATH_AGENT_TOOL_TIMEOUT":"%s","MATH_AGENT_MAX_TOOL_CALLS":"%s","TENSORBOARD_DIR":"%s"}}' \
  "${PYTHONPATH}" "${MATH_AGENT_TOOL_IMAGE}" "${MATH_AGENT_TOOL_CONCURRENCY}" \
  "${MATH_AGENT_TOOL_TIMEOUT}" "${MATH_AGENT_MAX_TOOL_CALLS}" "${TENSORBOARD_DIR}")

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${TRAIN_CMD[@]}"
