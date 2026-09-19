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
RL_DATA=${RL_DATA:-"${PROJECT_ROOT}/data/processed/dapo_math_rl_smoke.jsonl"}
RL_MAX_ROWS=${RL_MAX_ROWS:-64}
NUM_ROLLOUT=${NUM_ROLLOUT:-2}
HF_CHECKPOINT=${HF_CHECKPOINT:-"${MODEL_ROOT}/Qwen3-8B"}
MEGATRON_CHECKPOINT=${MEGATRON_CHECKPOINT:-"${MODEL_ROOT}/Qwen3-8B_torch_dist"}
REF_CHECKPOINT=${REF_CHECKPOINT:-"${MEGATRON_CHECKPOINT}"}
SAVE_DIR=${SAVE_DIR:-"${PROJECT_ROOT}/checkpoints/qwen3-8b-dapo-grpo-smoke"}
NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
MATH_AGENT_TOOL_IMAGE=${MATH_AGENT_TOOL_IMAGE:-python@sha256:be1575ed968de893bd54f4c56315ff7c4736ce522c1bca08fd521731aafc0d76}

for path in "${SLIME_ROOT}/train.py" "${SLIME_ROOT}/scripts/models/qwen3-8B.sh" "${RAW_DATA}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done
for path in "${MEGATRON_ROOT}" "${HF_CHECKPOINT}" "${MEGATRON_CHECKPOINT}" "${REF_CHECKPOINT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "Missing required directory: ${path}" >&2
    exit 1
  fi
done
for command in ray docker nvidia-smi "${PYTHON_BIN}"; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing command: ${command}" >&2
    exit 1
  fi
done
if ! docker image inspect "${MATH_AGENT_TOOL_IMAGE}" >/dev/null 2>&1; then
  echo "Sandbox image is not local: ${MATH_AGENT_TOOL_IMAGE}. Pull and pin it before training." >&2
  exit 1
fi
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
export no_proxy="127.0.0.1,${MASTER_ADDR}${no_proxy:+,${no_proxy}}"
export MATH_AGENT_TOOL_IMAGE
export MATH_AGENT_TOOL_CONCURRENCY=${MATH_AGENT_TOOL_CONCURRENCY:-8}
export MATH_AGENT_TOOL_TIMEOUT=${MATH_AGENT_TOOL_TIMEOUT:-10}
export MATH_AGENT_MAX_TOOL_CALLS=${MATH_AGENT_MAX_TOOL_CALLS:-4}

"${PYTHON_BIN}" -m math_agent.data dapo \
  --input "${RAW_DATA}" \
  --output "${RL_DATA}" \
  --max-rows "${RL_MAX_ROWS}"

# shellcheck source=/dev/null
source "${SLIME_ROOT}/scripts/models/qwen3-8B.sh"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_CHECKPOINT}"
  --load "${MEGATRON_CHECKPOINT}"
  --save "${SAVE_DIR}"
  --save-interval 1
)
ROLLOUT_ARGS=(
  --prompt-data "${RL_DATA}"
  --input-key prompt
  --label-key label
  --tool-key tools
  --rollout-shuffle
  --reward-key score
  # A release checkpoint resumes at rollout_id=1; the upper bound is exclusive.
  # Therefore num_rollout=2 executes exactly one learner update.
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-4}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
  --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-16384}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
  --global-batch-size "${GLOBAL_BATCH_SIZE:-32}"
  --num-steps-per-rollout 1
  --balance-data
)
GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef "${KL_LOSS_COEF:-0.001}"
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.2
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
  --lr "${LR:-1e-6}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.7}"
  --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-32}"
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

RUNTIME_ENV_JSON=$(printf '{"env_vars":{"PYTHONPATH":"%s","CUDA_DEVICE_MAX_CONNECTIONS":"1","MATH_AGENT_TOOL_IMAGE":"%s","MATH_AGENT_TOOL_CONCURRENCY":"%s","MATH_AGENT_TOOL_TIMEOUT":"%s","MATH_AGENT_MAX_TOOL_CALLS":"%s"}}' \
  "${PYTHONPATH}" "${MATH_AGENT_TOOL_IMAGE}" "${MATH_AGENT_TOOL_CONCURRENCY}" "${MATH_AGENT_TOOL_TIMEOUT}" "${MATH_AGENT_MAX_TOOL_CALLS}")

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${PYTHON_BIN}" "${SLIME_ROOT}/train.py" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${NUM_GPUS}" \
  --colocate \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}"
