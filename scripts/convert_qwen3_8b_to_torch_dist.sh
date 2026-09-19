#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

MATH_SHARED_ROOT=${MATH_SHARED_ROOT:-/data/dhsun/math-agent}
MODEL_ROOT=${MODEL_ROOT:-"${MATH_SHARED_ROOT}/models"}
CONVERT_GPU=${CONVERT_GPU:-0}
RUNTIME_IMAGE=${RUNTIME_IMAGE:-math-agent/slime-runtime:20260903-cu129}

usage() {
  cat <<'EOF'
Usage: bash scripts/convert_qwen3_8b_to_torch_dist.sh [base|instruct|all]...

With no arguments, converts both Qwen3-8B-Base and Qwen3-8B sequentially.

Environment variables:
  MODEL_ROOT        Model directory (default: /data/dhsun/math-agent/models)
  CONVERT_GPU       Single GPU used by the converter (default: 0)
  RUNTIME_IMAGE     Pinned slime runtime image
  ALLOW_BUSY_GPUS   Set to 1 only to bypass the selected-GPU occupancy check
EOF
}

if [[ $# -eq 0 ]]; then
  targets=(base instruct)
else
  targets=()
  for target in "$@"; do
    case "${target}" in
      base|instruct)
        targets+=("${target}")
        ;;
      all)
        targets+=(base instruct)
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown conversion target: ${target}" >&2
        usage >&2
        exit 2
        ;;
    esac
  done
fi

for command in docker nvidia-smi; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing command: ${command}" >&2
    exit 1
  fi
done

if [[ ! -d "${MODEL_ROOT}" ]]; then
  echo "Missing model root: ${MODEL_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/third_party/slime/tools/convert_hf_to_torch_dist.py" ]]; then
  echo "Missing slime converter under ${PROJECT_ROOT}/third_party/slime" >&2
  exit 1
fi
if ! docker image inspect "${RUNTIME_IMAGE}" >/dev/null 2>&1; then
  echo "Runtime image is not local: ${RUNTIME_IMAGE}" >&2
  exit 1
fi
if ! nvidia-smi --id="${CONVERT_GPU}" --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
  echo "GPU ${CONVERT_GPU} is not available" >&2
  exit 1
fi
if [[ ${ALLOW_BUSY_GPUS:-0} != 1 ]] && \
  nvidia-smi --id="${CONVERT_GPU}" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null | grep -Eq '[0-9]'; then
  echo "GPU ${CONVERT_GPU} already has a compute process; refusing to interfere." >&2
  exit 1
fi

checkpoint_is_complete() {
  local checkpoint=$1
  [[ -f "${checkpoint}/latest_checkpointed_iteration.txt" ]] &&
    [[ "$(tr -d '[:space:]' < "${checkpoint}/latest_checkpointed_iteration.txt")" == release ]] &&
    find "${checkpoint}/release" -type f -name '*.distcp' -print -quit 2>/dev/null | grep -q .
}

convert_one() {
  local target=$1
  local model_name
  case "${target}" in
    base)
      model_name=Qwen3-8B-Base
      ;;
    instruct)
      model_name=Qwen3-8B
      ;;
  esac

  local hf_checkpoint="${MODEL_ROOT}/${model_name}"
  local output_checkpoint="${MODEL_ROOT}/${model_name}_torch_dist"
  local partial_checkpoint="${output_checkpoint}.partial"

  if checkpoint_is_complete "${output_checkpoint}"; then
    echo "Checkpoint already complete; skipping: ${output_checkpoint}"
    return 0
  fi
  if [[ -e "${output_checkpoint}" ]]; then
    echo "Output exists but is not a complete torch_dist checkpoint: ${output_checkpoint}" >&2
    echo "Move it aside after inspection, then rerun the conversion." >&2
    return 1
  fi
  if [[ -e "${partial_checkpoint}" ]]; then
    echo "Partial output already exists: ${partial_checkpoint}" >&2
    echo "Move it aside after inspection, then rerun the conversion." >&2
    return 1
  fi
  for required_file in config.json model.safetensors.index.json tokenizer.json; do
    if [[ ! -f "${hf_checkpoint}/${required_file}" ]]; then
      echo "Missing Hugging Face file: ${hf_checkpoint}/${required_file}" >&2
      return 1
    fi
  done

  local output_uid
  local output_gid
  output_uid=$(stat -c %u "${MODEL_ROOT}")
  output_gid=$(stat -c %g "${MODEL_ROOT}")
  local timestamp
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  local log_dir="${PROJECT_ROOT}/logs/conversion"
  local log_file="${log_dir}/${model_name}-${timestamp}.log"
  mkdir -p "${log_dir}"

  echo "Converting ${hf_checkpoint}"
  echo "Output: ${output_checkpoint}"
  echo "GPU: ${CONVERT_GPU}"
  echo "Log: ${log_file}"

  docker run --rm \
    --gpus "device=${CONVERT_GPU}" \
    --ipc=host \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${PROJECT_ROOT}:/workspace/math-agent:ro" \
    -v "${MODEL_ROOT}:${MODEL_ROOT}" \
    -w /workspace/math-agent \
    -e HF_CHECKPOINT="${hf_checkpoint}" \
    -e SAVE_PATH="${partial_checkpoint}" \
    -e OUTPUT_UID="${output_uid}" \
    -e OUTPUT_GID="${output_gid}" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint bash \
    "${RUNTIME_IMAGE}" \
    -lc '
      set -uo pipefail
      source third_party/slime/scripts/models/qwen3-8B.sh
      conversion_status=0
      PYTHONPATH=/root/Megatron-LM:third_party/slime \
        python third_party/slime/tools/convert_hf_to_torch_dist.py \
          "${MODEL_ARGS[@]}" \
          --hf-checkpoint "${HF_CHECKPOINT}" \
          --save "${SAVE_PATH}" || conversion_status=$?
      if [[ -e "${SAVE_PATH}" ]]; then
        chown -R "${OUTPUT_UID}:${OUTPUT_GID}" "${SAVE_PATH}" || conversion_status=$?
      fi
      exit "${conversion_status}"
    ' 2>&1 | tee "${log_file}"

  if ! checkpoint_is_complete "${partial_checkpoint}"; then
    echo "Conversion finished without a valid release checkpoint: ${partial_checkpoint}" >&2
    return 1
  fi
  mv "${partial_checkpoint}" "${output_checkpoint}"
  echo "Conversion complete: $(du -sh "${output_checkpoint}" | cut -f1) ${output_checkpoint}"
}

declare -A converted=()
for target in "${targets[@]}"; do
  if [[ -z ${converted[${target}]+x} ]]; then
    convert_one "${target}"
    converted[${target}]=1
  fi
done
