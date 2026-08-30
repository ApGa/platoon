#!/bin/bash
#SBATCH --job-name=swe-rebench-preflight
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=cpu_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00

# Metadata scan:
#   sbatch --partition=cpu_short --time=1:00:00 \
#     slurm-scripts/openreward-swe-rebench-preflight.sh metadata
#
# Full execution scan (6,272 default images; requires an explicit array):
#   sbatch --array=0-255%32 \
#     slurm-scripts/openreward-swe-rebench-preflight.sh execute
#
# A targeted cache-warming validation should name its tasks and explicitly
# retain images:
#   SWE_PREFLIGHT_INDICES=12,90,400-410 \
#   SWE_PREFLIGHT_RETAIN_IMAGES=1 \
#   sbatch slurm-scripts/openreward-swe-rebench-preflight.sh execute
#
# A bounded audit over already-warm images:
#   SWE_PREFLIGHT_SAMPLE_PER_LANGUAGE=4 \
#   SWE_PREFLIGHT_CACHED_ONLY=1 \
#   SWE_PREFLIGHT_RETAIN_IMAGES=1 \
#   sbatch slurm-scripts/openreward-swe-rebench-preflight.sh execute

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -n "${PLATOON_REPO_ROOT:-}" ]]; then
  REPO_ROOT=${PLATOON_REPO_ROOT}
elif [[
  -n "${SLURM_SUBMIT_DIR:-}"
  && -d "${SLURM_SUBMIT_DIR}/external/swe-rebench-v2-openrewardenv"
]]; then
  # sbatch executes a spool copy of this script, so BASH_SOURCE no longer
  # identifies the checkout. Slurm preserves the submission directory.
  REPO_ROOT=${SLURM_SUBMIT_DIR}
else
  REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
fi
ENV_ROOT=${REPO_ROOT}/external/swe-rebench-v2-openrewardenv
SWE_RUNTIME_GUARD=${REPO_ROOT}/plugins/openreward/swe-rebench-runtime-guard.sh
PYTHON=${ENV_ROOT}/.venv-openreward/bin/python
DATA_DIR=${SWE_REBENCH_DATA_DIR:-${REPO_ROOT}/.cache/swe-rebench-v2-filtered-verified}
MODE=${1:-${SWE_PREFLIGHT_MODE:-metadata}}

[[ -r "${SWE_RUNTIME_GUARD}" ]] || {
  echo "ERROR: missing SWE-rebench runtime guard: ${SWE_RUNTIME_GUARD}" >&2
  exit 2
}
# Never scan or execute tasks from a stale, caller-selected, or dirty checkout.
# shellcheck source=../plugins/openreward/swe-rebench-runtime-guard.sh
source "${SWE_RUNTIME_GUARD}"
swe_rebench_require_validated_source \
  "${REPO_ROOT}" \
  "${ENV_ROOT}" \
  "${SWE_REBENCH_SOURCE_REVISION:-}" \
  "SWE-rebench preflight" || exit $?
export SWE_REBENCH_SOURCE_REVISION=${SWE_REBENCH_VALIDATED_SOURCE_REVISION}

[[ -x "${PYTHON}" ]] || {
  echo "ERROR: missing SWE-rebench runtime: ${PYTHON}" >&2
  exit 2
}
[[ -f "${DATA_DIR}/task_index.json" ]] || {
  echo "ERROR: missing SWE-rebench task index: ${DATA_DIR}/task_index.json" >&2
  exit 2
}
case "${MODE}" in
  metadata|execute) ;;
  *)
    echo "ERROR: mode must be metadata or execute; got ${MODE@Q}." >&2
    exit 2
    ;;
esac

RUN_ID=${SWE_PREFLIGHT_RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}
RESULT_ROOT=${SWE_PREFLIGHT_RESULT_ROOT:-${REPO_ROOT}/.cache/swe-rebench-preflight/${RUN_ID}}
mkdir -p "${RESULT_ROOT}"

COMMON_ARGS=(
  --data-dir "${DATA_DIR}"
  --mode "${MODE}"
  --resume
)
if [[ -n "${SWE_PREFLIGHT_INDICES:-}" ]]; then
  COMMON_ARGS+=(--indices "${SWE_PREFLIGHT_INDICES}")
fi
if [[ -n "${SWE_PREFLIGHT_MAX_TASKS:-}" ]]; then
  COMMON_ARGS+=(--max-tasks "${SWE_PREFLIGHT_MAX_TASKS}")
fi
if [[ -n "${SWE_PREFLIGHT_SAMPLE_PER_LANGUAGE:-}" ]]; then
  COMMON_ARGS+=(
    --sample-per-language "${SWE_PREFLIGHT_SAMPLE_PER_LANGUAGE}"
    --sample-seed "${SWE_PREFLIGHT_SAMPLE_SEED:-0}"
  )
fi
if [[ "${SWE_PREFLIGHT_CACHED_ONLY:-0}" == 1 ]]; then
  COMMON_ARGS+=(
    --cached-image-dir "${SWE_PREFLIGHT_EXISTING_IMAGE_CACHE:-${REPO_ROOT}/.cache/swe-rebench-enroot-images}"
  )
fi
if [[ "${SWE_PREFLIGHT_FAIL_ON_INVALID:-0}" == 1 ]]; then
  COMMON_ARGS+=(--fail-on-invalid)
fi

if [[ "${MODE}" == metadata ]]; then
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: metadata mode is a single streaming scan, not an array job." >&2
    exit 2
  fi
  exec "${PYTHON}" "${ENV_ROOT}/preflight.py" \
    "${COMMON_ARGS[@]}" \
    --output "${RESULT_ROOT}/metadata.jsonl"
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  NUM_SHARDS=${SWE_PREFLIGHT_NUM_SHARDS:-256}
  SHARD_ID=${SLURM_ARRAY_TASK_ID}
elif [[ -n "${SWE_PREFLIGHT_INDICES:-}" ]]; then
  NUM_SHARDS=1
  SHARD_ID=0
elif [[ -n "${SWE_PREFLIGHT_SAMPLE_PER_LANGUAGE:-}" ]]; then
  NUM_SHARDS=1
  SHARD_ID=0
else
  echo "ERROR: a full execution scan requires an explicit Slurm array." >&2
  echo "Example: sbatch --array=0-255%32 $0 execute" >&2
  exit 2
fi
if ((SHARD_ID < 0 || SHARD_ID >= NUM_SHARDS)); then
  echo "ERROR: array task ${SHARD_ID} is outside 0..$((NUM_SHARDS - 1))." >&2
  exit 2
fi

JOB_ID=${SLURM_JOB_ID:-manual}
LOCAL_ROOT=${SLURM_TMPDIR:-/tmp}/swe-rebench-preflight-${JOB_ID}
export ENROOT_CACHE_PATH=${LOCAL_ROOT}/enroot-cache
export ENROOT_DATA_PATH=${LOCAL_ROOT}/enroot-data
export ENROOT_RUNTIME_PATH=${LOCAL_ROOT}/enroot-runtime
export ENROOT_TEMP_PATH=${LOCAL_ROOT}/enroot-tmp
export SWE_ENROOT_SESSION_TMP_ROOT=${LOCAL_ROOT}/session-tmp
export ENROOT_MOUNT_HOME=
export ENROOT_MAX_PROCESSORS=${SWE_PREFLIGHT_ENROOT_PROCESSORS:-4}
export SWE_SANDBOX_RUNTIME=enroot
export SWE_ENROOT_IMPORT_ATTEMPTS=${SWE_ENROOT_IMPORT_ATTEMPTS:-3}
export SWE_ENROOT_IMPORT_RETRY_DELAY_SECONDS=${SWE_ENROOT_IMPORT_RETRY_DELAY_SECONDS:-2}
export SWE_ENROOT_IMPORT_TIMEOUT_SECONDS=${SWE_ENROOT_IMPORT_TIMEOUT_SECONDS:-1800}
export SWE_TEST_TIMEOUT_SECONDS=${SWE_TEST_TIMEOUT_SECONDS:-1200}
mkdir -p \
  "${ENROOT_CACHE_PATH}" \
  "${ENROOT_DATA_PATH}" \
  "${ENROOT_RUNTIME_PATH}" \
  "${ENROOT_TEMP_PATH}" \
  "${SWE_ENROOT_SESSION_TMP_ROOT}"
chmod 0700 "${SWE_ENROOT_SESSION_TMP_ROOT}"
swe_rebench_require_enroot_runtime \
  "${ENROOT_TEMP_PATH}" \
  "SWE-rebench execute preflight" || exit $?

EXECUTION_ARGS=(
  --num-shards "${NUM_SHARDS}"
  --shard-id "${SHARD_ID}"
  --workers "${SWE_PREFLIGHT_WORKERS:-2}"
  --test-timeout "${SWE_TEST_TIMEOUT_SECONDS}"
  --output "${RESULT_ROOT}/execute-shard-$(printf '%05d' "${SHARD_ID}").jsonl"
)

if [[ "${SWE_PREFLIGHT_RETAIN_IMAGES:-0}" == 1 ]]; then
  # Retaining the verified set would require roughly 18 TiB; retaining the
  # original 32K-image dataset would require roughly 90 TiB.
  # This mode is intended only for an explicit, small SWE_PREFLIGHT_INDICES set.
  [[ -n "${SWE_PREFLIGHT_INDICES:-}" || -n "${SWE_PREFLIGHT_SAMPLE_PER_LANGUAGE:-}" ]] || {
    echo "ERROR: image retention requires explicit indices or a bounded stratified sample." >&2
    exit 2
  }
  export SWE_ENROOT_IMAGE_CACHE=${SWE_ENROOT_IMAGE_CACHE:-${REPO_ROOT}/.cache/swe-rebench-enroot-images}
else
  dedicated_cache=${LOCAL_ROOT}/images
  if [[ -n "${SWE_ENROOT_IMAGE_CACHE:-}" && "${SWE_ENROOT_IMAGE_CACHE}" != "${dedicated_cache}" ]]; then
    echo "ERROR: refusing to delete images from non-dedicated cache ${SWE_ENROOT_IMAGE_CACHE@Q}." >&2
    exit 2
  fi
  export SWE_ENROOT_IMAGE_CACHE=${dedicated_cache}
  EXECUTION_ARGS+=(--delete-images)
fi
mkdir -p "${SWE_ENROOT_IMAGE_CACHE}"

exec "${PYTHON}" "${ENV_ROOT}/preflight.py" \
  "${COMMON_ARGS[@]}" \
  "${EXECUTION_ARGS[@]}"
