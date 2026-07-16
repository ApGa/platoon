#!/bin/bash

# Start one of the host-native OpenReward environments used by the mixed
# training launcher.  The Python server stays on the host because both TMax and
# SWE-rebench create writable Enroot task containers themselves.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${PLATOON_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
USER_ROOT=${PLATOON_USER_ROOT:-$(cd "${REPO_ROOT}/../.." && pwd)}
KIND=${OPENREWARD_SUPPLEMENTAL_KIND:?Set OPENREWARD_SUPPLEMENTAL_KIND to tmax or swe_rebench}
JOB_ID=${SLURM_JOB_ID:-manual}
LOCAL_ROOT=${SLURM_TMPDIR:-/tmp}/platoon-openreward-${JOB_ID}/${KIND}

export OPENREWARD_DISABLE_UPDATE_CHECK=1
export OPENHANDS_SUPPRESS_BANNER=1
export ENROOT_CACHE_PATH=${LOCAL_ROOT}/enroot-cache
export ENROOT_DATA_PATH=${LOCAL_ROOT}/enroot-data
export ENROOT_RUNTIME_PATH=${LOCAL_ROOT}/enroot-runtime
export ENROOT_MAX_PROCESSORS=${OPENREWARD_ENROOT_MAX_PROCESSORS:-4}
# Never let Enroot's site hook bind the submitter's home into untrusted task
# containers. Both environment backends enforce this again per start.
export ENROOT_MOUNT_HOME=
mkdir -p "${ENROOT_CACHE_PATH}" "${ENROOT_DATA_PATH}" "${ENROOT_RUNTIME_PATH}"

require_source_revision() {
  local repository=$1
  local expected=$2
  local actual
  actual=$(git -C "${repository}" rev-parse HEAD)
  if [[ "${actual}" != "${expected}" ]]; then
    echo "ERROR: verified OpenReward source revision mismatch:" >&2
    echo "       repository: ${repository}" >&2
    echo "       expected:   ${expected}" >&2
    echo "       actual:     ${actual}" >&2
    echo "Set the corresponding *_SOURCE_REVISION only after validating the new commit." >&2
    exit 2
  fi
}

case "${KIND}" in
  tmax)
    ENV_ROOT=${REPO_ROOT}/external/tmax
    TMAX_SOURCE_REVISION=${TMAX_SOURCE_REVISION:-00607489e4a433f24db3b791185b0d1f652246cb}
    require_source_revision "${ENV_ROOT}" "${TMAX_SOURCE_REVISION}"
    PYTHON=${ENV_ROOT}/.venv-openreward/bin/python
    [[ -x "${PYTHON}" ]] || {
      echo "ERROR: missing TMax runtime: ${PYTHON}" >&2
      exit 2
    }
    export OPENREWARD_PORT=${OPENREWARD_TMAX_PORT:-8083}
    export OPENREWARD_REWARD_MODE=binary
    export HF_HUB_OFFLINE=${TMAX_HF_HUB_OFFLINE:-${HF_HUB_OFFLINE:-1}}
    export TMAX_SANDBOX_RUNTIME=enroot
    export TMAX_HF_REVISION=${TMAX_HF_REVISION:-7b090eca98bf351356bc1c64290c5c4a09f2f98c}
    export TMAX_HF_CACHE_DIR=${TMAX_HF_CACHE_DIR:-${REPO_ROOT}/.cache/tmax-hf/hub}
    export TMAX_HF_EXTRACT_DIR=${TMAX_HF_EXTRACT_DIR:-${REPO_ROOT}/.cache/tmax-hf/tasks}
    export TMAX_ENROOT_IMAGE_CACHE=${TMAX_ENROOT_IMAGE_CACHE:-${REPO_ROOT}/.cache/tmax-enroot-images}
    mkdir -p "${TMAX_HF_CACHE_DIR}" "${TMAX_HF_EXTRACT_DIR}" "${TMAX_ENROOT_IMAGE_CACHE}"
    cd "${ENV_ROOT}"
    exec "${PYTHON}" -m openreward_env.server
    ;;
  swe_rebench)
    ENV_ROOT=${REPO_ROOT}/external/swe-rebench-v2-openrewardenv
    SWE_REBENCH_SOURCE_REVISION=${SWE_REBENCH_SOURCE_REVISION:-035d99666931ae39395938f0fdf2deb1e1d2038f}
    require_source_revision "${ENV_ROOT}" "${SWE_REBENCH_SOURCE_REVISION}"
    PYTHON=${ENV_ROOT}/.venv-openreward/bin/python
    DATA_DIR=${SWE_REBENCH_DATA_DIR:-${REPO_ROOT}/.cache/swe-rebench-v2}
    [[ -x "${PYTHON}" ]] || {
      echo "ERROR: missing SWE-rebench runtime: ${PYTHON}" >&2
      exit 2
    }
    [[ -f "${DATA_DIR}/task_index.json" ]] || {
      echo "ERROR: missing SWE-rebench task index: ${DATA_DIR}/task_index.json" >&2
      exit 2
    }
    export OPENREWARD_PORT=${OPENREWARD_SWE_REBENCH_PORT:-8084}
    export OPENREWARD_REWARD_MODE=${SWE_REBENCH_REWARD_MODE:-binary}
    export DATA_DIR
    export SWE_SANDBOX_RUNTIME=enroot
    export SWE_ENROOT_IMAGE_CACHE=${SWE_ENROOT_IMAGE_CACHE:-${REPO_ROOT}/.cache/swe-rebench-enroot-images}
    mkdir -p "${SWE_ENROOT_IMAGE_CACHE}"
    cd "${ENV_ROOT}"
    exec "${PYTHON}" server.py
    ;;
  *)
    echo "ERROR: unsupported OpenReward environment kind: ${KIND}" >&2
    exit 2
    ;;
esac
