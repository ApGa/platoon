#!/bin/bash

# Start one of the host-native OpenReward environments used by the mixed
# training launcher.  The Python server stays on the host because both TMax and
# SWE-rebench create writable Enroot task containers themselves.
#
# Both backends put every model-authored command in a session-private PID
# namespace. Keep the server and its restart supervisor in the job's original
# user namespace: Enroot's image importer needs the file capabilities on
# enroot-aufs2ovlfs to translate OCI whiteouts, and Linux intentionally drops
# those host capabilities inside a nested user namespace.

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

MAX_RESTARTS=${OPENREWARD_SUPPLEMENTAL_SERVER_MAX_RESTARTS:-20}
RESTART_RESET_SECS=${OPENREWARD_SUPPLEMENTAL_SERVER_RESTART_RESET_SECS:-300}
BACKOFF_INITIAL_SECS=${OPENREWARD_SUPPLEMENTAL_SERVER_RESTART_BACKOFF_INITIAL_SECS:-1}
BACKOFF_MAX_SECS=${OPENREWARD_SUPPLEMENTAL_SERVER_RESTART_BACKOFF_MAX_SECS:-30}

require_uint() {
  local name=$1
  local value=$2
  local minimum=$3
  if [[ -z "${value}" || "${value}" == *[!0-9]* ]] || ((10#${value} < minimum)); then
    echo "ERROR: ${name} must be an integer >= ${minimum}; got ${value@Q}." >&2
    exit 2
  fi
}

require_uint OPENREWARD_SUPPLEMENTAL_SERVER_MAX_RESTARTS "${MAX_RESTARTS}" 0
require_uint OPENREWARD_SUPPLEMENTAL_SERVER_RESTART_RESET_SECS "${RESTART_RESET_SECS}" 1
require_uint OPENREWARD_SUPPLEMENTAL_SERVER_RESTART_BACKOFF_INITIAL_SECS "${BACKOFF_INITIAL_SECS}" 0
require_uint OPENREWARD_SUPPLEMENTAL_SERVER_RESTART_BACKOFF_MAX_SECS "${BACKOFF_MAX_SECS}" 0

server_pid=
stop_requested=0

request_stop() {
  stop_requested=1
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM "${server_pid}" 2>/dev/null || true
  fi
}

trap request_stop TERM INT

restart_backoff() {
  local attempt=$1
  local delay=${BACKOFF_INITIAL_SECS}
  local n
  for ((n = 1; n < attempt; n++)); do
    if ((delay >= BACKOFF_MAX_SECS)); then
      break
    fi
    delay=$((delay * 2))
  done
  if ((delay > BACKOFF_MAX_SECS)); then
    delay=${BACKOFF_MAX_SECS}
  fi
  printf '%s' "${delay}"
}

supervise_server() {
  local failures=0
  local started_at
  local runtime
  local status
  local delay

  while ((stop_requested == 0)); do
    started_at=${SECONDS}
    "$@" &
    server_pid=$!
    set +e
    wait "${server_pid}"
    status=$?
    set -e

    if ((stop_requested)); then
      kill -TERM "${server_pid}" 2>/dev/null || true
      wait "${server_pid}" 2>/dev/null || true
      server_pid=
      return 0
    fi
    server_pid=

    runtime=$((SECONDS - started_at))
    if ((runtime >= RESTART_RESET_SECS)); then
      failures=0
    fi
    failures=$((failures + 1))
    if ((failures > MAX_RESTARTS)); then
      echo "ERROR: ${KIND} server exhausted its restart budget after status=${status}." >&2
      ((status == 0)) && status=1
      return "${status}"
    fi
    delay=$(restart_backoff "${failures}")
    echo "WARNING: ${KIND} server exited unexpectedly (status=${status}, runtime=${runtime}s); restart ${failures}/${MAX_RESTARTS} in ${delay}s." >&2
    sleep "${delay}"
  done
}

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
    TMAX_SOURCE_REVISION=${TMAX_SOURCE_REVISION:-b8436c80b29957ecc73fc49bf4da018132c6e952}
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
    SERVER_COMMAND=("${PYTHON}" -m openreward_env.server)
    ;;
  swe_rebench)
    ENV_ROOT=${REPO_ROOT}/external/swe-rebench-v2-openrewardenv
    SWE_REBENCH_SOURCE_REVISION=${SWE_REBENCH_SOURCE_REVISION:-25b14c06b9236c075a4ede25bff6979e5783bb09}
    require_source_revision "${ENV_ROOT}" "${SWE_REBENCH_SOURCE_REVISION}"
    PYTHON=${ENV_ROOT}/.venv-openreward/bin/python
    # Train on Prime Intellect's full verified-solvable subset by default. The
    # environment name remains stable; DATA_DIR selects its 6,272-task catalog.
    DATA_DIR=${SWE_REBENCH_DATA_DIR:-${REPO_ROOT}/.cache/swe-rebench-v2-filtered-verified}
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
    SERVER_COMMAND=("${PYTHON}" server.py)
    ;;
  *)
    echo "ERROR: unsupported OpenReward environment kind: ${KIND}" >&2
    exit 2
    ;;
esac

supervise_server "${SERVER_COMMAND[@]}"
