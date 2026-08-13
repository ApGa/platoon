#!/usr/bin/env bash
# Host-managed SWE-rebench OpenReward server for one ABCI node.  The Python
# server must remain outside a container because it launches persistent
# Singularity task instances itself.

set -euo pipefail
umask 077

ABCI_ENV=${PLATOON_ABCI_ENV_FILE:-/home/aci18912it/rasp/config/abci.env}
[[ -r "${ABCI_ENV}" ]] || {
    echo "[swe-server] ERROR: cluster environment is unreadable: ${ABCI_ENV}" >&2
    exit 2
}
# shellcheck disable=SC1090
source "${ABCI_ENV}"

# This host-managed environment server never needs training/judging secrets.
for secret_name in \
    WANDB_API_KEY OPENAI_API_KEY OPENAI_BASE_URL \
    LITELLM_API_KEY LITELLM_BASE_URL RUBRIC_DEFAULT_LLM HF_TOKEN \
    AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN; do
    unset "${secret_name}"
done
unset secret_name

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_ROOT=${PLATOON_REPO_ROOT}/external/swe-rebench-v2-openrewardenv
CLEANUP_HELPER=${SCRIPT_DIR}/cleanup_swe_singularity_sessions.py
RUNTIME_VERIFY=${SCRIPT_DIR}/verify_openreward_v8_swe_env.py
VERIFY_PINNED_RUNTIME=${OPENREWARD_VERIFY_PINNED_SWE_RUNTIME:-0}
EXPECTED_REVISION=${SWE_REBENCH_SOURCE_REVISION:-538b5c5c9c529d3c78d5ef7fd3df7e54e7902ad7}
PYTHON=${ENV_ROOT}/.venv-openreward/bin/python
RUNTIME=${SWE_CONTAINER_RUNTIME_BIN:-$(command -v singularity || true)}
PORT=${OPENREWARD_SWE_REBENCH_PORT:-8084}
MAX_RESTARTS=${OPENREWARD_SWE_SERVER_MAX_RESTARTS:-5}
RESTART_RESET_SECS=${OPENREWARD_SWE_SERVER_RESTART_RESET_SECS:-300}
BACKOFF_MAX_SECS=${OPENREWARD_SWE_SERVER_BACKOFF_MAX_SECS:-30}
SERVER_SHUTDOWN_SECS=${OPENREWARD_SWE_SERVER_SHUTDOWN_SECS:-60}

die() {
    echo "[swe-server] ERROR: $*" >&2
    exit 2
}

require_uint() {
    local name=$1 value=$2 minimum=$3
    if [[ -z "${value}" || "${value}" == *[!0-9]* ]] || ((10#${value} < minimum)); then
        die "${name} must be an integer >= ${minimum}; got ${value@Q}"
    fi
}

require_uint OPENREWARD_SWE_REBENCH_PORT "${PORT}" 1
require_uint OPENREWARD_SWE_SERVER_MAX_RESTARTS "${MAX_RESTARTS}" 0
require_uint OPENREWARD_SWE_SERVER_RESTART_RESET_SECS "${RESTART_RESET_SECS}" 1
require_uint OPENREWARD_SWE_SERVER_BACKOFF_MAX_SECS "${BACKOFF_MAX_SECS}" 0
require_uint OPENREWARD_SWE_SERVER_SHUTDOWN_SECS "${SERVER_SHUTDOWN_SECS}" 1
((PORT <= 65535)) || die "OPENREWARD_SWE_REBENCH_PORT must be <= 65535"
[[ -x "${PYTHON}" ]] || die "SWE runtime is missing: ${PYTHON}"
[[ -x "${RUNTIME}" ]] || die "Singularity runtime is missing: ${RUNTIME}"
[[ -x "${CLEANUP_HELPER}" ]] || die "session cleanup helper is missing: ${CLEANUP_HELPER}"
[[ "${VERIFY_PINNED_RUNTIME}" == 0 || "${VERIFY_PINNED_RUNTIME}" == 1 ]] || \
    die "OPENREWARD_VERIFY_PINNED_SWE_RUNTIME must be 0 or 1"
if [[ "${VERIFY_PINNED_RUNTIME}" == 1 ]]; then
    [[ -x "${RUNTIME_VERIFY}" ]] || die "SWE runtime verifier is missing: ${RUNTIME_VERIFY}"
fi
[[ -f "${SWE_REBENCH_DATA_DIR}/task_index.json" ]] || \
    die "verified SWE task index is missing: ${SWE_REBENCH_DATA_DIR}/task_index.json"

actual_revision=$(git -C "${ENV_ROOT}" rev-parse HEAD)
[[ "${actual_revision}" == "${EXPECTED_REVISION}" ]] || \
    die "SWE source revision ${actual_revision} does not match ${EXPECTED_REVISION}"
[[ -z "$(git -C "${ENV_ROOT}" status --porcelain --untracked-files=all)" ]] || \
    die "SWE source worktree is dirty at ${EXPECTED_REVISION}"
if [[ "${VERIFY_PINNED_RUNTIME}" == 1 ]]; then
    "${RUNTIME_VERIFY}" --release-root "${PLATOON_REPO_ROOT}" || \
        die "SWE release runtime verification failed"
fi

export OPENREWARD_DISABLE_UPDATE_CHECK=1
export OPENHANDS_SUPPRESS_BANNER=1
export PYTHONDONTWRITEBYTECODE=1
export OPENREWARD_PORT=${PORT}
export OPENREWARD_HOST=0.0.0.0
export OPENREWARD_REWARD_MODE=${SWE_REBENCH_REWARD_MODE:-binary}
export DATA_DIR=${SWE_REBENCH_DATA_DIR}
export SWE_SANDBOX_RUNTIME=singularity
export SWE_CONTAINER_RUNTIME_BIN=${RUNTIME}
export SWE_APPTAINER_IMAGE_CACHE=${SWE_APPTAINER_IMAGE_CACHE}
export SWE_SINGULARITY_IMAGE_CACHE=${SWE_APPTAINER_IMAGE_CACHE}
export SWE_APPTAINER_SESSION_ROOT=${SWE_APPTAINER_SESSION_ROOT}
export SWE_SINGULARITY_SESSION_ROOT=${SWE_APPTAINER_SESSION_ROOT}
export SWE_APPTAINER_FAKEROOT=0
export SWE_APPTAINER_NETWORK=${SWE_APPTAINER_NETWORK:-host}
export SWE_APPTAINER_OVERLAY_SIZE_MB=${SWE_APPTAINER_OVERLAY_SIZE_MB:-8192}
export SWE_APPTAINER_OVERLAY_SPARSE=${SWE_APPTAINER_OVERLAY_SPARSE:-1}
export SWE_APPTAINER_PULL_TIMEOUT_SECONDS=${SWE_APPTAINER_PULL_TIMEOUT_SECONDS:-1800}
export SWE_SANDBOX_CREATE_TIMEOUT_SECONDS=${SWE_SANDBOX_CREATE_TIMEOUT_SECONDS:-1800}
export SWE_SANDBOX_COMMAND_TIMEOUT_SECONDS=${SWE_SANDBOX_COMMAND_TIMEOUT_SECONDS:-1800}
export SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR}
export SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR}
export APPTAINER_CACHEDIR=${SINGULARITY_CACHEDIR}
export APPTAINER_TMPDIR=${SINGULARITY_TMPDIR}

mkdir -p \
    "${SWE_APPTAINER_IMAGE_CACHE}" \
    "${SWE_APPTAINER_SESSION_ROOT}" \
    "${SINGULARITY_CACHEDIR}" \
    "${SINGULARITY_TMPDIR}"
chmod 0700 "${SWE_APPTAINER_SESSION_ROOT}" "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"

server_pid=
stop_requested=0

cleanup_sessions() {
    /usr/bin/python3 "${CLEANUP_HELPER}" \
        --session-root "${SWE_APPTAINER_SESSION_ROOT}" \
        --runtime "${RUNTIME}" \
        --stop-grace-seconds 10
}

stop_server() {
    local deadline
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill -TERM "${server_pid}" 2>/dev/null || true
        deadline=$((SECONDS + SERVER_SHUTDOWN_SECS))
        while kill -0 "${server_pid}" 2>/dev/null && ((SECONDS < deadline)); do
            sleep 0.2
        done
        if kill -0 "${server_pid}" 2>/dev/null; then
            kill -KILL "${server_pid}" 2>/dev/null || true
        fi
        wait "${server_pid}" 2>/dev/null || true
    fi
    server_pid=
}

request_stop() {
    stop_requested=1
    stop_server
}

final_cleanup() {
    local status=$?
    trap - EXIT INT TERM
    stop_server
    if ! cleanup_sessions; then
        echo "[swe-server] ERROR: an exact per-session Singularity cleanup failed" >&2
        status=1
    fi
    exit "${status}"
}

trap request_stop INT TERM
trap final_cleanup EXIT

failures=0
while ((stop_requested == 0)); do
    # Clean only UUID sessions whose lease, hostname, pid-file, runtime record,
    # process title, and Linux PID start time agree. No wildcard is used.
    cleanup_sessions
    started_at=${SECONDS}
    echo "[swe-server] starting revision=${actual_revision} host=$(hostname) port=${PORT} runtime=${RUNTIME} network=${SWE_APPTAINER_NETWORK}"
    echo "[swe-server] note: Singularity host-network sessions have no cgroup resource enforcement; concurrency must pass the migration smoke gate"
    (
        cd "${ENV_ROOT}"
        exec "${PYTHON}" server.py
    ) &
    server_pid=$!

    set +e
    wait "${server_pid}"
    status=$?
    set -e
    server_pid=
    ((stop_requested)) && break

    runtime_seconds=$((SECONDS - started_at))
    if ((runtime_seconds >= RESTART_RESET_SECS)); then
        failures=0
    fi
    failures=$((failures + 1))
    if ((failures > MAX_RESTARTS)); then
        echo "[swe-server] ERROR: server exhausted restart budget after status=${status}" >&2
        ((status == 0)) && status=1
        exit "${status}"
    fi
    delay=$((1 << (failures - 1)))
    ((delay > BACKOFF_MAX_SECS)) && delay=${BACKOFF_MAX_SECS}
    echo "[swe-server] WARNING: server exited status=${status}; restart ${failures}/${MAX_RESTARTS} in ${delay}s" >&2
    sleep "${delay}"
done

exit 0
