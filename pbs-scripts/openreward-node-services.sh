#!/usr/bin/env bash
# Start one Toolathlon service and one host SWE server on one physical PBS node.
# One copy of this helper is held by one long-lived pbsdsh connection.

set -euo pipefail
umask 077

ABCI_ENV=${PLATOON_ABCI_ENV_FILE:-/home/aci18912it/rasp/config/abci.env}
[[ -r "${ABCI_ENV}" ]] || {
    echo "[node-services] ERROR: cluster environment is unreadable: ${ABCI_ENV}" >&2
    exit 2
}
# shellcheck disable=SC1090
source "${ABCI_ENV}"

# abci.env is shared with the controller and may chain to secrets.env.  This
# host helper needs paths only: remove every accepted credential before it runs
# another command or starts either environment service.
for secret_name in \
    WANDB_API_KEY OPENAI_API_KEY OPENAI_BASE_URL \
    LITELLM_API_KEY LITELLM_BASE_URL RUBRIC_DEFAULT_LLM HF_TOKEN \
    AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN; do
    unset "${secret_name}"
done
unset secret_name

if command -v module >/dev/null 2>&1; then
    module load singularity-ce/4.4.1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TOOLATHLON_ENTRYPOINT=${PLATOON_REPO_ROOT}/plugins/openreward/scripts/openreward-toolathlon-resilient-entrypoint.sh
TOOLATHLON_STAGER=${PLATOON_REPO_ROOT}/plugins/openreward/scripts/openreward-toolathlon-stage-pgdata.sh
SWE_HELPER=${SCRIPT_DIR}/openreward-swe-singularity-server.sh
SWE_CLEANUP=${SCRIPT_DIR}/cleanup_swe_singularity_sessions.py
CLEANUP_RECEIPT_HELPER=${SCRIPT_DIR}/openreward-service-cleanup-receipt.py
RUNTIME=${PLATOON_CONTAINER_RUNTIME:-singularity}
MKFS_EXT3=${PLATOON_MKFS_EXT3_BIN:-/usr/sbin/mkfs.ext3}
TOOLATHLON_PORT=${OPENREWARD_PORT:-8082}
SWE_PORT=${OPENREWARD_SWE_REBENCH_PORT:-8084}
TOOLATHLON_SHA256=${OPENREWARD_TOOLATHLON_SIF_SHA256:-e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267}
TOOLATHLON_SOURCE_REVISION=${OPENREWARD_TOOLATHLON_SOURCE_REVISION:-}
TOOLATHLON_SOURCE_URL=${OPENREWARD_TOOLATHLON_SOURCE_URL:-}
TOOLATHLON_OCI_AMD64_DIGEST=${OPENREWARD_TOOLATHLON_OCI_AMD64_DIGEST:-}
LOCAL_ROOT=${PLATOON_NODE_LOCAL_ROOT:?PLATOON_NODE_LOCAL_ROOT is required}
TOOLATHLON_ROOT=${LOCAL_ROOT}/toolathlon
TOOLATHLON_STATE=${TOOLATHLON_ROOT}/state
TOOLATHLON_OVERLAY=${TOOLATHLON_ROOT}/rootfs-overlay.img
# Sparse allocation on ABCI's 14-TB node-local NVMe.  Toolathlon creates
# session-scoped MCP environments during a long job; 16 GiB avoids reproducing
# the old small-overlay exhaustion while consuming blocks only as written.
TOOLATHLON_OVERLAY_SIZE_GB=${OPENREWARD_TOOLATHLON_OVERLAY_SIZE_GB:-16}
STAGE_TOOLATHLON_SIF=${PLATOON_STAGE_TOOLATHLON_SIF:-1}
STAGE_TRAINING_SIF=${PLATOON_STAGE_TRAINING_SIF:-0}
TRAINING_SIF_SHA256=${PLATOON_TRAINING_SIF_SHA256:-}
CLEANUP_RECEIPT=${PLATOON_NODE_SERVICES_CLEANUP_RECEIPT:-}
EXPECTED_HOST=${PLATOON_NODE_SERVICES_EXPECTED_HOST:-}
EXPECTED_JOB_ID=${PLATOON_NODE_SERVICES_EXPECTED_JOB_ID:-}

die() {
    echo "[node-services] ERROR: $*" >&2
    exit 2
}

[[ -n "${PBS_JOBID:-}" ]] || die "this helper must run inside PBS"
[[ -d "${PBS_LOCALDIR:-}" ]] || die "PBS_LOCALDIR is unavailable"
[[ -x "${TOOLATHLON_ENTRYPOINT}" ]] || die "Toolathlon entrypoint is missing"
[[ -x "${TOOLATHLON_STAGER}" ]] || die "Toolathlon PGDATA stager is missing"
[[ -x "${SWE_HELPER}" ]] || die "SWE helper is missing"
[[ -x "${SWE_CLEANUP}" ]] || die "SWE cleanup helper is missing"
[[ -r "${OPENREWARD_TOOLATHLON_SIF}" ]] || die "Toolathlon SIF is missing"
[[ "${MKFS_EXT3}" == /* ]] || \
    die "PLATOON_MKFS_EXT3_BIN must be an absolute path: ${MKFS_EXT3}"
[[ -x "${MKFS_EXT3}" ]] || die "mkfs.ext3 is missing or not executable: ${MKFS_EXT3}"
[[ "${STAGE_TOOLATHLON_SIF}" == 0 || "${STAGE_TOOLATHLON_SIF}" == 1 ]] || \
    die "PLATOON_STAGE_TOOLATHLON_SIF must be 0 or 1"
[[ "${STAGE_TRAINING_SIF}" == 0 || "${STAGE_TRAINING_SIF}" == 1 ]] || \
    die "PLATOON_STAGE_TRAINING_SIF must be 0 or 1"
if [[ "${STAGE_TRAINING_SIF}" == 1 ]]; then
    [[ "${TRAINING_SIF_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
        die "PLATOON_TRAINING_SIF_SHA256 is required when staging the training SIF"
    [[ -r "${PLATOON_TRAINING_SIF}" ]] || die "training SIF is missing"
fi
[[ "${TOOLATHLON_OVERLAY_SIZE_GB}" =~ ^[1-9][0-9]*$ ]] || \
    die "OPENREWARD_TOOLATHLON_OVERLAY_SIZE_GB must be a positive integer"
if [[ -n "${CLEANUP_RECEIPT}" ]]; then
    [[ -x "${CLEANUP_RECEIPT_HELPER}" ]] || die "service cleanup receipt helper is missing"
    [[ "${EXPECTED_JOB_ID}" == "${PBS_JOBID}" ]] || die "service cleanup receipt job mismatch"
    [[ -n "${EXPECTED_HOST}" && "${EXPECTED_HOST%%.*}" == "$(hostname -s)" ]] || \
        die "service cleanup receipt host mismatch"
    [[ "${CLEANUP_RECEIPT}" == /* && ! -e "${CLEANUP_RECEIPT}" && ! -L "${CLEANUP_RECEIPT}" ]] || \
        die "service cleanup receipt path is stale or indirect"
fi

actual_toolathlon_sha=$(sha256sum "${OPENREWARD_TOOLATHLON_SIF}" | awk '{print $1}')
[[ "${actual_toolathlon_sha}" == "${TOOLATHLON_SHA256}" ]] || \
    die "Toolathlon SIF digest mismatch: ${actual_toolathlon_sha}"
toolathlon_labels=$("${RUNTIME}" inspect --labels "${OPENREWARD_TOOLATHLON_SIF}")
if [[ -n "${TOOLATHLON_SOURCE_URL}" ]]; then
    grep -Fq "org.opencontainers.image.source: ${TOOLATHLON_SOURCE_URL}" \
        <<<"${toolathlon_labels}" || die "Toolathlon SIF source URL mismatch"
fi
if [[ -n "${TOOLATHLON_SOURCE_REVISION}" ]]; then
    grep -Fq "org.opencontainers.image.revision: ${TOOLATHLON_SOURCE_REVISION}" \
        <<<"${toolathlon_labels}" || die "Toolathlon SIF source revision mismatch"
fi
if [[ -n "${TOOLATHLON_OCI_AMD64_DIGEST}" ]]; then
    grep -Fq "org.label-schema.usage.singularity.deffile.from: ghcr.io/apga/toolathlon-gym@${TOOLATHLON_OCI_AMD64_DIGEST}" \
        <<<"${toolathlon_labels}" || die "Toolathlon SIF OCI digest mismatch"
fi
unset toolathlon_labels

mkdir -p \
    "${LOCAL_ROOT}/images" \
    "${LOCAL_ROOT}/worker-home" \
    "${LOCAL_ROOT}/worker-tmp" \
    "${TOOLATHLON_STATE}" \
    "${SINGULARITY_CACHEDIR}" \
    "${SINGULARITY_TMPDIR}" \
    "${SWE_APPTAINER_SESSION_ROOT}"
chmod 0700 \
    "${LOCAL_ROOT}" \
    "${LOCAL_ROOT}/images" \
    "${LOCAL_ROOT}/worker-home" \
    "${SINGULARITY_CACHEDIR}" \
    "${SINGULARITY_TMPDIR}" \
    "${SWE_APPTAINER_SESSION_ROOT}"
chmod 1777 "${LOCAL_ROOT}/worker-tmp"

toolathlon_sif=${OPENREWARD_TOOLATHLON_SIF}
if [[ "${STAGE_TOOLATHLON_SIF}" == 1 ]]; then
    toolathlon_sif=${LOCAL_ROOT}/images/openreward-toolathlon.sif
    if [[ ! -f "${toolathlon_sif}" ]] || \
        [[ "$(sha256sum "${toolathlon_sif}" | awk '{print $1}')" != "${TOOLATHLON_SHA256}" ]]; then
        stage_tmp=${toolathlon_sif}.partial.$$
        cp "${OPENREWARD_TOOLATHLON_SIF}" "${stage_tmp}"
        [[ "$(sha256sum "${stage_tmp}" | awk '{print $1}')" == "${TOOLATHLON_SHA256}" ]] || \
            die "node-local Toolathlon SIF changed while staging"
        chmod 0400 "${stage_tmp}"
        mv "${stage_tmp}" "${toolathlon_sif}"
    fi
fi

if [[ "${STAGE_TRAINING_SIF}" == 1 ]]; then
    local_training_sif=${LOCAL_ROOT}/images/platoon-training.sif
    if [[ ! -f "${local_training_sif}" ]] || \
        [[ "$(sha256sum "${local_training_sif}" | awk '{print $1}')" != "${TRAINING_SIF_SHA256}" ]]; then
        training_stage_tmp=${local_training_sif}.partial.$$
        cp "${PLATOON_TRAINING_SIF}" "${training_stage_tmp}"
        [[ "$(sha256sum "${training_stage_tmp}" | awk '{print $1}')" == "${TRAINING_SIF_SHA256}" ]] || \
            die "node-local training SIF changed while staging"
        chmod 0400 "${training_stage_tmp}"
        mv "${training_stage_tmp}" "${local_training_sif}"
    fi
    echo "[node-services] staged training SIF sha256=${TRAINING_SIF_SHA256}"
fi

pgdata_container=/mnt/state/postgresql/14/main
echo "[node-services] staging Toolathlon PGDATA on $(hostname)"
"${RUNTIME}" exec --fakeroot --cleanenv --no-home \
    --bind "${TOOLATHLON_STATE}:/mnt/state,${TOOLATHLON_STAGER}:/app/openreward-toolathlon-stage-pgdata.sh:ro" \
    "${toolathlon_sif}" \
    /app/openreward-toolathlon-stage-pgdata.sh "${pgdata_container}"

overlay_layout=${TOOLATHLON_ROOT}/overlay-layout
mkdir -p "${overlay_layout}/upper" "${overlay_layout}/work"
truncate -s "${TOOLATHLON_OVERLAY_SIZE_GB}G" "${TOOLATHLON_OVERLAY}"
"${MKFS_EXT3}" -q -F -d "${overlay_layout}" "${TOOLATHLON_OVERLAY}"
rmdir "${overlay_layout}/upper" "${overlay_layout}/work" "${overlay_layout}"

toolathlon_pid=
swe_pid=
cleanup_started=0

cleanup() {
    local primary_status=${1:-$?}
    local status=${primary_status}
    local pid deadline forced_kill=0 cleanup_ok=1 janitor_log
    if ((cleanup_started)); then
        return
    fi
    cleanup_started=1
    trap - EXIT INT TERM
    for pid in "${toolathlon_pid}" "${swe_pid}"; do
        if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
            echo "[node-services] ERROR: a service child was already dead before requested cleanup" >&2
            cleanup_ok=0
            status=1
        elif ! kill -TERM "${pid}" 2>/dev/null; then
            echo "[node-services] ERROR: could not request service-child cleanup" >&2
            cleanup_ok=0
            status=1
        fi
    done
    deadline=$((SECONDS + 90))
    for pid in "${toolathlon_pid}" "${swe_pid}"; do
        while [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && ((SECONDS < deadline)); do
            sleep 0.2
        done
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -KILL "${pid}" 2>/dev/null || true
            forced_kill=1
            cleanup_ok=0
            status=1
        fi
        [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
    done
    if ! /usr/bin/python3 -I "${CLEANUP_RECEIPT_HELPER}" assert-tcp-closed \
        --host 127.0.0.1 --port "${TOOLATHLON_PORT}" --port "${SWE_PORT}" --timeout-seconds 2; then
        echo "[node-services] ERROR: TCP endpoint closure could not be attested" >&2
        cleanup_ok=0
        status=1
    fi
    janitor_log=${LOCAL_ROOT}/node-services-final-swe-cleanup.log
    if ! /usr/bin/python3 "${SWE_CLEANUP}" \
        --session-root "${SWE_APPTAINER_SESSION_ROOT}" \
        --runtime "${RUNTIME}" \
        --stop-grace-seconds 10 >"${janitor_log}" 2>&1 || \
        ! grep -Fxq swe_singularity_sessions_cleaned=0 "${janitor_log}" || \
        ! /usr/bin/python3 -I "${CLEANUP_RECEIPT_HELPER}" assert-empty-swe \
            --session-root "${SWE_APPTAINER_SESSION_ROOT}"; then
        echo "[node-services] ERROR: SWE sessions were not already clean at supervisor exit" >&2
        tail -n 40 "${janitor_log}" >&2 || true
        cleanup_ok=0
        status=1
    fi
    if ((cleanup_ok)) && [[ -n "${CLEANUP_RECEIPT}" ]]; then
        if ! /usr/bin/python3 -I "${CLEANUP_RECEIPT_HELPER}" publish \
            --receipt "${CLEANUP_RECEIPT}" \
            --pbs-job-id "${EXPECTED_JOB_ID}" \
            --host "${EXPECTED_HOST}" \
            --primary-status "${primary_status}"; then
            echo "[node-services] ERROR: could not publish the cleanup receipt" >&2
            status=1
        fi
    fi
    exit "${status}"
}

trap cleanup EXIT
trap 'cleanup 143' TERM
trap 'cleanup 130' INT

mkdir -p \
    "${TOOLATHLON_STATE}/cache/uv" \
    "${TOOLATHLON_STATE}/cache/xdg" \
    "${TOOLATHLON_STATE}/home" \
    "${TOOLATHLON_STATE}/tmp"

echo "[node-services] starting Toolathlon host=$(hostname) port=${TOOLATHLON_PORT} image_sha256=${TOOLATHLON_SHA256}"
# The Toolathlon application uses PostgreSQL's standard localhost port; there
# is exactly one service per physical allocation node.
"${RUNTIME}" exec --cleanenv --no-home --overlay "${TOOLATHLON_OVERLAY}" \
    --bind "${TOOLATHLON_STATE}:/mnt/state,${TOOLATHLON_ENTRYPOINT}:/app/openreward-toolathlon-resilient-entrypoint.sh:ro" \
    "${toolathlon_sif}" \
    env \
        HOME=/mnt/state/home \
        UV_CACHE_DIR=/mnt/state/cache/uv \
        XDG_CACHE_HOME=/mnt/state/cache/xdg \
        TMPDIR=/mnt/state/tmp \
        VIRTUAL_ENV=/opt/venv \
        PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
        OPENREWARD_RUNTIME_MODE=user \
        OPENREWARD_STATE_DIR=/mnt/state \
        OPENREWARD_PGDATA="${pgdata_container}" \
        OPENREWARD_PG_PORT="${OPENREWARD_TOOLATHLON_PG_PORT:-5432}" \
        OPENREWARD_PG_MAX_CONNECTIONS="${OPENREWARD_PG_MAX_CONNECTIONS:-512}" \
        OPENREWARD_PORT="${TOOLATHLON_PORT}" \
        OPENREWARD_WORKER_BASE_PORT="${OPENREWARD_WORKER_BASE_PORT:-28100}" \
        OPENREWARD_WORKERS="${OPENREWARD_WORKERS:-8}" \
        OPENREWARD_WORKER_PID_ISOLATION=0 \
        /app/openreward-toolathlon-resilient-entrypoint.sh &
toolathlon_pid=$!

echo "[node-services] starting SWE-rebench host=$(hostname) port=${SWE_PORT}"
export SWE_APPTAINER_MKFS_BIN=${MKFS_EXT3}
OPENREWARD_SWE_REBENCH_PORT=${SWE_PORT} "${SWE_HELPER}" &
swe_pid=$!

# A long-lived pbsdsh connection owns this supervisor.  If either pool dies,
# tear down its sibling and let the controller health monitor fail the job.
while kill -0 "${toolathlon_pid}" 2>/dev/null && kill -0 "${swe_pid}" 2>/dev/null; do
    sleep 2
done

status=1
if ! kill -0 "${toolathlon_pid}" 2>/dev/null; then
    wait "${toolathlon_pid}" || status=$?
    echo "[node-services] ERROR: Toolathlon exited status=${status}" >&2
else
    wait "${swe_pid}" || status=$?
    echo "[node-services] ERROR: SWE-rebench exited status=${status}" >&2
fi
((status == 0)) && status=1
exit "${status}"
