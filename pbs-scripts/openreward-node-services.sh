#!/usr/bin/env bash
# Start one Toolathlon service and, in the default mixed mode, one host SWE
# server on one physical PBS node. One copy of this helper is held by one
# long-lived pbsdsh connection.

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
MEMORY_GUARD=${SCRIPT_DIR}/openreward-node-memory-guard.py
CLEANUP_RECEIPT_HELPER=${SCRIPT_DIR}/openreward-service-cleanup-receipt.py
SHUTDOWN_REQUEST_HELPER=${SCRIPT_DIR}/openreward-service-shutdown-request.py
RUNTIME=${PLATOON_CONTAINER_RUNTIME:-singularity}
MKFS_EXT3=${PLATOON_MKFS_EXT3_BIN:-/usr/sbin/mkfs.ext3}
TOOLATHLON_PORT=${OPENREWARD_PORT:-8082}
SWE_PORT=${OPENREWARD_SWE_REBENCH_PORT:-8084}
TOOLATHLON_SHA256=${OPENREWARD_TOOLATHLON_SIF_SHA256:-e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267}
TOOLATHLON_SOURCE_REVISION=${OPENREWARD_TOOLATHLON_SOURCE_REVISION:-}
TOOLATHLON_SOURCE_URL=${OPENREWARD_TOOLATHLON_SOURCE_URL:-}
TOOLATHLON_OCI_AMD64_DIGEST=${OPENREWARD_TOOLATHLON_OCI_AMD64_DIGEST:-}
TOOLATHLON_BASE_SIF_SHA256=${OPENREWARD_TOOLATHLON_BASE_SIF_SHA256:-}
TOOLATHLON_PORTABILITY_PATCH_SHA256=${OPENREWARD_TOOLATHLON_PORTABILITY_PATCH_SHA256:-}
TOOLATHLON_PLAYWRIGHT_BROWSERS_PATH=${OPENREWARD_TOOLATHLON_PLAYWRIGHT_BROWSERS_PATH:-}
EXPECTED_CONTAINER_RUNTIME_PATH=/usr/bin/singularity
EXPECTED_CONTAINER_RUNTIME_SHA256=085a757f172dcabda66b44d086913314d6aa574055a8fea90523be82b770cec6
EXPECTED_CONTAINER_RUNTIME_VERSION=4.4.1-1.el9
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
SHUTDOWN_REQUEST=${PLATOON_NODE_SERVICES_SHUTDOWN_REQUEST:-}
EXPECTED_HOST=${PLATOON_NODE_SERVICES_EXPECTED_HOST:-}
EXPECTED_JOB_ID=${PLATOON_NODE_SERVICES_EXPECTED_JOB_ID:-}
ENVIRONMENT_SET=${OPENREWARD_ENVIRONMENT_SET:-toolathlon_swe}
ENVIRONMENT_SET_BINDING=${OPENREWARD_ENVIRONMENT_SET_BINDING:-}
REWARD_MODE=${OPENREWARD_REWARD_MODE:-}
EXPECTED_NNODES=${OPENREWARD_EXPECTED_NNODES:-}
EXPECTED_RECURSIVE=${OPENREWARD_EXPECTED_RECURSIVE:-}
EXPECTED_PTC=${OPENREWARD_EXPECTED_PTC:-}
EXPECTED_EXPERIMENT=${OPENREWARD_EXPECTED_EXPERIMENT_NAME:-}
EXPECTED_TRIAL=${OPENREWARD_EXPECTED_TRIAL_NAME:-}
NODE_MEMORY_GUARD_ENABLED=${OPENREWARD_NODE_MEMORY_GUARD_ENABLED:-1}
NODE_MEMORY_GUARD_DISABLED_BINDING=${OPENREWARD_NODE_MEMORY_GUARD_DISABLED_BINDING:-}
NODE_MEMORY_LIMIT_BYTES=${OPENREWARD_NODE_MEMORY_LIMIT_BYTES:-2061584302080}
NODE_MEMORY_TRIGGER_PERCENT=${OPENREWARD_NODE_MEMORY_TRIGGER_PERCENT:-85}
NODE_MEMORY_FALLBACK_AVAILABLE_BYTES=${OPENREWARD_NODE_MEMORY_FALLBACK_AVAILABLE_BYTES:-274877906944}
NODE_MEMORY_POLL_SECONDS=${OPENREWARD_NODE_MEMORY_POLL_SECONDS:-2}
NODE_MEMORY_TELEMETRY_SECONDS=${OPENREWARD_NODE_MEMORY_TELEMETRY_SECONDS:-60}

die() {
    echo "[node-services] ERROR: $*" >&2
    exit 2
}

[[ -n "${PBS_JOBID:-}" ]] || die "this helper must run inside PBS"
[[ -d "${PBS_LOCALDIR:-}" ]] || die "PBS_LOCALDIR is unavailable"
# BEGIN OPENREWARD_ENVIRONMENT_SET_SUPERVISOR_AUTHORIZATION
case "${ENVIRONMENT_SET}" in
    toolathlon_swe)
        [[ -z "${ENVIRONMENT_SET_BINDING}" ]] || \
            die "mixed environment services must not carry a reduced-service binding"
        ;;
    toolathlon_only)
        case "${ENVIRONMENT_SET_BINDING}" in
            qwen36-32-v27-rootprop64)
                [[ "${EXPECTED_NNODES}:${EXPECTED_RECURSIVE}:${EXPECTED_PTC}:${OPENREWARD_MODEL_FAMILY:-}" == 32:1:1:qwen36 && \
                    "${EXPECTED_EXPERIMENT}" == abci-or-ta32-q36-cp2-rootprop-v27 && \
                    "${EXPECTED_TRIAL}" == abci-ta32-rec-q36m-cp2-c12-d2-rootprop-v27-t0 ]] || \
                    die "Toolathlon-only root-propagation supervisor binding mismatch"
                ;;
            qwen36-32-v27-behavior64)
                [[ "${EXPECTED_NNODES}:${EXPECTED_RECURSIVE}:${EXPECTED_PTC}:${OPENREWARD_MODEL_FAMILY:-}" == 32:1:1:qwen36 && \
                    "${EXPECTED_EXPERIMENT}" == abci-or-ta32-q36-cp2-behavior-gate-v27 && \
                    "${EXPECTED_TRIAL}" == abci-ta32-rec-q36m-cp2-c12-d2-behavior-gate-v27-t0 ]] || \
                    die "Toolathlon-only Qwen3.6 behavior-gate supervisor binding mismatch"
                ;;
            qwen36-32-v33-behavior-usage64)
                [[ "${EXPECTED_NNODES}:${EXPECTED_RECURSIVE}:${EXPECTED_PTC}:${OPENREWARD_MODEL_FAMILY:-}" == 32:1:1:qwen36 && \
                    "${EXPECTED_EXPERIMENT}" == abci-or-ta32-q36-cp2-behavior-usage-gate-v33 && \
                    "${EXPECTED_TRIAL}" == abci-ta32-rec-q36m-cp2-c12-d2-behavior-usage-gate-v33-t0 ]] || \
                    die "Toolathlon-only Qwen3.6 behavior/usage-gate supervisor binding mismatch"
                ;;
            qwen36-15-v34-behavior-usage-partial64)
                [[ "${EXPECTED_NNODES}:${EXPECTED_RECURSIVE}:${EXPECTED_PTC}:${OPENREWARD_MODEL_FAMILY:-}" == 15:1:1:qwen36 && \
                    "${EXPECTED_EXPERIMENT}" == abci-or-ta15-q36-cp1-behavior-usage-partial-v34 && \
                    "${EXPECTED_TRIAL}" == abci-ta15-rec-q36m-cp1-c24-s3-d2-behavior-usage-partial-v34-t0 ]] || \
                    die "Toolathlon-only optimized Qwen3.6 behavior/usage-gate supervisor binding mismatch"
                ;;
            qwen36-32-v31-behavior-gs13)
                [[ "${EXPECTED_NNODES}:${EXPECTED_RECURSIVE}:${EXPECTED_PTC}:${OPENREWARD_MODEL_FAMILY:-}" == 32:1:1:qwen36 && \
                    "${EXPECTED_EXPERIMENT}" == abci-or-ta32-q36-cp2-behavior-gate-v27 && \
                    "${EXPECTED_TRIAL}" == abci-ta32-rec-q36m-cp2-c12-d2-behavior-gate-v27-t0 ]] || \
                    die "Toolathlon-only Qwen3.6 behavior-gate resume supervisor binding mismatch"
                ;;
            qwen36-32-v32-rootprop-gs13)
                [[ "${EXPECTED_NNODES}:${EXPECTED_RECURSIVE}:${EXPECTED_PTC}:${OPENREWARD_MODEL_FAMILY:-}" == 32:1:1:qwen36 && \
                    "${EXPECTED_EXPERIMENT}" == abci-or-ta32-q36-cp2-rootprop-v27 && \
                    "${EXPECTED_TRIAL}" == abci-ta32-rec-q36m-cp2-c12-d2-rootprop-v27-t0 ]] || \
                    die "Toolathlon-only Qwen3.6 root-propagated resume supervisor binding mismatch"
                ;;
            qwen35-16-v27-behavior64)
                [[ "${EXPECTED_NNODES}:${EXPECTED_RECURSIVE}:${EXPECTED_PTC}:${OPENREWARD_MODEL_FAMILY:-}" == 16:1:1:qwen35 && \
                    "${EXPECTED_EXPERIMENT}" == abci-or-ta16-q35-9b-dp-behavior-v27 && \
                    "${EXPECTED_TRIAL}" == abci-ta16-rec-q359b-dp-c12-d2-behavior-v27-t0 ]] || \
                    die "Toolathlon-only Qwen3.5 behavior-gate supervisor binding mismatch"
                ;;
            qwen35-16-v28-cp2-behavior64)
                [[ "${EXPECTED_NNODES}:${EXPECTED_RECURSIVE}:${EXPECTED_PTC}:${OPENREWARD_MODEL_FAMILY:-}" == 16:1:1:qwen35 && \
                    "${EXPECTED_EXPERIMENT}" == abci-or-ta16-q35-9b-cp2-behavior-v28 && \
                    "${EXPECTED_TRIAL}" == abci-ta16-rec-q359b-cp2-c12-d2-behavior-v28-t0 ]] || \
                    die "Toolathlon-only Qwen3.5 CP2 behavior-gate supervisor binding mismatch"
                ;;
            qwen35-16-v30-tp2cp2-behavior64)
                [[ "${EXPECTED_NNODES}:${EXPECTED_RECURSIVE}:${EXPECTED_PTC}:${OPENREWARD_MODEL_FAMILY:-}" == 16:1:1:qwen35 && \
                    "${EXPECTED_EXPERIMENT}" == abci-or-ta16-q35-9b-tp2cp2-behavior-v30 && \
                    "${EXPECTED_TRIAL}" == abci-ta16-rec-q359b-tp2cp2-c12-d2-behavior-v30-t0 ]] || \
                    die "Toolathlon-only Qwen3.5 TP2/CP2 behavior-gate supervisor binding mismatch"
                ;;
            *)
                die "Toolathlon-only supervisor mode lacks an authorized fresh binding"
                ;;
        esac
        unset OPENREWARD_SESSION_URLS_SWE_REBENCH OPENREWARD_SWE_REBENCH_PORT
        ;;
    *)
        die "OPENREWARD_ENVIRONMENT_SET must be toolathlon_swe or toolathlon_only"
        ;;
esac
# END OPENREWARD_ENVIRONMENT_SET_SUPERVISOR_AUTHORIZATION
[[ "${REWARD_MODE}" == partial ]] || \
    die "OPENREWARD_REWARD_MODE must be exactly partial for Toolathlon services"
[[ -x "${TOOLATHLON_ENTRYPOINT}" ]] || die "Toolathlon entrypoint is missing"
[[ -x "${TOOLATHLON_STAGER}" ]] || die "Toolathlon PGDATA stager is missing"
if [[ "${ENVIRONMENT_SET}" == toolathlon_swe ]]; then
    [[ -x "${SWE_HELPER}" ]] || die "SWE helper is missing"
    [[ -x "${SWE_CLEANUP}" ]] || die "SWE cleanup helper is missing"
fi
[[ -x "${MEMORY_GUARD}" ]] || die "node memory guard is missing"
[[ -r "${OPENREWARD_TOOLATHLON_SIF}" ]] || die "Toolathlon SIF is missing"
[[ ( "${NODE_MEMORY_GUARD_ENABLED}" == 0 || "${NODE_MEMORY_GUARD_ENABLED}" == 1 ) && \
    "${NODE_MEMORY_LIMIT_BYTES}" == 2061584302080 && \
    "${NODE_MEMORY_TRIGGER_PERCENT}" == 85 && \
    "${NODE_MEMORY_FALLBACK_AVAILABLE_BYTES}" == 274877906944 && \
    "${NODE_MEMORY_POLL_SECONDS}" == 2 && \
    "${NODE_MEMORY_TELEMETRY_SECONDS}" == 60 ]] || \
    die "node memory guard policy must use the exact enabled-or-disabled PBS service-node contract"
if [[ "${NODE_MEMORY_GUARD_ENABLED}" == 0 ]]; then
    case "${NODE_MEMORY_GUARD_DISABLED_BINDING}" in
        qwen38-32-v22-gs10|qwen36-32-v22-depth2|qwen38-16-v25-gs12|qwen36-32-v26-fresh-d2-noeff) ;;
        qwen36-32-v27-rootprop64|qwen36-32-v27-behavior64|qwen36-32-v33-behavior-usage64|qwen36-15-v34-behavior-usage-partial64|qwen36-32-v31-behavior-gs13|qwen36-32-v32-rootprop-gs13|qwen35-16-v27-behavior64|qwen35-16-v28-cp2-behavior64|qwen35-16-v30-tp2cp2-behavior64)
            [[ "${ENVIRONMENT_SET}" == toolathlon_only && \
                "${ENVIRONMENT_SET_BINDING}" == "${NODE_MEMORY_GUARD_DISABLED_BINDING}" ]] || \
                die "guard disablement requires its exact Toolathlon-only supervisor binding"
            ;;
        *) die "node memory guard disablement lacks an exact successor binding" ;;
    esac
elif [[ -n "${NODE_MEMORY_GUARD_DISABLED_BINDING}" ]]; then
    die "enabled node memory guard must not carry a disabled binding"
fi
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
if [[ -n "${CLEANUP_RECEIPT}" || -n "${SHUTDOWN_REQUEST}" ]]; then
    [[ -n "${CLEANUP_RECEIPT}" && -n "${SHUTDOWN_REQUEST}" ]] || \
        die "cleanup receipt and shutdown request must be configured together"
    [[ -x "${CLEANUP_RECEIPT_HELPER}" ]] || die "service cleanup receipt helper is missing"
    [[ -x "${SHUTDOWN_REQUEST_HELPER}" ]] || die "service shutdown request helper is missing"
    [[ "${EXPECTED_JOB_ID}" == "${PBS_JOBID}" ]] || die "service cleanup receipt job mismatch"
    [[ -n "${EXPECTED_HOST}" && "${EXPECTED_HOST%%.*}" == "$(hostname -s)" ]] || \
        die "service cleanup receipt host mismatch"
    [[ "${CLEANUP_RECEIPT}" == /* && ! -e "${CLEANUP_RECEIPT}" && ! -L "${CLEANUP_RECEIPT}" ]] || \
        die "service cleanup receipt path is stale or indirect"
    [[ "${SHUTDOWN_REQUEST}" == /* && ! -e "${SHUTDOWN_REQUEST}" && ! -L "${SHUTDOWN_REQUEST}" ]] || \
        die "service shutdown request path is stale or indirect"
fi

if [[ -n "${TOOLATHLON_PORTABILITY_PATCH_SHA256}" ]]; then
    expected_node_local_root=${PBS_LOCALDIR}/platoon/${PBS_JOBID}
    [[ "${PLATOON_NODE_LOCAL_ROOT}" == "${expected_node_local_root}" && \
        "${LOCAL_ROOT}" == "${expected_node_local_root}" && \
        "${SINGULARITY_CACHEDIR:-}" == "${expected_node_local_root}/singularity-cache" && \
        "${SINGULARITY_TMPDIR:-}" == "${expected_node_local_root}/singularity-tmp" && \
        "${APPTAINER_CACHEDIR:-}" == "${SINGULARITY_CACHEDIR}" && \
        "${APPTAINER_TMPDIR:-}" == "${SINGULARITY_TMPDIR}" ]] || \
        die "derived v9 service paths do not match the canonical PBS job-local root"
    if [[ "${ENVIRONMENT_SET}" == toolathlon_swe ]]; then
        [[ "${SWE_APPTAINER_SESSION_ROOT:-}" == "${expected_node_local_root}/swe-sessions" && \
            "${SWE_SINGULARITY_SESSION_ROOT:-}" == "${SWE_APPTAINER_SESSION_ROOT}" ]] || \
            die "mixed SWE service paths do not match the canonical PBS job-local root"
    fi
    unset expected_node_local_root
fi

if [[ -n "${TOOLATHLON_PORTABILITY_PATCH_SHA256}" ]]; then
    runtime_uid=$(stat -c %u "${RUNTIME}")
    runtime_mode=$(stat -c %a "${RUNTIME}")
    [[ "${RUNTIME}" == "${EXPECTED_CONTAINER_RUNTIME_PATH}" && \
        "$(readlink -f "${RUNTIME}")" == "${EXPECTED_CONTAINER_RUNTIME_PATH}" && \
        -f "${RUNTIME}" && ! -L "${RUNTIME}" && -x "${RUNTIME}" && \
        ( "${runtime_uid}" == 0 || "${runtime_uid}" == 65534 ) && \
        $((8#${runtime_mode} & 8#022)) -eq 0 && \
        "$(sha256sum "${RUNTIME}" | awk '{print $1}')" == "${EXPECTED_CONTAINER_RUNTIME_SHA256}" && \
        "$("${RUNTIME}" version)" == "${EXPECTED_CONTAINER_RUNTIME_VERSION}" ]] || \
        die "derived v9 service requires the exact trusted SingularityCE runtime"
    unset runtime_uid runtime_mode
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
if [[ -n "${TOOLATHLON_BASE_SIF_SHA256}" || -n "${TOOLATHLON_PORTABILITY_PATCH_SHA256}" ]]; then
    [[ "${TOOLATHLON_BASE_SIF_SHA256}" =~ ^[0-9a-f]{64}$ && \
        "${TOOLATHLON_PORTABILITY_PATCH_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
        die "derived Toolathlon SIF requires exact base and portability SHA-256 identities"
    [[ -n "${TOOLATHLON_OCI_AMD64_DIGEST}" ]] || \
        die "derived Toolathlon SIF requires the exact base OCI digest"
    grep -Fq "org.openreward.toolathlon.base-sif-sha256: ${TOOLATHLON_BASE_SIF_SHA256}" \
        <<<"${toolathlon_labels}" || die "Toolathlon derived SIF base digest mismatch"
    grep -Fq "org.openreward.toolathlon.base-oci-digest: ${TOOLATHLON_OCI_AMD64_DIGEST}" \
        <<<"${toolathlon_labels}" || die "Toolathlon derived SIF base OCI digest mismatch"
    grep -Fq "org.openreward.toolathlon.pbs-portability: ${TOOLATHLON_PORTABILITY_PATCH_SHA256}" \
        <<<"${toolathlon_labels}" || die "Toolathlon derived SIF portability patch mismatch"
elif [[ -n "${TOOLATHLON_OCI_AMD64_DIGEST}" ]]; then
    grep -Fq "org.label-schema.usage.singularity.deffile.from: ghcr.io/apga/toolathlon-gym@${TOOLATHLON_OCI_AMD64_DIGEST}" \
        <<<"${toolathlon_labels}" || die "Toolathlon SIF OCI digest mismatch"
fi
unset toolathlon_labels

toolathlon_portable_env=()
if [[ -n "${TOOLATHLON_PORTABILITY_PATCH_SHA256}" ]]; then
    [[ "${TOOLATHLON_PLAYWRIGHT_BROWSERS_PATH}" == /opt/ms-playwright ]] || \
        die "derived Toolathlon SIF requires PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright"
    toolathlon_portable_env=(
        PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright
        UV_PYTHON_INSTALL_DIR=/opt/uv/python
        UV_PYTHON_PREFERENCE=only-managed
        UV_PYTHON_DOWNLOADS=never
    )
fi

node_service_directories=(
    "${LOCAL_ROOT}/images" \
    "${LOCAL_ROOT}/worker-home" \
    "${LOCAL_ROOT}/worker-tmp" \
    "${TOOLATHLON_STATE}" \
    "${SINGULARITY_CACHEDIR}" \
    "${SINGULARITY_TMPDIR}"
)
private_node_service_directories=(
    "${LOCAL_ROOT}" \
    "${LOCAL_ROOT}/images" \
    "${LOCAL_ROOT}/worker-home" \
    "${SINGULARITY_CACHEDIR}" \
    "${SINGULARITY_TMPDIR}"
)
if [[ "${ENVIRONMENT_SET}" == toolathlon_swe ]]; then
    node_service_directories+=("${SWE_APPTAINER_SESSION_ROOT}")
    private_node_service_directories+=("${SWE_APPTAINER_SESSION_ROOT}")
fi
mkdir -p "${node_service_directories[@]}"
chmod 0700 "${private_node_service_directories[@]}"
chmod 1777 "${LOCAL_ROOT}/worker-tmp"
unset node_service_directories private_node_service_directories

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
memory_guard_pid=
memory_guard_reaped=0
memory_guard_status=
cleanup_started=0
natural_shutdown_requested=0

cleanup() {
    local primary_status=${1:-$?}
    local status=${primary_status}
    local index name pid deadline kill_deadline wait_status elapsed child_forced_kill
    local guard_deadline guard_wait_status guard_forced_kill
    local forced_kill=0 cleanup_ok=1 janitor_log cleanup_begin
    local -a child_names=(toolathlon swe-rebench)
    local -a child_pids=("${toolathlon_pid}" "${swe_pid}")
    local -a closed_endpoint_args=(
        --host 127.0.0.1 --port "${TOOLATHLON_PORT}" --port "${SWE_PORT}"
    )
    if [[ "${ENVIRONMENT_SET}" == toolathlon_only ]]; then
        child_names=(toolathlon)
        child_pids=("${toolathlon_pid}")
        closed_endpoint_args=(--host 127.0.0.1 --port "${TOOLATHLON_PORT}")
    fi
    if ((cleanup_started)); then
        return
    fi
    cleanup_started=1
    trap - EXIT INT TERM
    cleanup_begin=${SECONDS}
    if [[ "${primary_status}" -eq 0 ]]; then
        if ((natural_shutdown_requested != 1)) || \
            ! /usr/bin/python3 -I "${SHUTDOWN_REQUEST_HELPER}" verify \
                --request "${SHUTDOWN_REQUEST}" \
                --pbs-job-id "${EXPECTED_JOB_ID}" --host "${EXPECTED_HOST}"; then
            echo "[node-services] ERROR: natural cleanup lacks an exact verified shutdown request" >&2
            cleanup_ok=0
            status=1
        fi
        if [[ "${NODE_MEMORY_GUARD_ENABLED}" == 1 ]] && \
            { [[ -z "${memory_guard_pid}" ]] || ! kill -0 "${memory_guard_pid}" 2>/dev/null; }; then
            echo "[node-services] ERROR: memory guard was already dead before natural cleanup" >&2
            cleanup_ok=0
            status=1
        fi
    fi
    for index in "${!child_pids[@]}"; do
        name=${child_names[index]}
        pid=${child_pids[index]}
        if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
            echo "[node-services] ERROR: service child ${name} pid=${pid:-absent} was already dead before requested cleanup" >&2
            cleanup_ok=0
            status=1
        elif ! kill -TERM "${pid}" 2>/dev/null; then
            echo "[node-services] ERROR: could not send TERM to service child ${name} pid=${pid}" >&2
            cleanup_ok=0
            status=1
        else
            echo "[node-services] cleanup-child name=${name} pid=${pid} TERM=sent elapsed_seconds=$((SECONDS - cleanup_begin))"
        fi
    done
    guard_wait_status=not-started
    guard_forced_kill=0
    if [[ -n "${memory_guard_pid}" ]]; then
        if kill -0 "${memory_guard_pid}" 2>/dev/null; then
            if kill -TERM "${memory_guard_pid}" 2>/dev/null; then
                echo "[node-services] cleanup-guard pid=${memory_guard_pid} TERM=sent elapsed_seconds=$((SECONDS - cleanup_begin))"
            else
                echo "[node-services] ERROR: could not send TERM to memory guard pid=${memory_guard_pid}" >&2
                cleanup_ok=0
                status=1
            fi
        fi
        guard_deadline=$((SECONDS + 10))
        while kill -0 "${memory_guard_pid}" 2>/dev/null && ((SECONDS < guard_deadline)); do
            sleep 0.2
        done
        if kill -0 "${memory_guard_pid}" 2>/dev/null; then
            kill -KILL "${memory_guard_pid}" 2>/dev/null || true
            guard_forced_kill=1
            cleanup_ok=0
            status=1
            kill_deadline=$((SECONDS + 10))
            while kill -0 "${memory_guard_pid}" 2>/dev/null && ((SECONDS < kill_deadline)); do
                sleep 0.2
            done
        fi
        if ((memory_guard_reaped)); then
            guard_wait_status=${memory_guard_status}
        elif ! kill -0 "${memory_guard_pid}" 2>/dev/null; then
            guard_wait_status=0
            wait "${memory_guard_pid}" 2>/dev/null || guard_wait_status=$?
            memory_guard_status=${guard_wait_status}
            memory_guard_reaped=1
        else
            echo "[node-services] ERROR: memory guard pid=${memory_guard_pid} remained live after bounded cleanup" >&2
            cleanup_ok=0
            status=1
        fi
        if [[ "${primary_status}" -eq 0 && "${guard_wait_status}" != 0 ]]; then
            echo "[node-services] ERROR: memory guard cleanup status was ${guard_wait_status}, expected 0" >&2
            cleanup_ok=0
            status=1
        fi
        echo "[node-services] cleanup-guard pid=${memory_guard_pid} exit_status=${guard_wait_status} forced_KILL=${guard_forced_kill}"
    fi
    deadline=$((SECONDS + 90))
    for index in "${!child_pids[@]}"; do
        name=${child_names[index]}
        pid=${child_pids[index]}
        child_forced_kill=0
        while [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && ((SECONDS < deadline)); do
            sleep 0.2
        done
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -KILL "${pid}" 2>/dev/null || true
            forced_kill=1
            child_forced_kill=1
            cleanup_ok=0
            status=1
            kill_deadline=$((SECONDS + 10))
            while kill -0 "${pid}" 2>/dev/null && ((SECONDS < kill_deadline)); do
                sleep 0.2
            done
        fi
        wait_status=unreaped
        if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
            wait_status=0
            wait "${pid}" 2>/dev/null || wait_status=$?
        elif [[ -n "${pid}" ]]; then
            echo "[node-services] ERROR: service child ${name} pid=${pid} remained live after bounded cleanup" >&2
            cleanup_ok=0
            status=1
        fi
        elapsed=$((SECONDS - cleanup_begin))
        echo "[node-services] cleanup-child name=${name} pid=${pid:-absent} TERM=requested elapsed_seconds=${elapsed} exit_status=${wait_status} forced_KILL=${child_forced_kill}"
    done
    if ! /usr/bin/python3 -I "${CLEANUP_RECEIPT_HELPER}" assert-tcp-closed \
        "${closed_endpoint_args[@]}" --timeout-seconds 2; then
        echo "[node-services] ERROR: TCP endpoint closure could not be attested" >&2
        cleanup_ok=0
        status=1
    fi
    if [[ "${ENVIRONMENT_SET}" == toolathlon_swe ]]; then
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

echo "[node-services] starting Toolathlon host=$(hostname) port=${TOOLATHLON_PORT} reward_mode=${REWARD_MODE} image_sha256=${TOOLATHLON_SHA256}"
if [[ "${NODE_MEMORY_GUARD_ENABLED}" == 1 ]]; then
    # The guard watches the aggregate PBS allocation cgroup when it can identify
    # that cgroup by the exact node memory limit.  Its only fallback is explicit
    # host MemAvailable telemetry.
    echo "[node-services] starting memory guard host=$(hostname) limit_bytes=${NODE_MEMORY_LIMIT_BYTES} trigger_percent=${NODE_MEMORY_TRIGGER_PERCENT}"
    /usr/bin/python3 -I "${MEMORY_GUARD}" \
        --node-limit-bytes "${NODE_MEMORY_LIMIT_BYTES}" \
        --trigger-percent "${NODE_MEMORY_TRIGGER_PERCENT}" \
        --fallback-available-bytes "${NODE_MEMORY_FALLBACK_AVAILABLE_BYTES}" \
        --poll-seconds "${NODE_MEMORY_POLL_SECONDS}" \
        --telemetry-seconds "${NODE_MEMORY_TELEMETRY_SECONDS}" &
    memory_guard_pid=$!
else
    echo "[node-services] memory guard disabled binding=${NODE_MEMORY_GUARD_DISABLED_BINDING} host=$(hostname)"
fi

# The Toolathlon application uses PostgreSQL's standard localhost port; there
# is exactly one service per physical allocation node.
"${RUNTIME}" exec --cleanenv --no-home --overlay "${TOOLATHLON_OVERLAY}" \
    --bind "${TOOLATHLON_STATE}:/mnt/state,${TOOLATHLON_ENTRYPOINT}:/app/openreward-toolathlon-resilient-entrypoint.sh:ro" \
    "${toolathlon_sif}" \
    env \
        HOME=/mnt/state/home \
        UV_CACHE_DIR=/mnt/state/cache/uv \
        "${toolathlon_portable_env[@]}" \
        XDG_CACHE_HOME=/mnt/state/cache/xdg \
        TMPDIR=/mnt/state/tmp \
        VIRTUAL_ENV=/opt/venv \
        PATH=/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
        OPENREWARD_RUNTIME_MODE=user \
        OPENREWARD_REWARD_MODE="${REWARD_MODE}" \
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

if [[ "${ENVIRONMENT_SET}" == toolathlon_swe ]]; then
    echo "[node-services] starting SWE-rebench host=$(hostname) port=${SWE_PORT}"
    export SWE_APPTAINER_MKFS_BIN=${MKFS_EXT3}
    OPENREWARD_SWE_REBENCH_PORT=${SWE_PORT} "${SWE_HELPER}" &
    swe_pid=$!
fi

# A long-lived pbsdsh connection owns this supervisor.  If either pool or an
# enabled memory guard dies, tear down the live children and fail the
# controller's node-service health check.
memory_guard_healthy() {
    [[ "${NODE_MEMORY_GUARD_ENABLED}" == 0 ]] || \
        { [[ -n "${memory_guard_pid}" ]] && kill -0 "${memory_guard_pid}" 2>/dev/null; }
}
service_children_healthy() {
    kill -0 "${toolathlon_pid}" 2>/dev/null && \
        { [[ "${ENVIRONMENT_SET}" == toolathlon_only ]] || \
            { [[ -n "${swe_pid}" ]] && kill -0 "${swe_pid}" 2>/dev/null; }; }
}
while service_children_healthy && memory_guard_healthy; do
    if [[ -e "${SHUTDOWN_REQUEST}" || -L "${SHUTDOWN_REQUEST}" ]]; then
        if ! /usr/bin/python3 -I "${SHUTDOWN_REQUEST_HELPER}" verify \
            --request "${SHUTDOWN_REQUEST}" \
            --pbs-job-id "${EXPECTED_JOB_ID}" --host "${EXPECTED_HOST}"; then
            echo "[node-services] ERROR: shutdown request failed exact job/host verification" >&2
            exit 2
        fi
        natural_shutdown_requested=1
        echo "[node-services] verified natural shutdown request job=${EXPECTED_JOB_ID} host=${EXPECTED_HOST}"
        exit 0
    fi
    sleep 2
done

status=1
if [[ "${NODE_MEMORY_GUARD_ENABLED}" == 1 ]] && ! kill -0 "${memory_guard_pid}" 2>/dev/null; then
    memory_guard_status=0
    wait "${memory_guard_pid}" || memory_guard_status=$?
    memory_guard_reaped=1
    status=${memory_guard_status}
    echo "[node-services] ERROR: node memory guard exited status=${status}" >&2
elif ! kill -0 "${toolathlon_pid}" 2>/dev/null; then
    wait "${toolathlon_pid}" || status=$?
    echo "[node-services] ERROR: Toolathlon exited status=${status}" >&2
elif [[ "${ENVIRONMENT_SET}" == toolathlon_swe ]]; then
    wait "${swe_pid}" || status=$?
    echo "[node-services] ERROR: SWE-rebench exited status=${status}" >&2
else
    echo "[node-services] ERROR: Toolathlon-only supervisor lost an unclassified child" >&2
fi
((status == 0)) && status=1
exit "${status}"
