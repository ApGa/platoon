#!/usr/bin/env bash
# Common ABCI lifecycle for Toolathlon + SWE-rebench AReaL jobs.
#
# This script deliberately has no PBS header, continuation submission, deadline
# resubmission, or synthetic GPU keepalive.  Topology-specific .pbs wrappers own
# allocation requests and delegate one allocation to this lifecycle.

set -euo pipefail
umask 077

ABCI_ENV=${PLATOON_ABCI_ENV_FILE:-/home/aci18912it/rasp/config/abci.env}
SECRETS_ENV=${PLATOON_SECRETS_ENV_FILE:-/home/aci18912it/rasp/config/secrets.env}
[[ -r "${ABCI_ENV}" ]] || {
    echo "[pbs-openreward] ERROR: cluster environment is unreadable: ${ABCI_ENV}" >&2
    exit 2
}
# shellcheck disable=SC1090
source "${ABCI_ENV}"
export PLATOON_ABCI_ENV_FILE=${ABCI_ENV}

# abci.env currently chains to this file, but source the protected file
# explicitly as part of this launcher's contract as well.  Reject permissive or
# foreign ownership before reading it, and never print its contents.
[[ -r "${SECRETS_ENV}" ]] || {
    echo "[pbs-openreward] ERROR: protected secrets file is unreadable: ${SECRETS_ENV}" >&2
    exit 2
}
secrets_owner=$(stat -c %u "${SECRETS_ENV}")
secrets_mode=$(stat -c %a "${SECRETS_ENV}")
[[ "${secrets_owner}" == "$(id -u)" && "${secrets_mode}" == 600 ]] || {
    echo "[pbs-openreward] ERROR: secrets file must be owned by uid $(id -u) with mode 0600: ${SECRETS_ENV}" >&2
    exit 2
}
# shellcheck disable=SC1090
source "${SECRETS_ENV}"
unset secrets_owner secrets_mode

source /etc/profile.d/modules.sh
module load singularity-ce/4.4.1
module load cuda/12.9/12.9.1
module load hpcx/2.20

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OPENREWARD_DIR=${PLATOON_REPO_ROOT}/plugins/openreward
PREPARE_ENV=${SCRIPT_DIR}/prepare_openreward_env.sh
PREPARE_CONTROLLER_ENV=${SCRIPT_DIR}/prepare_openreward_controller_env.sh
NODE_SERVICES=${SCRIPT_DIR}/openreward-node-services.sh
SERVICE_CLEANUP_RECEIPT=${SCRIPT_DIR}/openreward-service-cleanup-receipt.py
SERVICE_SHUTDOWN_REQUEST=${SCRIPT_DIR}/openreward-service-shutdown-request.py
CONTROLLER_RUNTIME_RECEIPT_HELPER=${SCRIPT_DIR}/openreward-controller-runtime-receipt.py
SWE_CLEANUP=${SCRIPT_DIR}/cleanup_swe_singularity_sessions.py
NODE_LOCAL_INIT=${SCRIPT_DIR}/openreward-node-local-init.py
UPDATE_GATE=${SCRIPT_DIR}/openreward_update_gate.py
TRAIN_MODULE=platoon.openreward.train_scripts.areal.train_areal
EXPECTED_NNODES=${OPENREWARD_EXPECTED_NNODES:?OPENREWARD_EXPECTED_NNODES is required}
EXPECTED_RECURSIVE=${OPENREWARD_EXPECTED_RECURSIVE:?OPENREWARD_EXPECTED_RECURSIVE is required}
EXPECTED_PTC=${OPENREWARD_EXPECTED_PTC:-}
EXPECTED_EXPERIMENT=${OPENREWARD_EXPECTED_EXPERIMENT_NAME:?OPENREWARD_EXPECTED_EXPERIMENT_NAME is required}
EXPECTED_TRIAL=${OPENREWARD_EXPECTED_TRIAL_NAME:?OPENREWARD_EXPECTED_TRIAL_NAME is required}
CONFIG=${1:?usage: $0 CONFIG}
RUNTIME=${PLATOON_CONTAINER_RUNTIME:-singularity}
PBSDSH=${PLATOON_PBSDSH_BIN:-$(command -v pbsdsh || true)}
UV_BIN=${OPENREWARD_UV_BIN:-/home/aci18912it/.local/bin/uv}
CONTROLLER_PYTHON_BASE=${OPENREWARD_CONTROLLER_PYTHON:-/home/aci18912it/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12}
NODE_LOCAL_INIT_PYTHON=${PLATOON_NODE_LOCAL_INIT_PYTHON:-/usr/bin/python3}
TOOLATHLON_SHA256=${OPENREWARD_TOOLATHLON_SIF_SHA256:-e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267}
TOOLATHLON_SOURCE_REVISION=${OPENREWARD_TOOLATHLON_SOURCE_REVISION:-}
TOOLATHLON_SOURCE_URL=${OPENREWARD_TOOLATHLON_SOURCE_URL:-}
TOOLATHLON_OCI_AMD64_DIGEST=${OPENREWARD_TOOLATHLON_OCI_AMD64_DIGEST:-}
TOOLATHLON_BASE_SIF_SHA256=${OPENREWARD_TOOLATHLON_BASE_SIF_SHA256:-}
TOOLATHLON_PORTABILITY_PATCH_SHA256=${OPENREWARD_TOOLATHLON_PORTABILITY_PATCH_SHA256:-}
EXPECTED_CONTAINER_RUNTIME_PATH=/usr/bin/singularity
EXPECTED_CONTAINER_RUNTIME_SHA256=085a757f172dcabda66b44d086913314d6aa574055a8fea90523be82b770cec6
EXPECTED_CONTAINER_RUNTIME_VERSION=4.4.1-1.el9
ATTEST_CONTAINER_RUNTIME=0
APEX_PIN=6424da3b4faa6c8f062da4a48c424fff3f02d42d
TE_PIN=2.12.0
TE_COMMIT_PIN=5671fd3675906cda1ade26c24a65d3dedd88eb89
TRAINING_BASE_PIN=sha256:829404319e32cc15c1a1fa0e8a47b807f98154c6287079b0548d031f1af30882

die() {
    echo "[pbs-openreward] ERROR: $*" >&2
    exit 2
}

require_uint() {
    local name=$1 value=$2 minimum=$3
    if [[ -z "${value}" || "${value}" == *[!0-9]* ]] || ((10#${value} < minimum)); then
        die "${name} must be an integer >= ${minimum}; got ${value@Q}"
    fi
}

require_uint OPENREWARD_EXPECTED_NNODES "${EXPECTED_NNODES}" 1
[[ "${EXPECTED_RECURSIVE}" == 0 || "${EXPECTED_RECURSIVE}" == 1 ]] || \
    die "OPENREWARD_EXPECTED_RECURSIVE must be 0 or 1"
[[ -z "${EXPECTED_PTC}" || "${EXPECTED_PTC}" == 0 || "${EXPECTED_PTC}" == 1 ]] || \
    die "OPENREWARD_EXPECTED_PTC must be 0 or 1 when set"
[[ -n "${PBS_JOBID:-}" && -r "${PBS_NODEFILE:-}" ]] || \
    die "this lifecycle must run inside a PBS allocation with PBS_NODEFILE"
[[ -x "${PBSDSH}" ]] || die "pbsdsh is unavailable"
[[ -x "${RUNTIME}" ]] || command -v "${RUNTIME}" >/dev/null 2>&1 || \
    die "Singularity runtime is unavailable: ${RUNTIME}"
RUNTIME=$(command -v "${RUNTIME}")
if [[ -n "${TOOLATHLON_PORTABILITY_PATCH_SHA256}" ]]; then
    ATTEST_CONTAINER_RUNTIME=1
    runtime_uid=$(stat -c %u "${RUNTIME}")
    runtime_mode=$(stat -c %a "${RUNTIME}")
    [[ "${RUNTIME}" == "${EXPECTED_CONTAINER_RUNTIME_PATH}" && \
        "$(readlink -f "${RUNTIME}")" == "${EXPECTED_CONTAINER_RUNTIME_PATH}" && \
        -f "${RUNTIME}" && ! -L "${RUNTIME}" && -x "${RUNTIME}" && \
        ( "${runtime_uid}" == 0 || "${runtime_uid}" == 65534 ) && \
        $((8#${runtime_mode} & 8#022)) -eq 0 && \
        "$(sha256sum "${RUNTIME}" | awk '{print $1}')" == "${EXPECTED_CONTAINER_RUNTIME_SHA256}" && \
        "$("${RUNTIME}" version)" == "${EXPECTED_CONTAINER_RUNTIME_VERSION}" ]] || \
        die "derived v9 launch requires the exact trusted SingularityCE runtime"
    unset runtime_uid runtime_mode
fi
[[ -x "${UV_BIN}" ]] || die "uv is unavailable: ${UV_BIN}"
[[ -x "${CONTROLLER_PYTHON_BASE}" ]] || \
    die "pinned host-controller Python is unavailable: ${CONTROLLER_PYTHON_BASE}"
[[ "${NODE_LOCAL_INIT_PYTHON}" == /* && -x "${NODE_LOCAL_INIT_PYTHON}" ]] || \
    die "node-local initializer Python must be an absolute executable: ${NODE_LOCAL_INIT_PYTHON}"
for required_file in \
    "${PREPARE_ENV}" "${PREPARE_CONTROLLER_ENV}" "${NODE_SERVICES}" \
    "${SERVICE_CLEANUP_RECEIPT}" "${SERVICE_SHUTDOWN_REQUEST}" \
    "${CONTROLLER_RUNTIME_RECEIPT_HELPER}" "${SWE_CLEANUP}" \
    "${NODE_LOCAL_INIT}" "${UPDATE_GATE}"; do
    [[ -x "${required_file}" ]] || die "required helper is not executable: ${required_file}"
done
unset required_file

CONFIG=$(readlink -f "${CONFIG}")
[[ -f "${CONFIG}" ]] || die "config does not exist: ${CONFIG}"
config_experiment=$(sed -n 's/^experiment_name:[[:space:]]*//p' "${CONFIG}" | head -n 1)
config_trial=$(sed -n 's/^trial_name:[[:space:]]*//p' "${CONFIG}" | head -n 1)
[[ "${config_experiment}" == "${EXPECTED_EXPERIMENT}" ]] || \
    die "config experiment ${config_experiment@Q} does not match ${EXPECTED_EXPERIMENT@Q}"
[[ "${config_trial}" == "${EXPECTED_TRIAL}" ]] || \
    die "config trial ${config_trial@Q} does not match ${EXPECTED_TRIAL@Q}"
[[ "${config_trial}" == abci-* ]] || die "ABCI production trial must start with 'abci-'"
grep -q 'type:[[:space:]]*pbs_prealloc' "${CONFIG}" || \
    die "ABCI leaf config must select scheduler.type=pbs_prealloc"
grep -q 'fileroot:[[:space:]]*${oc.env:PLATOON_EXPERIMENT_ROOT}' "${CONFIG}" || \
    die "ABCI leaf config must interpolate PLATOON_EXPERIMENT_ROOT"
if [[ "${EXPECTED_RECURSIVE}" == 1 ]]; then
    grep -q 'enable_recursive_subagents:[[:space:]]*true' "${CONFIG}" || \
        die "recursive wrapper requires a recursive leaf config"
else
    grep -q 'enable_recursive_subagents:[[:space:]]*false' "${CONFIG}" || \
        die "nonrecursive wrapper requires a nonrecursive leaf config"
fi
if [[ "${EXPECTED_PTC}" == 1 ]]; then
    grep -q 'enable_programmatic_tool_calling:[[:space:]]*true' "${CONFIG}" || \
        die "PTC-enabled wrapper requires a literal PTC-enabled leaf config"
    grep -q 'programmatic_tool_calling_mode:[[:space:]]*orchestration_only' "${CONFIG}" || \
        die "PTC-enabled wrapper requires orchestration_only mode"
    grep -q 'programmatic_tool_calling_max_tool_calls_per_execution:[[:space:]]*1024' "${CONFIG}" || \
        die "PTC-enabled wrapper requires the 1024-call budget"
elif [[ "${EXPECTED_PTC}" == 0 ]]; then
    grep -q 'enable_programmatic_tool_calling:[[:space:]]*false' "${CONFIG}" || \
        die "no-PTC wrapper requires a literal no-PTC leaf config"
fi

[[ -f "${SWE_REBENCH_DATA_DIR}/task_index.json" ]] || \
    die "verified SWE task index is missing: ${SWE_REBENCH_DATA_DIR}/task_index.json"
for model_file in config.json tokenizer.json tokenizer_config.json model.safetensors.index.json; do
    [[ -f "${PLATOON_MODEL_SNAPSHOT}/${model_file}" ]] || \
        die "model snapshot is incomplete; missing ${model_file}"
done
model_shards=$(find "${PLATOON_MODEL_SNAPSHOT}" -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l)
[[ "${model_shards}" -eq 26 ]] || die "model snapshot has ${model_shards} shards; expected 26"

verify_sif_with_sidecar() {
    local image=$1 sidecar=${1}.sha256 expected actual
    [[ -r "${image}" ]] || die "training SIF is missing: ${image}"
    [[ -r "${sidecar}" ]] || die "training SIF digest sidecar is missing: ${sidecar}"
    expected=$(awk 'NF {print $1; exit}' "${sidecar}")
    [[ "${expected}" =~ ^[0-9a-f]{64}$ ]] || die "invalid digest sidecar: ${sidecar}"
    actual=$(sha256sum "${image}" | awk '{print $1}')
    [[ "${actual}" == "${expected}" ]] || die "training SIF digest mismatch"
    "${RUNTIME}" sif list "${image}" >/dev/null
    printf '%s' "${actual}"
}

TRAINING_SIF_SHA256=$(verify_sif_with_sidecar "${PLATOON_TRAINING_SIF}")
training_labels=$("${RUNTIME}" inspect --labels "${PLATOON_TRAINING_SIF}")
grep -Fq "${TRAINING_BASE_PIN}" <<<"${training_labels}" || \
    die "training SIF labels do not name the pinned amd64 base manifest"
grep -Fq "${APEX_PIN}" <<<"${training_labels}" || \
    die "training SIF labels do not name the pinned Apex commit"
grep -Fq "${TE_COMMIT_PIN}" <<<"${training_labels}" || \
    die "training SIF labels do not name the pinned Transformer Engine commit"
unset training_labels
export PLATOON_TRAINING_SIF_SHA256=${TRAINING_SIF_SHA256}
export PLATOON_STAGE_TRAINING_SIF=${PLATOON_STAGE_TRAINING_SIF:-0}
[[ "${PLATOON_STAGE_TRAINING_SIF}" == 0 || "${PLATOON_STAGE_TRAINING_SIF}" == 1 ]] || \
    die "PLATOON_STAGE_TRAINING_SIF must be 0 or 1"
actual_toolathlon_sha=$(sha256sum "${OPENREWARD_TOOLATHLON_SIF}" | awk '{print $1}')
[[ "${actual_toolathlon_sha}" == "${TOOLATHLON_SHA256}" ]] || \
    die "Toolathlon SIF digest mismatch: ${actual_toolathlon_sha}"
"${RUNTIME}" sif list "${OPENREWARD_TOOLATHLON_SIF}" >/dev/null
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
export OPENREWARD_TOOLATHLON_SIF_SHA256=${TOOLATHLON_SHA256}

# The controller owns rollout/judging calls and W&B.  These names are the only
# training secrets accepted by this launcher.  They are never printed and are
# explicitly removed by node-service helpers; PBS worker containers receive no
# secret names at all.
if [[ -z "${HF_TOKEN:-}" && -r "${HF_TOKEN_PATH:-}" ]]; then
    IFS= read -r HF_TOKEN <"${HF_TOKEN_PATH}"
    export HF_TOKEN
fi
required_secret_names=(WANDB_API_KEY OPENAI_API_KEY LITELLM_API_KEY)
required_endpoint_names=(OPENAI_BASE_URL LITELLM_BASE_URL RUBRIC_DEFAULT_LLM)
for required_name in "${required_secret_names[@]}" "${required_endpoint_names[@]}"; do
    [[ -n "${!required_name:-}" ]] || \
        die "${required_name} is not set in the protected cluster environment"
done
unset required_name

if ! awk -v slots=8 '
    NF { count[$1]++ }
    END {
        failed=0
        for (host in count) if (count[host] != slots) {
            printf "invalid PBS vnode count host=%s count=%d expected=%d\n", host, count[host], slots > "/dev/stderr"
            failed=1
        }
        exit failed
    }
' "${PBS_NODEFILE}"; then
    die "PBS_NODEFILE does not match select=N:mpiprocs=8"
fi

JOB_ID_SAFE=${PBS_JOBID//[^A-Za-z0-9_.-]/_}
JOB_STATE_DIR=${PLATOON_JOB_STATE_ROOT}/openreward/${JOB_ID_SAFE}
LOG_DIR=${PLATOON_USER_ROOT}/logs/openreward/${JOB_ID_SAFE}
NODE_MAP=${JOB_STATE_DIR}/ordered-nodes.tsv
mkdir -p \
    "${JOB_STATE_DIR}" \
    "${LOG_DIR}" \
    "${PLATOON_EXPERIMENT_ROOT}" \
    "${PLATOON_NAME_RESOLVE_ROOT}" \
    "${OPENREWARD_ENV_CACHE_ROOT}" \
    "${PLATOON_PROJECT_ROOT}/cache/openreward-controller-venvs" \
    "${UV_CACHE_DIR}" \
    "${PLATOON_PROJECT_ROOT}/cache/uv-controller-rhel9-glibc234" \
    "${PLATOON_PROJECT_ROOT}/cache/te-wheels" \
    "${PLATOON_PROJECT_ROOT}/cache/apex-wheels"
chmod 0700 "${JOB_STATE_DIR}" "${LOG_DIR}"

awk '!seen[$1]++ { printf "%s\t%d\n", $1, NR - 1 }' "${PBS_NODEFILE}" >"${NODE_MAP}"
mapfile -t ALLOC_NODES < <(cut -f1 "${NODE_MAP}")
[[ "${#ALLOC_NODES[@]}" -eq "${EXPECTED_NNODES}" ]] || \
    die "expected ${EXPECTED_NNODES} ordered unique PBS nodes, found ${#ALLOC_NODES[@]}"
CONTROLLER_NODE=${ALLOC_NODES[0]}
controller_short=${CONTROLLER_NODE%%.*}
actual_short=$(hostname -s)
[[ "${actual_short}" == "${controller_short}" ]] || \
    die "PBS batch shell is on ${actual_short}, expected first node ${CONTROLLER_NODE}"

# Environment services may be restricted to the ordered tail reserved for the
# GPU8 rollout role.  This keeps their loop-backed Toolathlon SIF/overlay away
# from GPU1 actor nodes.  The default remains every allocation node so existing
# wrappers preserve their behavior.
SERVICE_NODE_COUNT=${OPENREWARD_SERVICE_NODE_COUNT:-${EXPECTED_NNODES}}
require_uint OPENREWARD_SERVICE_NODE_COUNT "${SERVICE_NODE_COUNT}" 1
SERVICE_NODE_COUNT=$((10#${SERVICE_NODE_COUNT}))
((SERVICE_NODE_COUNT <= EXPECTED_NNODES)) || \
    die "OPENREWARD_SERVICE_NODE_COUNT=${SERVICE_NODE_COUNT} exceeds the ${EXPECTED_NNODES}-node allocation"
SERVICE_NODE_START=$((EXPECTED_NNODES - SERVICE_NODE_COUNT))
SERVICE_NODE_MAP=${JOB_STATE_DIR}/ordered-service-nodes.tsv
awk -v first="$((SERVICE_NODE_START + 1))" 'NR >= first' "${NODE_MAP}" >"${SERVICE_NODE_MAP}"
mapfile -t SERVICE_NODES < <(cut -f1 "${SERVICE_NODE_MAP}")
[[ "${#SERVICE_NODES[@]}" -eq "${SERVICE_NODE_COUNT}" ]] || \
    die "expected ${SERVICE_NODE_COUNT} ordered service nodes, found ${#SERVICE_NODES[@]}"
for service_index in "${!SERVICE_NODES[@]}"; do
    allocation_index=$((SERVICE_NODE_START + service_index))
    [[ "${SERVICE_NODES[service_index]}" == "${ALLOC_NODES[allocation_index]}" ]] || \
        die "service-node tail does not preserve PBS allocation order at index ${allocation_index}"
done
unset service_index allocation_index
service_node_csv=$(IFS=,; printf '%s' "${SERVICE_NODES[*]}")
echo "[pbs-openreward] environment service nodes count=${SERVICE_NODE_COUNT} ordered_tail=${service_node_csv}"
SERVICE_CLEANUP_RECEIPT_DIR=${JOB_STATE_DIR}/node-services-cleanup
SERVICE_SHUTDOWN_REQUEST_DIR=${JOB_STATE_DIR}/node-services-shutdown-request
CONTROLLER_RUNTIME_RECEIPT=${JOB_STATE_DIR}/controller-runtime.json
[[ ! -e "${CONTROLLER_RUNTIME_RECEIPT}" && ! -L "${CONTROLLER_RUNTIME_RECEIPT}" ]] || \
    die "refusing stale controller runtime receipt"
[[ ! -e "${SERVICE_CLEANUP_RECEIPT_DIR}" && ! -L "${SERVICE_CLEANUP_RECEIPT_DIR}" ]] || \
    die "refusing stale node-service cleanup receipt directory"
[[ ! -e "${SERVICE_SHUTDOWN_REQUEST_DIR}" && ! -L "${SERVICE_SHUTDOWN_REQUEST_DIR}" ]] || \
    die "refusing stale node-service shutdown request directory"
mkdir "${SERVICE_CLEANUP_RECEIPT_DIR}"
mkdir "${SERVICE_SHUTDOWN_REQUEST_DIR}"
chmod 0700 "${SERVICE_CLEANUP_RECEIPT_DIR}" "${SERVICE_SHUTDOWN_REQUEST_DIR}"

# `${PLATOON_NODE_LOCAL_ROOT}` is the same allocation-specific spelling on
# every ABCI host, but /local is a different filesystem on each host.  Create
# and attest every literal worker bind/preamble source before any immutable
# build, environment service, or role container can start.  One synchronous
# pbsdsh task is addressed to each ordered host; controller-side correctness
# relies on private, fsync-backed receipts rather than pbsdsh output ordering.
NODE_LOCAL_INIT_GENERATION=$("${CONTROLLER_PYTHON_BASE}" -I -c \
    'import secrets; print(secrets.token_hex(16))')
[[ "${NODE_LOCAL_INIT_GENERATION}" =~ ^[0-9a-f]{32}$ ]] || \
    die "could not create a node-local initialization generation"
NODE_LOCAL_INIT_SNAPSHOT=${JOB_STATE_DIR}/openreward-node-local-init.snapshot.py
node_local_init_snapshot_tmp=${NODE_LOCAL_INIT_SNAPSHOT}.partial.$$
node_local_init_source_sha=$(sha256sum "${NODE_LOCAL_INIT}" | awk '{print $1}')
cp -- "${NODE_LOCAL_INIT}" "${node_local_init_snapshot_tmp}"
chmod 0500 "${node_local_init_snapshot_tmp}"
node_local_init_snapshot_sha=$(sha256sum "${node_local_init_snapshot_tmp}" | awk '{print $1}')
node_local_init_current_sha=$(sha256sum "${NODE_LOCAL_INIT}" | awk '{print $1}')
[[ "${node_local_init_source_sha}" == "${node_local_init_snapshot_sha}" &&
    "${node_local_init_source_sha}" == "${node_local_init_current_sha}" ]] || \
    die "node-local initializer changed while creating the job-private snapshot"
mv -T -- "${node_local_init_snapshot_tmp}" "${NODE_LOCAL_INIT_SNAPSHOT}"
[[ ! -L "${NODE_LOCAL_INIT_SNAPSHOT}" && -f "${NODE_LOCAL_INIT_SNAPSHOT}" &&
    "$(stat -c %a "${NODE_LOCAL_INIT_SNAPSHOT}")" == 500 ]] || \
    die "job-private node-local initializer snapshot is not a mode-0500 plain file"
NODE_LOCAL_RECEIPT_DIR=${JOB_STATE_DIR}/node-local-ready/${NODE_LOCAL_INIT_GENERATION}
mkdir -p "${NODE_LOCAL_RECEIPT_DIR}"
chmod 0700 "${JOB_STATE_DIR}/node-local-ready" "${NODE_LOCAL_RECEIPT_DIR}"

echo "[pbs-openreward] initializing worker bind sources on all ${EXPECTED_NNODES} ordered nodes"
node_init_count=0
while IFS=$'\t' read -r host vnode_index; do
    [[ "${host}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || \
        die "ordered node name has an invalid format"
    [[ "${vnode_index}" =~ ^[0-9]+$ ]] || \
        die "ordered node task index is invalid for host ${host}"
    node_init_log=${LOG_DIR}/node-local-init-${host}.log
    node_init_receipt=${NODE_LOCAL_RECEIPT_DIR}/${host}.json
    if ! env \
        -u WANDB_API_KEY \
        -u OPENAI_API_KEY \
        -u OPENAI_BASE_URL \
        -u LITELLM_API_KEY \
        -u LITELLM_BASE_URL \
        -u RUBRIC_DEFAULT_LLM \
        -u HF_TOKEN \
        -u AWS_ACCESS_KEY_ID \
        -u AWS_SECRET_ACCESS_KEY \
        -u AWS_SESSION_TOKEN \
        "${PBSDSH}" -n "${vnode_index}" -- \
        "${NODE_LOCAL_INIT_PYTHON}" "${NODE_LOCAL_INIT_SNAPSHOT}" init \
        --expected-host "${host}" \
        --task-index "${vnode_index}" \
        --expected-root "${PLATOON_NODE_LOCAL_ROOT}" \
        --pbs-job-id "${PBS_JOBID}" \
        --generation "${NODE_LOCAL_INIT_GENERATION}" \
        --receipt "${node_init_receipt}" \
        --stage-training-sif "${PLATOON_STAGE_TRAINING_SIF}" \
        --training-sif "${PLATOON_TRAINING_SIF}" \
        --training-sif-sha256 "${TRAINING_SIF_SHA256}" \
        --attest-container-runtime "${ATTEST_CONTAINER_RUNTIME}" \
        --container-runtime "${RUNTIME}" \
        --container-runtime-sha256 "${EXPECTED_CONTAINER_RUNTIME_SHA256}" \
        --container-runtime-version "${EXPECTED_CONTAINER_RUNTIME_VERSION}" \
        >"${node_init_log}" 2>&1; then
        echo "[pbs-openreward] ERROR: node-local initialization failed on ${host}" >&2
        tail -n 40 "${node_init_log}" >&2 || true
        exit 2
    fi
    node_init_count=$((node_init_count + 1))
done <"${NODE_MAP}"
[[ "${node_init_count}" -eq "${EXPECTED_NNODES}" ]] || \
    die "launched ${node_init_count} node-local initializers; expected ${EXPECTED_NNODES}"

"${CONTROLLER_PYTHON_BASE}" -I "${NODE_LOCAL_INIT_SNAPSHOT}" verify \
    --node-map "${NODE_MAP}" \
    --receipt-dir "${NODE_LOCAL_RECEIPT_DIR}" \
    --expected-nodes "${EXPECTED_NNODES}" \
    --expected-root "${PLATOON_NODE_LOCAL_ROOT}" \
    --pbs-job-id "${PBS_JOBID}" \
    --generation "${NODE_LOCAL_INIT_GENERATION}" \
    --stage-training-sif "${PLATOON_STAGE_TRAINING_SIF}" \
    --training-sif-sha256 "${TRAINING_SIF_SHA256}" \
    --attest-container-runtime "${ATTEST_CONTAINER_RUNTIME}" \
    --container-runtime "${RUNTIME}" \
    --container-runtime-sha256 "${EXPECTED_CONTAINER_RUNTIME_SHA256}" \
    --container-runtime-version "${EXPECTED_CONTAINER_RUNTIME_VERSION}"
unset \
    host vnode_index node_init_log node_init_receipt node_init_count \
    node_local_init_snapshot_tmp node_local_init_source_sha \
    node_local_init_snapshot_sha node_local_init_current_sha

urls_for_port() {
    local port=$1 result= node
    for node in "${SERVICE_NODES[@]}"; do
        result="${result:+${result},}http://${node}:${port}"
    done
    printf '%s' "${result}"
}

export OPENREWARD_PORT=${OPENREWARD_PORT:-8082}
export OPENREWARD_SWE_REBENCH_PORT=${OPENREWARD_SWE_REBENCH_PORT:-8084}
export OPENREWARD_SESSION_URLS_TOOLATHLON
OPENREWARD_SESSION_URLS_TOOLATHLON=$(urls_for_port "${OPENREWARD_PORT}")
export OPENREWARD_SESSION_URLS_SWE_REBENCH
OPENREWARD_SESSION_URLS_SWE_REBENCH=$(urls_for_port "${OPENREWARD_SWE_REBENCH_PORT}")
export OPENREWARD_SESSION_URLS=${OPENREWARD_SESSION_URLS_TOOLATHLON}
export OPENREWARD_ENABLE_TMAX=0

export OPENREWARD_PROJECT_DIR=${OPENREWARD_DIR}
export OPENREWARD_UV_CACHE_DIR=${UV_CACHE_DIR}
export OPENREWARD_UV_LINK_MODE=hardlink
export OPENREWARD_BUILD_TE=1
export OPENREWARD_BUILD_APEX=1
export TE_VERSION=${TE_PIN}
export APEX_GIT_REF=${APEX_PIN}
export TORCH_CUDA_ARCH_LIST=9.0+PTX
export TE_WHEEL_CACHE=${PLATOON_PROJECT_ROOT}/cache/te-wheels
export APEX_WHEEL_CACHE=${PLATOON_PROJECT_ROOT}/cache/apex-wheels
export OPENREWARD_CONTAINER_ENV_ID=sha256:${TRAINING_SIF_SHA256}

echo "[pbs-openreward] resolving immutable SIF worker environment from source snapshot"
env_resolution=$(
    OPENREWARD_CONTAINER_IMAGE="${PLATOON_TRAINING_SIF}" \
    OPENREWARD_UV_BIN="${UV_BIN}" \
    "${PREPARE_ENV}" resolve
)
IFS=$'\t' read -r OPENREWARD_ENV_KEY OPENREWARD_SOURCE_SHA OPENREWARD_SOURCE_ARCHIVE OPENREWARD_JOB_VENV <<<"${env_resolution}"
[[ -n "${OPENREWARD_ENV_KEY}" && -r "${OPENREWARD_SOURCE_ARCHIVE}" && -n "${OPENREWARD_JOB_VENV}" ]] || \
    die "environment resolver returned a malformed record"
OPENREWARD_JOB_PYTHON=${OPENREWARD_JOB_VENV}/bin/python
export OPENREWARD_ENV_KEY OPENREWARD_SOURCE_SHA OPENREWARD_SOURCE_ARCHIVE OPENREWARD_JOB_VENV

# Use explicit names after satisfying the unchanged worker builder's variable
# contract.  Worker Python only runs inside the SIF; it is never used for a PBS
# host launcher or the controller process.
OPENREWARD_WORKER_ENV_KEY=${OPENREWARD_ENV_KEY}
OPENREWARD_WORKER_SOURCE_SHA=${OPENREWARD_SOURCE_SHA}
OPENREWARD_WORKER_SOURCE_ARCHIVE=${OPENREWARD_SOURCE_ARCHIVE}
OPENREWARD_WORKER_JOB_VENV=${OPENREWARD_JOB_VENV}
OPENREWARD_WORKER_JOB_PYTHON=${OPENREWARD_JOB_PYTHON}
export \
    OPENREWARD_WORKER_ENV_KEY \
    OPENREWARD_WORKER_SOURCE_SHA \
    OPENREWARD_WORKER_SOURCE_ARCHIVE \
    OPENREWARD_WORKER_JOB_VENV

export OPENREWARD_CONTROLLER_ENV_CACHE_ROOT=${OPENREWARD_CONTROLLER_ENV_CACHE_ROOT:-${PLATOON_PROJECT_ROOT}/cache/openreward-controller-venvs}
export OPENREWARD_CONTROLLER_UV_CACHE_DIR=${OPENREWARD_CONTROLLER_UV_CACHE_DIR:-${PLATOON_PROJECT_ROOT}/cache/uv-controller-rhel9-glibc234}
export OPENREWARD_CONTROLLER_UV_BIN=${UV_BIN}
export OPENREWARD_CONTROLLER_PYTHON=${CONTROLLER_PYTHON_BASE}
echo "[pbs-openreward] resolving separately pinned host-controller environment"
controller_resolution=$(
    env \
        -u WANDB_API_KEY \
        -u OPENAI_API_KEY \
        -u OPENAI_BASE_URL \
        -u LITELLM_API_KEY \
        -u LITELLM_BASE_URL \
        -u RUBRIC_DEFAULT_LLM \
        -u HF_TOKEN \
        -u AWS_ACCESS_KEY_ID \
        -u AWS_SECRET_ACCESS_KEY \
        -u AWS_SESSION_TOKEN \
        "${PREPARE_CONTROLLER_ENV}" resolve
)
IFS=$'\t' read -r OPENREWARD_CONTROLLER_ENV_KEY OPENREWARD_CONTROLLER_JOB_VENV \
    <<<"${controller_resolution}"
[[ -n "${OPENREWARD_CONTROLLER_ENV_KEY}" && -n "${OPENREWARD_CONTROLLER_JOB_VENV}" ]] || \
    die "controller environment resolver returned a malformed record"
OPENREWARD_CONTROLLER_JOB_PYTHON=${OPENREWARD_CONTROLLER_JOB_VENV}/bin/python
export \
    OPENREWARD_CONTROLLER_ENV_KEY \
    OPENREWARD_CONTROLLER_JOB_VENV

helper_snapshot=${JOB_STATE_DIR}/prepare_openreward_env.snapshot.sh
"${PREPARE_ENV}" materialize "${OPENREWARD_SOURCE_ARCHIVE}" "${helper_snapshot}"
controller_helper_snapshot=${JOB_STATE_DIR}/prepare_openreward_controller_env.snapshot.sh
"${PREPARE_CONTROLLER_ENV}" materialize \
    "${OPENREWARD_SOURCE_ARCHIVE}" "${controller_helper_snapshot}"
mkdir -p \
    "${PLATOON_NODE_LOCAL_ROOT}/env-build-tmp" \
    "${PLATOON_NODE_LOCAL_ROOT}/controller-build-tmp" \
    "${PLATOON_NODE_LOCAL_ROOT}/controller-build-home" \
    "${PLATOON_NODE_LOCAL_ROOT}/worker-home" \
    "${PLATOON_NODE_LOCAL_ROOT}/controller-home" \
    "${PLATOON_NODE_LOCAL_ROOT}/worker-tmp"
chmod 0700 \
    "${PLATOON_NODE_LOCAL_ROOT}/controller-build-home" \
    "${PLATOON_NODE_LOCAL_ROOT}/worker-home" \
    "${PLATOON_NODE_LOCAL_ROOT}/controller-home"
chmod 1777 "${PLATOON_NODE_LOCAL_ROOT}/worker-tmp"

echo "[pbs-openreward] building/validating immutable worker venv key=${OPENREWARD_WORKER_ENV_KEY} in the digest-pinned SIF"
"${RUNTIME}" exec --nv --cleanenv --containall --no-home \
    --no-mount home,cwd,bind-paths \
    --bind /etc/resolv.conf:/etc/resolv.conf:ro \
    --bind /etc/hosts:/etc/hosts:ro \
    --bind "${PLATOON_REPO_ROOT}:${PLATOON_REPO_ROOT}:ro" \
    --bind "${PLATOON_PROJECT_ROOT}:${PLATOON_PROJECT_ROOT}:rw" \
    --bind "${PLATOON_NODE_LOCAL_ROOT}:${PLATOON_NODE_LOCAL_ROOT}:rw" \
    --pwd "${JOB_STATE_DIR}" \
    --env "PLATOON_REPO_ROOT=${PLATOON_REPO_ROOT}" \
    --env "OPENREWARD_ENV_CACHE_ROOT=${OPENREWARD_ENV_CACHE_ROOT}" \
    --env "OPENREWARD_ENV_KEY=${OPENREWARD_ENV_KEY}" \
    --env "OPENREWARD_SOURCE_SHA=${OPENREWARD_SOURCE_SHA}" \
    --env "OPENREWARD_SOURCE_ARCHIVE=${OPENREWARD_SOURCE_ARCHIVE}" \
    --env "OPENREWARD_JOB_VENV=${OPENREWARD_JOB_VENV}" \
    --env "OPENREWARD_UV_CACHE_DIR=${OPENREWARD_UV_CACHE_DIR}" \
    --env "OPENREWARD_UV_LINK_MODE=${OPENREWARD_UV_LINK_MODE}" \
    --env OPENREWARD_UV_BIN=uv \
    --env OPENREWARD_BUILD_TE=1 \
    --env OPENREWARD_BUILD_APEX=1 \
    --env "TE_VERSION=${TE_VERSION}" \
    --env "APEX_GIT_REF=${APEX_GIT_REF}" \
    --env "TE_WHEEL_CACHE=${TE_WHEEL_CACHE}" \
    --env "APEX_WHEEL_CACHE=${APEX_WHEEL_CACHE}" \
    --env TORCH_CUDA_ARCH_LIST=9.0+PTX \
    --env "TMPDIR=${PLATOON_NODE_LOCAL_ROOT}/env-build-tmp" \
    --env "TE_BUILD_SCRATCH=${PLATOON_NODE_LOCAL_ROOT}/env-build-tmp" \
    --env "APEX_BUILD_SCRATCH=${PLATOON_NODE_LOCAL_ROOT}/env-build-tmp" \
    --env CUDA_VISIBLE_DEVICES= \
    "${PLATOON_TRAINING_SIF}" \
    /usr/bin/env \
        "HOME=${PLATOON_NODE_LOCAL_ROOT}/worker-home" \
        "XDG_CACHE_HOME=${PLATOON_NODE_LOCAL_ROOT}/env-build-cache/xdg" \
        "FLASHINFER_CACHEDIR=${PLATOON_NODE_LOCAL_ROOT}/env-build-cache/flashinfer" \
        "TORCH_EXTENSIONS_DIR=${PLATOON_NODE_LOCAL_ROOT}/env-build-cache/torch-extensions" \
        /bin/bash "${helper_snapshot}" build

[[ -L "${OPENREWARD_WORKER_JOB_PYTHON}" ]] || \
    die "immutable worker venv Python link is missing: ${OPENREWARD_WORKER_JOB_PYTHON}"
[[ "$(readlink "${OPENREWARD_WORKER_JOB_PYTHON}")" == /usr/bin/python3.12 ]] || \
    die "immutable worker venv Python does not target the SIF interpreter"
[[ -f "${OPENREWARD_WORKER_JOB_VENV}/.platoon-ready" ]] || \
    die "immutable worker venv has no readiness marker"
cuda_lib_dirs=$(find "${OPENREWARD_WORKER_JOB_VENV}/lib/python3.12/site-packages/nvidia" \
    -mindepth 2 -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: - || true)
[[ -n "${cuda_lib_dirs}" ]] || die "immutable worker venv has no pinned NVIDIA library directories"

echo "[pbs-openreward] building/validating separately pinned host-controller venv key=${OPENREWARD_CONTROLLER_ENV_KEY}"
env \
    -u WANDB_API_KEY \
    -u OPENAI_API_KEY \
    -u OPENAI_BASE_URL \
    -u LITELLM_API_KEY \
    -u LITELLM_BASE_URL \
    -u RUBRIC_DEFAULT_LLM \
    -u HF_TOKEN \
    -u AWS_ACCESS_KEY_ID \
    -u AWS_SECRET_ACCESS_KEY \
    -u AWS_SESSION_TOKEN \
    HOME="${PLATOON_NODE_LOCAL_ROOT}/controller-build-home" \
    XDG_CACHE_HOME="${PLATOON_NODE_LOCAL_ROOT}/controller-build-cache/xdg" \
    TMPDIR="${PLATOON_NODE_LOCAL_ROOT}/controller-build-tmp" \
    OPENREWARD_CONTROLLER_ENV_CACHE_ROOT="${OPENREWARD_CONTROLLER_ENV_CACHE_ROOT}" \
    OPENREWARD_CONTROLLER_ENV_KEY="${OPENREWARD_CONTROLLER_ENV_KEY}" \
    OPENREWARD_CONTROLLER_JOB_VENV="${OPENREWARD_CONTROLLER_JOB_VENV}" \
    OPENREWARD_CONTROLLER_PYTHON="${OPENREWARD_CONTROLLER_PYTHON}" \
    OPENREWARD_CONTROLLER_UV_BIN="${OPENREWARD_CONTROLLER_UV_BIN}" \
    OPENREWARD_CONTROLLER_UV_CACHE_DIR="${OPENREWARD_CONTROLLER_UV_CACHE_DIR}" \
    OPENREWARD_SOURCE_SHA="${OPENREWARD_WORKER_SOURCE_SHA}" \
    OPENREWARD_SOURCE_ARCHIVE="${OPENREWARD_WORKER_SOURCE_ARCHIVE}" \
    /bin/bash "${controller_helper_snapshot}" build

[[ -x "${OPENREWARD_CONTROLLER_JOB_PYTHON}" ]] || \
    die "immutable host-controller Python is missing: ${OPENREWARD_CONTROLLER_JOB_PYTHON}"
[[ -f "${OPENREWARD_CONTROLLER_JOB_VENV}/.platoon-controller-ready" ]] || \
    die "immutable host-controller venv has no readiness marker"

# Prove the host entrypoint remains independent of container-only compiled
# extensions.  Actual workers resolve the emitted Megatron class path inside
# the SIF, where TE/APEX were built and validated above.
env \
    -u WANDB_API_KEY \
    -u OPENAI_API_KEY \
    -u OPENAI_BASE_URL \
    -u LITELLM_API_KEY \
    -u LITELLM_BASE_URL \
    -u RUBRIC_DEFAULT_LLM \
    -u HF_TOKEN \
    -u AWS_ACCESS_KEY_ID \
    -u AWS_SECRET_ACCESS_KEY \
    -u AWS_SESSION_TOKEN \
    -u VIRTUAL_ENV \
    -u UV_PROJECT_ENVIRONMENT \
    -u PYTHONHOME \
    -u PYTHONPATH \
    HOME="${PLATOON_NODE_LOCAL_ROOT}/controller-build-home" \
    XDG_CACHE_HOME="${PLATOON_NODE_LOCAL_ROOT}/controller-build-cache/xdg" \
    CUDA_VISIBLE_DEVICES= \
    PLATOON_EXPECTED_VENV="${OPENREWARD_CONTROLLER_JOB_VENV}" \
    "${OPENREWARD_CONTROLLER_JOB_PYTHON}" -I - <<'PY'
import os
import sys
from pathlib import Path

prefix = Path(os.environ["PLATOON_EXPECTED_VENV"]).resolve()
assert Path(sys.prefix).resolve() == prefix, (sys.prefix, prefix)
import platoon.openreward.train_scripts.areal.train_areal as train_entry
from platoon.train.areal.rl import _REMOTE_PLATOON_MEGATRON_PPO_ACTOR

assert callable(train_entry.main)
assert _REMOTE_PLATOON_MEGATRON_PPO_ACTOR.__module__ == "platoon.train.areal.actor"
assert _REMOTE_PLATOON_MEGATRON_PPO_ACTOR.__name__ == "PlatoonMegatronPPOActor"
for forbidden in ("transformer_engine", "apex", "fused_weight_gradient_mlp_cuda"):
    assert not any(name == forbidden or name.startswith(forbidden + ".") for name in sys.modules), forbidden
print("host_controller_runtime=ok")
PY

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=

environment_runtime_healthy() {
    [[ -f "${OPENREWARD_WORKER_JOB_VENV}/.platoon-ready" ]] &&
        [[ -f "${OPENREWARD_WORKER_JOB_VENV}/.build-complete" ]] &&
        [[ -f "${OPENREWARD_WORKER_JOB_VENV}/.platoon-env-manifest" ]] &&
        cmp -s "${OPENREWARD_ENV_CACHE_ROOT}/manifests/${OPENREWARD_WORKER_ENV_KEY}.txt" \
            "${OPENREWARD_WORKER_JOB_VENV}/.platoon-env-manifest" &&
        [[ -f "${OPENREWARD_CONTROLLER_JOB_VENV}/.platoon-controller-ready" ]] &&
        [[ -f "${OPENREWARD_CONTROLLER_JOB_VENV}/.build-complete" ]] &&
        [[ -f "${OPENREWARD_CONTROLLER_JOB_VENV}/.platoon-controller-manifest" ]] &&
        cmp -s "${OPENREWARD_CONTROLLER_ENV_CACHE_ROOT}/manifests/${OPENREWARD_CONTROLLER_ENV_KEY}.txt" \
            "${OPENREWARD_CONTROLLER_JOB_VENV}/.platoon-controller-manifest" &&
        PLATOON_EXPECTED_VENV="${OPENREWARD_CONTROLLER_JOB_VENV}" \
            env -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT -u PYTHONHOME -u PYTHONPATH \
            "${OPENREWARD_CONTROLLER_JOB_PYTHON}" -I -c '
import importlib.metadata as metadata
import os
import sys
from pathlib import Path
prefix = Path(os.environ["PLATOON_EXPECTED_VENV"]).resolve()
assert Path(sys.prefix).resolve() == prefix
for name in ("areal", "platoon", "platoon-openreward", "torch"):
    assert Path(metadata.distribution(name).locate_file("")).resolve().is_relative_to(prefix)
' >/dev/null 2>&1
}

pbs_cleanup_barrier_timeout=${PLATOON_AREAL_PBS_CLEANUP_BARRIER_TIMEOUT:-180}
require_uint PLATOON_AREAL_PBS_CLEANUP_BARRIER_TIMEOUT "${pbs_cleanup_barrier_timeout}" 1

RUNTIME_GUARD_BIN=${JOB_STATE_DIR}/runtime-guard-bin
mkdir -p "${RUNTIME_GUARD_BIN}"
printf '%s\n' \
    '#!/bin/sh' \
    'echo "Package installation is disabled inside the immutable training runtime." >&2' \
    'exit 2' >"${RUNTIME_GUARD_BIN}/package-manager-disabled"
chmod 0555 "${RUNTIME_GUARD_BIN}/package-manager-disabled"
for package_manager in pip pip3 pip3.12; do
    ln -sfn package-manager-disabled "${RUNTIME_GUARD_BIN}/${package_manager}"
done

service_pids=()
service_hosts=()
service_vnode_indices=()
service_receipts=()
service_shutdown_requests=()
health_pid=
gate_pid=
controller_pid=
health_failure_file=${JOB_STATE_DIR}/environment-health-failed
launch_epoch=$(date +%s)

terminate_pid() {
    local pid=$1 grace=${2:-30} deadline
    [[ -n "${pid}" ]] || return 0
    if kill -0 "${pid}" 2>/dev/null; then
        kill -TERM "${pid}" 2>/dev/null || true
        deadline=$((SECONDS + grace))
        while kill -0 "${pid}" 2>/dev/null && ((SECONDS < deadline)); do
            sleep 0.2
        done
        if kill -0 "${pid}" 2>/dev/null; then
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    fi
    wait "${pid}" 2>/dev/null || true
}

service_cleanup_failed=0
service_shutdown_mode=error
publish_all_service_shutdown_requests() {
    local index pid host request receipt request_pid
    local -a request_pids=()
    local -a request_logs=()
    for index in "${!service_pids[@]}"; do
        pid=${service_pids[index]}
        host=${service_hosts[index]}
        request=${service_shutdown_requests[index]}
        receipt=${service_receipts[index]}
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "[pbs-openreward] ERROR: node-service task died before natural shutdown request on ${host}" >&2
            return 1
        fi
        if [[ -e "${request}" || -L "${request}" ]]; then
            echo "[pbs-openreward] ERROR: stale node-service shutdown request for ${host}" >&2
            return 1
        fi
        if [[ -e "${receipt}" || -L "${receipt}" ]]; then
            echo "[pbs-openreward] ERROR: node-service receipt preceded natural shutdown request on ${host}" >&2
            return 1
        fi
    done
    for index in "${!service_pids[@]}"; do
        host=${service_hosts[index]}
        request=${service_shutdown_requests[index]}
        request_logs[index]=${LOG_DIR}/services-shutdown-request-${host//[^A-Za-z0-9_.-]/_}.log
        /usr/bin/timeout --signal=TERM --kill-after=5 30 \
            "${CONTROLLER_PYTHON_BASE}" -I "${SERVICE_SHUTDOWN_REQUEST}" publish \
            --request "${request}" --pbs-job-id "${PBS_JOBID}" --host "${host}" \
            >"${request_logs[index]}" 2>&1 &
        request_pids[index]=$!
    done
    for index in "${!request_pids[@]}"; do
        request_pid=${request_pids[index]}
        if ! wait "${request_pid}"; then
            echo "[pbs-openreward] ERROR: create-only shutdown request publication failed on ${service_hosts[index]}" >&2
            tail -n 40 "${request_logs[index]}" >&2 || true
            service_cleanup_failed=1
        fi
    done
    ((service_cleanup_failed == 0)) || return 1
    service_shutdown_mode=natural
    echo "[pbs-openreward] published exact natural shutdown requests for all ${#service_pids[@]} node-service supervisors"
}

request_all_service_shutdowns() {
    local index pid host receipt
    # Error/signal fallback only. Natural success uses the request handshake so
    # each original pbsdsh remains connected until it reaps the remote obit.
    for index in "${!service_pids[@]}"; do
        pid=${service_pids[index]}
        host=${service_hosts[index]}
        receipt=${service_receipts[index]}
        if [[ -e "${receipt}" || -L "${receipt}" ]]; then
            echo "[pbs-openreward] ERROR: stale node-service cleanup receipt for ${host}" >&2
            service_cleanup_failed=1
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "[pbs-openreward] ERROR: node-service task was already dead before requested cleanup on ${host}" >&2
            service_cleanup_failed=1
        fi
    done
    for index in "${!service_pids[@]}"; do
        pid=${service_pids[index]}
        host=${service_hosts[index]}
        if kill -0 "${pid}" 2>/dev/null && ! kill -TERM "${pid}" 2>/dev/null; then
            echo "[pbs-openreward] ERROR: could not request node-service cleanup on ${host}" >&2
            service_cleanup_failed=1
        fi
    done
}

audit_service_shutdown() {
    local index=$1 host vnode_index audit_log audit_marker
    host=${service_hosts[index]}
    vnode_index=${service_vnode_indices[index]}
    audit_log=${LOG_DIR}/services-cleanup-audit-${host//[^A-Za-z0-9_.-]/_}.log
    audit_marker=${SERVICE_CLEANUP_RECEIPT_DIR}/${host}.controller-audit
    if ! "${CONTROLLER_PYTHON_BASE}" -I "${SERVICE_CLEANUP_RECEIPT}" assert-tcp-closed \
        --host "${host}" --port "${OPENREWARD_PORT}" \
        --port "${OPENREWARD_SWE_REBENCH_PORT}" --timeout-seconds 2; then
        echo "[pbs-openreward] ERROR: TCP endpoint closure failed on ${host}" >&2
        return 1
    fi
    /usr/bin/timeout --signal=TERM --kill-after=10 70 env \
        -u WANDB_API_KEY -u OPENAI_API_KEY -u OPENAI_BASE_URL \
        -u LITELLM_API_KEY -u LITELLM_BASE_URL -u RUBRIC_DEFAULT_LLM -u HF_TOKEN \
        -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN \
        "${PBSDSH}" -n "${vnode_index}" -- /usr/bin/env \
        "PLATOON_ABCI_ENV_FILE=${ABCI_ENV}" \
        /bin/bash -c '
            set -euo pipefail
            helper=$1 receipt_helper=$2 session_root=$3 runtime=$4 output=$5 marker=$6
            /usr/bin/python3 "${helper}" --session-root "${session_root}" \
                --runtime "${runtime}" --stop-grace-seconds 10 >"${output}"
            grep -Fxq swe_singularity_sessions_cleaned=0 "${output}"
            /usr/bin/python3 -I "${receipt_helper}" assert-empty-swe --session-root "${session_root}"
            printf "node_services_remote_audit=passed\npbs_job_id=%s\nhost=%s\n" \
                "${PBS_JOBID}" "$(hostname -s)" >"${marker}.partial"
            chmod 0600 "${marker}.partial"
            ln "${marker}.partial" "${marker}"
            rm -f "${marker}.partial"
        ' node-service-audit "${SWE_CLEANUP}" "${SERVICE_CLEANUP_RECEIPT}" \
        "${SWE_APPTAINER_SESSION_ROOT}" "${RUNTIME}" \
        "${PLATOON_NODE_LOCAL_ROOT}/node-services-controller-audit.log" "${audit_marker}" \
        >"${audit_log}" 2>&1
    [[ -f "${audit_marker}" && ! -L "${audit_marker}" && \
        "$(stat -c %u:%a "${audit_marker}")" == "$(id -u):600" ]] && \
        grep -Fxq node_services_remote_audit=passed "${audit_marker}" && \
        grep -Fxq "pbs_job_id=${PBS_JOBID}" "${audit_marker}" && \
        grep -Fxq "host=${host%%.*}" "${audit_marker}"
}

cleanup() {
    local status=$? index pid wait_status service_shutdown_deadline receipts_ready audit_pid
    local expected_pbsdsh_status expected_primary_status kill_deadline
    local -a service_audit_pids=()
    trap - EXIT INT TERM
    terminate_pid "${gate_pid}" 5
    terminate_pid "${health_pid}" 5
    terminate_pid "${controller_pid}" 75
    # A controller can die after a role launcher is spawned but before Python
    # enters the trainer context (or while handling SIGTERM).  Keep the PBS
    # allocation alive until every exact launcher generation has published a
    # successful cleanup receipt and its manifest-listed control/discovery
    # paths are gone.  The barrier is read-only: identity-checked launchers own
    # all signalling and deletion.
    if ! "${OPENREWARD_CONTROLLER_JOB_PYTHON}" \
        "${OPENREWARD_CONTROLLER_JOB_VENV}/lib/python3.12/site-packages/platoon/train/areal/pbs_cleanup_barrier.py" \
        --state-dir "${JOB_STATE_DIR}/pbs-prealloc/${JOB_ID_SAFE}/${EXPECTED_TRIAL}" \
        --pbs-job-id "${PBS_JOBID}" \
        --experiment "${EXPECTED_EXPERIMENT}" \
        --trial "${EXPECTED_TRIAL}" \
        --timeout-seconds "${pbs_cleanup_barrier_timeout}"; then
        echo "[pbs-openreward] ERROR: PBS role cleanup was not durably confirmed" >&2
        status=1
    fi
    if [[ "${status}" -eq 0 ]]; then
        if ! publish_all_service_shutdown_requests; then
            status=1
            request_all_service_shutdowns
        fi
    else
        request_all_service_shutdowns
    fi
    if [[ "${service_shutdown_mode}" == natural ]]; then
        expected_pbsdsh_status=0
        expected_primary_status=0
    else
        expected_pbsdsh_status=1
        expected_primary_status=143
    fi
    service_shutdown_deadline=$((SECONDS + 180))
    for index in "${!service_pids[@]}"; do
        pid=${service_pids[index]}
        while kill -0 "${pid}" 2>/dev/null && ((SECONDS < service_shutdown_deadline)); do
            sleep 0.2
        done
        if kill -0 "${pid}" 2>/dev/null; then
            echo "[pbs-openreward] ERROR: local pbsdsh required forced KILL for ${service_hosts[index]}" >&2
            kill -KILL "${pid}" 2>/dev/null || true
            service_cleanup_failed=1
            kill_deadline=$((SECONDS + 10))
            while kill -0 "${pid}" 2>/dev/null && ((SECONDS < kill_deadline)); do
                sleep 0.2
            done
        fi
        wait_status=unreaped
        if ! kill -0 "${pid}" 2>/dev/null; then
            wait_status=0
            wait "${pid}" 2>/dev/null || wait_status=$?
        fi
        if [[ "${wait_status}" != "${expected_pbsdsh_status}" ]]; then
            echo "[pbs-openreward] ERROR: pbsdsh cleanup status was ${wait_status}, expected ${expected_pbsdsh_status} on ${service_hosts[index]}" >&2
            service_cleanup_failed=1
        fi
    done
    receipts_ready=0
    while ((SECONDS < service_shutdown_deadline)); do
        receipts_ready=1
        for index in "${!service_receipts[@]}"; do
            [[ -f "${service_receipts[index]}" && ! -L "${service_receipts[index]}" ]] || receipts_ready=0
        done
        ((receipts_ready)) && break
        sleep 0.2
    done
    for index in "${!service_receipts[@]}"; do
        if ! "${CONTROLLER_PYTHON_BASE}" -I "${SERVICE_CLEANUP_RECEIPT}" verify \
            --receipt "${service_receipts[index]}" --pbs-job-id "${PBS_JOBID}" \
            --host "${service_hosts[index]}" --primary-status "${expected_primary_status}"; then
            echo "[pbs-openreward] ERROR: node-service cleanup receipt failed on ${service_hosts[index]}" >&2
            service_cleanup_failed=1
        fi
    done
    for index in "${!service_pids[@]}"; do
        audit_service_shutdown "${index}" &
        service_audit_pids+=("$!")
    done
    for audit_pid in "${service_audit_pids[@]}"; do
        if ! wait "${audit_pid}"; then
            service_cleanup_failed=1
        fi
    done
    if ((service_cleanup_failed)); then
        status=1
    fi
    if [[ "${status}" -eq 0 ]]; then
        if ! "${CONTROLLER_PYTHON_BASE}" -I "${CONTROLLER_RUNTIME_RECEIPT_HELPER}" publish \
            --receipt "${CONTROLLER_RUNTIME_RECEIPT}" \
            --pbs-job-id "${PBS_JOBID}" --experiment "${EXPECTED_EXPERIMENT}" \
            --trial "${EXPECTED_TRIAL}" --source-sha256 "${OPENREWARD_WORKER_SOURCE_SHA}" \
            --controller-environment-key "${OPENREWARD_CONTROLLER_ENV_KEY}" \
            --controller-cache-root "${OPENREWARD_CONTROLLER_ENV_CACHE_ROOT}"; then
            echo "[pbs-openreward] ERROR: post-cleanup controller runtime receipt publication failed" >&2
            status=1
        fi
    fi
    exit "${status}"
}

handle_signal() {
    local signal=$1 status=143
    [[ "${signal}" == INT ]] && status=130
    echo "[pbs-openreward] received ${signal}; stopping controller and node services" >&2
    terminate_pid "${controller_pid}" 75
    exit "${status}"
}

trap cleanup EXIT
trap 'handle_signal TERM' TERM
trap 'handle_signal INT' INT

echo "[pbs-openreward] starting one Toolathlon and one SWE service on ${SERVICE_NODE_COUNT} ordered rollout-tail nodes: ${service_node_csv}"
while IFS=$'\t' read -r host vnode_index; do
    safe_host=${host//[^A-Za-z0-9_.-]/_}
    service_receipt=${SERVICE_CLEANUP_RECEIPT_DIR}/${host}.json
    service_shutdown_request=${SERVICE_SHUTDOWN_REQUEST_DIR}/${host}.json
    [[ ! -e "${service_receipt}" && ! -L "${service_receipt}" ]] || \
        die "refusing stale node-service cleanup receipt for ${host}"
    [[ ! -e "${service_shutdown_request}" && ! -L "${service_shutdown_request}" ]] || \
        die "refusing stale node-service shutdown request for ${host}"
    env \
        -u WANDB_API_KEY \
        -u OPENAI_API_KEY \
        -u OPENAI_BASE_URL \
        -u LITELLM_API_KEY \
        -u LITELLM_BASE_URL \
        -u RUBRIC_DEFAULT_LLM \
        -u HF_TOKEN \
        -u AWS_ACCESS_KEY_ID \
        -u AWS_SECRET_ACCESS_KEY \
        -u AWS_SESSION_TOKEN \
        "${PBSDSH}" -n "${vnode_index}" -- /usr/bin/env \
        "PLATOON_ABCI_ENV_FILE=${ABCI_ENV}" \
        "PLATOON_NODE_SERVICES_CLEANUP_RECEIPT=${service_receipt}" \
        "PLATOON_NODE_SERVICES_SHUTDOWN_REQUEST=${service_shutdown_request}" \
        "PLATOON_NODE_SERVICES_EXPECTED_HOST=${host}" \
        "PLATOON_NODE_SERVICES_EXPECTED_JOB_ID=${PBS_JOBID}" \
        "${NODE_SERVICES}" \
        >"${LOG_DIR}/services-${safe_host}.log" 2>&1 &
    service_pids+=("$!")
    service_hosts+=("${host}")
    service_vnode_indices+=("${vnode_index}")
    service_receipts+=("${service_receipt}")
    service_shutdown_requests+=("${service_shutdown_request}")
done <"${SERVICE_NODE_MAP}"
[[ "${#service_pids[@]}" -eq "${SERVICE_NODE_COUNT}" &&
    "${#service_hosts[@]}" -eq "${SERVICE_NODE_COUNT}" &&
    "${#service_vnode_indices[@]}" -eq "${SERVICE_NODE_COUNT}" &&
    "${#service_receipts[@]}" -eq "${SERVICE_NODE_COUNT}" &&
    "${#service_shutdown_requests[@]}" -eq "${SERVICE_NODE_COUNT}" ]] || \
    die "launched ${#service_pids[@]} service tasks for ${SERVICE_NODE_COUNT} service nodes"

endpoint_healthy() {
    local url=$1 status
    status=$(curl --silent --show-error --output /dev/null \
        --connect-timeout 2 --max-time 5 --write-out '%{http_code}' \
        --noproxy '*' "${url}" 2>/dev/null) || return 1
    [[ "${status}" =~ ^[1-4][0-9][0-9]$ ]]
}

service_pools_healthy() {
    local index host
    for index in "${!service_pids[@]}"; do
        kill -0 "${service_pids[index]}" 2>/dev/null || return 1
        host=${service_hosts[index]}
        endpoint_healthy "http://${host}:${OPENREWARD_PORT}/" || return 1
        endpoint_healthy "http://${host}:${OPENREWARD_SWE_REBENCH_PORT}/" || return 1
    done
}

server_wait_seconds=${OPENREWARD_SERVER_WAIT_SECS:-1200}
require_uint OPENREWARD_SERVER_WAIT_SECS "${server_wait_seconds}" 1
server_deadline=$((SECONDS + server_wait_seconds))
until service_pools_healthy; do
    for index in "${!service_pids[@]}"; do
        if ! kill -0 "${service_pids[index]}" 2>/dev/null; then
            host=${service_hosts[index]}
            echo "[pbs-openreward] ERROR: node services exited during startup on ${host}" >&2
            tail -n 100 "${LOG_DIR}/services-${host//[^A-Za-z0-9_.-]/_}.log" >&2 || true
            exit 1
        fi
    done
    if ((SECONDS >= server_deadline)); then
        echo "[pbs-openreward] ERROR: environment services did not become healthy in ${server_wait_seconds}s" >&2
        tail -n 60 "${LOG_DIR}"/services-*.log >&2 || true
        exit 1
    fi
    sleep 5
done
echo "[pbs-openreward] all Toolathlon and SWE service endpoints are healthy"

export OPENREWARD_DISABLE_UPDATE_CHECK=1
export OPENHANDS_SUPPRESS_BANNER=1
export OPENREWARD_REWARD_MODE=partial
export OPENREWARD_MCP_TIMEOUT=${OPENREWARD_MCP_TIMEOUT:-1800}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=online
# Qwen3.5 compatibility and padded-vs-packed layout are selected in the actor
# worker from the checkpoint model family and actual context-parallel topology.
# Launcher environment must not override those correctness decisions.
export PLATOON_QWEN35_GDN_CP_CONV_BACKEND=${PLATOON_QWEN35_GDN_CP_CONV_BACKEND:-fla}
export PLATOON_MEGATRON_ATTENTION_BACKEND=${PLATOON_MEGATRON_ATTENTION_BACKEND:-flash}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-4096}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG:-1}
unset PLATOON_TRAINING_DEADLINE_EPOCH PLATOON_TRAINING_DRAIN_FILE

export PLATOON_AREAL_PBS_CONTAINER_RUNTIME=${RUNTIME}
export PLATOON_AREAL_PBS_GPU_VISIBILITY_MODE=nvidia-container-cli
export PLATOON_AREAL_PBS_NVIDIA_SMI_BIN=/usr/bin/nvidia-smi
worker_training_sif=${PLATOON_TRAINING_SIF}
if [[ "${PLATOON_STAGE_TRAINING_SIF}" == 1 ]]; then
    # PBS_LOCALDIR has the same allocation-specific path on each ABCI node;
    # openreward-node-services.sh stages and verifies this path before workers.
    worker_training_sif=${PLATOON_NODE_LOCAL_ROOT}/images/platoon-training.sif
fi
export PLATOON_AREAL_PBS_CONTAINER_IMAGE=${worker_training_sif}
export PLATOON_AREAL_PBS_CONTAINER_BINDS="/etc/resolv.conf:/etc/resolv.conf:ro /etc/hosts:/etc/hosts:ro ${PLATOON_PROJECT_ROOT}:${PLATOON_PROJECT_ROOT}:rw ${PLATOON_NODE_LOCAL_ROOT}:${PLATOON_NODE_LOCAL_ROOT}:rw"
export PLATOON_AREAL_PBS_CONTAINER_WORKDIR=${JOB_STATE_DIR}
export PLATOON_AREAL_PBS_CONTAINER_ARGS="--containall --no-home --no-mount home,cwd,bind-paths --bind ${PLATOON_NODE_LOCAL_ROOT}/worker-tmp:/tmp:rw"
export PLATOON_AREAL_PBS_STATE_DIR=${JOB_STATE_DIR}/pbs-prealloc
export PLATOON_AREAL_PBS_HOST_PYTHON=${OPENREWARD_CONTROLLER_JOB_PYTHON}
# Forked inline proxies execute inside the contained rollout worker.  AReaL's
# Slurm default is the host controller's sys.executable, which is intentionally
# absent there; give the PBS scheduler the immutable SIF-built worker Python.
export PLATOON_AREAL_PBS_WORKER_PYTHON=${OPENREWARD_WORKER_JOB_PYTHON}
export PLATOON_AREAL_PBS_PBSDSH_BIN=${PBSDSH}
export PLATOON_AREAL_PBS_FORWARD_ENV="HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE OPENREWARD_DISABLE_UPDATE_CHECK OPENHANDS_SUPPRESS_BANNER OPENREWARD_SESSION_URLS OPENREWARD_SESSION_URLS_TOOLATHLON OPENREWARD_SESSION_URLS_SWE_REBENCH PLATOON_MODEL_SNAPSHOT PLATOON_QWEN35_GDN_CP_CONV_BACKEND PLATOON_MEGATRON_ATTENTION_BACKEND TORCH_NCCL_TRACE_BUFFER_SIZE TORCH_NCCL_DUMP_ON_TIMEOUT TORCH_NCCL_DESYNC_DEBUG"
export PLATOON_AREAL_PBS_WORKER_PREAMBLE="
set -euo pipefail
export HOME=${PLATOON_NODE_LOCAL_ROOT}/worker-home
export TMPDIR=/tmp
export XDG_CACHE_HOME=/tmp/xdg-\${PLATOON_ROLE}-\${PLATOON_WORKER_INDEX}
export FLASHINFER_CACHEDIR=/tmp/flashinfer-\${PLATOON_ROLE}-\${PLATOON_WORKER_INDEX}
export TORCH_EXTENSIONS_DIR=/tmp/torch-extensions-\${PLATOON_ROLE}-\${PLATOON_WORKER_INDEX}
export AREAL_CACHE_DIR=/tmp/areal-\${PLATOON_ROLE}-\${PLATOON_WORKER_INDEX}
mkdir -p \${XDG_CACHE_HOME} \${FLASHINFER_CACHEDIR} \${TORCH_EXTENSIONS_DIR} \${AREAL_CACHE_DIR}
export PATH=${RUNTIME_GUARD_BIN}:${OPENREWARD_WORKER_JOB_VENV}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT PYTHONHOME PYTHONPATH
export LD_LIBRARY_PATH=${cuda_lib_dirs}:\${LD_LIBRARY_PATH:-}
cd ${JOB_STATE_DIR}
"

monitor_services() {
    local failures=0 threshold=${OPENREWARD_HEALTH_FAILURE_THRESHOLD:-3}
    require_uint OPENREWARD_HEALTH_FAILURE_THRESHOLD "${threshold}" 1
    while [[ -n "${controller_pid}" ]] && kill -0 "${controller_pid}" 2>/dev/null; do
        if service_pools_healthy && environment_runtime_healthy; then
            failures=0
        else
            failures=$((failures + 1))
            echo "[pbs-openreward] WARNING: environment pool unhealthy ${failures}/${threshold}" >&2
            if ((failures >= threshold)); then
                : >"${health_failure_file}"
                echo "[pbs-openreward] ERROR: environment pool remained unhealthy; terminating controller" >&2
                kill -TERM "${controller_pid}" 2>/dev/null || true
                return 1
            fi
        fi
        sleep "${OPENREWARD_HEALTH_CHECK_SECS:-20}"
    done
}

export HOME=${PLATOON_NODE_LOCAL_ROOT}/controller-home
export XDG_CACHE_HOME=${PLATOON_NODE_LOCAL_ROOT}/controller-cache/xdg
export FLASHINFER_CACHEDIR=${PLATOON_NODE_LOCAL_ROOT}/controller-cache/flashinfer
export TORCH_EXTENSIONS_DIR=${PLATOON_NODE_LOCAL_ROOT}/controller-cache/torch-extensions
export AREAL_CACHE_DIR=${PLATOON_NODE_LOCAL_ROOT}/controller-cache/areal
mkdir -p "${XDG_CACHE_HOME}" "${FLASHINFER_CACHEDIR}" "${TORCH_EXTENSIONS_DIR}" "${AREAL_CACHE_DIR}"
export PATH=${RUNTIME_GUARD_BIN}:${OPENREWARD_CONTROLLER_JOB_VENV}/bin:${PATH}
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT PYTHONHOME PYTHONPATH
cd "${JOB_STATE_DIR}"

echo "[pbs-openreward] starting host controller experiment=${EXPECTED_EXPERIMENT} trial=${EXPECTED_TRIAL} nodes=${EXPECTED_NNODES} training_sif_sha256=${TRAINING_SIF_SHA256}"
"${OPENREWARD_CONTROLLER_JOB_PYTHON}" -m "${TRAIN_MODULE}" --config "${CONFIG}" \
    "cluster.n_nodes=${EXPECTED_NNODES}" \
    "actor.path=${PLATOON_MODEL_SNAPSHOT}" \
    "tokenizer_path=${PLATOON_MODEL_SNAPSHOT}" \
    "sglang.model_path=${PLATOON_MODEL_SNAPSHOT}" \
    stats_logger.wandb.mode=online &
controller_pid=$!

monitor_services &
health_pid=$!

if [[ -n "${OPENREWARD_GATE_MARKER_TO_PUBLISH:-}" ]]; then
    "${UPDATE_GATE}" monitor \
        --marker "${OPENREWARD_GATE_MARKER_TO_PUBLISH}" \
        --fileroot "${PLATOON_EXPERIMENT_ROOT}" \
        --experiment "${EXPECTED_EXPERIMENT}" \
        --trial "${EXPECTED_TRIAL}" \
        --min-global-step 0 \
        --training-sif "${PLATOON_TRAINING_SIF}" \
        --training-sif-sha256 "${TRAINING_SIF_SHA256}" \
        --config "${CONFIG}" \
        --pbs-job-id "${PBS_JOBID}" \
        --not-before-epoch "${launch_epoch}" \
        --controller-pid "${controller_pid}" &
    gate_pid=$!
fi

set +e
wait "${controller_pid}"
status=$?
set -e
controller_pid=
if [[ -f "${health_failure_file}" ]]; then
    status=1
fi
if [[ -n "${OPENREWARD_GATE_MARKER_TO_PUBLISH:-}" && "${status}" -eq 0 && \
      ! -f "${OPENREWARD_GATE_MARKER_TO_PUBLISH}" ]]; then
    echo "[pbs-openreward] ERROR: controller ended without a checkpoint-backed optimizer-update gate" >&2
    status=1
fi
exit "${status}"
