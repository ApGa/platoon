#!/bin/bash
#SBATCH --job-name=openreward-multienv-full-ptc-tracker
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --nodes=16
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --signal=B:USR1@300

# Add the requested supplemental OpenReward services per allocated node, then
# delegate Toolathlon and AReaL lifecycle management to the established
# preallocated launcher. TMax can be disabled for Toolathlon+SWE curricula with
# OPENREWARD_ENABLE_TMAX=0. Each rollout remains pinned to one environment
# server through a label-specific URL pool.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

platoon_repo_root_is_valid() {
  local candidate=${1:-}
  [[ -n "${candidate}" ]] || return 1
  [[ -f "${candidate}/pyproject.toml" ]] || return 1
  [[ -x "${candidate}/slurm-scripts/openreward-toolathlon-prealloc-base.sh" ]] || return 1
  [[ -x "${candidate}/slurm-scripts/openreward-multienv-prealloc-16node-ptc-task-tracker-full.sh" ]] || return 1
  [[ -x "${candidate}/slurm-scripts/openreward-multienv-server.sh" ]] || return 1
}

if [[ -n "${PLATOON_REPO_ROOT:-}" ]]; then
  repo_root_candidate=${PLATOON_REPO_ROOT}
  repo_root_source=PLATOON_REPO_ROOT
elif platoon_repo_root_is_valid "${SLURM_SUBMIT_DIR:-}"; then
  # Slurm executes a copied script below its spool directory inside a batch job.
  # Jobs submitted from the checkout retain its path in SLURM_SUBMIT_DIR.
  repo_root_candidate=${SLURM_SUBMIT_DIR}
  repo_root_source=SLURM_SUBMIT_DIR
else
  repo_root_candidate=${SCRIPT_DIR}/..
  repo_root_source="script-relative fallback"
fi

if ! REPO_ROOT=$(cd "${repo_root_candidate}" 2>/dev/null && pwd -P) || \
  ! platoon_repo_root_is_valid "${REPO_ROOT}"; then
  echo "ERROR: could not locate a valid Platoon checkout from ${repo_root_source}: ${repo_root_candidate}." >&2
  echo "Set PLATOON_REPO_ROOT to the checkout root (the directory containing pyproject.toml and slurm-scripts/)." >&2
  exit 2
fi

USER_ROOT=${PLATOON_USER_ROOT:-$(cd "${REPO_ROOT}/../.." && pwd)}
BASE_LAUNCHER=${REPO_ROOT}/slurm-scripts/openreward-toolathlon-prealloc-base.sh
SERVER_HELPER=${REPO_ROOT}/slurm-scripts/openreward-multienv-server.sh
SWE_RUNTIME_GUARD=${REPO_ROOT}/plugins/openreward/swe-rebench-runtime-guard.sh
SWE_SOURCE_ROOT=${REPO_ROOT}/external/swe-rebench-v2-openrewardenv
CREDENTIAL_SOURCE=${REPO_ROOT}/slurm-scripts/openreward-toolathlon-prealloc.sh
DEFAULT_CONFIG=${REPO_ROOT}/plugins/openreward/platoon/openreward/configs/areal/toolathlon_tmax_swe_openhands_areal_prealloc_16node-cp-ptc-task-tracker-full-r3-fp32-lm-head-hardened-v1.yaml
CONFIG=${1:-${DEFAULT_CONFIG}}

# BASH_SOURCE points into Slurm's ephemeral spool directory inside a batch job.
# Continuations must always submit a dedicated launcher from the checkout. A
# topology-specific outer wrapper can override this while reusing the mixed
# environment lifecycle below.
WRAPPER=${OPENREWARD_JOB_SCRIPT:-${REPO_ROOT}/slurm-scripts/openreward-multienv-prealloc-16node-ptc-task-tracker-full.sh}
export OPENREWARD_JOB_SCRIPT=${WRAPPER}
[[ -x "${BASE_LAUNCHER}" ]] || {
  echo "ERROR: base launcher is not executable: ${BASE_LAUNCHER}" >&2
  exit 2
}
[[ -x "${SERVER_HELPER}" ]] || {
  echo "ERROR: server helper is not executable: ${SERVER_HELPER}" >&2
  exit 2
}
[[ -r "${SWE_RUNTIME_GUARD}" ]] || {
  echo "ERROR: missing SWE-rebench runtime guard: ${SWE_RUNTIME_GUARD}" >&2
  exit 2
}
[[ -f "${CONFIG}" ]] || {
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 2
}
CONFIG=$(readlink -f "${CONFIG}")

# Reject stale submission/continuation environments before launching any
# allocation-wide server steps, then propagate the one reviewed source pin to
# every node. The per-node helper independently verifies the same pin.
# shellcheck source=../plugins/openreward/swe-rebench-runtime-guard.sh
source "${SWE_RUNTIME_GUARD}"
swe_rebench_require_validated_source \
  "${REPO_ROOT}" \
  "${SWE_SOURCE_ROOT}" \
  "${SWE_REBENCH_SOURCE_REVISION:-}" \
  "mixed-training launcher" || exit $?
export SWE_REBENCH_SOURCE_REVISION=${SWE_REBENCH_VALIDATED_SOURCE_REVISION}

# Reuse the established launcher's credentials without printing or duplicating
# secret values in this experiment wrapper. Existing submission-environment
# values win; missing values are loaded from simple `export NAME=value` lines.
for credential_name in \
  WANDB_API_KEY \
  OPENAI_API_KEY OPENAI_BASE_URL \
  LITELLM_API_KEY LITELLM_BASE_URL \
  HF_TOKEN; do
  if [[ -z "${!credential_name:-}" && -r "${CREDENTIAL_SOURCE}" ]]; then
    credential_value=$(grep -m1 "^export ${credential_name}=" "${CREDENTIAL_SOURCE}" | cut -d= -f2-)
    if [[ -n "${credential_value}" ]]; then
      printf -v "${credential_name}" '%s' "${credential_value}"
      export "${credential_name}"
    fi
  fi
done
unset credential_name credential_value

for required_name in WANDB_API_KEY OPENAI_API_KEY OPENAI_BASE_URL LITELLM_API_KEY LITELLM_BASE_URL; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "ERROR: ${required_name} is unavailable; refusing to start an incomplete mixed run." >&2
    exit 2
  fi
done
unset required_name
export OPENREWARD_WANDB_MODE=online

# This experiment's YAML owns both the unique trial name and untouched base
# actor path. Prevent inherited submission variables from replacing either.
unset OPENREWARD_TRIAL_NAME OPENREWARD_ACTOR_PATH

EXPECTED_NNODES=${OPENREWARD_MIXED_EXPECTED_NNODES:-16}
NNODES=${SLURM_NNODES:-${EXPECTED_NNODES}}
if [[ "${NNODES}" -ne "${EXPECTED_NNODES}" ]]; then
  echo "ERROR: this mixed wrapper requires exactly ${EXPECTED_NNODES} nodes; got ${NNODES}." >&2
  exit 2
fi
TMAX_PORT=${OPENREWARD_TMAX_PORT:-8083}
SWE_PORT=${OPENREWARD_SWE_REBENCH_PORT:-8084}
TOOLATHLON_PORT=${OPENREWARD_PORT:-8082}
ENABLE_TMAX=${OPENREWARD_ENABLE_TMAX:-1}
if [[ "${ENABLE_TMAX}" != 0 && "${ENABLE_TMAX}" != 1 ]]; then
  echo "ERROR: OPENREWARD_ENABLE_TMAX must be 0 or 1; got ${ENABLE_TMAX}." >&2
  exit 2
fi
SERVER_CPUS=${OPENREWARD_SUPPLEMENTAL_SERVER_CPUS:-8}
SERVER_MEM=${OPENREWARD_SUPPLEMENTAL_SERVER_MEM:-48G}
SERVER_WAIT_SECS=${OPENREWARD_SUPPLEMENTAL_SERVER_WAIT_SECS:-900}
HEALTH_CHECK_SECS=${OPENREWARD_SUPPLEMENTAL_HEALTH_CHECK_SECS:-20}
HEALTH_FAILURE_THRESHOLD=${OPENREWARD_SUPPLEMENTAL_HEALTH_FAILURE_THRESHOLD:-3}
# A mixed batch has at most eight concurrent roots across all environments.
# Eight Toolathlon workers per node leave ample headroom while avoiding 512
# mostly idle Python servers across the 16-node allocation.
export OPENREWARD_SERVER_CPUS=${OPENREWARD_SERVER_CPUS:-8}
export OPENREWARD_WORKERS=${OPENREWARD_WORKERS:-${OPENREWARD_SERVER_CPUS}}
# Cold TMax/SWE tasks import multi-GB Enroot images while OpenHands waits for
# the MCP bridge to list tools. Match the configured per-step timeout instead
# of the rollout module's 120-second default.
export OPENREWARD_MCP_TIMEOUT=${OPENREWARD_MCP_TIMEOUT:-1800}

RUN_ID=${OPENREWARD_RUN_ID:-${SLURM_JOB_ID:-manual}}

mapfile -t ALLOC_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}")
if [[ "${#ALLOC_NODES[@]}" -ne "${NNODES}" ]]; then
  echo "ERROR: expected ${NNODES} allocated nodes, found ${#ALLOC_NODES[@]}." >&2
  exit 2
fi
NODELIST=$(IFS=,; echo "${ALLOC_NODES[*]}")

urls_for_port() {
  local port=$1
  local result=
  local node
  for node in "${ALLOC_NODES[@]}"; do
    result="${result:+${result},}http://${node}:${port}"
  done
  printf '%s' "${result}"
}

export OPENREWARD_SESSION_URLS_TOOLATHLON
OPENREWARD_SESSION_URLS_TOOLATHLON=$(urls_for_port "${TOOLATHLON_PORT}")
export OPENREWARD_SESSION_URLS_TMAX
OPENREWARD_SESSION_URLS_TMAX=$(urls_for_port "${TMAX_PORT}")
export OPENREWARD_SESSION_URLS_SWE_REBENCH
OPENREWARD_SESSION_URLS_SWE_REBENCH=$(urls_for_port "${SWE_PORT}")
export OPENREWARD_TMAX_PORT=${TMAX_PORT}
export OPENREWARD_SWE_REBENCH_PORT=${SWE_PORT}
export OPENREWARD_PORT=${TOOLATHLON_PORT}
# The environment name is stable across SWE catalogs; pin the self-hosted
# service to Prime Intellect's complete verified-solvable subset explicitly so
# neither this allocation nor a continuation silently falls back to the larger
# unverified source catalog.
export SWE_REBENCH_DATA_DIR=${REPO_ROOT}/.cache/swe-rebench-v2-filtered-verified
[[ -f "${SWE_REBENCH_DATA_DIR}/task_index.json" ]] || {
  echo "ERROR: missing verified SWE-rebench task index: ${SWE_REBENCH_DATA_DIR}/task_index.json" >&2
  exit 2
}

mkdir -p "${USER_ROOT}/logs"
LOG_INSTANCE_ID=${SLURM_JOB_ID:-manual}
TMAX_LOG=${USER_ROOT}/logs/openreward-multienv-tmax-${RUN_ID}-job${LOG_INSTANCE_ID}
SWE_LOG=${USER_ROOT}/logs/openreward-multienv-swe-rebench-${RUN_ID}-job${LOG_INSTANCE_ID}
tmax_pid=
swe_pid=
base_pid=
health_pid=

cleanup() {
  local pid
  for pid in "${health_pid}" "${base_pid}" "${tmax_pid}" "${swe_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}

forward_signal() {
  local signal=$1
  if [[ -n "${base_pid}" ]] && kill -0 "${base_pid}" 2>/dev/null; then
    kill -s "${signal}" "${base_pid}" 2>/dev/null || true
  fi
}

trap cleanup EXIT
trap 'forward_signal USR1' USR1
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT

start_server_pool() {
  local kind=$1
  local log_prefix=$2
  local pid_variable=$3
  OPENREWARD_SUPPLEMENTAL_KIND="${kind}" srun \
    --overlap \
    --unbuffered \
    --nodes="${NNODES}" \
    --kill-on-bad-exit=0 \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    --nodelist="${NODELIST}" \
    --gpus-per-node=0 \
    --cpus-per-task="${SERVER_CPUS}" \
    --mem="${SERVER_MEM}" \
    --output="${log_prefix}-%N.log" \
    --error="${log_prefix}-%N.log" \
    "${SERVER_HELPER}" &
  printf -v "${pid_variable}" '%s' "$!"
}

SERVER_POOL_FAILURE_DETAIL=
SERVER_POOL_SRUN_DIED=0

server_pool_healthy() {
  local name=$1
  local port=$2
  local pid=$3
  local failed_endpoint=
  local node

  for node in "${ALLOC_NODES[@]}"; do
    if ! (exec 3<>"/dev/tcp/${node}/${port}") 2>/dev/null; then
      failed_endpoint="${node}:${port}"
      break
    fi
    exec 3>&- 3<&- || true
  done

  local srun_state=alive
  SERVER_POOL_SRUN_DIED=0
  if ! kill -0 "${pid}" 2>/dev/null; then
    srun_state=died
    SERVER_POOL_SRUN_DIED=1
  fi

  if [[ -n "${failed_endpoint}" ]]; then
    SERVER_POOL_FAILURE_DETAIL="pool=${name} endpoint=${failed_endpoint} srun_pid=${pid} srun_state=${srun_state}"
    return 1
  fi
  if [[ "${SERVER_POOL_SRUN_DIED}" -eq 1 ]]; then
    SERVER_POOL_FAILURE_DETAIL="pool=${name} endpoints=accepting port=${port} srun_pid=${pid} srun_state=died"
    return 1
  fi

  SERVER_POOL_FAILURE_DETAIL=
  return 0
}

wait_for_server_pool() {
  local name=$1
  local port=$2
  local pid=$3
  local log_prefix=$4
  local waited=0
  while ! server_pool_healthy "${name}" "${port}" "${pid}"; do
    if [[ "${SERVER_POOL_SRUN_DIED}" -eq 1 ]]; then
      echo "ERROR: ${name} server pool exited during startup: ${SERVER_POOL_FAILURE_DETAIL}." >&2
      tail -n 80 "${log_prefix}"-*.log 2>/dev/null || true
      return 1
    fi
    if [[ "${waited}" -ge "${SERVER_WAIT_SECS}" ]]; then
      echo "ERROR: timed out waiting for ${name}: ${SERVER_POOL_FAILURE_DETAIL}." >&2
      tail -n 80 "${log_prefix}"-*.log 2>/dev/null || true
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "All ${NNODES} ${name} servers are accepting connections on port ${port}."
}

if [[ "${ENABLE_TMAX}" -eq 1 ]]; then
  echo "Starting TMax and SWE-rebench server pools on ${NNODES} nodes."
  start_server_pool tmax "${TMAX_LOG}" tmax_pid
else
  echo "Starting SWE-rebench server pool on ${NNODES} nodes (TMax disabled)."
fi
start_server_pool swe_rebench "${SWE_LOG}" swe_pid
if [[ "${ENABLE_TMAX}" -eq 1 ]]; then
  wait_for_server_pool TMax "${TMAX_PORT}" "${tmax_pid}" "${TMAX_LOG}"
fi
wait_for_server_pool SWE-rebench "${SWE_PORT}" "${swe_pid}" "${SWE_LOG}"

# The delegated launcher's continuation logic must resubmit this wrapper so the
# supplemental services are restored along with Toolathlon and the trainer.
"${BASE_LAUNCHER}" "${CONFIG}" &
base_pid=$!

monitor_supplemental_servers() {
  local failures=0
  local issue_summary
  while kill -0 "${base_pid}" 2>/dev/null; do
    issue_summary=
    if [[ "${ENABLE_TMAX}" -eq 1 ]] && \
       ! server_pool_healthy TMax "${TMAX_PORT}" "${tmax_pid}"; then
      issue_summary="${SERVER_POOL_FAILURE_DETAIL}"
    fi
    if ! server_pool_healthy SWE-rebench "${SWE_PORT}" "${swe_pid}"; then
      issue_summary="${issue_summary:+${issue_summary}; }${SERVER_POOL_FAILURE_DETAIL}"
    fi

    if [[ -z "${issue_summary}" ]]; then
      failures=0
    else
      failures=$((failures + 1))
      echo "WARNING: supplemental environment health check failed (${failures}/${HEALTH_FAILURE_THRESHOLD}): ${issue_summary}." >&2
      if [[ "${failures}" -ge "${HEALTH_FAILURE_THRESHOLD}" ]]; then
        echo "ERROR: supplemental environment servers remained unhealthy (${issue_summary}); terminating the base launcher." >&2
        kill -TERM "${base_pid}" 2>/dev/null || true
        return 1
      fi
    fi
    sleep "${HEALTH_CHECK_SECS}"
  done
}

monitor_supplemental_servers &
health_pid=$!

set +e
while true; do
  wait "${base_pid}"
  status=$?
  if [[ "${status}" -ge 128 ]] && kill -0 "${base_pid}" 2>/dev/null; then
    continue
  fi
  break
done
set -e
base_pid=
exit "${status}"
