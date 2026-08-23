#!/bin/bash
#SBATCH --job-name=openreward-ta32-rec-behavior-gated
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --nodes=32
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/apurvag/logs/openreward-ta32-rec-behavior-gated-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/apurvag/logs/openreward-ta32-rec-behavior-gated-%j.err
#SBATCH --mail-user=apurvag@nvidia.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -n "${PLATOON_REPO_ROOT:-}" ]]; then
  repo_root_candidate=${PLATOON_REPO_ROOT}
elif [[ -f "${SLURM_SUBMIT_DIR:-}/pyproject.toml" ]]; then
  repo_root_candidate=${SLURM_SUBMIT_DIR}
else
  repo_root_candidate=${SCRIPT_DIR}/..
fi
if ! REPO_ROOT=$(cd "${repo_root_candidate}" 2>/dev/null && pwd -P) || \
  [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "ERROR: could not locate the Platoon checkout from ${repo_root_candidate}." >&2
  exit 2
fi
export PLATOON_REPO_ROOT=${REPO_ROOT}
USER_ROOT=${PLATOON_USER_ROOT:-$(cd "${REPO_ROOT}/../.." && pwd -P)}
export PLATOON_USER_ROOT=${USER_ROOT}

BASE_LAUNCHER=${REPO_ROOT}/slurm-scripts/openreward-toolathlon-prealloc-base.sh
DEFAULT_CONFIG=${REPO_ROOT}/plugins/openreward/platoon/openreward/configs/areal/toolathlon_openhands_areal_prealloc_32node-cp-ptc-recursive-behavior-gated-r3-fp32-lm-head-bs8.yaml
CONFIG=${1:-${DEFAULT_CONFIG}}

# The credential-bearing launcher is intentionally gitignored, so it is absent
# from clean, commit-pinned worktrees. Prefer an explicit source, then a local
# source when present, and finally the user's canonical Platoon checkout.
if [[ -n "${PLATOON_CREDENTIAL_SOURCE:-}" ]]; then
  CREDENTIAL_SOURCE=${PLATOON_CREDENTIAL_SOURCE}
elif [[ -r "${REPO_ROOT}/slurm-scripts/openreward-toolathlon-prealloc.sh" ]]; then
  CREDENTIAL_SOURCE=${REPO_ROOT}/slurm-scripts/openreward-toolathlon-prealloc.sh
else
  CREDENTIAL_SOURCE=${USER_ROOT}/source/platoon/slurm-scripts/openreward-toolathlon-prealloc.sh
fi

if [[ "${SLURM_NNODES:-32}" -ne 32 ]]; then
  echo "ERROR: this wrapper requires exactly 32 nodes; got ${SLURM_NNODES}." >&2
  exit 2
fi
[[ -x "${BASE_LAUNCHER}" ]] || {
  echo "ERROR: missing Toolathlon launcher: ${BASE_LAUNCHER}" >&2
  exit 2
}
[[ -f "${CONFIG}" ]] || {
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 2
}

# Load literal export assignments without eval and without printing secrets.
# Existing submission values win; one matching shell quote pair is removed
# from the checked-in credential source when this job starts with --export=NONE.
load_credential() {
  local name=$1 line value first last
  [[ -z "${!name:-}" ]] || return 0
  line=$(grep -m1 "^export ${name}=" "${CREDENTIAL_SOURCE}" 2>/dev/null || true)
  [[ -n "${line}" ]] || return 0
  value=${line#*=}
  if ((${#value} >= 2)); then
    first=${value:0:1}
    last=${value: -1}
    if [[ ("${first}" == '"' && "${last}" == '"') || \
          ("${first}" == "'" && "${last}" == "'") ]]; then
      value=${value:1:${#value}-2}
    fi
  fi
  printf -v "${name}" '%s' "${value}"
  export "${name}"
}

for credential_name in \
  WANDB_API_KEY \
  OPENAI_API_KEY OPENAI_BASE_URL \
  LITELLM_API_KEY LITELLM_BASE_URL \
  HF_TOKEN; do
  load_credential "${credential_name}"
done
unset credential_name
unset -f load_credential

for required_name in \
  WANDB_API_KEY \
  OPENAI_API_KEY OPENAI_BASE_URL \
  LITELLM_API_KEY LITELLM_BASE_URL; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "ERROR: ${required_name} is unavailable; refusing to start an incomplete trial." >&2
    exit 2
  fi
done
unset required_name

# Preserve the proven 32-node recursive topology and allocation safeguards.
export OPENREWARD_CONTROLLER_CPUS=${OPENREWARD_CONTROLLER_CPUS:-64}
export OPENREWARD_CONTROLLER_MEM=${OPENREWARD_CONTROLLER_MEM:-128G}
export PLATOON_AREAL_PREALLOC_SRUN_ARGS=${PLATOON_AREAL_PREALLOC_SRUN_ARGS:-"--unbuffered --mpi=pmi2 -K --overlap --cpu-bind=none"}
export NCCL_RAS_ENABLE=${NCCL_RAS_ENABLE:-0}

# The cold first update is excluded from adaptive history. Use the established
# 30-minute bootstrap floor and a ten-minute boundary safety margin.
export OPENREWARD_DEADLINE_INITIAL_STEP_SECONDS=${OPENREWARD_DEADLINE_INITIAL_STEP_SECONDS:-1800}
export OPENREWARD_DEADLINE_SAFETY_SECONDS=${OPENREWARD_DEADLINE_SAFETY_SECONDS:-600}

# Force a fresh behavior-gated lineage across automatic successors. The judge
# uses the rollout policy itself, so it needs no separate model credential.
export OPENREWARD_TRIAL_NAME=ta32-rec-behavior-gate-v1-trial0
export OPENREWARD_SUBAGENT_DELEGATION_REWARD_COEFFICIENT=0.0
export OPENREWARD_WANDB_MODE=online
export OPENREWARD_JOB_SCRIPT=${REPO_ROOT}/slurm-scripts/openreward-toolathlon-prealloc-32node-ptc-recursive-bs8-behavior-gated.sh

# Pin the validated structured-error Toolathlon image. TMax/SWE servers are not
# started because this wrapper uses the single-environment base launcher.
export OPENREWARD_SERVER_IMAGE=${USER_ROOT}/images/openreward/apga+toolathlon-gym+18e62c0d041.sqsh

exec "${BASE_LAUNCHER}" "${CONFIG}"
