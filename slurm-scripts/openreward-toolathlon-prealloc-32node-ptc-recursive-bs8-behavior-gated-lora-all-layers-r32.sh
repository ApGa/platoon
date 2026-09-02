#!/bin/bash
#SBATCH --job-name=openreward-ta32-rec-bg-lora-all-r32
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --nodes=32
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/apurvag/logs/openreward-ta32-rec-bg-lora-all-r32-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/apurvag/logs/openreward-ta32-rec-bg-lora-all-r32-%j.err
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
DEFAULT_CONFIG=${REPO_ROOT}/plugins/openreward/platoon/openreward/configs/areal/toolathlon_openhands_areal_prealloc_32node-cp-ptc-recursive-behavior-gated-lora-all-layers-r32-bs8.yaml
CONFIG=${1:-${DEFAULT_CONFIG}}

# Credentials remain in the gitignored canonical launcher. Existing submission
# values win, which keeps this wrapper portable to another cluster.
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

export OPENREWARD_CONTROLLER_CPUS=${OPENREWARD_CONTROLLER_CPUS:-64}
export OPENREWARD_CONTROLLER_MEM=${OPENREWARD_CONTROLLER_MEM:-256G}
export PLATOON_AREAL_PREALLOC_SRUN_ARGS=${PLATOON_AREAL_PREALLOC_SRUN_ARGS:-"--unbuffered --mpi=pmi2 -K --overlap --cpu-bind=none"}
export NCCL_RAS_ENABLE=${NCCL_RAS_ENABLE:-0}

# All-layer LoRA still imports Megatron Bridge's Transformer Engine bindings.
# Make that dependency invariant explicit in this wrapper instead of relying
# solely on best-effort backend detection in the generic launcher.  Both
# installers reuse ABI-matched wheels from the shared cache.
export OPENREWARD_BUILD_TE=1
export OPENREWARD_BUILD_APEX=1

# Preserve the non-radix behavior-gated deadline and successor policy.
export OPENREWARD_DEADLINE_INITIAL_STEP_SECONDS=${OPENREWARD_DEADLINE_INITIAL_STEP_SECONDS:-1800}
export OPENREWARD_DEADLINE_SAFETY_SECONDS=0

# Fresh adapter/optimizer lineage. Automatic four-hour successors retain this
# name and recover only checkpoints produced by this LoRA trial.
export OPENREWARD_TRIAL_NAME=ta32-rec-behavior-gated-lora-all-layers-r32-v1-trial0
export OPENREWARD_SUBAGENT_DELEGATION_REWARD_COEFFICIENT=0.0
export OPENREWARD_WANDB_MODE=online
export OPENREWARD_JOB_SCRIPT=${REPO_ROOT}/slurm-scripts/openreward-toolathlon-prealloc-32node-ptc-recursive-bs8-behavior-gated-lora-all-layers-r32.sh

export OPENREWARD_SERVER_IMAGE=${USER_ROOT}/images/openreward/apga+toolathlon-gym+18e62c0d041.sqsh

exec "${BASE_LAUNCHER}" "${CONFIG}"
