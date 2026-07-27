#!/bin/bash
#SBATCH --job-name=openreward-multienv-32n-recursive-efficiency
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --nodes=32
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/apurvag/logs/openreward-multienv-32n-recursive-efficiency-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/apurvag/logs/openreward-multienv-32n-recursive-efficiency-%j.err
#SBATCH --mail-user=apurvag@nvidia.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -n "${PLATOON_REPO_ROOT:-}" ]]; then
  repo_root_candidate=${PLATOON_REPO_ROOT}
elif [[ -f "${SLURM_SUBMIT_DIR:-}/pyproject.toml" ]]; then
  # Slurm runs a spool copy of this script; the submission directory remains
  # the checkout for an initial job.
  repo_root_candidate=${SLURM_SUBMIT_DIR}
else
  repo_root_candidate=${SCRIPT_DIR}/..
fi
if ! REPO_ROOT=$(cd "${repo_root_candidate}" 2>/dev/null && pwd -P) || \
  [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "ERROR: could not locate the Platoon checkout from ${repo_root_candidate}." >&2
  exit 2
fi
COMMON_MIXED_WRAPPER=${REPO_ROOT}/slurm-scripts/openreward-multienv-prealloc-16node-ptc-task-tracker-full.sh
DEFAULT_CONFIG=${REPO_ROOT}/plugins/openreward/platoon/openreward/configs/areal/toolathlon_tmax_swe_openhands_areal_prealloc_32node-cp-ptc-recursive-judged-r3-fp32-lm-head-bs8-efficiency.yaml
CONFIG=${1:-${DEFAULT_CONFIG}}

if [[ "${SLURM_NNODES:-32}" -ne 32 ]]; then
  echo "ERROR: this wrapper requires exactly 32 nodes; got ${SLURM_NNODES}." >&2
  exit 2
fi
[[ -x "${COMMON_MIXED_WRAPPER}" ]] || {
  echo "ERROR: missing common mixed launcher: ${COMMON_MIXED_WRAPPER}" >&2
  exit 2
}
[[ -f "${CONFIG}" ]] || {
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 2
}

# Preserve the proven 32-node recursive resource layout and make every
# continuation return through this topology-specific wrapper.
export OPENREWARD_MIXED_EXPECTED_NNODES=32
export OPENREWARD_JOB_SCRIPT=${REPO_ROOT}/slurm-scripts/openreward-multienv-prealloc-32node-ptc-recursive-bs8-efficiency.sh
export OPENREWARD_CONTROLLER_CPUS=${OPENREWARD_CONTROLLER_CPUS:-64}
export OPENREWARD_CONTROLLER_MEM=${OPENREWARD_CONTROLLER_MEM:-128G}
export PLATOON_AREAL_PREALLOC_SRUN_ARGS=${PLATOON_AREAL_PREALLOC_SRUN_ARGS:-"--unbuffered --mpi=pmi2 -K --overlap --cpu-bind=none"}
export NCCL_RAS_ENABLE=${NCCL_RAS_ENABLE:-0}

# The latest recursive run's complete steps ranged up to roughly 50 minutes.
# Do not start the first step of an allocation with less than one hour plus
# checkpoint/shutdown headroom; the adaptive guard becomes more conservative
# if recent mixed steps take longer.
export OPENREWARD_DEADLINE_INITIAL_STEP_SECONDS=${OPENREWARD_DEADLINE_INITIAL_STEP_SECONDS:-3600}
export OPENREWARD_DEADLINE_SAFETY_SECONDS=${OPENREWARD_DEADLINE_SAFETY_SECONDS:-600}

# Pin the no-delegation-bonus ablation across continuations. The YAML separately
# retains the verifier-excluding policy-subtree token-efficiency penalty.
export OPENREWARD_SUBAGENT_DELEGATION_REWARD_COEFFICIENT=0.0
export SWE_REBENCH_DATA_DIR=${REPO_ROOT}/.cache/swe-rebench-v2-filtered-verified

exec "${COMMON_MIXED_WRAPPER}" "${CONFIG}"
