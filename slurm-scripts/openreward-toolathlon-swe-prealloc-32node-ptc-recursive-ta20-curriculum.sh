#!/bin/bash
#SBATCH --job-name=openreward-ta-swe-32n-rec-ta20
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --nodes=32
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/apurvag/logs/openreward-ta-swe-32n-rec-ta20-%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/projects/nvr_lacr_llm/users/apurvag/logs/openreward-ta-swe-32n-rec-ta20-%j.err
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
COMMON_WRAPPER=${REPO_ROOT}/slurm-scripts/openreward-multienv-prealloc-16node-ptc-task-tracker-full.sh
DEFAULT_CONFIG=${REPO_ROOT}/plugins/openreward/platoon/openreward/configs/areal/toolathlon_swe_openhands_areal_prealloc_32node-cp-ptc-recursive-judged-r3-fp32-lm-head-bs8-efficiency-ta20-curriculum.yaml
CONFIG=${1:-${DEFAULT_CONFIG}}

if [[ "${SLURM_NNODES:-32}" -ne 32 ]]; then
  echo "ERROR: this wrapper requires exactly 32 nodes; got ${SLURM_NNODES}." >&2
  exit 2
fi

# Preserve the latest recursive topology, deadline, keepalive, and reward
# ablation settings while omitting the unused TMax service pool.
export OPENREWARD_MIXED_EXPECTED_NNODES=32
export OPENREWARD_JOB_SCRIPT=${REPO_ROOT}/slurm-scripts/openreward-toolathlon-swe-prealloc-32node-ptc-recursive-ta20-curriculum.sh
export OPENREWARD_CONTROLLER_CPUS=${OPENREWARD_CONTROLLER_CPUS:-64}
export OPENREWARD_CONTROLLER_MEM=${OPENREWARD_CONTROLLER_MEM:-128G}
export PLATOON_AREAL_PREALLOC_SRUN_ARGS=${PLATOON_AREAL_PREALLOC_SRUN_ARGS:-"--unbuffered --mpi=pmi2 -K --overlap --cpu-bind=none"}
export NCCL_RAS_ENABLE=${NCCL_RAS_ENABLE:-0}
export OPENREWARD_DEADLINE_INITIAL_STEP_SECONDS=${OPENREWARD_DEADLINE_INITIAL_STEP_SECONDS:-1800}
export OPENREWARD_DEADLINE_SAFETY_SECONDS=${OPENREWARD_DEADLINE_SAFETY_SECONDS:-600}
export OPENREWARD_SUBAGENT_DELEGATION_REWARD_COEFFICIENT=0.0
export OPENREWARD_ENABLE_TMAX=0
export SWE_REBENCH_DATA_DIR=${REPO_ROOT}/.cache/swe-rebench-v2-filtered-verified
export PLATOON_REPO_ROOT=${REPO_ROOT}

exec "${COMMON_WRAPPER}" "${CONFIG}"
