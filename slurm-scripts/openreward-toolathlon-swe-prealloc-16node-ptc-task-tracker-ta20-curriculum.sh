#!/bin/bash
#SBATCH --job-name=openreward-ta-swe-16n-ta20
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --nodes=16
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --signal=B:USR1@300

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
DEFAULT_CONFIG=${REPO_ROOT}/plugins/openreward/platoon/openreward/configs/areal/toolathlon_swe_openhands_areal_prealloc_16node-cp-ptc-task-tracker-full-r3-fp32-lm-head-ta20-curriculum.yaml
CONFIG=${1:-${DEFAULT_CONFIG}}
USER_ROOT=${PLATOON_USER_ROOT:-$(cd "${REPO_ROOT}/../.." && pwd -P)}

export OPENREWARD_ENABLE_TMAX=0
export OPENREWARD_JOB_SCRIPT=${REPO_ROOT}/slurm-scripts/openreward-toolathlon-swe-prealloc-16node-ptc-task-tracker-ta20-curriculum.sh
# Override an image path inherited through `sbatch --export=ALL` so automatic
# successors cannot silently keep serving the predecessor's Toolathlon image.
export OPENREWARD_SERVER_IMAGE=${USER_ROOT}/images/openreward/apga+toolathlon-gym+18e62c0d041.sqsh
export PLATOON_REPO_ROOT=${REPO_ROOT}

exec "${COMMON_WRAPPER}" "${CONFIG}"
