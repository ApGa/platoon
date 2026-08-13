#!/usr/bin/env bash
# Build the digest-pinned Platoon training SIF on an ABCI compute node, then
# publish it atomically to shared storage with a SHA-256 sidecar.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DEFINITION=${PLATOON_TRAINING_DEFINITION:-${REPO_ROOT}/containers/abci-training.def}
LOCK_FILE=${PLATOON_RUNTIME_LOCK_FILE:-${REPO_ROOT}/containers/abci-training-runtime.lock}

die() {
  echo "[build-training-sif] ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  build_training_sif.sh --output /shared/path/platoon-training.sif
  build_training_sif.sh --dry-run [--output PATH]
  build_training_sif.sh --print-provenance

The real build must run inside an ABCI PBS compute allocation because the OCI
unpack/build scratch is placed under PBS_LOCALDIR. Existing output images are
never overwritten.
EOF
}

[[ -r "${LOCK_FILE}" ]] || die "runtime provenance lock not found: ${LOCK_FILE}"
# shellcheck disable=SC1090
source "${LOCK_FILE}"

print_provenance() {
  printf 'base_tag=%s\n' "${SGLANG_BASE_TAG}"
  printf 'base_index_digest=%s\n' "${SGLANG_BASE_INDEX_DIGEST}"
  printf 'base_amd64_manifest_digest=%s\n' "${SGLANG_BASE_AMD64_MANIFEST_DIGEST}"
  printf 'cuda=%s\n' "${CUDA_TOOLKIT_VERSION}"
  printf 'python=%s\n' "${PYTHON_VERSION}"
  printf 'torch=%s\n' "${PYTORCH_VERSION}"
  printf 'apex_git_commit=%s\n' "${APEX_GIT_COMMIT}"
  printf 'transformer_engine=%s\n' "${TRANSFORMER_ENGINE_VERSION}"
  printf 'transformer_engine_git_commit=%s\n' "${TRANSFORMER_ENGINE_GIT_COMMIT}"
}

mode=build
output=${PLATOON_TRAINING_SIF:-}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      [[ $# -ge 2 ]] || die "--output requires a path"
      output=$2
      shift 2
      ;;
    --dry-run)
      mode=dry-run
      shift
      ;;
    --print-provenance)
      print_provenance
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ "${mode}" == dry-run ]]; then
  print_provenance
  printf 'definition=%s\n' "${DEFINITION}"
  printf 'output=%s\n' "${output:-<PLATOON_TRAINING_SIF or --output required>}"
  printf '%s\n' 'command=singularity build --fakeroot LOCAL_SIF containers/abci-training.def'
  printf '%s\n' 'publish=validate labels and runtime; copy to same-directory partial; verify SHA-256; atomic rename'
  exit 0
fi

[[ -n "${output}" ]] || die "set PLATOON_TRAINING_SIF or pass --output"
[[ -f "${DEFINITION}" ]] || die "Singularity definition not found: ${DEFINITION}"
[[ -n "${PBS_JOBID:-}" ]] || die "refusing a large build outside a PBS compute allocation"
[[ -d "${PBS_LOCALDIR:-}" ]] || die "PBS_LOCALDIR is not an existing node-local directory"
[[ ! -e "${output}" ]] || die "output already exists; refusing to overwrite it: ${output}"

if command -v module >/dev/null 2>&1; then
  module load singularity-ce/4.4.1
fi
singularity_bin=${SINGULARITY_BIN:-singularity}
if [[ "${singularity_bin}" == */* ]]; then
  [[ -x "${singularity_bin}" ]] || die "Singularity executable not found: ${singularity_bin}"
else
  command -v "${singularity_bin}" >/dev/null 2>&1 || die "Singularity executable not found: ${singularity_bin}"
fi
for command_name in flock sha256sum; do
  command -v "${command_name}" >/dev/null 2>&1 || die "required command is missing: ${command_name}"
done

output_dir=$(dirname -- "${output}")
output_name=$(basename -- "${output}")
mkdir -p "${output_dir}"
exec 9>"${output}.build.lock"
flock -w "${PLATOON_SIF_BUILD_LOCK_SECS:-21600}" 9 || die "timed out waiting for SIF build lock"
[[ ! -e "${output}" ]] || die "another builder published the output while this job waited: ${output}"

build_root=$(mktemp -d "${PBS_LOCALDIR}/platoon-training-sif.XXXXXXXX")
local_sif=${build_root}/platoon-training.sif
publish_tmp=${output_dir}/.${output_name}.partial.${PBS_JOBID}.$$
digest_tmp=${output_dir}/.${output_name}.sha256.partial.${PBS_JOBID}.$$
cleanup() {
  local status=$?
  rm -f -- "${publish_tmp}" "${digest_tmp}"
  if [[ -d "${build_root}" ]]; then
    rm -rf -- "${build_root}"
  fi
  return "${status}"
}
trap cleanup EXIT

export SINGULARITY_CACHEDIR=${build_root}/singularity-cache
export SINGULARITY_TMPDIR=${build_root}/singularity-tmp
mkdir -p "${SINGULARITY_CACHEDIR}" "${SINGULARITY_TMPDIR}"

echo "[build-training-sif] building on $(hostname) at $(date -Is)"
(
  cd "${REPO_ROOT}"
  "${singularity_bin}" build --fakeroot "${local_sif}" "${DEFINITION}"
)

"${singularity_bin}" inspect "${local_sif}" >/dev/null
labels=$("${singularity_bin}" inspect --labels "${local_sif}")
grep -Fq "${SGLANG_BASE_AMD64_MANIFEST_DIGEST}" <<<"${labels}" || \
  die "built SIF does not record the pinned base manifest"
grep -Fq "${APEX_GIT_COMMIT}" <<<"${labels}" || die "built SIF does not record the pinned Apex commit"
grep -Fq "${TRANSFORMER_ENGINE_GIT_COMMIT}" <<<"${labels}" || \
  die "built SIF does not record the pinned Transformer Engine commit"
"${singularity_bin}" exec --cleanenv --no-home "${local_sif}" /bin/bash -lc \
  'python3 -c "from importlib.metadata import version; import torch; assert torch.__version__ == '\''2.9.1+cu129'\''; assert torch.version.cuda == '\''12.9'\''; assert version('\''sglang'\'') == '\''0.5.10.post1'\''"; nvcc --version | grep -F "release 12.9"'

cp "${local_sif}" "${publish_tmp}"
local_sha=$(sha256sum "${local_sif}" | awk '{print $1}')
published_sha=$(sha256sum "${publish_tmp}" | awk '{print $1}')
[[ "${local_sha}" == "${published_sha}" ]] || die "SIF changed while copying to shared storage"
chmod 0444 "${publish_tmp}"
printf '%s  %s\n' "${published_sha}" "${output_name}" >"${digest_tmp}"
chmod 0444 "${digest_tmp}"
mv "${publish_tmp}" "${output}"
mv "${digest_tmp}" "${output}.sha256"

echo "[build-training-sif] published ${output}"
echo "[build-training-sif] sha256=${published_sha}"
