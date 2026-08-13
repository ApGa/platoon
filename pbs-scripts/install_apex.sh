#!/usr/bin/env bash
# Build and install the exact NVIDIA Apex revision required by the ABCI
# Megatron runtime. The compiled wheel is cached by all ABI-relevant inputs.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOCK_FILE=${PLATOON_RUNTIME_LOCK_FILE:-${SCRIPT_DIR}/../containers/abci-training-runtime.lock}

die() {
  echo "[install-apex] ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  install_apex.sh VENV
  install_apex.sh --verify-only VENV
  install_apex.sh --dry-run [VENV]
  install_apex.sh --print-provenance

Environment:
  APEX_WHEEL_CACHE       Shared compiled-wheel cache (required for install)
  OPENREWARD_UV_BIN      uv executable (default: uv)
  APEX_BUILD_SCRATCH     Node-local build root (default: PBS_LOCALDIR/TMPDIR)
  APEX_BUILD_LOCK_SECS   Cache-lock timeout (default: 7200)
EOF
}

[[ -r "${LOCK_FILE}" ]] || die "runtime provenance lock not found: ${LOCK_FILE}"
# shellcheck disable=SC1090
source "${LOCK_FILE}"

print_provenance() {
  printf 'apex_git_url=%s\n' "${APEX_GIT_URL}"
  printf 'apex_git_commit=%s\n' "${APEX_GIT_COMMIT}"
  printf 'build_base_amd64_manifest=%s\n' "${SGLANG_BASE_AMD64_MANIFEST_DIGEST}"
  printf 'python=%s\n' "${PYTHON_VERSION}"
  printf 'torch=%s\n' "${PYTORCH_VERSION}"
  printf 'cuda=%s\n' "${CUDA_TOOLKIT_VERSION}"
  printf 'torch_cuda_arch_list=%s\n' "${TORCH_CUDA_ARCH_LIST}"
}

emit_artifact_manifest() {
  local wheel_name=$1
  local wheel_sha=$2
  print_provenance
  printf 'wheel=%s\n' "${wheel_name}"
  printf 'wheel_sha256=%s\n' "${wheel_sha}"
}

validate_python_abi() {
  local python=$1
  "${python}" -I - "${PYTHON_VERSION}" "${PYTORCH_VERSION}" <<'PY'
import sys
from importlib import metadata

expected_python, expected_torch = sys.argv[1:]
actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
assert actual_python == expected_python, (actual_python, expected_python)
assert metadata.version("torch") == expected_torch

import torch

assert torch.version.cuda == "12.9", torch.version.cuda
PY
}

verify_install() {
  local target=$1
  local python=${target}/bin/python
  local marker=${target}/.platoon-apex-build
  [[ -x "${python}" ]] || die "target Python is not executable: ${python}"
  [[ -f "${marker}" ]] || die "Apex provenance marker is missing: ${marker}"
  grep -Fxq "apex_git_commit=${APEX_GIT_COMMIT}" "${marker}" || \
    die "Apex marker does not name the pinned commit"
  grep -Fxq "build_base_amd64_manifest=${SGLANG_BASE_AMD64_MANIFEST_DIGEST}" "${marker}" || \
    die "Apex marker does not name the pinned build base"
  validate_python_abi "${python}"
  CUDA_VISIBLE_DEVICES= "${python}" -I - <<'PY'
import apex
import fused_weight_gradient_mlp_cuda

print("Apex import smoke test passed", apex.__file__)
PY
}

mode=install
case "${1:-}" in
  --print-provenance)
    print_provenance
    exit 0
    ;;
  --dry-run)
    mode=dry-run
    shift
    ;;
  --verify-only)
    mode=verify-only
    shift
    ;;
  -h|--help)
    usage
    exit 0
    ;;
esac

target=${1:-}
[[ $# -le 1 ]] || die "unexpected arguments; run --help for usage"

if [[ "${mode}" == dry-run ]]; then
  target=${target:-/path/to/cp312-venv}
  print_provenance
  printf 'target=%s\n' "${target}"
  printf 'cache=%s\n' "${APEX_WHEEL_CACHE:-<APEX_WHEEL_CACHE must be set>}"
  printf '%s\n' 'plan=validate ABI; fetch exact Git commit; build CUDA/C++ wheel; atomically cache; install; import-smoke'
  exit 0
fi

[[ -n "${target}" ]] || die "VENV is required; run --help for usage"
if [[ "${mode}" == verify-only ]]; then
  verify_install "${target}"
  exit 0
fi

[[ -n "${APEX_WHEEL_CACHE:-}" ]] || die "APEX_WHEEL_CACHE must name a shared artifact cache"
[[ -z "${APEX_GIT_REF:-}" || "${APEX_GIT_REF}" == "${APEX_GIT_COMMIT}" ]] || \
  die "APEX_GIT_REF must be the pinned commit ${APEX_GIT_COMMIT}"

python=${target}/bin/python
[[ -x "${python}" ]] || die "target Python is not executable: ${python}"
validate_python_abi "${python}"

uv_bin=${OPENREWARD_UV_BIN:-${UV_BIN:-uv}}
if [[ "${uv_bin}" == */* ]]; then
  [[ -x "${uv_bin}" ]] || die "uv executable not found: ${uv_bin}"
else
  command -v "${uv_bin}" >/dev/null 2>&1 || die "uv executable not found: ${uv_bin}"
fi
for command_name in git flock nvcc sha256sum; do
  command -v "${command_name}" >/dev/null 2>&1 || die "required command is missing: ${command_name}"
done
nvcc --version | grep -Fq 'release 12.9' || die "nvcc is not CUDA 12.9"

cache_identity=$(print_provenance | sha256sum | awk '{print $1}')
cache_key="apex-${APEX_GIT_COMMIT:0:12}-${cache_identity}"
artifact_dir=${APEX_WHEEL_CACHE}/${cache_key}
lock_seconds=${APEX_BUILD_LOCK_SECS:-7200}
mkdir -p "${APEX_WHEEL_CACHE}"
exec 9>"${APEX_WHEEL_CACHE}/.${cache_key}.lock"
flock -w "${lock_seconds}" 9 || die "timed out waiting for Apex artifact lock"

cached_wheel=
if [[ -f "${artifact_dir}/MANIFEST" ]]; then
  shopt -s nullglob
  wheels=("${artifact_dir}"/*.whl)
  shopt -u nullglob
  if [[ ${#wheels[@]} -eq 1 ]]; then
    cached_wheel=${wheels[0]}
    expected_sha=$(sed -n 's/^wheel_sha256=//p' "${artifact_dir}/MANIFEST")
    actual_sha=$(sha256sum "${cached_wheel}" | awk '{print $1}')
    [[ -n "${expected_sha}" && "${actual_sha}" == "${expected_sha}" ]] || \
      die "cached Apex wheel failed its recorded digest: ${cached_wheel}"
    grep -Fxq "apex_git_commit=${APEX_GIT_COMMIT}" "${artifact_dir}/MANIFEST" || \
      die "cached Apex manifest has the wrong source commit"
    grep -Fxq "build_base_amd64_manifest=${SGLANG_BASE_AMD64_MANIFEST_DIGEST}" \
      "${artifact_dir}/MANIFEST" || die "cached Apex manifest has the wrong build base"
  fi
fi

build_root=
stage_artifact=
cleanup() {
  local status=$?
  if [[ -n "${stage_artifact}" && -d "${stage_artifact}" ]]; then
    rm -rf -- "${stage_artifact}"
  fi
  if [[ -n "${build_root}" && -d "${build_root}" ]]; then
    rm -rf -- "${build_root}"
  fi
  return "${status}"
}
trap cleanup EXIT

if [[ -z "${cached_wheel}" ]]; then
  if [[ -e "${artifact_dir}" ]]; then
    mv "${artifact_dir}" "${artifact_dir}.incomplete.$(date +%s).$$"
  fi

  scratch_parent=${APEX_BUILD_SCRATCH:-${PBS_LOCALDIR:-${TMPDIR:-/tmp}}}
  mkdir -p "${scratch_parent}/platoon-runtime-build"
  build_root=$(mktemp -d "${scratch_parent}/platoon-runtime-build/apex.XXXXXXXX")
  source_dir=${build_root}/source
  wheel_dir=${build_root}/wheel
  mkdir -p "${source_dir}" "${wheel_dir}"

  git -C "${source_dir}" init --quiet
  git -C "${source_dir}" remote add origin "${APEX_GIT_URL}"
  git -C "${source_dir}" fetch --quiet --depth=1 origin "${APEX_GIT_COMMIT}"
  git -C "${source_dir}" checkout --quiet --detach FETCH_HEAD
  [[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${APEX_GIT_COMMIT}" ]] || \
    die "fetched Apex revision does not match the provenance lock"
  git -C "${source_dir}" submodule update --init --recursive --depth=1

  export APEX_CPP_EXT=1
  export APEX_CUDA_EXT=1
  export APEX_PARALLEL_BUILD=${APEX_PARALLEL_BUILD:-${MAX_BUILD_JOBS}}
  export NVCC_APPEND_FLAGS=${NVCC_APPEND_FLAGS:---threads 4}
  export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
  export MAX_JOBS=${MAX_JOBS:-${MAX_BUILD_JOBS}}

  "${uv_bin}" build --wheel --no-build-isolation --python "${python}" \
    --out-dir "${wheel_dir}" "${source_dir}"

  shopt -s nullglob
  built_wheels=("${wheel_dir}"/*.whl)
  shopt -u nullglob
  [[ ${#built_wheels[@]} -eq 1 ]] || die "Apex build produced ${#built_wheels[@]} wheels, expected one"
  built_wheel=${built_wheels[0]}
  built_sha=$(sha256sum "${built_wheel}" | awk '{print $1}')

  stage_artifact=$(mktemp -d "${APEX_WHEEL_CACHE}/.${cache_key}.staging.XXXXXXXX")
  wheel_name=${built_wheel##*/}
  cp "${built_wheel}" "${stage_artifact}/${wheel_name}"
  [[ "$(sha256sum "${stage_artifact}/${wheel_name}" | awk '{print $1}')" == "${built_sha}" ]] || \
    die "Apex wheel changed while publishing"
  emit_artifact_manifest "${wheel_name}" "${built_sha}" >"${stage_artifact}/MANIFEST"
  chmod 0444 "${stage_artifact}/${wheel_name}" "${stage_artifact}/MANIFEST"
  mv "${stage_artifact}" "${artifact_dir}"
  stage_artifact=
  cached_wheel=${artifact_dir}/${wheel_name}
fi

"${uv_bin}" pip install --python "${python}" --no-deps --reinstall "${cached_wheel}"
cp "${artifact_dir}/MANIFEST" "${target}/.platoon-apex-build"
verify_install "${target}"
