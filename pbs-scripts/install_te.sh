#!/usr/bin/env bash
# Install Transformer Engine 2.12.0 from hash-locked PyPI artifacts. The Torch
# binding is source-built against the exact target PyTorch/CUDA ABI and cached.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOCK_FILE=${PLATOON_RUNTIME_LOCK_FILE:-${SCRIPT_DIR}/../containers/abci-training-runtime.lock}
CONSTRAINTS_FILE=${TE_CONSTRAINTS_FILE:-${SCRIPT_DIR}/../containers/transformer-engine-build-constraints.txt}

die() {
  echo "[install-te] ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  install_te.sh VENV
  install_te.sh --verify-only VENV
  install_te.sh --dry-run [VENV]
  install_te.sh --print-provenance

Environment:
  TE_WHEEL_CACHE       Shared download/compiled-wheel cache (required)
  OPENREWARD_UV_BIN    uv executable (default: uv)
  TE_BUILD_SCRATCH     Node-local build root (default: PBS_LOCALDIR/TMPDIR)
  TE_BUILD_LOCK_SECS   Cache-lock timeout (default: 7200)
EOF
}

[[ -r "${LOCK_FILE}" ]] || die "runtime provenance lock not found: ${LOCK_FILE}"
# shellcheck disable=SC1090
source "${LOCK_FILE}"

print_provenance() {
  printf 'transformer_engine_version=%s\n' "${TRANSFORMER_ENGINE_VERSION}"
  printf 'transformer_engine_git_tag=%s\n' "${TRANSFORMER_ENGINE_GIT_TAG}"
  printf 'transformer_engine_git_commit=%s\n' "${TRANSFORMER_ENGINE_GIT_COMMIT}"
  printf 'transformer_engine_wheel_sha256=%s\n' "${TRANSFORMER_ENGINE_WHEEL_SHA256}"
  printf 'transformer_engine_cu12_wheel_sha256=%s\n' "${TRANSFORMER_ENGINE_CU12_WHEEL_SHA256}"
  printf 'transformer_engine_torch_sdist_sha256=%s\n' "${TRANSFORMER_ENGINE_TORCH_SDIST_SHA256}"
  printf 'onnx_wheel_sha256=%s\n' "${ONNX_WHEEL_SHA256}"
  printf 'onnxscript_wheel_sha256=%s\n' "${ONNXSCRIPT_WHEEL_SHA256}"
  printf 'onnx_ir_wheel_sha256=%s\n' "${ONNX_IR_WHEEL_SHA256}"
  printf 'nvdlfw_inspect_wheel_sha256=%s\n' "${NVDLFW_INSPECT_WHEEL_SHA256}"
  printf 'build_base_amd64_manifest=%s\n' "${SGLANG_BASE_AMD64_MANIFEST_DIGEST}"
  printf 'python=%s\n' "${PYTHON_VERSION}"
  printf 'torch=%s\n' "${PYTORCH_VERSION}"
  printf 'cuda=%s\n' "${CUDA_TOOLKIT_VERSION}"
  printf 'nvte_cuda_archs=%s\n' "${NVTE_CUDA_ARCHS}"
  printf '%s\n' 'nvte_userbuffers=disabled'
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
  "${python}" -I - "${PYTHON_VERSION}" "${PYTORCH_VERSION}" "${NVIDIA_CUDNN_CU12_VERSION}" <<'PY'
import sys
from importlib import metadata

expected_python, expected_torch, expected_cudnn = sys.argv[1:]
actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
assert actual_python == expected_python, (actual_python, expected_python)
assert metadata.version("torch") == expected_torch
assert metadata.version("nvidia-cudnn-cu12") == expected_cudnn

import torch

assert torch.version.cuda == "12.9", torch.version.cuda
PY
}

verify_install() {
  local target=$1
  local python=${target}/bin/python
  local marker=${target}/.platoon-transformer-engine-build
  [[ -x "${python}" ]] || die "target Python is not executable: ${python}"
  [[ -f "${marker}" ]] || die "Transformer Engine provenance marker is missing: ${marker}"
  grep -Fxq "transformer_engine_version=${TRANSFORMER_ENGINE_VERSION}" "${marker}" || \
    die "Transformer Engine marker has the wrong version"
  grep -Fxq "transformer_engine_torch_sdist_sha256=${TRANSFORMER_ENGINE_TORCH_SDIST_SHA256}" "${marker}" || \
    die "Transformer Engine marker has the wrong Torch source artifact"
  grep -Fxq "build_base_amd64_manifest=${SGLANG_BASE_AMD64_MANIFEST_DIGEST}" "${marker}" || \
    die "Transformer Engine marker has the wrong build base"
  validate_python_abi "${python}"
  CUDA_VISIBLE_DEVICES= "${python}" -I - \
    "${TRANSFORMER_ENGINE_VERSION}" \
    "${ONNX_VERSION}" \
    "${ONNXSCRIPT_VERSION}" \
    "${ONNX_IR_VERSION}" \
    "${NVDLFW_INSPECT_VERSION}" <<'PY'
import sys
from importlib import metadata

expected, onnx_version, onnxscript_version, onnx_ir_version, inspect_version = sys.argv[1:]
for distribution in (
    "transformer-engine",
    "transformer-engine-cu12",
    "transformer-engine-torch",
):
    assert metadata.version(distribution) == expected, (distribution, metadata.version(distribution))
for distribution, version in (
    ("onnx", onnx_version),
    ("onnxscript", onnxscript_version),
    ("onnx-ir", onnx_ir_version),
    ("nvdlfw-inspect", inspect_version),
):
    assert metadata.version(distribution) == version, (distribution, metadata.version(distribution))

import onnx
import transformer_engine.pytorch

assert hasattr(onnx, "TensorProto")
print("Transformer Engine import smoke test passed")
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
  printf 'cache=%s\n' "${TE_WHEEL_CACHE:-<TE_WHEEL_CACHE must be set>}"
  printf 'constraints=%s\n' "${CONSTRAINTS_FILE}"
  printf '%s\n' 'plan=validate ABI; hash-check three official artifacts; pin missing dependencies; build Torch binding; atomically cache; install; import-smoke'
  exit 0
fi

[[ -n "${target}" ]] || die "VENV is required; run --help for usage"
if [[ "${mode}" == verify-only ]]; then
  verify_install "${target}"
  exit 0
fi

[[ -n "${TE_WHEEL_CACHE:-}" ]] || die "TE_WHEEL_CACHE must name a shared artifact cache"
[[ -r "${CONSTRAINTS_FILE}" ]] || die "Transformer Engine constraints not found: ${CONSTRAINTS_FILE}"
for constraint in \
  "onnx==${ONNX_VERSION}" \
  "onnxscript==${ONNXSCRIPT_VERSION}" \
  "onnx-ir==${ONNX_IR_VERSION}" \
  "nvdlfw-inspect==${NVDLFW_INSPECT_VERSION}"; do
  grep -Fxq "${constraint}" "${CONSTRAINTS_FILE}" || \
    die "dependency constraint and provenance lock disagree: ${constraint}"
done
[[ -z "${TE_VERSION:-}" || "${TE_VERSION}" == "${TRANSFORMER_ENGINE_VERSION}" ]] || \
  die "TE_VERSION must be the pinned version ${TRANSFORMER_ENGINE_VERSION}"

python=${target}/bin/python
[[ -x "${python}" ]] || die "target Python is not executable: ${python}"
validate_python_abi "${python}"

uv_bin=${OPENREWARD_UV_BIN:-${UV_BIN:-uv}}
if [[ "${uv_bin}" == */* ]]; then
  [[ -x "${uv_bin}" ]] || die "uv executable not found: ${uv_bin}"
else
  command -v "${uv_bin}" >/dev/null 2>&1 || die "uv executable not found: ${uv_bin}"
fi
for command_name in curl flock nvcc sha256sum; do
  command -v "${command_name}" >/dev/null 2>&1 || die "required command is missing: ${command_name}"
done
nvcc --version | grep -Fq 'release 12.9' || die "nvcc is not CUDA 12.9"

cache_identity=$(print_provenance | sha256sum | awk '{print $1}')
cache_key="te-${TRANSFORMER_ENGINE_VERSION}-${cache_identity}"
artifact_dir=${TE_WHEEL_CACHE}/${cache_key}
download_dir=${TE_WHEEL_CACHE}/downloads
lock_seconds=${TE_BUILD_LOCK_SECS:-7200}
mkdir -p "${TE_WHEEL_CACHE}"
exec 9>"${TE_WHEEL_CACHE}/.${cache_key}.lock"
flock -w "${lock_seconds}" 9 || die "timed out waiting for Transformer Engine artifact lock"

ensure_download() {
  local url=$1
  local expected_sha=$2
  local filename=${url##*/}
  local destination=${download_dir}/${filename}
  local temporary

  mkdir -p "${download_dir}"
  if [[ -f "${destination}" ]]; then
    [[ "$(sha256sum "${destination}" | awk '{print $1}')" == "${expected_sha}" ]] || \
      die "cached download failed its pinned digest: ${destination}"
  else
    temporary=${download_dir}/.${filename}.partial.$$
    curl --proto '=https' --tlsv1.2 --fail --location \
      --retry 5 --retry-all-errors --output "${temporary}" "${url}"
    [[ "$(sha256sum "${temporary}" | awk '{print $1}')" == "${expected_sha}" ]] || {
      rm -f -- "${temporary}"
      die "download failed its pinned digest: ${filename}"
    }
    chmod 0444 "${temporary}"
    mv "${temporary}" "${destination}"
  fi
  printf '%s\n' "${destination}"
}

te_wheel=$(ensure_download "${TRANSFORMER_ENGINE_WHEEL_URL}" "${TRANSFORMER_ENGINE_WHEEL_SHA256}")
te_cu12_wheel=$(ensure_download "${TRANSFORMER_ENGINE_CU12_WHEEL_URL}" "${TRANSFORMER_ENGINE_CU12_WHEEL_SHA256}")
te_torch_sdist=$(ensure_download "${TRANSFORMER_ENGINE_TORCH_SDIST_URL}" "${TRANSFORMER_ENGINE_TORCH_SDIST_SHA256}")
onnx_wheel=$(ensure_download "${ONNX_WHEEL_URL}" "${ONNX_WHEEL_SHA256}")
onnxscript_wheel=$(ensure_download "${ONNXSCRIPT_WHEEL_URL}" "${ONNXSCRIPT_WHEEL_SHA256}")
onnx_ir_wheel=$(ensure_download "${ONNX_IR_WHEEL_URL}" "${ONNX_IR_WHEEL_SHA256}")
nvdlfw_inspect_wheel=$(ensure_download "${NVDLFW_INSPECT_WHEEL_URL}" "${NVDLFW_INSPECT_WHEEL_SHA256}")

"${uv_bin}" pip install --no-config --python "${python}" --no-deps --reinstall \
  "${onnx_wheel}" \
  "${onnxscript_wheel}" \
  "${onnx_ir_wheel}" \
  "${nvdlfw_inspect_wheel}"
# Install and verify the metapackage independently.  uv 0.12.1 has been
# observed to resolve only the platform payload when both similarly named
# local TE wheels are supplied in one invocation inside the SIF, despite
# resolving both on the host.  Separate operations make omission impossible
# and give each pinned artifact its own postcondition.
"${uv_bin}" pip install --no-config --python "${python}" --no-deps --reinstall "${te_wheel}"
"${python}" -I - "${TRANSFORMER_ENGINE_VERSION}" <<'PY'
import sys
from importlib import metadata

assert metadata.version("transformer-engine") == sys.argv[1]
PY
"${uv_bin}" pip install --no-config --python "${python}" --no-deps --reinstall "${te_cu12_wheel}"
"${python}" -I - "${TRANSFORMER_ENGINE_VERSION}" <<'PY'
import sys
from importlib import metadata

expected = sys.argv[1]
assert metadata.version("transformer-engine") == expected
assert metadata.version("transformer-engine-cu12") == expected
PY

cached_wheel=
if [[ -f "${artifact_dir}/MANIFEST" ]]; then
  shopt -s nullglob
  wheels=("${artifact_dir}"/transformer_engine_torch-*.whl)
  shopt -u nullglob
  if [[ ${#wheels[@]} -eq 1 ]]; then
    cached_wheel=${wheels[0]}
    expected_sha=$(sed -n 's/^wheel_sha256=//p' "${artifact_dir}/MANIFEST")
    actual_sha=$(sha256sum "${cached_wheel}" | awk '{print $1}')
    [[ -n "${expected_sha}" && "${actual_sha}" == "${expected_sha}" ]] || \
      die "cached Transformer Engine Torch wheel failed its recorded digest"
    grep -Fxq "transformer_engine_torch_sdist_sha256=${TRANSFORMER_ENGINE_TORCH_SDIST_SHA256}" \
      "${artifact_dir}/MANIFEST" || die "cached Transformer Engine manifest has the wrong source artifact"
    grep -Fxq "build_base_amd64_manifest=${SGLANG_BASE_AMD64_MANIFEST_DIGEST}" \
      "${artifact_dir}/MANIFEST" || die "cached Transformer Engine manifest has the wrong build base"
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

  scratch_parent=${TE_BUILD_SCRATCH:-${PBS_LOCALDIR:-${TMPDIR:-/tmp}}}
  mkdir -p "${scratch_parent}/platoon-runtime-build"
  build_root=$(mktemp -d "${scratch_parent}/platoon-runtime-build/te.XXXXXXXX")
  wheel_dir=${build_root}/wheel
  mkdir -p "${wheel_dir}"

  cudnn_root=$("${python}" -I - <<'PY'
import site
from pathlib import Path

for root in map(Path, site.getsitepackages()):
    candidate = root / "nvidia" / "cudnn"
    if (candidate / "include").is_dir() and (candidate / "lib").is_dir():
        print(candidate)
        break
else:
    raise SystemExit("could not locate the nvidia-cudnn-cu12 package root")
PY
  )
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
  export CUDNN_PATH=${CUDNN_PATH:-${cudnn_root}}
  export CPATH=${CUDNN_PATH}/include${CPATH:+:${CPATH}}
  export LIBRARY_PATH=${CUDNN_PATH}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}
  export LD_LIBRARY_PATH=${CUDNN_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  export NVTE_FRAMEWORK=pytorch
  export NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS}
  export MAX_JOBS=${MAX_JOBS:-${MAX_BUILD_JOBS}}
  unset NVTE_WITH_USERBUFFERS

  "${uv_bin}" build --wheel --no-build-isolation --python "${python}" \
    --out-dir "${wheel_dir}" "${te_torch_sdist}"

  shopt -s nullglob
  built_wheels=("${wheel_dir}"/transformer_engine_torch-*.whl)
  shopt -u nullglob
  [[ ${#built_wheels[@]} -eq 1 ]] || \
    die "Transformer Engine build produced ${#built_wheels[@]} Torch wheels, expected one"
  built_wheel=${built_wheels[0]}
  built_sha=$(sha256sum "${built_wheel}" | awk '{print $1}')

  stage_artifact=$(mktemp -d "${TE_WHEEL_CACHE}/.${cache_key}.staging.XXXXXXXX")
  wheel_name=${built_wheel##*/}
  cp "${built_wheel}" "${stage_artifact}/${wheel_name}"
  [[ "$(sha256sum "${stage_artifact}/${wheel_name}" | awk '{print $1}')" == "${built_sha}" ]] || \
    die "Transformer Engine wheel changed while publishing"
  emit_artifact_manifest "${wheel_name}" "${built_sha}" >"${stage_artifact}/MANIFEST"
  chmod 0444 "${stage_artifact}/${wheel_name}" "${stage_artifact}/MANIFEST"
  mv "${stage_artifact}" "${artifact_dir}"
  stage_artifact=
  cached_wheel=${artifact_dir}/${wheel_name}
fi

"${uv_bin}" pip install --no-config --python "${python}" --no-deps --reinstall "${cached_wheel}"
# The OpenReward lock intentionally overrides several transitive metadata
# constraints (SGLang/OpenHands/Megatron pins and optional CUDA packages).
# A blanket `pip check` therefore rejects the known-good locked baseline and
# says nothing specific about this TE artifact.  `verify_install` below checks
# the exact TE/ONNX versions, build provenance, target Python/Torch/CUDA ABI,
# and imports the compiled binding.
cp "${artifact_dir}/MANIFEST" "${target}/.platoon-transformer-engine-build"
verify_install "${target}"
