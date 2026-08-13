#!/usr/bin/env bash
# Static and post-build checks for the ABCI training runtime bundle.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
LOCK_FILE=${PLATOON_RUNTIME_LOCK_FILE:-${REPO_ROOT}/containers/abci-training-runtime.lock}
DEFINITION=${PLATOON_TRAINING_DEFINITION:-${REPO_ROOT}/containers/abci-training.def}

die() {
  echo "[verify-training-runtime] ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  verify_training_runtime.sh --static
  verify_training_runtime.sh --sif IMAGE [--venv VENV] [--with-gpu]

--static performs no network access, image build, or package installation.
--with-gpu adds Singularity --nv and checks that CUDA sees an H200-class device.
EOF
}

static_check() {
  local file
  for file in \
    "${LOCK_FILE}" \
    "${DEFINITION}" \
    "${REPO_ROOT}/containers/transformer-engine-build-constraints.txt" \
    "${SCRIPT_DIR}/install_apex.sh" \
    "${SCRIPT_DIR}/install_te.sh" \
    "${SCRIPT_DIR}/build-training-runtime.pbs" \
    "${SCRIPT_DIR}/build_training_sif.sh"; do
    [[ -f "${file}" ]] || die "required runtime artifact is missing: ${file}"
  done

  # shellcheck disable=SC1090
  source "${LOCK_FILE}"
  [[ "${SGLANG_BASE_INDEX_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || \
    die "invalid SGLang OCI index digest in provenance lock"
  [[ "${SGLANG_BASE_AMD64_MANIFEST_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]] || \
    die "invalid SGLang amd64 manifest digest in provenance lock"
  [[ "${SGLANG_BASE_REFERENCE}" == "docker.io/lmsysorg/sglang@${SGLANG_BASE_AMD64_MANIFEST_DIGEST}" ]] || \
    die "SGLang base reference and platform manifest disagree"
  [[ "${APEX_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || die "invalid Apex commit in provenance lock"
  [[ "${TRANSFORMER_ENGINE_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || \
    die "invalid Transformer Engine commit in provenance lock"
  for digest in \
    "${TRANSFORMER_ENGINE_WHEEL_SHA256}" \
    "${TRANSFORMER_ENGINE_CU12_WHEEL_SHA256}" \
    "${TRANSFORMER_ENGINE_TORCH_SDIST_SHA256}" \
    "${ONNX_WHEEL_SHA256}" \
    "${ONNXSCRIPT_WHEEL_SHA256}" \
    "${ONNX_IR_WHEEL_SHA256}" \
    "${NVDLFW_INSPECT_WHEEL_SHA256}"; do
    [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || die "invalid Transformer Engine artifact digest: ${digest}"
  done
  grep -Fq "From: docker.io/lmsysorg/sglang@${SGLANG_BASE_AMD64_MANIFEST_DIGEST}" "${DEFINITION}" || \
    die "definition base does not match the provenance lock"
  grep -Fq "org.platoon.apex.commit ${APEX_GIT_COMMIT}" "${DEFINITION}" || \
    die "definition Apex label does not match the provenance lock"
  grep -Fq "org.platoon.transformer-engine.commit ${TRANSFORMER_ENGINE_GIT_COMMIT}" "${DEFINITION}" || \
    die "definition Transformer Engine label does not match the provenance lock"
  grep -Fq "uv==${UV_VERSION}" "${DEFINITION}" || die "definition uv pin does not match the provenance lock"

  bash -n \
    "${SCRIPT_DIR}/install_apex.sh" \
    "${SCRIPT_DIR}/install_te.sh" \
    "${SCRIPT_DIR}/build-training-runtime.pbs" \
    "${SCRIPT_DIR}/build_training_sif.sh" \
    "${BASH_SOURCE[0]}"
  "${SCRIPT_DIR}/install_apex.sh" --dry-run >/dev/null
  "${SCRIPT_DIR}/install_te.sh" --dry-run >/dev/null
  "${SCRIPT_DIR}/build_training_sif.sh" --dry-run >/dev/null

  grep -Fxq '#PBS -P gai51740' "${SCRIPT_DIR}/build-training-runtime.pbs" || \
    die "runtime build PBS script has the wrong project"
  grep -Fxq '#PBS -q R9920261300' "${SCRIPT_DIR}/build-training-runtime.pbs" || \
    die "runtime build PBS script has the wrong queue"
  grep -Fxq '#PBS -v RTYPE=rt_HF' "${SCRIPT_DIR}/build-training-runtime.pbs" || \
    die "runtime build PBS script does not request rt_HF"
  grep -Fxq '#PBS -l select=1:mpiprocs=8' "${SCRIPT_DIR}/build-training-runtime.pbs" || \
    die "runtime build PBS script does not request one full H200 node"

  while IFS= read -r constraint; do
    [[ -z "${constraint}" || "${constraint}" == \#* || "${constraint}" == *'=='* ]] || \
      die "unpinned Transformer Engine constraint: ${constraint}"
  done <"${REPO_ROOT}/containers/transformer-engine-build-constraints.txt"
  grep -Fxq "onnx==${ONNX_VERSION}" "${REPO_ROOT}/containers/transformer-engine-build-constraints.txt" || \
    die "ONNX build constraint and provenance lock disagree"
  grep -Fxq "onnxscript==${ONNXSCRIPT_VERSION}" "${REPO_ROOT}/containers/transformer-engine-build-constraints.txt" || \
    die "ONNXScript build constraint and provenance lock disagree"
  grep -Fxq "onnx-ir==${ONNX_IR_VERSION}" "${REPO_ROOT}/containers/transformer-engine-build-constraints.txt" || \
    die "ONNX IR build constraint and provenance lock disagree"
  grep -Fxq "nvdlfw-inspect==${NVDLFW_INSPECT_VERSION}" \
    "${REPO_ROOT}/containers/transformer-engine-build-constraints.txt" || \
    die "nvdlfw-inspect build constraint and provenance lock disagree"

  echo "Static ABCI training runtime validation passed"
}

mode=
sif=
venv=
with_gpu=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --static)
      mode=static
      shift
      ;;
    --sif)
      [[ $# -ge 2 ]] || die "--sif requires a path"
      mode=sif
      sif=$2
      shift 2
      ;;
    --venv)
      [[ $# -ge 2 ]] || die "--venv requires a path"
      venv=$2
      shift 2
      ;;
    --with-gpu)
      with_gpu=1
      shift
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

case "${mode}" in
  static)
    [[ -z "${sif}" && -z "${venv}" && "${with_gpu}" == 0 ]] || \
      die "--static cannot be combined with image options"
    static_check
    exit 0
    ;;
  sif)
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

[[ -f "${sif}" ]] || die "SIF not found: ${sif}"
# shellcheck disable=SC1090
source "${LOCK_FILE}"
singularity_bin=${SINGULARITY_BIN:-singularity}
command -v "${singularity_bin}" >/dev/null 2>&1 || die "Singularity executable not found: ${singularity_bin}"

"${singularity_bin}" inspect "${sif}" >/dev/null
labels=$("${singularity_bin}" inspect --labels "${sif}")
grep -Fq "${SGLANG_BASE_AMD64_MANIFEST_DIGEST}" <<<"${labels}" || die "SIF base digest label is wrong"
grep -Fq "${APEX_GIT_COMMIT}" <<<"${labels}" || die "SIF Apex label is wrong"
grep -Fq "${TRANSFORMER_ENGINE_GIT_COMMIT}" <<<"${labels}" || die "SIF Transformer Engine label is wrong"

runtime_args=(exec --cleanenv --no-home)
if [[ "${with_gpu}" == 1 ]]; then
  runtime_args+=(--nv)
fi
runtime_args+=("${sif}")
"${singularity_bin}" "${runtime_args[@]}" /bin/bash -lc \
  'python3 -c "from importlib.metadata import version; import torch; assert torch.__version__ == '\''2.9.1+cu129'\''; assert torch.version.cuda == '\''12.9'\''; assert version('\''sglang'\'') == '\''0.5.10.post1'\''"; nvcc --version | grep -F "release 12.9"'

if [[ "${with_gpu}" == 1 ]]; then
  "${singularity_bin}" "${runtime_args[@]}" python3 -c \
    'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
fi
if [[ -n "${venv}" ]]; then
  [[ -x "${venv}/bin/python" ]] || die "venv Python is not executable: ${venv}/bin/python"
  "${singularity_bin}" "${runtime_args[@]}" \
    /opt/platoon-runtime/pbs-scripts/install_te.sh --verify-only "${venv}"
  "${singularity_bin}" "${runtime_args[@]}" \
    /opt/platoon-runtime/pbs-scripts/install_apex.sh --verify-only "${venv}"
fi

echo "ABCI training runtime validation passed: ${sif}"
