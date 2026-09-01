#!/bin/bash
# Build and reuse an immutable, content-addressed OpenReward training venv.
#
# The launcher calls `resolve` on the host to snapshot the exact local package
# sources and derive the cache key.  It calls `build` once inside the CUDA
# container.  Builders for the same key are serialized with flock; consumers
# only accept an environment after its final readiness marker is published.

set -euo pipefail

die() {
  echo "[openreward-env] ERROR: $*" >&2
  exit 1
}

require_env() {
  local name=$1
  [[ -n "${!name:-}" ]] || die "required environment variable ${name} is not set"
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

atomic_publish_file() {
  local temporary=$1
  local destination=$2

  # link(2) is an atomic no-clobber publication on this shared filesystem.  If
  # another resolver won the race, its file must have exactly the same digest.
  if ln "${temporary}" "${destination}" 2>/dev/null; then
    rm -f "${temporary}"
    return 0
  fi
  [[ -f "${destination}" ]] || die "could not publish ${destination}"
  [[ "$(sha256_file "${temporary}")" == "$(sha256_file "${destination}")" ]] || \
    die "content collision while publishing ${destination}"
  rm -f "${temporary}"
}

validate_environment_packages() {
  local environment=$1
  local python=${environment}/bin/python

  [[ -x "${python}" ]] || {
    echo "[openreward-env] cached environment has no executable Python: ${python}" >&2
    return 1
  }

  # Use isolated mode so a caller's PYTHONPATH, user site, or checkout cannot
  # hide a damaged cache entry.  Metadata checks are deliberately cheaper than
  # importing torch/SGLang, while module-spec checks also catch missing package
  # trees when stale dist-info directories happen to remain.
  PYTHONDONTWRITEBYTECODE=1 "${python}" -I - "${environment}" <<'PY'
import importlib.util
import sys
from importlib import metadata
from pathlib import Path

environment = Path(sys.argv[1]).resolve()
required_distributions = (
    "areal",
    "megatron-core",
    "openhands-sdk",
    "openhands-tools",
    "platoon",
    "platoon-openhands",
    "platoon-openreward",
    "ray",
    "sglang",
    "torch",
    "transformers",
)
required_modules = (
    "areal",
    "megatron",
    "openhands",
    "platoon",
    "ray",
    "sglang",
    "torch",
    "transformers",
)

problems = []
if Path(sys.prefix).resolve() != environment:
    problems.append(f"Python prefix is {Path(sys.prefix).resolve()}, expected {environment}")

for name in required_distributions:
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        problems.append(f"missing distribution: {name}")
        continue
    distribution_root = Path(distribution.locate_file("")).resolve()
    if not distribution_root.is_relative_to(environment):
        problems.append(f"distribution outside cached environment: {name} ({distribution_root})")

for name in required_modules:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as exc:
        problems.append(f"module lookup failed: {name} ({exc!r})")
        continue
    if spec is None:
        problems.append(f"missing module: {name}")
        continue
    locations = []
    if spec.origin not in (None, "built-in", "frozen"):
        locations.append(Path(spec.origin).resolve())
    if spec.submodule_search_locations:
        locations.extend(Path(path).resolve() for path in spec.submodule_search_locations)
    if locations and not any(path.is_relative_to(environment) for path in locations):
        problems.append(f"module outside cached environment: {name} ({locations[0]})")

if problems:
    print("[openreward-env] cached environment package validation failed:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    raise SystemExit(1)
PY
}

environment_is_healthy() {
  local environment=$1
  local manifest_file=$2

  [[ -f "${environment}/.platoon-ready" ]] &&
    [[ -f "${environment}/.build-complete" ]] &&
    [[ -f "${environment}/.platoon-env-manifest" ]] &&
    cmp -s "${manifest_file}" "${environment}/.platoon-env-manifest" &&
    grep -q '^relocatable = true$' "${environment}/pyvenv.cfg" &&
    validate_environment_packages "${environment}"
}

make_environment_immutable() {
  local environment=$1

  # Package files may be hardlinked from uv's shared cache.  Changing their
  # modes would mutate the same inode for other environments, so only lock the
  # directory entries that pip/uv would need to modify an installation.
  find "${environment}" -type d -exec chmod a-w {} +
}

resolve_environment() {
  require_env PLATOON_REPO_ROOT
  require_env OPENREWARD_ENV_CACHE_ROOT
  require_env OPENREWARD_PROJECT_DIR
  require_env OPENREWARD_CONTAINER_IMAGE

  local repo_root=${PLATOON_REPO_ROOT}
  local cache_root=${OPENREWARD_ENV_CACHE_ROOT}
  local project_dir=${OPENREWARD_PROJECT_DIR}
  local schema=${OPENREWARD_ENV_SCHEMA:-2}
  local uv_bin=${OPENREWARD_UV_BIN:-uv}
  local build_te=${OPENREWARD_BUILD_TE:-0}
  local build_apex=${OPENREWARD_BUILD_APEX:-0}
  local te_version=${TE_VERSION:-2.12.0}
  local apex_ref=${APEX_GIT_REF:-master}
  local apex_url=${APEX_GIT_URL:-https://github.com/NVIDIA/apex.git}
  local arch_list=${TORCH_CUDA_ARCH_LIST:-9.0+PTX}
  local python_tag=cp312
  local source_tmp source_sha source_archive
  local torch_version
  local container_id uv_version manifest_tmp manifest_file env_key final_env

  [[ -d "${repo_root}/platoon" ]] || die "invalid repository root: ${repo_root}"
  [[ -f "${project_dir}/uv.lock" ]] || die "OpenReward uv.lock not found under ${project_dir}"
  [[ -f "${OPENREWARD_CONTAINER_IMAGE}" ]] || die "container image not found: ${OPENREWARD_CONTAINER_IMAGE}"
  command -v "${uv_bin}" >/dev/null 2>&1 || [[ -x "${uv_bin}" ]] || die "uv executable not found: ${uv_bin}"
  [[ "${TE_REBUILD:-0}" != "1" && "${APEX_REBUILD:-0}" != "1" && \
     "${TE_FORCE:-0}" != "1" && "${APEX_FORCE:-0}" != "1" ]] || \
    die "TE/APEX force-rebuild flags are incompatible with immutable cached environments; change the version/ref instead"

  mkdir -p "${cache_root}/sources" "${cache_root}/envs" \
    "${cache_root}/manifests" "${cache_root}/.locks" \
    "${cache_root}/.staging" "${cache_root}/.failed"

  source_tmp=$(mktemp "${cache_root}/.staging/source.XXXXXXXX.tar")
  trap 'rm -f "${source_tmp:-}" "${manifest_tmp:-}"' RETURN

  # Include all build inputs for the three local distributions.  This captures
  # dirty and untracked package code, unlike a Git commit-only key.  Normalized
  # tar metadata makes repeated resolution byte-for-byte deterministic.
  local -a source_inputs=(
    pyproject.toml
    uv.lock
    README.md
    LICENSE
    platoon
    plugins/openhands/pyproject.toml
    plugins/openhands/platoon
    plugins/openreward/pyproject.toml
    plugins/openreward/uv.lock
    plugins/openreward/platoon
    slurm-scripts/install_te.sh
    slurm-scripts/install_apex.sh
    slurm-scripts/patches/areal-d991-megatron-merged-lora.patch
    slurm-scripts/patches/megatron-bridge-0.4.0-grouped-lora-merge.patch
    slurm-scripts/prepare_openreward_env.sh
  )
  (
    cd "${repo_root}"
    tar \
      --sort=name \
      --mtime='UTC 1970-01-01' \
      --owner=0 \
      --group=0 \
      --numeric-owner \
      --format=gnu \
      --exclude='*/__pycache__' \
      --exclude='*.pyc' \
      --exclude='*.pyo' \
      --exclude='*/.pytest_cache' \
      --exclude='*/.pytest_cache/*' \
      --exclude='*/.ruff_cache' \
      --exclude='*/.ruff_cache/*' \
      -cf "${source_tmp}" \
      "${source_inputs[@]}"
  )
  source_sha=$(sha256_file "${source_tmp}")
  source_archive=${cache_root}/sources/${source_sha}.tar
  if [[ -f "${source_archive}" ]]; then
    [[ "$(sha256_file "${source_archive}")" == "${source_sha}" ]] || \
      die "cached source archive has the wrong digest: ${source_archive}"
    rm -f "${source_tmp}"
  else
    atomic_publish_file "${source_tmp}" "${source_archive}"
    chmod a-w "${source_archive}" 2>/dev/null || true
  fi

  # Record the locked torch identity used by both compiled-extension caches. The
  # wheel cache is append-only in normal operation; explicit force/rebuild flags
  # are rejected above.
  torch_version=$(sed -n '/^name = "torch"$/ { n; s/^version = "\([^"]*\)"$/\1/p; q; }' "${project_dir}/uv.lock")
  [[ -n "${torch_version}" ]] || die "could not determine torch version from ${project_dir}/uv.lock"

  uv_version=$("${uv_bin}" --version)
  container_id=${OPENREWARD_CONTAINER_ENV_ID:-$(stat -Lc 'size=%s;mtime=%Y;inode=%i' "${OPENREWARD_CONTAINER_IMAGE}")}

  manifest_tmp=$(mktemp "${cache_root}/.staging/manifest.XXXXXXXX.txt")
  {
    printf 'schema=%s\n' "${schema}"
    printf 'source_sha256=%s\n' "${source_sha}"
    printf 'container=%s\n' "${container_id}"
    printf 'uv=%s\n' "${uv_version}"
    printf 'platform=linux-x86_64\n'
    printf 'python=%s\n' "${python_tag}"
    printf 'torch=%s\n' "${torch_version}"
    printf 'build_te=%s\n' "${build_te}"
    printf 'te_version=%s\n' "${te_version}"
    printf 'build_apex=%s\n' "${build_apex}"
    printf 'apex_ref=%s\n' "${apex_ref}"
    printf 'apex_url=%s\n' "${apex_url}"
    printf 'torch_cuda_arch_list=%s\n' "${arch_list}"
    printf 'nvidia_cuda_runtime_cu12=12.9.79\n'
    printf 'nvidia_cublas_cu12=12.9.1.4\n'
    printf 'nvidia_cuda_nvrtc_cu12=12.9.86\n'
    printf 'nvidia_cusparse_cu12=12.5.10.65\n'
    printf 'nvidia_nvjitlink_cu12=12.9.86\n'
    printf 'local_install_mode=wheel\n'
    printf 'bytecode=compiled-no-runtime-writes\n'
  } >"${manifest_tmp}"
  env_key=$(sha256_file "${manifest_tmp}")
  manifest_file=${cache_root}/manifests/${env_key}.txt
  if [[ -f "${manifest_file}" ]]; then
    cmp -s "${manifest_tmp}" "${manifest_file}" || die "environment-key manifest collision: ${env_key}"
    rm -f "${manifest_tmp}"
  else
    atomic_publish_file "${manifest_tmp}" "${manifest_file}"
    chmod a-w "${manifest_file}" 2>/dev/null || true
  fi

  final_env=${cache_root}/envs/${env_key}
  printf '%s\t%s\t%s\t%s\n' "${env_key}" "${source_sha}" "${source_archive}" "${final_env}"
}

build_environment() {
  require_env PLATOON_REPO_ROOT
  require_env OPENREWARD_ENV_CACHE_ROOT
  require_env OPENREWARD_ENV_KEY
  require_env OPENREWARD_SOURCE_SHA
  require_env OPENREWARD_SOURCE_ARCHIVE
  require_env OPENREWARD_JOB_VENV
  require_env OPENREWARD_UV_CACHE_DIR

  local repo_root=${PLATOON_REPO_ROOT}
  local cache_root=${OPENREWARD_ENV_CACHE_ROOT}
  local env_key=${OPENREWARD_ENV_KEY}
  local source_sha=${OPENREWARD_SOURCE_SHA}
  local source_archive=${OPENREWARD_SOURCE_ARCHIVE}
  local final_env=${OPENREWARD_JOB_VENV}
  local uv_bin=${OPENREWARD_UV_BIN:-uv}
  local lock_wait=${OPENREWARD_ENV_LOCK_WAIT_SECS:-7200}
  local expected_env=${cache_root}/envs/${env_key}
  local manifest_file=${cache_root}/manifests/${env_key}.txt
  local build_root= stage_env= source_root= promoted=0 ready=0

  [[ "${final_env}" == "${expected_env}" ]] || die "refusing unexpected final environment path: ${final_env}"
  [[ "$(sha256_file "${source_archive}")" == "${source_sha}" ]] || die "source archive digest mismatch"
  [[ -f "${manifest_file}" ]] || die "environment manifest is missing: ${manifest_file}"
  command -v "${uv_bin}" >/dev/null 2>&1 || [[ -x "${uv_bin}" ]] || die "uv executable not found: ${uv_bin}"

  mkdir -p "${cache_root}/.locks" "${cache_root}/.staging" "${cache_root}/.failed"
  exec 9>"${cache_root}/.locks/${env_key}.lock"
  echo "[openreward-env] waiting for cache lock ${env_key} at $(date -Is)"
  flock -w "${lock_wait}" 9 || die "timed out waiting ${lock_wait}s for environment lock ${env_key}"

  if environment_is_healthy "${final_env}" "${manifest_file}"; then
    echo "[openreward-env] cache hit: ${final_env}"
    return 0
  fi
  if [[ -e "${final_env}" ]]; then
    local failed_env=${cache_root}/.failed/${env_key}.unhealthy.$(date +%s).$$.${RANDOM}
    echo "[openreward-env] quarantining incomplete, mismatched, or unhealthy environment: ${failed_env}" >&2
    mv "${final_env}" "${failed_env}"
  fi

  # A SIGKILL cannot run the EXIT trap. The per-key lock proves that no live
  # builder for this key remains, so its abandoned private staging directories
  # are safe to remove before retrying.
  local -a stale_staging=()
  shopt -s nullglob
  stale_staging=("${cache_root}/.staging/${env_key}."*)
  shopt -u nullglob
  if [[ "${#stale_staging[@]}" -gt 0 ]]; then
    rm -rf -- "${stale_staging[@]}"
  fi

  build_root=$(mktemp -d "${cache_root}/.staging/${env_key}.XXXXXXXX")
  stage_env=${build_root}/venv
  source_root=${build_root}/source
  cleanup_build() {
    local status=$?
    if [[ "${promoted}" == "1" && "${ready}" != "1" && -e "${final_env}" ]]; then
      mv "${final_env}" "${cache_root}/.failed/${env_key}.promotion.$(date +%s).$$.${RANDOM}" 2>/dev/null || true
    fi
    [[ -n "${build_root}" && -d "${build_root}" ]] && rm -rf "${build_root}"
    return "${status}"
  }
  trap cleanup_build EXIT

  mkdir -p "${source_root}"
  tar -xf "${source_archive}" -C "${source_root}"

  export UV_CACHE_DIR=${OPENREWARD_UV_CACHE_DIR}
  export UV_LINK_MODE=${OPENREWARD_UV_LINK_MODE:-hardlink}
  export UV_PROJECT_ENVIRONMENT=${stage_env}
  export VIRTUAL_ENV=${stage_env}
  export PYTHONDONTWRITEBYTECODE=1
  export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0+PTX}
  export TE_WHEEL_CACHE=${TE_WHEEL_CACHE:-${repo_root}/.te-wheels}
  export APEX_WHEEL_CACHE=${APEX_WHEEL_CACHE:-${repo_root}/.apex-wheels}

  echo "[openreward-env] cache miss; building relocatable staging environment at $(date -Is)"
  "${uv_bin}" venv --relocatable --python 3.12 "${stage_env}"
  cd "${source_root}/plugins/openreward"
  "${uv_bin}" sync --locked --extra areal --no-editable --compile-bytecode

  # AReaL d991 has an experimental Megatron LoRA path, but it injects adapters
  # after DDP (so DP replicas never reduce adapter gradients), disables
  # distributed-optimizer recovery, and exposes adapters directly to the
  # rollout runtime. This backport injects adapters before DDP and explicitly
  # merges them into the ordinary full-model XCCL stream instead. The rollout
  # server therefore stays on its proven non-LoRA Qwen3.6 kernels.
  local areal_direct_url megatron_bridge_version
  areal_direct_url=$("${stage_env}/bin/python" -I -c \
    'from importlib.metadata import distribution; print(distribution("areal").read_text("direct_url.json") or "")')
  [[ "${areal_direct_url}" == *"d99124ec15102ca2fcd4960cc8beaef3950c2672"* ]] || \
    die "the merged-LoRA patch requires AReaL d99124ec15102ca2fcd4960cc8beaef3950c2672"
  megatron_bridge_version=$("${stage_env}/bin/python" -I -c \
    'from importlib.metadata import version; print(version("megatron-bridge"))')
  [[ "${megatron_bridge_version}" == "0.4.0" ]] || \
    die "the grouped-expert LoRA merge patch requires Megatron Bridge 0.4.0; got ${megatron_bridge_version}"
  command -v patch >/dev/null 2>&1 || die "the build container has no patch executable"
  patch \
    --batch \
    --forward \
    --strip=1 \
    --directory="${stage_env}/lib/python3.12/site-packages" \
    <"${source_root}/slurm-scripts/patches/areal-d991-megatron-merged-lora.patch"
  patch \
    --batch \
    --forward \
    --strip=1 \
    --directory="${stage_env}/lib/python3.12/site-packages" \
    <"${source_root}/slurm-scripts/patches/megatron-bridge-0.4.0-grouped-lora-merge.patch"
  "${stage_env}/bin/python" -m compileall -q \
    "${stage_env}/lib/python3.12/site-packages/areal" \
    "${stage_env}/lib/python3.12/site-packages/megatron/bridge/models/conversion"
  "${stage_env}/bin/python" -I -c \
    'from areal.api.cli_args import MegatronEngineConfig; c = MegatronEngineConfig(); assert c.merge_lora_for_update_weights is False'
  # Parse the patched source without importing Megatron Bridge here. Its LoRA
  # modules import Transformer Engine eagerly, while TE is installed below.
  # The full dependency smoke test after TE/APEX installation exercises the
  # real import path.
  "${stage_env}/bin/python" -I -c \
    'import ast, pathlib, sys; tree = ast.parse(pathlib.Path(sys.argv[1]).read_text()); assert any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "_merge_grouped_export_adapter_weights" for n in ast.walk(tree))' \
    "${stage_env}/lib/python3.12/site-packages/megatron/bridge/models/conversion/peft_bridge.py"

  echo "[openreward-env] installing pinned CUDA runtime wheels at $(date -Is)"
  "${uv_bin}" pip install --python "${stage_env}/bin/python" \
    nvidia-cuda-runtime-cu12==12.9.79 \
    nvidia-cublas-cu12==12.9.1.4 \
    nvidia-cuda-nvrtc-cu12==12.9.86 \
    nvidia-cusparse-cu12==12.5.10.65 \
    nvidia-nvjitlink-cu12==12.9.86

  local nvidia_pkg_dir=${stage_env}/lib/python3.12/site-packages/nvidia
  if [[ -d "${nvidia_pkg_dir}/cuda_runtime" && ! -e "${nvidia_pkg_dir}/cudart" ]]; then
    ln -s cuda_runtime "${nvidia_pkg_dir}/cudart"
  fi

  export PATH=${stage_env}/bin:${PATH}
  local cuda_lib_dirs
  cuda_lib_dirs=$(find "${nvidia_pkg_dir}" -mindepth 2 -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)
  if [[ -n "${cuda_lib_dirs}" ]]; then
    export LD_LIBRARY_PATH=${cuda_lib_dirs}:${LD_LIBRARY_PATH:-}
  fi
  hash -r

  if [[ "${OPENREWARD_BUILD_TE:-0}" == "1" ]]; then
    mkdir -p "${TE_WHEEL_CACHE}"
    echo "[openreward-env] installing Transformer Engine under shared artifact lock at $(date -Is)"
    flock -w "${lock_wait}" "${TE_WHEEL_CACHE}/.build.lock" \
      bash "${source_root}/slurm-scripts/install_te.sh" "${stage_env}"
  fi
  if [[ "${OPENREWARD_BUILD_APEX:-0}" == "1" ]]; then
    mkdir -p "${APEX_WHEEL_CACHE}"
    echo "[openreward-env] installing APEX under shared artifact lock at $(date -Is)"
    flock -w "${lock_wait}" "${APEX_WHEEL_CACHE}/.build.lock" \
      bash "${source_root}/slurm-scripts/install_apex.sh" "${stage_env}"
  fi

  # Exercise the exact dependency chain needed by the controller/workers before
  # publishing.  Local distributions must be ordinary wheels, never .pth links
  # back to the mutable source checkout.
  export CUDA_VISIBLE_DEVICES=
  if [[ "${OPENREWARD_BUILD_TE:-0}" == "1" && "${OPENREWARD_BUILD_APEX:-0}" == "1" ]]; then
    "${stage_env}/bin/python" -c "from platoon.train.areal import PlatoonArealRLTrainer; import onnx; assert hasattr(onnx, 'TensorProto'); import transformer_engine.pytorch; from megatron.bridge.models.conversion.auto_bridge import AutoBridge; from fla.ops.cp import build_cp_context; import apex, fused_weight_gradient_mlp_cuda; print('Megatron/APEX dependency smoke test passed')"
  elif [[ "${OPENREWARD_BUILD_TE:-0}" == "1" ]]; then
    "${stage_env}/bin/python" -c "from platoon.train.areal import PlatoonArealRLTrainer; import onnx; assert hasattr(onnx, 'TensorProto'); import transformer_engine.pytorch; from megatron.bridge.models.conversion.auto_bridge import AutoBridge; from fla.ops.cp import build_cp_context; print('Megatron dependency smoke test passed')"
  elif [[ "${OPENREWARD_BUILD_APEX:-0}" == "1" ]]; then
    "${stage_env}/bin/python" -c "import apex, fused_weight_gradient_mlp_cuda; print('APEX dependency smoke test passed')"
  fi
  if find "${stage_env}/lib/python3.12/site-packages" -maxdepth 1 -name '__editable__*' -print -quit | grep -q .; then
    die "editable local package metadata found in staged environment"
  fi
  if grep -Rqs '"editable"[[:space:]]*:[[:space:]]*true' \
      "${stage_env}/lib/python3.12/site-packages"/*.dist-info/direct_url.json 2>/dev/null; then
    die "editable local package direct_url metadata found in staged environment"
  fi
  grep -q '^relocatable = true$' "${stage_env}/pyvenv.cfg" || die "staged uv environment is not relocatable"

  cp "${manifest_file}" "${stage_env}/.platoon-env-manifest"
  touch "${stage_env}/.build-complete"

  echo "[openreward-env] atomically promoting ${final_env} at $(date -Is)"
  mv -T "${stage_env}" "${final_env}"
  promoted=1
  cd "${cache_root}"
  "${final_env}/bin/python" -c "from pathlib import Path; import platoon, platoon.openreward, platoon.train.areal, platoon.openreward.train_scripts.areal.train_areal as train_entry; prefix=Path('${final_env}').resolve(); modules=(platoon, platoon.openreward, platoon.train.areal, train_entry); assert all(Path(m.__file__).resolve().is_relative_to(prefix) for m in modules), [(m.__name__, m.__file__) for m in modules]; print('Relocated non-editable imports passed:', platoon.__file__)"
  validate_environment_packages "${final_env}"
  touch "${final_env}/.platoon-ready"
  # Lock directory entries after publishing the final marker.  The
  # non-writable directory tree prevents pip/uv from unlinking, replacing, or
  # adding packages to an environment that may be shared by multiple jobs.
  make_environment_immutable "${final_env}"
  ready=1
  echo "[openreward-env] environment ready: ${final_env} at $(date -Is)"
  rm -rf "${build_root}"
  build_root=
  trap - EXIT
}

usage() {
  cat >&2 <<'EOF'
Usage: prepare_openreward_env.sh resolve|build

resolve requires PLATOON_REPO_ROOT, OPENREWARD_ENV_CACHE_ROOT,
OPENREWARD_PROJECT_DIR, and OPENREWARD_CONTAINER_IMAGE. It prints one
tab-separated record: ENV_KEY SOURCE_SHA SOURCE_ARCHIVE FINAL_ENV.

build additionally requires OPENREWARD_ENV_KEY, OPENREWARD_SOURCE_SHA,
OPENREWARD_SOURCE_ARCHIVE, OPENREWARD_JOB_VENV, and OPENREWARD_UV_CACHE_DIR.
EOF
  exit 2
}

case "${1:-}" in
  resolve)
    resolve_environment
    ;;
  build)
    build_environment
    ;;
  *)
    usage
    ;;
esac
