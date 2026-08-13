#!/usr/bin/env bash
# Build the host-side controller environment used by PBS launch supervisors.
#
# Training workers use a CUDA/TE/APEX environment built inside the immutable
# training SIF.  That venv's interpreter is /usr/bin/python3.12 in the SIF and
# cannot execute on ABCI's RHEL host.  This helper consumes the *same immutable
# source archive* but builds a second content-addressed venv with an explicitly
# pinned uv-managed host CPython.  It deliberately does not install TE or Apex.

set -euo pipefail
umask 077

die() {
    echo "[openreward-controller-env] ERROR: $*" >&2
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
    local temporary=$1 destination=$2
    if ln "${temporary}" "${destination}" 2>/dev/null; then
        rm -f -- "${temporary}"
        return 0
    fi
    [[ -f "${destination}" ]] || die "could not publish ${destination}"
    [[ "$(sha256_file "${temporary}")" == "$(sha256_file "${destination}")" ]] || \
        die "content collision while publishing ${destination}"
    rm -f -- "${temporary}"
}

python_distribution_root() {
    local python=$1 resolved
    resolved=$(readlink -f -- "${python}")
    [[ "${resolved}" == */bin/python3.12 ]] || \
        die "controller Python must be an exact Python 3.12 distribution binary: ${resolved}"
    dirname -- "$(dirname -- "${resolved}")"
}

controller_python_identity() {
    local python=$1 distribution_root executable_hash libpython_hash build_hash identity
    distribution_root=$(python_distribution_root "${python}")
    executable_hash=$(sha256_file "${distribution_root}/bin/python3.12")
    [[ -f "${distribution_root}/lib/libpython3.12.so.1.0" ]] || \
        die "controller Python distribution has no libpython3.12.so.1.0"
    libpython_hash=$(sha256_file "${distribution_root}/lib/libpython3.12.so.1.0")
    [[ -f "${distribution_root}/BUILD" ]] || die "controller Python distribution has no BUILD identity"
    build_hash=$(sha256_file "${distribution_root}/BUILD")
    identity=$("${python}" -I - <<'PY'
import platform
import sys
import sysconfig

assert sys.version_info[:2] == (3, 12), sys.version
print("|".join((
    platform.python_version(),
    sys.implementation.cache_tag,
    sysconfig.get_config_var("SOABI") or "",
    platform.machine(),
)))
PY
    )
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${distribution_root}" "${identity}" "${executable_hash}" "${libpython_hash}" "${build_hash}"
}

validate_controller_environment() {
    local environment=$1 manifest_file=$2
    local python=${environment}/bin/python
    [[ -x "${python}" ]] || return 1
    [[ -f "${environment}/.platoon-controller-ready" ]] || return 1
    [[ -f "${environment}/.build-complete" ]] || return 1
    [[ "$(stat -c '%a' "${environment}/.platoon-controller-ready")" == 400 ]] || return 1
    [[ "$(stat -c '%a' "${environment}/.build-complete")" == 400 ]] || return 1
    [[ -f "${environment}/.platoon-controller-manifest" ]] || return 1
    cmp -s "${manifest_file}" "${environment}/.platoon-controller-manifest" || return 1
    grep -q '^relocatable = true$' "${environment}/pyvenv.cfg" || return 1
    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= \
        "${python}" -I - "${environment}" <<'PY'
import importlib.util
import sys
from importlib import metadata
from pathlib import Path

environment = Path(sys.argv[1]).resolve()
if Path(sys.prefix).resolve() != environment:
    raise SystemExit(1)
for name in (
    "areal",
    "datasets",
    "openhands-sdk",
    "platoon",
    "platoon-openreward",
    "torch",
):
    distribution = metadata.distribution(name)
    if not Path(distribution.locate_file("")).resolve().is_relative_to(environment):
        raise SystemExit(1)
for name in ("areal", "datasets", "openhands", "platoon", "torch"):
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise SystemExit(1)
PY
}

make_environment_immutable() {
    local environment=$1
    # uv may hardlink package files from its ABI-specific cache.  Only remove
    # directory write permission so an install cannot replace entries without
    # mutating modes on shared cache inodes.
    find "${environment}" -type d -exec chmod a-w {} +
}

resolve_environment() {
    require_env OPENREWARD_CONTROLLER_ENV_CACHE_ROOT
    require_env OPENREWARD_CONTROLLER_PYTHON
    require_env OPENREWARD_SOURCE_SHA
    require_env OPENREWARD_SOURCE_ARCHIVE

    local cache_root=${OPENREWARD_CONTROLLER_ENV_CACHE_ROOT}
    local controller_python=${OPENREWARD_CONTROLLER_PYTHON}
    local source_sha=${OPENREWARD_SOURCE_SHA}
    local source_archive=${OPENREWARD_SOURCE_ARCHIVE}
    local uv_bin=${OPENREWARD_CONTROLLER_UV_BIN:-${OPENREWARD_UV_BIN:-uv}}
    local schema=${OPENREWARD_CONTROLLER_ENV_SCHEMA:-pbs-host-controller-1}
    local helper_sha embedded_helper_sha source_actual_sha
    local python_root python_identity python_executable_sha python_lib_sha python_build_sha
    local uv_version host_os_sha glibc_identity kernel_machine
    local manifest_tmp manifest_file env_key final_env

    [[ -r "${source_archive}" ]] || die "immutable source archive is unreadable: ${source_archive}"
    source_actual_sha=$(sha256_file "${source_archive}")
    [[ "${source_actual_sha}" == "${source_sha}" ]] || die "immutable source archive digest mismatch"
    command -v "${uv_bin}" >/dev/null 2>&1 || [[ -x "${uv_bin}" ]] || \
        die "uv executable not found: ${uv_bin}"
    [[ -x "${controller_python}" ]] || die "controller Python is not executable: ${controller_python}"

    helper_sha=$(sha256_file "${BASH_SOURCE[0]}")
    embedded_helper_sha=$(tar -xOf "${source_archive}" \
        pbs-scripts/prepare_openreward_controller_env.sh | sha256sum | awk '{print $1}')
    [[ "${helper_sha}" == "${embedded_helper_sha}" ]] || \
        die "running controller helper is not the helper captured by the immutable source archive"

    IFS=$'\t' read -r \
        python_root python_identity python_executable_sha python_lib_sha python_build_sha \
        <<<"$(controller_python_identity "${controller_python}")"
    uv_version=$("${uv_bin}" --version)
    host_os_sha=$(sha256_file /etc/os-release)
    glibc_identity=$(getconf GNU_LIBC_VERSION)
    kernel_machine=$(uname -m)
    [[ "${kernel_machine}" == x86_64 ]] || die "unsupported controller architecture: ${kernel_machine}"

    mkdir -p \
        "${cache_root}/envs" \
        "${cache_root}/manifests" \
        "${cache_root}/.locks" \
        "${cache_root}/.staging" \
        "${cache_root}/.failed"
    manifest_tmp=$(mktemp "${cache_root}/.staging/controller-manifest.XXXXXXXX.txt")
    trap 'rm -f -- "${manifest_tmp:-}"' RETURN
    {
        printf 'schema=%s\n' "${schema}"
        printf 'role=host-controller-and-pbs-supervisors\n'
        printf 'source_sha256=%s\n' "${source_sha}"
        printf 'helper_sha256=%s\n' "${helper_sha}"
        printf 'uv=%s\n' "${uv_version}"
        printf 'python_root=%s\n' "${python_root}"
        printf 'python_identity=%s\n' "${python_identity}"
        printf 'python_executable_sha256=%s\n' "${python_executable_sha}"
        printf 'python_lib_sha256=%s\n' "${python_lib_sha}"
        printf 'python_build_sha256=%s\n' "${python_build_sha}"
        printf 'host_os_release_sha256=%s\n' "${host_os_sha}"
        printf 'host_glibc=%s\n' "${glibc_identity}"
        printf 'host_architecture=%s\n' "${kernel_machine}"
        printf 'local_install_mode=wheel\n'
        printf 'build_transformer_engine=0\n'
        printf 'build_apex=0\n'
    } >"${manifest_tmp}"
    env_key=$(sha256_file "${manifest_tmp}")
    manifest_file=${cache_root}/manifests/${env_key}.txt
    if [[ -f "${manifest_file}" ]]; then
        cmp -s "${manifest_tmp}" "${manifest_file}" || \
            die "controller environment-key manifest collision: ${env_key}"
        rm -f -- "${manifest_tmp}"
    else
        atomic_publish_file "${manifest_tmp}" "${manifest_file}"
        chmod a-w "${manifest_file}" 2>/dev/null || true
    fi
    final_env=${cache_root}/envs/${env_key}
    printf '%s\t%s\n' "${env_key}" "${final_env}"
}

build_environment() {
    require_env OPENREWARD_CONTROLLER_ENV_CACHE_ROOT
    require_env OPENREWARD_CONTROLLER_ENV_KEY
    require_env OPENREWARD_CONTROLLER_JOB_VENV
    require_env OPENREWARD_CONTROLLER_PYTHON
    require_env OPENREWARD_CONTROLLER_UV_CACHE_DIR
    require_env OPENREWARD_SOURCE_SHA
    require_env OPENREWARD_SOURCE_ARCHIVE

    local cache_root=${OPENREWARD_CONTROLLER_ENV_CACHE_ROOT}
    local env_key=${OPENREWARD_CONTROLLER_ENV_KEY}
    local final_env=${OPENREWARD_CONTROLLER_JOB_VENV}
    local controller_python=${OPENREWARD_CONTROLLER_PYTHON}
    local uv_cache=${OPENREWARD_CONTROLLER_UV_CACHE_DIR}
    local source_sha=${OPENREWARD_SOURCE_SHA}
    local source_archive=${OPENREWARD_SOURCE_ARCHIVE}
    local uv_bin=${OPENREWARD_CONTROLLER_UV_BIN:-${OPENREWARD_UV_BIN:-uv}}
    local lock_wait=${OPENREWARD_CONTROLLER_ENV_LOCK_WAIT_SECS:-7200}
    local expected_env=${cache_root}/envs/${env_key}
    local manifest_file=${cache_root}/manifests/${env_key}.txt
    local build_root stage_env source_root failed_env

    [[ "${final_env}" == "${expected_env}" ]] || die "refusing unexpected controller environment path"
    [[ "$(sha256_file "${source_archive}")" == "${source_sha}" ]] || \
        die "immutable source archive digest mismatch"
    [[ -f "${manifest_file}" ]] || die "controller environment manifest is missing"
    [[ -x "${controller_python}" ]] || die "controller Python is not executable: ${controller_python}"
    command -v "${uv_bin}" >/dev/null 2>&1 || [[ -x "${uv_bin}" ]] || \
        die "uv executable not found: ${uv_bin}"

    mkdir -p \
        "${cache_root}/.locks" "${cache_root}/.staging" "${cache_root}/.failed" \
        "${uv_cache}"
    exec 9>"${cache_root}/.locks/${env_key}.lock"
    echo "[openreward-controller-env] waiting for cache lock ${env_key} at $(date -Is)"
    flock -w "${lock_wait}" 9 || die "timed out waiting for controller environment lock"

    if validate_controller_environment "${final_env}" "${manifest_file}"; then
        echo "[openreward-controller-env] cache hit: ${final_env}"
        return 0
    fi
    if [[ -e "${final_env}" ]]; then
        failed_env=${cache_root}/.failed/${env_key}.unhealthy.$(date +%s).$$.${RANDOM}
        echo "[openreward-controller-env] quarantining unhealthy environment: ${failed_env}" >&2
        mv "${final_env}" "${failed_env}"
    fi

    # The per-key lock proves abandoned staging roots for this key are stale.
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
    _controller_cleanup_build_root=${build_root}
    _controller_cleanup_final_env=${final_env}
    _controller_cleanup_failed_root=${cache_root}/.failed
    _controller_cleanup_env_key=${env_key}
    _controller_cleanup_promoted=0
    _controller_cleanup_ready=0
    cleanup_build() {
        local status=$?
        if [[ "${_controller_cleanup_promoted:-0}" == 1 && \
              "${_controller_cleanup_ready:-0}" != 1 && \
              -e "${_controller_cleanup_final_env:-}" ]]; then
            mv "${_controller_cleanup_final_env}" \
                "${_controller_cleanup_failed_root}/${_controller_cleanup_env_key}.promotion.$(date +%s).$$.${RANDOM}" \
                2>/dev/null || true
        fi
        if [[ -n "${_controller_cleanup_build_root:-}" && -d "${_controller_cleanup_build_root}" ]]; then
            rm -rf -- "${_controller_cleanup_build_root}"
        fi
        return "${status}"
    }
    trap cleanup_build EXIT

    mkdir -p "${source_root}"
    tar -xf "${source_archive}" -C "${source_root}"
    export UV_CACHE_DIR=${uv_cache}
    export UV_LINK_MODE=${OPENREWARD_CONTROLLER_UV_LINK_MODE:-hardlink}
    export UV_PROJECT_ENVIRONMENT=${stage_env}
    export UV_PYTHON_DOWNLOADS=never
    export VIRTUAL_ENV=${stage_env}
    export PYTHONDONTWRITEBYTECODE=1
    export CUDA_VISIBLE_DEVICES=

    echo "[openreward-controller-env] cache miss; building host-compatible staging environment at $(date -Is)"
    "${uv_bin}" venv --relocatable --python "${controller_python}" "${stage_env}"
    cd "${source_root}/plugins/openreward"
    "${uv_bin}" sync --locked --extra areal --no-editable --compile-bytecode

    if find "${stage_env}/lib/python3.12/site-packages" -maxdepth 1 \
        -name '__editable__*' -print -quit | grep -q .; then
        die "editable local package metadata found in controller environment"
    fi
    if grep -Rqs '"editable"[[:space:]]*:[[:space:]]*true' \
        "${stage_env}/lib/python3.12/site-packages"/*.dist-info/direct_url.json 2>/dev/null; then
        die "editable local package direct_url metadata found in controller environment"
    fi
    grep -q '^relocatable = true$' "${stage_env}/pyvenv.cfg" || \
        die "controller environment is not relocatable"

    # This is the exact controller module import chain.  PyTorch is genuinely
    # imported by Platoon's controller, but Megatron/TE/Apex must remain lazy
    # and worker-only.  A blanket `uv pip check` is intentionally omitted: the
    # committed OpenReward lock contains documented override metadata conflicts.
    PLATOON_EXPECTED_VENV=${stage_env} "${stage_env}/bin/python" -I - <<'PY'
import sys
from pathlib import Path

expected = Path(__import__("os").environ["PLATOON_EXPECTED_VENV"]).resolve()
assert Path(sys.prefix).resolve() == expected
import platoon.openreward.train_scripts.areal.train_areal as train_entry
from platoon.train.areal.preallocated_pbs import PreallocatedPBSScheduler

assert callable(train_entry.main)
assert PreallocatedPBSScheduler.__name__ == "PreallocatedPBSScheduler"
for forbidden in ("megatron", "transformer_engine", "apex", "fused_weight_gradient_mlp_cuda"):
    assert not any(name == forbidden or name.startswith(forbidden + ".") for name in sys.modules), forbidden
print("host_controller_lazy_cuda_imports=ok")
PY

    cp "${manifest_file}" "${stage_env}/.platoon-controller-manifest"
    touch "${stage_env}/.build-complete"
    chmod 0400 "${stage_env}/.build-complete"
    echo "[openreward-controller-env] atomically promoting ${final_env} at $(date -Is)"
    mv -T "${stage_env}" "${final_env}"
    _controller_cleanup_promoted=1
    touch "${final_env}/.platoon-controller-ready"
    chmod 0400 "${final_env}/.platoon-controller-ready"
    validate_controller_environment "${final_env}" "${manifest_file}"
    make_environment_immutable "${final_env}"
    _controller_cleanup_ready=1
    echo "[openreward-controller-env] environment ready: ${final_env} at $(date -Is)"
    rm -rf -- "${build_root}"
    _controller_cleanup_build_root=
    trap - EXIT
}

materialize_helper() {
    [[ $# -eq 2 ]] || die "usage: $0 materialize SOURCE_ARCHIVE OUTPUT"
    local source_archive=$1 output=$2 temporary
    [[ -r "${source_archive}" ]] || die "source archive is unreadable: ${source_archive}"
    mkdir -p "$(dirname -- "${output}")"
    temporary=$(mktemp "$(dirname -- "${output}")/.controller-helper.XXXXXXXX")
    trap 'rm -f -- "${temporary:-}"' EXIT INT TERM
    tar -xOf "${source_archive}" pbs-scripts/prepare_openreward_controller_env.sh >"${temporary}"
    chmod 0555 "${temporary}"
    mv -f "${temporary}" "${output}"
    temporary=
    trap - EXIT INT TERM
}

usage() {
    cat >&2 <<'EOF'
Usage: prepare_openreward_controller_env.sh resolve|build
       prepare_openreward_controller_env.sh materialize SOURCE_ARCHIVE OUTPUT

resolve consumes an immutable worker source archive and prints a tab-separated
CONTROLLER_ENV_KEY CONTROLLER_FINAL_ENV record. build publishes that exact host
controller environment under its content key.
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
    materialize)
        shift
        materialize_helper "$@"
        ;;
    *)
        usage
        ;;
esac
