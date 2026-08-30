#!/bin/bash

# Shared source-integrity guard for SWE-Rebench launchers. The validated pin is
# deliberately data, not an inherited environment default, so an old Slurm
# continuation cannot silently select a different checkout.

swe_rebench_guard_error() {
  echo "ERROR: $*" >&2
  return 2
}

# Enroot's OCI whiteout converter is a host helper, not part of the Python
# environment.  Merely finding it on PATH is insufficient: its CAP_SYS_ADMIN
# and CAP_MKNOD file capabilities are ignored in a nested user namespace and
# can be lost when the binary is copied to a filesystem without xattr support.
# Exercise opaque and ordinary whiteouts in the same temporary filesystem that
# imports will use. This is deterministic and local; it does not contact a
# registry or import an image.
swe_rebench_require_enroot_runtime() {
  local probe_root=${1:-${ENROOT_TEMP_PATH:-${TMPDIR:-/tmp}}}
  local context=${2:-SWE-rebench runtime}
  local enroot_bin=${SWE_REBENCH_ENROOT_BIN:-}
  local helper_bin=${SWE_REBENCH_ENROOT_AUFS2OVLFS_BIN:-}
  local getcap_bin=${SWE_REBENCH_GETCAP_BIN:-}
  local capabilities="unavailable (getcap is not installed)"
  local probe_dir
  local helper_output

  if [[ -z "${enroot_bin}" ]] && ! enroot_bin=$(command -v enroot 2>/dev/null); then
    swe_rebench_guard_error \
      "${context} requires host Enroot, but 'enroot' is not on PATH; install NVIDIA Enroot on every SWE server node"
    return
  fi
  if [[ -z "${helper_bin}" ]] && ! helper_bin=$(command -v enroot-aufs2ovlfs 2>/dev/null); then
    swe_rebench_guard_error \
      "${context} requires the host enroot-aufs2ovlfs helper, but it is not on PATH; install Enroot's +caps package on every SWE server node"
    return
  fi
  for enroot_executable in "${enroot_bin}" "${helper_bin}"; do
    [[ "${enroot_executable}" == /* && -f "${enroot_executable}" && -x "${enroot_executable}" ]] || {
      swe_rebench_guard_error \
        "${context} resolved a non-executable Enroot dependency: ${enroot_executable}; install the host Enroot +caps package"
      return
    }
  done
  unset enroot_executable

  # Inspect the real file: distribution packages may expose the helper through
  # a symlink, while security.capability is attached to its target inode.
  helper_bin=$(readlink -f -- "${helper_bin}") || {
    swe_rebench_guard_error "${context} cannot resolve enroot-aufs2ovlfs: ${helper_bin}"
    return
  }
  [[ -f "${helper_bin}" && -x "${helper_bin}" ]] || {
    swe_rebench_guard_error "${context} resolved a non-executable enroot-aufs2ovlfs target: ${helper_bin}"
    return
  }

  # File-capability metadata is diagnostic only. A site may provide the same
  # privileges through an equivalent packaging mechanism; the behavioral
  # probe below is authoritative.
  if [[ -z "${getcap_bin}" ]]; then
    getcap_bin=$(command -v getcap 2>/dev/null || true)
  fi
  if [[ -n "${getcap_bin}" && -f "${getcap_bin}" && -x "${getcap_bin}" ]]; then
    capabilities=$("${getcap_bin}" "${helper_bin}" 2>&1 || true)
    capabilities=${capabilities:-no file capabilities reported}
  fi

  [[ -d "${probe_root}" && -w "${probe_root}" ]] || {
    swe_rebench_guard_error \
      "${context} requires a writable Enroot temporary directory for its capability probe: ${probe_root}"
    return
  }
  if ! probe_dir=$(mktemp -d "${probe_root%/}/swe-enroot-capability-probe.XXXXXX"); then
    swe_rebench_guard_error \
      "${context} cannot create an Enroot capability probe under ${probe_root}"
    return
  fi
  if ! mkdir -p "${probe_dir}/opaque"; then
    rm -rf -- "${probe_dir}" 2>/dev/null || true
    swe_rebench_guard_error \
      "${context} cannot prepare an Enroot capability probe under ${probe_root}"
    return
  fi
  if ! (
    : >"${probe_dir}/opaque/.wh..wh..opq" &&
      : >"${probe_dir}/.wh.deleted"
  ); then
    rm -rf -- "${probe_dir}" 2>/dev/null || true
    swe_rebench_guard_error \
      "${context} cannot create Enroot whiteout probes under ${probe_root}"
    return
  fi

  if ! helper_output=$("${helper_bin}" "${probe_dir}" 2>&1); then
    rm -rf -- "${probe_dir}" 2>/dev/null || true
    swe_rebench_guard_error \
      "${context} cannot use enroot-aufs2ovlfs on ${probe_root}: ${helper_output}. Detected capabilities: ${capabilities}. The helper needs effective cap_sys_admin and cap_mknod in the host initial user namespace; install Enroot's +caps package (normally cap_sys_admin,cap_mknod+pe), run the SWE server outside nested user namespaces, and ensure ${probe_root} supports trusted overlay xattrs"
    return
  fi
  if [[ -e "${probe_dir}/opaque/.wh..wh..opq" || \
        -e "${probe_dir}/.wh.deleted" || \
        ! -e "${probe_dir}/deleted" ]]; then
    rm -rf -- "${probe_dir}" 2>/dev/null || true
    swe_rebench_guard_error \
      "${context} enroot-aufs2ovlfs did not convert both opaque and ordinary whiteouts under ${probe_root}. Detected capabilities: ${capabilities}. Verify the host Enroot +caps installation, initial user namespace, and filesystem xattr/device-node support"
    return
  fi
  if ! rm -rf -- "${probe_dir}"; then
    swe_rebench_guard_error \
      "${context} converted whiteouts but could not clean its probe under ${probe_root}"
    return
  fi

  SWE_REBENCH_ENROOT_BIN=${enroot_bin}
  SWE_REBENCH_ENROOT_AUFS2OVLFS_BIN=${helper_bin}
}

swe_rebench_load_validated_source_revision() {
  local repo_root=$1
  local revision_file=${repo_root}/plugins/openreward/swe-rebench-source-revision.txt
  local revision

  [[ -r "${revision_file}" ]] || {
    swe_rebench_guard_error "missing validated SWE-rebench revision file: ${revision_file}"
    return
  }
  revision=$(<"${revision_file}")
  [[ "${revision}" =~ ^[0-9a-f]{40}$ ]] || {
    swe_rebench_guard_error \
      "validated SWE-rebench revision must be exactly one lowercase 40-hex commit: ${revision_file}"
    return
  }
  case "${revision}" in
    25b14c06b9236c075a4ede25bff6979e5783bb09|10c49ab856fd0e62097815ba5909dfc4f31e7f93)
      swe_rebench_guard_error \
        "validated SWE-rebench revision ${revision} is blocked by the sandbox-incident denylist; update ${revision_file} only after the replacement commit passes the sandbox gate"
      return
      ;;
  esac

  SWE_REBENCH_VALIDATED_SOURCE_REVISION=${revision}
  SWE_REBENCH_SOURCE_REVISION_FILE=${revision_file}
}

swe_rebench_require_validated_source() {
  local repo_root=$1
  local source_root=$2
  local requested_revision=${3:-}
  local context=${4:-SWE-rebench launcher}
  local actual_revision
  local source_status

  swe_rebench_load_validated_source_revision "${repo_root}" || return
  if [[ -n "${requested_revision}" && \
        "${requested_revision}" != "${SWE_REBENCH_VALIDATED_SOURCE_REVISION}" ]]; then
    swe_rebench_guard_error \
      "${context} requested SWE-rebench revision ${requested_revision}, but the validated pin is ${SWE_REBENCH_VALIDATED_SOURCE_REVISION}"
    return
  fi
  [[ -d "${source_root}/.git" ]] || {
    swe_rebench_guard_error "${context} cannot find a SWE-rebench Git checkout: ${source_root}"
    return
  }
  if ! actual_revision=$(git -C "${source_root}" rev-parse --verify HEAD); then
    swe_rebench_guard_error "${context} cannot resolve SWE-rebench HEAD: ${source_root}"
    return
  fi
  if [[ "${actual_revision}" != "${SWE_REBENCH_VALIDATED_SOURCE_REVISION}" ]]; then
    swe_rebench_guard_error \
      "${context} SWE-rebench source mismatch: expected ${SWE_REBENCH_VALIDATED_SOURCE_REVISION}, got ${actual_revision} at ${source_root}"
    return
  fi
  if ! source_status=$(git -C "${source_root}" status --porcelain=v1 --untracked-files=normal); then
    swe_rebench_guard_error "${context} cannot inspect SWE-rebench worktree: ${source_root}"
    return
  fi
  if [[ -n "${source_status}" ]]; then
    swe_rebench_guard_error \
      "${context} requires a clean SWE-rebench worktree at validated revision ${SWE_REBENCH_VALIDATED_SOURCE_REVISION}: ${source_root}"
    return
  fi
}
