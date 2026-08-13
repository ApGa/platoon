#!/usr/bin/env bash
# PBS/Singularity adapter for the established content-addressed OpenReward
# environment builder.  The Slurm helper remains byte-for-byte unchanged; this
# adapter substitutes only the cluster-owned CUDA extension installers before
# executing or materializing it.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
UPSTREAM_HELPER=${REPO_ROOT}/slurm-scripts/prepare_openreward_env.sh

die() {
    echo "[pbs-openreward-env] ERROR: $*" >&2
    exit 1
}

[[ -r "${UPSTREAM_HELPER}" ]] || die "upstream environment helper is missing: ${UPSTREAM_HELPER}"

transform_helper() {
    # These replacements affect both the immutable source-input manifest and
    # the later build calls.  Therefore the source tar contains the pinned PBS
    # installers, and install_te/install_apex are executed from that tar rather
    # than from the mutable checkout.
    sed \
        -e 's#slurm-scripts/install_te\.sh#pbs-scripts/install_te.sh#g' \
        -e 's#slurm-scripts/install_apex\.sh#pbs-scripts/install_apex.sh#g' \
        | awk '
            # The inherited helper installs an EXIT trap whose callback reads
            # function-local variables.  Bash unwinds those locals before an
            # EXIT trap after `set -e`, masking the real build error with an
            # unbound-variable error and leaving staging data behind.  Keep
            # the Slurm helper unchanged, but make the materialized PBS copy
            # retain private cleanup state until process exit.
            $0 == "  cleanup_build() {" {
                print "  _pbs_cleanup_build_root=${build_root}"
                print "  _pbs_cleanup_final_env=${final_env}"
                print "  _pbs_cleanup_failed_root=${cache_root}/.failed"
                print "  _pbs_cleanup_env_key=${env_key}"
                print "  _pbs_cleanup_promoted=0"
                print "  _pbs_cleanup_ready=0"
            }
            $0 == "    if [[ \"${promoted}\" == \"1\" && \"${ready}\" != \"1\" && -e \"${final_env}\" ]]; then" {
                print "    if [[ \"${_pbs_cleanup_promoted:-0}\" == \"1\" && \"${_pbs_cleanup_ready:-0}\" != \"1\" && -e \"${_pbs_cleanup_final_env:-}\" ]]; then"
                next
            }
            $0 == "      mv \"${final_env}\" \"${cache_root}/.failed/${env_key}.promotion.$(date +%s).$$.${RANDOM}\" 2>/dev/null || true" {
                print "      mv \"${_pbs_cleanup_final_env}\" \"${_pbs_cleanup_failed_root}/${_pbs_cleanup_env_key}.promotion.$(date +%s).$$.${RANDOM}\" 2>/dev/null || true"
                next
            }
            $0 == "    [[ -n \"${build_root}\" && -d \"${build_root}\" ]] && rm -rf \"${build_root}\"" {
                print "    [[ -n \"${_pbs_cleanup_build_root:-}\" && -d \"${_pbs_cleanup_build_root}\" ]] && rm -rf \"${_pbs_cleanup_build_root}\""
                next
            }
            { print }
            $0 == "  promoted=1" { print "  _pbs_cleanup_promoted=1" }
            $0 == "  ready=1" { print "  _pbs_cleanup_ready=1" }
            $0 == "  build_root=" { print "  _pbs_cleanup_build_root=" }
            $0 == "    pbs-scripts/install_apex.sh" {
                print "    pbs-scripts/prepare_openreward_controller_env.sh"
                print "    pbs-scripts/openreward-v8-abci.env"
                print "    pbs-scripts/openreward-v8-release.json"
                print "    pbs-scripts/verify_openreward_v8_release.py"
                print "    pbs-scripts/verify_openreward_v8_swe_env.py"
                print "    pbs-scripts/materialize_openreward_v8_swe_env.py"
                print "    containers/abci-training-runtime.lock"
                print "    containers/swe-rebench-538b5c5-py311.lock.txt"
                print "    containers/transformer-engine-build-constraints.txt"
            }
        '
}

# Include this adapter's exact content in the environment key even though the
# large, shared implementation comes from the unchanged Slurm helper.
adapter_sha=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')
export OPENREWARD_ENV_SCHEMA=${OPENREWARD_ENV_SCHEMA:-pbs-singularity-1-${adapter_sha}}

if [[ "${1:-}" == materialize ]]; then
    [[ $# -eq 3 ]] || die "usage: $0 materialize SOURCE_ARCHIVE OUTPUT"
    source_archive=$2
    output=$3
    [[ -r "${source_archive}" ]] || die "source archive is not readable: ${source_archive}"
    output_parent=$(dirname -- "${output}")
    mkdir -p "${output_parent}"
    temporary=$(mktemp "${output_parent}/.prepare-openreward-env.XXXXXXXX")
    cleanup() {
        rm -f -- "${temporary:-}"
    }
    trap cleanup EXIT INT TERM
    tar -xOf "${source_archive}" slurm-scripts/prepare_openreward_env.sh \
        | transform_helper >"${temporary}"
    chmod 0555 "${temporary}"
    mv -f "${temporary}" "${output}"
    temporary=
    exit 0
fi

transform_helper <"${UPSTREAM_HELPER}" | /bin/bash -s -- "$@"
