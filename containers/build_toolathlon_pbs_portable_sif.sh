#!/bin/bash
# Build a patch-addressed PBS-portable child of the exact pinned Toolathlon SIF.
# The raw SIF SHA-256 is authoritative and is recorded only after the build.
set -euo pipefail
umask 077

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd -P)
BASE_SIF_DEFAULT=/groups/gai51740/aci18912it/rasp/images/openreward-toolathlon-18e62c0d-c1e60b36.sif
BASE_SIF=${1:-${BASE_SIF_DEFAULT}}
OUTPUT_DIR=${2:-/groups/gai51740/aci18912it/rasp/images}
EXPECTED_BASE_SHA256=e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267
SINGULARITY=/usr/bin/singularity
EXPECTED_SINGULARITY_SHA256=085a757f172dcabda66b44d086913314d6aa574055a8fea90523be82b770cec6
EXPECTED_SINGULARITY_VERSION=4.4.1-1.el9

for asset in \
    toolathlon-pbs-portable.def \
    patch_toolathlon_pbs_portable.py \
    repair_toolathlon_snowflake_catalog.py \
    verify_toolathlon_pbs_portability.sh \
    publish_toolathlon_pbs_image.py \
    publish_toolathlon_pbs_build_receipt.py; do
    test -r "${SCRIPT_DIR}/${asset}"
done
test -r "${BASE_SIF}"
test ! -L "${BASE_SIF}"
test "$(realpath -e "${BASE_SIF}")" = "${BASE_SIF}"
test -f "${BASE_SIF}"
test "$(sha256sum "${BASE_SIF}" | cut -d ' ' -f 1)" = "${EXPECTED_BASE_SHA256}"
# Pin the trusted site runtime before it can create or publish any SIF.  The
# receipt publisher independently repeats this verification after the build.
test -f "${SINGULARITY}"
test ! -L "${SINGULARITY}"
test "$(realpath -e "${SINGULARITY}")" = "${SINGULARITY}"
case "$(stat -c %u "${SINGULARITY}")" in
    0|65534) ;;
    *) echo "untrusted Singularity owner" >&2; exit 2 ;;
esac
SINGULARITY_MODE=$(stat -c %a "${SINGULARITY}")
test $((8#${SINGULARITY_MODE} & 8#022)) -eq 0
test $((8#${SINGULARITY_MODE} & 8#111)) -ne 0
test "$(sha256sum "${SINGULARITY}" | cut -d ' ' -f 1)" = "${EXPECTED_SINGULARITY_SHA256}"
test "$("${SINGULARITY}" version)" = "${EXPECTED_SINGULARITY_VERSION}"

PATCH_ASSETS=(
    toolathlon-pbs-portable.def
    patch_toolathlon_pbs_portable.py
    repair_toolathlon_snowflake_catalog.py
    verify_toolathlon_pbs_portability.sh
)
PATCH_IDENTITY=$(
    for asset in "${PATCH_ASSETS[@]}"; do
        printf '%s  %s\n' "$(sha256sum "${SCRIPT_DIR}/${asset}" | cut -d ' ' -f 1)" "${asset}"
    done | sha256sum | cut -d ' ' -f 1
)
SHORT_ID=${PATCH_IDENTITY:0:12}
OUTPUT_NAME=openreward-toolathlon-18e62c0d-pbs-${SHORT_ID}.sif
OUTPUT_SIF=${OUTPUT_DIR}/${OUTPUT_NAME}
RECEIPT_DIR=${OUTPUT_DIR}/build-receipts
BUILD_RECEIPT=${RECEIPT_DIR}/${OUTPUT_NAME}.json
JOB_TOKEN=${PBS_JOBID:-local}
JOB_TOKEN=${JOB_TOKEN//[^A-Za-z0-9_.-]/_}
PENDING_SIF=${OUTPUT_SIF}.pending.${JOB_TOKEN}.$$
WORK_ROOT=${TMPDIR:-/tmp}/toolathlon-pbs-portable-build.$$
DEFINITION=${WORK_ROOT}/toolathlon-pbs-portable.def

cleanup() {
    rm -f "${PENDING_SIF}" 2>/dev/null || true
    rm -f "${WORK_ROOT}/patch_toolathlon_pbs_portable.py" \
        "${WORK_ROOT}/repair_toolathlon_snowflake_catalog.py" \
        "${WORK_ROOT}/verify_toolathlon_pbs_portability.sh" \
        "${DEFINITION}" 2>/dev/null || true
    rmdir "${WORK_ROOT}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
mkdir -m 700 "${WORK_ROOT}"
mkdir -p "${OUTPUT_DIR}"
python3 - "${OUTPUT_DIR}" <<'PY'
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_absolute() or path.resolve() != path:
    raise SystemExit(f"output directory is not direct/canonical: {path}")
value = path.lstat()
mode = stat.S_IMODE(value.st_mode)
groups = set(os.getgroups()) | {os.getgid()}
if not (
    stat.S_ISDIR(value.st_mode)
    and value.st_uid == os.getuid()
    and (
        (mode == 0o700 and value.st_gid == os.getgid())
        or (mode == 0o2770 and value.st_gid in groups)
    )
):
    raise SystemExit(
        "output directory must be user-owned mode 0700 or user-owned setgid "
        f"mode 2770 for one of the user's groups: {path}"
    )
PY
mkdir -p -m 0700 "${RECEIPT_DIR}"
python3 - "${RECEIPT_DIR}" <<'PY'
import os
import stat
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_absolute() or path.resolve() != path:
    raise SystemExit(f"receipt directory is not direct/canonical: {path}")
value = path.lstat()
mode = stat.S_IMODE(value.st_mode)
# A setgid parent on ABCI can cause mkdir(0700) to yield 2700.  Setgid is
# harmless here because all group/other permission bits remain clear.
if not (
    stat.S_ISDIR(value.st_mode)
    and value.st_uid == os.getuid()
    and mode & 0o077 == 0
):
    raise SystemExit(
        f"receipt directory must be user-owned and private (0700/2700): {path}"
    )
PY
cp "${SCRIPT_DIR}/patch_toolathlon_pbs_portable.py" \
    "${SCRIPT_DIR}/repair_toolathlon_snowflake_catalog.py" \
    "${SCRIPT_DIR}/verify_toolathlon_pbs_portability.sh" \
    "${WORK_ROOT}/"
sed \
    -e "s|BASE_SIF_PATH|${BASE_SIF}|" \
    -e "s|PATCH_IDENTITY|${PATCH_IDENTITY}|g" \
    "${SCRIPT_DIR}/toolathlon-pbs-portable.def" >"${DEFINITION}"

if [[ -e "${OUTPUT_SIF}" ]]; then
    python3 "${SCRIPT_DIR}/publish_toolathlon_pbs_build_receipt.py" \
        --image "${OUTPUT_SIF}" \
        --receipt "${BUILD_RECEIPT}" \
        --base-sif "${BASE_SIF}" \
        --patch-identity "${PATCH_IDENTITY}" \
        --asset-dir "${SCRIPT_DIR}" \
        --builder "${BASH_SOURCE[0]}" \
        --singularity "${SINGULARITY}"
    OUTPUT_SHA256=$(sha256sum "${OUTPUT_SIF}" | cut -d ' ' -f 1)
    printf '%s\n' 'toolathlon_pbs_portable_build=reused_verified_output'
    printf 'toolathlon_revision=%s\n' 18e62c0d0419da88209af911528f2f8a0b44a5ed
    printf 'base_sif_sha256=%s\n' "${EXPECTED_BASE_SHA256}"
    printf 'portability_patch_sha256=%s\n' "${PATCH_IDENTITY}"
    printf 'output_sif=%s\n' "${OUTPUT_SIF}"
    printf 'output_sif_sha256=%s\n' "${OUTPUT_SHA256}"
    printf 'build_receipt=%s\n' "${BUILD_RECEIPT}"
    printf 'build_receipt_sha256=%s\n' "$(sha256sum "${BUILD_RECEIPT}" | cut -d ' ' -f 1)"
    exit 0
fi
if [[ -e "${PENDING_SIF}" ]]; then
    echo "refusing unexpected pending output: ${PENDING_SIF}" >&2
    exit 2
fi

(
    cd "${WORK_ROOT}"
    "${SINGULARITY}" build --fakeroot "${PENDING_SIF}" "${DEFINITION}"
)
test -f "${PENDING_SIF}"
test ! -L "${PENDING_SIF}"
chmod 0550 "${PENDING_SIF}"
"${SINGULARITY}" sif list "${PENDING_SIF}" >/dev/null
"${SINGULARITY}" inspect --labels "${PENDING_SIF}" >/dev/null
"${SINGULARITY}" exec --cleanenv --no-home "${PENDING_SIF}" \
    /usr/local/libexec/openreward/verify_toolathlon_pbs_portability.sh
OUTPUT_SHA256=$(sha256sum "${PENDING_SIF}" | cut -d ' ' -f 1)

# Atomic, no-overwrite publication plus parent-directory fsync.
python3 "${SCRIPT_DIR}/publish_toolathlon_pbs_image.py" \
    --pending "${PENDING_SIF}" --output "${OUTPUT_SIF}"
test "$(sha256sum "${OUTPUT_SIF}" | cut -d ' ' -f 1)" = "${OUTPUT_SHA256}"
python3 "${SCRIPT_DIR}/publish_toolathlon_pbs_build_receipt.py" \
    --image "${OUTPUT_SIF}" \
    --receipt "${BUILD_RECEIPT}" \
    --base-sif "${BASE_SIF}" \
    --patch-identity "${PATCH_IDENTITY}" \
    --asset-dir "${SCRIPT_DIR}" \
    --builder "${BASH_SOURCE[0]}" \
    --singularity "${SINGULARITY}"
printf 'toolathlon_revision=%s\n' 18e62c0d0419da88209af911528f2f8a0b44a5ed
printf 'base_sif_sha256=%s\n' "${EXPECTED_BASE_SHA256}"
printf 'portability_patch_sha256=%s\n' "${PATCH_IDENTITY}"
printf 'output_sif=%s\n' "${OUTPUT_SIF}"
printf 'output_sif_sha256=%s\n' "${OUTPUT_SHA256}"
printf 'build_receipt=%s\n' "${BUILD_RECEIPT}"
printf 'build_receipt_sha256=%s\n' "$(sha256sum "${BUILD_RECEIPT}" | cut -d ' ' -f 1)"
