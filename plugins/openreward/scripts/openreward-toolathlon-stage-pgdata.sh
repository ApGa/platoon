#!/bin/bash

# Copy Toolathlon's image-baked PostgreSQL cluster into disposable node-local
# storage. On ABCI this script is run in a short `singularity exec --fakeroot`
# process because the seed directory in the SIF is readable only by image uid
# 0. Fakeroot maps that uid to the submitting uid on the host, so the published
# PGDATA is suitable for the subsequent non-fakeroot service process.

set -euo pipefail
umask 077

SOURCE=${OPENREWARD_PG_SEED_DIR:-/var/lib/postgresql/14/main}
DESTINATION=${1:-${OPENREWARD_PGDATA:-}}

if [[ -z "${DESTINATION}" ]]; then
    echo "usage: $0 DESTINATION_PGDATA" >&2
    exit 2
fi
if [[ "${DESTINATION}" != /* ]]; then
    echo "[pgdata-stage] destination must be an absolute path: ${DESTINATION@Q}" >&2
    exit 2
fi
if [[ -L "${DESTINATION}" ]]; then
    echo "[pgdata-stage] refusing symlink destination: ${DESTINATION}" >&2
    exit 2
fi
if [[ ! -r "${SOURCE}/PG_VERSION" || ! -r "${SOURCE}/global/pg_control" ]]; then
    echo "[pgdata-stage] seed is not readable as a PostgreSQL cluster: ${SOURCE}" >&2
    echo "[pgdata-stage] run this helper inside the Toolathlon SIF with Singularity --fakeroot" >&2
    exit 2
fi

if [[ -e "${DESTINATION}/PG_VERSION" && -e "${DESTINATION}/global/pg_control" ]]; then
    owner=$(stat -c %u "${DESTINATION}")
    if [[ "${owner}" != "$(id -u)" ]]; then
        echo "[pgdata-stage] existing destination is owned by uid ${owner}, expected $(id -u): ${DESTINATION}" >&2
        exit 2
    fi
    echo "[pgdata-stage] existing PostgreSQL cluster is ready: ${DESTINATION}"
    exit 0
fi
if [[ -e "${DESTINATION}" ]] && [[ -n "$(find "${DESTINATION}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "[pgdata-stage] refusing nonempty, invalid destination: ${DESTINATION}" >&2
    exit 2
fi

parent=$(dirname "${DESTINATION}")
mkdir -p "${parent}"
stage_dir=$(mktemp -d "${parent}/.openreward-pgdata-stage.XXXXXX")
cleanup() {
    if [[ -n "${stage_dir:-}" && -d "${stage_dir}" ]]; then
        rm -rf -- "${stage_dir}"
    fi
}
trap cleanup EXIT INT TERM

echo "[pgdata-stage] copying ${SOURCE} to ${DESTINATION}"
cp -a --no-preserve=ownership "${SOURCE}/." "${stage_dir}/"
chmod -R u+rwX,go-rwx "${stage_dir}"
chmod 0700 "${stage_dir}"
test -r "${stage_dir}/PG_VERSION"
test -r "${stage_dir}/global/pg_control"

if [[ -d "${DESTINATION}" ]]; then
    rmdir "${DESTINATION}"
fi
mv "${stage_dir}" "${DESTINATION}"
stage_dir=
echo "[pgdata-stage] staged PostgreSQL cluster: ${DESTINATION}"
