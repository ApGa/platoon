#!/bin/bash

# Toolathlon's nginx front end hashes each ORS session to a stable internal
# Uvicorn port. Keep those ports stable while supervising the worker processes:
# an isolated worker failure should only invalidate that worker's sessions, not
# stop every server on the node (and, through srun, every server in the job).

set -euo pipefail

PUBLIC_PORT=${OPENREWARD_PORT:-${PORT:-8080}}
BASE_PORT=${OPENREWARD_WORKER_BASE_PORT:-8100}
PG_MAX_CONN=${OPENREWARD_PG_MAX_CONNECTIONS:-1000}
WORKERS=${OPENREWARD_WORKERS:-0}
MAX_RESTARTS=${OPENREWARD_WORKER_RESTART_MAX_ATTEMPTS:-5}
RESTART_RESET_SECS=${OPENREWARD_WORKER_RESTART_RESET_SECS:-300}
BACKOFF_INITIAL_SECS=${OPENREWARD_WORKER_RESTART_BACKOFF_INITIAL_SECS:-1}
BACKOFF_MAX_SECS=${OPENREWARD_WORKER_RESTART_BACKOFF_MAX_SECS:-30}
SHUTDOWN_GRACE_SECS=${OPENREWARD_SERVER_SHUTDOWN_GRACE_SECS:-15}
WORKER_PYTHON=${OPENREWARD_SERVER_PYTHON:-python3}
WORKER_APP=${OPENREWARD_SERVER_APP:-/app/server.py}
NGINX_BIN=${OPENREWARD_NGINX_BIN:-nginx}
SKIP_POSTGRES=${OPENREWARD_ENTRYPOINT_SKIP_POSTGRES:-0}
WORKER_PID_ISOLATION=${OPENREWARD_WORKER_PID_ISOLATION:-1}
RUNTIME_MODE=${OPENREWARD_RUNTIME_MODE:-auto}
PG_START_TIMEOUT_SECS=${OPENREWARD_PG_START_TIMEOUT_SECS:-120}

require_uint() {
    local name=$1
    local value=$2
    local minimum=$3
    if [[ -z "${value}" || "${value}" == *[!0-9]* ]] || ((10#${value} < minimum)); then
        echo "[entrypoint] ${name} must be an integer >= ${minimum}; got ${value@Q}" >&2
        exit 2
    fi
}

require_nginx_config_path() {
    local name=$1
    local value=$2
    if [[ "${value}" != /* || ! "${value}" =~ ^/[A-Za-z0-9_./:@%+,=-]+$ ]]; then
        echo "[entrypoint] ${name} must be an absolute nginx-safe path; got ${value@Q}" >&2
        exit 2
    fi
}

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    WORKERS=$(nproc 2>/dev/null || echo 4)
    if ((WORKERS < 4)); then
        WORKERS=4
    fi
fi

require_uint OPENREWARD_PORT "${PUBLIC_PORT}" 1
require_uint OPENREWARD_WORKER_BASE_PORT "${BASE_PORT}" 1
require_uint OPENREWARD_WORKERS "${WORKERS}" 1
require_uint OPENREWARD_PG_MAX_CONNECTIONS "${PG_MAX_CONN}" 1
require_uint OPENREWARD_WORKER_RESTART_MAX_ATTEMPTS "${MAX_RESTARTS}" 0
require_uint OPENREWARD_WORKER_RESTART_RESET_SECS "${RESTART_RESET_SECS}" 1
require_uint OPENREWARD_WORKER_RESTART_BACKOFF_INITIAL_SECS "${BACKOFF_INITIAL_SECS}" 0
require_uint OPENREWARD_WORKER_RESTART_BACKOFF_MAX_SECS "${BACKOFF_MAX_SECS}" 0
require_uint OPENREWARD_SERVER_SHUTDOWN_GRACE_SECS "${SHUTDOWN_GRACE_SECS}" 0
require_uint OPENREWARD_PG_START_TIMEOUT_SECS "${PG_START_TIMEOUT_SECS}" 1

if ((PUBLIC_PORT > 65535 || BASE_PORT + WORKERS - 1 > 65535)); then
    echo "[entrypoint] invalid port range: public_port=${PUBLIC_PORT} base_port=${BASE_PORT} workers=${WORKERS}" >&2
    exit 2
fi
if [[ "${SKIP_POSTGRES}" != 0 && "${SKIP_POSTGRES}" != 1 ]]; then
    echo "[entrypoint] OPENREWARD_ENTRYPOINT_SKIP_POSTGRES must be 0 or 1" >&2
    exit 2
fi
if [[ "${WORKER_PID_ISOLATION}" != 0 && "${WORKER_PID_ISOLATION}" != 1 ]]; then
    echo "[entrypoint] OPENREWARD_WORKER_PID_ISOLATION must be 0 or 1" >&2
    exit 2
fi

case "${RUNTIME_MODE}" in
    auto)
        if ((EUID == 0)); then
            RUNTIME_MODE=privileged
        else
            RUNTIME_MODE=user
        fi
        ;;
    privileged | user) ;;
    *)
        echo "[entrypoint] OPENREWARD_RUNTIME_MODE must be auto, privileged, or user; got ${RUNTIME_MODE@Q}" >&2
        exit 2
        ;;
esac

# The privileged Slurm/Pyxis path keeps the historical private PID namespace
# default. Native, non-fakeroot Singularity on ABCI already supplies container
# PID isolation and may forbid creating a nested user namespace, so user mode
# defaults this off unless the caller explicitly requests it.
if [[ "${RUNTIME_MODE}" == user && -z "${OPENREWARD_WORKER_PID_ISOLATION+x}" ]]; then
    WORKER_PID_ISOLATION=0
fi

if [[ "${RUNTIME_MODE}" == user ]]; then
    STATE_DIR=${OPENREWARD_STATE_DIR:-${PBS_LOCALDIR:-${TMPDIR:-/tmp}}/openreward-toolathlon-${PUBLIC_PORT}}
    NGINX_CONFIG=${OPENREWARD_NGINX_CONFIG:-${STATE_DIR}/nginx/nginx.conf}
    NGINX_PID_FILE=${OPENREWARD_NGINX_PID_FILE:-${STATE_DIR}/nginx/nginx.pid}
    NGINX_LOG_DIR=${OPENREWARD_NGINX_LOG_DIR:-${STATE_DIR}/nginx/log}
    NGINX_TEMP_DIR=${OPENREWARD_NGINX_TEMP_DIR:-${STATE_DIR}/nginx/tmp}
    PGDATA=${OPENREWARD_PGDATA:-${STATE_DIR}/postgresql/14/main}
    PG_SEED_DIR=${OPENREWARD_PG_SEED_DIR:-/var/lib/postgresql/14/main}
    PG_SOCKET_DIR=${OPENREWARD_PG_SOCKET_DIR:-${STATE_DIR}/postgresql/socket}
    PG_LOG_FILE=${OPENREWARD_PG_LOG_FILE:-${STATE_DIR}/postgresql/postgresql.log}
    PG_PORT=${OPENREWARD_PG_PORT:-5432}
    PG_BIN_DIR=${OPENREWARD_PG_BIN_DIR:-}
    if [[ -z "${PG_BIN_DIR}" ]] && command -v pg_config >/dev/null 2>&1; then
        PG_BIN_DIR=$(pg_config --bindir)
    fi
    PG_BIN_DIR=${PG_BIN_DIR:-/usr/lib/postgresql/14/bin}
    PG_CTL=${OPENREWARD_PG_CTL:-${PG_BIN_DIR}/pg_ctl}
    PG_ISREADY=${OPENREWARD_PG_ISREADY:-${PG_BIN_DIR}/pg_isready}
    PSQL_BIN=${OPENREWARD_PSQL_BIN:-${PG_BIN_DIR}/psql}
    NGINX_USER_DIRECTIVE=
else
    STATE_DIR=${OPENREWARD_STATE_DIR:-/run/openreward-toolathlon-${PUBLIC_PORT}}
    NGINX_CONFIG=${OPENREWARD_NGINX_CONFIG:-/etc/nginx/nginx.conf}
    NGINX_PID_FILE=${OPENREWARD_NGINX_PID_FILE:-/run/nginx.pid}
    NGINX_LOG_DIR=${OPENREWARD_NGINX_LOG_DIR:-/var/log/nginx}
    NGINX_TEMP_DIR=${OPENREWARD_NGINX_TEMP_DIR:-/var/lib/nginx}
    NGINX_USER_DIRECTIVE="user ${OPENREWARD_NGINX_USER:-root};"
fi
NGINX_ERROR_LOG=${NGINX_LOG_DIR}/error.log

# These paths are emitted as unquoted nginx configuration tokens.  Keep the
# accepted alphabet deliberately narrow so an environment override cannot add
# whitespace, comments, or another directive to the generated configuration.
require_nginx_config_path NGINX_CONFIG "${NGINX_CONFIG}"
require_nginx_config_path NGINX_PID_FILE "${NGINX_PID_FILE}"
require_nginx_config_path NGINX_LOG_DIR "${NGINX_LOG_DIR}"
require_nginx_config_path NGINX_ERROR_LOG "${NGINX_ERROR_LOG}"
require_nginx_config_path NGINX_TEMP_DIR "${NGINX_TEMP_DIR}"

# Publish one private, host-visible PID record per worker.  Besides making
# production diagnostics precise, this gives the ABCI acceptance test a safe
# way to kill exactly one worker and prove that the supervisor replaces it.
# The state directory is node-local in both the PBS and historical Slurm paths.
WORKER_PID_DIR=${OPENREWARD_WORKER_PID_DIR:-${STATE_DIR}/worker-pids}
mkdir -p "${WORKER_PID_DIR}"
chmod 0700 "${WORKER_PID_DIR}"

if [[ "${RUNTIME_MODE}" == user ]]; then
    require_uint OPENREWARD_PG_PORT "${PG_PORT}" 1
    if ((PG_PORT > 65535)); then
        echo "[entrypoint] OPENREWARD_PG_PORT must be <= 65535; got ${PG_PORT}" >&2
        exit 2
    fi
fi

declare -a worker_namespace_args=()
if [[ "${WORKER_PID_ISOLATION}" == 1 ]]; then
    worker_namespace_args=(
        unshare
        --user
        --map-current-user
        --pid
        --fork
        --kill-child=SIGKILL
        --mount-proc
        --
    )
    if ! command -v unshare >/dev/null 2>&1; then
        echo "[entrypoint] private worker PID isolation requires unshare" >&2
        exit 2
    fi
    if ! "${worker_namespace_args[@]}" true >/dev/null 2>&1; then
        echo "[entrypoint] this host does not permit private worker PID namespaces" >&2
        exit 2
    fi
fi

declare -a worker_pids=()
declare -a worker_failures=()
declare -a worker_started_at=()
declare -A pid_to_worker=()
nginx_pid=
postgres_started=0
cleanup_started=0

cleanup() {
    local index
    local pid
    local deadline
    local alive

    if ((cleanup_started)); then
        return
    fi
    cleanup_started=1
    trap '' TERM INT
    echo "[entrypoint] shutting down"

    if [[ -n "${nginx_pid}" ]] && kill -0 "${nginx_pid}" 2>/dev/null; then
        "${NGINX_BIN}" -s quit -c "${NGINX_CONFIG}" 2>/dev/null || kill -TERM "${nginx_pid}" 2>/dev/null || true
    fi
    for pid in "${worker_pids[@]:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done
    for index in "${!worker_pids[@]}"; do
        clear_worker_pid "${index}" "${worker_pids[index]}"
    done

    deadline=$((SECONDS + SHUTDOWN_GRACE_SECS))
    while ((SECONDS < deadline)); do
        alive=0
        if [[ -n "${nginx_pid}" ]] && kill -0 "${nginx_pid}" 2>/dev/null; then
            alive=1
        fi
        for pid in "${worker_pids[@]:-}"; do
            if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
                alive=1
                break
            fi
        done
        if ((alive == 0)); then
            break
        fi
        sleep 0.1
    done

    if [[ -n "${nginx_pid}" ]] && kill -0 "${nginx_pid}" 2>/dev/null; then
        kill -KILL "${nginx_pid}" 2>/dev/null || true
    fi
    for pid in "${worker_pids[@]:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    done

    if [[ -n "${nginx_pid}" ]]; then
        wait "${nginx_pid}" 2>/dev/null || true
    fi
    for pid in "${worker_pids[@]:-}"; do
        if [[ -n "${pid}" ]]; then
            wait "${pid}" 2>/dev/null || true
        fi
    done
    if ((postgres_started)); then
        if [[ "${RUNTIME_MODE}" == user ]]; then
            "${PG_CTL}" -D "${PGDATA}" -m fast -w \
                -t "${SHUTDOWN_GRACE_SECS}" stop >/dev/null 2>&1 || true
        else
            service postgresql stop >/dev/null 2>&1 || true
        fi
    fi
}

report_nginx_error_tail() {
    if [[ ! -r "${NGINX_ERROR_LOG}" ]]; then
        return
    fi
    echo "[entrypoint] nginx error log tail (last 100 lines, at most 16384 bytes): ${NGINX_ERROR_LOG}" >&2
    LC_ALL=C tail -c 16384 -- "${NGINX_ERROR_LOG}" 2>/dev/null \
        | LC_ALL=C tail -n 100 \
        | LC_ALL=C tr -cd '\11\12\40-\176' >&2 || true
    printf '\n' >&2
}

handle_signal() {
    local signal=$1
    echo "[entrypoint] received ${signal}; stopping"
    exit 0
}

trap cleanup EXIT
trap 'handle_signal TERM' TERM
trap 'handle_signal INT' INT

echo "[entrypoint] runtime_mode=${RUNTIME_MODE} public_port=${PUBLIC_PORT} workers=${WORKERS} base_port=${BASE_PORT} pg_max_connections=${PG_MAX_CONN} frozen_date=${OPENREWARD_FROZEN_DATE:-<wall-clock>} max_worker_restarts=${MAX_RESTARTS}"

if [[ "${SKIP_POSTGRES}" == 0 ]]; then
    if [[ "${RUNTIME_MODE}" == user ]]; then
        for required_binary in "${PG_CTL}" "${PG_ISREADY}" "${PSQL_BIN}"; do
            if [[ ! -x "${required_binary}" ]]; then
                echo "[entrypoint] PostgreSQL binary is missing or not executable: ${required_binary}" >&2
                exit 2
            fi
        done
        if [[ ! -r "${PGDATA}/PG_VERSION" || ! -r "${PGDATA}/global/pg_control" ]]; then
            echo "[entrypoint] user-mode PGDATA is not a staged PostgreSQL cluster: ${PGDATA}" >&2
            echo "[entrypoint] stage ${PG_SEED_DIR} into node-local storage with openreward-toolathlon-stage-pgdata.sh before starting this service" >&2
            exit 2
        fi
        pgdata_owner=$(stat -c %u "${PGDATA}")
        if [[ "${pgdata_owner}" != "$(id -u)" ]]; then
            echo "[entrypoint] user-mode PGDATA must be owned by uid $(id -u); ${PGDATA} is owned by uid ${pgdata_owner}" >&2
            exit 2
        fi
        foreign_pg_path=$(find "${PGDATA}" -xdev ! -user "$(id -u)" -print -quit)
        if [[ -n "${foreign_pg_path}" ]]; then
            echo "[entrypoint] user-mode PGDATA contains a path not owned by uid $(id -u): ${foreign_pg_path}" >&2
            exit 2
        fi

        mkdir -p "${STATE_DIR}" "$(dirname "${PG_LOG_FILE}")" "${PG_SOCKET_DIR}"
        chmod 0700 "${PGDATA}" "${PG_SOCKET_DIR}"
        PG_CONF=${PGDATA}/postgresql.conf
        PG_HBA=${PGDATA}/pg_hba.conf
        PG_IDENT=${PGDATA}/pg_ident.conf
        PG_EXTERNAL_PID=${STATE_DIR}/postgresql/postgresql.pid
        pgdata_conf=${PGDATA//\\/\\\\}
        pgdata_conf=${pgdata_conf//\'/\'\'}
        pg_hba_conf=${PG_HBA//\\/\\\\}
        pg_hba_conf=${pg_hba_conf//\'/\'\'}
        pg_ident_conf=${PG_IDENT//\\/\\\\}
        pg_ident_conf=${pg_ident_conf//\'/\'\'}
        pg_external_pid_conf=${PG_EXTERNAL_PID//\\/\\\\}
        pg_external_pid_conf=${pg_external_pid_conf//\'/\'\'}
        pg_socket_conf=${PG_SOCKET_DIR//\\/\\\\}
        pg_socket_conf=${pg_socket_conf//\'/\'\'}
        cat >"${PG_HBA}" <<'EOF'
local all all trust
host all all 127.0.0.1/32 trust
host all all ::1/128 trust
EOF
        : >"${PG_IDENT}"
        cat >"${PG_CONF}" <<EOF
data_directory = '${pgdata_conf}'
hba_file = '${pg_hba_conf}'
ident_file = '${pg_ident_conf}'
external_pid_file = '${pg_external_pid_conf}'
listen_addresses = '127.0.0.1'
port = ${PG_PORT}
max_connections = ${PG_MAX_CONN}
unix_socket_directories = '${pg_socket_conf}'
ssl = off
EOF
        chmod 0600 "${PG_CONF}" "${PG_HBA}" "${PG_IDENT}"

        echo "[entrypoint] starting user PostgreSQL pgdata=${PGDATA} port=${PG_PORT} socket_dir=${PG_SOCKET_DIR}"
        if ! "${PG_CTL}" -D "${PGDATA}" -l "${PG_LOG_FILE}" -w \
            -t "${PG_START_TIMEOUT_SECS}" start; then
            "${PG_CTL}" -D "${PGDATA}" -m immediate stop >/dev/null 2>&1 || true
            echo "[entrypoint] PostgreSQL failed to start; log follows: ${PG_LOG_FILE}" >&2
            tail -n 100 "${PG_LOG_FILE}" >&2 2>/dev/null || true
            exit 1
        fi
        postgres_started=1
        until "${PG_ISREADY}" -h 127.0.0.1 -p "${PG_PORT}" -U eigent -d postgres >/dev/null 2>&1; do
            if ! "${PG_CTL}" -D "${PGDATA}" status >/dev/null 2>&1; then
                echo "[entrypoint] PostgreSQL exited before becoming ready" >&2
                tail -n 100 "${PG_LOG_FILE}" >&2 2>/dev/null || true
                exit 1
            fi
            sleep 1
        done
        psql_connection_args=(-h 127.0.0.1 -p "${PG_PORT}" -U eigent)
    else
        chown -R postgres:postgres /var/lib/postgresql 2>/dev/null || true
        chown -R postgres:postgres /etc/ssl/private 2>/dev/null || true
        chmod 0600 /etc/ssl/private/ssl-cert-snakeoil.key 2>/dev/null || true

        PG_CONF=$(find /etc/postgresql -name postgresql.conf -print -quit)
        if [[ -n "${PG_CONF}" ]]; then
            if grep -qE '^[[:space:]]*max_connections[[:space:]]*=' "${PG_CONF}"; then
                sed -i "s/^[[:space:]]*max_connections[[:space:]]*=.*/max_connections = ${PG_MAX_CONN}/" "${PG_CONF}"
            else
                echo "max_connections = ${PG_MAX_CONN}" >>"${PG_CONF}"
            fi
        fi

        service postgresql start
        postgres_started=1
        until pg_isready -U eigent 2>/dev/null; do
            sleep 1
        done
        PSQL_BIN=psql
        psql_connection_args=(-U eigent)
    fi

    "${PSQL_BIN}" "${psql_connection_args[@]}" -d postgres -tAc \
        "SELECT datname FROM pg_database WHERE datname LIKE 's\\_%' ESCAPE '\\'" |
        while IFS= read -r db; do
            [[ -z "${db}" ]] && continue
            echo "[entrypoint] dropping orphan DB: ${db}"
            "${PSQL_BIN}" "${psql_connection_args[@]}" -d postgres \
                -c "DROP DATABASE IF EXISTS \"${db}\" WITH (FORCE);" || true
        done
fi

worker_pid_path() {
    local index=$1
    printf '%s/worker-%s.pid' "${WORKER_PID_DIR}" "${index}"
}

publish_worker_pid() {
    local index=$1
    local pid=$2
    local path temporary

    path=$(worker_pid_path "${index}")
    temporary=${path}.tmp.$$
    printf '%s\n' "${pid}" >"${temporary}"
    chmod 0600 "${temporary}"
    mv -f "${temporary}" "${path}"
}

clear_worker_pid() {
    local index=$1
    local expected_pid=$2
    local path current_pid=

    path=$(worker_pid_path "${index}")
    [[ -f "${path}" ]] || return 0
    IFS= read -r current_pid <"${path}" || true
    if [[ "${current_pid}" == "${expected_pid}" ]]; then
        rm -f -- "${path}"
    fi
}

start_worker() {
    local index=$1
    local port=$((BASE_PORT + index))
    local pid

    echo "[entrypoint] starting worker ${index} on 127.0.0.1:${port}"
    "${worker_namespace_args[@]}" env -u OPENREWARD_PORT PORT="${port}" "${WORKER_PYTHON}" "${WORKER_APP}" &
    pid=$!
    worker_pids[index]=${pid}
    worker_started_at[index]=${SECONDS}
    pid_to_worker[${pid}]=${index}
    publish_worker_pid "${index}" "${pid}"
}

upstream_servers=
for ((i = 0; i < WORKERS; i++)); do
    worker_failures[i]=0
    start_worker "${i}"
    port=$((BASE_PORT + i))
    upstream_servers+="        server 127.0.0.1:${port} max_fails=0;"$'\n'
done

mkdir -p "$(dirname "${NGINX_PID_FILE}")" "${NGINX_LOG_DIR}" \
    "$(dirname "${NGINX_ERROR_LOG}")" \
    "$(dirname "${NGINX_CONFIG}")" \
    "${NGINX_TEMP_DIR}/client-body" "${NGINX_TEMP_DIR}/proxy" \
    "${NGINX_TEMP_DIR}/fastcgi" "${NGINX_TEMP_DIR}/uwsgi" \
    "${NGINX_TEMP_DIR}/scgi"
: >"${NGINX_ERROR_LOG}"
chmod 0600 "${NGINX_ERROR_LOG}"
cat >"${NGINX_CONFIG}" <<EOF
${NGINX_USER_DIRECTIVE}
worker_processes auto;
pid ${NGINX_PID_FILE};
error_log ${NGINX_ERROR_LOG} warn;
events { worker_connections 8192; }
http {
    access_log off;
    client_body_temp_path ${NGINX_TEMP_DIR}/client-body;
    proxy_temp_path ${NGINX_TEMP_DIR}/proxy;
    fastcgi_temp_path ${NGINX_TEMP_DIR}/fastcgi;
    uwsgi_temp_path ${NGINX_TEMP_DIR}/uwsgi;
    scgi_temp_path ${NGINX_TEMP_DIR}/scgi;
    upstream ors_backends {
        hash \$http_x_session_id consistent;
${upstream_servers}    }
    server {
        listen ${PUBLIC_PORT};
        client_max_body_size 0;
        location / {
            proxy_pass http://ors_backends;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host \$host;
            proxy_buffering off;
            proxy_cache off;
            proxy_read_timeout 3600s;
            proxy_send_timeout 3600s;
            chunked_transfer_encoding off;
        }
    }
}
EOF

if ! "${NGINX_BIN}" -t -c "${NGINX_CONFIG}"; then
    echo "[entrypoint] nginx configuration test failed" >&2
    report_nginx_error_tail
    exit 1
fi
echo "[entrypoint] starting nginx on :${PUBLIC_PORT}"
"${NGINX_BIN}" -c "${NGINX_CONFIG}" -g 'daemon off;' &
nginx_pid=$!

restart_backoff() {
    local attempt=$1
    local delay=${BACKOFF_INITIAL_SECS}
    local n

    for ((n = 1; n < attempt; n++)); do
        if ((delay >= BACKOFF_MAX_SECS)); then
            break
        fi
        delay=$((delay * 2))
    done
    if ((delay > BACKOFF_MAX_SECS)); then
        delay=${BACKOFF_MAX_SECS}
    fi
    printf '%s' "${delay}"
}

while true; do
    wait_pids=()
    for pid in "${worker_pids[@]:-}"; do
        if [[ -n "${pid}" ]]; then
            wait_pids+=("${pid}")
        fi
    done
    wait_pids+=("${nginx_pid}")

    exited_pid=
    if wait -n -p exited_pid "${wait_pids[@]}"; then
        status=0
    else
        status=$?
    fi

    if [[ -z "${exited_pid}" ]]; then
        echo "[entrypoint] supervisor wait failed without an exited pid (status=${status})" >&2
        exit 1
    fi
    if [[ "${exited_pid}" == "${nginx_pid}" ]]; then
        echo "[entrypoint] nginx exited unexpectedly (pid=${exited_pid}, status=${status}); failing node service" >&2
        nginx_pid=
        report_nginx_error_tail
        exit 1
    fi

    index=${pid_to_worker[${exited_pid}]:-}
    if [[ -z "${index}" ]]; then
        echo "[entrypoint] unknown supervised process exited (pid=${exited_pid}, status=${status})" >&2
        exit 1
    fi
    unset 'pid_to_worker['"${exited_pid}"']'
    clear_worker_pid "${index}" "${exited_pid}"
    worker_pids[index]=

    runtime=$((SECONDS - worker_started_at[index]))
    failures=${worker_failures[index]}
    if ((runtime >= RESTART_RESET_SECS)); then
        failures=0
    fi
    failures=$((failures + 1))
    worker_failures[index]=${failures}

    if ((failures > MAX_RESTARTS)); then
        echo "[entrypoint] worker ${index} exhausted restart budget after status=${status}; failing node service" >&2
        exit 1
    fi

    delay=$(restart_backoff "${failures}")
    echo "[entrypoint] worker ${index} exited unexpectedly (pid=${exited_pid}, status=${status}, runtime=${runtime}s); restart ${failures}/${MAX_RESTARTS} in ${delay}s"
    sleep "${delay}"
    start_worker "${index}"
done
