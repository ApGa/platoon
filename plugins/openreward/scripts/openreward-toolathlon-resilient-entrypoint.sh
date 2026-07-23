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
NGINX_CONFIG=${OPENREWARD_NGINX_CONFIG:-/etc/nginx/nginx.conf}
NGINX_PID_FILE=${OPENREWARD_NGINX_PID_FILE:-/run/nginx.pid}
NGINX_LOG_DIR=${OPENREWARD_NGINX_LOG_DIR:-/var/log/nginx}
SKIP_POSTGRES=${OPENREWARD_ENTRYPOINT_SKIP_POSTGRES:-0}
WORKER_PID_ISOLATION=${OPENREWARD_WORKER_PID_ISOLATION:-1}

require_uint() {
    local name=$1
    local value=$2
    local minimum=$3
    if [[ -z "${value}" || "${value}" == *[!0-9]* ]] || ((10#${value} < minimum)); then
        echo "[entrypoint] ${name} must be an integer >= ${minimum}; got ${value@Q}" >&2
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
        service postgresql stop >/dev/null 2>&1 || true
    fi
}

handle_signal() {
    local signal=$1
    echo "[entrypoint] received ${signal}; stopping"
    exit 0
}

trap cleanup EXIT
trap 'handle_signal TERM' TERM
trap 'handle_signal INT' INT

echo "[entrypoint] public_port=${PUBLIC_PORT} workers=${WORKERS} base_port=${BASE_PORT} pg_max_connections=${PG_MAX_CONN} frozen_date=${OPENREWARD_FROZEN_DATE:-<wall-clock>} max_worker_restarts=${MAX_RESTARTS}"

if [[ "${SKIP_POSTGRES}" == 0 ]]; then
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

    psql -U eigent -d postgres -tAc \
        "SELECT datname FROM pg_database WHERE datname LIKE 's\\_%' ESCAPE '\\'" |
        while IFS= read -r db; do
            [[ -z "${db}" ]] && continue
            echo "[entrypoint] dropping orphan DB: ${db}"
            psql -U eigent -d postgres -c "DROP DATABASE IF EXISTS \"${db}\" WITH (FORCE);" || true
        done
fi

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
}

upstream_servers=
for ((i = 0; i < WORKERS; i++)); do
    worker_failures[i]=0
    start_worker "${i}"
    port=$((BASE_PORT + i))
    upstream_servers+="        server 127.0.0.1:${port} max_fails=0;"$'\n'
done

mkdir -p "$(dirname "${NGINX_PID_FILE}")" "${NGINX_LOG_DIR}" "$(dirname "${NGINX_CONFIG}")"
cat >"${NGINX_CONFIG}" <<EOF
user root;
worker_processes auto;
pid ${NGINX_PID_FILE};
error_log /dev/stderr warn;
events { worker_connections 8192; }
http {
    access_log off;
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

"${NGINX_BIN}" -t -c "${NGINX_CONFIG}"
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
        exit 1
    fi

    index=${pid_to_worker[${exited_pid}]:-}
    if [[ -z "${index}" ]]; then
        echo "[entrypoint] unknown supervised process exited (pid=${exited_pid}, status=${status})" >&2
        exit 1
    fi
    unset 'pid_to_worker['"${exited_pid}"']'
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
