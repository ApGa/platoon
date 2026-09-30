#!/usr/bin/env bash
# The frontend stays on the host to manage isolated Singularity task/judge instances.
set -euo pipefail
umask 077
ENV_FILE=${PLATOON_ABCI_ENV_FILE:?Set a credential-free Frontier-CS release env file}
[[ -r $ENV_FILE ]] || { echo "Missing Frontier-CS environment: $ENV_FILE" >&2; exit 2; }
source "$ENV_FILE"
for secret_name in WANDB_API_KEY OPENAI_API_KEY OPENAI_BASE_URL LITELLM_API_KEY \
    LITELLM_BASE_URL RUBRIC_DEFAULT_LLM HF_TOKEN AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN TINKER_API_KEY; do
  unset "$secret_name"
done
unset secret_name

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CLEANUP_HELPER=$SCRIPT_DIR/cleanup_swe_singularity_sessions.py
SOURCE_ROOT=${FRONTIER_SOURCE_ROOT:?}
BACKEND_ROOT=${FRONTIER_SANDBOX_BACKEND_ROOT:?}
PYTHON=${FRONTIER_PYTHON:?}
RUNTIME=${SWE_CONTAINER_RUNTIME_BIN:-$(command -v singularity || command -v apptainer || true)}
PORT=${OPENREWARD_FRONTIER_CS_PORT:-8085}
MAX_RESTARTS=${OPENREWARD_FRONTIER_SERVER_MAX_RESTARTS:-5}
RESTART_RESET_SECS=${OPENREWARD_FRONTIER_SERVER_RESTART_RESET_SECS:-300}
BACKOFF_MAX_SECS=${OPENREWARD_FRONTIER_SERVER_BACKOFF_MAX_SECS:-30}
SERVER_SHUTDOWN_SECS=${OPENREWARD_FRONTIER_SERVER_SHUTDOWN_SECS:-60}
die() { echo "[frontier-cs-server] ERROR: $*" >&2; exit 2; }
require_uint() {
  local name=$1 value=$2 minimum=$3
  [[ -n $value && $value != *[!0-9]* ]] && ((10#$value >= minimum)) || \
    die "$name must be an integer >= $minimum"
}
require_uint OPENREWARD_FRONTIER_CS_PORT "$PORT" 1
require_uint OPENREWARD_FRONTIER_SERVER_MAX_RESTARTS "$MAX_RESTARTS" 0
require_uint OPENREWARD_FRONTIER_SERVER_RESTART_RESET_SECS "$RESTART_RESET_SECS" 1
require_uint OPENREWARD_FRONTIER_SERVER_BACKOFF_MAX_SECS "$BACKOFF_MAX_SECS" 0
require_uint OPENREWARD_FRONTIER_SERVER_SHUTDOWN_SECS "$SERVER_SHUTDOWN_SECS" 1
require_uint FRONTIER_JUDGE_CONCURRENCY "${FRONTIER_JUDGE_CONCURRENCY:?}" 1
((PORT <= 65535)) || die "Invalid service port"
[[ -x $PYTHON && -x $RUNTIME && -r $CLEANUP_HELPER ]] || die "Missing runtime or cleanup helper"
[[ -r ${FRONTIER_IMAGE_MANIFEST:?} && -d ${FRONTIER_MATERIALIZED_ROOT:?} ]] || \
  die "Missing image manifest or materialized catalog"
[[ $(git -C "$SOURCE_ROOT" rev-parse HEAD) == "${FRONTIER_ADAPTER_REVISION:?}" ]] || die "Adapter revision mismatch"
[[ -z $(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all) ]] || die "Adapter source is dirty"
[[ $(git -C "$BACKEND_ROOT" rev-parse HEAD) == "${FRONTIER_SANDBOX_REVISION:?}" ]] || die "Sandbox revision mismatch"
[[ -z $(git -C "$BACKEND_ROOT" status --porcelain --untracked-files=all) ]] || die "Sandbox source is dirty"

export PYTHONPATH=$SOURCE_ROOT/adapters/frontier-cs-openreward/src:$BACKEND_ROOT
export PYTHONDONTWRITEBYTECODE=1 OPENREWARD_DISABLE_UPDATE_CHECK=1 OPENHANDS_SUPPRESS_BANNER=1
export OPENREWARD_PORT=$PORT OPENREWARD_HOST=0.0.0.0
export SWE_CONTAINER_RUNTIME_BIN=$RUNTIME SWE_SANDBOX_RUNTIME=singularity
export SWE_APPTAINER_FAKEROOT=0 SWE_APPTAINER_NETWORK=host
export SWE_APPTAINER_SESSION_ROOT=${FRONTIER_SESSION_ROOT:?}
export SWE_SINGULARITY_SESSION_ROOT=$SWE_APPTAINER_SESSION_ROOT
export APPTAINER_CACHEDIR=${SINGULARITY_CACHEDIR:?} APPTAINER_TMPDIR=${SINGULARITY_TMPDIR:?}
mkdir -p "$FRONTIER_SESSION_ROOT" "$SWE_APPTAINER_IMAGE_CACHE" "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
chmod 0700 "$FRONTIER_SESSION_ROOT" "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

server_pid=; stop_requested=0
cleanup_sessions() {
  /usr/bin/python3 "$SCRIPT_DIR/cleanup_frontier_native_services.py" \
    --session-root "$FRONTIER_SESSION_ROOT" --stop-grace-seconds 10 || return $?
  /usr/bin/python3 "$CLEANUP_HELPER" --session-root "$FRONTIER_SESSION_ROOT" \
    --runtime "$RUNTIME" --stop-grace-seconds 10
}
stop_server() {
  local deadline
  if [[ -n $server_pid ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || true
    deadline=$((SECONDS + SERVER_SHUTDOWN_SECS))
    while kill -0 "$server_pid" 2>/dev/null && ((SECONDS < deadline)); do sleep 0.2; done
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -KILL "$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=
}
request_stop() { stop_requested=1; stop_server; }
final_cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_server
  if ! cleanup_sessions; then
    echo "[frontier-cs-server] exact-session cleanup failed" >&2
    status=1
  fi
  exit "$status"
}
trap request_stop INT TERM
trap final_cleanup EXIT
SERVER_ARGS=(--root "$SOURCE_ROOT" --materialized-root "$FRONTIER_MATERIALIZED_ROOT" --suite all)
read -r -a EXCLUDED_TASKS <<< "${FRONTIER_EXCLUDE_TASK_IDS:-}"
for task_id in "${EXCLUDED_TASKS[@]}"; do SERVER_ARGS+=(--exclude-task-id "$task_id"); done
failures=0
while ((stop_requested == 0)); do
  cleanup_sessions
  started_at=$SECONDS
  echo "[frontier-cs-server] starting revision=$FRONTIER_ADAPTER_REVISION host=$(hostname) port=$PORT"
  "$PYTHON" -m frontier_cs_openreward.server "${SERVER_ARGS[@]}" &
  server_pid=$!
  status=0
  wait "$server_pid" || status=$?
  server_pid=
  ((stop_requested)) && break
  ((SECONDS - started_at < RESTART_RESET_SECS)) || failures=0
  failures=$((failures + 1))
  if ((failures > MAX_RESTARTS)); then
    echo "[frontier-cs-server] exhausted restart budget; last status=$status" >&2
    exit 1
  fi
  delay=$((1 << (failures - 1)))
  ((delay <= BACKOFF_MAX_SECS)) || delay=$BACKOFF_MAX_SECS
  echo "[frontier-cs-server] restarting after status=$status ($failures/$MAX_RESTARTS) in ${delay}s" >&2
  sleep "$delay"
done
