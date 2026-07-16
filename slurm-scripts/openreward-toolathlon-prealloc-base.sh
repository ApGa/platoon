#!/bin/bash
#SBATCH --job-name=openreward-toolathlon-prealloc
#SBATCH --account=nvr_lacr_llm
#SBATCH --partition=batch
#SBATCH --nodes=16
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --signal=B:USR1@300

# Multinode AReaL training for the toolathlon OpenReward environment.
#
# Architecture (mirrors slurm-scripts/textcraft-prealloc.sh for the trainer, plus
# the toolathlon env server from slurm-scripts/openreward-toolathlon.sh):
#   * The trainer runs in AReaL "slurm_prealloc" mode: a NON-containerized host
#     controller (this sbatch's final srun) spawns actor/sglang worker srun steps
#     across all ${NNODES} nodes via PreallocatedSlurmScheduler.
#   * AReaL is single-controller: the rollout workflow (OpenHands agent + the
#     openreward mcp_bridge) runs in subprocesses on the controller (rank-0) node.
#     So every env-server call originates from rank-0.
#   * To keep the toolathlon env server from bottlenecking under high concurrency,
#     we run one gym env server on EVERY node (no GPUs, host network, port
#     ${OPENREWARD_PORT}) and shard rollouts across them: each rollout is exactly
#     one ORS session, so it is pinned to a single node's server (chosen by hash
#     in platoon.openreward.rollout via OPENREWARD_SESSION_URLS). The per-node
#     in-container nginx then keeps worker affinity via X-Session-ID. No extra
#     load balancer is needed and session affinity holds end-to-end.
#
# Node count is driven by #SBATCH --nodes (default 16; set to 2 to try 2 nodes).
# The matching config (backend parallelism sized for the GPU count) is selected
# automatically, or pass an explicit config path as $1.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
USER_ROOT=${PLATOON_USER_ROOT:-$(cd "${REPO_ROOT}/../.." && pwd)}
CONTAINER_IMAGE=${USER_ROOT}/images/platoon.sqsh
OPENREWARD_DIR=${REPO_ROOT}/plugins/openreward
JOB_SCRIPT=${OPENREWARD_JOB_SCRIPT:-$(readlink -f "${BASH_SOURCE[0]}")}
KEEPALIVE_SCRIPT=${REPO_ROOT}/slurm-scripts/gpu_keepalive.py
ROLLOUT_IDLE_GUARD_SCRIPT=${REPO_ROOT}/slurm-scripts/rollout_gpu_idle_guard.py
PREPARE_ENV_SCRIPT=${REPO_ROOT}/slurm-scripts/prepare_openreward_env.sh
TOOLATHLON_SERVER_ENTRYPOINT=${OPENREWARD_TOOLATHLON_SERVER_ENTRYPOINT:-${REPO_ROOT}/plugins/openreward/scripts/openreward-toolathlon-resilient-entrypoint.sh}
TOOLATHLON_CONTAINER_ENTRYPOINT=/app/openreward-toolathlon-resilient-entrypoint.sh
TRAIN_MODULE=platoon.openreward.train_scripts.areal.train_areal
CONFIG_DIR=${OPENREWARD_DIR}/platoon/openreward/configs/areal

# Credentials and proxy URLs are inherited from the submission environment.
export RUBRIC_DEFAULT_LLM=${RUBRIC_DEFAULT_LLM:-litellm_proxy/azure/gpt-5-mini}

NNODES=${SLURM_NNODES:-16}
GPUS_PER_NODE=${OPENREWARD_GPUS_PER_NODE:-8}

# Pick the config whose backend parallelism matches the allocation, unless the
# caller passes one explicitly. The 16-node launcher defaults to the CP config.
case "${NNODES}" in
  2)
    DEFAULT_CONFIG=${CONFIG_DIR}/toolathlon_openhands_areal_prealloc_2node.yaml
    ;;
  8)
    DEFAULT_CONFIG=${CONFIG_DIR}/toolathlon_openhands_areal_prealloc_8node.yaml
    ;;
  16)
    DEFAULT_CONFIG=${CONFIG_DIR}/toolathlon_openhands_areal_prealloc_16node-cp.yaml
    ;;
  *)
    DEFAULT_CONFIG=
    ;;
esac
# Accept the config as a positional arg (`sbatch this.sh <config>`) OR with an
# explicit flag (`sbatch this.sh --config <config>`). The script supplies the
# `--config` flag to the trainer itself, so strip a leading one here.
if [[ "${1:-}" == "--config" || "${1:-}" == "-c" ]]; then
  shift
fi
CONFIG=${1:-${DEFAULT_CONFIG}}
if [[ -z "${CONFIG}" ]]; then
  echo "ERROR: no default config for ${NNODES} nodes; pass an explicit config path." >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: '${CONFIG}'" >&2
  echo "       Usage: sbatch ${BASH_SOURCE[0]##*/} [CONFIG]   (positional, no --config flag)" >&2
  exit 2
fi
if [[ ! -x "${TOOLATHLON_SERVER_ENTRYPOINT}" ]]; then
  echo "ERROR: Toolathlon server supervisor is missing or not executable: ${TOOLATHLON_SERVER_ENTRYPOINT}" >&2
  exit 2
fi
if [[ ! -x "${ROLLOUT_IDLE_GUARD_SCRIPT}" ]]; then
  echo "ERROR: rollout GPU idle guard is missing or not executable: ${ROLLOUT_IDLE_GUARD_SCRIPT}" >&2
  exit 2
fi
# The controller and workers intentionally run from a neutral job-state
# directory so the live checkout cannot shadow installed packages. Preserve
# support for a caller-provided relative config by canonicalizing it first.
CONFIG=$(readlink -f "${CONFIG}")
# This script injects `openreward.*` Hydra overrides, so the config must be an
# OpenReward config (not e.g. a textcraft one) or those overrides fail to compose.
if ! grep -qE '^[[:space:]]*openreward:' "${CONFIG}"; then
  echo "ERROR: '${CONFIG}' has no top-level 'openreward:' section." >&2
  echo "       This launcher is for OpenReward configs only (it sets openreward.session_url etc.)." >&2
  echo "       Did you pass the wrong config? Expected one under ${CONFIG_DIR}/." >&2
  exit 2
fi

# Optional model-only branch overrides.  Use both together to start a new trial
# from an HF/model-only checkpoint without inheriting an optimizer created under
# different loss-scaling semantics.  Because the new trial subsequently writes
# and recovers its own optimizer, this is safer than leaving
# `recover.no_load_optim=true` enabled across wall-time resubmissions.
#
# Example:
#   sbatch --export=ALL,OPENREWARD_TRIAL_NAME=my-new-trial,OPENREWARD_ACTOR_PATH=/path/to/complete-model \
#     slurm-scripts/openreward-toolathlon-prealloc.sh
OPENREWARD_TRIAL_NAME=${OPENREWARD_TRIAL_NAME:-}
OPENREWARD_ACTOR_PATH=${OPENREWARD_ACTOR_PATH:-}
OPENREWARD_LOCAL_MODEL_SNAPSHOT=${OPENREWARD_LOCAL_MODEL_SNAPSHOT:-}
OPENREWARD_WANDB_MODE=${OPENREWARD_WANDB_MODE:-${WANDB_MODE:-}}
if [[ -n "${OPENREWARD_ACTOR_PATH}" && -z "${OPENREWARD_TRIAL_NAME}" ]]; then
  echo "ERROR: OPENREWARD_ACTOR_PATH requires OPENREWARD_TRIAL_NAME." >&2
  echo "       Reusing the configured trial could silently recover an incompatible optimizer." >&2
  exit 2
fi

# AReaL's proxy tokenizer helper currently hardcodes force_download=True. Its
# repo-ID lookup both stampedes Hugging Face at multinode startup and conflicts
# with offline mode. Resolve the complete shared Qwen snapshot to an actual
# directory so Transformers takes its local-directory path before Hub logic.
if [[ -z "${OPENREWARD_LOCAL_MODEL_SNAPSHOT}" ]] && \
   grep -qE '^[[:space:]]*path:[[:space:]]*Qwen/Qwen3\.6-35B-A3B' "${CONFIG}"; then
  qwen_cache=${USER_ROOT}/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B
  if [[ -f "${qwen_cache}/refs/main" ]]; then
    qwen_revision=$(tr -d '\r\n' <"${qwen_cache}/refs/main")
    OPENREWARD_LOCAL_MODEL_SNAPSHOT=${qwen_cache}/snapshots/${qwen_revision}
  fi
fi
if [[ -n "${OPENREWARD_LOCAL_MODEL_SNAPSHOT}" ]]; then
  for required_model_file in config.json tokenizer.json tokenizer_config.json model.safetensors.index.json; do
    [[ -e "${OPENREWARD_LOCAL_MODEL_SNAPSHOT}/${required_model_file}" ]] || {
      echo "ERROR: local model snapshot is incomplete; missing ${required_model_file}:" >&2
      echo "       ${OPENREWARD_LOCAL_MODEL_SNAPSHOT}" >&2
      exit 2
    }
  done
  model_shard_count=$(find "${OPENREWARD_LOCAL_MODEL_SNAPSHOT}" -maxdepth 1 -name 'model-*-of-*.safetensors' | wc -l)
  [[ "${model_shard_count}" -eq 26 ]] || {
    echo "ERROR: expected 26 Qwen model shards, found ${model_shard_count}:" >&2
    echo "       ${OPENREWARD_LOCAL_MODEL_SNAPSHOT}" >&2
    exit 2
  }
  echo "Using local Qwen snapshot for AReaL tokenizer startup: ${OPENREWARD_LOCAL_MODEL_SNAPSHOT}"
fi

# The checked-in training config requests online W&B logging, but credentials
# are intentionally not stored in this launcher. AReaL calls wandb.login()
# during trainer construction, after costly worker startup has already begun.
# Disable W&B when no key was inherited so missing optional telemetry cannot
# abort the training job or stall on the SDK's local offline service. Set
# OPENREWARD_WANDB_MODE=online to require online logging, or to offline/disabled
# to choose that mode explicitly.
if [[ -z "${OPENREWARD_WANDB_MODE}" && -z "${WANDB_API_KEY:-}" ]]; then
  OPENREWARD_WANDB_MODE=disabled
  echo "WARNING: WANDB_API_KEY is not set; disabling W&B logging." >&2
  echo "         Export WANDB_API_KEY before sbatch to retain online logging." >&2
fi
case "${OPENREWARD_WANDB_MODE}" in
  ""|offline|disabled)
    ;;
  online)
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
      echo "ERROR: OPENREWARD_WANDB_MODE=online requires WANDB_API_KEY." >&2
      exit 2
    fi
    ;;
  *)
    echo "ERROR: invalid OPENREWARD_WANDB_MODE=${OPENREWARD_WANDB_MODE@Q}; expected online, offline, or disabled." >&2
    exit 2
    ;;
esac

TRAIN_OVERRIDE_ARGS=()
if [[ -n "${OPENREWARD_TRIAL_NAME}" ]]; then
  TRAIN_OVERRIDE_ARGS+=("trial_name=${OPENREWARD_TRIAL_NAME}")
fi
if [[ -n "${OPENREWARD_ACTOR_PATH}" ]]; then
  TRAIN_OVERRIDE_ARGS+=("actor.path=${OPENREWARD_ACTOR_PATH}")
fi
if [[ -n "${OPENREWARD_LOCAL_MODEL_SNAPSHOT}" ]]; then
  TRAIN_OVERRIDE_ARGS+=("tokenizer_path=${OPENREWARD_LOCAL_MODEL_SNAPSHOT}")
fi
if [[ -n "${OPENREWARD_WANDB_MODE}" ]]; then
  export WANDB_MODE=${OPENREWARD_WANDB_MODE}
  TRAIN_OVERRIDE_ARGS+=("stats_logger.wandb.mode=${OPENREWARD_WANDB_MODE}")
fi
TRAIN_OVERRIDE_CMD=
if [[ "${#TRAIN_OVERRIDE_ARGS[@]}" -gt 0 ]]; then
  printf -v TRAIN_OVERRIDE_CMD ' %q' "${TRAIN_OVERRIDE_ARGS[@]}"
fi
RUN_ID=${OPENREWARD_RUN_ID:-${SLURM_JOB_ID:-manual}}
JOB_INSTANCE_ID=${SLURM_JOB_ID:-manual-$$}
OPENREWARD_INFRA_RESTART_COUNT=${OPENREWARD_INFRA_RESTART_COUNT:-0}
OPENREWARD_MAX_INFRA_RESTARTS=${OPENREWARD_MAX_INFRA_RESTARTS:-3}
# Environments are immutable and shared by their complete dependency/source
# identity. Keep them beside uv's cache (Lustre project ID 0), so uv can hardlink
# cached packages instead of copying hundreds of them into the experiments
# project. The prepare helper serializes cold builders and atomically publishes
# the completed relocatable venv.
OPENREWARD_ENV_CACHE_ROOT=${OPENREWARD_ENV_CACHE_ROOT:-${REPO_ROOT}/.cache/openreward-venvs}
OPENREWARD_UV_CACHE_DIR=${UV_CACHE_DIR:-${REPO_ROOT}/.uv-cache}
OPENREWARD_UV_BIN=${OPENREWARD_UV_BIN:-${USER_ROOT}/.local/bin/uv}
OPENREWARD_UV_LINK_MODE=${OPENREWARD_UV_LINK_MODE:-hardlink}
OPENREWARD_JOB_STATE_ROOT=${OPENREWARD_JOB_STATE_ROOT:-${USER_ROOT}/experiments/areal/job-state/openreward}
OPENREWARD_JOB_STATE_DIR=${OPENREWARD_JOB_STATE_ROOT}/${JOB_INSTANCE_ID}
OPENREWARD_ENV_READY_FILE=${OPENREWARD_JOB_STATE_DIR}/env-ready
OPENREWARD_RUNTIME_GUARD_BIN=${OPENREWARD_JOB_STATE_DIR}/runtime-guard-bin
KEEPALIVE_READY_DIR=${OPENREWARD_JOB_STATE_DIR}/keepalive-ready
ROLLOUT_IDLE_GUARD_READY_DIR=${OPENREWARD_JOB_STATE_DIR}/gpu-idle-guard-ready
# The old project venv is no longer mutated. It provides torch immediately to
# the keepalive while a first content-addressed training environment is built.
OPENREWARD_KEEPALIVE_PYTHON=${OPENREWARD_KEEPALIVE_PYTHON:-${OPENREWARD_DIR}/.venv/bin/python}
STOP_DIR=${USER_ROOT}/experiments/areal/stop
STOP_FILE=${OPENREWARD_STOP_FILE:-${STOP_DIR}/openreward-toolathlon-prealloc-${RUN_ID}.stop}
if [[ -f "${STOP_FILE}" ]]; then
  echo "Stop file exists; exiting before setup: ${STOP_FILE}"
  exit 0
fi
RESUBMITTED=0

# --- Megatron / Transformer Engine --------------------------------------------
# The Megatron backend (areal[megatron]) does an unconditional
# `import transformer_engine.pytorch`, but TE is deliberately excluded from the uv
# lock (`transformer-engine; sys_platform == 'never'`) and has no prebuilt wheel
# for AReaL HEAD's torch, so it must be source-built. slurm-scripts/install_te.sh
# compiles it ONCE where nvcc exists (the container) into a shared /lustre cache,
# then installs that cached wheel anywhere in seconds. The environment helper
# installs it only in its locked in-container staging venv; the host controller
# never mutates a published environment.
# Auto-enabled when the chosen config uses a megatron backend; force with
# OPENREWARD_BUILD_TE=0/1.
# Megatron's ColumnParallelLinear defaults to gradient_accumulation_fusion=True,
# which hard-requires APEX's `fused_weight_gradient_mlp_cuda` CUDA kernel (AReaL only
# disables that fusion for LoRA). APEX, like TE, isn't in the uv lock and must be
# source-built (--cpp_ext --cuda_ext) where nvcc exists, then cached on /lustre and
# installed after `uv sync`. install_apex.sh handles the build-once + cache.
if grep -qiE '^[[:space:]]*backend:[[:space:]]*megatron' "${CONFIG}" 2>/dev/null; then
  OPENREWARD_BUILD_TE=${OPENREWARD_BUILD_TE:-1}
  OPENREWARD_BUILD_APEX=${OPENREWARD_BUILD_APEX:-1}
else
  OPENREWARD_BUILD_TE=${OPENREWARD_BUILD_TE:-0}
  OPENREWARD_BUILD_APEX=${OPENREWARD_BUILD_APEX:-0}
fi

# Snapshot local package sources and resolve the immutable environment before
# any worker command is assembled. This is cheap (the local packages are only a
# few MB) and guarantees that dirty source changes produce a different cache
# key. The expensive build happens later inside the CUDA container.
[[ -x "${PREPARE_ENV_SCRIPT}" ]] || {
  echo "ERROR: environment preparation helper is missing or not executable: ${PREPARE_ENV_SCRIPT}" >&2
  exit 2
}
env_resolution=$(
  PLATOON_REPO_ROOT="${REPO_ROOT}" \
  OPENREWARD_ENV_CACHE_ROOT="${OPENREWARD_ENV_CACHE_ROOT}" \
  OPENREWARD_PROJECT_DIR="${OPENREWARD_DIR}" \
  OPENREWARD_CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
  OPENREWARD_UV_BIN="${OPENREWARD_UV_BIN}" \
  OPENREWARD_BUILD_TE="${OPENREWARD_BUILD_TE}" \
  OPENREWARD_BUILD_APEX="${OPENREWARD_BUILD_APEX}" \
  "${PREPARE_ENV_SCRIPT}" resolve
)
IFS=$'\t' read -r OPENREWARD_ENV_KEY OPENREWARD_SOURCE_SHA OPENREWARD_SOURCE_ARCHIVE OPENREWARD_JOB_VENV <<<"${env_resolution}"
if [[ -z "${OPENREWARD_ENV_KEY}" || -z "${OPENREWARD_SOURCE_ARCHIVE}" || -z "${OPENREWARD_JOB_VENV}" ]]; then
  echo "ERROR: malformed environment resolution from ${PREPARE_ENV_SCRIPT}: ${env_resolution}" >&2
  exit 2
fi
OPENREWARD_JOB_PYTHON=${OPENREWARD_JOB_VENV}/bin/python

# --- OpenReward toolathlon env server (runs alongside training, no GPUs) -------
OPENREWARD_PORT=${OPENREWARD_PORT:-8082}
OPENREWARD_SERVER_IMAGE=${OPENREWARD_SERVER_IMAGE:-${USER_ROOT}/images/openreward/apga+openreward-toolathlon-gym+latest.sqsh}
# Shard rollouts across one env server per node (1) or run a single server on the
# controller node only (0). See header for the affinity rationale.
OPENREWARD_SHARD=${OPENREWARD_SHARD:-1}
# CPU/mem soft-requests for each (overlapping) env-server step. The container's
# entrypoint sizes its worker count from the visible CPU count, so this also caps
# workers-per-node (e.g. 32 cpus -> ~32 server workers per node).
OPENREWARD_SERVER_CPUS=${OPENREWARD_SERVER_CPUS:-32}
OPENREWARD_SERVER_MEM=${OPENREWARD_SERVER_MEM:-128G}
SERVER_WAIT_SECS=${OPENREWARD_SERVER_WAIT_SECS:-900}
# Once training starts, fail the allocation if the per-node environment service
# is unreachable for several consecutive probes. A dead env-server srun used to
# leave the trainer alive indefinitely, rejecting zero-data rollouts while all
# GPUs remained allocated.
SERVER_HEALTH_CHECK_SECS=${OPENREWARD_SERVER_HEALTH_CHECK_SECS:-20}
SERVER_HEALTH_FAILURE_THRESHOLD=${OPENREWARD_SERVER_HEALTH_FAILURE_THRESHOLD:-3}
# A published training environment must remain usable for newly spawned rollout
# subprocesses.  Check it alongside the env servers so accidental runtime
# package operations fail the job promptly instead of leaving it rejecting
# rollouts indefinitely.
ENVIRONMENT_HEALTH_FAILURE_THRESHOLD=${OPENREWARD_ENVIRONMENT_HEALTH_FAILURE_THRESHOLD:-2}
# toolathlon is DB-backed: the image entrypoint starts PostgreSQL then exec's the
# env server. Under this cluster's single-UID enroot namespace that needs three
# image fixups (done as container-root before the entrypoint), then a nested user
# namespace so PostgreSQL can run non-root. (Identical to openreward-toolathlon.sh.)
#
# The env server rebuilds per-session MCP sub-server venvs with uv on every
# /create. uv's cache (~/.cache/uv) and build temp (TMPDIR) default into the
# container's small writable overlay, which fills under sustained concurrency
# ("No space left on device" -> MCP sub-servers fail -> session init drops the
# connection). Redirect both onto the host-mounted /tmp (large host disk) before
# starting the entrypoint; the exports survive the unshare exec. UV_LINK_MODE=
# symlink keeps the per-session venvs (built on the container overlay) as symlinks
# into that host-side cache instead of cross-filesystem copies, so the overlay
# footprint stays near-zero. Override OPENREWARD_SERVER_CACHE_DIR to point at a
# different host path if needed.
OPENREWARD_SERVER_CACHE_DIR=${OPENREWARD_SERVER_CACHE_DIR:-/tmp/openreward-server-cache}
OPENREWARD_SERVER_CMD=${OPENREWARD_SERVER_CMD:-'mkdir -p '"${OPENREWARD_SERVER_CACHE_DIR}"'/uv '"${OPENREWARD_SERVER_CACHE_DIR}"'/tmp '"${OPENREWARD_SERVER_CACHE_DIR}"'/xdg && chmod -R 0777 '"${OPENREWARD_SERVER_CACHE_DIR}"' 2>/dev/null || true; export UV_CACHE_DIR='"${OPENREWARD_SERVER_CACHE_DIR}"'/uv TMPDIR='"${OPENREWARD_SERVER_CACHE_DIR}"'/tmp XDG_CACHE_HOME='"${OPENREWARD_SERVER_CACHE_DIR}"'/xdg UV_LINK_MODE=symlink; for d in /root/.local/share/uv/python/cpython-3.12.*-linux-x86_64-gnu; do [ -d "$d" ] && ln -sfn "$(basename "$d")" /root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu; done; export VIRTUAL_ENV=/opt/venv; export PATH=/opt/venv/bin:$PATH; chmod 0600 /etc/ssl/private/ssl-cert-snakeoil.key 2>/dev/null || true; if command -v unshare >/dev/null && unshare --user --map-user=105 --map-group=108 true 2>/dev/null; then exec unshare --user --map-user=105 --map-group=108 -- '"${TOOLATHLON_CONTAINER_ENTRYPOINT}"'; else exec '"${TOOLATHLON_CONTAINER_ENTRYPOINT}"'; fi'}

# Credentials (WANDB_API_KEY, OPENAI_API_KEY, LITELLM_API_KEY, and HF_TOKEN)
# are inherited from the submission environment.
# Credentials and proxy URLs are inherited from the submission environment.
export RUBRIC_DEFAULT_LLM=${RUBRIC_DEFAULT_LLM:-litellm_proxy/azure/gpt-5-mini}

export OPENHANDS_SUPPRESS_BANNER=1
export OPENREWARD_DISABLE_UPDATE_CHECK=1
export OPENREWARD_REWARD_MODE=partial
export PLATOON_QWEN35_GDN_CP=1
export PLATOON_QWEN35_GDN_CP_CONV_BACKEND=${PLATOON_QWEN35_GDN_CP_CONV_BACKEND:-fla}

# GPU keepalive guards against idle reclamation for the full Slurm allocation.
export OPENREWARD_GPU_KEEPALIVE=${OPENREWARD_GPU_KEEPALIVE:-1}
export KEEPALIVE_TICK_SEC=${KEEPALIVE_TICK_SEC:-5}
export KEEPALIVE_MATMUL_DIM=${KEEPALIVE_MATMUL_DIM:-4096}
export KEEPALIVE_MATMUL_REPS=${KEEPALIVE_MATMUL_REPS:-2000}
export KEEPALIVE_START_DELAY_SEC=${KEEPALIVE_START_DELAY_SEC:-300}
export KEEPALIVE_MAX_SEC=${KEEPALIVE_MAX_SEC:-16200}
export KEEPALIVE_EXPECTED_GPUS=${KEEPALIVE_EXPECTED_GPUS:-${GPUS_PER_NODE}}
export KEEPALIVE_MAX_CONSECUTIVE_ERRORS=${KEEPALIVE_MAX_CONSECUTIVE_ERRORS:-3}
KEEPALIVE_WAIT_SECS=${KEEPALIVE_WAIT_SECS:-300}
# Runtime idle protection is separate from the setup-only keepalive above. The
# scheduler starts one sibling for each exact actor and rollout role, pinned to
# its resolved role nodes. The Python guard independently rejects
# configurations above a 25% duty cycle or a 2048 matrix dimension.
export PLATOON_AREAL_ROLLOUT_IDLE_GUARD=${PLATOON_AREAL_ROLLOUT_IDLE_GUARD:-1}
export ROLLOUT_IDLE_GUARD_INTERVAL_SECONDS=${ROLLOUT_IDLE_GUARD_INTERVAL_SECONDS:-10}
export ROLLOUT_IDLE_GUARD_INTERVAL_JITTER_SECONDS=${ROLLOUT_IDLE_GUARD_INTERVAL_JITTER_SECONDS:-2}
export ROLLOUT_IDLE_GUARD_SAMPLE_COUNT=${ROLLOUT_IDLE_GUARD_SAMPLE_COUNT:-2}
export ROLLOUT_IDLE_GUARD_SAMPLE_INTERVAL_SECONDS=${ROLLOUT_IDLE_GUARD_SAMPLE_INTERVAL_SECONDS:-2}
export ROLLOUT_IDLE_GUARD_UTILIZATION_THRESHOLD=${ROLLOUT_IDLE_GUARD_UTILIZATION_THRESHOLD:-10}
export ROLLOUT_IDLE_GUARD_BURST_SECONDS=${ROLLOUT_IDLE_GUARD_BURST_SECONDS:-2}
export ROLLOUT_IDLE_GUARD_MATRIX_DIM=${ROLLOUT_IDLE_GUARD_MATRIX_DIM:-1024}
export ROLLOUT_IDLE_GUARD_OPERATIONS_PER_SYNC=${ROLLOUT_IDLE_GUARD_OPERATIONS_PER_SYNC:-32}
export ROLLOUT_IDLE_GUARD_EXPECTED_DEVICES=1
export ROLLOUT_IDLE_GUARD_MAX_CONSECUTIVE_QUERY_ERRORS=${ROLLOUT_IDLE_GUARD_MAX_CONSECUTIVE_QUERY_ERRORS:-5}
export ROLLOUT_IDLE_GUARD_MAX_CONSECUTIVE_CUDA_ERRORS=${ROLLOUT_IDLE_GUARD_MAX_CONSECUTIVE_CUDA_ERRORS:-3}
export ROLLOUT_IDLE_GUARD_LOG_EVERY_CYCLES=${ROLLOUT_IDLE_GUARD_LOG_EVERY_CYCLES:-10}


unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN
export PLATOON_MEGATRON_ATTENTION_BACKEND=${PLATOON_MEGATRON_ATTENTION_BACKEND:-flash}

mkdir -p "${USER_ROOT}/logs"
mkdir -p "${USER_ROOT}/experiments/areal/experiments"
mkdir -p "${USER_ROOT}/experiments/areal/name_resolve"
mkdir -p "${OPENREWARD_JOB_STATE_DIR}"
rm -f "${OPENREWARD_ENV_READY_FILE}"
# Keep pip entrypoints out of the parent trainer process. A rollout subprocess
# prepends its own disposable environment and pip shim; uv must remain available
# there so programmatic tool calling can install task-specific packages safely.
mkdir -p "${OPENREWARD_RUNTIME_GUARD_BIN}"
printf '%s\n' \
  '#!/bin/sh' \
  'echo "Package installation is disabled inside the immutable training runtime." >&2' \
  'exit 2' >"${OPENREWARD_RUNTIME_GUARD_BIN}/package-manager-disabled"
chmod 0555 "${OPENREWARD_RUNTIME_GUARD_BIN}/package-manager-disabled"
for package_manager in pip pip3 pip3.12; do
  ln -sfn package-manager-disabled "${OPENREWARD_RUNTIME_GUARD_BIN}/${package_manager}"
done
mkdir -p "${KEEPALIVE_READY_DIR}"
rm -f "${KEEPALIVE_READY_DIR}"/*.ready "${KEEPALIVE_READY_DIR}"/.*.tmp
mkdir -p "${ROLLOUT_IDLE_GUARD_READY_DIR}/actor" "${ROLLOUT_IDLE_GUARD_READY_DIR}/rollout"
rm -f "${ROLLOUT_IDLE_GUARD_READY_DIR}/actor"/*.ready "${ROLLOUT_IDLE_GUARD_READY_DIR}/actor"/.*.tmp
rm -f "${ROLLOUT_IDLE_GUARD_READY_DIR}/rollout"/*.ready "${ROLLOUT_IDLE_GUARD_READY_DIR}/rollout"/.*.tmp
mkdir -p "${STOP_DIR}"

# --- Node list + sharded session URLs ----------------------------------------
mapfile -t ALLOC_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}")
if [[ "${#ALLOC_NODES[@]}" -eq 0 ]]; then
  ALLOC_NODES=("$(hostname)")
fi
CONTROLLER_NODE=${ALLOC_NODES[0]}

if [[ "${OPENREWARD_SHARD}" == "1" ]]; then
  SERVER_NODES=("${ALLOC_NODES[@]}")
else
  SERVER_NODES=("${CONTROLLER_NODE}")
fi

# URLs the rank-0 controller uses to reach each node's env server. The servers run
# host-networked (pyxis), so other nodes are reachable by hostname.
session_urls=""
for n in "${SERVER_NODES[@]}"; do
  session_urls="${session_urls:+${session_urls},}http://${n}:${OPENREWARD_PORT}"
done
SERVER_LOG_PREFIX=${USER_ROOT}/logs/openreward-toolathlon-prealloc-server-${RUN_ID}

# --- Lifecycle traps ----------------------------------------------------------
launcher_pid=$$
keepalive_pid=
keepalive_monitor_pid=
env_setup_pid=
server_pid=
server_health_pid=
srun_pid=
SERVER_HEALTH_FAILURE_FILE=/tmp/platoon-openreward-server-health-${RUN_ID}-$$.failed
ENVIRONMENT_HEALTH_FAILURE_FILE=/tmp/platoon-openreward-environment-health-${RUN_ID}-$$.failed
SUCCESSOR_CLAIM_DIR=${OPENREWARD_JOB_STATE_DIR}/successor-claimed

cleanup() {
  if [[ -n "${keepalive_monitor_pid}" ]] && kill -0 "${keepalive_monitor_pid}" 2>/dev/null; then
    kill "${keepalive_monitor_pid}" 2>/dev/null || true
    wait "${keepalive_monitor_pid}" 2>/dev/null || true
  fi
  if [[ -n "${env_setup_pid}" ]] && kill -0 "${env_setup_pid}" 2>/dev/null; then
    kill "${env_setup_pid}" 2>/dev/null || true
    wait "${env_setup_pid}" 2>/dev/null || true
  fi
  if [[ -n "${server_health_pid}" ]] && kill -0 "${server_health_pid}" 2>/dev/null; then
    kill "${server_health_pid}" 2>/dev/null || true
    wait "${server_health_pid}" 2>/dev/null || true
  fi
  if [[ -n "${srun_pid}" ]] && kill -0 "${srun_pid}" 2>/dev/null; then
    kill "${srun_pid}" 2>/dev/null || true
    wait "${srun_pid}" 2>/dev/null || true
  fi
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    echo "Stopping OpenReward env-server srun (pid ${server_pid})."
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  if [[ -n "${keepalive_pid}" ]] && kill -0 "${keepalive_pid}" 2>/dev/null; then
    kill "${keepalive_pid}" 2>/dev/null || true
    wait "${keepalive_pid}" 2>/dev/null || true
  fi
  rm -f "${SERVER_HEALTH_FAILURE_FILE}" "${ENVIRONMENT_HEALTH_FAILURE_FILE}"
}

submit_successor() {
  local reason=$1
  local infrastructure_restart=${2:-0}
  local next_infra_restart_count=${OPENREWARD_INFRA_RESTART_COUNT}
  local next_job_id

  if ! mkdir "${SUCCESSOR_CLAIM_DIR}" 2>/dev/null; then
    echo "A successor job has already been claimed for run ${RUN_ID}."
    return 0
  fi
  if [[ -f "${STOP_FILE}" ]]; then
    echo "Stop file exists; not resubmitting: ${STOP_FILE}"
    return 0
  fi
  if [[ "${infrastructure_restart}" -eq 1 ]]; then
    next_infra_restart_count=$((OPENREWARD_INFRA_RESTART_COUNT + 1))
    if [[ "${next_infra_restart_count}" -gt "${OPENREWARD_MAX_INFRA_RESTARTS}" ]]; then
      echo "ERROR: refusing another automatic infrastructure restart after ${OPENREWARD_INFRA_RESTART_COUNT} attempts." >&2
      return 1
    fi
  fi
  if ! next_job_id=$(sbatch --parsable \
    --dependency="afterany:${SLURM_JOB_ID}" \
    --export=ALL,OPENREWARD_RUN_ID="${RUN_ID}",OPENREWARD_STOP_FILE="${STOP_FILE}",OPENREWARD_INFRA_RESTART_COUNT="${next_infra_restart_count}" \
    "${JOB_SCRIPT}" "${CONFIG}"); then
    rmdir "${SUCCESSOR_CLAIM_DIR}" 2>/dev/null || true
    echo "ERROR: failed to submit successor for ${reason}." >&2
    return 1
  fi
  printf '%s\n' "${next_job_id}" >"${SUCCESSOR_CLAIM_DIR}/job-id"
  echo "Submitted successor job ${next_job_id} for run ${RUN_ID} (${reason}; infrastructure restart ${next_infra_restart_count}/${OPENREWARD_MAX_INFRA_RESTARTS})."
}

resubmit_on_walltime() {
  echo "Received Slurm walltime warning for job ${SLURM_JOB_ID:-unknown}."
  if [[ "${RESUBMITTED}" -eq 1 ]]; then
    echo "A successor job has already been submitted for run ${RUN_ID}."
    return
  fi
  RESUBMITTED=1
  submit_successor "walltime continuation" 0
}

resubmit_on_sigterm() {
  # Slurm can deliver more than one shutdown signal while job steps are being
  # torn down.  Do not let a second signal re-enter this handler or interrupt
  # successor submission / EXIT cleanup.
  trap '' TERM INT USR1
  echo "Received SIGTERM for job ${SLURM_JOB_ID:-unknown}; requesting an infrastructure-recovery successor."
  # submit_successor's atomic claim directory makes this safe when SIGTERM is
  # secondary to a walltime warning or one of the health monitors: whichever
  # path claims the successor first wins, and the other becomes a no-op.
  # A deliberate stop remains available through STOP_FILE, which is inherited
  # by every allocation in the continuation chain.
  submit_successor "SIGTERM recovery" 1 || true
  exit 143
}

trap resubmit_on_walltime USR1
trap cleanup EXIT
trap resubmit_on_sigterm TERM
trap 'exit 130' INT

# --- AReaL preallocated-Slurm scheduler wiring (mirrors textcraft-prealloc) ----
export PLATOON_AREAL_PREALLOC_CONTAINER_IMAGE="${CONTAINER_IMAGE}"
export PLATOON_AREAL_PREALLOC_CONTAINER_MOUNTS=/lustre:/lustre,/tmp:/tmp
export PLATOON_AREAL_PREALLOC_CONTAINER_WORKDIR="${OPENREWARD_JOB_STATE_DIR}"
export PLATOON_AREAL_PREALLOC_SRUN_BIN=${PLATOON_AREAL_PREALLOC_SRUN_BIN:-$(command -v srun || echo srun)}
export PLATOON_AREAL_PREALLOC_SRUN_ARGS=${PLATOON_AREAL_PREALLOC_SRUN_ARGS:-"--unbuffered --mpi=pmi2 -K --overlap"}
export PLATOON_AREAL_PREALLOC_GPU_FLAG=${PLATOON_AREAL_PREALLOC_GPU_FLAG:-"--gpus-per-node={gpus}"}
export PLATOON_AREAL_PREALLOC_CONFIGURE_CONCURRENCY=${PLATOON_AREAL_PREALLOC_CONFIGURE_CONCURRENCY:-16}
export PLATOON_AREAL_ROLLOUT_IDLE_GUARD_SCRIPT="${ROLLOUT_IDLE_GUARD_SCRIPT}"
export PLATOON_AREAL_ROLLOUT_IDLE_GUARD_PYTHON="${OPENREWARD_JOB_PYTHON}"
export PLATOON_AREAL_ROLLOUT_IDLE_GUARD_READY_DIR="${ROLLOUT_IDLE_GUARD_READY_DIR}"
export PLATOON_AREAL_ROLLOUT_IDLE_GUARD_LOG_PREFIX="${USER_ROOT}/logs/gpu-idle-guard-${RUN_ID}-${JOB_INSTANCE_ID}"
export PLATOON_AREAL_ROLLOUT_IDLE_GUARD_READY_TIMEOUT=${PLATOON_AREAL_ROLLOUT_IDLE_GUARD_READY_TIMEOUT:-120}
export PLATOON_AREAL_PREALLOC_WORKER_PREAMBLE="
  set -euo pipefail
  export HOME=${USER_ROOT}
  export UV_CACHE_DIR=${OPENREWARD_UV_CACHE_DIR}
  export UV_LINK_MODE=${OPENREWARD_UV_LINK_MODE}
  # The complete Qwen snapshot is shared in the HF cache. Avoid a metadata
  # request storm when all SGLang ranks start together (previously HTTP 429).
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  export PYTHONDONTWRITEBYTECODE=1
  # Retain the collective history needed to identify the first desynchronized
  # rank if NCCL times out. These are inert on the healthy path apart from the
  # bounded circular trace buffer.
  export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-4096}
  export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
  export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG:-1}
  source ${USER_ROOT}/.local/bin/env
  cd ${OPENREWARD_JOB_STATE_DIR}
  export PATH=${OPENREWARD_RUNTIME_GUARD_BIN}:${OPENREWARD_JOB_VENV}/bin:\${PATH}
  # Use the cached interpreter without advertising its shared prefix as an
  # install target. Rollout subprocesses create a writable, disposable overlay
  # and override PATH/VIRTUAL_ENV only for model-authored Python.
  unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
  cuda_lib_dirs=\$(find ${OPENREWARD_JOB_VENV}/lib/python3.12/site-packages/nvidia -mindepth 2 -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)
  if [[ -n \"\${cuda_lib_dirs}\" ]]; then
    export LD_LIBRARY_PATH=\"\${cuda_lib_dirs}:\${LD_LIBRARY_PATH:-}\"
  fi
  hash -r
"

# --- Env server: one copy per node (or just the controller node) --------------
start_env_servers() {
  local nodelist
  nodelist=$(IFS=,; echo "${SERVER_NODES[*]}")
  echo "Starting OpenReward env server on node(s): ${nodelist} (port ${OPENREWARD_PORT}, image ${OPENREWARD_SERVER_IMAGE})."
  echo "Server logs: ${SERVER_LOG_PREFIX}-<node>.log"
  # Overlapping pyxis step, no GPUs, host network -> rank-0 reaches every node at
  # http://<node>:${OPENREWARD_PORT}. One task per node, pinned via --nodelist.
  srun \
    --overlap \
    --kill-on-bad-exit=0 \
    --nodes="${#SERVER_NODES[@]}" \
    --ntasks="${#SERVER_NODES[@]}" \
    --ntasks-per-node=1 \
    --nodelist="${nodelist}" \
    --gpus-per-node=0 \
    --cpus-per-task="${OPENREWARD_SERVER_CPUS}" \
    --mem="${OPENREWARD_SERVER_MEM}" \
    --container-image="${OPENREWARD_SERVER_IMAGE}" \
    --container-mounts="/tmp:/tmp,${TOOLATHLON_SERVER_ENTRYPOINT}:${TOOLATHLON_CONTAINER_ENTRYPOINT}:ro" \
    --output="${SERVER_LOG_PREFIX}-%N.log" \
    --error="${SERVER_LOG_PREFIX}-%N.log" \
    /bin/bash -lc "export NVIDIA_VISIBLE_DEVICES=void; export OPENREWARD_PORT=${OPENREWARD_PORT}; ${OPENREWARD_SERVER_CMD}" &
  server_pid=$!
  echo "OpenReward env-server srun pid: ${server_pid}"
}

ENV_SERVER_FAILURE_DETAIL=
ENV_SERVER_SRUN_DIED=0

env_servers_healthy() {
  local failed_endpoint=
  local n
  local srun_state=alive

  for n in "${SERVER_NODES[@]}"; do
    if ! (exec 3<>"/dev/tcp/${n}/${OPENREWARD_PORT}") 2>/dev/null; then
      failed_endpoint="${n}:${OPENREWARD_PORT}"
      break
    fi
    exec 3>&- 3<&- || true
  done

  ENV_SERVER_SRUN_DIED=0
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    srun_state=died
    ENV_SERVER_SRUN_DIED=1
  fi

  if [[ -n "${failed_endpoint}" ]]; then
    ENV_SERVER_FAILURE_DETAIL="pool=Toolathlon endpoint=${failed_endpoint} srun_pid=${server_pid} srun_state=${srun_state}"
    return 1
  fi
  if [[ "${ENV_SERVER_SRUN_DIED}" -eq 1 ]]; then
    ENV_SERVER_FAILURE_DETAIL="pool=Toolathlon endpoints=accepting port=${OPENREWARD_PORT} srun_pid=${server_pid} srun_state=died"
    return 1
  fi

  ENV_SERVER_FAILURE_DETAIL=
  return 0
}

wait_for_env_servers() {
  echo "Waiting up to ${SERVER_WAIT_SECS}s for ${#SERVER_NODES[@]} env server(s)..."
  local waited=0
  local ready
  while true; do
    ready=1
    if ! env_servers_healthy; then
      ready=0
    fi
    if [[ "${ENV_SERVER_SRUN_DIED}" -eq 1 ]]; then
      echo "ERROR: OpenReward env-server srun exited during startup: ${ENV_SERVER_FAILURE_DETAIL}. Recent logs:"
      tail -n 50 "${SERVER_LOG_PREFIX}"-*.log 2>/dev/null || true
      return 1
    fi
    if grep -qiE "Data directory .* must not be owned by root|psql: error: connection .* failed: Connection refused" "${SERVER_LOG_PREFIX}"-*.log 2>/dev/null; then
      echo "ERROR: PostgreSQL failed to start in an env-server container (UID-mapping / data-dir ownership)."
      tail -n 50 "${SERVER_LOG_PREFIX}"-*.log 2>/dev/null || true
      return 1
    fi
    if [[ "${ready}" -eq 1 ]]; then
      echo "All ${#SERVER_NODES[@]} env server(s) are accepting connections on port ${OPENREWARD_PORT}."
      return 0
    fi
    if [[ "${waited}" -ge "${SERVER_WAIT_SECS}" ]]; then
      echo "ERROR: Timed out waiting for env servers after ${SERVER_WAIT_SECS}s: ${ENV_SERVER_FAILURE_DETAIL}. Recent logs:"
      tail -n 50 "${SERVER_LOG_PREFIX}"-*.log 2>/dev/null || true
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

monitor_env_servers() {
  local consecutive_failures=0
  local environment_failures=0
  local healthy

  while kill -0 "${srun_pid}" 2>/dev/null; do
    healthy=1
    if ! env_servers_healthy; then
      healthy=0
    fi

    if [[ "${healthy}" -eq 1 ]]; then
      consecutive_failures=0
    else
      consecutive_failures=$((consecutive_failures + 1))
      if [[ "${ENV_SERVER_SRUN_DIED}" -eq 1 ]]; then
        echo "WARNING: Toolathlon env-server step died (${consecutive_failures}/${SERVER_HEALTH_FAILURE_THRESHOLD}): ${ENV_SERVER_FAILURE_DETAIL}." >&2
      else
        echo "WARNING: Toolathlon env-server endpoint is unreachable (${consecutive_failures}/${SERVER_HEALTH_FAILURE_THRESHOLD}): ${ENV_SERVER_FAILURE_DETAIL}." >&2
      fi
      if [[ "${consecutive_failures}" -ge "${SERVER_HEALTH_FAILURE_THRESHOLD}" ]]; then
        echo "ERROR: env servers remained unhealthy (${ENV_SERVER_FAILURE_DETAIL}); terminating the trainer to avoid rejected zero-data rollouts." >&2
        : >"${SERVER_HEALTH_FAILURE_FILE}"
        kill -TERM "${srun_pid}" 2>/dev/null || true
        return 1
      fi
    fi

    if environment_runtime_healthy; then
      environment_failures=0
    else
      environment_failures=$((environment_failures + 1))
      echo "WARNING: immutable training environment health probe failed (${environment_failures}/${ENVIRONMENT_HEALTH_FAILURE_THRESHOLD})." >&2
      if [[ "${environment_failures}" -ge "${ENVIRONMENT_HEALTH_FAILURE_THRESHOLD}" ]]; then
        echo "ERROR: immutable training environment became unusable; terminating the trainer before further rollout loss." >&2
        : >"${ENVIRONMENT_HEALTH_FAILURE_FILE}"
        kill -TERM "${srun_pid}" 2>/dev/null || true
        return 1
      fi
    fi
    sleep "${SERVER_HEALTH_CHECK_SECS}"
  done
}

environment_cache_markers_ready() {
  [[ -x "${OPENREWARD_JOB_PYTHON}" ]] && \
    [[ -f "${OPENREWARD_JOB_VENV}/.platoon-ready" ]] && \
    [[ -f "${OPENREWARD_JOB_VENV}/.build-complete" ]] && \
    [[ -f "${OPENREWARD_JOB_VENV}/.platoon-env-manifest" ]] && \
    cmp -s "${OPENREWARD_ENV_CACHE_ROOT}/manifests/${OPENREWARD_ENV_KEY}.txt" \
      "${OPENREWARD_JOB_VENV}/.platoon-env-manifest" && \
    grep -q '^relocatable = true$' "${OPENREWARD_JOB_VENV}/pyvenv.cfg"
}

environment_runtime_healthy() {
  PLATOON_EXPECTED_VENV="${OPENREWARD_JOB_VENV}" \
    env -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT -u PYTHONHOME -u PYTHONPATH \
    "${OPENREWARD_JOB_PYTHON}" -I -c '
import importlib.metadata as metadata
import importlib.util as import_util
import os
from pathlib import Path
import sys

prefix = Path(os.environ["PLATOON_EXPECTED_VENV"]).resolve()
assert Path(sys.prefix).resolve() == prefix
modules = ("areal", "megatron", "openhands", "platoon", "ray", "sglang", "torch", "transformers")
for name in modules:
    spec = import_util.find_spec(name)
    assert spec is not None, name
    locations = []
    if spec.origin not in (None, "built-in", "frozen"):
        locations.append(Path(spec.origin).resolve())
    if spec.submodule_search_locations:
        locations.extend(Path(path).resolve() for path in spec.submodule_search_locations)
    assert not locations or any(path.is_relative_to(prefix) for path in locations), (name, locations)
distributions = (
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
for name in distributions:
    distribution = metadata.distribution(name)
    assert Path(distribution.locate_file("")).resolve().is_relative_to(prefix), name
try:
    metadata.distribution("pip")
except metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError("pip was added to the immutable runtime environment")
' >/dev/null 2>&1
}

environment_cache_ready() {
  environment_cache_markers_ready && environment_runtime_healthy
}

invalidate_unhealthy_environment_cache() {
  if [[ -f "${OPENREWARD_JOB_VENV}/.platoon-ready" ]] && ! environment_cache_ready; then
    echo "ERROR: published environment failed its core-package health probe: ${OPENREWARD_JOB_VENV}" >&2
    echo "       The locked builder will quarantine and rebuild this cache entry." >&2
  fi
}

seal_environment_runtime() {
  local site_packages=${OPENREWARD_JOB_VENV}/lib/python3.12/site-packages
  [[ -d "${OPENREWARD_JOB_VENV}/bin" && -d "${site_packages}" ]] || {
    echo "ERROR: immutable environment is missing runtime directories: ${OPENREWARD_JOB_VENV}" >&2
    return 1
  }
  # Removing write permission from the two package roots is cheap even for a
  # very large environment and makes accidental ensurepip/uv/pip operations
  # fail before they can add, remove, or replace a top-level package.
  chmod a-w "${OPENREWARD_JOB_VENV}/bin" "${site_packages}"
}

wait_for_gpu_keepalive() {
  local deadline=$((SECONDS + KEEPALIVE_WAIT_SECS))
  local task_id all_ready status=0

  echo "Waiting up to ${KEEPALIVE_WAIT_SECS}s for GPU keepalive readiness on ${NNODES} node(s)..."
  while (( SECONDS < deadline )); do
    if ! kill -0 "${keepalive_pid}" 2>/dev/null; then
      wait "${keepalive_pid}" || status=$?
      echo "ERROR: GPU keepalive srun exited before readiness (status ${status})." >&2
      tail -n 50 "${USER_ROOT}/logs/gpu-keepalive-prealloc-${RUN_ID}-${JOB_INSTANCE_ID}"-*.err 2>/dev/null || true
      return 1
    fi

    all_ready=1
    for ((task_id = 0; task_id < NNODES; task_id++)); do
      if [[ ! -f "${KEEPALIVE_READY_DIR}/${task_id}.ready" ]]; then
        all_ready=0
        break
      fi
    done
    if [[ "${all_ready}" -eq 1 ]]; then
      echo "GPU keepalive is ready on all ${NNODES} node(s) at $(date -Is)."
      return 0
    fi
    sleep 2
  done

  echo "ERROR: timed out waiting for GPU keepalive readiness after ${KEEPALIVE_WAIT_SECS}s." >&2
  echo "Ready markers found:" >&2
  find "${KEEPALIVE_READY_DIR}" -maxdepth 1 -type f -name '*.ready' -printf '  %f\n' 2>/dev/null | sort >&2 || true
  tail -n 50 "${USER_ROOT}/logs/gpu-keepalive-prealloc-${RUN_ID}-${JOB_INSTANCE_ID}"-*.err 2>/dev/null || true
  return 1
}

monitor_gpu_keepalive() {
  while kill -0 "${keepalive_pid}" 2>/dev/null; do
    sleep 10
  done
  echo "ERROR: GPU keepalive srun exited after readiness; terminating this unprotected job." >&2
  tail -n 50 "${USER_ROOT}/logs/gpu-keepalive-prealloc-${RUN_ID}-${JOB_INSTANCE_ID}"-*.err 2>/dev/null || true
  # Keep successor submission single-writer: the parent shell's SIGTERM
  # handler owns the atomic claim and sbatch call.
  kill -TERM "${launcher_pid}" 2>/dev/null || true
}

stop_gpu_keepalive_before_training() {
  # The keepalive owns all GPUs only to bridge server/environment setup. Once
  # the trainer controller starts, AReaL launches real actor and rollout steps
  # across the allocation. Leaving the synthetic 4096x4096 matmul workload
  # alive beside those workers both wastes capacity and has repeatedly ended in
  # a single keepalive rank being SIGKILLed, which the safety monitor correctly
  # treats as loss of idle-GPU protection. Stop the monitor first so this
  # intentional handoff cannot race its failure path.
  if [[ -n "${keepalive_monitor_pid}" ]] && kill -0 "${keepalive_monitor_pid}" 2>/dev/null; then
    kill "${keepalive_monitor_pid}" 2>/dev/null || true
    wait "${keepalive_monitor_pid}" 2>/dev/null || true
  fi
  keepalive_monitor_pid=

  if [[ -n "${keepalive_pid}" ]] && kill -0 "${keepalive_pid}" 2>/dev/null; then
    echo "Stopping setup-only GPU keepalive before trainer handoff (pid ${keepalive_pid})."
    kill "${keepalive_pid}" 2>/dev/null || true
    wait "${keepalive_pid}" 2>/dev/null || true
  fi
  keepalive_pid=
}

# A manifest match alone cannot prove that a supposedly immutable environment
# was not modified after publication.  Invalidate a poisoned warm cache before
# selecting the keepalive interpreter or entering the locked build path.
invalidate_unhealthy_environment_cache

echo "OpenReward toolathlon (prealloc) run id: ${RUN_ID}"
echo "Config: ${CONFIG}"
if [[ "${#TRAIN_OVERRIDE_ARGS[@]}" -gt 0 ]]; then
  echo "Training overrides: ${TRAIN_OVERRIDE_ARGS[*]}"
fi
echo "Train module: ${TRAIN_MODULE}"
echo "Qwen GDN CP convolution backend: ${PLATOON_QWEN35_GDN_CP_CONV_BACKEND}"
echo "Environment key: ${OPENREWARD_ENV_KEY}"
echo "Immutable training venv: ${OPENREWARD_JOB_VENV}"
echo "Nodes: ${NNODES} (gpus/node: ${GPUS_PER_NODE}); controller node: ${CONTROLLER_NODE}"
echo "Sharding: ${OPENREWARD_SHARD} -> session URLs: ${session_urls}"
echo "To stop after the current job: touch ${STOP_FILE}"
echo "To stop immediately without recovery: touch ${STOP_FILE}; scancel ${SLURM_JOB_ID:-<job-id>}"

# The rank-0 rollout controller reads this to spread sessions across node servers
# (see _select_session_url in platoon.openreward.rollout). Unset => single server.
if [[ "${OPENREWARD_SHARD}" == "1" ]]; then
  export OPENREWARD_SESSION_URLS="${session_urls}"
fi

# --- GPU keepalive (starts immediately, before server/env setup) ----------------
if [[ "${OPENREWARD_GPU_KEEPALIVE}" == "1" ]]; then
  if environment_cache_ready; then
    KEEPALIVE_RUNTIME_PYTHON=${OPENREWARD_JOB_PYTHON}
    echo "Using ready content-addressed environment for GPU keepalive."
  else
    KEEPALIVE_RUNTIME_PYTHON=${OPENREWARD_KEEPALIVE_PYTHON}
    if [[ ! -x "${KEEPALIVE_RUNTIME_PYTHON}" ]]; then
      echo "ERROR: cold environment requires an existing bootstrap Python for GPU keepalive:" >&2
      echo "       ${KEEPALIVE_RUNTIME_PYTHON}" >&2
      echo "       Set OPENREWARD_KEEPALIVE_PYTHON to a venv with working torch/CUDA." >&2
      exit 1
    fi
    echo "Using bootstrap Python for GPU keepalive: ${KEEPALIVE_RUNTIME_PYTHON}"
  fi
  srun \
    --overlap \
    --unbuffered \
    --kill-on-bad-exit=1 \
    --nodes="${NNODES}" \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${OPENREWARD_JOB_STATE_DIR}" \
    --output="${USER_ROOT}/logs/gpu-keepalive-prealloc-${RUN_ID}-${JOB_INSTANCE_ID}-%N-%t.log" \
    --error="${USER_ROOT}/logs/gpu-keepalive-prealloc-${RUN_ID}-${JOB_INSTANCE_ID}-%N-%t.err" \
    /bin/bash -lc "
      set -euo pipefail
      export HOME=${USER_ROOT}
      export UV_CACHE_DIR=${OPENREWARD_UV_CACHE_DIR}
      export UV_LINK_MODE=${OPENREWARD_UV_LINK_MODE}
      export PYTHONDONTWRITEBYTECODE=1
      export KEEPALIVE_START_DELAY_SEC=0
      export KEEPALIVE_EXPECTED_GPUS=${KEEPALIVE_EXPECTED_GPUS}
      export KEEPALIVE_READY_DIR=${KEEPALIVE_READY_DIR}
      source ${USER_ROOT}/.local/bin/env
      cd ${OPENREWARD_JOB_STATE_DIR}
      keepalive_python=${KEEPALIVE_RUNTIME_PYTHON}
      keepalive_venv=\$(cd \"\$(dirname \"\${keepalive_python}\")/..\" && pwd)
      export UV_PROJECT_ENVIRONMENT=\"\${keepalive_venv}\"
      export VIRTUAL_ENV=\"\${keepalive_venv}\"
      export PATH=\"\${keepalive_venv}/bin:\${PATH}\"
      cuda_lib_dirs=\$(find \"\${keepalive_venv}\"/lib/python3.12/site-packages/nvidia -mindepth 2 -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: - || true)
      if [[ -n \"\${cuda_lib_dirs}\" ]]; then
        export LD_LIBRARY_PATH=\"\${cuda_lib_dirs}:\${LD_LIBRARY_PATH:-}\"
      fi
      hash -r
      \"\${keepalive_python}\" -u ${KEEPALIVE_SCRIPT}
    " &
  keepalive_pid=$!
  echo "Started preallocated GPU keepalive step: ${keepalive_pid}"
fi

start_env_servers
if [[ "${OPENREWARD_GPU_KEEPALIVE}" == "1" ]]; then
  wait_for_gpu_keepalive
  monitor_gpu_keepalive &
  keepalive_monitor_pid=$!
fi
wait_for_env_servers

# --- Resolve/build the shared immutable venv inside the CUDA container ----------
# The controller step is intentionally NOT containerized (it must spawn nested
# worker processes), so it has no CUDA toolkit. Compiling CUDA sdists (e.g.
# flash-attn) needs nvcc/CUDA_HOME, which the bare host lacks. On a warm cache,
# the helper only validates the readiness marker; on a miss it builds a
# relocatable staging venv under flock and atomically publishes it.
echo "Preparing immutable venv ${OPENREWARD_JOB_VENV} at $(date -Is)"
if environment_cache_ready; then
  # A published environment is immutable, so a valid host-side marker/manifest
  # is sufficient. Avoid paying for an otherwise no-op pyxis step on warm jobs.
  echo "Immutable environment cache hit; skipping container setup at $(date -Is)"
else
  srun \
    --overlap \
    --unbuffered \
    --nodes=1 \
    --ntasks=1 \
    --nodelist="${CONTROLLER_NODE}" \
    --cpus-per-task=${OPENREWARD_CONTROLLER_CPUS:-8} \
    --mem=${OPENREWARD_CONTROLLER_MEM:-64G} \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts=/lustre:/lustre \
    --container-workdir="${OPENREWARD_JOB_STATE_DIR}" \
    /bin/bash -lc "
      set -euo pipefail
      echo \"Container environment preparation started at \$(date -Is) on \$(hostname)\"
      export HOME=${USER_ROOT}
      export PLATOON_REPO_ROOT=${REPO_ROOT}
      export OPENREWARD_ENV_CACHE_ROOT=${OPENREWARD_ENV_CACHE_ROOT}
      export OPENREWARD_ENV_KEY=${OPENREWARD_ENV_KEY}
      export OPENREWARD_SOURCE_SHA=${OPENREWARD_SOURCE_SHA}
      export OPENREWARD_SOURCE_ARCHIVE=${OPENREWARD_SOURCE_ARCHIVE}
      export OPENREWARD_JOB_VENV=${OPENREWARD_JOB_VENV}
      export OPENREWARD_UV_CACHE_DIR=${OPENREWARD_UV_CACHE_DIR}
      export OPENREWARD_UV_LINK_MODE=${OPENREWARD_UV_LINK_MODE}
      export OPENREWARD_UV_BIN=${OPENREWARD_UV_BIN}
      export OPENREWARD_BUILD_TE=${OPENREWARD_BUILD_TE}
      export OPENREWARD_BUILD_APEX=${OPENREWARD_BUILD_APEX}
      export PYTHONDONTWRITEBYTECODE=1
      export TORCH_CUDA_ARCH_LIST=\"${TORCH_CUDA_ARCH_LIST:-9.0+PTX}\"
      source ${USER_ROOT}/.local/bin/env
      cd ${OPENREWARD_JOB_STATE_DIR}
      # Execute the exact helper captured by this environment key, not a live
      # checkout that could be edited between resolution and this Slurm step.
      helper_snapshot=${OPENREWARD_JOB_STATE_DIR}/prepare_openreward_env.sh
      tar -xOf ${OPENREWARD_SOURCE_ARCHIVE} slurm-scripts/prepare_openreward_env.sh > \"\${helper_snapshot}\"
      chmod 0555 \"\${helper_snapshot}\"
      bash \"\${helper_snapshot}\" build
      echo \"Container environment preparation finished at \$(date -Is)\"
    " &
  env_setup_pid=$!
  set +e
  while true; do
    wait "${env_setup_pid}"
    env_setup_status=$?
    if [[ "${env_setup_status}" -ge 128 ]] && kill -0 "${env_setup_pid}" 2>/dev/null; then
      continue
    fi
    break
  done
  set -e
  env_setup_pid=
  if [[ "${env_setup_status}" -ne 0 ]]; then
    echo "ERROR: immutable environment preparation failed (status ${env_setup_status})." >&2
    exit "${env_setup_status}"
  fi
fi
if ! environment_cache_ready; then
  invalidate_unhealthy_environment_cache
  echo "ERROR: immutable environment failed validation after preparation: ${OPENREWARD_JOB_VENV}" >&2
  exit 1
fi
seal_environment_runtime
touch "${OPENREWARD_ENV_READY_FILE}"

# --- Read-only host controller: runs trainer and spawns workers ----------------
if [[ "${OPENREWARD_GPU_KEEPALIVE}" == "1" ]]; then
  stop_gpu_keepalive_before_training
fi
srun \
  --overlap \
  --unbuffered \
  --nodes=1 \
  --ntasks=1 \
  --nodelist="${CONTROLLER_NODE}" \
  --cpus-per-task=${OPENREWARD_CONTROLLER_CPUS:-8} \
  --mem=${OPENREWARD_CONTROLLER_MEM:-64G} \
  /bin/bash -lc "
    set -euo pipefail
    echo \"Controller entered host Slurm step at \$(date -Is) on \$(hostname)\"
    export HOME=${USER_ROOT}
    export UV_CACHE_DIR=${OPENREWARD_UV_CACHE_DIR}
    export UV_LINK_MODE=${OPENREWARD_UV_LINK_MODE}
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PYTHONDONTWRITEBYTECODE=1
    source ${USER_ROOT}/.local/bin/env
    cd ${OPENREWARD_JOB_STATE_DIR}
    [[ -f ${OPENREWARD_JOB_VENV}/.platoon-ready && -f ${OPENREWARD_ENV_READY_FILE} ]] || {
      echo \"ERROR: immutable environment is missing a readiness marker: ${OPENREWARD_JOB_VENV}\" >&2
      exit 1
    }
    export PATH=${OPENREWARD_RUNTIME_GUARD_BIN}:${OPENREWARD_JOB_VENV}/bin:\${PATH}
    # Keep the cached interpreter on PATH, but do not expose the shared cache as
    # a uv/pip project environment to rollout or recursive PTC subprocesses.
    unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
    cuda_lib_dirs=\$(find ${OPENREWARD_JOB_VENV}/lib/python3.12/site-packages/nvidia -mindepth 2 -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)
    if [[ -n \"\${cuda_lib_dirs}\" ]]; then
      export LD_LIBRARY_PATH=\"\${cuda_lib_dirs}:\${LD_LIBRARY_PATH:-}\"
    fi
    hash -r

    # The controller has no GPU/toolkit. Hiding devices skips optional DeepGEMM
    # probing; worker srun steps receive their own GPU visibility.
    export CUDA_VISIBLE_DEVICES=
    if [[ \"${OPENREWARD_BUILD_TE}\" == \"1\" && \"${OPENREWARD_BUILD_APEX}\" == \"1\" ]]; then
      ${OPENREWARD_JOB_PYTHON} -c \"from platoon.train.areal import PlatoonArealRLTrainer; import onnx; assert hasattr(onnx, 'TensorProto'); import transformer_engine.pytorch; from megatron.bridge.models.conversion.auto_bridge import AutoBridge; from fla.ops.cp import build_cp_context; from fla.modules.convolution import causal_conv1d; import apex, fused_weight_gradient_mlp_cuda; print('Host Megatron/APEX dependency smoke test passed')\"
    elif [[ \"${OPENREWARD_BUILD_TE}\" == \"1\" ]]; then
      ${OPENREWARD_JOB_PYTHON} -c \"from platoon.train.areal import PlatoonArealRLTrainer; import onnx; assert hasattr(onnx, 'TensorProto'); import transformer_engine.pytorch; from megatron.bridge.models.conversion.auto_bridge import AutoBridge; from fla.ops.cp import build_cp_context; from fla.modules.convolution import causal_conv1d; print('Host Megatron dependency smoke test passed')\"
    elif [[ \"${OPENREWARD_BUILD_APEX}\" == \"1\" ]]; then
      ${OPENREWARD_JOB_PYTHON} -c \"import apex, fused_weight_gradient_mlp_cuda; print('Host APEX dependency smoke test passed')\"
    fi

    echo \"Starting trainer at \$(date -Is)\"
    ${OPENREWARD_JOB_PYTHON} -m ${TRAIN_MODULE} --config ${CONFIG} \
      cluster.n_nodes=${NNODES} \
      openreward.session_url=http://localhost:${OPENREWARD_PORT}${TRAIN_OVERRIDE_CMD}
  " &

srun_pid=$!
rm -f "${SERVER_HEALTH_FAILURE_FILE}" "${ENVIRONMENT_HEALTH_FAILURE_FILE}"
monitor_env_servers &
server_health_pid=$!
set +e
while true; do
  wait "${srun_pid}"
  status=$?
  if [[ "${status}" -ge 128 ]] && kill -0 "${srun_pid}" 2>/dev/null; then
    continue
  fi
  break
done
set -e

restart_reason=
if [[ -f "${SERVER_HEALTH_FAILURE_FILE}" || -f "${ENVIRONMENT_HEALTH_FAILURE_FILE}" ]]; then
  status=1
  restart_reason="environment service/runtime health failure"
elif [[ "${status}" -eq 1 ]]; then
  # AReaL reports controller/worker runtime failures (including an exhausted
  # rollout RPC retry) through the trainer srun's ordinary exit status 1. Reuse
  # the bounded infrastructure-restart budget rather than silently ending the
  # continuation chain. Other nonzero statuses remain terminal.
  restart_reason="trainer/controller runtime failure (exit 1)"
fi

if [[ -n "${restart_reason}" ]]; then
  # The trainer srun has exited, so shutdown signals no longer need to interrupt
  # this shell. Keep the atomic claim -> sbatch section from being interrupted
  # after the claim is created but before its successor is submitted. The EXIT
  # cleanup trap remains installed, and the original trainer status is preserved
  # below regardless of whether submission succeeds.
  trap '' TERM INT USR1
  submit_successor "${restart_reason}" 1 || true
fi

exit "${status}"
