#!/usr/bin/env bash
set -euo pipefail

MONITOR_USER="${MONITOR_USER:-$USER}"

human_gib_from_kib() {
  awk -v kib="${1:-0}" 'BEGIN { printf "%.1f", kib / 1024 / 1024 }'
}

percent() {
  awk -v numerator="${1:-0}" -v denominator="${2:-0}" \
    'BEGIN { if (denominator > 0) printf "%.1f", 100.0 * numerator / denominator; else printf "0.0" }'
}

human_duration() {
  local seconds="${1:-0}"
  local days hours minutes

  days=$((seconds / 86400))
  seconds=$((seconds % 86400))
  hours=$((seconds / 3600))
  seconds=$((seconds % 3600))
  minutes=$((seconds / 60))
  seconds=$((seconds % 60))

  if (( days > 0 )); then
    printf '%dd%02dh%02dm%02ds' "$days" "$hours" "$minutes" "$seconds"
  elif (( hours > 0 )); then
    printf '%dh%02dm%02ds' "$hours" "$minutes" "$seconds"
  elif (( minutes > 0 )); then
    printf '%dm%02ds' "$minutes" "$seconds"
  else
    printf '%ds' "$seconds"
  fi
}

summarize_ps() {
  ps -u "$MONITOR_USER" -o pid=,stat=,nlwp=,pcpu=,rss=,etimes=,comm=,args= |
    awk '
      BEGIN {
        procs = threads = cpu = rss = zombies = dstate = 0
        apptainer = starter = squash = overlay = agent = tmux = bash = 0
        apptainer_runtime_sum = apptainer_runtime_max = 0
        trainer_threads = trainer_cpu = trainer_rss = 0
        trainer_pids = ""
      }
      {
        pid = $1
        stat = $2
        nlwp = $3
        pcpu = $4
        r = $5
        elapsed = $6
        comm = $7

        procs++
        threads += nlwp
        cpu += pcpu
        rss += r

        if (stat ~ /^Z/) zombies++
        if (stat ~ /^D/) dstate++
        if (comm == "apptainer" && $0 ~ /apptainer run/) {
          apptainer++
          apptainer_runtime_sum += elapsed
          if (elapsed > apptainer_runtime_max) apptainer_runtime_max = elapsed
        }
        if (comm == "starter") starter++
        if (comm == "squashfuse_ll") squash++
        if (comm == "fuse-overlayfs") overlay++
        if (comm == "tmux:") tmux++
        if (comm == "bash") bash++
        if (comm == "python" && $0 ~ /openhands.agent_server/) agent++
        if (comm == "python3" && $0 ~ /platoon.issue_resolution.train_tinker/) {
          trainer_threads += nlwp
          trainer_cpu += pcpu
          trainer_rss += r
          trainer_pids = trainer_pids (trainer_pids == "" ? "" : ",") pid
        }
      }
      END {
        printf "procs=%d\n", procs
        printf "threads=%d\n", threads
        printf "cpu_cores=%.1f\n", cpu / 100.0
        printf "rss_kib=%d\n", rss
        printf "zombies=%d\n", zombies
        printf "dstate=%d\n", dstate
        printf "apptainer=%d\n", apptainer
        apptainer_runtime_avg = 0
        if (apptainer > 0) apptainer_runtime_avg = apptainer_runtime_sum / apptainer
        printf "apptainer_runtime_avg_sec=%d\n", apptainer_runtime_avg
        printf "apptainer_runtime_max_sec=%d\n", apptainer_runtime_max
        printf "starter=%d\n", starter
        printf "squash=%d\n", squash
        printf "overlay=%d\n", overlay
        printf "agent=%d\n", agent
        printf "tmux=%d\n", tmux
        printf "bash=%d\n", bash
        printf "trainer_pids=%s\n", trainer_pids
        printf "trainer_threads=%d\n", trainer_threads
        printf "trainer_cpu_cores=%.1f\n", trainer_cpu / 100.0
        printf "trainer_rss_kib=%d\n", trainer_rss
      }
    '
}

summarize_lwps() {
  ps -u "$MONITOR_USER" -L -o stat= |
    awk '
      BEGIN { runnable = zombies = dstate = 0 }
      {
        if ($1 ~ /^R/) runnable++
        if ($1 ~ /^Z/) zombies++
        if ($1 ~ /^D/) dstate++
      }
      END {
        printf "lwp_runnable=%d\n", runnable
        printf "lwp_zombie=%d\n", zombies
        printf "lwp_dstate=%d\n", dstate
      }
    '
}

summarize_vmstat() {
  vmstat 1 2 | tail -n 1 |
    awk '{
      printf "runq=%s\n", $1
      printf "blocked=%s\n", $2
      printf "swap_kib=%s\n", $3
      printf "free_kib=%s\n", $4
      printf "buff_cache_kib=%s\n", $5 + $6
      printf "cpu_user=%s\n", $13
      printf "cpu_sys=%s\n", $14
      printf "cpu_idle=%s\n", $15
      printf "cpu_wait=%s\n", $16
    }'
}

summarize_iostat() {
  if ! command -v iostat >/dev/null 2>&1; then
    printf "disk=na\n"
    printf "disk_util=na\n"
    printf "disk_await=na\n"
    return
  fi

  iostat -xz 1 2 |
    awk '
      BEGIN { sample = 0; max_util = 0; max_await = 0; hot = "none" }
      /^Device/ { sample++; next }
      sample >= 2 && NF >= 23 {
        await = ($6 > $12 ? $6 : $12)
        util = $NF
        if (util > max_util) {
          max_util = util
          max_await = await
          hot = $1
        }
      }
      END {
        printf "disk=%s\n", hot
        printf "disk_util=%.1f\n", max_util
        printf "disk_await=%.1f\n", max_await
      }
    '
}

read_kv() {
  local key value
  while IFS='=' read -r key value; do
    [[ -n "${key:-}" ]] || continue
    printf -v "$key" '%s' "$value"
  done
}

count_trainer_fds() {
  local pids="$1"
  local total=0
  local pid count

  [[ -n "$pids" ]] || {
    echo 0
    return
  }

  IFS=',' read -ra pid_array <<< "$pids"
  for pid in "${pid_array[@]}"; do
    if [[ -d "/proc/$pid/fd" ]]; then
      count="$(find "/proc/$pid/fd" -maxdepth 1 -type l 2>/dev/null | wc -l)"
      total=$((total + count))
    fi
  done
  echo "$total"
}

read_kv < <(summarize_ps)
read_kv < <(summarize_lwps)
read_kv < <(summarize_vmstat)
read_kv < <(summarize_iostat)

rss_gib="$(human_gib_from_kib "${rss_kib:-0}")"
trainer_rss_gib="$(human_gib_from_kib "${trainer_rss_kib:-0}")"
free_gib="$(human_gib_from_kib "${free_kib:-0}")"
buff_cache_gib="$(human_gib_from_kib "${buff_cache_kib:-0}")"
swap_gib="$(human_gib_from_kib "${swap_kib:-0}")"
trainer_fds="$(count_trainer_fds "${trainer_pids:-}")"
online_cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)"
mem_total_kib="$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)"
mem_total_gib="$(human_gib_from_kib "$mem_total_kib")"
cpu_util_pct="$(percent "${cpu_cores:-0}" "$online_cpus")"
ram_util_pct="$(percent "${rss_kib:-0}" "$mem_total_kib")"
apptainer_runtime_avg="$(human_duration "${apptainer_runtime_avg_sec:-0}")"
apptainer_runtime_max="$(human_duration "${apptainer_runtime_max_sec:-0}")"

printf 'Tinker resources | user=%s | %s\n' "$MONITOR_USER" "$(date '+%Y-%m-%d %H:%M:%S %Z')"
printf '%-14s %-20s %12s  %s\n' "Scope" "Field" "Value" "Meaning"
printf '%-14s %-20s %12s  %s\n' "------------" "--------------------" "------------" "----------------------------------------"
printf '%-14s %-20s %12s  %s\n' "$MONITOR_USER" "# processes" "${procs:-0}" "Process IDs owned by $MONITOR_USER."
printf '%-14s %-20s %12s  %s\n' "$MONITOR_USER" "# threads" "${threads:-0}" "Linux LWPs across those processes."
printf '%-14s %-20s %12s  %s\n' "$MONITOR_USER" "# zombies" "${zombies:-0}" "Exited processes not reaped by parent."
printf '%-14s %-20s %11s%%  %s\n' "$MONITOR_USER" "CPU core util" "$cpu_util_pct" "User CPU as % of ${online_cpus} online cores; ${cpu_cores:-0} cores active."
printf '%-14s %-20s %11s%%  %s\n' "$MONITOR_USER" "RAM util" "$ram_util_pct" "User RSS ${rss_gib}/${mem_total_gib} GiB host RAM."
printf '%-14s %-20s %11s%%  %s\n' "host" "disk util" "${disk_util:-na}" "Busiest disk ${disk:-na}; not reliably per-user."

printf '\n%-14s %-20s %12s  %s\n' "Scope" "Detail" "Value" "Meaning"
printf '%-14s %-20s %12s  %s\n' "------------" "--------------------" "------------" "----------------------------------------"
printf '%-14s %-20s %12s  %s\n' "$MONITOR_USER" "apptainer" "${apptainer:-0}" "Active apptainer run processes."
printf '%-14s %-20s %12s  %s\n' "$MONITOR_USER" "appt avg/max" "${apptainer_runtime_avg}/${apptainer_runtime_max}" "Average and max elapsed runtime of apptainer run processes."
printf '%-14s %-20s %12s  %s\n' "$MONITOR_USER" "agent_server" "${agent:-0}" "OpenHands server processes inside workspaces."
printf '%-14s %-20s %12s  %s\n' "$MONITOR_USER" "starter/squash/ovl" "${starter:-0}/${squash:-0}/${overlay:-0}" "Apptainer runtime and FUSE helpers."
printf '%-14s %-20s %12s  %s\n' "$MONITOR_USER" "run/D LWPs" "${lwp_runnable:-0}/${lwp_dstate:-0}" "Runnable vs uninterruptible user threads."
printf '%-14s %-20s %12s  %s\n' "host" "runq/blocked" "${runq:-?}/${blocked:-?}" "Overall runnable and blocked tasks on node."
printf '%-14s %-20s %12s  %s\n' "$MONITOR_USER" "trainer thr/fd" "${trainer_threads:-0}/${trainer_fds}" "Trainer threads and open file descriptors."
printf '%-14s %-20s %12s  %s\n' "$MONITOR_USER" "trainer RSS" "${trainer_rss_gib} GiB" "Trainer resident memory."
printf '%-14s %-20s %12s  %s\n' "host" "free/cache/swap" "${free_gib}/${buff_cache_gib}/${swap_gib}" "GiB free, cache, and swap used."
printf '%-14s %-20s %12s  %s\n' "host" "cpu u/s/i/w" "${cpu_user:-?}/${cpu_sys:-?}/${cpu_idle:-?}/${cpu_wait:-?}" "Host CPU user/sys/idle/iowait percentages."
printf '%-14s %-20s %12s  %s\n' "host" "await_ms" "${disk_await:-na}" "Busiest disk await from iostat."
printf '%-14s %-20s %12s  %s\n' "$MONITOR_USER" "trainer_pid" "${trainer_pids:-none}" "Trainer process IDs, if running."