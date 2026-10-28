#!/bin/sh
# FrontierSmith's judge pool, as a platoon node service: the Frontier-CS judge (go-judge 1.11.1 and Frontier-CS's judge
# engine of 2026-09-28, which scores a case from the checker's output alone) and
# judge_front.py, which grades a rollout's code for the problem its scope names, on $PLATOON_NODE_SERVICE_SOCKET.
#
# go-judge runs as your own uid, so it limits programs with rlimits rather than cgroups (cgroupType 0), as the ORS
# server ran it; judge_front.py checks that it does.  Its parallelism and the engine's workers follow the CPUs platoon
# pinned the service to ($PLATOON_NODE_SERVICE_CPUS): half of them run programs at once, through a quarter of them as
# the engine's workers, as the ORS server ran 8 and 4 on 16 CPUs; GJ_PARALLELISM and JUDGE_WORKERS override them.
set -eu
cpus=${PLATOON_NODE_SERVICE_CPUS:-$(nproc)}
half=$((cpus / 2)); [ "$half" -ge 1 ] || half=1
quarter=$((cpus / 4)); [ "$quarter" -ge 1 ] || quarter=1
GJ_PARALLELISM=${GJ_PARALLELISM:-$half}
JUDGE_WORKERS=${JUDGE_WORKERS:-$quarter}
work=${TMPDIR:-/tmp}/frontiersmith-judge
rm -rf "$work" && mkdir -p "$work/files"
GJ_AUTH_TOKEN=$(od -An -N24 -tx1 /dev/urandom | tr -d ' \n')
FRONTIER_ENGINE_AUTH_TOKEN=$(od -An -N24 -tx1 /dev/urandom | tr -d ' \n')
export GJ_PARALLELISM JUDGE_WORKERS GJ_AUTH_TOKEN FRONTIER_ENGINE_AUTH_TOKEN
export GJ_ADDR=http://127.0.0.1:5050 SAVE_OUTPUTS=false ENGINE_SOCKET="$work/engine.sock"
echo "[start-judge] cpus=$cpus GJ_PARALLELISM=$GJ_PARALLELISM JUDGE_WORKERS=$JUDGE_WORKERS"
# As the ORS server patched the engine: a program that go-judge could not run is the judge's failure, not a zero.
python3 /node/judge_front.py --patch /app/src/gojudge.js
# An engine whose go-judge client sends a token gets a go-judge that requires it (upstream's sends none).
auth=""
if grep -q GJ_AUTH_TOKEN /app/src/gojudge.js; then auth="-auth-token $GJ_AUTH_TOKEN"; fi
mounts=/usr/local/share/frontiercs/gojudge-mount.yaml
[ -f "$mounts" ] && mounts="-mount-conf $mounts" || mounts=""
# shellcheck disable=SC2086 # $auth and $mounts are options or nothing
/usr/local/bin/go-judge -http-addr 127.0.0.1:5050 $auth -silent -parallelism "$GJ_PARALLELISM" $mounts \
    -dir "$work/files" &
(cd /app && PORT="$ENGINE_SOCKET" exec node /app/server.js) &
exec python3 /node/judge_front.py
