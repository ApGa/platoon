#!/bin/bash
# Runs in the workdir. PLATOON_SANDBOX_ROOT is set only by the process provider.
root="${PLATOON_SANDBOX_ROOT:-}"
problems=""
[ -f "$root/opt/task-src/task.toml" ] || problems="$problems the task is not bound;"
for masked in solution tests; do
  [ -d "$root/opt/task-src/$masked" ] || problems="$problems /opt/task-src/$masked is not a directory;"
  [ -z "$(ls -A "$root/opt/task-src/$masked" 2>/dev/null)" ] || problems="$problems /opt/task-src/$masked is not empty;"
done
[ -d "$root/opt/setup_workspace" ] && [ -z "$(ls -A "$root/opt/setup_workspace")" ] || problems="$problems no empty setup_workspace;"
[ "$(cat "$root/opt/scratch/seed.txt" 2>/dev/null)" = seed ] || problems="$problems no scratch copy;"
[ "$(cat output.txt 2>/dev/null)" = "COMPOSE BINDS THE TASK" ] || problems="$problems output.txt is wrong;"
if [ -z "$problems" ]; then
  echo 1 > "$root/logs/verifier/reward.txt"
else
  echo "$problems"
  echo 0 > "$root/logs/verifier/reward.txt"
fi
