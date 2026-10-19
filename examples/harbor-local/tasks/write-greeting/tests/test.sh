#!/bin/bash
# The verifier: runs in the task's working directory after the agent finishes, and writes the reward
# where Harbor expects it. PLATOON_SANDBOX_ROOT is set only by platoon's process provider (no container),
# where /logs is a directory below it; real sandboxes leave it unset.
verifier_dir="${PLATOON_SANDBOX_ROOT:-}/logs/verifier"
if [ "$(cat greeting.txt 2>/dev/null)" = "hello from platoon" ]; then
  echo 1 > "$verifier_dir/reward.txt"
else
  echo 0 > "$verifier_dir/reward.txt"
fi
