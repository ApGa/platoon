#!/bin/bash
# Runs in the workdir. PLATOON_SANDBOX_ROOT is set only by the process provider.
verifier_dir="${PLATOON_SANDBOX_ROOT:-}/logs/verifier"
if [ "$(cat hello.txt 2>/dev/null)" = "$EXPECTED" ]; then
  echo "hello.txt is correct"
  echo 1 > "$verifier_dir/reward.txt"
else
  echo "hello.txt is missing or wrong"
  echo 0 > "$verifier_dir/reward.txt"
fi
