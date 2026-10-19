#!/bin/bash
# Runs in the workdir. PLATOON_SANDBOX_ROOT is set only by the process provider.
verifier_dir="${PLATOON_SANDBOX_ROOT:-}/logs/verifier"
if [ "$(cat primes.txt 2>/dev/null)" = "2,3,5,7,11" ]; then
  echo '{"reward": 1.0, "checks_passed": 1}' > "$verifier_dir/reward.json"
else
  echo '{"reward": 0.0, "checks_passed": 0}' > "$verifier_dir/reward.json"
fi
