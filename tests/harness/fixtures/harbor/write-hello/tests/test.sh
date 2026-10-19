#!/bin/bash
# Runs in the workdir. PLATOON_SANDBOX_ROOT is set only by the process provider.
verifier_dir="${PLATOON_SANDBOX_ROOT:-}/logs/verifier"
if [ "$(cat hello.txt 2>/dev/null)" = "hello" ]; then echo 1 > "$verifier_dir/reward.txt"; else echo 0 > "$verifier_dir/reward.txt"; fi
