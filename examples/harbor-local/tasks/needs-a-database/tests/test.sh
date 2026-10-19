#!/bin/bash
# Never runs under platoon: the compose file makes this task a multi-container stack, which platoon refuses.
verifier_dir="${PLATOON_SANDBOX_ROOT:-}/logs/verifier"
if [ "$(cat count.txt 2>/dev/null)" = "3" ]; then echo 1 > "$verifier_dir/reward.txt"; else echo 0 > "$verifier_dir/reward.txt"; fi
