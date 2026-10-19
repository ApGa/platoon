#!/bin/bash
# The judge is unreachable, so the verifier declares the grade invalid instead of scoring 0.
verifier_dir="${PLATOON_SANDBOX_ROOT:-}/logs/verifier"
echo '{"reward": 0, "invalid": 1, "failure_kind": "judge_unavailable"}' > "$verifier_dir/reward.json"
