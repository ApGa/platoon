#!/bin/bash
# The verifier: runs in the task's working directory once the agent is done, and writes the reward where Harbor
# expects it: 1 when answer.txt holds the expected answer, else 0. PLATOON_SANDBOX_ROOT is set only by platoon's
# process provider (no container), where /logs is a directory below it; real sandboxes leave it unset.
expected='3'
answer=""
[ -f answer.txt ] && answer="$(tr -d '\r' < answer.txt | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
echo "expected: $expected; answer.txt: $answer"
if [ "$answer" = "$expected" ]; then reward=1; else reward=0; fi
echo "$reward" > "${PLATOON_SANDBOX_ROOT:-}/logs/verifier/reward.txt"
