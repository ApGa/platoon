#!/bin/bash
# Runs as root in the agent's own sandbox: keeps a secret for itself, then runs the agent's script as nobody, which must
# not read it. The agent's files must be its own.
set -u
echo 0 > /logs/verifier/reward.txt
[ "$(id -u)" = 0 ] || { echo "not root: $(id -u)"; exit 0; }
[ "$(stat -c %u /app/answer.sh 2>/dev/null)" = 1000 ] || { echo "answer.sh is not the agent's"; exit 0; }
mkdir -p /verifier_data && echo 42 > /verifier_data/answer
chmod 700 /verifier_data && chmod 600 /verifier_data/answer
got=$(setpriv --reuid=65534 --regid=65534 --clear-groups sh /app/answer.sh 2>&1)
echo "the answer said: $got"
if [ "$got" = "uid=65534 secret=denied" ]; then echo 1 > /logs/verifier/reward.txt; fi
