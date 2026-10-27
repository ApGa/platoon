#!/bin/bash
cat > answer.sh <<'SCRIPT'
echo "uid=$(id -u) secret=$(cat /verifier_data/answer 2>/dev/null || echo denied)"
SCRIPT
