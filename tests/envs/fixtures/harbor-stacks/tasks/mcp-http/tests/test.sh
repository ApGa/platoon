#!/bin/bash
# Runs in main. PLATOON_SANDBOX_ROOT is set only by platoon's process provider.
root="${PLATOON_SANDBOX_ROOT:-}"
if [ "$(cat "$root/app/secret.txt" 2>/dev/null)" = "harbor-mcp-secret-12345" ]; then
  echo "the secret is in /app/secret.txt"
  echo 1 > "$root/logs/verifier/reward.txt"
else
  echo "/app/secret.txt does not hold the secret"
  echo 0 > "$root/logs/verifier/reward.txt"
fi
