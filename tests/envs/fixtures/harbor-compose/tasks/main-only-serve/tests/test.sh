#!/bin/bash
# Runs in the workdir. PLATOON_SANDBOX_ROOT is set only by the process provider.
root="${PLATOON_SANDBOX_ROOT:-}"
port=$(cat "$root/tmp/greeting.port" 2>/dev/null)
served=$(python3 -c '
import socket, sys
with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=5) as connection:
    print(connection.recv(1000).decode().strip())
' "$port" 2>/dev/null)
if [ -n "$served" ] && [ "$(cat answer.txt 2>/dev/null)" = "$served" ]; then
  echo "answer.txt holds what the service answers"
  echo 1 > "$root/logs/verifier/reward.txt"
else
  echo "the service answered: ${served:-nothing}; answer.txt: $(cat answer.txt 2>/dev/null)"
  echo 0 > "$root/logs/verifier/reward.txt"
fi
