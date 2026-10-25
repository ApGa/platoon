#!/bin/bash
# Runs in main. PLATOON_SANDBOX_ROOT is set only by platoon's process provider.
root="${PLATOON_SANDBOX_ROOT:-}"
stored=$(python3 -c '
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1] + "/answer", timeout=5) as response:
        print(response.read().decode())
except Exception as exc:
    print(f"nothing ({exc})")
' "$KV_URL")
writes=$(tr '\n' ' ' < "$root/logs/verifier/kv-writes.log" 2>/dev/null)
if [ "$stored" = "hello from the seed" ] && [ "$writes" = "greeting answer " ]; then
  echo "the store holds the seed's greeting as the answer"
  echo 1 > "$root/logs/verifier/reward.txt"
else
  echo "the store answered: $stored; its writes: $writes"
  echo 0 > "$root/logs/verifier/reward.txt"
fi
