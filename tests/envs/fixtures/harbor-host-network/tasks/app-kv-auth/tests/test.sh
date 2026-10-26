#!/bin/bash
# Runs in main, through [verifier].env's KV_CHECK_URL. PLATOON_SANDBOX_ROOT is set only by platoon's process provider.
root="${PLATOON_SANDBOX_ROOT:-}"
stored=$(python3 -c '
import os, sys, urllib.request
request = urllib.request.Request(sys.argv[1] + "/answer")
if os.environ.get("KV_TOKEN"):
    request.add_header("Authorization", "Bearer " + os.environ["KV_TOKEN"])
try:
    with urllib.request.urlopen(request, timeout=5) as response:
        print(response.read().decode())
except Exception as exc:
    print(f"nothing ({exc})")
' "$KV_CHECK_URL")
if [ "$stored" = "forty-two" ] && grep -qx answer "$root/logs/verifier/kv-writes.log" 2>/dev/null; then
  echo "the store holds the answer, and records its write"
  echo 1 > "$root/logs/verifier/reward.txt"
else
  echo "the store answered: $stored"
  echo 0 > "$root/logs/verifier/reward.txt"
fi
