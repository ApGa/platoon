"""Exit 0 when the URL given answers 200, with the store's token where KV_TOKEN is set: its healthcheck."""

import os
import sys
import urllib.request

request = urllib.request.Request(sys.argv[1])
if os.environ.get("KV_TOKEN"):
    request.add_header("Authorization", f"Bearer {os.environ['KV_TOKEN']}")
try:
    with urllib.request.urlopen(request, timeout=2) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception as exc:
    print(exc, file=sys.stderr)
    sys.exit(1)
