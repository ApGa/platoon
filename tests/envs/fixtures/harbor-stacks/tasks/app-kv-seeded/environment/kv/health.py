"""Exit 0 when the URL given answers 200: the key-value store's healthcheck."""

import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception as exc:
    print(exc, file=sys.stderr)
    sys.exit(1)
