"""Print what the URL given answers, and exit 0 when it answers 200: the api's healthcheck and its collect hook."""

import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
        print(response.read().decode())
        sys.exit(0 if response.status == 200 else 1)
except Exception as exc:
    print(exc, file=sys.stderr)
    sys.exit(1)
