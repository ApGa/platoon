"""Store a value in the key-value store at $KV_URL: the init job of platoon's app-kv-seeded fixture."""

import os
import sys
import urllib.request

key, value = sys.argv[1], sys.argv[2]
request = urllib.request.Request(f"{os.environ['KV_URL']}/{key}", data=value.encode(), method="PUT")
urllib.request.urlopen(request, timeout=5).close()
print(f"stored {key}")
