#!/bin/bash
python3 -c '
import sys, urllib.request
request = urllib.request.Request(sys.argv[1] + "/answer", data=b"forty-two", method="PUT")
urllib.request.urlopen(request, timeout=5).close()
' "$KV_URL"
