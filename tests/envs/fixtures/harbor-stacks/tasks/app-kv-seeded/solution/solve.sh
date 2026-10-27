#!/bin/bash
python3 -c '
import sys, urllib.request
greeting = urllib.request.urlopen(sys.argv[1] + "/greeting", timeout=5).read()
request = urllib.request.Request(sys.argv[1] + "/answer", data=greeting, method="PUT")
urllib.request.urlopen(request, timeout=5).close()
' "$KV_URL"
