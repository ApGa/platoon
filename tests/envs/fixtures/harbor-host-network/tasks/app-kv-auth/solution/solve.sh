#!/bin/bash
python3 -c '
import os, sys, urllib.request
request = urllib.request.Request(sys.argv[1] + "/answer", data=b"forty-two", method="PUT")
if os.environ.get("KV_TOKEN"):
    request.add_header("Authorization", "Bearer " + os.environ["KV_TOKEN"])
urllib.request.urlopen(request, timeout=5).close()
' "$KV_URL"
