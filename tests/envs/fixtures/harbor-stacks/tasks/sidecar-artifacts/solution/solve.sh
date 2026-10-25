#!/bin/bash
for item in apple banana cherry; do
  python3 -c '
import json, sys, urllib.request
request = urllib.request.Request(sys.argv[1] + "/orders", data=json.dumps({"item": sys.argv[2]}).encode(), method="POST")
urllib.request.urlopen(request, timeout=5).close()
' "$API_URL" "$item"
done
