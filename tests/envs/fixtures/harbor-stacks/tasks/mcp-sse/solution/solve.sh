#!/bin/bash
# Runs in main: get the secret from the task's MCP server over the SSE transport, with the standard library alone.
root="${PLATOON_SANDBOX_ROOT:-}"
python3 - "$MCP_URL" "$root/app/secret.txt" <<'PY'
import json, sys, urllib.parse, urllib.request

url, out = sys.argv[1], sys.argv[2]
stream = urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "text/event-stream"}), timeout=10)


def next_event():
    name, data = "message", []
    while True:
        line = stream.readline().decode().rstrip("\r\n")
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif not line and data:
            return name, "\n".join(data)


_, endpoint = next_event()
endpoint = urllib.parse.urljoin(url, endpoint)


def send(message):
    headers = {"Content-Type": "application/json"}
    request = urllib.request.Request(endpoint, data=json.dumps(message).encode(), headers=headers, method="POST")
    urllib.request.urlopen(request, timeout=10).close()
    while "id" in message:
        name, data = next_event()
        if name == "message" and json.loads(data).get("id") == message["id"]:
            return json.loads(data)


hello = {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "oracle", "version": "1"}}
send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": hello})
send({"jsonrpc": "2.0", "method": "notifications/initialized"})
reply = send({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "get_secret", "arguments": {}}})
with open(out, "w") as handle:
    handle.write(reply["result"]["content"][0]["text"])
PY
