#!/bin/bash
# Runs in main: get the secret from the task's MCP server over streamable HTTP, with the standard library alone.
root="${PLATOON_SANDBOX_ROOT:-}"
python3 - "$MCP_URL" "$root/app/secret.txt" <<'PY'
import json, sys, urllib.request

url, out = sys.argv[1], sys.argv[2]
session = None


def post(message):
    global session
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if session:
        headers["Mcp-Session-Id"] = session
    request = urllib.request.Request(url, data=json.dumps(message).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        session = response.headers.get("Mcp-Session-Id") or session
        body = response.read().decode()
    data = [line[5:].strip() for line in body.splitlines() if line.startswith("data:")]
    return json.loads(data[0] if data else body) if body else None


hello = {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "oracle", "version": "1"}}
post({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": hello})
post({"jsonrpc": "2.0", "method": "notifications/initialized"})
reply = post({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "get_secret", "arguments": {}}})
with open(out, "w") as handle:
    handle.write(reply["result"]["content"][0]["text"])
PY
