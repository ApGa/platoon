"""The harness ``test-mcp-direct`` (``mcp-direct.toml``): in the task's sandbox, it reaches the server ``mcp-server``
that ``{task_mcp_servers}`` lists (``TASK_MCP_SERVERS``) at its URL, over its transport, calls ``get_secret``, and
writes what it answers to ``/app/secret.txt`` (below ``$PLATOON_SANDBOX_ROOT`` on the process provider).  Standard
library alone, for the Python 3 of ``main``."""

import json
import os
import urllib.parse
import urllib.request

HELLO = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "direct", "version": "1"}},
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}
CALL = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "get_secret", "arguments": {}}}


def events(lines):
    """The (type, data) of each event of an event stream, from its lines."""

    event, data = "message", []
    for raw in lines:
        line = raw.decode().rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:") :].strip())


def post(url, message, session=None):
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if session:
        headers["Mcp-Session-Id"] = session
    request = urllib.request.Request(url, data=json.dumps(message).encode(), headers=headers, method="POST")
    return urllib.request.urlopen(request, timeout=30)


def streamable_http(url):
    """The answer to the call, over streamable HTTP: a POST per message, answered with JSON or an event stream."""

    with post(url, HELLO) as response:
        session = response.headers.get("Mcp-Session-Id")
        response.read()
    post(url, INITIALIZED, session).close()
    with post(url, CALL, session) as response:
        if not response.headers.get("Content-Type", "").startswith("text/event-stream"):
            return json.loads(response.read())
        replies = (json.loads(data) for event, data in events(response) if event == "message")
        return next(reply for reply in replies if reply.get("id") == CALL["id"])


def sse(url):
    """The answer to the call, over the legacy SSE transport: the stream names where to post, and the answers come on
    it."""

    request = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    with urllib.request.urlopen(request, timeout=30) as stream:
        received = events(stream)
        event, endpoint = next(received)
        assert event == "endpoint", event
        endpoint = urllib.parse.urljoin(url, endpoint)
        replies = (json.loads(data) for event, data in received if event == "message")
        post(endpoint, HELLO).close()
        next(reply for reply in replies if reply.get("id") == HELLO["id"])
        post(endpoint, INITIALIZED).close()
        post(endpoint, CALL).close()
        return next(reply for reply in replies if reply.get("id") == CALL["id"])


def main():
    servers = json.loads(os.environ["TASK_MCP_SERVERS"])
    server = next(server for server in servers if server["name"] == "mcp-server")
    reply = (sse if server["transport"] == "sse" else streamable_http)(server["url"])
    with open(os.environ.get("PLATOON_SANDBOX_ROOT", "") + "/app/secret.txt", "w") as handle:
        handle.write(reply["result"]["content"][0]["text"])
    print("wrote the secret that", server["url"], "answered")


if __name__ == "__main__":
    main()
