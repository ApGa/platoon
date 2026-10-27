"""An MCP server for platoon's stack fixtures, after Harbor's hello-mcp: ``get_secret`` (read-only) answers the
secret, and ``rotate_secret`` replaces it.  ``--transport`` says how it serves: ``streamable-http`` at ``/mcp`` (a
session id given at initialize and asked for after it, a call answered as an event stream, anything else as JSON), or
the legacy ``sse`` transport (a GET of ``/sse`` opens a stream that names where to POST, and the answers come on it).
It listens on its listen variable (``PLATOON_PORT_8000``) on the host's loopback when that is set, as platoon's
process provider sets it, else on port 8000 of every address."""

import argparse
import json
import os
import queue
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SECRET = "harbor-mcp-secret-12345"
VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
TOOLS = [
    {
        "name": "get_secret",
        "description": "Returns a secret value.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "rotate_secret",
        "description": "Replaces the secret with a new one.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]
STATE = {"secret": SECRET}
SESSIONS = set()  # streamable-http's session ids
STREAMS = {}  # sse: each session's queue of events to send
LOCK = threading.Lock()


def answer(message):
    """The JSON-RPC answer to ``message``; None for a notification."""

    if "id" not in message:
        return None
    method, params = message.get("method"), message.get("params") or {}
    if method == "initialize":
        version = params.get("protocolVersion")
        result = {
            "protocolVersion": version if version in VERSIONS else VERSIONS[-1],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fixture-mcp", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call" and params.get("name") == "get_secret":
        result = {"content": [{"type": "text", "text": STATE["secret"]}], "isError": False}
    elif method == "tools/call" and params.get("name") == "rotate_secret":
        with LOCK:
            STATE["secret"] = f"rotated-{uuid.uuid4().hex[:8]}"
        result = {"content": [{"type": "text", "text": "rotated"}], "isError": False}
    elif method == "tools/call":
        return error(message["id"], -32602, f"Unknown tool: {params.get('name')}")
    elif method == "ping":
        result = {}
    else:
        return error(message["id"], -32601, f"Method not found: {method}")
    return {"jsonrpc": "2.0", "id": message["id"], "result": result}


def error(request_id, code, text):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": text}}


def event(name, data):
    return f"event: {name}\ndata: {data}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if TRANSPORT != "sse" or self.path != "/sse":
            self.reply(405 if self.path == "/mcp" else 404, b"")
            return
        session = uuid.uuid4().hex
        events = STREAMS[session] = queue.Queue()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(event("endpoint", f"/messages/?session_id={session}"))
            self.wfile.flush()
            while True:
                try:
                    self.wfile.write(events.get(timeout=15))
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except OSError:  # the client went away
            pass
        finally:
            STREAMS.pop(session, None)

    def do_POST(self):
        try:
            message = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except ValueError:
            self.reply(400, b"not JSON")
            return
        if TRANSPORT == "sse":
            self.post_on_stream(message)
        else:
            self.post_streamable(message)

    def do_DELETE(self):
        SESSIONS.discard(self.headers.get("Mcp-Session-Id"))
        self.reply(200, b"")

    def post_on_stream(self, message):
        """The legacy transport: accept the message, and send its answer on its session's stream."""

        _, _, query = self.path.partition("?session_id=")
        events = STREAMS.get(query)
        if not self.path.startswith("/messages/") or events is None:
            self.reply(404, b"unknown session")
            return
        self.reply(202, b"")
        reply = answer(message)
        if reply is not None:
            events.put(event("message", json.dumps(reply)))

    def post_streamable(self, message):
        """Streamable HTTP: a session from initialize on, a call answered as an event stream, the rest as JSON."""

        if self.path != "/mcp":
            self.reply(404, b"")
            return
        session = self.headers.get("Mcp-Session-Id")
        headers = {}
        if message.get("method") == "initialize":
            session = uuid.uuid4().hex
            SESSIONS.add(session)
            headers["Mcp-Session-Id"] = session
        elif session not in SESSIONS:
            self.reply(400 if session is None else 404, b"no such session")
            return
        reply = answer(message)
        if reply is None:
            self.reply(202, b"")
        elif message.get("method") == "tools/call":
            self.reply(200, event("message", json.dumps(reply)), "text/event-stream", headers)
        else:
            self.reply(200, json.dumps(reply).encode(), "application/json", headers)

    def reply(self, status, body, content_type="text/plain", headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--transport", choices=("streamable-http", "sse"), required=True)
    TRANSPORT = parser.parse_args().transport
    listen = os.environ.get("PLATOON_PORT_8000")
    server = ThreadingHTTPServer(("127.0.0.1", int(listen)) if listen else ("0.0.0.0", 8000), Handler)
    server.daemon_threads = True
    server.serve_forever()
