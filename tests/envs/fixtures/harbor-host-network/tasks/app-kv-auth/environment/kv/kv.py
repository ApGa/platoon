"""A key-value store over HTTP, the "database" of platoon's host-network fixtures: PUT /<key> stores the body, GET
/<key> answers it (404 when unset) and GET / answers "ok".  Where KV_TOKEN is set it answers only the requests that
carry it (``Authorization: Bearer <token>``), and 401 the others.  It keeps its data in /var/lib/kv and records each
write's key in /logs/verifier/kv-writes.log.

It listens on KV_HOST at KV_PORT, which the host-network fallback sets (127.0.0.1, and the port its stack holds); else
on the process provider's listen variable (PLATOON_PORT_6379) on the host's loopback; else on 6379 of every address (a
stack's own network, where the others reach it by name)."""

import hmac
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = os.environ.get("PLATOON_SANDBOX_ROOT", "")  # set by platoon's process provider only
DATA = Path(f"{ROOT}/var/lib/kv")
WRITES = Path(f"{ROOT}/logs/verifier/kv-writes.log")
KEY = re.compile(r"[A-Za-z0-9_-]+")
TOKEN = os.environ.get("KV_TOKEN", "")


class Store(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if not self._allowed():
            return
        if self.path == "/":
            self._answer(200, b"ok")
            return
        key = self.path.strip("/")
        path = DATA / key
        if not KEY.fullmatch(key) or not path.is_file():
            self._answer(404, b"")
            return
        self._answer(200, path.read_bytes())

    def do_PUT(self) -> None:
        if not self._allowed():
            return
        key = self.path.strip("/")
        if not KEY.fullmatch(key):
            self._answer(400, b"bad key")
            return
        (DATA / key).write_bytes(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        with WRITES.open("a") as log:
            log.write(f"{key}\n")
        self._answer(204, b"")

    def _allowed(self) -> bool:
        given = self.headers.get("Authorization", "")
        if TOKEN and not hmac.compare_digest(given.encode(), f"Bearer {TOKEN}".encode()):
            self._answer(401, b"a token is needed")
            return False
        return True

    def _answer(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def log_message(self, *args: object) -> None:
        pass


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    WRITES.parent.mkdir(parents=True, exist_ok=True)
    listen = os.environ.get("PLATOON_PORT_6379")
    host = os.environ.get("KV_HOST") or ("127.0.0.1" if listen else "0.0.0.0")
    port = int(os.environ.get("KV_PORT") or listen or 6379)
    ThreadingHTTPServer((host, port), Store).serve_forever()
