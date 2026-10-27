"""A key-value store over HTTP, the "database" of platoon's stack fixtures: PUT /<key> stores the body, GET /<key>
answers it (404 when unset) and GET / answers "ok".  It keeps its data in /var/lib/kv and records each write's key
in /logs/verifier/kv-writes.log; with KV_CRASH_ON_WRITE=1 it exits right after its first write."""

import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = os.environ.get("PLATOON_SANDBOX_ROOT", "")  # set by platoon's process provider only
DATA = Path(f"{ROOT}/var/lib/kv")
WRITES = Path(f"{ROOT}/logs/verifier/kv-writes.log")
KEY = re.compile(r"[A-Za-z0-9_-]+")


class Store(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
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
        key = self.path.strip("/")
        if not KEY.fullmatch(key):
            self._answer(400, b"bad key")
            return
        (DATA / key).write_bytes(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        with WRITES.open("a") as log:
            log.write(f"{key}\n")
        self._answer(204, b"")
        if os.environ.get("KV_CRASH_ON_WRITE") == "1":
            os._exit(1)

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
    # The process provider's listen variable, on the host's loopback; else every address, where the others reach it.
    listen = os.environ.get("PLATOON_PORT_6379")
    address = ("127.0.0.1", int(listen)) if listen else ("0.0.0.0", 6379)
    ThreadingHTTPServer(address, Store).serve_forever()
