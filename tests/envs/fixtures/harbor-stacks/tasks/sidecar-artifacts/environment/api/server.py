"""An order API, the sidecar of platoon's sidecar-artifacts fixture: POST /orders takes {"item": ...} and appends it
to /tmp/api/orders.log, GET /stats answers how many orders it took, a count only this process knows."""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = os.environ.get("PLATOON_SANDBOX_ROOT", "")  # set by platoon's process provider only
ORDERS = Path(f"{ROOT}/tmp/api/orders.log")
counts = {"orders": 0, "invalid": 0}


class Orders(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        try:
            item = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))["item"]
        except (ValueError, KeyError, TypeError):
            counts["invalid"] += 1
            self._answer(400, {"error": "the body must be a JSON object with an item"})
            return
        if self.path != "/orders":
            self._answer(404, {"error": "not found"})
            return
        counts["orders"] += 1
        with ORDERS.open("a") as log:
            log.write(json.dumps({"item": item}) + "\n")
        self._answer(201, {"item": item})

    def do_GET(self) -> None:
        self._answer(200, counts) if self.path == "/stats" else self._answer(404, {"error": "not found"})

    def _answer(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


if __name__ == "__main__":
    ORDERS.parent.mkdir(parents=True, exist_ok=True)
    ThreadingHTTPServer(("127.0.0.1", int(os.environ.get("PLATOON_PORT_8000", "8000"))), Orders).serve_forever()
