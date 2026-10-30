"""A judge for platoon's node-service fixtures, written for platoon: an HTTP server on the Unix socket that
$PLATOON_NODE_SERVICE_SOCKET names, which answers only requests that carry $PLATOON_NODE_SERVICE_TOKEN.

- GET /health answers "ok", or 503 while /state/unhealthy exists;
- POST /evaluate takes {"code": "..."} and grades it against the problem that the X-Platoon-Scope header names
  (/problems/<problem_id>/answer): {"status": "done", "score": 100.0} when the code is the answer, else 0.0; code
  that starts with "slow" is graded 60 s later.  Each request appends its problem to /state/requests.log as it begins;
- GET /admin answers "admin": a route the front must keep from every rollout.

Each start appends its pid to /state/starts.log; with $CRASH_AFTER_S it exits that long after it starts.  Under
platoon's process provider, which translates the paths it gives in the environment and binds as links, it works below
$PLATOON_SANDBOX_ROOT; real sandboxes leave that unset."""

import json
import os
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler

ROOT = os.environ.get("PLATOON_SANDBOX_ROOT", "")
TOKEN = os.environ["PLATOON_NODE_SERVICE_TOKEN"]
SLOW_S = 60.0


class Judge(BaseHTTPRequestHandler):
    def do_GET(self):
        if not self._authorized():
            return
        if self.path == "/health":
            healthy = not os.path.exists(f"{ROOT}/state/unhealthy")
            self._answer(200 if healthy else 503, {"status": "ok" if healthy else "unhealthy"})
        elif self.path == "/admin":
            self._answer(200, {"admin": True})
        else:
            self._answer(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorized():
            return
        if self.path != "/evaluate":
            self._answer(404, {"error": "not found"})
            return
        scope = json.loads(self.headers.get("X-Platoon-Scope") or "{}")
        problem = scope.get("problem_id", "")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        with open(f"{ROOT}/state/requests.log", "a") as log:
            log.write(f"{problem}\n")
        if str(body.get("code", "")).startswith("slow"):
            time.sleep(SLOW_S)
        try:
            with open(f"{ROOT}/problems/{os.path.basename(problem)}/answer") as handle:
                answer = handle.read().strip()
        except OSError:
            self._answer(404, {"error": f"no problem {problem!r}"})
            return
        score = 100.0 if str(body.get("code", "")).strip() == answer else 0.0
        self._answer(200, {"status": "done", "score": score, "problem_id": problem})

    def _authorized(self):
        if self.headers.get("Authorization") == f"Bearer {TOKEN}":
            return True
        self._answer(401, {"error": "unauthorized"})
        return False

    def _answer(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def address_string(self):
        return "unix"

    def log_message(self, *args):
        pass


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def main():
    socket_path = os.environ["PLATOON_NODE_SERVICE_SOCKET"]
    # The socket's path may be too long for an address under the process provider's root: bind it by its name.
    os.chdir(os.path.dirname(socket_path))
    name = os.path.basename(socket_path)
    if os.path.exists(name):
        os.unlink(name)
    with open(f"{ROOT}/state/starts.log", "a") as log:
        log.write(f"{os.getpid()}\n")
    if crash_after := os.environ.get("CRASH_AFTER_S"):
        threading.Timer(float(crash_after), os._exit, args=(3,)).start()
    with Server(name, Judge) as server:
        server.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
