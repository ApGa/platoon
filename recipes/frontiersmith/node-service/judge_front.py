#!/usr/bin/env python3
"""FrontierSmith's judge front: the only route into the node's Frontier-CS judge pool, which platoon's front reaches
over $PLATOON_NODE_SERVICE_SOCKET with $PLATOON_NODE_SERVICE_TOKEN and a rollout's scope (``X-Platoon-Scope``).
Standard library only; it runs in the judge's image, next to the engine and go-judge (start-judge.sh starts them).

    python3 judge_front.py --patch /app/src/gojudge.js   # once per start, before the engine: see guard() below
    python3 judge_front.py                               # serve

Routes, each refused without the token:

- ``GET /health``: the engine lists its problems, and go-judge answers and runs without cgroups (``cgroupType`` 0,
  rlimits), as the ORS server ran it; else 503.
- ``POST /evaluate`` ``{"code": "<C++>", "timeout": <seconds>}``: grades the code for the problem the scope names
  (``problem_id``; one the request names is ignored), through the engine (``/submit``, then ``/result/<sid>`` until
  it is done), and answers what the rollout may see: ``{"status": "done", "score": <0-100>, "score_unbounded": ...,
  "passed": ..., "cases": <n>, "perfect_cases": <n>}``, with no checker message or program output; ``score`` 0 for
  code that does not compile (``"message": "compile error"``) or is empty.  When the judge cannot grade it (the
  engine's own error, a malformed or out-of-range score, no result within the timeout, the engine unreachable) it
  answers 503 ``{"status": "error", "invalid": true, ...}``: a judge's failure, never a zero.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import socket
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler

TOKEN = os.environ.get("PLATOON_NODE_SERVICE_TOKEN", "")
ENGINE_SOCKET = os.environ.get("ENGINE_SOCKET", "/tmp/frontiersmith-judge/engine.sock")
ENGINE_TOKEN = os.environ.get("FRONTIER_ENGINE_AUTH_TOKEN", "")
GJ_ADDR = os.environ.get("GJ_ADDR", "http://127.0.0.1:5050")
GJ_TOKEN = os.environ.get("GJ_AUTH_TOKEN", "")
MAX_CODE_BYTES = 10_000_000
DEFAULT_TIMEOUT_S = 3600.0
POLL_S = 0.5
HEALTH_CACHE_S = 5.0  # how long a health check that passed answers the next ones (a failed one answers none)
# go-judge's own failures, which the engine would otherwise grade as a wrong answer (the ORS server's guard).
INFRA_STATUSES = ("Internal Error", "System Error", "Runner Error")
GUARD = """// platoon (FrontierSmith recipe): a program that go-judge could not run is the judge's failure, not a zero.
function platoonCheckResults(results) {
    if (!Array.isArray(results) || !results.length) {
        throw new Error('judge infrastructure: a malformed go-judge result');
    }
    for (const result of results) {
        if (!result || typeof result.status !== 'string' || %s.includes(result.status)) {
            const status = result && result.status;
            throw new Error('judge infrastructure: go-judge could not run a program (' + status + ')');
        }
    }
    return results;
}
""" % json.dumps(list(INFRA_STATUSES))
GUARDED = {
    "return data[0];": "return platoonCheckResults(data)[0];",
    "return data;\n    }\n\n    // Delete file": "return platoonCheckResults(data);\n    }\n\n    // Delete file",
}


def guard(path: str) -> None:
    """Make the engine's go-judge client throw on go-judge's own failures, so that the engine reports an error rather
    than a case it failed to run as a wrong answer; refuses an engine whose code is not the one it knows."""

    with open(path) as handle:
        source = handle.read()
    if source.startswith("// platoon (FrontierSmith recipe)"):
        return
    for original, replacement in GUARDED.items():
        if source.count(original) != 1:
            raise SystemExit(f"{path}: not the go-judge client this front knows ({original!r})")
        source = source.replace(original, replacement)
    with open(path, "w") as handle:
        handle.write(GUARD + source)


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("engine", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


class JudgeError(RuntimeError):
    """The judge cannot grade the code: an infrastructure failure."""


def engine(method: str, path: str, body: dict | None = None, *, timeout: float = 60.0) -> tuple[int, object]:
    """One request to the engine over its socket: its status and its JSON answer."""

    connection = _UnixConnection(ENGINE_SOCKET, timeout)
    try:
        payload = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {ENGINE_TOKEN}"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        data = response.read()
        try:
            return response.status, json.loads(data or b"null")
        except ValueError:
            return response.status, None
    except OSError as exc:
        raise JudgeError(f"the engine cannot be reached: {exc}") from exc
    finally:
        connection.close()


def gojudge_config() -> dict:
    """go-judge's configuration (``GET /config``)."""

    host, _, port = GJ_ADDR.removeprefix("http://").partition(":")
    connection = http.client.HTTPConnection(host, int(port or 80), timeout=10)
    try:
        connection.request("GET", "/config", headers={"Authorization": f"Bearer {GJ_TOKEN}"})
        response = connection.getresponse()
        if response.status != 200:
            raise JudgeError(f"go-judge's /config answered {response.status}")
        return json.loads(response.read())
    except (OSError, ValueError) as exc:
        raise JudgeError(f"go-judge cannot be reached: {exc}") from exc
    finally:
        connection.close()


def _found(value: object, key: str) -> object:
    """The first value of ``key`` anywhere in ``value``."""

    if isinstance(value, dict):
        if key in value:
            return value[key]
        value = list(value.values())
    if isinstance(value, list):
        for item in value:
            if (found := _found(item, key)) is not None:
                return found
    return None


_health_lock = threading.Lock()
_health_passed = -HEALTH_CACHE_S  # when a check last passed (monotonic)


def health() -> str:
    """'' when the judge can grade, else why not.  A check that passed in the last few seconds answers; a failed one
    never answers for the next: each check after it looks again, so that one bad moment is one failed check, as the
    supervisor counts them."""

    global _health_passed
    with _health_lock:
        if time.monotonic() - _health_passed < HEALTH_CACHE_S:
            return ""
        problem = _check()
        if not problem:
            _health_passed = time.monotonic()
        return problem


def _check() -> str:
    try:
        cgroup_type = _found(gojudge_config(), "cgroupType")
        if cgroup_type != 0:
            return f"go-judge runs with cgroupType {cgroup_type!r}, not 0 (rlimits, as the ORS server ran it)"
        status, _ = engine("GET", "/problems", timeout=30)
    except JudgeError as exc:
        return str(exc)
    return "" if status == 200 else f"the engine's /problems answered {status}"


def evaluate(problem: str, code: str, timeout_s: float) -> dict:
    """The grade of ``code`` for ``problem``, as the rollout may see it; :class:`JudgeError` when there is none."""

    if not code.strip():
        return {"status": "done", "score": 0.0, "message": "empty submission"}
    status, answer = engine("POST", "/submit", {"pid": problem, "lang": "cpp", "code": code}, timeout=60)
    sid = answer.get("sid") if isinstance(answer, dict) else None
    if status != 200 or not isinstance(sid, int) or isinstance(sid, bool):
        raise JudgeError(f"the engine took no submission ({status}: {str(answer)[:300]})")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status, raw = engine("GET", f"/result/{sid}", timeout=60)
        if status == 200 and isinstance(raw, dict) and raw.get("status") in ("done", "error"):
            return _sanitized(raw)
        if status not in (200, 404):  # 404: not graded yet
            raise JudgeError(f"the engine's result of {sid} answered {status}")
        time.sleep(POLL_S)
    raise JudgeError(f"no result within {timeout_s:g} s")


def _sanitized(raw: dict) -> dict:
    if raw.get("status") == "error":
        if str(raw.get("error", "")).startswith("Error: compile failed:"):
            return {"status": "done", "score": 0.0, "message": "compile error"}
        raise JudgeError(f"the engine failed to grade it: {str(raw.get('error'))[:300]}")
    score = raw.get("score")
    number = isinstance(score, (int, float)) and not isinstance(score, bool)
    if not number or not math.isfinite(score) or not 0 <= score <= 100:
        raise JudgeError(f"the engine gave the score {score!r}")
    cases = raw.get("cases")
    if not isinstance(cases, list):
        cases = []
    perfect = sum(1 for case in cases if isinstance(case, dict) and case.get("scoreRatio") == 1.0)
    unbounded = raw.get("scoreUnbounded", score)
    return {
        "status": "done",
        "score": float(score),
        "score_unbounded": float(unbounded) if isinstance(unbounded, (int, float)) else float(score),
        "passed": bool(raw.get("passed")),
        "cases": len(cases),
        "perfect_cases": perfect,
    }


class Front(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if not self._authorized():
            return
        if self.path != "/health":
            self._answer(404, {"error": "not found"})
            return
        problem = health()
        self._answer(503 if problem else 200, {"status": "error" if problem else "ok", "detail": problem})

    def do_POST(self) -> None:
        if not self._authorized():
            return
        if self.path != "/evaluate":
            self._answer(404, {"error": "not found"})
            return
        try:
            scope = json.loads(self.headers.get("X-Platoon-Scope") or "{}")
            problem = scope.get("problem_id") if isinstance(scope, dict) else None
            length = int(self.headers.get("Content-Length") or 0)
            if not isinstance(problem, str) or not problem or "/" in problem or length > MAX_CODE_BYTES + 100_000:
                raise ValueError("a scope with a problem_id, and a body within the size limit")
            body = json.loads(self.rfile.read(length) or b"{}")
            code = body.get("code") if isinstance(body, dict) else None
            if not isinstance(code, str) or len(code.encode()) > MAX_CODE_BYTES:
                raise ValueError("a JSON body with the code, within the size limit")
            timeout = body.get("timeout", DEFAULT_TIMEOUT_S)
            timeout_s = DEFAULT_TIMEOUT_S
            if isinstance(timeout, (int, float)) and timeout > 0:
                timeout_s = min(float(timeout), 4 * DEFAULT_TIMEOUT_S)
        except (ValueError, TypeError) as exc:
            self._answer(400, {"status": "error", "message": f"expected {exc}"})
            return
        try:
            self._answer(200, evaluate(problem, code, timeout_s))
        except JudgeError as exc:
            print(f"[judge-front] {problem}: {exc}", flush=True)
            self._answer(503, {"status": "error", "invalid": True, "message": f"the judge cannot grade it: {exc}"})

    def _authorized(self) -> bool:
        if TOKEN and self.headers.get("Authorization") == f"Bearer {TOKEN}":
            return True
        self._answer(401, {"error": "unauthorized"})
        return False

    def _answer(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def address_string(self) -> str:
        return "platoon"

    def log_message(self, format: str, *args: object) -> None:
        pass


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def main() -> int:
    parser = argparse.ArgumentParser(description="FrontierSmith's judge front (see the module docstring)")
    parser.add_argument("--patch", help="guard the engine's go-judge client at this path, then exit")
    args = parser.parse_args()
    if args.patch:
        guard(args.patch)
        return 0
    if not TOKEN:
        raise SystemExit("PLATOON_NODE_SERVICE_TOKEN is not set: platoon starts this front as a node service")
    path = os.environ["PLATOON_NODE_SERVICE_SOCKET"]
    if os.path.exists(path):
        os.unlink(path)
    with Server(path, Front) as server:
        print(f"[judge-front] serving on {path}", flush=True)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
