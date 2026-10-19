"""Toy ORS environments for the shared-env concurrency tests (``test_ors_concurrency.py``).

They reproduce what real environments do when agents share a live env, with only the API that openreward
0.1.125 and 0.1.163 share (plus ``@tool(concurrent=True)`` where the SDK has it):

* ``TrackEnv``: tools that log when they run (``slow``, ``slow_b``, ``slow_sync`` in a worker thread, the
  concurrent ``scan`` and ``scan_declared``); read-modify-write counters (``incr_sync`` and ``incr_sync_b`` in
  worker threads, ``incr_async`` across an await), which lose updates when calls overlap; a ``wait_flag`` /
  ``set_flag`` pair; a terminal ``submit`` that grades the ``write``-able solution over a pause, as Frontier-CS's
  judge does; and ``fill``, a sync tool that grows a dict.
* ``GradedTrackEnv``: adds a sync ``platoon_grade`` that walks the dict ``fill`` grows.
* ``UndeclaredTrackEnv``: ``TrackEnv`` without its terminal declaration, so any of its calls may end an episode.
* ``TrackForkEnv``: ``platoon_fork`` children on their parent's state (TextCraft's shared inventory).
* ``PipeEnv``: Toolathlon's per-session MCP pipe.  ``call_tool`` and ``mcp_health`` share one stdio JSON-RPC
  subprocess through ``_MCPProcess.send_recv``, copied verbatim from the Toolathlon environment server's
  ``/app/server.py``; a call that reads while another one waits fails with "readuntil() called while another
  coroutine is already waiting for incoming data", although the subprocess still executes it.

``RUNS``, ``INSTANCES`` and ``TEARDOWNS`` are module-level, so a test sees every instance: platoon_fork children,
fresh forks, and envs an ORS server in the test process runs.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import threading
import time
from typing import Any

from openreward.environments import Environment, TextBlock, ToolOutput, tool
from pydantic import BaseModel, Field

from platoon.envs.ors import sdk

HAS_CONCURRENT = "concurrent" in inspect.signature(tool).parameters

RUNS: list[tuple[str, str, float, float]] = []  # (tool, tag, start, end) of every run, time.monotonic()
INSTANCES: list[Any] = []  # every TrackEnv, in creation order
TEARDOWNS: list[tuple[int, float]] = []  # (id of the instance, time.monotonic()) of every teardown
_COUNTS = threading.Lock()  # guards RUNNING/PEAK: sync tools run in worker threads
_STATE = {"running": 0, "peak": 0}


def reset() -> None:
    RUNS.clear()
    INSTANCES.clear()
    TEARDOWNS.clear()
    with _COUNTS:
        _STATE.update(running=0, peak=0)


def peak() -> int:
    """The most tool runs at once since :func:`reset`, over every TrackEnv instance."""
    return _STATE["peak"]


def torn_down(env: Any) -> bool:
    return any(instance == id(env) for instance, _ in TEARDOWNS)


def runs(*names: str) -> list[tuple[str, str, float, float]]:
    return [run for run in RUNS if not names or run[0] in names]


def tags(*names: str) -> list[str]:
    return [tag for _, tag, _, _ in runs(*names)]


def overlapping(selected: list[tuple[str, str, float, float]]) -> bool:
    spans = sorted((start, end) for _, _, start, end in selected)
    return any(later[0] < earlier[1] - 1e-4 for earlier, later in zip(spans, spans[1:]))


def concurrent_tool(fn: Any) -> Any:
    """``@tool(concurrent=True)`` on 0.1.163; a plain ``@tool`` on SDKs without the flag."""
    return tool(concurrent=True)(fn) if HAS_CONCURRENT else tool(fn)


class Run(BaseModel):
    seconds: float = 0.0
    tag: str = ""


class Increment(BaseModel):
    n: int = 1
    pause: float = 0.0


class Wait(BaseModel):
    timeout: float = 1.0


class Write(BaseModel):
    content: str
    pause: float = 0.0


class Fill(BaseModel):
    n: int = 100
    pause: float = 0.0


class TrackEnv(Environment):
    """See the module docstring."""

    platoon_terminal_tools = ("submit",)
    platoon_concurrent_tools = ("scan_declared",)

    def __init__(self, task_spec: dict = {}, secrets: dict = {}) -> None:  # noqa: B006 - the SDK's signature
        super().__init__(task_spec, secrets)
        self.state: dict[str, Any] = {"count": 0, "flag": False, "solution": "", "data": {}}
        INSTANCES.append(self)

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[dict]:
        return [{"name": "track-0"}]

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(text="Track the calls.")]

    def teardown(self) -> None:
        TEARDOWNS.append((id(self), time.monotonic()))

    def platoon_copy(self) -> TrackEnv:
        """A copy fork's env.  The teardown only records the instance, so a deep copy shares nothing it releases."""
        return sdk.deepcopy_env(self)

    # ------------------------------------------------------------------ calls that log when they run

    def _enter(self) -> float:
        with _COUNTS:
            _STATE["running"] += 1
            _STATE["peak"] = max(_STATE["peak"], _STATE["running"])
        return time.monotonic()

    def _exit(self, name: str, tag: str, started: float) -> None:
        with _COUNTS:
            _STATE["running"] -= 1
        RUNS.append((name, tag, started, time.monotonic()))

    async def _sleep(self, name: str, params: Run, **output: Any) -> ToolOutput:
        started = self._enter()
        try:
            await asyncio.sleep(params.seconds)
        finally:
            self._exit(name, params.tag, started)
        return ToolOutput(blocks=[TextBlock(text=f"{name}:{params.tag}")], **output)

    @tool
    async def slow(self, params: Run) -> ToolOutput:
        """Take ``seconds`` on the event loop."""
        return await self._sleep("slow", params)

    @tool
    async def slow_b(self, params: Run) -> ToolOutput:
        """``slow`` under another name (the SDK's per-tool locks do not order the two)."""
        return await self._sleep("slow_b", params)

    @tool
    def slow_sync(self, params: Run) -> ToolOutput:
        """Take ``seconds`` in a worker thread (a sync tool)."""
        started = self._enter()
        try:
            time.sleep(params.seconds)
        finally:
            self._exit("slow_sync", params.tag, started)
        return ToolOutput(blocks=[TextBlock(text=f"slow_sync:{params.tag}")])

    @concurrent_tool
    async def scan(self, params: Run) -> ToolOutput:
        """Take ``seconds``; declared ``concurrent=True`` where the SDK has the flag."""
        return await self._sleep("scan", params)

    @tool
    async def scan_declared(self, params: Run) -> ToolOutput:
        """Take ``seconds``; declared concurrent through ``platoon_concurrent_tools``."""
        return await self._sleep("scan_declared", params)

    # ------------------------------------------------------------------ read-modify-write

    def _increment_sync(self, params: Increment) -> ToolOutput:
        for _ in range(params.n):
            value = self.state["count"]
            time.sleep(params.pause)
            self.state["count"] = value + 1
        return ToolOutput(blocks=[TextBlock(text=str(self.state["count"]))])

    @tool
    def incr_sync(self, params: Increment) -> ToolOutput:
        """count += 1, n times, in a worker thread, pausing between the read and the write."""
        return self._increment_sync(params)

    @tool
    def incr_sync_b(self, params: Increment) -> ToolOutput:
        """``incr_sync`` under another name."""
        return self._increment_sync(params)

    @tool
    async def incr_async(self, params: Increment) -> ToolOutput:
        """count += 1, n times, awaiting between the read and the write."""
        for _ in range(params.n):
            value = self.state["count"]
            await asyncio.sleep(params.pause)
            self.state["count"] = value + 1
        return ToolOutput(blocks=[TextBlock(text=str(self.state["count"]))])

    @tool
    async def read_count(self) -> ToolOutput:
        """The counter."""
        return ToolOutput(blocks=[TextBlock(text=str(self.state["count"]))], metadata={"count": self.state["count"]})

    # ------------------------------------------------------------------ interdependent calls

    @tool
    async def wait_flag(self, params: Wait) -> ToolOutput:
        """Wait until ``set_flag`` is called, or ``timeout``."""
        started = self._enter()
        deadline = time.monotonic() + params.timeout
        try:
            while not self.state["flag"] and time.monotonic() < deadline:
                await asyncio.sleep(0.005)
        finally:
            self._exit("wait_flag", "", started)
        return ToolOutput(blocks=[TextBlock(text=f"saw_flag={self.state['flag']}")])

    @tool
    async def set_flag(self) -> ToolOutput:
        """Set the flag."""
        self.state["flag"] = True
        return ToolOutput(blocks=[TextBlock(text="set")])

    # ------------------------------------------------------------------ a judge that grades the solution

    @tool
    async def write(self, params: Write) -> ToolOutput:
        """Write the solution; it lands after ``pause``."""
        started = self._enter()
        try:
            await asyncio.sleep(params.pause)
            self.state["solution"] = params.content
        finally:
            self._exit("write", params.content, started)
        return ToolOutput(blocks=[TextBlock(text=f"wrote {params.content}")])

    @tool
    async def submit(self, params: Run) -> ToolOutput:
        """Grade the solution and end the episode: read it, judge for ``seconds``, and check it held still."""
        started = self._enter()
        try:
            seen = self.state["solution"]
            await asyncio.sleep(params.seconds)
            consistent = self.state["solution"] == seen
        finally:
            self._exit("submit", params.tag, started)
        return ToolOutput(
            blocks=[TextBlock(text=f"graded {seen!r}")],
            reward=1.0 if consistent and seen else 0.0,
            finished=True,
            metadata={"graded": seen, "consistent": consistent},
        )

    # ------------------------------------------------------------------ state a sync tool changes

    @tool
    def fill(self, params: Fill) -> ToolOutput:
        """Add ``n`` keys to the env's dict, one every ``pause`` seconds, in a worker thread."""
        started = self._enter()
        data = self.state["data"]
        try:
            for index in range(params.n):
                data[f"k{len(data)}-{index}"] = index
                time.sleep(params.pause)
        finally:
            self._exit("fill", str(params.n), started)
        return ToolOutput(blocks=[TextBlock(text=str(len(data)))])


class GradedTrackEnv(TrackEnv):
    def platoon_grade(self) -> dict:
        """A sync grader (it runs in a worker thread) that walks the dict ``fill`` grows."""
        walked = 0
        for _key in self.state["data"]:
            walked += 1
            time.sleep(0.0002)
        return {"reward": 1.0, "metadata": {"keys": walked}}


class UndeclaredTrackEnv(TrackEnv):
    platoon_terminal_tools = ()


class TrackForkEnv(TrackEnv):
    def platoon_fork(self, task_spec: dict) -> TrackForkEnv:
        child = TrackForkEnv(dict(task_spec))
        child.state = self.state  # shared, as TextCraft shares its inventory
        return child


# ---------------------------------------------------------------------------------------- Toolathlon's pipe

# A stdio JSON-RPC server that answers one request at a time, after ``arguments.delay`` seconds.
_ECHO_SERVER = r"""
import json, sys, time
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    arguments = (request.get("params") or {}).get("arguments") or {}
    time.sleep(float(arguments.get("delay", 0.05)))
    text = json.dumps({"echo": arguments.get("tag"), "id": request["id"]})
    reply = {"jsonrpc": "2.0", "id": request["id"], "result": {"content": [{"type": "text", "text": text}]}}
    sys.stdout.write(json.dumps(reply) + "\n")
    sys.stdout.flush()
"""


def _make_request(method: str, params: dict | None = None, req_id: int = 1) -> bytes:
    # verbatim from the Toolathlon environment server's /app/server.py
    msg: dict = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        msg["params"] = params
    return (json.dumps(msg) + "\n").encode()


class _MCPProcess:
    """``send_recv`` and ``call_tool`` copied verbatim from the Toolathlon environment server's /app/server.py
    (``_MCPProcess``)."""

    def __init__(self, name: str, proc: asyncio.subprocess.Process):
        self.name = name
        self.proc = proc
        self._req_id = 10

    async def send_recv(self, method: str, params: dict | None = None, timeout: float = 30) -> dict | None:
        self._req_id += 1
        req_id = self._req_id
        req = _make_request(method, params, req_id)
        self.proc.stdin.write(req)  # type: ignore[union-attr]
        await self.proc.stdin.drain()  # type: ignore[union-attr]

        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                line = await asyncio.wait_for(
                    self.proc.stdout.readline(),  # type: ignore[union-attr]
                    timeout=max(0.1, deadline - asyncio.get_event_loop().time()),
                )
            except asyncio.TimeoutError:
                break
            if not line:
                await asyncio.sleep(0.05)
                continue
            try:
                resp = json.loads(line.decode())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if resp.get("id") == req_id:
                return resp
        return None

    async def call_tool(self, tool_name: str, arguments: dict, timeout: float = 120) -> str:
        resp = await self.send_recv("tools/call", {"name": tool_name, "arguments": arguments}, timeout=timeout)
        if resp is None:
            raise TimeoutError(f"MCP tool call timed out after {timeout} seconds")
        if "error" in resp:
            error = resp["error"]
            detail = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            raise RuntimeError(f"MCP tool call failed: {detail}")
        result = resp.get("result", {})
        content_parts = result.get("content", [])
        texts = []
        for part in content_parts:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
            elif isinstance(part, dict):
                texts.append(json.dumps(part))
            else:
                texts.append(str(part))
        output = "\n".join(texts) if texts else json.dumps(result)
        if result.get("isError") is True:
            raise RuntimeError(output or "MCP tool reported an error")
        return output


class CallToolParams(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)


class PipeEnv(Environment):
    """One stdio MCP server subprocess per session, driven the way Toolathlon drives its MCP servers."""

    def __init__(self, task_spec: dict = {}, secrets: dict = {}) -> None:  # noqa: B006 - the SDK's signature
        super().__init__(task_spec, secrets)
        self.mcp: _MCPProcess | None = None
        self.failures: list[str] = []

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[dict]:
        return [{"name": "pipe-0"}]

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(text="Use the MCP tools.")]

    async def setup(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-c", _ECHO_SERVER,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )  # fmt: skip
        self.mcp = _MCPProcess("echo", proc)

    async def teardown(self) -> None:
        if self.mcp is not None and self.mcp.proc.returncode is None:
            self.mcp.proc.kill()
            await self.mcp.proc.wait()

    def _failed(self, exc: Exception) -> ToolOutput:
        message = f"{type(exc).__name__}: {exc}"
        self.failures.append(message)
        return ToolOutput(blocks=[TextBlock(text=message)], metadata={"openhands.dev/tool-error": True})

    @tool
    async def call_tool(self, params: CallToolParams) -> ToolOutput:
        """Call a tool of the session's MCP server (Toolathlon's catalog meta-tool)."""
        assert self.mcp is not None
        try:
            text = await self.mcp.call_tool(params.name, dict(params.arguments), timeout=10)
        except Exception as exc:  # Toolathlon reports these as tool errors
            return self._failed(exc)
        return ToolOutput(blocks=[TextBlock(text=text)])

    @tool
    async def mcp_health(self) -> ToolOutput:
        """Ask the session's MCP server for its tools again (read-only)."""
        assert self.mcp is not None
        try:
            reply = await self.mcp.send_recv("tools/list", {"arguments": {"delay": 0.05}}, timeout=10)
        except Exception as exc:
            return self._failed(exc)
        return ToolOutput(blocks=[TextBlock(text="healthy" if reply is not None else "no reply")])
