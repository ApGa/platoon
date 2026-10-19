"""Local fakes for harness tests: a scripted OpenAI Chat endpoint, a process-backed
sandbox, and an env service whose sessions grade files in that sandbox."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from aiohttp import web

from platoon.envs import service as envs_service
from platoon.sandbox.base import CWD_ERROR_EXIT_CODE, CWD_ERROR_PREFIX
from platoon.task import Task


def bash_call(command: str, call_id: str = "call-1") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
    }


def answer(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


class FakeChatServer:
    """OpenAI Chat Completions on loopback that replays scripted assistant messages."""

    def __init__(self, replies: list[dict[str, Any]], *, api_key: str = "agent-secret") -> None:
        self.replies = list(replies)
        self.api_key = api_key
        self.requests: list[dict[str, Any]] = []
        self.call_ids: list[str] = []
        self.base_url = ""
        self._runner: web.AppRunner | None = None

    async def start(self) -> str:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._chat)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001 - aiohttp exposes no public accessor
        self.base_url = f"http://127.0.0.1:{port}"
        return self.base_url

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def __aenter__(self) -> FakeChatServer:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()

    async def _chat(self, request: web.Request) -> web.Response:
        if request.headers.get("Authorization") != f"Bearer {self.api_key}":
            return web.json_response({"error": {"message": "unknown agent key"}}, status=401)
        self.requests.append(await request.json())
        message = self.replies.pop(0) if self.replies else answer("out of script")
        call_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self.call_ids.append(call_id)
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        return web.json_response(
            {
                "id": call_id,
                "object": "chat.completion",
                "model": "policy",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            }
        )


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class CommandDidNotStart(RuntimeError):
    """What hpc-sandbox raises (its ``SandboxError``) for a command that never started for a reason of the sandbox's
    own, and before 0.2.0 also for one whose working directory could not be entered."""


UnenterableCwd = Literal["raise", "exit-126"]


class ProcessSandbox:
    """A ``Sandbox`` over a host directory: commands are host subprocesses.

    ``unenterable_cwd`` says what a command gets whose working directory (``cwd``, else the workdir) cannot be
    entered: by default the host's OSError, as from the process provider; ``"raise"`` :class:`CommandDidNotStart`,
    as hpc-sandbox raised its ``SandboxError`` before 0.2.0; ``"exit-126"`` exit status 126 with empty stdout and only
    hpc-sandbox's line on stderr (``CWD_ERROR_PREFIX`` + ``"<directory>: <reason>"``), as hpc-sandbox 0.2 answers
    and docker exits.
    """

    def __init__(self, root: Path, *, unenterable_cwd: UnenterableCwd | None = None) -> None:
        self.id = f"sandbox-{uuid.uuid4().hex[:8]}"
        self.root = root
        self.workdir = root / "app"
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.unenterable_cwd = unenterable_cwd
        self.commands: list[list[str] | str] = []
        self.stopped = False

    async def exec(
        self,
        cmd: str | list[str],
        *,
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> ExecResult:
        self.commands.append(cmd)
        directory = cwd or str(self.workdir)
        if self.unenterable_cwd is not None and not (os.path.isdir(directory) and os.access(directory, os.X_OK)):
            if self.unenterable_cwd == "raise":
                raise CommandDidNotStart(
                    f"sandbox {self.id}: the command did not start (exit 2): /bin/sh: cd: can't cd to {directory}"
                )
            if os.path.isdir(directory):
                why = "permission denied"
            else:
                why = "not a directory" if os.path.exists(directory) else "it does not exist or cannot be reached"
            return ExecResult(CWD_ERROR_EXIT_CODE, "", f"{CWD_ERROR_PREFIX}{directory}: {why}\n")
        argv = ["/bin/sh", "-c", cmd] if isinstance(cmd, str) else list(cmd)
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=directory,
            env={**os.environ, **(env or {})},
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(stdin), timeout_s)
        except TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = await process.communicate()
            return ExecResult(-9, stdout.decode(), stderr.decode(), timed_out=True)
        except asyncio.CancelledError:
            os.killpg(process.pid, signal.SIGKILL)
            raise
        return ExecResult(process.returncode, stdout.decode(), stderr.decode())

    async def upload(self, src: str | Path, dst: str) -> None:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    async def download(self, src: str, dst: str | Path) -> None:
        shutil.copyfile(src, dst)

    async def stop(self) -> None:
        self.stopped = True


@dataclass
class FakeSession:
    task: Task
    instruction: str
    sandbox: ProcessSandbox | None
    expect_file: str | None = None
    expect_text: str | None = None
    grade_status: str = "scored"
    grade_error: bool = False
    grade_reason: str | None = None
    agent_timeout_s: float | None = None  # the task's own limit, as a Harbor session reports it
    graded: int = 0
    closed: bool = False

    async def tools(self) -> list[Any]:
        return []

    async def grade(self) -> envs_service.Outcome:
        self.graded += 1
        if self.grade_error:
            raise RuntimeError("grader crashed")
        if self.grade_status != "scored":
            return envs_service.Outcome(status=self.grade_status, reason="verifier_declared")
        path = (self.sandbox.workdir if self.sandbox else Path.cwd()) / (self.expect_file or "")
        solved = path.is_file() and path.read_text().strip() == self.expect_text
        return envs_service.Outcome(
            status="scored", rewards={"reward": 1.0 if solved else 0.0}, reason=self.grade_reason
        )

    async def close(self) -> None:
        self.closed = True
        if self.sandbox is not None:
            await self.sandbox.stop()


class FakeService:
    """An env service resolved by ``module:Class``; ``sessions`` records every session it opened."""

    name = "fake"
    sessions: ClassVar[list[FakeSession]] = []
    root: ClassVar[Path | None] = None

    def __init__(
        self,
        *,
        sandbox: bool = True,
        expect_file: str = "answer.txt",
        expect_text: str = "42",
        grade_status: str = "scored",
        grade_error: bool = False,
        grade_reason: str | None = None,
        fail_open: bool = False,
        agent_timeout_s: float | None = None,
    ) -> None:
        self.agent_timeout_s = agent_timeout_s
        self.sandbox = sandbox
        self.expect_file = expect_file
        self.expect_text = expect_text
        self.grade_status = grade_status
        self.grade_error = grade_error
        self.grade_reason = grade_reason
        self.fail_open = fail_open

    async def list_tasks(self, split: str) -> list[str]:
        return [f"{split}-{index}" for index in range(3)]

    async def get_task(self, task_id: str) -> Task:
        return Task(id=task_id, goal=f"Write {self.expect_text} to {self.expect_file}", misc={"service_task": task_id})

    async def open(self, task: Task, *, rollout_id: str) -> FakeSession:
        if self.fail_open:
            raise RuntimeError("sandbox start failed")
        assert FakeService.root is not None, "set FakeService.root before opening sessions"
        sandbox = ProcessSandbox(FakeService.root / rollout_id) if self.sandbox else None
        session = FakeSession(
            task=task,
            instruction=task.goal or "",
            sandbox=sandbox,
            expect_file=self.expect_file,
            expect_text=self.expect_text,
            grade_status=self.grade_status,
            grade_error=self.grade_error,
            grade_reason=self.grade_reason,
            agent_timeout_s=self.agent_timeout_s,
        )
        FakeService.sessions.append(session)
        return session


@dataclass
class FakeAgent:
    agent_id: str
    api_key: str
    rollout_id: str
    trajectory_id: str


@dataclass
class FakeGateway:
    """The harness rollout's gateway port over a ``FakeChatServer``; records what it was handed."""

    server: FakeChatServer
    agents: list[FakeAgent] = field(default_factory=list)
    collected: list[tuple[Task, Any]] = field(default_factory=list)
    started: bool = False
    stopped: bool = False

    async def start(self) -> str:
        self.started = True
        return await self.server.start()

    def open_agent(self, *, rollout_id: str, trajectory_id: str, sampling: Any = None) -> FakeAgent:
        agent = FakeAgent(
            agent_id=f"agent-{uuid.uuid4().hex[:8]}",
            api_key=self.server.api_key,
            rollout_id=rollout_id,
            trajectory_id=trajectory_id,
        )
        self.agents.append(agent)
        return agent

    def serve_tools(self, agent: FakeAgent, session: Any) -> str | None:
        return None  # no tool plane: harnesses run without MCP

    def collect(self, root: FakeAgent, *, task: Task, outcome: envs_service.Outcome) -> Any:
        from platoon.trajectory import Trajectory, TrajectoryCollection, TrajectoryStep

        self.collected.append((task, outcome))
        trajectory = Trajectory(id=root.trajectory_id, task=task, reward=outcome.reward or 0.0)
        trajectory.steps = [
            TrajectoryStep(misc={"action_misc": {"completion_id": call_id}}) for call_id in self.server.call_ids
        ]
        return TrajectoryCollection(id=root.rollout_id, trajectories={trajectory.id: trajectory})

    async def stop(self) -> None:
        self.stopped = True
        await self.server.stop()
