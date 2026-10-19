"""Harbor sessions: one task sandbox per rollout, graded by the task's ``tests/test.sh``.

Agents that share a sandbox take turns on it (:mod:`platoon.envs.gate`): their tool calls run one at a time
unless the service lists the tool in ``concurrent_tools``, and grading and copy snapshots run alone.  A call not
admitted within ``queue_timeout_s`` is not run; an admitted call runs to the end even when its caller gives up,
and the last close stops the sandbox only after the calls in flight end (at most ``close_timeout_s``).

A task with a separate verifier is graded in a sandbox of its own: grading kills every process the agents left in
their sandbox (``main``), collects the task's artifacts from it as directories and regular files only
(:mod:`platoon.envs.harbor.artifacts`), starts the verifier's sandbox, hands it the artifacts and runs the tests
there, and stops it on every path.  The agents never touch the grader's files, and no symlink of theirs reaches
it.  When nothing else holds ``main`` (no shared fork, no call in flight), grading releases it as soon as the
artifacts are collected, as Harbor stops it before verifying: the session then takes no more tool calls or forks,
and grading it again reuses the collected artifacts.  A verifier that runs as root (``verifier_user: honor``) gets a
sandbox that was checked to give it root and other users (:mod:`platoon.envs.harbor.users`); its commands run with
Docker's umask, and its artifacts are readable by every user, since its tests may run the agent's code as another
user, as they would in Docker.

Grading takes the sandbox the tests run in away from the agents.  In the agents' own sandbox (Harbor's ``shared``
verifier mode) it first kills every process they left running (``Sandbox.kill_processes``) and makes ``/tests`` and
``/logs/verifier`` afresh.  In either sandbox it kills again what the tests left running (the tests may run the
agent's code) before it reads their output, file by file and never through a symlink
(:mod:`platoon.envs.harbor.verifier`).

The agents' workdir is where the tests run (``shared``) and where the artifacts are (``separate``), so an attempt
whose agents removed or moved it, left something else in its place or locked it scores 0 (``workdir_missing``) in
either mode, without the tests: no command starts in a working directory that cannot be entered (docker and
hpc-sandbox exit 126, the ``process`` provider raises).  Should the tests' command in the agents' sandbox exit 126,
grading checks the workdir again, with a command run in ``/``, before it reads any reward file: a test may also exit
126 by itself, and with the workdir in place its reward counts.  Where the sandbox says that the tests' command never
entered its working directory (hpc-sandbox's line, :func:`platoon.sandbox.base.cwd_error`), the outcome records it
(``verifier_cwd_error``), in a separate verifier's sandbox too.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import time
import uuid
from collections.abc import Awaitable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from platoon.envs.gate import DEFAULT_QUEUE_TIMEOUT_S, EnvBusy, GateClosed, ToolGate, caller_or
from platoon.envs.harbor.artifacts import ArtifactError, CollectedArtifact, collect_artifacts, materialize_artifacts
from platoon.envs.harbor.task import HarborTask, McpServer, resolve_env
from platoon.envs.harbor.tools import TOOL_SPECS, SandboxTools
from platoon.envs.harbor.users import DOCKER_UMASK
from platoon.envs.harbor.verifier import TEST_STDOUT, collect, output_tail, suspect_outcome, write_local
from platoon.envs.service import EnvSetupError, ForkMode, Outcome, ToolResult, ToolSpec
from platoon.sandbox.base import CWD_ERROR_EXIT_CODE, ExecResult, Sandbox, SandboxSpec, cwd_error
from platoon.task import Task
from platoon.utils import scratch
from platoon.utils.stopping import let_finish

if TYPE_CHECKING:
    from platoon.envs.harbor.service import HarborEnvironmentService

logger = logging.getLogger(__name__)

TESTS_DIR = "/tests"
SOLUTION_DIR = "/solution"
VERIFIER_DIR = "/logs/verifier"
LOG_DIRS = ("/logs/agent", VERIFIER_DIR, "/logs/artifacts")
DEFAULT_CLOSE_TIMEOUT_S = 30.0
# Makes /tests and /logs/verifier afresh, whatever the agents left there or in their place (a symlink to elsewhere
# included: rm removes a symlink, not what it points to), after removing a /logs that is not a directory.  A
# directory an agent took the permissions from (chmod 000), which rm could not empty, gets them back first; chmod
# never follows a symlink there (the top directories are checked, and chmod -R skips the links it finds).
_RESET_SCRIPT = (
    "if [ -L /logs ] || { [ -e /logs ] && [ ! -d /logs ]; }; then rm -f /logs; fi; "
    "[ -d /logs ] && chmod u+rwx /logs 2>/dev/null; "
    f"for dir in {TESTS_DIR} {VERIFIER_DIR}; do "
    '[ -d "$dir" ] && [ ! -L "$dir" ] && chmod -R u+rwX "$dir" 2>/dev/null; done; '
    f"rm -rf {TESTS_DIR} {VERIFIER_DIR} && mkdir -p {TESTS_DIR} {VERIFIER_DIR}"
)
# Prints "ok" when the workdir ("$1") is a directory a command can start in (as hpc-sandbox, docker and the process
# provider enter it: with cd, or chdir), else what is there instead: "missing" (removed or moved, a symlink to
# nothing included), "not_a_directory", or "not_enterable" (its permissions were taken).
_WORKDIR_SCRIPT = (
    'if cd -- "$1" >/dev/null 2>&1; then echo ok; elif [ ! -e "$1" ]; then echo missing; '
    'elif [ ! -d "$1" ]; then echo not_a_directory; else echo not_enterable; fi'
)
_WORKDIR_STATES = frozenset({"ok", "missing", "not_a_directory", "not_enterable"})
_WORKDIR_TIMEOUT_S = 60.0
_MAX_SKIPPED_REPORTED = 20  # of the artifact paths left out, the ones an outcome names

_BACKGROUND: set[asyncio.Task[Any]] = set()  # teardowns whose callers stopped waiting
_UNKILLABLE: set[type] = set()  # sandbox types without kill_processes, warned about once


class SandboxLease:
    """A sandbox shared by a session and its shared forks; the last release stops it.

    ``gate`` orders the operations on the sandbox (:mod:`platoon.envs.gate`); ``inflight`` holds the tool calls
    still running, which the last release waits for (at most ``close_timeout_s``) before it stops the sandbox.
    """

    def __init__(
        self,
        sandbox: Sandbox,
        spec: SandboxSpec,
        *,
        queue_timeout_s: float | None = DEFAULT_QUEUE_TIMEOUT_S,
        close_timeout_s: float = DEFAULT_CLOSE_TIMEOUT_S,
    ) -> None:
        self.sandbox = sandbox
        self.spec = spec
        self.gate = ToolGate(queue_timeout_s=queue_timeout_s)
        self.close_timeout_s = close_timeout_s
        self.inflight: set[asyncio.Task[Any]] = set()
        self.closed = False  # the last holder released it: nothing new starts
        self._holders = 1
        self._closing: asyncio.Task[Any] | None = None

    @property
    def holders(self) -> int:
        """The sessions holding the sandbox: a session and its shared forks."""

        return self._holders

    def acquire(self) -> SandboxLease:
        self._holders += 1
        return self

    async def release(self) -> None:
        """Drop a holder.  The last one closes the lease at once and stops the sandbox in a background task the
        caller waits for, so a caller that stops waiting (a close under a timeout) does not leak the sandbox."""

        if self._holders > 0:
            self._holders -= 1
        if self._holders > 0:
            return
        if self._closing is None:
            self.closed = True
            self._closing = _in_background(self._teardown())
        await asyncio.shield(self._closing)

    async def _teardown(self) -> None:
        if self.inflight:
            _, running = await asyncio.wait(set(self.inflight), timeout=self.close_timeout_s)
            if running:
                logger.warning(
                    "Stopping sandbox %s with %d call(s) still running after %.0fs",
                    self.sandbox.id,
                    len(running),
                    self.close_timeout_s,
                )
        self.gate.close()
        await self.sandbox.stop()


class HarborSession:
    """A rollout's view of a Harbor task: instruction, task sandbox, generic tools, grading and forks.

    ``mcp_servers`` lists the task's own MCP servers for harnesses that can attach them.  ``network_isolated``
    says the sandbox has no network: the task's ``network_mode`` is ``no-network`` and the provider enforces it
    (its ``network_none`` capability), so a harness placed in the sandbox could not reach the model gateway.
    ``host_loopback`` says the sandbox shares this host's network, so a harness placed in it reaches the gateway on
    loopback: it has a network, and its provider declares the ``host_loopback`` capability.
    """

    # call_tool takes calls from every agent at once and orders them on the sandbox's gate (platoon.envs.gate).
    manages_concurrency = True

    def __init__(
        self,
        service: HarborEnvironmentService,
        task: Task,
        harbor_task: HarborTask,
        lease: SandboxLease,
        *,
        rollout_id: str,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.task = task
        self.instruction = task.goal or harbor_task.instruction
        self.harbor_task = harbor_task
        self.rollout_id = rollout_id
        self.sandbox: Sandbox | None = lease.sandbox
        self.mcp_servers: tuple[McpServer, ...] = harbor_task.mcp_servers
        capabilities = service.provider.capabilities
        self.network_isolated = lease.spec.network == "none" and "network_none" in capabilities
        self.host_loopback = not self.network_isolated and "host_loopback" in capabilities
        self._service = service
        self._lease = lease
        self._tools = SandboxTools(lease.sandbox, workdir=lease.spec.workdir, command_timeout_s=service.tool_timeout_s)
        # Separate verifier: the artifacts kept once main was released after collecting them (their directory, the
        # copies, and what was left out of them), and that release.
        self._kept_artifacts: tuple[Path, list[CollectedArtifact], dict[str, str]] | None = None
        self._main_release: asyncio.Task[Any] | None = None

    @property
    def routes(self) -> dict[str, str]:
        """The task sandbox's routes: upstream name -> the URL a process in it calls (``Sandbox.routes``)."""

        return dict(getattr(self._lease.sandbox, "routes", None) or {})

    @property
    def agent_timeout_s(self) -> float | None:
        """The task's wall clock for its agent: ``[agent].timeout_sec`` times the service's
        ``agent_timeout_multiplier`` (None when the task sets none).  Harness rollouts cap the harness with it."""

        if self.harbor_task.agent_timeout_s is None:
            return None
        return self.harbor_task.agent_timeout_s * self._service.agent_timeout_multiplier

    @property
    def _umask(self) -> str:
        """What the verifier's shell commands start with: a root verifier's set Docker's umask, so that what its tests
        write as root is readable by the users they switch to (:mod:`platoon.envs.harbor.users`)."""

        return f"umask {DOCKER_UMASK}; " if self.harbor_task.verifier_root else ""

    async def tools(self) -> list[ToolSpec]:
        return list(TOOL_SPECS)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run a tool in its turn on the sandbox: refused unrun when the session is closed or the call is not
        admitted within ``queue_timeout_s``; once admitted it runs to the end, even when its caller gives up."""

        lease = self._lease
        if self.sandbox is None or lease.closed:
            return _closed()
        call_class = "concurrent" if name in self._service.concurrent_tools else "serial"
        try:
            return await lease.gate.run(
                call_class,
                lambda grant: self._run_tool(name, arguments),
                view=caller_or(self),  # the calling agent takes its turn, this view's when the plane names none
                inflight=lease.inflight,
            )
        except EnvBusy as busy:
            logger.warning(
                "Harbor tool %s of session %s not run: env busy (waited %.1fs, the limit is %gs)",
                name,
                self.id,
                busy.waited_s,
                busy.timeout_s,
            )
            message = (
                f"Not run: env busy. The call waited {busy.waited_s:.0f} s for other calls on this environment to "
                f"finish (the limit is {busy.timeout_s:g} s); it was not executed. Try again."
            )
            return ToolResult([{"type": "text", "text": message}], is_error=True, metadata={"error_reason": "env_busy"})
        except GateClosed:
            return _closed()

    async def _run_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if self.sandbox is None or self._lease.closed:  # closed while the call waited for its turn
            return _closed()
        return await self._tools.call(name, arguments)

    async def grade(self) -> Outcome:
        """Run the task's tests against the sandbox's current state, in the sandbox itself or, for a task with a
        separate verifier, in a verifier sandbox given the task's artifacts.

        The tests are uploaded only now (so agents never see them) and removed
        afterwards. Infrastructure failures come back as error/invalid outcomes.
        Grading runs alone on the sandbox: it waits for the calls in flight, and
        no agent's call runs while the tests are in the sandbox (or while the
        artifacts are collected). Nothing the agents started runs alongside the
        tests either, nor anything the tests left running while their output is
        read (see the module docstring); a grade that cannot make sure of that
        is an infra ``error`` (``verifier_isolation``). An attempt that left the
        agents' workdir unusable scores 0 (``workdir_missing``) and the tests do
        not run. A grade that cannot start within ``queue_timeout_s`` is an
        infra ``error`` (``env_busy``); a cancelled grade stops the tests and
        removes them before the next call.
        """

        separate = self.harbor_task.verifier_mode == "separate"
        if self.sandbox is None and not (separate and self._kept_artifacts is not None):
            return Outcome.infra("session_closed")
        try:
            if separate:
                outcome = await self._grade_separately()
            else:
                sandbox = self._open_sandbox()
                async with self._lease.gate.hold("exclusive"):
                    outcome = await self._verify(sandbox)
        except EnvBusy as busy:
            logger.warning(
                "Grading Harbor session %s waited %.1fs for the calls in flight; gave up", self.id, busy.waited_s
            )
            return Outcome.infra("env_busy", reason=f"grading waited {busy.waited_s:.1f}s for the calls in flight")
        except GateClosed:
            return Outcome.infra("session_closed")
        except Exception as exc:
            logger.exception("Grading Harbor task %s failed", self.harbor_task.name)
            return Outcome.infra(f"unexpected:{type(exc).__name__}", metadata={"error": str(exc)})
        outcome.metadata["verifier_mode"] = self.harbor_task.verifier_mode
        if self.harbor_task.ignored_users:
            outcome.metadata["ignored_users"] = dict(self.harbor_task.ignored_users)  # ran as the sandbox's uid
        if self.harbor_task.verifier_root:
            outcome.metadata["verifier_user"] = "root"
        return outcome

    async def run_solution(self) -> ExecResult:
        """Run the task's reference ``solution/solve.sh`` (Harbor's oracle), e.g. to validate a dataset."""

        sandbox = self._open_sandbox()
        await sandbox.upload(self.harbor_task.solution_dir, SOLUTION_DIR)
        return await sandbox.exec(f"bash {SOLUTION_DIR}/solve.sh", timeout_s=self.agent_timeout_s)

    async def fork(self, mode: ForkMode = "shared", *, task: Task | None = None) -> HarborSession:
        """``shared`` shares this sandbox, ``copy`` restores a snapshot of it, ``fresh`` opens a new sandbox.

        A session whose ``main`` its separate verifier released forks no more: fork before grading."""

        sandbox = self._open_sandbox()
        task = task or self.task
        if mode == "fresh":
            return await self._service.open(task, rollout_id=self.rollout_id)
        harbor_task = self._service.harbor_task(task)
        if mode == "shared":
            return HarborSession(self._service, task, harbor_task, self._lease.acquire(), rollout_id=self.rollout_id)
        if mode == "copy":
            provider = self._service.provider
            if "snapshot" not in provider.capabilities:
                raise NotImplementedError(f"The {provider.name!r} sandbox provider cannot snapshot for copy forks")
            # The snapshot runs alone on the sandbox, so it never copies a call half-applied.
            snapshot = await self._lease.gate.run(
                "exclusive", lambda grant: sandbox.snapshot(), inflight=self._lease.inflight
            )
            restored = await provider.restore(snapshot, self._lease.spec)
            lease = SandboxLease(
                restored,
                self._lease.spec,
                queue_timeout_s=self._lease.gate.queue_timeout_s,
                close_timeout_s=self._lease.close_timeout_s,
            )
            return HarborSession(self._service, task, harbor_task, lease, rollout_id=self.rollout_id)
        raise ValueError(f"Unknown fork mode {mode!r}")

    async def close(self) -> None:
        kept, self._kept_artifacts = self._kept_artifacts, None
        if kept is not None:
            shutil.rmtree(kept[0], ignore_errors=True)
        if self._main_release is not None:
            await asyncio.shield(self._main_release)  # main stops in the background once its artifacts are out
            return
        if self.sandbox is None:
            return
        self.sandbox = None
        await self._lease.release()

    def _open_sandbox(self) -> Sandbox:
        if self.sandbox is None:
            if self._main_release is not None:
                raise RuntimeError(
                    f"Harbor session {self.id} released its sandbox once the separate verifier collected the "
                    "artifacts; fork or act before grading"
                )
            raise RuntimeError(f"Harbor session {self.id} is closed")
        return self.sandbox

    async def _verify(self, sandbox: Sandbox) -> Outcome:
        """Grade in the agents' own sandbox (``shared`` mode)."""

        try:
            env = resolve_env(self.harbor_task.verifier_env)
        except KeyError as exc:
            return Outcome.infra("verifier_setup", reason=f"host variable {exc} is not set")
        if (unsafe := await self._kill_processes(sandbox, "before the tests")) is not None:
            return unsafe
        # Grading's own commands run in "/", which an agent cannot remove, unlike the workdir, where the tests run.
        if (unusable := await self._check_workdir(sandbox)) is not None:
            return unusable
        reset = await sandbox.exec(["/bin/sh", "-c", _RESET_SCRIPT, "platoon-reset"], cwd="/")
        if reset.exit_code != 0:
            return Outcome.infra("verifier_setup", reason=reset.stderr.strip())
        try:
            try:
                await sandbox.upload(self.harbor_task.tests_dir, TESTS_DIR)
            except OSError as exc:
                return Outcome.infra("verifier_setup", reason=f"cannot upload the tests: {exc}")
            return await self._run_tests(sandbox, env, agents_sandbox=True)
        except Exception:
            # Where a provider raises for a working directory it cannot enter (the process provider's OSError, and
            # docker's upload, whose mkdir runs there), a process the kill missed may have removed the workdir since
            # the check: the upload or the tests' command failed for the agents' doing.  Every other failure stands,
            # also when the sandbox cannot be asked.
            with contextlib.suppress(Exception):
                if (unusable := await self._check_workdir(sandbox)) is not None and unusable.trainable:
                    return unusable
            raise
        finally:
            await self._remove_tests(sandbox)

    async def _grade_separately(self) -> Outcome:
        """Collect the artifacts from main (unless main was released and they were kept), then grade them in a
        verifier sandbox."""

        if self._kept_artifacts is not None:
            return await self._verify_separately(*self._kept_artifacts[1:])
        sandbox = self._open_sandbox()
        into = Path(scratch.mkdtemp("platoon-harbor-artifacts-"))
        skipped: dict[str, str] = {}
        collected: list[CollectedArtifact] = []
        try:
            async with self._lease.gate.hold("exclusive"):
                # Nothing the agents left running may change the artifacts while they are collected from the
                # workdir, and an unusable workdir scores 0 before anything is collected.
                early = await self._kill_processes(sandbox, "before collecting the artifacts")
                if early is None:
                    early = await self._check_workdir(sandbox)
                if early is None:
                    workdir = self._lease.spec.workdir
                    collected = await collect_artifacts(
                        sandbox,
                        self.harbor_task.artifacts,
                        workdir=workdir,
                        into=into,
                        skipped=skipped,
                        max_bytes=self._service.max_artifact_bytes,
                    )
        except ArtifactError as exc:
            shutil.rmtree(into, ignore_errors=True)
            return Outcome.infra("artifact_collect", reason=str(exc))
        except BaseException:
            shutil.rmtree(into, ignore_errors=True)
            raise
        if early is not None:  # unsafe to collect, or nothing to collect from
            shutil.rmtree(into, ignore_errors=True)
            return early
        if skipped:
            logger.info(
                "Harbor task %s: left out of the artifacts, as not directories or regular files: %s",
                self.harbor_task.name,
                skipped,
            )
        if self._lease.holders == 1 and not self._lease.inflight and self.sandbox is not None:
            # Nothing else needs main: let it go while the verifier starts, and keep the artifacts for regrading.
            self._kept_artifacts = (into, collected, skipped)
            self.sandbox = None
            self._main_release = _in_background(self._lease.release())
            return await self._verify_separately(collected, skipped)
        try:
            return await self._verify_separately(collected, skipped)
        finally:
            shutil.rmtree(into, ignore_errors=True)

    async def _verify_separately(self, collected: list[CollectedArtifact], skipped: dict[str, str]) -> Outcome:
        try:
            env = resolve_env(self.harbor_task.verifier_env)
        except KeyError as exc:
            return Outcome.infra("verifier_setup", reason=f"host variable {exc} is not set")
        try:
            verifier, spec = await self._service.open_verifier(self.harbor_task, rollout_id=self.rollout_id)
        except EnvSetupError as exc:
            return Outcome.infra("verifier_start", reason=str(exc), metadata={"setup_failure_kind": exc.failure_kind})
        try:
            bundled = (
                self.harbor_task.verifier_environment is not None
                and self.harbor_task.verifier_environment.bundled_tests
            )
            directories = [VERIFIER_DIR] if bundled else [TESTS_DIR, VERIFIER_DIR]
            reset = f"{self._umask}rm -rf {' '.join(directories)} && mkdir -p {' '.join(directories)}"
            result = await verifier.exec(["/bin/sh", "-c", reset], cwd="/")
            if result.exit_code != 0:
                return Outcome.infra("verifier_setup", reason=result.stderr.strip())
            try:
                await materialize_artifacts(
                    verifier, collected, workdir=spec.workdir, readable=self.harbor_task.verifier_root
                )
            except ArtifactError as exc:
                return Outcome.infra("verifier_setup", reason=str(exc))
            if not bundled:  # a verifier image built for the task holds its own /tests
                try:
                    await verifier.upload(self.harbor_task.tests_dir, TESTS_DIR)
                except OSError as exc:
                    return Outcome.infra("verifier_setup", reason=f"cannot upload the tests: {exc}")
            outcome = await self._run_tests(verifier, env, agents_sandbox=False)
            outcome.metadata["artifacts"] = sorted(artifact.target(spec.workdir) for artifact in collected)
            if skipped:
                outcome.metadata["artifacts_skipped"] = dict(sorted(skipped.items())[:_MAX_SKIPPED_REPORTED])
                outcome.metadata["artifacts_skipped_count"] = len(skipped)
            return outcome
        finally:
            await asyncio.shield(_in_background(verifier.stop()))

    async def _run_tests(self, sandbox: Sandbox, env: dict[str, str], *, agents_sandbox: bool) -> Outcome:
        """Run the tests in ``sandbox``, the agents' own (``shared``) or a separate verifier's, and read their
        verdict."""

        timeout_s = self.harbor_task.verifier_timeout_s * self._service.verifier_timeout_multiplier
        started = time.monotonic()
        result = await sandbox.exec(
            f"{self._umask}bash {TESTS_DIR}/test.sh > {VERIFIER_DIR}/{TEST_STDOUT} 2>&1", timeout_s=timeout_s, env=env
        )
        metadata: dict[str, Any] = {
            "verifier_exit_code": result.exit_code,
            "verifier_seconds": round(time.monotonic() - started, 3),
        }
        if (line := cwd_error(result)) is not None:  # the sandbox says the tests' command never entered the workdir
            metadata["verifier_cwd_error"] = line
        # The tests may have run the agent's code: nothing left of it may change their output while it is read.
        if (unsafe := await self._kill_processes(sandbox, "after the tests")) is not None:
            unsafe.metadata.update(metadata)
            return unsafe
        # A command that could not start in its working directory exits 126, and so may a command by itself (whose
        # output, the tests' here, could say anything): a check run in "/" confirms it.  When a process the kill missed
        # removed the agents' workdir since it was checked, the tests never ran and what /logs/verifier holds is not
        # their output.  A separate verifier's sandbox never held the agents' processes, and its workdir is the
        # verifier's own: its tests' exit status stands, and the reward files say what happened.
        if agents_sandbox and result.exit_code == CWD_ERROR_EXIT_CODE and not result.timed_out:
            if (unusable := await self._check_workdir(sandbox)) is not None:
                unusable.metadata.update(metadata)
                return unusable
        try:
            files = await collect(sandbox, VERIFIER_DIR)
        except RuntimeError as exc:
            return Outcome.infra("verifier_output", reason=str(exc), metadata=metadata)
        if result.timed_out:
            outcome = Outcome.infra("verifier_timeout", status="invalid", metadata={"timeout_s": timeout_s})
        elif (suspect := suspect_outcome(files, primary=self._service.reward.names[0])) is not None:
            outcome = suspect
        else:
            with scratch.temporary_directory() as local_dir:
                # A directory named verifier, as reward-contract hooks expect (platoon.envs.harbor.grading).
                local = write_local(files, Path(local_dir) / "verifier")
                outcome = self._service.verdict(local, self.harbor_task)
        if (output := output_tail(files)) is not None:
            metadata["test_output"] = output
        outcome.metadata.update(metadata)
        return outcome

    async def _kill_processes(self, sandbox: Sandbox, when: str) -> Outcome | None:
        """Kill every process left in the sandbox, so that nothing the agents started races the tests or the reading
        of their output; an infra outcome when that fails, since the grade could not be trusted."""

        kill = getattr(sandbox, "kill_processes", None)
        if kill is None:
            if type(sandbox) not in _UNKILLABLE:
                _UNKILLABLE.add(type(sandbox))
                logger.warning(
                    "%s has no kill_processes: Harbor grading cannot stop what agents leave running in its sandboxes",
                    type(sandbox).__name__,
                )
            return None
        try:
            await kill()
        except Exception as exc:
            logger.warning("Could not kill the processes left in sandbox %s %s: %s", sandbox.id, when, exc)
            return Outcome.infra(
                "verifier_isolation", reason=f"could not kill the processes left in the sandbox {when}: {exc}"
            )
        return None

    async def _check_workdir(self, sandbox: Sandbox) -> Outcome | None:
        """None when the agents' workdir is a directory a command can start in.  Else the agents made it unusable
        (grading never touches it): they removed or moved it, left something else in its place, or took its
        permissions, so the attempt scores 0 under the reward contract's first name (``workdir_missing``), an outcome
        whose ``metadata`` names the workdir and what is there instead.  An infra ``verifier_setup`` outcome when the
        sandbox's shell cannot tell, and the sandbox's own errors propagate: neither says anything about the
        policy."""

        workdir = self._lease.spec.workdir
        result = await sandbox.exec(
            ["/bin/sh", "-c", _WORKDIR_SCRIPT, "platoon-workdir", workdir], cwd="/", timeout_s=_WORKDIR_TIMEOUT_S
        )
        state = result.stdout.strip()
        if result.timed_out or result.exit_code != 0 or state not in _WORKDIR_STATES:
            detail = result.stderr.strip() or ("timed out" if result.timed_out else f"exit {result.exit_code}")
            return Outcome.infra("verifier_setup", reason=f"cannot check the workdir {workdir}: {detail}")
        if state == "ok":
            return None
        logger.info(
            "Harbor task %s: the workdir %s is %s, so the attempt scores 0 without its tests (workdir_missing)",
            self.harbor_task.name,
            workdir,
            state.replace("_", " "),
        )
        return self._service.reward.zero(
            reason="workdir_missing", metadata={"workdir": workdir, "workdir_state": state}
        )

    async def _remove_tests(self, sandbox: Sandbox) -> None:
        """Later agents sharing this sandbox must not see the tests, also after a failed or cancelled grade."""

        try:
            removed = (await sandbox.exec(["rm", "-rf", TESTS_DIR], cwd="/")).exit_code == 0
        except Exception:
            removed = False
        if not removed:
            logger.warning("Could not remove %s from sandbox %s after grading", TESTS_DIR, sandbox.id)


def _closed() -> ToolResult:
    return ToolResult.text("The session is closed", is_error=True)


def _in_background(coroutine: Awaitable[Any]) -> asyncio.Task[Any]:
    """Run a teardown as a task that lives on when its caller stops waiting (asyncio keeps only weak references).

    A stopping rollout worker lets it finish within its deadline (:func:`platoon.utils.stopping.let_finish`).
    """

    task = let_finish(asyncio.ensure_future(coroutine))
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)
    task.add_done_callback(_log_failure)
    return task


def _log_failure(task: asyncio.Task[Any]) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.warning("Stopping a Harbor sandbox failed", exc_info=task.exception())
