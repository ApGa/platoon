"""Run one harness for one gateway agent.

Two placements:

``task_sandbox``
    The harness runs inside ``session.sandbox`` through ``Sandbox.exec``, in the
    sandbox's own working directory, so its native tools act on task state.
    ``{workdir}`` renders as that directory as the sandbox reports it (``pwd``).
    Its state directory lives under
    :data:`SANDBOX_STATE_ROOT` in the sandbox, and every process it starts
    carries the :data:`AGENT_MARKER_ENV` marker, which is how the runner finds
    and kills the whole tree inside the sandbox on timeout, cancellation, or
    after the harness exits. The install commands and the harness run in the
    working directory; the runner's own commands (making directories for its
    files, the toolchain check, reading the final message file and the kill)
    run in ``/``, so they still work once the agent has removed or moved the
    working directory, where no command starts (docker and hpc-sandbox exit
    126, the ``process`` provider raises).
``local``
    The harness runs as a subprocess on the rollout host, in its own process
    group, inside a per-agent state directory that is removed afterwards. It
    gets a small allowlisted environment, never the worker's secrets. It gets
    SIGTERM, then SIGKILL after ``kill_grace_s``, when it ends, times out or is
    cancelled; that cleanup runs to its end even when the caller is cancelled
    again, and its group is registered with :mod:`platoon.utils.stopping` until
    then, so a stopping rollout worker ends it too.

In both, the gateway is reached on loopback, so ``NO_PROXY`` covers it. In a
task sandbox that has a route to the gateway (``sandbox.routes["gateway"]``,
:mod:`platoon.sandbox.routes`), ``{base_url}`` and ``{mcp_url}`` render through
the route; before each launch the runner has the sandbox start its route again if
the policy killed it (``sandbox.ensure_routes()``). A task sandbox without a
network (``session.network_isolated``) and without that route cannot reach the
gateway, so the runner refuses to place a harness there (``HarnessSetupError``)
rather than start one that would crash; so does one that does not share this
host's network (``session.host_loopback`` false, as on another node or in a
cloud) and has no route. When the agent serves its env tools over
MCP, ``mcp_url`` is the endpoint a spec's ``[mcp]`` table renders as
``{mcp_url}``. ``toolchain`` is where the launch finds the spec's toolchain
(``{toolchain}``), resolved by the rollout for a named one
(:mod:`platoon.harness.toolchains`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import posixpath
import re
import shutil
import signal
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from platoon.harness.spec import (
    HarnessSpec,
    HarnessSpecError,
    InstallSpec,
    RenderContext,
    RenderedLaunch,
    resolve_toolchain,
)
from platoon.sandbox.routes import GATEWAY
from platoon.utils import scratch
from platoon.utils.stopping import let_finish, register_process_group, unregister_process_group

if TYPE_CHECKING:
    from platoon.envs.service import EnvSession
    from platoon.sandbox.base import ExecResult, Sandbox

logger = logging.getLogger(__name__)

Placement = Literal["task_sandbox", "local"]
PlacementMode = Literal["auto", "task_sandbox", "local"]
BuiltinsMode = Literal["default", "keep", "disable"]

DEFAULT_MODEL = "policy"
SANDBOX_STATE_ROOT = "/tmp/platoon-harness"
AGENT_MARKER_ENV = "PLATOON_HARNESS_AGENT"
LOOPBACK_HOSTS = "127.0.0.1,localhost,::1"
LOCAL_ENV_PASSTHROUGH = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
)
OUTPUT_TAIL_CHARS = 4000

# Signals every process whose environment carries the marker line ("$1"): TERM,
# then KILL after "$2" seconds for whatever is left. GNU grep scans all of /proc
# in one process; other greps fall back to one check per process.
_KILL_MARKED_SCRIPT = r"""
marker="$1"; grace="$2"
marked() {
  if printf 'x\0' | grep -qz x 2>/dev/null; then
    grep -lsxzF -- "$marker" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3
  else
    for dir in /proc/[0-9]*; do
      tr '\0' '\n' < "$dir/environ" 2>/dev/null | grep -qxF -- "$marker" && echo "${dir#/proc/}"
    done
  fi
}
targets=$(marked)
[ -z "$targets" ] && exit 0
kill -s TERM $targets 2>/dev/null
sleep "$grace"
targets=$(marked)
[ -n "$targets" ] && kill -s KILL $targets 2>/dev/null
exit 0
"""


class AgentEndpoint(Protocol):
    """What the runner needs from a gateway agent: its id and its key."""

    @property
    def agent_id(self) -> str: ...

    @property
    def api_key(self) -> str: ...


class HarnessSetupError(RuntimeError):
    """The harness could not be installed or prepared: an infrastructure failure, not a policy one."""


@dataclass(frozen=True)
class HarnessResult:
    exit_code: int | None
    final_message: str | None
    timed_out: bool
    logs_path: str | None


def resolve_placement(requested: PlacementMode, session: EnvSession) -> Placement:
    """``auto`` runs in the task sandbox when the session offers one, else locally."""

    if requested == "auto":
        return "task_sandbox" if session.sandbox is not None else "local"
    if requested == "task_sandbox" and session.sandbox is None:
        raise HarnessSetupError("placement task_sandbox needs an env session with a sandbox")
    return requested


class HarnessRunner:
    """Installs (once per host), renders and launches harnesses, and reads their final message."""

    def __init__(
        self,
        *,
        logs_dir: str | os.PathLike[str],
        state_root: str | os.PathLike[str] | None = None,
        sandbox_state_root: str = SANDBOX_STATE_ROOT,
        kill_grace_s: float = 5.0,
    ) -> None:
        self._logs_dir = Path(logs_dir)
        self._sandbox_state_root = sandbox_state_root
        # In node-local scratch, which a node's users may share: a directory of this user's own.
        default = scratch.node_scratch() / f"platoon-harness-{os.getuid()}"
        self._state_root = Path(state_root) if state_root is not None else default
        self._kill_grace_s = kill_grace_s
        # (install spec, host, toolchain): what has been installed where, and the lock each install takes.
        self._installed: set[tuple[InstallSpec, str, str | None]] = set()
        self._install_locks: dict[tuple[InstallSpec, str, str | None], asyncio.Lock] = {}

    async def run(
        self,
        spec: HarnessSpec,
        *,
        session: EnvSession,
        agent: AgentEndpoint,
        base_url: str,
        placement: Placement,
        model: str = DEFAULT_MODEL,
        builtins: BuiltinsMode = "default",
        timeout_s: float | None = None,
        max_steps: int | None = None,
        mcp_url: str | None = None,
        toolchain: str | None = None,
        toolchain_root: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> HarnessResult:
        """Run ``spec`` to completion, a timeout, or cancellation; the harness never outlives this call.

        ``mcp_url`` is the agent's MCP endpoint when the gateway serves its env tools; specs whose ``[mcp]`` is
        enabled then get it as ``{mcp_url}``.  ``toolchain`` is where this launch finds the spec's toolchain (a
        rollout stages a named one and passes where it is); without it, ``[install] toolchain`` is an absolute path
        or a name under ``toolchain_root`` (default: the toolchain root, ``toolchains`` in the site profile's cache),
        and ``{toolchain}`` renders as that directory.  ``options`` are the entry's ``harness_options``, which the
        spec's ``[options]`` check and its launch renders (``{options.NAME}``); options it does not take are a
        ``HarnessSpecError``.
        """

        agent_slug = re.sub(r"[^A-Za-z0-9._-]", "-", agent.agent_id)
        log_path = self._logs_dir / f"{agent_slug}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        toolchain = toolchain if toolchain is not None else _toolchain_dir(spec, toolchain_root)
        if placement == "task_sandbox":
            if session.sandbox is None:
                raise HarnessSetupError("placement task_sandbox needs an env session with a sandbox")
            base_url, mcp_url = await _sandbox_urls(spec, session, base_url, mcp_url)
            host: _Host = await _SandboxHost.open(
                session.sandbox, posixpath.join(self._sandbox_state_root, agent_slug), agent_slug, self._kill_grace_s
            )
        else:
            host = _LocalHost.open(self._state_root / agent_slug, self._kill_grace_s)
        try:
            await self._install(spec, host, log_path, toolchain)
            rendered = spec.render(
                RenderContext(
                    instruction=session.instruction,
                    base_url=base_url,
                    api_key=agent.api_key,
                    model=model,
                    workdir=host.workdir,
                    state_dir=host.state_dir,
                    toolchain=toolchain,
                    max_steps=max_steps,
                    mcp_url=mcp_url,
                    options=options,
                ),
                disable_builtins=_disable_builtins(spec, builtins, placement),
            )
            for path, content in rendered.files.items():
                await host.write(host.resolve(path), content)
            for path, source in rendered.assets.items():
                await host.copy(source, host.resolve(path))
            no_proxy = ",".join(filter(None, (os.environ.get("NO_PROXY"), LOOPBACK_HOSTS)))
            env = {"NO_PROXY": no_proxy, "no_proxy": no_proxy, **rendered.env}
            exit_code, output, timed_out = await host.launch(
                rendered.argv, env=env, timeout_s=_earliest(spec.timeout_s, timeout_s), log_path=log_path
            )
            final_message = await _final_message(spec, rendered, host, output)
        finally:
            await host.close()
        return HarnessResult(
            exit_code=exit_code, final_message=final_message, timed_out=timed_out, logs_path=str(log_path)
        )

    async def _install(self, spec: HarnessSpec, host: _Host, log_path: Path, toolchain: str | None) -> None:
        key = (spec.install, host.key, toolchain)
        async with self._install_locks.setdefault(key, asyncio.Lock()):
            if key in self._installed:
                return
            for command in spec.install.commands:
                exit_code, output = await host.shell(command, timeout_s=spec.install.timeout_s)
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"[platoon] install {spec.ref}: exit {exit_code}\n{output}\n")
                if exit_code != 0:
                    raise HarnessSetupError(
                        f"installing {spec.ref} failed with exit code {exit_code}: {output[-OUTPUT_TAIL_CHARS:]}"
                    )
            if toolchain is not None and not await host.exists(toolchain):
                raise HarnessSetupError(f"toolchain {toolchain} for {spec.ref} is not present")
            self._installed.add(key)


async def _sandbox_urls(
    spec: HarnessSpec, session: EnvSession, base_url: str, mcp_url: str | None
) -> tuple[str, str | None]:
    """``{base_url}`` and ``{mcp_url}`` for a harness in the session's sandbox: through its gateway route when it
    has one (started again first if it is gone), else as given, which a sandbox without a network cannot reach."""

    sandbox = session.sandbox
    ensure = getattr(sandbox, "ensure_routes", None)
    try:
        routes = await ensure() if callable(ensure) else dict(getattr(sandbox, "routes", None) or {})
    except Exception as exc:
        raise HarnessSetupError(f"the task sandbox's route to the model gateway is down: {exc}") from exc
    route = routes.get(GATEWAY)
    if route is None:
        if getattr(session, "network_isolated", False):
            raise HarnessSetupError(
                f"harness {spec.ref} would run in the task sandbox, which has no network (a no-network task) and no "
                f"route to the model gateway, so it could not reach the gateway at {base_url}; run it with "
                "placement: local, or give the task a network, or use an environment service that gives its "
                "sandboxes a gateway route"
            )
        if not getattr(session, "host_loopback", True):
            raise HarnessSetupError(
                f"harness {spec.ref} would run in the task sandbox, which does not share this host's network (its "
                "provider does not declare host_loopback, or it is the main of a stack with a network of its own) and "
                f"has no route to the model gateway, so it could not reach the gateway at {base_url}; run it with "
                "placement: local, or use a sandbox provider that runs sandboxes on this host or gives them a gateway "
                "route"
            )
        return base_url, mcp_url
    route, origin = route.rstrip("/"), base_url.rstrip("/")
    if mcp_url is not None and mcp_url.startswith(origin):
        mcp_url = route + mcp_url[len(origin) :]
    return route, mcp_url


def _toolchain_dir(spec: HarnessSpec, root: str | None) -> str | None:
    """Where the spec's toolchain is; a named toolchain without a root is a setup error, not a spec error."""

    if spec.install.toolchain is None:
        return None
    try:
        return resolve_toolchain(spec.install.toolchain, root)
    except HarnessSpecError as exc:
        raise HarnessSetupError(f"{spec.ref}: {exc}") from None


def _disable_builtins(spec: HarnessSpec, mode: BuiltinsMode, placement: Placement) -> bool:
    """``default`` keeps built-ins in a task sandbox and disables them locally where the harness allows."""

    if mode != "default":
        return mode == "disable"
    if placement == "task_sandbox":
        return False
    if spec.builtins.disable is None:
        logger.warning(
            "harness %s cannot disable its built-in tools %s; they act on the local state directory",
            spec.ref,
            list(spec.builtins.tools),
        )
        return False
    return True


def _earliest(*timeouts: float | None) -> float | None:
    bounded = [timeout for timeout in timeouts if timeout is not None]
    return min(bounded) if bounded else None


async def _final_message(spec: HarnessSpec, rendered: RenderedLaunch, host: _Host, output: str) -> str | None:
    final = spec.final_message
    if final is None:
        return None
    if final.stdout_regex is not None:
        matches = list(re.finditer(final.stdout_regex, output, re.MULTILINE))
        if not matches:
            return None
        last = matches[-1]
        return last.group(1) if last.re.groups else last.group(0)
    if rendered.final_message_file is None:
        return None
    content = await host.read(host.resolve(rendered.final_message_file))
    if content is None or final.json_key is None:
        return content
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    for key in final.json_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value if isinstance(value, str) else json.dumps(value)


class _Host(Protocol):
    key: str
    workdir: str
    state_dir: str

    def resolve(self, path: str) -> str: ...
    async def shell(self, command: str, *, timeout_s: float) -> tuple[int | None, str]: ...
    async def write(self, path: str, content: str) -> None: ...
    async def copy(self, source: Path, path: str) -> None: ...
    async def exists(self, path: str) -> bool: ...
    async def read(self, path: str) -> str | None: ...
    async def launch(
        self, argv: tuple[str, ...], *, env: dict[str, str], timeout_s: float | None, log_path: Path
    ) -> tuple[int | None, str, bool]: ...
    async def close(self) -> None: ...


class _LocalHost:
    key = "local"

    def __init__(self, state_dir: Path, kill_grace_s: float) -> None:
        self.state_dir = str(state_dir)
        self.workdir = str(state_dir / "work")
        self._kill_grace_s = kill_grace_s

    @classmethod
    def open(cls, state_dir: Path, kill_grace_s: float) -> _LocalHost:
        (state_dir / "work").mkdir(parents=True, exist_ok=True)
        state_dir.chmod(0o700)
        return cls(state_dir, kill_grace_s)

    def resolve(self, path: str) -> str:
        return str(Path(self.state_dir, path))

    def _env(self, extra: dict[str, str]) -> dict[str, str]:
        env = {name: os.environ[name] for name in LOCAL_ENV_PASSTHROUGH if name in os.environ}
        return {**env, **extra}

    async def shell(self, command: str, *, timeout_s: float) -> tuple[int | None, str]:
        with scratch.temporary_file() as output:
            exit_code, _ = await self._spawn(
                ["/bin/sh", "-c", command], cwd=self.state_dir, env=self._env({}), stdout=output, timeout_s=timeout_s
            )
            output.seek(0)
            return exit_code, output.read().decode("utf-8", "replace")

    async def write(self, path: str, content: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    async def copy(self, source: Path, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)

    async def exists(self, path: str) -> bool:
        return Path(path).exists()

    async def read(self, path: str) -> str | None:
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    async def launch(
        self, argv: tuple[str, ...], *, env: dict[str, str], timeout_s: float | None, log_path: Path
    ) -> tuple[int | None, str, bool]:
        start = log_path.stat().st_size if log_path.exists() else 0
        with log_path.open("ab") as log:
            exit_code, timed_out = await self._spawn(
                list(argv), cwd=self.workdir, env=self._env(env), stdout=log, timeout_s=timeout_s
            )
        with log_path.open("rb") as log:
            log.seek(start)
            output = log.read().decode("utf-8", "replace")
        return exit_code, output, timed_out

    async def _spawn(
        self, argv: list[str], *, cwd: str, env: dict[str, str], stdout, timeout_s: float | None
    ) -> tuple[int | None, bool]:
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=stdout,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise HarnessSetupError(f"cannot start {argv[0]!r}: {exc}") from exc
        register_process_group(process.pid)  # its own session: out of reach of the worker's process-group kill
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout_s)
        except TimeoutError:
            timed_out = True
        finally:
            await _kill_process_group(process, self._kill_grace_s)
        return process.returncode, timed_out

    async def close(self) -> None:
        shutil.rmtree(self.state_dir, ignore_errors=True)


async def _kill_process_group(process: asyncio.subprocess.Process, grace_s: float) -> None:
    """TERM then KILL a live harness's group; KILL leftovers of one that already exited.

    A cleanup that runs to its end (:func:`platoon.utils.stopping.let_finish`): cancelling the caller, as a stopping
    rollout worker does again and again, neither cuts the grace short nor skips the KILL.
    """

    await asyncio.shield(let_finish(asyncio.ensure_future(_end_process_group(process, grace_s))))


async def _end_process_group(process: asyncio.subprocess.Process, grace_s: float) -> None:
    try:
        if process.returncode is None:
            _signal_group(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), grace_s)
            except TimeoutError:
                pass
    finally:  # also when this task is cancelled itself, as an event loop that ends cancels its tasks
        _signal_group(process.pid, signal.SIGKILL)
        unregister_process_group(process.pid)
    await process.wait()


def _signal_group(pgid: int, signum: signal.Signals) -> None:
    try:
        os.killpg(pgid, signum)
    except (ProcessLookupError, PermissionError):
        pass


class _SandboxHost:
    def __init__(self, sandbox: Sandbox, workdir: str, state_dir: str, agent_slug: str, kill_grace_s: float) -> None:
        self.key = f"sandbox:{sandbox.id}"
        self.workdir = workdir
        self.state_dir = state_dir
        self._sandbox = sandbox
        # The agent's id and a nonce: the kill finds this launch's processes and no others, also where the sandbox's
        # processes are this host's (the process provider), next to another launch for an agent of the same id.
        self._marker = f"{agent_slug}.{uuid.uuid4().hex[:12]}"
        self._kill_grace_s = kill_grace_s

    @classmethod
    async def open(cls, sandbox: Sandbox, state_dir: str, agent_slug: str, kill_grace_s: float) -> _SandboxHost:
        pwd = await sandbox.exec(["pwd"], timeout_s=60)
        if pwd.exit_code != 0:
            raise HarnessSetupError(f"cannot read the sandbox working directory: {pwd.stderr.strip()}")
        mkdir = await sandbox.exec(["mkdir", "-p", state_dir], cwd="/", timeout_s=60)
        if mkdir.exit_code != 0:
            raise HarnessSetupError(f"cannot create {state_dir} in the sandbox: {mkdir.stderr.strip()}")
        return cls(sandbox, pwd.stdout.strip() or "/", state_dir, agent_slug, kill_grace_s)

    def resolve(self, path: str) -> str:
        return posixpath.join(self.state_dir, path)

    async def shell(self, command: str, *, timeout_s: float) -> tuple[int | None, str]:
        result = await self._sandbox.exec(["sh", "-c", command], timeout_s=timeout_s)
        return (None if result.timed_out else result.exit_code), result.stdout + result.stderr

    async def write(self, path: str, content: str) -> None:
        with scratch.named_temporary_file("w", encoding="utf-8", suffix=".platoon", delete=False) as handle:
            handle.write(content)
        try:
            await self.copy(Path(handle.name), path)
        finally:
            os.unlink(handle.name)

    async def copy(self, source: Path, path: str) -> None:
        parent = posixpath.dirname(path)
        if parent != self.state_dir:
            await self._sandbox.exec(["mkdir", "-p", parent], cwd="/", timeout_s=60)
        await self._sandbox.upload(source, path)

    async def exists(self, path: str) -> bool:
        return (await self._sandbox.exec(["test", "-e", path], cwd="/", timeout_s=60)).exit_code == 0

    async def read(self, path: str) -> str | None:
        result = await self._sandbox.exec(["cat", path], cwd="/", timeout_s=60)
        return result.stdout if result.exit_code == 0 else None

    async def launch(
        self, argv: tuple[str, ...], *, env: dict[str, str], timeout_s: float | None, log_path: Path
    ) -> tuple[int | None, str, bool]:
        backstop = None if timeout_s is None else timeout_s + 2 * self._kill_grace_s + 60
        execution = asyncio.ensure_future(
            self._sandbox.exec(
                list(argv),
                env={**env, AGENT_MARKER_ENV: self._marker},
                timeout_s=backstop,
            )
        )
        timed_out = False
        try:
            try:
                result: ExecResult | None = await asyncio.wait_for(asyncio.shield(execution), timeout_s)
            except TimeoutError:
                timed_out = True
                await self._kill_marked()
                result = await _settle(execution, self._kill_grace_s + 60)
        finally:
            await self._kill_marked()
            execution.cancel()
        output = "" if result is None else result.stdout + result.stderr
        with log_path.open("a", encoding="utf-8") as log:
            log.write(output)
        exit_code = None if result is None else result.exit_code
        return exit_code, output, timed_out or (result is not None and result.timed_out)

    async def _kill_marked(self) -> None:
        """Kill every process in the sandbox that carries this launch's marker.

        It runs in :meth:`launch`'s ``finally``, so it never raises: a kill that fails or times out is logged, and
        what it missed runs on until something else kills it, such as Harbor grading or the sandbox's stop.
        """

        try:
            result = await self._sandbox.exec(
                [
                    "sh",
                    "-c",
                    _KILL_MARKED_SCRIPT,
                    "platoon-kill",
                    f"{AGENT_MARKER_ENV}={self._marker}",
                    str(self._kill_grace_s),
                ],
                cwd="/",
                timeout_s=self._kill_grace_s + 60,
            )
        except Exception as exc:
            logger.warning(
                "Could not kill the processes marked %s in sandbox %s: %s",
                self._marker,
                self._sandbox.id,
                exc,
            )
            return
        if result.timed_out or result.exit_code != 0:
            output = (result.stderr or result.stdout).strip()[-OUTPUT_TAIL_CHARS:]
            logger.warning(
                "Killing the processes marked %s in sandbox %s %s%s",
                self._marker,
                self._sandbox.id,
                "timed out" if result.timed_out else f"exited {result.exit_code}",
                f": {output}" if output else "",
            )

    async def close(self) -> None:
        """The state directory goes away with the sandbox."""


async def _settle(execution: asyncio.Future, timeout_s: float) -> ExecResult | None:
    try:
        return await asyncio.wait_for(execution, timeout_s)
    except TimeoutError:
        return None
