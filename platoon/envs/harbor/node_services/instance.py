"""One start of a node service (:class:`ServiceInstance`): its sandbox, its command, its health check and its log,
as the supervisor runs it (:mod:`platoon.envs.harbor.node_services.supervisor`).

The sandbox is one of the declared provider's, with the declared image, working directory, environment, network and
memory, the declared binds (a staged one from its copy), and the service's directory of this node (``run/`` of its
directory, which holds the socket the service listens on) bound read-write at
:data:`~platoon.envs.harbor.node_services.spec.SERVICE_DIR`.  Its command runs in the background there, with
``$PLATOON_NODE_SERVICE_SOCKET``, ``$PLATOON_NODE_SERVICE_TOKEN`` and ``$PLATOON_NODE_SERVICE_CPUS``; its output goes to
:data:`LOG` in the sandbox.  The health check is ``GET <health path>`` over the socket, with the token, answered 2xx
within its timeout, while the command runs.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from pathlib import Path

import aiohttp

from platoon.envs.harbor.node_services.spec import (
    CPUS_ENV,
    SERVICE_DIR,
    SERVICE_SOCKET,
    SOCKET_ENV,
    TOKEN_ENV,
    NodeServiceSpec,
)
from platoon.sandbox._processes import alive_processes, start_background
from platoon.sandbox.base import Mount, Sandbox, SandboxProvider, SandboxSpec

#: Where the service's command writes its output in its sandbox, and its exit status once it ends.
LOG = "/tmp/platoon-node-service/service.log"
STATUS = "/tmp/platoon-node-service/status"
#: The label of a node service's sandbox: the service's name.
NODE_SERVICE_LABEL = "platoon.node_service"
#: How much of the log a failure reports.
LOG_TAIL_BYTES = 2000


class ServiceInstance:
    """One start of ``spec``'s service: ``run_dir`` is its directory on the node (bound at the service directory),
    ``token`` what every request carries, ``mounts`` the declared binds as they mount, ``cpus`` the CPUs it is
    pinned to."""

    def __init__(self, spec: NodeServiceSpec, *, run_dir: Path, token: str, mounts: Sequence[Mount], cpus: int) -> None:
        self.spec = spec
        self.socket = run_dir / SERVICE_SOCKET
        self._run_dir = run_dir
        self._token = token
        self._mounts = list(mounts)
        self._cpus = cpus
        self.sandbox: Sandbox | None = None
        self._entry: str | None = None
        self._http: aiohttp.ClientSession | None = None

    async def start(self, provider: SandboxProvider) -> None:
        """Start the sandbox, then the command in it; on a failure, the sandbox stops again."""

        spec = self.spec
        self.socket.unlink(missing_ok=True)  # a socket a start before left: the service could not bind it again
        sandbox_spec = SandboxSpec(
            image=spec.image,
            workdir=spec.workdir,
            env={
                **spec.env,
                SOCKET_ENV: f"{SERVICE_DIR}/{SERVICE_SOCKET}",
                TOKEN_ENV: self._token,
                CPUS_ENV: str(self._cpus),
            },
            mounts=[Mount(source=str(self._run_dir), target=SERVICE_DIR, read_only=False), *self._mounts],
            network=spec.network,
            memory_mb=spec.memory_mb,
            labels={NODE_SERVICE_LABEL: spec.name},
        )
        self.sandbox = await provider.start(sandbox_spec)
        try:
            self._entry = await start_background(self.sandbox, list(spec.command), log=LOG, status=STATUS)
        except BaseException:
            await asyncio.shield(self.stop())
            raise

    async def check(self) -> tuple[bool, str]:
        """Whether the service serves now (the module docstring's health check), and what failed when it does not."""

        if not await self.running():
            return False, f"its command exited{await self._exit_status()}"
        if self._http is None:
            self._http = aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=str(self.socket)))
        health = self.spec.health
        try:
            async with self._http.get(
                f"http://node-service{health.path}",
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=aiohttp.ClientTimeout(total=health.timeout_s),
            ) as response:
                if 200 <= response.status < 300:
                    return True, ""
                body = (await response.text(errors="replace"))[:300]
                return False, f"GET {health.path} answered {response.status}: {body}"
        except TimeoutError:
            return False, f"GET {health.path} did not answer within {health.timeout_s:g} s"
        except (aiohttp.ClientError, OSError) as exc:
            return False, f"GET {health.path} failed: {type(exc).__name__}: {exc}"

    async def running(self) -> bool:
        """Whether its command still runs (False once its sandbox cannot be asked)."""

        if self.sandbox is None or self._entry is None:
            return False
        try:
            return bool(await alive_processes(self.sandbox, {self._entry}))
        except Exception:  # a sandbox that cannot be asked serves nothing
            return False

    async def logs(self, max_bytes: int = LOG_TAIL_BYTES) -> str:
        """The end of what its command wrote; empty when it cannot be read."""

        if self.sandbox is None:
            return ""
        tail = ["/bin/sh", "-c", 'tail -c "$1" -- "$2" 2>/dev/null', "platoon-log", str(max_bytes), LOG]
        try:
            result = await self.sandbox.exec(tail, cwd="/", timeout_s=30.0)
        except Exception:
            return ""
        return result.stdout[-max_bytes:]

    async def stop(self) -> None:
        """Stop the sandbox, and everything in it; idempotent."""

        http, self._http = self._http, None
        if http is not None:
            await http.close()
        sandbox, self.sandbox = self.sandbox, None
        if sandbox is not None:
            await sandbox.stop()
        with contextlib.suppress(OSError):
            self.socket.unlink(missing_ok=True)

    async def _exit_status(self) -> str:
        if self.sandbox is None:
            return ""
        try:
            result = await self.sandbox.exec(["cat", STATUS], cwd="/", timeout_s=30.0)
        except Exception:
            return ""
        text = result.stdout.strip()
        return f" with status {text}" if result.exit_code == 0 and text else ""


__all__ = ["LOG", "NODE_SERVICE_LABEL", "STATUS", "ServiceInstance"]
