"""Host subprocesses for sandbox providers: captured output, timeouts and process-group kills."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from collections.abc import Callable, Mapping, Sequence

from platoon.sandbox.base import ExecResult
from platoon.utils.stopping import register_process_group, unregister_process_group

DEFAULT_MAX_OUTPUT_BYTES = 8 * 1024 * 1024  # per stream
# After a command exits, wait this long for its pipes to close: a background
# process it started may hold them open, and its later output is dropped.
_PIPE_DRAIN_S = 1.0


class _Capture:
    """Keeps the first ``limit`` bytes of a stream and counts the rest."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._data = bytearray()
        self._dropped = 0

    def add(self, chunk: bytes) -> None:
        kept = chunk[: max(0, self._limit - len(self._data))]
        self._data += kept
        self._dropped += len(chunk) - len(kept)

    def text(self) -> str:
        text = self._data.decode("utf-8", errors="replace")
        if self._dropped:
            text += f"\n[{self._dropped} bytes of output dropped]"
        return text


class _CommandProtocol(asyncio.SubprocessProtocol):
    """Captures stdout/stderr and signals exit as soon as the process ends, even if its pipes stay open."""

    def __init__(self, max_output_bytes: int) -> None:
        loop = asyncio.get_running_loop()
        self.output = {1: _Capture(max_output_bytes), 2: _Capture(max_output_bytes)}
        self.exited: asyncio.Future[None] = loop.create_future()
        self.pipes_closed: asyncio.Future[None] = loop.create_future()

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        self.output[fd].add(data)

    def process_exited(self) -> None:
        if not self.exited.done():
            self.exited.set_result(None)

    def connection_lost(self, exc: Exception | None) -> None:
        if not self.pipes_closed.done():
            self.pipes_closed.set_result(None)


def kill_process_group(pgid: int) -> None:
    """SIGKILL a process group and stop tracking it; a group whose members are all gone is ignored."""

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    unregister_process_group(pgid)


def _has_members(pgid: int) -> bool:
    """Whether any process is left in group ``pgid``."""

    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # a member this process may not signal
        pass
    return True


async def run_command(
    argv: Sequence[str],
    *,
    timeout_s: float | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    stdin: bytes | None = None,
    process_groups: set[int] | None = None,
    on_group: Callable[[int], None] | None = None,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> ExecResult:
    """Run ``argv`` as the leader of a new process group and wait for it to exit.

    On timeout or cancellation the whole group is killed. The group id is added
    to ``process_groups`` so an owner can kill processes the command leaves
    running in the background, and passed to ``on_group`` (which must not raise)
    as soon as the group exists. Being in a session of its own, the group is out of
    reach of a signal to this process's group, so it is registered with
    :mod:`platoon.utils.stopping` while it has members: a stopping rollout worker
    ends it, also when its own cleanup here or the owner's is cut short.
    """

    loop = asyncio.get_running_loop()
    transport, protocol = await loop.subprocess_exec(
        lambda: _CommandProtocol(max_output_bytes),
        *argv,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=dict(env) if env is not None else None,
        start_new_session=True,
    )
    pgid = transport.get_pid()
    register_process_group(pgid)
    if process_groups is not None:
        process_groups.add(pgid)
    if on_group is not None:
        on_group(pgid)
    if stdin is not None:
        stdin_pipe = transport.get_pipe_transport(0)
        assert isinstance(stdin_pipe, asyncio.WriteTransport)
        stdin_pipe.write(stdin)
        stdin_pipe.close()  # flushes first; a command that exits without reading just drops it
    timed_out = False
    try:
        try:
            await asyncio.wait_for(asyncio.shield(protocol.exited), timeout_s)
        except TimeoutError:
            timed_out = True
            kill_process_group(pgid)
            await protocol.exited
        await asyncio.wait([protocol.pipes_closed], timeout=_PIPE_DRAIN_S)
    finally:
        if transport.get_returncode() is None:
            kill_process_group(pgid)
        elif not _has_members(pgid):  # else the processes it left in the background keep it registered
            unregister_process_group(pgid)
        transport.close()
    exit_code = transport.get_returncode()
    assert exit_code is not None
    return ExecResult(
        exit_code=exit_code,
        stdout=protocol.output[1].text(),
        stderr=protocol.output[2].text(),
        timed_out=timed_out,
    )
