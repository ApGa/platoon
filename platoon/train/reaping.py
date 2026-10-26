"""What a run leaves behind when its processes die, and reaping it.

Every sandbox of a ``platoon train`` or ``platoon eval`` run carries the run's label, ``platoon.run: <run
directory>`` (:func:`platoon.sandbox.cleanup.run_label`), in the run's own process and in its rollout workers.  A
process that dies without stopping its sandboxes (the watchdog's exit, a second signal's, a crash, SIGKILL, an
out-of-memory kill) leaves them running, and a trainer that dies may leave rollout workers running too.  So a run
reaps what it left (:class:`RunReaper`), and logs how many sandboxes went:

* ``platoon train`` when it starts, before each restart of a supervised run (:mod:`platoon.train.supervise`), and
  when it exits; ``platoon eval`` when it exits (:func:`held_run`);
* ``platoon sandbox reap RUN_DIR``, for a run that was killed (:func:`reap_stopped_run`).

**Which providers.**  The sandbox providers the run's config names (:func:`sandbox_providers`): each Harbor entry's
``service_kwargs.provider`` (default ``docker``) with its ``provider_options``.  Each is asked to reap the sandboxes
whose ``platoon.run`` label is the run's, matched exactly, so no other run's are touched.  A provider leaves alone the
sandboxes another live process holds, except Docker, which cannot tell whose a container is and removes every
container with the label.

**Whether a run is alive.**  So the command that runs a run records itself in the run directory
(:func:`record_process`, ``process.json``): its host, pid and start time, those of the training process it starts
when it supervises training (:func:`record_trainer`), and the label of the run's sandboxes.  ``platoon train`` and
``platoon eval`` refuse a run directory that a process the record names still runs on this host, before they write
anything there (:func:`refuse_running`), and ``platoon sandbox reap`` refuses such a run too, and one whose record
names another host, where nothing here can tell (:func:`live_process`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.utils.config import ConfigError
from platoon.utils.stopping import process_start_time

logger = logging.getLogger(__name__)

__all__ = [
    "HARBOR_DEFAULT_PROVIDER",
    "PROCESS_RECORD",
    "REAP_TIMEOUT_S",
    "HeldRun",
    "ProcessRecord",
    "RunReaper",
    "held_run",
    "live_process",
    "read_process",
    "reap_stopped_run",
    "record_process",
    "record_trainer",
    "refuse_running",
    "running_here",
    "sandbox_providers",
    "stop_orphaned_workers",
]

#: The record of the processes that run a run directory, in that directory.
PROCESS_RECORD = "process.json"
#: How long one reap may take in all, every provider at once.
REAP_TIMEOUT_S = 120.0
#: The provider a Harbor entry starts its sandboxes with when its ``service_kwargs`` name none (the ``provider``
#: default of :class:`platoon.envs.harbor.HarborEnvironmentService`).
HARBOR_DEFAULT_PROVIDER = "docker"
_HARBOR_SERVICE = "platoon.envs.harbor:HarborEnvironmentService"


# --------------------------------------------------------------------------------------------- the providers


def sandbox_providers(environments: Iterable[Any] | None) -> list[tuple[str, dict[str, Any]]]:
    """The sandbox providers a run's environments start sandboxes with, each once, as ``(provider, options)``: the
    ``provider`` and ``provider_options`` of every Harbor entry's ``service_kwargs``, and with its
    ``verifier_provider_options`` laid over those, the provider of its separate verifiers' sandboxes, unless those
    options would start no sandbox here (:func:`platoon.sandbox.rootfs_layout_refusal`, such as ``fakeroot`` on an
    hpc-sandbox that does not take it), so that no verifier's sandbox can have started with them.  An entry that
    names no provider (a record written before platoon recorded the site profile's) has the site profile's, with its
    options under the entry's, else ``docker`` (:mod:`platoon.site`).

    ``environments`` are a config's ``environments`` entries, as the loaded config holds them or as mappings, the
    form ``config.resolved.yaml`` holds them in.  Entries of other services start no sandboxes of their own.
    """

    from platoon.envs.service import service_target
    from platoon.sandbox import rootfs_layout_refusal

    providers: list[tuple[str, dict[str, Any]]] = []
    for entry in environments or ():
        service = _field(entry, "service")
        if not isinstance(service, str) or service_target(service) != _HARBOR_SERVICE:
            continue
        kwargs = _field(entry, "service_kwargs") or {}
        if not isinstance(kwargs, Mapping):
            continue
        provider, options = kwargs.get("provider"), kwargs.get("provider_options") or {}
        if provider is None:
            provider, defaults = _site_provider()
            if isinstance(options, Mapping):
                options = {**defaults, **options}
        if not isinstance(provider, str) or not isinstance(options, Mapping):
            continue  # not a config's: a provider object passed in Python
        found = [(provider, dict(options))]
        verifier = kwargs.get("verifier_provider_options")
        if isinstance(verifier, Mapping):  # separate verifiers' sandboxes, with these laid over the entry's options
            laid = {**options, **verifier}
            if rootfs_layout_refusal(provider, laid) is None:  # else a run refuses them, and none started
                found.append((provider, laid))
        for each in found:
            if each not in providers:
                providers.append(each)
    return providers


def _field(entry: Any, name: str) -> Any:
    return entry.get(name) if isinstance(entry, Mapping) else getattr(entry, name, None)


def _site_provider() -> tuple[str, dict[str, Any]]:
    """The provider of a Harbor entry that names none: the site profile's, with its options, else ``docker``.  A
    profile that cannot be read leaves ``docker``, with a warning: reaping must not fail for it."""

    from platoon.site import site_profile

    try:
        profile = site_profile()
    except ConfigError as exc:
        logger.warning("Reaping with the default sandbox provider, %s: %s", HARBOR_DEFAULT_PROVIDER, exc)
        return HARBOR_DEFAULT_PROVIDER, {}
    if profile.provider is None:
        return HARBOR_DEFAULT_PROVIDER, {}
    return profile.provider, dict(profile.provider_options)


# --------------------------------------------------------------------------------------------- the record


@dataclass(frozen=True)
class ProcessRecord:
    """What ``process.json`` holds: the command that runs the run directory (``train`` or ``eval``), its host, pid
    and process start time, the training process it supervises, if any, and the label of the run's sandboxes."""

    command: str
    host: str
    pid: int
    start_time: int | None
    label: dict[str, str] = field(default_factory=dict)
    trainer_pid: int | None = None
    trainer_start_time: int | None = None

    def processes(self) -> list[tuple[str, int, int | None]]:
        """``(what, pid, start time)`` of each process the record names."""

        named = [(f"platoon {self.command}", self.pid, self.start_time)]
        if self.trainer_pid is not None:
            named.append((f"the training process of platoon {self.command}", self.trainer_pid, self.trainer_start_time))
        return named


def record_process(run_dir: str | os.PathLike[str], command: str) -> ProcessRecord:
    """Record this process as the one that runs ``run_dir`` now (``process.json``, replaced whole); returns the
    record.  A failure to write it is logged, never raised: it only guards reaping."""

    record = ProcessRecord(
        command=command,
        host=socket.gethostname(),
        pid=os.getpid(),
        start_time=process_start_time(os.getpid()),
        label=sandbox_cleanup.run_label(run_dir),
    )
    _write(run_dir, record)
    return record


def record_trainer(run_dir: str | os.PathLike[str], record: ProcessRecord, pid: int | None) -> ProcessRecord:
    """Add the training process ``pid`` (None: none runs now) to ``record``, this process's, and write it; returns
    the new record."""

    start_time = process_start_time(pid) if pid is not None else None
    updated = replace(record, trainer_pid=pid, trainer_start_time=start_time)
    _write(run_dir, updated)
    return updated


def read_process(run_dir: str | os.PathLike[str]) -> ProcessRecord | None:
    """The record in ``run_dir``; None when there is none, or none that can be read.  Keys it does not know, which a
    later platoon may add, are left out."""

    try:
        data = json.loads((Path(run_dir) / PROCESS_RECORD).read_text(encoding="utf-8"))
        known = {item.name for item in fields(ProcessRecord)}
        return ProcessRecord(**{key: value for key, value in data.items() if key in known})
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def running_here(record: ProcessRecord | None, *, ignore: Sequence[int] = ()) -> str | None:
    """What still runs the run of ``record`` on this host (a process it names, ``ignore`` aside), or None."""

    if record is None or record.host != socket.gethostname():
        return None
    for what, pid, start_time in record.processes():
        if pid not in ignore and _running(pid, start_time):
            return f"{what} (pid {pid}) still runs it on this host"
    return None


def live_process(run_dir: str | os.PathLike[str]) -> str | None:
    """Why the run in ``run_dir`` may still be running, or None: a process its record names runs on this host, or
    the record names another host, where nothing here can tell."""

    record = read_process(run_dir)
    if record is not None and record.host != socket.gethostname():
        return (
            f"its last process, platoon {record.command} (pid {record.pid}), ran on host {record.host}, so whether it "
            f"still runs cannot be told on this host ({socket.gethostname()}), and the sandboxes it started are on "
            f"{record.host}: run this there"
        )
    return running_here(record)


def refuse_running(run_dir: str | os.PathLike[str], command: str) -> None:
    """Raise :class:`ConfigError` (exit 78) when a process that ``run_dir``'s record names still runs it on this host:
    two runs in one directory write over each other's checkpoints and records, and each would reap the other's
    sandboxes.  Called before anything is written there."""

    if (alive := running_here(read_process(run_dir), ignore=(os.getpid(),))) is not None:
        raise ConfigError(
            f"{run_dir} is in use: {alive}; wait for it to exit, or give this platoon {command} another run directory"
        )


def _running(pid: int, start_time: int | None) -> bool:
    """Whether the process ``pid`` that started at ``start_time`` runs (a zombie has stopped running)."""

    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
        fields = raw[raw.rindex(b")") + 2 :].split()
    except (OSError, ValueError):
        return False
    if fields[0] in (b"Z", b"X", b"x"):
        return False
    return start_time is None or int(fields[19]) == start_time


def _write(run_dir: str | os.PathLike[str], record: ProcessRecord) -> None:
    path = Path(run_dir) / PROCESS_RECORD
    partial = path.with_name(f".{path.name}.partial")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps(asdict(record), indent=2) + "\n", encoding="utf-8")
        partial.replace(path)  # readers never see half a record
    except OSError:
        logger.warning("Cannot record the run's process in %s", path, exc_info=True)


# --------------------------------------------------------------------------------------------- reaping


class RunReaper:
    """Reaps what a run left: the rollout workers of trainers that are gone, and the sandboxes labelled with the run.

    ``providers`` are the run's (:func:`sandbox_providers`), and ``label`` the label of its sandboxes (default: the
    run directory's, :func:`platoon.sandbox.cleanup.run_label`).
    """

    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        providers: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        label: Mapping[str, str] | None = None,
        timeout_s: float = REAP_TIMEOUT_S,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.providers = [(provider, dict(options)) for provider, options in providers]
        self.label = dict(label) if label else sandbox_cleanup.run_label(run_dir)
        self.timeout_s = float(timeout_s)

    def reap(self, when: str, *, workers: bool = False, own: bool = False) -> int:
        """Reap the run's sandboxes and log how many went, ``when`` saying when; returns that count; never raises.

        ``workers``: first stop the rollout workers whose trainer is gone, which would otherwise hold sandboxes.
        ``own``: first stop the sandboxes this process started and still holds, which a reap leaves to a live owner
        (call it once nothing uses them).  Runs an event loop of its own: call it where none runs.
        """

        try:
            return asyncio.run(self.areap(when, workers=workers, own=own))
        except Exception:
            logger.warning("Could not reap the sandboxes labelled %s %s", _label_text(self.label), when, exc_info=True)
            return 0

    async def areap(self, when: str, *, workers: bool = False, own: bool = False) -> int:
        """:meth:`reap`, in a running event loop."""

        if workers:
            await asyncio.to_thread(stop_orphaned_workers)
        counts: dict[str, int] = {}
        if own:
            counts["this process's own"] = await sandbox_cleanup.stop_live(self.timeout_s)
        records = [
            sandbox_cleanup.SandboxRecord(provider=provider, options=options, labels=self.label)
            for provider, options in self.providers
        ]
        reaped = await asyncio.gather(*(sandbox_cleanup.reap_records([record], self.timeout_s) for record in records))
        for record, count in zip(records, reaped, strict=True):
            name = record.provider if record.provider not in counts else f"{record.provider} {record.options}"
            counts[name] = count
        total = sum(counts.values())
        if self.providers:
            detail = ", ".join(f"{name}: {count}" for name, count in counts.items())
        else:
            detail = "the run's config names no sandbox provider"
        logger.info("Reaped %d sandbox(es) labelled %s %s (%s)", total, _label_text(self.label), when, detail)
        return total


def stop_orphaned_workers() -> list[int]:
    """Kill the rollout workers this host's trainers recorded whose trainer is gone (any run's: they belong to
    nobody), from the registry under ``$TMPDIR`` where trainers keep it; returns their pids; never raises."""

    try:
        from platoon.train.workers.protocol import ipc_dir
        from platoon.train.workers.registry import registry_dir, sweep_orphaned_workers

        stopped = sweep_orphaned_workers(registry_dir(ipc_dir()))
    except Exception:
        logger.warning("Could not look for rollout workers a dead trainer left running", exc_info=True)
        return []
    if stopped:
        logger.warning("Stopped %d rollout worker(s) a dead trainer left running: %s", len(stopped), stopped)
    return stopped


def _label_text(label: Mapping[str, str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in label.items())


@dataclass
class HeldRun:
    """A run directory this process runs (:func:`held_run`): its record, and the reaper of its leftovers."""

    run_dir: Path
    record: ProcessRecord
    reaper: RunReaper

    def record_trainer(self, pid: int | None) -> None:
        """Note the training process ``pid`` this process supervises now (None: none) in the run's record."""

        self.record = record_trainer(self.run_dir, self.record, pid)


@contextmanager
def held_run(
    run_dir: str | os.PathLike[str],
    command: str,
    environments: Iterable[Any] | None,
    *,
    workers: bool,
    reap_first: bool,
) -> Generator[HeldRun]:
    """Run ``run_dir`` in this process: record this process as the one that runs it (``process.json``), reap what
    the run left before the block when ``reap_first``, and reap what it leaves when the block exits, however it
    exits.  ``workers``: also stop the rollout workers whose trainer is gone (a training run's).

    Call :func:`refuse_running` first, before anything is written in ``run_dir``.
    """

    run_dir = Path(run_dir)
    record = record_process(run_dir, command)
    reaper = RunReaper(run_dir, sandbox_providers(environments))
    if reap_first:
        reaper.reap("before the run starts", workers=workers)
    try:
        yield HeldRun(run_dir, record, reaper)
    finally:
        reaper.reap("as the run exits", workers=workers, own=True)


def reap_stopped_run(run_dir: str | os.PathLike[str], environments: Iterable[Any] | None) -> tuple[int, str | None]:
    """Reap the sandboxes a run that no longer runs left (``platoon sandbox reap``): ``(how many, None)``, or ``(0,
    why)`` when it refuses because the run may still be running (:func:`live_process`).

    The label is the one the run's record names, else the run directory's; the providers are those its config's
    ``environments`` name.  Rollout workers are not looked for: a dead trainer's stop by themselves.
    """

    if (alive := live_process(run_dir)) is not None:
        return 0, alive
    record = read_process(run_dir)
    label = record.label if record is not None and record.label else None
    reaper = RunReaper(run_dir, sandbox_providers(environments), label=label)
    return reaper.reap("of a run that no longer runs"), None
