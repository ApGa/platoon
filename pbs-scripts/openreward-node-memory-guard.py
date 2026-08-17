#!/usr/bin/env python3
"""Fail closed before an OpenReward PBS service node exhausts host memory."""

from __future__ import annotations

import argparse
import re
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Optional, TextIO, Union

EXPECTED_NODE_LIMIT_BYTES = 2_061_584_302_080
EXPECTED_TRIGGER_PERCENT = 85
EXPECTED_FALLBACK_AVAILABLE_BYTES = 274_877_906_944
EXPECTED_POLL_SECONDS = 2
EXPECTED_TELEMETRY_SECONDS = 60

EXIT_CONFIGURATION = 2
EXIT_RUNTIME_FAILURE = 3
EXIT_THRESHOLD = 4

DEFAULT_PROC_SELF_CGROUP = Path("/proc/self/cgroup")
DEFAULT_CGROUP_ROOT = Path("/sys/fs/cgroup")
DEFAULT_PROC_MEMINFO = Path("/proc/meminfo")


class GuardFailure(RuntimeError):
    """A memory source could not be read or no longer matches its contract."""


@dataclass(frozen=True)
class Settings:
    node_limit_bytes: int
    trigger_percent: int
    fallback_available_bytes: int
    poll_seconds: int
    telemetry_seconds: int

    @property
    def cgroup_trigger_bytes(self) -> int:
        return self.node_limit_bytes * self.trigger_percent // 100

    def validate_exact_contract(self) -> None:
        actual = (
            self.node_limit_bytes,
            self.trigger_percent,
            self.fallback_available_bytes,
            self.poll_seconds,
            self.telemetry_seconds,
        )
        expected = (
            EXPECTED_NODE_LIMIT_BYTES,
            EXPECTED_TRIGGER_PERCENT,
            EXPECTED_FALLBACK_AVAILABLE_BYTES,
            EXPECTED_POLL_SECONDS,
            EXPECTED_TELEMETRY_SECONDS,
        )
        if actual != expected:
            raise GuardFailure(
                "memory guard settings do not match the exact PBS service-node contract: "
                f"actual={actual!r} expected={expected!r}"
            )


@dataclass(frozen=True)
class Reading:
    mode: str
    triggered: bool
    current_bytes: Optional[int] = None
    limit_bytes: Optional[int] = None
    trigger_bytes: Optional[int] = None
    available_bytes: Optional[int] = None
    available_threshold_bytes: Optional[int] = None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise GuardFailure(f"could not read {path}: {exc}") from exc


def _parse_nonnegative_integer(value: str, *, source: Path) -> int:
    if not re.fullmatch(r"[0-9]+", value):
        raise GuardFailure(f"{source} does not contain one nonnegative integer")
    return int(value)


def read_unified_cgroup_path(proc_self_cgroup: Path) -> Optional[PurePosixPath]:
    unified_paths = []
    for line in _read_text(proc_self_cgroup).splitlines():
        if not line:
            continue
        fields = line.split(":", 2)
        if len(fields) != 3:
            raise GuardFailure(f"malformed cgroup membership line in {proc_self_cgroup}")
        hierarchy_id, controllers, member_path = fields
        if hierarchy_id == "0":
            if controllers:
                raise GuardFailure(f"unified cgroup entry has controllers in {proc_self_cgroup}")
            parsed = PurePosixPath(member_path)
            if not parsed.is_absolute() or ".." in parsed.parts:
                raise GuardFailure(f"unified cgroup membership is not an absolute safe path: {member_path!r}")
            unified_paths.append(parsed)
    if not unified_paths:
        return None
    if len(unified_paths) != 1:
        raise GuardFailure(f"expected exactly one unified cgroup entry in {proc_self_cgroup}")
    return unified_paths[0]


def _ancestor_directories(cgroup_root: Path, member_path: PurePosixPath) -> list[Path]:
    relative_parts = tuple(part for part in member_path.parts if part != "/")
    leaf = cgroup_root.joinpath(*relative_parts)
    ancestors = []
    candidate = leaf
    while True:
        ancestors.append(candidate)
        if candidate == cgroup_root:
            break
        parent = candidate.parent
        if parent == candidate or cgroup_root not in parent.parents and parent != cgroup_root:
            raise GuardFailure("unified cgroup membership escaped the configured cgroup root")
        candidate = parent
    return ancestors


def _read_memory_max_if_present(cgroup_dir: Path) -> Optional[int]:
    memory_max = cgroup_dir / "memory.max"
    try:
        raw = memory_max.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        raise GuardFailure(f"could not read {memory_max}: {exc}") from exc
    if raw == "max":
        return None
    return _parse_nonnegative_integer(raw, source=memory_max)


@dataclass(frozen=True)
class CgroupMemorySource:
    cgroup_dir: Path
    settings: Settings

    @property
    def description(self) -> str:
        return f"mode=cgroup cgroup={self.cgroup_dir}"

    def sample(self) -> Reading:
        memory_max_path = self.cgroup_dir / "memory.max"
        memory_current_path = self.cgroup_dir / "memory.current"
        raw_max = _read_text(memory_max_path).strip()
        if raw_max == "max":
            raise GuardFailure(f"{memory_max_path} changed from finite to unlimited")
        memory_max = _parse_nonnegative_integer(raw_max, source=memory_max_path)
        if memory_max != self.settings.node_limit_bytes:
            raise GuardFailure(
                f"{memory_max_path} changed: actual={memory_max} expected={self.settings.node_limit_bytes}"
            )
        memory_current = _parse_nonnegative_integer(_read_text(memory_current_path).strip(), source=memory_current_path)
        trigger = self.settings.cgroup_trigger_bytes
        return Reading(
            mode="cgroup",
            triggered=memory_current >= trigger,
            current_bytes=memory_current,
            limit_bytes=memory_max,
            trigger_bytes=trigger,
        )


@dataclass(frozen=True)
class MeminfoMemorySource:
    meminfo_path: Path
    settings: Settings

    @property
    def description(self) -> str:
        return f"mode=meminfo meminfo={self.meminfo_path}"

    def sample(self) -> Reading:
        matches = []
        for line in _read_text(self.meminfo_path).splitlines():
            match = re.fullmatch(r"MemAvailable:[ \t]+([0-9]+)[ \t]+kB", line)
            if match:
                matches.append(int(match.group(1)) * 1024)
        if len(matches) != 1:
            raise GuardFailure(f"expected exactly one byte-convertible MemAvailable entry in {self.meminfo_path}")
        available = matches[0]
        threshold = self.settings.fallback_available_bytes
        return Reading(
            mode="meminfo",
            triggered=available < threshold,
            available_bytes=available,
            available_threshold_bytes=threshold,
        )


MemorySource = Union[CgroupMemorySource, MeminfoMemorySource]


def resolve_memory_source(
    settings: Settings,
    *,
    proc_self_cgroup: Path = DEFAULT_PROC_SELF_CGROUP,
    cgroup_root: Path = DEFAULT_CGROUP_ROOT,
    proc_meminfo: Path = DEFAULT_PROC_MEMINFO,
) -> MemorySource:
    member_path = read_unified_cgroup_path(proc_self_cgroup)
    exact_matches = []
    if member_path is not None:
        for cgroup_dir in _ancestor_directories(cgroup_root, member_path):
            if _read_memory_max_if_present(cgroup_dir) == settings.node_limit_bytes:
                exact_matches.append(cgroup_dir)
    if exact_matches:
        # The highest matching ancestor is the aggregate PBS allocation cgroup,
        # rather than a narrower pbsdsh/task child carrying the same limit.
        return CgroupMemorySource(exact_matches[-1], settings)
    return MeminfoMemorySource(proc_meminfo, settings)


def _reading_fields(reading: Reading) -> str:
    if reading.mode == "cgroup":
        return (
            f"memory_current_bytes={reading.current_bytes} "
            f"memory_max_bytes={reading.limit_bytes} "
            f"trigger_bytes={reading.trigger_bytes}"
        )
    return (
        f"memory_available_bytes={reading.available_bytes} "
        f"fallback_available_threshold_bytes={reading.available_threshold_bytes}"
    )


def run_monitor(
    source: MemorySource,
    settings: Settings,
    stop_event: threading.Event,
    *,
    output: TextIO = sys.stdout,
    error: TextIO = sys.stderr,
    monotonic: Callable[[], float] = time.monotonic,
    wait_for_stop: Optional[Callable[[float], bool]] = None,
) -> int:
    wait = wait_for_stop if wait_for_stop is not None else stop_event.wait
    try:
        reading = source.sample()
    except GuardFailure as exc:
        print(f"[node-memory-guard] ERROR: initial sample failed: {exc}", file=error, flush=True)
        return EXIT_RUNTIME_FAILURE

    print(
        f"[node-memory-guard] event=start {source.description} "
        f"{_reading_fields(reading)} poll_seconds={settings.poll_seconds} "
        f"telemetry_seconds={settings.telemetry_seconds}",
        file=output,
        flush=True,
    )
    if reading.triggered:
        print(
            f"[node-memory-guard] ERROR: event=threshold {_reading_fields(reading)}",
            file=error,
            flush=True,
        )
        return EXIT_THRESHOLD

    next_telemetry = monotonic() + settings.telemetry_seconds
    while not stop_event.is_set():
        if wait(settings.poll_seconds):
            break
        try:
            reading = source.sample()
        except GuardFailure as exc:
            print(f"[node-memory-guard] ERROR: sample failed: {exc}", file=error, flush=True)
            return EXIT_RUNTIME_FAILURE
        now = monotonic()
        if reading.triggered:
            print(
                f"[node-memory-guard] ERROR: event=threshold {_reading_fields(reading)}",
                file=error,
                flush=True,
            )
            return EXIT_THRESHOLD
        if now >= next_telemetry:
            print(
                f"[node-memory-guard] event=telemetry {source.description} {_reading_fields(reading)}",
                file=output,
                flush=True,
            )
            while next_telemetry <= now:
                next_telemetry += settings.telemetry_seconds

    print("[node-memory-guard] event=stop reason=signal", file=output, flush=True)
    return 0


def _positive_decimal(value: str) -> int:
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise argparse.ArgumentTypeError("must be a positive base-10 integer")
    return int(value)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-limit-bytes", required=True, type=_positive_decimal)
    parser.add_argument("--trigger-percent", required=True, type=_positive_decimal)
    parser.add_argument("--fallback-available-bytes", required=True, type=_positive_decimal)
    parser.add_argument("--poll-seconds", required=True, type=_positive_decimal)
    parser.add_argument("--telemetry-seconds", required=True, type=_positive_decimal)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    settings = Settings(
        node_limit_bytes=args.node_limit_bytes,
        trigger_percent=args.trigger_percent,
        fallback_available_bytes=args.fallback_available_bytes,
        poll_seconds=args.poll_seconds,
        telemetry_seconds=args.telemetry_seconds,
    )
    try:
        settings.validate_exact_contract()
        source = resolve_memory_source(settings)
    except GuardFailure as exc:
        print(f"[node-memory-guard] ERROR: startup failed: {exc}", file=sys.stderr, flush=True)
        return EXIT_CONFIGURATION

    stop_event = threading.Event()

    def request_stop(signum: int, _frame: object) -> None:
        del signum
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    return run_monitor(source, settings, stop_event)


if __name__ == "__main__":
    raise SystemExit(main())
