#!/usr/bin/env python3
"""Bounded-duty, utilization-aware idle guard for one training GPU.

Run one process per GPU. The guard samples ``nvidia-smi`` twice and only emits
a short BF16 matrix-multiply burst when every recent sample is below the
configured utilization threshold. Safety validation caps both burst size and
average duty cycle so this cannot regress into the old high-duty setup
keepalive workload.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

PREFIX = "[rollout_gpu_idle_guard]"
# PyTorch documents zero as its default stream priority; negative values request
# higher urgency. This public constructor contract is available in Torch 2.9.
LOW_PRIORITY_STREAM_PRIORITY = 0


@dataclass(frozen=True)
class GpuUtilization:
    index: int
    uuid: str
    utilization: int


@dataclass(frozen=True)
class GuardConfig:
    interval_seconds: float = 10.0
    interval_jitter_seconds: float = 2.0
    sample_count: int = 2
    sample_interval_seconds: float = 2.0
    utilization_threshold: int = 10
    burst_seconds: float = 2.0
    matrix_dim: int = 1024
    operations_per_sync: int = 32
    expected_devices: int = 1
    max_consecutive_query_errors: int = 5
    max_consecutive_cuda_errors: int = 3
    max_runtime_seconds: float = 0.0
    log_every_cycles: int = 10

    def validate(self) -> None:
        if self.interval_seconds < 10:
            raise ValueError("interval_seconds must be at least 10")
        if not 0 <= self.interval_jitter_seconds <= self.interval_seconds:
            raise ValueError(
                "interval_jitter_seconds must be between zero and interval_seconds"
            )
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least 2")
        if self.sample_interval_seconds < 0.1:
            raise ValueError("sample_interval_seconds must be at least 0.1")
        if not 1 <= self.utilization_threshold <= 100:
            raise ValueError("utilization_threshold must be between 1 and 100")
        if self.burst_seconds / self.interval_seconds > 0.25:
            raise ValueError("configured burst duty cycle must not exceed 25%")
        if not 0.05 <= self.burst_seconds <= 2.0:
            raise ValueError("burst_seconds must be between 0.05 and 2.0")
        if not 128 <= self.matrix_dim <= 2048:
            raise ValueError("matrix_dim must be between 128 and 2048")
        if self.operations_per_sync <= 0:
            raise ValueError("operations_per_sync must be positive")
        if self.expected_devices != 1:
            raise ValueError("run exactly one guard process per visible GPU")
        if self.max_consecutive_query_errors <= 0:
            raise ValueError("max_consecutive_query_errors must be positive")
        if self.max_consecutive_cuda_errors <= 0:
            raise ValueError("max_consecutive_cuda_errors must be positive")
        if self.max_runtime_seconds < 0:
            raise ValueError("max_runtime_seconds must not be negative")
        if self.log_every_cycles <= 0:
            raise ValueError("log_every_cycles must be positive")


@dataclass(frozen=True)
class CycleResult:
    samples: tuple[int, ...]
    action: str
    burst_elapsed_seconds: float = 0.0
    operations: int = 0


class UtilizationProbe(Protocol):
    def read(self) -> GpuUtilization: ...


class Burster(Protocol):
    low_priority: int

    def burst(self, seconds: float) -> tuple[float, int]: ...


def parse_nvidia_smi_output(output: str) -> list[GpuUtilization]:
    rows: list[GpuUtilization] = []
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ValueError(f"invalid nvidia-smi row {line_number}: {raw_line!r}")
        try:
            index = int(fields[0])
            utilization = int(fields[2])
        except ValueError as exc:
            raise ValueError(f"invalid nvidia-smi row {line_number}: {raw_line!r}") from exc
        if not 0 <= utilization <= 100:
            raise ValueError(f"invalid utilization in row {line_number}: {utilization}")
        rows.append(GpuUtilization(index=index, uuid=fields[1], utilization=utilization))
    if not rows:
        raise ValueError("nvidia-smi returned no GPU utilization rows")
    return rows


def visible_device_selectors(value: str | None = None) -> tuple[str, ...]:
    raw_value = os.environ.get("CUDA_VISIBLE_DEVICES") if value is None else value
    if raw_value is None:
        return ()
    return tuple(selector.strip() for selector in raw_value.split(",") if selector.strip())


def select_visible_gpu(rows: list[GpuUtilization], selectors: tuple[str, ...]) -> GpuUtilization:
    """Resolve the one GPU assigned to this Slurm task.

    Pyxis may expose only the assigned physical GPU while retaining a remapped
    ``CUDA_VISIBLE_DEVICES=0``. The single-row fallback handles that case. When
    multiple rows are visible, the physical index or UUID must match exactly so
    sibling tasks cannot all touch GPU zero by accident.
    """

    if len(rows) == 1:
        return rows[0]
    if len(selectors) != 1:
        raise RuntimeError(
            "expected exactly one CUDA_VISIBLE_DEVICES selector when nvidia-smi "
            f"reports {len(rows)} GPUs; got {selectors!r}"
        )
    selector = selectors[0]
    matches: list[GpuUtilization]
    if selector.isdigit():
        physical_index = int(selector)
        matches = [row for row in rows if row.index == physical_index]
    else:
        matches = [
            row
            for row in rows
            if row.uuid == selector or row.uuid.startswith(selector) or selector.startswith(row.uuid)
        ]
    if len(matches) != 1:
        raise RuntimeError(
            f"could not uniquely resolve visible GPU {selector!r} from {[(row.index, row.uuid) for row in rows]!r}"
        )
    return matches[0]


class NvidiaSmiProbe:
    def __init__(self, binary: str, timeout_seconds: float = 5.0):
        self._binary = binary
        self._timeout_seconds = timeout_seconds
        self._selectors = visible_device_selectors()
        self._selected_uuid: str | None = None

    def read(self) -> GpuUtilization:
        completed = subprocess.run(
            [
                self._binary,
                "--query-gpu=index,uuid,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=self._timeout_seconds,
        )
        rows = parse_nvidia_smi_output(completed.stdout)
        if self._selected_uuid is None:
            selected = select_visible_gpu(rows, self._selectors)
            self._selected_uuid = selected.uuid
            return selected
        matches = [row for row in rows if row.uuid == self._selected_uuid]
        if len(matches) != 1:
            raise RuntimeError(f"assigned GPU {self._selected_uuid!r} disappeared from nvidia-smi output")
        return matches[0]


class TorchBf16Burster:
    def __init__(self, matrix_dim: int, operations_per_sync: int):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"guard requires exactly one visible CUDA device; found {torch.cuda.device_count()}")
        self._torch = torch
        self._matrix_dim = matrix_dim
        self._operations_per_sync = operations_per_sync
        torch.cuda.set_device(0)
        self.low_priority = LOW_PRIORITY_STREAM_PRIORITY
        self._stream = torch.cuda.Stream(device=0, priority=self.low_priority)
        self._left = None
        self._right = None
        self._output = None

    def _ensure_operands(self) -> None:
        if self._left is not None:
            return
        torch = self._torch
        with torch.cuda.stream(self._stream):
            self._left = torch.randn(
                self._matrix_dim,
                self._matrix_dim,
                dtype=torch.bfloat16,
                device="cuda:0",
            )
            self._right = torch.randn(
                self._matrix_dim,
                self._matrix_dim,
                dtype=torch.bfloat16,
                device="cuda:0",
            )
            self._output = torch.empty_like(self._left)
        self._stream.synchronize()

    def burst(self, seconds: float) -> tuple[float, int]:
        torch = self._torch
        started = time.monotonic()
        self._ensure_operands()
        operations = 0
        with torch.inference_mode():
            while operations == 0 or time.monotonic() - started < seconds:
                with torch.cuda.stream(self._stream):
                    for _ in range(self._operations_per_sync):
                        torch.mm(self._left, self._right, out=self._output)
                self._stream.synchronize()
                operations += self._operations_per_sync
        return time.monotonic() - started, operations


def run_cycle(
    probe: UtilizationProbe,
    burster: Burster,
    config: GuardConfig,
    wait_between_samples=time.sleep,
) -> CycleResult:
    samples: list[int] = []
    gpu_uuid: str | None = None
    for sample_index in range(config.sample_count):
        sample = probe.read()
        if gpu_uuid is None:
            gpu_uuid = sample.uuid
        elif sample.uuid != gpu_uuid:
            raise RuntimeError(f"utilization probe changed GPUs from {gpu_uuid} to {sample.uuid}")
        samples.append(sample.utilization)
        if sample_index + 1 < config.sample_count:
            wait_between_samples(config.sample_interval_seconds)

    sample_tuple = tuple(samples)
    if max(sample_tuple) >= config.utilization_threshold:
        return CycleResult(samples=sample_tuple, action="skip-active")
    elapsed, operations = burster.burst(config.burst_seconds)
    return CycleResult(
        samples=sample_tuple,
        action="burst",
        burst_elapsed_seconds=elapsed,
        operations=operations,
    )


def deterministic_jitter_seed(
    environ: Mapping[str, str] | None = None,
) -> int:
    source = os.environ if environ is None else environ
    identity_parts = []
    for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_PROCID"):
        identity_parts.extend((name, source.get(name, "")))
    identity = "\0".join(identity_parts)
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def cycle_interval_jitter(
    config: GuardConfig, rng: random.Random
) -> float:
    return rng.uniform(0.0, config.interval_jitter_seconds)


def publish_ready(
    ready_dir: Path | None,
    gpu: GpuUtilization,
    config: GuardConfig,
    low_priority: int,
    jitter_seed: int,
) -> Path | None:
    if ready_dir is None:
        return None
    task_id = os.environ.get("SLURM_PROCID", str(os.getpid()))
    ready_dir.mkdir(parents=True, exist_ok=True)
    destination = ready_dir / f"{task_id}.ready"
    temporary = ready_dir / f".{task_id}.{os.getpid()}.tmp"
    temporary.write_text(
        "\n".join(
            [
                f"pid={os.getpid()}",
                f"host={os.uname().nodename}",
                f"task={task_id}",
                f"gpu_index={gpu.index}",
                f"gpu_uuid={gpu.uuid}",
                f"interval_seconds={config.interval_seconds}",
                f"interval_jitter_seconds={config.interval_jitter_seconds}",
                f"jitter_seed={jitter_seed}",
                f"burst_seconds={config.burst_seconds}",
                f"utilization_threshold={config.utilization_threshold}",
                f"stream_priority={low_priority}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def _env(name: str, default: str) -> str:
    return os.environ.get(f"ROLLOUT_IDLE_GUARD_{name}", default)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=float, default=float(_env("INTERVAL_SECONDS", "10")))
    parser.add_argument(
        "--interval-jitter-seconds",
        type=float,
        default=float(_env("INTERVAL_JITTER_SECONDS", "2")),
    )
    parser.add_argument("--sample-count", type=int, default=int(_env("SAMPLE_COUNT", "2")))
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=float(_env("SAMPLE_INTERVAL_SECONDS", "2")),
    )
    parser.add_argument(
        "--utilization-threshold",
        type=int,
        default=int(_env("UTILIZATION_THRESHOLD", "10")),
    )
    parser.add_argument("--burst-seconds", type=float, default=float(_env("BURST_SECONDS", "2")))
    parser.add_argument("--matrix-dim", type=int, default=int(_env("MATRIX_DIM", "1024")))
    parser.add_argument(
        "--operations-per-sync",
        type=int,
        default=int(_env("OPERATIONS_PER_SYNC", "32")),
    )
    parser.add_argument("--expected-devices", type=int, default=int(_env("EXPECTED_DEVICES", "1")))
    parser.add_argument(
        "--max-consecutive-query-errors",
        type=int,
        default=int(_env("MAX_CONSECUTIVE_QUERY_ERRORS", "5")),
    )
    parser.add_argument(
        "--max-consecutive-cuda-errors",
        type=int,
        default=int(_env("MAX_CONSECUTIVE_CUDA_ERRORS", "3")),
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=float(_env("MAX_RUNTIME_SECONDS", "0")),
    )
    parser.add_argument("--log-every-cycles", type=int, default=int(_env("LOG_EVERY_CYCLES", "10")))
    parser.add_argument("--nvidia-smi-bin", default=_env("NVIDIA_SMI_BIN", "nvidia-smi"))
    parser.add_argument("--ready-dir", type=Path, default=os.environ.get("ROLLOUT_IDLE_GUARD_READY_DIR"))
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> GuardConfig:
    config = GuardConfig(
        interval_seconds=args.interval_seconds,
        interval_jitter_seconds=args.interval_jitter_seconds,
        sample_count=args.sample_count,
        sample_interval_seconds=args.sample_interval_seconds,
        utilization_threshold=args.utilization_threshold,
        burst_seconds=args.burst_seconds,
        matrix_dim=args.matrix_dim,
        operations_per_sync=args.operations_per_sync,
        expected_devices=args.expected_devices,
        max_consecutive_query_errors=args.max_consecutive_query_errors,
        max_consecutive_cuda_errors=args.max_consecutive_cuda_errors,
        max_runtime_seconds=args.max_runtime_seconds,
        log_every_cycles=args.log_every_cycles,
    )
    config.validate()
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = config_from_args(args)
    except ValueError as exc:
        print(f"{PREFIX} invalid configuration: {exc}", file=sys.stderr, flush=True)
        return 2

    jitter_seed = deterministic_jitter_seed()
    jitter_rng = random.Random(jitter_seed)
    stop_event = threading.Event()

    def request_stop(signum, _frame) -> None:
        print(f"{PREFIX} received signal {signum}; stopping", flush=True)
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        probe = NvidiaSmiProbe(args.nvidia_smi_bin)
        initial_gpu = probe.read()
        burster = TorchBf16Burster(config.matrix_dim, config.operations_per_sync)
    except Exception as exc:
        print(f"{PREFIX} initialization failed: {exc}", file=sys.stderr, flush=True)
        return 1

    print(
        f"{PREFIX} START host={os.uname().nodename} pid={os.getpid()} "
        f"gpu={initial_gpu.uuid} index={initial_gpu.index} interval={config.interval_seconds}s "
        f"jitter=0..{config.interval_jitter_seconds}s jitter_seed={jitter_seed} "
        f"samples={config.sample_count}x{config.sample_interval_seconds}s "
        f"threshold={config.utilization_threshold}% burst={config.burst_seconds}s "
        f"dim={config.matrix_dim} duty_cap={config.burst_seconds / config.interval_seconds:.3%} "
        f"stream_priority={burster.low_priority}",
        flush=True,
    )

    started = time.monotonic()
    cycle = 0
    query_errors = 0
    cuda_errors = 0
    ready_published = False
    while not stop_event.is_set():
        if config.max_runtime_seconds and time.monotonic() - started >= config.max_runtime_seconds:
            print(f"{PREFIX} reached max runtime; stopping", flush=True)
            break
        cycle += 1
        cycle_jitter = cycle_interval_jitter(config, jitter_rng)
        if stop_event.wait(cycle_jitter):
            break
        cycle_started = time.monotonic()
        try:
            result = run_cycle(probe, burster, config, stop_event.wait)
            query_errors = 0
            if result.action == "burst":
                cuda_errors = 0
            if cycle == 1 or cycle % config.log_every_cycles == 0:
                print(
                    f"{PREFIX} cycle={cycle} gpu={initial_gpu.uuid} "
                    f"cadence_jitter={cycle_jitter:.3f}s samples={result.samples} action={result.action} "
                    f"burst_elapsed={result.burst_elapsed_seconds:.3f}s ops={result.operations}",
                    flush=True,
                )
            if not ready_published:
                marker = publish_ready(
                    args.ready_dir,
                    initial_gpu,
                    config,
                    burster.low_priority,
                    jitter_seed,
                )
                print(f"{PREFIX} READY marker={marker or 'disabled'}", flush=True)
                ready_published = True
        except subprocess.SubprocessError as exc:
            query_errors += 1
            print(
                f"{PREFIX} utilization query error ({query_errors}/{config.max_consecutive_query_errors}): {exc}",
                file=sys.stderr,
                flush=True,
            )
            if query_errors >= config.max_consecutive_query_errors:
                print(f"{PREFIX} persistent utilization query failures; exiting", file=sys.stderr, flush=True)
                return 2
        except Exception as exc:
            cuda_errors += 1
            print(
                f"{PREFIX} guard cycle error ({cuda_errors}/{config.max_consecutive_cuda_errors}): {exc}",
                file=sys.stderr,
                flush=True,
            )
            if cuda_errors >= config.max_consecutive_cuda_errors:
                print(f"{PREFIX} persistent guard failures; exiting", file=sys.stderr, flush=True)
                return 3

        remaining = config.interval_seconds - (time.monotonic() - cycle_started)
        if remaining > 0:
            stop_event.wait(remaining)

    print(f"{PREFIX} STOP cycles={cycle}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
