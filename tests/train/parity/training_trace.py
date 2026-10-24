"""Runs the golden trace's scenarios (``trace_scenarios.SCENARIOS``) on the fakes, in virtual time, and compares what
the trainer did with each scenario's golden files (:func:`trace_scenario`).  Two runs of a scenario must agree before
either is compared, and the scenario's checks must pass."""

from __future__ import annotations

import asyncio
import copy
import gzip
import hashlib
import json
import logging
import os
import re
import selectors
import signal
import sys
import time
import types
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tinker
import trace_fakes
import yaml
from trace_fakes import FakeRun, ServiceGone, Start, get_task, unused_rollout
from trace_format import (
    GOLDEN_DIR,
    Masks,
    OneLine,
    Trace,
    describe_difference,
    golden_names,
    metric_tables,
    render,
    to_data,
)
from trace_scenarios import BASE_CONFIG, SCENARIOS, Scenario, check_trains

from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.train.components import task_ids_to_dataset
from platoon.train.labels import label_column
from platoon.train.task_stats import EVENTS_NAME, TASKS_DIR
from platoon.train.tinker import proxy
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig, training_problems
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer
from platoon.train.tinker.rollout_metrics import reset_monitors_for_tests
from platoon.train.tinker.train import arun_tinker_training
from platoon.train.tinker.workers import run as worker_run
from platoon.train.tinker.workers import telemetry
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.utils import stats_tracker
from platoon.utils.config import resolve_config
from platoon.utils.stats_logger import StatsLogger

# The command that records a scenario's golden files again, as a failure names it.
RECORD = "uv run python tests/train/parity/record_golden.py --write"
# A run whose virtual clock passes this has stopped making progress (timers that re-arm forever).
VIRTUAL_TIME_LIMIT = 1e6
# A key the Tinker client would take: platoon train checks for one before it opens anything.  The fakes never send it.
FAKE_TINKER_KEY = "tml-golden-trace"
# Where a real Tinker client that escaped the fakes would connect: nothing listens there.
UNREACHABLE_TINKER = "http://127.0.0.1:9"


# --------------------------------------------------------------------------- virtual time


class _AutojumpSelector(selectors.DefaultSelector):  # type: ignore[misc, valid-type]
    """A selector that never blocks: with nothing ready, the clock jumps by the time the loop would wait."""

    def __init__(self, clock: SimpleNamespace) -> None:
        super().__init__()
        self._clock = clock

    def select(self, timeout: float | None = None):  # type: ignore[override]
        events = super().select(0)
        if events or timeout == 0:
            return events
        if timeout is None:
            raise RuntimeError(
                "the run's event loop has nothing scheduled and nothing ready: a deadlock, or work in a thread of its "
                "own, which a trace cannot order (VirtualTimeLoop runs run_in_executor's work inline)"
            )
        self._clock.now += timeout
        if self._clock.now > VIRTUAL_TIME_LIMIT:
            raise RuntimeError(f"the run's virtual clock passed {VIRTUAL_TIME_LIMIT:g} s: it does not finish")
        return events


class VirtualTimeLoop(asyncio.SelectorEventLoop):  # type: ignore[misc, valid-type]
    """An event loop on a virtual clock that jumps to the next timer whenever nothing is ready, and that runs
    ``run_in_executor``'s work (``asyncio.to_thread``'s) inline: a run on it is a function of its code alone."""

    def __init__(self) -> None:
        self._clock = SimpleNamespace(now=0.0)
        super().__init__(_AutojumpSelector(self._clock))

    def time(self) -> float:
        return self._clock.now

    def run_in_executor(self, executor: Any, func: Callable[..., Any], *args: Any) -> asyncio.Future:
        future = self.create_future()
        try:
            future.set_result(func(*args))
        except BaseException as exc:  # noqa: BLE001 - delivered to the awaiting caller, as an executor delivers it
            future.set_exception(exc)
        return future


def virtual_clock(real: Callable[[], float]) -> Callable[[], float]:
    """``real`` (``time.perf_counter``, ``time.monotonic``) on the virtual clock while a :class:`VirtualTimeLoop` runs
    in the calling thread, so that what the trainer measures with it, such as a group's rollout time, which a kept
    group carries into a checkpoint's files, is the same in every run; the real clock anywhere else."""

    def clock() -> float:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return real()
        return loop.time() if isinstance(loop, VirtualTimeLoop) else real()

    return clock


# --------------------------------------------------------------------------- the fakes in place


def _platoon_modules() -> list[types.ModuleType]:
    return [
        module
        for name, module in list(sys.modules.items())
        if module is not None and (name == "platoon" or name.startswith("platoon."))
    ]


def patch_everywhere(patch: pytest.MonkeyPatch, original: Any, replacement: Any) -> None:
    """Replace ``original`` with ``replacement`` in every loaded platoon module that holds it under a name, its own
    module among them, so that the fake reaches whichever module calls it, wherever a change moves that code."""

    for module in _platoon_modules():
        for attribute, value in list(vars(module).items()):
            if value is original:
                patch.setattr(module, attribute, replacement)


def _fakes_left() -> list[str]:
    """The attributes of platoon's modules that hold a fake although no start runs."""

    return [
        f"{module.__name__}.{attribute}"
        for module in _platoon_modules()
        for attribute, value in list(vars(module).items())
        if isinstance(value, types.MethodType) and isinstance(value.__self__, FakeRun)
    ]


class _Messages(logging.Handler):
    """Collects the INFO and higher records of platoon's loggers, masked."""

    def __init__(self, run: FakeRun) -> None:
        super().__init__(logging.INFO)
        self.run = run

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == "platoon" or record.name.startswith("platoon."):
            self.run.messages.append(f"{record.levelname} {self.run.masks.message(record.getMessage())}")


def _reset_process_state() -> None:
    """The process-wide state a start reads, as a fresh process has it: the stats trackers, the rollout client
    monitors and the rollout worker telemetry."""

    stats_tracker.export_all(reset=True)
    reset_monitors_for_tests()
    telemetry.reset_for_tests()


def _escaped_service_client(self: Any, *args: Any, **kwargs: Any) -> None:
    raise AssertionError("the trainer created a real tinker.ServiceClient: the fake service did not reach it")


def _stats_log(run: FakeRun) -> Callable[..., None]:
    def log(logger: StatsLogger, step: int, stats: dict, epoch: Any = None, prefix: str = "", force: bool = False):
        run.log_stats(step, stats, force)

    return log


def _binding_publication(run: FakeRun, publish: Callable[..., None]) -> Callable[..., None]:
    def publish_model_binding(workflow: Any, sampling_client: Any, version: int) -> None:
        run.binding(workflow, sampling_client, version)
        publish(workflow, sampling_client, version)

    return publish_model_binding


@contextmanager
def installed_fakes(run: FakeRun) -> Generator[None]:
    """The fakes in place for one start, its log messages collected and the process-wide state it reads reset; all
    put back afterwards, and nothing left holding a fake."""

    signals = [signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2]
    saved_signals = {number: signal.getsignal(number) for number in signals}
    platoon_logger = logging.getLogger("platoon")
    saved_level = platoon_logger.level
    saved_labels = sandbox_cleanup.labels()
    handler = _Messages(run)
    _reset_process_state()
    sandbox_cleanup.set_labels({})  # a fresh process's: the run adds its own, which its requests carry
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("TINKER_API_KEY", FAKE_TINKER_KEY)
            patch.setenv("TINKER_BASE_URL", UNREACHABLE_TINKER)
            for real in (time.perf_counter, time.monotonic):  # also where a module bound them by name
                clock = virtual_clock(real)
                patch_everywhere(patch, real, clock)
                patch.setattr(time, real.__name__, clock)
            service_client = tinker.ServiceClient  # also where a module bound it by name; the real one never runs
            patch_everywhere(patch, service_client, run.service_client)
            patch.setattr(tinker, "ServiceClient", run.service_client)
            patch.setattr(service_client, "__init__", _escaped_service_client)
            patch_everywhere(patch, proxy.register_tinker_llm, run.register_policy)
            patch_everywhere(patch, worker_run.run_rollout_request, run.rollout)
            patch.setattr(StatsLogger, "log", _stats_log(run))
            publish = GroupRolloutWorkflow.publish_model_binding
            patch.setattr(GroupRolloutWorkflow, "publish_model_binding", _binding_publication(run, publish))
            platoon_logger.addHandler(handler)
            platoon_logger.setLevel(logging.INFO)
            try:
                yield
            finally:
                platoon_logger.removeHandler(handler)
                platoon_logger.setLevel(saved_level)
    finally:
        for number, function in saved_signals.items():
            signal.signal(number, function)
        sandbox_cleanup.set_labels(saved_labels)
        _reset_process_state()
    if left := _fakes_left():
        raise AssertionError(
            f"{', '.join(left)} still hold(s) a fake of the golden trace after its start: the module was first "
            "imported during the start, after the fakes went in place, so nothing put the original back. Import it at "
            "the top of training_trace.py, so that it is loaded before the fakes replace what it holds."
        )


# --------------------------------------------------------------------------- a start


def write_config(scenario: Scenario, root: Path) -> Path:
    """``scenario``'s config, :data:`BASE_CONFIG` with its sections laid over it, written to ``root``."""

    data = _merged(BASE_CONFIG, scenario.config)
    data["log_path"] = str(root / "logs")
    path = root / "train.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _merged(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        both = isinstance(value, Mapping) and isinstance(merged.get(key), Mapping)
        merged[key] = _merged(merged[key], value) if both else copy.deepcopy(value)
    return merged


async def _train(scenario: Scenario, path: Path, config: PlatoonTinkerRLTrainerConfig) -> None:
    """One start of ``scenario``: a config with environments through ``platoon train``'s entry point, any other as a
    train script runs the trainer, on plain datasets and workflows it builds."""

    if "environments" in scenario.config:
        await arun_tinker_training(["--config", str(path)])
        return
    train_dataset = task_ids_to_dataset([f"train:{index}" for index in range(scenario.tasks)])
    eval_ids = [f"val:{index}" for index in range(scenario.eval_tasks)]
    problems = training_problems(config, num_train_tasks=len(train_dataset))
    assert not problems, f"platoon train would refuse the scenario's config: {problems}"
    trainer = PlatoonTinkerRLTrainer(
        config=config, train_dataset=train_dataset, eval_dataset=task_ids_to_dataset(eval_ids) if eval_ids else None
    )

    def workflow(section: Any, scope: str) -> GroupRolloutWorkflow:
        return GroupRolloutWorkflow(
            rollout_fn=unused_rollout,
            get_task_fn=get_task,
            config=section.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope=scope,
            filter_errors=section.workflow_config.filter_errors,
            label_by=label_column(config),
        )

    async with trainer:
        await trainer.train(train_workflow=workflow(config.train, "train"), eval_workflow=workflow(config.eval, "eval"))


def portable_save(row: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """A row of ``checkpoints.jsonl`` as a trace keeps it.  The dispatcher's state names the file of the groups it kept
    by the SHA-256 of its gzip bytes, which differ between Python builds (the gzip header) though the groups do not:
    so the state's own SHA-256 is taken with that digest replaced by the SHA-256 of the groups' uncompressed content."""

    dispatch = row.get("dispatch")
    if not isinstance(dispatch, dict) or not dispatch.get("kept"):
        return row
    state = (run_dir / dispatch["file"]).read_text(encoding="utf-8")
    kept = (run_dir / dispatch["kept"]).read_bytes()
    digest = hashlib.sha256(kept).hexdigest()
    assert digest in state, f"{dispatch['file']} does not name {dispatch['kept']} by its SHA-256"
    state = state.replace(digest, hashlib.sha256(gzip.decompress(kept)).hexdigest())
    return {**row, "dispatch": {**dispatch, "sha256": hashlib.sha256(state.encode()).hexdigest()}}


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []


def run_start(run: FakeRun, scenario: Scenario, path: Path, start: Start) -> dict[str, Any]:
    """One start of ``scenario``'s run from config ``path``, as a fresh process runs it, and what it did."""

    config = resolve_config(path, [], PlatoonTinkerRLTrainerConfig).config
    run_dir = Path(config.log_path, config.stats.experiment_name, config.stats.trial_name)
    checkpoints, events = run_dir / "checkpoints.jsonl", run_dir / TASKS_DIR / EVENTS_NAME
    saved_before, events_before = len(_lines(checkpoints)), len(_lines(events))
    run.begin(start)
    crashed = None
    with installed_fakes(run):
        try:
            asyncio.run(_train(scenario, path, config), loop_factory=VirtualTimeLoop)
        except (ServiceGone, BaseExceptionGroup) as exc:
            if start.crash_after_saves is None:
                raise
            if isinstance(exc, BaseExceptionGroup):
                gone, rest = exc.split(ServiceGone)
                if gone is None:
                    raise
                if rest is not None:  # an error beside the planned one
                    raise rest from None
            crashed = f"crashed: the Tinker service failed the first forward_backward after full save {run.full_saves}"
    stopped = "stopped: SIGUSR2 at {} {}".format(*start.stop_at) if run.stopped and start.stop_at else None
    outcome = crashed or stopped or "completed"
    _check_start(run, start, outcome)
    return {
        "outcome": outcome,
        "tinker": run.calls,
        "requests": {scope: rows for scope, rows in run.requests.items() if rows},
        "saves": [
            to_data(portable_save(json.loads(row), run_dir), run.masks) for row in _lines(checkpoints)[saved_before:]
        ],
        "events": [OneLine(to_data(json.loads(line), run.masks)) for line in _lines(events)[events_before:]],
        "metrics": metric_tables(run.logged),
        "messages": run.messages,
        "files": sorted(str(item.relative_to(run_dir)) for item in run_dir.rglob("*") if item.is_file()),
    }


def _check_start(run: FakeRun, start: Start, outcome: str) -> None:
    """That the start ended as planned, and that the fakes saw it whole: its first call of Tinker opened a training
    client of the fake service, and each step it trained logged its metrics through ``StatsLogger.log``."""

    if start.crash_after_saves is not None and not outcome.startswith("crashed"):
        raise AssertionError(f"the start was to crash after {start.crash_after_saves} full saves, but it did not")
    if start.stop_at is not None and not run.stopped:
        raise AssertionError("the start was to stop at {} {}, but it never made that call".format(*start.stop_at))
    first = run.calls[0][0] if run.calls else None
    assert first is not None and first.startswith("create_"), (
        f"the start's first Tinker call is {first!r}, not one that opens a training client of the fake service: "
        "the trainer reached Tinker some other way than tinker.ServiceClient"
    )
    if outcome != "completed":
        return
    span = next(
        (
            found
            for message in run.messages
            if (found := re.search(r"Starting training from batch (\d+) to (\d+)", message))
        ),
        None,
    )
    assert span is not None, "the start logged no 'Starting training from batch' message"
    logged = sum(not force for _, force, _ in run.logged)
    trained = int(span.group(2)) - int(span.group(1))
    assert logged == trained, (
        f"the start logged {logged} steps' metrics for {trained} steps: StatsLogger.log missed some"
    )


def run_scenario(name: str, root: Path) -> Trace:
    """The trace of scenario ``name``, run in the empty directory ``root``."""

    scenario = SCENARIOS[name]
    root.mkdir(parents=True, exist_ok=True)
    run = FakeRun(scenario.rollouts, Masks(root, trace_fakes.__name__))
    path = write_config(scenario, root)
    starts = [run_start(run, scenario, path, start) for start in scenario.starts]
    return {
        "scenario": name,
        "description": scenario.description,
        "request_settings": [json.loads(text) for text in run.settings],
        "starts": starts,
    }


# --------------------------------------------------------------------------- a scenario against its golden


@dataclass(frozen=True)
class TraceResult:
    """Scenario ``name`` run twice, checked and compared with its golden files (file name to text)."""

    name: str
    files: dict[str, str]
    second: dict[str, str]
    golden: dict[str, str]
    check_error: str | None
    seconds: float

    @property
    def deterministic(self) -> bool:
        return self.files == self.second

    @property
    def matches(self) -> bool:
        return self.golden == self.files

    @property
    def ok(self) -> bool:
        return self.deterministic and self.check_error is None and self.matches

    def write(self, directory: Path, *, second: bool = False) -> Path:
        """Write this run's files (the second run's, with ``second``) to ``directory``, which it returns."""

        directory.mkdir(parents=True, exist_ok=True)
        for file, text in (self.second if second else self.files).items():
            (directory / file).write_text(text, encoding="utf-8")
        return directory

    def problems(self, written: Path | None = None, *, compare: bool = True) -> str:
        """What is wrong, each with what to do: the runs, the check and, with ``compare``, the golden files;
        ``written`` is where :meth:`write` put this run's files."""

        found = []
        if not self.deterministic:
            found.append(
                f"Two runs of {self.name} differ, so its trace depends on more than the code (the wall clock, a "
                "thread, state left in the process): see README.md.\n"
                + describe_difference(self.files, self.second, labels=("first run", "second run"))
            )
        if self.check_error is not None:
            found.append(
                f"{self.name} no longer exercises what it is there for: {self.check_error}. Change its config or the "
                "fakes until it does, rather than recording a golden that tests less."
            )
        if compare and not self.golden:
            found.append(f"{self.name} has no golden files: record them with `{RECORD} {self.name}`.")
        elif compare and not self.matches:
            golden_dir = os.path.relpath(GOLDEN_DIR)
            diff = "; ".join(f"diff -u {golden_dir}/{file} {written}/{file}" for file in golden_names(self.name))
            found.append(
                f"The trace of {self.name} differs from its golden files. If the change of behavior is meant, record "
                f"them again with `{RECORD} {self.name}` and say why in the pull request; if not, the summary below "
                "says where training did something else."
                + (f" The whole diff: {diff}" if written is not None else "")
                + "\n"
                + describe_difference(self.golden, self.files)
            )
        return "\n\n".join(found)


def trace_scenario(name: str, workdir: Path) -> TraceResult:
    """Scenario ``name`` run twice in ``workdir``, checked (its own check, then :func:`check_trains`) and compared with
    its golden files."""

    began = time.perf_counter()
    trace = run_scenario(name, workdir / "first")
    files = render(trace)
    second = render(run_scenario(name, workdir / "second"))
    scenario = SCENARIOS[name]
    check_error = None
    try:
        scenario.check(trace)
        check_trains(trace)
    except Exception as exc:  # noqa: BLE001 - a check that cannot read the trace fails as one that asserts
        check_error = str(exc) if isinstance(exc, AssertionError) and str(exc) else f"{type(exc).__name__}: {exc}"
    golden = {
        file: (GOLDEN_DIR / file).read_text(encoding="utf-8")
        for file in golden_names(name)
        if (GOLDEN_DIR / file).exists()
    }
    return TraceResult(name, files, second, golden, check_error, time.perf_counter() - began)
