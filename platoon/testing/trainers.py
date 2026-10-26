"""The conformance check of a trainer (RFC 0021): :func:`check_trainer`.

It runs the trainer through RFC 0021's lifecycle: built from its setup, entered with ``async with``, and run (``platoon
train`` itself still builds the rl trainer with its constructor and calls its ``train()``, which runs the same
lifecycle).  It runs on CPU: on a fake Tinker service (:class:`FakePolicyService`), whose training clients step nothing
and save to ``tinker://`` paths of their own while the run's ``checkpoints.jsonl`` gets its rows as in a real run, under
a model gateway that samples nothing, and, for a trainer that rolls out (``needs().rollouts``), over 16 training tasks
with a workflow whose every group is one datum with a nonzero advantage.  It is made of named checks (:data:`CHECKS`),
run on three starts:

- ``trains``: a fresh run of two steps saves its launch checkpoint at step 0 and its final one at step 2, each row with
  the run's counters, its policy version and optimizer steps never going back;
- ``interrupts``: a run of four steps that a stop request reaches in its third optimizer step (``SIGUSR2``, which
  ``platoon train`` passes on to the start it runs) ends at a step boundary, and saves ``<step>_interrupted`` there;
- ``resumes``: the next start resumes from that row: its launch checkpoint has the row's step, policy version and
  optimizer steps, and it trains to the run's end, whose final row holds every key the interrupted row held.

The experimental trainer API (``platoon.experimental``) can change in a minor release, and this check changes with it.
"""

from __future__ import annotations

import asyncio
import json
import signal
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest import mock

if TYPE_CHECKING:
    from platoon.train.trainer.api import Trainer

# The checks, by name, in the order they run.
CHECKS = ("trains", "interrupts", "resumes")
# The run's counters, which every row of checkpoints.jsonl holds.
COUNTERS = ("batch", "wandb_run_id", "policy_version", "optimizer_steps")
# The optimizer step in which the interrupted start's stop request comes.
STOP_AT = 3
_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2)


def check_trainer(trainer: type[Trainer], *, options: Any = None) -> None:
    """Check a trainer class against RFC 0021's lifecycle (:data:`CHECKS`); raise ``AssertionError`` naming every
    check it fails, with what it saw.  ``options`` are the trainer's options, as its setup gives them.  It installs
    the signal handlers a start installs, and puts back the ones it found; call it outside a running event loop."""

    from platoon.utils.scratch import temporary_directory

    with temporary_directory(prefix="platoon-check-trainer-") as directory, _restored_signals():
        failures = _run_checks(trainer, options, Path(directory))
    if failures:
        lines = [f"{name}: {what}" for name, found in failures.items() for what in found]
        name = getattr(trainer, "__qualname__", repr(trainer))
        raise AssertionError(f"trainer {name} fails {len(failures)} check(s):\n  " + "\n  ".join(lines))


def _run_checks(trainer: type[Trainer], options: Any, directory: Path) -> dict[str, list[str]]:
    """Each check's problems, by name; the checks after one whose start fails do not run."""

    failures: dict[str, list[str]] = {}
    fresh = _start(trainer, options, directory / "fresh", steps=2)
    failures.update(_found("trains", fresh, _trains))
    stopped = _start(trainer, options, directory / "resumed", steps=4, stop_at=STOP_AT)
    failures.update(_found("interrupts", stopped, _interrupts))
    if "interrupts" not in failures:
        resumed = _start(trainer, options, directory / "resumed", steps=4)
        failures.update(_found("resumes", resumed, lambda rows: _resumes(stopped.rows, rows)))
    return failures


def _found(name: str, start: _Start, check: Any) -> dict[str, list[str]]:
    problems = [f"the start failed: {start.error}"] if start.error is not None else check(start.rows)
    return {name: problems} if problems else {}


# --- the checks ------------------------------------------------------------------------------------------------------


def _trains(rows: list[dict[str, Any]]) -> list[str]:
    problems = _counted(rows)
    if [(row.get("name"), row.get("batch")) for row in rows] != [("recovery-000000", 0), ("final", 2)]:
        problems.append(f"saved {_names(rows)}, not the launch checkpoint at step 0 and the final one at step 2")
    return problems


def _interrupts(rows: list[dict[str, Any]]) -> list[str]:
    last = rows[-1] if rows else {}
    step = last.get("batch")
    if not isinstance(step, int) or last.get("name") != f"{step:06d}_interrupted":
        return [f"saved {_names(rows)}: its last row is not <step>_interrupted at the step it stopped at"]
    if not 0 < step < 4:
        return [f"stopped at step {step}, not at a step boundary inside the run"]
    return _counted(rows)


def _resumes(before: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[str]:
    stopped, problems = before[-1], _counted(rows)
    launch = rows[0] if rows else {}
    resumed = {key: launch.get(key) for key in ("batch", "policy_version", "optimizer_steps")}
    if resumed != {key: stopped[key] for key in resumed}:
        problems.append(f"launched at {resumed}, not where {stopped['name']} left the run")
    final = rows[-1] if rows else {}
    if (final.get("name"), final.get("batch")) != ("final", 4):
        problems.append(f"saved {_names(rows)}, which ends without the final checkpoint at step 4")
    elif not final["optimizer_steps"] > stopped["optimizer_steps"]:
        problems.append(f"took no optimizer step after {stopped['name']}")
    if missing := sorted(set(stopped) - set(final)):
        problems.append(f"its final row lacks {missing}, which {stopped['name']} held")
    return problems


def _counted(rows: list[dict[str, Any]]) -> list[str]:
    """That each row holds the run's counters, its policy version and optimizer steps never going back."""

    problems = [
        f"{row.get('name')} lacks {missing}" for row in rows if (missing := [k for k in COUNTERS if k not in row])
    ]
    for key in ("policy_version", "optimizer_steps"):
        values = [row.get(key) for row in rows]
        if not all(isinstance(value, int) for value in values) or values != sorted(int(value or 0) for value in values):
            problems.append(f"{key} goes {values} over {_names(rows)}")
    return problems


def _names(rows: list[dict[str, Any]]) -> list[Any]:
    return [row.get("name") for row in rows]


# --- a start ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Start:
    """A start's rows of the run's ``checkpoints.jsonl``, and the error it failed with (None: it ended)."""

    rows: list[dict[str, Any]]
    error: BaseException | None


def _start(trainer: type[Trainer], options: Any, root: Path, *, steps: int, stop_at: int | None = None) -> _Start:
    """One start of the trainer, built from its setup and run, on a fake service that raises a stop request in its
    ``stop_at``-th optimizer step; the rows it added to the run's checkpoints.jsonl."""

    from platoon.train.components import task_ids_to_dataset
    from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.train.policy.spec import policy_specs
    from platoon.train.trainer.api import TaskSources, TrainerSetup, Workflows
    from platoon.utils.config import build_config

    config = build_config(PlatoonTinkerRLTrainerConfig, _settings(root, steps))
    run_dir = Path(config.log_path, config.stats.experiment_name, config.stats.trial_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    before = len(_rows(run_dir))
    rolls_out = trainer.needs(config, options).rollouts
    setup = TrainerSetup(
        config=config,
        policy_specs=policy_specs(config),
        options=options,
        tasks=TaskSources(train_dataset=task_ids_to_dataset([f"task-{index}" for index in range(16)]))
        if rolls_out
        else None,
        workflow_builder=lambda policies: Workflows(_OneDatumGroups(), None),
    )
    error = None
    with _faked(FakePolicyService(stop_at=stop_at)):
        try:
            asyncio.run(_run(trainer.from_setup(setup)))
        except Exception as exc:  # noqa: BLE001 - a check reports what failed
            error = exc
    return _Start(rows=_rows(run_dir)[before:], error=error)


async def _run(trainer: Any) -> None:
    async with trainer:
        await trainer.run()


def _settings(root: Path, steps: int) -> dict[str, Any]:
    """A run of ``steps`` steps of two groups each, with no evaluation, no watchdog and no W&B."""

    return {
        "train": {
            "model_name": "fake-model",
            "renderer_name": "fake-renderer",
            "batch_size": 2,
            "max_training_steps": steps,
            "max_concurrent_rollouts": 16,
        },
        "eval": {"every": 0},
        "watchdog": {"enabled": False},
        "stats": {"wandb": {"mode": "disabled"}},
        "log_path": str(root / "logs"),
    }


def _rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "checkpoints.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [json.loads(line) for line in lines if line]


@contextmanager
def _restored_signals() -> Generator[None]:
    saved = {number: signal.getsignal(number) for number in _SIGNALS}
    try:
        yield
    finally:
        for number, handler in saved.items():
            signal.signal(number, handler)


# --- the fakes -------------------------------------------------------------------------------------------------------


@contextmanager
def _faked(service: FakePolicyService) -> Generator[None]:
    """``service`` in place of the Tinker service, and a model gateway that samples nothing in place of the one a
    trainer registers for its policy."""

    import tinker

    from platoon.train.policy import set as set_module

    with (
        mock.patch.object(tinker, "ServiceClient", service.service_client),
        mock.patch.object(set_module, "register_tinker_llm", _registered),
    ):
        yield


def _registered(model_name: str, renderer_name: str, **kwargs: Any) -> Any:
    """A policy's model info whose gateway samples nothing, and counts the policy's versions."""

    return SimpleNamespace(
        llm=_FakeLLM(model_name),
        model_name=model_name,
        renderer_name=renderer_name,
        renderer_kwargs=kwargs.get("renderer_kwargs") or {},
        renderer=None,
        tokenizer=None,
    )


class _FakeLLM:
    def __init__(self, model_name: str) -> None:
        self.model_name, self.version = model_name, 0

    def set_version(self, version: int) -> None:
        self.version = version

    def update_sampling_client(self, sampling_client: Any, increment_version: bool = True) -> None:
        if increment_version:
            self.version += 1


class FakePolicyService:
    """A Tinker service on CPU, for one start: its training clients step nothing and save to ``tinker://fake/...``
    paths, while ``tinker_cookbook`` records each full save in the run's ``checkpoints.jsonl`` as it does for a real
    service.  ``stop_at``: the optimizer step, counted over the start, in which it raises ``SIGUSR2``, the stop request
    ``platoon train`` passes on.  ``calls`` lists what was asked of it, in order."""

    def __init__(self, *, stop_at: int | None = None) -> None:
        self.stop_at = stop_at
        self.optimizer_steps = 0
        self.calls: list[str] = []

    def service_client(self, **kwargs: Any) -> FakePolicyService:
        """``tinker.ServiceClient(base_url=...)``: this service."""

        return self

    async def create_lora_training_client_async(self, base_model: str, rank: int = 32, **kwargs: Any) -> Any:
        return self._open(f"create_lora_training_client {base_model}")

    async def create_training_client_from_state_with_optimizer_async(self, path: str, **kwargs: Any) -> Any:
        return self._open(f"create_training_client_from_state_with_optimizer {path}")

    async def create_training_client_from_state_async(self, path: str, **kwargs: Any) -> Any:
        return self._open(f"create_training_client_from_state {path}")

    def _open(self, call: str) -> _FakeTrainingClient:
        self.calls.append(call)
        return _FakeTrainingClient(self)


class _Done:
    """A Tinker call's future, already done."""

    def __init__(self, value: Any) -> None:
        self.value = value

    async def result_async(self) -> Any:
        return self.value


class _FakeTrainingClient:
    def __init__(self, service: FakePolicyService) -> None:
        self.service = service

    async def forward_backward_async(self, datums: list[Any], loss_fn: str, loss_fn_config: Any = None) -> _Done:
        import tinker
        import torch

        self.service.calls.append(f"forward_backward {len(datums)}")
        outputs = [
            {"logprobs": tinker.TensorData.from_torch(torch.zeros(len(datum.model_input.to_ints())))}
            for datum in datums
        ]
        return _Done(SimpleNamespace(loss_fn_outputs=outputs, metrics={}))

    async def optim_step_async(self, params: Any) -> _Done:
        service = self.service
        service.optimizer_steps += 1
        service.calls.append("optim_step")
        if service.optimizer_steps == service.stop_at:
            signal.raise_signal(signal.SIGUSR2)
        return _Done(SimpleNamespace(metrics={}))

    async def save_state_async(self, name: str, ttl_seconds: int | None = None) -> _Done:
        self.service.calls.append(f"save_state {name}")
        return _Done(SimpleNamespace(path=f"tinker://fake/weights/{name}"))

    async def save_weights_for_sampler_async(self, name: str, ttl_seconds: int | None = None) -> _Done:
        return _Done(SimpleNamespace(path=f"tinker://fake/sampler_weights/{name}"))

    async def save_weights_and_get_sampling_client_async(self) -> str:
        self.service.calls.append("save_weights_and_get_sampling_client")
        return f"sampler:fake/weights-only-{self.service.optimizer_steps}"

    def create_sampling_client(self, path: str) -> str:
        return f"sampler:{path}"


class _OneDatumGroups:
    """A training workflow whose every group is one datum with a nonzero advantage."""

    async def arun_episode(self, data: dict) -> list[Any]:
        import tinker
        import torch

        await asyncio.sleep(0)
        inputs = {"target_tokens": [2, 3, 4], "logprobs": [0.0, -0.1, -0.2], "advantages": [0.0, 1.0, 1.0]}
        inputs["mask"] = [0.0, 1.0, 1.0]
        return [
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints([1, 2, 3]),
                loss_fn_inputs={
                    key: tinker.TensorData.from_torch(torch.tensor(value)) for key, value in inputs.items()
                },
            )
        ]
