"""Drive ``PlatoonTinkerRLTrainer.train`` end to end on CPU, with Tinker, checkpoints and rollouts faked.

:func:`make_trainer` builds the trainer with its constructor, over ``task_ids_to_dataset`` tasks and a
:class:`FakePolicy` (main, as the config declares it, over a model gateway that samples nothing), whose service hands
out the test's training client and whose full saves are recorded by name, not written.  It records the steps the
trainer logs (:class:`StepLog`).  Every rollout group of :class:`Workflow` is one datum with a nonzero advantage, so
each step applies an optimizer update.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import tinker
import torch

from platoon.train.components import task_ids_to_dataset
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.policy.checkpoints import SAVE_CHECKPOINT_TIMEOUT_S, Waiter
from platoon.train.policy.policy import Policy
from platoon.train.policy.set import PolicySet
from platoon.train.policy.spec import MAIN, PolicySpec, policy_specs
from platoon.train.trainer.rl import PlatoonTinkerRLTrainer
from platoon.utils.stats_tracker import get as get_tracker


def datum(first_token: int = 1, *, advantage: float = 1.0, version: int | None = None) -> tinker.Datum:
    """A datum of two action tokens, each with ``advantage``, after ``first_token`` (in the tests that read which rows
    a step trained, the position of the row its group ran); sampled at policy ``version`` (``checkpoint_version``),
    when given."""

    inputs: dict[str, list[float]] = {
        "target_tokens": [2, 3, 4],
        "logprobs": [0.0, -0.1, -0.2],
        "advantages": [0.0, advantage, advantage],
        "mask": [0.0, 1.0, 1.0],
    }
    if version is not None:
        inputs["checkpoint_version"] = [version]
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([first_token, 2, 3]),
        loss_fn_inputs={name: tinker.TensorData.from_torch(torch.tensor(values)) for name, values in inputs.items()},
    )


class Future:
    """A Tinker API future that is already done: its result is ``value``."""

    def __init__(self, value):
        self.value = value

    async def result_async(self):
        return self.value


class TrainingClient:
    """Records the datums of each ``forward_backward`` (``forward_backwards``) and counts optimizer steps;
    ``on_optim_step(count)`` runs as each one is submitted."""

    def __init__(self, on_optim_step=None):
        self.optim_steps = 0
        self.on_optim_step = on_optim_step
        self.forward_backwards: list[list[tinker.Datum]] = []

    def create_sampling_client(self, path):
        return f"sampler:{path}"

    async def forward_backward_async(self, datums, loss_fn, loss_fn_config):
        self.forward_backwards.append(list(datums))
        outputs = [{"logprobs": item.loss_fn_inputs["target_tokens"]} for item in datums]
        return Future(SimpleNamespace(loss_fn_outputs=outputs))

    async def optim_step_async(self, params):
        self.optim_steps += 1
        if self.on_optim_step is not None:
            self.on_optim_step(self.optim_steps)
        return Future(SimpleNamespace(metrics={}))

    async def save_weights_and_get_sampling_client_async(self):
        return "weights-only-sampler"


class Workflow:
    """Each rollout group takes ``delay`` seconds and returns one datum (none with ``learn=False``);
    ``on_start(row, count)`` runs as each one starts (``count`` groups started so far), as in every workflow of the
    fakes."""

    def __init__(self, delay: float = 0.05, learn: bool = True, on_start=None):
        self.delay, self.learn, self.on_start = delay, learn, on_start
        self.started = 0

    async def arun_episode(self, data):
        self.started += 1
        if self.on_start is not None:
            self.on_start(data, self.started)
        await asyncio.sleep(self.delay)
        return [datum()] if self.learn else []


def interrupt_at(count: int, *signals: signal.Signals) -> Callable[[Any, int], None]:
    """An ``on_start`` hook that sends this process ``signals`` (SIGINT when none), in order, as the ``count``-th group
    starts."""

    def hook(row: Any, started: int) -> None:
        if started == count:
            for sig in signals or (signal.SIGINT,):
                signal.raise_signal(sig)

    return hook


class FakeLLM:
    """The model gateway's ``TinkerLLM`` as the trainer uses it: the policy's version, which each new sampler counts."""

    version = 0

    def set_version(self, version):
        self.version = version

    def update_sampling_client(self, client, increment_version=True):
        if increment_version:
            self.version += 1


@dataclass
class FakeModelInfo:
    """A policy's model info whose gateway samples nothing (:class:`FakeLLM`)."""

    llm: FakeLLM = field(default_factory=FakeLLM)
    model_name: str = "m"
    renderer_name: str = "r"
    renderer: Any = None
    tokenizer: Any = None


def named_policy(policy_id: str, *, url: str | None = None) -> Policy:
    """A trained policy ``policy_id`` of a run of several, on the service at ``url`` (None: the run's), over a model
    gateway that samples nothing (:class:`FakeModelInfo`), whose ``llm.version`` is its version: for the tests of the
    parts that read a policy's id, spec and version."""

    info: Any = FakeModelInfo()
    return Policy(PolicySpec(policy_id, model_name="m", renderer_name="r", tinker_base_url=url), info)


class OneClientService:
    """A Tinker service whose every training client, new or resumed, is ``training_client``."""

    def __init__(self, training_client: Any) -> None:
        self.training_client = training_client

    def service_client(self, **kwargs: Any) -> OneClientService:
        return self

    async def create_lora_training_client_async(self, base_model: str, rank: int = 32, **kwargs: Any) -> Any:
        return self.training_client

    async def create_training_client_from_state_with_optimizer_async(self, path: str, **kwargs: Any) -> Any:
        return self.training_client

    async def create_training_client_from_state_async(self, path: str, **kwargs: Any) -> Any:
        return self.training_client


class SaveLog(list):
    """The names of the full checkpoints a policy saved, in order; the ``loop_state`` each one saved, by name; and,
    given ``events``, each save also an event there (``save <name>``) as it starts, for the tests of what a start does
    in which order."""

    def __init__(self, events: list[str] | None = None) -> None:
        super().__init__()
        self.loop_states: dict[str, dict[str, Any]] = {}
        self.events = events

    def append(self, name: str) -> None:
        if self.events is not None:
            self.events.append(f"save {name}")
        super().append(name)


class FakePolicy(Policy):
    """Main, as ``config`` declares it, over a model gateway that samples nothing (``info``): its service clients are
    ``service_client()``'s, and with ``saves`` its full saves are recorded there, without writing the run's
    ``checkpoints.jsonl`` (each awaited under the run's waiter, as a save is); without, they save through
    tinker-cookbook, as a real run's do."""

    def __init__(
        self,
        config: PlatoonTinkerRLTrainerConfig,
        service_client: Callable[[], Any],
        *,
        saves: SaveLog | None = None,
        info: Any = None,
    ) -> None:
        super().__init__(policy_specs(config)[MAIN], info if info is not None else FakeModelInfo())
        self._service_client = service_client
        self.saves = saves

    def new_service_client(self) -> Any:
        return self._service_client()

    async def save(self, name: str, *, log_path: str, loop_state: dict[str, Any], wait: Waiter) -> dict[str, str]:
        if self.saves is None:
            return await super().save(name, log_path=log_path, loop_state=loop_state, wait=wait)
        return await wait(
            self._recorded(name, loop_state), name="save_checkpoint", total_timeout=SAVE_CHECKPOINT_TIMEOUT_S
        )

    async def _recorded(self, name: str, loop_state: dict[str, Any]) -> dict[str, str]:
        assert self.saves is not None
        self.saves.append(name)
        self.saves.loop_states[name] = loop_state
        return {"state_path": f"tinker://m/weights/{name}", "sampler_path": f"tinker://m/sampler_weights/{name}"}


def fake_policies(config: PlatoonTinkerRLTrainerConfig, service_client: Callable[[], Any] | None = None) -> PolicySet:
    """The run's policies of ``config``: one :class:`FakePolicy`, whose sessions ``service_client()`` opens (a service
    whose training client is a new :class:`TrainingClient` when None), and whose saves go through tinker-cookbook."""

    if service_client is None:
        service_client = OneClientService(TrainingClient()).service_client
    return PolicySet([FakePolicy(config, service_client)])


class StepLog(list):
    """The steps a trainer logged, in order; and the stats it logged with each (``stats``), or both together
    (:meth:`rows`).  Every trainer the fakes build logs into one (``logged``)."""

    def __init__(self) -> None:
        super().__init__()
        self.stats: list[dict[str, Any]] = []
        self.wandb_run_id: str | None = None

    def log(self, step: int, stats: dict[str, Any], force: bool = False) -> None:
        self.append(step)
        self.stats.append(dict(stats))

    def rows(self) -> list[tuple[int, dict[str, Any]]]:
        """Each step logged with its stats, ``(step, stats)``, in order."""

        return list(zip(self, self.stats))

    def close(self) -> None:
        pass


def make_trainer(
    config: PlatoonTinkerRLTrainerConfig,
    *,
    tasks: int = 8,
    eval_tasks: int = 0,
    training_client: Any = None,
    service_client: Callable[[], Any] | None = None,
    info: Any = None,
    saves: SaveLog | None = None,
    **trainer_options: Any,
) -> tuple[PlatoonTinkerRLTrainer, SaveLog, StepLog]:
    """``(trainer, saves, logged)``: the trainer, entered as ``async with`` enters it (its stats logger is ``logged``),
    over ``tasks`` training tasks and ``eval_tasks`` evaluation ones, on a :class:`FakePolicy` whose training client is
    ``training_client`` (a new :class:`TrainingClient` when None), or whose sessions ``service_client()`` opens, and
    whose model info is ``info`` (a :class:`FakeModelInfo` when None); the checkpoints it saves (``saves``, a new
    :class:`SaveLog` when None), and the steps it logs, in order.  ``trainer_options`` are the constructor's other
    arguments (``train_dataset`` in place of the ``tasks``, ``train_order``, ...)."""

    saves = saves if saves is not None else SaveLog()
    logged = StepLog()
    if service_client is None:
        service_client = OneClientService(training_client or TrainingClient()).service_client
    options = {
        "train_dataset": task_ids_to_dataset([f"train:{index}" for index in range(tasks)]),
        "eval_dataset": task_ids_to_dataset([f"val:{index}" for index in range(eval_tasks)]) if eval_tasks else None,
        **trainer_options,
    }
    policy = FakePolicy(config, service_client, saves=saves, info=info)
    trainer = PlatoonTinkerRLTrainer(config, policies=PolicySet([policy]), **options)
    enter(trainer, logged)
    return trainer, saves, logged


def enter(trainer: PlatoonTinkerRLTrainer, stats_logger: Any) -> None:
    """What ``async with trainer:`` opens, without W&B or the run's sandbox labels: ``stats_logger`` and the train
    tracker."""

    trainer._stats_logger = stats_logger
    trainer._train_tracker = get_tracker("train")
