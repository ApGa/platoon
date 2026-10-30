"""Runs of two policies that step independently (``train.stepping: independent``), whole starts on CPU: platoon's rl
trainer on two fake Tinker services, main's and the worker's, whose training clients remember their lineage, with a
training workflow whose every group holds a row of each policy, each row tagged with its group.

- :class:`Lineages`: what each policy trained, by the state path it was saved at, across starts: a training client
  opened from a saved state goes on with that state's lineage, so that :meth:`Lineages.trained` says which groups the
  weights a policy ends with have trained on, in order;
- :class:`Services`: the two fake services of one start, over the run's lineages;
- :class:`TaggedGroups`: the workflow, whose rows carry their group's tag as their first token;
- :func:`start`: one start of the rl trainer, ``SETTINGS`` with sections laid over them;
- :func:`rows`, :func:`run_dir`: what the run wrote; :func:`resumed_from`, :func:`shares_put_back`: the commit a resume
  starts from, and the shares it puts back.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest import mock

import tinker
import torch

from platoon.testing import trainers as checks
from platoon.testing.trainers import FakePolicyService, _Done, _FakeTrainingClient
from platoon.train.checkpoints.commits import Commit, newest_commit
from platoon.train.checkpoints.dispatch_state import read_state
from platoon.train.checkpoints.kept_groups import decode_group, read_kept
from platoon.train.components import task_ids_to_dataset
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.credit.api import TrainingSample
from platoon.train.policy.spec import MAIN, agent_bindings, policy_specs
from platoon.train.trainer.api import TaskSources, TrainerSetup, Workflows
from platoon.train.trainer.rl import PlatoonTinkerRLTrainer
from platoon.train.workflow.output import PolicyGroupOutput
from platoon.train.workflow.protocol import OutputReport
from platoon.utils.config import build_config

WORKER_URL = "http://worker:8000"
# Numbers each start's services, so that each start's training runs have names of their own, as on a real service.
_STARTS = itertools.count(1)
# Tags each group the workflow returns, over every start of every test: a row's first token.
_TAGS = itertools.count(1000)


class Lineages:
    """What each saved state of each policy's training runs has trained, by state path: the tags of the rows each
    optimizer step applied, in order."""

    def __init__(self) -> None:
        self.by_path: dict[str, list[int]] = {}


class _LineageClient(_FakeTrainingClient):
    """A fake training client that goes on with the lineage of the state it was opened from (``trained``): each
    optimizer step adds the tags of the rows sent since the last one, and each saved state remembers them."""

    def __init__(self, service: _LineageService, run: str, trained: list[int]) -> None:
        super().__init__(service, run)
        self.lineage_service = service
        self.trained = list(trained)
        self._pending: list[int] = []

    async def forward_backward_async(self, datums: list[Any], loss_fn: str, loss_fn_config: Any = None) -> _Done:
        self._pending.extend(datum.model_input.to_ints()[0] for datum in datums)
        return await super().forward_backward_async(datums, loss_fn, loss_fn_config)

    async def optim_step_async(self, params: Any) -> _Done:
        if delay := self.lineage_service.optim_delay:  # a slow service: its policy falls behind the others
            await asyncio.sleep(delay)
        self.trained.extend(self._pending)
        self._pending.clear()
        return await super().optim_step_async(params)

    async def save_state_async(self, name: str, ttl_seconds: int | None = None) -> _Done:
        if (hold := self.lineage_service.hold) is not None:  # a test holds this save back until it lets it go
            await hold(name)
        done = await super().save_state_async(name, ttl_seconds)
        self.lineage_service.lineages.by_path[done.value.path] = list(self.trained)
        return done


class _LineageService(FakePolicyService):
    """A fake service whose training clients remember their lineage (:class:`_LineageClient`); a test may hold its
    full saves back (``hold``) or slow its optimizer steps down (``optim_delay``)."""

    def __init__(self, lineages: Lineages, *, name: str, stop_at: int | None = None) -> None:
        super().__init__(name=name, stop_at=stop_at)
        self.lineages = lineages
        # What a full save awaits first, given the checkpoint's name (None: nothing).
        self.hold: Callable[[str], Awaitable[None]] | None = None
        # How long each optimizer step takes, in seconds.
        self.optim_delay = 0.0

    def _open(self, call: str) -> _FakeTrainingClient:
        self.calls.append(call)
        opened = len(self.training_clients)
        path = call.split()[-1] if "from_state" in call else None
        trained = self.lineages.by_path.get(path, []) if path is not None else []
        client = _LineageClient(self, self.name if not opened else f"{self.name}-{opened + 1}", trained)
        self.training_clients.append(client)
        return client


class Services:
    """Two fake Tinker services in ``tinker.ServiceClient``'s place, main's (the run's) and the worker's (at its URL),
    over the run's ``lineages``; ``stop_main``: main's optimizer step, counted over the start, in which its service
    asks for a stop (``SIGUSR2``).  Each start's are new: their training runs, and so the paths they save, are named
    apart from any earlier start's."""

    def __init__(self, lineages: Lineages, *, stop_main: int | None = None) -> None:
        start = next(_STARTS)
        self.main = _LineageService(lineages, name=f"main-{start}", stop_at=stop_main)
        self.worker = _LineageService(lineages, name=f"worker-{start}")
        # The training workflow the start built, once it has.
        self.workflow: Any = None

    def service_client(self, **kwargs: Any) -> FakePolicyService:
        service = self.worker if kwargs.get("base_url") == WORKER_URL else self.main
        return service.service_client(**kwargs)


def tagged_row(tag: int, version: int, logprob: float = -0.5) -> tinker.Datum:
    """A row whose first token is its group's ``tag``: two action tokens with an advantage of 1, at ``version``."""

    inputs = {
        "target_tokens": [2, 3, 4],
        "logprobs": [0.0, logprob, logprob],
        "advantages": [0.0, 1.0, 1.0],
        "mask": [0.0, 1.0, 1.0],
        "checkpoint_version": [version],
    }
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([tag, 2, 3]),
        loss_fn_inputs={key: tinker.TensorData.from_torch(torch.tensor(value)) for key, value in inputs.items()},
    )


class TaggedGroups:
    """A training workflow whose every group holds a row of main and a row of the worker, each at its policy's version
    now and tagged with the group (``tags``: every tag it gave, in order); it records each binding the trainer
    publishes for its rollout workers (``bindings``)."""

    binds_policies = True

    def __init__(self, policies: Any) -> None:
        self.policies = policies
        self.tags: list[int] = []
        self.bindings: list[tuple[int, str | None]] = []

    async def arun_episode(self, data: dict) -> PolicyGroupOutput:
        await asyncio.sleep(0)
        tag = next(_TAGS)
        self.tags.append(tag)
        samples = (
            TrainingSample(tagged_row(tag, self.policies[MAIN].version), MAIN, 0),
            TrainingSample(tagged_row(tag, self.policies["worker"].version, -0.875), "worker", 0),
        )
        return PolicyGroupOutput(samples, OutputReport())

    def publish_model_binding(self, sampling_client: Any, version: int, *, policy: str | None = None) -> None:
        self.bindings.append((version, policy))


# main steps on 4 shares, the worker on 1: about four worker steps per main step, its bound grown to match (S7 (m)).
SETTINGS: dict[str, Any] = {
    "train": {
        "model_name": "fake-model",
        "renderer_name": "fake-renderer",
        "batch_size": 4,
        "max_concurrent_rollouts": 32,
        "max_staleness": 1,
        "stepping": "independent",
    },
    "policies": {
        "worker": {
            "model_name": "fake-worker",
            "renderer_name": "fake-renderer",
            "tinker_base_url": WORKER_URL,
            "optimizer": {"learning_rate": 1e-5},
            "max_staleness": 7,
            "batch_size": 1,
            "checkpoint": {"recovery_every": 2},
        }
    },
    "agents": {"roles": {"subagent": {"policy": "worker"}}},
    "checkpoint": {"recovery_every": 1},
    "eval": {"every": 0},
    "watchdog": {"enabled": False},
    "stats": {"wandb": {"mode": "disabled"}},
}


def start(
    root: Path,
    lineages: Lineages,
    *,
    steps: int,
    workflow: Callable[[Any], Any] = TaggedGroups,
    trainer: Any = PlatoonTinkerRLTrainer,
    services: Services | None = None,
    timeout: float | None = None,
    **sections: Any,
) -> tuple[Services, Any]:
    """One start of ``trainer`` on two fake services over ``lineages`` (``services``, else new ones), with ``workflow``
    (built on the trainer's policies), to the plan's ``steps`` (main's), its config ``SETTINGS`` with ``sections`` laid
    over them (a section replaces the one it names whole, but ``train`` and ``policies``, which are merged; the
    worker's entry is merged too); ``TimeoutError`` once it ran ``timeout`` seconds (None: no limit)."""

    data = {**copy.deepcopy(SETTINGS), "log_path": str(root / "logs")}
    for name, section in sections.items():
        if name == "policies":
            section = {policy: {**data["policies"].get(policy, {}), **entry} for policy, entry in section.items()}
        data[name] = {**data.get(name, {}), **section} if name in ("train", "policies") else section
    data["train"] = {**data["train"], "max_training_steps": steps}
    config = build_config(PlatoonTinkerRLTrainerConfig, data)
    built: list[Any] = []

    def workflows(policies: Any) -> Workflows:
        built.append(workflow(policies))
        return Workflows(built[0], None)

    setup = TrainerSetup(
        config=config,
        policy_specs=policy_specs(config),
        agents=agent_bindings(config),
        tasks=TaskSources(train_dataset=task_ids_to_dataset([f"task-{index}" for index in range(16)])),
        workflow_builder=workflows,
    )
    services = services if services is not None else Services(lineages)
    with (
        checks._faked(services.main),
        mock.patch.object(tinker, "ServiceClient", services.service_client),
        checks._restored_signals(),
    ):
        try:
            asyncio.run(asyncio.wait_for(checks._run(trainer.from_setup(setup)), timeout))
        finally:
            services.workflow = built[0] if built else None
    return services, built[0] if built else None


def run_dir(root: Path) -> Path:
    return root / "logs" / "platoon_tinker" / "run"


def rows(path: Path) -> list[dict[str, Any]]:
    """The rows of the ``checkpoints.jsonl`` in ``path``."""

    file = path / "checkpoints.jsonl"
    return [json.loads(line) for line in file.read_text().splitlines()] if file.exists() else []


def final_lineage(lineages: Lineages, path: Path) -> list[int]:
    """What the weights of the newest row in ``path`` trained on, by group tag, in order."""

    return lineages.by_path[rows(path)[-1]["state_path"]]


def resumed_from(root: Path) -> Commit:
    """The commit a resume of the run in ``root`` starts from."""

    commit = newest_commit(str(run_dir(root)), (MAIN, "worker"))
    assert commit is not None
    return commit


def shares_put_back(root: Path, commit: Commit) -> dict[str, list[int]]:
    """The shares each policy trained after the row ``commit`` names for it (consumed, not dropped), by the tag of their
    group, in the order the groups settled: what the crash took from its weights, which a resume from the commit puts
    back."""

    run = str(run_dir(root))
    state = read_state(run, commit.record.extra["dispatch"])
    put_back: dict[str, list[int]] = {MAIN: [], "worker": []}
    for group in read_kept(run, state["kept"]) if state.get("kept") else ():  # none kept: nothing to put back
        output = decode_group(group)[4]
        for policy, share in (group.get("shares") or {}).items():
            if share and share.get("step") is not None and share["step"] >= commit.entries[policy]["step"]:
                if share.get("fate", "consumed") == "consumed":
                    put_back[policy].append(output.samples[0].datum.model_input.to_ints()[0])
    return put_back
