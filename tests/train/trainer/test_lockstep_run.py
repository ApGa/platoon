"""Several policies trained in lockstep, whole starts on CPU: platoon's rl trainer on two fake Tinker services, main's
and the worker's, with a training workflow whose every group holds a row of each policy, each stamped with a version
of its own policy.  Each step trains both policies, each on its own rows and service, and publishes each one it
updated, to the rollout workers too; each checkpoint saves both, the worker first, and the run's row names the
worker's; a second start resumes both from it; a policy without rows does not step, and says so; each policy's rows
are judged stale against its own version and bound; and a failing service fails the step, naming its policy.  A step
hook updates either policy through that policy's own lane and hears both publications, and a failing service in a hook's
update names its policy too; a trainer of its own that trains both through ``run.learner`` saves one checkpoint of
them at each step's end, and passes the trainer check (``check_trainer``)."""

from __future__ import annotations

import asyncio
import copy
import itertools
import json
import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest import mock

import pytest
import tinker
import torch

import platoon.experimental as px
from platoon.registry import register_component
from platoon.testing import trainers as checks
from platoon.testing.trainers import FakePolicyService
from platoon.train.components import task_ids_to_dataset
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.credit.api import TrainingSample
from platoon.train.policy.spec import MAIN, agent_bindings, policy_specs
from platoon.train.trainer.api import TaskSources, TrainerSetup, Workflows
from platoon.train.trainer.rl import PlatoonTinkerRLTrainer
from platoon.train.workflow.output import PolicyGroupOutput
from platoon.train.workflow.protocol import OutputReport
from platoon.utils.config import ConfigError, build_config
from platoon.utils.stats_logger import StatsLogger

WORKER_URL = "http://worker:8000"
# The logprob each policy samples its rows' tokens at, by which a training client's rows show whose they are.
LOGPROBS = {MAIN: -0.25, "worker": -0.875}


# Numbers each start's services, so that each start's training runs have names of their own, as on a real service.
_STARTS = itertools.count(1)


class Services:
    """Two fake Tinker services in ``tinker.ServiceClient``'s place, main's (the run's) and the worker's (at its URL):
    a session opens on the one its ``base_url`` names.  Each start's are new: their training runs, and so the paths
    they save, are named apart from any earlier start's (``main-<n>``, ``worker-<n>``)."""

    def __init__(self) -> None:
        start = next(_STARTS)
        self.main, self.worker = FakePolicyService(name=f"main-{start}"), FakePolicyService(name=f"worker-{start}")

    def service_client(self, **kwargs: Any) -> FakePolicyService:
        service = self.worker if kwargs.get("base_url") == WORKER_URL else self.main
        return service.service_client(**kwargs)


def _row(policy: str, version: int) -> tinker.Datum:
    """A row of ``policy``: two action tokens with an advantage of 1, sampled at its logprob, at ``version``."""

    inputs = {
        "target_tokens": [2, 3, 4],
        "logprobs": [0.0, LOGPROBS[policy], LOGPROBS[policy]],
        "advantages": [0.0, 1.0, 1.0],
        "mask": [0.0, 1.0, 1.0],
        "checkpoint_version": [version],
    }
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={key: tinker.TensorData.from_torch(torch.tensor(value)) for key, value in inputs.items()},
    )


class TwoPolicyGroups:
    """A training workflow whose every group holds a row of main, at main's version now, and a row of the worker, at
    ``worker_version`` of the worker (its version now by default), unless ``worker_rows`` is False; it records each
    binding the trainer publishes for its rollout workers (``bindings``)."""

    binds_policies = True

    def __init__(
        self, policies: Any, *, worker_rows: bool = True, worker_version: Callable[[Any], int] | None = None
    ) -> None:
        self.policies, self.worker_rows = policies, worker_rows
        self.worker_version = worker_version or (lambda worker: worker.version)
        self.bindings: list[tuple[Any, int, str | None]] = []

    async def arun_episode(self, data: dict) -> PolicyGroupOutput:
        await asyncio.sleep(0)
        samples = [TrainingSample(_row(MAIN, self.policies[MAIN].version), MAIN, 0)]
        if self.worker_rows:
            worker = self.policies["worker"]
            samples.append(TrainingSample(_row("worker", self.worker_version(worker)), "worker", 0))
        return PolicyGroupOutput(tuple(samples), OutputReport())

    def publish_model_binding(self, sampling_client: Any, version: int, *, policy: str | None = None) -> None:
        self.bindings.append((sampling_client, version, policy))


SETTINGS: dict[str, Any] = {
    "train": {
        "model_name": "fake-model",
        "renderer_name": "fake-renderer",
        "batch_size": 2,
        "max_concurrent_rollouts": 16,
        "max_staleness": 1,
    },
    "policies": {
        "worker": {
            "model_name": "fake-worker",
            "renderer_name": "fake-renderer",
            "tinker_base_url": WORKER_URL,
            "optimizer": {"learning_rate": 1e-5},
        }
    },
    "agents": {"roles": {"subagent": {"policy": "worker"}}},
    "checkpoint": {"recovery_every": 1},
    "eval": {"every": 0},
    "watchdog": {"enabled": False},
    "stats": {"wandb": {"mode": "disabled"}},
}


def start(
    root: Path, *, steps: int, workflow: Callable[[Any], TwoPolicyGroups] = TwoPolicyGroups, **sections: Any
) -> tuple[Services, TwoPolicyGroups]:
    """One start of the rl trainer on two fake services, with ``workflow`` (built on the trainer's policies), to step
    ``steps``, its config ``SETTINGS`` with ``sections`` laid over them."""

    services, built = start_trainer(PlatoonTinkerRLTrainer, root, steps=steps, workflow=workflow, **sections)
    return services, built[0]


def start_trainer(
    trainer: Any,
    root: Path,
    *,
    steps: int,
    workflow: Callable[[Any], TwoPolicyGroups] = TwoPolicyGroups,
    **sections: Any,
) -> tuple[Services, list[TwoPolicyGroups]]:
    """One start of ``trainer`` as :func:`start`'s; the workflows it built, none for a trainer that rolls out none."""

    data = {**copy.deepcopy(SETTINGS), "log_path": str(root / "logs"), **sections}
    data["train"] = {**data["train"], "max_training_steps": steps}
    config = build_config(PlatoonTinkerRLTrainerConfig, data)
    built: list[TwoPolicyGroups] = []

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
    services = Services()
    with (
        checks._faked(services.main),
        mock.patch.object(tinker, "ServiceClient", services.service_client),
        checks._restored_signals(),
    ):
        asyncio.run(checks._run(trainer.from_setup(setup)))
    return services, built


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (path / "checkpoints.jsonl").read_text().splitlines()]


def logprobs(client: Any) -> set[float]:
    """The logprobs of the action tokens of every row a training client was sent."""

    return {
        round(float(logprob), 3)
        for datum in client.datums
        for logprob, mask in zip(datum.loss_fn_inputs["logprobs"].to_torch(), [0, 1, 1], strict=True)
        if mask
    }


@pytest.fixture
def metrics(monkeypatch) -> list[dict[str, Any]]:
    """The metrics of every step the run logged, in order."""

    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(StatsLogger, "log", lambda self, step, stats, *args, **kwargs: logged.append(dict(stats)))
    return logged


def test_each_step_trains_both_policies_on_their_own_rows_and_each_checkpoint_saves_both(tmp_path, metrics):
    services, _ = start(tmp_path, steps=2)

    [main_client], [worker_client] = services.main.training_clients, services.worker.training_clients
    assert (main_client.optimizer_steps, worker_client.optimizer_steps) == (2, 2)
    assert logprobs(main_client) == {LOGPROBS[MAIN]} and logprobs(worker_client) == {LOGPROBS["worker"]}

    run = tmp_path / "logs" / "platoon_tinker" / "run"
    leader, worker = rows(run), rows(run / "policies" / "worker")
    names = ["recovery-000000", "recovery-000001", "recovery-000002", "final"]
    assert [row["name"] for row in leader] == names and [row["name"] for row in worker] == names
    for row, named in zip(leader, worker, strict=True):
        assert row["policies"]["worker"] == {
            "name": named["name"],
            "state_path": named["state_path"],
            "policy_version": named["policy_version"],
            "optimizer_steps": named["optimizer_steps"],
        }
        assert named["state_path"].startswith(f"tinker://{services.worker.name}/weights/")
    assert [(row["policy_version"], row["optimizer_steps"]) for row in worker] == [(0, 0), (1, 1), (2, 2), (2, 2)]

    # main's metrics keep their names; the worker's go under policies/worker/.
    assert [stats["optim/lr"] for stats in metrics] == [3e-5, 3e-5]
    assert [stats["policies/worker/optim/lr"] for stats in metrics] == [1e-5, 1e-5]
    assert [stats["train/policies/worker/batch/optimizer_updates"] for stats in metrics] == [1.0, 1.0]
    assert [stats["train/policies/worker/policy/version"] for stats in metrics] == [1.0, 2.0]

    # A second start resumes both from the run's last checkpoint, each from its own row.
    again, _ = start(tmp_path, steps=3)
    resumed = "create_training_client_from_state_with_optimizer tinker://{}/weights/final"
    assert again.main.calls[0] == resumed.format(services.main.name)
    assert again.worker.calls[0] == resumed.format(services.worker.name)
    [main_client], [worker_client] = again.main.training_clients, again.worker.training_clients
    assert (main_client.optimizer_steps, worker_client.optimizer_steps) == (1, 1)
    assert [row["batch"] for row in rows(run)[len(names) :]] == [2, 3, 3]
    assert [(row["policy_version"], row["batch"]) for row in rows(run / "policies" / "worker")[-1:]] == [(3, 3)]


def test_each_policy_publishes_its_samplers_to_the_rollout_workers_as_the_agents_bind_it(tmp_path):
    _, workflow = start(tmp_path / "main", steps=1)
    first, rest = workflow.bindings[:2], workflow.bindings[2:]
    # Before the first rollout: the agents' default, main, as their default binding, then the worker's own.
    assert [(version, policy) for _, version, policy in first] == [(0, None), (0, "worker")]
    # Each step: each policy it updated, main's as the default.
    assert [(version, policy) for _, version, policy in rest] == [(1, None), (1, "worker")]

    roots = {"policy": "worker", "roles": {"lead": {"policy": MAIN}}}
    _, workflow = start(tmp_path / "worker", steps=1, agents=roots)
    assert [(version, policy) for _, version, policy in workflow.bindings] == [
        (0, None),  # the root agents' policy, the worker, is the agents' default
        (0, MAIN),
        (1, MAIN),
        (1, None),
    ]


def test_a_policy_without_rows_in_a_step_does_not_step_and_says_so(tmp_path, metrics, caplog):
    with caplog.at_level(logging.INFO):
        services, _ = start(tmp_path, steps=2, workflow=lambda policies: TwoPolicyGroups(policies, worker_rows=False))

    assert [client.optimizer_steps for client in services.main.training_clients] == [2]
    assert [client.optimizer_steps for client in services.worker.training_clients] == [0]
    assert [stats["train/policies/worker/batch/skipped_optimizer"] for stats in metrics] == [1.0, 1.0]
    assert [stats["train/batch/skipped_optimizer"] for stats in metrics] == [0.0, 0.0]
    assert "Batch 0 (step 1) takes no optimizer update of policy worker: it had no datums to train on" in caplog.text
    final = rows(tmp_path / "logs" / "platoon_tinker" / "run")[-1]
    assert final["policies"]["worker"]["policy_version"] == 0  # it published nothing: its rows do not age
    how_to = (Path(__file__).parents[3] / "docs" / "how-to" / "train-and-eval.md").read_text()
    assert "(`train/policies/<id>/batch/skipped_optimizer`)" in how_to  # the name the docs give is the one logged


def test_each_policy_s_rows_are_judged_stale_against_its_own_version_and_bound(tmp_path, metrics, caplog):
    def old_worker_rows(policies: Any) -> TwoPolicyGroups:
        return TwoPolicyGroups(policies, worker_version=lambda worker: 0)  # every worker row at its first version

    with caplog.at_level(logging.INFO):
        services, _ = start(tmp_path, steps=3, workflow=old_worker_rows, policies=_worker(max_staleness=0))

    # The worker steps once, after which its rows are a version old, beyond its bound of 0; main's train on.
    assert [client.optimizer_steps for client in services.worker.training_clients] == [1]
    assert [client.optimizer_steps for client in services.main.training_clients] == [3]
    assert "Stale rollout detected in batch 1 of policy worker: policy version 1, rollout version 0" in caplog.text
    assert [stats.get("train/policies/worker/stale_rollouts", 0.0) for stats in metrics] == [0.0, 1.0, 1.0]
    assert [stats["train/batch/stale_groups"] for stats in metrics] == [0.0, 0.0, 0.0]  # main's rows still trained


class LateWorkerRows(TwoPolicyGroups):
    """Groups whose worker rows appear only once main has published twice, each at the worker's version then."""

    async def arun_episode(self, data: dict) -> PolicyGroupOutput:
        self.worker_rows = self.policies[MAIN].version >= 2
        return await super().arun_episode(data)


@pytest.mark.parametrize("stale_filter", ["group", "datum"])
def test_a_policy_s_rows_are_never_judged_against_another_policy_s_version(tmp_path, caplog, stale_filter):
    """main steps alone at first, so that its version stays two ahead of the worker's; the worker's rows, within its
    own bound (1) of its own version, train, by either filter: judged against main's version they would be stale."""

    train = {**SETTINGS["train"], "stale_filter": stale_filter}
    with caplog.at_level(logging.INFO):
        services, _ = start(tmp_path, steps=6, workflow=LateWorkerRows, train=train, policies=_worker(max_staleness=1))

    assert services.worker.training_clients[0].optimizer_steps > 0
    assert "Stale" not in caplog.text  # no stale rows, none masked


def _worker(**settings: Any) -> dict[str, Any]:
    return {"worker": {**SETTINGS["policies"]["worker"], **settings}}


def _errors(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [found for inner in error.exceptions for found in _errors(inner)]
    return [error]


def test_a_failing_service_fails_the_step_naming_its_policy(tmp_path, monkeypatch):
    forward_backward = checks._FakeTrainingClient.forward_backward_async

    async def failing(self: Any, datums: list[Any], loss_fn: str, loss_fn_config: Any = None) -> Any:
        if self.run.startswith("worker"):
            raise RuntimeError("503 Service Unavailable")
        return await forward_backward(self, datums, loss_fn, loss_fn_config)

    monkeypatch.setattr(checks._FakeTrainingClient, "forward_backward_async", failing)
    with pytest.raises(BaseException) as raised:
        start(tmp_path, steps=2)

    [error] = [error for error in _errors(raised.value) if "503" in str(error)]
    assert error.__notes__ == [f"policy worker, on {WORKER_URL}: forward_backward failed"]
    assert [row["name"] for row in rows(tmp_path / "logs" / "platoon_tinker" / "run")] == ["recovery-000000"]


def test_a_resume_that_finds_no_checkpoint_every_policy_still_has_stops_as_a_configuration_error(tmp_path):
    start(tmp_path, steps=1)
    worker_rows = tmp_path / "logs" / "platoon_tinker" / "run" / "policies" / "worker" / "checkpoints.jsonl"
    worker_rows.rename(worker_rows.with_suffix(".gone"))  # every row the run's checkpoints name is gone

    # A configuration error (exit 78), which no restart could mend, and which says what to do.
    with pytest.raises(ConfigError, match="Restore the rows each names .* or train a new trial"):
        start(tmp_path, steps=2)


class WorkerRowsOnly(TwoPolicyGroups):
    """A training workflow whose every group holds rows of the worker only: main has none to step on."""

    async def arun_episode(self, data: dict) -> PolicyGroupOutput:
        await asyncio.sleep(0)
        worker = self.policies["worker"]
        return PolicyGroupOutput((TrainingSample(_row("worker", worker.version), "worker", 0),), OutputReport())


@pytest.mark.parametrize(
    ("failing", "checkpoint", "doing"),
    [
        # Each checkpoint saves the worker before main, alone.
        ("save_state_async", {"recovery_every": 1}, "saving checkpoint recovery-000001"),
        # A step in which the worker stepped alone saves its weights alone.
        ("save_weights_and_get_sampling_client_async", {}, "saving its weights"),
    ],
)
def test_a_failing_save_of_the_one_policy_that_saves_names_it(tmp_path, monkeypatch, failing, checkpoint, doing):
    saving = getattr(checks._FakeTrainingClient, failing)

    async def failing_save(self: Any, *args: Any, **kwargs: Any) -> Any:
        if self.run.startswith("worker") and (not args or args[0] == "recovery-000001"):
            raise RuntimeError("503 worker save")
        return await saving(self, *args, **kwargs)

    monkeypatch.setattr(checks._FakeTrainingClient, failing, failing_save)
    with pytest.raises(BaseException) as raised:
        start(tmp_path, steps=2, workflow=WorkerRowsOnly, checkpoint=checkpoint)

    [error] = [error for error in _errors(raised.value) if "503" in str(error)]
    assert error.__notes__ == [f"policy worker, on {WORKER_URL}: {doing} failed"]


def _deleting_rest_client(service: FakePolicyService) -> Any:
    """A fake service's REST client, whose checkpoint deletion deletes every path it is asked to."""

    async def delete(path: str) -> None:
        await asyncio.sleep(0)

    return SimpleNamespace(delete_checkpoint_from_tinker_path_async=delete)


def kept_recovery(path: Path) -> list[str]:
    """The recovery checkpoints of a policy's rows (in ``path``) that its pruner did not delete, by name."""

    deleted_file = path / "checkpoints_deleted.jsonl"
    lines = deleted_file.read_text().splitlines() if deleted_file.exists() else []
    deleted = {json.loads(line)["path"] for line in lines}
    return [
        row["name"] for row in rows(path) if row["name"].startswith("recovery-") and row["state_path"] not in deleted
    ]


@pytest.mark.parametrize(
    ("workflow", "kept"),
    [
        (TwoPolicyGroups, {MAIN: ["recovery-000006"], "worker": ["recovery-000006"]}),
        # A policy that never steps samples its launch's checkpoint throughout, and keeps it with its newest; its
        # checkpoints' unchanged weights, which nothing samples, go as any older checkpoint does.
        (
            lambda policies: TwoPolicyGroups(policies, worker_rows=False),
            {MAIN: ["recovery-000006"], "worker": ["recovery-000000", "recovery-000006"]},
        ),
        (WorkerRowsOnly, {MAIN: ["recovery-000000", "recovery-000006"], "worker": ["recovery-000006"]}),
    ],
    ids=["both-step", "worker-stalls", "main-stalls"],
)
def test_a_policy_that_does_not_step_keeps_no_more_recovery_checkpoints_than_keep_recovery(
    tmp_path, monkeypatch, workflow, kept
):
    """keep_recovery 1, a recovery checkpoint at every step and bounds of 0: each policy keeps its newest recovery
    checkpoint and the one its rollouts may still sample, whether it steps or not, and a leader that does not step
    keeps no other policy's checkpoints from being pruned."""

    monkeypatch.setattr(FakePolicyService, "create_rest_client", _deleting_rest_client, raising=False)
    train = {**SETTINGS["train"], "max_staleness": 0}
    start(
        tmp_path,
        steps=6,
        workflow=workflow,
        train=train,
        policies=_worker(max_staleness=0),
        checkpoint={"recovery_every": 1, "keep_recovery": 1},
    )

    run = tmp_path / "logs" / "platoon_tinker" / "run"
    assert {MAIN: kept_recovery(run), "worker": kept_recovery(run / "policies" / "worker")} == kept


# --- step hooks ------------------------------------------------------------------------------------------------------


class _Scored:
    """Cross entropy, which the fake service takes as any loss, with a metric of its own, by which a learner's
    metrics show where they go."""

    loss_fn = "cross_entropy"
    loss_fn_config: dict[str, Any] = {}
    side_channels: frozenset[str] = frozenset()

    def metrics(self, originals: list[Any], result: Any) -> dict[str, float]:
        return {"scored": 1.0}


class WorkerUpdates(px.StepHookBase):
    """A step hook that adds one update of the worker at each step's end, and records what each step's end published
    (``heard``)."""

    name = "worker-updates"
    heard: ClassVar[list[tuple[str, int, dict[str, int]]]] = []

    async def update(self, end: px.StepEnd) -> None:
        await end.train("worker", [checks._probe_datum()], objective=_Scored())

    async def published(self, end: px.StepEnd) -> None:
        self.heard.append((end.kind, end.step, dict(end.published)))


register_component("step_hook", "test-lockstep-worker-updates", WorkerUpdates, exist_ok=True)


def test_a_step_hook_updates_another_policy_in_its_own_lane_and_hears_every_policy_s_publication(
    tmp_path, metrics, monkeypatch
):
    monkeypatch.setattr(WorkerUpdates, "heard", [])
    hooked = {"hooks": ["test-lockstep-worker-updates"]}
    services, workflow = start(
        tmp_path,
        steps=2,
        workflow=lambda policies: TwoPolicyGroups(policies, worker_rows=False),
        trainer_options=hooked,
    )

    # The worker has no rows: the hook's update alone steps it, on its own service, once a step.
    assert [client.optimizer_steps for client in services.worker.training_clients] == [2]
    assert [client.optimizer_steps for client in services.main.training_clients] == [2]
    # Each policy that the step or the hook updated publishes once, to the rollout workers too, and the hook hears
    # both, as it heard both at the launch.
    assert WorkerUpdates.heard == [
        ("launch", 0, {MAIN: 0, "worker": 0}),
        ("step", 0, {MAIN: 1, "worker": 1}),
        ("step", 1, {MAIN: 2, "worker": 2}),
    ]
    assert [(version, policy) for _, version, policy in workflow.bindings] == [
        (0, None),
        (0, "worker"),
        (1, None),
        (1, "worker"),
        (2, None),
        (2, "worker"),
    ]
    # Counted in the worker's lane: the step's own updates of it, the hook's, and the hook learner's metrics.
    assert [stats["train/policies/worker/batch/optimizer_updates"] for stats in metrics] == [0.0, 0.0]
    assert [stats["train/policies/worker/batch/hook_optimizer_updates"] for stats in metrics] == [1.0, 1.0]
    assert [stats["train/policies/worker/optim/updates_total"] for stats in metrics] == [1.0, 2.0]
    assert [stats["train/policies/worker/worker-updates/scored"] for stats in metrics] == [1.0, 1.0]
    assert [stats["train/batch/hook_optimizer_updates"] for stats in metrics] == [1.0, 1.0]
    assert "train/worker-updates/scored" not in metrics[0]
    worker = rows(tmp_path / "logs" / "platoon_tinker" / "run" / "policies" / "worker")
    assert [(row["name"], row["policy_version"], row["optimizer_steps"]) for row in worker] == [
        ("recovery-000000", 0, 0),
        ("recovery-000001", 1, 1),
        ("recovery-000002", 2, 2),
        ("final", 2, 2),
    ]


class _Hearing(px.StepHookBase):
    """A step hook that records what each step's end, and the launch, published."""

    name = "hearing"

    def __init__(self, heard: list[tuple[str, dict[str, int]]]) -> None:
        self.heard = heard

    async def published(self, end: px.StepEnd) -> None:
        self.heard.append((end.kind, dict(end.published)))


class TwoLearners(px.TrainerBase):
    """A trainer of its own that trains main and the worker, one update of each per step through ``run.learner``, and
    ends each step with ``run.end_step()``; its step hook records what each step's end published (``heard``), and it
    records the names of its steps' metrics (``recorded``)."""

    heard: ClassVar[list[tuple[str, dict[str, int]]]] = []
    recorded: ClassVar[list[str]] = []

    def check(self, run: Any) -> None:
        run.hooks.register(_Hearing(self.heard))

    async def prepare(self, run: Any, start: Any) -> px.StepPlan:
        return px.StepPlan(end_step=run.config.train.max_training_steps)

    async def run_steps(self, run: Any) -> None:
        learners = [run.learner(policy, objective=_Scored()) for policy in (MAIN, "worker")]
        while run.clock.step < run.plan.end_step:
            if run.shutdown.is_set():  # a stop request ends the run at a step boundary
                run.interrupted = True
                return
            for learner in learners:
                update = learner.update()
                await update.submit([checks._probe_datum()], [checks._probe_datum()], minibatch=0, microbatch=0)
                await update.apply()
            await run.end_step()
        assert self._train_tracker is not None
        self.recorded.extend(sorted(self._train_tracker.export()))


def test_a_trainer_of_its_own_publishes_each_policy_once_a_step_from_one_checkpoint_of_both(tmp_path, monkeypatch):
    monkeypatch.setattr(TwoLearners, "heard", [])
    monkeypatch.setattr(TwoLearners, "recorded", [])
    services, _ = start_trainer(TwoLearners, tmp_path, steps=2)

    assert [client.optimizer_steps for client in services.main.training_clients] == [2]
    assert [client.optimizer_steps for client in services.worker.training_clients] == [2]
    assert TwoLearners.heard == [
        ("launch", {MAIN: 0, "worker": 0}),
        ("step", {MAIN: 1, "worker": 1}),
        ("step", {MAIN: 2, "worker": 2}),
    ]
    # Each step's end saves one checkpoint (checkpoint.recovery_every: 1) of both, the worker's row first, whose
    # counters are those of the version each publishes from it, and main's row names it.
    run = tmp_path / "logs" / "platoon_tinker" / "run"
    leader, worker = rows(run), rows(run / "policies" / "worker")
    names = ["recovery-000000", "recovery-000001", "recovery-000002", "final"]
    assert [row["name"] for row in leader] == names and [row["name"] for row in worker] == names
    assert [(row["policy_version"], row["optimizer_steps"]) for row in leader] == [(0, 0), (1, 1), (2, 2), (2, 2)]
    assert [(row["policy_version"], row["optimizer_steps"]) for row in worker] == [(0, 0), (1, 1), (2, 2), (2, 2)]
    assert [row["policies"]["worker"]["state_path"] for row in leader] == [row["state_path"] for row in worker]
    assert services.worker.calls.count("save_state recovery-000001") == 1
    # main's learner's metrics keep their names, the worker's go under its prefix.
    assert {"train/scored", "train/policies/worker/scored"} <= set(TwoLearners.recorded)


class _HookLoss(_Scored):
    """A step hook's objective, under a loss of its own, by which the service tells a hook's calls from the step's."""

    loss_fn = "importance_sampling"


class BothUpdates(px.StepHookBase):
    """A step hook that adds one update of main, then one of the worker, at each step's end."""

    name = "both-updates"

    async def update(self, end: px.StepEnd) -> None:
        for policy in (MAIN, "worker"):
            await end.train(policy, [checks._probe_datum()], objective=_HookLoss())


register_component("step_hook", "test-lockstep-both-updates", BothUpdates, exist_ok=True)


class TwoLearnersAndAHook(TwoLearners):
    """:class:`TwoLearners` with :class:`BothUpdates` among its step hooks, whose learners are the run's
    (``run.learner``)."""

    def check(self, run: Any) -> None:
        super().check(run)
        run.hooks.register(BothUpdates())


@pytest.mark.parametrize(
    ("call", "doing"), [("forward_backward_async", "forward_backward"), ("optim_step_async", "optim_step")]
)
@pytest.mark.parametrize("policy", [MAIN, "worker"])
@pytest.mark.parametrize("trainer", ["rl", "own"])
def test_a_failing_service_in_a_step_hook_s_update_fails_the_step_naming_its_policy(
    tmp_path, monkeypatch, trainer, policy, call, doing
):
    """The hook updates main, then the worker, through each one's lane (rl) or the run's learners of them (a trainer
    of its own): a call of the hook's update that fails says whose service failed, as one of the step's own does."""

    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    monkeypatch.setattr(TwoLearners, "heard", [])
    monkeypatch.setattr(TwoLearners, "recorded", [])
    calling = getattr(checks._FakeTrainingClient, call)

    async def failing(self: Any, *args: Any, **kwargs: Any) -> Any:
        loss = kwargs.get("loss_fn") if call == "forward_backward_async" else self.service.losses[-1]
        if self.run.startswith(policy) and loss == _HookLoss.loss_fn:  # the hook's call, not the step's
            raise RuntimeError("503 Service Unavailable")
        return await calling(self, *args, **kwargs)

    monkeypatch.setattr(checks._FakeTrainingClient, call, failing)
    with pytest.raises(BaseException) as raised:
        if trainer == "rl":
            start(tmp_path, steps=1, trainer_options={"hooks": ["test-lockstep-both-updates"]})
        else:
            start_trainer(TwoLearnersAndAHook, tmp_path, steps=1)

    [error] = [error for error in _errors(raised.value) if "503" in str(error)]
    service = WORKER_URL if policy == "worker" else "hosted Tinker"
    assert error.__notes__ == [f"policy {policy}, on {service}: {doing} failed"]
    assert [row["name"] for row in rows(tmp_path / "logs" / "platoon_tinker" / "run")] == ["recovery-000000"]


def test_the_trainer_check_passes_trainers_of_two_policies(monkeypatch):
    """The trainer check (``check_trainer``) on a config that trains the worker beside main, on the check's one fake
    service: its step hook updates main and hears main's publications alone."""

    monkeypatch.setattr(TwoLearners, "heard", [])
    monkeypatch.setattr(TwoLearners, "recorded", [])
    settings = checks._settings

    def two(root: Path, steps: int) -> dict[str, Any]:
        worker = {"model_name": "fake-worker", "renderer_name": "fake-renderer"}
        return {**settings(root, steps), "policies": {"worker": worker}, "agents": SETTINGS["agents"]}

    monkeypatch.setattr(checks, "_settings", two)
    checks.check_trainer(TwoLearners)  # both policies publish at each step's end
    checks.check_trainer(PlatoonTinkerRLTrainer)
