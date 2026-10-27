"""A step's metrics follow the groups it consumes, even when producers finish ahead.

``batch/*`` always did.  The group workflow's own statistics -- ``rollout_completion/*``, ``credit/*``,
``workload/task/*`` -- used to be recorded when a group completed, so with prefetching a step reported the groups of
the next one too (a step counted 20 rollouts against the 16 of its batch, and the next step none).  They now travel
with the group (``group_stats``) and are recorded in the step that dequeues it.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest
import tinker
import torch
from fakes.config import make_config
from fakes.loop import loop_state, loop_trainer, run_train_loop

from platoon.envs.service import EnvSetupError
from platoon.sandbox.process import ProcessProvider
from platoon.task import Task
from platoon.train.components import ENVIRONMENT_COLUMN
from platoon.train.dispatch.settled import _Settled
from platoon.train.workflow.output import _TaskRolloutOutput
from platoon.utils.rollout_stats import (
    OUTCOME_STATUSES,
    REWARD_KEY_MISMATCH,
    outcome_counts,
    reward_key_counts,
    summarize_rollout_group,
)
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.stats_tracker import StatsTracker


def _group(rewards, *, empty=False, checkpoint_version=0, label=None, statuses=(), keys=()):
    # Uniform-reward groups have zero centered advantages but real outcomes.
    datums = (
        []
        if empty
        else [
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints([1, 2, 3]),
                loss_fn_inputs={
                    name: tinker.TensorData.from_torch(torch.tensor(values))
                    for name, values in {
                        "target_tokens": [2, 3, 4],
                        "logprobs": [0.0, -0.1, -0.2],
                        "advantages": [0.0, 0.0, 0.0],
                        "mask": [0.0, 1.0, 1.0],
                        "checkpoint_version": [checkpoint_version],
                    }.items()
                },
            )
            for _ in rewards
        ]
    )
    return _TaskRolloutOutput(
        datums,
        workload=RolloutWorkload(
            postmerge_datums=len(rewards),
            policy_eligible_datums=len(rewards),
            post_sampling_datums=len(rewards),
        ),
        requested_rollouts=len(rewards),
        observed_rollouts=len(rewards),
        trainable_rollouts=len(datums),
        task_retained_datums=len(datums),
        rollout_stats=_stats(rewards, statuses, keys),
        label=label,
    )


def _stats(rewards, statuses, keys=()):
    """A group's statistics, as ``GroupRolloutWorkflow`` summarizes its members: their rewards, their outcome
    statuses and the reward keys they trained."""

    stats = summarize_rollout_group({"task_reward": rewards})
    stats.counts.update(outcome_counts(statuses))
    stats.counts.update(reward_key_counts(keys))
    stats.counts[REWARD_KEY_MISMATCH] = int(len({key for key in keys if key}) > 1)
    return stats


def _run_batches(
    outputs,
    *,
    start_batch=0,
    max_staleness=None,
    filter_zero_advantage_datums=True,
    batch_size=1,
    environments=(),
    label_by=None,
    labels=(),
    transforms=(),
    settle_after=None,
):
    """Run the train loop over ``outputs``, ``batch_size`` groups a step (``fakes.loop``), with the batch
    ``transforms``; each step's logged stats.  The training tasks are labelled ``labels`` (by ``label_by``), and a
    mixture's ``environments`` each label one task by its entry.  The groups settle before the first step, or one each
    ``settle_after`` seconds."""

    column = label_by or ENVIRONMENT_COLUMN
    tasks = [{"task_id": f"{name}::t", ENVIRONMENT_COLUMN: name} for name in environments] + [
        {"task_id": f"t{index}", column: label} for index, label in enumerate(labels)
    ]
    train = {
        "batch_size": batch_size,
        "num_minibatches": 1,
        "num_microbatches": 1,
        "max_concurrent_rollouts": 8,
        "workflow_config": {"group_size": 8, "filter_zero_advantage_datums": filter_zero_advantage_datums},
        "max_staleness": max_staleness,
        "optimizer": {"learning_rate": 1e-5},
    }
    if label_by is not None:
        train["label_by"] = label_by
    entries = {"environments": [{"name": name} for name in environments]} if environments else {}
    trainer = loop_trainer(make_config(train=train, **entries), train_dataset=tasks)
    trainer.batch_transforms = list(transforms)
    state = loop_state(trainer, start_batch=start_batch, policy_version=start_batch)
    return run_train_loop(trainer, state, outputs, steps=len(outputs) // batch_size, settle_after=settle_after)


def test_prefetched_groups_are_logged_only_when_consumed_even_with_zero_advantages():
    first, second = _run_batches([_group([0.0] * 8), _group([1.0] * 2)])

    assert first["train/batch/task_reward/avg"] == 0
    assert first["train/batch/task_reward/count"] == 8
    assert second["train/batch/task_reward/avg"] == 1
    assert second["train/batch/task_reward/count"] == 2
    for metrics in (first, second):
        assert metrics["train/batch/groups"] == 1
        assert metrics["train/batch/groups_without_stats"] == 0
        assert metrics["train/workload/training_batch/total_submitted_datums"] == 0


def test_tinker_non_submitted_metric_requires_complete_workload_metadata():
    """Legacy plain-list workflows must not yield a negative exact count: a step logs how many of the datums it consumed
    it did not submit only when every group it consumed says what generating it took, as an output written to the
    contract does; a plain list of datums says nothing of that, alone or beside such an output."""

    key = "train/workload/training_batch/total_non_submitted_datums"
    (described,) = _run_batches([_group([1.0, 0.0])])
    (plain,) = _run_batches([list(_group([1.0, 0.0]))])
    (mixed,) = _run_batches([_group([1.0, 0.0]), list(_group([1.0]))], batch_size=2)

    assert described[key] == 2.0  # the group's 2 post-merge datums, none submitted (their advantages are zero)
    assert described["train/workload/training_batch/total_submitted_datums"] == 0
    assert key not in plain and key not in mixed
    assert mixed["train/workload/batch/total_tasks"] == 2
    assert mixed["train/workload/batch/total_tasks_with_workload_metadata"] == 1


def _completed_root(reward: float):
    """The conversion of a rollout whose root completed with ``reward``: one datum carrying it."""

    from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            name: tinker.TensorData.from_torch(torch.tensor(values))
            for name, values in {
                "target_tokens": [2, 3, 4],
                "logprobs": [0.0, -0.1, -0.2],
                "advantages": [0.0, reward, reward],
                "mask": [0.0, 1.0, 1.0],
                "checkpoint_version": [0],
            }.items()
        },
    )
    stats = TrajectoryStats("root", reward, num_steps=1, num_input_tokens=1, num_output_tokens=2, num_datums=1)
    return TrajectoryCollectionResult(
        datums=[datum], task_reward=reward, trajectory_stats=[stats], root_rewards_dict={}, task_reward_valid=True
    )


def _workflow_output(rewards: list[float]):
    """A real ``GroupRolloutWorkflow`` group of uniform completed roots (zero advantages: no model pass)."""

    from platoon.train.config_defs import WorkflowConfig
    from platoon.train.workflow.group import GroupRolloutWorkflow

    async def rollout(task, rollout_config):
        raise AssertionError("not run")

    workflow = GroupRolloutWorkflow(
        rollout_fn=rollout,
        get_task_fn=lambda task_id: SimpleNamespace(id=task_id, max_steps=None),
        config=WorkflowConfig(group_size=len(rewards)),
        model_info=SimpleNamespace(model_name="m", base_url=None, api_key=None, llm=SimpleNamespace(version=0)),
    )

    async def single(data, rollout_number):
        return _completed_root(rewards[rollout_number])

    workflow.arun_episode_single = single
    return asyncio.run(workflow.arun_episode({"task_id": "task"}))


def test_a_steps_credit_and_completion_metrics_cover_only_the_groups_it_consumes():
    """Both groups complete before step 1 trains, as when steps run ahead; each step reports only its own."""

    from platoon.utils.stats_tracker import get as get_tracker

    get_tracker("train").export(reset=True)
    outputs = [_workflow_output([0.0] * 4), _workflow_output([1.0] * 2)]
    assert get_tracker("train").export(reset=True) == {}  # nothing was recorded when the groups completed

    first, second = _run_batches(outputs)

    for metrics, baseline, rollouts in ((first, 0.0, 4), (second, 1.0, 2)):
        assert metrics["train/credit/baseline/avg"] == baseline
        assert metrics["train/credit/baseline/avg__count"] == 1
        assert metrics["train/credit/root_advantage/abs_avg"] == 0.0
        assert metrics["train/rollout_completion/groups"] == 1
        assert metrics["train/rollout_completion/task_reward/count"] == rollouts
        assert metrics["train/rollout_completion/task_reward/avg"] == baseline
        assert metrics["train/workload/task/count"] == 1
        assert metrics["train/workload/rollout/count"] == rollouts
        assert metrics["train/batch/task_reward/count"] == rollouts


def test_stale_groups_are_excluded_but_empty_group_outcomes_still_count():
    stale, empty, failed = _run_batches(
        [_group([1.0] * 8), _group([0.0] * 8, empty=True), None],
        start_batch=2,
        max_staleness=0,
    )
    assert stale["train/batch/stale_groups"] == 1
    assert stale["train/batch/groups"] == 0
    assert "train/batch/task_reward/avg" not in stale
    assert empty["train/batch/task_reward/avg"] == 0
    assert empty["train/batch/task_reward/count"] == 8
    assert empty["train/batch/groups"] == 1
    assert failed["train/batch/groups"] == 0
    assert failed["train/batch/groups_without_stats"] == 1
    assert "train/batch/task_reward/avg" not in failed


@pytest.mark.parametrize("filter_zero_advantage_datums", [True, False])
def test_a_step_whose_objective_is_zero_says_it_takes_no_optimizer_update(caplog, filter_zero_advantage_datums):
    """Uniform rewards give every datum a zero advantage: nothing is sent, no optimizer step is taken whatever the
    filter says, and the log says so in those words (the old "skipping forward/backward" read as a skipped
    microbatch only)."""

    from platoon.train.step.publish import _no_update_reason

    with caplog.at_level(logging.INFO):
        (metrics,) = _run_batches([_group([1.0] * 4)], filter_zero_advantage_datums=filter_zero_advantage_datums)

    assert metrics["train/batch/skipped_optimizer"] == 1.0 and metrics["train/batch/optimizer_updates"] == 0.0
    assert "Batch 0 (step 1) takes no optimizer update: its objective is zero" in caplog.text
    assert "whatever filter_zero_advantage_datums says" in caplog.text
    assert _no_update_reason(0, 2) == "it had no datums to train on"


def test_a_mixture_logs_each_environments_groups_rewards_and_outcome_counts_every_step():
    """Beside the batch's own, each mixture entry's metrics cover the groups of its tasks the step consumed, and an
    entry the step took no group of logs zero groups and zero outcomes, with no reward."""

    first, second = _run_batches(
        [
            _group([1.0, 0.0], label="math", statuses=["scored", "scored", "error"]),
            _group([0.5, 0.5], label="code", statuses=["scored", "scored", "timeout"]),
            _group([1.0], label="math", statuses=["scored", None]),
            # The second step takes three math groups and none of code.
            _group([0.0, 0.0], label="math", statuses=["scored", "scored"]),
            _group([1.0, 1.0], label="math", statuses=["scored", "invalid", "scored"]),
            _group([], label="math", statuses=["cancelled", None]),
        ],
        batch_size=3,
        environments=("math", "code"),
    )

    assert first["train/batch/environment/math/groups"] == 2
    assert first["train/batch/environment/math/task_reward/avg"] == pytest.approx(2 / 3)
    assert first["train/batch/environment/math/task_reward/count"] == 3
    assert first["train/batch/environment/math/outcome/scored"] == 3
    assert first["train/batch/environment/math/outcome/error"] == 1
    assert first["train/batch/environment/math/outcome/none"] == 1
    assert first["train/batch/environment/code/groups"] == 1
    assert first["train/batch/environment/code/task_reward/avg"] == 0.5
    assert first["train/batch/environment/code/outcome/timeout"] == 1
    assert first["train/batch/environment/code/outcome/error"] == 0
    # Mixture-wide: the sum of the entries.
    assert first["train/batch/task_reward/count"] == 5
    assert first["train/batch/outcome/scored"] == 5
    assert first["train/batch/outcome/error"] == first["train/batch/outcome/timeout"] == 1
    assert first["train/batch/outcome/none"] == 1
    assert first["train/batch/outcome/invalid"] == first["train/batch/outcome/cancelled"] == 0
    # Only the rewards and the counts of an entry are its own; its other statistics are the batch's.
    assert not any(key.startswith("train/batch/environment/math/task_reward_at_k") for key in first)

    assert second["train/batch/environment/math/groups"] == 3
    assert second["train/batch/environment/math/task_reward/avg"] == 0.5
    assert second["train/batch/environment/math/outcome/scored"] == 4
    assert second["train/batch/environment/math/outcome/invalid"] == 1
    assert second["train/batch/environment/math/outcome/cancelled"] == 1
    assert second["train/batch/environment/code/groups"] == 0
    assert all(second[f"train/batch/environment/code/outcome/{status}"] == 0 for status in OUTCOME_STATUSES)
    assert not any(key.startswith("train/batch/environment/code/task_reward") for key in second)


def test_a_single_environment_logs_outcome_counts_but_no_environment_of_its_own():
    (metrics,) = _run_batches([_group([1.0, 0.0], statuses=["scored", "scored", None])])

    assert metrics["train/batch/outcome/scored"] == 2
    assert metrics["train/batch/outcome/none"] == 1
    assert not any(key.startswith("train/batch/environment/") for key in metrics)


def test_a_stale_group_counts_in_no_environment():
    (stale,) = _run_batches(
        [_group([1.0] * 2, label="a", statuses=["scored", "scored"])],
        start_batch=2,
        max_staleness=0,
        environments=("a", "b"),
    )

    assert stale["train/batch/stale_groups"] == 1
    assert stale["train/batch/environment/a/groups"] == 0
    assert stale["train/batch/environment/a/outcome/scored"] == 0
    assert stale["train/batch/outcome/scored"] == 0


class _Rows:
    """A group's output written to the contract that is no list: its rows by policy, and its report."""

    def __init__(self, rows, report):
        self._rows, self.report = rows, report

    def rows(self):
        return self._rows


def test_a_step_trains_the_main_rows_of_an_output_written_to_the_contract_and_no_other_policy_s(caplog):
    """An output that is no list gives its rows by policy: the step trains its main rows and counts its report's
    statistics; it trains nothing of an empty group's, which gives no rows; and it leaves out another policy's rows,
    with a warning."""

    from platoon.train.workflow.protocol import OutputReport

    def output(policies):
        group = _group([1.0, 0.0])
        return _Rows({policy: list(group) for policy in policies}, OutputReport(rollout_stats=group.rollout_stats))

    with caplog.at_level(logging.WARNING, logger="platoon.train.step.datums"):
        main, empty, teacher = _run_batches([output(["main"]), output([]), output(["teacher"])])
    assert [step["train/total_datums_per_batch"] for step in (main, empty, teacher)] == [2.0, 0.0, 0.0]
    assert [step["train/batch/groups"] for step in (main, empty, teacher)] == [1, 1, 1]
    assert caplog.text.count("gives rows of teacher, which this run does not train") == 1


def _members_output(members, data):
    """A real ``GroupRolloutWorkflow`` group whose members end as ``members``: ``(reward, status)`` for a member with
    a completed root, ``(None, status)`` for one that left no result."""

    from platoon.train.config_defs import WorkflowConfig
    from platoon.train.workflow.group import GroupRolloutWorkflow
    from platoon.train.workflow.output import _SingleRolloutOutcome
    from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats

    def member(reward, status):
        if reward is None:
            return _SingleRolloutOutcome(result=None, workload=RolloutWorkload(), observed=False, outcome_status=status)
        stats = TrajectoryStats("root", reward, num_steps=1, num_input_tokens=1, num_output_tokens=2, num_datums=0)
        result = TrajectoryCollectionResult(
            datums=[], task_reward=reward, trajectory_stats=[stats], root_rewards_dict={}, task_reward_valid=True
        )
        return _SingleRolloutOutcome(result=result, workload=RolloutWorkload(), observed=True, outcome_status=status)

    async def rollout(task, rollout_config):
        raise AssertionError("not run")

    workflow = GroupRolloutWorkflow(
        rollout_fn=rollout,
        get_task_fn=lambda task_id: SimpleNamespace(id=task_id, max_steps=None),
        config=WorkflowConfig(group_size=len(members)),
        model_info=SimpleNamespace(model_name="m", base_url=None, api_key=None, llm=SimpleNamespace(version=0)),
    )

    async def single(data, rollout_number):
        return member(*members[rollout_number])

    workflow.arun_episode_single = single
    return asyncio.run(workflow.arun_episode(data))


def test_a_group_counts_every_members_outcome_and_names_its_mixture_entry():
    output = _members_output(
        [(1.0, "scored"), (0.0, "scored"), (None, "error"), (None, None)],
        {"task_id": "math::task-1", "_platoon_environment": "math"},
    )

    assert output.label == "math"
    assert output.rollout_stats.counts == {
        "outcome/scored": 2,
        "outcome/invalid": 0,
        "outcome/error": 1,  # a member that left no result but counts as an error, say one whose sandbox never started
        "outcome/timeout": 0,
        "outcome/cancelled": 0,
        "outcome/none": 1,  # a member that left nothing, say a dead worker
        "reward_key_mismatch": 0,  # no member recorded the reward key it trained
    }
    completion = StatsTracker("train")
    output.group_stats.record(completion)
    metrics = completion.export()
    assert metrics["train/rollout_completion/outcome/scored"] == 2
    assert metrics["train/rollout_completion/outcome/error"] == 1


def test_a_task_outside_a_mixture_names_no_environment():
    output = _members_output([(1.0, "scored")], {"task_id": "task-1"})

    assert output.label is None


class _Refusing(ProcessProvider):
    """A sandbox provider that refuses to start a sandbox."""

    async def start(self, spec):
        raise RuntimeError("no capacity left for a sandbox")


class _StillStarting(ProcessProvider):
    """A sandbox provider whose sandbox is still starting when the straggler limit cuts its rollout."""

    async def start(self, spec):
        await asyncio.sleep(3600)
        raise AssertionError("the straggler limit cuts the rollout first")


class _EnvServerDown:
    """An environment service whose env server does not answer, so that no session opens (an ORS service's
    ``env_connect`` failure)."""

    name = "remote"

    def __init__(self, **_options: Any) -> None:
        pass

    async def list_tasks(self, split: str) -> list[str]:
        return []

    async def get_task(self, task_id: str) -> Task:
        return Task(id=task_id)

    async def open(self, task: Task, *, rollout_id: str) -> Any:
        raise EnvSetupError("env_connect", f"no env server answers for task {task.id}")


async def _member_rollout(task: Task, config: Any) -> Any:
    """A scored root for a task without an environment route; for a routed task, the real harness rollout, whose
    session never opens here, so that it never reaches its model gateway."""

    from platoon.harness.rollout import run_harness_rollout
    from platoon.train.components import EnvironmentRoute

    if EnvironmentRoute.of_task(task) is None:
        outcome = {"status": "scored", "rewards": {"reward": 1.0}, "reason": None}
        return {"trajectories": {"root": {"id": "root", "steps": [{}], "reward": 1.0, "misc": {"outcome": outcome}}}}
    return await run_harness_rollout(task, config, gateway=SimpleNamespace())


def test_rollouts_whose_session_never_opened_count_as_errors_per_entry_and_in_the_batch(monkeypatch, tmp_path):
    """Real groups of a mixture, run in process.  A member whose sandbox provider refuses to start its sandbox, and
    one whose env server does not answer, raise EnvSetupError out of the harness rollout before any outcome is
    recorded: each counts as an error of its entry and of the batch.  A member whose sandbox was still starting when
    the straggler limit cut it counts as none, apart from them."""

    from platoon.config_defs import RolloutConfig
    from platoon.train.components import EnvironmentRoute
    from platoon.train.config_defs import StragglerPolicy, WorkflowConfig
    from platoon.train.workers import run as run_module
    from platoon.train.workflow.group import GroupRolloutWorkflow
    from platoon.utils.stats_tracker import get as get_tracker

    # A scored root converts to one datum of reward 1, so that every advantage is 0 and the step runs no model pass.
    monkeypatch.setattr(run_module, "get_train_data_for_trajectory_collection", lambda **_: _completed_root(1.0))
    task_dir = tmp_path / "tasks" / "hello"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "task.toml").write_text("")
    (task_dir / "instruction.md").write_text("Say hello.")
    (task_dir / "tests" / "test.sh").write_text("exit 0")

    def harbor(provider: ProcessProvider) -> Task:
        options = {"task_dirs": [str(task_dir)], "provider": provider, "default_image": "local"}
        route = EnvironmentRoute(name="harbor", service="harbor", service_kwargs=options, harness="platoon-bash")
        return route.attach(Task(id="hello"))

    remote = EnvironmentRoute(name="remote", service=f"{__name__}:_EnvServerDown", harness="platoon-bash")
    members = {  # what the members of each group play, one task each, in the order they start
        "harbor::hello": [
            Task(id="hello"),
            harbor(_Refusing(tmp_path / "sandboxes")),
            harbor(_StillStarting(tmp_path / "sandboxes")),
            Task(id="hello"),
        ],
        "remote::q1": [Task(id="q1"), remote.attach(Task(id="q1")), Task(id="q1"), Task(id="q1")],
    }
    planned = {task_id: iter(tasks) for task_id, tasks in members.items()}
    scope = "test_sessions_never_opened"
    workflow = GroupRolloutWorkflow(
        rollout_fn=_member_rollout,
        get_task_fn=lambda task_id: next(planned[task_id]),
        config=WorkflowConfig(
            group_size=4,
            rollout_config=RolloutConfig(output_dir=str(tmp_path / "rollouts"), timeout=60),
            # Once the others settled, a Harbor group waits 0.5 s for its last member; a remote one waits it out.
            straggler_policies={"harbor": StragglerPolicy(timeout_seconds=0.5)},
        ),
        model_info=SimpleNamespace(model_name="m", llm=SimpleNamespace(version=0)),
        stats_scope=scope,
    )

    async def run_groups() -> list:
        return [
            await workflow.arun_episode({"task_id": task_id, ENVIRONMENT_COLUMN: task_id.split("::")[0]})
            for task_id in members
        ]

    outputs = asyncio.run(run_groups())
    assert get_tracker(scope).export(reset=True)[f"{scope}/straggler_cancelled"] == 1

    (metrics,) = _run_batches(outputs, batch_size=2, environments=("harbor", "remote"))

    def counts(prefix: str) -> dict[str, float]:
        """The outcome counts logged under ``prefix`` that are not zero."""

        logged = {status: metrics[f"{prefix}/outcome/{status}"] for status in OUTCOME_STATUSES}
        return {status: count for status, count in logged.items() if count}

    assert counts("train/batch/environment/harbor") == {"scored": 2, "error": 1, "none": 1}  # none: the straggler
    assert counts("train/batch/environment/remote") == {"scored": 3, "error": 1}
    assert counts("train/batch") == counts("train/rollout_completion") == {"scored": 5, "error": 2, "none": 1}
    # The members whose session never opened add no reward.
    assert metrics["train/batch/environment/harbor/task_reward/count"] == 2
    assert metrics["train/batch/environment/remote/task_reward/count"] == 3


def test_each_step_logs_every_label_of_the_label_column_also_at_zero():
    """With ``train.label_by`` naming a column, such as a Harbor dataset's source, each step logs each of its labels'
    groups, reward and outcome counts, a label the step took no group of at zero; a ``/`` is written ``--``."""

    first, second = _run_batches(
        [
            _group([1.0, 0.0], label="Fine/Smol", statuses=["scored", "scored"]),
            _group([0.5], label="amphora/MCO", statuses=["scored", "timeout"]),
            _group([1.0, 1.0], label="Fine/Smol", statuses=["scored", "scored"]),
            _group([0.0], label="Fine/Smol", statuses=["scored"]),
        ],
        batch_size=2,
        label_by="source_dataset",
        labels=["Fine/Smol", "Fine/Smol", "amphora/MCO"],
    )

    smol, mco = "train/batch/source_dataset/Fine--Smol", "train/batch/source_dataset/amphora--MCO"
    assert (first[f"{smol}/groups"], first[f"{mco}/groups"]) == (1, 1)
    assert first[f"{smol}/task_reward/avg"] == first[f"{mco}/task_reward/avg"] == 0.5
    assert (first[f"{mco}/outcome/scored"], first[f"{mco}/outcome/timeout"]) == (1, 1)
    assert (second[f"{smol}/groups"], second[f"{smol}/task_reward/count"], second[f"{mco}/groups"]) == (2, 3, 0)
    assert all(second[f"{mco}/outcome/{status}"] == 0 for status in OUTCOME_STATUSES)
    assert not any(key.startswith("train/batch/environment/") for key in first)


def test_a_column_with_more_than_64_labels_logs_only_the_most_frequent(caplog):
    from platoon.train.labels import MAX_METRIC_LABELS

    labels = ["common"] * 3 + [f"rare-{index:02d}" for index in range(70)]
    with caplog.at_level(logging.WARNING, logger="platoon.train.labels"):
        (metrics,) = _run_batches(
            [_group([1.0], label="common", statuses=["scored"]), _group([1.0], label="rare-69", statuses=["scored"])],
            batch_size=2,
            label_by="item",
            labels=labels,
        )

    logged = {key.split("/")[3] for key in metrics if key.startswith("train/batch/item/")}
    assert len(logged) == MAX_METRIC_LABELS and "common" in logged and "rare-69" not in logged
    assert metrics["train/batch/item/common/groups"] == 1
    assert (
        metrics["train/batch/groups"] == 2
    )  # the batch counts every group, a label's whose metrics are not logged too
    assert "the training tasks have 71 'item' values; their metrics break down by the 64 most frequent" in caplog.text


def test_each_step_counts_its_rollouts_by_the_reward_key_they_trained_once_a_key_has_trained():
    """Rollouts count under the reward key they trained, in the batch and in each label; a key that trained once is
    logged every step after, also at zero; and each step counts the groups that trained different keys."""

    first, second = _run_batches(
        [
            _group([1.0, 0.0], label="smol", statuses=["scored", "scored"], keys=["correctness", "correctness"]),
            _group([1.0], label="mco", statuses=["scored"], keys=["reward"]),
            _group([0.5, 0.0], label="mco", statuses=["scored", "scored"], keys=["reward", None]),  # None: platoon's 0
            _group([0.0, 1.0], label="mco", statuses=["scored", "scored"], keys=["reward", "correctness"]),
        ],
        batch_size=2,
        label_by="source",
        labels=["smol", "mco"],
    )

    assert (first["train/batch/reward_key/correctness"], first["train/batch/reward_key/reward"]) == (2, 1)
    assert first["train/batch/source/smol/reward_key/correctness"] == 2
    assert first["train/batch/source/smol/reward_key/reward"] == 0  # a key the batch trained: every label logs it
    assert first["train/batch/reward_key_mismatch"] == 0
    assert (second["train/batch/reward_key/correctness"], second["train/batch/reward_key/reward"]) == (1, 2)
    assert second["train/batch/reward_key_mismatch"] == second["train/batch/source/mco/reward_key_mismatch"] == 1
    assert second["train/batch/source/smol/reward_key_mismatch"] == 0


# --- what a step counts of the groups it takes ----------------------------------------------------------------------


def test_a_group_counts_under_its_row_s_label_and_not_its_output_s():
    """The label metrics read a group's label from the row it ran, which the group holds, before its output's."""

    output = _group([1.0], label="amphora/MCO", statuses=["scored"])
    (metrics,) = _run_batches(
        [_Settled(None, "Fine/Smol", output)], label_by="source_dataset", labels=["Fine/Smol", "amphora/MCO"]
    )

    smol, mco = "train/batch/source_dataset/Fine--Smol", "train/batch/source_dataset/amphora--MCO"
    assert (metrics[f"{smol}/groups"], metrics[f"{mco}/groups"]) == (1, 0)


def test_a_group_the_step_waited_for_is_not_counted_as_ready_in_the_queue():
    """``trainer_queue_ready_fraction``: the share of the step's dequeues that found a group waiting; here the one
    group settles 50 ms into the step."""

    (metrics,) = _run_batches([_group([1.0])], settle_after=0.05)

    assert metrics["rollout_client/trainer_queue_ready_fraction"] == 0.0


def _no_update_lines(caplog) -> list[str]:
    """The step's line that says why it took no optimizer update."""

    lines = [record.getMessage() for record in caplog.records]
    return [line for line in lines if line.startswith("Batch ") and "takes no optimizer update" in line]


def test_a_group_without_datums_leaves_its_microbatch_empty_and_the_step_without_an_update(caplog):
    with caplog.at_level(logging.INFO):
        _run_batches([_group([1.0], empty=True)])

    assert "No rollouts found for microbatch 0 (minibatch 0)" in caplog.text
    [line] = _no_update_lines(caplog)
    assert line.endswith("it had no datums to train on; the policy stays at version 0")


@pytest.mark.parametrize("emptied", [[], None], ids=["no datums", "none"])
def test_a_microbatch_its_transforms_empty_trains_nothing_and_says_so(caplog, emptied):
    with caplog.at_level(logging.INFO):
        _run_batches([_group([1.0])], transforms=[lambda datums, context: emptied])

    assert "All datums were filtered out for microbatch 0 (minibatch 0)" in caplog.text
    [line] = _no_update_lines(caplog)
    assert line.endswith("it had no datums to train on; the policy stays at version 0")


def test_a_step_logs_the_rollout_threads_cell_guard_counters(monkeypatch):
    """What the guard of programmatic tool calls counts (the ``rollout_worker.telemetry`` hook), in each step's
    stats."""

    import platoon.hooks

    monkeypatch.setattr(platoon.hooks, "_HOOKS", {})
    platoon.hooks.register("rollout_worker.telemetry", lambda: {"ptc_cells_stuck": 2})
    (metrics,) = _run_batches([_group([1.0])])

    assert metrics["ptc_cells_stuck"] == 2.0


def test_a_step_that_submits_more_datums_than_its_groups_report_stops_the_run():
    """A group whose report says it merged no datum, beside the one it trains: the step's count of datums it consumed
    and did not submit would be negative, which stops the run before the step commits."""

    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            name: tinker.TensorData.from_torch(torch.tensor(values))
            for name, values in {
                "target_tokens": [2, 3, 4],
                "logprobs": [0.0, -0.1, -0.2],
                "advantages": [0.0, 1.0, 1.0],
                "mask": [0.0, 1.0, 1.0],
                "checkpoint_version": [0],
            }.items()
        },
    )
    claims = _TaskRolloutOutput(
        [datum],
        workload=RolloutWorkload(postmerge_datums=0, policy_eligible_datums=0, post_sampling_datums=0),
        requested_rollouts=1,
        observed_rollouts=1,
        trainable_rollouts=1,
        task_retained_datums=1,
        rollout_stats=_stats([1.0], ()),
    )

    with pytest.raises(Exception) as raised:
        _run_batches([claims])
    errors = [raised.value]
    while isinstance(errors[-1], BaseExceptionGroup):
        errors.append(errors[-1].exceptions[0])
    assert isinstance(errors[-1], ValueError) and "exceed the consumed post-merge datum count" in str(errors[-1])


class _Older(list):
    """A workflow's own list subclass, written before the contract: its counts, but no rollout statistics."""


def test_a_group_with_malformed_counts_and_no_statistics_is_warned_of_once(caplog):
    output = _Older([])
    vars(output).update(
        workload=RolloutWorkload(trajectories=1),
        requested_rollouts=True,  # a flag, not a count
        observed_rollouts=1,
        trainable_rollouts=0,
        task_retained_datums=0,
    )
    with caplog.at_level(logging.WARNING):
        (metrics,) = _run_batches([output])

    assert caplog.text.count("Ignoring malformed task workload rollout counts") == 1
    assert metrics["train/batch/groups_without_stats"] == 1
