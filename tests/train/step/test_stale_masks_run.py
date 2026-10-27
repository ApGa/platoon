"""Per-datum stale masks (RFC 0020, P5), in the trainer's own loop and the group workflow (Tinker faked).

- ``train.stale_filter: datum``: a rollout that spans versions keeps exactly its datums within the bound, with the
  advantages its group's credit gave them; a group none of whose datums is within the bound is dropped whole; the
  group workflow stamps each datum with its own version, the members keep their rollouts'.
- ``group``, the default: the same runs train what they trained before, and a run whose groups are all within the
  bound or all beyond it trains, counts and records the same under either setting.
- The parts: the filter's verdicts and how a run makes it, the datums a step keeps (a masked trajectory start moves to
  the next datum kept), the trainer telling the workflow (and warning of one that does not take it), the config key.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import tinker
import torch
from fakes.config import trainer_config
from fakes.trainer import TrainingClient, make_trainer
from fakes.workers import FakeProxySession, worker_request
from tinker import TensorData

from platoon.train.config_defs import TrainConfig, WorkflowConfig
from platoon.train.dispatch.dispatcher import POSITION_COLUMN
from platoon.train.entry_settings import EntrySettings
from platoon.train.step.datums import _kept_datums, _step_datums
from platoon.train.step.staleness import StaleFilter, StaleVerdict, datum_version
from platoon.train.task_stats import read_events
from platoon.train.trainer import rl as rl_module
from platoon.train.workers import run as run_module
from platoon.train.workflow import output as output_module
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.train.workflow.protocol import WorkflowCapabilities, capabilities_of
from platoon.utils.config import ConfigError
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats


def _datum(position: int, member: int, index: int, version: int | None, reward: float = 1.0, start=None):
    """A datum whose first three tokens are its group's position, its member and its index, its action tokens loaded
    with ``reward`` as the converter loads them; ``start``: its ``traj_start``, under depth-level weighting."""

    inputs = {
        "target_tokens": [1, 2, 3],
        "logprobs": [0.0, -0.1, -0.2],
        "advantages": [0.0, reward, reward],
        "mask": [0.0, 1.0, 1.0] if start != "no-action" else [0.0, 0.0, 0.0],
    }
    if version is not None:
        inputs["checkpoint_version"] = [version]
    loss_fn_inputs = {name: TensorData.from_torch(torch.tensor(values)) for name, values in inputs.items()}
    if start is not None:
        loss_fn_inputs["traj_start"] = TensorData.from_torch(torch.tensor([1.0 if start is True else 0.0]))
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([position, member, index]), loss_fn_inputs=loss_fn_inputs
    )


def _ids(datums) -> list[tuple[int, ...]]:
    return [tuple(datum.model_input.to_ints()) for datum in datums]


# --- a rollout's conversion: each datum's own version beside its rollout's ------------------------------------------


def test_a_rollouts_response_gives_each_datums_own_version_beside_its_rollouts(tmp_path, monkeypatch):
    """In process, the rollout's version is its request's (3); its two datums, each of one call, sampled at 5 and 7."""

    def call(prompt: list[int], version: int) -> Any:
        return SimpleNamespace(
            obs=tinker.ModelInput.from_ints(prompt),
            action=SimpleNamespace(tokens=[40, 41], logprobs=[-0.5, -0.5]),
            version=version,
        )

    monkeypatch.setattr(run_module, "TinkerLLMProxySession", FakeProxySession)
    monkeypatch.setattr(FakeProxySession, "interactions_to_inject", {"a": call([1, 2], 5), "b": call([9], 7)})

    async def rollout_fn(_task, _config):
        steps = [{"misc": {"action_misc": {"completion_id": completion}}} for completion in ("a", "b")]
        return {"id": "r", "trajectories": {"root": {"steps": steps, "reward": 1.0}}}

    request = worker_request(tmp_path)
    request = replace(request, model=replace(request.model, checkpoint_version=3))
    response = asyncio.run(run_module.run_rollout_request(request, None, rollout_fn=rollout_fn))
    assert response.status == "completed" and response.result is not None
    assert [datum_version(datum) for datum in response.result.datums] == [3, 3]
    assert response.result.own_versions == (5, 7)


# --- the group workflow, whose members' datums each carry their own version -------------------------------------------


class _Spanning(GroupRolloutWorkflow):
    """The group workflow over members whose rollout is ``versions(position, member, now) -> (rollout's version, each
    datum's own)``: one datum per own version, every one stamped with the rollout's (as the converter stamps them), the
    own versions given beside them (``own_versions``).  Member 0 scores 1 and member 1 scores 0."""

    def __init__(self, config: WorkflowConfig, llm: Any, versions: Any, *, scope: str = "train") -> None:
        # No rollout function or task source: the members' rollouts are ``versions``.
        unused: Any = None
        model_info: Any = SimpleNamespace(model_name="m", base_url="u", api_key="k", llm=llm)
        super().__init__(rollout_fn=unused, get_task_fn=unused, config=config, model_info=model_info, stats_scope=scope)
        self.llm = llm
        self.versions = versions

    async def arun_episode_single(self, data: dict, rollout_number: int = 0):
        await asyncio.sleep(0)
        position = data[POSITION_COLUMN]
        reward = 1.0 if rollout_number == 0 else 0.0
        rollout, own = self.versions(position, rollout_number, self.llm.version)
        result = TrajectoryCollectionResult(
            datums=[_datum(position, rollout_number, index, rollout, reward) for index in range(len(own))],
            task_reward=reward,
            trajectory_stats=[
                TrajectoryStats(
                    trajectory_id="root",
                    reward=reward,
                    num_steps=len(own),
                    num_input_tokens=3,
                    num_output_tokens=2,
                    num_datums=len(own),
                    is_root=True,
                )
            ],
            root_rewards_dict={},
            own_versions=tuple(own),
        )
        count = len(own)
        workload = RolloutWorkload(
            environment_steps=count,
            model_calls=count,
            input_tokens=3,
            output_tokens=2,
            trajectories=1,
            postmerge_datums=count,
            policy_eligible_datums=count,
            post_sampling_datums=count,
        )
        return output_module._SingleRolloutOutcome(result=result, workload=workload, observed=True)


def _spanning(position: int):
    """Members of ``position``'s group whose first datum was sampled five versions before the second (each rollout's
    version the first's); every other group's datums sampled now."""

    def versions(at: int, member: int, now: int) -> tuple[int, list[int]]:
        return (now - 5, [now - 5, now]) if at == position else (now, [now, now])

    return versions


def test_a_group_workflow_stamps_each_datum_with_its_own_version_under_the_per_datum_filter():
    """Under ``datum`` each datum carries its own version, with the advantages the group's credit gives it under
    ``group``, and the members, read for the per-task statistics, keep their rollouts' versions."""

    outputs: dict[str, Any] = {}
    for stale_filter in ("group", "datum"):
        workflow = _Spanning(
            WorkflowConfig(group_size=2), SimpleNamespace(version=9), _spanning(0), scope=f"test_p5_{stale_filter}"
        )
        workflow.stale_filter = stale_filter
        outputs[stale_filter] = asyncio.run(
            asyncio.wait_for(workflow.arun_episode({"task_id": "t", POSITION_COLUMN: 0}), 10)
        )
    group, datum = outputs["group"], outputs["datum"]
    assert [datum_version(row) for row in group] == [4, 4, 4, 4]
    assert [datum_version(row) for row in datum] == [4, 9, 4, 9]
    advantages = [[row.loss_fn_inputs["advantages"].to_torch().tolist() for row in output] for output in (group, datum)]
    assert advantages[0] == advantages[1] == [[0.0, 0.5, 0.5]] * 2 + [[0.0, -0.5, -0.5]] * 2
    assert group.members == datum.members and [member.versions for member in datum.members] == [{"main": (4, 4)}] * 2
    assert group.verdict == datum.verdict == "trained"


# --- the trainer: what a step trains under each setting -----------------------------------------------------------


def _run(tmp_path, stale_filter: str, workflow_of, *, steps: int = 4):
    """A run of ``steps`` steps of one group of 2 at ``max_staleness: 1`` under ``stale_filter``: what it sent (each
    datum by its first three tokens), each step's logged metrics, and the fates its groups got, in the order the event
    log records them."""

    tmp_path.mkdir(exist_ok=True)
    config = trainer_config(
        tmp_path,
        "train.batch_size=1",
        f"train.max_training_steps={steps}",
        "train.max_staleness=1",
        "train.max_concurrent_rollouts=2",
        "train.workflow_config.group_size=2",
        f"train.stale_filter={stale_filter}",
    )
    client = TrainingClient()
    trainer, _saves, logged = make_trainer(config, tasks=8, training_client=client)
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow_of(trainer, config)), 60))
    fates = [event["fate"] for event in read_events(trainer.run_log_path) if "fate" in event]
    return [ids for datums in client.forward_backwards for ids in _ids(datums)], logged.stats, fates


def _group_workflow(versions):
    return lambda trainer, config: _Spanning(config.train.workflow_config, trainer._model_info.llm, versions)


def test_a_rollout_that_spans_versions_keeps_exactly_its_datums_within_the_bound(tmp_path, caplog):
    """Under ``datum``, the group at position 1 trains its members' second datums, sampled now, and not their first,
    five versions old; under ``group`` the whole group is dropped, by its oldest datum."""

    with caplog.at_level(logging.INFO):
        sent, logged, fates = _run(tmp_path / "datum", "datum", _group_workflow(_spanning(1)))
    assert (1, 0, 1) in sent and (1, 1, 1) in sent
    assert (1, 0, 0) not in sent and (1, 1, 0) not in sent
    assert sum(stats["train/batch/stale_datums"] for stats in logged) == 2
    assert sum(stats["train/batch/stale_groups"] for stats in logged) == 0
    assert "stale" not in fates
    [line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Stale datums masked")]
    assert "2 of the group's 4, the oldest at policy version" in line and "beyond the bound of 1 versions" in line

    sent, logged, fates = _run(tmp_path / "group", "group", _group_workflow(_spanning(1)))
    assert not [ids for ids in sent if ids[0] == 1]
    assert sum(stats["train/batch/stale_groups"] for stats in logged) == 1
    assert not [key for stats in logged for key in stats if key.endswith("stale_datums")]
    assert fates.count("stale") == 1


def test_a_group_with_no_datum_within_the_bound_is_dropped_whole(tmp_path, caplog):
    def versions(at: int, member: int, now: int) -> tuple[int, list[int]]:
        return (now - 5, [now - 5, now - 4]) if at == 2 else (now, [now, now])

    with caplog.at_level(logging.INFO):
        sent, logged, fates = _run(tmp_path, "datum", _group_workflow(versions))
    assert not [ids for ids in sent if ids[0] == 2]
    assert sum(stats["train/batch/stale_groups"] for stats in logged) == 1
    assert sum(stats["train/batch/stale_datums"] for stats in logged) == 4
    assert fates.count("stale") == 1
    [line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Stale rollout detected")]
    assert "oldest datum version" in line and "rollout version" not in line


def _compared(stats: dict[str, Any]) -> dict[str, Any]:
    """A step's metrics that say what it trained and counted, without the per-datum filter's own and without times."""

    kept = ("train/batch/", "train/workload/", "train/credit/")
    return {
        key: value
        for key, value in stats.items()
        if (key.startswith(kept) or key in ("train/stale_rollouts", "train/total_datums_per_batch"))
        and not key.startswith("train/batch/stale_datums")  # and its __count
        and "time" not in key
    }


def test_groups_within_the_bound_or_beyond_it_whole_train_and_count_the_same_under_either_setting(tmp_path):
    """With every group's datums all within the bound or all beyond it, ``datum`` sends the same datums, logs the same
    metrics (beside its count of stale datums) and gives the same fates as ``group``."""

    def versions(at: int, member: int, now: int) -> tuple[int, list[int]]:
        return (now - 5, [now - 5, now - 5]) if at in (1, 3) else (now, [now, now])

    runs = {name: _run(tmp_path / name, name, _group_workflow(versions)) for name in ("group", "datum")}
    (group_sent, group_logged, group_fates), (datum_sent, datum_logged, datum_fates) = runs["group"], runs["datum"]
    assert group_sent == datum_sent and group_fates == datum_fates and group_fates.count("stale") == 2
    assert len(group_logged) == len(datum_logged)
    for group_stats, datum_stats in zip(map(_compared, group_logged), map(_compared, datum_logged), strict=True):
        keys = sorted(group_stats.keys() | datum_stats.keys())
        differ = [(key, group_stats.get(key), datum_stats.get(key)) for key in keys]
        assert [row for row in differ if row[1] != row[2]] == []


class _OwnWorkflow:
    """A workflow of its own, which stamps its datums itself and does not take the stale filter: the group at
    position 1 gives one datum five versions old and one sampled now."""

    def __init__(self, trainer) -> None:
        self.trainer = trainer

    async def arun_episode(self, row):
        await asyncio.sleep(0)
        position, now = row[POSITION_COLUMN], self.trainer._model_info.llm.version
        return [_datum(position, 0, 0, now - 5 if position == 1 else now), _datum(position, 0, 1, now)]


def test_a_workflow_that_does_not_take_the_filter_is_judged_by_the_versions_its_datums_carry(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        sent, logged, _fates = _run(tmp_path, "datum", lambda trainer, config: _OwnWorkflow(trainer))
    assert (1, 0, 1) in sent and (1, 0, 0) not in sent
    assert sum(stats["train/batch/stale_datums"] for stats in logged) == 1
    warnings = [r.getMessage() for r in caplog.records if "train.stale_filter is datum" in r.getMessage()]
    assert len(warnings) == 1 and "_OwnWorkflow" in warnings[0] and "takes_stale_filter" in warnings[0]


class _Trimming(_Spanning):
    """A subclass whose members drop their last datum after the conversion (a filter of its own), leaving their own
    versions as the conversion gave them."""

    async def arun_episode_single(self, data: dict, rollout_number: int = 0):
        outcome = await super().arun_episode_single(data, rollout_number)
        outcome.result.datums.pop()
        return outcome


def test_datums_that_no_longer_match_their_own_versions_keep_their_rollouts_and_the_run_warns_once(tmp_path, caplog):
    """The group at position 1 keeps its members' first datums, five versions old by their rollout's version: under
    ``datum`` they are judged by it, as under ``group``, and every other group trains."""

    def trimming(trainer, config):
        return _Trimming(config.train.workflow_config, trainer._model_info.llm, _spanning(1))

    runs = {}
    for name in ("group", "datum"):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            runs[name] = _run(tmp_path / name, name, trimming)
        warned = [r.getMessage() for r in caplog.records if "do not match their own versions" in r.getMessage()]
        assert len(warned) == (1 if name == "datum" else 0)
    (group_sent, _, group_fates), (datum_sent, _, datum_fates) = runs["group"], runs["datum"]
    assert datum_sent == group_sent and len(datum_sent) == 6
    assert datum_fates == group_fates == ["consumed", "stale", "consumed", "consumed"]
    assert "the datums of 2 of the 2 members of task" in warned[0]


# --- the parts ------------------------------------------------------------------------------------------------------


def test_the_trainer_tells_a_workflow_that_takes_it_how_the_stale_filter_judges_and_warns_of_one_that_does_not(caplog):
    def config(stale_filter: str) -> Any:
        return SimpleNamespace(train=SimpleNamespace(stale_filter=stale_filter))

    workflow = _Spanning(WorkflowConfig(group_size=2), SimpleNamespace(version=0), _spanning(0), scope="test_p5_set")
    assert capabilities_of(workflow).takes_stale_filter and workflow.stale_filter == "group"
    rl_module._set_stale_filter(config("datum"), workflow)
    assert workflow.stale_filter == "datum"
    rl_module._set_stale_filter(config("group"), workflow)  # a later start under the default
    assert workflow.stale_filter == "group"

    own = _OwnWorkflow(None)
    assert not capabilities_of(own).takes_stale_filter
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        rl_module._set_stale_filter(config("group"), own)
        assert not caplog.records
        rl_module._set_stale_filter(config("datum"), own)
    assert len(caplog.records) == 1 and not hasattr(own, "stale_filter")

    class Declared:
        capabilities = WorkflowCapabilities(takes_stale_filter=True)

    declared = Declared()
    rl_module._set_stale_filter(config("datum"), declared)
    assert declared.stale_filter == "datum"  # type: ignore[attr-defined]


def _rows(*versions: int | None) -> list[tinker.Datum]:
    return [_datum(0, 0, index, version) for index, version in enumerate(versions)]


def test_the_per_datum_filter_masks_the_datums_beyond_the_bound_and_drops_a_group_only_when_all_are():
    per_datum = StaleFilter(2, {"slow": 4}, per_datum=True)
    rows = _rows(3, 7, None, 8)  # at version 8 with the bound 2: 3 is 5 versions old; a datum without one is new
    assert per_datum.check(None, rows, 8) == StaleVerdict(stale=False, version=3, now=8, bound=2, masked=(0,))
    assert StaleFilter(2, {"slow": 4}).check(None, rows, 8) == StaleVerdict(stale=True, version=3, now=8, bound=2)
    assert per_datum.check("slow", rows, 8).masked == (0,)  # 5 versions old, beyond the entry's own bound
    assert per_datum.check("slow", _rows(4, 7), 8) == StaleVerdict(stale=False)  # 4 old: within it
    assert per_datum.check(None, _rows(1, 5), 8) == StaleVerdict(stale=True, version=1, now=8, bound=2)
    for output in (_rows(6, 7), _rows(None, None), [], None):
        assert per_datum.check(None, output, 8) == StaleVerdict(stale=False)
    assert StaleFilter(None, {}, per_datum=True).check(None, _rows(0), 99) == StaleVerdict(stale=False)


def test_a_step_keeps_the_datums_not_masked_and_a_masked_trajectory_start_moves_to_its_next_datum_kept():
    """Trajectory a: its start and the next datum masked, its third kept; b: kept whole; c: its start masked, its next
    datum kept without an action token, its third kept.  The settled rows keep their starts."""

    rows = [
        _datum(0, 0, 0, 1, start=True),
        _datum(0, 0, 1, 1, start=False),
        _datum(0, 0, 2, 9, start=False),
        _datum(0, 1, 0, 9, start=True),
        _datum(0, 1, 1, 9, start=False),
        _datum(0, 2, 0, 1, start=True),
        _datum(0, 2, 1, 9, start="no-action"),
        _datum(0, 2, 2, 9, start=False),
    ]
    starts = [float(row.loss_fn_inputs["traj_start"].to_torch().item()) for row in rows]
    kept = _kept_datums(_step_datums(rows), (0, 1, 5))
    assert _ids(kept) == [(0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 2, 1), (0, 2, 2)]
    assert [float(datum.loss_fn_inputs["traj_start"].to_torch().item()) for datum in kept] == [1, 1, 0, 0, 1]
    assert [float(row.loss_fn_inputs["traj_start"].to_torch().item()) for row in rows] == starts
    # Without depth-level weighting the datums carry no start, and none is added.
    plain = _kept_datums(_step_datums(_rows(1, 9)), (0,))
    assert _ids(plain) == [(0, 0, 1)] and "traj_start" not in plain[0].loss_fn_inputs


def test_a_runs_stale_filter_is_per_datum_under_the_setting_with_each_entrys_bound():
    settings = {label: EntrySettings(label, None, None, bound) for label, bound in (("quick", 2), ("slow", 6))}
    for stale_filter, per_datum in (("group", False), ("datum", True)):
        train = TrainConfig(model_name="m", renderer_name="r", max_staleness=2, stale_filter=cast(Any, stale_filter))
        made = StaleFilter.of_run(train, settings)
        assert (made.run_bound, made.bounds, made.per_datum) == (2, {"quick": 2, "slow": 6}, per_datum)
    partial = cast(Any, SimpleNamespace(max_staleness=None))  # a config made without the key
    assert not StaleFilter.of_run(partial, {}).per_datum


def test_stale_filter_takes_group_or_datum_and_defaults_to_group(tmp_path):
    assert trainer_config(tmp_path).train.stale_filter == "group"
    assert trainer_config(tmp_path, "train.stale_filter=datum").train.stale_filter == "datum"
    with pytest.raises(ConfigError, match="stale_filter"):
        trainer_config(tmp_path, "train.stale_filter=rollout")
    with pytest.raises(ValueError, match="train.stale_filter must be 'group' or 'datum'"):
        TrainConfig(model_name="m", renderer_name="r", stale_filter=cast(Any, "token"))
    with pytest.raises(ConfigError, match="a key of train, not of a workflow config: train.stale_filter"):
        trainer_config(tmp_path, "train.workflow_config.stale_filter=datum")
