"""The groups a full checkpoint keeps (``platoon.train.tinker.kept_groups``): a kept group reads back as it settled, its
datums exactly, a failed group as failed, and a file that is lost, changed or not what it says is refused with why."""

from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import math

import pytest
import tinker
import torch

from platoon.train.task_stats import GroupRecord, MemberResult
from platoon.train.tinker.kept_groups import (
    _encode_output,
    decode_group,
    encode_group,
    kept_name,
    read_kept,
    write_kept,
)
from platoon.train.tinker.workflows.group_rollout_workflow import _TaskRolloutOutput
from platoon.utils.rollout_stats import RolloutStats
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.stats_tracker import DeferredStats, ReduceType, StatsTracker


def _datum(tokens: list[int], logprobs: list[float]) -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(tokens[1:] + [7])),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(logprobs)),
            "advantages": tinker.TensorData(data=[0.1, -0.25, 1e-40], dtype="float32", shape=[3]),  # not all float32
            "mask": tinker.TensorData.from_torch(torch.tensor([0.0, 1.0, 1.0])),
            "checkpoint_version": tinker.TensorData.from_torch(torch.tensor([3])),
        },
    )


def _member(valid: bool) -> MemberResult:
    return MemberResult(
        reward=0.98 if valid else None,
        scored=1.0 if valid else None,
        valid=valid,
        outcome="scored" if valid else "error",
        disposition="completed",
        reward_key="reward" if valid else None,
        steps=4,
        output_tokens=120,
        seconds=2.5,
        versions={"main": (2, 3)},
        flag=not valid,
    )


def _group_output() -> _TaskRolloutOutput:
    group_stats = DeferredStats()
    group_stats.scalar(**{"rollout_completion/rollout_time": 1.5})
    group_stats.denominator(**{"credit/n": torch.tensor([True, False, True])})
    group_stats.stat(
        denominator="credit/n", reduce_type=ReduceType.AVG_MIN_MAX, **{"credit/x": torch.tensor([1.0, 2.0, 3.0])}
    )
    rollout_stats = RolloutStats(groups=1, values={"task_reward": [0.98, 0.0]}, counts={"outcome/scored": 1})
    return _TaskRolloutOutput(
        [_datum([5, 6, 7], [-0.5, float("-inf"), float("nan")]), _datum([9, 8, 7], [0.0, -1.25, -2.0])],
        workload=RolloutWorkload(environment_steps=4, model_calls=4, trajectories=2, postmerge_datums=2),
        requested_rollouts=2,
        observed_rollouts=2,
        trainable_rollouts=1,
        task_retained_datums=2,
        rollout_stats=rollout_stats,
        group_stats=group_stats,
        label="quick",
        entry="quick",
        members=(_member(True), _member(False)),
        verdict="trained",
    )


def _record() -> GroupRecord:
    return GroupRecord(
        start=2,
        seq=5,
        task_id="quick::q1",
        entry="quick",
        labels={"_platoon_environment": "quick"},
        harness=None,
        group_size=2,
        sampler="weighted",
        reason="order",
        score=None,
        position=11,
        drawn_at={"main": 3},
        members=(_member(True), _member(False)),
        verdict="trained",
    )


def _same_tensor(left: tinker.TensorData, right: tinker.TensorData) -> bool:
    if (left.dtype, left.shape) != (right.dtype, right.shape) or len(left.data) != len(right.data):
        return False
    return all((math.isnan(a) and math.isnan(b)) or a == b for a, b in zip(left.data, right.data))


def test_a_kept_group_reads_back_as_it_settled(tmp_path):
    output = _group_output()
    group = encode_group((2, 5), entry="quick", label="quick", record=_record(), output=output)
    assert group is not None
    record = write_kept(tmp_path, kept_name(2, "recovery-000004"), [group])
    assert record["file"] == "tasks/kept-s2-recovery-000004.json.gz" and record["groups"] == [[2, 5]]

    [data] = read_kept(tmp_path, record)
    key, entry, label, read_record, read_output = decode_group(data)

    assert (key, entry, label, read_record) == ((2, 5), "quick", "quick", _record())
    assert type(read_output) is _TaskRolloutOutput and len(read_output) == 2
    for read, kept in zip(read_output, output):
        assert read.model_input == kept.model_input
        assert read.loss_fn_inputs.keys() == kept.loss_fn_inputs.keys()
        assert all(_same_tensor(read.loss_fn_inputs[name], kept.loss_fn_inputs[name]) for name in kept.loss_fn_inputs)
    for name in ("workload", "requested_rollouts", "observed_rollouts", "trainable_rollouts", "task_retained_datums"):
        assert getattr(read_output, name) == getattr(output, name), name
    assert (read_output.label, read_output.entry, read_output.members, read_output.verdict) == (
        "quick",
        "quick",
        output.members,
        "trained",
    )
    assert read_output.rollout_stats == output.rollout_stats
    # The group's own statistics record in the consuming step as they would have.
    recorded, expected = StatsTracker("group"), StatsTracker("group")
    read_output.group_stats.record(recorded)
    output.group_stats.record(expected)
    assert recorded.export() == expected.export()


def test_a_workflow_s_plain_datums_and_a_failed_group_are_kept_and_other_outputs_are_not():
    datums = [_datum([1, 2], [0.0, -0.5])]
    group = encode_group((1, 0), entry=None, label=None, record=None, output=datums)
    assert group is not None
    _key, _entry, _label, record, output = decode_group(json.loads(json.dumps(group)))
    assert record is None and type(output) is list and output[0].model_input == datums[0].model_input
    # A group whose workflow failed reads back as failed, so the step after a resume consumes it as failed.
    failed = encode_group((1, 1), entry="quick", label="quick", record=_record(), output=None)
    assert failed is not None and decode_group(json.loads(json.dumps(failed)))[1:] == (
        "quick",
        "quick",
        _record(),
        None,
    )

    class Custom(list):
        pass

    assert encode_group((1, 2), entry=None, label=None, record=None, output=Custom(datums)) is None
    assert encode_group((1, 3), entry=None, label=None, record=None, output={"datums": datums}) is None


def test_every_field_of_a_group_s_output_is_kept():
    """A field that GroupRolloutWorkflow's output gains must be kept too, or a resume trains the group without it."""

    fields = [name for name in inspect.signature(_TaskRolloutOutput.__init__).parameters if name != "self"]
    encoded = _encode_output(_group_output())
    assert encoded is not None
    assert [name for name in fields if name not in encoded] == []


def test_a_kept_groups_file_that_is_lost_changed_or_not_kept_groups_is_refused(tmp_path):
    group = encode_group((1, 0), entry=None, label=None, record=None, output=[])
    assert group is not None
    record = write_kept(tmp_path, "kept-s1-final.json.gz", [group])
    path = tmp_path / record["file"]
    content = path.read_bytes()
    path.write_bytes(gzip.compress(json.dumps({"format": 1, "groups": []}).encode()))
    with pytest.raises(ValueError, match="does not have the SHA-256 its checkpoint recorded"):
        read_kept(tmp_path, record)
    path.unlink()
    with pytest.raises(ValueError, match="cannot be read"):
        read_kept(tmp_path, record)
    path.write_bytes(content)
    assert len(read_kept(tmp_path, record)) == 1
    with pytest.raises(ValueError, match="names no kept-groups file"):
        read_kept(tmp_path, {"sha256": record["sha256"]})
    other = gzip.compress(b'{"format": 2, "groups": []}\n')
    path.write_bytes(other)
    with pytest.raises(ValueError, match="holds no kept groups of format 1"):
        read_kept(tmp_path, {**record, "sha256": hashlib.sha256(other).hexdigest()})
    with pytest.raises(ValueError, match="cannot read a kept group"):
        decode_group({"key": [1, 0]})
    assert not list(tmp_path.glob("tasks/.*partial"))


def test_deferred_statistics_read_back_and_are_checked_again():
    stats = DeferredStats()
    stats.denominator(n=torch.tensor([True, True]))
    stats.stat(denominator="n", value=torch.tensor([0.5, 1.5]))
    copy = DeferredStats.from_json(json.loads(json.dumps(stats.to_json())))
    recorded, expected = StatsTracker("group"), StatsTracker("group")
    copy.record(recorded)
    stats.record(expected)
    assert recorded.export() == expected.export()
    with pytest.raises(ValueError, match="not registered"):  # a stat whose denominator is missing
        DeferredStats.from_json(
            [["stat", {"denominator": "m", "x": {"tensor": "float32", "shape": [1], "values": [1]}}]]
        )
    with pytest.raises(ValueError, match="not a statistics call"):
        DeferredStats.from_json([["export", {}]])
