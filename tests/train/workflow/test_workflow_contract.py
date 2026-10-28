"""The workflow contract (``platoon.train.workflow.protocol``): a workflow's capabilities read from a ``capabilities``
attribute or from its class flags as the trainer read them before, each by the one place that reads it; a group's
report read from its ``report`` or from the attributes an older output carries, a malformed one or one that cannot be
read left out; its rows by policy, a list of datums being its own, and a step's warning for rows of a policy it does
not train; a rejected group's marker keeping its report; and a full checkpoint keeping an output by the codec of its
exact type, in a trainer's process too."""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import tinker
from fakes.config import make_config

from platoon.train import launch
from platoon.train.checkpoints import kept_groups
from platoon.train.checkpoints.kept_groups import decode_group, encode_group, register_codec
from platoon.train.dispatch.settled import _settled, emptied
from platoon.train.dispatch.source import RolloutSource
from platoon.train.entry_settings import workflow_problems
from platoon.train.step.collect import _get_task_workload_payload
from platoon.train.step.datums import trained_rows
from platoon.train.task_stats import MAIN, MemberResult
from platoon.train.trainer.api import TaskSources
from platoon.train.trainer.rl import _limit_starts
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.train.workflow.output import GroupOutput, _TaskRolloutOutput
from platoon.train.workflow.protocol import (
    OutputReport,
    WorkflowCapabilities,
    capabilities_of,
    malformed_counts,
    report_of,
    rows_by_policy,
)
from platoon.utils.rollout_stats import RolloutStats
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.stats_tracker import DeferredStats

COUNTS = {"requested_rollouts": 4, "observed_rollouts": 4, "trainable_rollouts": 3, "task_retained_datums": 2}
FLAGS = tuple(flag.name for flag in dataclasses.fields(WorkflowCapabilities))


def _report(**fields) -> OutputReport:
    """A report with the workload and the counts above, and ``fields``."""

    return OutputReport(
        workload=RolloutWorkload(trajectories=2),
        requested_rollouts=4,
        observed_rollouts=4,
        trainable_rollouts=3,
        task_retained_datums=2,
        **fields,
    )


def _datum(token: int) -> tinker.Datum:
    return tinker.Datum(model_input=tinker.ModelInput.from_ints([token]), loss_fn_inputs={})


def _member() -> MemberResult:
    return MemberResult(
        reward=1.0,
        scored=1.0,
        valid=True,
        outcome="scored",
        disposition="completed",
        reward_key=None,
        steps=1,
        output_tokens=3,
        seconds=None,
        versions={},
        flag=False,
    )


def _group_output() -> GroupOutput:
    return _TaskRolloutOutput(
        [_datum(1), _datum(2)],
        workload=RolloutWorkload(trajectories=2),
        requested_rollouts=4,
        observed_rollouts=4,
        trainable_rollouts=3,
        task_retained_datums=2,
        rollout_stats=RolloutStats(groups=1),
        members=(_member(),),
        verdict="trained",
    )


class _Older(list):
    """A workflow's own list subclass, written before the contract: it carries the attributes the trainer reads."""


class _Rows:
    """An output written to the contract that is no list: its rows by policy, and its report."""

    def __init__(self, rows, report: OutputReport | None = None) -> None:
        self._rows, self.report = rows, report or OutputReport()

    def rows(self):
        return self._rows


# --- capabilities -----------------------------------------------------------------------------------------------------


def test_capabilities_are_the_class_flags_a_group_rollout_workflow_sets_and_none_for_another_class():
    every = WorkflowCapabilities(
        applies_entry_settings=True,
        gives_verdicts=True,
        runs_rollouts_on_lease=True,
        takes_start_limit=True,
        takes_stale_filter=True,
        runs_rollout_workers=True,
        applies_credit=True,
    )
    assert capabilities_of(GroupRolloutWorkflow) == every
    none = WorkflowCapabilities()
    assert capabilities_of(object) == capabilities_of(SimpleNamespace()) == capabilities_of(None) == none

    class OwnMembers(GroupRolloutWorkflow):  # starts its members its own way: off the lease, as before
        async def _run_group_member(self, *args, **kwargs):
            return None

    assert capabilities_of(OwnMembers) == WorkflowCapabilities(**{**vars(every), "runs_rollouts_on_lease": False})


def test_each_flag_reads_as_the_trainer_read_it_and_a_declared_capabilities_wins():
    class Flags:
        applies_entry_settings = 1  # read by its truth, as loading read it
        runs_rollouts_on_lease = "yes"  # likewise, as the dispatcher's setup read it
        gives_verdicts = 1  # read only when True, as resolution read it
        takes_start_limit = 1  # read only when True, as the isinstance checks were True only for the workflow
        runs_rollout_workers = 1
        binds_policies = 1

    assert capabilities_of(Flags) == WorkflowCapabilities(applies_entry_settings=True, runs_rollouts_on_lease=True)
    declared = WorkflowCapabilities(gives_verdicts=True)

    class Declared(GroupRolloutWorkflow):
        capabilities = declared

    assert capabilities_of(Declared) is declared and capabilities_of(SimpleNamespace(capabilities=declared)) is declared
    # Something else named capabilities is no declaration: the flags are read.
    assert capabilities_of(SimpleNamespace(capabilities="all", gives_verdicts=True)).gives_verdicts


def _only(flag: str) -> type:
    """A workflow class, no GroupRolloutWorkflow, that declares exactly ``flag``."""

    return type(f"Only_{flag}", (), {flag: True})


def _workflow(cls: type):
    """A workflow of ``cls``, unconstructed: the readers read only its class's flags."""

    return object.__new__(cls)


def _takes_the_start_limit(cls: type) -> bool:
    workflow = _workflow(cls)
    workflow.start_limit = "unset"
    _limit_starts(make_config(), workflow, None)  # no train.max_starting_rollouts: no limit
    return workflow.start_limit is None


def _gets_packages(cls: type) -> bool:
    return launch.worker_package_kwargs(cls, make_config()) != {}


def _gets_the_label_column(cls: type) -> bool:
    return launch.label_kwargs(cls, make_config(train={"label_by": "suite"})) != {}


def _gets_a_preload(cls: type) -> bool:
    read: list[str] = []

    class Settings:  # its settings, none set: whether the preload reads them
        def __getattr__(self, name):
            read.append(name)
            raise AttributeError(name)

    workflow = _workflow(cls)
    workflow.config = Settings()
    launch.set_worker_preload(workflow, [], phase="train", tasks=[])
    return bool(read)


def _starts_rolling(cls: type) -> bool:
    class Dispatcher:
        permit_release, rolling = "group", None

        def bind(self, results):
            pass

        async def run(self, **kwargs):
            pass

    class Tasks:
        def create_task(self, coroutine, name=None):
            coroutine.close()

    dispatcher: Any = Dispatcher()
    tasks = TaskSources(train_dataset=[])
    source = RolloutSource(None, run_dir="", tasks=tasks, workflow=_workflow(cls), versions=dict)
    source.dispatcher = dispatcher
    task_group: Any = Tasks()
    source.start(task_group)
    return dispatcher.rolling is True


def _applies_entry_settings(cls: type) -> bool:
    entry = SimpleNamespace(label="a", workflow_config={"group_size": 2}, eval_overrides=SimpleNamespace())
    return workflow_problems([entry], cls, "Only") == []


# Each place that reads a capability, by the capability it reads (the admission rules' read of gives_verdicts is
# tests/train/tasks/test_admission_config.py's).
READERS = [
    ("takes_start_limit", _takes_the_start_limit),
    ("runs_rollout_workers", _gets_packages),
    ("runs_rollout_workers", _gets_the_label_column),
    ("runs_rollout_workers", _gets_a_preload),
    ("runs_rollouts_on_lease", _starts_rolling),
    ("applies_entry_settings", _applies_entry_settings),
]


@pytest.mark.parametrize("flag", FLAGS)
def test_each_reader_reads_its_own_capability_of_a_class_that_declares_only_one(flag, monkeypatch):
    monkeypatch.setattr(launch.AutoEnvironment, "packages", classmethod(lambda cls, config: ("acme",)))
    cls = _only(flag)
    assert {reader.__name__: reader(cls) for _, reader in READERS} == {
        reader.__name__: capability == flag for capability, reader in READERS
    }
    assert all(reader(GroupRolloutWorkflow) for _, reader in READERS)


# --- the report -------------------------------------------------------------------------------------------------------


def test_a_group_output_reports_its_attributes_as_they_are_when_read():
    output = _group_output()
    assert report_of(output) == _report(
        rollout_stats=output.rollout_stats, group_stats=output.group_stats, members=(_member(),), verdict="trained"
    )
    output.label, output.entry = "quick", "quick"  # as GroupRolloutWorkflow.arun_episode sets them
    assert (report_of(output).label, report_of(output).entry) == ("quick", "quick")
    assert isinstance(report_of(output).group_stats, DeferredStats)
    assert report_of(output).rollout_stats is output.rollout_stats  # the same object, which the runner adds to


def test_an_older_output_s_attributes_are_read_and_a_malformed_one_is_left_out():
    output = _Older([_datum(1)])
    vars(output).update(workload=RolloutWorkload(trajectories=2), label="quick", verdict="trained", **COUNTS)
    assert report_of(output) == _report(label="quick", verdict="trained") and malformed_counts(output) is None
    for name, value in (("observed_rollouts", -1), ("requested_rollouts", True), ("trainable_rollouts", 1.0)):
        vars(output).update(COUNTS, **{name: value})
        malformed = report_of(output)
        assert malformed.workload is None and malformed.requested_rollouts is None, name
        assert malformed_counts(output) == {**COUNTS, name: value}
    odd = SimpleNamespace(rollout_stats={}, group_stats=[], members=5, verdict=None, label=7)
    assert report_of(odd) == OutputReport(label=odd.label)  # labels are read as they are
    assert report_of([_datum(1)]) == report_of(None) == OutputReport()
    given = OutputReport(verdict="uniform")
    assert report_of(SimpleNamespace(report=given, verdict="trained")) is given


def test_a_verdict_that_is_no_text_reads_as_unknown():
    output = _Older([_datum(1)])
    vars(output).update(verdict=7)
    assert report_of(output).verdict == "unknown"


def test_an_attribute_that_cannot_be_read_is_left_out_of_the_report():
    class Raising(_Older):
        @property
        def verdict(self):
            raise RuntimeError("no verdict")

        @property
        def report(self):
            raise RuntimeError("no report")

    output = Raising([_datum(1)])
    vars(output).update(label="quick", members=(_member(),))
    assert report_of(output) == OutputReport(label="quick", members=(_member(),))


def test_the_step_warns_of_malformed_counts_once_and_takes_no_workload(caplog):
    counts = {**COUNTS, "observed_rollouts": -1}
    output = _Older([_datum(1)])
    vars(output).update(workload=RolloutWorkload(trajectories=2), **counts)
    with caplog.at_level(logging.WARNING, logger="platoon.train.step.collect"):
        assert _get_task_workload_payload(output) is None
        assert _get_task_workload_payload(_Older([_datum(1)])) is None  # no workload: nothing to warn of
    assert [record.getMessage() for record in caplog.records] == [
        f"Ignoring malformed task workload rollout counts: {counts}"
    ]


def test_a_bare_output_settles_with_the_entry_and_label_of_its_report():
    output = _Older([_datum(1)])
    vars(output).update(entry="entry", label="label")
    settled = _settled(output)
    assert (settled.entry, settled.label, settled.rollout) == ("entry", "label", output)


def test_a_rejected_group_s_marker_keeps_its_output_s_report():
    group = _group_output()
    assert emptied(group) is group and group == [] and report_of(group).verdict == "trained"  # emptied in place
    report = OutputReport(label="quick", rollout_stats=RolloutStats(groups=1), verdict="trained")
    marker = emptied(_Rows({MAIN: [_datum(1)]}, report))
    assert len(marker) == 0 and rows_by_policy(marker, MAIN) == {MAIN: []} and report_of(marker) is report
    assert report_of(emptied((_datum(1),))) == OutputReport()  # a tuple of datums gives no report


# --- the rows ---------------------------------------------------------------------------------------------------------


def test_rows_by_policy_reads_a_sequence_of_datums_as_the_trained_policy_s_and_asks_anything_else():
    group, plain, pair = _group_output(), [_datum(1)], (_datum(1), _datum(2))
    for output in (group, plain, pair):
        assert rows_by_policy(output, MAIN)[MAIN] is output
    assert group.rows() == {MAIN: group} and GroupOutput is _TaskRolloutOutput
    teacher = [_datum(3)]
    tagged = _Rows({MAIN: plain, "teacher": teacher})
    assert rows_by_policy(tagged, MAIN) == {MAIN: plain, "teacher": teacher}


def test_the_step_trains_the_main_rows_and_leaves_out_another_policy_s_with_a_warning(caplog):
    plain, teacher = [_datum(1)], [_datum(3)]
    with caplog.at_level(logging.WARNING, logger="platoon.train.step.datums"):
        assert trained_rows(plain) is plain
        assert trained_rows(_Rows({MAIN: plain})) is plain
        assert trained_rows(_Rows({})) == ()  # an empty group
        assert not caplog.records
        assert trained_rows(_Rows({MAIN: plain, "teacher": teacher})) is plain
        assert trained_rows(_Rows({"teacher": teacher})) == ()
    assert [record.getMessage() for record in caplog.records] == [
        "A group's output gives rows of teacher, which this run does not train (it trains one policy, main): they "
        "are left out"
    ] * 2


# --- kept groups ------------------------------------------------------------------------------------------------------


class Scored(list):
    score = 0.0


def _encode_scored(output: Scored) -> dict:
    return {"tokens": [datum.model_input.to_ints() for datum in output], "score": output.score}


def _decode_scored(data) -> Scored:
    output = Scored(_datum(tokens[0]) for tokens in data["tokens"])
    output.score = data["score"]
    return output


@pytest.fixture
def codecs(monkeypatch):
    """The codec tables as they are, restored after the test, so that what it registers stays in it."""

    monkeypatch.setattr(kept_groups, "_BY_TYPE", dict(kept_groups._BY_TYPE))
    monkeypatch.setattr(kept_groups, "_BY_TAG", dict(kept_groups._BY_TAG))


def _kept(output) -> dict | None:
    group = encode_group((1, 0), entry=None, label=None, record=None, output=output)
    return None if group is None else json.loads(json.dumps(group))


def test_a_full_checkpoint_keeps_an_output_by_the_codec_of_its_exact_type(codecs):
    register_codec(Scored, "test_scored", _encode_scored, _decode_scored)
    output = Scored([_datum(4)])
    output.score = 0.5
    group = _kept(output)
    assert group is not None and group["output"]["type"] == "test_scored"
    read = decode_group(group)[-1]
    assert type(read) is Scored and read.score == 0.5 and read[0].model_input.to_ints() == [4]

    class Subclass(Scored):  # not its base: its own fields would be lost
        pass

    assert _kept(Subclass()) is None
    with pytest.raises(ValueError, match="the kept-group tag 'test_scored' is"):  # a tag is one class's
        register_codec(Subclass, "test_scored", _encode_scored, _decode_scored)
    with pytest.raises(ValueError, match="the kept-group tag 'failed' is a failed group's"):
        register_codec(Subclass, "failed", _encode_scored, _decode_scored)
    with pytest.raises(ValueError, match="not a kept output: 'unknown'"):
        decode_group({"key": [1, 2], "entry": None, "label": None, "record": None, "output": {"type": "unknown"}})


def test_a_codec_s_data_names_no_type_of_its_own(codecs):
    register_codec(Scored, "test_scored", lambda output: {"type": "datums", "datums": []}, _decode_scored)
    with pytest.raises(ValueError, match="the 'test_scored' codec's data names a type of its own"):
        _kept(Scored())


def test_a_class_registered_again_reads_back_with_its_first_codec(codecs):
    """A module run again (a reload, a test that loads it afresh against stubs) registers a new class of the same name:
    its objects keep with its codec, and a kept group reads back with the first, the class the trainer's code uses."""

    register_codec(Scored, "test_scored", _encode_scored, _decode_scored)
    again = type("Scored", (list,), {"__module__": Scored.__module__, "__qualname__": Scored.__qualname__})
    register_codec(again, "test_scored", lambda output: {"tokens": [], "score": -1.0}, lambda data: again())
    kept_again, kept_first = _kept(again([_datum(4)])), _kept(Scored([_datum(4)]))
    assert kept_again is not None and kept_again["output"] == {"type": "test_scored", "tokens": [], "score": -1.0}
    assert kept_first is not None and type(decode_group(kept_first)[-1]) is Scored


def test_a_trainer_process_reads_a_group_output_back():
    """The trainer's module registers the group codec, which a resume reads with: a process that imports it, and no
    workflow, reads a kept ``GroupOutput`` back as it settled."""

    output = _group_output()
    output.label, output.entry = "quick", "quick"
    group = json.dumps(_kept(output))
    code = (
        "import json, sys\n"
        "import platoon.train.trainer.rl\n"
        "from platoon.train.checkpoints.kept_groups import decode_group\n"
        "output = decode_group(json.loads(sys.stdin.read()))[-1]\n"
        "print(type(output).__module__, type(output).__qualname__, len(output), output.verdict, output.label)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], input=group, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["platoon.train.workflow.output", "_TaskRolloutOutput", "2", "trained", "quick"]


def test_the_contract_loads_neither_tinker_nor_torch_and_neither_do_its_readers():
    code = (
        "import sys\n"
        "from types import SimpleNamespace\n"
        "from platoon.train.workflow.protocol import capabilities_of, malformed_counts, report_of, rows_by_policy\n"
        "for output in (None, [], (), SimpleNamespace(workload=1, rollout_stats=2, group_stats=3)):\n"
        "    report_of(output), rows_by_policy(output, 'main'), malformed_counts(output), capabilities_of(output)\n"
        "print(sorted(name for name in ('tinker', 'torch') if name in sys.modules))\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"
