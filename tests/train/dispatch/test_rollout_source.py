"""The run's training draws (RFC 0021, ``RolloutSource``): a trainer other than rl trains on the run's rollouts through
``run.rollouts()`` and passes the trainer conformance check, its draws taking part in every full checkpoint and taken up
on resume, the horizon moving on with its steps, its takes ending on a stop request, and the run closing its draws; the
rl trainer's draws are the run's too, through its methods; each policy's version, as the source and the dispatcher read
them; and the workflow's setup (``WorkflowSetup``), which ``platoon train`` builds a workflow class with."""

from __future__ import annotations

import asyncio
import json
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

import platoon.experimental as px
from platoon.testing import check_trainer
from platoon.testing import trainers as checks
from platoon.train import launch
from platoon.train.checkpoints import saving
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig, WorkflowConfig
from platoon.train.dispatch import runner
from platoon.train.dispatch.settled import _Settled, _settled
from platoon.train.dispatch.source import RolloutSource
from platoon.train.policy.policy import Policy
from platoon.train.policy.set import PolicySet
from platoon.train.policy.spec import MAIN, PolicySpec, policy_specs
from platoon.train.step.collect import StepDispatch
from platoon.train.task_records import TaskRecorder
from platoon.train.task_stats import read_events
from platoon.train.trainer.rl import PlatoonTinkerRLTrainer
from platoon.train.trainer.run import TrainerRun
from platoon.train.trainer.signals import TerminateTrainLoop, TrainInterrupted
from platoon.train.workflow import protocol
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.utils.config import build_config

_SETTINGS, _RUN, _GROUPS = checks._settings, checks._run, checks._OneDatumGroups


class _Drawing(px.TrainerBase):
    """A trainer of its own that trains on the run's rollouts without the rl trainer, as RFC 0021's S4 sketches one:
    it draws the start's groups through the run's rollout source, taking them until the run's shutdown request, and
    each step is one optimizer step on the fake service over the step's groups, published as a new version.  It
    records each start's draws as they opened, with the groups the resume took up (``opened``); its running groups feed
    ``watchdog`` when one is set; with ``direct``, a step dequeues its groups from the source's queue itself."""

    opened: ClassVar[list[tuple[RolloutSource, int]]] = []
    watchdog: ClassVar[Any] = None
    direct: ClassVar[bool] = False

    @classmethod
    def needs(cls, config: Any, options: Any) -> px.TrainerNeeds:
        return px.TrainerNeeds(rollouts=True)

    async def prepare(self, run: TrainerRun, start: Any) -> px.StepPlan:
        self.source = run.rollouts()
        assert self.source.dispatcher is not None
        self.opened.append((self.source, len(self.source.dispatcher.kept())))
        return px.StepPlan(end_step=self.source.end_step)

    async def run_steps(self, run: TrainerRun) -> None:
        try:
            async with asyncio.TaskGroup() as group:
                self.source.start(group, watchdog=self.watchdog or run.clock.watchdog, stop=run.shutdown)
                while run.clock.step < run.plan.end_step and not run.shutdown.is_set():
                    await self._step(run)
                raise TerminateTrainLoop()  # the dispatcher and its groups end with the task group
        except* TerminateTrainLoop:
            pass
        except* TrainInterrupted:  # a stop request while the step waited for its groups: the run saves where it is
            pass

    async def _step(self, run: TrainerRun) -> None:
        groups = await self._take(run.config.train.batch_size, run.shutdown)
        policy = run.policies[run.policies.leader]
        rows = [datum for group in groups for datum in px.rows_by_policy(group.rollout, policy.id)[policy.id]]
        client: Any = policy.training_client  # the fake service's
        await (await client.forward_backward_async(rows, "cispo")).result_async()
        await (await client.optim_step_async(None)).result_async()
        policy.optimizer_steps += 1
        self.source.committed([group.key for group in groups])
        policy.info.llm.update_sampling_client(await policy.save_for_sampler())
        run.clock.advance(consumed=self.source)

    async def _take(self, n: int, stop: asyncio.Event) -> list[_Settled]:
        if not self.direct:
            return await self.source.take(n, stop=stop)
        source: Any = self.source
        groups = []
        for _ in range(n):  # as a step that reads the source's queue itself, the rl trainer's, does
            groups.append(_settled(await source.results.get()))
            source.dispatcher.dequeued()
        return groups


class _StopInGroup:
    """One-datum groups, of which the third raises the stop request that ``platoon train`` passes on (``SIGUSR2``)."""

    def __init__(self) -> None:
        self.groups = 0
        self.workflow = _GROUPS()

    async def arun_episode(self, data: dict) -> Any:
        self.groups += 1
        if self.groups == 3:
            signal.raise_signal(signal.SIGUSR2)
            await asyncio.sleep(0.05)
        return await self.workflow.arun_episode(data)


def _start(
    monkeypatch: Any,
    root: Path,
    trainer: Any,
    *,
    steps: int = 4,
    stop_at: int | None = None,
    train: dict[str, Any] | None = None,
    workflow: Any = None,
    timeout: float = 30.0,
) -> Any:
    """One start of ``trainer`` under the trainer check's harness, with ``train`` laid over its ``train.*`` settings
    and ``workflow`` (a class) in place of its one-datum groups; a start still running ``timeout`` seconds after it
    began ends with ``TimeoutError``."""

    def settings(directory: Path, count: int) -> dict[str, Any]:
        given = _SETTINGS(directory, count)
        return {**given, "train": {**given["train"], **(train or {})}}

    monkeypatch.setattr(checks, "_settings", settings)
    monkeypatch.setattr(checks, "_OneDatumGroups", workflow or _GROUPS)
    monkeypatch.setattr(checks, "_run", lambda built: asyncio.wait_for(_RUN(built), timeout))
    with checks._restored_signals():
        return checks._start(trainer, None, root, steps=steps, stop_at=stop_at)


def _run_dir(root: Path) -> Path:
    return root / "logs" / "platoon_tinker" / "run"


def _rows(root: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (_run_dir(root) / "checkpoints.jsonl").read_text().splitlines()]


@pytest.fixture(autouse=True)
def _fresh_drawing(monkeypatch):
    monkeypatch.setattr(_Drawing, "opened", [])
    monkeypatch.setattr(_Drawing, "watchdog", None)
    monkeypatch.setattr(_Drawing, "direct", False)


def test_a_trainer_that_draws_through_the_run_s_rollout_source_passes_the_trainer_check():
    check_trainer(_Drawing)


def test_its_draws_take_part_in_every_full_checkpoint_and_a_resume_takes_them_up(tmp_path, monkeypatch):
    stopped = _start(monkeypatch, tmp_path, _Drawing, stop_at=checks.STOP_AT)
    resumed = _start(monkeypatch, tmp_path, _Drawing)

    assert stopped.error is None and resumed.error is None
    rows = _rows(tmp_path)
    assert [row["name"] for row in rows] == ["recovery-000000", "000003_interrupted", "recovery-000003", "final"]
    assert all({"tasks", "dispatch"} <= set(row) for row in rows)
    # Every group of the run started at once (no staleness bound): the two the interrupted start's steps left settled
    # were kept, and the resumed start took them up as it opened, to train them without rolling them out again.
    assert [kept for _, kept in _Drawing.opened] == [0, 2]
    events = list(read_events(_run_dir(tmp_path)))
    drawn = {(event["start"], event["seq"]): event["position"] for event in events if event["event"] == "draw"}
    consumed = [(event["start"], event["seq"]) for event in events if event.get("fate") == "consumed"]
    assert sorted(drawn[key] for key in consumed) == list(range(8))  # each position of the four steps trained once
    assert len(drawn) == 8  # and drawn once


def test_the_horizon_moves_on_with_each_step(tmp_path, monkeypatch):
    start = _start(monkeypatch, tmp_path, _Drawing, train={"max_staleness": 0})

    assert start.error is None and [row["name"] for row in start.rows][-1] == "final"
    ((source, _),) = _Drawing.opened
    assert source.dispatcher is not None and source.dispatcher.peak == 2  # one batch at a time: (0 + 1) × 2 groups


def test_its_takes_end_on_a_stop_request_while_no_group_can_start(tmp_path, monkeypatch):
    # The rollout permits fit one group of eight at a time, and the stop request comes inside the first group of the
    # second step: the stopped dispatcher starts no other group, so the step's take ends on the request.
    start = _start(
        monkeypatch,
        tmp_path,
        _Drawing,
        train={"max_concurrent_rollouts": 8, "workflow_config": {"group_size": 8}},
        workflow=_StopInGroup,
    )

    assert start.error is None
    assert [row["name"] for row in start.rows] == ["recovery-000000", "000001_interrupted"]


def test_a_step_commits_the_groups_it_dequeued_from_the_source_s_queue_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(_Drawing, "direct", True)
    start = _start(monkeypatch, tmp_path, _Drawing)

    assert start.error is None
    events = list(read_events(_run_dir(tmp_path)))
    assert sum(event.get("fate") == "consumed" for event in events) == 8


def test_its_running_groups_feed_the_watchdog_it_is_started_with(tmp_path, monkeypatch):
    beats: list[None] = []
    monkeypatch.setattr(_Drawing, "watchdog", SimpleNamespace(heartbeat=lambda: beats.append(None)))
    start = _start(monkeypatch, tmp_path, _Drawing, steps=2)

    assert start.error is None and len(beats) >= 4  # one as each of the start's four groups ended


def test_the_run_closes_its_draws_after_the_trainer_s_close(tmp_path, monkeypatch):
    closed: list[str] = []

    class _Closing(_Drawing):
        async def close(self, run: TrainerRun) -> None:
            closed.append("trainer")

    def close(recorder: TaskRecorder) -> None:
        closed.append("statistics")

    monkeypatch.setattr(TaskRecorder, "close", close)
    start = _start(monkeypatch, tmp_path, _Closing, steps=2)

    assert start.error is None and closed == ["trainer", "statistics"]


def test_its_files_go_to_the_run_s_directory_a_trainer_names(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    class _Elsewhere(_Drawing):
        @property
        def run_log_path(self) -> str:
            return str(elsewhere)

    start = _start(monkeypatch, tmp_path, _Elsewhere, steps=2)

    assert start.error is None
    rows = [json.loads(line) for line in (elsewhere / "checkpoints.jsonl").read_text().splitlines()]
    assert all((elsewhere / row["dispatch"]["file"]).exists() for row in rows)
    assert list(read_events(elsewhere)) and not (_run_dir(tmp_path) / "tasks").exists()


def test_only_a_trainer_that_rolls_out_has_the_run_s_draws(tmp_path):
    config = build_config(PlatoonTinkerRLTrainerConfig, checks._settings(tmp_path, 2))

    class _Steps(px.TrainerBase):
        pass

    with checks._faked(checks.FakePolicyService()):
        trainer = _Steps.from_setup(px.TrainerSetup(config=config, policy_specs=policy_specs(config)))
    run = TrainerRun(trainer)
    with pytest.raises(RuntimeError, match="rolls out nothing"):
        run.rollouts()
    assert run._built_workflows() is None  # nor are workflows built for it


class _RLRecording(PlatoonTinkerRLTrainer):
    """An rl trainer that records each of its methods of the start's draws as the start calls it, and whether its draws
    are the run's."""

    calls: ClassVar[list[str]] = []
    SEAMS = (
        "_begin_task_records",
        "_training_tasks",
        "_new_dispatcher",
        "_task_records_started",
        "_checkpoint_state",
        "_restore_dispatch",
        "_dispatch",
    )

    async def prepare(self, run: TrainerRun, start: Any) -> px.StepPlan:
        plan = await super().prepare(run, start)
        self.calls.append(f"the run's draws: {run.source is self._rollouts}")
        return plan


def _recorded(name: str) -> Any:
    seam = getattr(PlatoonTinkerRLTrainer, name)

    def recorded(self: Any, *args: Any, **kwargs: Any) -> Any:
        _RLRecording.calls.append(name)
        return seam(self, *args, **kwargs)

    return recorded


def test_the_rl_trainer_s_draws_are_the_run_s_and_each_of_their_steps_goes_through_its_methods(tmp_path, monkeypatch):
    monkeypatch.setattr(_RLRecording, "calls", [])
    for name in _RLRecording.SEAMS:
        monkeypatch.setattr(_RLRecording, name, _recorded(name))
    stopped = _start(monkeypatch, tmp_path, _RLRecording, stop_at=checks.STOP_AT)
    resumed = _start(monkeypatch, tmp_path, _RLRecording)

    assert stopped.error is None and resumed.error is None
    assert set(_RLRecording.calls) == {*_RLRecording.SEAMS, "the run's draws: True"}
    # Each start's statistics, dispatcher and task samplers read the trainer's tasks, as its loop does.
    assert _RLRecording.calls.count("_training_tasks") >= 3 * 2


def _policy(version: int) -> Policy:
    info: Any = SimpleNamespace(llm=SimpleNamespace(version=version))  # the model info's gateway counts the versions
    return Policy(PolicySpec(MAIN, model_name="m", renderer_name="r"), info)


def _source(versions: Any = dict, leader: str = MAIN) -> Any:
    tasks = px.TaskSources(train_dataset=[])
    return RolloutSource(None, run_dir="", tasks=tasks, workflow=None, versions=versions, leader=leader)


def test_each_policy_s_version_is_the_count_its_gateway_keeps_and_the_source_reads_its_leader_s():
    policy = _policy(3)
    policies = PolicySet([policy])
    assert policy.version == 3 and policies.versions() == {MAIN: 3}
    llm: Any = policy.info.llm
    llm.version = 4
    assert policies.versions() == {MAIN: 4} and _source(policies.versions).version() == 4
    assert _source(lambda: {MAIN: 1, "student": 7}, leader="student").version() == 7


def test_a_group_admission_rule_decides_a_group_at_the_policy_s_version_as_it_settles():
    class Rule:
        def at_settle(self, draw: Any, rollout: Any, record: Any, *, now: int) -> Any:
            self.now = now
            return SimpleNamespace(admit=True, reason="informative"), "informative"

    settled: list[Any] = []
    source = _source(lambda: {MAIN: 7})
    source.dispatcher = SimpleNamespace(settle=lambda draw, item: settled.append(item))
    rule: Any = Rule()
    draw: Any = SimpleNamespace(entry=None, label=None, key=(0, 0))
    runner._settle_under_rule(source, rule, draw, ["datum"], None)
    assert rule.now == 7 and [group.admitted for group in settled] == ["informative"]


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, str]] = []

    def fate(self, record: Any, fate: str) -> None:
        self.calls.append(("fate", record, fate))

    def fate_of(self, key: Any, fate: str) -> None:
        self.calls.append(("fate_of", key, fate))


def test_a_commit_gives_each_group_the_fate_the_step_decided_else_consumed_or_failed():
    groups = {
        (0, 0): _Settled(None, None, ["datum"], "record-0", (0, 0)),
        (0, 1): _Settled(None, None, ["datum"], "record-1", (0, 1)),
        (0, 3): _Settled(None, None, ["datum"], "record-3", (0, 3)),
        (0, 4): _Settled(None, None, None, None, (0, 4)),  # failed, kept after the statistics stopped: no record
    }
    committed: list[Any] = []
    source, recorder = _source(), _Recorder()
    source.recorder = recorder
    source.dispatcher = SimpleNamespace(holding=groups.get, committed=committed.append)

    # (0, 2): a marker of a group a rule rejected as it settled, which has its fate already and the dispatcher no
    # longer holds; None: a group without a draw.
    source.committed([(0, 0), (0, 1), (0, 2), None], fates={(0, 0): "stale", (0, 1): "consumed"})
    source.committed([(0, 3), (0, 4)])

    assert recorder.calls == [
        ("fate", "record-0", "stale"),
        ("fate", "record-1", "consumed"),
        ("fate", "record-3", "consumed"),
        ("fate_of", (0, 4), "failed"),
    ]
    assert committed == [[(0, 0), (0, 1), (0, 2), None], [(0, 3), (0, 4)]]


def test_a_step_tells_its_draws_what_it_committed_and_without_a_dispatcher_only_their_statistics():
    groups = {(0, 0): _Settled(None, None, ["datum"], "record-0", (0, 0))}
    told: list[Any] = []
    source, recorder = _source(), _Recorder()
    dispatcher: Any = SimpleNamespace(
        holding=groups.get, committed=told.append, step_started=lambda: told.append("step started")
    )
    source.recorder, source.dispatcher = recorder, dispatcher

    # As a test's _dispatch leaves the loop: no dispatcher to tell, the start's statistics still hear the fates.
    undispatched = StepDispatch(source, None)
    undispatched.committed([(0, 0)], {(0, 0): "consumed"})
    undispatched.step_started()
    assert recorder.calls == [("fate", "record-0", "consumed")] and told == []

    dispatched = StepDispatch(source, dispatcher)
    dispatched.committed([(0, 0)], {(0, 0): "stale"})
    dispatched.step_started()
    assert recorder.calls[1:] == [("fate", "record-0", "stale")] and told == [[(0, 0)], "step started"]


def test_a_checkpoint_snapshots_the_statistics_before_it_writes_the_dispatcher_s_state(monkeypatch):
    order: list[str] = []
    source = _source()
    source.recorder = SimpleNamespace(snapshot=lambda name: order.append("snapshot") or {"file": "t"})
    source.dispatcher = SimpleNamespace(
        state=lambda: order.append("state") or {"next_position": 4}, kept=lambda: order.append("kept") or []
    )

    def save_state(run_dir: str, name: str, state: Any, kept: Any) -> dict[str, str]:
        order.append("save_state")
        return {"file": "d"}

    monkeypatch.setattr(saving, "save_state", save_state)
    assert asyncio.run(source.checkpoint_state("000001")) == {"tasks": {"file": "t"}, "dispatch": {"file": "d"}}
    assert order == ["snapshot", "state", "kept", "save_state"]


def _setup(**fields: Any) -> px.WorkflowSetup:
    async def rollout(task: Any, rollout_config: Any) -> dict:
        raise AssertionError("not run")

    given: dict[str, Any] = {
        "phase": "train",
        "rollout_fn": rollout,
        "get_task_fn": lambda task_id: SimpleNamespace(id=task_id, max_steps=None),
        "config": WorkflowConfig(group_size=2),
        "entry_configs": None,
        "policies": {
            MAIN: SimpleNamespace(model_name="m", base_url=None, api_key=None, llm=SimpleNamespace(version=0))
        },
        "log_path": "logs",
        "stats_scope": "train",
        "filter_errors": False,
        "reward_processor": None,
    }
    return px.WorkflowSetup(**{**given, **fields})


def test_a_setup_stands_for_the_group_workflow_s_keyword_arguments_in_their_order():
    setup = _setup(packages=("acme",), label_by="suite", options={"label_by": "env", "extra": 1})
    keywords = protocol._keywords(setup)
    assert list(keywords) == [
        "rollout_fn",
        "get_task_fn",
        "config",
        "model_info",
        "log_path",
        "stats_scope",
        "filter_errors",
        "reward_processor",
        "packages",
        "label_by",
        "extra",
    ]
    assert keywords["model_info"] is setup.policies[MAIN] and keywords["label_by"] == "env"  # an option wins
    assert "packages" not in protocol._keywords(_setup())  # not given: the constructor's default
    with pytest.raises(TypeError, match="'config'"):
        protocol._keywords(_setup(options={"config": None}))


def test_the_group_workflow_is_built_from_its_setup_as_from_its_keyword_arguments():
    entries = {"a": WorkflowConfig(group_size=4)}
    setup = _setup(packages=(), label_by="suite", entry_configs=entries)
    built = GroupRolloutWorkflow.from_setup(setup)
    keywords = GroupRolloutWorkflow(**protocol._keywords(setup))
    assert type(built) is GroupRolloutWorkflow and dict(built.entry_configs) == entries
    shown = ("config", "model_info", "log_path", "stats_scope", "filter_errors", "packages", "label_by")
    assert {name: getattr(built, name) for name in shown} == {name: getattr(keywords, name) for name in shown}


def test_platoon_train_builds_a_workflow_class_without_from_setup_with_the_keyword_arguments():
    class Own:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    entries = {"a": WorkflowConfig(group_size=4)}
    setup = _setup(entry_configs=entries)
    built: Any = launch._built(Own, setup)
    assert built.kwargs == protocol._keywords(setup) and built.entry_configs == entries


def test_the_environments_packages_and_label_column_reach_the_run_s_workflow(tmp_path):
    config = build_config(PlatoonTinkerRLTrainerConfig, checks._settings(tmp_path, 2))
    rollout = _setup().rollout_fn
    components: Any = SimpleNamespace(
        rollout_fn=rollout,
        eval_rollout_fn=rollout,
        get_task_fn=_setup().get_task_fn,
        eval_get_task_fn=_setup().get_task_fn,
        reward_processor=None,
        workflow_kwargs={},
        eval_workflow_kwargs={},
    )
    info = _setup().policies[MAIN]
    for phase in ("train", "eval"):
        setup = launch._workflow_setup(
            phase, config, components, info, {}, {"packages": ("acme",), "label_by": "suite"}
        )
        built = launch._built(GroupRolloutWorkflow, setup)
        assert (setup.packages, built.packages, built.label_by) == (("acme",), ("acme",), "suite")
