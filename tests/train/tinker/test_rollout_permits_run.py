"""Rollout permits that come back as each rollout ends, in the trainer's own loop and the group workflow (Tinker
faked; RFC 0020).

- With permits for two groups and one straggling member, a third group starts before the straggler ends under
  ``train.permit_release: rollout`` and only after it ends under ``group``.
- Which ``GroupRolloutWorkflow`` subclasses run their rollouts on the lease; a ``_run_group_rollouts`` override that
  bypasses ``_run_group_member`` keeps co-start and group-held permits; a custom workflow without the flag that runs
  its rollouts in lease blocks returns permits per rollout under co-start; an evaluation's workflow gets the no-op
  lease.
- A ``k = 0`` run trains the same datums in the same order under both settings.
- Stragglers: a cancelled straggler returns its permit only once its cancellation has unwound; each member's deadline
  carries its own start delay, so a late member gets its full grace; an override of ``_run_group_member`` is still cut.
- A recursive tree is one rollout on one permit, held until its last child is graded; a group that a member runs in
  turn takes no permits of its own.
- The step metrics (permits held, the dispatcher's blocked shares, the straggler policy's censoring) and the warnings
  (stale groups above 2% of the last 10 steps', rollouts beyond the budget), once each.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tinker
import torch
from test_dispatch_run import _mixture, _mixture_config, _Recording
from test_permit_pool import _quiet
from tinker import TensorData
from trainer_fakes import make_trainer, trainer_config

from platoon.train.leases import NO_LEASE, holding, rollout_lease
from platoon.train.task_stats import MemberResult
from platoon.train.tinker import rollout_metrics
from platoon.train.tinker.censoring import Censoring
from platoon.train.tinker.config_defs import StragglerPolicy, WorkflowConfig
from platoon.train.tinker.dispatcher import POSITION_COLUMN
from platoon.train.tinker.permits import DrawLease, PermitPool
from platoon.train.tinker.staleness import StaleWatch
from platoon.train.tinker.workflows import group_rollout_workflow as grw
from platoon.utils.config import ConfigError
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.stats_tracker import get as get_tracker
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """``train()`` installs SIGINT, SIGTERM, SIGUSR1 and SIGUSR2 handlers; put the test process's back."""

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _datum(position: int, member: int, version: int) -> tinker.Datum:
    """A datum whose first two tokens are the position of its group's row and its member."""

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([position, member, 2]),
        loss_fn_inputs={
            name: TensorData.from_torch(torch.tensor(values))
            for name, values in {
                "target_tokens": [1, 2, 3],
                "logprobs": [0.0, -0.1, -0.2],
                "advantages": [0.0, 1.0, 1.0],
                "mask": [0.0, 1.0, 1.0],
                "checkpoint_version": [version],
            }.items()
        },
    )


def _outcome(
    position: int, member: int, *, reward: float, version: int, seconds: float | None = None
) -> grw._SingleRolloutOutcome:
    result = TrajectoryCollectionResult(
        datums=[_datum(position, member, version)],
        task_reward=reward,
        trajectory_stats=[
            TrajectoryStats(
                trajectory_id="root",
                reward=reward,
                num_steps=1,
                num_input_tokens=3,
                num_output_tokens=2,
                num_datums=1,
                is_root=True,
            )
        ],
        root_rewards_dict={},
    )
    workload = RolloutWorkload(
        environment_steps=1,
        model_calls=1,
        input_tokens=3,
        output_tokens=2,
        trajectories=1,
        postmerge_datums=1,
        policy_eligible_datums=1,
        post_sampling_datums=1,
    )
    return grw._SingleRolloutOutcome(result=result, workload=workload, observed=True, seconds=seconds)


class _Members(grw.GroupRolloutWorkflow):
    """The group workflow, each member's rollout a script: ``await script(row, member)`` is the rollout, and returns
    its reward, or ``(reward, seconds)``.  It runs inside the member's lease block, as a rollout does, counted as a
    rollout in flight; ``log`` records each member's start, end and cancellation."""

    def __init__(self, config: WorkflowConfig, llm: Any, script: Any, *, scope: str = "train") -> None:
        super().__init__(
            rollout_fn=None,
            get_task_fn=None,
            config=config,
            model_info=SimpleNamespace(model_name="m", base_url="u", api_key="k", llm=llm),
            stats_scope=scope,
        )
        self.llm = llm
        self.script = script
        self.log: list[tuple[str, int, int]] = []

    async def arun_episode_single(self, data: dict, rollout_number: int = 0):
        position = data[POSITION_COLUMN]
        version = self.llm.version
        self.log.append(("start", position, rollout_number))
        try:
            with self._monitor().rollout_in_flight():
                answer = await self.script(data, rollout_number)
        except asyncio.CancelledError:
            self.log.append(("cancelled", position, rollout_number))
            raise
        self.log.append(("end", position, rollout_number))
        reward, seconds = answer if isinstance(answer, tuple) else (answer, None)
        return _outcome(position, rollout_number, reward=reward, version=version, seconds=seconds)

    def started(self, position: int) -> list[int]:
        return sorted(member for kind, where, member in self.log if kind == "start" and where == position)


class _Bypass(_Members):
    """A subclass whose ``_run_group_rollouts`` starts its members itself, without ``_run_group_member``."""

    async def _run_group_rollouts(self, data):
        return list(await asyncio.gather(*(self.arun_episode_single(data, i) for i in range(self.config.group_size))))


async def _until(condition, timeout: float = 20.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not condition():
        if loop.time() > end:
            raise AssertionError("the run never got there")
        await asyncio.sleep(0.01)


def _logging_stats(trainer) -> list[dict[str, Any]]:
    logged: list[dict[str, Any]] = []
    trainer._stats_logger = SimpleNamespace(
        wandb_run_id=None, log=lambda step, stats, force=False: logged.append(dict(stats))
    )
    return logged


def _group(config: WorkflowConfig, script: Any, scope: str, cls: Any = _Members) -> Any:
    """A group workflow outside the trainer, in its own stats scope."""

    return cls(config, SimpleNamespace(version=0), script, scope=scope)


# --- permits per rollout: a third group starts before a straggler ends ---------------------------------------------


def _straggler_run(tmp_path: Path, monkeypatch, release: str, workflow_class: Any = _Members, *, third: bool):
    """Permits for two groups of 4; group 0's member 3 straggles and group 1's members are held: once group 1 runs,
    wait for a third group to start (``third``), or 0.3 s for one not to, and read the permits held."""

    config = trainer_config(
        tmp_path,
        "train.batch_size=2",
        "train.max_training_steps=2",
        "train.max_staleness=1",
        "train.max_concurrent_rollouts=8",
        "train.workflow_config.group_size=4",
        f"train.permit_release={release}",
    )
    trainer, _saves, logged = make_trainer(monkeypatch, config, tasks=8)
    straggler, held = asyncio.Event(), asyncio.Event()

    async def script(row, member):
        position = row[POSITION_COLUMN]
        if position == 0 and member == 3:
            await straggler.wait()
        elif position == 1:
            await held.wait()
        else:
            await asyncio.sleep(0)
        return float(member % 2)

    workflow = workflow_class(config.train.workflow_config, trainer._model_info.llm, script)
    seen: dict[str, Any] = {}

    async def run() -> None:
        train = asyncio.ensure_future(trainer.train(train_workflow=workflow))
        await _until(lambda: workflow.started(1) == [0, 1, 2, 3] and ("end", 0, 2) in workflow.log)
        permits = trainer._dispatcher._rows.permits
        if third:  # it starts, and once it and the last group have ended, the permits held are the running members'
            await _until(lambda: workflow.started(2) == [0, 1, 2, 3] and permits.held == 5)
        else:
            await asyncio.sleep(0.3)
        seen["third"] = workflow.started(2)
        seen["held"] = permits.held
        straggler.set()
        held.set()
        await asyncio.wait_for(train, 60)

    asyncio.run(run())
    assert logged == [1, 2]
    return workflow, seen


@pytest.mark.parametrize("release", ["group", "rollout"])
def test_with_permits_for_two_groups_a_third_starts_before_a_straggler_ends_only_under_rollout(
    tmp_path, monkeypatch, release
):
    workflow, seen = _straggler_run(tmp_path, monkeypatch, release, third=release == "rollout")

    if release == "rollout":  # on the straggler's group's three returned permits, then its own
        assert workflow.log.index(("start", 2, 0)) < workflow.log.index(("end", 0, 3))
        assert seen["held"] == 5  # the straggler's permit and group 1's four
    else:
        assert seen["third"] == [] and seen["held"] == 8  # finished members hold their permits until the group ends
        assert workflow.log.index(("start", 2, 0)) > workflow.log.index(("end", 0, 3))


def test_a_run_group_rollouts_override_that_bypasses_the_member_keeps_co_start_and_group_held_permits(
    tmp_path, monkeypatch
):
    _workflow, seen = _straggler_run(tmp_path, monkeypatch, "rollout", _Bypass, third=False)

    assert seen["third"] == [] and seen["held"] == 8  # its members ran on no block: its permits came back whole


def test_which_subclasses_run_their_rollouts_on_the_lease():
    """Beside the guide's table (``test_extending_docs.py``): a flag set in a class body covers the lease methods of
    that class and its bases, so a grandchild inherits its parent's reading, and an override from a mixin or from
    another branch of a diamond reads as False unless the subclass's own body sets the flag."""

    class Credit(grw.GroupRolloutWorkflow):
        def compute_baselines(self, results):
            return super().compute_baselines(results)

    class Rollouts(grw.GroupRolloutWorkflow):
        async def _run_group_rollouts(self, data):
            return []

    class Declared(grw.GroupRolloutWorkflow):
        runs_rollouts_on_lease = True

        async def _run_group_member(self, data, rollout_number, admission):
            return await super()._run_group_member(data, rollout_number, admission)

    class Bypassing:
        async def _run_group_rollouts(self, data):
            return []

    class DeclaringMixin:
        runs_rollouts_on_lease = True  # it vouches for its own override

        async def _run_group_member(self, data, rollout_number, admission):
            return None

    class GrandchildOfCredit(Credit): ...

    class GrandchildOfRollouts(Rollouts): ...

    class GrandchildOfDeclared(Declared): ...

    class Mixin(Bypassing, grw.GroupRolloutWorkflow): ...

    class MixinDeclared(Bypassing, grw.GroupRolloutWorkflow):
        runs_rollouts_on_lease = True

    class OwnMixin(DeclaringMixin, grw.GroupRolloutWorkflow): ...

    class Diamond(Declared, Rollouts): ...

    on = (GrandchildOfCredit, GrandchildOfDeclared, MixinDeclared, OwnMixin)
    off = (GrandchildOfRollouts, Mixin, Diamond)
    assert all(cls.runs_rollouts_on_lease for cls in on) and not any(cls.runs_rollouts_on_lease for cls in off)


class _OwnRollouts:
    """A custom workflow (no ``GroupRolloutWorkflow``, no flag) that runs its own rollouts, each in a lease block."""

    def __init__(self, llm: Any, script: Any, group_size: int) -> None:
        self.llm = llm
        self.script = script
        self.group_size = group_size
        self.log: list[tuple[str, int, int]] = []

    async def arun_episode(self, row):
        lease = rollout_lease()
        position = row[POSITION_COLUMN]

        async def one(member: int) -> tinker.Datum:
            async with lease.rollout(member):
                self.log.append(("start", position, member))
                await self.script(row, member)
                return _datum(position, member, self.llm.version)

        return list(await asyncio.gather(*(one(member) for member in range(self.group_size))))


def test_a_custom_workflow_without_the_flag_that_uses_the_lease_returns_permits_per_rollout_under_co_start(
    tmp_path, monkeypatch
):
    """Group 0's straggler and group 1's three held members run: the four permits their finished members returned
    start the third group whole, its members together, before either ends."""

    config = trainer_config(
        tmp_path,
        "train.batch_size=2",
        "train.max_training_steps=2",
        "train.max_staleness=1",
        "train.max_concurrent_rollouts=8",
        "train.workflow_config.group_size=4",
        "train.permit_release=rollout",
    )
    trainer, _saves, logged = make_trainer(monkeypatch, config, tasks=8)
    straggler, held = asyncio.Event(), asyncio.Event()

    async def script(row, member):
        position = row[POSITION_COLUMN]
        if (position, member) == (0, 3):
            await straggler.wait()
        elif position == 1 and member > 0:
            await held.wait()
        else:
            await asyncio.sleep(0)

    workflow = _OwnRollouts(trainer._model_info.llm, script, 4)

    def third() -> list[int]:
        return sorted(member for _kind, position, member in workflow.log if position == 2)

    async def run() -> None:
        train = asyncio.ensure_future(trainer.train(train_workflow=workflow))
        await _until(lambda: third() == [0, 1, 2, 3])
        assert trainer._dispatcher.rolling is False and ("start", 0, 3) in workflow.log
        straggler.set()
        held.set()
        await asyncio.wait_for(train, 60)

    asyncio.run(run())
    assert logged == [1, 2]


def test_an_evaluation_gets_the_no_op_lease_and_training_its_draw_s(tmp_path, monkeypatch):
    config = trainer_config(tmp_path, "eval.every=1", "train.permit_release=rollout")
    trainer, _saves, _logged = make_trainer(monkeypatch, config, eval_tasks=2)

    class Leases:
        def __init__(self) -> None:
            self.seen: list[Any] = []

        async def arun_episode(self, row):
            self.seen.append(rollout_lease())
            await asyncio.sleep(0)
            return [_datum(0, 0, 0)]

    train, evaluate = Leases(), Leases()
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=train, eval_workflow=evaluate), 60))

    assert evaluate.seen and all(lease is NO_LEASE for lease in evaluate.seen)
    assert train.seen and all(isinstance(lease, DrawLease) and lease.size == 8 for lease in train.seen)


# --- a k = 0 run replays: the same datums, in the same order --------------------------------------------------------


@pytest.mark.parametrize("mixture", [False, True], ids=["one-environment", "mixture"])
def test_at_k_0_both_settings_train_the_same_datums_in_the_same_order(tmp_path, monkeypatch, mixture):
    """Members that take a varied number of event-loop turns, groups of 2 with the default permits (every rollout of a
    batch): each step trains exactly its batch, and ``rollout`` trains the datums ``group`` trains, in the same
    order."""

    steps: dict[str, list[list[int]]] = {}
    for release in ("group", "rollout"):
        where = tmp_path / release
        where.mkdir()
        train: dict[str, Any] = {
            "batch_size": 4,
            "max_training_steps": 5,
            "max_staleness": 0,
            "permit_release": release,
        }
        if mixture:
            train["workflow_config"] = {"group_size": 2}
            config = _mixture_config(where, train, [{"name": "a"}, {"name": "b"}])
        else:
            overrides = [f"train.{key}={value}" for key, value in train.items()]
            config = trainer_config(
                where, *overrides, "train.workflow_config.group_size=2", "train.max_concurrent_rollouts=8"
            )
        client = _Recording()
        trainer, _saves, logged = make_trainer(monkeypatch, config, tasks=10, training_client=client)
        if mixture:
            trainer.train_dataset, trainer.train_order = _mixture(a=7, b=5)

        async def script(row, member):
            for _ in range((row[POSITION_COLUMN] * 7 + member * 3) % 5):
                await asyncio.sleep(0)
            return float(member % 2)

        workflow = _Members(config.train.workflow_config, trainer._model_info.llm, script)
        asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 60))
        assert logged == [1, 2, 3, 4, 5]
        steps[release] = client.steps

    assert steps["rollout"] == steps["group"]
    assert [sorted(set(step)) for step in steps["group"]] == [list(range(4 * step, 4 * step + 4)) for step in range(5)]


# --- stragglers -----------------------------------------------------------------------------------------------------


def test_a_cancelled_straggler_returns_its_permit_only_once_its_cancellation_has_unwound():
    config = WorkflowConfig(
        group_size=4, straggler_timeout_seconds=0.05, straggler_quorum=3, filter_zero_advantage_datums=False
    )
    unwinding, unwound = asyncio.Event(), asyncio.Event()

    async def script(row, member):
        if member == 3:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                unwinding.set()
                await unwound.wait()  # its worker and sandboxes going away
                raise
        return float(member % 2)

    workflow = _group(config, script, "test_permit_straggler_unwinds")

    async def run():
        pool = PermitPool(4)
        lease = pool.reserve(4, rolling=True, active=True)
        with holding(lease):
            group = asyncio.ensure_future(workflow.arun_episode({"task_id": "t", POSITION_COLUMN: 0}))
        await asyncio.wait_for(unwinding.wait(), 10)
        await _quiet()
        assert pool.free == 3  # the straggler's permit is still held while its cancellation unwinds
        unwound.set()
        output = await asyncio.wait_for(group, 10)
        assert pool.free == 4 and output.observed_rollouts == 3
        lease()

    asyncio.run(run())
    assert ("cancelled", 0, 3) in workflow.log
    get_tracker("test_permit_straggler_unwinds").export(reset=True)


def test_a_member_s_start_delay_is_its_offset_behind_the_group_s_first_start():
    """The deadline arithmetic, on no clock: a member's offset is its block's wait less the group's least one, None
    while its block waits, and 0 for a member that never entered a block; the group started together only when no
    block waits and none waited; a waiting member's offset is at least the time it has waited so far."""

    async def run():
        starts = grw._MemberStarts()
        starts.created = 100.0
        assert starts.together() and starts.offset(2) == 0.0  # an override of _run_group_member records nothing
        for index in range(4):
            starts.waiting.add(index)
        starts.started(0, 0.0)
        starts.started(1, 0.0)
        assert not starts.together() and starts.offset(1) == 0.0 and starts.offset(2) is None
        starts.started(2, 0.5)
        assert starts.offset(2) == 0.5 and starts.least_offset(now=100.8) == pytest.approx(0.8)
        starts.started(3, 0.75)
        assert starts.offset(3) == 0.75 and not starts.together()
        together = grw._MemberStarts()
        together.started(0, 0.0)
        together.started(1, 0.0)
        assert together.together()

    asyncio.run(run())


def test_under_rolling_starts_late_members_get_their_full_grace_and_are_cut_each_at_its_own_deadline(caplog):
    """A group of 5 on 2 of 5 permits (another draw holds 3), quorum 2, grace 0.3 s.  Members 0 and 1 end at 0.5 and
    0.6 s, settling the quorum, and their permits start members 2 and 3; the other draw's permits start member 4 at
    0.8 s.  Member 2 runs past the group's shared deadline (0.9 s), to 1.1 s, and ends: its own deadline carries its
    0.5 s start delay.  Members 3 and 4 run on and are cut at their own deadlines, 1.5 and 1.7 s, with one warning."""

    config = WorkflowConfig(
        group_size=5, straggler_timeout_seconds=0.3, straggler_quorum=2, filter_zero_advantage_datums=False
    )
    quorum, shared_deadline_passed = asyncio.Event(), asyncio.Event()

    async def script(row, member):
        if member < 2:
            await asyncio.sleep(0.5 + 0.1 * member)
            if member == 1:
                quorum.set()
        elif member == 2:
            await shared_deadline_passed.wait()
        else:
            await asyncio.sleep(30)
        return float(member % 2)

    scope = "test_permit_late_member_grace"
    workflow = _group(config, script, scope)

    async def run():
        pool = PermitPool(5)
        other = pool.reserve(3)
        lease = pool.reserve(5, rolling=True, active=True)
        with holding(lease):
            group = asyncio.ensure_future(workflow.arun_episode({"task_id": "t", POSITION_COLUMN: 0}))
        await quorum.wait()
        await asyncio.sleep(0.2)
        other.withdraw()  # member 4 starts
        await asyncio.sleep(0.3)  # past the group's shared deadline, 0.3 s after the quorum, which member 2 outlives
        assert ("cancelled", 0, 2) not in workflow.log
        shared_deadline_passed.set()
        output = await asyncio.wait_for(group, 10)
        lease()
        assert pool.free == 5
        return output

    with caplog.at_level(logging.WARNING):
        output = asyncio.run(run())
    log = workflow.log
    assert ("end", 0, 2) in log and log.index(("end", 0, 2)) < log.index(("cancelled", 0, 3))
    assert log.index(("cancelled", 0, 3)) < log.index(("cancelled", 0, 4))
    assert output.observed_rollouts == 3
    assert get_tracker(scope).export(reset=True)[f"{scope}/straggler_cancelled"] == 2.0
    warnings = [record for record in caplog.records if "Cancelling straggler" in record.getMessage()]
    assert len(warnings) == 1  # one per cut group


class _MemberWrapper(_Members):
    """Overrides ``_run_group_member`` and runs the member itself, without ``super()``, keeping the startup slot."""

    async def _run_group_member(self, data, rollout_number, admission):
        slot = admission.member_slot()
        try:
            return await self.arun_episode_single(data, rollout_number)
        finally:
            slot()


def test_an_override_of_run_group_member_that_skips_the_lease_block_is_still_cut_by_the_straggler_policy():
    """Under the default ``permit_release: group``: its members never enter a lease block, so each counts as started
    with the group, and a member that never ends is cut once the quorum's grace has passed."""

    config = WorkflowConfig(
        group_size=4, straggler_timeout_seconds=0.2, straggler_quorum=3, filter_zero_advantage_datums=False
    )

    async def script(row, member):
        await asyncio.sleep(30 if member == 3 else 0.01 * (member + 1))
        return float(member % 2)

    workflow = _group(config, script, "test_permit_member_override", _MemberWrapper)

    async def run():
        lease = PermitPool(4).reserve(4)  # a draw's lease under group: its blocks hold nothing
        with holding(lease):
            return await asyncio.wait_for(workflow.arun_episode({"task_id": "t", POSITION_COLUMN: 0}), 10)

    assert asyncio.run(run()).observed_rollouts == 3
    assert [member for kind, _position, member in workflow.log if kind == "cancelled"] == [3]
    get_tracker("test_permit_member_override").export(reset=True)


def test_a_recursive_tree_is_one_rollout_on_one_permit_until_its_last_child_is_graded():
    """A group of one tree whose root runs three children at once, graded once their work ends: the tree holds its
    one permit until the last child is graded, and only then does the draw waiting for it start."""

    graded = asyncio.Event()

    async def tree(row, member):
        async def child() -> None:
            await asyncio.sleep(0)  # its work
            await graded.wait()  # its grading, by a forked env session or a verifier agent

        await asyncio.gather(child(), child(), child())
        return 1.0

    workflow = _group(WorkflowConfig(group_size=1, filter_zero_advantage_datums=False), tree, "test_permit_trees")

    async def run():
        pool = PermitPool(1)
        lease = pool.reserve(1, rolling=True, active=True)
        waiting = pool.reserve(1)
        with holding(lease):
            group = asyncio.ensure_future(workflow.arun_episode({"task_id": "t", POSITION_COLUMN: 0}))
        await _quiet()
        assert not waiting.startable  # the children's work is done; they wait to be graded
        graded.set()
        await group
        assert waiting.startable  # graded: the tree's permit started the waiting draw
        lease()

    asyncio.run(run())
    get_tracker("test_permit_trees").export(reset=True)


@pytest.mark.parametrize("release", ["group", "rollout"])
def test_a_group_that_a_member_runs_in_turn_takes_no_permits_of_its_own(release):
    """A draw of 2 on a pool of 2 whose members each run a group of 3 of another group workflow: the member's rollout
    sees no lease, so the nested groups reuse none of the draw's indices and run on their member's permit."""

    pool = PermitPool(2)
    held: list[int] = []

    async def nested(row, member):
        held.append(pool.held)
        await asyncio.sleep(0)
        return float(member % 2)

    inner = _group(WorkflowConfig(group_size=3, filter_zero_advantage_datums=False), nested, f"test_nested_{release}")

    async def member(row, index):
        assert rollout_lease() is NO_LEASE
        output = await inner.arun_episode({"task_id": f"t{index}", POSITION_COLUMN: index})
        return float(output.observed_rollouts == 3)

    outer = _group(WorkflowConfig(group_size=2, filter_zero_advantage_datums=False), member, f"test_outer_{release}")

    async def run():
        lease = pool.reserve(2, rolling=release == "rollout", active=release == "rollout")
        with holding(lease):
            output = await asyncio.wait_for(outer.arun_episode({"task_id": "t", POSITION_COLUMN: 0}), 10)
        lease()
        return output

    assert asyncio.run(run()).observed_rollouts == 2
    assert held == [2] * 6 and pool.free == 2  # the draw's two permits, and no more, throughout
    for scope in (f"test_nested_{release}", f"test_outer_{release}"):
        get_tracker(scope).export(reset=True)


# --- the step metrics and the warnings -------------------------------------------------------------------------------


@pytest.mark.parametrize("release", ["group", "rollout"])
def test_a_run_logs_the_permits_held_the_blocked_shares_and_what_the_straggler_policy_censored(
    tmp_path, monkeypatch, release
):
    """Two entries, groups of 4 whose member 3 the straggler policy always cuts, and whose members 0 to 2 score 1, 0,
    1 in 1, 2 and 3 s: each step's censoring metrics are exact, for the batch and each entry; the blocked shares are
    shares; and the ceiling ``concurrency_achieved_fraction`` reads against is the budget under ``rollout``, whole
    groups under ``group``."""

    train = {
        "batch_size": 2,
        "max_training_steps": 3,
        "max_staleness": 1,
        "max_concurrent_rollouts": 10,
        "permit_release": release,
        "workflow_config": {"group_size": 4, "straggler_timeout_seconds": 0.05, "straggler_quorum": 3},
    }
    config = _mixture_config(tmp_path, train, [{"name": "a"}, {"name": "b"}])
    trainer, _saves, _logged = make_trainer(monkeypatch, config)
    trainer.train_dataset, trainer.train_order = _mixture(a=4, b=4)
    logged = _logging_stats(trainer)

    async def script(row, member):
        if member == 3:
            await asyncio.sleep(30)
        return (float(member != 1), float(member + 1))

    workflow = _Members(config.train.workflow_config, trainer._model_info.llm, script)
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 60))

    assert len(logged) == 3
    for stats in logged:
        assert stats["train/batch/censoring/censored_share"] == pytest.approx(0.25)
        assert stats["train/batch/censoring/mixed_groups"] == 2.0
        assert stats["train/batch/censoring/slowest_pass_rate"] == 1.0  # member 2, the slowest valid one, passed
        assert stats["train/batch/censoring/others_pass_rate"] == pytest.approx(0.5)
        groups = [stats.get(f"train/batch/environment/{entry}/censoring/mixed_groups", 0.0) for entry in ("a", "b")]
        assert sum(groups) == 2.0
        for reason in ("permits", "horizon", "labels", "plan", "sampler"):
            assert 0.0 <= stats[f"train/dispatch/blocked_share/{reason}"] <= 1.0
        assert 0.0 <= stats["rollout_client/permits_held_mean"] <= 10.0
        assert stats["rollout_client/rollouts_in_flight_configured"] == (10.0 if release == "rollout" else 8.0)


def test_the_permits_held_match_a_hand_computed_run_of_the_permit_pool(monkeypatch):
    """On a clock the test moves, a pool of 6 permits reports to a monitor made 2 s before its first grant: none held
    for 2 s, 4 for 2 s, 6 for 3 s, then 2 for 5 s, a mean of (8 + 18 + 10) / 12 over the monitor's whole window, the
    rollouts' too.  Neither gauge shows before it is first reported."""

    clock = SimpleNamespace(now=50.0)
    monkeypatch.setattr(rollout_metrics, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monitor = rollout_metrics.ClientThroughputMonitor("test_permits_held")
    pool = PermitPool(6)
    clock.now += 1.0
    assert "rollout_client/permits_held_mean" not in monitor.export(reset=False)
    pool.on_held = monitor.set_permits_held
    clock.now += 1.0
    first = pool.reserve(4)
    clock.now += 2.0
    pool.reserve(2)
    clock.now += 3.0
    first.withdraw()
    clock.now += 5.0
    metrics = monitor.export()
    assert metrics["rollout_client/permits_held_mean"] == pytest.approx((4 * 2 + 6 * 3 + 2 * 5) / 12)
    assert "rollout_client/startup_gate_waiting_mean" not in metrics  # no group reached a startup gate
    with monitor.startup_gate_wait():
        clock.now += 1.0
    clock.now += 3.0
    assert monitor.export()["rollout_client/startup_gate_waiting_mean"] == pytest.approx(1 / 4)


def _record(entry: str, *members: tuple[str, bool, float | None, float | None]) -> SimpleNamespace:
    """A group record with ``(disposition, valid, scored, seconds)`` members."""

    return SimpleNamespace(
        entry=entry,
        members=tuple(
            MemberResult(
                reward=scored,
                scored=scored,
                valid=valid,
                outcome="scored" if valid else "none",
                disposition=disposition,
                reward_key=None,
                steps=None,
                output_tokens=None,
                seconds=seconds,
            )
            for disposition, valid, scored, seconds in members
        ),
    )


def test_the_censoring_metrics_match_hand_computed_groups_where_a_straggler_policy_applies():
    run = WorkflowConfig(straggler_timeout_seconds=60.0, straggler_policies={"c": StragglerPolicy(None)})
    censoring = Censoring({}, run, threshold=lambda entry: 0.5 if entry == "half" else 1.0)
    group_a = _record(
        "a",
        ("completed", True, 1.0, 10.0),
        ("completed", True, 0.0, 30.0),  # the slowest valid member, a failure
        ("straggler_killed", False, None, None),
        ("completed", True, 1.0, 20.0),
    )
    censoring.add("a", group_a)
    half = _record("half", ("completed", True, 0.6, 5.0), ("completed", True, 0.4, 1.0), ("completed", True, 0.7, None))
    censoring.add("b", half)  # stale groups count too: the straggler policy ran before the stale filter
    censoring.add("a", _record("a", ("completed", True, 1.0, 1.0), ("completed", True, 1.0, 2.0)))
    censoring.add("c", group_a)  # its label's policy sets no cutoff: no straggler policy applies
    prefixes = {label: f"environment/{label}" for label in "abc"}
    metrics = censoring.step_metrics(prefixes)
    assert metrics["batch/censoring/censored_share"] == pytest.approx(1 / 9)
    assert metrics["batch/censoring/mixed_groups"] == 2.0  # the all-pass third group is not mixed
    assert metrics["batch/censoring/slowest_pass_rate"] == pytest.approx(1 / 2)  # a's failed, b's (0.6 of 0.5) passed
    assert metrics["batch/censoring/others_pass_rate"] == pytest.approx(3 / 4)  # a: 2 of 2; b: 0.4 fails, 0.7 passes
    assert metrics["batch/environment/a/censoring/censored_share"] == pytest.approx(1 / 6)
    assert metrics["batch/environment/b/censoring/slowest_pass_rate"] == 1.0
    assert not [key for key in metrics if "/c/" in key]
    assert censoring.step_metrics(prefixes) == {}  # the next step starts afresh
    unpoliced = Censoring({}, WorkflowConfig(), threshold=None)
    unpoliced.add("a", group_a)
    assert unpoliced.step_metrics(prefixes) == {}  # a run without a straggler policy logs none of them


def _stale_steps(watch: StaleWatch, stale: list[int], *, horizon: float, groups: int = 10) -> None:
    for count in stale:
        for index in range(groups):
            watch.dequeued(index < count)
        shares = {"permits": 0.9 - horizon, "horizon": horizon}
        watch.step_ended({f"dispatch/blocked_share/{reason}": share for reason, share in shares.items()})


def test_the_staleness_warning_fires_once_above_2_percent_names_the_binding_constraint_and_both_fixes(caplog):
    early = StaleWatch(3)
    _stale_steps(early, [5] * 9, horizon=0.0)
    assert not early.warned  # half of each step's groups went stale, but over fewer than 10 steps
    watch = StaleWatch(3)
    with caplog.at_level(logging.WARNING):
        _stale_steps(watch, [0, 1, 0, 0, 0, 1, 0, 0, 0, 0], horizon=0.0)  # 2 of 100: not above 2%
        assert not watch.warned
        _stale_steps(watch, [1, 1, 0, 0, 0], horizon=0.0)  # 3 of the last 100
    warnings = [r.getMessage() for r in caplog.records if "dropped as stale" in r.getMessage()]
    assert len(warnings) == 1
    assert "3.0% of the groups the last 10 steps consumed (3 of 100)" in warnings[0]
    assert "the rollout permits (train.max_concurrent_rollouts) (90%)" in warnings[0]
    assert "train.max_staleness: 4 changes only the stale filter" in warnings[0]
    assert "per-datum stale filter" in warnings[0] and "later release" in warnings[0]

    bound = StaleWatch(2)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _stale_steps(bound, [1] * 10, horizon=0.6)
    [warning] = [r.getMessage() for r in caplog.records if "dropped as stale" in r.getMessage()]
    assert "the horizon (train.max_staleness) (60%)" in warning
    assert "train.max_staleness: 3 also lets rollouts run one more step ahead" in warning


def test_a_run_with_3_percent_stale_groups_warns_once(tmp_path, monkeypatch, caplog):
    """Twelve steps of 10 groups, of which 3 of the first 100 come back older than the bound."""

    config = trainer_config(
        tmp_path,
        "train.batch_size=10",
        "train.max_training_steps=12",
        "train.max_staleness=1",
        "train.max_concurrent_rollouts=10",
        "train.workflow_config.group_size=1",
    )
    trainer, _saves, logged = make_trainer(monkeypatch, config, tasks=120)
    stale = {7, 41, 77}

    class Groups:
        async def arun_episode(self, row):
            position = row[POSITION_COLUMN]
            await asyncio.sleep(0)
            version = trainer._model_info.llm.version - (5 if position in stale else 0)
            return [_datum(position, 0, version)]

    with caplog.at_level(logging.WARNING):
        asyncio.run(asyncio.wait_for(trainer.train(train_workflow=Groups()), 60))
    assert len(logged) == 12
    warnings = [r.getMessage() for r in caplog.records if "dropped as stale" in r.getMessage()]
    assert len(warnings) == 1 and "(3 of 100)" in warnings[0]


class _SaysSoButBypasses(_Bypass):
    """Says it runs its rollouts on the lease, but starts its members itself, outside any lease block."""

    runs_rollouts_on_lease = True


def test_a_workflow_that_runs_rollouts_outside_their_blocks_is_warned_of_once(tmp_path, monkeypatch, caplog):
    """Under rolling starts, a group of 2 that starts on its one free permit runs both its members at once: more
    rollouts run than the budget of 3, and the run warns once.  It sets no straggler policy, so no censoring shows."""

    config = trainer_config(
        tmp_path,
        "train.batch_size=2",
        "train.max_training_steps=3",
        "train.max_staleness=1",
        "train.max_concurrent_rollouts=3",
        "train.workflow_config.group_size=2",
        "train.permit_release=rollout",
    )
    trainer, _saves, _logged = make_trainer(monkeypatch, config, tasks=8)
    logged = _logging_stats(trainer)

    async def script(row, member):
        await asyncio.sleep(0.05)
        return float(member)

    workflow = _SaysSoButBypasses(config.train.workflow_config, trainer._model_info.llm, script)
    with caplog.at_level(logging.WARNING):
        asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 60))
    assert max(stats["rollout_client/rollouts_in_flight_max"] for stats in logged) > 3
    warnings = [r.getMessage() for r in caplog.records if "more than train.max_concurrent_rollouts" in r.getMessage()]
    assert len(warnings) == 1 and "rollouts ran at once" in warnings[0]
    assert not [key for stats in logged for key in stats if "censoring" in key]


def test_permit_release_takes_group_or_rollout(tmp_path):
    assert trainer_config(tmp_path).train.permit_release == "group"
    assert trainer_config(tmp_path, "train.permit_release=rollout").train.permit_release == "rollout"
    with pytest.raises(ConfigError, match="permit_release"):
        trainer_config(tmp_path, "train.permit_release=member")
