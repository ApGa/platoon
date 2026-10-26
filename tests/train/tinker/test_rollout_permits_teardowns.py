"""In process, a rollout's permit waits for the teardowns its rollout left running in the background (RFC 0020).

Under ``train.permit_release: rollout`` a member's lease block holds its permit until the sandbox and env session
teardowns that platoon finishes in the background (``platoon.utils.stopping.let_finish``) have ended, at most
``CLEANUP_WAIT_SECONDS``, so no more sandboxes are alive at once than the budget: with every teardown deferred, and
with a straggler cut that lands in a teardown.  A teardown that hangs holds the permit only that long; a cancellation
ends the wait and no teardown; a member whose rollout had ended keeps its outcome.  Under ``group`` no member waits.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from test_rollout_permits_run import _group

from platoon.train.config_defs import WorkflowConfig
from platoon.train.dispatch.dispatcher import POSITION_COLUMN
from platoon.train.dispatch.permits import PermitPool
from platoon.train.leases import holding
from platoon.train.workflow import members as members_module
from platoon.utils.stats_tracker import get as get_tracker
from platoon.utils.stopping import let_finish


class _Sandboxes:
    """The sandboxes alive, the most at once, and the teardowns that stop them."""

    def __init__(self) -> None:
        self.alive = 0
        self.peak = 0
        self.teardowns: set[asyncio.Task] = set()

    def up(self) -> None:
        self.alive += 1
        self.peak = max(self.peak, self.alive)

    def stop_in_background(self, seconds: float | asyncio.Event) -> asyncio.Task:
        """Stop a sandbox in a teardown that platoon finishes in the background, which takes ``seconds`` or runs until
        the event is set."""

        async def stop() -> None:
            try:
                await (asyncio.sleep(seconds) if isinstance(seconds, float) else seconds.wait())
            finally:
                self.alive -= 1

        teardown = let_finish(asyncio.ensure_future(stop()))
        self.teardowns.add(teardown)
        return teardown


def _three_draws(shape: str) -> int:
    """Three draws of 4 on a budget of 4 under ``rollout``, each member's rollout a sandbox stopped by a 0.3 s teardown:
    ``deferred`` starts it and returns, as an env torn down once the code still running in it ends; ``shielded`` waits
    for it, as a session's close does, and member 3's straggler cut (quorum 3 at ~0.31 s, grace 0.1 s) lands in its
    teardown.  Returns the most sandboxes alive at once."""

    sandboxes = _Sandboxes()
    if shape == "shielded":
        work = {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.35}
        config = WorkflowConfig(
            group_size=4, straggler_timeout_seconds=0.1, straggler_quorum=3, filter_zero_advantage_datums=False
        )
    else:
        work = {0: 0.05, 1: 0.10, 2: 0.15, 3: 0.40}
        config = WorkflowConfig(group_size=4, filter_zero_advantage_datums=False)

    async def script(row, member):
        sandboxes.up()
        try:
            await asyncio.sleep(work[member])
        finally:
            teardown = sandboxes.stop_in_background(0.3)
            if shape == "shielded":
                await asyncio.shield(teardown)
        return float(member % 2)

    scope = f"test_teardowns_{shape}"
    workflow = _group(config, script, scope)

    async def run() -> None:
        pool = PermitPool(4)
        groups = []
        for draw in range(3):
            lease = pool.reserve(4, rolling=True, active=True)
            await lease.ready()

            async def group(lease=lease, draw=draw) -> None:
                try:
                    with holding(lease):
                        row = {"task_id": f"t{draw}", POSITION_COLUMN: draw}
                        await asyncio.ensure_future(workflow.arun_episode(row))
                finally:
                    lease()

            groups.append(asyncio.ensure_future(group()))
        await asyncio.wait_for(asyncio.gather(*groups), 30)
        await asyncio.gather(*sandboxes.teardowns)
        assert sandboxes.alive == 0 and pool.free == 4

    asyncio.run(run())
    if shape == "shielded":
        assert [member for kind, _, member in workflow.log if kind == "cancelled"] == [3, 3, 3]
    get_tracker(scope).export(reset=True)
    return sandboxes.peak


@pytest.mark.parametrize("shape", ["deferred", "shielded"])
def test_under_rollout_no_more_sandboxes_are_alive_at_once_than_the_budget(shape):
    assert _three_draws(shape) <= 4


def test_under_group_no_member_waits_for_the_teardowns_its_rollout_left_running():
    """The blocks of a draw under ``group`` hold nothing: the group returns while its teardowns run on, as before."""

    sandboxes = _Sandboxes()

    async def run() -> None:
        released = asyncio.Event()

        async def script(row, member):
            sandboxes.up()
            sandboxes.stop_in_background(released)
            return float(member % 2)

        config = WorkflowConfig(group_size=2, filter_zero_advantage_datums=False)
        workflow = _group(config, script, "test_teardowns_group")
        lease = PermitPool(2).reserve(2)
        with holding(lease):
            output = await asyncio.wait_for(workflow.arun_episode({"task_id": "t", POSITION_COLUMN: 0}), 10)
        lease()
        assert output.observed_rollouts == 2 and sandboxes.alive == 2  # its teardowns still run
        released.set()
        await asyncio.gather(*sandboxes.teardowns)

    asyncio.run(run())
    get_tracker("test_teardowns_group").export(reset=True)


def test_a_teardown_that_hangs_holds_the_permit_only_for_the_bound(monkeypatch, caplog):
    monkeypatch.setattr(members_module, "CLEANUP_WAIT_SECONDS", 0.2)
    sandboxes = _Sandboxes()

    async def run() -> None:
        hung = asyncio.Event()  # never set while the group runs

        async def script(row, member):
            sandboxes.up()
            sandboxes.stop_in_background(hung)
            return 1.0

        config = WorkflowConfig(group_size=1, filter_zero_advantage_datums=False)
        workflow = _group(config, script, "test_teardowns_hang")
        pool = PermitPool(1)
        lease = pool.reserve(1, rolling=True, active=True)
        loop = asyncio.get_running_loop()
        started = loop.time()
        with holding(lease):
            output = await asyncio.wait_for(workflow.arun_episode({"task_id": "t", POSITION_COLUMN: 0}), 10)
        assert loop.time() - started >= 0.2 and pool.free == 1  # its permit came back after the bound
        assert output.observed_rollouts == 1
        (teardown,) = sandboxes.teardowns
        assert not teardown.done()  # and the teardown runs on
        lease()
        hung.set()
        await teardown

    with caplog.at_level(logging.WARNING):
        asyncio.run(run())
    assert any("1 of its teardowns still running after 0.2s" in record.getMessage() for record in caplog.records)
    get_tracker("test_teardowns_hang").export(reset=True)


def test_cancelling_a_member_that_waits_for_its_teardowns_ends_the_wait_and_no_teardown():
    sandboxes = _Sandboxes()

    async def run() -> None:
        released, waiting = asyncio.Event(), asyncio.Event()

        async def script(row, member):
            sandboxes.up()
            sandboxes.stop_in_background(released)
            waiting.set()
            return 1.0

        workflow = _group(
            WorkflowConfig(group_size=1, filter_zero_advantage_datums=False), script, "test_teardowns_cancel"
        )
        pool = PermitPool(1)
        lease = pool.reserve(1, rolling=True, active=True)
        with holding(lease):
            group = asyncio.ensure_future(workflow.arun_episode({"task_id": "t", POSITION_COLUMN: 0}))
        await waiting.wait()
        await asyncio.sleep(0.05)
        assert not group.done() and pool.free == 0  # the member waits for its teardown, on its permit
        group.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(group, 5)
        assert pool.free == 1  # the wait ended: the permit is back
        (teardown,) = sandboxes.teardowns
        assert not teardown.done()  # the teardown was not cancelled
        lease()
        released.set()
        await teardown
        assert sandboxes.alive == 0

    asyncio.run(run())
    get_tracker("test_teardowns_cancel").export(reset=True)


def test_a_straggler_cut_while_a_member_waits_for_its_teardowns_keeps_its_outcome():
    """Its rollout had ended within its grace: the cut ends only the wait, and the teardown runs on."""

    sandboxes = _Sandboxes()
    config = WorkflowConfig(
        group_size=2, straggler_timeout_seconds=0.1, straggler_quorum=1, filter_zero_advantage_datums=False
    )

    async def run() -> None:
        released = asyncio.Event()

        async def script(row, member):
            if member == 1:
                await asyncio.sleep(0.05)
                sandboxes.up()
                sandboxes.stop_in_background(released)  # it outlasts the grace
            return float(member)

        workflow = _group(config, script, "test_teardowns_cut_in_wait")
        lease = PermitPool(2).reserve(2, rolling=True, active=True)
        with holding(lease):
            output = await asyncio.wait_for(workflow.arun_episode({"task_id": "t", POSITION_COLUMN: 0}), 10)
        lease()
        assert output.observed_rollouts == 2 and ("cancelled", 0, 1) not in workflow.log
        (teardown,) = sandboxes.teardowns
        assert not teardown.done()
        released.set()
        await teardown

    asyncio.run(run())
    get_tracker("test_teardowns_cut_in_wait").export(reset=True)
