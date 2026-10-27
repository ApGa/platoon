"""A group's startup permit under ``rollout_isolation: process`` (``platoon.train.workflow.members``): one permit for
the whole group, which comes back once every member reported its worker ready or ended.  So it is held while a member
that is still starting unwinds a cancellation, whatever cancelled it (a sibling that raised, the group's own
cancellation, the straggler cut), and it comes back by the group's end even when no member reports anything (an
override of ``_run_group_member``), not on its timer.

``test_group_rollout_isolation.py`` checks that whole groups are admitted, and that the readiness a worker reports
gives the permit back.  A subclass that overrides the gate's methods under their names before platoon 0.5.0, which the
workflow no longer calls, is warned of."""

from __future__ import annotations

import asyncio
import warnings

import pytest
from fakes.members import ScriptedMembers, scripted_group, until

from platoon.deprecations import PlatoonDeprecationWarning
from platoon.train.config_defs import WorkflowConfig
from platoon.train.dispatch.dispatcher import POSITION_COLUMN
from platoon.utils.stats_tracker import get as get_tracker

#: How a member that is still starting is cancelled, and the straggler policy it runs under.
ENDS = {
    "a sibling raised": {},
    "the group was cancelled": {"straggler_timeout_seconds": 30.0, "straggler_quorum": 1},
    "cut as a straggler": {"straggler_timeout_seconds": 0.05, "straggler_quorum": 1},
}


def _config(**straggler_policy) -> WorkflowConfig:
    """Groups of 2 on one startup permit, whose timer (the startup budget plus 60 s) never fires in a test."""

    return WorkflowConfig(
        group_size=2,
        rollout_isolation="process",
        worker_launch_concurrent_groups=1,
        worker_startup_budget_seconds=30.0,
        filter_zero_advantage_datums=False,
        **straggler_policy,
    )


def _permit_held(workflow) -> bool:
    return workflow._startup_semaphore().locked()


@pytest.mark.parametrize("end", list(ENDS))
def test_the_permit_is_held_while_a_member_still_starting_unwinds(end):
    """Member 0 ends at once (it raises, when its sibling's end is that it did); member 1 never reports its worker
    ready, and takes 50 ms to unwind its cancellation, as a worker's termination grace does."""

    held_while_unwinding: list[bool] = []

    async def script(row, member):
        if member == 0:
            if end == "a sibling raised":
                raise RuntimeError("member 0 failed")
            return 1.0
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
            held_while_unwinding.append(_permit_held(workflow))
            raise
        return 0.0

    scope = "test_permit_unwinding"
    workflow = scripted_group(_config(**ENDS[end]), script, scope)

    async def run():
        group = asyncio.ensure_future(workflow._run_group_rollouts({"task_id": "t", POSITION_COLUMN: 0}))
        if end == "the group was cancelled":
            await until(lambda: ("end", 0, 0) in workflow.log)
            group.cancel()
        outcome = await asyncio.gather(group, return_exceptions=True)
        assert not _permit_held(workflow)  # and back once the group has ended
        return outcome

    (outcome,) = asyncio.run(run())
    assert held_while_unwinding == [True]
    assert workflow.log[-1] == ("cancelled", 0, 1)
    expected = {"a sibling raised": RuntimeError, "the group was cancelled": asyncio.CancelledError}.get(end, list)
    assert isinstance(outcome, expected)
    get_tracker(scope).export(reset=True)


class _Unreported(ScriptedMembers):
    """Overrides ``_run_group_member`` and runs the member itself, without ``super()``, reporting nothing to the group's
    startup permit: neither readiness nor an end."""

    async def _run_group_member(self, data, rollout_number, startup_pass):
        return await self.arun_episode_single(data, rollout_number)


@pytest.mark.parametrize(
    ("straggler_policy", "cancelled"),
    [({}, False), ({}, True), (ENDS["cut as a straggler"], False), (ENDS["the group was cancelled"], True)],
    ids=["returned", "cancelled", "returned-under-a-straggler-policy", "cancelled-under-a-straggler-policy"],
)
def test_a_group_whose_members_report_nothing_gives_the_permit_back_at_its_end(straggler_policy, cancelled):
    async def script(row, member):
        await asyncio.sleep(30 if cancelled else 0.2)
        return float(member)

    scope = "test_permit_unreported"
    workflow = scripted_group(_config(**straggler_policy), script, scope, _Unreported)

    async def run():
        group = asyncio.ensure_future(workflow._run_group_rollouts({"task_id": "t", POSITION_COLUMN: 0}))
        await until(lambda: len(workflow.started(0)) == 2)
        assert _permit_held(workflow)  # the members run, and none reported its worker ready
        if cancelled:
            group.cancel()
        await asyncio.gather(group, return_exceptions=True)
        return _permit_held(workflow)

    assert asyncio.run(run()) is False
    get_tracker(scope).export(reset=True)


def _never_called(old: str, new: str) -> str:
    """The warning of a subclass that defines ``old``, which platoon 0.5.0 renamed ``new``."""

    return rf"\.{old} is never called: platoon 0\.5\.0 renamed GroupRolloutWorkflow\.{old} to {new}, so override {new}"


def test_an_override_of_the_gate_under_its_name_before_platoon_0_5_0_is_warned_of_since_nothing_calls_it():
    with pytest.warns(PlatoonDeprecationWarning, match=_never_called("_admit_group", "_enter_startup_gate")):

        class Old(ScriptedMembers):
            async def _admit_group(self, group_size):
                return await self._enter_startup_gate(group_size)

    with pytest.warns(PlatoonDeprecationWarning, match=_never_called("_admission_semaphore", "_startup_semaphore")):

        class Older(ScriptedMembers):
            def _admission_semaphore(self):
                return self._startup_semaphore()

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class New(ScriptedMembers):  # the names the workflow calls: no warning
            async def _enter_startup_gate(self, group_size):
                return await super()._enter_startup_gate(group_size)
