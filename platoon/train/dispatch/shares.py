"""The shares of a run's groups when its trained policies step independently (RFC 0021, section 2.11): what a trainer's
steps take each policy's shares from (:class:`ShareLedger`, which ``platoon.experimental`` exports), and the run's
ledger of them (:class:`Shares`).

Stepping in lockstep, a step trains every trained policy's rows of the same groups.  Stepping independently, each group
that settles is credited once, over all its agents, and then split into **shares**: one per trained policy with rows in
it, each holding that policy's rows only, and always one of the leader's, with none of its rows in a group that has none
(a group with no trained policy's rows, or one that failed, is the leader's share alone).  The leader's steps so take
every group, as a lockstep step does, and the plan, ``train.batch_size`` groups per leader step, stays in draws.  Each
policy's steps take its own shares in the order they settled, on its own clock.  The ledger keeps every group that has
not retired: for each of its shares, whether its policy consumed it, at which of its steps, at what age (the policy's
version then minus the oldest version of the policy stamped on the share's rows), and with what fate.

A group **retires** (it leaves the ledger, the kept groups and the dispatcher's outstanding draws, so that no later
checkpoint carries it) once every share was consumed and each consuming policy has a full checkpoint row from a step
after the one that consumed its share (:meth:`Shares.durable`).  Its fate goes to the per-task statistics once, when
its last share is consumed (``recorded``).  So every full checkpoint of any policy holds, for every group that has not
retired, what each policy did with its share, and a resume from it hands each policy back exactly the shares it
consumed after the row it resumes from (:meth:`Shares.revert`), each judged by the age it had when first consumed: it
passed the stale filter once, at the same point of the same lineage, so it trains again whole.

The ledger is the rollout source's (``source.ledger``, :mod:`platoon.train.dispatch.source`), built when the run's
policies step independently.  It takes the place of the result queue (the dispatcher hands it each group that settles,
:meth:`Shares.put_nowait`), counts each policy's shares on the dispatcher's horizons
(:mod:`platoon.train.dispatch.horizons`), and holds the commit lock: the settle section, each step's consumption and
each commit's capture take it, so that a capture sees a group wholly settled and consumed, or not at all.  The rl
trainer's steps take shares one at a time (:meth:`Shares.queue`); a trainer of its own takes them in batches
(:meth:`ShareLedger.take`).

Imports neither Tinker nor torch, so that ``platoon.experimental`` re-exports it.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Protocol

from platoon.train.policy.spec import PolicyId
from platoon.train.trainer.waiting import _next_rollout
from platoon.train.workflow.protocol import OutputReport, report_of, rows_by_policy

if TYPE_CHECKING:
    from platoon.train.checkpoints.resume import StartPoint
    from platoon.train.dispatch.dispatcher import Dispatcher
    from platoon.train.dispatch.horizons import Horizons
    from platoon.train.dispatch.settled import _Settled as Settled  # a settled group, as a step takes it
    from platoon.train.policy.set import PolicySet

DrawKey = tuple[int, int]


class ShareLedger(Protocol):
    """The shares of a run's groups, as a trainer of its own whose policies step independently takes them
    (``run.rollouts().ledger``; ``TrainerNeeds.stepping``): each step of policy ``p`` takes ``p``'s shares
    (:meth:`take`), trains them (a trainer that drops stale ones judges each at :meth:`judge_version`), records them
    consumed under the commit lock (``async with ledger.lock:`` :meth:`consumed`), then ends with
    ``await run.end_step(policy=p)`` on ``p``'s own clock (``run.clocks[p]``).  The run keeps the rest: which shares a
    checkpoint holds, which a resume puts back, and when a group retires."""

    # The commit lock, which a step holds while it records what it consumed, so that a commit captures all of it or
    # none of it.
    lock: asyncio.Lock

    async def take(self, policy: PolicyId, n: int, *, stop: asyncio.Event | None = None) -> list[Settled]:
        """``n`` of ``policy``'s shares, in the order they settled, those a resume put back first: each a settled
        group, as the source's ``take`` gives one in lockstep, whose output (``rollout``) holds ``policy``'s rows of
        the group only (the leader's share of a group without its rows holds none; that of a group that failed is
        None), and whose ``key`` names its draw.  Fewer (none at all, perhaps) once no share will come
        (:meth:`done`): at the plan's end, or once a stop request ended another policy's loop and no group starts
        again.  Once ``stop`` is set (the run's shutdown request, ``run.shutdown``), ``TrainInterrupted`` instead:
        the shares it took stay unconsumed, which the checkpoint at exit keeps.  A trainer that drops stale shares
        judges each at :meth:`judge_version`."""
        ...

    def judge_version(self, policy: PolicyId, group: Settled) -> int:
        """The version of ``policy`` that the staleness of its share ``group`` (one :meth:`take` gave, not yet
        consumed) is judged at: the policy's version now, but for a share a resume put back, the version at which it
        has the age it had when the policy first consumed it (the oldest version stamped on its rows plus that age).
        The resume's version rule moves the policy's version past the row it resumes from, so a share put back,
        judged against the version now, could be dropped as stale, though it trained once on the same lineage."""
        ...

    def consumed(
        self, policy: PolicyId, keys: Iterable[DrawKey | None], *, step: int, fates: Mapping[DrawKey, str] | None = None
    ) -> None:
        """``policy``'s step ``step`` (its clock's) trained or dropped its shares of the groups of ``keys``, each with
        its fate (``fates``: ``consumed``, ``stale`` or ``failed``; default ``consumed``).  Call it under :attr:`lock`,
        before the step ends."""
        ...

    def done(self, policy: PolicyId) -> bool:
        """Whether ``policy``'s loop may end: no group runs that may still give it a share, it has none left to take,
        and none will start: the plan is drawn, or a stop request ended another policy's loop, whose full horizon holds
        back every group for good."""
        ...


class _PlanEnd:
    """What a policy's share queue gives once no share will come to it any more (:data:`PLAN_END`)."""

    def __repr__(self) -> str:
        return "PLAN_END"


# What a policy's share queue gives once no share will come to it any more (Shares.done).
PLAN_END: Final = _PlanEnd()


@dataclass(frozen=True)
class Share:
    """One policy's part of a settled group, as the rl trainer's step of that policy takes it.

    - ``group``: the share as a settled group, its draw's key, entry and label, with an output that holds this
      policy's rows only (the group's report goes with the leader's share, which every group has; the other shares keep
      its statistics and verdict, without its workload), and the group's record for the share that carries the report;
    - ``age``: for a share that a resume put back, the age it had when its policy first consumed it, by which it is
      judged again (None for a fresh share);
    - ``peers``: the oldest version of each other trained policy stamped on the group's rows (a peer's lag); none for
      the leader's share of a group without its rows."""

    policy: PolicyId
    group: Settled
    age: int | None = None
    peers: Mapping[PolicyId, int] = field(default_factory=dict)

    def judged_at(self) -> int | None:
        """For a share a resume put back, the version its stale filter judges it at: the oldest version stamped on its
        rows plus its recorded age, so that it is judged as when its policy first consumed it; None for a fresh share,
        which is judged at its policy's version now."""

        return _judged_at(_rows_of(self), self.age)


@dataclass
class _ShareState:
    """What a policy did with its share of a group: the step that consumed it, its age then and its fate (consumed,
    stale or failed); all None until it is consumed.  A share a resume put back is unconsumed (no step), and keeps the
    age it had when first consumed, by which it is judged again."""

    step: int | None = None
    age: int | None = None
    fate: str | None = None

    def to_json(self) -> dict[str, Any] | None:
        if self.step is None and self.age is None:
            return None
        return {"step": self.step, "age": self.age, "fate": self.fate}

    @classmethod
    def from_json(cls, data: Any) -> _ShareState:
        if data is None:
            return cls()
        step, age = data.get("step"), data.get("age")
        return cls(None if step is None else int(step), None if age is None else int(age), data.get("fate"))


@dataclass(eq=False)
class _Group:
    """A group the ledger keeps: the group as it settled (``item``), its shares and their states, by policy, and whether
    its fate went to the per-task statistics (``recorded``)."""

    item: Settled
    shares: dict[PolicyId, Share]
    states: dict[PolicyId, _ShareState]
    recorded: bool = False

    def consumed(self) -> bool:
        return all(state.step is not None for state in self.states.values())

    def fate(self) -> str:
        """The group's fate once every share was consumed: failed, else consumed when any share trained, else stale.
        The leader's share of a group without its rows counts only when no share holds rows, so that a group whose rows
        all went stale is stale, as a lockstep step judges it."""

        held = [policy for policy, share in self.shares.items() if _rows_of(share)] or list(self.states)
        fates = {self.states[policy].fate for policy in held}
        return "failed" if "failed" in fates else "consumed" if "consumed" in fates else "stale"


class Shares:
    """The run's ledger of the shares of the groups that have not retired (see the module docstring; a trainer uses it
    as a :class:`ShareLedger`), over the run's ``policies``: its trained ones, the leader first, whose versions age
    each share; the ``dispatcher``, whose horizons count each policy's shares and which drops each group that retires;
    and ``record_fates``, which hands a group's fate to the per-task statistics (the source's
    :meth:`~platoon.train.dispatch.source.RolloutSource.record_fates`)."""

    def __init__(
        self,
        policies: PolicySet,
        *,
        dispatcher: Dispatcher,
        record_fates: Callable[[list[DrawKey], Mapping[DrawKey, str]], None],
    ) -> None:
        assert dispatcher.horizons is not None  # a dispatcher of policies that step independently has their horizons
        self.policies = policies
        self.trained = [policy.id for policy in policies.trainable()]
        self.leader = policies.leader
        self.dispatcher = dispatcher
        self.horizons: Horizons = dispatcher.horizons
        self._record_fates = record_fates
        # The commit lock (see the module docstring).
        self.lock = asyncio.Lock()
        self._groups: dict[DrawKey, _Group] = {}
        # Each policy's shares to take: those a resume put back first, then those that settled, in order.
        self._reverted: dict[PolicyId, deque[Share]] = {policy: deque() for policy in self.trained}
        self._fresh: dict[PolicyId, deque[Share]] = {policy: deque() for policy in self.trained}
        # Each policy's newest durable full row, by the step it resumes at: a share it consumed at an earlier step is
        # in that row's weights.
        self._durable: dict[PolicyId, int] = dict.fromkeys(self.trained, 0)
        # What a resume read of the kept groups' shares, by draw, until each group settles again (put_nowait).
        self._restored: dict[DrawKey, tuple[dict[PolicyId, _ShareState], bool]] = {}
        # What each policy consumed at its latest consumption, by draw, with its fate (an outside trainer's step end).
        self.last_consumed: dict[PolicyId, dict[DrawKey, str]] = {policy: {} for policy in self.trained}
        self._changed = asyncio.Event()

    # --- the settle section ------------------------------------------------------------------------------------------

    def put_nowait(self, item: Settled) -> None:
        """A group settled (the dispatcher hands it here in place of the result queue, after credit, in the settle
        section, with no await): split it into shares, one per trained policy with rows in it and always the leader's
        (:meth:`_split`), each queued for its policy's steps and counted on its horizon.  A group a resume kept takes
        back what its policies did with each share (:meth:`revert` hands those over, and the dispatcher's hand-over of
        them again is passed over)."""

        assert item.key is not None  # every group the dispatcher starts has a draw
        if item.key in self._groups:
            return
        shares = self._split(item)
        states, recorded = self._restored.pop(item.key, ({policy: _ShareState() for policy in shares}, False))
        group = _Group(item, shares, {policy: states.get(policy, _ShareState()) for policy in shares}, recorded)
        self._groups[item.key] = group
        for policy, share in shares.items():
            state = group.states[policy]
            if state.step is not None:
                continue  # consumed, at a step its newest durable row holds
            if state.age is not None:  # put back by a resume: judged by its recorded age
                self._reverted[policy].append(dataclasses.replace(share, age=state.age))
            else:
                self._fresh[policy].append(share)
            self.horizons.queued(policy)
        self._retire(group)
        self._wake()

    def _split(self, item: Settled) -> dict[PolicyId, Share]:
        """A group's shares, by policy, in the run's order, the leader's first: each trained policy's rows of it, and
        the leader's share even without its rows, which carries the group's report and record, so that the leader's
        steps take every group, as the plan counts them; the leader's share alone, as the group is, for a group that
        failed or holds no trained policy's rows."""

        from platoon.train.dispatch.settled import _Settled  # what a step takes; it loads the statistics' tensors

        output = item.rollout
        rows = {} if output is None else dict(rows_by_policy(output, self.leader))
        holders = [policy for policy in self.trained if rows.get(policy)]
        if not holders:
            return {self.leader: Share(self.leader, item)}
        if self.leader not in holders:  # its share holds none of the group's rows: its step counts the group
            holders.insert(0, self.leader)
        report = report_of(output)
        oldest = {policy: _oldest_version(rows.get(policy, ())) for policy in holders}
        shares = {}
        for index, policy in enumerate(holders):
            own = report if index == 0 else _without_workload(report)
            record = item.record if index == 0 else None
            group = _Settled(item.entry, item.label, _ShareRows(rows.get(policy, ()), own), record, item.key)
            peers = {peer: version for peer, version in oldest.items() if peer != policy and version is not None}
            shares[policy] = Share(policy, group, peers=peers if rows.get(policy) else {})
        return shares

    # --- taking and consuming shares ---------------------------------------------------------------------------------

    def queue(self, policy: PolicyId) -> ShareQueue:
        """``policy``'s shares as the rl trainer's steps take them, one at a time (``await queue.get()``): those a
        resume put back first, then those that settled, in order; :data:`PLAN_END` once none will come
        (:meth:`done`)."""

        return ShareQueue(self, policy)

    async def take(self, policy: PolicyId, n: int, *, stop: asyncio.Event | None = None) -> list[Settled]:
        """:meth:`ShareLedger.take`: ``n`` of ``policy``'s shares (:meth:`queue`), each as its settled group; fewer once
        no share will come (:meth:`done`); ``TrainInterrupted`` once ``stop`` is set."""

        queue, taken = self.queue(policy), []
        while len(taken) < n:
            share = await _next_rollout(queue, stop, interruptible=stop is not None)
            if share is PLAN_END:
                break
            taken.append(share.group)
        return taken

    def judge_version(self, policy: PolicyId, group: Settled) -> int:
        """:meth:`ShareLedger.judge_version`: for a share a resume put back (unconsumed, with a recorded age), the
        oldest version stamped on its rows plus that age, as :meth:`Share.judged_at`; else ``policy``'s version now."""

        kept = self._groups.get(group.key) if group.key is not None else None
        state = kept.states.get(policy) if kept is not None else None
        if state is not None and state.step is None:  # unconsumed: put back by a resume when it has an age
            judged = _judged_at(() if group.rollout is None else group.rollout, state.age)
            if judged is not None:
                return judged
        return self.policies[policy].version

    def next_share(self, policy: PolicyId) -> Share | None:
        """The next share of ``policy``, counted as taken on its horizon; None when it has none left to take."""

        queue = self._reverted[policy] or self._fresh[policy]
        if not queue:
            return None
        share = queue.popleft()
        self.horizons.taken(policy)
        return share

    def has_shares(self, policy: PolicyId) -> bool:
        """Whether ``policy`` has shares left to take."""

        return bool(self._reverted[policy] or self._fresh[policy])

    def done(self, policy: PolicyId) -> bool:
        """:meth:`ShareLedger.done`: ``policy`` has no share left to take, no group runs, and none will start: the plan
        is drawn, or the full horizon of a policy whose loop a stop request ended holds every group back for good
        (:meth:`~platoon.train.dispatch.horizons.Horizons.held_for_good`), so that a step that applied an update closes
        short with the shares it has, as at the plan's end."""

        if self.has_shares(policy) or self.dispatcher.pipeline:
            return False
        return not self.horizons.plan_has_room(0) or self.horizons.held_for_good()

    def consumed(
        self, policy: PolicyId, keys: Iterable[DrawKey | None], *, step: int, fates: Mapping[DrawKey, str] | None = None
    ) -> None:
        """:meth:`ShareLedger.consumed`: each share's age is recorded (a share a resume put back keeps the one it had),
        a group whose last share this was hands its fate to the per-task statistics, and a group durable for every
        policy retires.  Run under the commit lock, with no await."""

        version = self.policies[policy].version
        consumed: dict[DrawKey, str] = {}
        for key in keys:
            if key is None or (group := self._groups.get(key)) is None or policy not in group.states:
                continue
            age = group.states[policy].age  # a share a resume put back keeps the age it had when first consumed
            if age is None and (oldest := _oldest_version(_rows_of(group.shares[policy]))) is not None:
                age = version - oldest
            fate = fates.get(key, "consumed") if fates is not None else "consumed"
            group.states[policy] = _ShareState(step, age, fate)
            consumed[key] = fate
            if group.consumed() and not group.recorded:
                group.recorded = True
                self._record_fates([key], {key: group.fate()})
            self._retire(group)
        self.last_consumed[policy] = consumed

    def step_started(self, policy: PolicyId) -> None:
        """``policy``'s next step started: its shares the last one took leave its horizon (the leader's counts one
        step of the plan), and its loop, or another's, may find the plan drawn."""

        self.dispatcher.policy_step_started(policy)
        self._wake()

    def stopped(self, policy: PolicyId) -> None:
        """A stop request ended ``policy``'s loop: it takes no more shares, so once its horizon is full no group starts
        again (:meth:`~platoon.train.dispatch.horizons.Horizons.stopped`); each waiting step looks again whether any
        share will come (:meth:`done`)."""

        self.horizons.stopped(policy)
        self._wake()

    # --- durability, capture and resume ------------------------------------------------------------------------------

    def durable(self, policy: PolicyId, step: int) -> list[DrawKey]:
        """``policy`` appended a full row that resumes at its step ``step``: each share it consumed before then is in
        durable weights.  The groups that retire now (:meth:`_retire`)."""

        self._durable[policy] = max(self._durable[policy], step)
        return [key for key, group in list(self._groups.items()) if self._retire(group)]

    def _retire(self, group: _Group) -> bool:
        """Retire ``group`` when every share was consumed at a step its policy's newest durable row holds: it leaves the
        ledger and the dispatcher's kept groups and outstanding draws.  Whether it retired."""

        if not all(
            state.step is not None and state.step < self._durable[policy] for policy, state in group.states.items()
        ):
            return False
        key = group.item.key
        assert key is not None  # every group the ledger keeps has a draw
        del self._groups[key]
        self.dispatcher.committed([key])
        return True

    def snapshot(self) -> dict[DrawKey, dict[str, Any]]:
        """What a commit's capture saves of the groups that have not retired, by draw (each kept group's ``shares`` and
        ``recorded``, :mod:`platoon.train.checkpoints.kept_groups`): taken at once, a copy."""

        return {
            key: {
                "shares": {policy: state.to_json() for policy, state in group.states.items()},
                "recorded": group.recorded,
            }
            for key, group in self._groups.items()
        }

    def restore(self, kept: Mapping[DrawKey, Mapping[str, Any]]) -> None:
        """Take up what the checkpoint a start resumes from saved of each kept group's shares (:meth:`snapshot`'s, as
        the kept-groups file reads back), until :meth:`revert` decides which go back and the group settles again."""

        for key, saved in kept.items():
            states = {policy: _ShareState.from_json(data) for policy, data in (saved.get("shares") or {}).items()}
            self._restored[key] = (states, bool(saved.get("recorded")))

    def revert(self, start: StartPoint) -> None:
        """A resume from ``start``: each policy resumes at its own step (``start.policies``), its row durable.  The
        groups the checkpoint kept (the dispatcher's, restored) are split again at once, in the order they settled:
        each share a policy consumed at that step or later goes back to the front of its queue, with its recorded age;
        one consumed before stays consumed; a group durable for every policy retires."""

        for policy in self.trained:
            begins = start.policies.get(policy)
            self._durable[policy] = begins.step if begins is not None and begins.step is not None else start.step
        for states, _recorded in self._restored.values():
            for policy, state in states.items():
                if state.step is not None and state.step >= self._durable.get(policy, 0):
                    states[policy] = _ShareState(age=state.age if state.age is not None else 0)
        for _key, item in self.dispatcher.kept():
            self.put_nowait(item)

    def retained(self) -> int:
        """The groups the ledger keeps, which every commit writes (``run/retained_groups``)."""

        return len(self._groups)

    def _wake(self) -> None:
        """Wake every step waiting for a share or for the plan's end."""

        self._changed.set()
        self._changed = asyncio.Event()

    async def changed(self) -> None:
        """Wait until a share is queued, a step starts, or a loop stopped: a waiting step looks again."""

        await self._changed.wait()


class ShareQueue:
    """A policy's shares as the rl trainer's steps take them: ``get()`` waits for the next one, or :data:`PLAN_END`.  It
    stands in for the result queue that a lockstep step dequeues from (``get``, ``get_nowait``, ``empty``)."""

    def __init__(self, ledger: Shares, policy: PolicyId) -> None:
        self.ledger, self.policy = ledger, policy

    async def get(self) -> Share | _PlanEnd:
        ledger = self.ledger
        while True:
            if (share := ledger.next_share(self.policy)) is not None:
                return share
            if ledger.done(self.policy):
                return PLAN_END
            await ledger.changed()

    def get_nowait(self) -> Share:
        if (share := self.ledger.next_share(self.policy)) is None:
            raise asyncio.QueueEmpty
        return share

    def empty(self) -> bool:
        return not self.ledger.has_shares(self.policy)


class _ShareRows(list):
    """A share's output: its policy's rows of the group, with the report it carries (``report``)."""

    def __init__(self, rows: Sequence[Any], report: OutputReport) -> None:
        super().__init__(rows)
        self.report = report


def _without_workload(report: OutputReport) -> OutputReport:
    """The report of a share other than the one that carries the group's: its statistics, members and verdict, without
    the workload and the group's own statistics, which the group's first share's step records once."""

    return dataclasses.replace(
        report,
        workload=None,
        requested_rollouts=None,
        observed_rollouts=None,
        trainable_rollouts=None,
        task_retained_datums=None,
        group_stats=None,
    )


def _rows_of(share: Share) -> Sequence[Any]:
    """A share's rows (none for a group that failed)."""

    output = share.group.rollout
    return () if output is None else output


def _judged_at(rows: Sequence[Any], age: int | None) -> int | None:
    """The version a share that a resume put back is judged at, whose ``rows`` its policy first consumed at ``age``: the
    oldest version stamped on them plus that age, so that it is judged as it was then; None for a fresh share (no age)
    or rows that carry no version."""

    if age is None:
        return None
    oldest = _oldest_version(rows)
    return None if oldest is None else oldest + age


def _oldest_version(rows: Sequence[Any]) -> int | None:
    """The oldest policy version stamped on ``rows`` (each datum's ``checkpoint_version``); None when none is."""

    versions = []
    for datum in rows:
        stamp = getattr(datum, "loss_fn_inputs", {}).get("checkpoint_version")
        if stamp is not None:
            versions.append(int(stamp.to_torch().reshape(-1)[0].item()))
    return min(versions, default=None)
