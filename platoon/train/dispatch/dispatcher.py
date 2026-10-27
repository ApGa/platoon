"""The dispatcher of a training run: it starts each rollout group when the group can start (RFC 0019, section 4).

The start's rollout source (:mod:`platoon.train.dispatch.source`) runs it beside the training loop.
:meth:`Dispatcher.run` draws a task only when its group can start.  It starts each group as a task of its own, whose
output settles into the queue the training steps dequeue from.  A group starts when three things allow it:

- **The horizon.**  Let ``k`` be ``train.max_staleness``.  The pipeline is the groups dispatched and not yet dequeued
  by a step: running, or settled and waiting for a step.  The pipeline plus the groups the current step has dequeued
  stay below ``(k + 1) × batch_size``, so the rollouts of a step run at most ``k`` steps ahead of training.  The count
  drops back when a step starts, after its policy is published.  Without a bound there is no horizon.  Either way the
  dispatcher never starts more groups than the run's remaining steps consume, so no group runs past the plan (under a
  group admission rule, a step that closes short consumes fewer: see below).  With ``k = 0`` a step trains exactly the
  groups drawn for it, unless a label's limit holds one of them back.
- **Rollout permits** (``train.max_concurrent_rollouts``, :mod:`platoon.train.dispatch.permits`): a group reserves its
  entry's group size of them, first come first served, and starts once the permits can grant it something: all of
  them at once, or, under ``train.permit_release: rollout`` with a workflow that runs its rollouts on the lease, the
  first, its members starting as the rest free.  It returns them when its workflow returns, or, under ``rollout``,
  each as its rollout ends (:class:`~platoon.train.leases.RolloutLease`).
- **Label capacity** (``train.rollout_group_limits``): at most that many groups of a label run at once.

**The order.**  Each new position of the run's order is drawn through the run's task samplers
(:class:`~platoon.train.sampler_runtime.SamplerSet`): the run's sampler decides position ``n`` (its decision ``n``),
and when it hands the slot to an environment entry, as ``weighted`` does, that entry's sampler decides the task.  The
built-in samplers draw the rows platoon 0.3.0's orders drew, position for position; an order handed to the dispatcher
directly is drawn as it is.

**Adaptive samplers.**  An adaptive sampler (one that reads the run's per-task statistics, such as ``pass_rate``)
decides only when a group can start: within the horizon, with some label of its pool below its limit, and with the
group's rollout permits already held, so that a decision is never made a group's duration before its group starts.
Under ``weighted``, a slot whose entry samples adaptively consumes its position at once.  It waits as a **token** in
the entry's own first-in, first-out queue until the entry can start a group; past the backlog limit it is skipped, and
its position is used up.  The token being started waits for its permits, and is saved with the dispatcher's state
first among its entry's, so no full save loses its position.  An adaptive run sampler decides over the whole pool,
holding permits for the run's largest group, and its draw takes the next position when it binds.  A decision that
finds nothing eligible decides nothing, and the dispatcher waits for a change.

A draw of the order whose label is at its limit waits in the label's first-in, first-out backlog, which does not count
toward the horizon, so the dispatcher draws on for the other labels.  Once :data:`BACKLOG_BATCHES` batches of a label's
draws wait, a new draw of that label is skipped: its position is used up, its fate is ``skipped``, and a warning says so
once.  The dispatcher draws only while some label of the run has room, so it never draws only to skip, and a skip
always leaves the slot to a label that can use it.  Each draw it starts is, in this order: the oldest waiting draw whose
label has room (the draws a resume reissued come first), then the next position of the order.  It wakes when something
it waits for changes, never by polling: a group's release of its label and permits, a step that dequeues a group, a step
that starts.  Each step's metrics say how long it was held back from starting a group, and by what
(:meth:`Dispatcher.blocked_shares`).

**Slots held for a capped label.**  The horizon's slots free when a step starts, and the draws that start then take
them, so a capped label whose group ends in the middle of a step would wait for the next step to start its next group:
a limit of ``n`` would let the label start at most ``n`` groups a step, however short its groups.  So while a capped
label has draws waiting, each of its running groups holds one more slot of the horizon, which the other labels' draws
leave free, for the draw that follows it: when the group ends, that draw starts at once.  It holds them only as far as
the capped labels' running groups and the slots they hold fit in ``k`` batches, so that the other labels always have a
batch of the horizon; with ``k = 0`` there is no slot to hold, and a capped label starts at most its limit of groups
each step.

**Independent stepping.**  When the run's trained policies step each on its own clock (``train.stepping:
independent``), each settled group is split into shares, one per trained policy with rows in it and always one of the
leader's (:mod:`platoon.train.dispatch.shares`), and each such policy has a horizon of its own instead of the one above
(:mod:`platoon.train.dispatch.horizons`): a group starts only when every horizon has room for it, and the leader's
remaining steps bound the plan, ``batch_size`` groups each.  ``pipeline`` then counts the groups running, and each
horizon the shares settled and taken; a group's settling hands it to the run's share ledger in place of the result
queue.  Each step's metrics name the horizon that held the dispatcher back
(``dispatch/blocked_share/horizon:<policy>``).

**Exact resume.**  What a full checkpoint saves of the dispatcher, and how a start that resumes takes it up, is
:mod:`platoon.train.checkpoints.dispatch_state`'s: every draw is numbered and stamped on its row, and the
dispatcher keeps every draw no committed step consumed (outstanding) until a committed step consumes it.

**Group admission** (RFC 0019, section 6; :mod:`platoon.train.admissions`).  In a run where some entry's rule is not
``all``, a group is decided when it settles.  An admitted group goes to the result queue.  A rejected or failed one
leaves the outstanding draws, and only a marker with its statistics goes to the queue (:meth:`Dispatcher.reject`).
A rejected group that carries datums frees its slot of the horizon at once, so that the next group starts while the
trainer trains.  A failed group, or one that settled without a datum (``no_results`` and the like), keeps its slot
until a step dequeues its marker, as in all mode: such groups can end at once (an outage), and freeing their slots
would start groups as fast as they end for as long as the trainer awaits an update or a save.  The horizon then
counts the groups that can still train: ``pipeline`` the groups running, the admitted ones waiting and the markers that
keep their slots, ``admitted`` those the current step counted toward its batch.  A step counts each group it dequeues
and each marker of a group rejected while it was current, up to its cap of ``ceil(refill.max_groups_per_step ×
batch_size)`` (:meth:`Dispatcher.took`); a marker of an earlier step's passes over.  At ``max_staleness: 0``, and on
the run's last step, no group starts that the current step could not count (what it counted, the groups in the
pipeline and its queued markers stay below the cap), so none of the groups it starts is left running when it closes.
The cap bounds what a step counts, not what starts while the trainer trains: at ``k >= 1`` the horizon and the rollout
permits bound those (after a step that admitted ``a`` groups, about ``((k + 1) × batch_size - a) / p`` start during
its update, ``p`` the share of groups admitted).  And since a step can close short of its batch, a run's last steps can
consume fewer groups than the end-of-plan bound counts on: up to ``batch_size`` groups started before the last step can
still be running or waiting when it closes, and the final checkpoint keeps them.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, defaultdict, deque
from collections.abc import Awaitable, Callable, Container, Coroutine, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from platoon.train.dispatch.horizons import Horizons
from platoon.train.dispatch.permits import DrawLease
from platoon.train.sampler_runtime import Owed, SamplerSet
from platoon.train.task_stats import MAIN

logger = logging.getLogger(__name__)


# How many draws of a label at its limit may wait, as a multiple of the batch size, before its new draws are skipped.
BACKLOG_BATCHES = 4
# The columns the dispatcher stamps on each training row it draws: the start's draw number and the row's place in the
# order (its position; a reissued draw keeps the position of the draw it replaces).
SEQ_COLUMN = "_platoon_seq"
POSITION_COLUMN = "_platoon_position"

Key = tuple[int, int]
# An entry whose tasks carry several labels: its tokens hold no label's slot.
_NO_LABEL = object()
# What held the dispatcher back from starting a group (blocked_shares): it awaited rollout permits; the horizon was
# full; every label it could start was at its limit; the run's remaining steps needed no more groups; an adaptive
# sampler found nothing eligible.
BLOCKED_REASONS = ("permits", "horizon", "labels", "plan", "sampler")


class Permits(Protocol):
    """Rollout permits that a group reserves before it starts (:class:`~platoon.train.dispatch.permits.PermitPool`)."""

    def reserve(self, count: int, *, rolling: bool = False, active: bool = False) -> DrawLease: ...


class RowReader(Protocol):
    """What the dispatcher reads of each row it draws (``of``: its entry, its label and its group size), and the run's
    rollout permits."""

    @property
    def permits(self) -> Permits: ...

    def of(self, data: Mapping[str, Any]) -> tuple[str | None, str | None, int]: ...


class SettledSink(Protocol):
    """Where the dispatcher hands each group that settles: the result queue the steps dequeue from, or, when the run's
    policies step independently, the share ledger, which splits it into shares."""

    def put_nowait(self, item: Any) -> None: ...


class DrawRecorder(Protocol):
    """The run's per-task statistics, as the dispatcher tells them of each draw and skip
    (:class:`~platoon.train.task_records.TaskRecorder`)."""

    def draw(
        self,
        row: Mapping[str, Any],
        *,
        seq: int,
        position: int | None,
        version: int,
        reissues: tuple[int, int] | None = None,
        sampler: str | None = None,
        reason: str | None = None,
        score: float | None = None,
    ) -> None: ...

    def skipped(self, row: Mapping[str, Any]) -> None: ...


@dataclass(eq=False)
class Token:
    """A slot of the run's order owed to an entry whose sampler is adaptive: its position, the entry, and its ticket
    among the waiting items.  The entry's sampler decides its task when the entry can start a group.  ``fallback``:
    the run sampler's fallback handed the slot, so the draw that fills it is the run's fallback."""

    position: int
    entry: str
    ticket: int
    fallback: bool = False


class _Ask:
    """A decision of the run's adaptive sampler, over the whole pool, when a group can start."""


ASK = _Ask()


@dataclass(eq=False)
class BoundDraw:
    """One drawn task: its row, stamped with its draw number and position, its key in the run's statistics
    (``(start, seq)``), and what the dispatcher reads of it: its entry, its label and its group size; with its row's
    index in the run's task pool (None for a row an order handed in) and the sampler that drew it."""

    row: dict[str, Any]
    start: int
    seq: int
    position: int | None
    task_id: str
    entry: str | None
    label: str | None
    group_size: int
    # Taken in the order the draws were made (or restored): of the waiting draws whose label has room, the one with the
    # smallest ticket starts first.
    ticket: int
    index: int | None = None
    sampler: str | None = None

    @property
    def key(self) -> Key:
        return (self.start, self.seq)


# A training group's coroutine: it runs the group of ``draw`` on its lease (``DrawLease``), calls the lease once its
# workflow has returned (so that the group's label slot and the rollout permits it still holds are free before anything
# else), then hands its output to :meth:`Dispatcher.settle`.
GroupRunner = Callable[[BoundDraw, DrawLease], Awaitable[None]]


class Dispatcher:
    """Starts a training run's rollout groups as the horizon, the rollout permits and the label limits allow (see the
    module docstring).  Its arguments:

    - ``rows`` reads each row's entry, label and group size, and holds the run's rollout permits.
    - ``samplers`` draws each position of the run's order (:class:`~platoon.train.sampler_runtime.SamplerSet`).
      Instead, ``order(position)`` may give the row at each position, which is drawn as it is.
    - ``steps``: the training steps this start runs.
    - ``labels``: every label of the run's training tasks.  The list must be complete: the dispatcher draws only while
      one of them has room, and learns of another label only by drawing it.  So a label it is not given never counts
      as having room before its first draw.
    - ``recorder``: the run's per-task statistics.  It is told of each skip, and of each draw at the version of the
      sampled policy (``leader``).
    - ``versions()``: the policies' versions now (``{main: v}``); ``step``: the training steps committed when this start
      began.  The samplers' contexts carry both.
    - ``start`` numbers this start's draws, and the order starts at ``next_position``.
    - ``groups_per_step``: the cap of a run under a group admission rule.  With it, the dispatcher counts its steps as
      the module docstring says (None: every group counts toward the batch).
    - ``permit_release`` is ``train.permit_release``.
    - ``horizons``: each trained policy's horizon, when the run's policies step independently (None: the one horizon
      of a run that steps in lockstep)."""

    def __init__(
        self,
        *,
        rows: RowReader,
        samplers: SamplerSet | None = None,
        order: Callable[[int], Mapping[str, Any]] | None = None,
        batch_size: int,
        max_staleness: int | None,
        steps: int,
        limits: Mapping[str, int] | None = None,
        labels: Iterable[str | None] = (),
        start: int = 0,
        next_position: int = 0,
        backlog_limit: int | None = None,
        recorder: DrawRecorder | None = None,
        versions: Callable[[], Mapping[str, int]] = lambda: {MAIN: 0},
        leader: str = MAIN,
        finish_batches: bool = False,
        sampler: str = "auto",
        fingerprint: Mapping[str, Any] | None = None,
        step: int = 0,
        groups_per_step: int | None = None,
        permit_release: str = "group",
        horizons: Horizons | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if samplers is None:
            if order is None:
                raise TypeError("a Dispatcher draws through samplers or an order: give one")
            samplers = SamplerSet.of_order(order, name=sampler, batch_size=batch_size)
        if backlog_limit is not None and (
            isinstance(backlog_limit, bool) or not isinstance(backlog_limit, int) or backlog_limit < 1
        ):
            raise ValueError("backlog_limit must be a positive integer")
        self._rows = rows
        self.samplers = samplers
        self.batch_size = batch_size
        self.max_staleness = max_staleness
        self._limits: dict[str | None, int] = {label: limit for label, limit in (limits or {}).items()}
        self._labels: set[str | None] = set(labels)
        self.start = start
        self.next_position = next_position
        self.backlog_limit = backlog_limit if backlog_limit is not None else BACKLOG_BATCHES * batch_size
        self._recorder = recorder
        self._versions, self._leader = versions, leader
        # With several updates per batch, a step that applied one finishes after a stop request, so the groups it still
        # needs must start: the dispatcher then goes on until the train loop's task group cancels it.
        self.finish_batches = finish_batches
        self.sampler = sampler
        self.fingerprint = dict(fingerprint or {})
        # The training steps committed, which each decision's context carries.
        self.step = step
        # The groups dispatched and not yet dequeued by a step (running, or settled and waiting in the result queue),
        # and those the current step has dequeued: together, the horizon's count.
        self.pipeline = 0
        self.admitted = 0
        # The most the horizon's count reached, which the tests check.
        self.peak = 0
        self._steps_left = steps
        self._active: Counter[str | None] = Counter()
        # The draws that wait, by label, each in the order it was drawn: those a resume reissued, and those whose label
        # was at its limit (the label's backlog).
        self._waiting: dict[str | None, deque[BoundDraw]] = {}
        # The settled groups no committed step consumed yet, by key, in the order they settled; a checkpoint keeps them.
        self._settled: dict[Key, Any] = {}
        # Every draw no committed step consumed, by key, in the order it was drawn: its task, its position, its row's
        # index in the task pool and the sampler that drew it.
        self._outstanding: dict[Key, tuple[str, int | None, int | None, str | None]] = {}
        # Draws of each label skipped because its backlog was full.
        self.skipped: Counter[str | None] = Counter()
        # Each adaptive entry's tokens, oldest first, and the token being started (waiting for its permits), which is
        # in no queue; and the slots skipped, by a full queue or a resume.
        self._tokens: dict[str, deque[Token]] = {}
        self._starting: Token | None = None
        self.skipped_slots = 0
        self._next_seq = 0
        self._next_ticket = 0
        self._changed = asyncio.Event()
        self.results: SettledSink | None = None
        # Under a group admission rule: the step's cap, what it counted toward it, and the markers of the groups
        # rejected while it was current that wait in the result queue and freed their slots (a marker that keeps its
        # slot counts in the pipeline instead).
        self.groups_per_step = groups_per_step
        self.counted = 0
        self._markers_now = 0
        # Under permit_release: rollout each group's lease returns a rollout's permit as the rollout ends; with rolling
        # (set for a workflow that runs its rollouts on the lease) a group is granted its permits one at a time.
        self.permit_release = permit_release
        self.rolling = False
        # The seconds it was held back since the metrics last read them, by reason (BLOCKED_REASONS), and the wait in
        # progress: its reason and when it began.
        self._held_back: defaultdict[str, float] = defaultdict(float)
        self._held_back_reason: str | None = None
        self._held_back_since = 0.0
        self._shares_since = time.monotonic()
        # Each trained policy's horizon, under independent stepping (None: lockstep), which wakes the dispatcher.
        self.horizons = horizons
        if horizons is not None:
            horizons.changed = self._changed.set

    # --- what the train loop tells it -------------------------------------------------------------------------------

    def dequeued(self) -> None:
        """A step dequeued a group: it leaves the pipeline and counts toward the step."""

        self.pipeline = max(0, self.pipeline - 1)
        self.admitted += 1
        self._changed.set()

    def took(self, kind: str, window: int | None = None, *, held: bool = False) -> bool:
        """Under a group admission rule, a step dequeued an item (instead of :meth:`dequeued`): an ``admitted`` group
        counts toward its batch, a ``stale`` group or a kept ``failed`` one toward its cap only, and a ``marker`` of a
        group rejected while step ``window`` was current toward its cap when that is this step.  A ``held`` marker
        (one whose group kept its slot, :meth:`reject`) frees its slot now, whichever step's it is.  Whether the item
        counts toward the cap."""

        self._changed.set()
        if kind == "marker":
            if held:
                self.pipeline = max(0, self.pipeline - 1)
            if window != self.step:
                return False  # an earlier step's: its slot was freed when it settled, or just now
            if not held:
                self._markers_now = max(0, self._markers_now - 1)
        else:
            self.pipeline = max(0, self.pipeline - 1)
            self.admitted += kind == "admitted"
        self.counted += 1
        return True

    def committed(self, keys: Iterable[Key | None]) -> None:
        """The step that dequeued the groups of ``keys`` committed: they were consumed, so no checkpoint keeps them."""

        for key in keys:
            if key is not None:
                self._settled.pop(key, None)
                self._outstanding.pop(key, None)

    def policy_step_started(self, policy: str) -> None:
        """Under independent stepping, the next step of ``policy`` started: its shares the last one took leave its
        horizon, and the leader's counts one step of the plan, and of the steps the samplers' contexts carry."""

        assert self.horizons is not None  # a run that steps independently built them
        self.horizons.step_started(policy)
        if policy == self.horizons.leader:
            self.step += 1

    def step_started(self) -> None:
        """A new step starts, its policy published: the groups the last one dequeued no longer count."""

        self.admitted = 0
        self.counted = self._markers_now = 0  # the markers still queued are an earlier step's now
        self._steps_left = max(0, self._steps_left - 1)
        self.step += 1
        self._changed.set()

    # --- what a group tells it --------------------------------------------------------------------------------------

    def settle(self, draw: BoundDraw, item: Any) -> None:
        """The group of ``draw`` settled: ``item`` (what a step dequeues) goes to the result queue, and is kept until a
        committed step consumes it."""

        self._settled[draw.key] = item
        if self.horizons is not None:  # it no longer runs; the share ledger counts its shares on their horizons
            self.pipeline -= 1
        if self.results is not None:
            self.results.put_nowait(item)
        self._changed.set()

    def reject(self, draw: BoundDraw, item: Any, *, hold: bool = False) -> None:
        """Under a group admission rule, the group of ``draw`` was rejected, or failed, when it settled: its fate is
        final, so it leaves the outstanding draws (no checkpoint keeps or saves it), and ``item``, its marker, goes to
        the result queue.  It frees its slot of the horizon at once, the marker counted among the current step's queued
        markers; with ``hold`` (a group that failed or settled without a datum, which can end at once) it keeps its slot
        until a step dequeues the marker (:meth:`took`), as a group does in all mode."""

        self._outstanding.pop(draw.key, None)
        if not hold:
            self.pipeline = max(0, self.pipeline - 1)
            self._markers_now += 1
        if self.results is not None:
            self.results.put_nowait(item)
        self._changed.set()

    # --- the loop ---------------------------------------------------------------------------------------------------

    def bind(self, results: SettledSink) -> None:
        """Hand the dispatcher the queue its groups settle into, the groups a resume kept first, in the order they
        settled."""

        self.results = results
        for item in list(self._settled.values()):  # the share ledger retires a kept group that is wholly durable
            results.put_nowait(item)

    async def run(
        self,
        *,
        spawn: Callable[[Coroutine[Any, Any, None]], Any],
        run_group: GroupRunner,
        stop: asyncio.Event | None = None,
    ) -> None:
        """Start groups for as long as the run needs them: ``spawn`` runs a coroutine as a task (the train loop's task
        group), ``run_group`` runs one group, and ``stop`` asks to stop starting groups, unless a step that applied an
        update must still finish (``finish_batches``)."""

        self._held_back.clear()
        self._shares_since = time.monotonic()  # the first step's shares count from here
        while not self._stopping(stop):
            item = self._choose()
            if item is None:
                self._changed.clear()
                await self._held_back_while(self._held_back_by(), self._changed.wait())
                continue
            if isinstance(item, BoundDraw):
                await self._start(item, spawn, run_group, stop)
            elif not await self._decide(item, spawn, run_group, stop):
                self._changed.clear()  # nothing was eligible: wait for a change
                await self._held_back_while("sampler", self._changed.wait())

    def _stopping(self, stop: asyncio.Event | None) -> bool:
        return stop is not None and stop.is_set() and not self.finish_batches

    async def _held_back_while(self, reason: str, waiting: Awaitable[Any]) -> None:
        """Await ``waiting``, the time counting as held back by ``reason`` (:meth:`blocked_shares`)."""

        self._hold_back(reason)
        try:
            await waiting
        finally:
            self._hold_back(None)

    def _hold_back(self, reason: str | None) -> None:
        """Count the time since what holds it back last changed, and note what does now (None: nothing)."""

        now = time.monotonic()
        if self._held_back_reason is not None:
            self._held_back[self._held_back_reason] += now - self._held_back_since
        self._held_back_reason, self._held_back_since = reason, now

    def _held_back_by(self) -> str:
        """What holds it back when :meth:`_choose` found nothing to start: the end of the plan (under a group admission
        rule, the last step's cap too), the horizon (at ``k = 0`` under a rule, the step's cap too), else the labels'
        limits."""

        if self.horizons is not None:  # independent stepping: the plan, else the first horizon without room
            if not self.horizons.plan_has_room(self.pipeline):
                return "plan"
            binding = self.horizons.binding(self.pipeline, self._capped_running())
            return "labels" if binding is None else f"horizon:{binding}"
        if self.pipeline + self.admitted >= self._steps_left * self.batch_size:
            return "plan"
        if self._horizon_has_room():
            return "labels"
        return "plan" if self.groups_per_step is not None and self._steps_left == 1 else "horizon"

    def blocked_shares(self) -> dict[str, float]:
        """The step's ``dispatch/blocked_share/<reason>`` metrics: the share of the time since the last call (the run's
        last step) in which it was held back from starting a group, by reason (:data:`BLOCKED_REASONS`).  A wait in
        progress counts up to now, and on in the next window."""

        now = time.monotonic()
        self._hold_back(self._held_back_reason)
        window = max(1e-9, now - self._shares_since)
        shares = {f"dispatch/blocked_share/{why}": min(1.0, self._held_back[why] / window) for why in BLOCKED_REASONS}
        if self.horizons is not None:  # each horizon's share, and theirs together as the horizon's
            held = {policy: self._held_back[f"horizon:{policy}"] for policy in self.horizons.policies}
            shares["dispatch/blocked_share/horizon"] = min(1.0, sum(held.values()) / window)
            shares.update({f"dispatch/blocked_share/horizon:{p}": min(1.0, s / window) for p, s in held.items()})
        self._held_back.clear()
        self._shares_since = now
        return shares

    def _horizon_has_room(self, label: str | None = None, *, waiting: bool = False) -> bool:
        """Whether another group may be dispatched.  It may when all of these hold:

        1. The run's remaining steps consume more groups than those in the pipeline and those the current step
           dequeued.
        2. Under a group admission rule, at ``k = 0`` or on the last step: the current step's cap is more than what it
           counted, the groups in the pipeline and its queued markers.
        3. The horizon, ``(k + 1) × batch_size``, is more than those groups and the slots held for the capped labels'
           next groups (:meth:`_held`).  For one of ``label``'s ``waiting`` draws, the slots held for ``label`` do not
           count: they are held for that draw.  Without a bound there is no horizon."""

        if self.horizons is not None:
            return self._horizons_have_room(label, waiting=waiting)
        in_use = self.pipeline + self.admitted
        if in_use >= self._steps_left * self.batch_size:
            return False
        if self.groups_per_step is not None and (self.max_staleness == 0 or self._steps_left == 1):
            # Under a rule, at k = 0 and on the last step: no group the current step could not count.
            if self.counted + self.pipeline + self._markers_now >= self.groups_per_step:
                return False
        if self.max_staleness is None:
            return True
        held = self._held()
        if waiting and held and label in self._limits and label in self._waiting_labels():
            held -= min(self._active[label], held)
        return in_use + held < (self.max_staleness + 1) * self.batch_size

    def _horizons_have_room(self, label: str | None, *, waiting: bool) -> bool:
        """Under independent stepping, whether another group may be dispatched: the leader's remaining steps take
        more shares than the groups running and its own queued and taken may give it, and every policy's horizon has
        room (:meth:`~platoon.train.dispatch.horizons.Horizons.binding`), the slots held for the capped labels counted
        on each as :meth:`_held` counts them on the one horizon of lockstep."""

        assert self.horizons is not None  # the caller checked
        if not self.horizons.plan_has_room(self.pipeline):
            return False
        own = self._active[label] if waiting and label in self._limits and label in self._waiting_labels() else 0
        return self.horizons.binding(self.pipeline, self._capped_running(), own) is None

    def _capped_running(self) -> int:
        """The running groups of the capped labels that have draws waiting, each holding a slot for its label's next
        draw (:meth:`_held`); none without limits."""

        if not self._limits:
            return 0
        return sum(self._active[label] for label in self._waiting_labels() if label in self._limits)

    def _held(self) -> int:
        """The slots of the horizon held for the groups that follow the capped labels' running ones: one for each
        running group of a capped label that has draws waiting (or tokens, of an entry whose tasks all carry that
        label), as far as those groups and the slots held for them fit in ``k`` batches (none with ``k = 0``).  The
        label's oldest waiting draw or token takes the slot when its label has room."""

        if not self.max_staleness or not self._limits:
            return 0
        running = sum(self._active[label] for label in self._waiting_labels() if label in self._limits)
        return min(running, max(0, self.max_staleness * self.batch_size - running))

    def _waiting_labels(self) -> list[str | None]:
        """The labels with draws waiting, or tokens of an entry of that one label."""

        labels = [label for label, waiting in self._waiting.items() if waiting]
        for entry, tokens in self._tokens.items():
            label = self._token_label(entry)
            if tokens and label is not _NO_LABEL and label not in labels:
                labels.append(label)  # type: ignore[arg-type]
        return labels

    def _token_label(self, entry: str) -> str | None | object:
        """The one label every task of ``entry`` carries, which its tokens count as waiting draws of (the default
        label column's: the entry's own); :data:`_NO_LABEL` for an entry whose tasks carry several."""

        labels = self.samplers.entry_labels(entry)
        return labels[0] if len(labels) == 1 else _NO_LABEL

    def _entry_can_start(self, entry: str) -> bool:
        """Whether a group of ``entry`` could start: some label its tasks carry has room."""

        return any(self._has_room(label) for label in self.samplers.entry_labels(entry))

    def _has_room(self, label: str | None) -> bool:
        return label not in self._limits or self._active[label] < self._limits[label]

    def _some_label_has_room(self) -> bool:
        """Whether a group of some label of the run could start: one without a limit, or below it.  With no limits,
        or no label known, nothing is known to be full."""

        if not self._limits or not self._labels:
            return True
        return any(self._has_room(label) for label in self._labels)

    def _choose(self) -> BoundDraw | Token | _Ask | None:
        """The next item to start, with room in the horizon for it; None when nothing can start now.  In this order:

        1. The oldest of the waiting draws whose label has room and the tokens whose entry can start.
        2. Else, under an adaptive run sampler, a decision over the whole pool (:data:`ASK`), while some label has
           room.
        3. Else a new draw of the order, while some label has room.  A new draw whose label is at its limit waits, or
           is skipped past its backlog.  A slot owed to an adaptive entry becomes a token.  The token starts at once
           when its entry can start and none of its tokens waits; else it waits in its entry's queue, or is skipped
           past the backlog limit."""

        best: deque[Any] | None = None
        for label, waiting in self._waiting.items():
            if (
                waiting
                and self._has_room(label)
                and self._horizon_has_room(label, waiting=True)
                and (best is None or waiting[0].ticket < best[0].ticket)
            ):
                best = waiting
        for entry, tokens in self._tokens.items():
            if (
                tokens
                and (best is None or tokens[0].ticket < best[0].ticket)
                and self._entry_can_start(entry)
                and self._token_horizon(entry)
            ):
                best = tokens
        if best is not None:
            return best.popleft()
        if not self._horizon_has_room():
            return None
        if self.samplers.adaptive():
            return ASK if self._some_label_has_room() else None
        while self._some_label_has_room():
            item = self._new_draw()
            if isinstance(item, Token):
                tokens = self._tokens.setdefault(item.entry, deque())
                if not tokens and self._entry_can_start(item.entry):
                    return item
                if len(tokens) < self.backlog_limit:
                    tokens.append(item)  # waits until its entry can start
                    if not self._horizon_has_room():  # its label's running groups now hold a slot for it
                        return None
                    continue
                self._skip_slot(item.position, item.entry, "backlog")
                continue
            draw = item
            if self._has_room(draw.label):
                return draw
            waiting = self._waiting.setdefault(draw.label, deque())
            if len(waiting) < self.backlog_limit:
                waiting.append(draw)  # waits for its label's limit
                if not self._horizon_has_room():  # its label's running groups now hold a slot for it
                    return None
                continue
            self._skip(draw)
        return None

    def _token_horizon(self, entry: str) -> bool:
        """Whether the horizon has room for a token of ``entry``: as for a waiting draw of its label, when all its
        tasks carry one."""

        label = self._token_label(entry)
        if label is _NO_LABEL:
            return self._horizon_has_room()
        return self._horizon_has_room(label, waiting=True)  # type: ignore[arg-type]

    async def _start(
        self,
        draw: BoundDraw,
        spawn: Callable[[Coroutine[Any, Any, None]], Any],
        run_group: GroupRunner,
        stop: asyncio.Event | None = None,
    ) -> None:
        """Take the draw's label slot, reserve its group's permits (first come first served) and start its group once
        it can start (:meth:`_reserve`); a stop request that came while it waited for permits puts the draw back, first
        in line."""

        if draw.label in self._limits:
            self._active[draw.label] += 1
        self.pipeline += 1
        self.peak = max(self.peak, self.pipeline + self.admitted)
        lease = self._reserve(draw.group_size)
        try:
            await self._held_back_while("permits", lease.ready())
        except BaseException:
            lease.withdraw()
            self._unstart(draw)
            raise
        if self._stopping(stop):
            lease.withdraw()
            self._unstart(draw)
            self._waiting.setdefault(draw.label, deque()).appendleft(draw)
            return
        spawn(self._group(draw, run_group, lease))

    def _reserve(self, size: int) -> DrawLease:
        """Reserve a group's permits: whole, or under ``permit_release: rollout`` one at a time for a ``rolling``
        workflow, whose group can then start on the first; under ``rollout`` its lease returns each rollout's permit
        as the rollout ends."""

        leases = self.permit_release == "rollout"
        return self._rows.permits.reserve(size, rolling=leases and self.rolling, active=leases)

    async def _decide(
        self,
        item: Token | _Ask,
        spawn: Callable[[Coroutine[Any, Any, None]], Any],
        run_group: GroupRunner,
        stop: asyncio.Event | None = None,
    ) -> bool:
        """Start the group of a token, or of a decision of the run's adaptive sampler.  False when nothing was decided.

        It reserves the horizon's slot and the group's rollout permits: the token's entry's group size, or the run's
        largest for a decision over the pool.  Once the group can start (:meth:`_reserve`), it does all of this at
        once:

        1. ask the sampler;
        2. bind its draw (a decision over the pool takes the next position);
        3. resize the reservation to the drawn task's group size;
        4. take its label's slot and start the group.

        While it waits for its permits, the token being started is saved first among its entry's.  A stop request or a
        cancellation that comes meanwhile puts it back, first in its queue; so does a decision that finds nothing
        eligible."""

        token = item if isinstance(item, Token) else None
        entry = token.entry if token is not None else None
        size = (self.samplers.group_size(entry) if token is not None else self._ask_size()) or 1
        self._starting = token
        self.pipeline += 1
        self.peak = max(self.peak, self.pipeline + self.admitted)
        lease = self._reserve(size)
        try:
            await self._held_back_while("permits", lease.ready())
        except BaseException:
            lease.withdraw()
            self.pipeline -= 1
            self._put_back(token)
            raise
        if self._stopping(stop):
            lease.withdraw()
            self.pipeline -= 1
            self._put_back(token)
            return True
        decided = self.samplers.decide(
            entry,
            room=self._has_room,
            step=self.step,
            versions=self._versions(),
            fallback=token is not None and token.fallback,
            most=None if token is not None else size,
        )
        if decided is None:
            lease.withdraw()
            self.pipeline -= 1
            self._put_back(token)
            return False
        self._starting = None
        if token is not None:
            position = token.position
        else:
            position = self.next_position
            self.next_position += 1
        draw = self._draw(decided.row, position, index=decided.index, sampler=decided.sampler)
        if size > draw.group_size:
            lease.resize(draw.group_size)
        if draw.label in self._limits:
            self._active[draw.label] += 1
        self._record(draw, reason=decided.reason, score=decided.score)
        if size < draw.group_size:  # a group larger than the set said: its other permits, after the decision
            lease.resize(draw.group_size)
            try:
                await self._held_back_while("permits", lease.ready())
            except BaseException:
                lease.withdraw()
                self._unstart(draw)
                self._waiting.setdefault(draw.label, deque()).appendleft(draw)
                raise
        spawn(self._group(draw, run_group, lease))
        return True

    def _ask_size(self) -> int | None:
        """The rollout permits a decision over the whole pool holds: the largest group of the entries that can start
        a group now (some label of their tasks has room), so that permits a capped entry's group could not use stay
        free for the others; its draw is one of those entries' (:meth:`SamplerSet.decide`'s ``most``)."""

        sizes = [info.group_size for label, info in self.samplers.infos.items() if self._entry_can_start(label)]
        return max(sizes) if sizes else self.samplers.group_size(None)

    def _put_back(self, token: Token | None) -> None:
        """A token whose decision was not made goes back, first in its entry's queue."""

        self._starting = None
        if token is not None:
            self._tokens.setdefault(token.entry, deque()).appendleft(token)

    def _unstart(self, draw: BoundDraw) -> None:
        if draw.label in self._limits:
            self._active[draw.label] -= 1
        self.pipeline -= 1

    async def _group(self, draw: BoundDraw, run_group: GroupRunner, lease: DrawLease) -> None:
        """Run the group of ``draw`` on its lease, whose release (calling it) frees the group's label slot too, and
        which is called again here, so that a runner that raises or is cancelled still frees what it holds."""

        def free_label_slot() -> None:
            if draw.label in self._limits:
                self._active[draw.label] -= 1
            self._changed.set()

        lease.on_release, lease.name = free_label_slot, draw.task_id
        try:
            await run_group(draw, lease)
        finally:
            lease()

    # --- draws ------------------------------------------------------------------------------------------------------

    def _new_draw(self) -> BoundDraw | Token:
        """Draw the order's next position: the run's samplers decide its task (:meth:`SamplerSet.static`), or owe it
        to an entry whose sampler is adaptive (a token)."""

        position = self.next_position
        self.next_position += 1
        decided = self.samplers.static(position, step=self.step, versions=self._versions())
        if isinstance(decided, Owed):
            ticket = self._next_ticket
            self._next_ticket += 1
            return Token(position, decided.entry, ticket, decided.fallback)
        draw = self._draw(decided.row, position, index=decided.index, sampler=decided.sampler)
        self._record(draw, reason=decided.reason, score=decided.score)
        return draw

    def _draw(
        self, row: dict[str, Any], position: int | None, *, index: int | None = None, sampler: str | None = None
    ) -> BoundDraw:
        """Number ``row`` (pool row ``index``, which ``sampler`` drew) as this start's next draw, at ``position`` in
        the order: it is outstanding until a committed step consumes it, or it is skipped."""

        seq = self._next_seq
        self._next_seq += 1
        row[SEQ_COLUMN] = seq
        row[POSITION_COLUMN] = position
        entry, label, group_size = self._rows.of(row)
        self._labels.add(label)
        ticket = self._next_ticket
        self._next_ticket += 1
        draw = BoundDraw(
            row, self.start, seq, position, str(row.get("task_id")), entry, label, group_size, ticket, index, sampler
        )
        self._outstanding[draw.key] = (draw.task_id, position, index, sampler)
        return draw

    def _record(
        self,
        draw: BoundDraw,
        *,
        reason: str = "order",
        score: float | None = None,
        reissues: Key | None = None,
        version: int | None = None,
    ) -> None:
        reason = "reissue" if reissues is not None else reason
        self.samplers.count(draw.sampler, reason)
        if self._recorder is not None:
            self._recorder.draw(
                draw.row,
                seq=draw.seq,
                position=draw.position,
                version=self._versions()[self._leader] if version is None else version,
                reissues=reissues,
                sampler=draw.sampler,
                reason=reason,
                score=score,
            )

    def _skip(self, draw: BoundDraw) -> None:
        self._outstanding.pop(draw.key, None)
        self.skipped[draw.label] += 1
        if self._recorder is not None:
            self._recorder.skipped(draw.row)
        if self.skipped[draw.label] == 1:
            logger.warning(
                "Label %r has %d rows waiting for its limit (train.rollout_group_limits: %d at once): its new rows are "
                "skipped, so that the other labels' rows keep the batches full; its share of the order asks for more "
                "groups than its limit runs",
                draw.label,
                self.backlog_limit,
                self._limits.get(draw.label, 0),
            )

    def _skip_slot(self, position: int, entry: str, why: str) -> None:
        """A token's position is used up without a draw: its entry's queue was full (``backlog``), or a resume found
        that its entry no longer gets slots (``no_slots``) or left the run (``entry_left``)."""

        self.skipped_slots += 1
        slot_skipped = getattr(self._recorder, "slot_skipped", None)
        if callable(slot_skipped):
            slot_skipped(position=position, entry=entry, why=why)
        if why == "backlog" and self.skipped_slots == 1:
            logger.warning(
                "Environment %r has %d slots waiting for its task sampler, whose labels are at their limit "
                "(train.rollout_group_limits): its new slots are skipped, so that the other labels' rows keep the "
                "batches full; its share of the order asks for more groups than its limit runs",
                entry,
                self.backlog_limit,
            )

    # --- what the run's metrics read ----------------------------------------------------------------------------------

    def parked_by_label(self) -> Counter[str | None]:
        """The draws that wait for their label's limit, by label, and an adaptive entry's tokens under its label."""

        parked = Counter(
            {label: len(waiting) for label, waiting in self._waiting.items() if waiting and label in self._limits}
        )
        for entry, tokens in self._tokens.items():
            label = self._token_label(entry)
            if tokens and label is not _NO_LABEL and label in self._limits:
                parked[label] += len(tokens)  # type: ignore[index]
        return parked

    def tokens_by_entry(self) -> dict[str, int]:
        """Each adaptive entry's tokens waiting now, the one being started included."""

        found = {entry: len(tokens) for entry, tokens in self._tokens.items()}
        if self._starting is not None:
            found[self._starting.entry] = found.get(self._starting.entry, 0) + 1
        return found

    # --- checkpoints and resume ---------------------------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """What a full checkpoint saves of the dispatcher, beside the groups it keeps (:meth:`kept`):

        - ``start``, ``sampler`` and ``fingerprint``: this start's number, its sampler and the fingerprint of its
          inputs;
        - ``next_position``: the next position of the order;
        - ``outstanding``: the outstanding draws, ``[start, seq, task id, position]`` each, in the order they were
          drawn;
        - ``drawn``: the row each of them drew and the sampler that drew it, ``[start, seq, pool index, sampler]``
          each, in the same order (the index is None for a row an order handed in);
        - ``waiting``: each label's waiting draws, ``[start, seq]`` each, oldest first;
        - ``tokens``: the positions of each adaptive entry's tokens, oldest first, the one waiting for its permits
          first;
        - ``fallback_slots``: the positions of the tokens whose slot the run sampler's fallback handed (their draws
          are the run's fallback);
        - ``samplers``: the samplers' counts and states.

        A resume draws again the outstanding draws that neither wait nor are kept, from the rows they drew."""

        outstanding = self._outstanding.items()
        return {
            "start": self.start,
            "sampler": self.sampler,
            "fingerprint": self.fingerprint,
            "next_position": self.next_position,
            "outstanding": [[start, seq, task, position] for (start, seq), (task, position, _, _) in outstanding],
            "drawn": [[start, seq, index, by] for (start, seq), (_, _, index, by) in outstanding],
            "waiting": {
                "" if label is None else label: [list(draw.key) for draw in waiting]
                for label, waiting in self._waiting.items()
                if waiting
            },
            "tokens": self._token_positions(),
            "fallback_slots": [token.position for token in self._token_list() if token.fallback],
            "samplers": self.samplers.state(),
        }

    def _token_list(self) -> list[Token]:
        """Every token, the one being started first."""

        found = [self._starting] if self._starting is not None else []
        return found + [token for tokens in self._tokens.values() for token in tokens]

    def _token_positions(self) -> dict[str, list[int]]:
        """Each entry's tokens by position, oldest first, the token being started first among its entry's."""

        found: dict[str, list[int]] = {}
        if self._starting is not None:
            found[self._starting.entry] = [self._starting.position]
        for entry, tokens in self._tokens.items():
            if tokens:
                found.setdefault(entry, []).extend(token.position for token in tokens)
        return found

    def kept(self) -> list[tuple[Key, Any]]:
        """The groups that settled and no committed step consumed, ``(key, item)`` each, in the order they settled."""

        return list(self._settled.items())

    def holding(self, key: Key) -> Any:
        """The group of draw ``key`` that settled and no committed step consumed yet, as :meth:`kept` lists it; None
        when it holds none (a group a rule rejected as it settled is not held)."""

        return self._settled.get(key)

    def restore(
        self,
        *,
        kept: Sequence[tuple[Key, str, int | None, Any]],
        reissue: Sequence[tuple[Key, Mapping[str, Any], int | None]],
        waiting: Sequence[tuple[Key, Mapping[str, Any], int | None]],
        version: int,
        drawn: Mapping[Key, tuple[int | None, str | None]] | None = None,
    ) -> None:
        """Take up what a checkpoint saved (:func:`~platoon.train.checkpoints.dispatch_state.plan_resume`): ``kept``
        groups, ``(key, task id, position, item)`` each, which the steps dequeue first and which stay outstanding until
        a committed step consumes them; the draws to draw again, ``(old key, row, position)`` each, oldest first,
        before any new draw (``reissue``); and the draws that waited for their label, in their labels' order
        (``waiting``).  Each draw drawn again is a new draw of this start, at its old position, at policy ``version``,
        and replaces the old one among the outstanding draws.
        ``drawn`` gives, by old key, the pool index of the row each one is drawn from (a kept group's: of the row it
        drew) and the sampler that drew it; the dispatcher's own sampler name stands in for one it does not give."""

        drawn = drawn or {}
        for key, task_id, position, item in kept:
            index, by = drawn.get(key, (None, None))
            self._settled[key] = item
            self._outstanding[key] = (task_id, position, index, by)
        if self.horizons is None:  # independent stepping: the share ledger counts the kept groups' shares
            self.pipeline += len(kept)
        for old, row, position in [*reissue, *waiting]:
            index, by = drawn.get(old, (None, None))
            draw = self._draw(dict(row), position, index=index, sampler=by or self.sampler)
            self._record(draw, reissues=old, version=version)
            self._waiting.setdefault(draw.label, deque()).append(draw)

    def restore_tokens(
        self, tokens: Mapping[str, Sequence[int]], *, version: int, fallbacks: Container[int] = ()
    ) -> dict[str, int]:
        """Take up the tokens a checkpoint saved (:meth:`state`'s ``tokens``: each entry's positions, the one that
        waited for its permits first), after the outstanding draws.  Returns how many tokens were restored, bound and
        dropped.

        - An entry whose sampler is adaptive gets its tokens back in its queue, in order.
        - An entry whose sampler is now static binds them by it, in order, as waiting draws of their labels.
        - The tokens of an entry that no longer gets slots, or left the run, are dropped, each with a ``slot_skipped``
          event (``no_slots``, ``entry_left``).

        A token whose position is in ``fallbacks`` (``fallback_slots``) had its slot from the run sampler's fallback,
        and its draw is the run's fallback."""

        found = {"restored": 0, "bound": 0, "dropped": 0}
        for entry, positions in tokens.items():
            member = self.samplers.entries.get(entry)
            if member is None:
                why = "no_slots" if entry in self.samplers.infos else "entry_left"
                for position in positions:
                    self._skip_slot(int(position), entry, why)
                found["dropped"] += len(positions)
            elif member.adaptive:
                queue = self._tokens.setdefault(entry, deque())
                for position in positions:
                    queue.append(Token(int(position), entry, self._next_ticket, int(position) in fallbacks))
                    self._next_ticket += 1
                found["restored"] += len(positions)
            else:
                # Its count stands at the slots its entry received, these included: they take the decisions before.
                member.decisions = max(0, member.decisions - len(positions))
                for position in positions:
                    decided = self.samplers.entry_static(
                        entry, step=self.step, versions={MAIN: version}, fallback=int(position) in fallbacks
                    )
                    draw = self._draw(decided.row, int(position), index=decided.index, sampler=decided.sampler)
                    self._record(draw, reason=decided.reason, score=decided.score, version=version)
                    self._waiting.setdefault(draw.label, deque()).append(draw)
                found["bound"] += len(positions)
        return found
