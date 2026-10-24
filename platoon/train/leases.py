"""Rollout leases (RFC 0020): how a training workflow runs each of its rollouts on one of its draw's rollout permits.

:class:`RolloutLease` states the contract, and ``docs/extending/rollout-leases.md`` is the guide.  The trainer sets the
lease of each training draw in a context variable around the draw's ``arun_episode(row)`` (:func:`holding`), and
:func:`rollout_lease` reads it.  This module imports nothing of ``tinker``, so that ``platoon.experimental`` can export
it; the trainer's leases live in :mod:`platoon.train.tinker.permits`.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import AbstractAsyncContextManager, contextmanager
from contextvars import ContextVar
from typing import Protocol

__all__ = ["NO_BLOCK", "NO_LEASE", "RolloutLease", "holding", "rollout_lease"]


class RolloutLease(Protocol):
    """The rollout permits of the training draw whose workflow runs in this asyncio context (:func:`rollout_lease`).

    A draw reserves ``size`` rollout permits of the run's ``train.max_concurrent_rollouts``: its entry's group size.
    ``async with lease.rollout(i):`` runs rollout ``i`` on one of them: the block waits until the draw holds a permit
    that no other block uses (at once when the run grants the draw's permits together), runs on it, and returns it to
    the run as it exits, so that another draw can start.  Its value is the seconds it waited for that permit: 0.0 when
    it entered at once.  Under ``train.permit_release: group`` a draw's blocks hold nothing and its permits come back
    together when its workflow returns; outside a training draw the lease has size 0 and its blocks hold nothing,
    whatever their index.

    The rules a workflow keeps:

    1. **One block per rollout, around the whole rollout.**  The permit stands for the rollout's worker and sandboxes:
       exit the block only once they are gone, and a cancelled rollout only once its cancellation has been awaited.  A
       rollout's retry stays inside its block.
    2. **Each index once, at most ``size`` of them.**  In a training draw, ``rollout(i)`` raises ``IndexError`` for an
       index outside ``0 .. size - 1``, and a second block for an index raises ``RuntimeError`` as it enters.
    3. **The flag.**  A workflow class whose every rollout runs in a block says so with
       ``runs_rollouts_on_lease = True``, and its draws are then granted their permits one at a time as they free
       (rolling starts); any other draw is granted whole before ``arun_episode`` starts, and each of its blocks still
       returns its permit.
    4. **The rest comes back with the draw.**  Whatever the draw still holds comes back when ``arun_episode``
       returns, raises or is cancelled; a block that enters after that, or still waits for its permit then, holds
       nothing, and the trainer warns.
    5. **Read it in the task.**  The lease lives in a context variable, which asyncio tasks copy when they are
       created and ``loop.run_in_executor`` does not: read ``rollout_lease()`` in the workflow's task, before handing
       work to a thread.

    Asking for permits beyond ``size``, for a draw whose size varies or for an episode's child agents, is reserved for
    a later release (RFC 0020).
    """

    @property
    def size(self) -> int:
        """The rollout permits the draw reserved: its entry's group size; 0 outside a training draw."""
        ...

    def rollout(self, index: int) -> AbstractAsyncContextManager[float]:
        """A block that runs rollout ``index`` of the draw on one of its permits (see the class docstring)."""
        ...


class _NoBlock:
    """A block that holds no permit: it enters at once, its value is 0.0, and it returns nothing when it exits."""

    async def __aenter__(self) -> float:
        return 0.0

    async def __aexit__(self, *exc: object) -> bool:
        return False


# The block of every rollout of the no-op lease: it has no state, so one serves every block.
NO_BLOCK = _NoBlock()


class _NoLease:
    """The lease outside a training draw: size 0, and blocks that hold nothing."""

    size = 0

    def rollout(self, index: int) -> AbstractAsyncContextManager[float]:
        return NO_BLOCK


NO_LEASE: RolloutLease = _NoLease()

# The lease of the training draw that runs in this context (holding); tasks the workflow creates copy it.
_current: ContextVar[RolloutLease | None] = ContextVar("platoon_rollout_lease", default=None)


def rollout_lease() -> RolloutLease:
    """The lease of the training draw whose workflow runs in this asyncio context (:class:`RolloutLease`).

    Outside a training draw (evaluation, a test, a direct call) a lease of size 0 whose blocks enter at once and hold
    nothing, so a workflow runs its rollouts the same way everywhere.  Read it in the workflow's own task: a thread of
    ``loop.run_in_executor`` does not see it unless the context is copied."""

    lease = _current.get()
    return NO_LEASE if lease is None else lease


@contextmanager
def holding(lease: RolloutLease | None) -> Generator[None, None, None]:
    """Make ``lease`` the lease of this context while the ``with`` block runs (None: the no-op lease)."""

    token = _current.set(lease)
    try:
        yield
    finally:
        _current.reset(token)
