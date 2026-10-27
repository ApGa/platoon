"""The file of the dispatcher's state that a full checkpoint saves, and what a resume takes up from it.

**Exact resume.**  Every draw is numbered ``(start, seq)`` and stamped on its row (``_platoon_seq``,
``_platoon_position``), and the dispatcher keeps every draw no committed step consumed (outstanding).  What a full
checkpoint saves of it (:func:`save_state`) is a file of its own, ``tasks/dispatch-s<start>-<checkpoint>.json``, which
the checkpoint's row names (``dispatch``), so that it does not depend on the run's per-task statistics: the next
position of the order, the outstanding draws, each label's waiting draws and a fingerprint of the sampling inputs
(:meth:`Dispatcher.state`), with each outstanding draw's row in the run's task pool and the sampler that drew it,
each entry's tokens, and each sampler's decision count and state, and, in ``tasks/kept-s<start>-<checkpoint>.json.gz``
(:mod:`platoon.train.checkpoints.kept_groups`), the groups that settled and no committed step consumed
(:meth:`Dispatcher.kept`).  A start that resumes reads it back
(:func:`read_state`) and rebuilds the dispatcher from it (:func:`plan_resume`, :meth:`Dispatcher.restore`): the kept
groups go to the result queue first, so the steps after the resume train them without running them again; every other
outstanding draw (one that was running, waiting for its label, or whose group could not be kept) is drawn again first,
as a new draw of this start (reason ``reissue``) at its old place in the order, from the row it drew; and the order
goes on from where the checkpoint left it, every sampler at its saved count (a state of platoon 0.3.0's format 1,
which saved none, has the counts recomputed from the position), each entry's tokens back in its queue
(:meth:`Dispatcher.restore_tokens`).  So every position of the order is consumed exactly once: trained, dropped as
stale, failed, skipped or rejected, or, for a token, skipped by a full queue or dropped by a resume whose entry no
longer gets slots (a ``slot_skipped`` event each).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from platoon.train.dispatch.dispatcher import Key
from platoon.train.task_stats import read_tasks_file, write_tasks_file

logger = logging.getLogger(__name__)


# The format of the file of the dispatcher's state that a full checkpoint saves, which its row names: 2 adds the
# samplers' counts and states, each entry's tokens and each outstanding draw's pool row and sampler to platoon 0.3.0's
# format 1, which a resume still reads.
STATE_FORMAT = 2
STATE_FORMATS = (1, 2)


# --- what a full checkpoint saves of it, and what a resume takes up --------------------------------------------------


def state_name(start: int, checkpoint: str) -> str:
    """The file of the dispatcher's state that a start's checkpoint saves: ``dispatch-s<start>-<checkpoint>.json``."""

    return f"dispatch-s{start}-{checkpoint}.json"


def save_state(
    run_dir: str | os.PathLike[str],
    checkpoint: str,
    state: Mapping[str, Any],
    kept: Sequence[tuple[Key, Any]],
    shares: Mapping[Key, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Write what checkpoint ``checkpoint`` saves of the dispatcher: the groups it keeps (``kept``, as
    :meth:`Dispatcher.kept` gives them; those that can be kept, with what each policy did with its share when the
    policies step independently, ``shares``) to ``tasks/kept-s<start>-<checkpoint>.json.gz``, then
    its state (``state``, :meth:`Dispatcher.state`'s, which names that file with its SHA-256) to
    ``tasks/dispatch-s<start>-<checkpoint>.json``, and return the record the checkpoint's row keeps in ``loop_state``
    (``dispatch``): ``{file, sha256, format, start}``, and the kept groups' file (``kept``) when there is one.  Kept
    groups that cannot be written leave the state without them, with a warning, so that a resume draws their tasks
    again; a state that cannot be written leaves the checkpoint without one, with a warning, so that a resume from it
    draws from its batch.  Never raises."""

    from platoon.train.checkpoints.kept_groups import write_kept_groups

    try:
        start = int(state["start"])
        kept_record = write_kept_groups(run_dir, start, checkpoint, kept, shares)
        content = json.dumps(
            {"format": STATE_FORMAT, **state, "kept": kept_record}, separators=(",", ":"), allow_nan=False
        )
        record = {
            **write_tasks_file(run_dir, state_name(start, checkpoint), (content + "\n").encode()),
            "format": STATE_FORMAT,
            "start": start,
        }
    except OSError as exc:  # the disk, not the state
        logger.warning(
            "Could not write where the run's draws stand with checkpoint %s (%s): a resume from it draws from its "
            "batch, and rolls out again what had not trained",
            checkpoint,
            exc,
        )
        return None
    except Exception:  # what a resume takes up never stops training
        logger.warning("Could not save where the run's draws stand with checkpoint %s", checkpoint, exc_info=True)
        return None
    if kept_record is not None:
        record["kept"] = kept_record["file"]
    return record


def read_state(run_dir: str | os.PathLike[str], record: Mapping[str, Any]) -> dict[str, Any]:
    """The dispatcher's state that a checkpoint row's ``dispatch`` record names, checked against its SHA-256; a
    ``ValueError`` that says why when it is missing, changed or unreadable."""

    content = read_tasks_file(run_dir, record)
    try:
        data = json.loads(content)
    except ValueError as exc:
        raise ValueError(f"{record.get('file')} is not JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") not in STATE_FORMATS:
        raise ValueError(
            f"{record.get('file')} holds no dispatcher's state of format {' or '.join(map(str, STATE_FORMATS))}"
        )
    return data


def saved_draws(state: Mapping[str, Any]) -> dict[Key, tuple[int | None, str | None]]:
    """The pool index of the row each outstanding draw of a saved state drew and the sampler that drew it, by
    ``(start, seq)`` (``drawn``); none for a state of format 1, which saved neither.  A ``ValueError`` for data it
    cannot read."""

    try:
        drawn: dict[Key, tuple[int | None, str | None]] = {}
        for start, seq, index, by in state.get("drawn") or []:
            if index is not None and (isinstance(index, bool) or not isinstance(index, int) or index < 0):
                raise ValueError(f"pool index {index!r}")
            drawn[(int(start), int(seq))] = (index, None if by is None else str(by))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot read the outstanding draws' rows: {type(exc).__name__}: {exc}") from exc
    return drawn


def saved_fallback_slots(state: Mapping[str, Any]) -> frozenset[int]:
    """The positions of a saved state's tokens whose slot the run sampler's fallback handed (``fallback_slots``); none
    for a state that saved none.  A ``ValueError`` for data it cannot read."""

    positions = state.get("fallback_slots") or []
    if not isinstance(positions, list) or any(
        isinstance(position, bool) or not isinstance(position, int) or position < 0 for position in positions
    ):
        raise ValueError(f"cannot read the slots of the run sampler's fallback: {positions!r}")
    return frozenset(positions)


def saved_token_positions(state: Mapping[str, Any]) -> dict[str, list[int]]:
    """Each entry's tokens of a saved state, by position, the one that waited for its permits first (``tokens``); none
    for a state of format 1.  A ``ValueError`` for data it cannot read."""

    tokens = state.get("tokens") or {}
    try:
        if not isinstance(tokens, Mapping):
            raise TypeError(f"tokens is a {type(tokens).__name__}")
        found: dict[str, list[int]] = {}
        for entry, positions in tokens.items():
            if not isinstance(positions, list) or any(
                isinstance(position, bool) or not isinstance(position, int) or position < 0 for position in positions
            ):
                raise ValueError(f"the positions of entry {entry!r}: {positions!r}")
            found[str(entry)] = list(positions)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot read the tokens: {exc}") from exc
    return found


@dataclass(frozen=True)
class ResumePlan:
    """What a resume does with each draw a checkpoint left outstanding (:func:`plan_resume`): ``(key, task id,
    position)`` each."""

    # The groups the checkpoint kept, which the steps train first, in the order they settled.
    kept: list[tuple[Key, str, int | None]]
    # The draws to draw again first, in (start, seq) order: their groups were running, or could not be kept.
    reissue: list[tuple[Key, str, int | None]]
    # The draws that waited for their label's limit, back to their labels' backlogs in order.
    waiting: list[tuple[Key, str, int | None]]
    # The draws whose task left the run's training tasks.
    lost: list[tuple[Key, str, int | None]]

    def keys(self) -> list[Key]:
        """Every outstanding draw the plan takes up or loses."""

        return [key for part in (self.kept, self.reissue, self.waiting, self.lost) for key, _task, _position in part]


def plan_resume(
    *,
    outstanding: Iterable[Sequence[Any]],
    waiting: Mapping[str, Iterable[Sequence[Any]]],
    kept: Iterable[Key],
    pool: Container[str],
) -> ResumePlan:
    """Split the draws a checkpoint left outstanding (``outstanding``: ``[start, seq, task id, position]`` each) by
    what a resume does with them, given the keys of the groups the checkpoint kept that read back (``kept``, in the
    order they settled), each label's waiting draws (``waiting``, ``[start, seq]`` each, oldest first) and the run's
    training tasks now (``pool``): a kept group trains as it is; a draw that waited for its label goes back to its
    backlog; every other one is drawn again first; and a draw whose task left ``pool`` is lost.  A kept group or a
    waiting draw that is not outstanding is passed over.  A ``ValueError`` for data it cannot read."""

    try:
        drawn: dict[Key, tuple[str, int | None]] = {
            (int(start), int(seq)): (str(task), None if position is None else int(position))
            for start, seq, task, position in outstanding
        }
        waits = [(int(start), int(seq)) for keys in waiting.values() for start, seq in keys]
        kept_keys = [(int(start), int(seq)) for start, seq in kept]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot read the outstanding draws: {type(exc).__name__}: {exc}") from exc
    plan = ResumePlan(kept=[], reissue=[], waiting=[], lost=[])
    taken: set[Key] = set()
    others = sorted(set(drawn) - set(kept_keys) - set(waits))
    for name, keys in (("kept", kept_keys), ("waiting", waits), ("reissue", others)):
        for key in keys:
            if key not in drawn or key in taken:
                continue
            taken.add(key)
            task, position = drawn[key]
            getattr(plan, name if task in pool else "lost").append((key, task, position))
    return plan
