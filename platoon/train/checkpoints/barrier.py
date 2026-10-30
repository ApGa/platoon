"""A run's checkpoint when it trains several policies in lockstep: a barrier of their saves, and the start point that
a resume finds in them.

The leader (the run's first trained policy) keeps its rows in the run's ``checkpoints.jsonl``, as a run's one policy
always did.  Every other trained policy keeps its own, in ``<run>/policies/<id>/checkpoints.jsonl``
(:func:`policy_dir`), on its own service.  A checkpoint of the run saves them all: first every other policy, at the
same time (:func:`save_others`), then the leader, whose row names each other policy's row by its ``state_path``
(``loop_state.policies``, :data:`POLICIES_KEY`), so that the leader's row, written last, commits the checkpoint.  A
crash between the saves leaves rows that no leader row names: a resume never reads them, and their policy's pruner
deletes them later.  Each policy's pruner keeps the rows that the leader rows a resume may take name: the leader's
newest, and its newest ``checkpoint.keep_recovery`` recovery rows (:func:`named_state_paths`).

A resume takes the newest leader row whose named rows every policy still has (:func:`newest_barrier`): each named
row is in its policy's ``checkpoints.jsonl``, and its pruner has not deleted it.  A newer leader row that does not hold
is skipped with a warning, which ``lock.json`` records (``policies_restore``).

A run of one trained policy has none of this: no ``policies`` directory, and rows without ``policies``.
"""

import json
import logging
import os
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from tinker_cookbook import checkpoint_utils

from platoon.train.checkpoints.participants import POLICIES_KEY
from platoon.train.checkpoints.pruning import (
    live_recovery_checkpoints,
    read_checkpoint_records,
    read_deleted_paths,
    recorded_checkpoints,
)
from platoon.train.policy.spec import PolicyId
from platoon.utils.config import LOCK_NAME, ConfigError, update_run_lock

if TYPE_CHECKING:
    from platoon.train.policy.checkpoints import Waiter
    from platoon.train.policy.set import PolicySet

logger = logging.getLogger(__name__)

# The list lock.json keeps of the starts that skipped newer checkpoints whose rows did not hold.
RESTORE_KEY = "policies_restore"


def policy_dir(run_dir: str, policy: PolicyId) -> str:
    """Where a trained policy other than the leader keeps its checkpoint rows: ``<run>/policies/<id>``."""

    return os.path.join(run_dir, "policies", policy)


class Counters(NamedTuple):
    """What a policy's row of a checkpoint records of it: the version its saved weights are published at, and the
    optimizer steps it took; stepping independently, its own step at the row (None: the checkpoint's)."""

    version: int
    optimizer_steps: int
    step: int | None = None


async def save_others(
    policies: "PolicySet",
    run_dir: str,
    name: str,
    *,
    step: int,
    counters: Mapping[PolicyId, Counters],
    wait: "Waiter",
) -> tuple[dict[PolicyId, dict[str, str]], dict[PolicyId, dict[str, Any]]]:
    """Save checkpoint ``name`` of every trained policy but the leader, at the same time, each on its own service into
    its own directory (:func:`policy_dir`), with its row's ``loop_state``: ``batch`` (``step``), its version and its
    optimizer steps (``counters``, else its own now).  The paths each save wrote, and what the leader's row records of
    each (``name``, ``state_path``, ``policy_version``, ``optimizer_steps``), by policy; neither in a run of one
    trained policy.  A save that fails stops the checkpoint, naming its policy
    (:meth:`~platoon.train.policy.set.PolicySet.each`)."""

    others = [policy for policy in policies.trainable() if policy.id != policies.leader]
    if not others:
        return {}, {}
    states = {}
    for policy in others:
        os.makedirs(policy_dir(run_dir, policy.id), exist_ok=True)
        counted = counters[policy.id] if policy.id in counters else Counters(policy.version, policy.optimizer_steps)
        states[policy.id] = {
            "batch": step,
            "policy_version": counted.version,
            "optimizer_steps": counted.optimizer_steps,
        }
    saves = {
        policy.id: policy.save(name, log_path=policy_dir(run_dir, policy.id), loop_state=states[policy.id], wait=wait)
        for policy in others
    }
    saved = await policies.each(saves, doing=f"saving checkpoint {name}")
    named = {
        policy: {
            "name": name,
            "state_path": paths["state_path"],
            "policy_version": states[policy]["policy_version"],
            "optimizer_steps": states[policy]["optimizer_steps"],
        }
        for policy, paths in saved.items()
    }
    return saved, named


@dataclass(frozen=True)
class Barrier:
    """A checkpoint of every trained policy: the leader's row (``record``), and the row it names of each other policy
    (``rows``), each with the version and optimizer steps it recorded."""

    record: checkpoint_utils.CheckpointRecord
    rows: Mapping[PolicyId, checkpoint_utils.CheckpointRecord]


def newest_barrier(run_dir: str, others: Collection[PolicyId]) -> Barrier | None:
    """The newest checkpoint of the run in ``run_dir`` whose leader row names a row of each policy of ``others`` that
    the policy still has (:func:`_named_rows`); None for a run that saved none.  Each newer leader row that does not
    hold is skipped with a warning, which ``lock.json`` records (``policies_restore``).
    :class:`~platoon.utils.config.ConfigError` (exit 78: no restart can mend it) when the run saved checkpoints and
    none holds, naming what each lacks and what to do: a resume never trains on from rows that disagree."""

    leader_rows = [row for row in read_checkpoint_records(run_dir) if isinstance(row.get("state_path"), str)]
    leader_deleted = read_deleted_paths(run_dir)
    rows = {policy: _state_rows(policy_dir(run_dir, policy)) for policy in others}
    skipped: list[str] = []
    for row in reversed(leader_rows):
        record = checkpoint_utils.CheckpointRecord.from_dict(row)
        named, missing = _named_rows(record, rows, leader_deleted)
        if not missing:
            if skipped:
                _record_skipped(run_dir, record.name, skipped)
            return Barrier(record, named)
        skipped.append(f"checkpoint {record.name}: {'; '.join(missing)}")
    if skipped:
        raise ConfigError(
            f"No checkpoint of the run in {run_dir} names rows that every trained policy still has, so no start can "
            f"resume it: {' | '.join(skipped)}. Restore the rows each names (in <run>/policies/<id>/checkpoints.jsonl, "
            "with their checkpoints on the policy's service), or train a new trial in a new run directory"
        )
    return None


def _state_rows(directory: str) -> dict[str, checkpoint_utils.CheckpointRecord]:
    """The rows a policy's ``checkpoints.jsonl`` holds that a resume can load, by ``state_path``, without those its
    pruner deleted (``checkpoints_deleted.jsonl``): a later row of the same path wins."""

    deleted = read_deleted_paths(directory)
    return {
        row["state_path"]: checkpoint_utils.CheckpointRecord.from_dict(row)
        for row in read_checkpoint_records(directory)
        if isinstance(row.get("state_path"), str) and row["state_path"] not in deleted
    }


def _named_rows(
    record: checkpoint_utils.CheckpointRecord,
    rows: Mapping[PolicyId, Mapping[str, checkpoint_utils.CheckpointRecord]],
    leader_deleted: Collection[str],
) -> tuple[dict[PolicyId, checkpoint_utils.CheckpointRecord], list[str]]:
    """The row of each policy that a leader row (``record``) names, among each policy's ``rows``; and, one line each,
    what does not hold: the leader's own state deleted, a policy it names no row of, or a named row its policy no
    longer has."""

    missing = [f"its state {record.state_path} was deleted"] if record.state_path in leader_deleted else []
    named_rows = record.extra.get(POLICIES_KEY)
    named = named_rows if isinstance(named_rows, Mapping) else {}
    found: dict[PolicyId, checkpoint_utils.CheckpointRecord] = {}
    for policy, by_path in rows.items():
        entry = named.get(policy)
        state_path = entry.get("state_path") if isinstance(entry, Mapping) else None
        if state_path is None:
            missing.append(f"it names no row of policy {policy}")
        elif state_path not in by_path:
            missing.append(f"policy {policy} no longer has the row it names ({state_path})")
        else:
            found[policy] = by_path[state_path]
    return found, missing


def _record_skipped(run_dir: str, checkpoint: str, skipped: list[str]) -> None:
    """Warn that the start resumes from ``checkpoint``, older than the newer ones ``skipped`` (why each, one line), and
    add it to ``lock.json``'s ``policies_restore``.  Never raises."""

    logger.warning(
        "Resuming from checkpoint %s: the newer checkpoints name rows that a trained policy no longer has, so they "
        "are skipped: %s",
        checkpoint,
        " | ".join(skipped),
    )
    try:
        path = Path(run_dir) / LOCK_NAME
        lock = json.loads(path.read_text()) if path.is_file() else None
        earlier = lock.get(RESTORE_KEY) if isinstance(lock, dict) else None
        record = {"checkpoint": checkpoint, "skipped": skipped}
        update_run_lock(run_dir, {RESTORE_KEY: [*(earlier if isinstance(earlier, list) else []), record]})
    except Exception:  # what the record says never stops a resume
        logger.warning("Could not record in lock.json that this start skipped newer checkpoints", exc_info=True)


def named_state_paths(run_dir: str, policy: PolicyId, *, keep: int) -> set[str]:
    """The ``state_path`` of each row of ``policy`` that the policy's pruner keeps for a resume: the rows named by the
    leader rows that the leader's own pruner keeps to resume from, its newest row whose state is not deleted and its
    newest ``keep`` recovery rows (``checkpoint.keep_recovery``).  An older leader row that is only kept because the
    leader's rollouts may still sample it keeps none of another policy's rows: no resume would take it."""

    records = read_checkpoint_records(run_dir)
    deleted = read_deleted_paths(run_dir)
    kept = [records[found.index] for found in live_recovery_checkpoints(recorded_checkpoints(records), deleted)[-keep:]]
    live = [row for row in records if isinstance(row.get("state_path"), str) and row["state_path"] not in deleted]
    paths = set()
    for row in (*live[-1:], *kept):
        entry = row[POLICIES_KEY].get(policy) if isinstance(row.get(POLICIES_KEY), Mapping) else None
        if isinstance(entry, Mapping) and isinstance(entry.get("state_path"), str):
            paths.add(entry["state_path"])
    return paths
