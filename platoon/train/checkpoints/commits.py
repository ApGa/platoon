"""Commits: the full checkpoints of a run whose trained policies step independently (RFC 0021, section 2.11; D14).

Stepping in lockstep, a checkpoint is a barrier of every trained policy's saves, which the leader's row commits
(:mod:`platoon.train.checkpoints.barrier`).  Stepping independently, each policy saves on its own clock, with no
barrier, and **every full save of any policy is a commit**: a point a resume can start from.  A commit is three parts
(:meth:`Commits.commit`):

1. **Capture**, under the commit lock (which the settle section of each group and each step's consumption of its
   shares take too, :mod:`platoon.train.dispatch.shares`), once no step's end runs its step hooks
   (:meth:`Commits.hook_phase`: a hook's state is captured between its works, never part-way through one, rule 10):
   the next ``commit_seq``; each checkpoint participant's capture (the rollout source takes its kept groups, the ledger
   of their shares and its dispatcher's state at once, by reference; a participant without ``capture`` runs its
   ``checkpoint_state`` here); and each other trained policy's newest durable row and published version.  The rows it
   names are pinned.
2. **Write**, after the lock is released: the participants' files, then the policy's Tinker save.  The other policies
   keep stepping meanwhile.
3. **Append**: the policy's row in its own ``checkpoints.jsonl``, with what was captured.  It is now that file's newest
   commit, whose rows stay pinned; the rows its file's previous newest commit named are pinned no more, unless
   another newest commit names them.  Each share the policy consumed before the row's step is durable now.

The row's ``loop_state`` is ``{batch, wandb_run_id, policy_version, optimizer_steps, commit_seq, <participants' keys>,
policies}``: ``batch`` the leader's step (its own when the leader commits, else that of the leader's row it names);
``policy_version`` and ``optimizer_steps`` the committing policy's; ``policies`` an entry for every trained policy,
``{name, state_path, policy_version, published_version, optimizer_steps, step}``, ``step`` the policy's own step at the
row (where it resumes), and the committing policy's naming the row itself (no ``state_path``).  The start's launch and
exit saves are barriers of every policy, as in lockstep, whose leader row is a commit naming them all.  A policy other
than the leader names its own commits apart from the barriers' rows in its file, which the leader's step names
(:class:`~platoon.train.trainer.clocks.OwnClocks`), so that no two rows of one training run share a state path.

A resume starts from the commit with the highest ``commit_seq`` in any policy's file whose named rows its policies
still have (:func:`newest_commit`): a named row whose ``state_path`` its policy's ``checkpoints_deleted.jsonl`` lists,
or another of its rows holds too (one save wrote over the other's), is missing, and a newer commit that does not hold
is skipped with a warning that ``lock.json`` records
(``policies_restore``).  Each policy resumes from the row the commit names, at the row's ``policy_version``, unless the
commit records a ``published_version`` above it: then at one more than that, so that no new sampler shares a number
with one the crash lost (:func:`resumed_version`).  Numbers go on from the highest ``commit_seq`` in any file, never
from the chosen commit's.

Each policy's pruner skips the pinned rows (:meth:`Commits.pinned`), so that no commit names a row that was deleted
while its save ran.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
from collections import Counter
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tinker_cookbook import checkpoint_utils

from platoon.train.checkpoints.barrier import Counters, _record_skipped, _state_rows, policy_dir
from platoon.train.checkpoints.participants import POLICIES_KEY
from platoon.train.checkpoints.pruning import read_checkpoint_records
from platoon.train.policy.spec import PolicyId
from platoon.utils.config import ConfigError

if TYPE_CHECKING:
    from platoon.train.dispatch.shares import Shares
    from platoon.train.policy.checkpoints import Waiter
    from platoon.train.policy.set import PolicySet

# The key of a commit's loop_state that orders the commits a run's policies saved.
COMMIT_SEQ_KEY = "commit_seq"


def rows_dir(run_dir: str, leader: PolicyId, policy: PolicyId) -> str:
    """Where ``policy`` keeps its rows: the run's directory for the leader, else ``<run>/policies/<id>``."""

    return run_dir if policy == leader else policy_dir(run_dir, policy)


def resumed_version(entry: Mapping[str, Any], row_version: int) -> int:
    """The version a policy resumes at from a commit's ``entry`` for it and its row's ``row_version``: the row's,
    unless the commit records a published version above it (samplers published after the row, which the crash lost):
    then one more than that."""

    published = entry.get("published_version")
    if isinstance(published, int) and not isinstance(published, bool) and published > row_version:
        return published + 1
    return row_version


@dataclass(frozen=True)
class Commit:
    """A commit a resume may start from: its row (``record``), the policy that saved it and its ``seq``, and for every
    trained policy the row it names (``rows``; its own, the committing policy's) and the entry that names it."""

    record: checkpoint_utils.CheckpointRecord
    policy: PolicyId
    seq: int
    rows: Mapping[PolicyId, checkpoint_utils.CheckpointRecord]
    entries: Mapping[PolicyId, Mapping[str, Any]]


def _commit_rows(directory: str) -> list[dict[str, Any]]:
    """The commits among a policy's rows: those with a ``commit_seq`` and a state to resume from, in order."""

    return [
        row
        for row in read_checkpoint_records(directory)
        if _is_seq(row.get(COMMIT_SEQ_KEY)) and isinstance(row.get("state_path"), str)
    ]


def _is_seq(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def newest_commit(run_dir: str, policies: Sequence[PolicyId]) -> Commit | None:
    """The commit a resume of the run in ``run_dir`` starts from, over its trained ``policies`` (the leader first): the
    one with the highest ``commit_seq`` in any policy's file whose named rows hold (see the module docstring); None for
    a run that saved nothing yet.  :class:`~platoon.utils.config.ConfigError` (exit 78: no restart can mend it) when the
    run saved rows and no commit holds."""

    leader = policies[0]
    held = {policy: _resumable_rows(rows_dir(run_dir, leader, policy)) for policy in policies}
    found = sorted(
        (
            (row[COMMIT_SEQ_KEY], policy, row)
            for policy in policies
            for row in _commit_rows(rows_dir(run_dir, leader, policy))
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    skipped: list[str] = []
    for seq, policy, row in found:
        commit, missing = _named_rows(row, policy, seq, held, policies)
        if commit is not None:
            if skipped:
                _record_skipped(run_dir, f"{commit.record.name} (commit {seq}, policy {policy})", skipped)
            return commit
        skipped.append(f"commit {seq} (policy {policy}'s checkpoint {row.get('name')}): {'; '.join(missing)}")
    if skipped:
        raise ConfigError(
            f"No commit of the run in {run_dir} names rows that every trained policy still has, so no start can resume "
            f"it: {' | '.join(skipped)}. Restore the rows each names (in <run>/policies/<id>/checkpoints.jsonl, with "
            "their checkpoints on the policy's service), or train a new trial in a new run directory"
        )
    if held[leader]:
        raise ConfigError(
            f"The run in {run_dir} saved its checkpoints with its trained policies stepping in lockstep (no row holds "
            "a commit_seq), and this start steps them independently: a resume keeps the stepping its run began with "
            "(train.stepping: lockstep, or a trainer whose needs() say so), else a new trial in a new run directory"
        )
    return None


def _resumable_rows(directory: str) -> dict[str, checkpoint_utils.CheckpointRecord]:
    """The rows of a policy's ``checkpoints.jsonl`` that a commit may name, by ``state_path``: those a resume can load
    (not deleted), but a state path that two rows hold, since a later save under one name writes over the earlier
    one's state, and no row then says whose weights the path holds."""

    paths = Counter(row.get("state_path") for row in read_checkpoint_records(directory))
    return {path: record for path, record in _state_rows(directory).items() if paths[path] == 1}


def _named_rows(
    row: Mapping[str, Any],
    policy: PolicyId,
    seq: int,
    held: Mapping[PolicyId, Mapping[str, checkpoint_utils.CheckpointRecord]],
    policies: Sequence[PolicyId],
) -> tuple[Commit | None, list[str]]:
    """The commit of ``policy``'s ``row`` when every row it names holds (``held``: :func:`_resumable_rows`), else None
    and, one line each, what does not: its own state deleted or saved over, a policy it names no row of, or a named row
    its policy no longer has."""

    record = checkpoint_utils.CheckpointRecord.from_dict(dict(row))
    named = row.get(POLICIES_KEY)
    if not isinstance(named, Mapping):
        named = {}
    missing = [] if record.state_path in held[policy] else [f"its state {record.state_path} was deleted or saved over"]
    rows: dict[PolicyId, checkpoint_utils.CheckpointRecord] = {policy: record}
    entries: dict[PolicyId, Mapping[str, Any]] = {}
    for other in policies:
        entry = named.get(other)
        if not isinstance(entry, Mapping):
            missing.append(f"it names no row of policy {other}")
            continue
        entries[other] = entry
        if other == policy:
            continue
        state_path = entry.get("state_path")
        if state_path not in held[other]:
            missing.append(f"policy {other} no longer has the row it names ({state_path}): deleted or saved over")
        else:
            rows[other] = held[other][state_path]
    if missing:
        return None, missing
    return Commit(record, policy, seq, rows, entries), []


def next_commit_seq(run_dir: str, policies: Sequence[PolicyId]) -> int:
    """The ``commit_seq`` of the run's next commit: one more than the highest in any policy's file (0 for the first),
    whichever commit the start resumed from, so that a new commit never ties with one an earlier start left."""

    leader = policies[0]
    seqs = [row[COMMIT_SEQ_KEY] for policy in policies for row in _commit_rows(rows_dir(run_dir, leader, policy))]
    return max(seqs, default=-1) + 1


def _names(row: Mapping[str, Any], policy: PolicyId) -> dict[PolicyId, str]:
    """The state path of each row a commit names, its own included, by policy."""

    named = row.get(POLICIES_KEY)
    if not isinstance(named, Mapping):
        named = {}
    paths = {
        other: entry["state_path"]
        for other, entry in named.items()
        if isinstance(entry, Mapping) and isinstance(entry.get("state_path"), str)
    }
    return {**paths, policy: row["state_path"]}


async def _written(pending: Sequence[Any]) -> list[Any]:
    """Each participant's entries of a commit, in order: what it captured, or what its writing gives once awaited (part
    2).  When one writing fails, or the commit is cancelled while it runs, the writings after it are closed, never run,
    and the error is raised."""

    entries = []
    for index, item in enumerate(pending):
        try:
            entries.append(await item if inspect.isawaitable(item) else item)
        except BaseException:
            _close(pending[index + 1 :])
            raise
    return entries


def _close(writings: Sequence[Any]) -> None:
    """Close the writings of a commit that will never run (a coroutine never awaited warns when it is collected)."""

    for writing in writings:
        if inspect.iscoroutine(writing):
            writing.close()


class Commits:
    """The commits of one start of a run whose trained policies (``policies``) step independently: the commit lock,
    the next ``commit_seq``, each policy's newest durable row, the rows each file's newest commit and each commit in
    flight name (the pins), and the step ends whose hooks run now (:meth:`hook_phase`).  ``participants`` are the
    run's checkpoint participants (its checkpoints' list, read at each commit), ``wandb_run_id`` reads the W&B run each
    commit goes on with, and ``ledger`` (the rollout source's share ledger, once it is open, :meth:`attach`) takes the
    commit lock and hears of each durable row."""

    def __init__(
        self,
        run_dir: str,
        policies: PolicySet,
        *,
        participants: Sequence[Any],
        wandb_run_id: Callable[[], str | None],
    ) -> None:
        self.run_dir = run_dir
        self.policies = policies
        self.trained = [policy.id for policy in policies.trainable()]
        self.leader = policies.leader
        self.participants = participants
        self._wandb_run_id = wandb_run_id
        self.ledger: Shares | None = None
        # The commit lock (see the module docstring), which the share ledger takes too (attach).
        self.lock = asyncio.Lock()
        self.next_seq = next_commit_seq(run_dir, self.trained)
        # Each policy's newest durable row: its entry as a commit names it (with its state_path).
        self.newest: dict[PolicyId, dict[str, Any]] = {}
        # The rows each policy's file's newest commit names, and those each commit in flight names, by policy.
        self._newest_names: dict[PolicyId, dict[PolicyId, str]] = {}
        for policy in self.trained:
            commits = _commit_rows(rows_dir(run_dir, self.leader, policy))
            if commits:
                self._newest_names[policy] = _names(commits[-1], policy)
        self._in_flight: dict[int, dict[PolicyId, str]] = {}
        # How many step ends run their hooks now (hook_phase), and the condition over the commit lock that a commit
        # waits on until none does.
        self._hook_phases = 0
        self._quiet: asyncio.Condition | None = None

    def attach(self, ledger: Shares) -> None:
        """The run's draws opened with ``ledger``, which takes the commit lock (its settle sections and the steps'
        consumption hold it) and hears of each durable row."""

        self.ledger = ledger
        ledger.lock = self.lock

    def _condition(self) -> asyncio.Condition:
        """The condition over the commit lock that a commit waits on until no step's end runs its hooks."""

        if self._quiet is None:
            self._quiet = asyncio.Condition(self.lock)
        return self._quiet

    @contextlib.asynccontextmanager
    async def hook_phase(self) -> AsyncGenerator[None, None]:
        """Held while a step's end runs its step hooks' ``drain`` and ``update``, and while they hear its publication
        (the run's ``hooks.running``): a commit captures nothing meanwhile, so that it never saves a hook's state
        part-way through its work at another policy's step end (rule 10), and a phase begins only once a capture in
        progress is taken.  A commit's own step end publishes between its phases, so a commit never waits for
        itself."""

        condition = self._condition()
        async with condition:
            self._hook_phases += 1
        try:
            yield
        finally:
            async with condition:
                self._hook_phases -= 1
                condition.notify_all()

    def pinned(self, policy: PolicyId) -> set[str]:
        """The state paths of ``policy``'s rows that a commit in flight, or the newest commit of some policy's file,
        names: its pruner never deletes them."""

        names = [*self._in_flight.values(), *self._newest_names.values()]
        return {named[policy] for named in names if policy in named}

    def take_seq(self) -> int:
        """The next ``commit_seq``, which a commit takes at its capture."""

        seq, self.next_seq = self.next_seq, self.next_seq + 1
        return seq

    def path_of(self, policy: PolicyId) -> str:
        return rows_dir(self.run_dir, self.leader, policy)

    async def commit(
        self, policy: PolicyId, name: str, *, step: int, counters: Counters, wait: Waiter
    ) -> dict[str, str]:
        """``policy``'s full checkpoint ``name`` at its step ``step`` (where a resume from it starts the policy), with
        its ``counters``, as a commit: capture, write, append (see the module docstring).  The paths its save wrote."""

        condition = self._condition()
        async with condition:  # the commit lock, once no step's end runs its hooks
            await condition.wait_for(lambda: not self._hook_phases)
            seq, pending, others = await self._capture(policy, name)
        try:
            entries = await _written(pending)
            own = {
                "name": name,
                "policy_version": counters.version,
                "published_version": counters.version,
                "optimizer_steps": counters.optimizer_steps,
                "step": step,
            }
            batch = step if policy == self.leader else others[self.leader]["step"]  # the leader's row it names
            loop_state: dict[str, Any] = {
                "batch": batch,
                "wandb_run_id": self._wandb_run_id(),
                "policy_version": counters.version,
                "optimizer_steps": counters.optimizer_steps,
                COMMIT_SEQ_KEY: seq,
            }
            for entry in entries:
                loop_state.update(entry)
            loop_state[POLICIES_KEY] = {other: own if other == policy else others[other] for other in self.trained}
            os.makedirs(self.path_of(policy), exist_ok=True)
            saved = await self.policies[policy].save(
                name, log_path=self.path_of(policy), loop_state=loop_state, wait=wait
            )
        finally:
            del self._in_flight[seq]
        self._appended(policy, {**own, "state_path": saved["state_path"]}, others)
        return saved

    async def _capture(self, policy: PolicyId, name: str) -> tuple[int, list[Any], dict[PolicyId, dict[str, Any]]]:
        """Part 1, under the commit lock: the commit's ``commit_seq``; each participant's entries, or the writing that
        gives them (its ``capture``); each other trained policy's newest durable row with its published version, whose
        rows are pinned until the commit is appended or fails."""

        seq = self.take_seq()
        pending: list[Any] = []
        try:
            for participant in self.participants:
                capture = getattr(participant, "capture", None)
                pending.append(capture(name) if callable(capture) else await participant.checkpoint_state(name))
        except BaseException:
            _close(pending)  # the writings captured before the failure never run
            raise
        others = {
            other: {**self.newest[other], "published_version": self.policies[other].version}
            for other in self.trained
            if other != policy
        }
        self._in_flight[seq] = {other: str(entry["state_path"]) for other, entry in others.items()}
        return seq, pending, others

    def _appended(self, policy: PolicyId, own: dict[str, Any], others: Mapping[PolicyId, Mapping[str, Any]]) -> None:
        """Part 3, once the row is appended: it is the policy's newest durable row and its file's newest commit (whose
        rows stay pinned), and each share it consumed before the row's step is durable."""

        self.newest[policy] = {
            key: own[key] for key in ("name", "state_path", "policy_version", "optimizer_steps", "step")
        }
        self._newest_names[policy] = {
            **{other: entry["state_path"] for other, entry in others.items()},
            policy: own["state_path"],
        }
        if self.ledger is not None:
            self.ledger.durable(policy, own["step"])

    # --- the start's barriers (launch and exit) ----------------------------------------------------------------------

    def barrier_entries(
        self,
        name: str,
        named: Mapping[PolicyId, Mapping[str, Any]],
        *,
        leader: Counters,
        step: int,
        others: Mapping[PolicyId, Counters],
    ) -> dict[PolicyId, dict[str, Any]]:
        """The ``policies`` entries of a barrier's leader row ``name`` (a launch or an exit save of every policy): each
        other policy's row as the barrier saved it (``named``), at its own step and published version (``others``),
        and the leader's own (``leader``, at ``step``), which names the row itself."""

        entries: dict[PolicyId, dict[str, Any]] = {}
        for policy in self.trained:
            counted = leader if policy == self.leader else others[policy]
            saved = {"name": name} if policy == self.leader else named[policy]
            entries[policy] = {
                **saved,
                "policy_version": counted.version,
                "published_version": counted.version,
                "optimizer_steps": counted.optimizer_steps,
                "step": counted.step if counted.step is not None else step,
            }
        return entries

    def barrier_saved(self, entries: Mapping[PolicyId, Mapping[str, Any]], leader_paths: Mapping[str, str]) -> None:
        """A barrier's leader row is appended (``leader_paths``, what its save wrote): every policy's row it names is
        that policy's newest durable row, the leader's file's newest commit names them all, and each policy's shares
        consumed before its step are durable."""

        rows = {policy: dict(entry) for policy, entry in entries.items()}
        rows[self.leader]["state_path"] = leader_paths["state_path"]
        for policy, entry in rows.items():
            self.newest[policy] = {
                key: entry[key] for key in ("name", "state_path", "policy_version", "optimizer_steps", "step")
            }
        self._newest_names[self.leader] = {policy: entry["state_path"] for policy, entry in rows.items()}
        if self.ledger is not None:
            for policy, entry in rows.items():
                self.ledger.durable(policy, entry["step"])
