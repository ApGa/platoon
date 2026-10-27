"""Deleting a run's older recovery checkpoints through the Tinker API (``checkpoint.keep_recovery``).

A recovery checkpoint (``recovery-<batch>``) is a full save of the weights and the optimizer that only exists so that
a crash loses few steps: every start of a run writes one, and ``checkpoint.recovery_every`` writes one every that many
steps. For a large model trained in full each is hundreds of gigabytes, so a long run fills the service's disk unless
the older ones go. With ``checkpoint.keep_recovery: N``, once a recovery save is recorded in ``checkpoints.jsonl``,
:class:`RecoveryCheckpointPruner` deletes the run's recovery checkpoints beyond the newest N, both the state and the
sampler weights of each, through the service's checkpoint deletion (``RestClient.delete_checkpoint_from_tinker_path``).

What is never deleted (:func:`plan_deletions`):

* a permanent checkpoint (``<batch>``, ``<batch>_interrupted``, ``final``), or anything a permanent record names;
* the resume point: the newest record with a ``state_path``, which a start resumes from;
* a recovery checkpoint a sampling client of this start may still use: one this start saved and published within the
  run's largest staleness bound of the policy (``train.max_staleness``, or an environment entry's ``max_staleness``
  above it), or whose version is unknown. A checkpoint an earlier start saved serves no sampling client of this one,
  and neither does one that saved a policy a step did not update, in a run of several trained policies (its unchanged
  weights, saved for the checkpoint, whose sampler no rollout samples), and under a trainer that rolls nothing out
  (``trainer: sft``) none does;
* anything that ``checkpoints.jsonl`` of this run does not list: another run's checkpoints, or the checkpoint
  ``load_checkpoint_path`` names;
* in a run of several trained policies, a row of another policy than the leader that a leader row a resume may take
  names: the leader's newest, or one of its newest ``keep_recovery`` recovery rows
  (:func:`platoon.train.checkpoints.barrier.named_state_paths`). Each such policy has a pruner of its own, in its own
  directory, on a session of its own on its own service.

A deleted recovery checkpoint's files in the run's ``tasks`` directory go with it: the snapshot of the run's per-task
statistics (``tasks``), and the state of the dispatcher's draws with the groups it kept (``dispatch``,
:mod:`platoon.train.dispatch.dispatcher`): the files each of its own rows names, never one found by the checkpoint's
name, which a later start may have saved again.

Deletes run beside training, so a step never waits for one; each has :data:`DELETE_TIMEOUT_S` to answer. They go
through a Tinker session of their own, never the training session: the Tinker SDK reports a failed delete as the end of
the session it ran in, and a self-hosted service such as miles-tinker then releases that session's models. Each path
the service deletes, or answers it does not have (404, as below), is appended to ``checkpoints_deleted.jsonl`` in the
run directory, so a resumed run goes on from where its earlier starts left off and never asks for one twice. A failed
delete is counted (``checkpoint/delete_failures``) and tried again after the next recovery save. A 404 is believed only
once the service has deleted a checkpoint in this start: a service without the delete route answers 404 for its path
too, and recording such answers would leave those checkpoints to no later start. Until then a 404 is asked again, and
counts as failed for the state of a recovery checkpoint this start saved, which nothing but a delete removes. After
:data:`MAX_FAILURES_IN_A_ROW` failures in a row, or at once when the service cannot delete checkpoints at all, pruning
stops for the rest of the start with a warning that says what failed, and training goes on.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from platoon.utils.config import ConfigError
from platoon.utils.jsonl import append_jsonl_line, read_jsonl

logger = logging.getLogger(__name__)

# Name prefix of recovery checkpoints: full weights+optimizer saves written at every start and every
# ``checkpoint.recovery_every`` steps so that a crash loses few steps. They are disposable, while the permanent
# ``<batch>`` / ``<batch>_interrupted`` / ``final`` checkpoints are kept forever.
RECOVERY_CHECKPOINT_PREFIX = "recovery-"
# The record, in the run directory, of the checkpoint paths pruning has deleted (or the service no longer had).
DELETED_CHECKPOINTS_NAME = "checkpoints_deleted.jsonl"
# Failed deletes in a row after which pruning stops for the rest of the start.
MAX_FAILURES_IN_A_ROW = 3
# Seconds one delete may take before it counts as failed: a little over the Tinker SDK's own budget for retrying a
# request (5 minutes), so that the SDK's last error is the one reported.
DELETE_TIMEOUT_S = 330.0
# Seconds the end of a run waits for a delete still in flight; the next start tries again what it cut short.
EXIT_WAIT_S = 30.0
# HTTP statuses with which a service says it has no checkpoint deletion: Method Not Allowed, Not Implemented.
_NO_DELETION_STATUSES = (405, 501)
_NOT_FOUND = 404
_TINKER_SCHEME = "tinker://"
_CHECKPOINT_KINDS = ("weights", "sampler_weights")


def recovery_pruning_problems(
    keep_recovery: int | None,
    recovery_every: int | None,
    max_staleness: int | None,
    *,
    bound: str = "train.max_staleness",
) -> list[str]:
    """What stops ``checkpoint.keep_recovery`` from keeping its promise, one entry each, with the keys' dotted paths;
    checked when a training config loads.

    Pruning never deletes a recovery checkpoint that rollouts may still sample from: one within ``max_staleness``
    versions of the policy, the run's largest staleness bound, which the key ``bound`` sets (``train.max_staleness``,
    or an environment entry's ``max_staleness`` above it). So it needs that limit, and with a recovery cadence the
    newest ``keep_recovery`` recovery checkpoints must span more than it (``keep_recovery × recovery_every >
    max_staleness``): otherwise the checkpoints within reach of rollouts outnumber ``keep_recovery``, and more stay
    than it keeps. Without ``recovery_every`` the only recovery checkpoints are the ones each start writes.
    """

    if keep_recovery is None:
        return []
    if max_staleness is None:
        return [
            "checkpoint.keep_recovery needs train.max_staleness: without a staleness limit rollouts may still sample "
            "from any older checkpoint, so pruning could never delete one; set train.max_staleness (how many policy "
            "versions old a rollout may be and still train)"
        ]
    if recovery_every is None or keep_recovery * recovery_every > max_staleness:
        return []
    largest = ", the run's largest staleness bound" if bound.startswith("environments[") else ""
    return [
        f"checkpoint.keep_recovery: {keep_recovery} × checkpoint.recovery_every: {recovery_every} = "
        f"{keep_recovery * recovery_every} must exceed {bound}: {max_staleness}{largest}: a recovery checkpoint "
        f"within {max_staleness} policy versions may still serve rollouts and is never deleted, so more would stay "
        f"than keep_recovery keeps; raise checkpoint.keep_recovery to {max_staleness // recovery_every + 1} or "
        f"checkpoint.recovery_every to {max_staleness // keep_recovery + 1}, or lower {bound}"
    ]


def config_pruning_problems(config: Any) -> list[str]:
    """The :func:`recovery_pruning_problems` of a training config, against the largest staleness bound of the run and
    its environment entries (:func:`platoon.train.entry_settings.largest_staleness`): a slow entry's group may still
    sample from an old checkpoint; and against each trained ``policies:`` entry's own bound, which its own pruner
    keeps.  None under a trainer that rolls nothing out, which no rollout samples."""

    from platoon.train.entry_settings import largest_staleness
    from platoon.train.trainer.registry import rolls_out

    checkpoint = getattr(config, "checkpoint", None)
    if getattr(config, "train", None) is None or not rolls_out(config):
        return []
    keep, every = getattr(checkpoint, "keep_recovery", None), getattr(checkpoint, "recovery_every", None)
    largest, bound = largest_staleness(config)
    problems = recovery_pruning_problems(keep, every, largest, bound=bound)
    for policy_id, policy in (getattr(config, "policies", None) or {}).items():
        own = getattr(policy, "max_staleness", None)
        if getattr(policy, "trainable", True) and own is not None and largest is not None:
            problems += recovery_pruning_problems(keep, every, own, bound=f"policies.{policy_id}.max_staleness")
    return problems


def check_recovery_pruning(config: Any) -> None:
    """Raise :class:`~platoon.utils.config.ConfigError` (exit 78) with the :func:`config_pruning_problems` of a
    training config. The trainer checks again before it opens the training service, for a config class whose own
    checks skip the training config's."""

    if problems := config_pruning_problems(config):
        raise ConfigError(problems)


@dataclass(frozen=True)
class RecordedCheckpoint:
    """One checkpoint ``checkpoints.jsonl`` lists: its newest record's name, its paths (state first), the policy
    version it holds (``policy_version``, else ``batch``; None when it records neither), the position of its
    newest record in the file, and the files in the run's ``tasks`` directory its records name (:func:`_record_files`).
    """

    name: str
    paths: tuple[str, ...]
    version: int | None
    index: int
    files: tuple[str, ...] = ()

    @property
    def recovery(self) -> bool:
        return self.name.startswith(RECOVERY_CHECKPOINT_PREFIX)


def read_checkpoint_records(run_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """The records of ``checkpoints.jsonl`` in ``run_dir``, in order; a line that does not parse (one being written)
    is skipped, and a missing file has none."""

    return read_jsonl(os.path.join(run_dir, "checkpoints.jsonl"))


def read_deleted_paths(run_dir: str | os.PathLike[str]) -> set[str]:
    """The checkpoint paths ``checkpoints_deleted.jsonl`` in ``run_dir`` lists: deleted by an earlier pass or start."""

    return {
        record["path"]
        for record in read_jsonl(os.path.join(run_dir, DELETED_CHECKPOINTS_NAME))
        if isinstance(record.get("path"), str)
    }


def recorded_checkpoints(records: Sequence[Mapping[str, Any]]) -> list[RecordedCheckpoint]:
    """The checkpoints ``records`` name, each once, in the order of their newest record: two records with one path
    (the same name saved twice by one training run) are one checkpoint."""

    checkpoints: dict[str, RecordedCheckpoint] = {}
    for index, record in enumerate(records):
        if paths := _record_paths(record):
            earlier = checkpoints.pop(paths[0], None)
            files = (*(earlier.files if earlier is not None else ()), *_record_files(record))
            checkpoints[paths[0]] = RecordedCheckpoint(_record_name(record), paths, _version(record), index, files)
    return list(checkpoints.values())


def _record_files(record: Mapping[str, Any]) -> tuple[str, ...]:
    """The files in the run's ``tasks`` directory that a record names: the snapshot of the per-task statistics (its
    ``tasks`` record's ``file``), and the state of the dispatcher's draws and the groups the checkpoint kept (its
    ``dispatch`` record's ``file`` and ``kept``)."""

    return tuple(
        name
        for section, keys in (("tasks", ("file",)), ("dispatch", ("file", "kept")))
        if isinstance(names := record.get(section), Mapping)
        for key in keys
        if isinstance(name := names.get(key), str) and name
    )


def _record_paths(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(path for key in ("state_path", "sampler_path") if isinstance(path := record.get(key), str))


def _record_name(record: Mapping[str, Any]) -> str:
    name = record.get("name")
    return "" if name is None else str(name)


def _version(record: Mapping[str, Any]) -> int | None:
    for key in ("policy_version", "batch"):
        value = record.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def resume_paths(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """The paths of the resume point: the newest record with a ``state_path`` (tinker-cookbook's
    ``get_last_checkpoint``), which a start resumes from."""

    for record in reversed(records):
        if isinstance(record.get("state_path"), str):
            return _record_paths(record)
    return ()


def live_recovery_checkpoints(
    checkpoints: Sequence[RecordedCheckpoint], deleted: Collection[str]
) -> list[RecordedCheckpoint]:
    """The recovery checkpoints among ``checkpoints`` that a start could still resume from, oldest first: those whose
    first path (the state) is not in ``deleted``."""

    return [checkpoint for checkpoint in checkpoints if checkpoint.recovery and checkpoint.paths[0] not in deleted]


def plan_deletions(
    records: Sequence[Mapping[str, Any]],
    *,
    keep: int,
    deleted: Collection[str],
    policy_version: int,
    max_staleness: int | None,
    this_start: Collection[str],
    named: Collection[str] = (),
) -> list[tuple[RecordedCheckpoint, str]]:
    """The paths to delete now, oldest first, each with its checkpoint: the paths left of every recovery checkpoint
    but the newest ``keep`` a start could resume from, except what is never deleted (see the module docstring).

    ``records`` are ``checkpoints.jsonl``'s, in order; ``deleted`` the paths already deleted; ``this_start`` the
    first path (the state path) of each recovery checkpoint this start saved; ``policy_version`` the policy's
    version now; ``max_staleness`` the run's largest staleness bound, or None when no rollout samples the policy;
    ``named`` the state paths of the checkpoints that a leader row a resume may take names (another policy's rows,
    which the leader's commit).
    """

    if keep < 1:
        raise ValueError("keep must be at least 1")
    checkpoints = recorded_checkpoints(records)
    kept = live_recovery_checkpoints(checkpoints, deleted)[-keep:]
    protected = set(resume_paths(records))
    for record in records:  # every permanent record, also one a later record of the same path renamed
        if not _record_name(record).startswith(RECOVERY_CHECKPOINT_PREFIX):
            protected.update(_record_paths(record))
    for checkpoint in checkpoints:
        sampled = (
            max_staleness is not None
            and checkpoint.paths[0] in this_start
            and _maybe_sampled(checkpoint, policy_version, max_staleness)
        )
        if sampled or checkpoint in kept or checkpoint.paths[0] in named:
            protected.update(checkpoint.paths)
    return [
        (checkpoint, path)
        for checkpoint in checkpoints
        if checkpoint.recovery and checkpoint not in kept
        for path in checkpoint.paths
        if path not in protected and path not in deleted and _is_checkpoint_path(path)
    ]


def _maybe_sampled(checkpoint: RecordedCheckpoint, policy_version: int, max_staleness: int) -> bool:
    """Whether a sampling client of this start may still use ``checkpoint``, which this start saved: rollouts that
    sampled from it still train while it is within ``max_staleness`` versions of the policy."""

    return checkpoint.version is None or policy_version - checkpoint.version <= max_staleness


def _is_checkpoint_path(path: str) -> bool:
    """A ``tinker://<training run>/<weights|sampler_weights>/<checkpoint>`` path: the only kind pruning deletes."""

    parts = path[len(_TINKER_SCHEME) :].split("/") if path.startswith(_TINKER_SCHEME) else []
    return len(parts) == 3 and all(parts) and parts[1] in _CHECKPOINT_KINDS


def _is_state_path(path: str) -> bool:
    """A ``tinker://<training run>/weights/<checkpoint>`` path: weights and optimizer, which a resume loads."""

    return _is_checkpoint_path(path) and path[len(_TINKER_SCHEME) :].split("/")[1] == _CHECKPOINT_KINDS[0]


class RecoveryCheckpointPruner:
    """Deletes a run's older recovery checkpoints beside training (``checkpoint.keep_recovery``).

    The trainer calls :meth:`recovery_saved` after each recovery save that ``checkpoints.jsonl`` records, adds
    :meth:`step_metrics` to each step's metrics, and awaits :meth:`aclose` when the run ends.
    """

    def __init__(
        self,
        *,
        run_dir: str,
        keep: int,
        max_staleness: int | None,
        service_client_factory: Callable[[], Any],
        named: Callable[[], Collection[str]] | None = None,
    ) -> None:
        """``max_staleness`` is the run's largest staleness bound, or None when no rollout samples the policy (an sft
        run: ``k = 0``, and since its version never moves, none of this start's checkpoints is kept for a sampling
        client either).  ``service_client_factory`` opens the Tinker session the deletes go through (a
        ``tinker.ServiceClient`` of the training service): pruning calls it once, in a worker thread, before its first
        delete.  ``named``, for the pruner of another trained policy than the leader: the state paths of its rows that
        a leader row a resume may take names, read at each pass, which it never deletes."""

        if keep < 1:
            raise ValueError("keep must be at least 1")
        self.run_dir = run_dir
        self.keep = keep
        self.max_staleness = max_staleness
        self._service_client_factory = service_client_factory
        self._named = named
        self._rest_client: Any = None
        self._deleted = read_deleted_paths(run_dir)
        self._this_start: set[str] = set()
        self._policy_version = 0
        self._task: asyncio.Task[None] | None = None
        self._again = False
        self._failures_in_a_row = 0
        self._stopped = False
        # Whether the service deleted a checkpoint in this start: until it has, a 404 may only mean it has no route.
        self._service_deletes = False
        # Since the last step_metrics: paths deleted and deletes that failed.
        self._deleted_since = 0
        self._failed_since = 0
        self._recovery_kept: int | None = None
        logger.info(
            "checkpoint.keep_recovery: %d: keeping this run's newest %d recovery checkpoints and deleting older ones "
            "through the Tinker service (%d checkpoint paths deleted before this start, in %s)",
            keep,
            keep,
            len(self._deleted),
            DELETED_CHECKPOINTS_NAME,
        )

    @property
    def stopped(self) -> bool:
        """Whether pruning stopped for the rest of this start (failed deletes, or a service that cannot delete)."""

        return self._stopped

    def recovery_saved(self, paths: Mapping[str, str], policy_version: int, *, published: bool = True) -> None:
        """Note a recovery save of this start, which ``checkpoints.jsonl`` now records, at ``policy_version`` (the
        policy's version now), and prune beside training. A pass still running runs once more when it ends.
        ``published`` False: a save of weights the step did not update (a policy of several that did not step), whose
        sampler is never published, so no rollout samples it and it is pruned as any older checkpoint is."""

        first = paths.get("state_path") or paths.get("sampler_path")
        if first and published:
            self._this_start.add(first)
        self._policy_version = max(self._policy_version, policy_version)
        if self._stopped:
            return
        if self._task is not None and not self._task.done():
            self._again = True
            return
        self._task = asyncio.get_running_loop().create_task(self._run(), name="prune_recovery_checkpoints")

    def step_metrics(self) -> dict[str, float]:
        """The metrics a training step logs: paths deleted and deletes failed since the step before, how many
        recovery checkpoints the run keeps after the last pass, and whether pruning is on."""

        metrics = {
            "checkpoint/deleted": float(self._deleted_since),
            "checkpoint/delete_failures": float(self._failed_since),
            "checkpoint/pruning": 0.0 if self._stopped else 1.0,
        }
        if self._recovery_kept is not None:
            metrics["checkpoint/recovery_kept"] = float(self._recovery_kept)
        self._deleted_since = self._failed_since = 0
        return metrics

    async def aclose(self, timeout: float | None = None) -> None:
        """Wait up to ``timeout`` seconds (default :data:`EXIT_WAIT_S`) for the passes due, the one in flight and the
        one a later save asked for, then cancel them: the next start finishes their work."""

        task = self._task
        if task is None or task.done():
            return
        timeout = EXIT_WAIT_S if timeout is None else timeout
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if not done:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            logger.warning(
                "Stopped deleting recovery checkpoints after waiting %g s at the end of the run; the next start "
                "of this run deletes what is left",
                timeout,
            )

    async def _run(self) -> None:
        while True:
            self._again = False
            try:
                await self._prune()
            except asyncio.CancelledError:
                raise
            except Exception:  # never let pruning end training
                logger.exception("Pruning recovery checkpoints failed; it tries again after the next recovery save")
            if not self._again or self._stopped:
                return

    async def _prune(self) -> None:
        records = read_checkpoint_records(self.run_dir)
        doomed = plan_deletions(
            records,
            keep=self.keep,
            deleted=self._deleted,
            policy_version=self._policy_version,
            max_staleness=self.max_staleness,
            this_start=self._this_start,
            named=self._named() if self._named is not None else (),
        )
        for checkpoint, path in doomed:
            if self._stopped:
                break
            await self._delete(checkpoint, path)
        self._recovery_kept = len(live_recovery_checkpoints(recorded_checkpoints(records), self._deleted))

    async def _delete(self, checkpoint: RecordedCheckpoint, path: str) -> None:
        try:
            async with asyncio.timeout(DELETE_TIMEOUT_S):
                delete = await self._deletion()
                await delete(path)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if getattr(exc, "status_code", None) == _NOT_FOUND:
                self._not_found(checkpoint, path, exc)
            else:
                self._failed_delete(checkpoint, path, exc)
            return
        self._record(checkpoint, path, "deleted")
        self._service_deletes = True
        self._deleted_since += 1
        self._failures_in_a_row = 0
        logger.info(
            "Deleted recovery checkpoint %s (%s, policy version %s; checkpoint.keep_recovery: %d)",
            path,
            checkpoint.name,
            checkpoint.version,
            self.keep,
        )

    async def _deletion(self) -> Callable[[str], Any]:
        """The service's checkpoint deletion, ``RestClient.delete_checkpoint_from_tinker_path_async`` of the Tinker
        SDK, on a session of pruning's own, which the first call opens; a ``NotImplementedError`` says the client has
        none."""

        if self._rest_client is None:
            service_client = await asyncio.to_thread(self._service_client_factory)
            create_rest_client = getattr(service_client, "create_rest_client", None)
            if create_rest_client is None:
                raise NotImplementedError(f"{type(service_client).__name__} has no create_rest_client")
            self._rest_client = create_rest_client()
        delete = getattr(self._rest_client, "delete_checkpoint_from_tinker_path_async", None)
        if delete is None:
            raise NotImplementedError(
                f"{type(self._rest_client).__name__} has no delete_checkpoint_from_tinker_path_async"
            )
        return delete

    def _not_found(self, checkpoint: RecordedCheckpoint, path: str, exc: BaseException) -> None:
        """The service answers that it has no checkpoint at ``path`` (404).

        Before the service has deleted a checkpoint in this start, the answer may only mean that it has no delete
        route, since a service without one answers 404 for the route's path too: nothing is recorded, the next pass
        asks again, and for the state of a recovery checkpoint this start saved, which nothing but a delete removes,
        the delete counts as failed. Once the service has deleted one, a 404 means the checkpoint is gone (someone
        deleted it, the service retired it as a LoRA gateway does old sampler weights, a retry found the delete done,
        or a start deleted it and stopped before recording that), or that it was saved under another API key, which
        the service answers alike. It is recorded and never asked for again.
        """

        if not self._service_deletes:
            if path in self._this_start:
                self._failed_delete(
                    checkpoint,
                    path,
                    exc,
                    hint="a state this start saved, and nothing deleted yet: the service may have no delete route",
                )
                return
            logger.info(
                "Recovery checkpoint %s was not found (%s); it is asked for again once the service has deleted a "
                "checkpoint, since a service without the delete route answers 404 too",
                path,
                _describe(exc),
            )
            return
        self._record(checkpoint, path, "absent")
        if _is_state_path(path):  # only a delete removes a state: someone else deleted it, or another API key saved it
            logger.warning(
                "Recovery checkpoint %s is not on the service (%s), so it is not asked for again. If it still exists, "
                "saved under another API key, delete it yourself",
                path,
                _describe(exc),
            )
        else:
            logger.info("Recovery checkpoint %s is already gone (%s); it is not asked for again", path, _describe(exc))

    def _failed_delete(self, checkpoint: RecordedCheckpoint, path: str, exc: BaseException, *, hint: str = "") -> None:
        self._failed_since += 1
        self._failures_in_a_row += 1
        status = getattr(exc, "status_code", None)
        if isinstance(exc, NotImplementedError) or status in _NO_DELETION_STATUSES:
            self._stop(
                f"the Tinker service cannot delete checkpoints: deleting {path} failed with {_describe(exc)}. This "
                "client or service offers no checkpoint deletion (the Tinker API's DELETE "
                "/api/v1/training_runs/<id>/checkpoints/<checkpoint>, which the SDK's "
                "RestClient.delete_checkpoint_from_tinker_path calls)",
                error=True,
            )
            return
        error = _describe(exc) + (f" ({hint})" if hint else "")
        logger.warning(
            "Could not delete recovery checkpoint %s: %s (%d failed delete(s) in a row; pruning stops at %d)",
            path,
            error,
            self._failures_in_a_row,
            MAX_FAILURES_IN_A_ROW,
        )
        if self._failures_in_a_row >= MAX_FAILURES_IN_A_ROW:
            self._stop(f"{self._failures_in_a_row} deletes failed in a row, the last for {path}: {error}")

    def _stop(self, why: str, *, error: bool = False) -> None:
        self._stopped = True
        log = logger.error if error else logger.warning
        log(
            "checkpoint.keep_recovery: stopped deleting recovery checkpoints for the rest of this start: %s. Training "
            "goes on, and the run's recovery checkpoints now accumulate, one per start and one every "
            "checkpoint.recovery_every steps; delete the older ones yourself (checkpoints.jsonl lists them, and %s "
            "what was deleted), or fix the service and restart the run, which resumes pruning",
            why,
            DELETED_CHECKPOINTS_NAME,
        )

    def _record(self, checkpoint: RecordedCheckpoint, path: str, result: str) -> None:
        """Remember that ``path`` is gone, in memory and in ``checkpoints_deleted.jsonl``, for later starts."""

        self._deleted.add(path)
        record = {
            "path": path,
            "name": checkpoint.name,
            "policy_version": checkpoint.version,
            "result": result,
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        try:
            append_jsonl_line(os.path.join(self.run_dir, DELETED_CHECKPOINTS_NAME), json.dumps(record))
        except OSError as exc:  # a later start asks the service once more, which answers that it is gone
            logger.warning("Could not record the deletion of %s in %s: %s", path, DELETED_CHECKPOINTS_NAME, exc)
        if _is_state_path(path):  # no start resumes from it now: its files in tasks/ go with it
            self._delete_files(checkpoint)

    def _delete_files(self, checkpoint: RecordedCheckpoint) -> None:
        """Delete the files in the run's ``tasks`` directory that ``checkpoint``'s own records name (its snapshots of
        the per-task statistics, its states of the dispatcher's draws and the groups it kept); a file already gone, or
        one outside that directory, is left alone."""

        tasks = (Path(self.run_dir) / "tasks").resolve()
        for name in checkpoint.files:
            path = (Path(self.run_dir) / name).resolve()
            if path.parent != tasks:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not delete %s, a file of a deleted checkpoint: %s", path, exc)


def _describe(exc: BaseException) -> str:
    status = getattr(exc, "status_code", None)
    if isinstance(exc, TimeoutError) and not str(exc):
        return f"no answer within {DELETE_TIMEOUT_S:.0f} s"
    text = str(exc).strip() or type(exc).__name__
    return f"{type(exc).__name__} (HTTP {status}): {text}" if status is not None else f"{type(exc).__name__}: {text}"
