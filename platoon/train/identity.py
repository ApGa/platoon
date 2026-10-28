"""One run per run directory: ``identity.json``, and the warning for an evaluation that lists training tasks.

A run directory holds one run: its checkpoints, rollouts and metrics.  A resume that trains another model, or that
would move tasks between training and evaluation, would mix two runs in one directory and one curve.  So the first
start of a run writes ``identity.json`` into its directory: the model and the renderer, in a run that declares
``policies:`` or ``agents:`` the table of its policies (each one's model, renderer and whether it trains, and a frozen
one's ``load_checkpoint_path``, the sampler it serves) and which policy each agent role samples, and for each
environment entry with a hold-out (``service_kwargs.holdout``) the rule, the dataset and the held-out tasks, refused
ones included, as the service records them (``split_record``).  The table records no service URL, so a resume may
reach its services elsewhere.  A start that resumes (its directory holds a checkpoint to resume from) compares its own
with it after the tasks are listed, before the training service is opened, and stops with a configuration error (exit
78) when:

- the model or the renderer is another;
- a policy was added or removed, or one's model, renderer, training or sampler changed, or a role samples another
  policy (each named, with the field that changed);
- a task the run held out would now train;
- a task is now held out that was not, a hold-out added to a running run included.

The refused start records why in its ``lock.json`` (``resume_refused``).  A refusal that changed moves no held-out
task (the hold-out walks over refused tasks too): the resume goes on, and logs the held-out tasks that left the
evaluation or joined it again, and those that only left the selection.  A start without a checkpoint writes the file
afresh, and a resume of a run that has none (started before platoon wrote it) writes it then, unless the run has a
hold-out now, which it may not add.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from platoon.train.components import split_task_id
from platoon.train.policy.spec import MAIN, AgentBindings, PolicySpec, agent_bindings, policy_specs
from platoon.utils.config import ConfigError, update_run_lock

logger = logging.getLogger(__name__)

IDENTITY_NAME = "identity.json"
IDENTITY_VERSION = 1
# The lock.json key under which a refused start records why.
LOCK_KEY = "resume_refused"
_SHOWN = 3  # task ids a problem names


def check_training_run(config: Any, run_dir: str | os.PathLike[str], train_dataset: Any, eval_dataset: Any) -> None:
    """What ``platoon train`` checks once it has listed its tasks, before it opens the training service: the run's
    identity (:func:`check_run_identity`), and with an evaluation, whether it lists training tasks
    (:func:`warn_eval_overlap`)."""

    from platoon.train.auto import AutoEnvironment

    environments = AutoEnvironment.environments(config)
    check_run_identity(run_dir, run_identity(config, environments), entry_task_ids(train_dataset, environments))
    if eval_dataset is not None:
        warn_eval_overlap(environments, train_dataset, eval_dataset)


def run_identity(config: Any, environments: Sequence[Any]) -> dict[str, Any]:
    """What ``identity.json`` records of a training config: ``model``, ``renderer``, and ``environments``, each entry
    with a hold-out by its label.  The services of those entries are built to ask for their record, after the tasks
    were listed with the same settings."""

    from platoon.envs.service import get_service

    entries: dict[str, Any] = {}
    for environment in environments:
        if environment.service is None or "holdout" not in (environment.service_kwargs or {}):
            continue
        service = get_service(environment.service, **environment.service_kwargs)
        split_record = getattr(service, "split_record", None)
        record = split_record() if callable(split_record) else None
        if record is not None:
            entries[environment.label] = record
    return {
        "version": IDENTITY_VERSION,
        "model": config.train.model_name,
        "renderer": config.train.renderer_name,
        **policy_table(config),
        "environments": entries,
    }


def policy_table(config: Any) -> dict[str, Any]:
    """What ``identity.json`` records of the run's policies, in a run that declares ``policies:`` or ``agents:``
    (nothing in any other, whose identity stays as it was): each policy's record by id (``policies``,
    :func:`_policy_record`), and the agents' default policy and each role's (``agents``)."""

    agents = agent_bindings(config)
    declared = getattr(config, "policies", None) or {}
    if not declared and agents == AgentBindings():
        return {}
    return {
        "policies": {spec.id: _policy_record(spec) for spec in policy_specs(config).values()},
        "agents": {"policy": agents.default, "roles": dict(agents.roles)},
    }


def _policy_record(spec: PolicySpec) -> dict[str, Any]:
    """A policy as ``identity.json`` records it, under its config keys: its model, its renderer, whether it trains,
    and for a frozen policy the sampler it serves (``load_checkpoint_path``, a ``tinker://`` path, never a URL; None:
    its base model).  A trained policy's ``load_checkpoint_path`` only starts a new run, so a resume ignores it."""

    record: dict[str, Any] = {
        "model_name": spec.model_name,
        "renderer_name": spec.renderer_name,
        "trainable": spec.trainable,
    }
    if not spec.trainable:
        record["load_checkpoint_path"] = spec.load_checkpoint_path
    return record


def check_run_identity(
    run_dir: str | os.PathLike[str], identity: Mapping[str, Any], train_ids: Mapping[str, Iterable[str]]
) -> None:
    """Write ``identity`` into ``run_dir``, or, when the run resumes, refuse a start whose identity moved (see the
    module docstring).  ``train_ids`` are the task ids each entry trains on now, by label.

    Raises :class:`~platoon.utils.config.ConfigError`, after recording the refusal in ``lock.json``.
    """

    run_dir = Path(run_dir)
    path = run_dir / IDENTITY_NAME
    if not resumes(run_dir):
        _write(path, identity)
        return
    previous = _read(path)
    problems, notes = identity_changes(previous, identity, train_ids)
    if problems:
        update_run_lock(run_dir, {LOCK_KEY: {"identity": str(path), "problems": problems}})
        raise ConfigError(
            [
                f"{run_dir} holds a run this start would change, so it does not resume it:",
                *problems,
                "A run directory holds one run: start a new one under another stats.trial_name (or log_path), or "
                "resume with the run's own model, renderer and held-out tasks",
            ]
        )
    for note in notes:
        logger.info("%s", note)
    if previous is None:
        _write(path, identity)


def identity_changes(
    previous: Mapping[str, Any] | None, identity: Mapping[str, Any], train_ids: Mapping[str, Iterable[str]]
) -> tuple[list[str], list[str]]:
    """What a resume with ``identity`` would change in a run that recorded ``previous`` (None: no record): the
    problems that refuse it, and what it changes that a resume may change, to log: held-out tasks that left the
    selection, and held-out tasks that a changed refusal took out of the evaluation or put back into it."""

    problems: list[str] = []
    notes: list[str] = []
    if previous is None:
        for label in sorted(identity.get("environments") or {}):
            problems.append(
                f"environments[{label}] holds tasks out, and the run recorded no hold-out ({IDENTITY_NAME} is "
                "missing): it may have trained on them"
            )
        return problems, notes
    for key, what in (("model", "model"), ("renderer", "renderer")):
        if previous.get(key) != identity.get(key):
            problems.append(f"the run trains {what} {previous.get(key)!r}, and this start {identity.get(key)!r}")
    problems.extend(policy_changes(previous, identity))
    before = previous.get("environments") or {}
    now = identity.get("environments") or {}
    for label in sorted(set(before) | set(now)):
        held_before, held_now = _ids(before.get(label), "held_out"), _ids(now.get(label), "held_out")
        training = {str(task_id) for task_id in train_ids.get(label) or ()}
        if would_train := held_before & training:
            problems.append(
                f"environments[{label}]: {len(would_train)} task(s) the run held out would now train, such as "
                f"{sorted(would_train)[:_SHOWN]}"
            )
        if newly_held := held_now - held_before:
            what = "the run held out no task" if label not in before else "the run did not hold them out"
            problems.append(
                f"environments[{label}]: {len(newly_held)} task(s) are now held out, and {what}, such as "
                f"{sorted(newly_held)[:_SHOWN]}"
            )
        if gone := held_before - held_now - training:
            notes.append(
                f"environments[{label}]: {len(gone)} task(s) the run held out left the selection since it started, "
                f"such as {sorted(gone)[:_SHOWN]}; the evaluation goes on without them"
            )
        evaluated_before, evaluated_now = _ids(before.get(label), "eval") & held_now, _ids(now.get(label), "eval")
        if refused := evaluated_before - evaluated_now:
            notes.append(
                f"environments[{label}]: {len(refused)} held-out task(s) are refused now and leave the evaluation, "
                f"such as {sorted(refused)[:_SHOWN]}; they stay held out"
            )
        if back := (evaluated_now - evaluated_before) & held_before:
            notes.append(
                f"environments[{label}]: {len(back)} held-out task(s) are runnable again and join the evaluation, "
                f"such as {sorted(back)[:_SHOWN]}"
            )
    return problems, notes


def policy_changes(previous: Mapping[str, Any], identity: Mapping[str, Any]) -> list[str]:
    """What a resume with ``identity`` changes of the policy table of a run that recorded ``previous``, one line each: a
    policy other than ``main`` (whose model and renderer are the run's, compared on their own) added or removed, a
    field of one's record changed, the agents' default policy or a role's.  A record without the table is a run of
    ``main`` alone, which every agent samples."""

    problems = []
    before, now = previous.get("policies") or {}, identity.get("policies") or {}
    for policy in _in_order(before, now):
        if policy == MAIN:
            continue
        if policy not in now:
            problems.append(f"policies.{policy}: the run has it, and this start does not declare it")
        elif policy not in before:
            problems.append(f"policies.{policy}: this start declares it, and the run does not have it")
        else:
            old, new = before[policy], now[policy]
            problems.extend(
                f"policies.{policy}.{key}: the run's is {old.get(key)!r}, and this start's {new.get(key)!r}"
                for key in _in_order(old, new)
                if old.get(key) != new.get(key)
            )
    agents_before, agents_now = previous.get("agents") or {}, identity.get("agents") or {}
    if (default := agents_before.get("policy", MAIN)) != (now_default := agents_now.get("policy", MAIN)):
        problems.append(f"agents.policy: the run's is {default!r}, and this start's {now_default!r}")
    roles_before, roles_now = agents_before.get("roles") or {}, agents_now.get("roles") or {}
    problems.extend(
        f"agents.roles.{role}.policy: the run's is {roles_before.get(role)!r}, and this start's {roles_now.get(role)!r}"
        for role in _in_order(roles_before, roles_now)
        if roles_before.get(role) != roles_now.get(role)
    )
    return problems


def _in_order(before: Iterable[str], now: Iterable[str]) -> list[str]:
    """The keys of ``before``, then those of ``now`` that ``before`` lacks, each once, in their order: the run's
    record's, then this start's new ones."""

    return list(dict.fromkeys([*before, *now]))


def _ids(record: Mapping[str, Any] | None, key: str) -> set[str]:
    """The task ids an entry's record lists under ``key`` (``held_out`` or ``eval``); none without a record."""

    return {str(task_id) for task_id in (record or {}).get(key) or ()}


def resumes(run_dir: str | os.PathLike[str]) -> bool:
    """Whether a start in ``run_dir`` resumes: its ``checkpoints.jsonl`` lists a checkpoint training can resume
    from, as the trainer decides."""

    from tinker_cookbook import checkpoint_utils

    return checkpoint_utils.get_last_checkpoint(str(run_dir)) is not None


def entry_task_ids(dataset: Any, environments: Sequence[Any]) -> dict[str, set[str]]:
    """The task ids of a listed dataset (a training or an evaluation split), by entry label: a mixture's ids carry
    their entry's label, a single entry's do not."""

    task_ids = _task_ids(dataset)
    if len(environments) == 1:
        return {environments[0].label: set(task_ids)}
    by_label: dict[str, set[str]] = {}
    for task_id in task_ids:
        label, raw = split_task_id(task_id)
        by_label.setdefault(label, set()).add(raw)
    return by_label


def warn_eval_overlap(environments: Sequence[Any], train_dataset: Any, eval_dataset: Any) -> list[str]:
    """Warn, once per entry, when the evaluation lists tasks that training draws from: its numbers would then
    measure tasks training sees.  Returns the labels warned about."""

    train = entry_task_ids(train_dataset, environments)
    evaluate = entry_task_ids(eval_dataset, environments)
    warned = []
    for environment in environments:
        label = environment.label
        shared = train.get(label, set()) & evaluate.get(label, set())
        if not shared:
            continue
        warned.append(label)
        logger.warning(
            "environments[%s]: the evaluation lists %d of its %d task(s) among the tasks training draws from, such as "
            "%s, so its numbers measure tasks training sees; hold tasks out for evaluation (a Harbor entry's "
            "service_kwargs.holdout, or a split manifest)",
            label,
            len(shared),
            len(evaluate.get(label, ())),
            sorted(shared)[:_SHOWN],
        )
    return warned


def _task_ids(dataset: Any) -> list[str]:
    if dataset is None:
        return []
    if isinstance(dataset, list):
        return [str(row["task_id"]) if isinstance(row, Mapping) else str(row) for row in dataset]
    columns = getattr(dataset, "column_names", None) or ()
    if "task_id" in columns:
        return [str(task_id) for task_id in dataset["task_id"]]
    return []


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ConfigError(
            f"{path} cannot be read ({exc}); the run's identity is unknown, so it is not resumed"
        ) from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} holds no JSON object; the run's identity is unknown, so it is not resumed")
    return data


def _write(path: Path, identity: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {**identity, "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(record, indent=1) + "\n")
    partial.replace(path)
