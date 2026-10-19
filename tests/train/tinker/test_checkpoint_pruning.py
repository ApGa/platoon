"""``checkpoint.keep_recovery``: a run deletes its older recovery checkpoints through the Tinker API.

After each recovery save that ``checkpoints.jsonl`` records, the run deletes its recovery checkpoints beyond the newest
``keep_recovery``, beside training and on a Tinker session of its own; never a permanent checkpoint, the resume point,
or one this start saved within ``train.max_staleness`` policy versions.  ``checkpoints_deleted.jsonl`` lets a resumed
run go on where it left off.  Failed deletes are counted and three in a row stop pruning; a service that cannot delete
stops it at once with an error that names what is missing.  The config's loading refuses ``keep_recovery`` when
``keep_recovery × recovery_every`` does not exceed ``max_staleness`` (exit 78).

The trainer tests run ``train()`` on a fake Tinker service (``tinker_service_fakes``) with tinker-cookbook's own
checkpoint saving.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from tinker_service_fakes import FakeTinkerService, make_pruning_trainer, status_error
from trainer_fakes import Workflow, trainer_config

from platoon import cli
from platoon.train.tinker import checkpoint_pruning
from platoon.train.tinker.checkpoint_pruning import (
    DELETED_CHECKPOINTS_NAME,
    RecoveryCheckpointPruner,
    plan_deletions,
    read_deleted_paths,
)
from platoon.train.tinker.config_defs import CheckpointConfig, PlatoonTinkerRLTrainerConfig, training_problems
from platoon.utils.config import ConfigError


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """``train()`` installs SIGINT and SIGTERM handlers; put the test process's back."""

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def record(name: str, version: int | None, *, run: int = 1, sampler: bool = True, state: bool = True) -> dict:
    """A ``checkpoints.jsonl`` record as platoon writes it, of training run ``run``."""

    entry: dict = {"name": name}
    if version is not None:
        entry.update(batch=version, policy_version=version)
    if state:
        entry["state_path"] = f"tinker://run-{run}/weights/{name}"
    if sampler:
        entry["sampler_path"] = f"tinker://run-{run}/sampler_weights/{name}"
    return entry


def paths(*names: str, run: int = 1) -> list[str]:
    return [f"tinker://run-{run}/{kind}/{name}" for name in names for kind in ("weights", "sampler_weights")]


def planned(records, *, keep=2, deleted=(), policy_version=None, max_staleness=0, this_start=None) -> list[str]:
    """The paths ``plan_deletions`` deletes; by default every record is this start's and the policy is at the newest
    record's version."""

    versions = [entry["policy_version"] for entry in records if "policy_version" in entry]
    starts = {entry.get("state_path") or entry.get("sampler_path") for entry in records}
    plan = plan_deletions(
        records,
        keep=keep,
        deleted=set(deleted),
        policy_version=max(versions) if policy_version is None else policy_version,
        max_staleness=max_staleness,
        this_start=starts if this_start is None else set(this_start),
    )
    return [path for _checkpoint, path in plan]


# --- what is deleted -------------------------------------------------------------------------------------------


def test_the_newest_recovery_checkpoints_and_every_permanent_one_stay():
    records = [
        record("recovery-000000", 0),
        record("recovery-000001", 1),
        record("recovery-000002", 2),
        record("000003", 3),
        record("recovery-000004", 4),
        record("000005_interrupted", 5),
        record("recovery-000005", 5),
        record("final", 6),
    ]

    assert planned(records, keep=2) == paths("recovery-000000", "recovery-000001", "recovery-000002")
    assert planned(records, keep=1) == paths("recovery-000000", "recovery-000001", "recovery-000002", "recovery-000004")
    assert planned(records, keep=5) == []


def test_the_resume_point_stays_even_beyond_the_newest():
    # The newest record with a state path is the resume point, whatever newer records without one say.
    records = [
        record("recovery-000004", 4),
        record("recovery-000005", 5),
        record("recovery-000006", 6, state=False),
        record("recovery-000007", 7, state=False),
    ]

    assert planned(records, keep=2) == paths("recovery-000004")


def test_what_rollouts_of_this_start_may_still_sample_stays():
    records = [record(f"recovery-{version:06d}", version) for version in range(6)]

    # Policy version 5 with max_staleness 2: rollouts sampled from versions 3 to 5 still train.
    assert planned(records, keep=1, max_staleness=2) == paths("recovery-000000", "recovery-000001", "recovery-000002")
    assert planned(records, keep=1, max_staleness=0) == paths(*(f"recovery-{version:06d}" for version in range(5)))
    # A checkpoint of this start whose version nobody recorded may be the one a rollout samples.
    unversioned = [record("recovery-a", None), record("recovery-b", None), record("recovery-c", 9)]
    assert planned(unversioned, keep=1, policy_version=20, max_staleness=1) == []


def test_an_earlier_starts_checkpoints_serve_no_sampler_of_this_one():
    # The previous start saved recovery-000005 (run-1) at version 5; this one resumed there, and its launch save
    # (run-2) holds the same version.  Rollouts of this start sample only what this start saved.
    records = [record("recovery-000004", 4), record("recovery-000005", 5), record("recovery-000005", 5, run=2)]
    this_start = {"tinker://run-2/weights/recovery-000005"}

    plan = planned(records, keep=1, policy_version=5, max_staleness=3, this_start=this_start)

    assert plan == paths("recovery-000004", "recovery-000005")


def test_deleted_paths_are_not_asked_for_again_and_a_half_deleted_checkpoint_is_finished():
    records = [record(f"recovery-{version:06d}", version) for version in range(4)]
    state_0, sampler_0, state_1, _sampler_1 = paths("recovery-000000", "recovery-000001")

    # recovery-000000 is gone; only the state of recovery-000001 is, so it no longer counts among the kept.
    plan = planned(records, keep=2, deleted={state_0, sampler_0, state_1})

    assert plan == ["tinker://run-1/sampler_weights/recovery-000001"]


def test_two_records_of_one_path_are_one_checkpoint():
    records = [record("recovery-000003", 3), record("recovery-000004", 4), record("recovery-000003", 5)]

    # The second save of recovery-000003 is the newest: recovery-000004 is the one beyond keep 1.
    assert planned(records, keep=1) == paths("recovery-000004")


def test_only_tinker_checkpoint_paths_are_deleted():
    records = [
        {"name": "recovery-000000", "batch": 0, "state_path": "/disk/recovery-000000"},
        {"name": "recovery-000001", "batch": 1, "state_path": "tinker://run-1/weights/a/b"},
        record("recovery-000002", 2),
        record("recovery-000003", 3),
    ]

    assert planned(records, keep=1) == paths("recovery-000002")


# --- the config ------------------------------------------------------------------------------------------------


def _load(tmp_path: Path, *overrides: str) -> PlatoonTinkerRLTrainerConfig:
    return trainer_config(tmp_path, *overrides)


@pytest.mark.parametrize("value", ["0", "-1", "true"])
def test_keep_recovery_keeps_at_least_one(tmp_path, value):
    with pytest.raises(ConfigError, match="checkpoint.keep_recovery"):
        _load(tmp_path, f"checkpoint.keep_recovery={value}", "train.max_staleness=1")


def test_keep_recovery_needs_a_staleness_limit(tmp_path):
    with pytest.raises(ConfigError) as error:
        _load(tmp_path, "checkpoint.keep_recovery=2", "checkpoint.recovery_every=5")

    assert "checkpoint.keep_recovery needs train.max_staleness" in str(error.value)


def test_keep_recovery_times_recovery_every_must_exceed_max_staleness(tmp_path):
    with pytest.raises(ConfigError) as error:
        _load(tmp_path, "checkpoint.keep_recovery=1", "checkpoint.recovery_every=3", "train.max_staleness=3")

    message = str(error.value)
    assert (
        "checkpoint.keep_recovery: 1 × checkpoint.recovery_every: 3 = 3 must exceed train.max_staleness: 3" in message
    )
    assert "raise checkpoint.keep_recovery to 2 or checkpoint.recovery_every to 4" in message
    for good in (("1", "4", "3"), ("2", "2", "3"), ("2", "5", "3")):
        keep, every, staleness = good
        config = _load(
            tmp_path,
            f"checkpoint.keep_recovery={keep}",
            f"checkpoint.recovery_every={every}",
            f"train.max_staleness={staleness}",
        )
        assert config.checkpoint.keep_recovery == int(keep)


def test_without_a_recovery_cadence_keep_recovery_prunes_the_launch_saves(tmp_path):
    config = _load(tmp_path, "checkpoint.keep_recovery=1", "train.max_staleness=0")

    assert (config.checkpoint.keep_recovery, config.checkpoint.recovery_every) == (1, None)
    assert _load(tmp_path).checkpoint.keep_recovery is None  # off by default


def test_the_command_line_refuses_an_unkeepable_promise_with_exit_78(tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "m", "renderer_name": "r", "max_staleness": 4, "max_training_steps": 3},
                "eval": {"every": 0},
                "checkpoint": {"every": 50, "recovery_every": 2, "keep_recovery": 2},
                "log_path": str(tmp_path / "logs"),
            }
        )
    )
    trained = []
    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", trained.append)

    with pytest.raises(SystemExit) as exited:
        cli.main(["config", "validate", "--no-resolve", str(path)])
    assert exited.value.code == cli.EXIT_CONFIG
    assert "checkpoint.keep_recovery: 2 × checkpoint.recovery_every: 2 = 4 must exceed" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exited:
        cli.main(["train", str(path)])
    assert exited.value.code == cli.EXIT_CONFIG
    assert "checkpoint.keep_recovery: 2 × checkpoint.recovery_every: 2 = 4 must exceed" in capsys.readouterr().err
    assert trained == []


@dataclass
class _ConfigWithoutTheChecks(PlatoonTinkerRLTrainerConfig):
    """A plugin's config class whose ``__post_init__`` does not call the training config's."""

    def __post_init__(self):
        pass


def test_platoon_train_refuses_before_anything_runs_what_a_config_class_let_through(tmp_path):
    # training_problems runs before the trainer registers its sampler, starts W&B or lists a task.
    loaded = trainer_config(tmp_path)
    config = _ConfigWithoutTheChecks(
        train=loaded.train,
        eval=loaded.eval,
        log_path=loaded.log_path,
        checkpoint=CheckpointConfig(every=0, recovery_every=2, keep_recovery=1),
        watchdog=loaded.watchdog,
    )
    config.train.max_staleness = 2

    problems = training_problems(config)

    assert any("checkpoint.keep_recovery: 1 × checkpoint.recovery_every: 2 = 2 must exceed" in p for p in problems)
    assert training_problems(trainer_config(tmp_path, "checkpoint.keep_recovery=2", "train.max_staleness=1")) == []


def test_the_trainer_refuses_before_opening_the_service_what_a_config_class_let_through(tmp_path, monkeypatch):
    service = FakeTinkerService()
    loaded = trainer_config(tmp_path)
    config = _ConfigWithoutTheChecks(
        train=loaded.train,
        eval=loaded.eval,
        log_path=loaded.log_path,
        checkpoint=CheckpointConfig(every=0, recovery_every=1, keep_recovery=2),
        watchdog=loaded.watchdog,
    )
    trainer, _logged = make_pruning_trainer(monkeypatch, config, service)

    with pytest.raises(ConfigError, match="checkpoint.keep_recovery needs train.max_staleness"):
        asyncio.run(trainer.train(train_workflow=Workflow()))
    assert service.sessions == []


# --- training --------------------------------------------------------------------------------------------------

PRUNE_EVERY_STEP = (
    "checkpoint={every: 100, recovery_every: 1, keep_recovery: 2}",
    "train.max_staleness=1",
)


def _train(trainer, workflow: Workflow | None = None) -> None:
    """Run ``train()`` to its end; one cut short by a timeout would save an interrupt checkpoint and return."""

    async def run() -> None:
        task = asyncio.ensure_future(trainer.train(train_workflow=workflow or Workflow(delay=0.02)))
        done, _ = await asyncio.wait({task}, timeout=60)
        if not done:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise AssertionError("training did not finish within 60 s")
        task.result()

    asyncio.run(run())


def _deletion_records(trainer) -> list[dict]:
    """The records of ``checkpoints_deleted.jsonl``, but a line a crash cut short."""

    path = Path(trainer.run_log_path) / DELETED_CHECKPOINTS_NAME
    records = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def _recovery_stored(service: FakeTinkerService) -> set[str]:
    return {path for path in service.stored if "/recovery-" in path}


def _pruner(trainer) -> RecoveryCheckpointPruner:
    pruner = trainer._recovery_pruner
    assert pruner is not None
    return pruner


def _total(trainer, logged, key: str) -> float:
    """A count over the run: what its steps logged, and what the pruner counted after the last of them."""

    return sum(stats.get(key, 0.0) for _step, stats in logged) + _pruner(trainer).step_metrics()[key]


def test_training_keeps_the_newest_recovery_checkpoints(tmp_path, monkeypatch):
    service = FakeTinkerService()
    config = trainer_config(tmp_path, "train.max_training_steps=5", *PRUNE_EVERY_STEP)
    trainer, logged = make_pruning_trainer(monkeypatch, config, service)

    _train(trainer)

    # The launch save recovery-000000, then recovery-000001 .. recovery-000005, then final.
    assert _recovery_stored(service) == service.paths(1, "recovery-000004", "recovery-000005")
    assert service.paths(1, "final") <= service.stored
    doomed = paths("recovery-000000", "recovery-000001", "recovery-000002", "recovery-000003")
    assert sorted(service.deleted) == sorted(doomed)
    assert sorted(service.delete_calls) == sorted(doomed)  # each asked for once
    assert [(entry["path"], entry["result"]) for entry in _deletion_records(trainer)] == [
        (path, "deleted") for path in service.deleted
    ]
    assert read_deleted_paths(trainer.run_log_path) == set(doomed)
    # Every step logs the pruning metrics (the pass after the launch save has run by the first step's end).
    assert all(stats["checkpoint/pruning"] == 1.0 and "checkpoint/recovery_kept" in stats for _step, stats in logged)
    assert _total(trainer, logged, "checkpoint/deleted") == len(doomed)
    assert _total(trainer, logged, "checkpoint/delete_failures") == 0
    assert _pruner(trainer).step_metrics()["checkpoint/recovery_kept"] == 2.0


def test_permanent_checkpoints_are_never_deleted(tmp_path, monkeypatch):
    service = FakeTinkerService()
    config = trainer_config(
        tmp_path,
        "train.max_training_steps=6",
        "checkpoint={every: 2, recovery_every: 1, keep_recovery: 1}",
        "train.max_staleness=0",
    )
    trainer, _logged = make_pruning_trainer(monkeypatch, config, service)

    _train(trainer)

    permanent = service.paths(1, "000002", "000004", "000006", "final")
    assert permanent <= service.stored
    assert not permanent & set(service.delete_calls)
    assert _recovery_stored(service) == service.paths(1, "recovery-000005")
    assert sorted(service.deleted) == sorted(paths("recovery-000000", "recovery-000001", "recovery-000003"))


def test_pruning_is_off_by_default(tmp_path, monkeypatch):
    service = FakeTinkerService()
    config = trainer_config(tmp_path, "train.max_training_steps=4", "checkpoint={every: 100, recovery_every: 1}")
    trainer, logged = make_pruning_trainer(monkeypatch, config, service)

    _train(trainer)

    assert service.delete_calls == []
    assert len(service.sessions) == 1  # training's own
    assert not any(key.startswith("checkpoint/") for _step, stats in logged for key in stats)
    assert not (Path(trainer.run_log_path) / DELETED_CHECKPOINTS_NAME).exists()


def test_deletes_go_through_a_session_of_their_own(tmp_path, monkeypatch):
    # A failed delete ends the session it ran in (the SDK reports it as the session's end, and a self-hosted service
    # then releases the session's models): on training's session it would end training.
    service = FakeTinkerService(failure=lambda path: status_error(500, "disk busy") if "000000" in path else None)
    config = trainer_config(tmp_path, "train.max_training_steps=5", *PRUNE_EVERY_STEP)
    trainer, logged = make_pruning_trainer(monkeypatch, config, service)

    _train(trainer)

    training_session, pruning_session = service.sessions
    assert not training_session.ended
    assert pruning_session.ended
    assert service.paths(1, "final") <= service.stored  # training went on to the end
    assert _total(trainer, logged, "checkpoint/delete_failures") >= 2


def test_failed_deletes_are_counted_and_three_in_a_row_stop_pruning(tmp_path, monkeypatch, caplog):
    service = FakeTinkerService(failure=lambda path: status_error(500, "storage unavailable"))
    config = trainer_config(tmp_path, "train.max_training_steps=8", *PRUNE_EVERY_STEP)
    trainer, logged = make_pruning_trainer(monkeypatch, config, service)

    with caplog.at_level(logging.WARNING, logger=checkpoint_pruning.__name__):
        _train(trainer)

    assert len(service.delete_calls) == 3
    assert _pruner(trainer).stopped
    assert _total(trainer, logged, "checkpoint/delete_failures") == 3
    assert _pruner(trainer).step_metrics()["checkpoint/pruning"] == 0.0
    warnings = [entry.getMessage() for entry in caplog.records]
    assert sum("Could not delete recovery checkpoint" in message for message in warnings) == 3
    stopped = [message for message in warnings if "stopped deleting recovery checkpoints" in message]
    assert len(stopped) == 1
    assert "3 deletes failed in a row" in stopped[0] and "storage unavailable" in stopped[0]
    assert service.paths(1, "final") <= service.stored  # training went on
    assert _deletion_records(trainer) == []


def test_a_delete_that_succeeds_resets_the_count(tmp_path, monkeypatch):
    answers = iter([status_error(500, "busy"), status_error(500, "busy")])
    service = FakeTinkerService(failure=lambda path: next(answers, None))
    config = trainer_config(tmp_path, "train.max_training_steps=6", *PRUNE_EVERY_STEP)
    trainer, logged = make_pruning_trainer(monkeypatch, config, service)

    _train(trainer)

    assert not _pruner(trainer).stopped
    assert _total(trainer, logged, "checkpoint/delete_failures") == 2
    assert _recovery_stored(service) == service.paths(1, "recovery-000005", "recovery-000006")


@pytest.mark.parametrize(
    "service",
    [
        pytest.param(FakeTinkerService(rest_deletes=False), id="no-delete-method"),
        pytest.param(FakeTinkerService(failure=lambda path: status_error(405, "Method Not Allowed")), id="405"),
        pytest.param(FakeTinkerService(failure=lambda path: status_error(501, "Not Implemented")), id="501"),
    ],
)
def test_a_service_that_cannot_delete_stops_pruning_with_an_error_not_a_crash(tmp_path, monkeypatch, caplog, service):
    config = trainer_config(tmp_path, "train.max_training_steps=5", *PRUNE_EVERY_STEP)
    trainer, logged = make_pruning_trainer(monkeypatch, config, service)

    with caplog.at_level(logging.WARNING, logger=checkpoint_pruning.__name__):
        _train(trainer)

    errors = [entry.getMessage() for entry in caplog.records if entry.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "the Tinker service cannot delete checkpoints" in errors[0]
    assert "DELETE /api/v1/training_runs/<id>/checkpoints/<checkpoint>" in errors[0]
    assert "RestClient.delete_checkpoint_from_tinker_path" in errors[0]
    assert len(service.delete_calls) <= 1 and service.deleted == []
    assert _pruner(trainer).stopped
    assert _pruner(trainer).step_metrics()["checkpoint/pruning"] == 0.0
    assert _total(trainer, logged, "checkpoint/delete_failures") == 1
    assert service.paths(1, "final") <= service.stored  # training went on to the end
    assert _recovery_stored(service) == service.paths(1, *(f"recovery-{step:06d}" for step in range(6)))


def test_a_checkpoint_the_service_no_longer_has_is_not_asked_for_again(tmp_path, monkeypatch):
    service = FakeTinkerService()  # answers 404 for a checkpoint it does not have, as hosted Tinker does
    gone = "tinker://run-1/sampler_weights/recovery-000000"
    config = trainer_config(tmp_path, "train.max_training_steps=5", *PRUNE_EVERY_STEP)
    trainer, logged = make_pruning_trainer(monkeypatch, config, service)

    # Someone deletes the launch save's sampler weights by hand once training has started.
    _train(trainer, Workflow(delay=0.02, on_start=lambda count: service.stored.discard(gone)))

    assert service.delete_calls.count(gone) == 1
    assert {entry["path"]: entry["result"] for entry in _deletion_records(trainer)}[gone] == "absent"
    assert gone in read_deleted_paths(trainer.run_log_path)  # nor will a later start ask for it
    assert _total(trainer, logged, "checkpoint/delete_failures") == 0  # gone is gone
    assert not _pruner(trainer).stopped
    assert _recovery_stored(service) == service.paths(1, "recovery-000004", "recovery-000005")


def _no_delete_route(path: str):
    """What a service without the delete route answers: no route matches the path, so FastAPI says 404 Not Found."""

    return status_error(404, "Not Found")


def test_a_service_without_the_states_this_start_saved_cannot_be_deleting_and_pruning_stops(
    tmp_path, monkeypatch, caplog
):
    # A service whose delete route answers 404 to everything: only a delete removes the state of a recovery
    # checkpoint, so one this start saved and the service does not have counts as a failed delete.
    service = FakeTinkerService(failure=_no_delete_route)
    config = trainer_config(tmp_path, "train.max_training_steps=8", *PRUNE_EVERY_STEP)
    trainer, logged = make_pruning_trainer(monkeypatch, config, service)

    with caplog.at_level(logging.WARNING, logger=checkpoint_pruning.__name__):
        _train(trainer)

    assert _pruner(trainer).stopped
    assert _total(trainer, logged, "checkpoint/delete_failures") == 3
    assert len([path for path in service.delete_calls if "/weights/" in path]) == 3
    stopped = [entry.getMessage() for entry in caplog.records if "stopped deleting" in entry.getMessage()]
    assert len(stopped) == 1 and "the service may have no delete route" in stopped[0]
    assert service.deleted == [] and service.paths(1, "final") <= service.stored
    # Nothing is recorded as gone: the 404s may only mean the service has no delete route.
    assert _deletion_records(trainer) == []


def test_a_service_without_the_delete_route_leaves_its_checkpoints_to_a_later_start(tmp_path, monkeypatch):
    # The first start runs on a service without the delete route; the second, on one that has it, deletes what the
    # first left, its states and sampler weights alike.
    service = FakeTinkerService(failure=_no_delete_route)
    first, _logged = make_pruning_trainer(
        monkeypatch, trainer_config(tmp_path, "train.max_training_steps=6", *PRUNE_EVERY_STEP), service
    )
    _train(first)
    assert _pruner(first).stopped and service.deleted == []
    assert read_deleted_paths(first.run_log_path) == set()

    service.failure = None
    second, logged = make_pruning_trainer(
        monkeypatch, trainer_config(tmp_path, "train.max_training_steps=8", *PRUNE_EVERY_STEP), service
    )
    _train(second)

    assert _recovery_stored(service) == service.paths(2, "recovery-000007", "recovery-000008")
    assert service.paths(1, *(f"recovery-{step:06d}" for step in range(7))) <= set(service.deleted)
    assert _total(second, logged, "checkpoint/delete_failures") == 0


def test_a_resumed_start_without_the_delete_route_leaves_earlier_leftovers_to_a_later_start(tmp_path, monkeypatch):
    # A 404 for what an earlier start saved is believed only once the service has deleted something in this start.
    service = FakeTinkerService()
    first, _logged = make_pruning_trainer(
        monkeypatch, trainer_config(tmp_path, "train.max_training_steps=4", *PRUNE_EVERY_STEP), service
    )
    _train(first)
    left_by_first = _recovery_stored(service)
    assert left_by_first == service.paths(1, "recovery-000003", "recovery-000004")

    service.failure = _no_delete_route
    second, _logged = make_pruning_trainer(
        monkeypatch, trainer_config(tmp_path, "train.max_training_steps=8", *PRUNE_EVERY_STEP), service
    )
    _train(second)
    assert _pruner(second).stopped
    assert not set(read_deleted_paths(second.run_log_path)) & left_by_first  # asked, not believed

    service.failure = None
    third, _logged = make_pruning_trainer(
        monkeypatch, trainer_config(tmp_path, "train.max_training_steps=10", *PRUNE_EVERY_STEP), service
    )
    _train(third)

    assert _recovery_stored(service) == service.paths(3, "recovery-000009", "recovery-000010")


def test_once_the_service_has_deleted_a_state_it_no_longer_has_is_recorded_with_a_warning(
    tmp_path, monkeypatch, caplog
):
    # Someone deletes the state of this start's recovery-000001 by hand; by the time pruning gets to it the service
    # has deleted recovery-000000, so its 404 means the state is gone (or another API key saved it): a warning.
    gone = "tinker://run-1/weights/recovery-000001"
    service = FakeTinkerService()
    config = trainer_config(tmp_path, "train.max_training_steps=5", *PRUNE_EVERY_STEP)
    trainer, logged = make_pruning_trainer(monkeypatch, config, service)

    with caplog.at_level(logging.WARNING, logger=checkpoint_pruning.__name__):
        _train(trainer, Workflow(delay=0.02, on_start=lambda count: service.stored.discard(gone)))

    assert {entry["path"]: entry["result"] for entry in _deletion_records(trainer)}[gone] == "absent"
    assert service.delete_calls.count(gone) == 1
    assert _total(trainer, logged, "checkpoint/delete_failures") == 0 and not _pruner(trainer).stopped
    warnings = [entry.getMessage() for entry in caplog.records if gone in entry.getMessage()]
    assert len(warnings) == 1 and "saved under another API key, delete it yourself" in warnings[0]
    assert _recovery_stored(service) == service.paths(1, "recovery-000004", "recovery-000005")


def test_an_error_inside_the_client_delete_is_an_ordinary_failure(tmp_path, monkeypatch):
    # Only a client without a delete method stops pruning at once; an error raised by its delete is tried again.
    answers = iter([AttributeError("'NoneType' object has no attribute 'post'")])
    service = FakeTinkerService(failure=lambda path: next(answers, None))
    config = trainer_config(tmp_path, "train.max_training_steps=5", *PRUNE_EVERY_STEP)
    trainer, logged = make_pruning_trainer(monkeypatch, config, service)

    _train(trainer)

    assert not _pruner(trainer).stopped
    assert _total(trainer, logged, "checkpoint/delete_failures") == 1
    assert _recovery_stored(service) == service.paths(1, "recovery-000004", "recovery-000005")


def test_a_resumed_run_goes_on_pruning_where_it_left_off(tmp_path, monkeypatch):
    # The first start cannot delete the state of recovery-000001, and leaves it behind.
    left_behind = "tinker://run-1/weights/recovery-000001"
    service = FakeTinkerService(failure=lambda path: status_error(500, "busy") if path == left_behind else None)
    first, _logged = make_pruning_trainer(
        monkeypatch, trainer_config(tmp_path, "train.max_training_steps=4", *PRUNE_EVERY_STEP), service
    )
    _train(first)
    first_deleted = set(service.deleted)
    asked_by_first = len(service.delete_calls)
    assert first_deleted == {
        *paths("recovery-000000", "recovery-000002"),
        "tinker://run-1/sampler_weights/recovery-000001",
    }
    assert left_behind in service.stored

    # The second start resumes from final (step 4) with deletes working again, and trains to step 6.
    service.failure = None
    second, _logged = make_pruning_trainer(
        monkeypatch, trainer_config(tmp_path, "train.max_training_steps=6", *PRUNE_EVERY_STEP), service
    )
    _train(second)

    asked_by_second = service.delete_calls[asked_by_first:]
    assert not first_deleted & set(asked_by_second)  # what the first start deleted is not asked for again
    assert asked_by_second.count(left_behind) == 1  # what it could not delete is
    assert set(service.deleted) - first_deleted == {
        left_behind,
        *paths("recovery-000003", "recovery-000004"),
        *paths("recovery-000004", run=2),
    }
    assert _recovery_stored(service) == service.paths(2, "recovery-000005", "recovery-000006")
    assert service.paths(1, "final") | service.paths(2, "final") <= service.stored


def test_a_restart_after_a_crash_resumes_from_a_recovery_checkpoint_and_prunes_what_the_crash_left(
    tmp_path, monkeypatch
):
    # A first start saved recovery-000000 .. recovery-000004 of run-1, deleted the first two, failed to delete the
    # third, and crashed: no final checkpoint.  The second start resumes from recovery-000004.
    service = FakeTinkerService()
    config = trainer_config(tmp_path, "train.max_training_steps=7", *PRUNE_EVERY_STEP)
    second, logged = make_pruning_trainer(monkeypatch, config, service)
    run_dir = Path(second.run_log_path)
    records = [record(f"recovery-{version:06d}", version) for version in range(5)]
    (run_dir / "checkpoints.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in records))
    deleted = paths("recovery-000000", "recovery-000001")
    (run_dir / DELETED_CHECKPOINTS_NAME).write_text(
        "".join(json.dumps({"path": path, "result": "deleted"}) + "\n" for path in deleted) + '{"path": "tinker://'
    )  # a line cut short by the crash
    service.stored |= service.paths(1, "recovery-000002", "recovery-000003", "recovery-000004")
    hand_deleted = "tinker://run-1/weights/recovery-000002"
    service.stored.discard(hand_deleted)  # someone deleted it by hand
    service.runs = 1

    _train(second)

    assert second._model_info.llm.version == 7  # resumed at policy version 4
    assert not set(deleted) & set(service.delete_calls)
    left = set(paths("recovery-000002", "recovery-000003", "recovery-000004")) - {hand_deleted}
    assert sorted(service.deleted) == sorted([*left, *paths("recovery-000004", "recovery-000005", run=2)])
    assert _recovery_stored(service) == service.paths(2, "recovery-000006", "recovery-000007")
    # The checkpoint someone deleted is gone, which is no failure: an earlier start saved it.
    assert {entry["path"]: entry["result"] for entry in _deletion_records(second)}[hand_deleted] == "absent"
    lines = (run_dir / DELETED_CHECKPOINTS_NAME).read_text().splitlines()
    assert lines[len(deleted)] == '{"path": "tinker://' and len(_deletion_records(second)) == len(lines) - 1
    assert read_deleted_paths(run_dir) == set(deleted) | set(service.deleted) | {hand_deleted}
    assert _total(second, logged, "checkpoint/delete_failures") == 0


def test_another_runs_checkpoints_are_never_touched(tmp_path, monkeypatch):
    service = FakeTinkerService()
    other_dir, run_dir = tmp_path / "other", tmp_path / "pruned"
    other_dir.mkdir()
    run_dir.mkdir()
    other, _logged = make_pruning_trainer(
        monkeypatch,
        trainer_config(other_dir, "train.max_training_steps=3", "checkpoint={every: 0, recovery_every: 1}"),
        service,
    )
    _train(other)
    other_paths = set(service.stored)
    assert len(other_paths) == 10  # recovery-000000 .. recovery-000003 and final

    pruned, _logged = make_pruning_trainer(
        monkeypatch,
        trainer_config(
            run_dir,
            "train.max_training_steps=5",
            *PRUNE_EVERY_STEP,
            "checkpoint.load_checkpoint_path=tinker://run-1/weights/recovery-000001",
        ),
        service,
    )
    _train(pruned)

    assert other_paths <= service.stored
    assert not other_paths & set(service.delete_calls)
    assert {path.split("/")[2] for path in service.delete_calls} == {"run-2"}


def test_a_step_never_waits_for_a_delete(tmp_path, monkeypatch, caplog):
    service = FakeTinkerService(delete_gate=asyncio.Event())  # every delete hangs
    monkeypatch.setattr(checkpoint_pruning, "EXIT_WAIT_S", 0.2)
    config = trainer_config(tmp_path, "train.max_training_steps=6", *PRUNE_EVERY_STEP)
    trainer, logged = make_pruning_trainer(monkeypatch, config, service)

    with caplog.at_level(logging.WARNING, logger=checkpoint_pruning.__name__):
        _train(trainer)

    assert [step for step, _stats in logged] == [1, 2, 3, 4, 5, 6]
    assert service.paths(1, "final") <= service.stored
    assert service.delete_calls == ["tinker://run-1/weights/recovery-000000"]  # the one delete in flight
    assert service.deleted == [] and _deletion_records(trainer) == []  # the next start asks for it again
    assert any("Stopped deleting recovery checkpoints after waiting" in entry.getMessage() for entry in caplog.records)


def test_step_metrics_count_since_the_step_before(tmp_path):
    service = FakeTinkerService(
        tolerant=True, failure=lambda path: status_error(500, "x") if "sampler" in path else None
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lines = [record(f"recovery-{version:06d}", version) for version in range(4)]
    (run_dir / "checkpoints.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    pruner = RecoveryCheckpointPruner(
        run_dir=str(run_dir), keep=2, max_staleness=0, service_client_factory=service.service_client
    )

    async def prune_once():
        pruner.recovery_saved({"state_path": lines[-1]["state_path"]}, 3)
        await pruner.aclose()
        return pruner.step_metrics(), pruner.step_metrics()

    first, second = asyncio.run(prune_once())

    assert first == {
        "checkpoint/deleted": 2.0,
        "checkpoint/delete_failures": 2.0,
        "checkpoint/pruning": 1.0,
        "checkpoint/recovery_kept": 2.0,
    }
    assert second == {**first, "checkpoint/deleted": 0.0, "checkpoint/delete_failures": 0.0}
