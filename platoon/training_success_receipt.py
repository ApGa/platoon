"""Fail-closed evidence for the ABCI one-update training smoke.

The ordinary recovery marker is intentionally not sufficient evidence of a
successful optimizer update: Platoon advances logical versions and may write a
checkpoint after an empty or non-finite update.  This module binds direct
trainer control-flow evidence to the durable recovery tree and to uniquely
nonced summary fields observed in the remote W&B run.

The trainer-facing entry point imports W&B lazily. Local verification uses only
the Python standard library so PBS monitors can use ABCI's host Python.
"""

from __future__ import annotations

import argparse
import atexit
import datetime as dt
import getpass
import hashlib
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 2
GATE_NAME = "abci-openreward-one-update-success"
RECEIPT_ENV = "PLATOON_TRAINING_SUCCESS_RECEIPT"
REMOTE_TIMEOUT_ENV = "PLATOON_TRAINING_SUCCESS_RECEIPT_REMOTE_TIMEOUT"
FINALIZE_ENV = "PLATOON_TRAINING_SUCCESS_RECEIPT_FINALIZE"
MIN_GLOBAL_STEP_ENV = "PLATOON_TRAINING_SUCCESS_RECEIPT_MIN_GLOBAL_STEP"
METRIC_PREFIX = "migration_gate5"
CHECKPOINT_SAMPLE_BYTES = 1024 * 1024
RECOVER_INFO_FILES = (
    "step_info.json",
    "saver_info.json",
    "evaluator_info.json",
    "stats_logger_info.json",
    "checkpoint_info.json",
    "dataloader_info.pkl",
)

_REGISTERED_RECEIPTS: set[Path] = set()
_PUBLISHED_RECEIPTS: set[Path] = set()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _normalized_path(value: str | os.PathLike[str]) -> Path:
    """Normalize a configured path lexically, without filesystem I/O."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def candidate_publication_enabled(global_step: int = 0) -> bool:
    """Return whether trainer evidence arguments should be evaluated."""

    configured_receipt = os.environ.get(RECEIPT_ENV)
    if not configured_receipt:
        return False
    raw_minimum = os.environ.get(MIN_GLOBAL_STEP_ENV, "0")
    if not raw_minimum.isdigit():
        raise RuntimeError(f"{MIN_GLOBAL_STEP_ENV} must be a nonnegative integer")
    minimum = int(raw_minimum)
    if global_step < minimum:
        return False
    return _normalized_path(configured_receipt) not in _PUBLISHED_RECEIPTS


def _finalization_enabled() -> bool:
    value = os.environ.get(FINALIZE_ENV, "1").strip().lower()
    if value in {"1", "true"}:
        return True
    if value in {"0", "false"}:
        return False
    raise RuntimeError(f"{FINALIZE_ENV} must be one of 0, 1, false, or true")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sampled_sha256(path: Path, *, sample_bytes: int = CHECKPOINT_SAMPLE_BYTES) -> str:
    """Hash a bounded payload sample plus its size.

    A 35B distributed checkpoint is hundreds of GiB.  Re-reading every shard
    immediately after the synchronous save would turn the smoke proof into a
    second checkpoint-sized I/O pass.  Size/mtime plus a first-and-last MiB
    digest detects incomplete or subsequently changed shards without doing so.
    Small files (including DCP's ``.metadata``) are hashed in full.
    """

    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(b"platoon-checkpoint-sample-v1\0")
    digest.update(str(size).encode())
    digest.update(b"\0")
    with path.open("rb") as stream:
        if size <= 2 * sample_bytes:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        else:
            digest.update(stream.read(sample_bytes))
            stream.seek(size - sample_bytes)
            digest.update(stream.read(sample_bytes))
    return digest.hexdigest()


def _atomic_private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and create-only.  Unlike replace(),
        # it cannot silently bless a stale artifact from an earlier job.
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(f"refusing to overwrite existing evidence: {path}") from error
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _read_private_json(path: Path) -> dict[str, Any]:
    stat = path.stat()
    if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
        raise ValueError(f"evidence file is not private and owned by uid {os.getuid()}: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"evidence file is not a JSON object: {path}")
    return value


def _checkpoint_root(fileroot: Path, experiment: str, trial: str) -> Path:
    return fileroot / "checkpoints" / getpass.getuser() / experiment / trial


def _nonnegative_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if result < minimum or numeric != float(result):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _recover_evidence(
    fileroot: Path,
    experiment: str,
    trial: str,
    global_step: int,
) -> dict[str, Any]:
    fileroot = fileroot.expanduser().resolve()
    root = _checkpoint_root(fileroot, experiment, trial)
    recover_info = root / "recover_info"
    checkpoint = root / "default" / "recover_checkpoint"
    if not recover_info.is_dir():
        raise ValueError(f"recovery metadata directory is missing: {recover_info}")
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise ValueError(f"recovery checkpoint directory is missing or unsafe: {checkpoint}")

    metadata: dict[str, dict[str, Any]] = {}
    for name in RECOVER_INFO_FILES:
        path = recover_info / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise ValueError(f"recovery metadata is missing or empty: {path}")
        metadata[name] = {
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }

    try:
        step = json.loads((recover_info / "step_info.json").read_text())
        recorded_step = _nonnegative_int(step["global_step"], "recover global_step")
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("recovery step_info.json is malformed") from error
    if recorded_step != global_step:
        raise ValueError(f"recovery checkpoint step {recorded_step} does not match successful step {global_step}")

    checkpoint_files: list[Path] = []
    checkpoint_manifest: dict[str, dict[str, Any]] = {}
    for item in checkpoint.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"recovery checkpoint contains a symlink: {item}")
        if item.is_file():
            if item.stat().st_size <= 0:
                raise ValueError(f"recovery checkpoint contains an empty file: {item}")
            checkpoint_files.append(item)
            relative = str(item.relative_to(checkpoint))
            item_stat = item.stat()
            checkpoint_manifest[relative] = {
                "size": item_stat.st_size,
                "mtime_ns": item_stat.st_mtime_ns,
                "sample_sha256": _sampled_sha256(item),
            }
    if not checkpoint_files:
        raise ValueError(f"recovery checkpoint contains no payload files: {checkpoint}")

    return {
        "fileroot": str(fileroot),
        "checkpoint_root": str(root.resolve()),
        "recover_info_path": str(recover_info.resolve()),
        "recover_info": metadata,
        "recover_checkpoint_path": str(checkpoint.resolve()),
        "recover_checkpoint_file_count": len(checkpoint_files),
        "recover_checkpoint_total_bytes": sum(item.stat().st_size for item in checkpoint_files),
        "recover_checkpoint_manifest": checkpoint_manifest,
        "global_step": recorded_step,
    }


def _candidate_path(receipt: Path) -> Path:
    if receipt.suffix:
        return receipt.with_name(f"{receipt.stem}.candidate{receipt.suffix}")
    return receipt.with_name(f"{receipt.name}.candidate.json")


def _error_path(receipt: Path) -> Path:
    if receipt.suffix:
        return receipt.with_name(f"{receipt.stem}.error{receipt.suffix}")
    return receipt.with_name(f"{receipt.name}.error.json")


def _wandb_identity(run: Any) -> dict[str, str]:
    identity = {
        "entity": str(getattr(run, "entity", "") or ""),
        "project": str(getattr(run, "project", "") or ""),
        "run_id": str(getattr(run, "id", "") or ""),
        "url": str(getattr(run, "url", "") or ""),
    }
    if not all(identity[name] for name in ("entity", "project", "run_id")):
        raise RuntimeError("W&B online run has no complete entity/project/run identity")
    identity["path"] = "/".join((identity["entity"], identity["project"], identity["run_id"]))
    return identity


def maybe_publish_training_success_candidate(
    *,
    fileroot: str | Path,
    experiment: str,
    trial: str,
    config_path: str | Path,
    pbs_job_id: str,
    global_step: int,
    optimizer_update_successful: bool,
    optimizer_batch_datums: int | float,
    optimizer_action_tokens: int | float,
    weight_sync_succeeded: bool,
    expected_version: int,
    actor_version: int,
    rollout_version: int,
    stats_logger: Any,
    wandb_mode: str,
    training_sif_path: str | Path,
    training_sif_sha256: str,
    not_before_epoch: float | str,
    receipt_path: str | Path | None = None,
    register_finalizer: bool = True,
) -> Path | None:
    """Publish a candidate only after the real update/checkpoint/stats path.

    The caller must invoke this immediately after ``_export_and_commit_stats``.
    When the receipt environment is absent, production training is unchanged.
    """

    configured_receipt = receipt_path or os.environ.get(RECEIPT_ENV)
    if not configured_receipt:
        return None

    receipt = _normalized_path(configured_receipt)
    candidate = _candidate_path(receipt)
    if receipt in _PUBLISHED_RECEIPTS:
        if not candidate.is_file():
            raise RuntimeError(f"published training success candidate disappeared: {candidate}")
        return candidate
    stale_paths = (receipt, candidate, _error_path(receipt))
    if any(os.path.lexists(path) for path in stale_paths):
        stale = next(path for path in stale_paths if os.path.lexists(path))
        raise RuntimeError(f"refusing stale training success evidence: {stale}")
    finalize_on_exit = _finalization_enabled() if register_finalizer else False
    global_step = _nonnegative_int(global_step, "global_step")
    expected_version = _nonnegative_int(expected_version, "expected_version", minimum=1)
    actor_version = _nonnegative_int(actor_version, "actor_version", minimum=1)
    rollout_version = _nonnegative_int(rollout_version, "rollout_version", minimum=1)
    datums = _nonnegative_int(optimizer_batch_datums, "optimizer_batch_datums", minimum=1)
    action_tokens = _nonnegative_int(
        optimizer_action_tokens,
        "optimizer_action_tokens",
        minimum=1,
    )
    if optimizer_update_successful is not True:
        raise RuntimeError("training success receipt requires a successful finite optimizer update")
    if weight_sync_succeeded is not True:
        raise RuntimeError("training success receipt requires the real weight-sync branch")
    if expected_version != global_step + 1:
        raise RuntimeError("successful actor version must be global_step + 1")
    if actor_version != expected_version or rollout_version != expected_version:
        raise RuntimeError(
            "actor and rollout versions do not attest the completed weight sync: "
            f"expected={expected_version}, actor={actor_version}, rollout={rollout_version}"
        )
    if wandb_mode != "online":
        raise RuntimeError(f"training success receipt requires W&B online mode, got {wandb_mode!r}")
    if not pbs_job_id:
        raise RuntimeError("training success receipt requires PBS_JOBID")
    try:
        not_before_epoch = float(not_before_epoch)
    except (TypeError, ValueError) as error:
        raise RuntimeError("training success receipt requires a finite launch epoch") from error
    if not_before_epoch <= 0 or not_before_epoch > time.time() + 300:
        raise RuntimeError("training success receipt launch epoch is invalid")

    config = Path(config_path).expanduser().resolve()
    training_sif = Path(training_sif_path).expanduser().resolve()
    if not config.is_file():
        raise RuntimeError(f"training config is missing: {config}")
    if not training_sif.is_file():
        raise RuntimeError(f"training SIF is missing: {training_sif}")
    if len(training_sif_sha256) != 64 or any(character not in "0123456789abcdef" for character in training_sif_sha256):
        raise RuntimeError("training SIF digest is not a lowercase SHA-256 value")

    stats_state = stats_logger.state_dict()
    committed_step = _nonnegative_int(
        stats_state.get("last_commit_step"),
        "stats_logger.last_commit_step",
    )
    if committed_step < global_step:
        raise RuntimeError(f"stats logger has not committed successful step {global_step}: {committed_step}")

    import wandb  # Imported only in the pinned host-controller environment.

    run = wandb.run
    if run is None:
        raise RuntimeError("W&B online run is not initialized")
    wandb_identity = _wandb_identity(run)
    nonce = secrets.randbelow(2**48 - 1) + 1
    remote_metrics: dict[str, int] = {
        f"{METRIC_PREFIX}/nonce": nonce,
        f"{METRIC_PREFIX}/global_step": global_step,
        f"{METRIC_PREFIX}/optimizer_update_succeeded": 1,
        f"{METRIC_PREFIX}/optimizer_batch_datums": datums,
        f"{METRIC_PREFIX}/optimizer_action_tokens": action_tokens,
        f"{METRIC_PREFIX}/weight_sync_succeeded": 1,
        f"{METRIC_PREFIX}/actor_version": actor_version,
        f"{METRIC_PREFIX}/rollout_version": rollout_version,
        f"{METRIC_PREFIX}/recover_checkpoint_durable": 1,
    }

    recover = _recover_evidence(Path(fileroot), experiment, trial, global_step)
    recover_paths = [
        Path(recover["recover_checkpoint_path"]),
        *(Path(recover["recover_info_path"]) / name for name in RECOVER_INFO_FILES),
    ]
    if any(path.stat().st_mtime < not_before_epoch for path in recover_paths):
        raise RuntimeError("training success receipt refuses recovery evidence older than this PBS launch")
    # Summary fields do not advance W&B's explicit training-history step.  The
    # nonce prevents an older job/run summary from satisfying this candidate.
    # The atexit finalizer publishes the final receipt only after the finished
    # remote run exposes every exact field.
    run.summary.update(remote_metrics)

    record = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "kind": "candidate",
        "created_at_utc": _utc_now(),
        "not_before_epoch": not_before_epoch,
        "hostname": os.uname().nodename,
        "pbs_job_id": pbs_job_id,
        "experiment_name": experiment,
        "trial_name": trial,
        "config_path": str(config),
        "config_sha256": _sha256(config),
        "global_step": global_step,
        "optimizer_update_successful": True,
        "optimizer_batch_datums": datums,
        "optimizer_action_tokens": action_tokens,
        "weight_sync_succeeded": True,
        "expected_version": expected_version,
        "actor_version": actor_version,
        "rollout_version": rollout_version,
        "stats_logger_last_commit_step": committed_step,
        "training_sif_path": str(training_sif),
        "training_sif_sha256": training_sif_sha256,
        "recover": recover,
        "wandb": {
            **wandb_identity,
            "mode": wandb_mode,
            "summary_metrics": remote_metrics,
        },
    }
    _atomic_private_json(candidate, record)
    _PUBLISHED_RECEIPTS.add(receipt)
    print(f"training_success_candidate={candidate}", flush=True)

    if finalize_on_exit and receipt not in _REGISTERED_RECEIPTS:
        _REGISTERED_RECEIPTS.add(receipt)
        timeout = _nonnegative_int(
            os.environ.get(REMOTE_TIMEOUT_ENV, "300"),
            REMOTE_TIMEOUT_ENV,
            minimum=1,
        )
        atexit.register(_finalize_at_exit, candidate, receipt, float(timeout))
    return candidate


def _metrics_match(values: dict[str, Any], expected: dict[str, Any]) -> bool:
    try:
        for name, expected_value in expected.items():
            if float(values[name]) != float(expected_value):
                return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return True


def _observe_finished_wandb_run(candidate: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    return _observe_wandb_run(candidate, timeout_seconds, allowed_states={"finished"})


def _observe_wandb_run(
    candidate: dict[str, Any],
    timeout_seconds: float,
    *,
    allowed_states: set[str],
) -> dict[str, Any]:
    import wandb  # Imported only in the pinned host-controller environment.

    wandb_record = candidate["wandb"]
    path = str(wandb_record["path"])
    metrics = dict(wandb_record["summary_metrics"])
    deadline = time.monotonic() + timeout_seconds
    last_error = "remote run was not yet visible"
    while time.monotonic() < deadline:
        try:
            api = wandb.Api(timeout=min(max(int(timeout_seconds), 10), 60))
            run = api.run(path)
            state = str(getattr(run, "state", "") or "").lower()
            matched = _metrics_match(dict(getattr(run, "summary", {}) or {}), metrics)
            if state in allowed_states and matched:
                return {
                    "path": path,
                    "url": str(getattr(run, "url", "") or wandb_record.get("url", "")),
                    "state": state,
                    "summary_metrics": metrics,
                    "observed_at_utc": _utc_now(),
                }
            last_error = f"state={state!r}, unique_summary_metrics_visible={matched}"
        except Exception as error:  # W&B exposes several transport exception types.
            last_error = f"{type(error).__name__}: {error}"
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(10.0, remaining))
    raise RuntimeError(f"W&B remote attestation timed out for {path}: {last_error}")


def _validate_candidate(candidate_path: Path) -> dict[str, Any]:
    candidate = _read_private_json(candidate_path)
    if candidate.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("training success candidate schema mismatch")
    if candidate.get("gate") != GATE_NAME or candidate.get("kind") != "candidate":
        raise ValueError("training success candidate gate/kind mismatch")
    if candidate.get("optimizer_update_successful") is not True:
        raise ValueError("candidate does not attest a successful optimizer update")
    if candidate.get("weight_sync_succeeded") is not True:
        raise ValueError("candidate does not attest the real weight-sync branch")
    not_before_epoch = float(candidate.get("not_before_epoch", 0))
    if not_before_epoch <= 0:
        raise ValueError("candidate launch epoch is invalid")
    global_step = _nonnegative_int(candidate.get("global_step"), "candidate global_step")
    expected_version = global_step + 1
    if any(
        _nonnegative_int(candidate.get(name), name, minimum=1) != expected_version
        for name in ("expected_version", "actor_version", "rollout_version")
    ):
        raise ValueError("candidate actor/rollout version evidence is inconsistent")
    _nonnegative_int(candidate.get("optimizer_batch_datums"), "optimizer_batch_datums", minimum=1)
    _nonnegative_int(candidate.get("optimizer_action_tokens"), "optimizer_action_tokens", minimum=1)

    config = Path(str(candidate["config_path"]))
    if not config.is_file() or _sha256(config) != candidate.get("config_sha256"):
        raise ValueError("candidate config is absent or changed")
    current_recover = _recover_evidence(
        Path(str(candidate["recover"]["fileroot"])),
        str(candidate["experiment_name"]),
        str(candidate["trial_name"]),
        global_step,
    )
    current_paths = [
        Path(current_recover["recover_checkpoint_path"]),
        *(Path(current_recover["recover_info_path"]) / name for name in RECOVER_INFO_FILES),
    ]
    if any(path.stat().st_mtime < not_before_epoch for path in current_paths):
        raise ValueError("candidate recovery evidence predates its PBS launch")
    for key in (
        "fileroot",
        "checkpoint_root",
        "recover_info",
        "recover_checkpoint_path",
        "recover_checkpoint_file_count",
        "recover_checkpoint_total_bytes",
        "recover_checkpoint_manifest",
        "global_step",
    ):
        if current_recover[key] != candidate["recover"][key]:
            raise ValueError(f"candidate recovery evidence changed: {key}")
    return candidate


def finalize_candidate(
    candidate_path: str | Path,
    receipt_path: str | Path,
    *,
    timeout_seconds: float = 300,
    remote_observer: Callable[[dict[str, Any], float], dict[str, Any]] | None = None,
) -> Path:
    """Observe the finished W&B run and atomically publish the final receipt."""

    candidate_path = Path(candidate_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    if os.path.lexists(receipt_path):
        raise RuntimeError(f"refusing stale training success receipt: {receipt_path}")
    candidate = _validate_candidate(candidate_path)
    observer = remote_observer or _observe_finished_wandb_run
    remote = observer(candidate, timeout_seconds)
    if str(remote.get("state", "")).lower() != "finished":
        raise RuntimeError("remote W&B run is not finished")
    wandb_record = candidate["wandb"]
    if remote.get("path") != wandb_record.get("path"):
        raise RuntimeError("remote W&B run path differs from the candidate")
    if remote.get("summary_metrics") != wandb_record.get("summary_metrics"):
        raise RuntimeError("remote W&B unique summary proof differs from the candidate")

    final = {
        "schema_version": SCHEMA_VERSION,
        "gate": GATE_NAME,
        "kind": "final",
        "created_at_utc": _utc_now(),
        "candidate_path": str(candidate_path),
        "candidate_sha256": _sha256(candidate_path),
        "pbs_job_id": candidate["pbs_job_id"],
        "experiment_name": candidate["experiment_name"],
        "trial_name": candidate["trial_name"],
        "config_path": candidate["config_path"],
        "config_sha256": candidate["config_sha256"],
        "not_before_epoch": candidate["not_before_epoch"],
        "global_step": candidate["global_step"],
        "optimizer_update_successful": True,
        "optimizer_batch_datums": candidate["optimizer_batch_datums"],
        "optimizer_action_tokens": candidate["optimizer_action_tokens"],
        "weight_sync_succeeded": True,
        "actor_version": candidate["actor_version"],
        "rollout_version": candidate["rollout_version"],
        "recover": candidate["recover"],
        "wandb_remote": remote,
        "training_sif_path": candidate["training_sif_path"],
        "training_sif_sha256": candidate["training_sif_sha256"],
    }
    _atomic_private_json(receipt_path, final)
    print(f"training_success_receipt={receipt_path}", flush=True)
    return receipt_path


def verify_candidate(
    candidate_path: str | Path,
    *,
    pbs_job_id: str | None = None,
    experiment: str | None = None,
    trial: str | None = None,
    global_step: int | None = None,
    require_remote_wandb: bool = False,
    remote_timeout_seconds: float = 60,
) -> dict[str, Any]:
    """Verify first-update evidence while a long-running job is still active."""

    candidate = _validate_candidate(Path(candidate_path).resolve())
    expectations = {
        "pbs_job_id": pbs_job_id,
        "experiment_name": experiment,
        "trial_name": trial,
        "global_step": global_step,
    }
    for name, expected in expectations.items():
        if expected is not None and candidate.get(name) != expected:
            raise ValueError(f"training success candidate {name}={candidate.get(name)!r}, expected {expected!r}")
    if require_remote_wandb:
        remote = _observe_wandb_run(
            candidate,
            remote_timeout_seconds,
            allowed_states={"running", "finished"},
        )
        candidate = {**candidate, "wandb_remote_observed": remote}
    return candidate


def _finalize_at_exit(candidate: Path, receipt: Path, timeout_seconds: float) -> None:
    try:
        finalize_candidate(candidate, receipt, timeout_seconds=timeout_seconds)
    except Exception as error:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "gate": GATE_NAME,
            "kind": "finalization_error",
            "created_at_utc": _utc_now(),
            "candidate_path": str(candidate),
            "receipt_path": str(receipt),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        try:
            _atomic_private_json(_error_path(receipt), failure)
        except Exception:
            pass
        print(f"training success receipt finalization failed: {error}", file=sys.stderr, flush=True)


def verify_receipt(
    receipt_path: str | Path,
    *,
    pbs_job_id: str | None = None,
    experiment: str | None = None,
    trial: str | None = None,
    global_step: int | None = None,
) -> dict[str, Any]:
    """Verify the final receipt and all still-current local evidence."""

    receipt_path = Path(receipt_path).resolve()
    receipt = _read_private_json(receipt_path)
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("training success receipt schema mismatch")
    if receipt.get("gate") != GATE_NAME or receipt.get("kind") != "final":
        raise ValueError("training success receipt gate/kind mismatch")
    expectations = {
        "pbs_job_id": pbs_job_id,
        "experiment_name": experiment,
        "trial_name": trial,
        "global_step": global_step,
    }
    for name, expected in expectations.items():
        if expected is not None and receipt.get(name) != expected:
            raise ValueError(f"training success receipt {name}={receipt.get(name)!r}, expected {expected!r}")
    if receipt.get("optimizer_update_successful") is not True:
        raise ValueError("receipt does not attest a successful optimizer update")
    if receipt.get("weight_sync_succeeded") is not True:
        raise ValueError("receipt does not attest the real weight-sync branch")
    if str(receipt.get("wandb_remote", {}).get("state", "")).lower() != "finished":
        raise ValueError("receipt does not attest a finished remote W&B run")

    candidate_path = Path(str(receipt["candidate_path"]))
    if not candidate_path.is_file() or _sha256(candidate_path) != receipt.get("candidate_sha256"):
        raise ValueError("receipt candidate is absent or changed")
    candidate = _validate_candidate(candidate_path)
    if candidate["pbs_job_id"] != receipt["pbs_job_id"]:
        raise ValueError("receipt and candidate PBS job IDs differ")
    if receipt["config_sha256"] != candidate["config_sha256"]:
        raise ValueError("receipt and candidate config digests differ")
    if receipt["recover"] != candidate["recover"]:
        raise ValueError("receipt and candidate recovery evidence differ")
    remote = receipt["wandb_remote"]
    if remote.get("path") != candidate["wandb"].get("path"):
        raise ValueError("receipt W&B path differs from the candidate")
    if remote.get("summary_metrics") != candidate["wandb"].get("summary_metrics"):
        raise ValueError("receipt W&B summary proof differs from the candidate")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--receipt", type=Path, required=True)
    verify_parser.add_argument("--pbs-job-id")
    verify_parser.add_argument("--experiment")
    verify_parser.add_argument("--trial")
    verify_parser.add_argument("--global-step", type=int)
    candidate_parser = subparsers.add_parser("verify-candidate")
    candidate_parser.add_argument("--candidate", type=Path, required=True)
    candidate_parser.add_argument("--pbs-job-id")
    candidate_parser.add_argument("--experiment")
    candidate_parser.add_argument("--trial")
    candidate_parser.add_argument("--global-step", type=int)
    candidate_parser.add_argument("--require-remote-wandb", action="store_true")
    candidate_parser.add_argument("--remote-timeout-seconds", type=float, default=60)
    finalize_parser = subparsers.add_parser("finalize-candidate")
    finalize_parser.add_argument("--candidate", type=Path, required=True)
    finalize_parser.add_argument("--receipt", type=Path, required=True)
    finalize_parser.add_argument("--remote-timeout-seconds", type=float, default=300)
    args = parser.parse_args(argv)

    if args.command == "finalize-candidate":
        try:
            result = finalize_candidate(
                args.candidate,
                args.receipt,
                timeout_seconds=args.remote_timeout_seconds,
            )
        except (OSError, KeyError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"training success candidate finalization failed: {error}", file=sys.stderr)
            return 1
        print(f"finalized_training_success_candidate={result}", flush=True)
        return 0

    if args.command == "verify-candidate":
        try:
            candidate = verify_candidate(
                args.candidate,
                pbs_job_id=args.pbs_job_id,
                experiment=args.experiment,
                trial=args.trial,
                global_step=args.global_step,
                require_remote_wandb=args.require_remote_wandb,
                remote_timeout_seconds=args.remote_timeout_seconds,
            )
        except (
            OSError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            print(f"training success candidate verification failed: {error}", file=sys.stderr)
            return 1
        print(
            f"verified_training_success_candidate={args.candidate.resolve()} "
            f"global_step={candidate['global_step']} "
            f"actor_version={candidate['actor_version']} "
            f"rollout_version={candidate['rollout_version']}",
            flush=True,
        )
        return 0

    if args.command == "verify":
        try:
            receipt = verify_receipt(
                args.receipt,
                pbs_job_id=args.pbs_job_id,
                experiment=args.experiment,
                trial=args.trial,
                global_step=args.global_step,
            )
        except (
            OSError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            print(f"training success receipt verification failed: {error}", file=sys.stderr)
            return 1
        print(
            f"verified_training_success_receipt={args.receipt.resolve()} "
            f"global_step={receipt['global_step']} "
            f"actor_version={receipt['actor_version']} "
            f"wandb={receipt['wandb_remote']['path']}"
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
