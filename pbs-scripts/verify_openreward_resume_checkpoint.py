#!/usr/bin/env python3
"""Authorize one exact AReaL recovery checkpoint for an immutable resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

RECOVER_INFO_FILES = {
    "checkpoint_info.json",
    "dataloader_info.pkl",
    "evaluator_info.json",
    "saver_info.json",
    "stats_logger_info.json",
    "step_info.json",
}
CHECKPOINT_METADATA = {
    "sharded_backend": "torch_dist",
    "sharded_backend_version": 1,
    "common_backend": "torch",
    "common_backend_version": 1,
}
CHECKPOINT_SAMPLE_BYTES = 1024 * 1024
PREUPDATE_BINDING_ID = "qwen38-16-v18-job2155389"
PREUPDATE_EXPERIMENT = "abci-or-ta-swe16-q38-v18"
PREUPDATE_TRIAL = "abci-ta-swe16-nr-q38d-c12-v18-t0"
PREUPDATE_SOURCE_JOB = "2155389.pbs1"
QWEN38_SUCCESSOR_BINDING_ID = "qwen38-32-v20-post-step6"
QWEN38_SUCCESSOR_EXPERIMENT = "abci-or-ta-swe32-q38-v16"
QWEN38_SUCCESSOR_TRIAL = "abci-ta-swe32-rec-q38d-c24-v16-t0"
QWEN38_SUCCESSOR_SOURCE_CONFIG = Path(
    "plugins/openreward/platoon/openreward/configs/areal/"
    "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_"
    "qwen38_dense_c24_ta20_v16.yaml"
)
QWEN38_SUCCESSOR_SOURCE_CONFIG_SHA256 = (
    "a3f8c04b2a3e59edd708f4f3a39eb38e1a22fa72b63a1f962afb54ac80183ffa"
)
QWEN38_SUCCESSOR_CURRENT_CONFIG_SHA256 = (
    "f3ffb459d1ed6e9427124e8e7d6daafebb2709f965f86c5629a99e508449fe5c"
)
QWEN38_SUCCESSOR_GLOBAL_STEP = 6
QWEN38_SUCCESSOR_RECOVER_STATS_COMMIT_STEP = 4
QWEN38_SUCCESSOR_ANCESTOR_SHA256 = (
    "b432e3a12ff78921e7aac6cff3914f1c4c5aeffd9f08e3c9918ad5daa7714a22"
)
QWEN38_SUCCESSOR_RECEIPT = (
    "abci-ta-swe32-rec-q38d-c24-v16-t0."
    "resume-post-step6-v8r18.success.json"
)
QWEN38_SUCCESSOR_ANCESTOR_RECEIPT = (
    "abci-ta-swe32-rec-q38d-c24-v16-t0.success.candidate.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _size_manifest_sha256(paths: list[Path]) -> str:
    """Hash sorted entry-name/NUL/decimal-size/newline records."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _content_manifest_sha256(paths: list[Path]) -> str:
    """Hash sorted entry-name/NUL/content-SHA256/newline records."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sampled_sha256(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(b"openreward-resume-checkpoint-sample-v2\0")
    digest.update(str(size).encode("ascii"))
    digest.update(b"\0")
    with path.open("rb") as stream:
        if size <= 3 * CHECKPOINT_SAMPLE_BYTES:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        else:
            digest.update(stream.read(CHECKPOINT_SAMPLE_BYTES))
            stream.seek((size - CHECKPOINT_SAMPLE_BYTES) // 2)
            digest.update(stream.read(CHECKPOINT_SAMPLE_BYTES))
            stream.seek(size - CHECKPOINT_SAMPLE_BYTES)
            digest.update(stream.read(CHECKPOINT_SAMPLE_BYTES))
    return digest.hexdigest()


def _candidate_sampled_sha256(path: Path) -> str:
    """Reproduce training-success receipt checkpoint sampling exactly."""

    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(b"platoon-checkpoint-sample-v1\0")
    digest.update(str(size).encode())
    digest.update(b"\0")
    with path.open("rb") as stream:
        if size <= 2 * CHECKPOINT_SAMPLE_BYTES:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        else:
            digest.update(stream.read(CHECKPOINT_SAMPLE_BYTES))
            stream.seek(size - CHECKPOINT_SAMPLE_BYTES)
            digest.update(stream.read(CHECKPOINT_SAMPLE_BYTES))
    return digest.hexdigest()


def _sample_manifest_sha256(paths: list[Path]) -> str:
    """Hash sorted entry-name/NUL/size/NUL/bounded-content-digest records."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sampled_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _current_candidate_recover_evidence(
    *,
    checkpoint_root: Path,
    checkpoint_files: list[Path],
    recover_info_paths: list[Path],
    expected_experiment: str,
    expected_trial: str,
    expected_global_step: int,
) -> dict[str, Any]:
    """Reconstruct the full same-step receipt evidence from current files."""

    if (
        checkpoint_root.name != expected_trial
        or checkpoint_root.parent.name != expected_experiment
        or checkpoint_root.parent.parent.parent.name != "checkpoints"
    ):
        raise RuntimeError("same-step checkpoint root has an invalid lineage shape")
    fileroot = checkpoint_root.parent.parent.parent.parent
    checkpoint = checkpoint_root / "default" / "recover_checkpoint"
    recover_info = checkpoint_root / "recover_info"
    return {
        "fileroot": str(fileroot),
        "checkpoint_root": str(checkpoint_root),
        "recover_info_path": str(recover_info),
        "recover_info": {
            path.name: {"size": path.stat().st_size, "sha256": _sha256(path)}
            for path in recover_info_paths
        },
        "recover_checkpoint_path": str(checkpoint),
        "recover_checkpoint_file_count": len(checkpoint_files),
        "recover_checkpoint_total_bytes": sum(path.stat().st_size for path in checkpoint_files),
        "recover_checkpoint_manifest": {
            path.name: {
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sample_sha256": _candidate_sampled_sha256(path),
            }
            for path in checkpoint_files
        },
        "global_step": expected_global_step,
    }


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def _require_canonical_directory(path: Path, *, label: str) -> os.stat_result:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise RuntimeError(f"{label} is not an exact canonical directory: {path}")
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.getuid():
        raise RuntimeError(f"{label} identity/owner mismatch: {path}")
    return value


def _require_owned_regular(path: Path, *, label: str) -> os.stat_result:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise RuntimeError(f"{label} is not an exact canonical file: {path}")
    value = path.lstat()
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.getuid()
        or value.st_nlink != 1
        or value.st_size <= 0
    ):
        raise RuntimeError(f"{label} identity/owner/size mismatch: {path}")
    return value


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    _require_owned_regular(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain one JSON object: {path}")
    return value


def _require_exact_children(path: Path, expected: set[str], *, label: str) -> None:
    actual = {entry.name for entry in path.iterdir()}
    if actual != expected:
        raise RuntimeError(
            f"{label} child set mismatch: missing={sorted(expected - actual)!r} "
            f"unexpected={sorted(actual - expected)!r}"
        )


def _checkpoint_snapshot(path: Path) -> tuple[tuple[str, int, int, int, int], ...]:
    return tuple(
        sorted(
            (
                entry.name,
                entry.stat().st_size,
                entry.stat().st_mtime_ns,
                entry.stat().st_dev,
                entry.stat().st_ino,
            )
            for entry in path.iterdir()
        )
    )


def _verify_source_release(
    *,
    source_release_root: Path,
    source_release_ref: str,
    source_config_relative: Path,
    current_config: Path,
    expected_source_config_sha256: str,
    expected_config_sha256: str,
    expected_experiment: str,
    expected_trial: str,
    model_repository: str,
    model_revision: str,
    model_snapshot: Path,
) -> Path:
    _require_canonical_directory(source_release_root, label="source release root")
    source_head = _git(source_release_root, "rev-parse", "--verify", "HEAD")
    source_commit = _git(
        source_release_root,
        "rev-parse",
        "--verify",
        f"{source_release_ref}^{{commit}}",
    )
    if source_head != source_commit:
        raise RuntimeError("source release HEAD does not match its pinned tag")
    if _git(source_release_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("source release checkout is dirty")

    if source_config_relative.is_absolute() or ".." in source_config_relative.parts:
        raise RuntimeError("source config relative path is unsafe")
    source_config = source_release_root / source_config_relative
    _require_owned_regular(source_config, label="source config")
    _require_owned_regular(current_config, label="current resume config")
    if _sha256(source_config) != expected_source_config_sha256:
        raise RuntimeError("source resume config identity mismatch")
    if _sha256(current_config) != expected_config_sha256:
        raise RuntimeError("current resume config identity mismatch")

    manifest = _load_object(
        source_release_root / "pbs-scripts/openreward-v9-release.json",
        label="source release manifest",
    )
    config_name = source_config_relative.name
    if (
        "swe32" in expected_experiment
        and "32node_recursive" in config_name
        and "swe32" in expected_trial
    ):
        model_artifact_key = "recursive_32node"
        wrapper_name = "openreward-ta-swe-32node-recursive-h200-v9-abci.pbs"
    elif (
        "swe16" in expected_experiment
        and "16node_nonrecursive" in config_name
        and "swe16" in expected_trial
    ):
        model_artifact_key = "nonrecursive_16node"
        wrapper_name = "openreward-ta-swe-16node-nonrecursive-h200-v9-abci.pbs"
    else:
        raise RuntimeError("source config/experiment/trial topology lineage mismatch")
    model = manifest.get("model_artifacts", {}).get(model_artifact_key)
    if (
        manifest.get("release_root") != str(source_release_root)
        or manifest.get("release_ref") != source_release_ref
        or not isinstance(model, dict)
        or model.get("repository") != model_repository
        or model.get("revision") != model_revision
        or model.get("snapshot_path") != str(model_snapshot)
        or model.get("architecture") != "dense_transformer"
        or model.get("safetensors_shard_count") != 18
    ):
        raise RuntimeError("source release/model lineage mismatch")

    overlay = source_release_root / "pbs-scripts/openreward-v9-abci.env"
    wrapper = source_release_root / "pbs-scripts" / wrapper_name
    _require_owned_regular(overlay, label="source release overlay")
    _require_owned_regular(wrapper, label="source release wrapper")
    overlay_text = overlay.read_text(encoding="utf-8")
    wrapper_text = wrapper.read_text(encoding="utf-8")
    required_overlay_lines = {
        f"export OPENREWARD_QWEN38_MODEL_REPOSITORY={model_repository}",
        f"export OPENREWARD_QWEN38_MODEL_REVISION={model_revision}",
        f"export OPENREWARD_QWEN38_MODEL_SNAPSHOT={model_snapshot}",
        "export PLATOON_MODEL_SNAPSHOT=${OPENREWARD_QWEN38_MODEL_SNAPSHOT}",
    }
    if not required_overlay_lines.issubset(set(overlay_text.splitlines())):
        raise RuntimeError("source release overlay does not bind the expected model")
    required_wrapper_lines = {
        f"REPO_ROOT={source_release_root}",
        f"RELEASE_REF={source_release_ref}",
        f"EXPERIMENT={expected_experiment}",
        f"TRIAL={expected_trial}",
        f"CONFIG=${{REPO_ROOT}}/{source_config_relative}",
    }
    if not required_wrapper_lines.issubset(set(wrapper_text.splitlines())):
        raise RuntimeError("source release wrapper lineage mismatch")
    return source_config


def _verify_receipts(
    *,
    source_candidate: Path,
    expected_source_candidate_sha256: str,
    source_pbs_job_id: str,
    expected_checkpoint_root: Path,
    source_config: Path,
    expected_source_config_sha256: str,
    expected_experiment: str,
    expected_trial: str,
    expected_global_step: int,
    target_receipt: Path,
    source_candidate_at_recover_step: bool,
    current_candidate_recover_evidence: dict[str, Any] | None,
    authorized_ancestor_receipts: list[Path],
    expected_authorized_ancestor_receipt_sha256s: list[str],
) -> None:
    if not source_candidate.name.endswith(".success.candidate.json"):
        raise RuntimeError("source receipt is not an exact success candidate path")
    _require_owned_regular(source_candidate, label="source success candidate")
    if _sha256(source_candidate) != expected_source_candidate_sha256:
        raise RuntimeError("source success candidate content digest mismatch")
    candidate = _load_object(source_candidate, label="source success candidate")
    recover = candidate.get("recover")
    wandb = candidate.get("wandb")
    expected_wandb_id = f"{expected_experiment}_{expected_trial}_train"
    if (
        candidate.get("schema_version") != 2
        or candidate.get("gate") != "abci-openreward-one-update-success"
        or candidate.get("kind") != "candidate"
        or candidate.get("pbs_job_id") != source_pbs_job_id
        or candidate.get("experiment_name") != expected_experiment
        or candidate.get("trial_name") != expected_trial
        or candidate.get("config_path") != str(source_config)
        or candidate.get("config_sha256") != expected_source_config_sha256
        or not isinstance(candidate.get("global_step"), int)
        or candidate["global_step"] < 0
        or (
            candidate["global_step"] != expected_global_step
            if source_candidate_at_recover_step
            else candidate["global_step"] >= expected_global_step
        )
        or not isinstance(recover, dict)
        or recover.get("checkpoint_root") != str(expected_checkpoint_root)
        or recover.get("global_step") != candidate.get("global_step")
        or not isinstance(wandb, dict)
        or wandb.get("run_id") != expected_wandb_id
    ):
        raise RuntimeError("source success candidate lineage/state mismatch")
    if source_candidate_at_recover_step:
        if current_candidate_recover_evidence is None:
            raise RuntimeError("same-step candidate recovery evidence was not reconstructed")
        if recover != current_candidate_recover_evidence:
            raise RuntimeError(
                "same-step candidate does not exactly bind the current recovery checkpoint"
            )
        if (
            candidate.get("stats_logger_last_commit_step") != expected_global_step
            or candidate.get("optimizer_update_successful") is not True
            or candidate.get("weight_sync_succeeded") is not True
            or isinstance(candidate.get("optimizer_batch_datums"), bool)
            or not isinstance(candidate.get("optimizer_batch_datums"), int)
            or candidate["optimizer_batch_datums"] < 1
            or isinstance(candidate.get("optimizer_action_tokens"), bool)
            or not isinstance(candidate.get("optimizer_action_tokens"), int)
            or candidate["optimizer_action_tokens"] < 1
            or candidate.get("expected_version") != expected_global_step + 1
            or candidate.get("actor_version") != expected_global_step + 1
            or candidate.get("rollout_version") != expected_global_step + 1
        ):
            raise RuntimeError(
                "same-step candidate lacks the exact post-checkpoint commit/update evidence"
            )
    elif current_candidate_recover_evidence is not None:
        raise RuntimeError("current candidate recovery evidence is restricted to same-step resumes")

    source_final = source_candidate.with_name(
        source_candidate.name.removesuffix(".candidate.json") + ".json"
    )
    source_error = source_candidate.with_name(
        source_candidate.name.removesuffix(".candidate.json") + ".error.json"
    )
    if source_final.exists() or source_final.is_symlink() or source_error.exists() or source_error.is_symlink():
        raise RuntimeError("source receipt state is ambiguous or already final")

    if not target_receipt.is_absolute() or target_receipt.parent != source_candidate.parent:
        raise RuntimeError("target resume receipt must share the exact source receipt directory")
    target_paths = {
        target_receipt,
        target_receipt.with_name(target_receipt.name.removesuffix(".json") + ".candidate.json"),
        target_receipt.with_name(target_receipt.name.removesuffix(".json") + ".error.json"),
    }
    if any(path.exists() or path.is_symlink() for path in target_paths):
        raise RuntimeError("target resume receipt state is stale")

    if len(authorized_ancestor_receipts) != len(
        expected_authorized_ancestor_receipt_sha256s
    ):
        raise RuntimeError("authorized ancestor receipt/hash counts differ")
    authorized_names: set[str] = set()
    for path, digest in zip(
        authorized_ancestor_receipts,
        expected_authorized_ancestor_receipt_sha256s,
    ):
        if path.parent != source_candidate.parent or path == source_candidate:
            raise RuntimeError("authorized ancestor receipt path is outside the exact lineage")
        _require_owned_regular(path, label="authorized ancestor receipt")
        if _sha256(path) != digest:
            raise RuntimeError("authorized ancestor receipt content digest mismatch")
        authorized_names.add(path.name)

    matching = {
        entry.name
        for entry in source_candidate.parent.glob(f"{expected_trial}*")
    }
    if matching != {source_candidate.name, *authorized_names}:
        raise RuntimeError(f"receipt state is missing or ambiguous: {sorted(matching)!r}")


def _verify_failed_preupdate_source(
    *,
    source_log: Path,
    expected_source_log_sha256: str,
    source_pbs_job_id: str,
    source_release_root: Path,
    expected_checkpoint_root: Path,
    expected_experiment: str,
    expected_trial: str,
    target_receipt: Path,
) -> None:
    """Bind a step-zero recovery to one failed, candidate-free source job."""

    if source_log.name != f"{source_pbs_job_id}.OU":
        raise RuntimeError("failed source log name does not match its PBS job ID")
    _require_owned_regular(source_log, label="failed source job log")
    if _sha256(source_log) != expected_source_log_sha256:
        raise RuntimeError("failed source job log content digest mismatch")
    source_commit = _git(source_release_root, "rev-parse", "--verify", "HEAD")
    log_text = source_log.read_text(encoding="utf-8", errors="strict")
    required_fragments = {
        f"openreward_v9_release=PASS release_commit={source_commit}",
        "[h200-16-v9] accepted gate2=",
        (
            "[pbs-openreward] starting host controller "
            f"experiment={expected_experiment} trial={expected_trial} nodes=16"
        ),
        (
            "Recover INFO: Saved recover checkpoint to "
            f"{expected_checkpoint_root}/default/recover_checkpoint"
        ),
        (
            "RLTrainer ERROR: Training failed with exception: "
            "optimizer_batch_datums must be an integer >= 1"
        ),
    }
    missing = sorted(fragment for fragment in required_fragments if fragment not in log_text)
    if missing:
        raise RuntimeError(f"failed source job log lineage evidence is incomplete: {missing!r}")
    if "training_success_candidate=published" in log_text:
        raise RuntimeError("failed preupdate source log unexpectedly published a success candidate")

    if (
        not target_receipt.is_absolute()
        or target_receipt.name
        != f"{expected_trial}.resume-gs0-v8r18.success.json"
    ):
        raise RuntimeError("target preupdate resume receipt identity mismatch")
    _require_canonical_directory(
        target_receipt.parent, label="target resume receipt directory"
    )
    source_paths = {
        target_receipt.parent / f"{expected_trial}.success.json",
        target_receipt.parent / f"{expected_trial}.success.candidate.json",
        target_receipt.parent / f"{expected_trial}.success.error.json",
    }
    if any(path.exists() or path.is_symlink() for path in source_paths):
        raise RuntimeError("failed preupdate source receipt/candidate state is not absent")
    target_paths = {
        target_receipt,
        target_receipt.with_name(target_receipt.name.removesuffix(".json") + ".candidate.json"),
        target_receipt.with_name(target_receipt.name.removesuffix(".json") + ".error.json"),
    }
    if any(path.exists() or path.is_symlink() for path in target_paths):
        raise RuntimeError("target resume receipt state is stale")
    matching = {entry.name for entry in target_receipt.parent.glob(f"{expected_trial}*")}
    if matching:
        raise RuntimeError(f"candidate-free receipt state is ambiguous: {sorted(matching)!r}")


def verify(args: argparse.Namespace) -> None:
    if args.allow_preupdate_global_step_zero and (
        args.preupdate_binding != PREUPDATE_BINDING_ID
        or args.expected_experiment != PREUPDATE_EXPERIMENT
        or args.expected_trial != PREUPDATE_TRIAL
        or args.source_pbs_job_id != PREUPDATE_SOURCE_JOB
        or args.expected_global_step != 0
        or args.expected_steps_per_epoch != 846
        or args.expected_shards != 80
    ):
        raise RuntimeError("preupdate resume binding identity/topology mismatch")
    checkpoint_root = args.checkpoint_root
    if checkpoint_root != args.expected_checkpoint_root:
        raise RuntimeError("checkpoint root does not equal the authorized absolute path")
    _require_canonical_directory(checkpoint_root, label="checkpoint root")
    _require_exact_children(checkpoint_root, {"default", "recover_info"}, label="checkpoint root")

    default_root = checkpoint_root / "default"
    recover_info = checkpoint_root / "recover_info"
    checkpoint = default_root / "recover_checkpoint"
    _require_canonical_directory(default_root, label="checkpoint default root")
    _require_canonical_directory(recover_info, label="recover info")
    _require_canonical_directory(checkpoint, label="recover checkpoint")
    _require_exact_children(default_root, {"recover_checkpoint"}, label="checkpoint default root")
    _require_exact_children(recover_info, RECOVER_INFO_FILES, label="recover info")

    before = _checkpoint_snapshot(checkpoint)
    pattern = re.compile(r"__([0-9]+)_0[.]distcp")
    shard_paths: dict[int, Path] = {}
    fixed_names = {".metadata", "common.pt", "metadata.json"}
    for entry in checkpoint.iterdir():
        match = pattern.fullmatch(entry.name)
        if match is not None:
            shard_paths[int(match.group(1))] = entry
        elif entry.name not in fixed_names:
            raise RuntimeError(f"recover checkpoint contains an unexpected entry: {entry.name}")
    expected_ordinals = set(range(args.expected_shards))
    if set(shard_paths) != expected_ordinals:
        raise RuntimeError(
            "recover checkpoint shard ordinals mismatch: "
            f"missing={sorted(expected_ordinals - set(shard_paths))!r} "
            f"unexpected={sorted(set(shard_paths) - expected_ordinals)!r}"
        )
    checkpoint_files = [checkpoint / name for name in fixed_names] + list(shard_paths.values())
    checkpoint_stats = [
        _require_owned_regular(path, label="recover checkpoint file")
        for path in checkpoint_files
    ]
    if len(checkpoint_files) != args.expected_shards + len(fixed_names):
        raise RuntimeError("recover checkpoint file count mismatch")
    total_bytes = sum(value.st_size for value in checkpoint_stats)
    if total_bytes != args.expected_total_bytes:
        raise RuntimeError(
            f"recover checkpoint byte count mismatch: {total_bytes} != {args.expected_total_bytes}"
        )
    if _size_manifest_sha256(checkpoint_files) != args.expected_checkpoint_layout_sha256:
        raise RuntimeError("recover checkpoint exact name/size manifest digest mismatch")
    if _sample_manifest_sha256(checkpoint_files) != args.expected_checkpoint_sample_manifest_sha256:
        raise RuntimeError("recover checkpoint bounded-sample manifest digest mismatch")

    metadata_json = _load_object(checkpoint / "metadata.json", label="checkpoint metadata JSON")
    if metadata_json != CHECKPOINT_METADATA:
        raise RuntimeError("recover checkpoint backend metadata mismatch")
    fixed_paths = [checkpoint / name for name in fixed_names]
    if _content_manifest_sha256(fixed_paths) != args.expected_fixed_content_sha256:
        raise RuntimeError("recover checkpoint fixed-file content manifest digest mismatch")
    dcp_metadata = checkpoint / ".metadata"
    atomic_save_path = (str(checkpoint) + "_new").encode("utf-8")
    if atomic_save_path not in dcp_metadata.read_bytes():
        raise RuntimeError("DCP metadata does not name the atomically promoted checkpoint path")

    step_info = _load_object(recover_info / "step_info.json", label="step info")
    if step_info != {
        "epoch": 0,
        "epoch_step": args.expected_global_step,
        "global_step": args.expected_global_step,
        "steps_per_epoch": args.expected_steps_per_epoch,
    }:
        raise RuntimeError("recover step info does not equal the authorized global step")
    stats_info = _load_object(recover_info / "stats_logger_info.json", label="stats logger info")
    expected_stats_commit_step = (
        QWEN38_SUCCESSOR_RECOVER_STATS_COMMIT_STEP
        if args.successor_binding == QWEN38_SUCCESSOR_BINDING_ID
        else args.expected_global_step - 1
    )
    if stats_info != {"last_commit_step": expected_stats_commit_step}:
        raise RuntimeError("stats logger recovery state is not coherent with the checkpoint step")
    checkpoint_info = _load_object(recover_info / "checkpoint_info.json", label="checkpoint timer info")
    step_timer = checkpoint_info.get("step")
    if (
        not isinstance(step_timer, dict)
        or step_timer.get("frequency_steps") != 1
        or step_timer.get("steps") != args.expected_global_step + 1
        or step_timer.get("last_steps") != args.expected_global_step + 1
        or step_timer.get("interval_steps") != 1
    ):
        raise RuntimeError("checkpoint timer state is not coherent with the checkpoint step")
    for name in RECOVER_INFO_FILES - {
        "step_info.json",
        "stats_logger_info.json",
        "checkpoint_info.json",
    }:
        _require_owned_regular(recover_info / name, label="recover info file")
    recover_info_paths = [recover_info / name for name in RECOVER_INFO_FILES]
    if _content_manifest_sha256(recover_info_paths) != args.expected_recover_info_content_sha256:
        raise RuntimeError("recover info content manifest digest mismatch")
    newest_checkpoint_ns = max(value.st_mtime_ns for value in checkpoint_stats)
    oldest_recover_ns = min((recover_info / name).stat().st_mtime_ns for name in RECOVER_INFO_FILES)
    if oldest_recover_ns < newest_checkpoint_ns:
        raise RuntimeError("recover info predates the promoted recover checkpoint")

    candidate_recover_evidence = None
    if args.successor_binding == QWEN38_SUCCESSOR_BINDING_ID:
        candidate_recover_evidence = _current_candidate_recover_evidence(
            checkpoint_root=checkpoint_root,
            checkpoint_files=checkpoint_files,
            recover_info_paths=recover_info_paths,
            expected_experiment=args.expected_experiment,
            expected_trial=args.expected_trial,
            expected_global_step=args.expected_global_step,
        )

    model_snapshot = args.model_snapshot
    _require_canonical_directory(model_snapshot, label="model snapshot")
    source_config = _verify_source_release(
        source_release_root=args.source_release_root,
        source_release_ref=args.source_release_ref,
        source_config_relative=args.source_config_relative,
        current_config=args.current_config,
        expected_source_config_sha256=args.expected_source_config_sha256,
        expected_config_sha256=args.expected_config_sha256,
        expected_experiment=args.expected_experiment,
        expected_trial=args.expected_trial,
        model_repository=args.model_repository,
        model_revision=args.model_revision,
        model_snapshot=model_snapshot,
    )
    if args.allow_preupdate_global_step_zero:
        _verify_failed_preupdate_source(
            source_log=args.source_log,
            expected_source_log_sha256=args.expected_source_log_sha256,
            source_pbs_job_id=args.source_pbs_job_id,
            source_release_root=args.source_release_root,
            expected_checkpoint_root=checkpoint_root,
            expected_experiment=args.expected_experiment,
            expected_trial=args.expected_trial,
            target_receipt=args.target_receipt,
        )
    else:
        _verify_receipts(
            source_candidate=args.source_candidate,
            expected_source_candidate_sha256=args.expected_source_candidate_sha256,
            source_pbs_job_id=args.source_pbs_job_id,
            expected_checkpoint_root=checkpoint_root,
            source_config=source_config,
            expected_source_config_sha256=args.expected_source_config_sha256,
            expected_experiment=args.expected_experiment,
            expected_trial=args.expected_trial,
            expected_global_step=args.expected_global_step,
            target_receipt=args.target_receipt,
            source_candidate_at_recover_step=(
                args.successor_binding == QWEN38_SUCCESSOR_BINDING_ID
            ),
            current_candidate_recover_evidence=candidate_recover_evidence,
            authorized_ancestor_receipts=args.authorized_ancestor_receipt,
            expected_authorized_ancestor_receipt_sha256s=(
                args.expected_authorized_ancestor_receipt_sha256
            ),
        )
    if _checkpoint_snapshot(checkpoint) != before:
        raise RuntimeError("recover checkpoint changed during authorization")
    print(
        "openreward_resume_checkpoint=PASS "
        f"global_step={args.expected_global_step} shards={args.expected_shards} "
        f"bytes={total_bytes} source_job={args.source_pbs_job_id}"
    )


def _verify_runtime_objects(
    *,
    checkpoint: Path,
    metadata: Any,
    common: Any,
    dataloader: Any,
    expected_state_keys: int,
    expected_storage_entries: int,
    expected_shards: int,
    expected_lr_num_steps: int,
    expected_samples_yielded: int,
    expected_iterator_yielded: int,
) -> None:
    state_metadata = getattr(metadata, "state_dict_metadata", None)
    storage_data = getattr(metadata, "storage_data", None)
    storage_meta = getattr(metadata, "storage_meta", None)
    if not isinstance(state_metadata, dict) or len(state_metadata) != expected_state_keys:
        raise RuntimeError("DCP state-dict metadata key count mismatch")
    if not isinstance(storage_data, dict) or len(storage_data) != expected_storage_entries:
        raise RuntimeError("DCP storage entry count mismatch")
    if getattr(metadata, "version", None) != "1.0.0":
        raise RuntimeError("DCP metadata version mismatch")
    if (
        storage_meta is None
        or getattr(storage_meta, "checkpoint_id", None) != str(checkpoint) + "_new"
        or re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            str(getattr(storage_meta, "save_id", "")),
        )
        is None
        or getattr(storage_meta, "load_id", None) is not None
        or list(getattr(storage_meta, "modules", ()))
    ):
        raise RuntimeError("DCP atomic-save storage metadata mismatch")

    expected_paths = {f"__{ordinal}_0.distcp" for ordinal in range(expected_shards)}
    intervals: dict[str, list[tuple[int, int]]] = {}
    indexed_fqns: set[str] = set()
    for index, storage in storage_data.items():
        fqn = getattr(index, "fqn", None)
        relative = str(getattr(storage, "relative_path", ""))
        offset = getattr(storage, "offset", None)
        length = getattr(storage, "length", None)
        if not isinstance(fqn, str) or not fqn or fqn not in state_metadata:
            raise RuntimeError("DCP storage entry names an unknown state key")
        if relative not in expected_paths or Path(relative).name != relative:
            raise RuntimeError("DCP storage entry names an unauthorized shard path")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
        ):
            raise RuntimeError("DCP storage entry has invalid byte bounds")
        indexed_fqns.add(fqn)
        intervals.setdefault(relative, []).append((offset, length))
    if indexed_fqns != set(state_metadata) or set(intervals) != expected_paths:
        raise RuntimeError("DCP storage coverage does not match its state keys/shards")

    for relative, values in intervals.items():
        expected_offset = 0
        for offset, length in sorted(values):
            if offset != expected_offset:
                raise RuntimeError(f"DCP storage coverage is not gap-free: {relative}")
            expected_offset += length
        if expected_offset != (checkpoint / relative).stat().st_size:
            raise RuntimeError(f"DCP storage coverage is not exactly in bounds: {relative}")

    if (
        not isinstance(common, dict)
        or set(common) != {"optimizer", "lr_scheduler"}
        or common.get("optimizer") != {"param_state_sharding_type": "dp_reshardable"}
        or not isinstance(common.get("lr_scheduler"), dict)
        or common["lr_scheduler"].get("num_steps") != expected_lr_num_steps
    ):
        raise RuntimeError("common.pt optimizer/LR scheduler recovery state mismatch")
    if not isinstance(dataloader, list) or len(dataloader) != 1 or not isinstance(dataloader[0], dict):
        raise RuntimeError("dataloader recovery state shape mismatch")
    loader = dataloader[0]
    sampler_state = loader.get("_sampler_iter_state")
    if (
        not isinstance(sampler_state, dict)
        or sampler_state.get("samples_yielded") != expected_samples_yielded
        or loader.get("_sampler_iter_yielded") != expected_iterator_yielded
        or loader.get("_num_yielded") != expected_iterator_yielded
        or loader.get("_iterator_finished") is not False
    ):
        raise RuntimeError("dataloader recovery progress mismatch")


def runtime_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog=f"{Path(sys.argv[0]).name} runtime")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-layout-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sample-manifest-sha256", required=True)
    parser.add_argument("--expected-fixed-content-sha256", required=True)
    parser.add_argument("--expected-recover-info-content-sha256", required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
    parser.add_argument("--allow-preupdate-global-step-zero", action="store_true")
    parser.add_argument("--preupdate-binding")
    parser.add_argument("--expected-state-keys", type=int, required=True)
    parser.add_argument("--expected-storage-entries", type=int, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-lr-num-steps", type=int, required=True)
    parser.add_argument("--expected-samples-yielded", type=int, required=True)
    parser.add_argument("--expected-iterator-yielded", type=int, required=True)
    args = parser.parse_args(argv)
    digests = (
        args.expected_checkpoint_layout_sha256,
        args.expected_checkpoint_sample_manifest_sha256,
        args.expected_fixed_content_sha256,
        args.expected_recover_info_content_sha256,
    )
    if (
        any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in digests)
        or min(
            args.expected_state_keys,
            args.expected_storage_entries,
            args.expected_shards,
            args.expected_samples_yielded,
            args.expected_iterator_yielded,
        )
        < 1
        or (
            args.allow_preupdate_global_step_zero
            and (
                args.preupdate_binding != PREUPDATE_BINDING_ID
                or args.expected_global_step != 0
                or args.expected_lr_num_steps != 0
                or args.expected_state_keys != 1456
                or args.expected_storage_entries != 9169
                or args.expected_shards != 80
                or args.expected_samples_yielded != 72
                or args.expected_iterator_yielded != 9
            )
        )
        or (
            not args.allow_preupdate_global_step_zero
            and (
                args.preupdate_binding is not None
                or args.expected_global_step < 1
                or args.expected_lr_num_steps != args.expected_global_step + 1
            )
        )
    ):
        parser.error("runtime counts/digests are invalid")
    try:
        root = args.checkpoint_root
        _require_canonical_directory(root, label="runtime checkpoint root")
        checkpoint = root / "default" / "recover_checkpoint"
        recover_info = root / "recover_info"
        _require_canonical_directory(checkpoint, label="runtime recover checkpoint")
        _require_canonical_directory(recover_info, label="runtime recover info")
        checkpoint_paths = list(checkpoint.iterdir())
        fixed_paths = [checkpoint / name for name in {".metadata", "common.pt", "metadata.json"}]
        recover_paths = [recover_info / name for name in RECOVER_INFO_FILES]
        if _size_manifest_sha256(checkpoint_paths) != args.expected_checkpoint_layout_sha256:
            raise RuntimeError("runtime checkpoint name/size manifest changed")
        if (
            _sample_manifest_sha256(checkpoint_paths)
            != args.expected_checkpoint_sample_manifest_sha256
        ):
            raise RuntimeError("runtime checkpoint bounded-sample manifest changed")
        if _content_manifest_sha256(fixed_paths) != args.expected_fixed_content_sha256:
            raise RuntimeError("runtime checkpoint fixed contents changed")
        if _content_manifest_sha256(recover_paths) != args.expected_recover_info_content_sha256:
            raise RuntimeError("runtime recover info contents changed")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("runtime verifier requires the pinned training torch") from exc
        with (checkpoint / ".metadata").open("rb") as stream:
            metadata = pickle.load(stream)
        common = torch.load(checkpoint / "common.pt", map_location="cpu", weights_only=False)
        with (recover_info / "dataloader_info.pkl").open("rb") as stream:
            dataloader = pickle.load(stream)
        _verify_runtime_objects(
            checkpoint=checkpoint,
            metadata=metadata,
            common=common,
            dataloader=dataloader,
            expected_state_keys=args.expected_state_keys,
            expected_storage_entries=args.expected_storage_entries,
            expected_shards=args.expected_shards,
            expected_lr_num_steps=args.expected_lr_num_steps,
            expected_samples_yielded=args.expected_samples_yielded,
            expected_iterator_yielded=args.expected_iterator_yielded,
        )
    except (OSError, RuntimeError, pickle.PickleError) as exc:
        parser.error(str(exc))
    print(
        "openreward_resume_checkpoint_runtime=PASS "
        f"state_keys={args.expected_state_keys} storage_entries={args.expected_storage_entries} "
        f"shards={args.expected_shards} lr_num_steps={args.expected_lr_num_steps}"
    )


def main() -> None:
    if sys.argv[1:2] == ["runtime"]:
        runtime_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-global-step", type=int, required=True)
    parser.add_argument("--expected-steps-per-epoch", type=int, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-total-bytes", type=int, required=True)
    parser.add_argument("--expected-checkpoint-layout-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sample-manifest-sha256", required=True)
    parser.add_argument("--expected-fixed-content-sha256", required=True)
    parser.add_argument("--expected-recover-info-content-sha256", required=True)
    parser.add_argument("--expected-experiment", required=True)
    parser.add_argument("--expected-trial", required=True)
    parser.add_argument("--current-config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-source-config-sha256")
    parser.add_argument("--source-candidate", type=Path)
    parser.add_argument("--expected-source-candidate-sha256")
    parser.add_argument(
        "--authorized-ancestor-receipt", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--expected-authorized-ancestor-receipt-sha256", action="append", default=[]
    )
    parser.add_argument("--source-log", type=Path)
    parser.add_argument("--expected-source-log-sha256")
    parser.add_argument("--allow-preupdate-global-step-zero", action="store_true")
    parser.add_argument("--preupdate-binding")
    parser.add_argument("--successor-binding")
    parser.add_argument("--source-pbs-job-id", required=True)
    parser.add_argument("--source-release-root", type=Path, required=True)
    parser.add_argument("--source-release-ref", required=True)
    parser.add_argument("--source-config-relative", type=Path, required=True)
    parser.add_argument("--model-repository", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--target-receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_source_config_sha256 is None:
        args.expected_source_config_sha256 = args.expected_config_sha256
    successor_binding = args.successor_binding == QWEN38_SUCCESSOR_BINDING_ID
    if (
        (
            args.allow_preupdate_global_step_zero
            and (
                args.expected_global_step != 0
                or args.preupdate_binding != PREUPDATE_BINDING_ID
                or args.expected_experiment != PREUPDATE_EXPERIMENT
                or args.expected_trial != PREUPDATE_TRIAL
                or args.source_pbs_job_id != PREUPDATE_SOURCE_JOB
                or args.expected_steps_per_epoch != 846
                or args.expected_shards != 80
                or args.source_log is None
                or re.fullmatch(r"[0-9a-f]{64}", args.expected_source_log_sha256 or "") is None
                or args.source_candidate is not None
                or args.expected_source_candidate_sha256 is not None
                or args.successor_binding is not None
                or args.authorized_ancestor_receipt
                or args.expected_authorized_ancestor_receipt_sha256
            )
        )
        or (
            not args.allow_preupdate_global_step_zero
            and (
                args.expected_global_step < 1
                or args.preupdate_binding is not None
                or args.source_candidate is None
                or re.fullmatch(
                    r"[0-9a-f]{64}", args.expected_source_candidate_sha256 or ""
                )
                is None
                or args.source_log is not None
                or args.expected_source_log_sha256 is not None
            )
        )
        or (
            args.successor_binding is not None
            and not successor_binding
        )
        or (
            successor_binding
            and (
                args.allow_preupdate_global_step_zero
                or args.expected_experiment != QWEN38_SUCCESSOR_EXPERIMENT
                or args.expected_trial != QWEN38_SUCCESSOR_TRIAL
                or args.source_config_relative != QWEN38_SUCCESSOR_SOURCE_CONFIG
                or args.target_receipt.name != QWEN38_SUCCESSOR_RECEIPT
                or args.expected_global_step != QWEN38_SUCCESSOR_GLOBAL_STEP
                or args.expected_source_config_sha256
                != QWEN38_SUCCESSOR_SOURCE_CONFIG_SHA256
                or args.expected_config_sha256
                != QWEN38_SUCCESSOR_CURRENT_CONFIG_SHA256
                or len(args.authorized_ancestor_receipt) != 1
                or len(args.expected_authorized_ancestor_receipt_sha256) != 1
                or args.authorized_ancestor_receipt[0].name
                != QWEN38_SUCCESSOR_ANCESTOR_RECEIPT
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    args.expected_authorized_ancestor_receipt_sha256[0],
                )
                is None
                or args.expected_authorized_ancestor_receipt_sha256[0]
                != QWEN38_SUCCESSOR_ANCESTOR_SHA256
            )
        )
        or (
            not successor_binding
            and (
                args.expected_source_config_sha256 != args.expected_config_sha256
                or args.authorized_ancestor_receipt
                or args.expected_authorized_ancestor_receipt_sha256
            )
        )
        or args.expected_steps_per_epoch < 1
        or args.expected_shards < 1
        or args.expected_total_bytes < 1
        or re.fullmatch(r"[0-9a-f]{64}", args.expected_config_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", args.expected_source_config_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", args.expected_checkpoint_layout_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", args.expected_checkpoint_sample_manifest_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", args.expected_fixed_content_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", args.expected_recover_info_content_sha256) is None
    ):
        parser.error("expected steps/counts/digests are invalid")
    try:
        verify(args)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
