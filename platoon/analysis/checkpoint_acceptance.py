"""Read-only acceptance checks for Megatron recovery and HF export checkpoints.

The checks intentionally inspect metadata only. They never deserialize tensor
shards, copy checkpoints, or modify the supplied directories.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

MAX_DCP_METADATA_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024

_OPTIMIZER_KEY_RE = re.compile(r"^(?P<prefix>.+)[./](?P<kind>exp_avg_sq|exp_avg|param)$")
_LAYER_RE = re.compile(r"(^|\.)layers\.(?P<index>\d+)\.")
_LM_HEAD_RE = re.compile(r"(^|\.)lm_head(\.|$)")
_MTP_WEIGHT_RE = re.compile(
    r"(^|[._])(?:mtp(?:_layers?)?|next_?n(?:_predict(?:_layers?)?)?|"
    r"multi_token_prediction)(?=$|[._])",
    re.IGNORECASE,
)
_MTP_CONFIG_FIELDS = ("mtp_num_hidden_layers", "num_nextn_predict_layers")


class CheckpointAcceptanceError(ValueError):
    """Raised when checkpoint metadata violates an acceptance invariant."""


@dataclass(frozen=True)
class DCPAcceptanceSummary:
    metadata_keys: int
    optimizer_state_prefixes: int
    param_keys: int
    exp_avg_keys: int
    exp_avg_sq_keys: int


@dataclass(frozen=True)
class HFAcceptanceSummary:
    indexed_weights: int
    transformer_layers: int
    layer_namespace: str
    lm_head_keys: int
    configured_mtp_layers: int
    mtp_weight_keys: int


def _example(values: Iterable[str], limit: int = 3) -> str:
    selected = sorted(values)[:limit]
    return ", ".join(selected) if selected else "<none>"


def validate_dcp_metadata_keys(keys: Iterable[str]) -> DCPAcceptanceSummary:
    """Validate optimizer-state triplets from DCP state_dict_metadata keys.

    An exact terminal component is required, so exp_avg_sq cannot accidentally
    satisfy the exp_avg check. Every detected optimizer bucket must have three
    distinct metadata entries with the same prefix: param, exp_avg, and
    exp_avg_sq.
    """

    normalized = tuple(keys)
    if not normalized:
        raise CheckpointAcceptanceError("DCP metadata contains no state-dict keys.")
    if any(not isinstance(key, str) for key in normalized):
        raise CheckpointAcceptanceError("DCP metadata keys must all be strings.")

    prefixes_by_kind: dict[str, set[str]] = {
        "param": set(),
        "exp_avg": set(),
        "exp_avg_sq": set(),
    }
    keys_by_kind: dict[str, set[str]] = {kind: set() for kind in prefixes_by_kind}
    for key in normalized:
        match = _OPTIMIZER_KEY_RE.match(key)
        if match is None or "optimizer" not in match.group("prefix").lower():
            continue
        kind = match.group("kind")
        prefixes_by_kind[kind].add(match.group("prefix"))
        keys_by_kind[kind].add(key)

    errors: list[str] = []
    missing_kinds = [kind for kind, matches in keys_by_kind.items() if not matches]
    if missing_kinds:
        errors.append(
            "missing exact optimizer metadata key suffix(es): "
            + ", ".join(missing_kinds)
            + "; expected distinct keys ending in .param, .exp_avg, and .exp_avg_sq"
        )

    all_prefixes = set().union(*prefixes_by_kind.values())
    for kind, prefixes in prefixes_by_kind.items():
        missing = all_prefixes - prefixes
        if missing:
            errors.append(f"{len(missing)} optimizer state prefix(es) lack .{kind}; examples: {_example(missing)}")

    all_state_keys = set().union(*keys_by_kind.values())
    if len(all_state_keys) != sum(len(matches) for matches in keys_by_kind.values()):
        errors.append("optimizer param/exp_avg/exp_avg_sq metadata keys are not distinct")

    if errors:
        raise CheckpointAcceptanceError("DCP optimizer metadata rejected:\n- " + "\n- ".join(errors))

    return DCPAcceptanceSummary(
        metadata_keys=len(normalized),
        optimizer_state_prefixes=len(all_prefixes),
        param_keys=len(keys_by_kind["param"]),
        exp_avg_keys=len(keys_by_kind["exp_avg"]),
        exp_avg_sq_keys=len(keys_by_kind["exp_avg_sq"]),
    )


def _require_bounded_file(path: Path, *, max_bytes: int, description: str) -> None:
    if not path.is_file():
        raise CheckpointAcceptanceError(f"{description} does not exist: {path}")
    size = path.stat().st_size
    if size > max_bytes:
        raise CheckpointAcceptanceError(
            f"{description} is {size} bytes, exceeding the metadata-only safety limit of {max_bytes} bytes: {path}"
        )


def _safe_relative_storage_path(directory: Path, raw_path: Any) -> Path:
    """Resolve a DCP storage path without permitting escape from its root."""

    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path or "\\" in raw_path:
        raise CheckpointAcceptanceError(f"DCP storage_data contains unsafe relative_path {raw_path!r}.")
    posix = PurePosixPath(raw_path)
    windows = PureWindowsPath(raw_path)
    unsafe = (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part == ".." for part in posix.parts)
        or any(part == ".." for part in windows.parts)
        or not posix.parts
    )
    if unsafe:
        raise CheckpointAcceptanceError(f"DCP storage_data contains unsafe relative_path {raw_path!r}.")

    root = directory.resolve()
    candidate = directory.joinpath(*posix.parts)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CheckpointAcceptanceError(
            f"DCP storage_data relative_path escapes the recovery directory: {raw_path!r}."
        ) from exc
    return resolved


def _validate_dcp_storage_data(directory: Path, metadata: Any) -> None:
    """Validate DCP file references and byte ranges without reading payloads."""

    storage_data = getattr(metadata, "storage_data", None)
    if not isinstance(storage_data, Mapping) or not storage_data:
        raise CheckpointAcceptanceError("DCP metadata has no non-empty storage_data mapping.")

    storage_files: dict[str, tuple[Path, int]] = {}
    for storage_index, storage_info in storage_data.items():
        raw_path = getattr(storage_info, "relative_path", None)
        if isinstance(raw_path, str) and raw_path in storage_files:
            path, file_size = storage_files[raw_path]
        else:
            path = _safe_relative_storage_path(directory, raw_path)
            if not path.is_file():
                raise CheckpointAcceptanceError(f"DCP storage file referenced by metadata does not exist: {raw_path!r}")
            try:
                file_size = path.stat().st_size
            except OSError as exc:
                raise CheckpointAcceptanceError(f"Could not stat DCP storage file {path}: {exc}") from exc
            storage_files[raw_path] = (path, file_size)

        offset = getattr(storage_info, "offset", None)
        length = getattr(storage_info, "length", None)
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length < 0
        ):
            raise CheckpointAcceptanceError(
                "DCP storage_data contains an invalid byte range for "
                f"{storage_index!r}: offset={offset!r}, length={length!r}."
            )
        if offset > file_size or length > file_size - offset:
            raise CheckpointAcceptanceError(
                "DCP storage_data byte range exceeds its file for "
                f"{storage_index!r}: offset={offset}, length={length}, "
                f"file={raw_path!r}, file_size={file_size}."
            )


def read_dcp_metadata_keys(recovery_dir: str | Path) -> tuple[str, ...]:
    """Read only the DCP .metadata file and return state-dict keys."""

    directory = Path(recovery_dir)
    if not directory.is_dir():
        raise CheckpointAcceptanceError(f"DCP recovery directory does not exist: {directory}")
    metadata_path = directory / ".metadata"
    _require_bounded_file(
        metadata_path,
        max_bytes=MAX_DCP_METADATA_BYTES,
        description="DCP .metadata file",
    )
    try:
        from torch.distributed.checkpoint import FileSystemReader
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise CheckpointAcceptanceError(
            "Reading DCP metadata requires a PyTorch installation with torch.distributed.checkpoint."
        ) from exc
    try:
        metadata = FileSystemReader(str(directory)).read_metadata()
        state_metadata = metadata.state_dict_metadata
    except Exception as exc:
        raise CheckpointAcceptanceError(
            f"Could not read DCP metadata from {metadata_path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(state_metadata, Mapping):
        raise CheckpointAcceptanceError(f"DCP metadata at {metadata_path} has no state_dict_metadata mapping.")
    _validate_dcp_storage_data(directory, metadata)
    return tuple(state_metadata.keys())


def validate_dcp_recovery_directory(recovery_dir: str | Path) -> DCPAcceptanceSummary:
    return validate_dcp_metadata_keys(read_dcp_metadata_keys(recovery_dir))


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    _require_bounded_file(path, max_bytes=MAX_JSON_BYTES, description=description)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointAcceptanceError(f"Could not parse {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointAcceptanceError(f"{description} must contain a JSON object: {path}")
    return value


def _language_config(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    for key in ("text_config", "language_config", "llm_config"):
        candidate = config.get(key)
        if isinstance(candidate, Mapping):
            return candidate, key
    return config, "<root>"


def _configured_layer_count(config: Mapping[str, Any]) -> tuple[int, str]:
    language, namespace = _language_config(config)
    value = language.get("num_hidden_layers")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CheckpointAcceptanceError(f"HF config must define a positive integer num_hidden_layers in {namespace}.")
    return value, namespace


def _configured_mtp_layers(config: Mapping[str, Any]) -> tuple[int, list[str]]:
    language, language_namespace = _language_config(config)
    declarations: list[tuple[str, int]] = []
    config_nodes = [("<root>", config)]
    if language is not config:
        config_nodes.append((language_namespace, language))
    for namespace, node in config_nodes:
        for field in _MTP_CONFIG_FIELDS:
            if field not in node:
                continue
            raw = node[field]
            value = 0 if raw is None else raw
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CheckpointAcceptanceError(
                    f"HF config {namespace}.{field} must be a non-negative integer or null; got {raw!r}."
                )
            declarations.append((f"{namespace}.{field}", value))

    values = {value for _, value in declarations}
    if len(values) > 1:
        rendered = ", ".join(f"{path}={value}" for path, value in declarations)
        raise CheckpointAcceptanceError(f"HF config has conflicting MTP layer declarations: {rendered}")
    return (next(iter(values)) if values else 0), [path for path, _ in declarations]


def _layer_namespaces(weight_keys: Iterable[str]) -> dict[str, set[int]]:
    namespaces: dict[str, set[int]] = {}
    for key in weight_keys:
        match = _LAYER_RE.search(key)
        if match is None:
            continue
        layers_start = match.start() + len(match.group(1))
        namespace = key[:layers_start] + "layers."
        namespaces.setdefault(namespace, set()).add(int(match.group("index")))
    return namespaces


def _weight_map_shard_filenames(weight_map: Mapping[Any, Any]) -> set[str]:
    """Return safe, basename-only safetensors shard filenames."""

    errors: list[str] = []
    filenames: set[str] = set()
    for weight_key, raw_filename in weight_map.items():
        if not isinstance(raw_filename, str):
            errors.append(f"{weight_key!r} references non-string shard filename {raw_filename!r}")
            continue
        filename = raw_filename
        posix = PurePosixPath(filename)
        windows = PureWindowsPath(filename)
        unsafe = (
            not filename
            or "\x00" in filename
            or posix.is_absolute()
            or windows.is_absolute()
            or bool(windows.drive)
            or len(posix.parts) != 1
            or len(windows.parts) != 1
            or filename in {".", ".."}
            or not filename.endswith(".safetensors")
        )
        if unsafe:
            errors.append(
                f"{weight_key!r} references unsafe shard filename {filename!r}; "
                "expected a basename ending in .safetensors"
            )
            continue
        filenames.add(filename)
    if errors:
        raise CheckpointAcceptanceError("HF weight_map shard references rejected:\n- " + "\n- ".join(errors))
    return filenames


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    """Read a bounded safetensors JSON header and return payload capacity."""

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise CheckpointAcceptanceError(f"Could not stat safetensors shard {path}: {exc}") from exc
    if file_size < 8:
        raise CheckpointAcceptanceError(
            f"Safetensors shard is too short to contain a header: {path} ({file_size} bytes)."
        )

    try:
        with path.open("rb") as shard:
            prefix = shard.read(8)
            if len(prefix) != 8:
                raise CheckpointAcceptanceError(f"Could not read safetensors header length from {path}.")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size == 0 or header_size > MAX_SAFETENSORS_HEADER_BYTES:
                raise CheckpointAcceptanceError(
                    f"Safetensors header size {header_size} is outside the allowed range "
                    f"1..{MAX_SAFETENSORS_HEADER_BYTES}: {path}"
                )
            if header_size > file_size - 8:
                raise CheckpointAcceptanceError(
                    f"Safetensors header exceeds shard size: header={header_size}, file={file_size}, path={path}"
                )
            header_bytes = shard.read(header_size)
    except CheckpointAcceptanceError:
        raise
    except OSError as exc:
        raise CheckpointAcceptanceError(f"Could not read safetensors header from {path}: {exc}") from exc

    if len(header_bytes) != header_size:
        raise CheckpointAcceptanceError(
            f"Safetensors header is truncated: expected {header_size} bytes, read {len(header_bytes)}: {path}"
        )
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointAcceptanceError(f"Could not parse safetensors header in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise CheckpointAcceptanceError(f"Safetensors header must be a JSON object: {path}")
    return header, file_size - 8 - header_size


def _validate_hf_shard_headers(directory: Path, weight_map: Mapping[str, str]) -> None:
    """Match every indexed key to bounded metadata in its assigned shard."""

    assigned_by_filename: dict[str, set[str]] = {}
    for key, filename in weight_map.items():
        assigned_by_filename.setdefault(filename, set()).add(key)

    errors: list[str] = []
    for filename, expected_keys in sorted(assigned_by_filename.items()):
        shard_path = directory / filename
        try:
            header, payload_size = _read_safetensors_header(shard_path)
        except CheckpointAcceptanceError as exc:
            errors.append(str(exc))
            continue

        actual_keys = set(header) - {"__metadata__"}
        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys
        if missing:
            errors.append(f"{filename} header lacks {len(missing)} weight_map key(s); examples: {_example(missing)}")
        if unexpected:
            errors.append(
                f"{filename} header contains {len(unexpected)} tensor key(s) not assigned to it by weight_map; "
                f"examples: {_example(unexpected)}"
            )

        for key in sorted(actual_keys):
            tensor_metadata = header[key]
            if not isinstance(tensor_metadata, Mapping):
                errors.append(f"{filename} header entry {key!r} is not an object")
                continue
            offsets = tensor_metadata.get("data_offsets")
            valid_offsets = (
                isinstance(offsets, list)
                and len(offsets) == 2
                and all(isinstance(value, int) and not isinstance(value, bool) for value in offsets)
            )
            if not valid_offsets:
                errors.append(f"{filename} header entry {key!r} has invalid data_offsets {offsets!r}")
                continue
            begin, end = offsets
            if begin < 0 or end < begin or end > payload_size:
                errors.append(
                    f"{filename} header entry {key!r} has out-of-bounds data_offsets "
                    f"{offsets!r} for payload size {payload_size}"
                )

    if errors:
        raise CheckpointAcceptanceError("HF safetensors shard metadata rejected:\n- " + "\n- ".join(errors))


def validate_hf_export_documents(
    config: Mapping[str, Any],
    index: Mapping[str, Any],
) -> HFAcceptanceSummary:
    """Validate an HF config and safetensors index without reading any shard."""

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise CheckpointAcceptanceError("HF model.safetensors.index.json must contain a non-empty weight_map object.")
    if any(not isinstance(key, str) for key in weight_map):
        raise CheckpointAcceptanceError("HF weight_map keys must all be strings.")
    _weight_map_shard_filenames(weight_map)
    weight_keys = tuple(weight_map.keys())

    expected_layers, config_namespace = _configured_layer_count(config)
    expected_indices = set(range(expected_layers))
    namespaces = _layer_namespaces(weight_keys)
    covering = [(namespace, indices) for namespace, indices in namespaces.items() if expected_indices.issubset(indices)]
    best_namespace = "<none>"
    best_indices: set[int] = set()
    if covering:
        best_namespace, best_indices = max(covering, key=lambda item: len(item[1]))
    elif namespaces:
        best_namespace, best_indices = max(
            namespaces.items(), key=lambda item: (len(item[1] & expected_indices), len(item[1]))
        )

    lm_head_keys = [key for key in weight_keys if _LM_HEAD_RE.search(key)]
    mtp_weight_keys = [key for key in weight_keys if _MTP_WEIGHT_RE.search(key)]
    configured_mtp_layers, mtp_declarations = _configured_mtp_layers(config)

    errors: list[str] = []
    missing_layers = sorted(expected_indices - best_indices)
    if missing_layers:
        errors.append(
            f"transformer namespace {best_namespace!r} is missing configured layer "
            f"indices {missing_layers} (expected 0..{expected_layers - 1} from "
            f"{config_namespace}.num_hidden_layers={expected_layers})"
        )
    if not lm_head_keys:
        errors.append("weight_map contains no lm_head weight key")
    if configured_mtp_layers > 0 and not mtp_weight_keys:
        declaration_text = ", ".join(mtp_declarations) or "MTP config"
        errors.append(
            f"config declares {configured_mtp_layers} MTP layer(s) via {declaration_text}, "
            "but weight_map contains no MTP/next-token-prediction weights"
        )
    if configured_mtp_layers == 0 and mtp_weight_keys:
        errors.append(
            f"config disables MTP, but weight_map contains {len(mtp_weight_keys)} MTP "
            f"weight key(s); examples: {_example(mtp_weight_keys)}"
        )
    if errors:
        raise CheckpointAcceptanceError("HF export rejected:\n- " + "\n- ".join(errors))

    return HFAcceptanceSummary(
        indexed_weights=len(weight_keys),
        transformer_layers=expected_layers,
        layer_namespace=best_namespace,
        lm_head_keys=len(lm_head_keys),
        configured_mtp_layers=configured_mtp_layers,
        mtp_weight_keys=len(mtp_weight_keys),
    )


def validate_hf_export(export: str | Path) -> HFAcceptanceSummary:
    path = Path(export)
    if path.is_dir():
        directory = path
        index_path = directory / "model.safetensors.index.json"
    else:
        index_path = path
        directory = path.parent
    config_path = directory / "config.json"
    config = _read_json_object(config_path, "HF config")
    index = _read_json_object(index_path, "HF safetensors index")
    errors: list[str] = []
    summary: HFAcceptanceSummary | None = None
    try:
        summary = validate_hf_export_documents(config, index)
    except CheckpointAcceptanceError as exc:
        errors.append(str(exc))

    weight_map = index.get("weight_map")
    if isinstance(weight_map, Mapping) and weight_map:
        try:
            filenames = _weight_map_shard_filenames(weight_map)
        except CheckpointAcceptanceError as exc:
            # The document validator already reports the same unsafe references.
            if not errors:
                errors.append(str(exc))
        else:
            missing = sorted(filename for filename in filenames if not (directory / filename).is_file())
            if missing:
                errors.append(
                    f"HF export is missing {len(missing)} shard file(s) referenced by weight_map: {_example(missing)}"
                )
            elif all(isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()):
                try:
                    _validate_hf_shard_headers(directory, weight_map)
                except CheckpointAcceptanceError as exc:
                    errors.append(str(exc))
    if errors:
        raise CheckpointAcceptanceError("HF export directory rejected:\n- " + "\n- ".join(errors))
    assert summary is not None
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate DCP optimizer metadata and/or an HF export index without loading tensor shards."
    )
    parser.add_argument("--dcp-recovery", type=Path, help="DCP recovery directory containing .metadata")
    parser.add_argument(
        "--hf-export",
        type=Path,
        help="HF export directory or its model.safetensors.index.json",
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable result")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.dcp_recovery is None and args.hf_export is None:
        parser.error("provide --dcp-recovery, --hf-export, or both")

    checks: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name, path, validator in (
        ("dcp", args.dcp_recovery, validate_dcp_recovery_directory),
        ("hf", args.hf_export, validate_hf_export),
    ):
        if path is None:
            continue
        try:
            checks[name] = asdict(validator(path))
        except CheckpointAcceptanceError as exc:
            errors.append(f"{name.upper()}: {exc}")

    result = {"ok": not errors, "checks": checks, "errors": errors}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for name, summary in checks.items():
            print(f"PASS {name.upper()}: {json.dumps(summary, sort_keys=True)}")
        for error in errors:
            print(f"FAIL {error}", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
