from __future__ import annotations

import json
import struct
from types import SimpleNamespace

import pytest

from platoon.analysis.checkpoint_acceptance import (
    MAX_SAFETENSORS_HEADER_BYTES,
    CheckpointAcceptanceError,
    _validate_dcp_storage_data,
    main,
    read_dcp_metadata_keys,
    validate_dcp_metadata_keys,
    validate_hf_export,
    validate_hf_export_documents,
)


def _optimizer_keys(prefix: str) -> list[str]:
    return [f"{prefix}.param", f"{prefix}.exp_avg", f"{prefix}.exp_avg_sq"]


def _hf_documents(*, layers: int = 3, mtp_layers: int = 0, include_mtp: bool = False):
    config = {
        "text_config": {
            "num_hidden_layers": layers,
            "mtp_num_hidden_layers": mtp_layers,
        }
    }
    keys = [f"model.language_model.layers.{layer}.self_attn.q_proj.weight" for layer in range(layers)]
    keys.append("lm_head.weight")
    if include_mtp:
        keys.append("model.language_model.mtp.layers.0.proj.weight")
    return config, {"weight_map": {key: "model-00001-of-00001.safetensors" for key in keys}}


def _write_safetensors_shard(path, keys, *, offsets_by_key=None, payload_size=None):
    offsets_by_key = offsets_by_key or {}
    header = {}
    next_offset = 0
    for key in keys:
        offsets = offsets_by_key.get(key, [next_offset, next_offset + 4])
        header[key] = {"dtype": "F32", "shape": [1], "data_offsets": offsets}
        next_offset = max(next_offset, offsets[1])
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    if payload_size is None:
        payload_size = next_offset
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * payload_size)


def test_dcp_optimizer_metadata_accepts_complete_distinct_triplets():
    keys = ["model.layers.0.weight"]
    keys += _optimizer_keys("chained_0.optimizer.distributed.bucket_idx_0")
    keys += _optimizer_keys("chained_1.optimizer.distributed.bucket_idx_0")

    summary = validate_dcp_metadata_keys(keys)

    assert summary.metadata_keys == 7
    assert summary.optimizer_state_prefixes == 2
    assert summary.param_keys == summary.exp_avg_keys == summary.exp_avg_sq_keys == 2


def test_dcp_exp_avg_sq_does_not_count_as_exp_avg():
    keys = [
        "optimizer.distributed.bucket.param",
        "optimizer.distributed.bucket.exp_avg_sq",
    ]

    with pytest.raises(CheckpointAcceptanceError, match=r"missing exact.*exp_avg"):
        validate_dcp_metadata_keys(keys)


def test_dcp_rejects_incomplete_bucket_triplets_with_prefix_example():
    keys = _optimizer_keys("optimizer.bucket0")
    keys += ["optimizer.bucket1.param", "optimizer.bucket1.exp_avg"]

    with pytest.raises(CheckpointAcceptanceError) as exc_info:
        validate_dcp_metadata_keys(keys)

    message = str(exc_info.value)
    assert "lack .exp_avg_sq" in message
    assert "optimizer.bucket1" in message


def test_dcp_storage_data_accepts_safe_existing_in_bounds_ranges(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    (shard_dir / "__0_0.distcp").write_bytes(b"0123456789abcdef")
    metadata = SimpleNamespace(
        storage_data={
            "first": SimpleNamespace(relative_path="shards/__0_0.distcp", offset=0, length=8),
            "second": SimpleNamespace(relative_path="shards/__0_0.distcp", offset=8, length=8),
        }
    )

    _validate_dcp_storage_data(tmp_path, metadata)


@pytest.mark.parametrize(
    "relative_path",
    ["../outside.distcp", "/tmp/outside.distcp", r"..\outside.distcp", r"C:\outside.distcp"],
)
def test_dcp_storage_data_rejects_unsafe_relative_paths(tmp_path, relative_path):
    metadata = SimpleNamespace(storage_data={"item": SimpleNamespace(relative_path=relative_path, offset=0, length=1)})

    with pytest.raises(CheckpointAcceptanceError, match="unsafe relative_path|escapes"):
        _validate_dcp_storage_data(tmp_path, metadata)


def test_dcp_storage_data_rejects_missing_files_and_out_of_bounds_ranges(tmp_path):
    missing = SimpleNamespace(
        storage_data={"item": SimpleNamespace(relative_path="missing.distcp", offset=0, length=1)}
    )
    with pytest.raises(CheckpointAcceptanceError, match="does not exist"):
        _validate_dcp_storage_data(tmp_path, missing)

    (tmp_path / "__0_0.distcp").write_bytes(b"12345678")
    out_of_bounds = SimpleNamespace(
        storage_data={"item": SimpleNamespace(relative_path="__0_0.distcp", offset=7, length=2)}
    )
    with pytest.raises(CheckpointAcceptanceError, match="exceeds its file"):
        _validate_dcp_storage_data(tmp_path, out_of_bounds)

    invalid = SimpleNamespace(storage_data={"item": SimpleNamespace(relative_path="__0_0.distcp", offset=-1, length=1)})
    with pytest.raises(CheckpointAcceptanceError, match="invalid byte range"):
        _validate_dcp_storage_data(tmp_path, invalid)


def test_read_dcp_metadata_keys_enforces_storage_validation(tmp_path, monkeypatch):
    import torch.distributed.checkpoint as dcp

    (tmp_path / ".metadata").write_bytes(b"fake metadata")
    shard = tmp_path / "__0_0.distcp"
    shard.write_bytes(b"01234567")
    metadata = SimpleNamespace(
        state_dict_metadata={"optimizer.bucket.param": object()},
        storage_data={"item": SimpleNamespace(relative_path=shard.name, offset=0, length=8)},
    )

    class FakeFileSystemReader:
        def __init__(self, _path):
            pass

        def read_metadata(self):
            return metadata

    monkeypatch.setattr(dcp, "FileSystemReader", FakeFileSystemReader)
    assert read_dcp_metadata_keys(tmp_path) == ("optimizer.bucket.param",)

    shard.unlink()
    with pytest.raises(CheckpointAcceptanceError, match="does not exist"):
        read_dcp_metadata_keys(tmp_path)


def test_hf_export_accepts_nested_qwen_config_with_mtp_stripped():
    config, index = _hf_documents()

    summary = validate_hf_export_documents(config, index)

    assert summary.transformer_layers == 3
    assert summary.layer_namespace == "model.language_model.layers."
    assert summary.lm_head_keys == 1
    assert summary.configured_mtp_layers == summary.mtp_weight_keys == 0


def test_hf_export_reports_missing_lm_head_and_configured_layers_together():
    config, index = _hf_documents(layers=4)
    del index["weight_map"]["lm_head.weight"]
    del index["weight_map"]["model.language_model.layers.2.self_attn.q_proj.weight"]

    with pytest.raises(CheckpointAcceptanceError) as exc_info:
        validate_hf_export_documents(config, index)

    message = str(exc_info.value)
    assert "missing configured layer indices [2]" in message
    assert "no lm_head weight key" in message


@pytest.mark.parametrize(
    ("mtp_layers", "include_mtp", "expected"),
    [
        (1, False, "declares 1 MTP layer"),
        (0, True, "config disables MTP"),
    ],
)
def test_hf_export_rejects_mtp_config_weight_mismatch(mtp_layers, include_mtp, expected):
    config, index = _hf_documents(mtp_layers=mtp_layers, include_mtp=include_mtp)

    with pytest.raises(CheckpointAcceptanceError, match=expected):
        validate_hf_export_documents(config, index)


def test_hf_export_accepts_enabled_mtp_when_weights_are_indexed():
    config, index = _hf_documents(mtp_layers=1, include_mtp=True)

    summary = validate_hf_export_documents(config, index)

    assert summary.configured_mtp_layers == 1
    assert summary.mtp_weight_keys == 1


def test_hf_export_reads_root_level_nextn_configuration():
    config = {"num_hidden_layers": 1, "num_nextn_predict_layers": 1}
    index = {
        "weight_map": {
            "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
            "lm_head.weight": "model.safetensors",
            "model.nextn_predict.layers.0.weight": "model.safetensors",
        }
    }

    summary = validate_hf_export_documents(config, index)

    assert summary.configured_mtp_layers == 1
    assert summary.mtp_weight_keys == 1


@pytest.mark.parametrize("filename", [123, "../outside.safetensors", "/tmp/model.safetensors"])
def test_hf_export_rejects_non_string_or_unsafe_shard_filenames(filename):
    config, index = _hf_documents()
    index["weight_map"]["lm_head.weight"] = filename

    with pytest.raises(CheckpointAcceptanceError, match="shard filename"):
        validate_hf_export_documents(config, index)


def test_hf_path_validation_rejects_missing_referenced_shards(tmp_path):
    config, index = _hf_documents()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(CheckpointAcceptanceError, match="missing 1 shard file"):
        validate_hf_export(tmp_path)


def test_hf_path_validation_rejects_empty_and_oversized_headers(tmp_path):
    config, index = _hf_documents()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.touch()

    with pytest.raises(CheckpointAcceptanceError, match="too short"):
        validate_hf_export(tmp_path)

    shard.write_bytes(struct.pack("<Q", MAX_SAFETENSORS_HEADER_BYTES + 1))
    with pytest.raises(CheckpointAcceptanceError, match="outside the allowed range"):
        validate_hf_export(tmp_path)


def test_hf_path_validation_matches_index_keys_to_shard_header(tmp_path):
    config, index = _hf_documents()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    expected_keys = set(index["weight_map"])
    omitted = "lm_head.weight"
    _write_safetensors_shard(
        tmp_path / "model-00001-of-00001.safetensors",
        (expected_keys - {omitted}) | {"unexpected.weight"},
    )

    with pytest.raises(CheckpointAcceptanceError) as exc_info:
        validate_hf_export(tmp_path)

    message = str(exc_info.value)
    assert f"header lacks 1 weight_map key(s); examples: {omitted}" in message
    assert "header contains 1 tensor key(s) not assigned" in message


def test_hf_path_validation_rejects_out_of_bounds_tensor_offsets(tmp_path):
    config, index = _hf_documents()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    keys = list(index["weight_map"])
    _write_safetensors_shard(
        tmp_path / "model-00001-of-00001.safetensors",
        keys,
        offsets_by_key={keys[0]: [0, 8]},
        payload_size=4,
    )

    with pytest.raises(CheckpointAcceptanceError, match="out-of-bounds data_offsets"):
        validate_hf_export(tmp_path)


def test_hf_path_validation_and_json_cli(tmp_path, capsys):
    config, index = _hf_documents()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    _write_safetensors_shard(tmp_path / "model-00001-of-00001.safetensors", index["weight_map"])

    assert validate_hf_export(tmp_path).indexed_weights == 4
    assert main(["--hf-export", str(tmp_path), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["errors"] == []


def test_cli_returns_failure_without_reading_tensor_payloads(tmp_path, capsys):
    config, index = _hf_documents()
    del index["weight_map"]["lm_head.weight"]
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    _write_safetensors_shard(tmp_path / "model-00001-of-00001.safetensors", index["weight_map"])

    assert main(["--hf-export", str(tmp_path)]) == 1
    assert "no lm_head weight key" in capsys.readouterr().err
