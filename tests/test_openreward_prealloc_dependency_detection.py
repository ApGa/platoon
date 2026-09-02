from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_LAUNCHER = REPO_ROOT / "slurm-scripts/openreward-toolathlon-prealloc-base.sh"
LORA_LAUNCHER = REPO_ROOT / (
    "slurm-scripts/"
    "openreward-toolathlon-prealloc-32node-ptc-recursive-bs8-"
    "behavior-gated-lora-all-layers-r32.sh"
)


def _detect_binary_dependency_flags(
    tmp_path: Path,
    config_text: str,
    *,
    initial_te: str | None = None,
    initial_apex: str | None = None,
) -> tuple[str, str]:
    """Execute the launcher's real dependency-detection block in isolation."""
    source = BASE_LAUNCHER.read_text()
    section_start = source.index("# --- Megatron / Transformer Engine")
    block_start = source.index("if grep -qiE", section_start)
    block_end = source.index("# Snapshot local package sources", block_start)
    detection_block = source[block_start:block_end]

    config = tmp_path / "config.yaml"
    config.write_text(config_text)
    env = os.environ.copy()
    if initial_te is None:
        env.pop("OPENREWARD_BUILD_TE", None)
    else:
        env["OPENREWARD_BUILD_TE"] = initial_te
    if initial_apex is None:
        env.pop("OPENREWARD_BUILD_APEX", None)
    else:
        env["OPENREWARD_BUILD_APEX"] = initial_apex

    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail\n"
            "CONFIG=$1\n"
            f"{detection_block}\n"
            "printf '%s\\n%s\\n' \"$OPENREWARD_BUILD_TE\" \"$OPENREWARD_BUILD_APEX\"",
            "bash",
            str(config),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )
    te, apex = result.stdout.splitlines()
    return te, apex


def test_detects_quoted_megatron_backend(tmp_path: Path) -> None:
    flags = _detect_binary_dependency_flags(
        tmp_path,
        'actor:\n  backend: "megatron:(attn:d10p2t4c2|ffn:d10p2t1e8)"\n',
    )
    assert flags == ("1", "1")


def test_detects_single_quoted_and_unquoted_megatron_backends(tmp_path: Path) -> None:
    single_quoted = _detect_binary_dependency_flags(
        tmp_path,
        "actor:\n  backend: 'megatron:d4p2t8'\n",
    )
    unquoted = _detect_binary_dependency_flags(
        tmp_path,
        "actor:\n  backend: megatron:d4p2t8\n",
    )
    assert single_quoted == ("1", "1")
    assert unquoted == ("1", "1")


def test_does_not_enable_megatron_dependencies_for_sglang(tmp_path: Path) -> None:
    flags = _detect_binary_dependency_flags(
        tmp_path,
        "rollout:\n  backend: sglang:d12p1t8\n",
    )
    assert flags == ("0", "0")


def test_generic_launcher_preserves_explicit_dependency_overrides(tmp_path: Path) -> None:
    flags = _detect_binary_dependency_flags(
        tmp_path,
        'actor:\n  backend: "megatron:d4p2t8"\n',
        initial_te="0",
        initial_apex="0",
    )
    assert flags == ("0", "0")


def test_all_layer_lora_wrapper_forces_required_binary_dependencies() -> None:
    source = LORA_LAUNCHER.read_text()
    assert "export OPENREWARD_BUILD_TE=1" in source
    assert "export OPENREWARD_BUILD_APEX=1" in source
