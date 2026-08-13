"""Lightweight AReaL RPC-server entrypoint for Platoon rollout workers.

Load worker-only compatibility patches directly without importing
``platoon.train.areal`` and its controller/Megatron training stack, then
delegate argument parsing and serving to AReaL unchanged.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path


def _apply_platoon_areal_rollout_worker_patches() -> None:
    patches_path = Path(__file__).resolve().parent / "train" / "areal" / "patches.py"
    spec = importlib.util.spec_from_file_location(
        "platoon.train.areal.patches",
        patches_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Platoon AReaL patches from {patches_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_rollout_worker_patches()


def main() -> None:
    _apply_platoon_areal_rollout_worker_patches()
    upstream = importlib.import_module("areal.infra.rpc.rpc_server")
    upstream.main()


if __name__ == "__main__":
    main()
