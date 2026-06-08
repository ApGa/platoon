"""Lightweight entrypoint for forked AReaL proxy rollout workers.

Loads ``platoon.train.areal.patches`` directly so we do not import
``platoon.train.areal`` (which pulls in the FSDP training stack).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _apply_platoon_areal_patches() -> None:
    patches_path = Path(__file__).resolve().parent / "train" / "areal" / "patches.py"
    spec = importlib.util.spec_from_file_location(
        "platoon.train.areal.patches",
        patches_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Platoon AReaL patches from {patches_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_all_patches()


_apply_platoon_areal_patches()

from areal.experimental.openai.proxy.proxy_rollout_server import main  # pyright: ignore[reportMissingImports]

if __name__ == "__main__":
    main()
