"""Lightweight entrypoint for forked AReaL proxy rollout workers.

This module loads Platoon's proxy-specific compatibility patches directly,
without importing ``platoon.train.areal`` and its training stack.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _apply_platoon_areal_proxy_patches() -> None:
    patches_path = Path(__file__).resolve().parent / "train" / "areal" / "patches.py"
    spec = importlib.util.spec_from_file_location(
        "platoon.train.areal.patches",
        patches_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Platoon AReaL patches from {patches_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_proxy_patches()


_apply_platoon_areal_proxy_patches()

main = importlib.import_module(
    "areal.experimental.openai.proxy.proxy_rollout_server"
).main


if __name__ == "__main__":
    main()
