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

_upstream_main = importlib.import_module("areal.experimental.openai.proxy.proxy_rollout_server").main


def _reset_name_resolve() -> None:
    """Remove this fork child's discovery records before it exits."""

    importlib.import_module("areal.utils.name_resolve").reset()


def main():
    """Delegate unchanged and clean this proxy's discovery records on exit."""

    try:
        result = _upstream_main()
    except BaseException as primary_error:
        try:
            _reset_name_resolve()
        except BaseException as reset_error:
            # Cleanup failure must not replace the exception that stopped the
            # proxy. Preserve it as diagnostic context on Python 3.11+.
            primary_error.add_note(
                f"Platoon proxy name-resolution cleanup also failed: {type(reset_error).__name__}: {reset_error}"
            )
        raise
    _reset_name_resolve()
    return result


main.__platoon_name_resolve_cleanup__ = True


if __name__ == "__main__":
    main()
