"""Platoon entrypoint for AReaL proxy rollout workers.

Forked proxy workers run ``python -m <module>``. Platoon redirects the default
AReaL module to this entrypoint so compatibility patches are applied before the
proxy server imports the OpenAI client stack.
"""

from __future__ import annotations

from platoon.train.areal.patches import apply_all_patches

apply_all_patches()

from areal.experimental.openai.proxy.proxy_rollout_server import main  # pyright: ignore[reportMissingImports]

if __name__ == "__main__":
    main()
