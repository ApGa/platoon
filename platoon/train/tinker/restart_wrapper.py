"""Removed: platoon 0.3.0 removed ``python -m platoon.train.tinker.restart_wrapper`` and ``run_with_restart``, which
restarted a training command after the watchdog's exit only.  Set ``watchdog.max_restarts`` in the training config
instead, and ``platoon train`` restarts training after any failure (:mod:`platoon.train.supervise`).

What is left says so: run as a module, it prints what replaces it and exits 78, a configuration error, which no
supervisor restarts; imported, it raises ``ImportError`` with the same words.
"""

import sys

from platoon.deprecations import removed_hint

#: What the removed wrapper's users see, with what replaces it.
REMOVED = "python -m platoon.train.tinker.restart_wrapper (run_with_restart): " + removed_hint(
    "0.3.0",
    "watchdog.max_restarts in the training config, with which platoon train restarts training after any failure",
)

if __name__ == "__main__":
    print(f"platoon: {REMOVED}", file=sys.stderr)
    sys.exit(78)
raise ImportError(REMOVED)
