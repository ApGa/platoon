"""Moved: platoon 0.5.0 moved this module to :mod:`platoon.train.workers.entrypoint`.  Its names still import from
here, each with a ``PlatoonDeprecationWarning``, until platoon 0.6.0 removes this module.  ``python -m
platoon.train.tinker.workers.entrypoint`` still runs a rollout worker, with the same warning."""

from platoon.train.tinker._moved import moved as _moved

__getattr__, __dir__ = _moved(
    globals(), "platoon.train.workers.entrypoint", name="platoon.train.tinker.workers.entrypoint"
)


if __name__ == "__main__":
    import sys

    from platoon.train.tinker._moved import warn_run
    from platoon.train.workers.entrypoint import main

    warn_run("platoon.train.tinker.workers.entrypoint", "platoon.train.workers.entrypoint")
    sys.exit(main())
