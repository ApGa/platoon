"""Moved: platoon 0.5.0 moved this module to :mod:`platoon.train.launch`.  Its names still import from here, each
with a ``PlatoonDeprecationWarning``, until platoon 0.6.0 removes this module.  ``python -m
platoon.train.tinker.train`` still trains, with the same warning."""

from platoon.train.tinker._moved import moved as _moved

__getattr__, __dir__ = _moved(globals(), "platoon.train.launch", name="platoon.train.tinker.train")


if __name__ == "__main__":
    from platoon.train.launch import run_tinker_training
    from platoon.train.tinker._moved import warn_run

    warn_run("platoon.train.tinker.train", "platoon.train.launch")
    run_tinker_training()
