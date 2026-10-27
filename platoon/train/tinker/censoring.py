"""Moved: platoon 0.5.0 moved this module into :mod:`platoon.train.step.metrics`, beside the step's other metrics.  Its
names still import from here, and so do the names it imported, each with a ``PlatoonDeprecationWarning`` that names
its new module, until platoon 0.6.0 removes this module."""

from platoon.train.tinker._moved import moved as _moved

__getattr__, __dir__ = _moved(
    globals(),
    {
        "_Counts": "platoon.train.step.metrics",
        "Censoring": "platoon.train.step.metrics",
    },
    homes=("platoon.train.step.metrics",),
)
