"""Moved: platoon 0.5.0 moved this module to :mod:`platoon.train.workflow.start_limit`.  Its names still import from
here, each with a ``PlatoonDeprecationWarning``, until platoon 0.6.0 removes this module."""

from platoon.train.tinker._moved import moved as _moved

__getattr__, __dir__ = _moved(globals(), "platoon.train.workflow.start_limit")
