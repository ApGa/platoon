"""Keeps an inheritable descriptor open: not fork-safe, and one that an exec of the server would pass on to it."""

import os

FD = os.open(__file__, os.O_RDONLY)
os.set_inheritable(FD, True)
