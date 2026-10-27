"""Keeps a file open: not fork-safe (every worker would share its offset)."""

HANDLE = open(__file__, "rb")  # noqa: SIM115
