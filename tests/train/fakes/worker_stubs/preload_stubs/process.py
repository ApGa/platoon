"""Starts a process that runs on: not fork-safe.  Its pid goes to ``$PRELOAD_STUB_PIDS``, for the test to end it."""

import os
import subprocess

PROCESS = subprocess.Popen(["sleep", "3600"])
if os.environ.get("PRELOAD_STUB_PIDS"):
    with open(os.environ["PRELOAD_STUB_PIDS"], "a") as handle:
        handle.write(f"{PROCESS.pid}\n")
