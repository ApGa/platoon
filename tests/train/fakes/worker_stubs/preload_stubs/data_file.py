"""Keeps a data file open, read only, whose name holds ".so" (``tasks.solutions.jsonl``): not fork-safe (the workers
would share its offset); only a shared object's descriptor is."""

import os
from pathlib import Path

PATH = Path(os.environ["PRELOAD_STUB_PIDS"]).with_name("tasks.solutions.jsonl")
PATH.write_text('{"task": 0}\n{"task": 300}\n')
HANDLE = open(PATH, "rb")  # noqa: SIM115
