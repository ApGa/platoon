"""Sets an environment variable: not fork-safe (each worker's own environment replaces the server's, so a worker
forked from it would not have the variable that an exec'd worker's import sets)."""

import os

os.environ["PRELOAD_STUB_FLAG"] = "set at import"
