"""A module that takes ``FORKSERVER_SLOW_IMPORT_SECONDS`` to import, as the worker's stack can on a node whose CPUs are
all busy: a fork server that imports it says hello that much later."""

from __future__ import annotations

import os
import time

time.sleep(float(os.environ.get("FORKSERVER_SLOW_IMPORT_SECONDS", "0")))
