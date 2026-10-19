"""A module that starts a thread when imported: a fork server that imports it must not fork."""

from __future__ import annotations

import threading
import time

threading.Thread(target=time.sleep, args=(3600,), name="stray-import-thread", daemon=True).start()
