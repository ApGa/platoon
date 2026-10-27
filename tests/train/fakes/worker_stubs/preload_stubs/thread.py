"""Starts a thread that runs on: not fork-safe."""

import threading
import time

threading.Thread(target=time.sleep, args=(3600,), name="preload-stub-thread", daemon=True).start()
