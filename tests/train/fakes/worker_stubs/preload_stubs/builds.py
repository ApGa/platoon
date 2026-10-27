"""Builds a fork server makes ahead of its workers (``prepare``): one safe, one that starts a thread."""

import threading
import time

BUILT: dict[str, str] = {}


def remember(value: str) -> None:
    BUILT["value"] = value


def start_a_thread() -> None:
    threading.Thread(target=time.sleep, args=(3600,), name="preload-stub-build", daemon=True).start()
