"""Starts a thread that ends at once: fork-safe once it has ended."""

import threading

threading.Thread(target=lambda: None, name="preload-stub-short").start()
