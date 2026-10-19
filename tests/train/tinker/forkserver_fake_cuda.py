"""Stands in for torch's CUDA check where an NVIDIA driver is installed, for the fork server's tests.

Imported, it does what ``FAKE_CUDA_AT_IMPORT`` says, as transformers asks ``torch.cuda.is_available()`` at import:

    is_available    torch's check as it behaves with the 595 driver: it starts the CUDA driver (``cuInit``, or
                    ``nvmlInit`` with ``PYTORCH_NVML_BASED_CUDA_CHECK=1``), which starts a thread of its own
                    (``cuda00001400006``) whether a GPU is visible or not, unless the environment both hides the GPUs
                    (``CUDA_VISIBLE_DEVICES=``) and asks NVML: torch then answers from the environment alone
    initialize      any other CUDA call: the driver's thread starts whatever the environment says
    (unset)         nothing

The driver's thread is a native one, which no ``threading.Thread`` stands for, and is named as the driver names it
before the call returns.
"""

from __future__ import annotations

import _thread
import os
import time

DRIVER_THREAD = "cuda00001400006"
#: What this module did at import: "environment" (answered from the environment) or "driver" (started its thread).
DID: list[str] = []


def _start_driver_thread() -> None:
    named = _thread.allocate_lock()
    named.acquire()

    def driver() -> None:
        with open(f"/proc/self/task/{_thread.get_native_id()}/comm", "w") as handle:
            handle.write(DRIVER_THREAD)
        named.release()
        time.sleep(3600)

    _thread.start_new_thread(driver, ())
    named.acquire()
    DID.append("driver")


def is_available() -> bool:
    if os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK") == "1" and os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        DID.append("environment")
    else:
        _start_driver_thread()
    return False


_AT_IMPORT = os.environ.get("FAKE_CUDA_AT_IMPORT", "")
if _AT_IMPORT == "is_available":
    is_available()
elif _AT_IMPORT == "initialize":
    _start_driver_thread()
