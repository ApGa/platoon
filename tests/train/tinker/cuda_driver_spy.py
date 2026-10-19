"""Imports modules with spies on the calls through which torch reaches the CUDA driver, for the fork server's tests.

``python cuda_driver_spy.py <module>...`` imports torch, wraps the calls that reach the driver, imports the modules,
asks ``torch.cuda.is_available()`` and ``torch.cuda.device_count()``, and prints one JSON line: the wrapped calls made,
in order, and the two answers.  The calls:

    _cuda_getDeviceCount, _cuda_init                ``cudaGetDeviceCount`` and lazy initialization: ``cuInit``
    _raw_device_count_nvml, _raw_device_uuid_nvml   ``nvmlInit`` (``PYTORCH_NVML_BASED_CUDA_CHECK=1``, device_count)

Wherever an NVIDIA driver is installed, each of them starts a thread of the driver's (with driver 595, ``nvmlInit``
too), in the process that makes it.  The spies see the calls on a machine without a driver as well.
"""

from __future__ import annotations

import importlib
import json
import sys

import torch

CALLS: list[str] = []


def _spy(owner: object, name: str) -> None:
    original = getattr(owner, name)

    def spied(*args, **kwargs):
        CALLS.append(name)
        return original(*args, **kwargs)

    setattr(owner, name, spied)


def main(modules: list[str]) -> None:
    for name in ("_cuda_getDeviceCount", "_cuda_init"):
        _spy(torch._C, name)
    for name in ("_raw_device_count_nvml", "_raw_device_uuid_nvml"):
        _spy(torch.cuda, name)
    for module in modules:
        importlib.import_module(module)
    answers = [torch.cuda.is_available(), torch.cuda.device_count()]
    print(json.dumps({"calls": CALLS, "answers": answers}))


if __name__ == "__main__":
    main(sys.argv[1:])
