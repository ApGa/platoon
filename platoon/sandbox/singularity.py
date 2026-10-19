"""The singularity provider: rootless SingularityCE or Apptainer instances through ``hpc-sandbox``.

An adapter over ``hpc_sandbox.SingularityProvider`` (:mod:`platoon.sandbox.hpc`): private PID and IPC
namespaces, root filesystem, ``/tmp`` and environment; ``docker://`` references or local ``.sif`` images.
Its options (``scratch_root``, ``image_cache``, ``allow_network_none``, ``gpu``, ``rootfs_layout``, ...) pass
through.  Without ``allow_network_none: true`` hpc-sandbox refuses every sandbox without a network, which
:meth:`SingularityProvider.network_none_refusal` says before any is started.

``rootfs_layout`` (or ``HPC_SANDBOX_ROOTFS_LAYOUT`` for a whole job) picks how hpc-sandbox gives each sandbox its
root filesystem, where the installed hpc-sandbox has layouts: ``copy`` (the default) extracts the image into a
private directory per sandbox; ``fuse-overlay`` mounts a per-sandbox writable layer over one extraction per node
with ``fuse-overlayfs`` and runs the instance unprivileged, so a start costs about a second whatever the image's
size.  :meth:`SingularityProvider.rootfs_layout_refusal` says before any sandbox starts what would stop a layout.
"""

from __future__ import annotations

import importlib
import inspect
import os
import shutil
from collections.abc import Mapping
from typing import Any

from platoon.sandbox.hpc import HpcProvider

#: The root filesystem layouts of hpc-sandbox's Singularity provider (``rootfs_layout``).
ROOTFS_LAYOUTS = ("copy", "fuse-overlay")


class SingularityProvider(HpcProvider):
    name = "singularity"
    inner_class = "SingularityProvider"
    executables = ("singularity", "apptainer")

    @classmethod
    def network_none_refusal(cls, options: Mapping[str, Any]) -> str | None:
        if options.get("allow_network_none"):
            return None
        return (
            "the singularity provider starts a sandbox without a network only with allow_network_none: true in its "
            "provider_options (setuid Singularity reserves network namespaces for root unless the site permits them)"
        )

    @classmethod
    def rootfs_layout_refusal(cls, options: Mapping[str, Any]) -> str | None:
        """Why a provider built with ``options`` could not give its sandboxes the root filesystem layout they ask for
        (``rootfs_layout``, else ``HPC_SANDBOX_ROOTFS_LAYOUT``), read from the options and this host without
        building it; None when nothing stops it.  An hpc-sandbox without layouts takes no ``rootfs_layout`` at all,
        not even ``copy``: the provider would fail to build in every rollout."""

        explicit = options.get("rootfs_layout")
        if "rootfs_layout" in options and _hpc_sandbox_has_layouts() is False:
            return "the installed hpc-sandbox has no rootfs layouts: leave rootfs_layout out, or install one that has"
        layout = explicit or os.environ.get("HPC_SANDBOX_ROOTFS_LAYOUT") or "copy"
        if layout not in ROOTFS_LAYOUTS:
            return f"rootfs_layout must be one of {', '.join(ROOTFS_LAYOUTS)}, not {layout!r}"
        if layout == "copy":
            return None
        if _hpc_sandbox_has_layouts() is False:
            return None  # it ignores the job's HPC_SANDBOX_ROOTFS_LAYOUT (an explicit rootfs_layout is refused above)
        # As hpc-sandbox (0.3) chooses: what the layout cannot take refuses a rootfs_layout the options give, even an
        # empty one, and leaves a provider on copies when the layout is the job's default.
        if (
            options.get("gpu")
            or options.get("cgroup_limits")
            or any(arg in ("--fakeroot", "-f") for arg in options.get("extra_start_args") or ())
        ):
            conflict = "runs unprivileged instances: GPUs, cgroup limits and --fakeroot need copy"
        elif options.get("rootfs_templates") is False:
            conflict = "lies over the node's template of the image: leave rootfs_templates on"
        else:
            conflict = None
        if conflict:
            return f"the fuse-overlay layout {conflict}" if explicit is not None else None
        missing = [tool for tool in ("fuse-overlayfs",) if shutil.which(tool) is None]
        if shutil.which("fusermount3") is None and shutil.which("fusermount") is None:
            missing.append("fusermount3")
        if missing:
            return f"the fuse-overlay layout needs {' and '.join(missing)} on PATH"
        return None


def _hpc_sandbox_has_layouts() -> bool | None:
    """Whether the installed hpc-sandbox's Singularity provider takes ``rootfs_layout``; None when that cannot be
    told (no hpc-sandbox: building the provider says so)."""

    try:
        provider = importlib.import_module("hpc_sandbox").SingularityProvider
        return "rootfs_layout" in inspect.signature(provider).parameters
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


__all__ = ["ROOTFS_LAYOUTS", "SingularityProvider"]
