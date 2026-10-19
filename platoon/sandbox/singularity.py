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

A sandbox's commands run as your own user, unless ``fakeroot: true`` (where the installed hpc-sandbox takes it,
0.4 on; ``extra_start_args: ["--fakeroot"]`` means the same) starts its instance with Singularity's ``--fakeroot``,
on the copy layout: they then run as root of a user namespace that maps root to your uid and your subordinate ids
(from ``/etc/subuid``) to the users after it, so a command can switch users as in Docker.  What those users write
belongs to your subordinate ids on the host, which only an hpc-sandbox that hands such files back before it deletes
a sandbox (0.4 on) can delete: :meth:`SingularityProvider.root_refusal` refuses root sandboxes on any other.
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
        if "fakeroot" in options and _hpc_sandbox_has_fakeroot() is False:  # it would fail to build in every rollout
            return _NO_FAKEROOT
        layout = explicit or os.environ.get("HPC_SANDBOX_ROOTFS_LAYOUT") or "copy"
        if layout not in ROOTFS_LAYOUTS:
            return f"rootfs_layout must be one of {', '.join(ROOTFS_LAYOUTS)}, not {layout!r}"
        if layout == "copy":
            return None
        if _hpc_sandbox_has_layouts() is False:
            return None  # it ignores the job's HPC_SANDBOX_ROOTFS_LAYOUT (an explicit rootfs_layout is refused above)
        # As hpc-sandbox (0.3 on) chooses: what the layout cannot take refuses a rootfs_layout the options give, even an
        # empty one, and leaves a provider on copies when the layout is the job's default.
        if options.get("gpu") or options.get("cgroup_limits") or _fakeroot(options):
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

    @classmethod
    def root_refusal(cls, options: Mapping[str, Any]) -> str | None:
        """Why a provider built with ``options`` could not run a sandbox's commands as root with other users to
        switch to (:func:`platoon.sandbox.root_refusal`), read from the options without building it; None when it
        can.  Only ``--fakeroot`` gives a sandbox root and your subordinate ids, and only on the copy layout; what
        those other users write belongs to your subordinate ids on the host, so only an hpc-sandbox that hands their
        files back before it deletes a sandbox (0.4 on: one whose provider takes ``fakeroot``) may run it."""

        if not _fakeroot(options):
            return (
                "the singularity provider runs a sandbox's commands as your own user, who cannot switch users, unless "
                "its options set fakeroot: true (Singularity's --fakeroot, on the copy layout)"
            )
        if _hpc_sandbox_has_fakeroot() is False:
            return _NO_FAKEROOT
        if options.get("rootfs_layout") == "fuse-overlay":
            return "the fuse-overlay layout runs unprivileged instances: --fakeroot needs rootfs_layout: copy"
        return None


# Why an hpc-sandbox whose Singularity provider takes no fakeroot option cannot run sandboxes whose root switches users,
# whichever way the options ask for --fakeroot.
_NO_FAKEROOT = (
    "the installed hpc-sandbox takes no fakeroot option: it cannot delete what the other users of a --fakeroot sandbox "
    "write, which belongs to your subordinate ids on the host, so each such sandbox would leave its files on the node. "
    "Install hpc-sandbox 0.4 (the hpc extra's), whose singularity provider takes the fakeroot option and hands those "
    "files back before it deletes a sandbox"
)


def _fakeroot(options: Mapping[str, Any]) -> bool:
    """Whether a provider built with ``options`` starts its instances with ``--fakeroot``: ``fakeroot: true``, or the
    flag in ``extra_start_args``."""

    return bool(options.get("fakeroot")) or any(
        arg in ("--fakeroot", "-f") for arg in options.get("extra_start_args") or ()
    )


def _hpc_sandbox_has_fakeroot() -> bool | None:
    """Whether the installed hpc-sandbox's Singularity provider takes ``fakeroot`` (0.4 on), which comes with handing
    back what the other users of a ``--fakeroot`` sandbox write before the sandbox is deleted; None when that cannot
    be told (no hpc-sandbox: building the provider says so)."""

    return _takes("fakeroot")


def _hpc_sandbox_has_layouts() -> bool | None:
    """Whether the installed hpc-sandbox's Singularity provider takes ``rootfs_layout``; None when that cannot be
    told (no hpc-sandbox: building the provider says so)."""

    return _takes("rootfs_layout")


def _takes(option: str) -> bool | None:
    """Whether the installed hpc-sandbox's Singularity provider takes ``option``; None when that cannot be told."""

    try:
        provider = importlib.import_module("hpc_sandbox").SingularityProvider
        return option in inspect.signature(provider).parameters
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


__all__ = ["ROOTFS_LAYOUTS", "SingularityProvider"]
