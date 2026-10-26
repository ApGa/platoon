"""The enroot provider: rootless enroot containers through ``hpc-sandbox``, for NVIDIA sites (Slurm with pyxis).

An adapter over ``hpc_sandbox.EnrootProvider`` (:mod:`platoon.sandbox.hpc`), like the singularity provider: one
long-lived container per sandbox in its own user, mount, PID and IPC namespaces, which each command enters;
``docker://`` references, digest pins (``docker://repo@sha256:<hex>``, which hpc-sandbox imports as
``repo:sha256:<hex>``) included, or local ``.sqsh`` images.  Harbor environments select it with
``provider: enroot``.

Its options pass through to ``hpc_sandbox.EnrootProvider`` (``scratch_root`` on node-local storage,
``image_cache``, ``gpus``, ``remap_root``, ...), which also takes the singularity provider's names, so a
``provider_options`` block written for a Singularity site serves an enroot site too, except ``binary`` and
``extra_start_args``, which each runtime reads as its own.  ``root`` is one more name for ``remap_root``, and
``rootfs_layout``, whose layouts only the singularity provider has, is ignored (:func:`enroot_options`).

Multi-container stacks run as hpc-sandbox pods here too (:mod:`platoon.sandbox.hpc_stacks`), with hpc-sandbox 0.5:
enroot maps one id into a container, so their sidecars run as your own uid, never as a root that switches users.
enroot pods are experimental until a site runs platoon's pod tests with them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from platoon.sandbox.hpc import HpcProvider

# The singularity provider's option that enroot has no use for: ``rootfs_layout`` (enroot creates each sandbox's root
# filesystem its own way).  hpc-sandbox's EnrootProvider takes it from 0.3 on, but refuses ``fuse-overlay``, which a
# block shared with a Singularity site may set.
_IGNORED = ("rootfs_layout",)


def enroot_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """``hpc_sandbox.EnrootProvider`` options for a ``provider_options`` block: the block as it is, with ``root``,
    another name for ``remap_root`` (uid 0 inside, your uid outside; the default), renamed, and without
    ``rootfs_layout``.  ``root`` is an enroot-only alias: the singularity provider does not take it.

    The singularity provider's names pass through, since ``hpc_sandbox.EnrootProvider`` takes them with the same
    meaning: ``gpu: nv`` is ``gpus: all`` (every GPU this process may use; ``gpus`` also takes a list such as
    ``0,1``), ``allow_network_none: false`` refuses ``network: none`` (which enroot otherwise always offers: every
    sandbox has its own network namespace), ``bind_resolv_conf`` and ``cgroup_limits: false`` mean what they mean
    there, and ``stop_grace_s`` is ignored (a sandbox stops by SIGKILL).  ``rootfs_layout``, whose ``fuse-overlay``
    hpc-sandbox's ``EnrootProvider`` refuses, is the one singularity name left out, so a block shared with a Singularity
    site may set it.  What enroot cannot honor raises ``ValueError`` when the provider is built: in hpc-sandbox,
    ``gpu: rocm``, ``cgroup_limits: true`` (``cpus`` stays advisory, ``memory_mb`` arms the memory watchdog),
    ``fakeroot: true`` (0.4 on: enroot maps only your uid, so root there cannot switch users) and both ``gpu`` and
    ``gpus`` set; here, both ``root`` and ``remap_root`` given.
    """

    translated = dict(options)
    for name in _IGNORED:
        translated.pop(name, None)
    if "root" in translated:
        if "remap_root" in translated:
            raise ValueError("Set remap_root or root for the enroot provider, not both")
        translated["remap_root"] = bool(translated.pop("root"))
    return translated


class EnrootProvider(HpcProvider):
    name = "enroot"
    inner_class = "EnrootProvider"
    executables = ("enroot",)

    def __init__(self, **options: Any) -> None:
        super().__init__(**enroot_options(options))

    @classmethod
    def network_none_refusal(cls, options: Mapping[str, Any]) -> str | None:
        """hpc-sandbox's EnrootProvider offers ``network: none`` unless ``allow_network_none`` is false, which a
        ``provider_options`` block shared with a Singularity site may say; then it refuses every such sandbox."""

        if options.get("allow_network_none") is not False:
            return None
        return (
            "the enroot provider starts no sandbox without a network while its provider_options set "
            "allow_network_none: false (enroot needs no site permission for one: set it to true or leave it out)"
        )

    @classmethod
    def root_refusal(cls, options: Mapping[str, Any]) -> str | None:
        """enroot maps a sandbox's root (``remap_root``) to your uid and maps no other user: a command there runs as
        root but cannot switch to another user (:func:`platoon.sandbox.root_refusal`)."""

        return (
            "the enroot provider maps only your own uid into a sandbox (as root, with remap_root), so a command "
            "there cannot switch to another user"
        )


__all__ = ["EnrootProvider", "enroot_options"]
