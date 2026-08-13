#!/usr/bin/env python3
"""Atomically hard-link a verified pending SIF into its final path."""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path


def validate_directory(path: Path) -> None:
    if not path.is_absolute() or path.resolve() != path:
        raise RuntimeError(f"output directory is not direct/canonical: {path}")
    value = path.lstat()
    mode = stat.S_IMODE(value.st_mode)
    groups = set(os.getgroups()) | {os.getgid()}
    allowed = (
        stat.S_ISDIR(value.st_mode)
        and value.st_uid == os.getuid()
        and ((mode == 0o700 and value.st_gid == os.getgid()) or (mode == 0o2770 and value.st_gid in groups))
    )
    if not allowed:
        raise RuntimeError(
            f"output directory must be user-owned mode 0700 or a user-owned "
            f"setgid mode 2770 directory for one of the user's groups: {path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pending", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pending = args.pending.absolute()
    output = args.output.absolute()
    validate_directory(output.parent)
    if pending.parent != output.parent:
        raise RuntimeError("pending and output SIF are not in the same directory")
    if pending.resolve() != pending or output.parent.resolve() != output.parent:
        raise RuntimeError("pending/output paths are indirect")
    pending_stat = pending.lstat()
    if (
        not stat.S_ISREG(pending_stat.st_mode)
        or pending_stat.st_uid != os.getuid()
        or stat.S_IMODE(pending_stat.st_mode) != 0o550
        or pending_stat.st_nlink != 1
    ):
        raise RuntimeError("pending SIF is not a single-link user-owned mode 0550 file")
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing existing output SIF: {output}")
    try:
        os.link(pending, output, follow_symlinks=False)
    except FileExistsError as exc:
        raise RuntimeError("output SIF appeared during atomic publication") from exc
    directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    output_stat = output.lstat()
    if (
        not stat.S_ISREG(output_stat.st_mode)
        or output_stat.st_ino != pending_stat.st_ino
        or output_stat.st_dev != pending_stat.st_dev
        or output_stat.st_nlink != 2
    ):
        raise RuntimeError("published SIF identity/link count mismatch")
    pending.unlink()
    directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    final_stat = output.lstat()
    if final_stat.st_nlink != 1:
        raise RuntimeError("published SIF did not return to a single link")
    print(f"toolathlon_pbs_image_published={output}")


if __name__ == "__main__":
    main()
