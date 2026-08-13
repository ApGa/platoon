#!/usr/bin/env python3
"""Copy a verified node-local SIF and publish it create-only."""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import stat
from pathlib import Path

COPY_BUFFER_BYTES = 8 * 1024 * 1024


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


def validate_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("expected SHA-256 is not a lowercase digest")


def require_source(path: Path) -> tuple[int, os.stat_result]:
    if not path.is_absolute() or path.resolve() != path:
        raise RuntimeError(f"pending SIF path is not direct/canonical: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        value = os.fstat(descriptor)
        if (
            not stat.S_ISREG(value.st_mode)
            or value.st_uid != os.getuid()
            or stat.S_IMODE(value.st_mode) != 0o550
            or value.st_nlink != 1
        ):
            raise RuntimeError("pending SIF is not a single-link user-owned mode 0550 file")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, value


def sha256_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, COPY_BUFFER_BYTES):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def copy_descriptor(source: int, destination: int) -> None:
    while chunk := os.read(source, COPY_BUFFER_BYTES):
        view = memoryview(chunk)
        while view:
            written = os.write(destination, view)
            if written <= 0:
                raise RuntimeError("short write while staging derived SIF")
            view = view[written:]


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def unlink_if_identity(path: Path, identity: tuple[int, int]) -> bool:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return False
    if (value.st_dev, value.st_ino) != identity:
        return False
    path.unlink()
    return True


def create_staging(output: Path) -> tuple[Path, int, os.stat_result]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    for _ in range(128):
        staging = output.with_name(f".{output.name}.publish.{secrets.token_hex(16)}")
        try:
            descriptor = os.open(staging, flags, 0o600)
        except FileExistsError:
            continue
        value = os.fstat(descriptor)
        identity = (value.st_dev, value.st_ino)
        try:
            os.fchmod(descriptor, 0o600)
            value = os.fstat(descriptor)
            if (
                not stat.S_ISREG(value.st_mode)
                or value.st_uid != os.getuid()
                or stat.S_IMODE(value.st_mode) != 0o600
                or value.st_nlink != 1
            ):
                raise RuntimeError("shared staging SIF is not a single-link user-owned mode 0600 file")
        except BaseException:
            os.close(descriptor)
            unlink_if_identity(staging, identity)
            raise
        return staging, descriptor, value
    raise RuntimeError("could not create a unique shared staging SIF")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pending", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    pending = args.pending.absolute()
    output = args.output.absolute()
    expected_sha256 = args.expected_sha256
    validate_sha256(expected_sha256)
    validate_directory(output.parent)
    if output.parent.resolve() != output.parent:
        raise RuntimeError("output path is indirect")
    if os.path.lexists(output):
        raise RuntimeError(f"refusing existing output SIF: {output}")

    source, pending_stat = require_source(pending)
    staging: Path | None = None
    staging_identity: tuple[int, int] | None = None
    staging_created = False
    output_identity: tuple[int, int] | None = None
    publication_complete = False
    try:
        if sha256_descriptor(source) != expected_sha256:
            raise RuntimeError("pending SIF digest differs from expected SHA-256")
        source_after_hash = os.fstat(source)
        if (
            source_after_hash.st_dev != pending_stat.st_dev
            or source_after_hash.st_ino != pending_stat.st_ino
            or source_after_hash.st_size != pending_stat.st_size
            or source_after_hash.st_mtime_ns != pending_stat.st_mtime_ns
        ):
            raise RuntimeError("pending SIF identity changed while hashing")

        staging, staging_descriptor, staging_stat = create_staging(output)
        staging_identity = (staging_stat.st_dev, staging_stat.st_ino)
        staging_created = True
        try:
            copy_descriptor(source, staging_descriptor)
            os.fsync(staging_descriptor)
            os.fchmod(staging_descriptor, 0o550)
            os.fsync(staging_descriptor)
        finally:
            os.close(staging_descriptor)
        os.close(source)
        source = -1

        staged_descriptor, staged_stat = require_source(staging)
        try:
            if (staged_stat.st_dev, staged_stat.st_ino) != staging_identity:
                raise RuntimeError("shared staging SIF identity changed before verification")
            if sha256_descriptor(staged_descriptor) != expected_sha256:
                raise RuntimeError("shared staging SIF digest differs from verified local SIF")
        finally:
            os.close(staged_descriptor)
        if staged_stat.st_size != pending_stat.st_size:
            raise RuntimeError("shared staging SIF size differs from verified local SIF")
        current_staged = staging.lstat()
        if (
            (current_staged.st_dev, current_staged.st_ino) != staging_identity
            or stat.S_IMODE(current_staged.st_mode) != 0o550
            or current_staged.st_nlink != 1
        ):
            raise RuntimeError("shared staging SIF identity changed before publication")

        try:
            os.link(staging, output, follow_symlinks=False)
        except FileExistsError as exc:
            raise RuntimeError("output SIF appeared during atomic publication") from exc
        output_identity = staging_identity
        linked_stat = output.lstat()
        if (
            not stat.S_ISREG(linked_stat.st_mode)
            or linked_stat.st_uid != os.getuid()
            or stat.S_IMODE(linked_stat.st_mode) != 0o550
            or linked_stat.st_ino != staged_stat.st_ino
            or linked_stat.st_dev != staged_stat.st_dev
            or linked_stat.st_nlink != 2
        ):
            raise RuntimeError("published SIF identity/link count mismatch")
        fsync_directory(output.parent)
        if not unlink_if_identity(staging, staging_identity):
            raise RuntimeError("shared staging SIF identity changed after publication")
        staging_created = False
        fsync_directory(output.parent)
        final_stat = output.lstat()
        if (
            not stat.S_ISREG(final_stat.st_mode)
            or final_stat.st_uid != os.getuid()
            or stat.S_IMODE(final_stat.st_mode) != 0o550
            or final_stat.st_nlink != 1
            or final_stat.st_dev != staged_stat.st_dev
            or final_stat.st_ino != staged_stat.st_ino
        ):
            raise RuntimeError("published SIF final identity mismatch")
        final_descriptor, _ = require_source(output)
        try:
            if sha256_descriptor(final_descriptor) != expected_sha256:
                raise RuntimeError("published SIF digest mismatch")
        finally:
            os.close(final_descriptor)
        publication_complete = True
    finally:
        if source >= 0:
            os.close(source)
        changed_directory = False
        if not publication_complete and output_identity is not None:
            changed_directory = unlink_if_identity(output, output_identity) or changed_directory
        if staging_created and staging is not None and staging_identity is not None:
            changed_directory = unlink_if_identity(staging, staging_identity) or changed_directory
        if changed_directory:
            fsync_directory(output.parent)
    print(f"toolathlon_pbs_image_published={output}")


if __name__ == "__main__":
    main()
