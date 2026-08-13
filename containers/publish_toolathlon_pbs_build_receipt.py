#!/usr/bin/env python3
"""Verify a derived Toolathlon SIF and publish its create-only build receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

BASE_SIF_SHA256 = "e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267"
BASE_OCI_DIGEST = "sha256:c1e60b3673c848b758bda1b32ec39898feb5d6ff4c7e55ead8661adb04b327bd"
TOOLATHLON_REVISION = "18e62c0d0419da88209af911528f2f8a0b44a5ed"
BASE_SERVER_SHA256 = "719ec82fb3efada36c01115df3f636f7311b40778ba7c52103d1e6950800152e"
PATCHED_SERVER_SHA256 = "f38415553a88d16a9b7327ce241c20eea1ef172ca9ae4fb9da797b8e06c76485"
BASE_CATALOG_SHA256 = "48c56aabcf85833c53a967b1927bdf8a3e96c600983e9aa4f830ed99ab48f057"
REPAIRED_CATALOG_SHA256 = "a6bdfea0a5524ec8d198a0b2eb9b107aba89e308b50659a106abaff0002fd64d"
PLAYWRIGHT_BROWSERS_PATH = "/opt/ms-playwright"
PLAYWRIGHT_BROWSER_TREE_SHA256 = "84d979d1a301e24d7ee04009ff85b806cda195cce9d6bd1a8a686f06887805e5"
PLAYWRIGHT_BROWSER_FILE_COUNT = 156
PLAYWRIGHT_BROWSER_TOTAL_BYTES = 959511402
PLAYWRIGHT_BROWSER_SYMLINK_COUNT = 0
PLAYWRIGHT_BROWSER_DIRECTORY_COUNT = 13
PLAYWRIGHT_CHROMIUM_REVISION = "1182"
PLAYWRIGHT_CHROMIUM_EXECUTABLE = "/opt/ms-playwright/chromium-1182/chrome-linux/chrome"
PLAYWRIGHT_CHROMIUM_SHA256 = "b93030bc068a3850d4d668e69816fe4a5f04ec93db9b2f63f32f58a39f207db8"
PLAYWRIGHT_HEADLESS_SHELL_EXECUTABLE = "/opt/ms-playwright/chromium_headless_shell-1182/chrome-linux/headless_shell"
PLAYWRIGHT_HEADLESS_SHELL_SHA256 = "71ebe8760eb72a99b265470d4030fef296f119e25b9aab5d3efbb6ac821483a5"
PLAYWRIGHT_HEADLESS_SHELL_BYTES = 305228784
BASE_PLAYWRIGHT_CONFIG_SHA256 = "66f5bc92c182bf365e499d8237308452a016d8ca9c5e878d1628393647c82d1a"
PATCHED_PLAYWRIGHT_CONFIG_SHA256 = "47841318775a9afc122097e769a3fcc02d3af7089d372b60fc11da1ea44f0de9"
SINGULARITY_PATH = Path("/usr/bin/singularity")
SINGULARITY_SHA256 = "085a757f172dcabda66b44d086913314d6aa574055a8fea90523be82b770cec6"
SINGULARITY_VERSION = "4.4.1-1.el9"

PATCH_ASSETS = (
    "toolathlon-pbs-portable.def",
    "patch_toolathlon_pbs_portable.py",
    "repair_toolathlon_snowflake_catalog.py",
    "verify_toolathlon_pbs_portability.sh",
)
EMBEDDED_ASSETS = {
    "patch_toolathlon_pbs_portable.py": ("/usr/local/libexec/openreward/patch_toolathlon_pbs_portable.py"),
    "repair_toolathlon_snowflake_catalog.py": ("/usr/local/libexec/openreward/repair_toolathlon_snowflake_catalog.py"),
    "verify_toolathlon_pbs_portability.sh": ("/usr/local/libexec/openreward/verify_toolathlon_pbs_portability.sh"),
}
MANIFEST_PATH = "/usr/local/share/openreward/toolathlon-pbs-portable.json"
VERIFY_PATH = "/usr/local/libexec/openreward/verify_toolathlon_pbs_portability.sh"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular(
    path: Path,
    *,
    mode: int | None = None,
    private: bool = False,
) -> os.stat_result:
    if not path.is_absolute() or path.resolve() != path:
        raise RuntimeError(f"path is not direct/canonical: {path}")
    value = path.lstat()
    actual_mode = stat.S_IMODE(value.st_mode)
    if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid():
        raise RuntimeError(f"path is not a user-owned regular file: {path}")
    if mode is not None and actual_mode != mode:
        raise RuntimeError(f"path mode is {actual_mode:04o}, expected {mode:04o}: {path}")
    if private and actual_mode & 0o077:
        raise RuntimeError(f"path is not private: {path}")
    return value


def require_private_parent(path: Path) -> None:
    if not path.is_absolute() or path.parent.resolve() != path.parent:
        raise RuntimeError(f"receipt path is not direct and absolute: {path}")
    value = path.parent.lstat()
    if not stat.S_ISDIR(value.st_mode) or value.st_uid != os.getuid() or stat.S_IMODE(value.st_mode) & 0o077:
        raise RuntimeError(f"receipt parent is not private and user-owned: {path.parent}")


def require_trusted_executable(path: Path) -> os.stat_result:
    if not path.is_absolute() or path.resolve() != path:
        raise RuntimeError(f"trusted executable path is not direct/canonical: {path}")
    value = path.lstat()
    mode = stat.S_IMODE(value.st_mode)
    if (
        not stat.S_ISREG(value.st_mode)
        # ABCI's immutable system image may present root-owned packaged files
        # as uid 65534 on the login node.  Never trust a caller-owned binary:
        # it could forge inspect/exec output and authorize an arbitrary file.
        or value.st_uid not in {0, 65534}
        or mode & 0o022
        or not mode & 0o111
    ):
        raise RuntimeError(f"executable is not trusted: {path}")
    if path != SINGULARITY_PATH or sha256(path) != SINGULARITY_SHA256:
        raise RuntimeError(f"Singularity executable identity is not pinned: {path}")
    version = run_bytes([str(path), "version"], timeout=30).decode().strip()
    if version != SINGULARITY_VERSION:
        raise RuntimeError(f"Singularity version mismatch: {version!r} != {SINGULARITY_VERSION!r}")
    return value


def run_bytes(command: list[str], *, timeout: int = 120) -> bytes:
    try:
        result = subprocess.run(command, check=False, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command timed out after {timeout}s: {command!r}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command!r}: {result.stderr.decode(errors='replace')[-1000:]}"
        )
    return result.stdout


def singularity_exec_bytes(singularity: Path, image: Path, *command: str) -> bytes:
    return run_bytes(
        [
            str(singularity),
            "exec",
            "--cleanenv",
            "--no-home",
            str(image),
            *command,
        ]
    )


def logical_asset_hashes(asset_dir: Path) -> tuple[dict[str, str], str]:
    values: dict[str, str] = {}
    identity_input = bytearray()
    for name in PATCH_ASSETS:
        path = asset_dir / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"build asset is missing or indirect: {path}")
        digest = sha256(path)
        values[name] = digest
        identity_input.extend(f"{digest}  {name}\n".encode())
    return values, sha256_bytes(bytes(identity_input))


def labels_from_inspect(value: bytes) -> dict[str, str]:
    try:
        parsed = json.loads(value)
        labels = parsed["data"]["attributes"]["labels"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("Singularity inspect JSON has no label map") from exc
    if not isinstance(labels, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in labels.items()
    ):
        raise RuntimeError("Singularity label map is malformed")
    return dict(sorted(labels.items()))


def validate_manifest(manifest: dict[str, Any], patch_identity: str) -> None:
    expected = {
        "schema_version": 1,
        "portability_patch_sha256": patch_identity,
        "base_sif_sha256": BASE_SIF_SHA256,
        "base_oci_digest": BASE_OCI_DIGEST.removeprefix("sha256:"),
        "toolathlon_revision": TOOLATHLON_REVISION,
        "base_server_sha256": BASE_SERVER_SHA256,
        "patched_server_sha256": PATCHED_SERVER_SHA256,
        "base_tool_schemas_sha256": BASE_CATALOG_SHA256,
        "repaired_tool_schemas_sha256": REPAIRED_CATALOG_SHA256,
        "mcp_catalog_server_count": 25,
        "snowflake_tool_count": 14,
        "mcp_startup_policy": "fail_closed",
        "mcp_catalog_policy": "live_name_set_parity",
        "uv_python_downloads": "never",
        "playwright_browsers_path": PLAYWRIGHT_BROWSERS_PATH,
        "base_playwright_browser_tree_sha256": PLAYWRIGHT_BROWSER_TREE_SHA256,
        "portable_playwright_browser_tree_sha256": PLAYWRIGHT_BROWSER_TREE_SHA256,
        "playwright_browser_file_count": PLAYWRIGHT_BROWSER_FILE_COUNT,
        "playwright_browser_total_bytes": PLAYWRIGHT_BROWSER_TOTAL_BYTES,
        "playwright_browser_symlink_count": PLAYWRIGHT_BROWSER_SYMLINK_COUNT,
        "playwright_browser_directory_count": PLAYWRIGHT_BROWSER_DIRECTORY_COUNT,
        "playwright_chromium_revision": PLAYWRIGHT_CHROMIUM_REVISION,
        "playwright_chromium_executable": PLAYWRIGHT_CHROMIUM_EXECUTABLE,
        "playwright_chromium_sha256": PLAYWRIGHT_CHROMIUM_SHA256,
        "playwright_headless_shell_executable": PLAYWRIGHT_HEADLESS_SHELL_EXECUTABLE,
        "playwright_headless_shell_sha256": PLAYWRIGHT_HEADLESS_SHELL_SHA256,
        "playwright_headless_shell_bytes": PLAYWRIGHT_HEADLESS_SHELL_BYTES,
        "base_playwright_config_sha256": BASE_PLAYWRIGHT_CONFIG_SHA256,
        "patched_playwright_config_sha256": PATCHED_PLAYWRIGHT_CONFIG_SHA256,
        "playwright_chromium_sandbox": "disabled_for_unprivileged_singularity",
        "playwright_rootless_launch_policy": "required",
    }
    mismatches = {
        key: (manifest.get(key), expected_value)
        for key, expected_value in expected.items()
        if manifest.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"embedded portability manifest mismatch: {mismatches}")


def expected_receipt(
    *,
    image: Path,
    base_sif: Path,
    patch_identity: str,
    asset_hashes: dict[str, str],
    builder: Path,
    publisher: Path,
    image_publisher: Path,
    manifest_bytes: bytes,
    embedded_hashes: dict[str, str],
    labels: dict[str, str],
    singularity: Path,
    completed_at_epoch: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "gate": "toolathlon_pbs_portable_build",
        "status": "passed",
        "authorization_scope": "image_artifact_only",
        "pbs_accounting_attested": False,
        "toolathlon_source_revision": TOOLATHLON_REVISION,
        "base_image": {
            "path": str(base_sif),
            "sha256": BASE_SIF_SHA256,
            "oci_digest": BASE_OCI_DIGEST,
        },
        "derived_image": {
            "path": str(image),
            "sha256": sha256(image),
            "size": image.stat().st_size,
            "device": image.stat().st_dev,
            "inode": image.stat().st_ino,
            "mode": 0o550,
            "uid": os.getuid(),
        },
        "portability_patch_sha256": patch_identity,
        "patch_assets": asset_hashes,
        "build_tooling": {
            "builder": {"name": builder.name, "sha256": sha256(builder)},
            "publisher": {"name": publisher.name, "sha256": sha256(publisher)},
            "image_publisher": {
                "name": image_publisher.name,
                "sha256": sha256(image_publisher),
            },
            "singularity": {
                "path": str(singularity),
                "sha256": sha256(singularity),
                "version": SINGULARITY_VERSION,
            },
        },
        "embedded": {
            "manifest_path": MANIFEST_PATH,
            "manifest_sha256": sha256_bytes(manifest_bytes),
            "assets": embedded_hashes,
            "base_server_sha256": BASE_SERVER_SHA256,
            "patched_server_sha256": PATCHED_SERVER_SHA256,
            "base_tool_schemas_sha256": BASE_CATALOG_SHA256,
            "repaired_tool_schemas_sha256": REPAIRED_CATALOG_SHA256,
            "playwright": {
                "browsers_path": PLAYWRIGHT_BROWSERS_PATH,
                "base_browser_tree_sha256": PLAYWRIGHT_BROWSER_TREE_SHA256,
                "portable_browser_tree_sha256": PLAYWRIGHT_BROWSER_TREE_SHA256,
                "browser_file_count": PLAYWRIGHT_BROWSER_FILE_COUNT,
                "browser_total_bytes": PLAYWRIGHT_BROWSER_TOTAL_BYTES,
                "browser_symlink_count": PLAYWRIGHT_BROWSER_SYMLINK_COUNT,
                "browser_directory_count": PLAYWRIGHT_BROWSER_DIRECTORY_COUNT,
                "chromium_revision": PLAYWRIGHT_CHROMIUM_REVISION,
                "chromium_executable": PLAYWRIGHT_CHROMIUM_EXECUTABLE,
                "chromium_sha256": PLAYWRIGHT_CHROMIUM_SHA256,
                "headless_shell_executable": PLAYWRIGHT_HEADLESS_SHELL_EXECUTABLE,
                "headless_shell_sha256": PLAYWRIGHT_HEADLESS_SHELL_SHA256,
                "headless_shell_bytes": PLAYWRIGHT_HEADLESS_SHELL_BYTES,
                "base_config_sha256": BASE_PLAYWRIGHT_CONFIG_SHA256,
                "patched_config_sha256": PATCHED_PLAYWRIGHT_CONFIG_SHA256,
                "chromium_sandbox": "disabled_for_unprivileged_singularity",
                "rootless_launch": "passed",
            },
            "verify": "passed",
        },
        "labels": labels,
        "completed_at_epoch": completed_at_epoch,
    }


def validate_receipt(value: dict[str, Any], expected: dict[str, Any]) -> None:
    # An existing read-only receipt has its original completion time.  Every
    # identity-bearing field must otherwise match a fresh verification.
    expected_without_time = dict(expected)
    expected_without_time.pop("completed_at_epoch")
    actual_without_time = dict(value)
    completed = actual_without_time.pop("completed_at_epoch", None)
    if not isinstance(completed, int) or completed <= 0:
        raise RuntimeError("existing build receipt completion time is invalid")
    if actual_without_time != expected_without_time:
        raise RuntimeError("existing build receipt does not match verified image")


def publish_create_only(path: Path, value: dict[str, Any]) -> None:
    require_private_parent(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o400)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError("build receipt appeared during publication") from exc
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    require_regular(path, mode=0o400)


def verify_and_publish(args: argparse.Namespace) -> dict[str, Any]:
    image = args.image.absolute()
    receipt = args.receipt.absolute()
    base_sif = args.base_sif.absolute()
    asset_dir = args.asset_dir.absolute()
    builder = args.builder.absolute()
    publisher = Path(__file__).absolute()
    image_publisher = asset_dir / "publish_toolathlon_pbs_image.py"
    singularity = args.singularity.absolute()

    require_regular(image, mode=0o550)
    if base_sif.is_symlink() or not base_sif.is_file():
        raise RuntimeError(f"base SIF is missing or indirect: {base_sif}")
    if base_sif.resolve() != base_sif:
        raise RuntimeError(f"base SIF path is not canonical: {base_sif}")
    if sha256(base_sif) != BASE_SIF_SHA256:
        raise RuntimeError("base SIF digest mismatch")
    require_regular(builder)
    require_regular(publisher)
    require_regular(image_publisher)
    require_trusted_executable(singularity)
    if not asset_dir.is_dir() or asset_dir.resolve() != asset_dir:
        raise RuntimeError(f"asset directory is not direct/canonical: {asset_dir}")

    asset_hashes, patch_identity = logical_asset_hashes(asset_dir)
    if patch_identity != args.patch_identity:
        raise RuntimeError(f"patch identity mismatch: {patch_identity} != {args.patch_identity}")

    run_bytes([str(singularity), "sif", "list", str(image)])
    labels = labels_from_inspect(run_bytes([str(singularity), "inspect", "--json", str(image)]))
    expected_labels = {
        "org.opencontainers.image.source": "https://github.com/ApGa/toolathlon-gym",
        "org.opencontainers.image.revision": TOOLATHLON_REVISION,
        "org.openreward.toolathlon.base-sif-sha256": BASE_SIF_SHA256,
        "org.openreward.toolathlon.base-oci-digest": BASE_OCI_DIGEST,
        "org.openreward.toolathlon.pbs-portability": patch_identity,
        "org.openreward.toolathlon.playwright-browsers-sha256": PLAYWRIGHT_BROWSER_TREE_SHA256,
    }
    mismatches = {key: (labels.get(key), value) for key, value in expected_labels.items() if labels.get(key) != value}
    if mismatches:
        raise RuntimeError(f"derived SIF label mismatch: {mismatches}")

    verify_output = singularity_exec_bytes(singularity, image, VERIFY_PATH)
    if b"toolathlon_pbs_portability_verify=ok" not in verify_output:
        raise RuntimeError("embedded portability verifier did not pass")
    if b"toolathlon_playwright_rootless_launch=ok" not in verify_output:
        raise RuntimeError("embedded Playwright rootless launch did not pass")

    manifest_bytes = singularity_exec_bytes(singularity, image, "/bin/cat", MANIFEST_PATH)
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise RuntimeError("embedded portability manifest is malformed") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("embedded portability manifest is not an object")
    validate_manifest(manifest, patch_identity)

    embedded_hashes: dict[str, str] = {}
    for name, embedded_path in EMBEDDED_ASSETS.items():
        embedded_hash = sha256_bytes(singularity_exec_bytes(singularity, image, "/bin/cat", embedded_path))
        if embedded_hash != asset_hashes[name]:
            raise RuntimeError(f"embedded build asset digest mismatch: {name}")
        embedded_hashes[name] = embedded_hash

    value = expected_receipt(
        image=image,
        base_sif=base_sif,
        patch_identity=patch_identity,
        asset_hashes=asset_hashes,
        builder=builder,
        publisher=publisher,
        image_publisher=image_publisher,
        manifest_bytes=manifest_bytes,
        embedded_hashes=embedded_hashes,
        labels=labels,
        singularity=singularity,
        completed_at_epoch=int(time.time()),
    )
    if receipt.exists() or receipt.is_symlink():
        require_regular(receipt, mode=0o400)
        try:
            existing = json.loads(receipt.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError("existing build receipt is malformed") from exc
        if not isinstance(existing, dict):
            raise RuntimeError("existing build receipt is not an object")
        validate_receipt(existing, value)
        return existing

    publish_create_only(receipt, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--base-sif", type=Path, required=True)
    parser.add_argument("--patch-identity", required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--builder", type=Path, required=True)
    parser.add_argument("--singularity", type=Path, required=True)
    args = parser.parse_args()
    if len(args.patch_identity) != 64 or any(char not in "0123456789abcdef" for char in args.patch_identity):
        raise RuntimeError("patch identity is not a lowercase SHA-256 digest")
    value = verify_and_publish(args)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
