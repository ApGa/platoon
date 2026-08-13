"""Focused tests for the PBS-portable Toolathlon image derivation."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.util
import json
import os
import py_compile
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTAINERS = REPO_ROOT / "containers"
TOOLATHLON = REPO_ROOT / "external/toolathlon-gym"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_verifier_python_function(name: str):
    source = (CONTAINERS / "verify_toolathlon_pbs_portability.sh").read_text()
    embedded = source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    tree = ast.parse(embedded)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Path": Path, "re": re, "stat": stat}
    exec(compile(module, "<portable-verifier-function>", "exec"), namespace)
    return namespace[name]


PATCH = load_module(
    "test_patch_toolathlon_pbs_portable",
    CONTAINERS / "patch_toolathlon_pbs_portable.py",
)
REPAIR = load_module(
    "test_repair_toolathlon_snowflake_catalog",
    CONTAINERS / "repair_toolathlon_snowflake_catalog.py",
)
BUILD_RECEIPT = load_module(
    "test_publish_toolathlon_pbs_build_receipt",
    CONTAINERS / "publish_toolathlon_pbs_build_receipt.py",
)
IMAGE_PUBLISHER = CONTAINERS / "publish_toolathlon_pbs_image.py"
IMAGE_PUBLISHER_MODULE = load_module("test_publish_toolathlon_pbs_image", IMAGE_PUBLISHER)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_patch_identity_is_independent_of_asset_directory(tmp_path: Path) -> None:
    roots = [tmp_path / "first/deep", tmp_path / "second"]
    for root in roots:
        root.mkdir(parents=True)
        for name in BUILD_RECEIPT.PATCH_ASSETS:
            shutil.copy2(CONTAINERS / name, root / name)

    first_hashes, first_identity = BUILD_RECEIPT.logical_asset_hashes(roots[0])
    second_hashes, second_identity = BUILD_RECEIPT.logical_asset_hashes(roots[1])

    assert first_hashes == second_hashes
    assert first_identity == second_identity
    absolute_names = "\n".join(str(root) for root in roots)
    assert absolute_names.encode() not in bytes.fromhex(first_identity)


def test_patch_repair_chain_seals_complete_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "rootfs"
    uv = root / "root/.local/bin/uv"
    uv.parent.mkdir(parents=True)
    uv.write_text("#!/bin/sh\necho uv fixture\n")
    uv.chmod(0o755)

    managed = root / ("root/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11")
    managed.parent.mkdir(parents=True)
    managed.write_text("#!/bin/sh\nexit 0\n")
    managed.chmod(0o755)
    portable_uv_parent = root / "opt/uv"
    portable_uv_parent.mkdir(parents=True)
    portable_uv_parent.chmod(0o750)
    venv = root / "opt/local_servers/fixture/.venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to("/root/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11")
    (venv / "bin/python3").symlink_to("python")
    (venv / "pyvenv.cfg").write_text("home = /root/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin\n")

    app = root / "app"
    app.mkdir()
    shutil.copy2(TOOLATHLON / "server.py", app / "server.py")
    configs = app / "configs/mcp_servers"
    configs.mkdir(parents=True)
    shutil.copy2(
        TOOLATHLON / "configs/mcp_servers/playwright_with_chunk.yaml",
        configs / "playwright_with_chunk.yaml",
    )
    browser_source = root / "root/.cache/ms-playwright/chromium-fixture/chrome-linux/chrome"
    browser_source.parent.mkdir(parents=True)
    browser_source.write_bytes(b"fixture chromium\n")
    browser_source.chmod(0o750)
    browser_identity = PATCH.tree_content_identity(root / "root/.cache/ms-playwright")
    monkeypatch.setattr(PATCH, "PLAYWRIGHT_BROWSER_TREE_SHA256", browser_identity["tree_sha256"])
    monkeypatch.setattr(PATCH, "PLAYWRIGHT_BROWSER_FILE_COUNT", 1)
    monkeypatch.setattr(PATCH, "PLAYWRIGHT_BROWSER_TOTAL_BYTES", browser_source.stat().st_size)
    monkeypatch.setattr(PATCH, "PLAYWRIGHT_BROWSER_DIRECTORY_COUNT", 2)
    monkeypatch.setattr(PATCH, "PLAYWRIGHT_CHROMIUM_RELATIVE", "chromium-fixture/chrome-linux/chrome")
    monkeypatch.setattr(PATCH, "PLAYWRIGHT_CHROMIUM_SHA256", digest(browser_source))
    monkeypatch.setattr(
        PATCH,
        "PLAYWRIGHT_HEADLESS_SHELL_RELATIVE",
        "chromium-fixture/chrome-linux/chrome",
    )
    monkeypatch.setattr(PATCH, "PLAYWRIGHT_HEADLESS_SHELL_SHA256", digest(browser_source))
    PATCH.install_portable_uv(root, digest(uv))
    assert portable_uv_parent.stat().st_mode & 0o777 == 0o755
    playwright_stats = PATCH.install_portable_playwright(root, browser_identity["tree_sha256"])
    PATCH.patch_playwright_config(root, PATCH.BASE_PLAYWRIGHT_CONFIG_SHA256)
    stats = PATCH.rewrite_mcp_venvs(root)
    PATCH.patch_server_fail_closed(root, PATCH.BASE_SERVER_SHA256)
    identity = "1" * 64
    PATCH.write_manifest(root, identity, stats, playwright_stats)
    manifest = root / "usr/local/share/openreward/toolathlon-pbs-portable.json"
    assert manifest.stat().st_mode & 0o777 == 0o644

    # Supply the exact 25-name config universe and a small base catalog.  The
    # repair module's source AST extraction still operates on the real pinned
    # Snowflake server source and its exact expected 14 tools.
    for index, server in enumerate(sorted(REPAIR.EXPECTED_SERVERS)):
        if server == "playwright_with_chunk":
            continue
        (configs / f"{index:02d}.yaml").write_text(yaml.safe_dump({"name": server}))
    base_catalog = {
        server: [
            {
                "name": f"fixture_{index}",
                "description": "fixture",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]
        for index, server in enumerate(sorted(REPAIR.BASE_SERVERS))
    }
    catalog = app / "tool_schemas.json"
    catalog.write_text(json.dumps(base_catalog, indent=2) + "\n")
    snowflake = root / ("opt/local_servers/mcp-snowflake-server/src/mcp_snowflake_server/server.py")
    snowflake.parent.mkdir(parents=True)
    shutil.copy2(
        TOOLATHLON / "local_servers/mcp-snowflake-server/src/mcp_snowflake_server/server.py",
        snowflake,
    )

    monkeypatch.setattr(REPAIR, "BASE_CATALOG_SHA256", digest(catalog))
    monkeypatch.setattr(REPAIR, "REPAIRED_CATALOG_SHA256", "TO_BE_FILLED")
    monkeypatch.setattr(sys, "argv", ["repair", "--root", str(root)])
    REPAIR.main()

    repaired = json.loads(catalog.read_text())
    completed_manifest = json.loads(manifest.read_text())
    assert set(repaired) == REPAIR.EXPECTED_SERVERS
    assert len(repaired["snowflake"]) == 14
    assert manifest.stat().st_mode & 0o777 == 0o444
    assert completed_manifest["patched_server_sha256"] == PATCH.PATCHED_SERVER_SHA256
    assert completed_manifest["repaired_tool_schemas_sha256"] == digest(catalog)
    assert completed_manifest["mcp_catalog_server_count"] == 25
    assert completed_manifest["portability_patch_sha256"] == identity
    assert completed_manifest["uv_python_downloads"] == "never"
    assert completed_manifest["playwright_browsers_path"] == "/opt/ms-playwright"
    assert completed_manifest["portable_playwright_browser_tree_sha256"] == browser_identity["tree_sha256"]
    assert completed_manifest["patched_playwright_config_sha256"] == PATCH.PATCHED_PLAYWRIGHT_CONFIG_SHA256
    assert (root / "opt/ms-playwright/chromium-fixture/chrome-linux/chrome").stat().st_mode & 0o777 == 0o755
    patched_playwright = (configs / "playwright_with_chunk.yaml").read_text()
    assert patched_playwright.count('    - "--no-sandbox"') == 1
    assert not os.readlink(venv / "bin/python").startswith("/root")
    assert "/root" not in (venv / "pyvenv.cfg").read_text()
    patched_server = (app / "server.py").read_text()
    assert "async def live_health(self) -> dict:" in patched_server
    assert "async def mcp_health(self) -> ToolOutput:" in patched_server
    assert 'name="mcp_health"' in patched_server
    assert "await self._mcp_bridge.live_health()" in patched_server
    py_compile.compile(str(app / "server.py"), doraise=True)

    # Execute the generated bridge diagnostic itself with a fake live MCP.
    tree = ast.parse(patched_server)
    bridge = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MCPBridge")
    health_method = next(
        node for node in bridge.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "live_health"
    )
    harness = ast.Module(
        body=[
            ast.ClassDef(
                name="HealthHarness",
                bases=[],
                keywords=[],
                body=[health_method],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(harness)
    namespace = {"ALL_TOOL_SCHEMAS": {"fixture": [{"name": "ping"}]}}
    exec(compile(harness, "<health-harness>", "exec"), namespace)

    class FakeProcess:
        proc = SimpleNamespace(returncode=None)

        async def list_tools(self):
            return [{"name": "ping"}]

    fake_bridge = SimpleNamespace(
        needed_servers=["fixture"],
        _processes={"fixture": FakeProcess()},
    )
    live = asyncio.run(namespace["HealthHarness"].live_health(fake_bridge))
    assert live == {
        "status": "passed",
        "required_servers": ["fixture"],
        "server_count": 1,
        "servers": {
            "fixture": {
                "process_running": True,
                "live_tool_names": ["ping"],
            }
        },
    }

    class DriftedProcess(FakeProcess):
        async def list_tools(self):
            return [{"name": "different"}]

    fake_bridge._processes = {"fixture": DriftedProcess()}
    with pytest.raises(RuntimeError, match="parity failed"):
        asyncio.run(namespace["HealthHarness"].live_health(fake_bridge))


def test_cross_directory_image_publication_is_create_only(tmp_path: Path) -> None:
    local_dir = tmp_path / "node-local"
    output_dir = tmp_path / "images"
    local_dir.mkdir(mode=0o700)
    output_dir.mkdir(mode=0o700)
    pending = local_dir / "image.sif.pending"
    output = output_dir / "image.sif"
    pending.write_bytes(b"verified-sif")
    pending.chmod(0o550)
    expected_sha256 = digest(pending)

    subprocess.run(
        [
            sys.executable,
            str(IMAGE_PUBLISHER),
            "--pending",
            str(pending),
            "--output",
            str(output),
            "--expected-sha256",
            expected_sha256,
        ],
        check=True,
    )

    assert output.read_bytes() == b"verified-sif"
    assert output.stat().st_mode & 0o777 == 0o550
    assert output.stat().st_nlink == 1
    assert pending.read_bytes() == b"verified-sif"
    assert not list(output_dir.glob(".*.publish.*"))
    second = local_dir / "second.pending"
    second.write_bytes(b"other")
    second.chmod(0o550)
    failed = subprocess.run(
        [
            sys.executable,
            str(IMAGE_PUBLISHER),
            "--pending",
            str(second),
            "--output",
            str(output),
            "--expected-sha256",
            digest(second),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert output.read_bytes() == b"verified-sif"
    assert second.read_bytes() == b"other"


def test_image_publication_rejects_digest_mismatch_and_cleans_staging(tmp_path: Path) -> None:
    local_dir = tmp_path / "node-local"
    output_dir = tmp_path / "images"
    local_dir.mkdir(mode=0o700)
    output_dir.mkdir(mode=0o700)
    pending = local_dir / "image.pending"
    output = output_dir / "image.sif"
    pending.write_bytes(b"verified")
    pending.chmod(0o550)

    failed = subprocess.run(
        [
            sys.executable,
            str(IMAGE_PUBLISHER),
            "--pending",
            str(pending),
            "--output",
            str(output),
            "--expected-sha256",
            "0" * 64,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert failed.returncode != 0
    assert "digest differs" in failed.stderr
    assert pending.read_bytes() == b"verified"
    assert not output.exists()
    assert not list(output_dir.glob(".*.publish.*"))


def test_image_publisher_rejects_private_setgid_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "images"
    output_dir.mkdir()
    output_dir.chmod(0o2700)
    with pytest.raises(RuntimeError, match="output directory"):
        IMAGE_PUBLISHER_MODULE.validate_directory(output_dir)


@pytest.mark.parametrize("directory_mode", [0o750, 0o770, 0o2775])
def test_image_publisher_rejects_unsafe_directory_modes(tmp_path: Path, directory_mode: int) -> None:
    output_dir = tmp_path / f"images-{directory_mode:o}"
    output_dir.mkdir()
    output_dir.chmod(directory_mode)
    with pytest.raises(RuntimeError, match="output directory"):
        IMAGE_PUBLISHER_MODULE.validate_directory(output_dir)


def test_image_publication_loses_create_race_without_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local_dir = tmp_path / "node-local"
    output_dir = tmp_path / "images"
    local_dir.mkdir(mode=0o700)
    output_dir.mkdir(mode=0o700)
    pending = local_dir / "image.pending"
    output = output_dir / "image.sif"
    pending.write_bytes(b"verified")
    pending.chmod(0o550)
    real_link = IMAGE_PUBLISHER_MODULE.os.link

    def competing_link(source, destination, **kwargs):
        competitor = output_dir / "competitor"
        competitor.write_bytes(b"foreign")
        real_link(competitor, destination)
        raise FileExistsError(destination)

    monkeypatch.setattr(IMAGE_PUBLISHER_MODULE.os, "link", competing_link)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publisher",
            "--pending",
            str(pending),
            "--output",
            str(output),
            "--expected-sha256",
            digest(pending),
        ],
    )
    with pytest.raises(RuntimeError, match="appeared"):
        IMAGE_PUBLISHER_MODULE.main()
    assert output.read_bytes() == b"foreign"
    assert pending.read_bytes() == b"verified"
    assert not list(output_dir.glob(".*.publish.*"))


def test_image_publication_rejects_tampered_shared_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    local_dir = tmp_path / "node-local"
    output_dir = tmp_path / "images"
    local_dir.mkdir(mode=0o700)
    output_dir.mkdir(mode=0o700)
    pending = local_dir / "image.pending"
    output = output_dir / "image.sif"
    pending.write_bytes(b"verified")
    pending.chmod(0o550)
    real_copy = IMAGE_PUBLISHER_MODULE.copy_descriptor

    def tampering_copy(source: int, destination: int) -> None:
        real_copy(source, destination)
        os.lseek(destination, 0, os.SEEK_SET)
        os.write(destination, b"X")

    monkeypatch.setattr(IMAGE_PUBLISHER_MODULE, "copy_descriptor", tampering_copy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publisher",
            "--pending",
            str(pending),
            "--output",
            str(output),
            "--expected-sha256",
            digest(pending),
        ],
    )
    with pytest.raises(RuntimeError, match="staging SIF digest"):
        IMAGE_PUBLISHER_MODULE.main()
    assert pending.read_bytes() == b"verified"
    assert not output.exists()
    assert not list(output_dir.glob(".*.publish.*"))


def test_image_publisher_rejects_multilink_pending(tmp_path: Path) -> None:
    output_dir = tmp_path / "images"
    output_dir.mkdir(mode=0o700)
    pending = output_dir / "image.pending"
    second_link = output_dir / "second-link"
    output = output_dir / "image.sif"
    pending.write_bytes(b"verified")
    pending.chmod(0o550)
    os.link(pending, second_link)
    failed = subprocess.run(
        [
            sys.executable,
            str(IMAGE_PUBLISHER),
            "--pending",
            str(pending),
            "--output",
            str(output),
            "--expected-sha256",
            digest(pending),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert not output.exists()


@pytest.mark.parametrize("source_kind", ["symlink", "wrong-mode"])
def test_image_publisher_rejects_indirect_or_malformed_source(tmp_path: Path, source_kind: str) -> None:
    local_dir = tmp_path / "node-local"
    output_dir = tmp_path / "images"
    local_dir.mkdir(mode=0o700)
    output_dir.mkdir(mode=0o700)
    real_source = local_dir / "real.sif"
    real_source.write_bytes(b"verified")
    real_source.chmod(0o550)
    pending = real_source
    if source_kind == "symlink":
        pending = local_dir / "indirect.sif"
        pending.symlink_to(real_source)
    else:
        pending.chmod(0o500)
    output = output_dir / "image.sif"

    failed = subprocess.run(
        [
            sys.executable,
            str(IMAGE_PUBLISHER),
            "--pending",
            str(pending),
            "--output",
            str(output),
            "--expected-sha256",
            digest(real_source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert failed.returncode != 0
    assert not output.exists()
    assert not list(output_dir.glob(".*.publish.*"))


def test_site_singularity_is_accepted_as_trusted_executable() -> None:
    singularity = Path("/usr/bin/singularity")
    if not singularity.is_file():
        pytest.skip("site Singularity is unavailable")
    try:
        BUILD_RECEIPT.require_trusted_executable(singularity)
    except RuntimeError as exc:
        # The host-side test sandbox has no NSS entry for the submitting uid,
        # which makes Singularity exit before printing its version.  The PBS
        # build job has normal NSS and the production validator remains
        # fail-closed there.
        if "Couldn't determine user account information" in str(exc):
            pytest.skip("test sandbox cannot execute Singularity without NSS")
        raise


def test_user_owned_fake_singularity_is_rejected(tmp_path: Path) -> None:
    fake = tmp_path / "singularity"
    fake.write_text("#!/bin/sh\necho forged\n")
    fake.chmod(0o755)
    with pytest.raises(RuntimeError, match="trusted|pinned"):
        BUILD_RECEIPT.require_trusted_executable(fake)


def test_build_receipt_is_private_create_only_and_tamper_evident(
    tmp_path: Path,
) -> None:
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    receipt_dir.chmod(0o2700)
    BUILD_RECEIPT.require_private_parent(receipt_dir / "probe")
    receipt = receipt_dir / "image.json"
    expected = {
        "schema_version": 1,
        "gate": "toolathlon_pbs_portable_build",
        "completed_at_epoch": 123,
        "derived_image": {"sha256": "a" * 64},
    }
    BUILD_RECEIPT.publish_create_only(receipt, expected)
    assert receipt.stat().st_mode & 0o7777 == 0o400
    assert json.loads(receipt.read_text()) == expected
    BUILD_RECEIPT.validate_receipt(json.loads(receipt.read_text()), expected)

    with pytest.raises(FileExistsError):
        BUILD_RECEIPT.publish_create_only(receipt, expected)
    tampered = json.loads(receipt.read_text())
    tampered["derived_image"]["sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="does not match"):
        BUILD_RECEIPT.validate_receipt(tampered, expected)


def test_build_receipt_detects_asset_and_patch_identity_mismatch(
    tmp_path: Path,
) -> None:
    asset_dir = tmp_path / "assets"
    asset_dir.mkdir()
    for name in BUILD_RECEIPT.PATCH_ASSETS:
        shutil.copy2(CONTAINERS / name, asset_dir / name)
    original_hashes, original_identity = BUILD_RECEIPT.logical_asset_hashes(asset_dir)
    changed = asset_dir / BUILD_RECEIPT.PATCH_ASSETS[0]
    changed.write_text(changed.read_text() + "\n# tamper\n")
    changed_hashes, changed_identity = BUILD_RECEIPT.logical_asset_hashes(asset_dir)
    assert changed_identity != original_identity
    assert changed_hashes != original_hashes


def test_build_receipt_binds_pinned_singularity_identity(tmp_path: Path) -> None:
    image = tmp_path / "image.sif"
    base = tmp_path / "base.sif"
    builder = tmp_path / "builder.sh"
    publisher = tmp_path / "publisher.py"
    image_publisher = tmp_path / "image_publisher.py"
    for path, value in (
        (image, b"image"),
        (base, b"base"),
        (builder, b"builder"),
        (publisher, b"publisher"),
        (image_publisher, b"image-publisher"),
    ):
        path.write_bytes(value)
    image.chmod(0o550)
    value = BUILD_RECEIPT.expected_receipt(
        image=image,
        base_sif=base,
        patch_identity="c" * 64,
        asset_hashes={name: "d" * 64 for name in BUILD_RECEIPT.PATCH_ASSETS},
        builder=builder,
        publisher=publisher,
        image_publisher=image_publisher,
        manifest_bytes=b"{}\n",
        embedded_hashes={name: "e" * 64 for name in BUILD_RECEIPT.EMBEDDED_ASSETS},
        labels={},
        singularity=BUILD_RECEIPT.SINGULARITY_PATH,
        completed_at_epoch=123,
    )
    assert value["build_tooling"]["singularity"] == {
        "path": "/usr/bin/singularity",
        "sha256": BUILD_RECEIPT.SINGULARITY_SHA256,
        "version": BUILD_RECEIPT.SINGULARITY_VERSION,
    }


def test_build_scripts_reference_exact_existing_asset_names() -> None:
    builder = (CONTAINERS / "build_toolathlon_pbs_portable_sif.sh").read_text()
    definition = (CONTAINERS / "toolathlon-pbs-portable.def").read_text()
    for name in (
        *BUILD_RECEIPT.PATCH_ASSETS,
        "publish_toolathlon_pbs_image.py",
        "publish_toolathlon_pbs_build_receipt.py",
    ):
        assert (CONTAINERS / name).is_file()
        assert name in builder or name in definition
    assert "verify_toolathlon_pbs-portability.sh" not in builder
    assert "SINGULARITY=/usr/bin/singularity" in builder
    assert BUILD_RECEIPT.SINGULARITY_SHA256 in builder
    assert f"EXPECTED_SINGULARITY_VERSION={BUILD_RECEIPT.SINGULARITY_VERSION}" in builder
    assert "command -v singularity" not in builder
    assert definition.count("verify_toolathlon_pbs_portability.sh --structural-only") == 2
    assert "exec --cleanenv --no-home" in builder
    assert "verify_toolathlon_pbs_portability.sh\n" in builder
    assert "LOCAL_BASE_SIF=${WORK_ROOT}/openreward-toolathlon-base.sif" in builder
    assert "LOCAL_OUTPUT_SIF=${WORK_ROOT}/${OUTPUT_NAME}" in builder
    assert 'cp -- "${BASE_SIF}" "${LOCAL_BASE_SIF}"' in builder
    assert "s|BASE_SIF_PATH|${LOCAL_BASE_SIF}|" in builder
    assert 'build --fakeroot "${LOCAL_OUTPUT_SIF}"' in builder
    assert '--pending "${LOCAL_OUTPUT_SIF}"' in builder
    assert '--expected-sha256 "${OUTPUT_SHA256}"' in builder
    assert '--base-sif "${BASE_SIF}"' in builder


def test_embedded_verifier_covers_every_portable_runtime_tree() -> None:
    verifier = (CONTAINERS / "verify_toolathlon_pbs_portability.sh").read_text()
    assert "EXPECTED_UV_VERSION='uv 0.12.3 (x86_64-unknown-linux-gnu)'" in verifier
    assert 'test "$(uv --version)" = "${EXPECTED_UV_VERSION}"' in verifier
    assert 'portable_python = Path("/opt/uv/python")' in verifier
    assert "portable uv parent must be mode 0755" in verifier
    assert 're.fullmatch(r"python[0-9]+\\.[0-9]+", candidate.name)' in verifier
    assert 'magic != b"\\x7fELF"' in verifier
    assert 'done < "${MANAGED_INTERPRETER_LIST}"' in verifier
    assert "setpriv --reuid=65534 --regid=65534 --clear-groups" in verifier
    for runtime_tree in ("/opt/local_servers", "/opt/venv", "/opt/uv/python", "/opt/ms-playwright"):
        assert f'Path("{runtime_tree}")' in verifier
    assert "root-backed runtime symlink" in verifier
    assert "root-backed executable shebang" in verifier
    assert "root reference in" in verifier
    assert "MCP command is not executable by uid 65534" in verifier
    assert "tree_content_identity" in verifier
    assert 'playwright_args.count("--no-sandbox") != 1' in verifier
    assert "toolathlon_playwright_rootless_launch=ok" in verifier
    assert "executablePath: selected" in verifier


def test_managed_interpreter_selector_matches_exact_uv_root_shape(tmp_path: Path) -> None:
    portable_python = tmp_path / "opt/uv/python"
    expected = []
    for version in ("3.11", "3.13"):
        exact = portable_python / f"cpython-{version}.15-linux-x86_64-gnu"
        binary = exact / f"bin/python{version}"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"\x7fELFfixture")
        binary.chmod(0o755)
        config = binary.with_name(f"python{version}-config")
        config.write_text("#!/bin/sh\nexit 1\n")
        config.chmod(0o755)
        (binary.parent / "python").symlink_to(binary.name)
        (binary.parent / "python3").symlink_to(binary.name)
        (binary.parent / "python3-config").symlink_to(config.name)
        alias = portable_python / f"cpython-{version}-linux-x86_64-gnu"
        alias.symlink_to(exact.name, target_is_directory=True)
        expected.append(binary.resolve())

    select = load_verifier_python_function("select_managed_python_interpreters")
    selected = select(portable_python)

    assert selected == expected
    assert len({(path.stat().st_dev, path.stat().st_ino) for path in selected}) == 2
    assert all(not path.name.endswith("-config") for path in selected)


def test_managed_interpreter_selector_rejects_config_only_tree(tmp_path: Path) -> None:
    portable_python = tmp_path / "opt/uv/python"
    config = portable_python / "cpython-3.11.15-linux-x86_64-gnu/bin/python3.11-config"
    config.parent.mkdir(parents=True)
    config.write_text("#!/bin/sh\nexit 1\n")
    config.chmod(0o755)

    select = load_verifier_python_function("select_managed_python_interpreters")
    with pytest.raises(SystemExit, match="no directly executable uv-managed Python ELF"):
        select(portable_python)


def test_playwright_tree_identity_detects_content_and_logical_path_drift(tmp_path: Path) -> None:
    root = tmp_path / "browsers"
    payload = root / "chromium-1182/chrome-linux/chrome"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"pinned chromium")
    original = PATCH.tree_content_identity(root)
    payload.write_bytes(b"tampered chromium")
    assert PATCH.tree_content_identity(root)["tree_sha256"] != original["tree_sha256"]
    payload.write_bytes(b"pinned chromium")
    payload.rename(payload.with_name("renamed-chrome"))
    assert PATCH.tree_content_identity(root)["tree_sha256"] != original["tree_sha256"]
