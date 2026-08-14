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
import signal
import stat
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
LAUNCHER = load_module(
    "test_launch_toolathlon_python_mcp",
    CONTAINERS / "launch_toolathlon_python_mcp.py",
)
PROBE = load_module(
    "test_probe_toolathlon_python_mcp",
    CONTAINERS / "probe_toolathlon_python_mcp_launchers.py",
)


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
    excel_source = TOOLATHLON / "local_servers/excel-mcp-server/src/excel_mcp"
    excel_target = root / "opt/local_servers/excel-mcp-server/src/excel_mcp"
    excel_target.mkdir(parents=True)
    shutil.copy2(excel_source / "server.py", excel_target / "server.py")
    shutil.copy2(excel_source / "__main__.py", excel_target / "__main__.py")
    word_source = TOOLATHLON / "local_servers/Office-Word-MCP-Server/word_document_server/main.py"
    word_target = root / "opt/local_servers/Office-Word-MCP-Server/word_document_server/main.py"
    word_target.parent.mkdir(parents=True)
    shutil.copy2(word_source, word_target)
    yahoo_target = root / (
        "opt/local_servers/yahoo-finance-mcp/.venv/lib/python3.11/site-packages/mcp/server/session.py"
    )
    yahoo_target.parent.mkdir(parents=True)
    yahoo_base = "        return True\n\n    async def _received_request(\n"
    yahoo_target.write_text(yahoo_base)
    monkeypatch.setattr(PATCH, "BASE_YAHOO_MCP_SESSION_SHA256", digest(yahoo_target))
    yahoo_patched = yahoo_base.replace(
        "        return True\n\n    async def _received_request(\n",
        "        return True\n\n"
        "    async def _receive_loop(self) -> None:\n"
        "        async with self._incoming_message_stream_writer:\n"
        "            await super()._receive_loop()\n\n"
        "    async def _received_request(\n",
    )
    monkeypatch.setattr(
        PATCH,
        "PATCHED_YAHOO_MCP_SESSION_SHA256",
        hashlib.sha256(yahoo_patched.encode()).hexdigest(),
    )
    PATCH.patch_excel_logging(root)
    PATCH.patch_word_teardown(root)
    PATCH.patch_yahoo_teardown(root)
    stats = PATCH.rewrite_mcp_venvs(root)
    monkeypatch.setattr(PATCH, "PATCHED_SERVER_SHA256", "TO_BE_FILLED")
    PATCH.patch_server_fail_closed(root, PATCH.BASE_SERVER_SHA256)
    patched_server_digest = digest(app / "server.py")
    identity = "1" * 64
    PATCH.write_manifest(
        root,
        identity,
        stats,
        playwright_stats,
        {"fixture": "python-mcp-launchers"},
        {"fixture": "snowflake-runtime"},
    )
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
    assert completed_manifest["patched_server_sha256"] == patched_server_digest
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
        snowflake_mcp_runtime={"fixture": "snowflake-runtime"},
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


def test_definition_normalizes_embedded_assets_for_submit_uid(tmp_path: Path) -> None:
    definition = (CONTAINERS / "toolathlon-pbs-portable.def").read_text()
    root = tmp_path / "rootfs"
    embedded = root / "usr/local/libexec/openreward"
    embedded.mkdir(parents=True)
    manifest = root / "usr/local/share/openreward/toolathlon-pbs-portable.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n")
    manifest.chmod(0o400)
    assets = {
        "patch_toolathlon_pbs_portable.py": 0o644,
        "repair_toolathlon_snowflake_catalog.py": 0o644,
        "launch_toolathlon_python_mcp.py": 0o755,
        "probe_toolathlon_python_mcp_launchers.py": 0o755,
        "verify_toolathlon_pbs_portability.sh": 0o755,
    }
    for name in assets:
        path = embedded / name
        shutil.copy2(CONTAINERS / name, path)
        # Model the canonical clone observed by the PBS builder: %files
        # preserves root-owned source assets with an umask-derived 0750 mode.
        path.chmod(0o750)
    normalized_directories = (
        root / "usr",
        root / "usr/local",
        root / "usr/local/libexec",
        embedded,
        root / "usr/local/share",
        manifest.parent,
    )
    for path in normalized_directories:
        path.chmod(0o750)

    verifier = embedded / "verify_toolathlon_pbs_portability.sh"

    def accessible_as_submit_uid(path: Path, *, read: bool = False, execute: bool = False) -> bool:
        requested = (0o004 if read else 0) | (0o001 if execute else 0)
        if path.stat().st_mode & requested != requested:
            return False
        current = path.parent
        while current != root:
            if current.stat().st_mode & 0o001 == 0:
                return False
            current = current.parent
        return True

    assert not accessible_as_submit_uid(verifier, execute=True)
    assert not accessible_as_submit_uid(manifest, read=True)

    # Bind the behavioral model to the exact fail-closed normalization in the
    # definition.  Root ownership is asserted textually because tests run as
    # an unprivileged host uid; modes are applied and evaluated as uid 65534.
    for logical in (
        "/usr",
        "/usr/local",
        "/usr/local/libexec",
        "/usr/local/libexec/openreward",
        "/usr/local/share",
        "/usr/local/share/openreward",
    ):
        assert logical in definition
        assert 'chown 0:0 "${directory}"' in definition
        (root / logical.lstrip("/")).chmod(0o755)
    assert 'test "$(stat -c %u:%g "${directory}")" = 0:0' in definition
    assert 'test "$(stat -c %a "${directory}")" = 755' in definition
    assert 'chown 0:0 "${asset}"' in definition
    assert 'test "$(stat -c %u:%g "${asset}")" = 0:0' in definition
    assert 'test "$(stat -c %a "${asset}")" = 644' in definition
    assert 'chown 0:0 "${executable}"' in definition
    assert 'test "$(stat -c %u:%g "${executable}")" = 0:0' in definition
    assert 'test "$(stat -c %a "${executable}")" = 755' in definition
    assert "manifest=/usr/local/share/openreward/toolathlon-pbs-portable.json" in definition
    assert 'chown 0:0 "${manifest}"' in definition
    assert 'test "$(stat -c %u:%g "${manifest}")" = 0:0' in definition
    assert 'test "$(stat -c %a "${manifest}")" = 444' in definition
    for name, mode in assets.items():
        (embedded / name).chmod(mode)
    manifest.chmod(0o444)

    assert accessible_as_submit_uid(verifier, read=True, execute=True)
    for name in ("patch_toolathlon_pbs_portable.py", "repair_toolathlon_snowflake_catalog.py"):
        assert accessible_as_submit_uid(embedded / name, read=True)
        assert not accessible_as_submit_uid(embedded / name, execute=True)
    for name in (
        "launch_toolathlon_python_mcp.py",
        "probe_toolathlon_python_mcp_launchers.py",
        "verify_toolathlon_pbs_portability.sh",
    ):
        assert accessible_as_submit_uid(embedded / name, read=True, execute=True)
    assert accessible_as_submit_uid(manifest, read=True)
    assert not accessible_as_submit_uid(manifest, execute=True)


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


def test_all_python_mcp_configs_use_exact_baked_launchers_without_uv(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    configs = root / "app/configs/mcp_servers"
    configs.mkdir(parents=True)
    source_configs = TOOLATHLON / "configs/mcp_servers"
    filenames = {
        *PATCH.PYTHON_MCP_LAUNCH_SPECS,
        "yahoo-finance.yaml",
        "youtube_transcript.yaml",
    }
    for filename in filenames:
        shutil.copy2(source_configs / filename, configs / filename)

    embedded = root / "usr/local/libexec/openreward"
    embedded.mkdir(parents=True)
    shutil.copy2(CONTAINERS / "launch_toolathlon_python_mcp.py", embedded)
    shutil.copy2(CONTAINERS / "probe_toolathlon_python_mcp_launchers.py", embedded)
    managed = root / "opt/uv/python/fixture/bin/python3.10"
    managed.parent.mkdir(parents=True)
    managed.write_bytes(b"fixture-python")
    managed.chmod(0o755)

    launcher_paths = {
        *(
            f"/opt/local_servers/{spec['project']}/.venv/bin/{spec['launcher']}"
            for spec in PATCH.PYTHON_MCP_LAUNCH_SPECS.values()
        ),
        "/opt/local_servers/yahoo-finance-mcp/.venv/bin/python",
        "/opt/local_servers/mcp-youtube-transcript/.venv/bin/python3",
    }
    for image_path in sorted(launcher_paths):
        path = root / image_path.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name in {"python", "python3"}:
            path.symlink_to("/opt/uv/python/fixture/bin/python3.10")
        else:
            path.write_text(f"#!{image_path.rsplit('/', 1)[0]}/python\n")
            path.chmod(0o755)

    result = PATCH.patch_python_mcp_launchers(root)

    assert result["rewritten_uv_config_count"] == 12
    assert result["audited_python_server_count"] == 13
    assert result["base_config_set_sha256"] == ("ed8f303ce1bebf763f45bc838e58e68477bb4cded54bcb455471782705dc9511")
    assert result["patched_config_set_sha256"] == ("0be2dc5984be8280f5ba5e6d781dad56e765a62f72fb9ebd791f51dcc80b1d60")
    assert result["launcher_mapping_sha256"] == ("23497715f4e3388420e1842558e4017cad3aec62fdf8662e294d4480b19412cc")
    assert set(LAUNCHER.LAUNCHERS) == set(result["launcher_mapping"]) - {"youtube-transcript"}
    for name, mapping in result["launcher_mapping"].items():
        value = yaml.safe_load(
            (
                configs
                / next(
                    filename
                    for filename, expected_hash in result["patched_config_sha256"].items()
                    if expected_hash == mapping["config_sha256"]
                )
            ).read_text()
        )
        assert Path(value["params"]["command"]).name != "uv"
        if name == "youtube-transcript":
            assert value["params"]["command"].endswith("/.venv/bin/python3")
            assert value["params"]["cwd"] == ("${local_servers_paths}/mcp-youtube-transcript")
        else:
            assert value["params"]["command"] == PATCH.PYTHON_MCP_WRAPPER
            assert value["params"]["args"][0] == name
            assert value["params"].get("cwd", "${agent_workspace}") == ("${agent_workspace}")
            assert LAUNCHER.LAUNCHERS[name] == (
                mapping["project_cwd"],
                mapping["exec_command"],
            )

    resolve_command = load_verifier_python_function("resolve_configured_command")
    youtube_config = configs / "youtube_transcript.yaml"
    youtube_command = yaml.safe_load(youtube_config.read_text())["params"]["command"]
    assert youtube_command == ("${local_servers_paths}/mcp-youtube-transcript/.venv/bin/python3")
    assert resolve_command(youtube_command, youtube_config) == (
        "/opt/local_servers/mcp-youtube-transcript/.venv/bin/python3"
    )
    for invalid in (
        "${agent_workspace}/python3",
        "${local_servers_paths}/../python3",
        "${local_servers_paths}/${local_servers_paths}/python3",
    ):
        with pytest.raises(SystemExit):
            resolve_command(invalid, youtube_config)


def test_probe_creates_every_absent_server_runtime_privately_and_atomically(
    tmp_path: Path,
) -> None:
    probe_root = tmp_path / "run"
    probe_root.mkdir(mode=0o700)
    names = tuple(PROBE.PYTHON_CONFIGS)
    assert len(names) == 13
    workspaces = [probe_root / name for name in names]
    assert all(not path.exists() and not path.is_symlink() for path in workspaces)

    with ThreadPoolExecutor(max_workers=len(names)) as executor:
        prepared = list(executor.map(PROBE.prepare_server_runtime, workspaces))

    for workspace, runtime in zip(workspaces, prepared, strict=True):
        assert runtime == {
            "workspace": workspace,
            "home": workspace / ".home",
            "tmp": workspace / ".tmp",
            "cache": workspace / ".cache",
            "uv_cache": workspace / ".uv-cache",
        }
        for path in runtime.values():
            metadata = path.lstat()
            assert stat.S_ISDIR(metadata.st_mode)
            assert not path.is_symlink()
            assert metadata.st_uid == os.geteuid()
            assert stat.S_IMODE(metadata.st_mode) == 0o700

    race_root = tmp_path / "race"
    race_root.mkdir(mode=0o700)
    barrier = threading.Barrier(2)

    def race_create():
        barrier.wait(timeout=5)
        return PROBE.prepare_server_runtime(race_root / "same-server")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(race_create) for _ in range(2)]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except RuntimeError as exc:
            outcomes.append(exc)
    assert sum(isinstance(item, dict) for item in outcomes) == 1
    assert sum("already exists" in str(item) for item in outcomes if isinstance(item, RuntimeError)) == 1

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir(mode=0o700)
    indirect_parent = tmp_path / "indirect-parent"
    indirect_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="not direct"):
        PROBE.create_private_child_directory(indirect_parent, "child")

    attack_parent = tmp_path / "attack-parent"
    attack_parent.mkdir(mode=0o700)
    attack_target = tmp_path / "attack-target"
    attack_target.mkdir(mode=0o700)
    (attack_parent / "child").symlink_to(attack_target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="already exists"):
        PROBE.create_private_child_directory(attack_parent, "child")
    assert list(attack_target.iterdir()) == []


def test_probe_one_prepares_absent_immutable_cwd_runtime_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir(mode=0o700)
    immutable_project = tmp_path / "immutable-youtube-project"
    immutable_project.mkdir(mode=0o700)
    config_path = configs / "youtube_transcript.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "youtube-transcript",
                "params": {
                    "command": sys.executable,
                    "args": ["-c", "raise SystemExit"],
                    "cwd": str(immutable_project),
                },
            },
            sort_keys=False,
        )
    )
    run_root = tmp_path / "run"
    run_root.mkdir(mode=0o700)
    workspace = run_root / "youtube-transcript"
    assert not workspace.exists() and not workspace.is_symlink()

    class SpawnSentinel(RuntimeError):
        pass

    async def fail_after_runtime_setup(
        command,
        arguments,
        *,
        stderr,
        cwd,
        environment,
    ):
        assert command == sys.executable
        assert arguments == ["-c", "raise SystemExit"]
        assert cwd == immutable_project
        assert stderr is not None
        assert environment["HOME"] == str(workspace / ".home")
        assert environment["TMPDIR"] == str(workspace / ".tmp")
        assert environment["XDG_CACHE_HOME"] == str(workspace / ".cache")
        assert environment["UV_CACHE_DIR"] == str(workspace / ".uv-cache")
        raise SpawnSentinel("spawn reached after runtime setup")

    monkeypatch.setattr(PROBE, "CONFIGS", configs)
    monkeypatch.setattr(PROBE, "project_from_command", lambda _command: immutable_project)
    monkeypatch.setattr(PROBE, "spawn_registered_process", fail_after_runtime_setup)
    expected_launcher = {
        "config_sha256": digest(config_path),
        "config_command": sys.executable,
        "exec_command": sys.executable,
        "project_cwd": str(immutable_project),
    }
    with pytest.raises(RuntimeError, match="spawn reached after runtime setup") as caught:
        asyncio.run(
            PROBE.probe_one(
                "youtube-transcript",
                config_path.name,
                workspace,
                expected_launcher,
                [],
                {},
            )
        )
    assert "FileNotFoundError" not in str(caught.value)
    for path in (
        workspace,
        workspace / ".home",
        workspace / ".tmp",
        workspace / ".cache",
        workspace / ".uv-cache",
    ):
        metadata = path.lstat()
        assert stat.S_ISDIR(metadata.st_mode)
        assert not path.is_symlink()
        assert stat.S_IMODE(metadata.st_mode) == 0o700


def test_probe_workspace_creation_rolls_back_and_main_cleans_pending_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rollback_parent = tmp_path / "rollback-parent"
    rollback_parent.mkdir(mode=0o700)
    real_open = PROBE.os.open

    def fail_child_open(path, flags, mode=0o777, *, dir_fd=None):
        if path == "child" and dir_fd is not None:
            raise OSError("injected post-mkdir open failure")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    with monkeypatch.context() as scoped:
        scoped.setattr(PROBE.os, "open", fail_child_open)
        with pytest.raises(OSError, match="injected post-mkdir"):
            PROBE.create_private_child_directory(rollback_parent, "child")
    assert not (rollback_parent / "child").exists()
    assert not PROBE.ACTIVE_PROCESS_GROUPS and PROBE.ACTIVE_POSTGRES is None

    cleanup_parent = tmp_path / "cleanup-parent"
    cleanup_parent.mkdir(mode=0o700)
    exact = PROBE.create_private_child_directory(cleanup_parent, "exact")
    exact_identity = PROBE.private_directory_identity(exact.lstat())
    saved = cleanup_parent / "saved-exact"
    exact.rename(saved)
    exact.mkdir(mode=0o700)
    replacement_marker = exact / "replacement"
    replacement_marker.write_text("do not delete")
    with pytest.raises(RuntimeError, match="refusing to remove replaced"):
        PROBE.remove_pinned_private_workspace(exact, exact_identity)
    assert replacement_marker.read_text() == "do not delete"

    disappeared = PROBE.create_private_child_directory(cleanup_parent, "disappeared")
    disappeared_identity = PROBE.private_directory_identity(disappeared.lstat())
    disappeared.rmdir()
    with pytest.raises(RuntimeError, match="unavailable"):
        PROBE.remove_pinned_private_workspace(disappeared, disappeared_identity)

    outer = tmp_path / "signal-parent"
    outer.mkdir(mode=0o700)
    workspace = outer / "run"
    original_create = PROBE.create_private_child_directory
    original_term_handler = signal.getsignal(signal.SIGTERM)
    original_int_handler = signal.getsignal(signal.SIGINT)

    def create_then_interrupt(parent: Path, name: str) -> Path:
        created = original_create(parent, name)
        os.kill(os.getpid(), signal.SIGTERM)
        return created

    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(PROBE, "create_private_child_directory", create_then_interrupt)
            scoped.setattr(sys, "argv", ["probe", "--workspace-root", str(workspace)])
            with pytest.raises(PROBE.ProbeInterrupted):
                PROBE.main()
    finally:
        signal.signal(signal.SIGTERM, original_term_handler)
        signal.signal(signal.SIGINT, original_int_handler)
    assert not workspace.exists() and not workspace.is_symlink()
    assert not PROBE.ACTIVE_PROCESS_GROUPS and PROBE.ACTIVE_POSTGRES is None
    blocked = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    assert signal.SIGTERM not in blocked and signal.SIGINT not in blocked

    preexisting = outer / "preexisting"
    preexisting.mkdir(mode=0o700)
    marker = preexisting / "keep"
    marker.write_text("owned by caller")
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(sys, "argv", ["probe", "--workspace-root", str(preexisting)])
            with pytest.raises(RuntimeError, match="must not already exist"):
                PROBE.main()
    finally:
        signal.signal(signal.SIGTERM, original_term_handler)
        signal.signal(signal.SIGINT, original_int_handler)
    assert marker.read_text() == "owned by caller"


def test_exact_excel_word_and_yahoo_source_fixes_are_hash_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "rootfs"
    excel_source = TOOLATHLON / "local_servers/excel-mcp-server/src/excel_mcp"
    excel_target = root / "opt/local_servers/excel-mcp-server/src/excel_mcp"
    excel_target.mkdir(parents=True)
    shutil.copy2(excel_source / "server.py", excel_target / "server.py")
    shutil.copy2(excel_source / "__main__.py", excel_target / "__main__.py")
    word_source = TOOLATHLON / "local_servers/Office-Word-MCP-Server/word_document_server/main.py"
    word_target = root / "opt/local_servers/Office-Word-MCP-Server/word_document_server/main.py"
    word_target.parent.mkdir(parents=True)
    shutil.copy2(word_source, word_target)
    yahoo_target = root / (
        "opt/local_servers/yahoo-finance-mcp/.venv/lib/python3.11/site-packages/mcp/server/session.py"
    )
    yahoo_target.parent.mkdir(parents=True)
    yahoo_base = "        return True\n\n    async def _received_request(\n"
    yahoo_target.write_text(yahoo_base)
    monkeypatch.setattr(PATCH, "BASE_YAHOO_MCP_SESSION_SHA256", digest(yahoo_target))
    yahoo_patched = yahoo_base.replace(
        "        return True\n\n    async def _received_request(\n",
        "        return True\n\n"
        "    async def _receive_loop(self) -> None:\n"
        "        async with self._incoming_message_stream_writer:\n"
        "            await super()._receive_loop()\n\n"
        "    async def _received_request(\n",
    )
    monkeypatch.setattr(
        PATCH,
        "PATCHED_YAHOO_MCP_SESSION_SHA256",
        hashlib.sha256(yahoo_patched.encode()).hexdigest(),
    )

    PATCH.patch_excel_logging(root)
    PATCH.patch_word_teardown(root)
    PATCH.patch_yahoo_teardown(root)

    assert digest(excel_target / "server.py") == PATCH.PATCHED_EXCEL_SERVER_SHA256
    assert digest(excel_target / "__main__.py") == PATCH.PATCHED_EXCEL_MAIN_SHA256
    assert digest(word_target) == PATCH.PATCHED_WORD_MAIN_SHA256
    assert "logging.FileHandler(LOG_FILE)" not in (excel_target / "server.py").read_text()
    assert 'print("Service stopped.", file=sys.stderr)' in (excel_target / "__main__.py").read_text()
    assert "    return mcp\n" not in word_target.read_text()
    assert "async with self._incoming_message_stream_writer:" in yahoo_target.read_text()
    source = (CONTAINERS / "patch_toolathlon_pbs_portable.py").read_text()
    verifier = (CONTAINERS / "verify_toolathlon_pbs_portability.sh").read_text()
    receipt = (CONTAINERS / "publish_toolathlon_pbs_build_receipt.py").read_text()
    for value in (
        "e37de044b79f6a048f5580e425c2a69628f6bd41dc940ce7dbbb0e3d4e3239f2",
        "d8412e3a9a75d1c2a36407ecd560169b0a049f9ef29feb6bb508753dd433e67b",
    ):
        assert value in source
        assert value in verifier
        assert value in receipt


def test_distribution_record_verifier_covers_console_scripts_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    site = tmp_path / "venv/lib/python3.10/site-packages"
    dist = site / "fixture-1.0.dist-info"
    script = tmp_path / "venv/bin/fixture"
    dist.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    script.write_bytes(b"#!/bin/sh\nexit 0\n")
    payload = dist / "METADATA"
    payload.write_bytes(b"Name: fixture\nVersion: 1.0\n")

    def record_digest(path: Path) -> str:
        import base64

        return base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b"=").decode()

    record = dist / "RECORD"
    record.write_text(
        "../../../bin/fixture,sha256="
        f"{record_digest(script)},{script.stat().st_size}\n"
        "fixture-1.0.dist-info/METADATA,sha256="
        f"{record_digest(payload)},{payload.stat().st_size}\n"
        "fixture-1.0.dist-info/RECORD,,\n"
    )
    assert PATCH.verify_distribution_record(site, record) == {
        "rows": 3,
        "verified_hashed_rows": 2,
        "unhashed_self_rows": 1,
    }
    script.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="RECORD mismatch"):
        PATCH.verify_distribution_record(site, record)


def test_python_mcp_probe_has_fail_closed_lifecycle_and_cleanenv_gate() -> None:
    probe = (CONTAINERS / "probe_toolathlon_python_mcp_launchers.py").read_text()
    verifier = (CONTAINERS / "verify_toolathlon_pbs_portability.sh").read_text()
    assert "@contextmanager\ndef private_postgres" in probe
    assert 'str(POSTGRES_BIN / "postgres")' in probe
    assert "registered_callback=register_postgres" in probe
    assert "validate_foreground_postgres(state, command)" in probe
    assert "process.wait(timeout=30)" in probe
    assert "start_new_session=True" in probe
    assert "preexec_fn=restore_child_signal_mask" in probe
    assert "os.killpg(process_group, signal.SIGTERM)" in probe
    assert "os.killpg(process_group, signal.SIGKILL)" in probe
    assert "dict(os.environ)" not in probe
    assert '"PYTHONDONTWRITEBYTECODE": "1"' in probe
    assert '"XDG_CACHE_HOME": str(child_cache)' in probe
    assert '"UV_CACHE_DIR": str(child_uv_cache)' in probe
    assert "toolathlon_python_mcp_rootless_probe" in verifier
    assert "toolathlon_python_mcp_rootless_probe=deferred_post_build" in verifier
    assert '"${PYTHON_MCP_ENV[@]}" /usr/bin/timeout --foreground --kill-after=120s 900s' in verifier
    assert 'process.on("SIGTERM"' in verifier
    assert "/usr/bin/env -i" in verifier
    assert '--workspace-root "${PYTHON_MCP_TMP}/run"' in verifier


def test_probe_process_lifecycle_behaviors(tmp_path: Path) -> None:
    async def scenario() -> None:
        stderr_path = tmp_path / "probe.stderr"
        with stderr_path.open("wb") as stderr:
            child = await PROBE.spawn_registered_process(
                sys.executable,
                [
                    "-c",
                    (
                        "import json,signal,sys; "
                        "print(json.dumps(sorted(int(s) for s in "
                        "signal.pthread_sigmask(signal.SIG_BLOCK,set()))),flush=True); "
                        "sys.stdin.buffer.read()"
                    ),
                ],
                stderr=stderr,
                cwd=tmp_path,
                environment=dict(os.environ),
            )
            assert child.stdout is not None
            blocked = json.loads((await child.stdout.readline()).decode())
            assert int(signal.SIGTERM) not in blocked
            assert int(signal.SIGINT) not in blocked
            await PROBE.require_stable_before_eof("normal", child, duration=0.05)
            teardown = await PROBE.close_process("normal", child, protocol_succeeded=True)
            assert teardown == {
                "method": "eof",
                "returncode": 0,
                "process_group_absent": True,
            }

            early = await PROBE.spawn_registered_process(
                "/bin/true",
                [],
                stderr=stderr,
                cwd=tmp_path,
                environment={"PATH": "/usr/bin:/bin"},
            )
            await asyncio.sleep(0.1)
            with pytest.raises(RuntimeError, match="self-exited before client EOF"):
                await PROBE.require_stable_before_eof("early", early, duration=0)
            with pytest.raises(RuntimeError, match="self-exited before client EOF"):
                await PROBE.close_process("early", early, protocol_succeeded=True)

            stubborn = await PROBE.spawn_registered_process(
                sys.executable,
                [
                    "-c",
                    (
                        "import signal,subprocess,sys,time; "
                        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                        "subprocess.Popen([sys.executable,'-c',"
                        "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                        "time.sleep(60)']); time.sleep(60)"
                    ),
                ],
                stderr=stderr,
                cwd=tmp_path,
                environment=dict(os.environ),
            )
            await asyncio.sleep(0.1)
            _returncode, forced_kill = await PROBE.terminate_process_group(
                stubborn,
                stubborn.pid,
                term_wait=0.1,
                term_group_wait=0.1,
                kill_wait=2,
                kill_group_wait=2,
            )
            PROBE.ACTIVE_PROCESS_GROUPS.discard(stubborn.pid)
            assert forced_kill is True
            assert await PROBE.process_group_absent(stubborn.pid, timeout=0)

    asyncio.run(scenario())
    assert not PROBE.ACTIVE_PROCESS_GROUPS


def test_probe_sync_helpers_and_signal_mask_fail_closed() -> None:
    environment = {"PATH": "/usr/bin:/bin"}
    PROBE.run_checked(["/bin/true"], environment=environment)
    with pytest.raises(RuntimeError, match="timed out"):
        PROBE.run_checked(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            environment=environment,
            timeout=0.1,
        )

    def rejected_callback(_process):
        raise RuntimeError("callback rejected")

    with pytest.raises(RuntimeError, match="callback rejected"):
        PROBE.spawn_registered_sync(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            environment=environment,
            registered_callback=rejected_callback,
        )
    assert not PROBE.ACTIVE_PROCESS_GROUPS

    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    try:
        with pytest.raises(RuntimeError, match="blocked cleanup signals"):
            PROBE.require_cleanup_signals_unblocked()
        with pytest.raises(RuntimeError, match="blocked cleanup signals"):
            BUILD_RECEIPT.require_cleanup_signals_unblocked()
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def test_postgres_group_validator_rejects_reused_leader_identity(tmp_path: Path) -> None:
    class FakeProcess:
        pid = os.getpid()

    state = PROBE.PostgresState(
        tmp_path,
        tmp_path / "postgres.log",
        {},
        FakeProcess(),
        0,
    )
    state.start_ticks = PROBE.process_start_ticks(os.getpid()) - 1
    with pytest.raises(RuntimeError, match="leader start identity changed"):
        PROBE.validate_postgres_group_members(state)


def test_pending_term_during_sync_and_async_spawn_cleans_registered_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_term = signal.getsignal(signal.SIGTERM)
    old_int = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, PROBE.interrupt_probe)
    signal.signal(signal.SIGINT, PROBE.interrupt_probe)
    try:
        with pytest.raises(PROBE.ProbeInterrupted):
            PROBE.spawn_registered_sync(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                environment={"PATH": "/usr/bin:/bin"},
                registered_callback=lambda _process: os.kill(os.getpid(), signal.SIGTERM),
            )
        assert not PROBE.ACTIVE_PROCESS_GROUPS

        original_spawn = PROBE.asyncio.create_subprocess_exec

        async def injected_spawn(*args, **kwargs):
            process = await original_spawn(*args, **kwargs)
            os.kill(os.getpid(), signal.SIGTERM)
            return process

        monkeypatch.setattr(PROBE.asyncio, "create_subprocess_exec", injected_spawn)
        signal.signal(signal.SIGTERM, PROBE.interrupt_probe)
        signal.signal(signal.SIGINT, PROBE.interrupt_probe)

        async def scenario() -> None:
            with (tmp_path / "spawn.stderr").open("wb") as stderr:
                with pytest.raises(PROBE.ProbeInterrupted):
                    await PROBE.spawn_registered_process(
                        sys.executable,
                        ["-c", "import time; time.sleep(60)"],
                        stderr=stderr,
                        cwd=tmp_path,
                        environment=dict(os.environ),
                    )

        asyncio.run(scenario())
        assert not PROBE.ACTIVE_PROCESS_GROUPS
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)


def test_publisher_timeout_reaps_its_complete_process_group(tmp_path: Path) -> None:
    pid_file = tmp_path / "descendant.pid"
    program = (
        "import os,signal,subprocess,sys,time; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)']); "
        f"open({str(pid_file)!r},'w').write(str(child.pid)); time.sleep(60)"
    )
    with pytest.raises(RuntimeError, match="timed out"):
        BUILD_RECEIPT.run_bytes(
            [sys.executable, "-c", program],
            timeout=1,
            terminate_timeout=0.2,
            kill_timeout=2,
        )
    assert not BUILD_RECEIPT.ACTIVE_COMMANDS
    assert pid_file.is_file()
    descendant = int(pid_file.read_text())
    deadline = time.monotonic() + 2
    while Path(f"/proc/{descendant}").exists() and time.monotonic() < deadline:
        state = Path(f"/proc/{descendant}/stat").read_text().split()[2]
        if state == "Z":
            break
        time.sleep(0.05)
    if Path(f"/proc/{descendant}").exists():
        assert Path(f"/proc/{descendant}/stat").read_text().split()[2] == "Z"


def test_outer_publisher_term_reaches_foreground_probe_and_escaped_child(
    tmp_path: Path,
) -> None:
    child_pid_file = tmp_path / "probe-child.pid"
    harness = tmp_path / "probe-harness.py"
    harness.write_text(
        textwrap.dedent(
            f"""
            import asyncio
            import importlib.util
            import os
            import signal
            import sys
            from pathlib import Path

            spec = importlib.util.spec_from_file_location(
                "nested_probe", {str(CONTAINERS / "probe_toolathlon_python_mcp_launchers.py")!r}
            )
            probe = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(probe)
            probe.require_cleanup_signals_unblocked()
            signal.signal(signal.SIGTERM, probe.interrupt_probe)
            signal.signal(signal.SIGINT, probe.interrupt_probe)
            probe.atexit.register(probe.emergency_cleanup)

            async def main():
                stderr_path = Path({str(tmp_path / "nested.stderr")!r})
                with stderr_path.open("wb") as stderr:
                    process = await probe.spawn_registered_process(
                        sys.executable,
                        ["-c", "import sys; sys.stdin.buffer.read()"],
                        stderr=stderr,
                        cwd=Path({str(tmp_path)!r}),
                        environment=dict(os.environ),
                    )
                    Path({str(child_pid_file)!r}).write_text(str(process.pid))
                    try:
                        await asyncio.sleep(60)
                    finally:
                        await probe.close_process(
                            "nested", process, protocol_succeeded=False
                        )

            asyncio.run(main())
            """
        )
    )
    with pytest.raises(RuntimeError, match="timed out"):
        BUILD_RECEIPT.run_bytes(
            [
                "/usr/bin/timeout",
                "--foreground",
                "--kill-after=5s",
                "30s",
                sys.executable,
                str(harness),
            ],
            timeout=1,
            terminate_timeout=5,
            kill_timeout=2,
        )
    assert child_pid_file.is_file()
    child_pid = int(child_pid_file.read_text())
    deadline = time.monotonic() + 2
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        if Path(f"/proc/{child_pid}/stat").read_text().split()[2] == "Z":
            break
        time.sleep(0.05)
    if Path(f"/proc/{child_pid}").exists():
        assert Path(f"/proc/{child_pid}/stat").read_text().split()[2] == "Z"
    assert not BUILD_RECEIPT.ACTIVE_COMMANDS


def test_receipt_rejects_structural_or_deferred_verifier_output() -> None:
    with pytest.raises(RuntimeError, match="deferred structural output"):
        BUILD_RECEIPT.validate_full_verifier_output(
            b"toolathlon_pbs_portability_verify=ok\ntoolathlon_python_mcp_rootless_probe=deferred_post_build\n"
        )
    BUILD_RECEIPT.validate_full_verifier_output(
        b"toolathlon_pbs_portability_verify=ok\n"
        b"toolathlon_playwright_rootless_launch=ok\n"
        b'{"toolathlon_python_mcp_rootless_probe": "ok"}\n'
    )


def patched_server_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from collections import defaultdict

    root = tmp_path / "patched-server-root"
    (root / "app").mkdir(parents=True)
    shutil.copy2(TOOLATHLON / "server.py", root / "app/server.py")
    monkeypatch.setattr(PATCH, "PATCHED_SERVER_SHA256", "TO_BE_FILLED")
    PATCH.patch_server_fail_closed(root, PATCH.BASE_SERVER_SHA256)
    source = (root / "app/server.py").read_text()
    tree = ast.parse(source)
    classes = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in {"_MCPProcess", "MCPBridge"}
    ]
    module = ast.Module(body=classes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "ALL_TOOL_SCHEMAS": {},
        "Any": object,
        "CONFIGS_DIR": Path("/nonexistent"),
        "LOCAL_SERVERS": "/opt/local_servers",
        "MCP_SUBPROCESS_STREAM_LIMIT": 1024 * 1024,
        "Path": Path,
        "asyncio": asyncio,
        "defaultdict": defaultdict,
        "json": json,
        "os": os,
        "signal": signal,
        "yaml": yaml,
    }
    exec(compile(module, "<patched-server-classes>", "exec"), namespace)
    return namespace["_MCPProcess"], namespace["MCPBridge"], source


def test_production_mcp_close_bounds_noisy_streams_and_reaps_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_process, _bridge, source = patched_server_classes(tmp_path, monkeypatch)

    async def scenario() -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            (
                "import sys; sys.stderr.buffer.write(b'x'*200000+b'NOISY_END'); "
                "sys.stderr.flush(); "
                "sys.stdout.buffer.write(b'y'*2000000+b'STDOUT_END'); "
                "sys.stdout.flush(); sys.stdin.buffer.read(); "
                "import time; time.sleep(0.2)"
            ),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        wrapped = mcp_process("noisy", process)
        deadline = asyncio.get_running_loop().time() + 3
        while b"NOISY_END" not in wrapped._stderr_tail:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("background stderr drain did not consume the noisy server")
            await asyncio.sleep(0.02)
        started = asyncio.get_running_loop().time()
        await wrapped.close()
        assert asyncio.get_running_loop().time() - started >= 0.15
        assert process.returncode == 0
        assert wrapped.stderr_tail().endswith("NOISY_END")
        assert len(wrapped._stderr_tail) <= 65536
        assert await wrapped._group_absent(0)

        failed = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            ("import sys; sys.stderr.write('EXPECTED_FAIL'); sys.stderr.flush(); sys.stdin.buffer.read(); sys.exit(7)"),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        failed_wrapper = mcp_process("failed", failed)
        with pytest.raises(RuntimeError, match="rc=7: EXPECTED_FAIL"):
            await failed_wrapper.close()
        assert await failed_wrapper._group_absent(0)

    asyncio.run(scenario())
    assert "start_new_session=True" in source
    assert 'full_env.pop("VIRTUAL_ENV", None)' in source
    assert 'full_env.pop("UV_PROJECT_ENVIRONMENT", None)' in source
    assert 'full_env["PYTHONPATH"] = "/app/runtime_patches"' in source
    assert "async def _discard_stdout" in source
    assert "while await self.proc.stdout.read(65536)" in source
    assert "asyncio.create_task(self.proc.stdout.read())" not in source
    assert "await asyncio.wait_for(self.proc.wait(), timeout=20)" in source


def test_production_bridge_cleanup_attempts_all_after_failure_and_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_process, bridge_type, _source = patched_server_classes(tmp_path, monkeypatch)

    class FakeProcess:
        def __init__(self, *, fail: bool = False, delay: float = 0) -> None:
            self.fail = fail
            self.delay = delay
            self.closed = False

        async def close(self) -> None:
            try:
                if self.delay:
                    await asyncio.sleep(self.delay)
                if self.fail:
                    raise RuntimeError("expected close failure")
            finally:
                self.closed = True

    async def failure_scenario() -> None:
        bridge = object.__new__(bridge_type)
        first = FakeProcess(fail=True)
        second = FakeProcess()
        bridge._processes = {"first": first, "second": second}
        with pytest.raises(RuntimeError, match="first: cleanup failed"):
            await bridge.close()
        assert first.closed and second.closed
        assert bridge._processes == {}

    async def repeated_cancellation_scenario() -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import sys,time; sys.stdin.buffer.read(); time.sleep(0.2)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        bridge = object.__new__(bridge_type)
        wrapped = mcp_process("repeated-cancel", process)
        bridge._processes = {"repeated-cancel": wrapped}
        cleanup = asyncio.create_task(bridge.close())
        for _ in range(3):
            await asyncio.sleep(0.03)
            cleanup.cancel()
        with pytest.raises(RuntimeError, match=r"cancelled 3 time\(s\)"):
            await cleanup
        assert process.returncode == 0
        assert await wrapped._group_absent(0)
        assert bridge._processes == {}

    asyncio.run(failure_scenario())
    asyncio.run(repeated_cancellation_scenario())


def test_production_close_validation_failure_still_finalizes_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mcp_process, _bridge, _source = patched_server_classes(tmp_path, monkeypatch)

    async def scenario() -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import os,time; os.close(1); os.close(2); time.sleep(60)",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        wrapped = mcp_process("untrusted", process)
        stdout_finished = asyncio.Event()
        original_discard = wrapped._discard_stdout

        async def tracked_discard() -> None:
            try:
                await original_discard()
            finally:
                stdout_finished.set()

        states = iter((False, False, True))

        async def controlled_group_absent(_timeout: float) -> bool:
            return next(states)

        def reject_group() -> bool:
            raise RuntimeError("untrusted group member")

        wrapped._discard_stdout = tracked_discard
        wrapped._group_absent = controlled_group_absent
        wrapped._validate_group_members = reject_group
        try:
            with pytest.raises(RuntimeError, match="TERM group validation failed") as caught:
                await wrapped.close()
            assert "KILL group validation failed" in str(caught.value)
            assert stdout_finished.is_set()
            assert wrapped._stderr_task.done()
        finally:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

    asyncio.run(scenario())


def test_launcher_pins_project_cwd_and_venv_first_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_execve(executable, arguments, environment):
        captured.update(
            executable=executable,
            arguments=arguments,
            environment=environment,
        )
        raise RuntimeError("exec captured")

    monkeypatch.setattr(sys, "argv", ["launcher", "emails", "--fixture"])
    monkeypatch.setenv("VIRTUAL_ENV", "/untrusted/venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/untrusted/project")
    monkeypatch.setattr(LAUNCHER.os, "chdir", lambda value: captured.update(cwd=value))
    monkeypatch.setattr(LAUNCHER.os, "execve", fake_execve)
    with pytest.raises(RuntimeError, match="exec captured"):
        LAUNCHER.main()
    project, executable = LAUNCHER.LAUNCHERS["emails"]
    assert captured["cwd"] == project
    assert captured["executable"] == executable
    assert captured["arguments"] == [executable, "--fixture"]
    environment = captured["environment"]
    assert environment["PATH"] == f"{Path(executable).parent}:{LAUNCHER.PRODUCTION_PATH}"
    assert "VIRTUAL_ENV" not in environment
    assert "UV_PROJECT_ENVIRONMENT" not in environment


def test_snowflake_lock_and_runtime_contracts_are_exact_and_no_overlay_remains() -> None:
    patcher = (CONTAINERS / "patch_toolathlon_pbs_portable.py").read_text()
    verifier = (CONTAINERS / "verify_toolathlon_pbs_portability.sh").read_text()
    receipt = (CONTAINERS / "publish_toolathlon_pbs_build_receipt.py").read_text()
    for token in (
        "2945a5f989fd610b8a31017cd29c41cd3a662e5267f1f17d4871afb7f1ace81c",
        "e7b1bf8895eec6e95d738b68a53cd4f0eaad9672284c9cdbe5ec3613399c2eec",
        "b618e4359067acb69893820bae6a970c4d8a3714d77550c972346f6120497bc5",
        "ca2c7c3ba32778fb8de51822f75f8bef3cfcd83696e50229cdec9a72c029ddd0",
    ):
        assert token in patcher
        assert token in verifier or token in receipt
    combined = "\n".join((patcher, verifier, receipt))
    assert "pinned_complete_donor_precedence" not in combined
    assert "snowflake-mcp126-site-packages" not in combined
    assert "PYTHONPATH" not in (CONTAINERS / "launch_toolathlon_python_mcp.py").read_text()
