#!/usr/bin/env python3
"""Make the pinned Toolathlon SIF portable to an unprivileged PBS user.

The upstream OCI image is built as root.  Its ``uv`` executable, several
uv-managed Python interpreters, and the complete Playwright browser payload
live below ``/root``, which is mode 0700 in the resulting SIF.  SingularityCE
on PBS runs the payload as the submitting uid, so those paths cannot be
traversed.  This patch copies the pinned runtimes to system paths, rewrites
every MCP venv interpreter, and makes per-session MCP startup fail closed.

The source SIF and ``server.py`` are content-addressed.  Refuse to touch any
other input rather than applying a plausible-looking patch to a changed image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path

BASE_SIF_SHA256 = "e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267"
BASE_OCI_DIGEST = "c1e60b3673c848b758bda1b32ec39898feb5d6ff4c7e55ead8661adb04b327bd"
TOOLATHLON_REVISION = "18e62c0d0419da88209af911528f2f8a0b44a5ed"
BASE_SERVER_SHA256 = "719ec82fb3efada36c01115df3f636f7311b40778ba7c52103d1e6950800152e"
PATCHED_SERVER_SHA256 = "f38415553a88d16a9b7327ce241c20eea1ef172ca9ae4fb9da797b8e06c76485"
UV_VERSION = "0.12.3"
UV_BINARY_SHA256 = "729d27dbea534ee540a2d3ef43a62fa1a10af7fcbb6d57a70d5859509f624578"

ROOT_UV_PREFIX = "/root/.local/share/uv/python"
PORTABLE_UV_PREFIX = "/opt/uv/python"
ROOT_PLAYWRIGHT_PREFIX = "/root/.cache/ms-playwright"
PORTABLE_PLAYWRIGHT_PREFIX = "/opt/ms-playwright"
PLAYWRIGHT_BROWSER_TREE_SHA256 = "84d979d1a301e24d7ee04009ff85b806cda195cce9d6bd1a8a686f06887805e5"
PLAYWRIGHT_BROWSER_FILE_COUNT = 156
PLAYWRIGHT_BROWSER_TOTAL_BYTES = 959511402
PLAYWRIGHT_BROWSER_SYMLINK_COUNT = 0
PLAYWRIGHT_BROWSER_DIRECTORY_COUNT = 13
PLAYWRIGHT_CHROMIUM_REVISION = "1182"
PLAYWRIGHT_CHROMIUM_RELATIVE = "chromium-1182/chrome-linux/chrome"
PLAYWRIGHT_CHROMIUM_SHA256 = "b93030bc068a3850d4d668e69816fe4a5f04ec93db9b2f63f32f58a39f207db8"
PLAYWRIGHT_HEADLESS_SHELL_RELATIVE = "chromium_headless_shell-1182/chrome-linux/headless_shell"
PLAYWRIGHT_HEADLESS_SHELL_SHA256 = "71ebe8760eb72a99b265470d4030fef296f119e25b9aab5d3efbb6ac821483a5"
PLAYWRIGHT_HEADLESS_SHELL_BYTES = 305228784
BASE_PLAYWRIGHT_CONFIG_SHA256 = "66f5bc92c182bf365e499d8237308452a016d8ca9c5e878d1628393647c82d1a"
PATCHED_PLAYWRIGHT_CONFIG_SHA256 = "47841318775a9afc122097e769a3fcc02d3af7089d372b60fc11da1ea44f0de9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rooted(root: Path, absolute_path: str) -> Path:
    if not absolute_path.startswith("/"):
        raise ValueError(f"expected an absolute image path: {absolute_path!r}")
    return root / absolute_path.lstrip("/")


def assert_digest(path: Path, expected: str, description: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{description} digest mismatch: expected {expected}, got {actual}: {path}")


def make_world_readable(root: Path) -> None:
    """Grant read/traverse access and remove group/other write access."""

    for path in [root, *root.rglob("*")]:
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod((mode & ~0o022) | 0o055)
        else:
            new_mode = (mode & ~0o022) | 0o044
            if mode & 0o111:
                new_mode |= 0o111
            path.chmod(new_mode)


def tree_content_identity(root: Path) -> dict[str, int | str]:
    """Hash a tree by logical path, type, size, content, and link target.

    Ownership and modes deliberately do not enter the content identity because
    the derived image widens read/execute access after the byte-for-byte copy.
    The verifier audits the final permission policy independently.
    """

    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"browser payload is missing or indirect: {root}")
    identity = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    symlink_count = 0
    directory_count = 0
    paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if "\t" in relative or "\n" in relative:
            raise RuntimeError(f"browser payload has unsafe path: {relative!r}")
        value = path.lstat()
        if stat.S_ISDIR(value.st_mode):
            record = f"D\t{relative}\n"
            directory_count += 1
        elif stat.S_ISLNK(value.st_mode):
            target = os.readlink(path)
            if "\t" in target or "\n" in target:
                raise RuntimeError(f"browser payload has unsafe link target: {path}")
            record = f"L\t{relative}\t{target}\n"
            symlink_count += 1
        elif stat.S_ISREG(value.st_mode):
            record = f"F\t{relative}\t{value.st_size}\t{sha256(path)}\n"
            file_count += 1
            total_bytes += value.st_size
        else:
            raise RuntimeError(f"browser payload has unsupported entry: {path}")
        identity.update(record.encode())
    return {
        "tree_sha256": identity.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "symlink_count": symlink_count,
        "directory_count": directory_count,
    }


def install_portable_playwright(root: Path, expected_tree_sha256: str) -> dict[str, int | str]:
    source = rooted(root, ROOT_PLAYWRIGHT_PREFIX)
    target = rooted(root, PORTABLE_PLAYWRIGHT_PREFIX)
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"portable Playwright destination already exists: {target}")
    source_identity = tree_content_identity(source)
    if source_identity != {
        "tree_sha256": expected_tree_sha256,
        "file_count": PLAYWRIGHT_BROWSER_FILE_COUNT,
        "total_bytes": PLAYWRIGHT_BROWSER_TOTAL_BYTES,
        "symlink_count": PLAYWRIGHT_BROWSER_SYMLINK_COUNT,
        "directory_count": PLAYWRIGHT_BROWSER_DIRECTORY_COUNT,
    }:
        raise RuntimeError(f"base Playwright payload identity mismatch: {source_identity}")
    chromium = source / PLAYWRIGHT_CHROMIUM_RELATIVE
    assert_digest(chromium, PLAYWRIGHT_CHROMIUM_SHA256, "base Playwright Chromium")
    headless_shell = source / PLAYWRIGHT_HEADLESS_SHELL_RELATIVE
    assert_digest(headless_shell, PLAYWRIGHT_HEADLESS_SHELL_SHA256, "base Playwright headless shell")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, symlinks=True)
    make_world_readable(target)
    target_identity = tree_content_identity(target)
    if target_identity != source_identity:
        raise RuntimeError(
            f"portable Playwright payload differs from the pinned base payload: {target_identity} != {source_identity}"
        )
    return target_identity


def patch_playwright_config(root: Path, expected_sha256: str) -> None:
    """Disable Chromium's process sandbox for unprivileged SingularityCE."""

    config = rooted(root, "/app/configs/mcp_servers/playwright_with_chunk.yaml")
    assert_digest(config, expected_sha256, "base Playwright MCP config")
    source = config.read_text()
    anchor = '    - "--headless"      # Headless mode\n'
    if source.count(anchor) != 1 or "--no-sandbox" in source:
        raise RuntimeError("Playwright MCP no-sandbox patch anchor mismatch")
    source = source.replace(
        anchor,
        anchor + '    - "--no-sandbox"    # Required for unprivileged SingularityCE on PBS\n',
    )
    config.write_text(source)
    config.chmod(0o644)
    assert_digest(config, PATCHED_PLAYWRIGHT_CONFIG_SHA256, "patched Playwright MCP config")


def install_portable_uv(root: Path, expected_uv_sha256: str) -> None:
    source_uv = rooted(root, "/root/.local/bin/uv")
    source_python = rooted(root, ROOT_UV_PREFIX)
    target_uv = rooted(root, "/usr/local/bin/uv")
    target_python = rooted(root, PORTABLE_UV_PREFIX)

    assert_digest(source_uv, expected_uv_sha256, "base uv binary")
    if not source_python.is_dir():
        raise RuntimeError(f"base uv-managed Python directory is missing: {source_python}")
    if target_python.exists():
        raise RuntimeError(f"portable uv-managed Python destination already exists: {target_python}")

    target_uv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_uv, target_uv)
    target_uv.chmod(0o755)
    target_python.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_python, target_python, symlinks=True)
    make_world_readable(target_python)


def rewrite_mcp_venvs(root: Path) -> dict[str, int]:
    local_servers = rooted(root, "/opt/local_servers")
    if not local_servers.is_dir():
        raise RuntimeError(f"MCP server directory is missing: {local_servers}")

    rewritten_links = 0
    python_links = sorted(local_servers.glob("*/.venv/bin/python*"))
    if not python_links:
        raise RuntimeError("no MCP virtual-environment Python aliases were found")
    for link in python_links:
        if not link.is_symlink():
            continue
        target = os.readlink(link)
        if target == ROOT_UV_PREFIX or target.startswith(ROOT_UV_PREFIX + "/"):
            link.unlink()
            link.symlink_to(PORTABLE_UV_PREFIX + target[len(ROOT_UV_PREFIX) :])
            rewritten_links += 1

    rewritten_configs = 0
    configs = sorted(local_servers.glob("*/.venv/pyvenv.cfg"))
    if not configs:
        raise RuntimeError("no MCP virtual-environment pyvenv.cfg files were found")
    for config in configs:
        old = config.read_text()
        new = old.replace(ROOT_UV_PREFIX, PORTABLE_UV_PREFIX)
        if new != old:
            config.write_text(new)
            rewritten_configs += 1

    # The base image is known to contain uv-managed links.  Requiring at least
    # one rewrite makes a changed image fail instead of yielding a false pass.
    if rewritten_links == 0:
        raise RuntimeError("no /root-backed MCP Python interpreter link was rewritten")

    make_world_readable(local_servers)
    return {
        "python_aliases": len(python_links),
        "rewritten_python_aliases": rewritten_links,
        "pyvenv_configs": len(configs),
        "rewritten_pyvenv_configs": rewritten_configs,
    }


def patch_server_fail_closed(root: Path, expected_server_sha256: str) -> None:
    server = rooted(root, "/app/server.py")
    assert_digest(server, expected_server_sha256, "base Toolathlon server.py")
    source = server.read_text()

    replacements = (
        (
            """    async def call_tool(self, tool_name: str, arguments: dict) -> str:
""",
            '''    async def list_tools(self) -> list[dict]:
        """Return the live MCP tool catalog after initialization."""
        resp = await self.send_recv("tools/list", {}, timeout=30)
        if resp is None:
            raise TimeoutError("MCP tools/list timed out after 30 seconds")
        if "error" in resp:
            error = resp["error"]
            detail = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            raise RuntimeError(f"MCP tools/list failed: {detail}")
        tools = resp.get("result", {}).get("tools")
        if not isinstance(tools, list):
            raise RuntimeError("MCP tools/list returned no tool list")
        return tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
''',
        ),
        (
            '''    async def start(self):
        """Launch and initialize all needed MCP servers."""
        for yaml_path in sorted(CONFIGS_DIR.glob("*.yaml")):
''',
            '''    async def start(self):
        """Launch every required MCP server or fail the whole session."""
        failures: list[str] = []
        if len(self.needed_servers) != len(set(self.needed_servers)):
            raise RuntimeError(
                f"Required MCP server list contains duplicates: {self.needed_servers}"
            )
        for yaml_path in sorted(CONFIGS_DIR.glob("*.yaml")):
''',
        ),
        (
            """            os.makedirs(cwd, exist_ok=True)

            try:
""",
            """            os.makedirs(cwd, exist_ok=True)

            mcp_proc: _MCPProcess | None = None
            try:
""",
        ),
        (
            """                    print(f"[MCPBridge] WARNING: Failed to initialize {name}{detail}", flush=True)
                    await mcp_proc.close()
            except Exception as e:
                print(f"[MCPBridge] WARNING: Failed to launch {name}: {e}", flush=True)

    async def call_tool""",
            """                    failure = f"{name}: initialization failed{detail}"
                    print(f"[MCPBridge] ERROR: {failure}", flush=True)
                    failures.append(failure)
                    await mcp_proc.close()
            except Exception as e:
                failure = f"{name}: launch failed: {e}"
                print(f"[MCPBridge] ERROR: {failure}", flush=True)
                failures.append(failure)
                if mcp_proc is not None:
                    await mcp_proc.close()

        missing_servers = set(self.needed_servers) - set(self._processes)
        unexpected_servers = set(self._processes) - set(self.needed_servers)
        if missing_servers or unexpected_servers:
            failures.append(
                "MCP process set mismatch: "
                f"missing={sorted(missing_servers)}, "
                f"unexpected={sorted(unexpected_servers)}"
            )
        if failures:
            for process in self._processes.values():
                await process.close()
            self._processes.clear()
            raise RuntimeError(
                "Required MCP startup failed: " + " | ".join(failures)
            )

    async def call_tool""",
        ),
        (
            """                if ok:
                    self._processes[name] = mcp_proc
                else:
""",
            """                if ok:
                    expected_tools = ALL_TOOL_SCHEMAS.get(name)
                    if not expected_tools:
                        failure = f"{name}: baked tool catalog is missing or empty"
                        print(f"[MCPBridge] ERROR: {failure}", flush=True)
                        failures.append(failure)
                        await mcp_proc.close()
                        continue
                    live_tools = await mcp_proc.list_tools()
                    live_names = [tool.get("name") for tool in live_tools]
                    expected_names = [tool.get("name") for tool in expected_tools]
                    live_set = set(live_names)
                    expected_set = set(expected_names)
                    if (
                        None in live_set
                        or len(live_names) != len(live_set)
                        or len(expected_names) != len(expected_set)
                        or live_set != expected_set
                    ):
                        failure = (
                            f"{name}: live tools/list differs from baked catalog "
                            f"(missing={sorted(expected_set - live_set)}, "
                            f"unexpected={sorted(live_set - expected_set)}, "
                            f"live_count={len(live_names)}, "
                            f"baked_count={len(expected_names)})"
                        )
                        print(f"[MCPBridge] ERROR: {failure}", flush=True)
                        failures.append(failure)
                        await mcp_proc.close()
                        continue
                    self._processes[name] = mcp_proc
                else:
""",
        ),
        (
            """        return await proc.call_tool(tool_name, arguments)

    async def close(self):
""",
            '''        return await proc.call_tool(tool_name, arguments)

    async def live_health(self) -> dict:
        """Reissue tools/list to every required live MCP and prove parity."""
        required = list(self.needed_servers)
        if len(required) != len(set(required)):
            raise RuntimeError(f"Required MCP server list contains duplicates: {required}")
        expected_processes = set(required)
        actual_processes = set(self._processes)
        if actual_processes != expected_processes:
            raise RuntimeError(
                "MCP health process set mismatch: "
                f"missing={sorted(expected_processes - actual_processes)}, "
                f"unexpected={sorted(actual_processes - expected_processes)}"
            )

        server_health = {}
        for name in sorted(required):
            process = self._processes[name]
            if process.proc.returncode is not None:
                raise RuntimeError(
                    f"MCP health process exited: {name} rc={process.proc.returncode}"
                )
            expected_tools = ALL_TOOL_SCHEMAS.get(name)
            if not expected_tools:
                raise RuntimeError(f"MCP health baked catalog is missing or empty: {name}")
            live_tools = await process.list_tools()
            live_names = [tool.get("name") for tool in live_tools]
            expected_names = [tool.get("name") for tool in expected_tools]
            live_set = set(live_names)
            expected_set = set(expected_names)
            if (
                None in live_set
                or len(live_names) != len(live_set)
                or len(expected_names) != len(expected_set)
                or live_set != expected_set
            ):
                raise RuntimeError(
                    f"MCP health tools/list parity failed for {name}: "
                    f"missing={sorted(expected_set - live_set)}, "
                    f"unexpected={sorted(live_set - expected_set)}, "
                    f"live_count={len(live_names)}, "
                    f"baked_count={len(expected_names)}"
                )
            server_health[name] = {
                "process_running": True,
                "live_tool_names": sorted(live_set),
            }
        return {
            "status": "passed",
            "required_servers": sorted(required),
            "server_count": len(server_health),
            "servers": server_health,
        }

    async def close(self):
''',
        ),
        (
            """                },
            ),
        ])

    async def _call_tool(self, name: str, input: dict) -> RunToolOutput:
""",
            """                },
            ),
            _with_tool_routing(
                ToolSpec(
                    name="mcp_health",
                    description=(
                        "Reissue live tools/list against every required MCP server "
                        "and fail unless all processes and baked tool names match."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                ),
                capabilities=("tool.health",),
                invocation={"kind": "direct"},
            ),
        ])

    async def _call_tool(self, name: str, input: dict) -> RunToolOutput:
""",
        ),
        (
            """    @tool
    async def claim_done(self) -> ToolOutput:
""",
            '''    @tool(shared=False)
    async def mcp_health(self) -> ToolOutput:
        """Prove every required MCP process remains alive with exact tool parity."""
        if self._mcp_bridge is None:
            raise RuntimeError("MCP bridge is not running for this task")
        payload = await self._mcp_bridge.live_health()
        return ToolOutput(
            blocks=[TextBlock(text=json.dumps(payload, indent=2, sort_keys=True))]
        )

    @tool
    async def claim_done(self) -> ToolOutput:
''',
        ),
    )
    for old, new in replacements:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(f"server.py exact patch anchor occurred {count} times, expected once")
        source = source.replace(old, new)

    server.write_text(source)
    if PATCHED_SERVER_SHA256 != "TO_BE_FILLED":
        assert_digest(server, PATCHED_SERVER_SHA256, "patched Toolathlon server.py")


def validate_identity(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("identity must be a lowercase SHA-256 digest")
    return value


def write_manifest(
    root: Path,
    identity: str,
    venv_stats: dict[str, int],
    playwright_stats: dict[str, int | str],
) -> None:
    manifest = rooted(root, "/usr/local/share/openreward/toolathlon-pbs-portable.json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "portability_patch_sha256": identity,
                "base_sif_sha256": BASE_SIF_SHA256,
                "base_oci_digest": BASE_OCI_DIGEST,
                "toolathlon_revision": TOOLATHLON_REVISION,
                "base_server_sha256": BASE_SERVER_SHA256,
                "patched_server_sha256": sha256(rooted(root, "/app/server.py")),
                "uv_version": UV_VERSION,
                "uv_binary_sha256": sha256(rooted(root, "/usr/local/bin/uv")),
                "uv_python_install_dir": PORTABLE_UV_PREFIX,
                "uv_python_downloads": "never",
                "playwright_browsers_path": PORTABLE_PLAYWRIGHT_PREFIX,
                "base_playwright_browser_tree_sha256": PLAYWRIGHT_BROWSER_TREE_SHA256,
                "portable_playwright_browser_tree_sha256": playwright_stats["tree_sha256"],
                "playwright_browser_file_count": playwright_stats["file_count"],
                "playwright_browser_total_bytes": playwright_stats["total_bytes"],
                "playwright_browser_symlink_count": playwright_stats["symlink_count"],
                "playwright_browser_directory_count": playwright_stats["directory_count"],
                "playwright_chromium_revision": PLAYWRIGHT_CHROMIUM_REVISION,
                "playwright_chromium_executable": (f"{PORTABLE_PLAYWRIGHT_PREFIX}/{PLAYWRIGHT_CHROMIUM_RELATIVE}"),
                "playwright_chromium_sha256": PLAYWRIGHT_CHROMIUM_SHA256,
                "playwright_headless_shell_executable": (
                    f"{PORTABLE_PLAYWRIGHT_PREFIX}/{PLAYWRIGHT_HEADLESS_SHELL_RELATIVE}"
                ),
                "playwright_headless_shell_sha256": PLAYWRIGHT_HEADLESS_SHELL_SHA256,
                "playwright_headless_shell_bytes": PLAYWRIGHT_HEADLESS_SHELL_BYTES,
                "base_playwright_config_sha256": BASE_PLAYWRIGHT_CONFIG_SHA256,
                "patched_playwright_config_sha256": PATCHED_PLAYWRIGHT_CONFIG_SHA256,
                "playwright_chromium_sandbox": "disabled_for_unprivileged_singularity",
                "playwright_rootless_launch_policy": "required",
                "mcp_venvs": venv_stats,
                "mcp_startup_policy": "fail_closed",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    # The catalog-repair stage appends its independently verified hashes, then
    # seals the completed manifest read-only.  Keep this intermediate file
    # owner-writable only for that immediately following build stage.
    manifest.chmod(0o644)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/"))
    parser.add_argument("--identity", type=validate_identity, required=True)
    parser.add_argument("--expected-uv-sha256", default=UV_BINARY_SHA256)
    parser.add_argument("--expected-server-sha256", default=BASE_SERVER_SHA256)
    args = parser.parse_args()
    root = args.root.resolve()

    install_portable_uv(root, args.expected_uv_sha256)
    playwright_stats = install_portable_playwright(root, PLAYWRIGHT_BROWSER_TREE_SHA256)
    patch_playwright_config(root, BASE_PLAYWRIGHT_CONFIG_SHA256)
    venv_stats = rewrite_mcp_venvs(root)
    patch_server_fail_closed(root, args.expected_server_sha256)
    write_manifest(root, args.identity, venv_stats, playwright_stats)

    print(
        json.dumps(
            {
                "toolathlon_pbs_portable_patch": "ok",
                "identity": args.identity,
                "playwright": playwright_stats,
                "venvs": venv_stats,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
