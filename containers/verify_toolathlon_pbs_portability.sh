#!/bin/bash
# Fail-closed structural/runtime audit for the derived Toolathlon image.
set -euo pipefail
umask 077
VERIFY_MODE=${1:-full}
case "${VERIFY_MODE}" in
    full|--structural-only) ;;
    *) printf 'invalid verifier mode: %s\n' "${VERIFY_MODE}" >&2; exit 2 ;;
esac

EXPECTED_UV_SHA256=729d27dbea534ee540a2d3ef43a62fa1a10af7fcbb6d57a70d5859509f624578
EXPECTED_UV_VERSION='uv 0.12.3 (x86_64-unknown-linux-gnu)'
EXPECTED_SERVER_SHA256=f38415553a88d16a9b7327ce241c20eea1ef172ca9ae4fb9da797b8e06c76485
EXPECTED_CATALOG_SHA256=a6bdfea0a5524ec8d198a0b2eb9b107aba89e308b50659a106abaff0002fd64d
EXPECTED_SERVERS=25
EXPECTED_SNOWFLAKE_TOOLS=14
EXPECTED_PLAYWRIGHT_TREE_SHA256=84d979d1a301e24d7ee04009ff85b806cda195cce9d6bd1a8a686f06887805e5
EXPECTED_PLAYWRIGHT_FILES=156
EXPECTED_PLAYWRIGHT_BYTES=959511402
EXPECTED_CHROMIUM_SHA256=b93030bc068a3850d4d668e69816fe4a5f04ec93db9b2f63f32f58a39f207db8
EXPECTED_HEADLESS_SHELL_SHA256=71ebe8760eb72a99b265470d4030fef296f119e25b9aab5d3efbb6ac821483a5
PLAYWRIGHT_ROOT=/opt/ms-playwright
CHROMIUM_EXECUTABLE=${PLAYWRIGHT_ROOT}/chromium-1182/chrome-linux/chrome
HEADLESS_SHELL_EXECUTABLE=${PLAYWRIGHT_ROOT}/chromium_headless_shell-1182/chrome-linux/headless_shell
MANIFEST=/usr/local/share/openreward/toolathlon-pbs-portable.json
PRODUCTION_PATH=/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
COMMAND_LIST=$(mktemp /tmp/toolathlon-pbs-portable-commands.XXXXXX)
MANAGED_INTERPRETER_LIST=$(mktemp /tmp/toolathlon-pbs-portable-interpreters.XXXXXX)
BROWSER_TMP=$(mktemp -d /tmp/toolathlon-playwright-probe.XXXXXX)
cleanup() {
    rm -f -- "${COMMAND_LIST}" "${MANAGED_INTERPRETER_LIST}"
    case "${BROWSER_TMP}" in
        /tmp/toolathlon-playwright-probe.*) rm -r -- "${BROWSER_TMP}" ;;
        *) printf 'refusing unexpected browser temp path: %s\n' "${BROWSER_TMP}" >&2 ;;
    esac
}
trap cleanup EXIT

test "${PATH}" = "${PRODUCTION_PATH}"
test "$(command -v uv)" = /usr/local/bin/uv
test "${UV_PYTHON_INSTALL_DIR:-}" = /opt/uv/python
test "${UV_PYTHON_PREFERENCE:-}" = only-managed
test "${UV_PYTHON_DOWNLOADS:-}" = never
test "${UV_CACHE_DIR:-}" = /tmp/openreward-uv-cache
test "${PLAYWRIGHT_BROWSERS_PATH:-}" = "${PLAYWRIGHT_ROOT}"
test "$(uv --version)" = "${EXPECTED_UV_VERSION}"
test "$(sha256sum /usr/local/bin/uv | cut -d ' ' -f 1)" = "${EXPECTED_UV_SHA256}"
test -x /opt/venv/bin/python
test -x /usr/local/bin/uv
test -d /opt/uv/python
test -d "${PLAYWRIGHT_ROOT}"
test -x "${CHROMIUM_EXECUTABLE}"
test -x "${HEADLESS_SHELL_EXECUTABLE}"
test -r "${MANIFEST}"
test "$(sha256sum /app/server.py | cut -d ' ' -f 1)" = "${EXPECTED_SERVER_SHA256}"
test "$(sha256sum /app/tool_schemas.json | cut -d ' ' -f 1)" = "${EXPECTED_CATALOG_SHA256}"
test "$(sha256sum "${CHROMIUM_EXECUTABLE}" | cut -d ' ' -f 1)" = "${EXPECTED_CHROMIUM_SHA256}"
test "$(sha256sum "${HEADLESS_SHELL_EXECUTABLE}" | cut -d ' ' -f 1)" = "${EXPECTED_HEADLESS_SHELL_SHA256}"

python3 - "${MANIFEST}" "${COMMAND_LIST}" "${PRODUCTION_PATH}" \
    "${MANAGED_INTERPRETER_LIST}" <<'PY'
import json
import hashlib
import os
import re
import shutil
import stat
import sys
from pathlib import Path

import yaml

manifest = json.loads(Path(sys.argv[1]).read_text())
command_list = Path(sys.argv[2])
production_path = sys.argv[3]
managed_interpreter_list = Path(sys.argv[4])
expected = {
    "base_sif_sha256": "e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267",
    "toolathlon_revision": "18e62c0d0419da88209af911528f2f8a0b44a5ed",
    "patched_server_sha256": "f38415553a88d16a9b7327ce241c20eea1ef172ca9ae4fb9da797b8e06c76485",
    "base_tool_schemas_sha256": "48c56aabcf85833c53a967b1927bdf8a3e96c600983e9aa4f830ed99ab48f057",
    "repaired_tool_schemas_sha256": "a6bdfea0a5524ec8d198a0b2eb9b107aba89e308b50659a106abaff0002fd64d",
    "uv_binary_sha256": "729d27dbea534ee540a2d3ef43a62fa1a10af7fcbb6d57a70d5859509f624578",
    "uv_python_install_dir": "/opt/uv/python",
    "uv_python_downloads": "never",
    "playwright_browsers_path": "/opt/ms-playwright",
    "base_playwright_browser_tree_sha256": "84d979d1a301e24d7ee04009ff85b806cda195cce9d6bd1a8a686f06887805e5",
    "portable_playwright_browser_tree_sha256": "84d979d1a301e24d7ee04009ff85b806cda195cce9d6bd1a8a686f06887805e5",
    "playwright_browser_file_count": 156,
    "playwright_browser_total_bytes": 959511402,
    "playwright_browser_symlink_count": 0,
    "playwright_browser_directory_count": 13,
    "playwright_chromium_revision": "1182",
    "playwright_chromium_executable": "/opt/ms-playwright/chromium-1182/chrome-linux/chrome",
    "playwright_chromium_sha256": "b93030bc068a3850d4d668e69816fe4a5f04ec93db9b2f63f32f58a39f207db8",
    "playwright_headless_shell_executable": "/opt/ms-playwright/chromium_headless_shell-1182/chrome-linux/headless_shell",
    "playwright_headless_shell_sha256": "71ebe8760eb72a99b265470d4030fef296f119e25b9aab5d3efbb6ac821483a5",
    "playwright_headless_shell_bytes": 305228784,
    "base_playwright_config_sha256": "66f5bc92c182bf365e499d8237308452a016d8ca9c5e878d1628393647c82d1a",
    "patched_playwright_config_sha256": "47841318775a9afc122097e769a3fcc02d3af7089d372b60fc11da1ea44f0de9",
    "playwright_chromium_sandbox": "disabled_for_unprivileged_singularity",
    "playwright_rootless_launch_policy": "required",
    "mcp_startup_policy": "fail_closed",
    "mcp_catalog_policy": "live_name_set_parity",
    "mcp_catalog_server_count": 25,
    "snowflake_tool_count": 14,
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"manifest mismatch {key}: {manifest.get(key)!r} != {value!r}")
identity = manifest.get("portability_patch_sha256", "")
if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
    raise SystemExit(f"invalid portability patch identity: {identity!r}")

catalog = json.loads(Path("/app/tool_schemas.json").read_text())
if len(catalog) != 25 or any(not tools for tools in catalog.values()):
    raise SystemExit("MCP catalog must contain 25 nonempty server entries")
if len(catalog.get("snowflake", [])) != 14:
    raise SystemExit("Snowflake catalog must contain 14 tools")


def tree_content_identity(root: Path) -> dict[str, int | str]:
    if not root.is_dir() or root.is_symlink():
        raise SystemExit(f"Playwright payload is missing or indirect: {root}")
    root_mode = stat.S_IMODE(root.lstat().st_mode)
    if root_mode & 0o022 or root_mode & 0o055 != 0o055:
        raise SystemExit(f"unsafe Playwright payload root mode: {root_mode:04o}")
    for ancestor in (root.parent, *root.parents):
        value = ancestor.stat()
        mode = stat.S_IMODE(value.st_mode)
        if not stat.S_ISDIR(value.st_mode) or mode & 0o001 == 0 or mode & 0o022:
            raise SystemExit(f"Playwright payload ancestor is writable/untraversable: {ancestor}")
    identity = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    symlink_count = 0
    directory_count = 0
    permission_errors = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if "\t" in relative or "\n" in relative:
            raise SystemExit(f"Playwright payload has unsafe path: {relative!r}")
        value = path.lstat()
        mode = stat.S_IMODE(value.st_mode)
        if mode & 0o022:
            permission_errors.append(f"writable Playwright payload entry: {path} mode={mode:04o}")
        if stat.S_ISDIR(value.st_mode):
            if mode & 0o055 != 0o055:
                permission_errors.append(f"untraversable Playwright directory: {path}")
            record = f"D\t{relative}\n"
            directory_count += 1
        elif stat.S_ISLNK(value.st_mode):
            target = os.readlink(path)
            if "\t" in target or "\n" in target or "/root/" in target:
                raise SystemExit(f"unsafe Playwright link {path} -> {target!r}")
            record = f"L\t{relative}\t{target}\n"
            symlink_count += 1
        elif stat.S_ISREG(value.st_mode):
            if mode & 0o044 != 0o044:
                permission_errors.append(f"unreadable Playwright file: {path}")
            content = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    content.update(chunk)
            record = f"F\t{relative}\t{value.st_size}\t{content.hexdigest()}\n"
            file_count += 1
            total_bytes += value.st_size
        else:
            raise SystemExit(f"unsupported Playwright payload entry: {path}")
        identity.update(record.encode())
    if permission_errors:
        raise SystemExit("\n".join(permission_errors))
    return {
        "tree_sha256": identity.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "symlink_count": symlink_count,
        "directory_count": directory_count,
    }


playwright_identity = tree_content_identity(Path("/opt/ms-playwright"))
expected_playwright_identity = {
    "tree_sha256": "84d979d1a301e24d7ee04009ff85b806cda195cce9d6bd1a8a686f06887805e5",
    "file_count": 156,
    "total_bytes": 959511402,
    "symlink_count": 0,
    "directory_count": 13,
}
if playwright_identity != expected_playwright_identity:
    raise SystemExit(
        f"Playwright payload identity mismatch: {playwright_identity} != {expected_playwright_identity}"
    )
playwright_config = Path("/app/configs/mcp_servers/playwright_with_chunk.yaml")
if hashlib.sha256(playwright_config.read_bytes()).hexdigest() != (
    "47841318775a9afc122097e769a3fcc02d3af7089d372b60fc11da1ea44f0de9"
):
    raise SystemExit("patched Playwright MCP config digest mismatch")
playwright_config_value = yaml.safe_load(playwright_config.read_text())
playwright_args = playwright_config_value.get("params", {}).get("args", [])
if not isinstance(playwright_args, list) or playwright_args.count("--no-sandbox") != 1:
    raise SystemExit("Playwright MCP config does not disable Chromium sandbox exactly once")


def contains_root_reference(value: str) -> bool:
    return "/root" in value


# Console scripts and virtual environments can retain a root-only interpreter
# even when the well-known python aliases were repaired.  Audit every symlink,
# executable shebang, and pyvenv.cfg in both runtime trees.
runtime_errors = []
portable_python = Path("/opt/uv/python")
for ancestor in (portable_python, *portable_python.parents):
    try:
        value = ancestor.lstat()
    except OSError as exc:
        runtime_errors.append(f"cannot stat managed Python ancestor {ancestor}: {exc}")
        continue
    mode = stat.S_IMODE(value.st_mode)
    if not stat.S_ISDIR(value.st_mode) or mode & 0o022 or mode & 0o005 != 0o005:
        runtime_errors.append(
            f"unsafe managed Python ancestor {ancestor}: mode={mode:04o}"
        )
if portable_python.parent.is_dir():
    uv_parent_mode = stat.S_IMODE(portable_python.parent.lstat().st_mode)
    if uv_parent_mode != 0o755:
        runtime_errors.append(
            f"portable uv parent must be mode 0755: {portable_python.parent} "
            f"mode={uv_parent_mode:04o}"
        )
for tree in (
    Path("/opt/local_servers"),
    Path("/opt/venv"),
    Path("/opt/uv/python"),
    Path("/opt/ms-playwright"),
):
    if not tree.is_dir():
        runtime_errors.append(f"missing runtime tree: {tree}")
        continue
    for path in [tree, *tree.rglob("*")]:
        try:
            value = path.lstat()
        except OSError as exc:
            runtime_errors.append(f"cannot stat runtime path {path}: {exc}")
            continue
        if stat.S_ISLNK(value.st_mode):
            try:
                target = os.readlink(path)
            except OSError as exc:
                runtime_errors.append(f"cannot read runtime symlink {path}: {exc}")
                continue
            if contains_root_reference(target):
                runtime_errors.append(f"root-backed runtime symlink {path} -> {target}")
            continue
        if not stat.S_ISREG(value.st_mode):
            continue
        if path.name == "pyvenv.cfg":
            try:
                config_text = path.read_text()
            except (OSError, UnicodeError) as exc:
                runtime_errors.append(f"cannot read {path}: {exc}")
            else:
                if contains_root_reference(config_text):
                    runtime_errors.append(f"root reference in {path}")
        if value.st_mode & 0o111:
            try:
                with path.open("rb") as stream:
                    first_line = stream.readline(8192)
            except OSError as exc:
                runtime_errors.append(f"cannot inspect executable {path}: {exc}")
            else:
                if first_line.startswith(b"#!") and b"/root/" in first_line:
                    runtime_errors.append(f"root-backed executable shebang in {path}")
if runtime_errors:
    raise SystemExit("\n".join(runtime_errors))


def select_managed_python_interpreters(portable_python: Path) -> list[Path]:
    """Select each real, versioned uv-managed Python ELF exactly once."""

    try:
        portable_root = portable_python.resolve(strict=True)
    except OSError as exc:
        raise SystemExit(f"cannot resolve managed Python root {portable_python}: {exc}") from exc
    selected = []
    selected_paths = set()
    selected_inodes = set()
    for candidate in sorted(portable_python.glob("*/bin/python*")):
        # The managed distributions also contain executable python*-config
        # shell scripts.  They are not interpreters and reject the -c probe.
        if re.fullmatch(r"python[0-9]+\.[0-9]+", candidate.name) is None:
            continue
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(portable_root)
            value = resolved.stat()
            with resolved.open("rb") as stream:
                magic = stream.read(4)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"invalid managed Python interpreter {candidate}: {exc}") from exc
        if not stat.S_ISREG(value.st_mode) or value.st_mode & 0o111 == 0:
            raise SystemExit(f"managed Python interpreter is not executable: {resolved}")
        if magic != b"\x7fELF":
            raise SystemExit(f"managed Python interpreter is not an ELF binary: {resolved}")
        inode = (value.st_dev, value.st_ino)
        if resolved in selected_paths or inode in selected_inodes:
            continue
        selected.append(resolved)
        selected_paths.add(resolved)
        selected_inodes.add(inode)
    if not selected:
        raise SystemExit("no directly executable uv-managed Python ELF interpreters found")
    return sorted(selected)


managed_interpreters = select_managed_python_interpreters(portable_python)
managed_interpreter_list.write_text(
    "\n".join(str(interpreter) for interpreter in managed_interpreters) + "\n"
)


def executable_by_uid(path: Path, uid: int = 65534, gid: int = 65534) -> bool:
    """Check regular-file execute and ancestor traversal as uid/gid nobody."""

    if contains_root_reference(str(path)):
        return False
    try:
        value = path.stat()
    except OSError:
        return False
    if not stat.S_ISREG(value.st_mode):
        return False

    def allowed(item: os.stat_result, owner: int, group: int, other: int) -> bool:
        if item.st_uid == uid:
            return bool(item.st_mode & owner)
        if item.st_gid == gid:
            return bool(item.st_mode & group)
        return bool(item.st_mode & other)

    if not allowed(value, stat.S_IXUSR, stat.S_IXGRP, stat.S_IXOTH):
        return False
    # stat() above follows the final symlink.  Walking the lexical ancestors
    # proves the configured image path itself is traversable; setpriv below is
    # the definitive kernel-level check of any absolute/relative symlink chain.
    for parent in (path.parent, *path.parents):
        try:
            parent_value = parent.stat()
        except OSError:
            return False
        if not allowed(parent_value, stat.S_IXUSR, stat.S_IXGRP, stat.S_IXOTH):
            return False
    return True


# Resolve every one of the 25 launch commands with exactly the PATH baked into
# the production image.  Reject unresolved interpolation and commands that are
# only executable as root.
configs = sorted(Path("/app/configs/mcp_servers").glob("*.yaml"))
if len(configs) != 25:
    raise SystemExit(f"expected 25 MCP YAML files, found {len(configs)}")
configured_names = set()
resolved_commands = []
for config in configs:
    try:
        data = yaml.safe_load(config.read_text())
        name = data["name"]
        command = data["params"]["command"]
    except (OSError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise SystemExit(f"malformed MCP config {config}: {exc}") from exc
    if not isinstance(name, str) or not name or name in configured_names:
        raise SystemExit(f"invalid or duplicate MCP name in {config}: {name!r}")
    if not isinstance(command, str) or not command:
        raise SystemExit(f"invalid MCP command in {config}: {command!r}")
    configured_names.add(name)
    command = command.replace("${local_servers_paths}", "/opt/local_servers")
    if "${" in command:
        raise SystemExit(f"unresolved MCP command interpolation in {config}: {command!r}")
    if "/" in command:
        candidate = Path(command)
        if not candidate.is_absolute():
            raise SystemExit(f"relative MCP command path in {config}: {command!r}")
    else:
        found = shutil.which(command, path=production_path)
        if found is None:
            raise SystemExit(f"MCP command is absent from production PATH: {command!r}")
        candidate = Path(found)
    try:
        candidate.stat()
    except OSError as exc:
        raise SystemExit(f"unresolvable MCP command in {config}: {candidate}: {exc}") from exc
    if not executable_by_uid(candidate):
        raise SystemExit(f"MCP command is not executable by uid 65534: {candidate}")
    resolved_commands.append(str(candidate))
if configured_names != set(catalog):
    raise SystemExit(
        "MCP config/catalog name mismatch: "
        f"missing={sorted(set(catalog) - configured_names)}, "
        f"unexpected={sorted(configured_names - set(catalog))}"
    )
command_list.write_text("\n".join(sorted(set(resolved_commands))) + "\n")

venvs = sorted(Path("/opt/local_servers").glob("*/.venv"))
if not venvs:
    raise SystemExit("no MCP virtual environments found")
errors = []
for venv in venvs:
    config = venv / "pyvenv.cfg"
    if not config.is_file():
        errors.append(f"missing {config}")
    elif "/root" in config.read_text():
        errors.append(f"root-backed interpreter in {config}")
    aliases = sorted((venv / "bin").glob("python*"))
    if not aliases:
        errors.append(f"no Python aliases in {venv / 'bin'}")
    for alias in aliases:
        try:
            target = alias.resolve(strict=True)
        except OSError as exc:
            errors.append(f"unresolvable {alias}: {exc}")
            continue
        if str(target).startswith("/root/"):
            errors.append(f"root-backed {alias} -> {target}")
        if not os.access(target, os.R_OK | os.X_OK):
            errors.append(f"inaccessible {alias} -> {target}")
if errors:
    raise SystemExit("\n".join(errors))
print(
    json.dumps(
        {
            "toolathlon_pbs_portability": "ok",
            "venv_count": len(venvs),
            "configured_servers": len(configured_names),
            "resolved_commands": len(set(resolved_commands)),
        },
        sort_keys=True,
    )
)
PY

# Run the system-path uv, every configured executable, and every venv
# interpreter as an unprivileged uid.  This is the definitive access check;
# the Python permission audit above also produces useful path-level errors.
if command -v setpriv >/dev/null 2>&1 && [[ $(id -u) -eq 0 ]]; then
    setpriv --reuid=65534 --regid=65534 --clear-groups \
        /usr/local/bin/uv --version >/dev/null
    while IFS= read -r executable; do
        setpriv --reuid=65534 --regid=65534 --clear-groups \
            /usr/bin/test -x "${executable}"
    done < "${COMMAND_LIST}"
    while IFS= read -r interpreter; do
        setpriv --reuid=65534 --regid=65534 --clear-groups \
            "${interpreter}" -c 'import sys; assert sys.version_info >= (3, 10)' \
            >/dev/null
    done < <(find /opt/local_servers -path '*/.venv/bin/python' \( -type f -o -type l \) -print | sort)
    while IFS= read -r interpreter; do
        setpriv --reuid=65534 --regid=65534 --clear-groups \
            "${interpreter}" -c 'import sys; assert sys.version_info >= (3, 10)' \
            >/dev/null
    done < "${MANAGED_INTERPRETER_LIST}"
else
    while IFS= read -r executable; do
        test -x "${executable}"
    done < "${COMMAND_LIST}"
    while IFS= read -r interpreter; do
        "${interpreter}" -c 'import sys; assert sys.version_info >= (3, 10)' \
            >/dev/null
    done < "${MANAGED_INTERPRETER_LIST}"
fi

# Launch a real local page through the exact pinned Playwright runtime.  This
# is deliberately network-free, bounded, and run as uid 65534 when the image
# build invokes the verifier as root.  The post-build verifier repeats it as
# the submitting uid, and the all-MCP gate later calls browser_tab_list through
# the actual MCP process.
BROWSER_PROBE_JS='const { chromium } = require("/opt/local_servers/playwright-mcp/node_modules/playwright-core");
(async () => {
  const installed = "/opt/ms-playwright/chromium-1182/chrome-linux/chrome";
  const selected = "/opt/ms-playwright/chromium_headless_shell-1182/chrome-linux/headless_shell";
  if (chromium.executablePath() !== installed)
    throw new Error(`unexpected Chromium path: ${chromium.executablePath()}`);
  let browser;
  try {
    browser = await chromium.launch({
      headless: true,
      chromiumSandbox: false,
      executablePath: selected,
      args: ["--disable-dev-shm-usage"],
    });
    const page = await browser.newPage();
    await page.setContent("<!doctype html><title>toolathlon-portable-smoke</title>");
    if (await page.title() !== "toolathlon-portable-smoke")
      throw new Error("local page title mismatch");
  } finally {
    if (browser) await browser.close();
  }
  console.log("toolathlon_playwright_rootless_launch=ok");
})().catch(error => { console.error(error); process.exit(1); });'
BROWSER_ENV=(
    /usr/bin/env -i
    "PATH=${PRODUCTION_PATH}"
    "HOME=${BROWSER_TMP}"
    "TMPDIR=${BROWSER_TMP}"
    "PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_ROOT}"
    "UV_PYTHON_DOWNLOADS=never"
)
if [[ "${VERIFY_MODE}" = --structural-only ]]; then
    printf '%s\n' 'toolathlon_playwright_rootless_launch=deferred_post_build'
else
    if command -v setpriv >/dev/null 2>&1 && [[ $(id -u) -eq 0 ]]; then
        chown 65534:65534 "${BROWSER_TMP}"
        setpriv --reuid=65534 --regid=65534 --clear-groups \
            "${BROWSER_ENV[@]}" /usr/bin/timeout 90s /usr/bin/node -e "${BROWSER_PROBE_JS}" \
            | grep -Fx 'toolathlon_playwright_rootless_launch=ok'
    else
        "${BROWSER_ENV[@]}" /usr/bin/timeout 90s /usr/bin/node -e "${BROWSER_PROBE_JS}" \
            | grep -Fx 'toolathlon_playwright_rootless_launch=ok'
    fi
fi

printf '%s\n' 'toolathlon_pbs_portability_verify=ok'
