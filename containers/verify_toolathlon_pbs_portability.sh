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
EXPECTED_SERVER_SHA256=26f58c49d42975616c5bef19b654cc4483e665b2107533000861cecbcd74c8a5
EXPECTED_CATALOG_SHA256=a6bdfea0a5524ec8d198a0b2eb9b107aba89e308b50659a106abaff0002fd64d
EXPECTED_SERVERS=25
EXPECTED_SNOWFLAKE_TOOLS=14
EXPECTED_PLAYWRIGHT_TREE_SHA256=84d979d1a301e24d7ee04009ff85b806cda195cce9d6bd1a8a686f06887805e5
EXPECTED_PLAYWRIGHT_FILES=156
EXPECTED_PLAYWRIGHT_BYTES=959511402
EXPECTED_CHROMIUM_SHA256=b93030bc068a3850d4d668e69816fe4a5f04ec93db9b2f63f32f58a39f207db8
EXPECTED_HEADLESS_SHELL_SHA256=71ebe8760eb72a99b265470d4030fef296f119e25b9aab5d3efbb6ac821483a5
EXPECTED_EXCEL_SERVER_SHA256=ff9f1931ca5d0db8e4c8a63e07a5ea67a16024fb2fa337db55b0d596057cf26c
EXPECTED_EXCEL_MAIN_SHA256=f959bbb73eea5c18a74a7e59491e9f2305283443d6470cf10f34443dddb2401e
EXPECTED_WORD_MAIN_SHA256=f8e0fbd953d549890645b8a4435d99af32adb735c50177aa2308d8e652ced1f2
EXPECTED_YAHOO_MCP_SESSION_SHA256=d8412e3a9a75d1c2a36407ecd560169b0a049f9ef29feb6bb508753dd433e67b
EXPECTED_PYTHON_MCP_WRAPPER_SHA256=409c978c81ad5f8e89c6454878a1075820f936ac4a9014b141b526fdeefbfd00
EXPECTED_PYTHON_MCP_PROBE_SHA256=cf23c165adfc5d19c1d6a40117e0c257ecf519102140b053854af97f87534d65
PLAYWRIGHT_ROOT=/opt/ms-playwright
CHROMIUM_EXECUTABLE=${PLAYWRIGHT_ROOT}/chromium-1182/chrome-linux/chrome
HEADLESS_SHELL_EXECUTABLE=${PLAYWRIGHT_ROOT}/chromium_headless_shell-1182/chrome-linux/headless_shell
MANIFEST=/usr/local/share/openreward/toolathlon-pbs-portable.json
PRODUCTION_PATH=/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin
COMMAND_LIST=$(mktemp /tmp/toolathlon-pbs-portable-commands.XXXXXX)
MANAGED_INTERPRETER_LIST=$(mktemp /tmp/toolathlon-pbs-portable-interpreters.XXXXXX)
BROWSER_TMP=$(mktemp -d /tmp/toolathlon-playwright-probe.XXXXXX)
PYTHON_MCP_TMP=$(mktemp -d /tmp/toolathlon-python-mcp-probe.XXXXXX)
cleanup() {
    rm -f -- "${COMMAND_LIST}" "${MANAGED_INTERPRETER_LIST}"
    case "${BROWSER_TMP}" in
        /tmp/toolathlon-playwright-probe.*) rm -r -- "${BROWSER_TMP}" ;;
        *) printf 'refusing unexpected browser temp path: %s\n' "${BROWSER_TMP}" >&2 ;;
    esac
    case "${PYTHON_MCP_TMP}" in
        /tmp/toolathlon-python-mcp-probe.*) rm -r -- "${PYTHON_MCP_TMP}" ;;
        *) printf 'refusing unexpected Python MCP temp path: %s\n' "${PYTHON_MCP_TMP}" >&2 ;;
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
test "$(sha256sum /opt/local_servers/excel-mcp-server/src/excel_mcp/server.py | cut -d ' ' -f 1)" = "${EXPECTED_EXCEL_SERVER_SHA256}"
test "$(sha256sum /opt/local_servers/excel-mcp-server/src/excel_mcp/__main__.py | cut -d ' ' -f 1)" = "${EXPECTED_EXCEL_MAIN_SHA256}"
test "$(sha256sum /opt/local_servers/Office-Word-MCP-Server/word_document_server/main.py | cut -d ' ' -f 1)" = "${EXPECTED_WORD_MAIN_SHA256}"
test "$(sha256sum /opt/local_servers/yahoo-finance-mcp/.venv/lib/python3.11/site-packages/mcp/server/session.py | cut -d ' ' -f 1)" = "${EXPECTED_YAHOO_MCP_SESSION_SHA256}"
test "$(sha256sum /usr/local/libexec/openreward/launch_toolathlon_python_mcp.py | cut -d ' ' -f 1)" = "${EXPECTED_PYTHON_MCP_WRAPPER_SHA256}"
test "$(sha256sum /usr/local/libexec/openreward/probe_toolathlon_python_mcp_launchers.py | cut -d ' ' -f 1)" = "${EXPECTED_PYTHON_MCP_PROBE_SHA256}"
test -x /usr/local/libexec/openreward/launch_toolathlon_python_mcp.py
test -x /usr/local/libexec/openreward/probe_toolathlon_python_mcp_launchers.py

python3 - "${MANIFEST}" "${COMMAND_LIST}" "${PRODUCTION_PATH}" \
    "${MANAGED_INTERPRETER_LIST}" <<'PY'
import json
import base64
import csv
import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import yaml

manifest = json.loads(Path(sys.argv[1]).read_text())
command_list = Path(sys.argv[2])
production_path = sys.argv[3]
managed_interpreter_list = Path(sys.argv[4])
python_mcp_wrapper = "/usr/local/libexec/openreward/launch_toolathlon_python_mcp.py"
python_mcp_configs = {
    "arxiv-latex": (
        "arxiv-latex-mcp.yaml",
        "/opt/local_servers/arxiv-latex-mcp/.venv/bin/python",
        "/opt/local_servers/arxiv-latex-mcp",
    ),
    "arxiv_local": (
        "arxiv_local.yaml",
        "/opt/local_servers/arxiv-mcp-server/.venv/bin/arxiv-mcp-server",
        "/opt/local_servers/arxiv-mcp-server",
    ),
    "emails": (
        "emails.yaml",
        "/opt/local_servers/emails-mcp/.venv/bin/emails-mcp",
        "/opt/local_servers/emails-mcp",
    ),
    "excel": (
        "excel.yaml",
        "/opt/local_servers/excel-mcp-server/.venv/bin/excel-mcp-server",
        "/opt/local_servers/excel-mcp-server",
    ),
    "google_sheet": (
        "google_sheet.yaml",
        "/opt/local_servers/mcp-google-sheets/.venv/bin/mcp-google-sheets",
        "/opt/local_servers/mcp-google-sheets",
    ),
    "pdf-tools": (
        "pdf-tools.yaml",
        "/opt/local_servers/pdf-tools-mcp/.venv/bin/pdf-tools-mcp",
        "/opt/local_servers/pdf-tools-mcp",
    ),
    "pptx": (
        "pptx.yaml",
        "/opt/local_servers/Office-PowerPoint-MCP-Server/.venv/bin/ppt_mcp_server",
        "/opt/local_servers/Office-PowerPoint-MCP-Server",
    ),
    "scholarly": (
        "scholarly_search.yaml",
        "/opt/local_servers/mcp-scholarly/.venv/bin/mcp-scholarly",
        "/opt/local_servers/mcp-scholarly",
    ),
    "snowflake": (
        "snowflake.yaml",
        "/opt/local_servers/mcp-snowflake-server/.venv/bin/mcp_snowflake_server",
        "/opt/local_servers/mcp-snowflake-server",
    ),
    "terminal": (
        "terminal.yaml",
        "/opt/local_servers/cli-mcp-server/.venv/bin/cli-mcp-server",
        "/opt/local_servers/cli-mcp-server",
    ),
    "word": (
        "word.yaml",
        "/opt/local_servers/Office-Word-MCP-Server/.venv/bin/word_mcp_server",
        "/opt/local_servers/Office-Word-MCP-Server",
    ),
    "yahoo-finance": (
        "yahoo-finance.yaml",
        "/opt/local_servers/yahoo-finance-mcp/.venv/bin/python",
        "/opt/local_servers/yahoo-finance-mcp",
    ),
    "youtube-transcript": (
        "youtube_transcript.yaml",
        "/opt/local_servers/mcp-youtube-transcript/.venv/bin/python3",
        "/opt/local_servers/mcp-youtube-transcript",
    ),
}
python_mcp_config_hashes = {
    "arxiv-latex-mcp.yaml": "5f03d88fb523ac22916544e80741e558b1dcba3b526c15f38b238d3aa6604059",
    "arxiv_local.yaml": "2778c766fcfd923a9d1d258306099cd6e33d62cb9467ff963f6e74b4d82e6236",
    "emails.yaml": "d23fa2bb87ff05a7ff58a4d3284295011d1c63955084bcf9a90af56416827224",
    "excel.yaml": "4d3e13389e726a6f6bc4f2ad44c40b04209841228fa9683c1b7c02fe99626739",
    "google_sheet.yaml": "75ba539b71e1a5d552ee6a3e97c2cfa5f26b8dc075d3d906c28a570197a798d8",
    "pdf-tools.yaml": "5595252990c6b82e0c868ace99eae7c4ed55bbd6afdd915825a80281c86c2271",
    "pptx.yaml": "58714c846943cd860b432153feadb7445693490bafe0120ff1e92223ea7d6948",
    "scholarly_search.yaml": "c2aa0e9691aca4b4cabeb6a67316fb60105e85cb2d72ba6f829c0bb33ca22b5c",
    "snowflake.yaml": "9030af77cd414390353fc4c39d185122d019dde1817d1a30d6b9b655641a118e",
    "terminal.yaml": "6c40b77a472f4cb54377eed8d619c0f98c29ddb14da0aa8845c2c89779474e4a",
    "word.yaml": "17ce31b6a3639ecde005ff314901e2fa05641dabf565d3059bd0dee0186178d2",
    "yahoo-finance.yaml": "5fe25c1e236ed64196c487018990fb2ab4b305d2a9b6c3ecdc007af004d2fb57",
    "youtube_transcript.yaml": "ad8f1f3144698841b6e4e6b0485b9919c4684edac932274153e87363588f4972",
}
python_mcp_mapping = {
    name: {
        "config_command": executable if name == "youtube-transcript" else python_mcp_wrapper,
        "exec_command": executable,
        "project_cwd": project,
        "config_sha256": python_mcp_config_hashes[filename],
    }
    for name, (filename, executable, project) in sorted(python_mcp_configs.items())
}
expected = {
    "base_sif_sha256": "e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267",
    "toolathlon_revision": "18e62c0d0419da88209af911528f2f8a0b44a5ed",
    "patched_server_sha256": "26f58c49d42975616c5bef19b654cc4483e665b2107533000861cecbcd74c8a5",
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
    "base_excel_server_sha256": "bc862fc7b268446f7bec47ece78548f4c1d29f5499a2cd9290856f2d7ba9effc",
    "patched_excel_server_sha256": "ff9f1931ca5d0db8e4c8a63e07a5ea67a16024fb2fa337db55b0d596057cf26c",
    "base_excel_main_sha256": "c4b37f9fac83215030f8ed0178590e6d154f1a8249523f56f7483fb3dc710625",
    "patched_excel_main_sha256": "f959bbb73eea5c18a74a7e59491e9f2305283443d6470cf10f34443dddb2401e",
    "base_word_main_sha256": "967026a662e86b67d62994cbf020864995b66d48570ab0018dbe875a8002445d",
    "patched_word_main_sha256": "f8e0fbd953d549890645b8a4435d99af32adb735c50177aa2308d8e652ced1f2",
    "base_yahoo_mcp_session_sha256": "e37de044b79f6a048f5580e425c2a69628f6bd41dc940ce7dbbb0e3d4e3239f2",
    "patched_yahoo_mcp_session_sha256": "d8412e3a9a75d1c2a36407ecd560169b0a049f9ef29feb6bb508753dd433e67b",
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

launch_manifest = manifest.get("python_mcp_launchers")
if not isinstance(launch_manifest, dict):
    raise SystemExit("immutable Python MCP launcher manifest is missing")
launch_expected = {
    "policy": "baked_venv_no_uv",
    "rewritten_uv_config_count": 12,
    "audited_python_server_count": 13,
    "base_config_set_sha256": "ed8f303ce1bebf763f45bc838e58e68477bb4cded54bcb455471782705dc9511",
    "patched_config_set_sha256": "0be2dc5984be8280f5ba5e6d781dad56e765a62f72fb9ebd791f51dcc80b1d60",
    "wrapper_path": python_mcp_wrapper,
    "wrapper_sha256": "409c978c81ad5f8e89c6454878a1075820f936ac4a9014b141b526fdeefbfd00",
    "probe_path": "/usr/local/libexec/openreward/probe_toolathlon_python_mcp_launchers.py",
    "probe_sha256": "cf23c165adfc5d19c1d6a40117e0c257ecf519102140b053854af97f87534d65",
    "rootless_probe_policy": (
        "cleanenv_no_home_exact_launch_initialize_live_name_set_"
        "persistence_clean_teardown_no_mutation"
    ),
    "launcher_mapping_sha256": "23497715f4e3388420e1842558e4017cad3aec62fdf8662e294d4480b19412cc",
    "patched_config_sha256": python_mcp_config_hashes,
    "launcher_mapping": python_mcp_mapping,
}
for key, value in launch_expected.items():
    if launch_manifest.get(key) != value:
        raise SystemExit(f"immutable Python MCP manifest mismatch {key}")
base_config_hashes = launch_manifest.get("base_config_sha256")
if not isinstance(base_config_hashes, dict) or set(base_config_hashes) != set(python_mcp_config_hashes):
    raise SystemExit("immutable Python MCP base config hash map is malformed")
base_config_identity = "".join(
    f"{base_config_hashes[name]}  {name}\n" for name in sorted(base_config_hashes)
)
if hashlib.sha256(base_config_identity.encode()).hexdigest() != launch_expected["base_config_set_sha256"]:
    raise SystemExit("immutable Python MCP base config-set identity mismatch")
snowflake_mcp_runtime = manifest.get("snowflake_mcp_runtime")
if not isinstance(snowflake_mcp_runtime, dict):
    raise SystemExit("Snowflake MCP 1.26 runtime manifest is missing")
snowflake_scalars = {
    "policy": "exact_in_place_record_repaired_ten_tree_replacement",
    "site_packages": "/opt/local_servers/mcp-snowflake-server/.venv/lib/python3.10/site-packages",
    "base_mcp_version": "2.0.0",
    "base_mcp_record_sha256": "60c02ca68bf53784cd57a6db77587fb747594b05f1f52874919dc8cf4ecc12c6",
    "installed_mcp_version": "1.26.0",
    "donor_mcp_record_sha256": "9fb533e1b39001b3dbb42cd1923045e4311c0f6e4ae304caa96dcafa60c91a64",
    "donor_path": "/opt/local_servers/arxiv-latex-mcp/.venv/lib/python3.10/site-packages",
    "donor_closure_sha256": "e7b1bf8895eec6e95d738b68a53cd4f0eaad9672284c9cdbe5ec3613399c2eec",
    "installed_closure_sha256": "b618e4359067acb69893820bae6a970c4d8a3714d77550c972346f6120497bc5",
    "removed_mcp2_closure_sha256": "ca2c7c3ba32778fb8de51822f75f8bef3cfcd83696e50229cdec9a72c029ddd0",
}
for key, value in snowflake_scalars.items():
    if snowflake_mcp_runtime.get(key) != value:
        raise SystemExit(f"Snowflake MCP runtime manifest mismatch: {key}")
for key, identity in (
    ("donor_trees", snowflake_scalars["donor_closure_sha256"]),
    ("installed_trees", snowflake_scalars["installed_closure_sha256"]),
    ("removed_mcp2_trees", snowflake_scalars["removed_mcp2_closure_sha256"]),
):
    value = snowflake_mcp_runtime.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Snowflake MCP runtime tree map is missing: {key}")
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != identity:
        raise SystemExit(f"Snowflake MCP runtime tree-map identity mismatch: {key}")
project_metadata = snowflake_mcp_runtime.get("project_metadata")
expected_project_metadata = {
    "base_pyproject_sha256": "cf9e6b2e2b966d48c9b3b6615d66e00fdd8395dd39df9de49c2da9bc2c9cb3b7",
    "patched_pyproject_sha256": "8d8ed2e26ffe2621403a640fff0d598a35a78dbb06da2867e0bcf6ae80df2743",
    "base_lock_sha256": "83a120808c4458ed40505c395b6cbdd812a1534485169c86ce0ef39abbce0fe3",
    "patched_lock_sha256": "2945a5f989fd610b8a31017cd29c41cd3a662e5267f1f17d4871afb7f1ace81c",
    "donor_lock_sha256": "9e721de6dc27f29169af6af6e883ab40dc08f034fcd64e5f5a56219fbfcb6119",
    "offline_uv_lock_check": "passed",
    "resolved_package_count": 66,
}
if project_metadata != expected_project_metadata:
    raise SystemExit("Snowflake MCP project metadata manifest mismatch")

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
def component_tree_identity(root: Path) -> dict[str, int | str]:
    if not root.is_dir() or root.is_symlink():
        raise SystemExit(f"dependency component is missing or indirect: {root}")
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    directories = 0
    symlinks = 0
    paths = [root, *sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())]
    for path in paths:
        relative = "." if path == root else path.relative_to(root).as_posix()
        value = path.lstat()
        mode = stat.S_IMODE(value.st_mode)
        if mode & 0o022:
            raise SystemExit(f"writable dependency component entry: {path}: {mode:04o}")
        if stat.S_ISDIR(value.st_mode):
            if mode & 0o005 != 0o005:
                raise SystemExit(f"untraversable dependency component entry: {path}")
            record = f"D\t{relative}\n"
            directories += 1
        elif stat.S_ISLNK(value.st_mode):
            target = os.readlink(path)
            record = f"L\t{relative}\t{target}\n"
            symlinks += 1
        elif stat.S_ISREG(value.st_mode):
            if mode & 0o004 == 0:
                raise SystemExit(f"unreadable dependency component entry: {path}")
            payload = path.read_bytes()
            record = f"F\t{relative}\t{len(payload)}\t{hashlib.sha256(payload).hexdigest()}\n"
            files += 1
            total_bytes += len(payload)
        else:
            raise SystemExit(f"unsupported dependency component entry: {path}")
        digest.update(record.encode())
    return {
        "sha256": digest.hexdigest(),
        "files": files,
        "bytes": total_bytes,
        "dirs": directories,
        "symlinks": symlinks,
    }


snowflake_donor = Path(snowflake_mcp_runtime["donor_path"])
snowflake_site = Path(snowflake_mcp_runtime["site_packages"])
for key, root in (("donor_trees", snowflake_donor), ("installed_trees", snowflake_site)):
    expected_trees = snowflake_mcp_runtime[key]
    actual_trees = {
        name: component_tree_identity(root / name) for name in sorted(expected_trees)
    }
    if actual_trees != expected_trees:
        raise SystemExit(f"Snowflake MCP component identity mismatch: {key}")

obsolete = ["mcp-2.0.0.dist-info", *snowflake_mcp_runtime["removed_mcp2_trees"]]
for name in obsolete:
    path = snowflake_site / name
    if path.exists() or path.is_symlink():
        raise SystemExit(f"obsolete Snowflake MCP 2 tree remains installed: {path}")

distribution_records = snowflake_mcp_runtime.get("distribution_records")
expected_distribution_records = {
    "installed_entrypoints": {
        "httpx": {
            "path": "/opt/local_servers/mcp-snowflake-server/.venv/bin/httpx",
            "sha256": "962c10bb001b246ec1e8102231eab5463e9a7a097e1a85d26b84612b7a12cd17",
            "size": 327,
            "mode": "0755",
        },
        "mcp": {
            "path": "/opt/local_servers/mcp-snowflake-server/.venv/bin/mcp",
            "sha256": "24a32fc6c81d80fbadf99f687dc09be2738c8d0a2ad5e1a6f6c1ff0f482a6b96",
            "size": 327,
            "mode": "0755",
        },
    },
    "removed_entrypoints": {
        "httpx2": {
            "path": "/opt/local_servers/mcp-snowflake-server/.venv/bin/httpx2",
            "base_sha256": "889148a2acab37fd38379a49fe79d9e2ab19b00449377c7915e732f4733b2bbe",
            "status": "absent",
        }
    },
    "editable_distribution": {
        "path": (
            "/opt/local_servers/mcp-snowflake-server/.venv/lib/python3.10/"
            "site-packages/mcp_snowflake_server-0.4.0.dist-info"
        ),
        "base_metadata_sha256": "3557166e394b25d750fe183cce403f761d34c871638c0ee06b9ce9ae9772bf15",
        "patched_metadata_sha256": "e24695b398213d544f479fc814154ad217c34835f07cfaf7ae88297ca7b3cf97",
        "base_record_sha256": "0f4c2fc4320c7ae83ee3da5320c5f92ae6e016315b2a1f9d7fc148244dfe8247",
        "patched_record_sha256": "0860da95e18081bc17601ab4682638f2ce31eef4eab26b11e7ef11bc0c742f8a",
        "base_tree": {
            "sha256": "ed08e8fe37a61e766b6e4500fde8bfc38699a8b3b5eaa5484c48d87109cafcca",
            "files": 10, "bytes": 46391, "dirs": 2, "symlinks": 0,
        },
        "patched_tree": {
            "sha256": "1f7bc1de08185154cdd82df8fd7e4688ec75261e1d8f17ed5ba233df160f3879",
            "files": 10, "bytes": 46392, "dirs": 2, "symlinks": 0,
        },
    },
    "verified_records": {
        "httpx-0.28.1.dist-info/RECORD": {
            "rows": 32, "verified_hashed_rows": 31, "unhashed_self_rows": 1,
        },
        "mcp-1.26.0.dist-info/RECORD": {
            "rows": 118, "verified_hashed_rows": 117, "unhashed_self_rows": 1,
        },
        "mcp_snowflake_server-0.4.0.dist-info/RECORD": {
            "rows": 12, "verified_hashed_rows": 11, "unhashed_self_rows": 1,
        },
    },
    "full_record_audit": {
        "record_files": 61,
        "rows": 12245,
        "verified_hashed_rows": 12184,
        "unhashed_self_rows": 61,
        "unique_owned_paths": 12184,
    },
}
if distribution_records != expected_distribution_records:
    raise SystemExit("Snowflake distribution RECORD manifest mismatch")

for entrypoint in expected_distribution_records["installed_entrypoints"].values():
    path = Path(entrypoint["path"])
    value = path.lstat()
    if (
        not stat.S_ISREG(value.st_mode)
        or stat.S_IMODE(value.st_mode) != 0o755
        or value.st_size != entrypoint["size"]
        or hashlib.sha256(path.read_bytes()).hexdigest() != entrypoint["sha256"]
        or not path.read_bytes().startswith(
            b"#!/opt/local_servers/mcp-snowflake-server/.venv/bin/python\n"
        )
    ):
        raise SystemExit(f"Snowflake console entrypoint identity mismatch: {path}")
stale_httpx2 = Path(expected_distribution_records["removed_entrypoints"]["httpx2"]["path"])
if stale_httpx2.exists() or stale_httpx2.is_symlink():
    raise SystemExit("stale Snowflake httpx2 entrypoint remains installed")

editable = expected_distribution_records["editable_distribution"]
if component_tree_identity(Path(editable["path"])) != editable["patched_tree"]:
    raise SystemExit("Snowflake editable distribution component identity mismatch")
for name, expected_hash in (
    ("METADATA", editable["patched_metadata_sha256"]),
    ("RECORD", editable["patched_record_sha256"]),
):
    path = Path(editable["path"]) / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise SystemExit(f"Snowflake editable distribution file identity mismatch: {path}")

project = Path("/opt/local_servers/mcp-snowflake-server")
for name, expected_hash in (
    ("pyproject.toml", expected_project_metadata["patched_pyproject_sha256"]),
    ("uv.lock", expected_project_metadata["patched_lock_sha256"]),
):
    path = project / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise SystemExit(f"Snowflake project metadata identity mismatch: {path}")
lock_header, *lock_parts = (project / "uv.lock").read_text().split("\n[[package]]\n")
if len(lock_parts) != expected_project_metadata["resolved_package_count"]:
    raise SystemExit("Snowflake lock package count mismatch")
lock_versions = {}
for part in lock_parts:
    name_match = re.search(r'^name = "([^"]+)"$', part, re.MULTILINE)
    version_match = re.search(r'^version = "([^"]+)"$', part, re.MULTILINE)
    if name_match is None:
        raise SystemExit("Snowflake lock has an unnamed package block")
    lock_versions.setdefault(name_match.group(1), []).append(
        version_match.group(1) if version_match else "editable"
    )
for name, version in {
    "mcp": "1.26.0",
    "httpx": "0.28.1",
    "httpcore": "1.0.9",
    "httpx-sse": "0.4.3",
    "pydantic-settings": "2.13.1",
}.items():
    if lock_versions.get(name) != [version]:
        raise SystemExit(f"Snowflake lock version mismatch: {name}")
for name in ("mcp-types", "httpcore2", "httpx2", "httpx2-jsfetch", "opentelemetry-api", "truststore"):
    if name in lock_versions:
        raise SystemExit(f"Snowflake lock retained MCP 2-only package: {name}")


def verify_record_universe(site_packages: Path) -> dict[str, int]:
    records = sorted(site_packages.glob("*.dist-info/RECORD"))
    owners = {}
    rows = 0
    verified = 0
    for record in records:
        names = set()
        self_rows = 0
        with record.open(newline="") as stream:
            for row in csv.reader(stream):
                rows += 1
                if len(row) != 3 or not row[0] or row[0] in names:
                    raise SystemExit(f"malformed/duplicate RECORD row: {record}:{row!r}")
                names.add(row[0])
                lexical = Path(os.path.normpath(site_packages / row[0]))
                try:
                    lexical.relative_to(site_packages.parents[2])
                except ValueError as exc:
                    raise SystemExit(f"RECORD member escapes venv: {record}:{row[0]}") from exc
                candidate = lexical.resolve(strict=True)
                try:
                    candidate.relative_to(site_packages.parents[2])
                except ValueError as exc:
                    raise SystemExit(f"RECORD member resolves outside venv: {record}:{row[0]}") from exc
                value = candidate.lstat()
                if not stat.S_ISREG(value.st_mode):
                    raise SystemExit(f"RECORD member is not regular: {record}:{row[0]}")
                if not row[1] and not row[2]:
                    if (
                        row[0] != record.relative_to(site_packages).as_posix()
                        or candidate != record.resolve(strict=True)
                    ):
                        raise SystemExit(f"unexpected unhashed RECORD member: {record}:{row[0]}")
                    self_rows += 1
                    continue
                if not row[1].startswith("sha256=") or not row[2].isdigit():
                    raise SystemExit(f"unsupported RECORD digest: {record}:{row[0]}")
                previous = owners.setdefault(candidate, record)
                if previous != record:
                    raise SystemExit(f"duplicate RECORD ownership: {candidate}: {previous}, {record}")
                payload = candidate.read_bytes()
                digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
                if row[1] != f"sha256={digest}" or int(row[2]) != len(payload):
                    raise SystemExit(f"RECORD content mismatch: {record}:{row[0]}")
                verified += 1
        if self_rows != 1:
            raise SystemExit(f"RECORD self-row coverage mismatch: {record}")
    return {
        "record_files": len(records),
        "rows": rows,
        "verified_hashed_rows": verified,
        "unhashed_self_rows": len(records),
        "unique_owned_paths": len(owners),
    }


if verify_record_universe(snowflake_site) != expected_distribution_records["full_record_audit"]:
    raise SystemExit("Snowflake installed RECORD universe mismatch")
snowflake_python = project / ".venv/bin/python"
import_probe = subprocess.run(
    [
        str(snowflake_python),
        "-I",
        "-c",
        (
            "from importlib.metadata import version; "
            "from mcp.server import Server; "
            "expected={'mcp':'1.26.0','httpx':'0.28.1','httpcore':'1.0.9',"
            "'httpx-sse':'0.4.3','pydantic-settings':'2.13.1'}; "
            "actual={k:version(k) for k in expected}; "
            "assert actual == expected, (actual, expected); "
            "assert hasattr(Server, 'list_resources'); "
            "print('snowflake_mcp126_imports=ok')"
        ),
    ],
    cwd=project,
    env={
        "PATH": production_path,
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "/app/runtime_patches",
    },
    capture_output=True,
    text=True,
    timeout=30,
    check=False,
)
if import_probe.returncode != 0 or import_probe.stdout.strip() != "snowflake_mcp126_imports=ok":
    raise SystemExit(
        "Snowflake MCP 1.26 import/version probe failed: "
        f"rc={import_probe.returncode} stdout={import_probe.stdout!r} stderr={import_probe.stderr[-2000:]!r}"
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
    Path("/app/runtime_patches"),
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
def resolve_configured_command(command: str, config: Path) -> str:
    placeholder = "${local_servers_paths}"
    placeholder_count = command.count(placeholder)
    if placeholder_count:
        prefix = placeholder + "/"
        if placeholder_count != 1 or not command.startswith(prefix):
            raise SystemExit(
                f"invalid local-server command interpolation in {config}: {command!r}"
            )
        remainder = command[len(prefix) :]
        parts = remainder.split("/")
        if (
            not remainder
            or "\\" in remainder
            or "\x00" in remainder
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise SystemExit(
                f"escaping local-server command interpolation in {config}: {command!r}"
            )
        command = "/opt/local_servers/" + "/".join(parts)
    if "${" in command:
        raise SystemExit(f"unresolved MCP command interpolation in {config}: {command!r}")
    return command


configs = sorted(Path("/app/configs/mcp_servers").glob("*.yaml"))
if len(configs) != 25:
    raise SystemExit(f"expected 25 MCP YAML files, found {len(configs)}")
configured_names = set()
resolved_commands = []
python_servers_seen = set()
actual_python_config_hashes = {}
for config in configs:
    try:
        data = yaml.safe_load(config.read_text())
        name = data["name"]
        command = data["params"]["command"]
        arguments = data["params"].get("args", [])
    except (OSError, TypeError, KeyError, yaml.YAMLError) as exc:
        raise SystemExit(f"malformed MCP config {config}: {exc}") from exc
    if not isinstance(name, str) or not name or name in configured_names:
        raise SystemExit(f"invalid or duplicate MCP name in {config}: {name!r}")
    if not isinstance(command, str) or not command:
        raise SystemExit(f"invalid MCP command in {config}: {command!r}")
    if command == "uv" or Path(command).name == "uv":
        raise SystemExit(f"runtime uv project launcher remains configured: {config}")
    command = resolve_configured_command(command, config)
    if name in python_mcp_configs:
        expected_filename, expected_executable, _expected_project = python_mcp_configs[name]
        if config.name != expected_filename:
            raise SystemExit(f"Python MCP config filename mismatch for {name}: {config.name}")
        config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
        if config_hash != python_mcp_config_hashes[config.name]:
            raise SystemExit(f"Python MCP config digest mismatch: {config}")
        expected_command = expected_executable if name == "youtube-transcript" else python_mcp_wrapper
        if command != expected_command:
            raise SystemExit(f"Python MCP command mismatch for {name}: {command!r}")
        if name != "youtube-transcript" and (
            not isinstance(arguments, list) or not arguments or arguments[0] != name
        ):
            raise SystemExit(f"Python MCP whitelist key is absent or reordered for {name}")
        python_servers_seen.add(name)
        actual_python_config_hashes[config.name] = config_hash
    configured_names.add(name)
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
if python_servers_seen != set(python_mcp_configs):
    raise SystemExit(
        "immutable Python MCP config universe mismatch: "
        f"missing={sorted(set(python_mcp_configs) - python_servers_seen)}, "
        f"unexpected={sorted(python_servers_seen - set(python_mcp_configs))}"
    )
if actual_python_config_hashes != python_mcp_config_hashes:
    raise SystemExit("immutable Python MCP config hash map mismatch")
patched_config_identity = "".join(
    f"{actual_python_config_hashes[name]}  {name}\n" for name in sorted(actual_python_config_hashes)
)
if hashlib.sha256(patched_config_identity.encode()).hexdigest() != (
    "0be2dc5984be8280f5ba5e6d781dad56e765a62f72fb9ebd791f51dcc80b1d60"
):
    raise SystemExit("immutable Python MCP patched config-set identity mismatch")
for name, (_filename, executable, _project) in python_mcp_configs.items():
    if not executable_by_uid(Path(executable)):
        raise SystemExit(f"Python MCP exec target is not executable by uid 65534: {name}: {executable}")
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
# uses a local data URL, is bounded, and runs as uid 65534 when the image
# build invokes the verifier as root.  The post-build verifier repeats it as
# the submitting uid, and the all-MCP gate later calls browser_tab_list through
# the actual MCP process.
BROWSER_PROBE_JS='const { chromium } = require("/opt/local_servers/playwright-mcp/node_modules/playwright-core");
let browser;
let closePromise;
let interruptedCode = 0;
async function closeBrowser() {
  if (browser && !closePromise) closePromise = browser.close();
  if (closePromise) await closePromise;
}
function requestInterrupt(code) {
  if (!interruptedCode) interruptedCode = code;
  process.exitCode = interruptedCode;
  if (browser) void closeBrowser().catch(error => { console.error(error); });
}
process.on("SIGTERM", () => { requestInterrupt(143); });
process.on("SIGINT", () => { requestInterrupt(130); });
(async () => {
  const installed = "/opt/ms-playwright/chromium-1182/chrome-linux/chrome";
  const selected = "/opt/ms-playwright/chromium_headless_shell-1182/chrome-linux/headless_shell";
  if (chromium.executablePath() !== installed)
    throw new Error(`unexpected Chromium path: ${chromium.executablePath()}`);
  try {
    browser = await chromium.launch({
      headless: true,
      chromiumSandbox: false,
      executablePath: selected,
      args: ["--disable-dev-shm-usage"],
    });
    if (interruptedCode) throw new Error(`interrupted during browser launch: ${interruptedCode}`);
    const page = await browser.newPage();
    await page.setContent("<!doctype html><title>toolathlon-portable-smoke</title>");
    if (await page.title() !== "toolathlon-portable-smoke")
      throw new Error("local page title mismatch");
  } finally {
    await closeBrowser();
  }
  if (interruptedCode) throw new Error(`browser probe interrupted: ${interruptedCode}`);
  console.log("toolathlon_playwright_rootless_launch=ok");
})().catch(error => { console.error(error); process.exit(interruptedCode || 1); });'
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
            "${BROWSER_ENV[@]}" /usr/bin/timeout --foreground --kill-after=30s 90s /usr/bin/node -e "${BROWSER_PROBE_JS}" \
            | grep -Fx 'toolathlon_playwright_rootless_launch=ok'
    else
        "${BROWSER_ENV[@]}" /usr/bin/timeout --foreground --kill-after=30s 90s /usr/bin/node -e "${BROWSER_PROBE_JS}" \
            | grep -Fx 'toolathlon_playwright_rootless_launch=ok'
    fi
fi

# Exercise every exact configured Python MCP command through initialize and
# live tool-name parity without an inherited home, proxy, or credential.
# Preserve only the pinned production runtime-patch Python path.
# The probe itself supplies a private Unix-socket-only PostgreSQL fixture and
# proves pre-EOF persistence, clean teardown, and no project/.venv mutation.
mkdir -m 0700 -- "${PYTHON_MCP_TMP}/.cache" "${PYTHON_MCP_TMP}/.uv-cache"
PYTHON_MCP_ENV=(
    /usr/bin/env -i
    "PATH=${PRODUCTION_PATH}"
    "HOME=${PYTHON_MCP_TMP}"
    "TMPDIR=${PYTHON_MCP_TMP}"
    "LANG=C.UTF-8"
    "LC_ALL=C.UTF-8"
    "PYTHONNOUSERSITE=1"
    "PYTHONDONTWRITEBYTECODE=1"
    "PYTHONPATH=/app/runtime_patches"
    "XDG_CACHE_HOME=${PYTHON_MCP_TMP}/.cache"
    "UV_CACHE_DIR=${PYTHON_MCP_TMP}/.uv-cache"
    "UV_PYTHON_DOWNLOADS=never"
    "PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_ROOT}"
)
PYTHON_MCP_PROBE=/usr/local/libexec/openreward/probe_toolathlon_python_mcp_launchers.py
if [[ "${VERIFY_MODE}" = --structural-only ]]; then
    printf '%s\n' 'toolathlon_python_mcp_rootless_probe=deferred_post_build'
else
    if command -v setpriv >/dev/null 2>&1 && [[ $(id -u) -eq 0 ]]; then
        chown -R 65534:65534 "${PYTHON_MCP_TMP}"
        setpriv --reuid=65534 --regid=65534 --clear-groups \
            "${PYTHON_MCP_ENV[@]}" /usr/bin/timeout --foreground --kill-after=120s 900s "${PYTHON_MCP_PROBE}" \
            --workspace-root "${PYTHON_MCP_TMP}/run" \
            | grep -F '"toolathlon_python_mcp_rootless_probe": "ok"'
    else
        "${PYTHON_MCP_ENV[@]}" /usr/bin/timeout --foreground --kill-after=120s 900s "${PYTHON_MCP_PROBE}" \
            --workspace-root "${PYTHON_MCP_TMP}/run" \
            | grep -F '"toolathlon_python_mcp_rootless_probe": "ok"'
    fi
fi

printf '%s\n' 'toolathlon_pbs_portability_verify=ok'
