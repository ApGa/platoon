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
import base64
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 build hosts
    import tomli as tomllib

BASE_SIF_SHA256 = "e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267"
BASE_OCI_DIGEST = "c1e60b3673c848b758bda1b32ec39898feb5d6ff4c7e55ead8661adb04b327bd"
TOOLATHLON_REVISION = "18e62c0d0419da88209af911528f2f8a0b44a5ed"
BASE_SERVER_SHA256 = "719ec82fb3efada36c01115df3f636f7311b40778ba7c52103d1e6950800152e"
PATCHED_SERVER_SHA256 = "26f58c49d42975616c5bef19b654cc4483e665b2107533000861cecbcd74c8a5"
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

# ``uv run`` is a project/environment reconciler, not an immutable process
# launcher.  Even with downloads disabled it may select a Python absent from
# the portable managed-runtime tree or try to rewrite ``.venv``/build metadata
# in the read-only SIF.  Every Python MCP already has a complete baked virtual
# environment, so launch that environment directly.  Pin both sides of every
# rewrite so a changed upstream config cannot be silently transformed.
PYTHON_MCP_LAUNCH_SPECS = {
    "arxiv-latex-mcp.yaml": {
        "name": "arxiv-latex",
        "project": "arxiv-latex-mcp",
        "uv_target": "python",
        "launcher": "python",
        "forward_uv_args": True,
        "base_sha256": "79e37d8aeb972f7c366513a7bc94c4d5ecc07cf61cae45466e1f0b8953ad6b4b",
        "patched_sha256": "5f03d88fb523ac22916544e80741e558b1dcba3b526c15f38b238d3aa6604059",
    },
    "arxiv_local.yaml": {
        "name": "arxiv_local",
        "project": "arxiv-mcp-server",
        "uv_target": "arxiv-mcp-server",
        "launcher": "arxiv-mcp-server",
        "forward_uv_args": True,
        "base_sha256": "aca12e47c68294a3a6b46f42d57393b508ee514cb19a75e83e13719d36f41adf",
        "patched_sha256": "2778c766fcfd923a9d1d258306099cd6e33d62cb9467ff963f6e74b4d82e6236",
    },
    "emails.yaml": {
        "name": "emails",
        "project": "emails-mcp",
        "uv_target": "emails-mcp",
        "launcher": "emails-mcp",
        "forward_uv_args": True,
        "base_sha256": "b45d5f05ccc1b870ae5fbf8621d6c1c890484a0f1012f3ead8d3d0d3367d9589",
        "patched_sha256": "d23fa2bb87ff05a7ff58a4d3284295011d1c63955084bcf9a90af56416827224",
    },
    "excel.yaml": {
        "name": "excel",
        "project": "excel-mcp-server",
        "uv_target": "excel-mcp-server",
        "launcher": "excel-mcp-server",
        "forward_uv_args": True,
        "base_sha256": "91906d6172840533d0f85ca665b4ddc467127d2d51da96f74a4e658f2eadd282",
        "patched_sha256": "4d3e13389e726a6f6bc4f2ad44c40b04209841228fa9683c1b7c02fe99626739",
    },
    "google_sheet.yaml": {
        "name": "google_sheet",
        "project": "mcp-google-sheets",
        "uv_target": "mcp-google-sheets",
        "launcher": "mcp-google-sheets",
        "forward_uv_args": False,
        "base_sha256": "4037507d771eac6c505afe0052082ae6ca1f93887cdaddcc86789be3935074d3",
        "patched_sha256": "75ba539b71e1a5d552ee6a3e97c2cfa5f26b8dc075d3d906c28a570197a798d8",
    },
    "pdf-tools.yaml": {
        "name": "pdf-tools",
        "project": "pdf-tools-mcp",
        "uv_target": "pdf-tools-mcp",
        "launcher": "pdf-tools-mcp",
        "forward_uv_args": True,
        "base_sha256": "5cbf37d03f8da5950e576cd05df32edafee29e17cd558507adfdee205ec999f7",
        "patched_sha256": "5595252990c6b82e0c868ace99eae7c4ed55bbd6afdd915825a80281c86c2271",
    },
    "pptx.yaml": {
        "name": "pptx",
        "project": "Office-PowerPoint-MCP-Server",
        "uv_target": "ppt_mcp_server",
        "launcher": "ppt_mcp_server",
        "forward_uv_args": False,
        "base_sha256": "17f0c9fa660078954112367f46d5d63551848cc4d77d9e6d415c5ed73cc6d5f3",
        "patched_sha256": "58714c846943cd860b432153feadb7445693490bafe0120ff1e92223ea7d6948",
    },
    "scholarly_search.yaml": {
        "name": "scholarly",
        "project": "mcp-scholarly",
        "uv_target": "mcp-scholarly",
        "launcher": "mcp-scholarly",
        "forward_uv_args": False,
        "base_sha256": "ad539c7d3c0b34ffd4ea967b1871aa3a47945be51211f4d6c34d2c1e5247d3ed",
        "patched_sha256": "c2aa0e9691aca4b4cabeb6a67316fb60105e85cb2d72ba6f829c0bb33ca22b5c",
    },
    "snowflake.yaml": {
        "name": "snowflake",
        "project": "mcp-snowflake-server",
        "uv_target": "mcp_snowflake_server",
        "launcher": "mcp_snowflake_server",
        "forward_uv_args": True,
        "base_sha256": "186189f53715bdeb9674c6be484361b1452686ea88ed52d41154242e19441a7a",
        "patched_sha256": "9030af77cd414390353fc4c39d185122d019dde1817d1a30d6b9b655641a118e",
    },
    "terminal.yaml": {
        "name": "terminal",
        "project": "cli-mcp-server",
        "uv_target": "cli-mcp-server",
        "launcher": "cli-mcp-server",
        "forward_uv_args": False,
        "base_sha256": "f16afa1a516cfd03574e9fdf3e0379044e105e6b1237737fe4cc9775d2ef2153",
        "patched_sha256": "6c40b77a472f4cb54377eed8d619c0f98c29ddb14da0aa8845c2c89779474e4a",
    },
    "word.yaml": {
        "name": "word",
        "project": "Office-Word-MCP-Server",
        "uv_target": "word_mcp_server",
        "launcher": "word_mcp_server",
        "forward_uv_args": False,
        "base_sha256": "64832c8e3a6009df1c042d3632b830d49c88807ebf017895582d9f6628f25761",
        "patched_sha256": "17ce31b6a3639ecde005ff314901e2fa05641dabf565d3059bd0dee0186178d2",
    },
}
YAHOO_FINANCE_CONFIG_SHA256 = "cfa6b124a48b2df46b35656a0f98f49b6208c0f73f9d9924ee7f0c4be9091f73"
YAHOO_FINANCE_PATCHED_CONFIG_SHA256 = "5fe25c1e236ed64196c487018990fb2ab4b305d2a9b6c3ecdc007af004d2fb57"
YOUTUBE_TRANSCRIPT_CONFIG_SHA256 = "ad8f1f3144698841b6e4e6b0485b9919c4684edac932274153e87363588f4972"
PYTHON_MCP_WRAPPER = "/usr/local/libexec/openreward/launch_toolathlon_python_mcp.py"
PYTHON_MCP_PROBE = "/usr/local/libexec/openreward/probe_toolathlon_python_mcp_launchers.py"
BASE_EXCEL_SERVER_SHA256 = "bc862fc7b268446f7bec47ece78548f4c1d29f5499a2cd9290856f2d7ba9effc"
PATCHED_EXCEL_SERVER_SHA256 = "ff9f1931ca5d0db8e4c8a63e07a5ea67a16024fb2fa337db55b0d596057cf26c"
BASE_EXCEL_MAIN_SHA256 = "c4b37f9fac83215030f8ed0178590e6d154f1a8249523f56f7483fb3dc710625"
PATCHED_EXCEL_MAIN_SHA256 = "f959bbb73eea5c18a74a7e59491e9f2305283443d6470cf10f34443dddb2401e"
BASE_WORD_MAIN_SHA256 = "967026a662e86b67d62994cbf020864995b66d48570ab0018dbe875a8002445d"
PATCHED_WORD_MAIN_SHA256 = "f8e0fbd953d549890645b8a4435d99af32adb735c50177aa2308d8e652ced1f2"
BASE_YAHOO_MCP_SESSION_SHA256 = "e37de044b79f6a048f5580e425c2a69628f6bd41dc940ce7dbbb0e3d4e3239f2"
PATCHED_YAHOO_MCP_SESSION_SHA256 = "d8412e3a9a75d1c2a36407ecd560169b0a049f9ef29feb6bb508753dd433e67b"
SNOWFLAKE_MCP126_DONOR = "/opt/local_servers/arxiv-latex-mcp/.venv/lib/python3.10/site-packages"
SNOWFLAKE_SITE_PACKAGES = "/opt/local_servers/mcp-snowflake-server/.venv/lib/python3.10/site-packages"
SNOWFLAKE_PROJECT = "/opt/local_servers/mcp-snowflake-server"
SNOWFLAKE_BASE_PYPROJECT_SHA256 = "cf9e6b2e2b966d48c9b3b6615d66e00fdd8395dd39df9de49c2da9bc2c9cb3b7"
SNOWFLAKE_PATCHED_PYPROJECT_SHA256 = "8d8ed2e26ffe2621403a640fff0d598a35a78dbb06da2867e0bcf6ae80df2743"
SNOWFLAKE_BASE_LOCK_SHA256 = "83a120808c4458ed40505c395b6cbdd812a1534485169c86ce0ef39abbce0fe3"
SNOWFLAKE_PATCHED_LOCK_SHA256 = "2945a5f989fd610b8a31017cd29c41cd3a662e5267f1f17d4871afb7f1ace81c"
SNOWFLAKE_DONOR_LOCK_SHA256 = "9e721de6dc27f29169af6af6e883ab40dc08f034fcd64e5f5a56219fbfcb6119"
SNOWFLAKE_BASE_TREES = {
    "mcp": {
        "tree_sha256": "c402262b0535d45286ad181cfd36304edc0e20435420b66c3c0d216f61d0276c",
        "file_count": 230,
        "total_bytes": 1945318,
        "symlink_count": 0,
        "directory_count": 36,
    },
    "mcp-2.0.0.dist-info": {
        "tree_sha256": "972a8cd55b10cd9ad0a0ede619599f9c23cb54890fc49b24d76d8363d7c11813",
        "file_count": 7,
        "total_bytes": 19990,
        "symlink_count": 0,
        "directory_count": 1,
    },
}
SNOWFLAKE_MCP126_TREES = {
    "httpcore": {
        "sha256": "4979575d1d1387da735e5ca878f295b56a0f6613dae7787c318d23c2b726f763",
        "files": 32,
        "bytes": 261935,
        "dirs": 4,
        "symlinks": 0,
    },
    "httpcore-1.0.9.dist-info": {
        "sha256": "a440271dc5541f1c93025ebf150d843b19fba319cfae589816331d438d21d105",
        "files": 6,
        "bytes": 26259,
        "dirs": 2,
        "symlinks": 0,
    },
    "httpx": {
        "sha256": "dcab6e2a667adb72691431d65abb39d1b4f45f896bdca248c14bb1d7a4b1e13e",
        "files": 47,
        "bytes": 517126,
        "dirs": 4,
        "symlinks": 0,
    },
    "httpx-0.28.1.dist-info": {
        "sha256": "ba4344ded20f1905831db1c8dc3159a6ca4e45b0bcbe43856a256f257963d1fd",
        "files": 7,
        "bytes": 11155,
        "dirs": 2,
        "symlinks": 0,
    },
    "httpx_sse": {
        "sha256": "1b570184ac2bc1746bde2f53dc4c292156ac15fa220fc314ad1a7f627d9cd917",
        "files": 11,
        "bytes": 17379,
        "dirs": 2,
        "symlinks": 0,
    },
    "httpx_sse-0.4.3.dist-info": {
        "sha256": "8581def6ee925f4b5f7f189da8ca1cafc0d885b27aaace1411ebb205a5433a9a",
        "files": 7,
        "bytes": 11887,
        "dirs": 2,
        "symlinks": 0,
    },
    "mcp": {
        "sha256": "f8a26827d5a9fa1aa752ce689cfe94bac67db6730d0ca82f5ac4bf083d43c9c7",
        "files": 192,
        "bytes": 1272614,
        "dirs": 44,
        "symlinks": 0,
    },
    "mcp-1.26.0.dist-info": {
        "sha256": "7cd7b0a4500824b148345921c27b8a4a0135c55214e1531075a076b5d7b2e168",
        "files": 7,
        "bytes": 100954,
        "dirs": 2,
        "symlinks": 0,
    },
    "pydantic_settings": {
        "sha256": "b5fc4afa1a4fac6637e7a7b47341d3888a6cd80e90dab103026d2b17f27e118e",
        "files": 45,
        "bytes": 378393,
        "dirs": 6,
        "symlinks": 0,
    },
    "pydantic_settings-2.13.1.dist-info": {
        "sha256": "9e9610dd15656b0843bff454e6e112da96a00d0cff6fef5e143152c9e9440733",
        "files": 6,
        "bytes": 7322,
        "dirs": 2,
        "symlinks": 0,
    },
}
SNOWFLAKE_MCP126_CLOSURE_SHA256 = "e7b1bf8895eec6e95d738b68a53cd4f0eaad9672284c9cdbe5ec3613399c2eec"
SNOWFLAKE_INSTALLED_MCP126_TREES = {
    **SNOWFLAKE_MCP126_TREES,
    "httpx-0.28.1.dist-info": {
        "sha256": "e3d5c1a896da31543eacaf2feabd3d71e678f41b3e4ad8aeee466d730fc06f97",
        "files": 7,
        "bytes": 11155,
        "dirs": 2,
        "symlinks": 0,
    },
    "mcp-1.26.0.dist-info": {
        "sha256": "b2513aa50f095f094bbb22353385cf627e6e68b73c4e31a2472b73cbfe21b464",
        "files": 7,
        "bytes": 100954,
        "dirs": 2,
        "symlinks": 0,
    },
}
SNOWFLAKE_INSTALLED_MCP126_CLOSURE_SHA256 = "b618e4359067acb69893820bae6a970c4d8a3714d77550c972346f6120497bc5"
SNOWFLAKE_BASE_MCP2_RECORD_SHA256 = "60c02ca68bf53784cd57a6db77587fb747594b05f1f52874919dc8cf4ecc12c6"
SNOWFLAKE_DONOR_MCP126_RECORD_SHA256 = "9fb533e1b39001b3dbb42cd1923045e4311c0f6e4ae304caa96dcafa60c91a64"
SNOWFLAKE_INSTALLED_MCP126_RECORD_SHA256 = "6eb84b420a62e9c103767d7ac1b8e546e4aebde70728a935c8636db85f15efa7"
SNOWFLAKE_INSTALLED_HTTPX_RECORD_SHA256 = "a74f9379bd94e5b414e56a4c7e4437316dd09e0c8997121da43b9c099067522b"
SNOWFLAKE_STALE_MCP2_TREES = {
    "httpcore2": {
        "sha256": "a3a7a88e450d1fabd33315da01dcda159c01bd58b3af698516b7862ec483dc44",
        "files": 32,
        "bytes": 269014,
        "dirs": 4,
        "symlinks": 0,
    },
    "httpcore2-2.10.0.dist-info": {
        "sha256": "9eec80a3c918162966b7b5f15a1929c7ba0c1985ff089161473c5205b03c800b",
        "files": 6,
        "bytes": 30374,
        "dirs": 2,
        "symlinks": 0,
    },
    "httpx2": {
        "sha256": "9f99700a783ff92f1ce37a723543ce6b7fe50e4892841b360172acefbf8b6e05",
        "files": 55,
        "bytes": 630518,
        "dirs": 5,
        "symlinks": 0,
    },
    "httpx2-2.10.0.dist-info": {
        "sha256": "6bfbb647eeebbbbf470d5833f8e966a3b3daf61ede51d0ffb68258c631f56a71",
        "files": 7,
        "bytes": 15056,
        "dirs": 2,
        "symlinks": 0,
    },
    "mcp_types": {
        "sha256": "d277fadd8e06b95fa0f4b6370d2508207bf72ad90bd6db0fc67c5598bc57c77e",
        "files": 17,
        "bytes": 584978,
        "dirs": 6,
        "symlinks": 0,
    },
    "mcp_types-2.0.0.dist-info": {
        "sha256": "c467c6ee64f0e3cc0e05fe1035a9a02cf4d887bed49d221c8e50a4e2e77ab0cd",
        "files": 6,
        "bytes": 4288,
        "dirs": 2,
        "symlinks": 0,
    },
    "opentelemetry": {
        "sha256": "904385c3c61d97c58c7b7fdcc39c700b647cbef5a871e7f826cc40f06927697f",
        "files": 62,
        "bytes": 280206,
        "dirs": 27,
        "symlinks": 0,
    },
    "opentelemetry_api-1.44.0.dist-info": {
        "sha256": "ba54e3dc03672344621cd1b79a93c00717d2828ae103d72802d9d7c061be1f28",
        "files": 7,
        "bytes": 17902,
        "dirs": 2,
        "symlinks": 0,
    },
    "truststore": {
        "sha256": "967cfda76dc2cc91317d6e30251a5e504b86b9b28acc3b7773fc704ba2d0c451",
        "files": 7,
        "bytes": 54709,
        "dirs": 1,
        "symlinks": 0,
    },
    "truststore-0.10.4.dist-info": {
        "sha256": "a992e608b22932d34fc2d140eba7be46895676c5ffecef13730e10c8babddb17",
        "files": 6,
        "bytes": 6634,
        "dirs": 2,
        "symlinks": 0,
    },
}
SNOWFLAKE_STALE_MCP2_CLOSURE_SHA256 = "ca2c7c3ba32778fb8de51822f75f8bef3cfcd83696e50229cdec9a72c029ddd0"
SNOWFLAKE_STALE_HTTPX2_ENTRYPOINT_SHA256 = "889148a2acab37fd38379a49fe79d9e2ab19b00449377c7915e732f4733b2bbe"
SNOWFLAKE_DONOR_MCP_ENTRYPOINT_SHA256 = "d46fa568366451ff645afb9e4a8105f38b93d3d7ff22ab35a0cde18311bc3cb9"
SNOWFLAKE_DONOR_HTTPX_ENTRYPOINT_SHA256 = "af7b8d64642806a7bb700eb23607fb506547399921626c1b701edf73d33ea9e8"
SNOWFLAKE_INSTALLED_MCP_ENTRYPOINT_SHA256 = "24a32fc6c81d80fbadf99f687dc09be2738c8d0a2ad5e1a6f6c1ff0f482a6b96"
SNOWFLAKE_INSTALLED_HTTPX_ENTRYPOINT_SHA256 = "962c10bb001b246ec1e8102231eab5463e9a7a097e1a85d26b84612b7a12cd17"
SNOWFLAKE_BASE_EDITABLE_METADATA_SHA256 = "3557166e394b25d750fe183cce403f761d34c871638c0ee06b9ce9ae9772bf15"
SNOWFLAKE_PATCHED_EDITABLE_METADATA_SHA256 = "e24695b398213d544f479fc814154ad217c34835f07cfaf7ae88297ca7b3cf97"
SNOWFLAKE_BASE_EDITABLE_RECORD_SHA256 = "0f4c2fc4320c7ae83ee3da5320c5f92ae6e016315b2a1f9d7fc148244dfe8247"
SNOWFLAKE_PATCHED_EDITABLE_RECORD_SHA256 = "0860da95e18081bc17601ab4682638f2ce31eef4eab26b11e7ef11bc0c742f8a"
SNOWFLAKE_BASE_EDITABLE_TREE = {
    "sha256": "ed08e8fe37a61e766b6e4500fde8bfc38699a8b3b5eaa5484c48d87109cafcca",
    "files": 10,
    "bytes": 46391,
    "dirs": 2,
    "symlinks": 0,
}
SNOWFLAKE_PATCHED_EDITABLE_TREE = {
    "sha256": "1f7bc1de08185154cdd82df8fd7e4688ec75261e1d8f17ed5ba233df160f3879",
    "files": 10,
    "bytes": 46392,
    "dirs": 2,
    "symlinks": 0,
}
SNOWFLAKE_LOCK_REMOVED_PACKAGES = frozenset(
    {
        "httpcore2",
        "httpx2",
        "httpx2-jsfetch",
        "mcp",
        "mcp-types",
        "opentelemetry-api",
        "truststore",
    }
)
SNOWFLAKE_LOCK_DONOR_PACKAGES = (
    "mcp",
    "httpcore",
    "httpx",
    "httpx-sse",
    "pydantic-settings",
)


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


def component_tree_identity(root: Path) -> dict[str, int | str]:
    """Canonical identity used for the audited Snowflake dependency closure."""

    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"dependency component is missing or indirect: {root}")
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    directories = 0
    symlinks = 0
    paths = [root, *sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())]
    for path in paths:
        relative = "." if path == root else path.relative_to(root).as_posix()
        value = path.lstat()
        if stat.S_ISDIR(value.st_mode):
            record = f"D\t{relative}\n"
            directories += 1
        elif stat.S_ISLNK(value.st_mode):
            target = os.readlink(path)
            record = f"L\t{relative}\t{target}\n"
            symlinks += 1
        elif stat.S_ISREG(value.st_mode):
            record = f"F\t{relative}\t{value.st_size}\t{sha256(path)}\n"
            files += 1
            total_bytes += value.st_size
        else:
            raise RuntimeError(f"unsupported dependency component entry: {path}")
        digest.update(record.encode())
    return {
        "sha256": digest.hexdigest(),
        "files": files,
        "bytes": total_bytes,
        "dirs": directories,
        "symlinks": symlinks,
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
    # The base image's umask can create /opt/uv as 0750.  The managed Python
    # payload may itself be world-traversable, but every rewritten MCP venv
    # still fails for the unprivileged container uid if this parent is not.
    target_python.parent.chmod(0o755)
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


def config_set_sha256(values: dict[str, str]) -> str:
    identity = "".join(f"{values[name]}  {name}\n" for name in sorted(values))
    return hashlib.sha256(identity.encode()).hexdigest()


def assert_baked_launcher(root: Path, image_path: str) -> None:
    launcher = rooted(root, image_path)
    value = launcher.lstat()
    if stat.S_ISLNK(value.st_mode):
        target = os.readlink(launcher)
        if launcher.name not in {"python", "python3"}:
            raise RuntimeError(f"baked MCP launcher is an unexpected symlink: {image_path} -> {target}")
        if target.startswith("/"):
            target_path = rooted(root, target)
        else:
            target_path = launcher.parent / target
        target_value = target_path.lstat()
        if stat.S_ISLNK(target_value.st_mode):
            nested = os.readlink(target_path)
            target_path = rooted(root, nested) if nested.startswith("/") else target_path.parent / nested
            target_value = target_path.lstat()
        if not stat.S_ISREG(target_value.st_mode) or target_value.st_mode & 0o111 == 0:
            raise RuntimeError(f"baked MCP Python launcher target is not executable: {image_path}")
        return
    if not stat.S_ISREG(value.st_mode) or value.st_mode & 0o111 == 0:
        raise RuntimeError(f"baked MCP launcher is not an executable regular file: {image_path}")
    with launcher.open("rb") as stream:
        first_line = stream.readline(8192).decode(errors="strict").rstrip("\n")
    expected_shebang = f"#!{image_path.rsplit('/', 1)[0]}/python"
    if first_line != expected_shebang:
        raise RuntimeError(
            f"baked MCP entrypoint shebang mismatch: {image_path}: {first_line!r} != {expected_shebang!r}"
        )


def patch_excel_logging(root: Path) -> None:
    """Keep Excel output writable and make stdio EOF teardown clean."""

    server = rooted(root, "/opt/local_servers/excel-mcp-server/src/excel_mcp/server.py")
    assert_digest(server, BASE_EXCEL_SERVER_SHA256, "base Excel MCP server")
    source = server.read_text()
    old = "        logging.FileHandler(LOG_FILE)\n"
    new = "        logging.StreamHandler()\n"
    if source.count(old) != 1:
        raise RuntimeError("Excel MCP logging patch anchor mismatch")
    server.write_text(source.replace(old, new))
    server.chmod(0o644)
    if PATCHED_EXCEL_SERVER_SHA256 != "TO_BE_FILLED":
        assert_digest(server, PATCHED_EXCEL_SERVER_SHA256, "patched Excel MCP server")

    main = rooted(root, "/opt/local_servers/excel-mcp-server/src/excel_mcp/__main__.py")
    assert_digest(main, BASE_EXCEL_MAIN_SHA256, "base Excel MCP __main__")
    source = main.read_text()
    import_anchor = "import typer\n"
    if source.count(import_anchor) != 1 or "import sys\n" in source:
        raise RuntimeError("Excel MCP __main__ import patch anchor mismatch")
    source = source.replace(import_anchor, "import sys\n\nimport typer\n", 1)
    old_stdio = '''@app.command()
def stdio():
    """Start Excel MCP Server in stdio mode"""
    try:
        run_stdio()
    except KeyboardInterrupt:
        print("\\nShutting down server...")
    except Exception as e:
        print(f"\\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Service stopped.")
'''
    new_stdio = old_stdio.replace(
        '        print("Service stopped.")\n',
        '        print("Service stopped.", file=sys.stderr)\n',
    )
    if source.count(old_stdio) != 1:
        raise RuntimeError("Excel MCP stdio teardown patch anchor mismatch")
    main.write_text(source.replace(old_stdio, new_stdio))
    main.chmod(0o644)
    assert_digest(main, PATCHED_EXCEL_MAIN_SHA256, "patched Excel MCP __main__")


def patch_word_teardown(root: Path) -> None:
    """Do not pass a FastMCP object to the generated console-script sys.exit."""

    main = rooted(
        root,
        "/opt/local_servers/Office-Word-MCP-Server/word_document_server/main.py",
    )
    assert_digest(main, BASE_WORD_MAIN_SHA256, "base Word MCP main")
    source = main.read_text()
    old = "    return mcp\n\n\ndef main():\n"
    new = "    return None\n\n\ndef main():\n"
    if source.count(old) != 1:
        raise RuntimeError("Word MCP clean teardown patch anchor mismatch")
    main.write_text(source.replace(old, new))
    main.chmod(0o644)
    assert_digest(main, PATCHED_WORD_MAIN_SHA256, "patched Word MCP main")


def patch_yahoo_teardown(root: Path) -> None:
    """Close MCP 1.6's incoming writer so Yahoo exits cleanly on stdio EOF."""

    session = rooted(
        root,
        ("/opt/local_servers/yahoo-finance-mcp/.venv/lib/python3.11/site-packages/mcp/server/session.py"),
    )
    assert_digest(session, BASE_YAHOO_MCP_SESSION_SHA256, "base Yahoo MCP session")
    source = session.read_text()
    old = "        return True\n\n    async def _received_request(\n"
    new = (
        "        return True\n\n"
        "    async def _receive_loop(self) -> None:\n"
        "        async with self._incoming_message_stream_writer:\n"
        "            await super()._receive_loop()\n\n"
        "    async def _received_request(\n"
    )
    if source.count(old) != 1:
        raise RuntimeError("Yahoo MCP clean teardown patch anchor mismatch")
    session.write_text(source.replace(old, new))
    session.chmod(0o644)
    assert_digest(session, PATCHED_YAHOO_MCP_SESSION_SHA256, "patched Yahoo MCP session")


def uv_lock_blocks(source: str) -> tuple[str, list[tuple[str, str]]]:
    """Parse a uv lock into its header and stable ordered package blocks."""

    header, *parts = source.split("\n[[package]]\n")
    if not parts:
        raise RuntimeError("Snowflake uv.lock has no package blocks")
    blocks: list[tuple[str, str]] = []
    for part in parts:
        match = re.search(r'^name = "([^"]+)"$', part, re.MULTILINE)
        if match is None:
            raise RuntimeError("Snowflake uv.lock package block has no exact name")
        blocks.append((match.group(1), "[[package]]\n" + part.rstrip() + "\n"))
    return header.rstrip() + "\n", blocks


def patched_snowflake_metadata(root: Path) -> tuple[str, str]:
    """Return exact coherent pyproject/lock bytes without touching the tree."""

    project = rooted(root, SNOWFLAKE_PROJECT)
    pyproject = project / "pyproject.toml"
    target_lock = project / "uv.lock"
    donor_lock = rooted(root, f"{SNOWFLAKE_MCP126_DONOR.rsplit('/.venv/', 1)[0]}/uv.lock")
    assert_digest(pyproject, SNOWFLAKE_BASE_PYPROJECT_SHA256, "base Snowflake pyproject")
    assert_digest(target_lock, SNOWFLAKE_BASE_LOCK_SHA256, "base Snowflake uv.lock")
    assert_digest(donor_lock, SNOWFLAKE_DONOR_LOCK_SHA256, "Snowflake MCP donor uv.lock")

    pyproject_source = pyproject.read_text()
    old_requirement = '    "mcp>=1.0.0",\n'
    new_requirement = '    "mcp==1.26.0",\n'
    if pyproject_source.count(old_requirement) != 1:
        raise RuntimeError("Snowflake pyproject MCP requirement anchor mismatch")
    patched_pyproject = pyproject_source.replace(old_requirement, new_requirement)
    if hashlib.sha256(patched_pyproject.encode()).hexdigest() != SNOWFLAKE_PATCHED_PYPROJECT_SHA256:
        raise RuntimeError("patched Snowflake pyproject identity mismatch")

    header, target_blocks = uv_lock_blocks(target_lock.read_text())
    _, donor_blocks = uv_lock_blocks(donor_lock.read_text())
    donor_by_name: dict[str, str] = {}
    for name, block in donor_blocks:
        if name in SNOWFLAKE_LOCK_DONOR_PACKAGES:
            if name in donor_by_name:
                raise RuntimeError(f"duplicate Snowflake donor lock package: {name}")
            donor_by_name[name] = block
    if set(donor_by_name) != set(SNOWFLAKE_LOCK_DONOR_PACKAGES):
        raise RuntimeError("Snowflake donor lock package set is incomplete")

    removed_counts = {name: 0 for name in SNOWFLAKE_LOCK_REMOVED_PACKAGES}
    kept: list[tuple[str, str]] = []
    for name, block in target_blocks:
        if name in SNOWFLAKE_LOCK_REMOVED_PACKAGES:
            removed_counts[name] += 1
            continue
        if name == "mcp-snowflake-server":
            old_specifier = '{ name = "mcp", specifier = ">=1.0.0" },'
            new_specifier = '{ name = "mcp", specifier = "==1.26.0" },'
            if block.count(old_specifier) != 1:
                raise RuntimeError("Snowflake root lock MCP specifier anchor mismatch")
            block = block.replace(old_specifier, new_specifier)
        kept.append((name, block))
    if set(removed_counts.values()) != {1}:
        raise RuntimeError(f"Snowflake obsolete lock package multiplicity changed: {removed_counts}")

    optional_cli = (
        '\n[package.optional-dependencies]\ncli = [\n    { name = "python-dotenv" },\n    { name = "typer" },\n]\n'
    )
    for name in SNOWFLAKE_LOCK_DONOR_PACKAGES:
        block = donor_by_name[name]
        if name == "mcp":
            if block.count(optional_cli) != 1:
                raise RuntimeError("Snowflake donor MCP optional CLI lock anchor mismatch")
            block = block.replace(optional_cli, "\n")
        kept.append((name, block))
    kept.sort(key=lambda item: item[0])
    patched_lock = header + "\n" + "\n".join(block.rstrip() for _, block in kept) + "\n"
    if hashlib.sha256(patched_lock.encode()).hexdigest() != SNOWFLAKE_PATCHED_LOCK_SHA256:
        raise RuntimeError("patched Snowflake uv.lock identity mismatch")

    parsed = tomllib.loads(patched_lock)
    packages = parsed.get("package")
    if not isinstance(packages, list):
        raise RuntimeError("patched Snowflake uv.lock package list is malformed")
    versions: dict[str, list[str]] = {}
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str):
            raise RuntimeError("patched Snowflake uv.lock package record is malformed")
        versions.setdefault(package["name"], []).append(str(package.get("version", "editable")))
    expected_versions = {
        "httpcore": ["1.0.9"],
        "httpx": ["0.28.1"],
        "httpx-sse": ["0.4.3"],
        "mcp": ["1.26.0"],
        "pydantic-settings": ["2.13.1"],
    }
    for name, expected in expected_versions.items():
        if versions.get(name) != expected:
            raise RuntimeError(f"patched Snowflake lock version mismatch: {name}")
    if set(versions).intersection(SNOWFLAKE_LOCK_REMOVED_PACKAGES - {"mcp"}):
        raise RuntimeError("patched Snowflake uv.lock retained an MCP 2-only package")
    return patched_pyproject, patched_lock


def install_snowflake_metadata(root: Path) -> dict[str, object]:
    """Install the exact lock pair and prove image-pinned uv accepts it offline."""

    patched_pyproject, patched_lock = patched_snowflake_metadata(root)
    project = rooted(root, SNOWFLAKE_PROJECT)
    pyproject = project / "pyproject.toml"
    lock = project / "uv.lock"
    pyproject.write_text(patched_pyproject)
    lock.write_text(patched_lock)
    pyproject.chmod(0o644)
    lock.chmod(0o644)
    assert_digest(pyproject, SNOWFLAKE_PATCHED_PYPROJECT_SHA256, "patched Snowflake pyproject")
    assert_digest(lock, SNOWFLAKE_PATCHED_LOCK_SHA256, "patched Snowflake uv.lock")

    uv = rooted(root, "/usr/local/bin/uv")
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "UV_NO_CACHE": "1",
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_PYTHON_INSTALL_DIR": str(rooted(root, PORTABLE_UV_PREFIX)),
    }
    completed = subprocess.run(
        [str(uv), "lock", "--check", "--offline", "--project", str(project)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "image-pinned uv rejected patched Snowflake lock offline: "
            f"rc={completed.returncode} stderr={completed.stderr[-4000:]!r}"
        )
    assert_digest(pyproject, SNOWFLAKE_PATCHED_PYPROJECT_SHA256, "uv-checked Snowflake pyproject")
    assert_digest(lock, SNOWFLAKE_PATCHED_LOCK_SHA256, "uv-checked Snowflake uv.lock")
    return {
        "base_pyproject_sha256": SNOWFLAKE_BASE_PYPROJECT_SHA256,
        "patched_pyproject_sha256": SNOWFLAKE_PATCHED_PYPROJECT_SHA256,
        "base_lock_sha256": SNOWFLAKE_BASE_LOCK_SHA256,
        "patched_lock_sha256": SNOWFLAKE_PATCHED_LOCK_SHA256,
        "donor_lock_sha256": SNOWFLAKE_DONOR_LOCK_SHA256,
        "offline_uv_lock_check": "passed",
        "resolved_package_count": 66,
    }


def verify_distribution_record(
    site_packages: Path,
    record: Path,
    owners: dict[Path, Path] | None = None,
) -> dict[str, int]:
    """Verify every hashed RECORD member, including venv console scripts."""

    venv = site_packages.parents[2]
    rows = 0
    verified = 0
    self_rows = 0
    row_names: set[str] = set()
    with record.open(newline="") as stream:
        for row in csv.reader(stream):
            rows += 1
            if len(row) != 3 or not row[0]:
                raise RuntimeError(f"malformed installed distribution RECORD row: {record}:{rows}")
            if row[0] in row_names:
                raise RuntimeError(f"duplicate installed distribution RECORD row: {record}:{row[0]}")
            row_names.add(row[0])
            lexical = Path(os.path.normpath(site_packages / row[0]))
            try:
                lexical.relative_to(venv)
            except ValueError as exc:
                raise RuntimeError(f"installed RECORD escapes its venv: {record}:{row[0]}") from exc
            candidate = lexical.resolve(strict=True)
            try:
                candidate.relative_to(venv)
            except ValueError as exc:
                raise RuntimeError(f"installed RECORD resolves outside its venv: {record}:{row[0]}") from exc
            value = candidate.lstat()
            if not stat.S_ISREG(value.st_mode):
                raise RuntimeError(f"installed RECORD member is not a regular file: {record}:{row[0]}")
            if owners is not None and row[1]:
                previous = owners.setdefault(candidate, record)
                if previous != record:
                    raise RuntimeError(
                        f"installed RECORD member has duplicate ownership: {candidate}: {previous}, {record}"
                    )
            if not row[1] and not row[2]:
                expected_self = record.relative_to(site_packages).as_posix()
                if row[0] != expected_self or candidate != record.resolve(strict=True):
                    raise RuntimeError(f"unexpected unhashed installed RECORD member: {record}:{row[0]}")
                self_rows += 1
                continue
            if not row[1].startswith("sha256=") or not row[2].isdigit():
                raise RuntimeError(f"unsupported installed RECORD digest row: {record}:{row[0]}")
            payload = candidate.read_bytes()
            encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
            if row[1] != f"sha256={encoded}" or int(row[2]) != len(payload):
                raise RuntimeError(f"installed distribution RECORD mismatch: {record}:{row[0]}")
            verified += 1
    if rows == 0 or self_rows != 1 or verified != rows - 1:
        raise RuntimeError(f"installed distribution RECORD coverage mismatch: {record}")
    return {"rows": rows, "verified_hashed_rows": verified, "unhashed_self_rows": self_rows}


def verify_all_distribution_records(site_packages: Path) -> dict[str, int]:
    records = sorted(site_packages.glob("*.dist-info/RECORD"))
    owners: dict[Path, Path] = {}
    rows = 0
    verified = 0
    for record in records:
        result = verify_distribution_record(site_packages, record, owners)
        rows += result["rows"]
        verified += result["verified_hashed_rows"]
    if len(records) != 61 or rows != 12245 or verified != rows - len(records):
        raise RuntimeError(
            f"installed Snowflake RECORD universe changed: records={len(records)} rows={rows} verified={verified}"
        )
    return {
        "record_files": len(records),
        "rows": rows,
        "verified_hashed_rows": verified,
        "unhashed_self_rows": len(records),
        "unique_owned_paths": len(owners),
    }


def repair_snowflake_distribution_records(root: Path) -> dict[str, object]:
    """Retarget console scripts and make all patched distribution RECORDs exact."""

    donor_venv = rooted(root, SNOWFLAKE_MCP126_DONOR.rsplit("/lib/", 1)[0])
    target_venv = rooted(root, SNOWFLAKE_SITE_PACKAGES.rsplit("/lib/", 1)[0])
    target_site = rooted(root, SNOWFLAKE_SITE_PACKAGES)
    target_bin = target_venv / "bin"

    donor_scripts = {
        "mcp": (SNOWFLAKE_DONOR_MCP_ENTRYPOINT_SHA256, SNOWFLAKE_INSTALLED_MCP_ENTRYPOINT_SHA256),
        "httpx": (
            SNOWFLAKE_DONOR_HTTPX_ENTRYPOINT_SHA256,
            SNOWFLAKE_INSTALLED_HTTPX_ENTRYPOINT_SHA256,
        ),
    }
    old_shebang = b"#!/opt/local_servers/arxiv-latex-mcp/.venv/bin/python\n"
    new_shebang = b"#!/opt/local_servers/mcp-snowflake-server/.venv/bin/python\n"
    installed_scripts = {}
    for name, (donor_hash, installed_hash) in donor_scripts.items():
        source = donor_venv / "bin" / name
        assert_digest(source, donor_hash, f"Snowflake donor {name} entrypoint")
        value = source.read_bytes()
        if value.count(old_shebang) != 1 or not value.startswith(old_shebang):
            raise RuntimeError(f"Snowflake donor {name} shebang anchor mismatch")
        value = value.replace(old_shebang, new_shebang, 1)
        destination = target_bin / name
        if name == "mcp":
            assert_digest(destination, installed_hash, "base Snowflake mcp entrypoint")
        elif destination.exists() or destination.is_symlink():
            raise RuntimeError(f"unexpected preexisting Snowflake entrypoint: {destination}")
        destination.write_bytes(value)
        destination.chmod(0o755)
        assert_digest(destination, installed_hash, f"installed Snowflake {name} entrypoint")
        installed_scripts[name] = {
            "path": f"{SNOWFLAKE_PROJECT}/.venv/bin/{name}",
            "sha256": installed_hash,
            "size": 327,
            "mode": "0755",
        }

    stale_httpx2 = target_bin / "httpx2"
    assert_digest(stale_httpx2, SNOWFLAKE_STALE_HTTPX2_ENTRYPOINT_SHA256, "stale httpx2 entrypoint")
    stale_httpx2.unlink()

    record_patches = {
        "mcp-1.26.0.dist-info/RECORD": (
            "../../../bin/mcp,sha256=1G-laDZkUf9kWvueSoEF84uT09f_Iqs1oM3hgxG8PLk,322\n",
            "../../../bin/mcp,sha256=JKMvxsgdgPut-Z9ofcCb4nOMjQoq1eGm9sH_D0gqa5Y,327\n",
            SNOWFLAKE_INSTALLED_MCP126_RECORD_SHA256,
        ),
        "httpx-0.28.1.dist-info/RECORD": (
            "../../../bin/httpx,sha256=r3uNZGQoBqe7cA6yNgf7UGVHOZkhYmwbcB7fc9M-qeg,322\n",
            "../../../bin/httpx,sha256=liwQuwAbJG7B6BAiMeq1Rj6aegl-GoXSa4RhK3oSzRc,327\n",
            SNOWFLAKE_INSTALLED_HTTPX_RECORD_SHA256,
        ),
    }
    for relative, (old, new, expected_hash) in record_patches.items():
        record = target_site / relative
        source = record.read_text()
        if source.count(old) != 1:
            raise RuntimeError(f"Snowflake installed RECORD entrypoint anchor mismatch: {relative}")
        record.write_text(source.replace(old, new))
        record.chmod(0o644)
        assert_digest(record, expected_hash, f"repaired Snowflake RECORD {relative}")

    editable = target_site / "mcp_snowflake_server-0.4.0.dist-info"
    if component_tree_identity(editable) != SNOWFLAKE_BASE_EDITABLE_TREE:
        raise RuntimeError("base Snowflake editable distribution identity mismatch")
    metadata = editable / "METADATA"
    record = editable / "RECORD"
    assert_digest(metadata, SNOWFLAKE_BASE_EDITABLE_METADATA_SHA256, "base Snowflake METADATA")
    assert_digest(record, SNOWFLAKE_BASE_EDITABLE_RECORD_SHA256, "base Snowflake editable RECORD")
    metadata_source = metadata.read_text()
    old_requirement = "Requires-Dist: mcp>=1.0.0\n"
    new_requirement = "Requires-Dist: mcp==1.26.0\n"
    if metadata_source.count(old_requirement) != 1:
        raise RuntimeError("Snowflake installed METADATA requirement anchor mismatch")
    metadata.write_text(metadata_source.replace(old_requirement, new_requirement))
    metadata.chmod(0o644)
    assert_digest(metadata, SNOWFLAKE_PATCHED_EDITABLE_METADATA_SHA256, "patched Snowflake METADATA")
    record_source = record.read_text()
    old_record = (
        "mcp_snowflake_server-0.4.0.dist-info/METADATA,sha256=NVcWbjlLJddQ_hg8zkA_dh00yHFjjA7ga5zprpdyvxU,9642\n"
    )
    new_record = (
        "mcp_snowflake_server-0.4.0.dist-info/METADATA,sha256=4kaVs5ghPVRPR5_IFBVK0hfDSDXwfPr3rogpfKezz5c,9643\n"
    )
    if record_source.count(old_record) != 1:
        raise RuntimeError("Snowflake editable RECORD METADATA anchor mismatch")
    record.write_text(record_source.replace(old_record, new_record))
    record.chmod(0o644)
    assert_digest(record, SNOWFLAKE_PATCHED_EDITABLE_RECORD_SHA256, "patched editable RECORD")
    if component_tree_identity(editable) != SNOWFLAKE_PATCHED_EDITABLE_TREE:
        raise RuntimeError("patched Snowflake editable distribution identity mismatch")
    verified_records = {
        "httpx-0.28.1.dist-info/RECORD": verify_distribution_record(
            target_site, target_site / "httpx-0.28.1.dist-info/RECORD"
        ),
        "mcp-1.26.0.dist-info/RECORD": verify_distribution_record(
            target_site, target_site / "mcp-1.26.0.dist-info/RECORD"
        ),
        "mcp_snowflake_server-0.4.0.dist-info/RECORD": verify_distribution_record(target_site, editable / "RECORD"),
    }
    return {
        "installed_entrypoints": installed_scripts,
        "removed_entrypoints": {
            "httpx2": {
                "path": f"{SNOWFLAKE_PROJECT}/.venv/bin/httpx2",
                "base_sha256": SNOWFLAKE_STALE_HTTPX2_ENTRYPOINT_SHA256,
                "status": "absent",
            }
        },
        "editable_distribution": {
            "path": f"{SNOWFLAKE_SITE_PACKAGES}/mcp_snowflake_server-0.4.0.dist-info",
            "base_metadata_sha256": SNOWFLAKE_BASE_EDITABLE_METADATA_SHA256,
            "patched_metadata_sha256": SNOWFLAKE_PATCHED_EDITABLE_METADATA_SHA256,
            "base_record_sha256": SNOWFLAKE_BASE_EDITABLE_RECORD_SHA256,
            "patched_record_sha256": SNOWFLAKE_PATCHED_EDITABLE_RECORD_SHA256,
            "base_tree": SNOWFLAKE_BASE_EDITABLE_TREE,
            "patched_tree": SNOWFLAKE_PATCHED_EDITABLE_TREE,
        },
        "verified_records": verified_records,
    }


def install_snowflake_mcp126_runtime(root: Path) -> dict[str, object]:
    """Install a lock-coherent, RECORD-coherent MCP 1.26 runtime in place."""

    donor = rooted(root, SNOWFLAKE_MCP126_DONOR)
    target = rooted(root, SNOWFLAKE_SITE_PACKAGES)
    base_record = target / "mcp-2.0.0.dist-info/RECORD"
    donor_record = donor / "mcp-1.26.0.dist-info/RECORD"
    assert_digest(base_record, SNOWFLAKE_BASE_MCP2_RECORD_SHA256, "base Snowflake MCP 2 RECORD")
    assert_digest(donor_record, SNOWFLAKE_DONOR_MCP126_RECORD_SHA256, "donor MCP 1.26 RECORD")
    for name, expected in SNOWFLAKE_BASE_TREES.items():
        actual = tree_content_identity(target / name)
        if actual != expected:
            raise RuntimeError(f"base Snowflake dependency tree mismatch {name}: {actual}")
    for name, expected in SNOWFLAKE_STALE_MCP2_TREES.items():
        actual = component_tree_identity(target / name)
        if actual != expected:
            raise RuntimeError(f"stale Snowflake MCP 2 dependency mismatch {name}: {actual}")
    stale_identity = hashlib.sha256(
        json.dumps(SNOWFLAKE_STALE_MCP2_TREES, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if stale_identity != SNOWFLAKE_STALE_MCP2_CLOSURE_SHA256:
        raise RuntimeError("stale Snowflake MCP 2 dependency-set identity mismatch")

    expected_trees = SNOWFLAKE_MCP126_TREES
    for name, expected in expected_trees.items():
        source = donor / name
        if component_tree_identity(source) != expected:
            raise RuntimeError(f"Snowflake MCP 1.26 donor tree mismatch: {name}")
        destination = target / name
        if name not in SNOWFLAKE_BASE_TREES and (destination.exists() or destination.is_symlink()):
            raise RuntimeError(f"unexpected preexisting Snowflake dependency tree: {destination}")

    for name in (*SNOWFLAKE_BASE_TREES, *SNOWFLAKE_STALE_MCP2_TREES):
        shutil.rmtree(target / name)
    for name in sorted(expected_trees):
        shutil.copytree(donor / name, target / name, symlinks=True)
        make_world_readable(target / name)
        if component_tree_identity(target / name) != expected_trees[name]:
            raise RuntimeError(f"installed Snowflake MCP 1.26 tree mismatch: {name}")
    donor_closure_identity = hashlib.sha256(
        json.dumps(expected_trees, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if donor_closure_identity != SNOWFLAKE_MCP126_CLOSURE_SHA256:
        raise RuntimeError("Snowflake MCP 1.26 donor closure identity mismatch")
    record_repair = repair_snowflake_distribution_records(root)
    installed_trees = {name: component_tree_identity(target / name) for name in sorted(expected_trees)}
    if installed_trees != SNOWFLAKE_INSTALLED_MCP126_TREES:
        raise RuntimeError("record-repaired Snowflake MCP 1.26 tree set mismatch")
    installed_closure_identity = hashlib.sha256(
        json.dumps(installed_trees, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if installed_closure_identity != SNOWFLAKE_INSTALLED_MCP126_CLOSURE_SHA256:
        raise RuntimeError("record-repaired Snowflake MCP 1.26 closure identity mismatch")
    obsolete_trees = (
        "mcp-2.0.0.dist-info",
        *SNOWFLAKE_STALE_MCP2_TREES,
    )
    for name in obsolete_trees:
        if (target / name).exists() or (target / name).is_symlink():
            raise RuntimeError(f"obsolete Snowflake dependency survived replacement: {name}")
    metadata = install_snowflake_metadata(root)
    record_repair["full_record_audit"] = verify_all_distribution_records(target)
    return {
        "policy": "exact_in_place_record_repaired_ten_tree_replacement",
        "site_packages": SNOWFLAKE_SITE_PACKAGES,
        "base_mcp_version": "2.0.0",
        "base_mcp_record_sha256": SNOWFLAKE_BASE_MCP2_RECORD_SHA256,
        "base_trees": SNOWFLAKE_BASE_TREES,
        "installed_mcp_version": "1.26.0",
        "donor_mcp_record_sha256": SNOWFLAKE_DONOR_MCP126_RECORD_SHA256,
        "donor_path": SNOWFLAKE_MCP126_DONOR,
        "donor_trees": expected_trees,
        "donor_closure_sha256": donor_closure_identity,
        "installed_trees": installed_trees,
        "installed_closure_sha256": installed_closure_identity,
        "removed_mcp2_trees": SNOWFLAKE_STALE_MCP2_TREES,
        "removed_mcp2_closure_sha256": stale_identity,
        "distribution_records": record_repair,
        "project_metadata": metadata,
    }


def patch_python_mcp_launchers(root: Path) -> dict[str, object]:
    """Replace all 12 ``uv run`` MCP launches with their baked venv runtime."""

    configs = rooted(root, "/app/configs/mcp_servers")
    base_hashes: dict[str, str] = {}
    patched_hashes: dict[str, str] = {}
    launchers: dict[str, str] = {}
    configs_by_name: dict[str, str] = {}

    for filename, spec in sorted(PYTHON_MCP_LAUNCH_SPECS.items()):
        config = configs / filename
        base_sha256 = str(spec["base_sha256"])
        assert_digest(config, base_sha256, f"base Python MCP config {filename}")
        project = str(spec["project"])
        uv_target = str(spec["uv_target"])
        launcher = str(spec["launcher"])
        old = (
            "  command: uv\n"
            "  args:\n"
            '    - "--directory"\n'
            f'    - "${{local_servers_paths}}/{project}"\n'
            '    - "run"\n'
            f'    - "{uv_target}"\n'
        )
        image_launcher = f"/opt/local_servers/{project}/.venv/bin/{launcher}"
        new = f'  command: "{PYTHON_MCP_WRAPPER}"\n  args:\n    - "{spec["name"]}"\n'
        source = config.read_text()
        if source.count(old) != 1:
            raise RuntimeError(f"Python MCP launch patch anchor mismatch: {filename}")
        config.write_text(source.replace(old, new))
        config.chmod(0o644)
        expected_patched = str(spec["patched_sha256"])
        if expected_patched != "TO_BE_FILLED":
            assert_digest(config, expected_patched, f"patched Python MCP config {filename}")
        assert_baked_launcher(root, image_launcher)
        base_hashes[filename] = base_sha256
        patched_hashes[filename] = sha256(config)
        server_name = str(spec["name"])
        launchers[server_name] = image_launcher
        configs_by_name[server_name] = filename

    yahoo = configs / "yahoo-finance.yaml"
    assert_digest(yahoo, YAHOO_FINANCE_CONFIG_SHA256, "base Python MCP config yahoo-finance.yaml")
    yahoo_old = '  command: uv\n  args:\n    - "run"\n    - "${local_servers_paths}/yahoo-finance-mcp/server.py"\n'
    yahoo_new = (
        f'  command: "{PYTHON_MCP_WRAPPER}"\n'
        "  args:\n"
        '    - "yahoo-finance"\n'
        '    - "${local_servers_paths}/yahoo-finance-mcp/server.py"\n'
    )
    yahoo_source = yahoo.read_text()
    if yahoo_source.count(yahoo_old) != 1:
        raise RuntimeError("Python MCP launch patch anchor mismatch: yahoo-finance.yaml")
    yahoo.write_text(yahoo_source.replace(yahoo_old, yahoo_new))
    yahoo.chmod(0o644)
    if YAHOO_FINANCE_PATCHED_CONFIG_SHA256 != "TO_BE_FILLED":
        assert_digest(
            yahoo,
            YAHOO_FINANCE_PATCHED_CONFIG_SHA256,
            "patched Python MCP config yahoo-finance.yaml",
        )
    yahoo_launcher = "/opt/local_servers/yahoo-finance-mcp/.venv/bin/python"
    assert_baked_launcher(root, yahoo_launcher)
    base_hashes["yahoo-finance.yaml"] = YAHOO_FINANCE_CONFIG_SHA256
    patched_hashes["yahoo-finance.yaml"] = sha256(yahoo)
    launchers["yahoo-finance"] = yahoo_launcher
    configs_by_name["yahoo-finance"] = "yahoo-finance.yaml"

    youtube = configs / "youtube_transcript.yaml"
    assert_digest(
        youtube,
        YOUTUBE_TRANSCRIPT_CONFIG_SHA256,
        "direct Python MCP config youtube_transcript.yaml",
    )
    youtube_source = youtube.read_text()
    youtube_command = '  command: "${local_servers_paths}/mcp-youtube-transcript/.venv/bin/python3"\n'
    if youtube_source.count(youtube_command) != 1 or "command: uv" in youtube_source:
        raise RuntimeError("YouTube transcript MCP is not pinned to its baked venv Python")
    youtube_launcher = "/opt/local_servers/mcp-youtube-transcript/.venv/bin/python3"
    assert_baked_launcher(root, youtube_launcher)
    base_hashes["youtube_transcript.yaml"] = YOUTUBE_TRANSCRIPT_CONFIG_SHA256
    patched_hashes["youtube_transcript.yaml"] = YOUTUBE_TRANSCRIPT_CONFIG_SHA256
    launchers["youtube-transcript"] = youtube_launcher
    configs_by_name["youtube-transcript"] = "youtube_transcript.yaml"

    if len(PYTHON_MCP_LAUNCH_SPECS) + 1 != 12 or len(launchers) != 13:
        raise RuntimeError("immutable Python MCP launcher universe changed")
    launcher_mapping = {
        name: {
            "config_command": (launchers[name] if name == "youtube-transcript" else PYTHON_MCP_WRAPPER),
            "exec_command": launchers[name],
            "project_cwd": str(Path(launchers[name]).parents[2]),
            "config_sha256": patched_hashes[configs_by_name[name]],
        }
        for name in sorted(launchers)
    }
    mapping_identity = hashlib.sha256(
        json.dumps(launcher_mapping, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "policy": "baked_venv_no_uv",
        "rewritten_uv_config_count": 12,
        "audited_python_server_count": 13,
        "base_config_sha256": dict(sorted(base_hashes.items())),
        "patched_config_sha256": dict(sorted(patched_hashes.items())),
        "base_config_set_sha256": config_set_sha256(base_hashes),
        "patched_config_set_sha256": config_set_sha256(patched_hashes),
        "wrapper_path": PYTHON_MCP_WRAPPER,
        "wrapper_sha256": sha256(rooted(root, PYTHON_MCP_WRAPPER)),
        "probe_path": PYTHON_MCP_PROBE,
        "probe_sha256": sha256(rooted(root, PYTHON_MCP_PROBE)),
        "rootless_probe_policy": (
            "cleanenv_no_home_exact_launch_initialize_live_name_set_persistence_clean_teardown_no_mutation"
        ),
        "launcher_mapping": launcher_mapping,
        "launcher_mapping_sha256": mapping_identity,
    }


def patch_server_fail_closed(root: Path, expected_server_sha256: str) -> None:
    server = rooted(root, "/app/server.py")
    assert_digest(server, expected_server_sha256, "base Toolathlon server.py")
    source = server.read_text()

    replacements = (
        (
            "import os\nimport random\n",
            "import os\nimport random\nimport signal\n",
        ),
        (
            """    def __init__(self, name: str, proc: asyncio.subprocess.Process):
        self.name = name
        self.proc = proc
        self._req_id = 10

    async def send_recv""",
            """    def __init__(self, name: str, proc: asyncio.subprocess.Process):
        self.name = name
        self.proc = proc
        self._req_id = 10
        self._stderr_tail = bytearray()
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        self._close_lock = asyncio.Lock()
        self._closed = False

    async def _drain_stderr(self) -> None:
        if self.proc.stderr is None:
            return
        while True:
            chunk = await self.proc.stderr.read(65536)
            if not chunk:
                return
            self._stderr_tail.extend(chunk)
            if len(self._stderr_tail) > 65536:
                del self._stderr_tail[:-65536]

    def stderr_tail(self) -> str:
        return bytes(self._stderr_tail).decode(errors="replace")

    async def _discard_stdout(self) -> None:
        if self.proc.stdout is None:
            return
        while await self.proc.stdout.read(65536):
            pass

    async def _finish_stderr(self) -> None:
        try:
            await asyncio.wait_for(asyncio.shield(self._stderr_task), timeout=2)
        except asyncio.TimeoutError:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            pass

    async def _group_absent(self, timeout: float) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            try:
                os.killpg(self.proc.pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError as exc:
                raise RuntimeError(
                    f"Cannot verify MCP process group {self.proc.pid}"
                ) from exc
            if asyncio.get_running_loop().time() >= deadline:
                return False
            await asyncio.sleep(0.05)

    def _validate_group_members(self) -> bool:
        members = []
        for process_root in Path("/proc").iterdir():
            if not process_root.name.isdigit():
                continue
            pid = int(process_root.name)
            try:
                if os.getpgid(pid) != self.proc.pid:
                    continue
                if os.getsid(pid) != self.proc.pid or process_root.stat().st_uid != os.geteuid():
                    raise RuntimeError(
                        f"Untrusted process in MCP group {self.proc.pid}: pid={pid}"
                    )
                members.append(pid)
            except (FileNotFoundError, ProcessLookupError):
                continue
        return bool(members)

    async def send_recv""",
        ),
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
            """    async def close(self):
        try:
            self.proc.terminate()
            await asyncio.wait_for(self.proc.wait(), timeout=5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
""",
            """    async def close(self):
        async with self._close_lock:
            if self._closed:
                return
            used_sigterm = False
            used_sigkill = False
            failures: list[str] = []
            stdout_task = None
            if self.proc.stdout is not None:
                stdout_task = asyncio.create_task(self._discard_stdout())

            async def checked_group_absent(timeout: float, phase: str):
                try:
                    return await self._group_absent(timeout)
                except BaseException as exc:
                    failures.append(f"MCP {phase} group verification failed: {exc}")
                    return None

            alive_at_close = self.proc.returncode is None
            if not alive_at_close:
                failures.append(
                    f"MCP exited before graceful close: {self.name} "
                    f"rc={self.proc.returncode}"
                )
            try:
                if alive_at_close and self.proc.stdin is not None:
                    if not self.proc.stdin.is_closing():
                        self.proc.stdin.close()
                        await self.proc.stdin.wait_closed()
                    await asyncio.wait_for(self.proc.wait(), timeout=20)
            except asyncio.TimeoutError:
                failures.append(
                    f"MCP did not exit within 20s of stdin EOF: {self.name}"
                )
            except BaseException as exc:
                failures.append(f"graceful EOF failed: {exc}")

            group_state = await checked_group_absent(1, "pre-TERM")
            if group_state is False:
                try:
                    trusted_group = self._validate_group_members()
                except BaseException as exc:
                    failures.append(f"MCP TERM group validation failed: {exc}")
                    trusted_group = False
                if trusted_group:
                    try:
                        os.killpg(self.proc.pid, signal.SIGTERM)
                        used_sigterm = True
                    except ProcessLookupError:
                        pass
                    if self.proc.returncode is None:
                        try:
                            await asyncio.wait_for(self.proc.wait(), timeout=5)
                        except asyncio.TimeoutError:
                            pass

            group_state = await checked_group_absent(1, "pre-KILL")
            if group_state is False:
                try:
                    trusted_group = self._validate_group_members()
                except BaseException as exc:
                    failures.append(f"MCP KILL group validation failed: {exc}")
                    trusted_group = False
                if trusted_group:
                    try:
                        os.killpg(self.proc.pid, signal.SIGKILL)
                        used_sigkill = True
                    except ProcessLookupError:
                        pass
                    if self.proc.returncode is None:
                        try:
                            await asyncio.wait_for(self.proc.wait(), timeout=5)
                        except asyncio.TimeoutError as exc:
                            failures.append(
                                f"MCP leader survived process-group SIGKILL: {exc}"
                            )

            group_state = await checked_group_absent(2, "final")
            group_absent = group_state is True
            if self.proc.returncode is None and group_absent:
                try:
                    await asyncio.wait_for(self.proc.wait(), timeout=2)
                except asyncio.TimeoutError as exc:
                    failures.append(f"MCP leader was not reaped: {self.name}: {exc}")
            if stdout_task is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(stdout_task), timeout=2)
                except asyncio.TimeoutError:
                    stdout_task.cancel()
                    try:
                        await stdout_task
                    except asyncio.CancelledError:
                        pass
                except BaseException as exc:
                    failures.append(f"stdout drain failed: {exc}")
            try:
                await self._finish_stderr()
            except BaseException as exc:
                failures.append(f"stderr drain failed: {exc}")
            if self.proc.returncode not in (None, 0):
                tail = self.stderr_tail().strip()[-300:]
                detail = f": {tail}" if tail else ""
                failures.append(
                    f"MCP leader exited nonzero: {self.name} "
                    f"rc={self.proc.returncode}{detail}"
                )
            if not group_absent:
                failures.append(f"MCP process group remains after close: {self.name}")
            self._closed = group_absent
            if used_sigterm:
                failures.append(f"MCP process group required SIGTERM: {self.name}")
            if used_sigkill:
                failures.append(f"MCP process group required forbidden SIGKILL: {self.name}")
            if failures:
                raise RuntimeError(" | ".join(failures))
""",
        ),
        (
            '''    async def start(self):
        """Launch and initialize all needed MCP servers."""
        for yaml_path in sorted(CONFIGS_DIR.glob("*.yaml")):
''',
            '''    async def _close_record(
        self,
        name: str,
        process: _MCPProcess,
        failures: list[str],
    ) -> None:
        cleanup_task = asyncio.create_task(process.close())
        cancellation_count = 0
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                cancellation_count += 1
            except BaseException:
                break
        try:
            cleanup_task.result()
        except BaseException as exc:
            failures.append(f"{name}: cleanup failed: {exc}")
        if cancellation_count:
            failures.append(
                f"{name}: cleanup caller was cancelled "
                f"{cancellation_count} time(s)"
            )

    async def start(self):
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
            """            full_env = {**os.environ, **env_vars, **self.pg_env}
            os.makedirs(cwd, exist_ok=True)
""",
            """            full_env = {**os.environ, **env_vars, **self.pg_env}
            full_env.pop("VIRTUAL_ENV", None)
            full_env.pop("UV_PROJECT_ENVIRONMENT", None)
            full_env["PYTHONDONTWRITEBYTECODE"] = "1"
            full_env["PYTHONNOUSERSITE"] = "1"
            full_env["PYTHONPATH"] = "/app/runtime_patches"
            os.makedirs(cwd, exist_ok=True)
""",
        ),
        (
            """                    cwd=cwd,
                    limit=MCP_SUBPROCESS_STREAM_LIMIT,
                )
""",
            """                    cwd=cwd,
                    limit=MCP_SUBPROCESS_STREAM_LIMIT,
                    start_new_session=True,
                )
""",
        ),
        (
            """                    detail = ""
                    if proc.returncode is not None and proc.stderr is not None:
                        try:
                            err = await asyncio.wait_for(proc.stderr.read(2000), timeout=1)
                            detail = f" (rc={proc.returncode}): {err.decode(errors='replace').strip()[:300]}"
                        except Exception:
                            detail = f" (rc={proc.returncode})"
""",
            """                    detail = ""
                    if proc.returncode is not None:
                        await asyncio.sleep(0)
                        tail = mcp_proc.stderr_tail().strip()[-300:]
                        detail = f" (rc={proc.returncode}): {tail}" if tail else f" (rc={proc.returncode})"
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
                    await self._close_record(name, mcp_proc, failures)
            except asyncio.CancelledError:
                if mcp_proc is not None:
                    await self._close_record(name, mcp_proc, failures)
                for process_name, process in list(self._processes.items()):
                    await self._close_record(process_name, process, failures)
                self._processes.clear()
                raise
            except Exception as e:
                failure = f"{name}: launch failed: {e}"
                print(f"[MCPBridge] ERROR: {failure}", flush=True)
                failures.append(failure)
                if mcp_proc is not None:
                    await self._close_record(name, mcp_proc, failures)

        missing_servers = set(self.needed_servers) - set(self._processes)
        unexpected_servers = set(self._processes) - set(self.needed_servers)
        if missing_servers or unexpected_servers:
            failures.append(
                "MCP process set mismatch: "
                f"missing={sorted(missing_servers)}, "
                f"unexpected={sorted(unexpected_servers)}"
            )
        if failures:
            for process_name, process in list(self._processes.items()):
                await self._close_record(process_name, process, failures)
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
                        await self._close_record(name, mcp_proc, failures)
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
                        await self._close_record(name, mcp_proc, failures)
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
            """    async def close(self):
        for proc in self._processes.values():
            await proc.close()
        self._processes.clear()
""",
            """    async def close(self):
        failures: list[str] = []
        for name, process in list(self._processes.items()):
            await self._close_record(name, process, failures)
        self._processes.clear()
        if failures:
            raise RuntimeError("MCP cleanup failed: " + " | ".join(failures))
""",
        ),
        (
            """    async def teardown(self):
        if self._mcp_bridge:
            await self._mcp_bridge.close()
        shutil.rmtree(self.workspace_dir, ignore_errors=True)
        # Drop the per-session DB. Best-effort: a leak just means the next
        # entrypoint.sh sweep will collect it.
        try:
            await self._psql(
                f'DROP DATABASE IF EXISTS "{self._db_name}" WITH (FORCE);'
            )
        except Exception as e:
            print(f"[teardown] DROP DATABASE {self._db_name} failed: {e}", flush=True)
""",
            """    async def teardown(self):
        failures: list[str] = []
        if self._mcp_bridge:
            try:
                await self._mcp_bridge.close()
            except BaseException as exc:
                failures.append(f"MCP bridge cleanup failed: {exc}")
            finally:
                self._mcp_bridge = None
        try:
            await self._psql(
                f'DROP DATABASE IF EXISTS "{self._db_name}" WITH (FORCE);'
            )
        except BaseException as exc:
            failures.append(f"DROP DATABASE {self._db_name} failed: {exc}")
        try:
            shutil.rmtree(self.workspace_dir)
        except FileNotFoundError:
            pass
        except BaseException as exc:
            failures.append(f"workspace cleanup failed: {exc}")
        if os.path.exists(self.workspace_dir):
            failures.append(f"workspace remains after cleanup: {self.workspace_dir}")
        if failures:
            raise RuntimeError("Task teardown failed: " + " | ".join(failures))
""",
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
    python_mcp_launchers: dict[str, object],
    snowflake_mcp_runtime: dict[str, object],
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
                "base_excel_server_sha256": BASE_EXCEL_SERVER_SHA256,
                "patched_excel_server_sha256": sha256(
                    rooted(root, "/opt/local_servers/excel-mcp-server/src/excel_mcp/server.py")
                ),
                "base_excel_main_sha256": BASE_EXCEL_MAIN_SHA256,
                "patched_excel_main_sha256": sha256(
                    rooted(root, "/opt/local_servers/excel-mcp-server/src/excel_mcp/__main__.py")
                ),
                "base_word_main_sha256": BASE_WORD_MAIN_SHA256,
                "patched_word_main_sha256": sha256(
                    rooted(
                        root,
                        "/opt/local_servers/Office-Word-MCP-Server/word_document_server/main.py",
                    )
                ),
                "base_yahoo_mcp_session_sha256": BASE_YAHOO_MCP_SESSION_SHA256,
                "patched_yahoo_mcp_session_sha256": sha256(
                    rooted(
                        root,
                        (
                            "/opt/local_servers/yahoo-finance-mcp/.venv/lib/python3.11/"
                            "site-packages/mcp/server/session.py"
                        ),
                    )
                ),
                "mcp_venvs": venv_stats,
                "python_mcp_launchers": python_mcp_launchers,
                "snowflake_mcp_runtime": snowflake_mcp_runtime,
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
    snowflake_mcp_runtime = install_snowflake_mcp126_runtime(root)
    patch_excel_logging(root)
    patch_word_teardown(root)
    patch_yahoo_teardown(root)
    venv_stats = rewrite_mcp_venvs(root)
    python_mcp_launchers = patch_python_mcp_launchers(root)
    patch_server_fail_closed(root, args.expected_server_sha256)
    write_manifest(
        root,
        args.identity,
        venv_stats,
        playwright_stats,
        python_mcp_launchers,
        snowflake_mcp_runtime,
    )

    print(
        json.dumps(
            {
                "toolathlon_pbs_portable_patch": "ok",
                "identity": args.identity,
                "playwright": playwright_stats,
                "python_mcp_launchers": python_mcp_launchers,
                "snowflake_mcp_runtime": snowflake_mcp_runtime,
                "venvs": venv_stats,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
