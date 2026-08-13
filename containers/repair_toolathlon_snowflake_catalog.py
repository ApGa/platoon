#!/usr/bin/env python3
"""Reconstruct the Snowflake entry omitted from the pinned image catalog.

The 18e62c0d OCI/SIF contains all Snowflake server code and dependencies, but
its build-time discovery skipped that one server.  Extract the literal tool
definitions from the exact pinned server source, then rely on the portable
server patch's live ``tools/list`` parity check to prove those schemas name the
tools the subprocess actually serves.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import yaml

BASE_CATALOG_SHA256 = "48c56aabcf85833c53a967b1927bdf8a3e96c600983e9aa4f830ed99ab48f057"
SNOWFLAKE_SOURCE_SHA256 = "d23550dd90a469499fe392185613fc86e7672bed7fd468bce1ee4cc93acc076b"
REPAIRED_CATALOG_SHA256 = "a6bdfea0a5524ec8d198a0b2eb9b107aba89e308b50659a106abaff0002fd64d"

BASE_SERVERS = frozenset(
    {
        "arxiv-latex",
        "arxiv_local",
        "canvas",
        "emails",
        "excel",
        "fetch",
        "filesystem",
        "google_calendar",
        "google_forms",
        "google_sheet",
        "howtocook",
        "memory",
        "notion",
        "pdf-tools",
        "playwright_with_chunk",
        "pptx",
        "rail_12306",
        "scholarly",
        "terminal",
        "woocommerce",
        "word",
        "yahoo-finance",
        "youtube",
        "youtube-transcript",
    }
)
EXPECTED_SERVERS = BASE_SERVERS | {"snowflake"}
EXPECTED_SNOWFLAKE_TOOLS = frozenset(
    {
        "list_databases",
        "list_schemas",
        "list_tables",
        "describe_table",
        "read_query",
        "append_insight",
        "write_query",
        "create_table",
        "create_databases",
        "drop_databases",
        "create_schemas",
        "drop_schemas",
        "create_tables",
        "drop_tables",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_digest(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} digest mismatch: expected {expected}, got {actual}: {path}")


def rooted(root: Path, absolute_path: str) -> Path:
    return root / absolute_path.lstrip("/")


def configured_servers(configs: Path) -> set[str]:
    names: set[str] = set()
    for config_path in sorted(configs.glob("*.yaml")):
        config = yaml.safe_load(config_path.read_text())
        if not config:
            continue
        name = config.get("name", config_path.stem)
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"invalid MCP name in {config_path}: {name!r}")
        if name in names:
            raise RuntimeError(f"duplicate configured MCP server name: {name}")
        names.add(name)
    return names


def extract_snowflake_tools(snowflake_source: Path) -> list[dict]:
    assert_digest(snowflake_source, SNOWFLAKE_SOURCE_SHA256, "Snowflake MCP server source")
    tree = ast.parse(snowflake_source.read_text(), filename=str(snowflake_source))
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "all_tools" for target in node.targets)
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.List):
        raise RuntimeError(f"expected one literal Snowflake all_tools list, found {len(assignments)}")

    schemas: list[dict] = []
    for item in assignments[0].value.elts:
        if not isinstance(item, ast.Call):
            raise RuntimeError("Snowflake all_tools contains a non-call expression")
        keywords = {keyword.arg: keyword.value for keyword in item.keywords}
        try:
            name = ast.literal_eval(keywords["name"])
            description = ast.literal_eval(keywords["description"])
            input_schema = ast.literal_eval(keywords["input_schema"])
        except (KeyError, ValueError, TypeError) as exc:
            raise RuntimeError("Snowflake Tool name/description/input_schema stopped being literal") from exc
        if not isinstance(name, str) or not isinstance(description, str):
            raise RuntimeError(f"invalid Snowflake tool metadata: {name!r}")
        if not isinstance(input_schema, dict):
            raise RuntimeError(f"invalid Snowflake input schema for {name}")
        schemas.append(
            {
                "name": name,
                "description": description,
                "inputSchema": input_schema,
            }
        )

    names = [schema["name"] for schema in schemas]
    if len(names) != len(set(names)) or set(names) != EXPECTED_SNOWFLAKE_TOOLS:
        raise RuntimeError(
            "Snowflake source tool set mismatch: "
            f"missing={sorted(EXPECTED_SNOWFLAKE_TOOLS - set(names))}, "
            f"unexpected={sorted(set(names) - EXPECTED_SNOWFLAKE_TOOLS)}, "
            f"count={len(names)}"
        )
    return schemas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/"))
    args = parser.parse_args()
    root = args.root.resolve()
    catalog_path = rooted(root, "/app/tool_schemas.json")
    configs = rooted(root, "/app/configs/mcp_servers")
    snowflake_source = rooted(
        root,
        "/opt/local_servers/mcp-snowflake-server/src/mcp_snowflake_server/server.py",
    )
    manifest_path = rooted(root, "/usr/local/share/openreward/toolathlon-pbs-portable.json")

    assert_digest(catalog_path, BASE_CATALOG_SHA256, "base baked MCP catalog")
    catalog = json.loads(catalog_path.read_text())
    if set(catalog) != BASE_SERVERS:
        raise RuntimeError(
            "base baked MCP catalog keys changed: "
            f"missing={sorted(BASE_SERVERS - set(catalog))}, "
            f"unexpected={sorted(set(catalog) - BASE_SERVERS)}"
        )
    empty = sorted(name for name, tools in catalog.items() if not tools)
    if empty:
        raise RuntimeError(f"base baked MCP catalog has empty servers: {empty}")

    yaml_servers = configured_servers(configs)
    if yaml_servers != EXPECTED_SERVERS:
        raise RuntimeError(
            "configured MCP server set mismatch: "
            f"missing={sorted(EXPECTED_SERVERS - yaml_servers)}, "
            f"unexpected={sorted(yaml_servers - EXPECTED_SERVERS)}"
        )

    catalog["snowflake"] = extract_snowflake_tools(snowflake_source)
    if set(catalog) != EXPECTED_SERVERS or any(not tools for tools in catalog.values()):
        raise RuntimeError("repaired catalog does not have 25 nonempty MCP entries")
    catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    if REPAIRED_CATALOG_SHA256 != "TO_BE_FILLED":
        assert_digest(catalog_path, REPAIRED_CATALOG_SHA256, "repaired MCP catalog")

    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "base_tool_schemas_sha256": BASE_CATALOG_SHA256,
            "repaired_tool_schemas_sha256": sha256(catalog_path),
            "mcp_catalog_server_count": len(catalog),
            "snowflake_tool_count": len(catalog["snowflake"]),
            "mcp_catalog_policy": "live_name_set_parity",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_path.chmod(0o444)
    print(
        json.dumps(
            {
                "toolathlon_snowflake_catalog_repair": "ok",
                "server_count": len(catalog),
                "snowflake_tool_count": len(catalog["snowflake"]),
                "catalog_sha256": sha256(catalog_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
