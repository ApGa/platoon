#!/usr/bin/env python3
"""Exercise seven Toolathlon sessions whose MCP union is exactly all 25.

Session creation is meaningful only with the PBS-portable image patch: the
patched server initializes every requested MCP, calls its live ``tools/list``,
and exact-compares that name set with the baked catalog before returning the
session.  This driver additionally validates catalog exposure in the prompt,
discovers one representative tool per server through the public meta-tool, and
performs safe read-only calls plus a simulated PostgreSQL email send/readback.
No network email is sent: Toolathlon's ``PgSMTPBackend`` inserts into the
fresh per-session database, which is dropped when the session closes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

ENVIRONMENT = "toolathlongym"
EXPECTED_CATALOG_SHA256 = "a6bdfea0a5524ec8d198a0b2eb9b107aba89e308b50659a106abaff0002fd64d"
PLAYWRIGHT_BROWSERS_PATH = Path("/opt/ms-playwright")
PLAYWRIGHT_CONFIG_PATH = Path("/app/configs/mcp_servers/playwright_with_chunk.yaml")
PLAYWRIGHT_CHROMIUM_EXECUTABLE = PLAYWRIGHT_BROWSERS_PATH / "chromium-1182/chrome-linux/chrome"
PLAYWRIGHT_HEADLESS_SHELL_EXECUTABLE = (
    PLAYWRIGHT_BROWSERS_PATH / "chromium_headless_shell-1182/chrome-linux/headless_shell"
)
EXPECTED_PLAYWRIGHT_CHROMIUM_SHA256 = "b93030bc068a3850d4d668e69816fe4a5f04ec93db9b2f63f32f58a39f207db8"
EXPECTED_PLAYWRIGHT_HEADLESS_SHELL_SHA256 = "71ebe8760eb72a99b265470d4030fef296f119e25b9aab5d3efbb6ac821483a5"
EXPECTED_PLAYWRIGHT_CONFIG_SHA256 = "47841318775a9afc122097e769a3fcc02d3af7089d372b60fc11da1ea44f0de9"
EXPECTED_SERVERS = (
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
    "snowflake",
    "terminal",
    "woocommerce",
    "word",
    "yahoo-finance",
    "youtube",
    "youtube-transcript",
)

# Greedy exact set cover over the 503 pinned task configs.  Keep both the task
# identity and its expected MCP set literal so task-data drift fails the gate.
SESSION_COVER = {
    "financial-audit-cross-system-reconciliation": (
        "emails",
        "excel",
        "google_calendar",
        "pdf-tools",
        "snowflake",
        "terminal",
        "woocommerce",
        "word",
    ),
    "arxiv-lit-review-gsheet": (
        "arxiv-latex",
        "arxiv_local",
        "filesystem",
        "google_sheet",
        "memory",
        "scholarly",
    ),
    "terminal-canvas-howtocook-gsheet-notion-email": (
        "canvas",
        "emails",
        "filesystem",
        "google_sheet",
        "howtocook",
        "notion",
        "terminal",
    ),
    "pw-yf-risk-assessment-excel-ppt": (
        "excel",
        "filesystem",
        "playwright_with_chunk",
        "pptx",
        "terminal",
        "yahoo-finance",
    ),
    "fetch-sf-support-survey-gform-gcal-excel": (
        "excel",
        "fetch",
        "google_calendar",
        "google_forms",
        "pdf-tools",
        "snowflake",
        "terminal",
    ),
    "yt-train-conference-planning-notion-gcal-excel": (
        "emails",
        "excel",
        "filesystem",
        "google_calendar",
        "notion",
        "rail_12306",
        "youtube",
    ),
    "yt-transcript-word-gform": (
        "emails",
        "filesystem",
        "google_forms",
        "word",
        "youtube-transcript",
    ),
}

CONCURRENT_TASKS = (
    "terminal-canvas-howtocook-gsheet-notion-email",
    "yt-transcript-word-gform",
)
CONCURRENT_OVERLAP = ("emails", "filesystem")

# Safe live invocations.  Servers not listed here are still proven by patched
# session startup's initialize + live tools/list parity and by get_tool_details.
READ_ONLY_PROBES = {
    "arxiv_local": ("arxiv_local.list_papers", {}),
    "canvas": ("canvas.canvas_health_check", {}),
    "filesystem": ("filesystem.list_allowed_directories", {}),
    "google_sheet": ("google_sheet.list_spreadsheets", {}),
    "howtocook": ("howtocook.mcp_howtocook_getAllRecipes", {}),
    "memory": ("memory.read_graph", {}),
    "notion": ("notion.API-get-self", {}),
    "playwright_with_chunk": ("playwright_with_chunk.browser_tab_list", {}),
    "pptx": ("pptx.get_server_info", {}),
    "rail_12306": ("rail_12306.get-current-date", {}),
    "snowflake": ("snowflake.list_databases", {}),
    "terminal": ("terminal.show_security_rules", {}),
    "woocommerce": ("woocommerce.woo_system_status", {}),
    "word": ("word.list_available_documents", {"directory": "."}),
}

TOOL_ERROR_META_KEY = "openhands.dev/tool-error"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_of(output: object) -> str:
    return "\n".join(str(getattr(block, "text", "")) for block in getattr(output, "blocks", []))


def assert_tool_success(output: object, name: str) -> str:
    metadata = getattr(output, "metadata", None)
    marker = metadata.get(TOOL_ERROR_META_KEY) if isinstance(metadata, dict) else None
    if marker:
        raise RuntimeError(f"{name} returned typed tool error: {marker!r}")
    text = text_of(output)
    if not text.strip():
        raise RuntimeError(f"{name} returned no text")
    return text


def call_catalog_tool(session: object, name: str, arguments: dict[str, Any]) -> str:
    output = session.call_tool("call_tool", {"name": name, "arguments": arguments})
    return assert_tool_success(output, name)


def load_and_validate_catalog(path: Path) -> dict[str, list[dict[str, Any]]]:
    if sha256(path) != EXPECTED_CATALOG_SHA256:
        raise RuntimeError(f"Toolathlon catalog digest mismatch: {sha256(path)} != {EXPECTED_CATALOG_SHA256}")
    catalog = json.loads(path.read_text())
    if tuple(sorted(catalog)) != EXPECTED_SERVERS:
        raise RuntimeError(
            "catalog server set mismatch: "
            f"missing={sorted(set(EXPECTED_SERVERS) - set(catalog))}, "
            f"unexpected={sorted(set(catalog) - set(EXPECTED_SERVERS))}"
        )
    if any(not tools for tools in catalog.values()):
        raise RuntimeError("catalog contains an empty MCP server tool list")
    for server, tools in catalog.items():
        names = [tool.get("name") for tool in tools]
        if None in names or len(names) != len(set(names)):
            raise RuntimeError(f"catalog has invalid/duplicate tools for {server}")
    return catalog


def validate_playwright_runtime() -> dict[str, Any]:
    """Bind the relocated browser/config before the MCP launch probe."""

    if os.environ.get("PLAYWRIGHT_BROWSERS_PATH") != str(PLAYWRIGHT_BROWSERS_PATH):
        raise RuntimeError("PLAYWRIGHT_BROWSERS_PATH does not select the portable payload")
    for path, expected_digest, description in (
        (
            PLAYWRIGHT_CHROMIUM_EXECUTABLE,
            EXPECTED_PLAYWRIGHT_CHROMIUM_SHA256,
            "Chromium executable",
        ),
        (
            PLAYWRIGHT_CONFIG_PATH,
            EXPECTED_PLAYWRIGHT_CONFIG_SHA256,
            "Playwright MCP config",
        ),
        (
            PLAYWRIGHT_HEADLESS_SHELL_EXECUTABLE,
            EXPECTED_PLAYWRIGHT_HEADLESS_SHELL_SHA256,
            "Playwright headless shell",
        ),
    ):
        value = path.lstat()
        mode = stat.S_IMODE(value.st_mode)
        if not stat.S_ISREG(value.st_mode) or path.is_symlink() or mode & 0o022:
            raise RuntimeError(f"{description} is indirect, writable, or not regular: {path}")
        if sha256(path) != expected_digest:
            raise RuntimeError(f"{description} digest mismatch: {path}")
    for executable in (PLAYWRIGHT_CHROMIUM_EXECUTABLE, PLAYWRIGHT_HEADLESS_SHELL_EXECUTABLE):
        if not os.access(executable, os.R_OK | os.X_OK):
            raise RuntimeError(f"portable Playwright executable is inaccessible: {executable}")
    config_text = PLAYWRIGHT_CONFIG_PATH.read_text()
    if config_text.count('    - "--no-sandbox"') != 1:
        raise RuntimeError("Playwright MCP config lacks the pinned no-sandbox argument")
    return {
        "status": "passed",
        "browsers_path": str(PLAYWRIGHT_BROWSERS_PATH),
        "chromium_executable": str(PLAYWRIGHT_CHROMIUM_EXECUTABLE),
        "chromium_sha256": EXPECTED_PLAYWRIGHT_CHROMIUM_SHA256,
        "headless_shell_executable": str(PLAYWRIGHT_HEADLESS_SHELL_EXECUTABLE),
        "headless_shell_sha256": EXPECTED_PLAYWRIGHT_HEADLESS_SHELL_SHA256,
        "patched_config_sha256": EXPECTED_PLAYWRIGHT_CONFIG_SHA256,
        "chromium_sandbox": "disabled_for_unprivileged_singularity",
        "browser_tab_list": "pending_live_probe",
    }


def validate_cover(tasks_root: Path) -> None:
    union: set[str] = set()
    for task_name, expected in SESSION_COVER.items():
        matches = list(tasks_root.glob(f"*/{task_name}/task_config.json"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one task config for {task_name}, found {len(matches)}")
        configured = tuple(sorted(json.loads(matches[0].read_text()).get("needed_mcp_servers", [])))
        if configured != expected:
            raise RuntimeError(f"task MCP set drift for {task_name}: {configured!r} != {expected!r}")
        union.update(configured)
    if tuple(sorted(union)) != EXPECTED_SERVERS:
        raise RuntimeError(f"seven-session cover is not exact: {tuple(sorted(union))!r}")


def selected_task(environment: object, wanted: str) -> object:
    selected: list[object] = []
    for task in environment.list_tasks(split="train"):
        task_name = getattr(task, "task_spec", {}).get("task_name")
        if task_name == wanted:
            selected.append(task)
    if len(selected) != 1:
        raise RuntimeError(f"task API returned {len(selected)} entries for health task {wanted}")
    return selected[0]


def validate_session_catalog(
    session: object,
    expected_servers: tuple[str, ...],
    catalog: dict[str, list[dict[str, Any]]],
) -> tuple[int, list[str]]:
    direct_names = {str(getattr(tool, "name", "")) for tool in session.list_tools()}
    required_direct = {
        "get_tool_details",
        "call_tool",
        "mcp_health",
        "python_execute",
        "claim_done",
    }
    if not required_direct.issubset(direct_names):
        raise RuntimeError(f"session missing direct tools: {sorted(required_direct - direct_names)}")

    prompt = "\n".join(str(getattr(block, "text", "")) for block in session.get_prompt())
    expected_qualified: list[str] = []
    representatives: list[str] = []
    for server in expected_servers:
        tools = catalog[server]
        for tool in tools:
            qualified = f"{server}.{tool['name']}"
            expected_qualified.append(qualified)
            if f"`{qualified}`" not in prompt:
                raise RuntimeError(f"prompt omitted catalog tool {qualified}")

        representative = f"{server}.{tools[0]['name']}"
        output = session.call_tool("get_tool_details", {"name": representative})
        payload = json.loads(assert_tool_success(output, "get_tool_details"))
        if (
            payload.get("name") != representative
            or payload.get("server") != server
            or payload.get("input_schema") != tools[0].get("inputSchema", {})
        ):
            raise RuntimeError(f"get_tool_details mismatch for representative {representative}")
        representatives.append(representative)
    return len(expected_qualified), representatives


def validate_live_mcp_health(
    session: object,
    expected_servers: tuple[str, ...],
    catalog: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Re-probe every live MCP and compact its exact name-set evidence."""

    output = session.call_tool("mcp_health", {})
    payload = json.loads(assert_tool_success(output, "mcp_health"))
    if set(payload) != {"status", "required_servers", "server_count", "servers"}:
        raise RuntimeError("mcp_health returned an unexpected field set")
    expected_names = list(expected_servers)
    servers = payload.get("servers")
    if (
        payload.get("status") != "passed"
        or payload.get("required_servers") != expected_names
        or payload.get("server_count") != len(expected_names)
        or not isinstance(servers, dict)
        or set(servers) != set(expected_names)
    ):
        raise RuntimeError(f"mcp_health server identity mismatch: {payload!r}")

    compact: dict[str, dict[str, Any]] = {}
    for server in expected_names:
        evidence = servers[server]
        live_names = evidence.get("live_tool_names") if isinstance(evidence, dict) else None
        catalog_names = sorted(tool["name"] for tool in catalog[server])
        if (
            not isinstance(evidence, dict)
            or set(evidence) != {"process_running", "live_tool_names"}
            or evidence.get("process_running") is not True
            or live_names != catalog_names
        ):
            raise RuntimeError(f"mcp_health live parity mismatch for {server}")
        canonical = json.dumps(live_names, separators=(",", ":"), ensure_ascii=True)
        compact[server] = {
            "live_tool_count": len(live_names),
            "live_tool_names_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        }
    return {
        "status": "passed",
        "server_count": len(compact),
        "servers": compact,
    }


def run_one_session(
    openreward_class: object,
    task_name: str,
    expected_servers: tuple[str, ...],
    catalog: dict[str, list[dict[str, Any]]],
    probe_servers: tuple[str, ...],
    concurrency_barrier: Barrier | None = None,
) -> dict[str, Any]:
    tested_tools: list[str] = []
    mcp_health_checks: list[dict[str, Any]] = []
    concurrent = concurrency_barrier is not None
    with openreward_class(api_key="local") as client:
        environment = client.environments.get(name=ENVIRONMENT)
        task = selected_task(environment, task_name)
        with environment.session(task=task) as session:
            catalog_count, representatives = validate_session_catalog(session, expected_servers, catalog)

            # Both concurrent sessions have completed setup (including every
            # MCP initialize + tools/list parity check) and remain live here.
            if concurrency_barrier is not None:
                concurrency_barrier.wait(timeout=600)
                mcp_health_checks.append(validate_live_mcp_health(session, expected_servers, catalog))
                tested_tools.append("mcp_health")

            for server in probe_servers:
                name, arguments = READ_ONLY_PROBES[server]
                call_catalog_tool(session, name, dict(arguments))
                tested_tools.append(name)

            # Exercise the duplicated email MCP while both cover sessions are
            # simultaneously active. This is read-only and also proves each
            # session received its own usable PostgreSQL connection context.
            if concurrent:
                connection = call_catalog_tool(session, "emails.check_connection", {})
                if (
                    "IMAP: ✓ Connected" not in connection
                    or "SMTP: ✓ Connected" not in connection
                    or "All connections are working properly" not in connection
                ):
                    raise RuntimeError(f"concurrent email connection health failed: {connection!r}")
                tested_tools.append("emails.check_connection")

            # Exact regression probe for the reported failure.  This is a
            # PG-only simulated send in the disposable session DB.
            if task_name == "financial-audit-cross-system-reconciliation":
                connection = call_catalog_tool(session, "emails.check_connection", {})
                if (
                    "IMAP: ✓ Connected" not in connection
                    or "SMTP: ✓ Connected" not in connection
                    or "All connections are working properly" not in connection
                ):
                    raise RuntimeError(f"email connection health failed: {connection!r}")
                subject = "toolathlon-all-mcp-health-simulated-send"
                sent = call_catalog_tool(
                    session,
                    "emails.send_email",
                    {
                        "to": "health-probe@example.invalid",
                        "subject": subject,
                        "body": "Disposable PostgreSQL-backed health probe.",
                    },
                )
                if "Email sent successfully" not in sent:
                    raise RuntimeError(f"simulated email send failed: {sent!r}")
                readback = call_catalog_tool(
                    session,
                    "emails.get_emails",
                    {"folder": "Sent", "page": 1, "page_size": 20},
                )
                if subject not in readback:
                    raise RuntimeError(f"simulated email readback omitted subject: {readback!r}")
                tested_tools.extend(
                    (
                        "emails.check_connection",
                        "emails.send_email",
                        "emails.get_emails",
                    )
                )

            # Re-probe immediately before leaving the session context.  This
            # catches MCPs that initialized successfully but died during the
            # representative/read-only/email work above.
            mcp_health_checks.append(validate_live_mcp_health(session, expected_servers, catalog))
            tested_tools.append("mcp_health")

            return {
                "session": {
                    "task_name": task_name,
                    "sid": session.sid,
                    "expected_servers": list(expected_servers),
                    "catalog_tool_count": catalog_count,
                    "live_probes": sorted(set(tested_tools)),
                    "concurrency_barrier": concurrent,
                    "mcp_health_checks": mcp_health_checks,
                },
                "catalog_representatives": representatives,
                "tested_tools": sorted(set(tested_tools)),
            }


def run_health(catalog_path: Path, tasks_root: Path) -> dict[str, Any]:
    # Import lazily so the structural/unit checks run outside the Toolathlon SIF.
    from openreward import OpenReward

    catalog = load_and_validate_catalog(catalog_path)
    playwright = validate_playwright_runtime()
    validate_cover(tasks_root)

    # Deterministically assign every read-only representative call to the
    # first cover session that includes that server. Parallel workers therefore
    # never race on shared Python state.
    probe_owners: dict[str, str] = {}
    for task_name, servers in SESSION_COVER.items():
        for server in servers:
            if server in READ_ONLY_PROBES and server not in probe_owners:
                probe_owners[server] = task_name
    if set(probe_owners) != set(READ_ONLY_PROBES):
        raise RuntimeError("read-only probe set is not covered by the seven tasks")
    probes_by_task = {
        task_name: tuple(sorted(server for server, owner in probe_owners.items() if owner == task_name))
        for task_name in SESSION_COVER
    }

    pair_overlap = tuple(sorted(set(SESSION_COVER[CONCURRENT_TASKS[0]]) & set(SESSION_COVER[CONCURRENT_TASKS[1]])))
    if pair_overlap != CONCURRENT_OVERLAP:
        raise RuntimeError(f"concurrent task overlap drift: {pair_overlap!r} != {CONCURRENT_OVERLAP!r}")

    results: dict[str, dict[str, Any]] = {}
    barrier = Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            task_name: executor.submit(
                run_one_session,
                OpenReward,
                task_name,
                SESSION_COVER[task_name],
                catalog,
                probes_by_task[task_name],
                barrier,
            )
            for task_name in CONCURRENT_TASKS
        }
        for task_name in CONCURRENT_TASKS:
            results[task_name] = futures[task_name].result(timeout=1200)

    for task_name, expected_servers in SESSION_COVER.items():
        if task_name in results:
            continue
        results[task_name] = run_one_session(
            OpenReward,
            task_name,
            expected_servers,
            catalog,
            probes_by_task[task_name],
        )

    sessions = [results[task_name]["session"] for task_name in SESSION_COVER]
    catalog_representatives = sorted(
        {name for result in results.values() for name in result["catalog_representatives"]}
    )
    tested_tools = sorted({name for result in results.values() for name in result["tested_tools"]})

    if len(sessions) != 7:
        raise RuntimeError(f"expected seven completed sessions, got {len(sessions)}")
    if "playwright_with_chunk.browser_tab_list" not in tested_tools:
        raise RuntimeError("portable Playwright browser_tab_list live launch was not exercised")
    playwright["browser_tab_list"] = "passed"
    return {
        "schema_version": 1,
        "gate": "toolathlon_all_mcp_health_driver",
        "status": "passed",
        "expected_server_count": len(EXPECTED_SERVERS),
        "tested_servers": list(EXPECTED_SERVERS),
        "failure_count": 0,
        "sessions": sessions,
        "concurrency": {
            "status": "passed",
            "simultaneous_active_sessions_verified": 2,
            "tasks": list(CONCURRENT_TASKS),
            "overlapping_servers": list(CONCURRENT_OVERLAP),
            "both_called": ["emails.check_connection"],
        },
        "catalog_representatives": catalog_representatives,
        "tested_tools": tested_tools,
        "catalog_sha256": sha256(catalog_path),
        "playwright": playwright,
        "completed_at_epoch": int(time.time()),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_absolute() or path.parent.resolve() != path.parent:
        raise RuntimeError(f"health report path is not direct and absolute: {path}")
    parent = path.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.getuid() or stat.S_IMODE(parent.st_mode) & 0o077:
        raise RuntimeError(f"health report parent is not private/user-owned: {path.parent}")
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or path.is_symlink() or temp.exists() or temp.is_symlink():
        raise RuntimeError(f"refusing unsafe report path: {path}")
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, path)
        except FileExistsError as exc:
            raise RuntimeError(f"health report target appeared: {path}") from exc
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)
    value = path.lstat()
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.getuid()
        or stat.S_IMODE(value.st_mode) != 0o600
        or value.st_nlink != 1
    ):
        raise RuntimeError("published health report identity/mode mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--catalog", type=Path, default=Path("/app/tool_schemas.json"))
    parser.add_argument("--tasks-root", type=Path, default=Path("/app/tasks"))
    args = parser.parse_args()

    started = int(time.time())
    try:
        report = run_health(args.catalog, args.tasks_root)
    except BaseException as exc:
        failed = {
            "schema_version": 1,
            "gate": "toolathlon_all_mcp_health_driver",
            "status": "failed",
            "expected_server_count": len(EXPECTED_SERVERS),
            "tested_servers": list(EXPECTED_SERVERS),
            "failure_count": 1,
            "error": f"{type(exc).__name__}: {exc}",
            "started_at_epoch": started,
            "completed_at_epoch": int(time.time()),
        }
        write_report(args.output, failed)
        raise
    write_report(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
