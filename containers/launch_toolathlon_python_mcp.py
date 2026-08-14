#!/usr/bin/python3
"""Exec one pinned Toolathlon Python MCP from its baked virtual environment."""

from __future__ import annotations

import os
import sys

PRODUCTION_PATH = "/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"

LAUNCHERS = {
    "arxiv-latex": (
        "/opt/local_servers/arxiv-latex-mcp",
        "/opt/local_servers/arxiv-latex-mcp/.venv/bin/python",
    ),
    "arxiv_local": (
        "/opt/local_servers/arxiv-mcp-server",
        "/opt/local_servers/arxiv-mcp-server/.venv/bin/arxiv-mcp-server",
    ),
    "emails": (
        "/opt/local_servers/emails-mcp",
        "/opt/local_servers/emails-mcp/.venv/bin/emails-mcp",
    ),
    "excel": (
        "/opt/local_servers/excel-mcp-server",
        "/opt/local_servers/excel-mcp-server/.venv/bin/excel-mcp-server",
    ),
    "google_sheet": (
        "/opt/local_servers/mcp-google-sheets",
        "/opt/local_servers/mcp-google-sheets/.venv/bin/mcp-google-sheets",
    ),
    "pdf-tools": (
        "/opt/local_servers/pdf-tools-mcp",
        "/opt/local_servers/pdf-tools-mcp/.venv/bin/pdf-tools-mcp",
    ),
    "pptx": (
        "/opt/local_servers/Office-PowerPoint-MCP-Server",
        "/opt/local_servers/Office-PowerPoint-MCP-Server/.venv/bin/ppt_mcp_server",
    ),
    "scholarly": (
        "/opt/local_servers/mcp-scholarly",
        "/opt/local_servers/mcp-scholarly/.venv/bin/mcp-scholarly",
    ),
    "snowflake": (
        "/opt/local_servers/mcp-snowflake-server",
        "/opt/local_servers/mcp-snowflake-server/.venv/bin/mcp_snowflake_server",
    ),
    "terminal": (
        "/opt/local_servers/cli-mcp-server",
        "/opt/local_servers/cli-mcp-server/.venv/bin/cli-mcp-server",
    ),
    "word": (
        "/opt/local_servers/Office-Word-MCP-Server",
        "/opt/local_servers/Office-Word-MCP-Server/.venv/bin/word_mcp_server",
    ),
    "yahoo-finance": (
        "/opt/local_servers/yahoo-finance-mcp",
        "/opt/local_servers/yahoo-finance-mcp/.venv/bin/python",
    ),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in LAUNCHERS:
        raise SystemExit("usage: launch_toolathlon_python_mcp.py <pinned-server> [args ...]")
    project, executable = LAUNCHERS[sys.argv[1]]
    os.chdir(project)
    environment = dict(os.environ)
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("UV_PROJECT_ENVIRONMENT", None)
    environment["PATH"] = f"{os.path.dirname(executable)}:{PRODUCTION_PATH}"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    os.execve(executable, [executable, *sys.argv[2:]], environment)


if __name__ == "__main__":
    main()
