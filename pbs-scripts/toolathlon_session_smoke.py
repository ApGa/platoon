#!/usr/bin/env python3
"""Exercise concurrent Toolathlon sessions and deterministic terminal rewards."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from openreward import OpenReward


ENVIRONMENT = "toolathlongym"
TASK_NAME = "canvas-announcement-summary"
TOOL_ROUTING_SCHEMA_KEY = "x-openhands-tool-routing"
TOOL_ERROR_META_KEY = "openhands.dev/tool-error"
GOLD_FILE = Path(
    "/app/tasks/finalpool/canvas-announcement-summary/groundtruth_workspace/"
    "Canvas_Announcements.xlsx"
)


def text_of(output: object) -> str:
    blocks = getattr(output, "blocks", [])
    return "\n".join(str(getattr(block, "text", "")) for block in blocks)


def selected_task(environment: object) -> object:
    tasks = list(environment.list_tasks(split="train"))
    for task in tasks:
        task_spec = getattr(task, "task_spec", {})
        if task_spec.get("task_name") == TASK_NAME:
            return task
    raise RuntimeError(f"Toolathlon smoke task not found: {TASK_NAME}")


def run_session(label: str, barrier: Barrier, *, install_gold: bool) -> dict[str, object]:
    expected_marker = f"isolated-{label}"
    with OpenReward(api_key="local") as client:
        environment = client.environments.get(name=ENVIRONMENT)
        task = selected_task(environment)
        with environment.session(task=task) as session:
            tools = session.list_tools()
            tool_names = {str(getattr(tool, "name", "")) for tool in tools}
            required = {"python_execute", "claim_done"}
            if not required.issubset(tool_names):
                raise RuntimeError(
                    f"session {label} missing tools {sorted(required - tool_names)}"
                )

            by_name = {str(getattr(tool, "name", "")): tool for tool in tools}
            python_schema = getattr(by_name["python_execute"], "input_schema", {})
            routing = python_schema.get(TOOL_ROUTING_SCHEMA_KEY)
            if not isinstance(routing, dict) or routing.get("execution_domain") != "task":
                raise RuntimeError(
                    f"session {label} missing Toolathlon routing metadata"
                )
            if routing.get("invocation") != {"kind": "direct"}:
                raise RuntimeError(
                    f"session {label} has invalid python routing metadata: {routing!r}"
                )

            declared_failure = session.call_tool(
                "python_execute",
                {"code": "raise RuntimeError('v8 typed error probe')"},
            )
            failure_metadata = getattr(declared_failure, "metadata", None)
            marker = (
                failure_metadata.get(TOOL_ERROR_META_KEY)
                if isinstance(failure_metadata, dict)
                else None
            )
            if not isinstance(marker, dict) or marker.get("kind") != "nonzero_exit":
                raise RuntimeError(
                    f"session {label} did not return a typed execution error"
                )

            write = session.call_tool(
                "python_execute",
                {"code": f"from pathlib import Path; Path('isolation.txt').write_text({expected_marker!r})"},
            )
            if "error" in text_of(write).lower():
                raise RuntimeError(f"session {label} marker write failed: {text_of(write)}")
            barrier.wait(timeout=120)
            read = session.call_tool(
                "python_execute",
                {"code": "from pathlib import Path; print(Path('isolation.txt').read_text())"},
            )
            marker_output = text_of(read)
            if expected_marker not in marker_output:
                raise RuntimeError(
                    f"session {label} did not retain its marker: {marker_output!r}"
                )
            other = "isolated-b" if label == "a" else "isolated-a"
            if other in marker_output:
                raise RuntimeError(f"session {label} observed another session's marker")

            if install_gold:
                if not GOLD_FILE.is_file():
                    raise RuntimeError(f"gold smoke artifact is missing: {GOLD_FILE}")
                copy_result = session.call_tool(
                    "python_execute",
                    {
                        "code": (
                            "import shutil; "
                            f"shutil.copy2({str(GOLD_FILE)!r}, 'Canvas_Announcements.xlsx')"
                        )
                    },
                )
                if "error" in text_of(copy_result).lower():
                    raise RuntimeError(
                        f"session {label} deterministic setup failed: {text_of(copy_result)}"
                    )

            terminal = session.call_tool("claim_done", {})
            reward = getattr(terminal, "reward", None)
            finished = bool(getattr(terminal, "finished", False))
            if not finished:
                raise RuntimeError(f"session {label} terminal tool did not finish")
            expected_reward = 1.0 if install_gold else 0.0
            if reward != expected_reward:
                raise RuntimeError(
                    f"session {label} reward {reward!r}, expected {expected_reward}"
                )
            return {
                "label": label,
                "sid": session.sid,
                "reward": reward,
                "finished": finished,
                "tool_count": len(tools),
                "routing_metadata": True,
                "typed_error_metadata": True,
            }


def main() -> None:
    if not os.environ.get("OPENREWARD_SESSION_URL"):
        raise RuntimeError("OPENREWARD_SESSION_URL is required")
    barrier = Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_session, "a", barrier, install_gold=True),
            executor.submit(run_session, "b", barrier, install_gold=False),
        ]
        results = [future.result(timeout=600) for future in futures]
    print(json.dumps({"toolathlon_session_smoke": "ok", "sessions": results}, sort_keys=True))


if __name__ == "__main__":
    main()
