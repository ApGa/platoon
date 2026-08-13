#!/usr/bin/env python3
"""One-session latest-stack Toolathlon -> MCP -> OpenHands PTC acceptance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import stat
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

EXPECTED_OPENHANDS_REVISION = "b1acf52ee362884f092b9eaf5e701a1bb5d1b335"
EXPECTED_TOOLATHLON_REVISION = "18e62c0d0419da88209af911528f2f8a0b44a5ed"
TASK_NAME = "canvas-announcement-summary"
TYPED_ERROR_MARKER = "v8-ptc-typed-error-probe"
SIBLING_MARKER = "v8-ptc-sibling-survived"


@contextmanager
def _closing_executor(executor: Any):
    try:
        yield executor
    finally:
        executor.close()


def _private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_private_regular(path: Path) -> None:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid() or stat.S_IMODE(value.st_mode) & 0o077:
        raise PermissionError(f"unsafe PTC smoke artifact: {path}")


def _direct_url_revision(distribution: str) -> str:
    raw = importlib.metadata.distribution(distribution).read_text("direct_url.json")
    if not raw:
        raise RuntimeError(f"{distribution} has no direct_url provenance")
    value = json.loads(raw)
    revision = value.get("vcs_info", {}).get("commit_id")
    if not isinstance(revision, str):
        raise RuntimeError(f"{distribution} direct_url lacks a VCS commit")
    return revision


def _events(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError("bridge event is not an object")
        values.append(value)
    return values


def _result_metadata(event: dict[str, Any]) -> dict[str, Any]:
    result = event.get("result")
    if not isinstance(result, dict):
        return {}
    metadata = result.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    # LiteLLM otherwise attempts a network refresh of its model-cost map while
    # importing the exact controller stack. This gate has no external LLM call.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    from openhands.sdk.mcp import create_mcp_tools
    from openhands.tools.programmatic_tool_calling import (
        ProgrammaticToolCallingAction,
        ProgrammaticToolCallingMode,
        ProgrammaticToolCallingTool,
    )
    from openhands.tools.programmatic_tool_calling.impl import (
        ProgrammaticToolCallingExecutor,
    )
    from openreward import OpenReward
    from platoon.openreward.config_defs import (
        OpenRewardConfig,
        OpenRewardEnvironmentConfig,
    )
    from platoon.openreward.rollout import _build_mcp_config
    from platoon.openreward.tasks import (
        OPENREWARD_ENVIRONMENT_LABEL_KEY,
        OPENREWARD_TASK_INDEX_KEY,
        OPENREWARD_TASK_NAME_KEY,
        OPENREWARD_TASK_SPLIT_KEY,
        _configure_openreward_urls,
    )

    if Path(sys.executable).resolve() != args.controller_python.resolve():
        raise RuntimeError("PTC smoke is not executing the exact Gate2 controller Python")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != args.driver_sha256:
        raise RuntimeError("PTC smoke driver snapshot digest mismatch")
    for distribution in (
        "openhands-sdk",
        "openhands-tools",
        "openhands-workspace",
        "openhands-agent-server",
    ):
        if _direct_url_revision(distribution) != EXPECTED_OPENHANDS_REVISION:
            raise RuntimeError(f"{distribution} is not installed from the pinned OpenHands revision")

    environment = OpenRewardEnvironmentConfig(
        label="toolathlon",
        env_name="toolathlongym",
        split="train",
        api_key="local",
        session_urls=[args.session_url],
        max_tool_calls=8,
    )
    openreward = OpenRewardConfig(environments=[environment])
    _configure_openreward_urls(environment, allow_legacy_global_pool=False)
    if (
        os.environ.get("OPENREWARD_SESSION_URL") != args.session_url
        or os.environ.get("OPENREWARD_API_URL") != args.session_url
        or os.environ.get("OPENREWARD_DISABLE_UPDATE_CHECK") != "1"
    ):
        raise RuntimeError("production OpenReward URL configuration is not exact")
    with OpenReward(api_key="local") as client:
        catalog = client.environments.get(name="toolathlongym")
        matching_indices = [
            index
            for index, task in enumerate(catalog.list_tasks(split="train"))
            if getattr(task, "task_spec", {}).get("task_name") == TASK_NAME
        ]
    if len(matching_indices) != 1:
        raise RuntimeError(f"Toolathlon catalog did not contain one {TASK_NAME!r} task")

    class _Task:
        id = TASK_NAME
        misc = {
            OPENREWARD_ENVIRONMENT_LABEL_KEY: "toolathlon",
            OPENREWARD_TASK_INDEX_KEY: matching_indices[0],
            OPENREWARD_TASK_NAME_KEY: TASK_NAME,
            OPENREWARD_TASK_SPLIT_KEY: "train",
        }

    bridge_root = args.state_root / "bridge"
    bridge_root.mkdir(mode=0o700)
    mcp_config = _build_mcp_config(_Task(), openreward, environment, str(bridge_root))
    server = mcp_config["mcpServers"]["openreward"]
    if server["command"] != sys.executable:
        raise RuntimeError("production MCP config did not use the Gate2 controller Python")
    if server["args"][:2] != ["-m", "platoon.openreward.mcp_bridge"]:
        raise RuntimeError("production MCP config did not select the Platoon bridge")

    bad_code = f"raise RuntimeError({TYPED_ERROR_MARKER!r})"
    good_code = f"from pathlib import Path; Path('ptc_sibling.txt').write_text({SIBLING_MARKER!r})"

    with ExitStack() as resources:
        client = resources.enter_context(create_mcp_tools(mcp_config, timeout=180.0))
        mcp_tools = client.tools
        tools_map = {tool.name: tool for tool in mcp_tools}
        required = {"python_execute", "get_task", "get_status"}
        if not required.issubset(tools_map):
            raise RuntimeError(f"MCP bridge lacks required tools: {sorted(required - tools_map.keys())}")
        routing = tools_map["python_execute"].meta or {}
        route = routing.get("openhands.dev/tool-routing")
        if not isinstance(route, dict) or route.get("execution_domain") != "task":
            raise RuntimeError("python_execute lacks the latest routing metadata")
        if route.get("invocation") != {"kind": "direct"}:
            raise RuntimeError("python_execute routing metadata is not direct")

        programmatic_tools = ProgrammaticToolCallingTool.create(
            SimpleNamespace(),
            mode=ProgrammaticToolCallingMode.ORCHESTRATION_ONLY,
            max_tool_calls_per_execution=1024,
        )
        if len(programmatic_tools) != 1 or programmatic_tools[0].executor is None:
            raise RuntimeError("OpenHands did not create exactly one executable PTC tool")
        programmatic_tool = programmatic_tools[0]
        executor: ProgrammaticToolCallingExecutor = resources.enter_context(
            _closing_executor(programmatic_tool.executor)
        )

        class _SecretRegistry:
            @staticmethod
            def get_secret_value(_name: str) -> None:
                return None

            @staticmethod
            def mask_secrets_in_output(value: str) -> str:
                return value

        conversation = SimpleNamespace(
            agent=SimpleNamespace(tools_map={**tools_map, programmatic_tool.name: programmatic_tool}),
            state=SimpleNamespace(secret_registry=_SecretRegistry()),
        )
        first = executor(
            ProgrammaticToolCallingAction(
                code=(
                    "persistent_probe = 'ptc-state-preserved'\n"
                    "bad, good = await asyncio.gather(\n"
                    f"    atools.python_execute(code={bad_code!r}),\n"
                    f"    atools.python_execute(code={good_code!r}),\n"
                    ")\n"
                    "print(bad.is_error, good.is_error)"
                )
            ),
            conversation,
        )
        if not first.is_error or first.error_kind.value != "tool_error":
            raise RuntimeError("typed nested MCP error did not surface as a PTC tool_error")
        if first.failed_tool_names != ("python_execute",):
            raise RuntimeError("PTC failed-tool provenance is not exact")
        if (
            first.tool_call_attempts,
            first.tool_calls_admitted,
            first.tool_calls_completed,
            first.tool_calls_rejected,
        ) != (2, 2, 2, 0):
            raise RuntimeError("PTC concurrent-cell telemetry is not 2/2/2/0")
        if "True False" not in first.text:
            raise RuntimeError("PTC siblings did not return typed error/success observations")

        second = executor(
            ProgrammaticToolCallingAction(
                code=(
                    "followup = await atools.python_execute("
                    "code=\"from pathlib import Path; print(Path('ptc_sibling.txt').read_text())\")\n"
                    "print(persistent_probe, followup.is_error, followup.text)"
                )
            ),
            conversation,
        )
        if second.is_error:
            raise RuntimeError("follow-up PTC cell failed after the typed sibling error")
        if (
            second.tool_call_attempts,
            second.tool_calls_admitted,
            second.tool_calls_completed,
            second.tool_calls_rejected,
        ) != (1, 1, 1, 0):
            raise RuntimeError("PTC follow-up telemetry is not 1/1/1/0")
        if "ptc-state-preserved" not in second.text or SIBLING_MARKER not in second.text:
            raise RuntimeError("PTC or environment session state did not survive the typed error")

    events_path = bridge_root / "bridge_events.jsonl"
    state_path = bridge_root / "bridge_state.json"
    if not events_path.is_file() or not state_path.is_file():
        raise RuntimeError("bridge did not publish durable state")
    events = _events(events_path)
    calls = [event for event in events if event.get("type") == "tool_call"]
    results = [event for event in events if event.get("type") == "tool_result"]
    if len(calls) != 3 or len(results) != 3:
        raise RuntimeError("bridge did not record exactly three call/result pairs")
    typed = []
    for event in results:
        marker = _result_metadata(event).get("openhands.dev/tool-error")
        if marker is not None:
            typed.append(marker)
    if len(typed) != 1 or not isinstance(typed[0], dict) or typed[0].get("kind") != "nonzero_exit":
        raise RuntimeError("bridge did not record exactly one structured nonzero_exit")
    state = json.loads(state_path.read_text())
    if state.get("turn") != 3 or state.get("finished") is not False:
        raise RuntimeError("bridge state did not remain live at turn three")
    result = {
        "schema_version": 1,
        "status": "passed",
        "pbs_job_id": args.pbs_job_id,
        "source_sha256": args.source_sha256,
        "controller_environment_key": args.controller_environment_key,
        "driver_sha256": args.driver_sha256,
        "openhands_revision": EXPECTED_OPENHANDS_REVISION,
        "toolathlon_revision": EXPECTED_TOOLATHLON_REVISION,
        "ptc_mode": "orchestration_only",
        "ptc_max_tool_calls": 1024,
        "environment_max_tool_calls": 8,
        "routing_metadata": "direct",
        "first_cell": {
            "is_error": True,
            "error_kind": "tool_error",
            "failed_tool_names": ["python_execute"],
            "telemetry": [2, 2, 2, 0],
            "typed_error": True,
            "sibling_success": True,
        },
        "followup_cell": {"is_error": False, "telemetry": [1, 1, 1, 0]},
        "bridge_turn": 3,
        "bridge_finished": False,
        "typed_error_kind": "nonzero_exit",
    }
    result_path = args.state_root / "RESULT.json"
    _private_json(result_path, result)
    _assert_private_regular(result_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--session-url", required=True)
    parser.add_argument("--controller-python", type=Path, required=True)
    parser.add_argument("--controller-environment-key", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--pbs-job-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.state_root.is_absolute() or args.state_root.exists() or args.state_root.is_symlink():
        raise RuntimeError("PTC smoke requires a fresh absolute state root")
    args.state_root.mkdir(mode=0o700, parents=True)
    result = run(args)
    print(f"openreward_v8_ptc_e2e=PASS result={args.state_root / 'RESULT.json'}", flush=True)
    if result["status"] != "passed":
        raise RuntimeError("PTC result status is not passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
