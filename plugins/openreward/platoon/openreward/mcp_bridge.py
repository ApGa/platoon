from __future__ import annotations

import argparse
import atexit
import inspect
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

DECLARED_RESOURCES_META_KEY = "openhands.dev/declared_resources"


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return repr(value)


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")[:80] or "value"


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: list[str] = []
        for block in prompt:
            text = getattr(block, "text", None)
            parts.append(text if text is not None else str(block))
        return "\n".join(parts)
    text = getattr(prompt, "text", None)
    return text if text is not None else str(prompt)


def _tool_result_to_payload(result: Any) -> dict[str, Any]:
    blocks = getattr(result, "blocks", None) or []
    block_texts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        block_texts.append(text if isinstance(text, str) else str(block))
    data = getattr(result, "data", None)
    reward = getattr(result, "reward", None)
    finished = bool(getattr(result, "finished", False))

    payload: dict[str, Any] = {
        "finished": finished,
        "reward": reward,
    }
    if block_texts:
        payload["text"] = "\n".join(block_texts)
        payload["blocks"] = block_texts
    if data is not None:
        payload["data"] = data
    if not block_texts and data is None:
        payload["raw"] = repr(result)
    return payload


def _normalize_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        if arguments.strip() == "":
            return {}
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"arguments must be a JSON object or JSON string: {exc}") from exc
        if parsed is None:
            return {}
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("arguments must be a JSON object")


def _schema_type_to_annotation(schema: dict[str, Any]) -> type:
    schema_type = schema.get("type")
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "string":
        return str
    if schema_type == "array":
        return list
    if schema_type == "object":
        return dict
    return Any


def _tool_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    parameters = tool.get("parameters") or tool.get("input_schema") or tool.get("inputSchema")
    return parameters if isinstance(parameters, dict) else {}


def _lockfree_tool_meta() -> dict[str, Any]:
    return {DECLARED_RESOURCES_META_KEY: []}


def _make_environment_tool(runtime: "OpenRewardMCPBridge", tool: dict[str, Any]):
    tool_name = str(tool["name"])
    parameters_schema = _tool_parameters(tool)
    properties = parameters_schema.get("properties") or {}
    required = set(parameters_schema.get("required") or [])

    def _environment_tool(**kwargs) -> str:
        payload = runtime.call_openreward_tool(tool_name, kwargs)
        return json.dumps(payload, default=_json_default)

    _environment_tool.__name__ = _slug(tool_name).replace("-", "_")
    _environment_tool.__doc__ = tool.get("description") or f"Invoke OpenReward tool {tool_name}."
    signature_parameters = []
    for name, property_schema in properties.items():
        default = inspect.Parameter.empty if name in required else None
        annotation = _schema_type_to_annotation(property_schema if isinstance(property_schema, dict) else {})
        signature_parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )
    setattr(
        _environment_tool,
        "__signature__",
        inspect.Signature(signature_parameters, return_annotation=str),
    )
    return _environment_tool


def _task_name(task: Any, fallback: int) -> str:
    spec = getattr(task, "task_spec", None)
    if isinstance(spec, dict):
        for key in ("task_name", "id", "task_id", "name"):
            if spec.get(key):
                return str(spec[key])
    for key in ("task_name", "id", "task_id", "name"):
        value = getattr(task, key, None)
        if value:
            return str(value)
    return f"task-{fallback}"


@dataclass
class BridgeConfig:
    env_name: str
    split: str
    task_index: int
    task_name: str | None
    session_url: str | None
    api_url: str | None
    api_key: str
    output_dir: Path
    max_tool_calls: int


def _configure_openreward_urls(session_url: str | None, api_url: str | None) -> None:
    os.environ.setdefault("OPENREWARD_DISABLE_UPDATE_CHECK", "1")
    if session_url:
        os.environ["OPENREWARD_SESSION_URL"] = session_url
        os.environ["OPENREWARD_API_URL"] = api_url or session_url
    elif api_url:
        os.environ["OPENREWARD_API_URL"] = api_url


class OpenRewardMCPBridge:
    def __init__(self, config: BridgeConfig) -> None:
        self.config = config
        self.events_path = config.output_dir / "bridge_events.jsonl"
        self.state_path = config.output_dir / "bridge_state.json"
        self.turn = 0
        self.finished = False
        self.last_reward: float | None = None
        self.closed = False

        _configure_openreward_urls(config.session_url, config.api_url)

        from openreward import OpenReward

        self.client = OpenReward(api_key=config.api_key)
        self.environment = self.client.environments.get(name=config.env_name)
        self.tasks = list(self.environment.list_tasks(split=config.split))
        if not self.tasks:
            raise RuntimeError(f"No tasks found for {config.env_name} split={config.split!r}")

        self.task = self._select_task()
        self.task_name = _task_name(self.task, config.task_index)
        self.session_context = self.environment.session(task=self.task)
        self.session = self.session_context.__enter__()
        self.prompt_text = _prompt_to_text(self.session.get_prompt())
        self.tools = list(self.session.list_tools(format="openai"))

        self._record(
            "session_started",
            {
                "env": config.env_name,
                "split": config.split,
                "task_index": config.task_index,
                "task_name": self.task_name,
                "tool_names": [tool.get("name") for tool in self.tools],
                "session_url": config.session_url,
            },
        )

    def _select_task(self) -> Any:
        if self.config.task_name:
            for index, task in enumerate(self.tasks):
                if _task_name(task, index) == self.config.task_name:
                    self.config.task_index = index
                    return task
            raise RuntimeError(f"Task {self.config.task_name!r} not found in split {self.config.split!r}")

        if self.config.task_index < 0 or self.config.task_index >= len(self.tasks):
            raise RuntimeError(f"task-index {self.config.task_index} outside range 0..{len(self.tasks) - 1}")
        return self.tasks[self.config.task_index]

    def _record(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "type": event_type,
            "time": time.time(),
            **payload,
        }
        _write_jsonl(self.events_path, record)
        _write_json(
            self.state_path,
            {
                "env": self.config.env_name,
                "split": self.config.split,
                "task_index": self.config.task_index,
                "task_name": self.task_name,
                "turn": self.turn,
                "finished": self.finished,
                "last_reward": self.last_reward,
                "updated_at": time.time(),
            },
        )

    def get_task(self) -> str:
        payload = {
            "task_name": self.task_name,
            "task_index": self.config.task_index,
            "prompt": self.prompt_text,
            "environment_tools": self.tools,
            "policy": (
                "Use the listed environment tools directly by name. If this environment "
                "exposes a catalog/meta tool such as call_tool, follow that environment's "
                "prompt for when to use it. Call claim_done when the task is complete."
            ),
        }
        self._record(
            "task_requested",
            {"prompt_chars": len(self.prompt_text), "tool_count": len(self.tools)},
        )
        return json.dumps(payload, default=_json_default)

    def _call_environment_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.finished:
            return {
                "bridge_error": "environment_already_finished",
                "finished": True,
                "reward": self.last_reward,
            }
        if self.config.max_tool_calls > 0 and self.turn >= self.config.max_tool_calls:
            self.finished = True
            self._record(
                "max_tool_calls_exceeded",
                {"max_tool_calls": self.config.max_tool_calls, "tool_name": name},
            )
            return {
                "bridge_error": "max_tool_calls_exceeded",
                "finished": True,
                "reward": self.last_reward,
            }

        self.turn += 1
        call_id = f"call_{self.turn:04d}_{_slug(name)}"
        self._record(
            "tool_call",
            {
                "turn": self.turn,
                "call_id": call_id,
                "tool_name": name,
                "arguments": arguments,
            },
        )

        result = self.session.call_tool(name, arguments)
        payload = _tool_result_to_payload(result)
        self.finished = bool(payload.get("finished"))
        reward = payload.get("reward")
        if isinstance(reward, (int, float)):
            self.last_reward = float(reward)

        self._record(
            "tool_result",
            {
                "turn": self.turn,
                "call_id": call_id,
                "tool_name": name,
                "result": payload,
            },
        )
        return payload

    def call_openreward_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._call_environment_tool(name, arguments)

    def get_tool_details(self, name: str) -> str:
        payload = self._call_environment_tool("get_tool_details", {"name": name})
        return json.dumps(payload, default=_json_default)

    def call_tool(self, name: str, arguments: Any = None) -> str:
        payload = self._call_environment_tool(
            "call_tool",
            {"name": name, "arguments": _normalize_arguments(arguments)},
        )
        return json.dumps(payload, default=_json_default)

    def python_execute(self, code: str) -> str:
        payload = self._call_environment_tool("python_execute", {"code": code})
        return json.dumps(payload, default=_json_default)

    def claim_done(self) -> str:
        payload = self._call_environment_tool("claim_done", {})
        return json.dumps(payload, default=_json_default)

    def get_status(self) -> str:
        return json.dumps(
            {
                "env": self.config.env_name,
                "split": self.config.split,
                "task_index": self.config.task_index,
                "task_name": self.task_name,
                "turn": self.turn,
                "finished": self.finished,
                "last_reward": self.last_reward,
            },
            sort_keys=True,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self._record(
                "session_closing",
                {"finished": self.finished, "last_reward": self.last_reward},
            )
        finally:
            self.session_context.__exit__(None, None, None)
            self.client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expose one OpenReward session through stdio MCP tools.")
    parser.add_argument("--env-name", default="toolathlongym")
    parser.add_argument("--split", default="train")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-name")
    parser.add_argument("--session-url", default=os.getenv("OPENREWARD_SESSION_URL"))
    parser.add_argument("--api-url", default=os.getenv("OPENREWARD_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENREWARD_API_KEY", "local"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tool-calls", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = OpenRewardMCPBridge(
        BridgeConfig(
            env_name=args.env_name,
            split=args.split,
            task_index=args.task_index,
            task_name=args.task_name,
            session_url=args.session_url,
            api_url=args.api_url,
            api_key=args.api_key,
            output_dir=args.output_dir.resolve(),
            max_tool_calls=args.max_tool_calls,
        )
    )
    atexit.register(runtime.close)

    mcp = FastMCP("openreward")

    @mcp.tool(meta=_lockfree_tool_meta())
    def get_task() -> str:
        """Return the OpenReward task prompt and environment tool catalog."""

        return runtime.get_task()

    @mcp.tool(meta=_lockfree_tool_meta())
    def get_status() -> str:
        """Return bridge state including turn count, reward, and finished flag."""

        return runtime.get_status()

    for tool in runtime.tools:
        tool_name = tool.get("name")
        if not tool_name or tool_name in {"get_task", "get_status"}:
            continue
        mcp.add_tool(
            _make_environment_tool(runtime, tool),
            name=str(tool_name),
            description=tool.get("description") or f"Invoke OpenReward tool {tool_name}.",
            meta=_lockfree_tool_meta(),
        )

    try:
        mcp.run()
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
