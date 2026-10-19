"""platoon-bash: a minimal reference harness for CPU tests and smoke runs.

It loops OpenAI Chat Completions calls with a ``bash`` tool, runs each command
with ``bash -c`` in the current directory, and stops when the model answers
without a tool call. Standard library only, so it runs wherever ``python3``
exists. The API key is read from ``PLATOON_API_KEY``.

With ``--mcp-url`` it also connects to that MCP server (streamable HTTP, the
API key as a Bearer token), offers the server's tools next to ``bash``, and
routes their calls there. A tool result whose ``_meta.platoon.finished`` is
true (the env says the episode is over) ends the run with that result as the
final message.

Exit codes: 0 finished, 2 model endpoint error, 3 turn limit reached, 4 MCP
server error.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import urllib.error
import urllib.request

SYSTEM_PROMPT = (
    "You are an agent working in a Linux shell. Use the bash tool to inspect and change files "
    "in the current directory. When the task is done, reply with your final answer and no tool call."
)
MCP_PROMPT = " The environment also provides its own tools; use them as the task describes."
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in the working directory and return its exit code and output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
        },
    },
}
MAX_OUTPUT_CHARS = 16_000
EXIT_API_ERROR = 2
EXIT_TURN_LIMIT = 3
EXIT_MCP_ERROR = 4
MCP_PROTOCOL_VERSION = "2025-06-18"


class McpError(Exception):
    """The MCP server answered with a JSON-RPC error."""


class McpUnavailable(Exception):
    """The MCP server could not be reached, or its reply could not be read."""


class McpClient:
    """Just enough of an MCP streamable-HTTP client to list and call tools.

    Each JSON-RPC message is one POST; the server answers with JSON or with an
    SSE stream that ends in the reply (keepalive comments and notifications
    before it are skipped). The socket timeout bounds each read, not the call.
    """

    def __init__(self, url: str, api_key: str, timeout: float) -> None:
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.session_id: str | None = None
        self.protocol_version: str | None = None
        self._ids = itertools.count(1)

    def initialize(self) -> None:
        result = self.request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "platoon-bash", "version": "1"},
            },
        )
        self.protocol_version = result.get("protocolVersion") or MCP_PROTOCOL_VERSION
        self.notify("notifications/initialized")

    def list_tools(self) -> list[dict]:
        tools: list[dict] = []
        cursor = None
        while True:
            result = self.request("tools/list", {"cursor": cursor} if cursor else None)
            tools.extend(result.get("tools") or [])
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict) -> dict:
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def request(self, method: str, params: dict | None = None) -> dict:
        request_id = next(self._ids)
        message: dict = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        with self._post(message) as response:
            self.session_id = response.headers.get("Mcp-Session-Id") or self.session_id
            if response.headers.get("Content-Type", "").startswith("text/event-stream"):
                reply = _sse_reply(response, request_id)
            else:
                reply = json.load(response)
        error = reply.get("error")
        if error is not None:
            raise McpError(f"{method}: {error.get('message')} (code {error.get('code')})")
        return reply.get("result") or {}

    def notify(self, method: str) -> None:
        with self._post({"jsonrpc": "2.0", "method": method}):
            pass

    def close(self) -> None:
        if self.session_id is None:
            return
        request = urllib.request.Request(self.url, headers=self._headers(), method="DELETE")
        try:
            urllib.request.urlopen(request, timeout=10).close()
        except OSError:
            pass  # 405: the server keeps no sessions to end

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        return headers

    def _post(self, message: dict):
        request = urllib.request.Request(
            self.url, data=json.dumps(message).encode(), headers=self._headers(), method="POST"
        )
        return urllib.request.urlopen(request, timeout=self.timeout)


def _sse_reply(stream, request_id: int) -> dict:
    """Read SSE events until the reply to ``request_id``."""
    data: list[str] = []
    for raw in stream:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                message = json.loads("\n".join(data))
                data = []
                if message.get("id") == request_id and ("result" in message or "error" in message):
                    return message
            continue
        if line.startswith(":"):
            continue  # a keepalive comment
        name, _, value = line.partition(":")
        if name == "data":
            data.append(value[1:] if value.startswith(" ") else value)
    raise McpError("the server closed the SSE stream without a reply")


def openai_tool(tool: dict) -> dict:
    """An MCP tool as an OpenAI function tool."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description") or "",
            "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
        },
    }


def mcp_result_text(result: dict) -> str:
    parts = []
    for block in result.get("content") or []:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        else:
            parts.append(f"[{block.get('type', 'content')}]")
    text = "\n".join(parts)
    return f"error: {text}" if result.get("isError") else text


def mcp_finished(result: dict) -> bool:
    platoon = (result.get("_meta") or {}).get("platoon") or {}
    return bool(platoon.get("finished"))


def chat(args: argparse.Namespace, messages: list[dict], tools: list[dict]) -> dict:
    body = json.dumps({"model": args.model, "messages": messages, "tools": tools}).encode()
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {os.environ['PLATOON_API_KEY']}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
        return json.load(response)["choices"][0]["message"]


def run_bash(command: str, timeout: float) -> str:
    process = subprocess.Popen(
        ["bash", "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        return f"command timed out after {timeout:g}s\n{output[-MAX_OUTPUT_CHARS:]}"
    return f"exit code {process.returncode}\n{output[-MAX_OUTPUT_CHARS:]}"


def tool_result(call: dict, timeout: float, mcp: McpClient | None, mcp_tools: set[str]) -> tuple[str, bool]:
    """Run one tool call; returns its output and whether the env finished the episode.

    A JSON-RPC error is shown to the model; a transport failure raises :class:`McpUnavailable`.
    """
    function = call.get("function") or {}
    name = function.get("name")
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        arguments = None
    if name == "bash":
        if not isinstance(arguments, dict) or not isinstance(arguments.get("command"), str):
            return 'invalid arguments; expected {"command": "<bash command>"}', False
        return run_bash(arguments["command"], timeout), False
    if mcp is not None and name in mcp_tools:
        if not isinstance(arguments, dict):
            return "invalid arguments; expected a JSON object", False
        try:
            result = mcp.call_tool(name, arguments)
        except McpError as exc:
            return f"error: {exc}", False
        except (OSError, ValueError) as exc:  # URLError and HTTPError are OSErrors; a garbled reply a ValueError
            raise McpUnavailable(str(exc)) from exc
        return mcp_result_text(result), mcp_finished(result)
    available = ", ".join(["bash", *sorted(mcp_tools)])
    return f"unknown tool {name!r}; the tools are {available}", False


def connect_mcp(url: str, api_key: str, timeout: float) -> tuple[McpClient, list[dict]]:
    """Connect to the MCP server and return its tools as OpenAI tools (one named bash is skipped)."""
    client = McpClient(url, api_key, timeout)
    client.initialize()
    tools = []
    for tool in client.list_tools():
        if tool.get("name") == "bash":
            print("platoon-bash: skipping the MCP tool 'bash', which the built-in shadows", file=sys.stderr)
            continue
        tools.append(openai_tool(tool))
    return client, tools


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--instruction-file", required=True)
    parser.add_argument("--final-message-file")
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--command-timeout", type=float, default=120.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--mcp-url", help="MCP server (streamable HTTP) whose tools are offered next to bash")
    args = parser.parse_args(argv)

    with open(args.instruction_file, encoding="utf-8") as handle:
        instruction = handle.read()
    tools: list[dict] = [BASH_TOOL]
    mcp: McpClient | None = None
    if args.mcp_url:
        try:
            mcp, mcp_openai_tools = connect_mcp(args.mcp_url, os.environ["PLATOON_API_KEY"], args.request_timeout)
        except (OSError, LookupError, ValueError, McpError) as exc:
            print(f"platoon-bash: MCP server unavailable: {exc}", file=sys.stderr)
            return EXIT_MCP_ERROR
        tools.extend(mcp_openai_tools)
    mcp_tools = {tool["function"]["name"] for tool in tools[1:]}
    system_prompt = SYSTEM_PROMPT + (MCP_PROMPT if mcp_tools else "")
    messages: list[dict] = [{"role": "system", "content": system_prompt}, {"role": "user", "content": instruction}]
    try:
        for _ in range(args.max_turns):
            try:
                message = chat(args, messages, tools)
            except (OSError, LookupError, ValueError) as exc:  # URLError is an OSError; bad JSON a ValueError
                print(f"platoon-bash: model request failed: {exc}", file=sys.stderr)
                return EXIT_API_ERROR
            messages.append({key: value for key, value in message.items() if value is not None})
            calls = message.get("tool_calls") or []
            if not calls:
                return finish(args, message.get("content") or "")
            for call in calls:
                try:
                    result, finished = tool_result(call, args.command_timeout, mcp, mcp_tools)
                except McpUnavailable as exc:
                    print(f"platoon-bash: MCP tool call failed: {exc}", file=sys.stderr)
                    return EXIT_MCP_ERROR
                function = call.get("function") or {}
                prompt = "$" if function.get("name") == "bash" else f"> {function.get('name')}"
                print(f"{prompt} {function.get('arguments')}\n{result}", flush=True)
                messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": result})
                if finished:
                    return finish(args, result)
        print(f"platoon-bash: no final answer after {args.max_turns} turns", file=sys.stderr)
        return EXIT_TURN_LIMIT
    finally:
        if mcp is not None:
            mcp.close()


def finish(args: argparse.Namespace, final_message: str) -> int:
    if args.final_message_file:
        with open(args.final_message_file, "w", encoding="utf-8") as handle:
            handle.write(final_message)
    print(final_message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
