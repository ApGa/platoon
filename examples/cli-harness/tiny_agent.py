"""tiny-agent: a small command-line agent, the stand-in for your own in the cli-harness example.

It knows nothing about platoon. Like most CLI agents it talks to an OpenAI-compatible Chat Completions
endpoint (``OPENAI_BASE_URL`` and ``OPENAI_API_KEY``, as the OpenAI SDK reads them), has a built-in tool
(``shell``: ``bash -c`` in the current directory) that ``--no-shell`` turns off, and takes extra tools from MCP
servers listed in a JSON config file (``--mcp-config``, the common ``{"mcpServers": {...}}`` format, where
``${VAR}`` in a value is read from the environment).

It loops: one model call per turn, every tool call answered, until the model replies without a tool call. Then
it prints ``FINAL ANSWER: <reply>`` and exits 0. Exit codes: 0 answered, 2 model endpoint error, 3 turn limit
(``--max-turns``) reached, 4 MCP server error.

Standard library only, so it runs wherever ``python3`` exists.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import urllib.request

SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Run a bash command in the working directory; returns its exit code and output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}
EXIT_API_ERROR, EXIT_TURN_LIMIT, EXIT_MCP_ERROR = 2, 3, 4


def post(url: str, body: dict, headers: dict[str, str], timeout: float = 600) -> tuple[dict, str]:
    """POST JSON; returns the reply (a JSON body, or the last JSON event of an SSE stream) and the session id."""

    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", **headers}
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        session = response.headers.get("Mcp-Session-Id", "")
        if response.headers.get("Content-Type", "").startswith("text/event-stream"):
            reply: dict = {}
            for line in response:  # keep the last "data:" event; skip keepalive comments
                text = line.decode().strip()
                if text.startswith("data:"):
                    reply = json.loads(text[5:])
            return reply, session
        body = response.read()  # empty for a notification (HTTP 202)
        return (json.loads(body) if body.strip() else {}), session


class McpServer:
    """Just enough of an MCP streamable-HTTP client to list and call tools."""

    def __init__(self, url: str, headers: dict[str, str]) -> None:
        self.url, self.headers, self.ids = url, dict(headers), itertools.count(1)

    def request(self, method: str, params: dict | None = None) -> dict:
        reply, session = post(
            self.url, {"jsonrpc": "2.0", "id": next(self.ids), "method": method, "params": params or {}}, self.headers
        )
        if session:
            self.headers["Mcp-Session-Id"] = session
        if "error" in reply:
            raise RuntimeError(f"{method}: {reply['error'].get('message')}")
        return reply.get("result") or {}

    def connect(self) -> list[dict]:
        self.request(
            "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "tiny-agent"}}
        )
        post(self.url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, self.headers)
        return self.request("tools/list").get("tools") or []


def load_mcp_servers(path: str) -> list[McpServer]:
    with open(path, encoding="utf-8") as handle:
        servers = json.load(handle).get("mcpServers") or {}
    return [
        McpServer(
            server["url"], {key: os.path.expandvars(value) for key, value in (server.get("headers") or {}).items()}
        )
        for server in servers.values()
    ]


def chat(model: str, messages: list[dict], tools: list[dict]) -> dict:
    url = os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    body = {"model": model, "messages": messages, **({"tools": tools} if tools else {})}
    reply, _ = post(url, body, headers)
    return reply["choices"][0]["message"]


def run_shell(command: str) -> str:
    try:
        process = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "the command timed out after 120 s"
    return f"exit code {process.returncode}\n{(process.stdout + process.stderr)[-8000:]}"


def answer(call: dict, routes: dict[str, McpServer], *, shell: bool) -> tuple[str, bool]:
    """Run one tool call: its output, and whether the tool said the episode is over."""

    name = call["function"]["name"]
    try:
        arguments = json.loads(call["function"]["arguments"] or "{}")
    except ValueError as exc:
        return f"invalid arguments: {exc}", False
    if name == "shell" and shell:
        return run_shell(str(arguments.get("command", ""))), False
    if name not in routes:
        return f"unknown tool {name!r}", False
    result = routes[name].request("tools/call", {"name": name, "arguments": arguments})
    text = "\n".join(block.get("text", "") for block in result.get("content") or [])
    # Optional and platoon-specific: the result that ends the episode says so in _meta, so the agent can stop.
    return text, bool((result.get("_meta") or {}).get("platoon", {}).get("finished"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A small command-line agent.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-file", required=True, help="a file with the task")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--mcp-config", help='a JSON file of {"mcpServers": {name: {"url", "headers"}}}')
    parser.add_argument("--no-shell", action="store_true", help="turn the built-in shell tool off")
    args = parser.parse_args(argv)

    tools: list[dict] = [] if args.no_shell else [SHELL_TOOL]
    routes: dict[str, McpServer] = {}  # tool name -> the MCP server that serves it
    try:
        for server in load_mcp_servers(args.mcp_config) if args.mcp_config else []:
            for tool in server.connect():
                routes[tool["name"]] = server
                schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
                function = {"name": tool["name"], "description": tool.get("description", ""), "parameters": schema}
                tools.append({"type": "function", "function": function})
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"tiny-agent: MCP server error: {exc}", file=sys.stderr)
        return EXIT_MCP_ERROR

    with open(args.task_file, encoding="utf-8") as handle:
        messages: list[dict] = [{"role": "user", "content": handle.read()}]
    for _ in range(args.max_turns):
        try:
            message = chat(args.model, messages, tools)
        except (OSError, ValueError, LookupError) as exc:
            print(f"tiny-agent: model request failed: {exc}", file=sys.stderr)
            return EXIT_API_ERROR
        messages.append({key: value for key, value in message.items() if value is not None})
        if not message.get("tool_calls"):
            print(f"FINAL ANSWER: {message.get('content') or ''}")
            return 0
        for call in message["tool_calls"]:
            try:
                output, finished = answer(call, routes, shell=not args.no_shell)
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"tiny-agent: MCP tool call failed: {exc}", file=sys.stderr)
                return EXIT_MCP_ERROR
            print(f"$ {call['function']['name']} {call['function']['arguments']}\n{output}", flush=True)
            if finished:
                print(f"FINAL ANSWER: {output}")
                return 0
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
    print(f"tiny-agent: no answer after {args.max_turns} turns", file=sys.stderr)
    return EXIT_TURN_LIMIT


if __name__ == "__main__":
    sys.exit(main())
