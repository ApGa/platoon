"""dialect-agent: a minimal command-line agent for the hosted-model tests, which speaks Anthropic Messages or OpenAI
Responses to the gateway.

It plays a task with the environment's tools over MCP (platoon-bash's client, shipped next to it), calls the model in
its dialect, streamed with ``--stream``, and stops when the environment says the episode is over or the model
answers without a tool call.  Standard library only, like platoon-bash.  The agent key is read from
``PLATOON_API_KEY``.  Exit codes: 0 finished, 2 model endpoint error, 3 turn limit reached.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from platoon_bash import McpClient, mcp_finished, mcp_result_text

SYSTEM = "Find the hidden number. Call the guess tool with one integer per turn."
EMPTY = {"type": "object", "properties": {}}


def post(url: str, body: dict, headers: dict, stream: bool) -> dict:
    """The reply to ``body``: its JSON, or for a stream the reply its events make up."""

    request = urllib.request.Request(
        url,
        data=json.dumps({**body, "stream": stream}).encode(),
        headers={"Content-Type": "application/json", **headers},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        if not stream:
            return json.load(response)
        return assemble(read_events(response))


def read_events(response) -> list[dict]:
    events: list[dict] = []
    data: list[str] = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                events.append(json.loads("\n".join(data)))
                data = []
            continue
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        events.append(json.loads("\n".join(data)))
    return events


def assemble(events: list[dict]) -> dict:
    """A Messages message or a Responses response from its stream's events."""

    message: dict = {}
    blocks: dict[int, dict] = {}
    partial: dict[int, str] = {}
    for event in events:
        kind = event.get("type")
        if kind in ("response.completed", "response.incomplete"):
            return event["response"]
        if kind == "error" or kind == "response.failed":
            raise RuntimeError(f"the stream failed: {event}")
        if kind == "message_start":
            message = dict(event["message"])
        elif kind == "content_block_start":
            blocks[event["index"]] = dict(event["content_block"])
        elif kind == "content_block_delta":
            block, delta = blocks[event["index"]], event["delta"]
            if delta["type"] == "text_delta":
                block["text"] = block.get("text", "") + delta["text"]
            elif delta["type"] == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta["thinking"]
            elif delta["type"] == "signature_delta":
                block["signature"] = delta["signature"]
            elif delta["type"] == "input_json_delta":
                partial[event["index"]] = partial.get(event["index"], "") + delta["partial_json"]
        elif kind == "content_block_stop" and event["index"] in partial:
            blocks[event["index"]]["input"] = json.loads(partial.pop(event["index"]) or "{}")
        elif kind == "message_delta":
            message.update(event.get("delta") or {})
    message["content"] = [blocks[index] for index in sorted(blocks)]
    return message


def play_messages(
    args: argparse.Namespace, key: str, mcp: McpClient, tools: list[dict], instruction: str
) -> str | None:
    specs = [
        {"name": t["name"], "description": t.get("description") or "", "input_schema": t.get("inputSchema") or EMPTY}
        for t in tools
    ]
    turns: list[dict] = [{"role": "user", "content": instruction}]
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    for _ in range(args.max_turns):
        body = {"model": args.model, "max_tokens": 1024, "system": SYSTEM, "messages": turns, "tools": specs}
        message = post(f"{args.base_url}/messages", body, headers, args.stream)
        turns.append({"role": "assistant", "content": message["content"]})
        calls = [block for block in message["content"] if block.get("type") == "tool_use"]
        if not calls:
            return "".join(block.get("text", "") for block in message["content"])
        results, final = [], None
        for call in calls:
            result = mcp.call_tool(call["name"], call.get("input") or {})
            results.append({"type": "tool_result", "tool_use_id": call["id"], "content": mcp_result_text(result)})
            if mcp_finished(result):
                final = mcp_result_text(result)
        turns.append({"role": "user", "content": results})
        if final is not None:
            return final
    return None


def play_responses(
    args: argparse.Namespace, key: str, mcp: McpClient, tools: list[dict], instruction: str
) -> str | None:
    specs = [
        {
            "type": "function",
            "name": t["name"],
            "description": t.get("description") or "",
            "parameters": t.get("inputSchema") or EMPTY,
        }
        for t in tools
    ]
    items: list[dict] = [{"role": "user", "content": instruction}]
    headers = {"Authorization": f"Bearer {key}"}
    for _ in range(args.max_turns):
        body = {"model": args.model, "instructions": SYSTEM, "input": items, "tools": specs, "store": False}
        response = post(f"{args.base_url}/responses", body, headers, args.stream)
        items.extend(response["output"])
        calls = [item for item in response["output"] if item.get("type") == "function_call"]
        if not calls:
            return "".join(
                part.get("text", "")
                for item in response["output"]
                if item.get("type") == "message"
                for part in item.get("content") or []
            )
        final = None
        for call in calls:
            result = mcp.call_tool(call["name"], json.loads(call.get("arguments") or "{}"))
            items.append(
                {"type": "function_call_output", "call_id": call["call_id"], "output": mcp_result_text(result)}
            )
            if mcp_finished(result):
                final = mcp_result_text(result)
        if final is not None:
            return final
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dialect", required=True, choices=["anthropic_messages", "openai_responses"])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--instruction-file", required=True)
    parser.add_argument("--final-message-file", required=True)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args(argv)
    key = os.environ["PLATOON_API_KEY"]
    with open(args.instruction_file, encoding="utf-8") as handle:
        instruction = handle.read()
    mcp = McpClient(args.mcp_url, key, 600)
    mcp.initialize()
    play = play_messages if args.dialect == "anthropic_messages" else play_responses
    try:
        final = play(args, key, mcp, mcp.list_tools(), instruction)
    except (urllib.error.URLError, RuntimeError, KeyError, ValueError) as exc:
        print(f"dialect-agent: the model endpoint failed: {exc}", file=sys.stderr)
        return 2
    finally:
        mcp.close()
    if final is None:
        return 3
    with open(args.final_message_file, "w", encoding="utf-8") as handle:
        handle.write(final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
