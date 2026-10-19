"""The openhands harness: one OpenHands SDK agent, run headless against platoon's model gateway.

The ``openhands`` spec copies this file into the agent's state directory and runs it with its toolchain's own
interpreter (``{toolchain}/bin/python3 -I``), so it needs the pinned SDK of that toolchain and nothing else.  The
spec's options reach it as flags (``--reasoning-replay``, ``--llm-timeout``, ``--num-retries``, ``--condenser``).

The agent is the one Harbor's ``openhands-sdk`` adapter evaluates: the Terminal, FileEditor and TaskTracker tools
plus the SDK's own finish and think tools, the SDK's default system prompt, and the task's working directory as its
workspace.  With the spec's defaults, its training settings, it differs in five ways:

* no condenser, no skills, no security analyzer and no delegation, so every model call is one policy turn;
* ``LLM(timeout=3600, num_retries=0)``: one attempt per call, as a retry would send a call the gateway may already
  have recorded.  The SDK's ``num_retries`` counts attempts at a call that fails transiently, so 0 and 1 both make
  one, and 5 up to five; litellm itself makes no retries (the SDK passes it ``max_retries=0``);
* no sampling parameters of its own: the rollout's ``inference_params`` apply;
* reasoning replay.  Each assistant turn goes back to the gateway exactly as it was sampled: with its
  ``reasoning_content``, its text as one string, and whitespace-only text kept on tool-call turns.  The next
  prompt then extends the previous prompt and completion, so the calls of one episode merge into few training
  sequences.  The SDK does this for a fixed list of models only, so :class:`ReplayLLM` overrides two private
  methods; that is one reason the SDK version is pinned;
* ``--max-iterations`` is the step limit: one iteration is one model call of the agent (with a condenser, a
  condensation takes one too, below).

``--reasoning-replay false``, ``--llm-timeout 300`` and ``--num-retries 5`` run it as Harbor's adapter evaluates it:
without replay, with the SDK's default request timeout and retries, and with the SDK's own serialization of the
messages (their text parts as lists, where replay sends one string), so each prompt renders as the adapter's does.
A retry can then record a call twice, which an evaluation does not train on.

``--condenser llm_summarizing`` adds the SDK's ``LLMSummarizingCondenser`` (its defaults: it condenses past 240
events, keeping the first 2, or when the gateway answers that a prompt overflows the context window).  Its
summaries are model calls through the same gateway with the agent's key, so they are recorded as the agent's own
policy calls and trained on.  The SDK could mark them with platoon's call-kind header (``LLM.extra_headers``), but
the gateway refuses ``aux`` calls from an agent limited to policy calls, as every harness in a task sandbox is, so
they go unmarked.  A condensation takes one of ``--max-iterations``: that step makes the summary call instead of an
agent call, and a call the gateway refused for overflowing the context window took one too.  After a condensation
the next prompt no longer extends the last, so it starts a new training sequence.

Model endpoint: ``LLM_MODEL``, ``LLM_BASE_URL`` and ``LLM_API_KEY`` (the agent key).  Exit codes: 0 the agent
finished, 3 it reached ``--max-iterations``, 4 the SDK found it stuck in a loop (both policy outcomes, which the
spec grades), 1 anything else.  The first line it prints is its settings, as ``openhands: settings {json}``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import traceback

from openhands.sdk import LLM, Agent, AgentContext, Conversation, TextContent, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

EXIT_FINISHED, EXIT_ERROR, EXIT_ITERATION_LIMIT, EXIT_STUCK = 0, 1, 3, 4
CONDENSERS = ("none", "llm_summarizing")


class ReplayLLM(LLM):
    """An LLM that sends each assistant turn back as it was sampled (see the module docstring)."""

    def _model_features(self):  # type: ignore[override]
        return dataclasses.replace(super()._model_features(), send_reasoning_content=True)

    def _to_chat_dicts(self, messages):  # type: ignore[override]
        wire = super()._to_chat_dicts(messages)
        for message, serialized in zip(messages, wire, strict=True):
            if message.role != "assistant":
                continue
            texts = [item.text for item in message.content if isinstance(item, TextContent)]
            if texts:  # the SDK drops whitespace-only text from tool-call turns; it is part of the sampled prefix
                serialized["content"] = "\n".join(texts)
        return wire


def build_condenser(llm: LLM, kind: str):
    """The condenser ``--condenser`` names: None, or the SDK's summarizing condenser with its own defaults, whose
    calls go through ``llm`` (a copy, for the SDK's usage accounting)."""

    if kind == "none":
        return None
    from openhands.sdk.context.condenser import LLMSummarizingCondenser

    return LLMSummarizingCondenser(llm=llm.model_copy(update={"usage_id": "condenser"}))


def build_agent(llm_timeout_s: int, *, replay: bool = True, num_retries: int = 0, condenser: str = "none") -> Agent:
    llm_class = ReplayLLM if replay else LLM
    llm = llm_class(
        model=os.environ["LLM_MODEL"],
        base_url=os.environ["LLM_BASE_URL"],
        # Out of the environment the agent's terminal inherits; the policy has no use for its own key.
        api_key=os.environ.pop("LLM_API_KEY"),
        usage_id="agent",
        timeout=llm_timeout_s,
        num_retries=num_retries,
        # Replay sends each turn as text, as it was sampled: list content would change the prefix.  Without replay,
        # the SDK serializes the messages as it does for Harbor's adapter (text parts as a list).
        force_string_serializer=True if replay else None,
    )
    tools = [Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name), Tool(name=TaskTrackerTool.name)]
    context = AgentContext(skills=[], load_user_skills=False, load_public_skills=False)
    return Agent(llm=llm, tools=tools, agent_context=context, condenser=build_condenser(llm, condenser))


def final_message(events) -> str | None:
    """The finish tool's message, else the agent's last plain message (a budget stop answers with one)."""

    for event in reversed(events):
        action = getattr(event, "action", None)
        if action is not None and type(action).__name__ == "FinishAction":
            return action.message
    for event in reversed(events):
        message = getattr(event, "llm_message", None)
        if getattr(event, "source", None) == "agent" and message is not None:
            texts = [item.text for item in message.content if isinstance(item, TextContent)]
            if texts:
                return "\n".join(texts)
    return None


def exit_code(status: str, events) -> int:
    if status == "finished":
        return EXIT_FINISHED
    if status == "stuck":
        return EXIT_STUCK
    for event in reversed(events):
        if type(event).__name__ == "ConversationErrorEvent" and getattr(event, "code", None) == "MaxIterationsReached":
            return EXIT_ITERATION_LIMIT
    return EXIT_ERROR


def _boolean(text: str) -> bool:
    """``true`` or ``false``, as the spec renders a boolean option."""

    if text not in ("true", "false"):
        raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")
    return text == "true"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--instruction-file", required=True)
    parser.add_argument("--max-iterations", type=int, required=True, help="model calls the agent may make")
    parser.add_argument("--final-message-file", required=True)
    parser.add_argument(
        "--reasoning-replay",
        type=_boolean,
        default=True,
        metavar="{true,false}",
        help="send each assistant turn back with its reasoning (false: as the SDK serializes it, without)",
    )
    parser.add_argument("--llm-timeout", type=int, default=3600, help="seconds one model call may take")
    parser.add_argument(
        "--num-retries",
        type=int,
        default=0,
        help="the SDK's num_retries: the most attempts at a model call that fails transiently (0 and 1 make one)",
    )
    parser.add_argument("--condenser", choices=CONDENSERS, default="none", help="the SDK's context condenser")
    args = parser.parse_args(argv)
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")
    if args.llm_timeout < 1 or args.num_retries < 0:
        parser.error("--llm-timeout must be at least 1 and --num-retries at least 0")
    return args


def settings(args: argparse.Namespace) -> dict[str, object]:
    """What the agent runs with, as its first line of output reports it."""

    return {
        "max_iterations": args.max_iterations,
        "reasoning_replay": args.reasoning_replay,
        "llm_timeout": args.llm_timeout,
        "num_retries": args.num_retries,
        "condenser": args.condenser,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("openhands: settings " + json.dumps(settings(args)), flush=True)
    with open(args.instruction_file, encoding="utf-8") as handle:
        instruction = handle.read()

    started = time.monotonic()
    conversation = Conversation(
        agent=build_agent(
            args.llm_timeout, replay=args.reasoning_replay, num_retries=args.num_retries, condenser=args.condenser
        ),
        workspace=os.getcwd(),
        max_iteration_per_run=args.max_iterations,
    )
    error = None
    try:
        conversation.send_message(instruction)
        conversation.run()
    except Exception as exc:  # the SDK raises ConversationRunError for a failed model call or tool
        traceback.print_exc()
        error = f"{type(exc).__name__}: {exc}"
    events = list(conversation.state.events)
    status = str(getattr(conversation.state.execution_status, "value", conversation.state.execution_status))
    code = EXIT_ERROR if error else exit_code(status, events)
    final = final_message(events)
    with open(args.final_message_file, "w", encoding="utf-8") as handle:
        handle.write(final or "")
    summary = {"status": status, "exit_code": code, "error": error, "seconds": round(time.monotonic() - started, 1)}
    print("openhands: " + json.dumps(summary), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
