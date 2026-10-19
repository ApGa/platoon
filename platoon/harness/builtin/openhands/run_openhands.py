"""The openhands harness: one OpenHands SDK agent, run headless against platoon's model gateway.

The ``openhands`` spec copies this file into the agent's state directory and runs it with its toolchain's own
interpreter (``{toolchain}/bin/python3 -I``), so it needs the pinned SDK of that toolchain and nothing else.  The
spec's options reach it as flags (``--reasoning-replay``, ``--llm-timeout``, ``--num-retries``, ``--condenser``,
``--condenser-max-tokens``).

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

``--condenser llm_summarizing`` adds a summarizing condenser, :class:`SummarizingCondenser`: the SDK's
``LLMSummarizingCondenser`` with its defaults (it condenses past 240 events, keeping the first 2, the system prompt
and the task, or when the gateway answers that a prompt overflows the context window), changed so that a
condensation keeps the task and its summary call is shown the forgotten events whole (the class says how).  Its
summary calls go through the same gateway with the agent's key, marked ``aux`` with platoon's call-kind header
(:data:`CALL_KIND_HEADER`, which only the condenser's copy of the agent's LLM sends): the gateway records each as one
of the agent's ``aux`` calls, which train as its policy calls do, with the rollout's advantage.  The gateway takes
them from a harness in a task sandbox because no gateway step budget counts this agent's calls: the runner keeps its
own step limit, and OpenHands counts the condensations in it.  A condensation takes one of ``--max-iterations``,
since that step makes the summary call instead of an agent call, and a call the gateway refused for overflowing the
context window took one too.  After a condensation the next prompt no longer extends the last, so it starts a new
training sequence.

``--condenser-max-tokens N``, above 0, gives that condenser a token trigger (its ``max_tokens``) and runs it whatever
``--condenser`` says: before each model call it condenses when the prompt would be longer than N tokens, summarizing
the oldest events after the system prompt and the task until the rest fits in N/2 tokens, so that the next prompt
holds about N/2 tokens plus the summary.  The event count and the overflow fallback still condense too.  Below
twice the agent's first prompt (the system prompt with the tools, and the task), each condensation forgets every
event after the task; a trigger that a prompt passes with nothing left to forget but a summary can never be met, so
the condenser sets it aside for the rest of the run and says so.  The trigger counts each prompt as the gateway
renders it: the agent's LLM (:class:`GatewayCounted`) asks the gateway's token count, ``POST
/v1/messages/count_tokens``, for the messages and tools of the call it would make, sent as a Messages request that
the gateway reads back into the same chat messages (:func:`messages_count_body`), and a gateway that serves a
Tinker policy answers with the length of the prompt it would render with the policy's renderer.  That length is
exact for a renderer that writes tool results without their tool's name and serializes tool-call arguments itself,
as the Qwen chat templates do; one that writes either as the call sent it (the ``kimi_k2`` and ``gpt_oss``
renderers) counts a few tokens off for each tool call.  Where the gateway gives no count (a hosted model's provider
without that route), the SDK's own estimate counts: litellm's token counter, with the ``cl100k_base`` encoding for a
model it does not know, which is below a Qwen chat template's count by about a tenth on prose and code and by nearly
half on numbers (Qwen's tokenizers split numbers into single digits).  A prompt the estimate undercounts can then
pass N, and the overflow fallback still condenses.  Without a token trigger nothing counts tokens, as before.

Model endpoint: ``LLM_MODEL``, ``LLM_BASE_URL`` and ``LLM_API_KEY`` (the agent key).  Exit codes: 0 the agent
finished, 3 it reached ``--max-iterations``, 4 the SDK found it stuck in a loop (both policy outcomes, which the
spec grades), 1 anything else.  The first line it prints is its settings, as ``openhands: settings {json}``.
"""

from __future__ import annotations

import argparse
import dataclasses
import http.client
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from typing import TYPE_CHECKING, ClassVar

from openhands.sdk import LLM, Agent, AgentContext, Conversation, TextContent, Tool
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.context.condenser.base import NoCondensationAvailableException
from openhands.sdk.context.condenser.llm_summarizing_condenser import Reason
from openhands.sdk.event.condenser import Condensation, CondensationSummaryEvent
from openhands.sdk.utils import maybe_truncate
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool
from pydantic import PrivateAttr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openhands.sdk import Message, ToolDefinition

EXIT_FINISHED, EXIT_ERROR, EXIT_ITERATION_LIMIT, EXIT_STUCK = 0, 1, 3, 4
CONDENSERS = ("none", "llm_summarizing")
# platoon's call-kind header (``platoon.harness.agent.CALL_KIND_HEADER``: the runner imports nothing of platoon), with
# which the condenser marks its summary calls ``aux``.
CALL_KIND_HEADER = "X-Platoon-Call-Kind"
# The gateway's token count, under the agent's base URL (LLM_BASE_URL ends in /v1), and how long one count may take.
COUNT_TOKENS_PATH = "/messages/count_tokens"
COUNT_TOKENS_TIMEOUT_S = 60.0
# Answers that say the gateway gives no token count at all, so the run stops asking.
NO_COUNT_STATUSES = (404, 405, 501)
# What the agent reads in place of a summary when a condensation dropped events without one (SummarizingCondenser).
NO_SUMMARY = (
    "Earlier turns of this conversation were removed to keep it within the context window, and no summary of them "
    "could be made. Check the current state of your work before you go on."
)


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


def messages_count_body(messages: list[dict], tools: list[dict]) -> dict | None:
    """A Messages ``count_tokens`` body for a chat call's ``messages`` and ``tools`` (as the SDK formats them), which
    the gateway's Messages dialect reads back into the same chat messages and tools: the system message as
    ``system``, an assistant turn's reasoning, text and tool calls as ``thinking``, ``text`` and ``tool_use`` blocks,
    and each tool result and user message as a user turn of its own.  The dialect gives tools back in the order the
    SDK writes them (``type``, then ``function`` with ``name``, ``description`` and ``parameters``), which matters
    because a chat template writes tools out as JSON.  None for a call it cannot carry: a system message after the
    first, content that is not text, a role it does not know, or tool arguments that are not a JSON object."""

    def text_blocks(content: object) -> list[dict] | None:
        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, list) and all(isinstance(part, dict) and part.get("type") == "text" for part in content):
            return [{"type": "text", "text": part.get("text") or ""} for part in content]
        return None  # an image or another part the Messages count would read otherwise

    system: list[dict] = []
    turns: list[dict] = []
    for index, message in enumerate(messages):
        role, texts = message.get("role"), text_blocks(message.get("content"))
        if texts is None:
            return None
        if role == "system" and index == 0:
            system = texts
        elif role == "user":
            turns.append({"role": "user", "content": texts or [{"type": "text", "text": ""}]})
        elif role == "tool":
            result = {"type": "tool_result", "tool_use_id": message.get("tool_call_id"), "content": texts}
            turns.append({"role": "user", "content": [result]})
        elif role == "assistant":
            blocks = texts
            if isinstance(message.get("reasoning_content"), str):
                blocks = [{"type": "thinking", "thinking": message["reasoning_content"]}, *blocks]
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except (TypeError, ValueError):
                    return None
                if not isinstance(arguments, dict):
                    return None
                name = function.get("name")
                blocks.append({"type": "tool_use", "id": call.get("id"), "name": name, "input": arguments})
            turns.append({"role": "assistant", "content": blocks})
        else:
            return None
    if not turns:
        return None
    body: dict = {"messages": turns}
    if system:
        body["system"] = system
    functions = [tool.get("function", tool) for tool in tools]
    if functions:
        body["tools"] = [
            {"name": f.get("name"), "description": f.get("description"), "input_schema": f.get("parameters")}
            for f in functions
        ]
    return body


class GatewayTokenCount:
    """The model gateway's count of a chat call's prompt (``POST {base_url}/messages/count_tokens``), or None where
    it gives none: a call :func:`messages_count_body` cannot carry, an answer other than a count, or any call after
    the gateway answered that it counts none (:data:`NO_COUNT_STATUSES`, as a hosted model's provider without the
    route answers).  The runner reports the first call it gives no count for, and that it counts none.  The last
    count is kept, since the condenser counts one prompt more than once when it condenses."""

    def __init__(self, base_url: str, api_key: str | None, *, timeout_s: float = COUNT_TOKENS_TIMEOUT_S) -> None:
        self.url = base_url.rstrip("/") + COUNT_TOKENS_PATH
        self.headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
        if api_key:
            self.headers["authorization"] = f"Bearer {api_key}"
        self.timeout_s = timeout_s
        self.unsupported: str | None = None  # why the gateway counts none, once it said so
        self.uncounted: str | None = None  # why it gave no count for the first call it gave none for
        self._last: tuple[str, int] | None = None

    def count(self, messages: list[dict], tools: list[dict]) -> int | None:
        if self.unsupported is not None:
            return None
        body = messages_count_body(messages, tools)
        if body is None:
            return self._uncounted("the Messages count cannot carry the call")
        payload = json.dumps(body)
        if self._last is not None and self._last[0] == payload:
            return self._last[1]
        request = urllib.request.Request(self.url, data=payload.encode(), headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                status, answer = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, answer = error.code, b""
        except (OSError, http.client.HTTPException) as error:  # unreachable, reset or timed out (URLError: OSError)
            return self._uncounted(type(error).__name__)
        if status in NO_COUNT_STATUSES:
            self.unsupported = f"HTTP {status}"
            print(
                f"openhands: the gateway counts no tokens ({self.unsupported}), so the condenser counts with the "
                "SDK's estimate",
                flush=True,
            )
            return None
        try:
            tokens = json.loads(answer)["input_tokens"] if status == 200 else None
        except (ValueError, KeyError, TypeError):
            tokens = None
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            return self._uncounted(f"HTTP {status}" if status != 200 else "an answer that is no count")
        self._last = (payload, tokens)
        return tokens

    def _uncounted(self, why: str) -> None:
        if self.uncounted is None:
            self.uncounted = why
            print(
                f"openhands: the gateway gave no token count for a call ({why}), so the SDK's estimate counts it; "
                "later calls without a count are not reported",
                flush=True,
            )


_GATEWAY_COUNTS: dict[str, GatewayTokenCount] = {}


def gateway_token_count(llm: LLM) -> GatewayTokenCount:
    """The gateway count for ``llm``'s own endpoint and key: one per base URL, which the LLM's copies share."""

    url = llm.base_url or ""
    if url not in _GATEWAY_COUNTS:
        secret = llm.api_key
        api_key = secret.get_secret_value() if hasattr(secret, "get_secret_value") else secret
        _GATEWAY_COUNTS[url] = GatewayTokenCount(url, api_key)
    return _GATEWAY_COUNTS[url]


class GatewayCounted(LLM):
    """The agent's LLM when the condenser has a token trigger: :meth:`get_token_count`, which the condenser counts
    the agent's prompts with, is the gateway's count of the call the LLM would make with the same messages and tools,
    and the SDK's own estimate where the gateway gives none (see the module docstring)."""

    failure_reported: ClassVar[bool] = False  # a count that raised is reported once, with its traceback

    def get_token_count(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None = None,
        add_security_risk_prediction: bool = False,
    ) -> int:
        risk = add_security_risk_prediction
        try:
            functions = [tool.to_openai_tool(add_security_risk_prediction=risk) for tool in tools or []]
            if not self.should_mock_tool_calls(functions):
                counted = gateway_token_count(self).count(self.format_messages_for_llm(messages), functions)
                if counted is not None:
                    return counted
        except Exception:  # a count never stops the agent: the estimate takes over
            if not GatewayCounted.failure_reported:
                GatewayCounted.failure_reported = True
                print("openhands: a token count at the gateway failed; the SDK's estimate counts instead:", flush=True)
                traceback.print_exc(file=sys.stdout)
        return super().get_token_count(messages, tools, add_security_risk_prediction)


class CountedLLM(GatewayCounted):
    """The SDK's LLM, counting prompts at the gateway."""


class CountedReplayLLM(GatewayCounted, ReplayLLM):
    """:class:`ReplayLLM`, counting prompts at the gateway."""


def summary_input(event) -> str:
    """How the summary call reads one forgotten event: whole, as the message it puts in the agent's prompt but
    without its reasoning, that is its kind, a tool result's tool, its text and each tool call with its arguments.
    The SDK's own rendering (``str(event)``) shows the first 500 characters of each text and names a tool call's
    action without its arguments.  An event that is no message reads as the SDK renders it."""

    try:
        message = event.to_llm_message()
    except Exception:  # no message of its own
        return str(event)
    lines = [f"{type(event).__name__} ({event.source})"]
    if message.role == "tool" and message.name:
        lines.append(f"  Tool: {message.name}")
    text = "\n".join(
        item.text if isinstance(item, TextContent) else f"[{type(item).__name__}]" for item in message.content
    )
    if text:
        lines.append(f"  {message.role}: {text}")
    for call in message.tool_calls or []:
        lines.append(f"  Call: {call.name} {call.arguments}")
    return "\n".join(lines)


def event_length_cap(lengths: list[int], budget: float) -> int:
    """The longest any one of events of ``lengths`` characters may be for all of them to take at most ``budget``
    characters, at least 1."""

    low, high = 1, max(lengths, default=1)
    while low < high:
        middle = (low + high + 1) // 2
        if sum(min(length, middle) for length in lengths) <= budget:
            low = middle
        else:
            high = middle - 1
    return low


class SummarizingCondenser(LLMSummarizingCondenser):
    """The SDK's ``LLMSummarizingCondenser``, changed so that a condensation keeps what the agent needs and its
    summary call, which trains, is shown what happened:

    * The summary call reads each forgotten event whole (:func:`summary_input`), tool calls with their arguments and
      an earlier summary entire, where the SDK shows it the first 500 characters of each text, a tool call's action
      without its arguments, and so of an earlier summary only its start.
    * A condensation for the token trigger forgets the events it takes to bring the prompt under half the trigger,
      however few they are.  The SDK also asks it to forget a tenth of the events (``minimum_progress``) and, when
      that would forget fewer, resets the whole context: a few long early events then cost the agent its task.
    * A hard reset keeps the task: it summarizes every event after the system prompt and the task (the first
      ``keep_first``), where the SDK's keeps the system prompt only (with nothing after the task, it resets as the
      SDK does).  A summary call that fails, for example because the gateway refused a prompt that overflows the
      context window, is made again with the events cut, their length by a fifth each time (each keeps its head and
      tail); if none succeeds the events go without a summary (:data:`NO_SUMMARY`), where the SDK stops the agent.
    * A summary call answered without text (with a tool call, or a completion cut short in its reasoning) still
      condenses, with :data:`NO_SUMMARY` in its place, where the SDK drops the forgotten events and says nothing.

    A token trigger that a prompt passes with nothing left to forget, that is the system prompt and the task alone
    or with summaries of the rest, can never be met: the agent would condense at every step.  The condenser sets it
    aside for the rest of the run, and the runner says so; the event count and the overflow fallback still condense."""

    _trigger_set_aside: bool = PrivateAttr(default=False)

    def _effective_max_tokens(self, agent_llm):
        if self._trigger_set_aside:  # the agent LLM's own input limit only, as without a trigger
            return agent_llm.effective_max_input_tokens if agent_llm is not None else None
        return super()._effective_max_tokens(agent_llm)

    def get_condensation_reasons(self, view, agent_llm=None):
        reasons = super().get_condensation_reasons(view, agent_llm)
        if Reason.TOKENS in reasons and self.max_tokens and not self._trigger_set_aside:
            if all(isinstance(event, CondensationSummaryEvent) for event in view.events[self.keep_first :]):
                self._trigger_set_aside = True
                print(
                    f"openhands: the condenser's token trigger ({self.max_tokens} tokens) is set aside for the rest of "
                    "the run: a prompt passes it with nothing left to forget but the system prompt, the task and a "
                    "summary",
                    flush=True,
                )
                reasons = super().get_condensation_reasons(view, agent_llm)
        return reasons

    def get_condensation(self, view, agent_llm=None) -> Condensation:
        reasons = self.get_condensation_reasons(view, agent_llm)
        try:
            forgotten, offset = self._get_forgotten_events(view, agent_llm=agent_llm)
        except ValueError as error:
            raise NoCondensationAvailableException("Unable to compute forgotten events") from error
        if not forgotten:
            raise NoCondensationAvailableException("Cannot condense 0 events")
        if Reason.TOKENS not in reasons and len(forgotten) < len(view) * self.minimum_progress:
            raise NoCondensationAvailableException("Events forgotten below minimum progress threshold")
        return self._generate_condensation(forgotten_events=forgotten, summary_offset=offset)

    def hard_context_reset(self, view, agent_llm=None) -> Condensation | None:
        try:
            start = view.manipulation_indices.find_next(self.keep_first)
        except ValueError:
            start = len(view)
        forgotten = view[start:]
        if not forgotten:  # the system prompt and the task alone overflow: the SDK's reset, which summarizes the task
            return super().hard_context_reset(view, agent_llm)
        lengths = [len(summary_input(event)) for event in forgotten]
        cap = None
        for attempt in range(1, self.hard_context_reset_max_retries + 1):
            try:
                return self._generate_condensation(forgotten, start, max_event_str_length=cap)
            except Exception as error:
                print(f"openhands: the summary call of a context reset failed: {error}", flush=True)
                cap = event_length_cap(lengths, sum(lengths) * self.hard_context_reset_context_scaling**attempt)
        print(f"openhands: no summary of {len(forgotten)} events could be made, so they go without one", flush=True)
        return Condensation(
            forgotten_event_ids={event.id for event in forgotten},
            summary=NO_SUMMARY,
            summary_offset=start,
            llm_response_id=f"no-summary-{uuid.uuid4().hex}",
        )

    async def aget_condensation(self, view, agent_llm=None) -> Condensation:
        return self.get_condensation(view, agent_llm)

    async def ahard_context_reset(self, view, agent_llm=None) -> Condensation | None:
        return self.hard_context_reset(view, agent_llm)

    def _generate_condensation(self, forgotten_events, summary_offset, max_event_str_length=None) -> Condensation:
        """The SDK's summary call, on the :func:`summary_input` of each event (cut to ``max_event_str_length``),
        with :data:`NO_SUMMARY` for an answer without text."""

        inputs = [
            maybe_truncate(summary_input(event), truncate_after=max_event_str_length) for event in forgotten_events
        ]
        try:
            response = self.llm.generate(messages=self._build_summary_messages(inputs), store=False)
        except Exception as error:
            raise NoCondensationAvailableException(f"Summarization LLM call failed: {error}") from error
        summary = "\n".join(item.text for item in response.message.content if isinstance(item, TextContent))
        if not summary.strip():
            print(
                f"openhands: the summary call answered without text, so {len(forgotten_events)} events go without "
                "a summary",
                flush=True,
            )
            summary = NO_SUMMARY
        return Condensation(
            forgotten_event_ids={event.id for event in forgotten_events},
            summary=summary,
            summary_offset=summary_offset,
            llm_response_id=response.id,
        )


def build_condenser(llm: LLM, kind: str, max_tokens: int = 0) -> SummarizingCondenser | None:
    """The condenser ``--condenser`` and ``--condenser-max-tokens`` name: None, or :class:`SummarizingCondenser`
    with the SDK's defaults and, for ``max_tokens`` above 0, that token trigger.  Its summary calls go through a copy
    of ``llm`` (for the SDK's usage accounting) that marks them ``aux`` (:data:`CALL_KIND_HEADER`); ``llm`` itself, the
    agent's, sends no call kind, so the agent's own calls stay ``policy`` calls."""

    if kind == "none" and not max_tokens:
        return None
    trigger = {"max_tokens": max_tokens} if max_tokens else {}
    summarizer = llm.model_copy(update={"usage_id": "condenser", "extra_headers": {CALL_KIND_HEADER: "aux"}})
    return SummarizingCondenser(llm=summarizer, **trigger)


def build_agent(
    llm_timeout_s: int,
    *,
    replay: bool = True,
    num_retries: int = 0,
    condenser: str = "none",
    condenser_max_tokens: int = 0,
) -> Agent:
    if condenser_max_tokens:  # the trigger counts prompts as the gateway renders them
        llm_class = CountedReplayLLM if replay else CountedLLM
    else:
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
    return Agent(
        llm=llm,
        tools=tools,
        agent_context=context,
        condenser=build_condenser(llm, condenser, condenser_max_tokens),
    )


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
    parser.add_argument(
        "--condenser-max-tokens",
        type=int,
        default=0,
        help="condense when the next prompt would be longer than this many tokens, as the gateway counts them "
        "(runs the summarizing condenser); 0: no token trigger",
    )
    args = parser.parse_args(argv)
    if args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1")
    if args.llm_timeout < 1 or args.num_retries < 0:
        parser.error("--llm-timeout must be at least 1 and --num-retries at least 0")
    if args.condenser_max_tokens < 0:
        parser.error("--condenser-max-tokens must be at least 0")
    return args


def settings(args: argparse.Namespace) -> dict[str, object]:
    """What the agent runs with, as its first line of output reports it: the condenser that runs, which a token
    trigger makes the summarizing one whatever ``--condenser`` says (:func:`build_condenser`)."""

    return {
        "max_iterations": args.max_iterations,
        "reasoning_replay": args.reasoning_replay,
        "llm_timeout": args.llm_timeout,
        "num_retries": args.num_retries,
        "condenser": "llm_summarizing" if args.condenser_max_tokens else args.condenser,
        "condenser_max_tokens": args.condenser_max_tokens,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("openhands: settings " + json.dumps(settings(args)), flush=True)
    with open(args.instruction_file, encoding="utf-8") as handle:
        instruction = handle.read()

    started = time.monotonic()
    conversation = Conversation(
        agent=build_agent(
            args.llm_timeout,
            replay=args.reasoning_replay,
            num_retries=args.num_retries,
            condenser=args.condenser,
            condenser_max_tokens=args.condenser_max_tokens,
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
