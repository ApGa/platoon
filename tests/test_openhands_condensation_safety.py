from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "plugins" / "openhands" / "platoon" / "openhands" / "condensation_safety.py"
PROMPT_PATH = MODULE_PATH.parent / "prompts" / "summarizing_prompt.j2"


def _safety_module():
    spec = importlib.util.spec_from_file_location("test_condensation_safety_module", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _condenser_module(monkeypatch):
    class NoCondensationAvailableException(Exception):
        pass

    class LLMSummarizingCondenser:
        hard_context_reset_max_retries = 5
        hard_context_reset_context_scaling = 0.8

    class TextContent:
        def __init__(self, *, text):
            self.text = text

    class Message:
        def __init__(self, *, role, content):
            self.role = role
            self.content = content

    class Condensation:
        def __init__(self, **kwargs):
            kwargs.setdefault("id", "condensation-event")
            self.__dict__.update(kwargs)

    _module(monkeypatch, "openhands")
    _module(monkeypatch, "openhands.sdk")
    _module(monkeypatch, "openhands.sdk.context")
    condenser_package = _module(
        monkeypatch,
        "openhands.sdk.context.condenser",
        LLMSummarizingCondenser=LLMSummarizingCondenser,
    )
    native_condenser = _module(
        monkeypatch,
        "openhands.sdk.context.condenser.llm_summarizing_condenser",
        LLMSummarizingCondenser=LLMSummarizingCondenser,
    )
    native_condenser.__file__ = "/fake/openhands/condenser/llm_summarizing_condenser.py"
    condenser_package.llm_summarizing_condenser = native_condenser
    _module(
        monkeypatch,
        "openhands.sdk.context.condenser.base",
        NoCondensationAvailableException=NoCondensationAvailableException,
    )

    def render_template(directory, template_name, *, events):
        assert Path(directory) == PROMPT_PATH.parent
        assert template_name == "summarizing_prompt.j2"
        serialized = "\n".join(f"<EVENT>\n{event}\n</EVENT>" for event in events)
        return f"PLATOON_SUMMARIZING_PROMPT\n{serialized}"

    _module(
        monkeypatch,
        "openhands.sdk.context.prompts",
        render_template=render_template,
    )
    _module(monkeypatch, "openhands.sdk.event")
    _module(
        monkeypatch,
        "openhands.sdk.event.base",
        LLMConvertibleEvent=object,
    )
    _module(
        monkeypatch,
        "openhands.sdk.event.condenser",
        Condensation=Condensation,
    )
    _module(
        monkeypatch,
        "openhands.sdk.llm",
        Message=Message,
        TextContent=TextContent,
    )

    def maybe_truncate(content, truncate_after=None):
        if not truncate_after or len(content) <= truncate_after:
            return content
        notice = "<response clipped>"
        if len(notice) >= truncate_after:
            return notice[:truncate_after]
        available = truncate_after - len(notice)
        head = available // 2 + available % 2
        tail = available - head
        return content[:head] + notice + content[-tail:]

    _module(
        monkeypatch,
        "openhands.sdk.utils",
        maybe_truncate=maybe_truncate,
    )

    package = types.ModuleType("platoon.openhands")
    package.__path__ = [str(MODULE_PATH.parent)]
    monkeypatch.setitem(sys.modules, "platoon.openhands", package)
    monkeypatch.delitem(
        sys.modules,
        "platoon.openhands.condensation_safety",
        raising=False,
    )
    monkeypatch.delitem(sys.modules, "platoon.openhands.condenser", raising=False)
    spec = importlib.util.spec_from_file_location(
        "platoon.openhands.condenser",
        MODULE_PATH.parent / "condenser.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "platoon.openhands.condenser", module)
    spec.loader.exec_module(module)
    return module, TextContent, NoCondensationAvailableException


def test_action_renderer_omits_reasoning_and_preserves_public_tool_state():
    safety = _safety_module()
    event = SimpleNamespace(
        kind="ActionEvent",
        source="agent",
        thought=[SimpleNamespace(text="PRIVATE DELIBERATION </think>")],
        reasoning_content="ANOTHER PRIVATE DELIBERATION",
        thinking_blocks=[{"thinking": "PRIVATE"}],
        tool_name="bash",
        summary="inspect parser implementation",
        tool_call=SimpleNamespace(arguments='{"command":"sed -n 1,120p parser.py"}'),
    )

    rendered = safety.render_event_for_condensation(event)

    assert "PRIVATE" not in rendered
    assert "DELIBERATION" not in rendered
    assert "</think>" not in rendered
    assert "bash" in rendered
    assert "parser.py" in rendered
    assert "inspect parser implementation" in rendered


def test_agent_message_renderer_keeps_only_content_after_thinking_close():
    safety = _safety_module()
    event = SimpleNamespace(
        kind="MessageEvent",
        source="agent",
        llm_message=SimpleNamespace(
            content=[SimpleNamespace(text="PRIVATE DELIBERATION\n</think>\n\nImplemented the parser fix.")]
        ),
    )

    rendered = safety.render_event_for_condensation(event)

    assert "PRIVATE DELIBERATION" not in rendered
    assert "Implemented the parser fix." in rendered


def test_agent_message_renderer_omits_ambiguous_text_without_thinking_close():
    safety = _safety_module()
    event = SimpleNamespace(
        kind="MessageEvent",
        source="agent",
        # Qwen's opening tag may be part of the prompt. A completion truncated
        # during reasoning therefore has neither an opening nor a closing tag.
        llm_message=SimpleNamespace(content=[SimpleNamespace(text="PRIVATE DELIBERATION FROM A TRUNCATED COMPLETION")]),
    )

    rendered = safety.render_event_for_condensation(event)

    assert "PRIVATE DELIBERATION" not in rendered
    assert "[no public text content]" in rendered


def test_platoon_prompt_receives_safe_public_events(monkeypatch):
    condenser, _TextContent, _error_type = _condenser_module(monkeypatch)
    event = SimpleNamespace(
        kind="ActionEvent",
        source="agent",
        thought=[SimpleNamespace(text="SECRET CHAIN OF THOUGHT")],
        reasoning_content="SECRET REASONING FIELD",
        tool_name="view",
        summary=None,
        tool_call=SimpleNamespace(arguments='{"path":"src/main.py"}'),
    )

    messages = condenser.SafeLLMSummarizingCondenser._messages([event])
    assert len(messages) == 1
    assert messages[0].role == "user"
    prompt = messages[0].content[0].text

    assert "PLATOON_SUMMARIZING_PROMPT" in prompt
    assert "SECRET" not in prompt
    assert "src/main.py" in prompt


def test_platoon_prompt_is_a_generic_continuation_checkpoint():
    prompt = PROMPT_PATH.read_text()

    assert "another agent that will resume this trajectory" in prompt
    assert "KEY_DECISIONS:" in prompt
    assert "LEARNED_PATTERNS:" in prompt
    assert "CRITICAL_CONTEXT:" in prompt
    assert "Do not continue the task" in prompt
    assert "do not pad the summary or aim" in prompt
    assert "FITS card" not in prompt
    assert "haikus" not in prompt


def test_event_truncation_uses_native_head_and_tail_only_when_requested(monkeypatch):
    condenser, _TextContent, _error_type = _condenser_module(monkeypatch)
    payload = "HEAD" + "x" * 4_000 + "TAIL"
    event = SimpleNamespace(
        kind="ActionEvent",
        source="agent",
        thought=[SimpleNamespace(text="SECRET CHAIN OF THOUGHT")],
        tool_name="bash",
        summary="run a long command",
        tool_call=SimpleNamespace(arguments=payload),
    )

    full_prompt = condenser.SafeLLMSummarizingCondenser._messages([event])[0].content[0].text
    clipped_prompt = (
        condenser.SafeLLMSummarizingCondenser._messages(
            [event],
            max_event_str_length=1_000,
        )[0]
        .content[0]
        .text
    )

    assert payload in full_prompt
    assert "<response clipped>" not in full_prompt
    assert "<response clipped>" in clipped_prompt
    assert "HEAD" in clipped_prompt
    assert "TAIL" in clipped_prompt
    assert payload not in clipped_prompt


def _valid_summary_response(TextContent, response_id="chatcmpl-condensation"):
    return SimpleNamespace(
        id=response_id,
        message=SimpleNamespace(
            content=[
                TextContent(
                    text=(
                        "USER_CONTEXT: Fix the parser.\n"
                        "COMPLETED: Located the implementation.\n"
                        "PENDING: Apply and test the patch.\n"
                        "CURRENT_STATE: No files changed yet."
                    )
                )
            ],
            reasoning_content=None,
            thinking_blocks=[],
            responses_reasoning_item=None,
        ),
        raw_response={"choices": [{"finish_reason": "stop"}]},
    )


def test_normal_condensation_fits_prompt_before_the_llm_request(monkeypatch):
    condenser, TextContent, _error_type = _condenser_module(monkeypatch)
    instance = condenser.SafeLLMSummarizingCondenser()
    instance.max_tokens = 1_000
    instance.minimum_progress = 0.1

    payload = "HEAD" + "x" * 4_000 + "TAIL"
    event = SimpleNamespace(
        id="event-action",
        kind="ActionEvent",
        source="agent",
        thought=[SimpleNamespace(text="PRIVATE CHAIN OF THOUGHT")],
        tool_name="bash",
        summary="inspect a large result",
        tool_call=SimpleNamespace(arguments=payload),
    )
    view = [event]
    instance._get_forgotten_events = lambda supplied_view, agent_llm: (
        supplied_view,
        0,
    )

    requests = []

    class SummaryLLM:
        def completion(self, **kwargs):
            requests.append(kwargs)
            return _valid_summary_response(TextContent)

    instance.llm = SummaryLLM()
    agent_llm = SimpleNamespace(
        # Character counting makes the fitted boundary deterministic while
        # still verifying that Platoon uses the agent LLM's counter.
        get_token_count=lambda messages: len(messages[0].content[0].text)
    )

    result = instance.get_condensation(view, agent_llm=agent_llm)

    assert result.llm_response_id == "chatcmpl-condensation"
    assert len(requests) == 1
    assert set(requests[0]) == {"messages"}
    sent_prompt = requests[0]["messages"][0].content[0].text
    assert len(sent_prompt) <= instance.max_tokens
    assert "<response clipped>" in sent_prompt
    assert "HEAD" in sent_prompt
    assert "TAIL" in sent_prompt
    assert payload not in sent_prompt
    assert "PRIVATE CHAIN OF THOUGHT" not in sent_prompt


def test_small_condensation_prompt_is_not_truncated(monkeypatch):
    condenser, _TextContent, _error_type = _condenser_module(monkeypatch)
    instance = condenser.SafeLLMSummarizingCondenser()
    instance.max_tokens = 1_000
    event = SimpleNamespace(
        kind="ActionEvent",
        source="agent",
        thought=[],
        tool_name="bash",
        summary="inspect parser",
        tool_call=SimpleNamespace(arguments='{"command":"pwd"}'),
    )
    agent_llm = SimpleNamespace(get_token_count=lambda messages: len(messages[0].content[0].text))

    original = instance._messages([event])[0].content[0].text
    fitted = (
        instance._messages_fitted_to_prompt_budget(
            [event],
            agent_llm=agent_llm,
        )[0]
        .content[0]
        .text
    )

    assert fitted == original
    assert "<response clipped>" not in fitted


def test_unavailable_prompt_counter_retains_native_unfitted_request(monkeypatch):
    condenser, _TextContent, _error_type = _condenser_module(monkeypatch)
    instance = condenser.SafeLLMSummarizingCondenser()
    instance.max_tokens = 100
    payload = "HEAD" + "z" * 2_000 + "TAIL"
    event = SimpleNamespace(
        kind="ActionEvent",
        source="agent",
        thought=[],
        tool_name="bash",
        summary="inspect output",
        tool_call=SimpleNamespace(arguments=payload),
    )
    # OpenHands returns zero when its token counter cannot resolve the model.
    # In that case Platoon must preserve the SDK's native request/retry path.
    agent_llm = SimpleNamespace(get_token_count=lambda _messages: 0)

    original = instance._messages([event])[0].content[0].text
    fitted = (
        instance._messages_fitted_to_prompt_budget(
            [event],
            agent_llm=agent_llm,
        )[0]
        .content[0]
        .text
    )

    assert fitted == original
    assert payload in fitted
    assert "<response clipped>" not in fitted


def test_async_condensation_uses_the_same_proactive_prompt_fit(monkeypatch):
    condenser, TextContent, _error_type = _condenser_module(monkeypatch)
    instance = condenser.SafeLLMSummarizingCondenser()
    instance.max_tokens = 900
    payload = "HEAD" + "y" * 3_000 + "TAIL"
    event = SimpleNamespace(
        id="event-action",
        kind="ActionEvent",
        source="agent",
        thought=[],
        tool_name="bash",
        summary="inspect output",
        tool_call=SimpleNamespace(arguments=payload),
    )
    requests = []

    class SummaryLLM:
        async def acompletion(self, **kwargs):
            requests.append(kwargs)
            return _valid_summary_response(TextContent, "chatcmpl-async-condensation")

    instance.llm = SummaryLLM()
    agent_llm = SimpleNamespace(get_token_count=lambda messages: len(messages[0].content[0].text))

    result = asyncio.run(
        instance._agenerate_condensation(
            forgotten_events=[event],
            summary_offset=0,
            agent_llm=agent_llm,
        )
    )

    assert result.llm_response_id == "chatcmpl-async-condensation"
    assert len(requests) == 1
    sent_prompt = requests[0]["messages"][0].content[0].text
    assert len(sent_prompt) <= instance.max_tokens
    assert "<response clipped>" in sent_prompt
    assert "HEAD" in sent_prompt
    assert "TAIL" in sent_prompt


def test_hard_reset_does_not_send_or_retry_a_prompt_that_cannot_fit(monkeypatch):
    condenser, _TextContent, _error_type = _condenser_module(monkeypatch)
    instance = condenser.SafeLLMSummarizingCondenser()
    instance.max_tokens = 10
    event = SimpleNamespace(
        id="event-action",
        kind="ActionEvent",
        source="agent",
        thought=[],
        tool_name="bash",
        summary="inspect parser",
        tool_call=SimpleNamespace(arguments='{"command":"pwd"}'),
    )
    requests = []

    class SummaryLLM:
        def completion(self, **kwargs):
            requests.append(kwargs)
            raise AssertionError("an impossible prompt must not be sent")

    instance.llm = SummaryLLM()
    agent_llm = SimpleNamespace(get_token_count=lambda messages: len(messages[0].content[0].text))

    result = instance.hard_context_reset(
        SimpleNamespace(events=[event]),
        agent_llm=agent_llm,
    )

    assert requests == []
    assert result.llm_response_id.startswith(condenser.NONTRAINABLE_CONDENSATION_RESPONSE_PREFIX)


def test_retained_public_summary_has_a_character_safety_cap():
    safety = _safety_module()
    summary = (
        "USER_CONTEXT: Fix the parser.\n"
        "COMPLETED: Located the implementation.\n"
        "PENDING: Apply and test the patch.\n"
        "CURRENT_STATE: "
    )

    with pytest.raises(safety.UnsafeCondensationSummary, match="size limit"):
        safety.validate_condensation_summary(summary + "x" * safety.MAX_RETAINED_SUMMARY_CHARS)


def test_prior_condensation_summary_is_not_reduced_to_sdk_preview():
    safety = _safety_module()
    long_state = "task-state-" * 200
    event = SimpleNamespace(
        kind="CondensationSummaryEvent",
        source="environment",
        summary=long_state,
    )

    rendered = safety.render_event_for_condensation(event)

    assert long_state in rendered
    assert len(rendered) > 500


def test_validate_condensation_summary_extracts_exact_wrapper():
    safety = _safety_module()
    response = """<context_summary>
USER_CONTEXT: Fix the parser.
COMPLETED: Located the implementation.
PENDING: Apply and test the patch.
CURRENT_STATE: No files changed yet.
</context_summary>"""

    summary = safety.validate_condensation_summary(response)

    assert summary.startswith("USER_CONTEXT:")
    assert "<context_summary>" not in summary


def test_extract_visible_condensation_text_drops_in_band_qwen_reasoning():
    safety = _safety_module()
    response = """Private deliberation supplied after the template's opening tag.
</think>

<context_summary>
USER_CONTEXT: Fix the parser.
COMPLETED: Located the implementation.
PENDING: Apply and test the patch.
CURRENT_STATE: No files changed yet.
</context_summary>"""

    visible = safety.extract_visible_condensation_text(response)

    assert "Private deliberation" not in visible
    assert visible.startswith("<context_summary>")
    assert safety.validate_condensation_summary(visible).startswith("USER_CONTEXT:")


def test_extract_completion_reasoning_text_reads_in_band_qwen_prefix():
    safety = _safety_module()
    completion = (
        "Inspect the forgotten events and preserve exact task state.\n"
        "</think>\n\n"
        "USER_CONTEXT: Fix the parser.\n"
        "COMPLETED: Located the implementation.\n"
        "PENDING: Apply and test the patch."
    )

    reasoning = safety.extract_completion_reasoning_text(
        SimpleNamespace(message=SimpleNamespace(content=[])),
        completion_text=completion,
    )

    assert reasoning == "Inspect the forgotten events and preserve exact task state."
    assert "USER_CONTEXT" not in reasoning


def test_extract_completion_reasoning_text_reads_provider_field_without_redacted_data():
    safety = _safety_module()
    response = SimpleNamespace(
        message=SimpleNamespace(
            reasoning_content="Readable provider reasoning.",
            thinking_blocks=[
                {"type": "redacted_thinking", "data": "encrypted-secret"},
            ],
            responses_reasoning_item=None,
            content=[],
        ),
        raw_response=None,
    )

    reasoning = safety.extract_completion_reasoning_text(response)

    assert reasoning == "Readable provider reasoning."
    assert "encrypted-secret" not in reasoning


def test_extract_visible_condensation_text_rejects_unclosed_reasoning():
    safety = _safety_module()

    with pytest.raises(safety.UnsafeCondensationSummary, match="incomplete reasoning tag"):
        safety.extract_visible_condensation_text("<think>private and truncated")


@pytest.mark.parametrize(
    "response",
    [
        (
            "Here's a thinking process:\n"
            "1. Analyze the request.\n"
            "</think>\n\n"
            "<context_summary>USER_CONTEXT: Fix it.</context_summary>"
        ),
        "<think>private reasoning with no close",
        "preface\n<context_summary>USER_CONTEXT: Fix it.</context_summary>",
        "<context_summary>USER_CONTEXT: Fix it.",
        "I need to analyze the events before writing the summary.",
        (
            "<context_summary>\n"
            "USER_CONTEXT: Fix it.\n"
            "REASONING: First I inspected the prompt and decided what to retain.\n"
            "CURRENT_STATE: No files changed.\n"
            "</context_summary>"
        ),
        (
            "<context_summary>\n"
            "USER_CONTEXT: Fix it.\n"
            "<analysis>private deliberation</analysis>\n"
            "CURRENT_STATE: No files changed.\n"
            "</context_summary>"
        ),
        "<context_summary>arbitrary prose without durable state sections</context_summary>",
        ("<context_summary>\nUSER_CONTEXT: Fix it.\nCURRENT_STATE: No files changed.\n</context_summary>"),
        "TASK_TRACKING: item-1 is pending.",
        "Here is the summary.\nUSER_CONTEXT: Fix it.\nCURRENT_STATE: No files changed.",
    ],
)
def test_validate_condensation_summary_rejects_reasoning_or_incomplete_output(response):
    safety = _safety_module()

    with pytest.raises(safety.UnsafeCondensationSummary):
        safety.validate_condensation_summary(response)


def test_plain_structured_summary_remains_compatible():
    safety = _safety_module()
    response = """USER_CONTEXT: Fix the parser.
COMPLETED: Located the implementation.
PENDING: Apply and test the patch.
CURRENT_STATE: No files changed yet."""

    assert safety.validate_condensation_summary(response) == response
    assert safety.is_safe_condensation_summary(response)


def test_handoff_prefix_is_retained_and_structured_summary_remains_valid():
    safety = _safety_module()
    structured_summary = """USER_CONTEXT: Fix the parser.
COMPLETED: Located the implementation.
PENDING: Apply and test the patch.
CURRENT_STATE: No files changed yet."""
    handoff = safety.add_condensation_handoff_prefix(structured_summary)

    assert handoff.startswith(safety.CONDENSATION_HANDOFF_PREFIX)
    assert f"\n\n{structured_summary}" in handoff
    assert safety.validate_condensation_summary(handoff) == handoff
    assert safety.is_safe_condensation_summary(handoff)


def test_native_code_summary_format_can_omit_current_state():
    safety = _safety_module()
    response = """USER_CONTEXT: Fix the parser.
COMPLETED: Located the implementation.
PENDING: Apply and test the patch.
CODE_STATE: parser.py contains parse_step()."""

    assert safety.validate_condensation_summary(response) == response


@pytest.mark.parametrize(
    "raw_response",
    [
        {"choices": [{"finish_reason": "length"}]},
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
        SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="max_tokens")],
            status="completed",
        ),
    ],
)
def test_completion_was_truncated_across_response_shapes(raw_response):
    safety = _safety_module()
    response = SimpleNamespace(raw_response=raw_response)

    assert safety.completion_was_truncated(response)


def test_completed_response_is_not_marked_truncated():
    safety = _safety_module()
    response = SimpleNamespace(
        raw_response=SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop")],
            status="completed",
        )
    )

    assert not safety.completion_was_truncated(response)


@pytest.mark.parametrize(
    "message",
    [
        SimpleNamespace(
            reasoning_content="private chain of thought",
            thinking_blocks=[],
            responses_reasoning_item=None,
        ),
        SimpleNamespace(
            reasoning_content=None,
            thinking_blocks=[{"type": "thinking", "thinking": "private"}],
            responses_reasoning_item=None,
        ),
        SimpleNamespace(
            reasoning_content=None,
            thinking_blocks=[],
            responses_reasoning_item={"type": "reasoning"},
        ),
    ],
)
def test_completion_reasoning_metadata_is_rejected(message):
    safety = _safety_module()

    assert safety.completion_contains_reasoning(SimpleNamespace(message=message))


def test_completion_without_reasoning_metadata_is_allowed():
    safety = _safety_module()
    message = SimpleNamespace(
        reasoning_content=None,
        thinking_blocks=[],
        responses_reasoning_item=None,
    )

    assert not safety.completion_contains_reasoning(SimpleNamespace(message=message))


def _clean_message():
    return SimpleNamespace(
        content=[],
        reasoning_content=None,
        thinking_blocks=[],
        responses_reasoning_item=None,
    )


@pytest.mark.parametrize(
    "raw_response",
    [
        {
            "choices": [
                {
                    "message": {
                        "content": "clean summary",
                        "reasoning_content": "private reasoning",
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": SimpleNamespace(
                        content="clean summary",
                        provider_specific_fields=SimpleNamespace(
                            thinking_blocks=[{"type": "thinking", "thinking": "private"}]
                        ),
                    )
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "clean summary",
                        "provider_specific_fields": {"reasoningContentBlocks": [{"text": "private reasoning"}]},
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {
                                "type": "redacted_thinking",
                                "data": "encrypted-private-reasoning",
                            },
                            {"type": "text", "text": "clean summary"},
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "content": "clean summary",
                        "reasoning_details": [{"text": "private reasoning"}],
                    }
                }
            ]
        },
        {
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"text": "private reasoning summary"}],
                },
                {"type": "message", "content": []},
            ],
        },
    ],
)
def test_raw_provider_reasoning_variants_are_rejected(raw_response):
    safety = _safety_module()
    response = SimpleNamespace(
        message=_clean_message(),
        raw_response=raw_response,
    )

    assert safety.completion_contains_reasoning(response)


def test_reasoning_usage_counts_without_payload_are_allowed():
    safety = _safety_module()
    response = SimpleNamespace(
        message=_clean_message(),
        raw_response={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "clean summary",
                        "provider_specific_fields": {
                            "native_finish_reason": "stop",
                        },
                    },
                }
            ],
            "usage": {
                "completion_tokens_details": {
                    "reasoning_tokens": 17,
                }
            },
        },
    )

    assert not safety.completion_contains_reasoning(response)


def test_provider_reasoning_does_not_enter_context_and_retains_trainable_id(monkeypatch):
    condenser, TextContent, _error_type = _condenser_module(monkeypatch)
    response = SimpleNamespace(
        id="chatcmpl-private-reasoning",
        message=SimpleNamespace(
            content=[
                TextContent(
                    text=(
                        "<context_summary>\n"
                        "USER_CONTEXT: Fix the parser.\n"
                        "COMPLETED: Located the implementation.\n"
                        "PENDING: Apply and test the patch.\n"
                        "CURRENT_STATE: No files changed yet.\n"
                        "</context_summary>"
                    )
                )
            ],
            reasoning_content=None,
            thinking_blocks=[],
            responses_reasoning_item=None,
        ),
        raw_response={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "clean summary",
                        "provider_specific_fields": {"reasoning_content": "private chain of thought"},
                    },
                }
            ]
        },
    )

    event = condenser.SafeLLMSummarizingCondenser._event(
        forgotten_events=[SimpleNamespace(id="event-1")],
        summary_offset=0,
        llm_response=response,
    )

    safety = sys.modules["platoon.openhands.condensation_safety"]
    assert event.summary.startswith(safety.CONDENSATION_HANDOFF_PREFIX)
    assert "\n\nUSER_CONTEXT:" in event.summary
    assert "private reasoning" not in event.summary
    assert event.llm_response_id == response.id
    assert safety.take_condensation_reasoning(event.id) == "private chain of thought"


def test_in_band_reasoning_is_removed_from_context_and_retains_trainable_id(monkeypatch):
    condenser, TextContent, _error_type = _condenser_module(monkeypatch)
    response = SimpleNamespace(
        id="chatcmpl-in-band-reasoning",
        message=SimpleNamespace(
            content=[
                TextContent(
                    text=(
                        "Private chain of thought from the reasoning model.\n"
                        "</think>\n\n"
                        "<context_summary>\n"
                        "USER_CONTEXT: Fix the parser.\n"
                        "COMPLETED: Located the implementation.\n"
                        "PENDING: Apply and test the patch.\n"
                        "CURRENT_STATE: No files changed yet.\n"
                        "</context_summary>"
                    )
                )
            ]
        ),
        raw_response={"choices": [{"finish_reason": "stop"}]},
    )

    event = condenser.SafeLLMSummarizingCondenser._event(
        forgotten_events=[SimpleNamespace(id="event-1")],
        summary_offset=0,
        llm_response=response,
    )

    safety = sys.modules["platoon.openhands.condensation_safety"]
    assert "Private chain of thought" not in event.summary
    assert event.summary.startswith(safety.CONDENSATION_HANDOFF_PREFIX)
    assert "\n\nUSER_CONTEXT:" in event.summary
    assert event.llm_response_id == response.id
    assert safety.take_condensation_reasoning(event.id) == ("Private chain of thought from the reasoning model.")


def test_fully_public_condensation_retains_trainable_completion_id(monkeypatch):
    condenser, TextContent, _error_type = _condenser_module(monkeypatch)
    response = SimpleNamespace(
        id="chatcmpl-public-summary",
        message=SimpleNamespace(
            content=[
                TextContent(
                    text=(
                        "<context_summary>\n"
                        "USER_CONTEXT: Fix the parser.\n"
                        "COMPLETED: Located the implementation.\n"
                        "PENDING: Apply and test the patch.\n"
                        "CURRENT_STATE: No files changed yet.\n"
                        "</context_summary>"
                    )
                )
            ],
            reasoning_content=None,
            thinking_blocks=[],
            responses_reasoning_item=None,
        ),
        raw_response={"choices": [{"finish_reason": "stop"}]},
    )

    event = condenser.SafeLLMSummarizingCondenser._event(
        forgotten_events=[SimpleNamespace(id="event-1")],
        summary_offset=0,
        llm_response=response,
    )

    assert event.llm_response_id == response.id


def test_hard_reset_uses_native_retries_then_safe_nontrainable_fallback(monkeypatch):
    condenser, _TextContent, error_type = _condenser_module(monkeypatch)
    instance = condenser.SafeLLMSummarizingCondenser()
    attempts = []

    def fail_generation(**kwargs):
        attempts.append(kwargs)
        raise error_type("Condensation completion was truncated")

    instance._generate_condensation = fail_generation
    events = [
        SimpleNamespace(
            id="event-user",
            kind="MessageEvent",
            source="user",
            llm_message=SimpleNamespace(content=[]),
            __str__=lambda self: "Fix the parser.",
        ),
        SimpleNamespace(
            id="event-action",
            kind="ActionEvent",
            source="agent",
            thought=[SimpleNamespace(text="PRIVATE DELIBERATION")],
            tool_name="bash",
            summary="inspect parser",
            tool_call=SimpleNamespace(arguments='{"command":"sed -n 1,80p parser.py"}'),
        ),
    ]

    event = instance.hard_context_reset(SimpleNamespace(events=events))

    assert len(attempts) == instance.hard_context_reset_max_retries
    assert attempts[0]["max_event_str_length"] is None
    retry_limits = [attempt["max_event_str_length"] for attempt in attempts[1:]]
    assert all(isinstance(limit, int) and limit > 0 for limit in retry_limits)
    assert retry_limits == sorted(retry_limits, reverse=True)
    assert event.llm_response_id.startswith(condenser.NONTRAINABLE_CONDENSATION_RESPONSE_PREFIX)
    assert "PRIVATE DELIBERATION" not in event.summary
    assert "parser.py" in event.summary
    assert _safety_module().is_safe_condensation_summary(event.summary)
