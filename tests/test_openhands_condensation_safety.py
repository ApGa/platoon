from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "plugins"
    / "openhands"
    / "platoon"
    / "openhands"
    / "condensation_safety.py"
)


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

    class TextContent:
        def __init__(self, *, text):
            self.text = text

    class Message:
        def __init__(self, *, role, content):
            self.role = role
            self.content = content

    class Condensation:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    _module(monkeypatch, "openhands")
    _module(monkeypatch, "openhands.sdk")
    _module(monkeypatch, "openhands.sdk.context")
    _module(
        monkeypatch,
        "openhands.sdk.context.condenser",
        LLMSummarizingCondenser=object,
    )
    _module(
        monkeypatch,
        "openhands.sdk.context.condenser.base",
        NoCondensationAvailableException=NoCondensationAvailableException,
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
            content=[
                SimpleNamespace(
                    text="PRIVATE DELIBERATION\n</think>\n\nImplemented the parser fix."
                )
            ]
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
        llm_message=SimpleNamespace(
            content=[
                SimpleNamespace(
                    text="PRIVATE DELIBERATION FROM A TRUNCATED COMPLETION"
                )
            ]
        ),
    )

    rendered = safety.render_event_for_condensation(event)

    assert "PRIVATE DELIBERATION" not in rendered
    assert "[no public text content]" in rendered


def test_safe_prompt_does_not_include_action_reasoning():
    safety = _safety_module()
    event = SimpleNamespace(
        kind="ActionEvent",
        source="agent",
        thought=[SimpleNamespace(text="SECRET CHAIN OF THOUGHT")],
        reasoning_content="SECRET REASONING FIELD",
        tool_name="view",
        summary=None,
        tool_call=SimpleNamespace(arguments='{"path":"src/main.py"}'),
    )

    system_prompt, user_prompt = safety.build_safe_condensation_prompt([event])

    assert "Never reveal" in system_prompt
    assert safety.CONTEXT_SUMMARY_OPEN in system_prompt
    assert "SECRET" not in user_prompt
    assert "src/main.py" in user_prompt


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
        (
            "<context_summary>\n"
            "USER_CONTEXT: Fix it.\n"
            "CURRENT_STATE: No files changed.\n"
            "</context_summary>"
        ),
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
                            thinking_blocks=[
                                {"type": "thinking", "thinking": "private"}
                            ]
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
                        "provider_specific_fields": {
                            "reasoningContentBlocks": [
                                {"text": "private reasoning"}
                            ]
                        },
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


def test_provider_reasoning_metadata_does_not_enter_condensation_context(monkeypatch):
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
                        "provider_specific_fields": {
                            "reasoning_content": "private chain of thought"
                        },
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

    assert event.summary.startswith("USER_CONTEXT:")
    assert "private reasoning" not in event.summary
    assert event.llm_response_id.startswith(
        condenser.NONTRAINABLE_CONDENSATION_RESPONSE_PREFIX
    )


def test_in_band_reasoning_is_removed_before_condensation_event(monkeypatch):
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

    assert "Private chain of thought" not in event.summary
    assert event.summary.startswith("USER_CONTEXT:")
    assert event.llm_response_id.startswith(
        condenser.NONTRAINABLE_CONDENSATION_RESPONSE_PREFIX
    )


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


def test_hard_reset_uses_safe_nontrainable_fallback_after_one_failed_retry(monkeypatch):
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

    assert len(attempts) == 1
    assert event.llm_response_id.startswith(
        condenser.NONTRAINABLE_CONDENSATION_RESPONSE_PREFIX
    )
    assert "PRIVATE DELIBERATION" not in event.summary
    assert "parser.py" in event.summary
    assert _safety_module().is_safe_condensation_summary(event.summary)
