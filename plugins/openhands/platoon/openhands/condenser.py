"""OpenHands condenser that keeps model reasoning out of state summaries."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from uuid import uuid4

from openhands.sdk.context.condenser import (
    LLMSummarizingCondenser,
)
from openhands.sdk.context.condenser import (
    llm_summarizing_condenser as openhands_summarizing_condenser,
)
from openhands.sdk.context.condenser.base import NoCondensationAvailableException
from openhands.sdk.context.prompts import render_template
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.utils import maybe_truncate

from .condensation_safety import (
    NONTRAINABLE_CONDENSATION_RESPONSE_PREFIX,
    completion_contains_reasoning,
    completion_was_truncated,
    extract_visible_condensation_text,
    render_event_for_condensation,
    validate_condensation_summary,
)

logger = logging.getLogger(__name__)


class SafeLLMSummarizingCondenser(LLMSummarizingCondenser):
    """Summarize public task state without putting reasoning back into context."""

    @staticmethod
    def _messages(
        forgotten_events: Sequence[LLMConvertibleEvent],
        max_event_str_length: int | None = None,
    ) -> list[Message]:
        event_strings = [
            maybe_truncate(
                render_event_for_condensation(event),
                truncate_after=max_event_str_length,
            )
            for event in forgotten_events
        ]
        prompt = render_template(
            os.path.join(
                os.path.dirname(openhands_summarizing_condenser.__file__),
                "prompts",
            ),
            "summarizing_prompt.j2",
            events=event_strings,
        )
        return [
            Message(role="user", content=[TextContent(text=prompt)]),
        ]

    @staticmethod
    def _completion_text(llm_response) -> str:
        return "\n".join(item.text for item in llm_response.message.content if isinstance(item, TextContent))

    @classmethod
    def _summary_text(cls, llm_response) -> str:
        if completion_was_truncated(llm_response):
            raise NoCondensationAvailableException("Condensation completion was truncated")
        text = cls._completion_text(llm_response)
        try:
            # Reasoning is intentionally enabled for summary quality. Qwen may
            # return that reasoning in-band, followed by ``</think>`` and the
            # public summary. Only the public suffix enters future context.
            return validate_condensation_summary(extract_visible_condensation_text(text))
        except ValueError as exc:
            raise NoCondensationAvailableException(f"Unsafe condensation completion: {exc}") from exc

    @staticmethod
    def _fallback_summary(
        forgotten_events: Sequence[LLMConvertibleEvent],
    ) -> str:
        """Build a bounded public-state reset when model summarization fails.

        A failed hard reset must not terminate an otherwise valid rollout. This
        fallback retains the initial user request and a bounded tail of public
        tool/action state. Event text is JSON-escaped and angle brackets are
        escaped so embedded tool output cannot accidentally become a reasoning
        tag or summary-control wrapper.
        """

        def encoded(event: LLMConvertibleEvent, max_chars: int) -> str:
            rendered = maybe_truncate(
                render_event_for_condensation(event),
                truncate_after=max_chars,
            )
            return json.dumps(rendered, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e")

        user_context = "Retain the original task and constraints from the conversation."
        for event in forgotten_events:
            if getattr(event, "kind", None) == "MessageEvent" and str(getattr(event, "source", "")) == "user":
                user_context = f"Initial public task event: {encoded(event, 2_500)}"
                break

        tail = [event for event in forgotten_events if getattr(event, "kind", None) != "SystemPromptEvent"][-8:]
        snapshots = [f"- {getattr(event, 'kind', type(event).__name__)}: {encoded(event, 900)}" for event in tail]
        current_state = (
            "Recent public event excerpts:\n" + "\n".join(snapshots)
            if snapshots
            else "No public event excerpt was available; inspect the environment before continuing."
        )
        summary = (
            f"USER_CONTEXT: {user_context}\n"
            "COMPLETED: Earlier public interaction history was compacted after tool use.\n"
            "PENDING: Reinspect current environment state, continue the original task, "
            "verify outputs, and invoke the required completion tool.\n"
            f"CURRENT_STATE: {current_state}"
        )
        try:
            return validate_condensation_summary(summary)
        except ValueError:
            # This fixed form is intentionally independent of event-controlled
            # text and therefore remains a safe last resort.
            return (
                "USER_CONTEXT: Continue the original retained task and constraints.\n"
                "COMPLETED: Earlier public interaction history was compacted.\n"
                "PENDING: Reinspect the environment, continue the task, verify outputs, "
                "and invoke the required completion tool.\n"
                "CURRENT_STATE: A deterministic context reset was applied; no private "
                "reasoning was retained."
            )

    @classmethod
    def _fallback_event(
        cls,
        *,
        forgotten_events: Sequence[LLMConvertibleEvent],
        summary_offset: int,
    ) -> Condensation:
        return Condensation(
            forgotten_event_ids={event.id for event in forgotten_events},
            summary=cls._fallback_summary(forgotten_events),
            summary_offset=summary_offset,
            # Fallback summaries are not model completions and must not be
            # looked up in AReaL's interaction cache for policy loss.
            llm_response_id=f"{NONTRAINABLE_CONDENSATION_RESPONSE_PREFIX}{uuid4().hex}",
        )

    @classmethod
    def _event(
        cls,
        *,
        forgotten_events: Sequence[LLMConvertibleEvent],
        summary_offset: int,
        llm_response,
    ) -> Condensation:
        raw_text = cls._completion_text(llm_response)
        contains_nonpublic_reasoning = "</think>" in raw_text.lower() or completion_contains_reasoning(llm_response)
        return Condensation(
            forgotten_event_ids={event.id for event in forgotten_events},
            summary=cls._summary_text(llm_response),
            summary_offset=summary_offset,
            # The sanitized summary is safe for future context, but a raw
            # completion with stripped/provider reasoning should not receive
            # synthetic policy loss. Only fully public completions retain their
            # AReaL interaction ID.
            llm_response_id=(
                f"{NONTRAINABLE_CONDENSATION_RESPONSE_PREFIX}{uuid4().hex}"
                if contains_nonpublic_reasoning
                else llm_response.id
            ),
        )

    def _generate_condensation(
        self,
        forgotten_events: Sequence[LLMConvertibleEvent],
        summary_offset: int,
        max_event_str_length: int | None = None,
    ) -> Condensation:
        assert forgotten_events, "No events to condense."
        try:
            response = self.llm.completion(messages=self._messages(forgotten_events, max_event_str_length))
        except NoCondensationAvailableException:
            raise
        except Exception as exc:
            raise NoCondensationAvailableException(f"Summarization LLM call failed: {exc}") from exc
        return self._event(
            forgotten_events=forgotten_events,
            summary_offset=summary_offset,
            llm_response=response,
        )

    async def _agenerate_condensation(
        self,
        forgotten_events: Sequence[LLMConvertibleEvent],
        summary_offset: int,
        max_event_str_length: int | None = None,
    ) -> Condensation:
        assert forgotten_events, "No events to condense."
        try:
            response = await self.llm.acompletion(messages=self._messages(forgotten_events, max_event_str_length))
        except NoCondensationAvailableException:
            raise
        except Exception as exc:
            raise NoCondensationAvailableException(f"Summarization LLM call failed: {exc}") from exc
        return self._event(
            forgotten_events=forgotten_events,
            summary_offset=summary_offset,
            llm_response=response,
        )

    def hard_context_reset(self, view, agent_llm=None) -> Condensation | None:  # noqa: ARG002
        """Use OpenHands' native retry/truncation policy, then a safe fallback."""

        if not view.events:
            return None

        max_event_str_length: int | None = None
        attempts_remaining = self.hard_context_reset_max_retries
        last_error: Exception | None = None
        while attempts_remaining > 0:
            try:
                return self._generate_condensation(
                    forgotten_events=view.events,
                    summary_offset=0,
                    max_event_str_length=max_event_str_length,
                )
            except Exception as exc:
                last_error = exc
                if max_event_str_length is None:
                    max_event_str_length = max(len(render_event_for_condensation(event)) for event in view.events)
                max_event_str_length = int(max_event_str_length * self.hard_context_reset_context_scaling)
                attempts_remaining -= 1
                logger.warning(
                    "Hard context reset summarization failed (%s); reducing max "
                    "event size to %d and retrying (%d attempts remain).",
                    exc,
                    max_event_str_length,
                    attempts_remaining,
                )

        if last_error is not None:
            logger.warning(
                "Hard context reset model summary exhausted retries (%s); using deterministic public-state fallback.",
                last_error,
            )
        return self._fallback_event(
            forgotten_events=view.events,
            summary_offset=0,
        )

    async def ahard_context_reset(self, view, agent_llm=None) -> Condensation | None:  # noqa: ARG002
        """Async hard reset with the native retry policy and safe fallback."""

        if not view.events:
            return None

        max_event_str_length: int | None = None
        attempts_remaining = self.hard_context_reset_max_retries
        last_error: Exception | None = None
        while attempts_remaining > 0:
            try:
                return await self._agenerate_condensation(
                    forgotten_events=view.events,
                    summary_offset=0,
                    max_event_str_length=max_event_str_length,
                )
            except Exception as exc:
                last_error = exc
                if max_event_str_length is None:
                    max_event_str_length = max(len(render_event_for_condensation(event)) for event in view.events)
                max_event_str_length = int(max_event_str_length * self.hard_context_reset_context_scaling)
                attempts_remaining -= 1
                logger.warning(
                    "Async hard context reset summarization failed (%s); reducing "
                    "max event size to %d and retrying (%d attempts remain).",
                    exc,
                    max_event_str_length,
                    attempts_remaining,
                )

        if last_error is not None:
            logger.warning(
                "Async hard context reset model summary exhausted retries (%s); "
                "using deterministic public-state fallback.",
                last_error,
            )
        return self._fallback_event(
            forgotten_events=view.events,
            summary_offset=0,
        )
