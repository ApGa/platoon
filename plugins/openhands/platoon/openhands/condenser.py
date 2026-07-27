"""OpenHands condenser that keeps model reasoning out of state summaries."""

from __future__ import annotations

from collections.abc import Sequence

from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.context.condenser.base import NoCondensationAvailableException
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm import Message, TextContent

from .condensation_safety import (
    build_safe_condensation_prompt,
    completion_contains_reasoning,
    completion_was_truncated,
    validate_condensation_summary,
)


class SafeLLMSummarizingCondenser(LLMSummarizingCondenser):
    """Summarize public task state while rejecting reasoning-bearing output."""

    @staticmethod
    def _messages(
        forgotten_events: Sequence[LLMConvertibleEvent],
        max_event_str_length: int | None = None,
    ) -> list[Message]:
        prompt_kwargs = (
            {"max_event_chars": max_event_str_length}
            if max_event_str_length is not None
            else {}
        )
        system_prompt, user_prompt = build_safe_condensation_prompt(
            forgotten_events,
            **prompt_kwargs,
        )
        return [
            Message(role="system", content=[TextContent(text=system_prompt)]),
            Message(role="user", content=[TextContent(text=user_prompt)]),
        ]

    @staticmethod
    def _summary_text(llm_response) -> str:
        if completion_was_truncated(llm_response):
            raise NoCondensationAvailableException("Condensation completion was truncated")
        if completion_contains_reasoning(llm_response):
            raise NoCondensationAvailableException(
                "Condensation completion contained private reasoning"
            )
        text = "\n".join(
            item.text
            for item in llm_response.message.content
            if isinstance(item, TextContent)
        )
        try:
            return validate_condensation_summary(text)
        except ValueError as exc:
            raise NoCondensationAvailableException(f"Unsafe condensation completion: {exc}") from exc

    @classmethod
    def _event(
        cls,
        *,
        forgotten_events: Sequence[LLMConvertibleEvent],
        summary_offset: int,
        llm_response,
    ) -> Condensation:
        return Condensation(
            forgotten_event_ids={event.id for event in forgotten_events},
            summary=cls._summary_text(llm_response),
            summary_offset=summary_offset,
            llm_response_id=llm_response.id,
        )

    def _generate_condensation(
        self,
        forgotten_events: Sequence[LLMConvertibleEvent],
        summary_offset: int,
        max_event_str_length: int | None = None,
    ) -> Condensation:
        assert forgotten_events, "No events to condense."
        try:
            response = self.llm.completion(
                messages=self._messages(forgotten_events, max_event_str_length)
            )
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
            response = await self.llm.acompletion(
                messages=self._messages(forgotten_events, max_event_str_length)
            )
        except NoCondensationAvailableException:
            raise
        except Exception as exc:
            raise NoCondensationAvailableException(f"Summarization LLM call failed: {exc}") from exc
        return self._event(
            forgotten_events=forgotten_events,
            summary_offset=summary_offset,
            llm_response=response,
        )
