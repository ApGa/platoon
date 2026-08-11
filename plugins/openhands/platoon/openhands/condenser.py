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
from openhands.sdk.context.condenser.base import NoCondensationAvailableException
from openhands.sdk.context.prompts import render_template
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.condenser import Condensation
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.utils import maybe_truncate

from .condensation_safety import (
    NONTRAINABLE_CONDENSATION_RESPONSE_PREFIX,
    add_condensation_handoff_prefix,
    completion_contains_reasoning,
    completion_was_truncated,
    extract_completion_reasoning_text,
    extract_visible_condensation_text,
    remember_condensation_reasoning,
    render_event_for_condensation,
    validate_condensation_summary,
)

logger = logging.getLogger(__name__)


class _CondensationPromptCannotFit(NoCondensationAvailableException):
    """Raised before an LLM call when even maximally clipped events do not fit."""


class SafeLLMSummarizingCondenser(LLMSummarizingCondenser):
    """Summarize public task state without putting reasoning back into context."""

    @staticmethod
    def _messages_from_event_strings(event_strings: Sequence[str]) -> list[Message]:
        prompt = render_template(
            os.path.join(os.path.dirname(__file__), "prompts"),
            "summarizing_prompt.j2",
            events=event_strings,
        )
        return [
            Message(role="user", content=[TextContent(text=prompt)]),
        ]

    @classmethod
    def _messages(
        cls,
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
        return cls._messages_from_event_strings(event_strings)

    @staticmethod
    def _prompt_token_count(messages: list[Message], agent_llm) -> int | None:
        """Count with the agent's exact tokenizer/template when it is available."""

        if agent_llm is None:
            return None
        try:
            token_count = agent_llm.get_token_count(messages)
        except Exception:
            logger.warning(
                "Unable to count condensation prompt tokens; retaining native request/retry behavior.",
                exc_info=True,
            )
            return None
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
            logger.warning(
                "Condensation prompt token counter returned %r; retaining native request/retry behavior.",
                token_count,
            )
            return None
        return token_count

    def _messages_fitted_to_prompt_budget(
        self,
        forgotten_events: Sequence[LLMConvertibleEvent],
        *,
        agent_llm=None,
        max_event_str_length: int | None = None,
    ) -> list[Message]:
        """Fit a summary prompt before sending it to the completion endpoint.

        ``max_tokens`` is already configured to 80% of the model's 32K context
        for these rollouts. Reusing it as the maximum condensation *input*
        leaves at least the remaining 20% for reasoning plus summary text. The
        condenser LLM's separately configured 80%-of-context generation limit
        remains a ceiling, not a target; the AReaL endpoint clips it to the
        request's actual remaining context.

        Truncation deliberately uses OpenHands' native policy: one uniform
        character limit per event and ``maybe_truncate``'s head/notice/tail
        rendering. A binary search only chooses that character limit; it does
        not introduce a second event serialization policy.
        """

        rendered_events = [render_event_for_condensation(event) for event in forgotten_events]

        def messages_for_limit(limit: int | None) -> list[Message]:
            return self._messages_from_event_strings(
                [maybe_truncate(rendered, truncate_after=limit) for rendered in rendered_events]
            )

        requested_messages = messages_for_limit(max_event_str_length)
        prompt_budget = self.max_tokens
        if not prompt_budget or agent_llm is None:
            return requested_messages

        requested_tokens = self._prompt_token_count(requested_messages, agent_llm)
        if requested_tokens is None or requested_tokens <= prompt_budget:
            return requested_messages

        largest_event = max(len(rendered) for rendered in rendered_events)
        upper_limit = largest_event
        if max_event_str_length is not None:
            upper_limit = min(upper_limit, max_event_str_length)
        upper_limit = max(upper_limit, 1)

        # ``maybe_truncate(..., truncate_after=1)`` is the smallest prompt we
        # can form while retaining one placeholder character per native event.
        minimum_messages = messages_for_limit(1)
        minimum_tokens = self._prompt_token_count(minimum_messages, agent_llm)
        if minimum_tokens is None:
            return requested_messages
        if minimum_tokens > prompt_budget:
            raise _CondensationPromptCannotFit(
                "Condensation prompt cannot fit its input budget even with "
                f"one character per event (prompt_tokens={minimum_tokens}, "
                f"budget={prompt_budget}, events={len(forgotten_events)})"
            )

        best_limit = 1
        best_messages = minimum_messages
        best_tokens = minimum_tokens
        low = 2
        high = upper_limit
        while low <= high:
            candidate_limit = (low + high) // 2
            candidate_messages = messages_for_limit(candidate_limit)
            candidate_tokens = self._prompt_token_count(candidate_messages, agent_llm)
            if candidate_tokens is None:
                # We already have a verified fitting candidate. Prefer that
                # conservative prompt over sending the known-oversize original.
                break
            if candidate_tokens <= prompt_budget:
                best_limit = candidate_limit
                best_messages = candidate_messages
                best_tokens = candidate_tokens
                low = candidate_limit + 1
            else:
                high = candidate_limit - 1

        logger.info(
            "Fitted condensation prompt to context budget: prompt_tokens=%d->%d budget=%d max_event_chars=%d events=%d",
            requested_tokens,
            best_tokens,
            prompt_budget,
            best_limit,
            len(forgotten_events),
        )
        return best_messages

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
            summary = validate_condensation_summary(extract_visible_condensation_text(text))
            return add_condensation_handoff_prefix(summary)
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
            return add_condensation_handoff_prefix(validate_condensation_summary(summary))
        except ValueError:
            # This fixed form is intentionally independent of event-controlled
            # text and therefore remains a safe last resort.
            return add_condensation_handoff_prefix(
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
        event = Condensation(
            forgotten_event_ids={event.id for event in forgotten_events},
            summary=cls._summary_text(llm_response),
            summary_offset=summary_offset,
            # Train the actual sampled completion, including any reasoning
            # tokens, while retaining only the sanitized public summary in
            # future agent context. Deterministic fallbacks remain nontrainable
            # because they have no corresponding cached model completion.
            llm_response_id=llm_response.id,
        )
        reasoning = extract_completion_reasoning_text(
            llm_response,
            completion_text=raw_text,
        )
        if reasoning is None and completion_contains_reasoning(llm_response):
            reasoning = "[Reasoning payload was redacted or unavailable for display.]"
        remember_condensation_reasoning(str(event.id), reasoning)
        return event

    def _generate_condensation(
        self,
        forgotten_events: Sequence[LLMConvertibleEvent],
        summary_offset: int,
        max_event_str_length: int | None = None,
        agent_llm=None,
    ) -> Condensation:
        assert forgotten_events, "No events to condense."
        messages = self._messages_fitted_to_prompt_budget(
            forgotten_events,
            agent_llm=agent_llm,
            max_event_str_length=max_event_str_length,
        )
        try:
            response = self.llm.completion(messages=messages)
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
        agent_llm=None,
    ) -> Condensation:
        assert forgotten_events, "No events to condense."
        messages = self._messages_fitted_to_prompt_budget(
            forgotten_events,
            agent_llm=agent_llm,
            max_event_str_length=max_event_str_length,
        )
        try:
            response = await self.llm.acompletion(messages=messages)
        except NoCondensationAvailableException:
            raise
        except Exception as exc:
            raise NoCondensationAvailableException(f"Summarization LLM call failed: {exc}") from exc
        return self._event(
            forgotten_events=forgotten_events,
            summary_offset=summary_offset,
            llm_response=response,
        )

    def _condensation_inputs(self, view, agent_llm=None):
        """Apply the native forgotten-event and minimum-progress checks."""

        try:
            forgotten_events, summary_offset = self._get_forgotten_events(
                view,
                agent_llm=agent_llm,
            )
        except ValueError as exc:
            raise NoCondensationAvailableException("Unable to compute forgotten events") from exc

        if not forgotten_events:
            raise NoCondensationAvailableException(
                "Cannot condense 0 events. This typically occurs when a tool loop "
                "spans almost the entire view, leaving no valid range for forgetting "
                "events. Consider adjusting keep_first or max_size parameters."
            )
        if len(forgotten_events) < len(view) * self.minimum_progress:
            raise NoCondensationAvailableException(
                "Cannot apply condensation: events forgotten below minimum progress threshold."
            )
        return forgotten_events, summary_offset

    def get_condensation(self, view, agent_llm=None) -> Condensation:
        """Generate a normal condensation with proactive prompt fitting."""

        forgotten_events, summary_offset = self._condensation_inputs(view, agent_llm)
        return self._generate_condensation(
            forgotten_events=forgotten_events,
            summary_offset=summary_offset,
            agent_llm=agent_llm,
        )

    async def aget_condensation(self, view, agent_llm=None) -> Condensation:
        """Async normal condensation with proactive prompt fitting."""

        forgotten_events, summary_offset = self._condensation_inputs(view, agent_llm)
        return await self._agenerate_condensation(
            forgotten_events=forgotten_events,
            summary_offset=summary_offset,
            agent_llm=agent_llm,
        )

    def hard_context_reset(self, view, agent_llm=None) -> Condensation | None:
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
                    agent_llm=agent_llm,
                )
            except _CondensationPromptCannotFit as exc:
                last_error = exc
                logger.warning(
                    "Hard context reset prompt cannot fit after maximum native "
                    "event clipping (%s); using deterministic fallback without "
                    "an impossible LLM request.",
                    exc,
                )
                break
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

    async def ahard_context_reset(self, view, agent_llm=None) -> Condensation | None:
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
                    agent_llm=agent_llm,
                )
            except _CondensationPromptCannotFit as exc:
                last_error = exc
                logger.warning(
                    "Async hard context reset prompt cannot fit after maximum "
                    "native event clipping (%s); using deterministic fallback "
                    "without an impossible LLM request.",
                    exc,
                )
                break
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
