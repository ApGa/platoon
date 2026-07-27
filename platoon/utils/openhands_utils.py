import logging
from collections import defaultdict
from typing import Sequence

from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import (
    ActionEvent,
    AgentErrorEvent,
    Event,
    EventID,
    MessageEvent,
    ObservationBaseEvent,
)
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.tool.builtins.finish import FinishAction

from platoon.openhands.types import OpenHandsObservation

logger = logging.getLogger(__name__)


def _conversation_execution_status(conversation_state) -> ConversationExecutionStatus | None:
    return (
        getattr(conversation_state, "agent_status", None)
        or getattr(conversation_state, "execution_status", None)
        or getattr(conversation_state, "agent_state", None)
    )


def _is_terminal_status(conversation_state) -> bool:
    """Return True if the conversation execution status is a terminal state."""
    return _conversation_execution_status(conversation_state) in (
        ConversationExecutionStatus.FINISHED,
        ConversationExecutionStatus.STUCK,
        ConversationExecutionStatus.ERROR,
    )


def is_action(event: Event) -> bool:
    return isinstance(event, ActionEvent) or (isinstance(event, MessageEvent) and event.source == "agent")


def group_actions(events: Sequence[Event]):
    """Build a map of llm_response_id -> list of ActionEvent IDs."""
    batches: dict[EventID, list[EventID]] = defaultdict(list)
    action_id_to_response_id: dict[EventID, EventID] = {}
    tool_call_id_to_action_id = {}
    action_id_to_tool_call_id = {}

    for event in events:
        if is_action(event):
            llm_response_id = event.llm_response_id
            batches[llm_response_id].append(event.id)
            action_id_to_response_id[event.id] = llm_response_id
            tool_call_id = getattr(event, "tool_call_id", None)
            if isinstance(event, ActionEvent) and tool_call_id is not None:
                tool_call_id_to_action_id[tool_call_id] = event.id
                action_id_to_tool_call_id[event.id] = tool_call_id

    return batches, action_id_to_response_id, tool_call_id_to_action_id, action_id_to_tool_call_id


def _event_cursor(events: Sequence[Event], event_id: EventID | None) -> int:
    """Return the index immediately after event_id, or the start of the stream."""
    if event_id is None:
        return 0
    for index, event in enumerate(events):
        if event.id == event_id:
            return index + 1
    return 0


def _next_action_batch(
    events: Sequence[Event],
    start_index: int,
) -> tuple[list[Event], int | None, int]:
    """Return the oldest action batch after start_index and its event window."""
    actions: list[Event] = []
    batch_start: int | None = None
    llm_response_id: EventID | None = None

    for index in range(start_index, len(events)):
        event = events[index]
        if not is_action(event):
            continue

        event_response_id = event.llm_response_id
        if batch_start is None:
            batch_start = index
            llm_response_id = event_response_id
        elif event_response_id != llm_response_id:
            return actions, batch_start, index

        actions.append(event)

    return actions, batch_start, len(events)


def _batch_results(
    events: Sequence[Event],
    actions: Sequence[Event],
) -> tuple[dict[EventID, list[Event]], list[Event]]:
    """Associate result events with actions, retaining unrelated batch events."""
    action_ids = {action.id for action in actions}
    tool_call_id_to_action_id = {
        action.tool_call_id: action.id
        for action in actions
        if isinstance(action, ActionEvent) and getattr(action, "tool_call_id", None) is not None
    }
    results: dict[EventID, list[Event]] = defaultdict(list)
    unrelated: list[Event] = []

    for event in events:
        if is_action(event):
            continue

        action_id = None
        if isinstance(event, ObservationBaseEvent):
            action_id = getattr(event, "action_id", None)
            if action_id not in action_ids and isinstance(event, AgentErrorEvent):
                action_id = tool_call_id_to_action_id.get(getattr(event, "tool_call_id", None))

        if action_id in action_ids:
            results[action_id].append(event)
        else:
            unrelated.append(event)

    return results, unrelated


def _action_is_self_observing(action: Event, conversation_state) -> bool:
    if isinstance(action, MessageEvent) and action.source == "agent":
        return True
    return (
        isinstance(action, ActionEvent)
        and action.source == "agent"
        and (isinstance(action.action, FinishAction) or _is_terminal_status(conversation_state))
    )


def get_actions_for_last_obs(observation: OpenHandsObservation, require_same_llm_call_id: bool = True) -> list[Event]:
    """Collect all actions since the last observation, once each has a corresponding future observation."""
    events = observation.conversation_state.events
    start_index = _event_cursor(events, observation.last_step_observation_id)
    actions, batch_start, batch_end = _next_action_batch(events, start_index)
    if not actions or batch_start is None:
        return []

    if require_same_llm_call_id:
        llm_response_id = actions[0].llm_response_id
        if any(action.llm_response_id != llm_response_id for action in actions):
            raise ValueError(
                "Detected at least two actions in a step with differing llm_response_id. "
                "This is unexpected and can lead to undefined behavior."
            )

    batch_events = events[batch_start:batch_end]
    results, unrelated = _batch_results(batch_events, actions)
    conversation_error_seen = any(isinstance(event, ConversationErrorEvent) for event in unrelated)
    if not conversation_error_seen:
        for action in actions:
            if action.id not in results and not _action_is_self_observing(
                action,
                observation.conversation_state,
            ):
                logger.debug(
                    "Waiting for a result for action event %s from response %s",
                    action.id,
                    action.llm_response_id,
                )
                return []

    return actions


def _action_batch_containing(
    events: Sequence[Event],
    action_id: EventID,
) -> tuple[list[Event], int | None, int]:
    selected_index = next(
        (index for index, event in enumerate(events) if is_action(event) and event.id == action_id),
        None,
    )
    if selected_index is None:
        return [], None, len(events)

    llm_response_id = events[selected_index].llm_response_id
    batch_start = selected_index
    for index in range(selected_index - 1, -1, -1):
        event = events[index]
        if not is_action(event):
            continue
        if event.llm_response_id != llm_response_id:
            break
        batch_start = index

    actions: list[Event] = []
    batch_end = len(events)
    for index in range(batch_start, len(events)):
        event = events[index]
        if not is_action(event):
            continue
        if event.llm_response_id != llm_response_id:
            batch_end = index
            break
        actions.append(event)

    return actions, batch_start, batch_end


def get_obs_for_last_action(
    observation: OpenHandsObservation,
    action_events: Sequence[Event] | None = None,
) -> list[Event]:
    """Collect event(s) that immediately follow a past ActionEvent and are
    fully observed by a subsequent ObservationBaseEvent referencing them.
    """
    events = observation.conversation_state.events
    if observation.last_step_action_id is None:
        first_action_index = next((index for index, event in enumerate(events) if is_action(event)), None)
        if first_action_index is None:
            if not _is_terminal_status(observation.conversation_state):
                return []
            first_action_index = len(events)
        return [event for event in events[:first_action_index] if not is_action(event)]

    actions, batch_start, batch_end = _action_batch_containing(events, observation.last_step_action_id)
    if not actions or batch_start is None:
        return []

    if action_events:
        expected_action_ids = {action.id for action in action_events}
        batch_action_ids = {action.id for action in actions}
        if expected_action_ids != batch_action_ids:
            logger.warning(
                "Action events passed to observation parsing do not match response batch: expected=%s actual=%s",
                expected_action_ids,
                batch_action_ids,
            )

    batch_events = events[batch_start:batch_end]
    results, unrelated = _batch_results(batch_events, actions)
    conversation_error_seen = any(isinstance(event, ConversationErrorEvent) for event in unrelated)
    batch_is_closed = batch_end < len(events) or _is_terminal_status(observation.conversation_state)
    if not batch_is_closed and not conversation_error_seen:
        return []

    if not conversation_error_seen:
        for action in actions:
            if action.id not in results and not _action_is_self_observing(
                action,
                observation.conversation_state,
            ):
                return []

    ordered_results = [event for action in actions for event in results.get(action.id, [])]
    return ordered_results + unrelated


def is_finished(observation: OpenHandsObservation, last_event_seen: EventID | None = None) -> bool:
    conversation_state = observation.conversation_state
    oh_conversation_finished = _is_terminal_status(conversation_state)
    last_event_id = conversation_state.events[-1].id
    assert last_event_id is not None, "Last event in conversation must have a non-None ID"
    valid_ids = [
        event_id
        for event_id in [observation.last_step_action_id, observation.last_step_observation_id, last_event_seen]
        if event_id is not None
    ]
    platoon_episode_caught_up = last_event_id in valid_ids
    if oh_conversation_finished and platoon_episode_caught_up:
        try:
            logger.debug(
                "is_finished: conversation finished with status "
                f"{_conversation_execution_status(conversation_state)}, last_event_id: {last_event_id}, "
                f"valid_ids: {valid_ids}, last_event_seen: {conversation_state.events[-1].kind}"
            )
        except Exception as e:
            logger.debug(
                "is_finished: conversation finished with status "
                f"{_conversation_execution_status(conversation_state)}, last_event_id: {last_event_id}, "
                f"valid_ids: {valid_ids}, unable to print last event kind due to error: {e}"
            )
    return oh_conversation_finished and platoon_episode_caught_up
