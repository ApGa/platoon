    
from typing import Sequence
from openhands.sdk.event import ActionEvent, AgentErrorEvent, Event, EventID, MessageEvent
from openhands.sdk.conversation.base import ConversationStateProtocol
from openhands.sdk.conversation.state import AgentExecutionStatus
from platoon.envs.openhands.types import OpenHandsObservation


def is_action(event: Event) -> bool:
    return isinstance(event, ActionEvent) \
        or (isinstance(event, MessageEvent) and event.source == "agent")

# TODO: Logic can probably be simplified now, by looking at changes in llm_response_id. Anytime llm_response_id changes, we can consider it as a new action.
def get_actions_for_last_obs(observation: OpenHandsObservation, require_same_llm_call_id: bool = False) -> list[Event]:
    """Collect Event(s) we consider as actions that immediately follow a past ObservationEvent and are
    fully observed by a subsequent ObservationBaseEvent referencing them.
    """
    events = observation.conversation_state.events
    new_actions: list[Event] = list()
    new_actions_candidates: list[Event] = list()
    seen_action_ids: set[EventID] = set()
    at_least_one_future_obs_seen = False
    at_least_one_future_error_event_seen = False
    for event in reversed(events):
        if event.id == observation.last_step_observation_id:
            break
        if not is_action(event):
            new_actions.clear()
            at_least_one_future_obs_seen = True
            if hasattr(event, "action_id"):
                seen_action_ids.add(event.action_id)
            if isinstance(event, AgentErrorEvent):
                at_least_one_future_error_event_seen = True
            continue
        else:
            new_actions.append(event)
            new_actions_candidates.append(event)

    last_event_seen = new_actions[-1].id if new_actions else None
    if not is_finished(observation, last_event_seen=last_event_seen) and not at_least_one_future_error_event_seen:
        for action in new_actions:
            if isinstance(action, ActionEvent) and action.id not in seen_action_ids:
                new_actions.clear()
                break

        if not at_least_one_future_obs_seen:
            new_actions.clear()

    if require_same_llm_call_id and new_actions:
        llm_call_id = new_actions[0].llm_response_id
        if any(action.llm_response_id != llm_call_id for action in new_actions):
            raise ValueError("Detected at least two actions in a step with differing llm_response_id. "
            "This is unexpected and can lead to undefined behavior.")

    return list(reversed(new_actions))


def get_obs_for_last_action(observation: OpenHandsObservation) -> list[Event]:
    """Collect event(s) that immediately follow a past ActionEvent and are
    fully observed by a subsequent ObservationBaseEvent referencing them.
    """
    events = observation.conversation_state.events
    new_obs: list[Event] = list()
    at_least_one_future_action_seen = False
    for event in reversed(events):
        if event.id == observation.last_step_action_id:
            break

        if is_action(event):
            at_least_one_future_action_seen = True
            new_obs.clear()
            continue
        else:
            new_obs.append(event)

    # If not at least one future action seen and if this obs is not the final one, empty the list.
    last_event_seen = new_obs[-1].id if new_obs else None
    if not at_least_one_future_action_seen and not is_finished(observation, last_event_seen=last_event_seen):
        new_obs.clear()

    return list(reversed(new_obs))


def is_finished(observation: OpenHandsObservation, last_event_seen: EventID | None = None) -> bool:
    conversation_state = observation.conversation_state
    oh_conversation_finished = conversation_state.agent_status == AgentExecutionStatus.FINISHED \
        or conversation_state.agent_status == AgentExecutionStatus.STUCK \
        or conversation_state.agent_status == AgentExecutionStatus.ERROR
    last_event_id = conversation_state.events[-1].id
    platoon_episode_caught_up = last_event_id in (observation.last_step_action_id, observation.last_step_observation_id, last_event_seen)
    return oh_conversation_finished and platoon_episode_caught_up