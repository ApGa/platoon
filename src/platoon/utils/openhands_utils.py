    
from typing import Sequence
from openhands.sdk.event import ActionEvent, Event, EventID, MessageEvent
from openhands.sdk.conversation.base import ConversationStateProtocol
from openhands.sdk.conversation.state import AgentExecutionStatus


def is_action(event: Event) -> bool:
    return isinstance(event, ActionEvent) \
        or (isinstance(event, MessageEvent) and event.source == "agent")


# TODO: Need to consider LLM agent message to user as an action event.
def get_actions_for_last_obs(events: Sequence[Event], last_step_obs_id: EventID | None, require_same_llm_call_id: bool = False) -> list[Event]:
    """Collect ActionEvent(s) that immediately follow a past ObservationEvent and are
    fully observed by a subsequent ObservationBaseEvent referencing them.
    """
    new_actions: list[Event] = list()
    seen_action_ids: set[EventID] = set()
    at_least_one_future_obs_seen = False
    for event in reversed(events):
        if event.id == last_step_obs_id:
            break
        if not isinstance(event, ActionEvent):
            new_actions.clear()
            at_least_one_future_obs_seen = True
            if hasattr(event, "action_id"):
                seen_action_ids.add(event.action_id)
            continue
        else:
            new_actions.append(event)
    print(f"last_step_obs_id: {last_step_obs_id}")
    print(f"new_actions: {new_actions}")
    print(f"seen_action_ids: {seen_action_ids}")
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


def get_obs_for_last_action(conversation_state: ConversationStateProtocol, last_step_action_id: EventID | None) -> list[Event]:
    """Collect event(s) that immediately follow a past ActionEvent and are
    fully observed by a subsequent ObservationBaseEvent referencing them.
    """
    events = conversation_state.events
    finished = is_finished(conversation_state)
    new_obs: list[Event] = list()
    seen_obs_ids: set[EventID] = set()
    at_least_one_future_action_seen = False
    for event in reversed(events):
        if event.id == last_step_action_id:
            break

        if isinstance(event, ActionEvent):
            at_least_one_future_action_seen = True
            new_obs.clear()
            continue
        else:
            new_obs.append(event)

    # If not at least one future action seen and if this obs is not the final one, empty the list.
    if not at_least_one_future_action_seen and not finished:
        new_obs.clear()

    return list(reversed(new_obs))


def is_finished(conversation_state: ConversationStateProtocol) -> bool:
    return conversation_state.agent_status == AgentExecutionStatus.FINISHED \
        or conversation_state.agent_status == AgentExecutionStatus.STUCK \
        or conversation_state.agent_status == AgentExecutionStatus.ERROR