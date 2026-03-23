    
from typing import Sequence
from openhands.sdk.event import ActionEvent, AgentErrorEvent, Event, EventID, MessageEvent
from openhands.sdk.tool.builtins.finish import FinishAction
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.conversation import ConversationExecutionStatus
from platoon.openhands.types import OpenHandsObservation
from collections import defaultdict


def _is_terminal_status(conversation_state) -> bool:
    """Return True if the conversation execution status is a terminal state."""
    return conversation_state.execution_status in (
        ConversationExecutionStatus.FINISHED,
        ConversationExecutionStatus.STUCK,
        ConversationExecutionStatus.ERROR,
    )

def is_action(event: Event) -> bool:
    return isinstance(event, ActionEvent) \
        or (isinstance(event, MessageEvent) and event.source == "agent")


def group_actions(events: Sequence[Event]):
    """Build a map of llm_response_id -> list of ActionEvent IDs."""
    batches: dict[EventID, list[EventID]] = defaultdict(list)
    action_id_to_response_id: dict[EventID, EventID] = {}
    tool_call_id_to_action_id = {}
    action_id_to_tool_call_id = {}

    for event in events:
        if isinstance(event, ActionEvent) or (isinstance(event, MessageEvent) and event.source == "agent"):
            llm_response_id = event.llm_response_id
            batches[llm_response_id].append(event.id)
            action_id_to_response_id[event.id] = llm_response_id
            if isinstance(event, ActionEvent) and event.tool_call_id is not None:
                tool_call_id_to_action_id[event.tool_call_id] = event.id
                action_id_to_tool_call_id[event.id] = event.tool_call_id

    return batches, action_id_to_response_id, tool_call_id_to_action_id, action_id_to_tool_call_id

def get_actions_for_last_obs(observation: OpenHandsObservation, require_same_llm_call_id: bool = True) -> list[Event]:
    """Collect all Actions between the last observation.last_step_observation_id and the most recent observation, ensuring that all these Actions have a corresponding observation except for messages and finish actions from agent."""
    # from openhands.sdk.context.view import ActionBatch
    events = observation.conversation_state.events
    new_actions: list[Event] = list()
    seen_action_ids: set[EventID] = set()
    at_least_one_future_obs_seen = False
    at_least_one_future_error_event_seen = False
    # action_batch = ActionBatch.from_events(events)
    batches, action_id_to_response_id, tool_call_id_to_action_id, action_id_to_tool_call_id = group_actions(events)
    for event in reversed(events):
        # Only consider events after the last observed event
        if event.id == observation.last_step_observation_id:
            break

        if not is_action(event):
            new_actions.clear() # clear all accumulated actions till now if a non-action event happened before them
            at_least_one_future_obs_seen = True
            if hasattr(event, "action_id"):
                seen_action_ids.add(event.action_id) #always true for observation events
            
            if isinstance(event, AgentErrorEvent) and event.tool_call_id is not None and event.tool_call_id in tool_call_id_to_action_id:
                # If we see an agent error event that references a tool call id, we should consider the corresponding action as having a future observation, since agent error events are a type of observation event that LLM would see and react to and AgentErrorEvents don't terminate agent loop. 
                seen_action_ids.add(tool_call_id_to_action_id[event.tool_call_id])

            if isinstance(event, ConversationErrorEvent):
                at_least_one_future_error_event_seen = True
            continue
        else:
            new_actions.append(event)
            if isinstance(event, MessageEvent) and event.source == "agent":
                seen_action_ids.add(event.id)
                at_least_one_future_obs_seen = True
            elif isinstance(event, ActionEvent) and event.source == "agent" and (
                isinstance(event.action, FinishAction)
                or _is_terminal_status(observation.conversation_state)
            ):
                # The agent submitted a terminal action (built-in FinishAction or
                # a custom tool that set execution_status to FINISHED/STUCK/ERROR,
                # e.g. LocalizationFinishAction).  Treat this the same as a
                # message: mark it as "seen" so the downstream validation logic
                # doesn't clear it for lacking a corresponding observation.
                seen_action_ids.add(event.id)
                at_least_one_future_obs_seen = True

    if len(new_actions) == 0:
        return new_actions
    last_event_seen = new_actions[0].id if new_actions else None

    if not is_finished(observation, last_event_seen=last_event_seen) and not at_least_one_future_error_event_seen:
        for action in new_actions:
            if action.id not in seen_action_ids:
                print(f"Clearing new_actions due to action event that has not been observed in a future observation: {action.id} {action.kind}", flush=True)
                new_actions.clear()
                break

        if not at_least_one_future_obs_seen:
            new_actions.clear()
        
    if require_same_llm_call_id and new_actions:
        llm_call_id = new_actions[0].llm_response_id
        if any(action.llm_response_id != llm_call_id for action in new_actions):
            raise ValueError("Detected at least two actions in a step with differing llm_response_id. "
            "This is unexpected and can lead to undefined behavior.")
        if len(new_actions) != len(batches[llm_call_id]):
            print("Warning: The number of new actions detected does not match the number of actions in the batch for the corresponding llm_response_id. This could indicate that some actions are not being properly observed or that there are unexpected events in the conversation history.", flush=True)

    return list(reversed(new_actions))



# # TODO: Logic can probably be simplified now, by looking at changes in llm_response_id. Anytime llm_response_id changes, we can consider it as a new action.
# def get_actions_for_last_obs(observation: OpenHandsObservation, require_same_llm_call_id: bool = False) -> list[Event]:
#     """Collect Event(s) we consider as actions that immediately follow a past ObservationEvent and are
#     fully observed by a subsequent ObservationBaseEvent referencing them.
#     """
#     events = observation.conversation_state.events
#     new_actions: list[Event] = list()
#     seen_action_ids: set[EventID] = set()
#     at_least_one_future_obs_seen = False
#     at_least_one_future_error_event_seen = False
#     for event in reversed(events):
#         if event.id == observation.last_step_observation_id:
#             break
#         if not is_action(event):
#             new_actions.clear()
#             at_least_one_future_obs_seen = True
#             if hasattr(event, "action_id"):
#                 seen_action_ids.add(event.action_id)
#             # NOTE: Do we need to handle ConversationErrorEvent here?
#             if isinstance(event, AgentErrorEvent) or isinstance(event, ConversationErrorEvent):
#                 at_least_one_future_error_event_seen = True
#             continue
#         else:
#             new_actions.append(event)
#             if isinstance(event, MessageEvent) and event.source == "agent":
#                 # print("IMPORTANT: agent has finished rollout")
#                 seen_action_ids.add(event.id)
#                 at_least_one_future_obs_seen = True
#             elif isinstance(event, ActionEvent) and event.source == "agent" and isinstance(event.action, FinishAction):
#                 # print("IMPORTANT: agent has finished rollout")
#                 seen_action_ids.add(event.id)
    

#     if len(new_actions) == 0:
#         return new_actions
#     last_event_seen = new_actions[-1].id if new_actions else None
#     if not is_finished(observation, last_event_seen=last_event_seen) and not at_least_one_future_error_event_seen:
#         for action in new_actions:
#             if isinstance(action, ActionEvent) and action.id not in seen_action_ids:
#                 print(f"Clearing new_actions due to action event that has not been observed in a future observation: {action.id} {action.kind}", flush=True)
#                 new_actions.clear()
#                 break

#         if not at_least_one_future_obs_seen:
#             new_actions.clear()

#     if require_same_llm_call_id and new_actions:
#         llm_call_id = new_actions[0].llm_response_id
#         if any(action.llm_response_id != llm_call_id for action in new_actions):
#             raise ValueError("Detected at least two actions in a step with differing llm_response_id. "
#             "This is unexpected and can lead to undefined behavior.")

#     return list(reversed(new_actions))


def get_obs_for_last_action(observation: OpenHandsObservation) -> list[Event]:
    """Collect event(s) that immediately follow a past ActionEvent and are
    fully observed by a subsequent ObservationBaseEvent referencing them.
    """
    events = observation.conversation_state.events
    new_obs: list[Event] = list()
    at_least_one_future_action_seen = False
    future_action_seen = None
    for event in reversed(events):
        if event.id == observation.last_step_action_id:
            # print(f"Found matching last_step_action_id: {event.id} {observation.last_step_action_id}")
            break

        if is_action(event):
            at_least_one_future_action_seen = True
            future_action_seen = event
            # print(f"Clearing new_obs due to action event: {event.kind} {[ev.kind for ev in new_obs]}")
            new_obs.clear()
            continue
        else:
            new_obs.append(event)

    # If not at least one future action seen and if this obs is not the final one, empty the list.
    oh_conversation_finished = _is_terminal_status(observation.conversation_state)
    if oh_conversation_finished and len(new_obs) == 0:
        print(f"Conversation is finished and no new obs seen, returning empty obs list.")
        return []
    if len(new_obs) == 0:
        return new_obs
    # last_event_seen = new_obs[0].id
    
    # Check if conversation has finished WITHOUT calling is_finished (to avoid circular dependency)
    conversation_state = observation.conversation_state
    oh_conversation_finished = _is_terminal_status(conversation_state)
    
    if not at_least_one_future_action_seen and not oh_conversation_finished:
        new_obs.clear()

    # state_info = {'at_least_one_future_action_seen': at_least_one_future_action_seen,
    #               'oh_conversation_finished': oh_conversation_finished,
    #               'last_step_action_id': observation.last_step_action_id,
    #               'last_event_seen': last_event_seen,
    #               'future_action_seen': future_action_seen
    #             }

    # if len(new_obs) > 0:
    #     print(f"get_obs_for_last_action returning obs: {[ev.kind for ev in reversed(new_obs)]} oh_conversation_finished: {oh_conversation_finished}")

    return list(reversed(new_obs))


def is_finished(observation: OpenHandsObservation, last_event_seen: EventID | None = None) -> bool:
    conversation_state = observation.conversation_state
    oh_conversation_finished = _is_terminal_status(conversation_state)
    last_event_id = conversation_state.events[-1].id
    assert last_event_id is not None, "Last event in conversation must have a non-None ID"
    valid_ids = [event_id for event_id in [observation.last_step_action_id, observation.last_step_observation_id, last_event_seen] if event_id is not None]
    platoon_episode_caught_up = last_event_id in valid_ids
    if oh_conversation_finished and platoon_episode_caught_up:
        try:
            print(f"is_finished: conversation finished with status {conversation_state.execution_status}, last_event_id: {last_event_id}, valid_ids: {valid_ids}, last_event_seen: {conversation_state.events[-1].kind}")
        except Exception as e:
            print(f"is_finished: conversation finished with status {conversation_state.execution_status}, last_event_id: {last_event_id}, valid_ids: {valid_ids}, unable to print last event kind due to error: {e}")
    # if oh_conversation_finished and not platoon_episode_caught_up:
    #     try:
    #         print(f"is_finished: conversation finished with status {conversation_state.execution_status}, but platoon episode not caught up, last_event_id: {last_event_id}, valid_ids: {valid_ids}, last_event_seen: {conversation_state.events[-1].kind}", flush=True)
    #     except Exception as e:
    #         print(f"is_finished: conversation finished with status {conversation_state.execution_status}, but platoon episode not caught up, last_event_id: {last_event_id}, valid_ids: {valid_ids}, unable to print last event kind due to error: {e}", flush=True)
    return oh_conversation_finished and platoon_episode_caught_up