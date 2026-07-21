from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from copy import deepcopy
from pathlib import Path

from platoon.envs.base import Task
from platoon.openhands.types import OpenHandsAction, OpenHandsObservation
from platoon.utils.openhands_utils import get_actions_for_last_obs, is_finished


def _write_act_debug_snapshot(
    *,
    obs: OpenHandsObservation,
    step_actions: list,
    agent_debug_uuid: uuid.UUID,
    act_call_index: int,
    snapshot_index: int,
    elapsed: float,
    message: str,
) -> None:
    if os.environ.get("PLATOON_OPENHANDS_ACT_DEBUG_DIR") is None:
        return
    conversation_state = obs.conversation_state
    non_refreshed_execution_state = getattr(conversation_state, "execution_status", None)
    refresh_from_server = getattr(conversation_state, "refresh_from_server", None)
    if callable(refresh_from_server):
        try:
            refreshed_execution_state = refresh_from_server().get("execution_status")
        except Exception as exc:
            refreshed_execution_state = f"refresh_failed:{type(exc).__name__}:{exc}"
    else:
        refreshed_execution_state = "refresh_unavailable"
    events = list(conversation_state.events)
    event_ids = [str(event.id) for event in events]
    event_stream = [
        {
            "index": index,
            "id": str(event.id),
            "type": type(event).__name__,
            "kind": str(getattr(event, "kind", None)),
            "source": str(getattr(event, "source", None)),
            "action_id": str(event.action_id) if getattr(event, "action_id", None) is not None else None,
        }
        for index, event in enumerate(events)
    ]
    event_index_by_id = {event_id: index for index, event_id in enumerate(event_ids)}
    last_step_observation_id = str(obs.last_step_observation_id) if obs.last_step_observation_id is not None else None
    last_step_action_id = str(obs.last_step_action_id) if obs.last_step_action_id is not None else None
    debug_payload = {
        "message": message,
        "agent_debug_uuid": str(agent_debug_uuid),
        "act_call_index": act_call_index,
        "elapsed_seconds": round(elapsed, 3),
        "non_refreshed_execution_state": str(non_refreshed_execution_state),
        "refreshed_execution_state": str(refreshed_execution_state),
        "last_step_observation_id": last_step_observation_id,
        "last_step_observation_index": event_index_by_id.get(last_step_observation_id),
        "last_step_action_id": last_step_action_id,
        "last_step_action_index": event_index_by_id.get(last_step_action_id),
        "event_count": len(events),
        "event_ids": event_ids,
        "event_stream": event_stream,
        "step_action_ids": [str(action.id) for action in step_actions],
    }
    debug_dir = Path(os.environ.get("PLATOON_OPENHANDS_ACT_DEBUG_DIR", "openhands_act_poll_debug"))
    act_debug_dir = debug_dir / str(agent_debug_uuid) / f"{act_call_index:06d}"
    act_debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = act_debug_dir / f"{snapshot_index:04d}_{int(elapsed * 1000)}ms.json"
    debug_path.write_text(json.dumps(debug_payload, indent=2) + "\n")


class OpenHandsAgent:
    def __init__(self):
        self._debug_uuid = uuid.uuid4()
        self._act_call_count = 0

    async def act(self, obs: OpenHandsObservation) -> OpenHandsAction:
        self._act_call_count += 1
        act_call_index = self._act_call_count
        step_actions = get_actions_for_last_obs(obs, require_same_llm_call_id=True)
        poll_start = time.monotonic()
        next_debug_at = 30.0
        debug_snapshot_index = 0
        debug_snapshot_index += 1
        _write_act_debug_snapshot(
            obs=obs,
            step_actions=step_actions,
            agent_debug_uuid=self._debug_uuid,
            act_call_index=act_call_index,
            snapshot_index=debug_snapshot_index,
            elapsed=time.monotonic() - poll_start,
            message="OpenHandsAgent.act initial get_actions_for_last_obs result",
        )
        if step_actions:
            debug_snapshot_index += 1
            _write_act_debug_snapshot(
                obs=obs,
                step_actions=step_actions,
                agent_debug_uuid=self._debug_uuid,
                act_call_index=act_call_index,
                snapshot_index=debug_snapshot_index,
                elapsed=time.monotonic() - poll_start,
                message="OpenHandsAgent.act returned actions immediately",
            )
        while not step_actions and not is_finished(obs):
            await asyncio.sleep(0.2)
            step_actions = get_actions_for_last_obs(obs, require_same_llm_call_id=True)
            elapsed = time.monotonic() - poll_start
            if elapsed >= next_debug_at:
                debug_snapshot_index += 1
                _write_act_debug_snapshot(
                    obs=obs,
                    step_actions=step_actions,
                    agent_debug_uuid=self._debug_uuid,
                    act_call_index=act_call_index,
                    snapshot_index=debug_snapshot_index,
                    elapsed=elapsed,
                    message="OpenHandsAgent.act waiting for next action",
                )
                next_debug_at += 30.0

        action = OpenHandsAction(action_events=step_actions)

        if step_actions:
            action.misc["completion_id"] = step_actions[-1].llm_response_id

        # TODO: Consider logging usage and model here to be consistent with CodeActAgent.
        # Although, this info is probably already logged by OpenHands in the events.
        return action

    async def reset(self) -> None:
        pass

    async def close(self) -> None:
        pass

    # NOTE: OpenHands agents are stateless, so we can probably just return copy of self.
    # TODO: Need to verify above.
    async def fork(self, task: Task) -> OpenHandsAgent:
        return deepcopy(self)
