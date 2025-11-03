from __future__ import annotations

import asyncio
from platoon.envs.base import Task
from platoon.envs.openhands.types import OpenHandsObservation
from platoon.utils.openhands_utils import get_actions_for_last_obs
from copy import deepcopy
from platoon.envs.openhands.types import OpenHandsAction
from platoon.utils.openhands_utils import is_finished


class AgentSDKSWEAgent:
    def __init__(self):
        pass

    async def act(self, obs: OpenHandsObservation) -> OpenHandsAction:
        step_actions = get_actions_for_last_obs(
            obs, 
            require_same_llm_call_id=True
        )
        while not step_actions and not is_finished(obs):
            await asyncio.sleep(0.2)
            step_actions = get_actions_for_last_obs(
                obs, 
                require_same_llm_call_id=True
            )
        
        action = OpenHandsAction(action_events=step_actions)

        if step_actions:
            action.misc['completion_id'] = step_actions[-1].llm_response_id

        # TODO: Consider logging usage and model here to be consistent with CodeActAgent.
        # Although, this info is probably already logged by OpenHands in the events.
        return action

    async def reset(self) -> None:
        pass
    
    async def close(self) -> None:
        pass

    # NOTE: OpenHands agents are stateless, so we can probably just return copy of self.
    # TODO: Need to verify above.
    async def fork(self, task: Task) -> AgentSDKSWEAgent:
        return deepcopy(self)
