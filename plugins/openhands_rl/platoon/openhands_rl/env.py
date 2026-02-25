from pathlib import Path
from platoon.utils.openhands_utils import is_finished
from platoon.episode.context import current_trajectory_collection, current_trajectory, finish_message, error_message

from openhands.sdk import get_logger
from platoon.envs.base import Task
from openhands.sdk.agent import AgentBase
from openhands.sdk.workspace import BaseWorkspace
from openhands.sdk.conversation import Conversation, BaseConversation, get_agent_final_response
#TODO: check below imports
from platoon.openhands.env import OpenHandsEnv
from platoon.openhands.types import OpenHandsObservation, OpenHandsAction, OpenHandsTrajectoryStep
import threading
import asyncio
from platoon.utils.openhands_utils import is_finished

logger = get_logger(__name__)

# TODO: double-check if we really need to over-ride any other methods from OpenHandsEnv
# NOTE: The primary job of this class is to implement the step-wise reward functionality.
class OpenHandsRLEnv(OpenHandsEnv):
    async def evaluate(self) -> tuple[float, dict]:
        # return 0., {}
        import random
        # get a random reward between 0 and 0.5 and between 0.5 and 1

        low_reward = random.uniform(0, 0.5)
        high_reward = random.uniform(0.5, 1)
        if not is_finished(self._state):
            return low_reward, {}
        agent_final_msg: str | None = get_agent_final_response(self._conversation.state.events)
        if agent_final_msg is None or agent_final_msg.strip() == "" or "<tool_call>" in agent_final_msg:
            return low_reward, {}
        return high_reward, {}