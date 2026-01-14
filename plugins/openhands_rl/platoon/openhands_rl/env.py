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
from platoon.utils.openhands_utils import get_obs_for_last_action

logger = get_logger(__name__)

# TODO: double-check if we really need to over-ride any other methods from OpenHandsEnv
# NOTE: The primary job of this class is to implement the step-wise reward functionality.
class OpenHandsRLEnv(OpenHandsEnv):
    async def evaluate(self) -> tuple[float, dict]:
        raise NotImplementedError("OpenHandsRLEnv does not implement evaluate() --> this is something which Aditya will do soon.")
        return 0., {}
