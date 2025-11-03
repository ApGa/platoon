

from __future__ import annotations

from platoon.envs.base import Task
from platoon.envs.openhands.types import OpenHandsObservation
from openhands.sdk.conversation import get_agent_final_response
from openhands.sdk.conversation.base import BaseConversation
from openhands.sdk.agent.base import AgentBase
from openhands.sdk.workspace.base import BaseWorkspace
from copy import deepcopy
from openhands.sdk.conversation.conversation import Conversation
from platoon.episode.context import current_trajectory_collection, current_trajectory, finish_message, error_message
from platoon.utils.openhands_utils import get_obs_for_last_action
from platoon.envs.openhands.types import OpenHandsTrajectoryStep
from platoon.utils.openhands_utils import is_finished
from platoon.envs.openhands.types import OpenHandsAction
from openhands.sdk.conversation.state import AgentExecutionStatus
import threading
import asyncio

class OpenHandsEnv:
    def __init__(self, task: Task, agent: AgentBase, workspace: str | BaseWorkspace):
        self._task = task
        self._agent = agent
        if not isinstance(workspace, BaseWorkspace):
            workspace = str(workspace)
        self._workspace = workspace
        self._conversation = None
    
    async def reset(self) -> OpenHandsObservation:
        self._conversation: BaseConversation = Conversation(agent=self._agent, workspace=self._workspace, visualize=False, max_iteration_per_run=self._task.max_steps)
        self._state = OpenHandsObservation(task=self._task, conversation_state=self._conversation.state)
        self._conversation.send_message(self._task.goal)
        # NOTE: Run the conversation in a separate thread to avoid blocking the main thread.
        threading.Thread(target=self._conversation.run, daemon=True).start() 

        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj_collection.set_trajectory_task(traj.id, self._state.task)
        traj.reward = 0.0
        obs_events = get_obs_for_last_action(self._state)
        while not obs_events:
            await asyncio.sleep(1)
            obs_events = get_obs_for_last_action(self._state)
        traj_collection.add_trajectory_step(traj.id, OpenHandsTrajectoryStep(
            observation_events=obs_events,
        ))
        self._state.last_step_observation_id = obs_events[-1].id
        return await self.observe()

    async def evaluate(self) -> tuple[float, dict]:
        return 0., {}

    async def step(self, action: OpenHandsAction) -> OpenHandsObservation:
        if action.action_events:
            self._state.last_step_action_id = action.action_events[-1].id
        obs_events = get_obs_for_last_action(self._state)
        while not obs_events and not is_finished(self._state):
            await asyncio.sleep(0.2)
            obs_events = get_obs_for_last_action(self._state)
        if obs_events:
            self._state.last_step_observation_id = obs_events[-1].id
        step = OpenHandsTrajectoryStep(
            action_events=action,
            observation_events=obs_events,
        )
        step.misc['action_misc'] = action.misc
        step.reward, reward_info = await self.evaluate()
        step.misc['reward_misc'] = reward_info
        self._state.reward += step.reward
        
        if is_finished(self._state):
            self._state.finished = True
            finish_message.set(get_agent_final_response(self._conversation.state.events))
            self._state.misc["finish_message"] = finish_message.get()
            if self._state.conversation_state.agent_status == AgentExecutionStatus.STUCK:
                error_message.set("Agent got stuck")
                self._state.misc["error_message"] = error_message.get()

        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj_collection.add_trajectory_step(traj.id, step)
        if self._state.finished:
            traj.reward = self._state.reward
        return await self.observe()

    
    async def close(self) -> None:
        if self._conversation is not None:
            self._conversation.close()
        self._conversation = None

    # TODO: Consider adding a return_copy option here.
    async def observe(self) -> OpenHandsObservation:
        return self._state
    
    @property
    def task(self) -> Task:
        return self._task

    async def fork(self, task: Task) -> OpenHandsEnv:
        # NOTE: The agent might have state, during the copy, but should be reinitialized before use withenv.reset().
        # TODO: Need to double-check that this works for remote agent server case.
        # TODO: Consider explicitly resetting the agent here manually.
        return type(self)(task=task, agent=deepcopy(self._agent), workspace=self._workspace)
