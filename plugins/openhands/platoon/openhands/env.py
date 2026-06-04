from __future__ import annotations

import asyncio
import threading
from copy import deepcopy
from typing import Callable

from platoon.envs.base import Task
from platoon.episode.context import (
    current_trajectory,
    current_trajectory_collection,
    error_message,
    finish_message,
)
from platoon.utils.openhands_utils import get_obs_for_last_action, is_finished

from openhands.sdk.agent.base import AgentBase
from openhands.sdk.conversation import get_agent_final_response
from openhands.sdk.conversation.base import BaseConversation
from openhands.sdk.conversation.conversation import Conversation
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event.base import Event
from openhands.sdk.workspace.base import BaseWorkspace

from .types import OpenHandsAction, OpenHandsObservation, OpenHandsTrajectoryStep


def _conversation_execution_status(conversation_state) -> ConversationExecutionStatus | None:
    return (
        getattr(conversation_state, "agent_status", None)
        or getattr(conversation_state, "execution_status", None)
        or getattr(conversation_state, "agent_state", None)
    )


class OpenHandsEnv:
    def __init__(
        self,
        task: Task,
        agent: AgentBase,
        workspace: str | BaseWorkspace,
        callbacks: list[Callable[[Event], None]] | None = None,
    ):
        self._task = task
        self._agent = agent
        if not isinstance(workspace, BaseWorkspace):
            workspace = str(workspace)
        self._workspace = workspace
        self._callbacks = callbacks or []
        self._conversation = None

    async def reset(self) -> OpenHandsObservation:
        self._conversation: BaseConversation = Conversation(
            agent=self._agent,
            callbacks=self._callbacks,
            workspace=self._workspace,
            visualizer=None,
            max_iteration_per_run=self._task.max_steps or 500,
        )
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
        traj_collection.add_trajectory_step(
            traj.id,
            OpenHandsTrajectoryStep(
                observation_events=obs_events,
            ),
        )
        self._state.last_step_observation_id = obs_events[-1].id
        return await self.observe()

    async def evaluate(self) -> tuple[float, dict]:
        return 0.0, {}

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
        step.misc["action_misc"] = action.misc
        step.reward, reward_info = await self.evaluate()
        step.misc["reward_misc"] = reward_info
        self._state.reward += step.reward

        if is_finished(self._state):
            self._state.finished = True
            finish_message.set(get_agent_final_response(self._conversation.state.events))
            self._state.misc["finish_message"] = finish_message.get()
            if _conversation_execution_status(self._state.conversation_state) == ConversationExecutionStatus.STUCK:
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
        return type(self)(task=task, agent=deepcopy(self._agent), workspace=self._workspace, callbacks=self._callbacks)
