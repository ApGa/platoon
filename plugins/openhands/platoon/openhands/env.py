from __future__ import annotations

import asyncio
import threading
from copy import deepcopy
import concurrent.futures
from openhands.sdk.agent import AgentBase
from openhands.sdk.conversation import get_agent_final_response, BaseConversation, Conversation, ConversationExecutionStatus, RemoteConversation
from openhands.sdk.workspace import BaseWorkspace
from platoon.envs.base import Task
from platoon.episode.context import (
    current_trajectory,
    current_trajectory_collection,
    error_message,
    finish_message,
)
from platoon.utils.openhands_utils import get_obs_for_last_action, is_finished

from platoon.openhands.types import OpenHandsAction, OpenHandsObservation, OpenHandsTrajectoryStep


class OpenHandsEnv:
    def __init__(self, task: Task, agent: AgentBase, workspace: str | BaseWorkspace):
        self._task = task
        self._agent = agent
        if not isinstance(workspace, BaseWorkspace):
            workspace = str(workspace)
        self._workspace = workspace
        self._conversation = None
        self._run_thread: threading.Thread | None = None
    
    async def reset(self) -> OpenHandsObservation:
        self._conversation: BaseConversation = Conversation(
            agent=self._agent,
            workspace=self._workspace,
            visualizer=None,
            max_iteration_per_run=self._task.max_steps,
        )
        if isinstance(self._conversation, RemoteConversation):
            self._conversation.delete_on_close = True
        self._state = OpenHandsObservation(task=self._task, conversation_state=self._conversation.state)
        self._conversation.send_message(self._task.goal)
        # NOTE: Run the conversation in a separate thread to avoid blocking the main thread.
        self._run_thread = threading.Thread(target=self._conversation.run, kwargs={'timeout': 300}, daemon=True)
        self._run_thread.start()

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
            agent_final_msg: str | None = get_agent_final_response(self._conversation.state.events)
            if agent_final_msg is None or agent_final_msg.strip() == "":
                agent_final_msg = "No final response from agent."
            finish_message.set(agent_final_msg)
            self._state.misc["finish_message"] = finish_message.get()
            if self._state.conversation_state.execution_status == ConversationExecutionStatus.STUCK:
                error_message.set("Agent got stuck")
                self._state.misc["error_message"] = error_message.get()
            elif self._state.conversation_state.execution_status == ConversationExecutionStatus.ERROR:
                error_message.set("Agent encountered an error")
                self._state.misc["error_message"] = error_message.get()

        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj_collection.add_trajectory_step(traj.id, step)
        if self._state.finished:
            traj.reward = self._state.reward
        return await self.observe()

    async def close(self) -> None:
        if self._conversation is not None:
            conversation = self._conversation
            self._conversation = None
            # Fire-and-forget: submit close() to a thread pool so the DELETE
            # request completes even if this coroutine is cancelled by
            # asyncio.wait_for() or CancelledError from the parent task.
            # We use a standalone executor submit (not awaited) so cancellation
            # of this coroutine cannot prevent the HTTP DELETE from being sent.
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(self._close_conversation_sync, conversation, self._workspace)
            try:
                # Give it a reasonable amount of time, but don't block forever
                await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=120
                )
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
                # Even if we're cancelled or timed out, the thread-pool task
                # will still finish in the background (the DELETE gets sent).
                print(f"env.close() interrupted ({type(e).__name__}: {e}), "
                      f"cleanup thread will finish in background", flush=True)
            finally:
                # Don't call executor.shutdown(wait=True) which would block;
                # let the daemon thread finish on its own.
                executor.shutdown(wait=False)
        # Wait briefly for the run-polling thread to notice the conversation
        # was deleted and exit on its own.
        if self._run_thread is not None:
            self._run_thread.join(timeout=5)
            if self._run_thread.is_alive():
                print("Warning: conversation run thread still alive after close()", flush=True)
            self._run_thread = None

    @staticmethod
    def _close_conversation_sync(conversation: BaseConversation, workspace=None) -> None:
        """Synchronous helper that calls conversation.close() in a background thread.
        This runs outside the asyncio event loop so it cannot be cancelled by
        CancelledError. The DELETE request will always be sent."""
        try:
            conversation.close()
            if workspace is not None and not isinstance(workspace, str):
                workspace.cleanup()
        except Exception as e:
            print(f"Error in background conversation.close(): {e}", flush=True)

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
