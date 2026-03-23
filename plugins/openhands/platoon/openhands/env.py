

from __future__ import annotations

from platoon.envs.base import Task
from platoon.openhands.types import OpenHandsObservation, OpenHandsTrajectoryStep, OpenHandsAction
from openhands.sdk.conversation import get_agent_final_response, BaseConversation, Conversation, ConversationExecutionStatus, RemoteConversation
from openhands.sdk.agent import AgentBase
from openhands.sdk.workspace import BaseWorkspace
from copy import deepcopy
from platoon.episode.context import current_trajectory_collection, current_trajectory, finish_message, error_message
from platoon.utils.openhands_utils import get_obs_for_last_action
from platoon.utils.openhands_utils import is_finished
import threading
import asyncio
import concurrent.futures

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
        # NOTE: Do NOT re-import the tool module here — it must be imported via
        # "custom_tools.localization_finish" (the server-side path) at the top of
        # rollout.py so that register_tool stores the correct module qualname.
        self._conversation: BaseConversation = Conversation(agent=self._agent, visualizer=None, workspace=self._workspace, max_iteration_per_run=self._task.max_steps)
        if isinstance(self._conversation, RemoteConversation):
            self._conversation.delete_on_close = True
        self._state = OpenHandsObservation(task=self._task, conversation_state=self._conversation.state)
        # from openhands.sdk.tool import Tool, register_tool
        # from platoon.codescout.custom_tools.localization_finish import LocalizationFinishTool 
        # register_tool(LocalizationFinishTool.name, LocalizationFinishTool)
        # import platoon.codescout.custom_tools.localization_finish
        self._conversation.send_message(self._task.goal)
        # NOTE: Run the conversation in a separate thread to avoid blocking the main thread.
        # Set a 5-min timeout and pass it to conversation.run
        self._run_thread = threading.Thread(target=self._conversation.run, kwargs={'timeout': 300}, daemon=True)
        self._run_thread.start()
        # try:
        #     self._conversation.run()
        # except Exception as e:
        #     pass
        # for event in self._conversation.state.events:
        #     if event.source == "agent":
        #         print(f"Conv. state: {self._conversation.state.execution_status} Initial conversation event: {event}", flush=True)
        # obs_events = get_obs_for_last_action(self._state)
        # print("Observation events:", obs_events)
        # self._state.last_step_observation_id = obs_events[-1].id if obs_events else None
        # from platoon.utils.openhands_utils import get_actions_for_last_obs 
        # while True:
        #     action_events = get_actions_for_last_obs(self._state)
        #     for action_event in action_events:
        #         print(f"Conv. state: {self._conversation.state.execution_status} Initial conversation action event: {action_event}", flush=True)
        #     self._state.last_step_action_id = action_events[-1].id if action_events else None
        #     obs_events = get_obs_for_last_action(self._state)
        #     for obs_event in obs_events:
        #         print(f"Conv. state: {self._conversation.state.execution_status} Initial conversation observation event: {obs_event}", flush=True)
        #     # print("Observation events:", obs_events)
        #     self._state.last_step_observation_id = obs_events[-1].id if obs_events else None
        #     if not obs_events:
        #         break
        # exit()
        # self._run_thread = threading.Thread(target=self._conversation.run, daemon=True)
        # self._run_thread.start()

        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj_collection.set_trajectory_task(traj.id, self._state.task)
        traj.reward = 0.0
        # print(f"Starting env.reset, last step action id: {self._state.last_step_action_id}")
        obs_events = get_obs_for_last_action(self._state)
        while not obs_events:
            await asyncio.sleep(1)
            obs_events = get_obs_for_last_action(self._state)
        # print(f"env.reset adding observation events: {[e.kind for e in obs_events]} for last step action id: {self._state.last_step_action_id}")
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
        
        print(f"waiting to get obs events, curr agent state: {self._conversation.state.execution_status} last step action: {action.action_events[-1]}")
        obs_events = get_obs_for_last_action(self._state)
        while not obs_events and not is_finished(self._state):
            await asyncio.sleep(0.2)
            obs_events = get_obs_for_last_action(self._state)
        print(f"got obs events: {[e.kind for e in obs_events]}")
        # Update last_step_observation_id to the last event we collected
        # This includes all trailing system events when conversation finishes
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
            print("Environment detected finished conversation in env.step", flush=True)
            self._state.finished = True
            agent_final_msg: str | None = get_agent_final_response(self._conversation.state.events)
            if agent_final_msg is None or agent_final_msg.strip() == "":
                agent_final_msg = "No final response from agent."
            finish_message.set(agent_final_msg)
            self._state.misc["finish_message"] = finish_message.get()
            if self._state.conversation_state.execution_status == ConversationExecutionStatus.STUCK:
                error_message.set("Agent got stuck")
                self._state.misc["error_message"] = error_message.get()
            elif self._state.conversation_state.execution_status == ConversationExecutionStatus.ERROR: #TODO: check
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
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    asyncio.wrap_future(future),
                    timeout=180
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
        # TODO: check if cleaning up workspace manually is required -- causes errors -- check this later
        # if isinstance(self._workspace, BaseWorkspace):
        #     await self._workspace.cleanup()

    @staticmethod
    def _close_conversation_sync(conversation: BaseConversation, workspace=None) -> None:
        """Synchronous helper that calls conversation.close() in a background thread.
        This runs outside the asyncio event loop so it cannot be cancelled by
        CancelledError. The DELETE request will always be sent."""
        try:
            conversation.close()
            if workspace is not None:
                workspace.cleanup()
                # workspace.__exit__(None, None, None)
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
