from __future__ import annotations

import asyncio
import threading
from contextvars import copy_context
from dataclasses import replace
from typing import Any, Callable
from uuid import UUID, uuid4

from platoon.agents.actions.subagent import SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY
from platoon.envs.base import SubTask, Task
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

from .recursive import DEFAULT_SUBAGENT_MAX_STEPS, copy_agent_config_for_fork
from .types import OpenHandsAction, OpenHandsObservation, OpenHandsTrajectoryStep

_CONVERSATION_CLOSE_TIMEOUT_SECONDS = 10.0


async def _wait_for_thread(thread: threading.Thread, timeout: float) -> bool:
    """Wait without registering a possibly stuck thread in asyncio's executor.

    ``asyncio.to_thread`` cannot stop its worker when the timeout expires, and
    ``asyncio.run`` subsequently waits for every default-executor worker.  A
    hung SDK ``close()`` could therefore defeat the rollout deadline.  The
    OpenHands cleanup threads are daemon threads; bounded polling lets the
    rollout subprocess return, with its process-level deadline as a backstop.
    """

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while thread.is_alive():
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(0.1, remaining))
    return True


def _conversation_execution_status(conversation_state) -> ConversationExecutionStatus | None:
    return (
        getattr(conversation_state, "agent_status", None)
        or getattr(conversation_state, "execution_status", None)
        or getattr(conversation_state, "agent_state", None)
    )


def _condensation_completion_id(event: Event) -> str | None:
    if getattr(event, "kind", None) != "Condensation":
        return None
    llm_response_id = getattr(event, "llm_response_id", None)
    if llm_response_id is None:
        return None
    return str(llm_response_id)


class OpenHandsEnv:
    def __init__(
        self,
        task: Task,
        agent: AgentBase,
        workspace: str | BaseWorkspace,
        callbacks: list[Callable[[Event], None]] | None = None,
        persistence_dir: str | None = None,
        conversation_id: UUID | str | None = None,
        enable_recursive_subagents: bool = False,
        subagent_default_max_steps: int = DEFAULT_SUBAGENT_MAX_STEPS,
    ):
        self._task = task
        self._agent = agent
        if not isinstance(workspace, BaseWorkspace):
            workspace = str(workspace)
        self._workspace = workspace
        self._callbacks = callbacks or []
        self._persistence_dir = persistence_dir
        self._conversation_id = conversation_id
        self._conversation = None
        self._conversation_task: asyncio.Task[None] | None = None
        self._synthetic_condensation_step_event_ids: set[str] = set()
        self._enable_recursive_subagents = enable_recursive_subagents
        self._subagent_default_max_steps = subagent_default_max_steps
        self._launch_subagent_runtime: Any | None = None

    def _close_launch_subagent_runtime(self) -> None:
        if self._launch_subagent_runtime is not None:
            self._launch_subagent_runtime.close()
            self._launch_subagent_runtime = None

    async def _aclose_launch_subagent_runtime(self) -> None:
        runtime = self._launch_subagent_runtime
        self._launch_subagent_runtime = None
        if runtime is not None:
            await runtime.aclose()

    def _prepare_agent_for_conversation(self) -> AgentBase:
        configured_agent = self._agent
        if isinstance(self._task, SubTask):
            from platoon.openhands.recursive import with_finish_tool

            configured_agent = with_finish_tool(configured_agent)

        if not self._enable_recursive_subagents or self._task.misc.get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY):
            self._agent = configured_agent
            return self._agent

        from platoon.openhands.recursive import (
            LaunchSubagentRuntime,
            with_launch_subagent_tool,
        )

        self._close_launch_subagent_runtime()
        runtime = LaunchSubagentRuntime()
        runtime.bind(asyncio.get_running_loop(), copy_context())
        self._launch_subagent_runtime = runtime
        self._agent = with_launch_subagent_tool(
            configured_agent,
            runtime=runtime,
            default_max_steps=self._subagent_default_max_steps,
        )
        return self._agent

    def _add_trainable_condensation_steps(
        self,
        traj_collection,
        trajectory_id: str,
        obs_events: list[Event] | None,
    ) -> None:
        for event in obs_events or []:
            completion_id = _condensation_completion_id(event)
            event_id = getattr(event, "id", None)
            if completion_id is None or event_id is None:
                continue
            event_id = str(event_id)
            if event_id in self._synthetic_condensation_step_event_ids:
                continue
            self._synthetic_condensation_step_event_ids.add(event_id)

            step = OpenHandsTrajectoryStep(observation_events=[event])
            step.misc["action_misc"] = {"completion_id": completion_id}
            step.misc["reward_misc"] = {}
            step.misc["synthetic_step_type"] = "openhands_condensation"
            traj_collection.add_trajectory_step(trajectory_id, step)

    async def _initial_user_message(self) -> str:
        return self._task.goal or ""

    async def reset(self) -> OpenHandsObservation:
        self._conversation: BaseConversation = Conversation(
            agent=self._prepare_agent_for_conversation(),
            callbacks=self._callbacks,
            workspace=self._workspace,
            visualizer=None,
            max_iteration_per_run=self._task.max_steps or 500,
            persistence_dir=self._persistence_dir,
            conversation_id=self._conversation_id,
            delete_on_close=False,
        )
        initial_user_message = await self._initial_user_message()
        self._task = replace(self._task, goal=initial_user_message)
        self._state = OpenHandsObservation(task=self._task, conversation_state=self._conversation.state)
        self._conversation.send_message(initial_user_message)
        # The OpenHands fork's async runner tracks its task and cancellation
        # token, allowing interrupt() to stop an in-flight LLM request and
        # pending tools.  Keeping this task on the episode loop also makes
        # teardown observable instead of leaving an opaque sync daemon thread.
        self._conversation_task = asyncio.create_task(
            self._conversation.arun(),
            name=f"openhands-conversation-{self._conversation_id or 'ephemeral'}",
        )

        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj_collection.set_trajectory_task(traj.id, self._state.task)
        traj.reward = 0.0
        obs_events = get_obs_for_last_action(self._state)
        while not obs_events:
            self._raise_if_conversation_failed()
            await asyncio.sleep(1)
            obs_events = get_obs_for_last_action(self._state)
        traj_collection.add_trajectory_step(
            traj.id,
            OpenHandsTrajectoryStep(
                observation_events=obs_events,
            ),
        )
        self._add_trainable_condensation_steps(traj_collection, traj.id, obs_events)
        self._state.last_step_observation_id = obs_events[-1].id
        return await self.observe()

    async def evaluate(self) -> tuple[float, dict]:
        return 0.0, {}

    async def step(self, action: OpenHandsAction) -> OpenHandsObservation:
        if action.action_events:
            self._state.last_step_action_id = action.action_events[-1].id
        obs_events = get_obs_for_last_action(self._state)
        while not obs_events and not is_finished(self._state):
            self._raise_if_conversation_failed()
            await asyncio.sleep(0.2)
            obs_events = get_obs_for_last_action(self._state)
        if obs_events:
            self._state.last_step_observation_id = obs_events[-1].id
        step = OpenHandsTrajectoryStep(
            action_events=action.action_events,
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
        self._add_trainable_condensation_steps(traj_collection, traj.id, obs_events)
        if self._state.finished:
            traj.reward = self._state.reward
        return await self.observe()

    async def close(self) -> None:
        conversation = self._conversation
        conversation_task = self._conversation_task
        self._conversation = None
        self._conversation_task = None
        if conversation is not None:
            try:
                conversation.interrupt()
            except BaseException:
                pass
        if conversation_task is not None and not conversation_task.done():
            conversation_task.cancel()
            done, _pending = await asyncio.wait({conversation_task}, timeout=10.0)
            if done and not conversation_task.cancelled():
                conversation_task.exception()
        elif conversation_task is not None and not conversation_task.cancelled():
            # Retrieve a terminal exception so it does not become an unhandled
            # task warning during event-loop teardown.
            conversation_task.exception()

        # Stop recursive children only after the root conversation has been
        # interrupted, so it cannot interpret child cancellation as a tool
        # error and begin another model step while teardown is in progress.
        await self._aclose_launch_subagent_runtime()
        if conversation is not None:
            close_thread = threading.Thread(
                target=conversation.close,
                daemon=True,
                name=f"openhands-close-{self._conversation_id or 'ephemeral'}",
            )
            close_thread.start()
            await _wait_for_thread(close_thread, _CONVERSATION_CLOSE_TIMEOUT_SECONDS)

    def _raise_if_conversation_failed(self) -> None:
        task = self._conversation_task
        if task is None or not task.done():
            return
        if task.cancelled():
            raise RuntimeError("OpenHands conversation was cancelled before producing an observation")
        exception = task.exception()
        if exception is not None:
            raise exception

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
        return type(self)(
            task=task,
            agent=copy_agent_config_for_fork(self._agent),
            workspace=self._workspace,
            callbacks=self._callbacks,
            persistence_dir=self._persistence_dir,
            conversation_id=uuid4() if self._persistence_dir is not None else None,
            enable_recursive_subagents=self._enable_recursive_subagents,
            subagent_default_max_steps=self._subagent_default_max_steps,
        )
