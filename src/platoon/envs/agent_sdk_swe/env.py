from platoon.envs.openhands.env import OpenHandsEnv
from platoon.envs.base import Task
from platoon.envs.openhands.types import OpenHandsObservation
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
import os
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from openhands.sdk import get_logger
from openhands.sdk.conversation import get_agent_final_response

logger = get_logger(__name__)
# NOTE: The below code is borrowed from benchmarks repo.
def get_instruction(
    instance: dict,
    workspace_path: str,
    prompt_path: str
) -> str:
    """Generate user instruction for the agent for SWE-Bench-style tasks."""
    # Set up Jinja2 environment
    prompts_dir = os.path.dirname(prompt_path)
    template_name = os.path.basename(prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        "instance": instance,
        "actual_workspace_path": workspace_path,
    }
    context["test_instructions"] = ""

    # Render the instruction
    instruction = template.render(context)
    return instruction

class AgentSDKSWEEnv(OpenHandsEnv):
    def __init__(self, task: Task, agent: AgentBase, workspace: BaseWorkspace, max_steps: int=100, prompt_filename: str="default.j2"):
        super().__init__(task, agent, workspace)
        # NOTE: max_iterations are controlled via the Task in StepBudgetTracker
        self.max_iterations: int = max_steps
        self._task.max_steps = max_steps
        prompt_dir = (Path(__file__).parent / "prompts").resolve()
        self.prompt_path = prompt_dir / prompt_filename
        assert self.prompt_path.exists(), f"Default prompt path {self.prompt_path} not found"
        self.prompt_path = str(self.prompt_path)
    
    async def reset(self) -> OpenHandsObservation:
        self._conversation: BaseConversation = Conversation(
            agent=self._agent,
            workspace=self._workspace,
            # callbacks=[_log_event], #NOTE: we can configure custom callbacks for each event in the agent's event stream if we want.
            max_iteration_per_run=self.max_iterations, #NOTE: the default is 500 in OpenHands SDK codebase
        )
        self._state = OpenHandsObservation(task=self._task, conversation_state=self._conversation.state)
        instance: dict = self._task.misc
        repo_path = f"/workspace/{instance['repo'].split('/')[-1]}/"
        logger.info(f"Repo path in Remote workspace: {repo_path}")

        cp_testbed_repo = self._workspace.execute_command(
            (f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}")
        )
        assert cp_testbed_repo.exit_code == 0, (
            f"Failed to copy the the repo from /testbed/ to {repo_path}: {cp_testbed_repo.stderr}"
        )

        # git reset
        git_reset = self._workspace.execute_command(f"cd {repo_path} ; git reset --hard")
        assert git_reset.exit_code == 0, f"git reset --hard failed: {git_reset.stderr}"

        # get initial SWE-Bench instruction
        instruction = get_instruction(instance, repo_path, self.prompt_path)
        self._task.goal = instruction #TODO: do we need to set goal before this line for other parts of codebase to work correctly? --> most probably not

        self._conversation.send_message(instruction) #NOTE: this triggers a MessageEvent in the evenstream
        # NOTE: Run the conversation in a separate thread to avoid blocking the main thread.
        # TODO: should daemon be True? Don't we want the main process to wait for this thread to finish before exiting?
        threading.Thread(target=self._conversation.run, daemon=True).start() 
        

        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj_collection.set_trajectory_task(traj.id, self._state.task)
        traj.reward = 0.0
        obs_events = get_obs_for_last_action(self._state)
        # TODO: this loop will not exit till atleast one step is made by the agent. Is this desired??
        while not obs_events:
            await asyncio.sleep(1)
            obs_events = get_obs_for_last_action(self._state)
        traj_collection.add_trajectory_step(traj.id, OpenHandsTrajectoryStep(
            observation_events=obs_events,
        ))
        self._state.last_step_observation_id = obs_events[-1].id
        return await self.observe()

    async def evaluate(self) -> tuple[float, dict]:
        # return 0., {}
        # TODO: implement intermediate rewards here
        raise NotImplementedError("Evaluation for agent-sdk SWE bench is not yet implemented.")

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
        step.reward, step.misc['reward_misc'] = await self.evaluate()
        self._state.reward += step.reward
        
        if is_finished(self._state):
            self._state.finished = True
            finish_message.set(get_agent_final_response(self._conversation.state.events))
            self._state.misc["finish_message"] = finish_message.get()
            if self._state.conversation_state.agent_status == AgentExecutionStatus.STUCK:
                error_message.set("Agent got stuck")
                self._state.misc["error_message"] = error_message.get()
            elif self._state.conversation_state.agent_status == AgentExecutionStatus.ERROR:
                error_message.set("Agent encountered an error")
                self._state.misc["error_message"] = error_message.get()

        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj_collection.add_trajectory_step(traj.id, step)
        if self._state.finished:
            traj.reward = self._state.reward
        return await self.observe()

    async def close(self) -> None:
        self._conversation.close()
        self._workspace.cleanup()
