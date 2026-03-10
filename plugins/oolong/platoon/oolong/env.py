"""Oolong environment for long-context aggregation tasks with recursive agent support."""
from __future__ import annotations

from copy import deepcopy
from rubric.core.checklist import RubricChecklistFast

from platoon.envs.codeact import CodeActEnv, IPythonCodeExecutor
from platoon.envs.base import Task, SubTask
from platoon.agents.actions.common import finish
from platoon.agents.actions.subagent import launch_subagent
from platoon.episode.context import finish_message, current_trajectory, current_trajectory_collection, error_message
from platoon.envs.codeact import safe_asyncio
from platoon.oolong.agent import OolongPromptBuilder
from platoon.oolong.eval_helpers import dnd_process_response, synth_process_response

class OolongCodeExecutor(IPythonCodeExecutor):

    def __init__(self, task: Task):

        self.task = task
        self.context = task.misc['context']
        
        super().__init__(
            task,
            actions=(
                finish,
                safe_asyncio
            ),
            detect_unawaited_async_calls=False
        )

    async def describe_action_space(self) -> str:
        return """Available Actions (python functions):
1. def finish(message: str) -> str
    Complete the task with your answer.
"""
    async def reset(self) -> OolongCodeExecutor:
        await super().reset()
        return self

class OolongRecursiveCodeExecutor(OolongCodeExecutor):

    def __init__(self, task: Task, subagent_max_steps: int | None = 25):
        super().__init__(task)
        self.subagent_max_steps = subagent_max_steps
        self.shell.user_ns['context'] = self.task.misc['context']
        self.shell.user_ns['launch_subagent'] = self.launch_subagent
        self._subagent_stats_this_step: tuple[int, int] = (0, 0)
    
    async def describe_action_space(self) -> str:
        return """Available Actions (python functions):
1. async def launch_subagent(goal: str, context: str) -> str
    Launch a subagent to process a chunk of the context.
    Returns the finish message from the subagent.
    - goal: The goal/instruction for the subagent.
    - context: The chunk of the context to process.

Note: `asyncio` is already imported. Use `await asyncio.gather(...)` to run subtasks in parallel
or `await launch_subagent()` for a single subtask. **Do not forget to await** the results.

2. def finish(message: str) -> str
    Complete the task with your answer.
"""

    async def reset(self) -> OolongRecursiveCodeExecutor:
        await super().reset()
        # Re-inject bindings that are lost when the shell is recreated
        self.shell.user_ns['context'] = self.task.misc['context']
        self.shell.user_ns['launch_subagent'] = self.launch_subagent
        self._subagent_stats_this_step = (0, 0)
        return self

    def reset_subagent_stats(self) -> None:
        """Reset subagent tracking for a new step."""
        self._subagent_stats_this_step = (0, 0)

    def get_subagent_stats(self) -> tuple[int, int]:
        """Get (launched_count, success_count) for current step."""
        return self._subagent_stats_this_step

    async def launch_subagent(self, goal: str, context: str) -> str:
        
        task_misc = deepcopy(self.task.misc)
        task_misc['context'] = context

        # Track trajectories before launch to find the new child
        traj_collection = current_trajectory_collection.get()
        current_traj = current_trajectory.get()
        traj_ids_before = set(traj_collection.trajectories.keys())

        result = await launch_subagent(
            goal=goal,
            max_steps=self.subagent_max_steps,
            task_misc=task_misc
        )

        # --- subagent success tracking ---
        launched, succeeded = self._subagent_stats_this_step
        launched += 1
        for traj_id, traj in traj_collection.trajectories.items():
            if traj_id not in traj_ids_before:
                if traj.parent_info and traj.parent_info.id == current_traj.id:
                    if traj.steps:
                        final_step = traj.steps[-1]
                        reward_misc = final_step.misc.get("reward_misc", {})
                        if reward_misc.get("success", False):
                            succeeded += 1
                    break
        self._subagent_stats_this_step = (launched, succeeded)

        return result

    async def fork(self, task: Task) -> OolongRecursiveCodeExecutor:
        return OolongRecursiveCodeExecutor(
            task,
            subagent_max_steps=self.subagent_max_steps
        )



class OolongEnv(CodeActEnv):
    def __init__(self, task: Task):
        super().__init__(task, OolongCodeExecutor(task))

    async def evaluate(self) -> tuple[float, dict]:
        
        score = 0.0
        reward_misc = {}

        if self._state.finished:
            if isinstance(self._task, SubTask) and self._task.parent_tasks:
                try:
                    rubric_checklist = RubricChecklistFast(self._task.goal)
                    prompt_builder = OolongPromptBuilder()
                    action_history = prompt_builder.build_action_history_description(await self.observe())
                    # Pull messages from episode-level context vars first; fall back to last step if available
                    final_message = finish_message.get() or (self._state.history[-1].misc.get("finish_message") if self._state.history else None)
                    err_message = error_message.get() or (self._state.history[-1].misc.get("error_message") if self._state.history else None)

                    rubric_context = f"We need to judge the performance of an agent on the task.\n\n# Agent Trajectory Info\n## Action History\n{action_history}\n\n## Final Message\n{final_message}\n\n## Error Message\n{err_message}"
                    score, reason = rubric_checklist.evaluate(include_reason=True, context=rubric_context)

                    reward_misc["reason"] = reason
                    reward_misc["rubric_dict"] = rubric_checklist.to_dict()

                except Exception as e:
                    reward_misc["reason"] = f"Failed rubric-based evaluation: {e}"
                    score = 0.
            else:
                try:
                    eval_fn = dnd_process_response if 'real' in self._task.id else synth_process_response
                    eval_result = eval_fn(self._task.misc, finish_message.get())
                    score = eval_result['score']
                    reward_misc['reason'] = "Oolong environment evaluation result."
                    reward_misc['parse_confidence'] = eval_result['parse_confidence']

                except Exception as e:
                    reward_misc["reason"] = f"Failed to evaluate task: {e}"

        reward_misc["reward/success"] = score
        return score, reward_misc

class OolongRecursiveEnv(OolongEnv):
    def __init__(self, task: Task,
        subagent_max_steps: int | None = 25,
        per_step_subagent_success_reward: float = 0.0,
        per_step_subagent_reward_ceiling: float = float('inf')
    ):
        super().__init__(task)
        self._code_executor = OolongRecursiveCodeExecutor(
            task,
            subagent_max_steps=subagent_max_steps
        )
        self.subagent_max_steps = subagent_max_steps
        self._per_step_subagent_success_reward = per_step_subagent_success_reward
        self._per_step_subagent_reward_ceiling = per_step_subagent_reward_ceiling

    async def evaluate(self) -> tuple[float, dict]:
        score, reward_misc = await super().evaluate()

        launched, succeeded = self._code_executor.get_subagent_stats()
        subagent_reward = 0.0
        if self._per_step_subagent_success_reward > 0 and succeeded > 0:
            subagent_reward = min(
                self._per_step_subagent_success_reward * succeeded,
                self._per_step_subagent_reward_ceiling
            )
            score += subagent_reward

        reward_misc["subagent_launched"] = launched
        reward_misc["subagent_succeeded"] = succeeded
        reward_misc["reward/subagent_success"] = subagent_reward

        return score, reward_misc

    async def fork(self, task: Task) -> OolongRecursiveEnv:
        return OolongRecursiveEnv(
            task,
            subagent_max_steps=self.subagent_max_steps
        )


