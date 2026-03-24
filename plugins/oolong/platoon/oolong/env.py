"""Oolong environment for long-context aggregation tasks with recursive agent support."""

from __future__ import annotations

import json
import re
from copy import deepcopy

from rubric.utils.llm_tools import create_llm_client as rubric_create_llm_client

from platoon.agents.actions.common import finish
from platoon.agents.actions.subagent import launch_subagent
from platoon.envs.base import Task
from platoon.envs.codeact import CodeActEnv, IPythonCodeExecutor, safe_asyncio
from platoon.episode.context import (
    current_trajectory,
    current_trajectory_collection,
    error_message,
    finish_message,
)
from platoon.oolong.agent import OolongPromptBuilder
from platoon.oolong.eval_helpers import dnd_process_response, synth_process_response
from platoon.utils.span_profile import profile_span


class OolongCodeExecutor(IPythonCodeExecutor):
    def __init__(self, task: Task):
        self.task = task
        self.context = task.misc["context"]

        super().__init__(
            task,
            actions=(finish, safe_asyncio),
            detect_unawaited_async_calls=True,
            detect_while_loops=True,
            detect_interactive_input=True,
        )
        self.shell.user_ns["context"] = self.context

    async def describe_action_space(self) -> str:
        return """Available Actions (python functions):
1. def finish(message: str) -> str
    Complete the task with your answer.
"""

    async def reset(self) -> OolongCodeExecutor:
        await super().reset()
        # Re-inject bindings that are lost when the shell is recreated.
        self.shell.user_ns["context"] = self.context
        return self


class OolongRecursiveCodeExecutor(OolongCodeExecutor):
    def __init__(self, task: Task, subagent_max_steps: int | None = 15):
        super().__init__(task)
        self.subagent_max_steps = subagent_max_steps
        self.shell.user_ns["context"] = self.context
        self.shell.user_ns["launch_subagent"] = self.launch_subagent
        self._launched_subagent_ids_this_step: set[str] = set()
        self._subagent_success_by_child_this_step: dict[str, float] = {}

    async def describe_action_space(self) -> str:
        return """Available Actions (python functions):
1. async def launch_subagent(goal: str, context: str) -> Any
    Launch a subagent to process a chunk of the context.
    Returns the result of the subagent's execution with return type specified in the goal (if specified, else str)
    - goal: The goal/instruction for the subagent. Tell the subagent what information you want and specify the exact format and type in which you expect the subagent to return its result.
    - context: The chunk of the context to process.

    Examples:
    <python>
    result = await launch_subagent(goal="Find the last two spells mentioned in the context by printing out the context and manually reading it. Return the spells as a stringified list of spells that I can parse with json.loads()", context=context_chunk)
    print(f"Subagent result: {result}")
    </python>

Note: `asyncio` is already imported. Use `await asyncio.gather(...)` to run subtasks in parallel
or `await launch_subagent()` for a single subtask. **Do not forget to await** the function call.

2. def finish(result: Any) -> Any
    Complete the task with your result.
"""

    async def reset(self) -> OolongRecursiveCodeExecutor:
        await super().reset()
        # Re-inject bindings that are lost when the shell is recreated
        self.shell.user_ns["context"] = self.context
        self.shell.user_ns["launch_subagent"] = self.launch_subagent
        self.reset_subagent_stats()
        return self

    def reset_subagent_stats(self) -> None:
        """Reset subagent tracking for a new step."""
        self._launched_subagent_ids_this_step.clear()
        self._subagent_success_by_child_this_step.clear()

    def get_subagent_stats(self) -> tuple[int, float]:
        """Get (unique launched children, summed child success score) for current step."""
        return len(self._launched_subagent_ids_this_step), sum(
            self._subagent_success_by_child_this_step.values()
        )

    async def launch_subagent(self, goal: str, context: str = "") -> str:
        task_misc = deepcopy(self.task.misc)
        task_misc["context"] = context

        # Track trajectories before launch to find the new child
        traj_collection = current_trajectory_collection.get()
        current_traj = current_trajectory.get()
        traj_ids_before = set(traj_collection.trajectories.keys())

        result = await launch_subagent(
            goal=goal, max_steps=self.subagent_max_steps, task_misc=task_misc, verbose=False
        )

        # Count each new child trajectory once, even if multiple launch_subagent
        # coroutines complete concurrently during the same parent step.
        for traj_id, traj in traj_collection.trajectories.items():
            if traj_id not in traj_ids_before:
                if traj.parent_info and traj.parent_info.id == current_traj.id:
                    self._launched_subagent_ids_this_step.add(traj_id)
                    success_reward = 0.0
                    if traj.steps:
                        final_step = traj.steps[-1]
                        reward_misc = final_step.misc.get("reward_misc", {})
                        # Preserve partial-credit success scores; only exclude
                        # any separate delegation bonus tracked under other keys.
                        success_reward = float(reward_misc.get("reward/success", 0.0))
                    self._subagent_success_by_child_this_step[traj_id] = success_reward

        return result

    async def fork(self, task: Task) -> OolongRecursiveCodeExecutor:
        return OolongRecursiveCodeExecutor(task, subagent_max_steps=self.subagent_max_steps)


class OolongEnv(CodeActEnv):
    def __init__(self, task: Task, skip_subagent_reward_computation: bool = False):
        task.fork_strategy = "task"
        code_executor = OolongCodeExecutor(task)
        # Remove context task misc to avoid massive context logging in events
        self.context = task.misc.pop('context')
        self._skip_subagent_reward_computation = skip_subagent_reward_computation
        super().__init__(task, code_executor)

    def _parse_rubric_response(self, response: str) -> dict:
        """Parse the LLM response to extract structured data.

        Args:
            response: Raw response from the LLM.

        Returns:
            Parsed response as dictionary.
        """
        try:
            # Try to find JSON code block first
            json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)

            if json_match:
                json_str = json_match.group(1).strip()
            else:
                # Fallback: look for any ``` code block that might contain JSON
                code_match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
                if code_match:
                    json_str = code_match.group(1).strip()
                else:
                    # Last resort: try the entire response as JSON
                    json_str = response.strip()

            # Parse the JSON response
            parsed_response = json.loads(json_str)

            # Validate required fields
            if not isinstance(parsed_response, dict):
                raise ValueError("Response must be a JSON object")

            required_fields = ["reason", "success"]
            for field in required_fields:
                if field not in parsed_response:
                    raise ValueError(f"Missing required field: {field}")

            return parsed_response

        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON response: {str(e)}") from e

    def parse_rubric_response(self, response: str) -> tuple[float, str]:
        rubric_dict = self._parse_rubric_response(response)
        success_flag = rubric_dict["success"]
        if isinstance(success_flag, bool):
            score = 1.0 if success_flag else 0.0
            reason = rubric_dict["reason"]
        else:
            reason = "Success flag is not a boolean value"
            score = 0.0

        return score, reason

    async def evaluate(self) -> tuple[float, dict]:
        score = 0.0
        reward_misc = {}
        is_subagent_task = "oolong" not in (self._task.id or "")
        if self._skip_subagent_reward_computation and is_subagent_task:
            reward_misc["reason"] = "Skipped subagent reward computation"
            reward_misc["reward/success"] = 0.0
            return 0.0, reward_misc

        if self._state.finished:
            if "oolong" not in self._task.id:
                try:
                    # rubric_checklist = RubricChecklistFast(self._task.goal)
                    prompt_builder = OolongPromptBuilder()
                    action_history = prompt_builder.build_action_history_description(
                        await self.observe()
                    )
                    # Preserve falsy-but-meaningful outputs like 0 or ""; only fall back on None.
                    final_message = finish_message.get()
                    if final_message is None and self._state.history:
                        final_message = self._state.history[-1].misc.get("finish_message")
                    err_message = error_message.get()
                    if err_message is None and self._state.history:
                        err_message = self._state.history[-1].misc.get("error_message")

                    rubric_system_prompt = (
                        "We need to judge the performance of an agent on a task. The task relates to aggregating some information from a potentially very large context.\n"
                        "1. Read the context carefully and see if the agent's answer is accurate. 2. Do not mark the agent as successful unless it prints out the context and reads it manually or alternatively uses subagents to answer the question.\n"
                        "For example, if the agent uses regex or string matching/contains logic to answer the question, this is a heuristic that may not be reliable in general and thus should not be marked as successful.\n"
                        "Using subagents is okay since subagents can view the context on behalf of the agent.\n"
                        "Please provide a reason and success flag (boolean value) in the following format:\n"
                        "```json\n"
                        "{\n"
                        '    "reason": "Brief reasoning for success flag here.",\n'
                        '    "success": <true|false>\n'
                        "}\n"
                        "```"
                    )
                    if self.context:
                        prompt_start = f"# Task:\n{self._task.goal}\n\n# Context:\n{self.context}"
                    else:
                        prompt_start = f"# Task:\n{self._task.goal}"
                    rubric_client = rubric_create_llm_client()
                    rubric_context = (
                        f"{prompt_start}\n\n# Agent Trajectory Info\n## Action History\n{action_history}"
                        f"\n\n## Agent Output\n{final_message if final_message is not None else 'No output provided'}"
                        f"\n\n## Error Message\n{err_message if err_message is not None else 'No error message'}."
                    )
                    traj = current_trajectory.get()
                    async with profile_span(
                        "oolong_llm_judge",
                        metadata={
                            "task_id": self._task.id,
                            "trajectory_id": traj.id if traj is not None else None,
                            "parent_trajectory_id": (
                                traj.parent_info.id
                                if traj is not None and traj.parent_info is not None
                                else None
                            ),
                            "action_history_len": len(action_history),
                            "final_message_len": len(str(final_message) or ""),
                            "error_message_len": len(str(err_message) or ""),
                            "rubric_context_len": len(rubric_context),
                        },
                    ):
                        rubric_response = await rubric_client.asystem_completion(
                            system_prompt=rubric_system_prompt,
                            user_prompt=rubric_context,
                            temperature=1,
                        )
                    # rubric_response = "{\"reason\": \"The agent successfully completed the task.\", \"success\": true}"

                    score, reason = self.parse_rubric_response(rubric_response)
                    reward_misc["reason"] = reason
                    # reward_misc["rubric_dict"] = rubric_checklist.to_dict()
                except Exception as e:
                    reward_misc["reason"] = f"Failed rubric-based evaluation: {e}"
                    score = 0.0
            else:
                try:
                    eval_fn = (
                        dnd_process_response if "real" in self._task.id else synth_process_response
                    )
                    eval_result = eval_fn(self._task.misc, str(finish_message.get()))
                    score = eval_result["score"]
                    reward_misc["reason"] = "Oolong environment evaluation result."
                    reward_misc["parse_confidence"] = eval_result["parse_confidence"]

                except Exception as e:
                    reward_misc["reason"] = f"Failed to evaluate task: {e}"

        reward_misc["reward/success"] = score
        return score, reward_misc


class OolongRecursiveEnv(OolongEnv):
    def __init__(
        self,
        task: Task,
        subagent_max_steps: int | None = 25,
        skip_subagent_reward_computation: bool = False,
    ):
        code_executor = OolongRecursiveCodeExecutor(
            task,
            subagent_max_steps=subagent_max_steps
        )
        super().__init__(task, skip_subagent_reward_computation=skip_subagent_reward_computation)
        self._code_executor = code_executor
        self.subagent_max_steps = subagent_max_steps

    def _get_subagent_stats_and_reset(self) -> tuple[int, float]:
        """Get per-step unique launched children and summed child success score."""
        stats = self._code_executor.get_subagent_stats()
        self._code_executor.reset_subagent_stats()
        return stats

    async def evaluate(self) -> tuple[float, dict]:
        score, reward_misc = await super().evaluate()

        launched, success_total = self._get_subagent_stats_and_reset()
        reward_misc["reward/subagent_launched"] = launched
        reward_misc["reward/subagent_succeeded"] = success_total

        return score, reward_misc

    async def fork(self, task: Task) -> OolongRecursiveEnv:
        return OolongRecursiveEnv(
            task,
            subagent_max_steps=self.subagent_max_steps,
            skip_subagent_reward_computation=self._skip_subagent_reward_computation,
        )
