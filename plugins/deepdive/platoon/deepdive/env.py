from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any

from platoon.envs.codeact import IPythonCodeExecutor, safe_asyncio, CodeActEnv
from platoon.envs.base import SubTask, Task
from platoon.agents.actions.common import finish
from platoon.agents.actions.subagent import launch_subagent
from platoon.episode.context import current_trajectory, current_trajectory_collection, error_message, finish_message
from rubric.core.checklist import RubricChecklistFast
from rubric.utils.llm_tools import create_llm_client as rubric_create_llm_client

from .agent import DeepDivePromptBuilder
from .search_tools import search_web, view_webpage_content

class DeepDiveCodeExecutor(IPythonCodeExecutor):
    def __init__(self, task: Task):
        super().__init__(
            task,
            actions=(
                search_web,
                view_webpage_content,
                finish,
                safe_asyncio
            ),
            detect_unawaited_async_calls=True,
            detect_while_loops=True,
            detect_interactive_input=True,
        )

    async def describe_action_space(self) -> str:
        return """Available Actions (python functions):
1. async def search_web(query: str, max_results: int = 5) -> dict:
   Search the web for information related to the query.
    Args:
        query: The query to search for.
        max_results: (optional) The maximum number of results to return. Betweem 1 and 20. Defaults to 5.

    Returns:
        A dictionary containing the search results in the following format:
            {
                "query": str,
                "follow_up_questions": list[str],
                "answer": str,
                "images": list[str],
                "results": list[dict],
                "response_time": float,
                "request_id": str,
            }

            A single result is a dictionary with the following keys:
            {
                "url": str,
                "title": str,
                "content": str,
                "score": float,
                "raw_content": str | None,
            }

2. async def view_webpage_content(url: str) -> str:
View the content of a webpage.
Args:
    url: The URL of the webpage to view.

Returns:
    A string containing the content of the webpage. This may be very long.
    It is wise to first inspect the size of the response before deciding to print it out
    as it may exceed your context window. A good rule of thumb: if the response is greater than
    32K characters, you may want to just look at the first 32K characters or take some other 
    reasonable approach to avoid printing out the entire response.

3. def finish(message: str) -> str
    Complete the task with your answer. **Note that this is a synchronous function and so you should not await it.**
"""

    async def reset(self) -> DeepDiveCodeExecutor:
        await super().reset()
        return self


class DeepDiveRecursiveCodeExecutor(DeepDiveCodeExecutor):
    def __init__(self, task: Task, subagent_max_steps: int | None = 25):
        self.subagent_max_steps = subagent_max_steps
        self._launched_subagent_ids_this_step: set[str] = set()
        self._subagent_success_by_child_this_step: dict[str, float] = {}
        super().__init__(task)
        self.actions = (
            self.launch_subagent,
            search_web,
            view_webpage_content,
            finish,
            safe_asyncio,
        )
        self.shell = self._create_shell()

    async def describe_action_space(self) -> str:
        return """Available Actions (python functions):

1. async def launch_subagent(goal: str) -> Any:
    Launch a subagent to solve a subtask.
    Args:
        goal: The instruction for the subagent. This can be a simple or compound task. 
        Subagents have the ability to recursively delegate tasks to other subagents.
        Also specify the format and type of the answer you expect from the subageent.

    Returns:
        The answer from the subagent. Should be in the format and type specified in the goal.

    Example:

    ps5_price_range = launch_subagent("Find the price range of a PS5 across sony, bestbuy, amazon and gamestop. Return the answer as a string of the form '$$$ - $$$$'.")
    switch2_price_range = launch_subagent("Find the price range of a Switch 2 across nintendo, bestbuy, amazon and gamestop. Return the answer as a string of the form '$$$ - $$$$'.")
    results = await asyncio.gather(ps5_price_range, switch2_price_range, return_exceptions=True)
    print(f"The price range of a PS5 is {results[0]}")
    print(f"The price range of a Switch 2 is {results[1]}")
        

Note: `asyncio` is already imported. Use `await asyncio.gather(...)` to run subagents in parallel
or `await launch_subagent(goal)` for a single subagent. **Do not forget to await** the results.

2. async def search_web(query: str, max_results: int = 5) -> dict:
   Search the web for information related to the query.
    Args:
        query: The query to search for.
        max_results: (optional) The maximum number of results to return. Betweem 1 and 20. Defaults to 5.

    Returns:
        A dictionary containing the search results in the following format:
            {
                "query": str,
                "follow_up_questions": list[str],
                "answer": str,
                "images": list[str],
                "results": list[dict],
                "response_time": float,
                "request_id": str,
            }

            A single result is a dictionary with the following keys:
            {
                "url": str,
                "title": str,
                "content": str,
                "score": float,
                "raw_content": str | None,
            }

3. async def view_webpage_content(url: str) -> str:
View the content of a webpage.
Args:
    url: The URL of the webpage to view.

Returns:
    A string containing the content of the webpage. This may be very long.
    It is wise to first inspect the size of the response before deciding to print it out
    as it may exceed your context window. A good rule of thumb: if the response is greater than
    32K characters, you may want to just look at the first 32K characters or take some other 
    reasonable approach to avoid printing out the entire response.

4. def finish(message: str) -> str
    Complete the task with your answer. **Note that this is a synchronous function and so you should not await it.**
"""

    async def reset(self) -> DeepDiveRecursiveCodeExecutor:
        await super().reset()
        self.reset_subagent_stats()
        return self

    def reset_subagent_stats(self) -> None:
        self._launched_subagent_ids_this_step.clear()
        self._subagent_success_by_child_this_step.clear()

    def get_subagent_stats(self) -> tuple[int, float]:
        return len(self._launched_subagent_ids_this_step), sum(self._subagent_success_by_child_this_step.values())

    async def launch_subagent(self, goal: str) -> Any:
        traj_collection = current_trajectory_collection.get()
        current_traj = current_trajectory.get()
        traj_ids_before = set(traj_collection.trajectories.keys())

        result = await launch_subagent(
            goal=goal,
            max_steps=self.subagent_max_steps,
            task_misc=deepcopy(self.task.misc),
            verbose=False,
        )

        candidate_children: list[str] = []

        for traj_id, traj in traj_collection.trajectories.items():
            if traj_id in traj_ids_before or traj_id in self._launched_subagent_ids_this_step:
                continue
            if not traj.parent_info or traj.parent_info.id != current_traj.id:
                continue
            if traj.task is not None and traj.task.goal == goal:
                candidate_children.append(traj_id)

        if not candidate_children:
            for traj_id, traj in traj_collection.trajectories.items():
                if traj_id in traj_ids_before or traj_id in self._launched_subagent_ids_this_step:
                    continue
                if traj.parent_info and traj.parent_info.id == current_traj.id:
                    candidate_children.append(traj_id)

        if candidate_children:
            child_id = candidate_children[0]
            self._launched_subagent_ids_this_step.add(child_id)
            child_traj = traj_collection.trajectories[child_id]
            success_reward = 0.0
            if child_traj.steps:
                reward_misc = child_traj.steps[-1].misc.get("reward_misc", {})
                success_reward = float(reward_misc.get("reward/success", 0.0))
            self._subagent_success_by_child_this_step[child_id] = success_reward
        return result

    async def fork(self, task: Task) -> DeepDiveRecursiveCodeExecutor:
        return DeepDiveRecursiveCodeExecutor(
            task=task,
            subagent_max_steps=self.subagent_max_steps
        )

class DeepDiveEnv(CodeActEnv):
    def __init__(self, task: Task, skip_subagent_reward_computation: bool = False):
        #task.fork_strategy = "task"
        super().__init__(task, DeepDiveCodeExecutor(task))
        self._skip_subagent_reward_computation = skip_subagent_reward_computation

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
        success_flag = rubric_dict['success']
        if isinstance(success_flag, bool):
            score = 1.0 if success_flag else 0.0
            reason = rubric_dict['reason']
        else:
            reason = "Success flag is not a boolean value"
            score = 0.0

        return score, reason

    async def evaluate(self) -> tuple[float, dict]:
        score, reward_misc = 0., {}
        is_subagent_task = "deepdive" not in (self._task.id or "")
        if self._skip_subagent_reward_computation and is_subagent_task:
            reward_misc["reason"] = "Skipped subagent reward computation"
            reward_misc["success"] = False
            reward_misc["reward/success"] = 0.0
            return 0.0, reward_misc

        final_message = finish_message.get()
        if final_message is None and self._state.history:
            final_message = self._state.history[-1].misc.get("finish_message")
        err_message = error_message.get()
        if err_message is None and self._state.history:
            err_message = self._state.history[-1].misc.get("error_message")

        if self._state.finished:
            if 'deepdive' not in self._task.id:
                try:
                    rubric_checklist = RubricChecklistFast(self._task.goal)
                    prompt_builder = DeepDivePromptBuilder()
                    action_history = prompt_builder.build_action_history_description(await self.observe())
                    
                    rubric_context = f"We need to judge the performance of an agent on the task. The agent may use subagents to solve parts of the task. Do not penalize the model for relying on subagents, unless the subtasks delegated to the subagents are not meaningful or useful for the task.\n\n# Agent Trajectory Info\n## Action History\n{action_history}\n\n## Final Message\n{final_message}\n\n## Error Message\n{err_message}"
                    score, reason = await rubric_checklist.aevaluate(include_reason=True, context=rubric_context)

                    reward_misc["reason"] = reason
                    reward_misc["rubric_dict"] = rubric_checklist.to_dict()

                except Exception as e:
                    reward_misc["reason"] = f"Failed rubric-based evaluation: {e}"
                    score = 0.
            else:
                try:
                    if not final_message:
                        score, reason = 0.0, "No final message provided"
                    else:
                        ground_truth = self._task.misc["ground_truth"]
                        rubric_client = rubric_create_llm_client()
                        rubric_system_prompt = (
                            "We need to judge the performance of an deepresearch agent on a task. The task requires searching the web for information across various sources and synthesizing information together to answer a question.\n"
                            "The agent may use subagents to solve parts of the task. Do not penalize the model for relying on subagents, unless the subtasks delegated to the subagents are not meaningful or useful for the task.\n"
                            "You will be given the ground truth answer to the task and the agent's answer to the task.\n"
                            "When comparing the agent's answer to the ground truth answer, it is acceptable to have minor formatting differences as long as the core information is equivalent."
                            "Please provide a reason and success flag (boolean value) in the following format:\n"
                            "```json\n"
                            "{\n"
                            "    \"reason\": \"Brief reasoning for success flag here.\",\n"
                            "    \"success\": <true|false>\n"
                            "}\n"
                        )

                        rubric_user_prompt = f"Ground truth answer: {ground_truth}\n\nAgent's answer: {final_message}"
                        rubric_response = await rubric_client.asystem_completion(
                            system_prompt=rubric_system_prompt,
                            user_prompt=rubric_user_prompt,
                            temperature=1
                        )
                        score, reason = self.parse_rubric_response(rubric_response)
                        reward_misc["reason"] = reason
                except Exception as e:
                    reward_misc["reason"] = f"Failed to evaluate task: {e}"

        reward_misc["success"] = bool(score >= 1.0)
        reward_misc["reward/success"] = score
        return score, reward_misc


class DeepDiveRecursiveEnv(DeepDiveEnv):
    def __init__(
        self,
        task: Task,
        subagent_max_steps: int | None = 25,
        skip_subagent_reward_computation: bool = False,
    ):
        super().__init__(task, skip_subagent_reward_computation=skip_subagent_reward_computation)
        self._code_executor = DeepDiveRecursiveCodeExecutor(
            task=task,
            subagent_max_steps=subagent_max_steps
        )
        self.subagent_max_steps = subagent_max_steps

    def _get_subagent_stats_and_reset(self) -> tuple[int, float]:
        stats = self._code_executor.get_subagent_stats()
        self._code_executor.reset_subagent_stats()
        return stats

    async def evaluate(self) -> tuple[float, dict]:
        score, reward_misc = await super().evaluate()

        launched, success_total = self._get_subagent_stats_and_reset()
        reward_misc["reward/subagent_launched"] = launched
        reward_misc["reward/subagent_succeeded"] = success_total
        return score, reward_misc

    async def fork(self, task: Task) -> DeepDiveRecursiveEnv:
        return DeepDiveRecursiveEnv(
            task=task,
            subagent_max_steps=self.subagent_max_steps,
            skip_subagent_reward_computation=self._skip_subagent_reward_computation,
        )
