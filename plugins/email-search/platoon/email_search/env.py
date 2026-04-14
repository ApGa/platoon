from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import re
from typing import Any

from rubric.core.checklist import RubricChecklistFast
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

from .agent import EmailSearchPromptBuilder
from .email_search_tools import read_email as read_email_tool
from .email_search_tools import search_emails as search_emails_tool


@dataclass
class RootTaskMetrics:
    answer_correct: bool = False
    sources_correct: bool = False
    num_turns: int = 0
    attempted_answer: bool = False
    ever_found_right_email: bool = False
    ever_read_right_email: bool = False
    returned_i_dont_know: bool = False
    num_sources: int = 0
    ever_tried_to_read_invalid_email: bool = False

    def to_metrics(self) -> dict[str, float | int]:
        return {key: int(value) for key, value in asdict(self).items()}


class EmailSearchCodeExecutor(IPythonCodeExecutor):
    def __init__(self, task: Task):
        self.task = task
        self._gold_message_ids = {str(message_id) for message_id in task.misc.get("gold_message_ids", [])}
        self._inbox_address = str(task.misc["inbox_address"])
        self.found_target_email = False
        self.read_target_email = False
        self.read_invalid_email = False
        super().__init__(
            task,
            actions=(self.search_emails, self.read_email, finish, safe_asyncio),
            detect_unawaited_async_calls=True,
            detect_while_loops=True,
            detect_interactive_input=True,
        )

    def _create_shell(self):
        shell = super()._create_shell()
        shell.user_ns["asyncio"] = safe_asyncio
        shell.user_ns["json"] = json
        return shell

    async def search_emails(
        self,
        keywords: list[str],
        from_addr: str | None = None,
        to_addr: str | None = None,
        sent_after: str | None = None,
        sent_before: str | None = None,
        max_results: int = 10,
    ) -> list[dict[str, str]]:
        results = await search_emails_tool(
            inbox=self._inbox_address,
            keywords=keywords,
            from_addr=from_addr,
            to_addr=to_addr,
            sent_after=sent_after,
            sent_before=sent_before,
            max_results=max_results,
        )
        if any(result.message_id in self._gold_message_ids for result in results):
            self.found_target_email = True
        return [asdict(result) for result in results]

    async def read_email(self, message_id: str) -> dict[str, Any] | None:
        email = await read_email_tool(message_id)
        if email is None:
            self.read_invalid_email = True
            return None
        if message_id in self._gold_message_ids:
            self.read_target_email = True
        return email.model_dump()

    async def describe_action_space(self) -> str:
        return """Available Actions (python functions):
1. async def search_emails(
       keywords: list[str],
       from_addr: str | None = None,
       to_addr: str | None = None,
       sent_after: str | None = None,
       sent_before: str | None = None,
       max_results: int = 10,
   ) -> list[dict[str, str]]
   Search the current user's inbox.
   - The inbox address is implicit; do not pass it.
   - `keywords` is a list of required search terms; all keywords must match in the subject/body.
   - `from_addr` filters by exact sender email address.
   - `to_addr` filters by exact recipient email address across To/CC/BCC.
   - `sent_after` must be `YYYY-MM-DD` and is inclusive.
   - `sent_before` must be `YYYY-MM-DD` and is exclusive.
   - To search a single day, use `sent_after="YYYY-MM-DD"` and `sent_before` equal to the next day.
   - `max_results` cannot exceed 10.

2. async def read_email(message_id: str) -> dict | None
   Read a full email by message id.
   Returns fields such as `message_id`, `date`, `subject`, `from_address`,
   `to_addresses`, `cc_addresses`, `bcc_addresses`, `body`, and `file_name`.

3. def finish(message: str) -> str
   Complete the task.
   - `json` and `asyncio` are already available in the notebook; no import is needed.
   - For root tasks, `message` must be a JSON string, not a Python dict.
   - Prefer `finish(json.dumps({"answer": "<answer>", "sources": ["<message_id>", ...]}))`.
   - `answer` should directly answer the user question, or be `"I don't know"`.
   - `sources` should list the supporting email message IDs used for the answer.
   - For delegated subtasks, return the intermediate result in whatever format the parent requested.
"""

    async def reset(self) -> EmailSearchCodeExecutor:
        await super().reset()
        self.found_target_email = False
        self.read_target_email = False
        self.read_invalid_email = False
        return self


class EmailSearchRecursiveCodeExecutor(EmailSearchCodeExecutor):
    def __init__(self, task: Task, subagent_max_steps: int | None = 10):
        self.subagent_max_steps = subagent_max_steps
        self._launched_subagent_ids_this_step: set[str] = set()
        self._subagent_success_by_child_this_step: dict[str, float] = {}
        super().__init__(task)
        self.actions = (
            self.launch_subagent,
            self.search_emails,
            self.read_email,
            finish,
            safe_asyncio,
        )
        self.shell = self._create_shell()

    async def describe_action_space(self) -> str:
        return """Available Actions (python functions):
1. async def launch_subagent(goal: str) -> Any
   Launch a subagent to solve a focused email-search subtask.
   Tell the subagent exactly what to return and in what format.

2. async def search_emails(
       keywords: list[str],
       from_addr: str | None = None,
       to_addr: str | None = None,
       sent_after: str | None = None,
       sent_before: str | None = None,
       max_results: int = 10,
   ) -> list[dict[str, str]]
   Search the current user's inbox.
   - The inbox address is implicit; do not pass it.
   - `keywords` is a list of required search terms; all keywords must match in the subject/body.
   - `from_addr` filters by exact sender email address.
   - `to_addr` filters by exact recipient email address across To/CC/BCC.
   - `sent_after` must be `YYYY-MM-DD` and is inclusive.
   - `sent_before` must be `YYYY-MM-DD` and is exclusive.
   - To search a single day, use `sent_after="YYYY-MM-DD"` and `sent_before` equal to the next day.
   - `max_results` cannot exceed 10.

3. async def read_email(message_id: str) -> dict | None
   Read a full email by message id.
   Returns fields such as `message_id`, `date`, `subject`, `from_address`,
   `to_addresses`, `cc_addresses`, `bcc_addresses`, `body`, and `file_name`.

4. def finish(message: str) -> str
   Complete the current task.
   - `json` and `asyncio` are already available in the notebook; no import is needed.
   - For root tasks, `message` must be a JSON string, not a Python dict.
   - Prefer `finish(json.dumps({"answer": "<answer>", "sources": ["<message_id>", ...]}))`.
   - `answer` should directly answer the user question, or be `"I don't know"`.
   - `sources` should list the supporting email message IDs used for the answer.
   - For delegated subtasks, return the intermediate result in whatever format the parent requested.
"""

    async def reset(self) -> EmailSearchRecursiveCodeExecutor:
        await super().reset()
        self.reset_subagent_stats()
        return self

    def reset_subagent_stats(self) -> None:
        self._launched_subagent_ids_this_step.clear()
        self._subagent_success_by_child_this_step.clear()

    def get_subagent_stats(self) -> tuple[int, float]:
        return len(self._launched_subagent_ids_this_step), sum(
            self._subagent_success_by_child_this_step.values()
        )

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

    async def fork(self, task: Task) -> EmailSearchRecursiveCodeExecutor:
        return EmailSearchRecursiveCodeExecutor(task=task, subagent_max_steps=self.subagent_max_steps)


class EmailSearchEnv(CodeActEnv):
    def __init__(self, task: Task):
        task.fork_strategy = "task"
        super().__init__(task, EmailSearchCodeExecutor(task))

    def _parse_json_payload(self, response: str) -> dict[str, Any]:
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        if json_match:
            payload = json_match.group(1).strip()
        else:
            code_match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            payload = code_match.group(1).strip() if code_match else response.strip()
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object")
        return parsed

    def _parse_root_final_message(self, final_message: str | None) -> tuple[str, list[str]]:
        if final_message is None:
            return "", []

        parsed = self._parse_json_payload(final_message)
        answer = parsed.get("answer", "")
        sources = parsed.get("sources", [])
        if not isinstance(answer, str):
            answer = str(answer)
        if not isinstance(sources, list):
            sources = []
        return answer.strip(), [str(source) for source in sources]

    def _parse_rubric_response(self, response: str) -> tuple[float, str]:
        parsed = self._parse_json_payload(response)
        success = parsed.get("success")
        reason = str(parsed.get("reason", ""))
        if not isinstance(success, bool):
            raise ValueError("Rubric success field must be a boolean")
        return (1.0 if success else 0.0), reason

    def _is_root_task(self) -> bool:
        task_id = self._task.id or ""
        return task_id.startswith("email_search.")

    async def _evaluate_subtask(self, final_message: str | None, err_message: str | None) -> tuple[float, dict]:
        reward_misc: dict[str, Any] = {}
        try:
            #rubric_checklist = RubricChecklistFast(self._task.goal)
            rubric_client = rubric_create_llm_client()
            prompt_builder = EmailSearchPromptBuilder()
            action_history = prompt_builder.build_action_history_description(await self.observe())

            rubric_response = await rubric_client.asystem_completion(
                system_prompt=(
                    "We need to judge the performance of an email-search agent on a task. You should mark the task as successful if both the agent's final answer is correct AND if the agent delegates, its delegation strategy is useful and efficient. "
                    "The agent may use subagents to solve parts of the task, and good, useful delegation is allowed. But degenerate delegation strategies should not be rewarded."
                    "Reward delegation only when it is genuinely useful: the subtasks should be concrete, non-overlapping, and should help the agent search or read emails more effectively.\n\n"
                    "Mark the task as unsuccessful for degenerate delegation behavior even if the final answer is correct. "
                    "Degenerate behavior includes repeated delegation of (forwarding) nearly the same/whole goal to a subagent without doing any meaningful search/reading work or decomposition or wasteful failed launches caused by trying to delegate past the depth limit.\n\n"
                    "It is acceptable for the agent to do all the work itself, part of the work itself, or to use subagents heavily if those subagents do distinct useful work. "
                    "Do not give credit just because the wording of child tasks changes slightly from the agent's own goal; judge whether the decomposition is actually useful and efficient. "
                    "Please provide a reason and success flag (boolean value) in the following JSON format:\n"
                    '```json\n{"reason": "Brief reasoning here.", "success": true}\n```'
                ),
                user_prompt=(
                    f"# Agent Trajectory Info\n## Action History\n{action_history}\n\n"
                    f"## Final Message\n{final_message}\n\n"
                    f"## Error Message\n{err_message}"
                ),
                temperature=1,
            )
            score, reason = self._parse_rubric_response(rubric_response)
            reward_misc["reason"] = reason
            reward_misc["reward/success"] = score
            reward_misc["success"] = score >= 1.0
            return score, reward_misc
        except Exception as exc:
            reward_misc["reason"] = f"Failed rubric-based evaluation: {exc}"
            reward_misc["reward/success"] = 0.0
            reward_misc["success"] = False
            return 0.0, reward_misc

    async def _judge_root_answer(self, answer: str) -> tuple[float, str]:
        normalized_truth = " ".join(str(self._task.misc["ground_truth"]).lower().split())
        normalized_answer = " ".join(answer.lower().split())
        if normalized_truth == normalized_answer:
            return 1.0, "Exact normalized match."

        rubric_client = rubric_create_llm_client()
        rubric_system_prompt = (
            "We need to judge the performance of an email-search agent on a task. "
            "You will be given the ground truth answer and the agent answer. "
            "Minor differences are acceptable as long as the core information is equivalent and sufficiently answers the task."
            "Please provide a reason and success flag (boolean value) in the following JSON format:\n"
            '```json\n{"reason": "Brief reasoning here.", "success": true}\n```'
        )
        rubric_user_prompt = (
            f"Task:\n{self._task.goal}\n\n"
            f"Ground truth answer: {self._task.misc['ground_truth']}\n\n"
            f"Agent's answer: {answer}"
        )
        rubric_response = await rubric_client.asystem_completion(
            system_prompt=rubric_system_prompt,
            user_prompt=rubric_user_prompt,
            temperature=1,
        )
        return self._parse_rubric_response(rubric_response)

    async def evaluate(self) -> tuple[float, dict]:
        if not self._state.finished:
            return 0.0, {}

        final_message = finish_message.get()
        if final_message is None and self._state.history:
            final_message = self._state.history[-1].misc.get("finish_message")
        err_message = error_message.get()
        if err_message is None and self._state.history:
            err_message = self._state.history[-1].misc.get("error_message")

        if not self._is_root_task():
            return await self._evaluate_subtask(final_message, err_message)

        reward_misc: dict[str, Any] = {}
        score = 0.0

        try:
            answer, sources = self._parse_root_final_message(final_message)
        except Exception as exc:
            answer, sources = "", []
            reward_misc["reason"] = f"Failed to parse final answer JSON: {exc}"
        else:
            if answer:
                if answer.lower() == "i don't know":
                    reward_misc["reason"] = "Agent returned I don't know."
                else:
                    try:
                        score, reason = await self._judge_root_answer(answer)
                        reward_misc["reason"] = reason
                    except Exception as exc:
                        reward_misc["reason"] = f"Failed to evaluate root task: {exc}"

        gold_message_ids = {str(message_id) for message_id in self._task.misc.get("gold_message_ids", [])}
        metrics = RootTaskMetrics(
            answer_correct=bool(score >= 1.0),
            sources_correct=gold_message_ids.issubset(set(sources)),
            num_turns=len(self._state.history),
            attempted_answer=bool(answer and answer.lower() != "i don't know"),
            ever_found_right_email=self._code_executor.found_target_email,
            ever_read_right_email=self._code_executor.read_target_email,
            returned_i_dont_know=answer.lower() == "i don't know" if answer else False,
            num_sources=len(sources),
            ever_tried_to_read_invalid_email=self._code_executor.read_invalid_email,
        )

        reward_misc.update(metrics.to_metrics())
        reward_misc["answer"] = answer
        reward_misc["sources"] = sources
        reward_misc["error_message"] = err_message
        reward_misc["reward/success"] = score
        reward_misc["success"] = bool(score >= 1.0)
        return score, reward_misc


class EmailSearchRecursiveEnv(EmailSearchEnv):
    def __init__(self, task: Task, subagent_max_steps: int | None = 15):
        super().__init__(task)
        self._code_executor = EmailSearchRecursiveCodeExecutor(
            task=task,
            subagent_max_steps=subagent_max_steps,
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

    async def fork(self, task: Task) -> EmailSearchRecursiveEnv:
        return EmailSearchRecursiveEnv(task=task, subagent_max_steps=self.subagent_max_steps)
