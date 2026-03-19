from __future__ import annotations

import re
import uuid
from typing import cast

from openai.types.chat import ChatCompletionMessageParam

from platoon.config_defs import InferenceParams
from platoon.agents.codeact.prompt_builder import CodeActPromptBuilder, PromptMode
from platoon.envs.base import Task
from platoon.envs.codeact import CodeActAction, CodeActObservation
from platoon.episode.context import current_trajectory
from platoon.utils import async_hang_debug
from platoon.utils.llm_client import LLMClient
from platoon.utils.span_profile import profile_span


def extract_code_and_thought(raw_action: str) -> tuple[str, str]:
    # Try to extract both code and thought in the expected format
    match = re.search(r"<thought>(.*?)</thought>\n<python>(.*?)</python>", raw_action, re.DOTALL)
    if match:
        thought = match.group(1)
        code = match.group(2)
        return code, thought

    # If both aren't present, try to extract them separately
    thought = ""
    code = ""

    # Try to extract thought
    thought_match = re.search(r"<thought>(.*?)</thought>", raw_action, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1)

    # Try to extract code
    code_match = re.search(r"<python>(.*?)</python>", raw_action, re.DOTALL)
    if code_match:
        code = code_match.group(1)

    return code, thought


class CodeActAgent:
    """Agent that uses code actions in an interactive environment.

    Supports two prompt modes via the prompt_builder:
    - "sequence_extension" (default): Uses a multi-turn conversation format where
      each step appends to the conversation. This enables sequence extension for
      efficient training - consecutive observations are prefixes of each other,
      allowing tinker to merge consecutive steps into fewer Datums.
    - "no_sequence_extension": Uses a single user message with the full action
      history embedded. This is the legacy format that rebuilds the entire prompt
      each step.

    Args:
        prompt_builder: Custom prompt builder. If not provided, uses CodeActPromptBuilder
            with the specified prompt_mode and include_reasoning.
        prompt_mode: The prompt format to use. Only used if prompt_builder is not provided.
            Default is "sequence_extension".
        include_reasoning: If True, prompts instruct the agent to include <thought> tags.
            If False, only <python> tags are expected. Default is True.
        llm_client: LLM client for inference.
        stuck_in_loop_threshold: Number of repetitions to detect a loop.
        stuck_in_loop_window: Window size for loop detection.
    """

    def __init__(
        self,
        prompt_builder: CodeActPromptBuilder | None = None,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
        llm_client: LLMClient | None = None,
        inference_params: InferenceParams | None = None,
        stuck_in_loop_threshold: int = 4,
        stuck_in_loop_window: int = 3,
    ):
        if prompt_builder is None:
            prompt_builder = CodeActPromptBuilder(prompt_mode=prompt_mode, include_reasoning=include_reasoning)
        if llm_client is None:
            llm_client = LLMClient()

        self.prompt_builder = prompt_builder
        self.llm_client = llm_client
        self.include_reasoning = include_reasoning
        self.inference_params = inference_params or InferenceParams()
        self.stuck_in_loop_threshold = stuck_in_loop_threshold
        self.stuck_in_loop_window = stuck_in_loop_window

    def _stuck_in_loop(self, obs: CodeActObservation) -> bool:
        if len(obs.history) < self.stuck_in_loop_threshold:
            return False
        codes = [(step.code or "").strip() for step in obs.history]
        n = len(codes)
        max_period = min(self.stuck_in_loop_window, n // self.stuck_in_loop_threshold)
        if max_period <= 0:
            return False
        for period in range(1, max_period + 1):
            segment_len = period * self.stuck_in_loop_threshold
            segment = codes[-segment_len:]
            pattern = segment[:period]
            matches_pattern = True
            for i in range(segment_len):
                if segment[i] != pattern[i % period]:
                    matches_pattern = False
                    break
            if matches_pattern:
                return True
        return False

    def _stuck_in_loop_action(self) -> CodeActAction:
        desc = (
            f"Detected a repeating pattern (length ≤ {self.stuck_in_loop_window}) "
            f"repeated at least {self.stuck_in_loop_threshold} times."
        )

        stuck_action = CodeActAction(
            parsed_code="finish('Stuck in a loop, terminating early.')",
            parsed_thought=f"{desc} It seems that I'm stuck in a loop.",
            misc={"error_message": desc},
        )
        stuck_action.action = str(stuck_action)
        stuck_action.misc["usage"] = {}
        stuck_action.misc["model"] = None
        return stuck_action

    async def act(self, obs: CodeActObservation) -> CodeActAction:
        if self._stuck_in_loop(obs):
            return self._stuck_in_loop_action()

        prompt = cast(list[ChatCompletionMessageParam], self.prompt_builder.build_messages(obs))
        current_traj = current_trajectory.get(None)
        request_id = str(uuid.uuid4())
        request_kwargs = {
            "stop": ["</python>"],
            "max_completion_tokens": self.inference_params.max_completion_tokens,
            "timeout": 1800, # TODO: Make configurable and move to LiteLLM utils?
        }
        if self.inference_params.temperature is not None:
            request_kwargs["temperature"] = self.inference_params.temperature
        if self.inference_params.top_p is not None:
            request_kwargs["top_p"] = self.inference_params.top_p

        async_hang_debug.track_current_task(
            request_id=request_id,
            kind="agent_llm",
            metadata={
                "task_id": obs.task.id,
                "trajectory_id": current_traj.id if current_traj is not None else None,
                "parent_trajectory_id": (
                    current_traj.parent_info.id
                    if current_traj is not None and current_traj.parent_info is not None
                    else None
                ),
                "step_index": len(obs.history),
                "message_count": len(prompt),
                "model": getattr(self.llm_client, "model", None),
                "timeout": request_kwargs["timeout"],
            },
        )
        try:
            async with profile_span(
                "agent_act",
                metadata={
                    "task_id": obs.task.id,
                    "trajectory_id": current_traj.id if current_traj is not None else None,
                    "parent_trajectory_id": (
                        current_traj.parent_info.id
                        if current_traj is not None and current_traj.parent_info is not None
                        else None
                    ),
                    "step_index": len(obs.history),
                    "message_count": len(prompt),
                    "model": getattr(self.llm_client, "model", None),
                },
            ):
                response = await self.llm_client.async_chat_completion(
                    prompt,
                    # Stop sequence is agent-level behavior, not a global rollout knob.
                    **request_kwargs,
                )
        finally:
            async_hang_debug.untrack(request_id)
        response_text = response.choices[0].message.content or ""
        # NOTE: We only do this conditionally, because with Areal, stop words are not supported.
        # And so we might already have the stop word in the response.
        if "</python>" not in response_text:
            response_text += "</python>"
        action = self.parse_raw_action(response_text)
        action.misc["usage"] = response.usage.to_dict()
        action.misc["model"] = response.model
        action.misc["completion_id"] = response.id
        return action

    async def reset(self) -> None:
        pass

    async def fork(self, task: Task) -> CodeActAgent:
        return type(self)(
            prompt_builder=self.prompt_builder,
            include_reasoning=self.include_reasoning,
            llm_client=self.llm_client.fork(),
            inference_params=self.inference_params,
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        )

    async def close(self) -> None:
        await self.llm_client.aclose()

    def parse_raw_action(self, raw_action: str) -> CodeActAction:
        code, thought = extract_code_and_thought(raw_action)
        return CodeActAction(action=raw_action, parsed_code=code, parsed_thought=thought)
