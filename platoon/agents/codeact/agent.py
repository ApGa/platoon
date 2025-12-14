from __future__ import annotations

import re

from platoon.agents.codeact.prompt_builder import CodeActPromptBuilder
from platoon.envs.codeact import CodeActObservation, CodeActAction
from platoon.envs.base import Task
from platoon.utils.llm_client import LLMClient

def extract_code_and_thought(raw_action: str) -> tuple[str, str]:
    # Try to extract both code and thought in the expected format
    match = re.search(r"<think>(.*?)</think>\n<python>(.*?)</python>", raw_action, re.DOTALL)
    if match:
        thought = match.group(1)
        code = match.group(2)
        return code, thought
    
    # If both aren't present, try to extract them separately
    thought = ""
    code = ""

    # Try to extract thought
    thought_match = re.search(r"<think>(.*?)</think>", raw_action, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1)
    
    # Try to extract code
    code_match = re.search(r"<python>(.*?)</python>", raw_action, re.DOTALL)
    if code_match:
        code = code_match.group(1)
    
    return code, thought

# TODO: Need to add a factory for this.
# TODO: Let's probably get rid of kwargs
class CodeActAgent:
    def __init__(self, prompt_builder: CodeActPromptBuilder | None = None, llm_client: LLMClient | None = None, stuck_in_loop_threshold: int = 4, stuck_in_loop_window: int = 3):
        if prompt_builder is None:
            prompt_builder = CodeActPromptBuilder()
        if llm_client is None:
            llm_client = LLMClient()

        self.prompt_builder = prompt_builder
        self.llm_client = llm_client
        self.stuck_in_loop_threshold = stuck_in_loop_threshold
        self.stuck_in_loop_window = stuck_in_loop_window
    
    def _stuck_in_loop(self, obs: CodeActObservation) -> bool:
        if len(obs.history) < self.stuck_in_loop_threshold:
            return False
        codes = [ (step.code or "").strip() for step in obs.history ]
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
        desc = f"Detected a repeating pattern (length ≤ {self.stuck_in_loop_window}) repeated at least {self.stuck_in_loop_threshold} times."

        stuck_action = CodeActAction(
            parsed_code="finish('Stuck in a loop, terminating early.')",
            parsed_thought=f"{desc} It seems that I'm stuck in a loop.",
            misc={"error_message": desc}
        )
        stuck_action.action = str(stuck_action)
        stuck_action.misc["usage"] = {}
        stuck_action.misc["model"] = None
        return stuck_action

    async def act(self, obs: CodeActObservation) -> CodeActAction:
        if self._stuck_in_loop(obs):
            return self._stuck_in_loop_action()
        
        prompt = self.prompt_builder.build_messages(obs)
        # TODO: Make inference params configurable.
        response = await self.llm_client.async_chat_completion(prompt, stop=["</python>"], temperature=1.0, top_p=1, max_tokens=32000, extra_body={"return_token_ids": True})
        response_text = response.choices[0].message.content
        # NOTE: We only do this conditionally, because with Areal, stop words are not supported.
        # And so we might already have the stop word in the response.
        if '</python>' not in response_text:
            response_text += "</python>"
        action = self.parse_raw_action(response_text)
        action.misc["usage"] = response.usage.to_dict()
        action.misc["model"] = response.model
        action.misc["completion_id"] = response.id
        action.misc["completion_data"] = dict(
            input_token_ids=response.prompt_token_ids,
            output_token_ids=response.choices[0].provider_specific_fields["token_ids"],
        )
        return action
    
    async def reset(self) -> None:
        pass

    async def fork(self, task: Task) -> CodeActAgent:
        return type(self)(
            prompt_builder=self.prompt_builder,
            llm_client=self.llm_client.fork(),
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        )

    async def close(self) -> None:
        await self.llm_client.aclose()

    def parse_raw_action(self, raw_action: str) -> CodeActAction:
        code, thought = extract_code_and_thought(raw_action)
        return CodeActAction(action=raw_action, parsed_code=code, parsed_thought=thought)
