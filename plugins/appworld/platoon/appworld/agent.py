from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder, PromptMode
from platoon.envs.base import Task
from platoon.envs.codeact import CodeActObservation
from platoon.utils.prompt_retriever import PromptRetriever

@dataclass
class Supervisor:
    first_name: str
    last_name: str
    email: str
    phone_number: str
    
class AppWorldCodeActPromptBuilder(CodeActPromptBuilder):
    def __init__(
        self, 
        prompt_mode: PromptMode = "sequence_extension", 
        include_reasoning: bool = True, 
        prompts_dir: str | Path | None = None,
    ):
        super().__init__(prompt_mode=prompt_mode, include_reasoning=include_reasoning, prompts_dir=prompts_dir)
        self.appworld_prompt_retriever = PromptRetriever(prompts_dir=Path(__file__).parent / "prompts")
    
    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        if "env_specific_system_context" not in context:
            context["env_specific_system_context"] = self.appworld_prompt_retriever.get_prompt("system-env-specific-system-context")
        return super().build_system_prompt(obs, **context)

class AppWorldRecursiveCodeActPromptBuilder(AppWorldCodeActPromptBuilder):
    def __init__(self, 
        prompt_mode: PromptMode = "sequence_extension", 
        include_reasoning: bool = True, 
        prompts_dir: str | Path | None = None,
        use_parent_state: bool = False,
    ):
        super().__init__(prompt_mode=prompt_mode, include_reasoning=include_reasoning, prompts_dir=prompts_dir)
        self.appworld_prompt_retriever = PromptRetriever(prompts_dir=Path(__file__).parent / "prompts")
        self.use_parent_state = use_parent_state
        
    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        if "env_specific_system_context" not in context:
            context["env_specific_system_context"] = self.appworld_prompt_retriever.get_prompt("system-recursive-env-specific-system-context")
        return super().build_system_prompt(obs, **context)

class AppWorldAgent(CodeActAgent):
    def __init__(self,
        prompt_mode: PromptMode = "sequence_extension", 
        include_reasoning: bool = True, 
        prompts_dir: str | Path | None = None,
        prompt_builder: AppWorldCodeActPromptBuilder | None = None,
        **kwargs
    ):
        if prompt_builder is None:
            prompt_builder = AppWorldCodeActPromptBuilder(prompt_mode=prompt_mode, include_reasoning=include_reasoning, prompts_dir=prompts_dir)
        super().__init__(
            prompt_builder=prompt_builder,
            prompt_mode=prompt_mode,
            include_reasoning=include_reasoning,
            **kwargs,
        )
        
class AppWorldRecursiveAgent(AppWorldAgent):

    def __init__(self,
        prompt_mode: PromptMode = "sequence_extension", 
        include_reasoning: bool = True, 
        prompts_dir: str | Path | None = None,
        use_parent_state: bool = False,
        prompt_builder: AppWorldRecursiveCodeActPromptBuilder | None = None,
        **kwargs,
    ):
        if prompt_builder is None:
            prompt_builder = AppWorldRecursiveCodeActPromptBuilder(prompt_mode=prompt_mode, include_reasoning=include_reasoning, prompts_dir=prompts_dir)
        super().__init__(
            prompt_builder=prompt_builder,
            prompt_mode=prompt_mode,
            include_reasoning=include_reasoning,
            **kwargs,
        )

    async def fork(self, task: Task) -> AppWorldRecursiveAgent:
        return AppWorldRecursiveAgent(
            prompt_builder=self.prompt_builder,
            prompt_mode=self.prompt_builder.prompt_mode,
            include_reasoning=self.include_reasoning,
            llm_client=self.llm_client,
            inference_params=self.inference_params,
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        )
