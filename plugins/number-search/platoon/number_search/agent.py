from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder, PromptMode
from platoon.envs.codeact import CodeActObservation


class NumberSearchPromptBuilder(CodeActPromptBuilder):
    """Prompt builder for NumberSearch agent.
    
    Inherits prompt_mode and include_reasoning support from CodeActPromptBuilder:
    - prompt_mode: "sequence_extension" (default) or "no_sequence_extension"
    - include_reasoning: Whether to include <thought> tags (default True)
    """
    
    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        include_reasoning = context.get("include_reasoning", self.include_reasoning)
        if include_reasoning:
            return """Solve the problem step by step. Write your thoughts in <thought> </thought> tags and the final action in <python> </python> tags.
Your answer must call guess(number: int) with the guessed number as an integer.

Example:
<thought>
thought process here
</thought>
<python>
guess(42)
</python>
"""
        else:
            return """Solve the problem step by step. Write your action in <python> </python> tags.
Your answer must call guess(number: int) with the guessed number as an integer.

Example:
<python>
guess(42)
</python>
"""


class NumberSearchAgent(CodeActAgent):
    """Agent for NumberSearch environment.
    
    Args:
        prompt_mode: The prompt format to use ("sequence_extension" or "no_sequence_extension")
        include_reasoning: Whether to include <thought> tags in prompts (default True)
    """
    
    def __init__(
        self, 
        prompt_mode: PromptMode = "sequence_extension", 
        include_reasoning: bool = True,
        **kwargs
    ):
        if "prompt_builder" not in kwargs:
            kwargs["prompt_builder"] = NumberSearchPromptBuilder(
                prompt_mode=prompt_mode,
                include_reasoning=include_reasoning,
            )
        super().__init__(
            prompt_mode=prompt_mode, 
            include_reasoning=include_reasoning, 
            **kwargs
        )
