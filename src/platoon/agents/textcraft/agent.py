"""TextCraft agent with recursive spawning support."""
from __future__ import annotations

from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder
from platoon.envs.codeact import CodeActObservation


class TextCraftPromptBuilder(CodeActPromptBuilder):
    """Prompt builder for TextCraft agent."""
    
    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        return """You are an agent in a crafting game. Your goal is to craft items by combining ingredients.
You can perform an action by writing a block of code. You will get multiple steps to complete the task.
For your current step, first briefly reason about your next step in the <think> </think> tags and then output your code action in <python> </python> tags.
It is highly recommended to use subagents to parallelize (with asyncio) and craft intermediate items if needed.
"""

class TextCraftAgent(CodeActAgent):
    """Agent for TextCraft environment with recursive spawning support."""
    
    def __init__(self, *args, **kwargs):
        if "prompt_builder" not in kwargs:
            kwargs["prompt_builder"] = TextCraftPromptBuilder()
        super().__init__(*args, **kwargs)
    
    async def fork(self, task) -> 'TextCraftAgent':
        """Fork the agent for a subagent."""
        return TextCraftAgent(
            prompt_builder=self.prompt_builder,
            llm_client=self.llm_client.fork(),
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        )

