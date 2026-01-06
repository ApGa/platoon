"""TextCraft agent with recursive spawning support."""
from __future__ import annotations

from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder, PromptMode
from platoon.envs.codeact import CodeActObservation


class TextCraftPromptBuilder(CodeActPromptBuilder):
    """Prompt builder for TextCraft agent.
    
    Inherits prompt_mode and include_reasoning support from CodeActPromptBuilder:
    - prompt_mode: "sequence_extension" (default) or "no_sequence_extension"
    - include_reasoning: Whether to include <thought> tags (default True)
    """
    
    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        include_reasoning = context.get("include_reasoning", self.include_reasoning)
        if include_reasoning:
            return """You are an agent in a crafting game. Your goal is to craft items by combining ingredients.
You have access to an inventory of existing ingredients, which are sufficient to craft the target items; though, you may need to craft intermediate ingredients first.
Note: If you are already have one of the target items in your inventory, you should craft the requested number of the target on top of what you already have;
For example, if you already have 2 wooden_pickaxes but your goal is to craft 3, your inventory should end up with 5 wooden_pickaxes.

You can perform an action by writing a block of code. You will get multiple steps to complete the task.
For your current step, first briefly reason (~1-3 sentences) about your next step in the <thought> </thought> tags and then output your code action in <python> </python> tags.
Your code cell will be executed inside a jupyter notebook and the output will be shown to you. Craft ingredients if they are not already in your inventory.
"""
        else:
            return """You are an agent in a crafting game. Your goal is to craft items by combining ingredients.
You have access to an inventory of existing ingredients, which are sufficient to craft the target items; though, you may need to craft intermediate ingredients first.
Note: If you are already have one of the target items in your inventory, you should craft the requested number of the target on top of what you already have;
For example, if you already have 2 wooden_pickaxes but your goal is to craft 3, your inventory should end up with 5 wooden_pickaxes.

You can perform an action by writing a block of code. You will get multiple steps to complete the task.
Output your code action in <python> </python> tags.
Craft ingredients if they are not already in your inventory.
"""


class TextCraftAgent(CodeActAgent):
    """Agent for TextCraft environment.
    
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
            kwargs["prompt_builder"] = TextCraftPromptBuilder(
                prompt_mode=prompt_mode,
                include_reasoning=include_reasoning,
            )
        super().__init__(
            prompt_mode=prompt_mode, 
            include_reasoning=include_reasoning, 
            **kwargs
        )


class TextCraftRecursivePromptBuilder(TextCraftPromptBuilder):
    """Prompt builder for recursive TextCraft agent with subagent support."""
    
    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        include_reasoning = context.get("include_reasoning", self.include_reasoning)
        if include_reasoning:
            return """You are an agent in a crafting game. Your goal is to craft items by combining ingredients.
You have access to an inventory of existing ingredients, which are sufficient to craft the target items; though, you may need to craft intermediate ingredients first.
Note: If you are already have one of the target items in your inventory, you should craft the requested number of the target on top of what you already have;
For example, if you already have 2 wooden_pickaxes but your goal is to craft 3, your inventory should end up with 5 wooden_pickaxes.

You can perform an action by writing a block of code. You will get multiple steps to complete the task.
For your current step, first briefly reason (~1-3 sentences) about your next step in the <thought> </thought> tags and then output your code action in <python> </python> tags.
Your code cell will be executed inside a jupyter notebook and the output will be shown to you. Craft ingredients if they are not already in your inventory. 
It is **highly recommended** to use subagents to craft ingredients if they are not already in your inventory.
"""
        else:
            return """You are an agent in a crafting game. Your goal is to craft items by combining ingredients.
You have access to an inventory of existing ingredients, which are sufficient to craft the target items; though, you may need to craft intermediate ingredients first.
Note: If you are already have one of the target items in your inventory, you should craft the requested number of the target on top of what you already have;
For example, if you already have 2 wooden_pickaxes but your goal is to craft 3, your inventory should end up with 5 wooden_pickaxes.

You can perform an action by writing a block of code. You will get multiple steps to complete the task.
Output your code action in <python> </python> tags.
Craft ingredients if they are not already in your inventory. It is **highly recommended** to use subagents to craft ingredients if they are not already in your inventory.
"""


class TextCraftRecursiveAgent(TextCraftAgent):
    """Agent for TextCraft environment with recursive spawning support.
    
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
            kwargs["prompt_builder"] = TextCraftRecursivePromptBuilder(
                prompt_mode=prompt_mode,
                include_reasoning=include_reasoning,
            )
        super().__init__(
            prompt_mode=prompt_mode, 
            include_reasoning=include_reasoning, 
            **kwargs
        )
        
    async def fork(self, task) -> TextCraftRecursiveAgent:
        """Fork the agent for a subagent."""
        return TextCraftRecursiveAgent(
            prompt_mode=self.prompt_builder.prompt_mode,
            include_reasoning=self.include_reasoning,
            prompt_builder=self.prompt_builder,
            llm_client=self.llm_client.fork(),
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        )