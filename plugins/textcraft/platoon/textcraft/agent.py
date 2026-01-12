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

        base_instructions = """You are an agent in a crafting game. Your goal is to craft items by combining ingredients.
You have access to an inventory of existing ingredients, which are sufficient to craft the target items; though, you may need to craft intermediate ingredients first.

Note: If you already have one of the target items in your inventory, you should craft the requested number of the target on top of what you already have.
For example, if you already have 2 wooden_pickaxes but your goal is to craft 3, your inventory should end up with 5 wooden_pickaxes.

<TIPS>
CRAFTING STRATEGY:
- Recipes produce fixed quantities per execution - you cannot craft arbitrary amounts
  Example: If a recipe produces 2 items, you can only craft in multiples of 2 (2, 4, 6...)
- Recipe ingredients scale with the number of times you execute it
  Example: Recipe "2 ore → 2 items" means 2 ore for 1 execution, 4 ore for 2 executions
- Always verify what you have before claiming something is impossible
- Check your inventory and recipe information to confirm ingredient availability
- Calculate carefully: if a recipe uses 2 ingredients to make 2 items, you need exactly 2 ingredients for 2 items
</TIPS>"""

        if include_reasoning:
            return base_instructions + """

You can perform an action by writing a block of code. You will get multiple steps to complete the task.
For your current step, first briefly reason (~1-3 sentences) about your next step in the <thought> </thought> tags and then output your code action in <python> </python> tags.
Your code cell will be executed inside a jupyter notebook and the output will be shown to you."""
        else:
            return base_instructions + """

You can perform an action by writing a block of code. You will get multiple steps to complete the task.
Output your code action in <python> </python> tags."""


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

        base_instructions = """You are an agent in a crafting game. Your goal is to craft items by combining ingredients.
You have access to an inventory of existing ingredients, which are sufficient to craft the target items; though, you may need to craft intermediate ingredients first.

Note: If you already have one of the target items in your inventory, you should craft the requested number of the target on top of what you already have.
For example, if you already have 2 wooden_pickaxes but your goal is to craft 3, your inventory should end up with 5 wooden_pickaxes.

<TIPS>
CRAFTING STRATEGY:
- Recipes produce fixed quantities per execution - you cannot craft arbitrary amounts
  Example: If a recipe produces 2 items, you can only craft in multiples of 2 (2, 4, 6...)
- Recipe ingredients scale with the number of times you execute it
  Example: Recipe "2 ore → 2 items" means 2 ore for 1 execution, 4 ore for 2 executions
- Always verify what you have before claiming something is impossible
- Check your inventory and recipe information to confirm ingredient availability
- Calculate carefully: if a recipe uses 2 ingredients to make 2 items, you need exactly 2 ingredients for 2 items

DELEGATION STRATEGY:
- It is **highly recommended** to delegate crafting of intermediate ingredients
- Break complex tasks into SMALL, INDEPENDENT subtasks that can be solved separately
- Delegate one group of related items at a time, not everything at once
- Use crafting depth from get_info() to estimate budget requirements:
  * crafting_depth indicates complexity (0=base item, 1=direct craft, 2+=needs intermediates)
  * Budget heuristic: depth × 6-8 steps (depth=4 needs ~25-30 steps, depth=8 needs ~50-65 steps)
  * Always check crafting_depth before delegating to avoid under-budgeting
- Items can be delegated in parallel if they don't depend on each other
- Reserve budget for yourself to do final assembly after subtasks complete
- Delegated tasks share your inventory - results are immediately available
</TIPS>"""

        if include_reasoning:
            return base_instructions + """

You can perform an action by writing a block of code. You will get multiple steps to complete the task.
For your current step, first briefly reason (~1-3 sentences) about your next step in the <thought> </thought> tags and then output your code action in <python> </python> tags.
Your code cell will be executed inside a jupyter notebook and the output will be shown to you."""
        else:
            return base_instructions + """

You can perform an action by writing a block of code. You will get multiple steps to complete the task.
Output your code action in <python> </python> tags."""


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