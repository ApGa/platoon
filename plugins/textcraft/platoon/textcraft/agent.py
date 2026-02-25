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

        base_instructions = (
            "You are an agent in a crafting game. "
            "Your goal is to craft items by combining ingredients.\n"
            "You have access to an inventory of existing ingredients, "
            "which are sufficient to craft the target items; "
            "though, you may need to craft intermediate ingredients first.\n"
            "\n"
            "Note: If you already have one of the target items in your inventory, "
            "you should craft the requested number of the target on top of what you already have.\n"
            "For example, if you already have 2 wooden_pickaxes but your goal is to craft 3, "
            "your inventory should end up with 5 wooden_pickaxes.\n"
            "\n"
            "<TIPS>\n"
            "CRAFTING STRATEGY:\n"
            "- Recipes produce fixed quantities per execution - you cannot craft arbitrary amounts\n"
            "  Example: If a recipe produces 2 items, "
            "you can only craft in multiples of 2 (2, 4, 6...)\n"
            "- Recipe ingredients scale with the number of times you execute it\n"
            '  Example: Recipe "2 ore → 2 items" means 2 ore for 1 execution, 4 ore for 2 executions\n'
            "- Always verify what you have before claiming something is impossible\n"
            "- Check your inventory and recipe information to confirm ingredient availability\n"
            "- Calculate carefully: if a recipe uses 2 ingredients to make 2 items, "
            "you need exactly 2 ingredients for 2 items\n"
            "</TIPS>"
        )

        if include_reasoning:
            return (
                base_instructions + "\n\n"
                "You can perform an action by writing a block of code. "
                "You will get multiple steps to complete the task.\n"
                "For your current step, first briefly reason (~1-3 sentences) about your next step "
                "in the <thought> </thought> tags and then output your code action "
                "in <python> </python> tags.\n"
                "Your code cell will be executed inside a jupyter notebook "
                "and the output will be shown to you."
            )
        else:
            return (
                base_instructions + "\n\n"
                "You can perform an action by writing a block of code. "
                "You will get multiple steps to complete the task.\n"
                "Output your code action in <python> </python> tags."
            )


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
        **kwargs,
    ):
        if "prompt_builder" not in kwargs:
            kwargs["prompt_builder"] = TextCraftPromptBuilder(
                prompt_mode=prompt_mode,
                include_reasoning=include_reasoning,
            )
        super().__init__(prompt_mode=prompt_mode, include_reasoning=include_reasoning, **kwargs)


class TextCraftRecursivePromptBuilder(TextCraftPromptBuilder):
    """Prompt builder for recursive TextCraft agent with subagent support."""

    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        include_reasoning = context.get("include_reasoning", self.include_reasoning)

        base_instructions = (
            "You are an agent in a crafting game. "
            "Your goal is to craft items by combining ingredients.\n"
            "You have access to an inventory of existing ingredients, "
            "which are sufficient to craft the target items; "
            "though, you may need to craft intermediate ingredients first.\n"
            "\n"
            "Note: If you already have one of the target items in your inventory, "
            "you should craft the requested number of the target on top of what you already have.\n"
            "For example, if you already have 2 wooden_pickaxes but your goal is to craft 3, "
            "your inventory should end up with 5 wooden_pickaxes.\n"
            "\n"
            "<TIPS>\n"
            "CRAFTING STRATEGY:\n"
            "- Recipes produce fixed quantities per execution - you cannot craft arbitrary amounts\n"
            "  Example: If a recipe produces 2 items, "
            "you can only craft in multiples of 2 (2, 4, 6...)\n"
            "- Recipe ingredients scale with the number of times you execute it\n"
            '  Example: Recipe "2 ore → 2 items" means 2 ore for 1 execution, 4 ore for 2 executions\n'
            "- Always verify what you have before claiming something is impossible\n"
            "- Check your inventory and recipe information to confirm ingredient availability\n"
            "- Calculate carefully: if a recipe uses 2 ingredients to make 2 items, "
            "you need exactly 2 ingredients for 2 items\n"
            "\n"
            "DELEGATION STRATEGY:\n"
            "- It is **highly recommended** to delegate crafting of intermediate ingredients\n"
            "- Break complex tasks into INDEPENDENT subtasks that can be solved separately\n"
            "- For tasks that are sufficiently complex, it is recommended to recursively delegate; "
            "i.e., subagents can further delegate to other subagents.\n"
            "- Delegate one group of related items at a time, not everything at once\n"
            "- Use crafting depth from get_info() to estimate budget requirements:\n"
            "  * crafting_depth indicates complexity (0=base item, 1=direct craft, 2+=needs intermediates)\n"
            "  * Budget heuristic: depth × 8-10 steps "
            "(depth=4 needs ~32-40 steps, depth=8 needs ~64-80 steps)\n"
            "  * Always check crafting_depth before delegating to avoid under-budgeting\n"
            "- Items can be delegated in parallel if they don't depend on each other\n"
            "- Reserve budget for yourself to do final assembly after subtasks complete\n"
            "- Delegated tasks share your inventory - results are immediately available\n"
            "- **IMPORTANT**: `launch_subagent` is an async function, you MUST use `await`:\n"
            '  * CORRECT: `await launch_subagent({"item": 1}, 20)`\n'
            "  * CORRECT: `await asyncio.gather(launch_subagent(...), launch_subagent(...))`\n"
            '  * WRONG: `launch_subagent({"item": 1}, 20)` -- missing await, will error\n'
            "</TIPS>"
        )

        if include_reasoning:
            return (
                base_instructions + "\n\n"
                "You can perform an action by writing a block of code. "
                "You will get multiple steps to complete the task.\n"
                "For your current step, first briefly reason (~1-3 sentences) about your next step "
                "in the <thought> </thought> tags and then output your code action "
                "in <python> </python> tags.\n"
                "Your code cell will be executed inside a jupyter notebook "
                "and the output will be shown to you."
            )
        else:
            return (
                base_instructions + "\n\n"
                "You can perform an action by writing a block of code. "
                "You will get multiple steps to complete the task.\n"
                "Output your code action in <python> </python> tags."
            )


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
        **kwargs,
    ):
        if "prompt_builder" not in kwargs:
            kwargs["prompt_builder"] = TextCraftRecursivePromptBuilder(
                prompt_mode=prompt_mode,
                include_reasoning=include_reasoning,
            )
        super().__init__(prompt_mode=prompt_mode, include_reasoning=include_reasoning, **kwargs)

    async def fork(self, task) -> TextCraftRecursiveAgent:
        """Fork the agent for a subagent."""
        return TextCraftRecursiveAgent(
            prompt_mode=self.prompt_builder.prompt_mode,
            include_reasoning=self.include_reasoning,
            prompt_builder=self.prompt_builder,
            llm_client=self.llm_client.fork(),
            inference_params=self.inference_params,
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        )


class TextCraftDepthAwarePromptBuilder(TextCraftPromptBuilder):
    """Prompt builder for depth-aware recursive agents.

    Compared to ``TextCraftRecursivePromptBuilder`` this version removes all
    budget-estimation guidance (step counts, depth × N heuristics, reserving
    budget) because each agent has a fixed, pre-configured budget.
    """

    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        include_reasoning = context.get("include_reasoning", self.include_reasoning)

        base_instructions = (
            "You are an agent in a crafting game. "
            "Your goal is to craft items by combining ingredients.\n"
            "You have access to an inventory of existing ingredients, "
            "which are sufficient to craft the target items; "
            "though, you may need to craft intermediate ingredients first.\n"
            "\n"
            "Note: If you already have one of the target items in your inventory, "
            "you should craft the requested number of the target on top of what you already have.\n"
            "For example, if you already have 2 wooden_pickaxes but your goal is to craft 3, "
            "your inventory should end up with 5 wooden_pickaxes.\n"
            "\n"
            "<TIPS>\n"
            "CRAFTING STRATEGY:\n"
            "- Recipes produce fixed quantities per execution - you cannot craft arbitrary amounts\n"
            "  Example: If a recipe produces 2 items, "
            "you can only craft in multiples of 2 (2, 4, 6...)\n"
            "- Recipe ingredients scale with the number of times you execute it\n"
            '  Example: Recipe "2 ore → 2 items" means 2 ore for 1 execution, 4 ore for 2 executions\n'
            "- Always verify what you have before claiming something is impossible\n"
            "- Check your inventory and recipe information to confirm ingredient availability\n"
            "- Calculate carefully: if a recipe uses 2 ingredients to make 2 items, "
            "you need exactly 2 ingredients for 2 items\n"
            "\n"
            "DELEGATION STRATEGY:\n"
            "- It is **highly recommended** to delegate crafting of intermediate ingredients\n"
            "- Break complex tasks into INDEPENDENT subtasks that can be solved separately\n"
            "- For tasks that are sufficiently complex, it is recommended to recursively delegate; "
            "i.e., subagents can further delegate to other subagents.\n"
            "- Delegate one group of related items at a time, not everything at once\n"
            "- Items can be delegated in parallel if they don't depend on each other\n"
            "- Delegated tasks share your inventory - results are immediately available\n"
            "- **IMPORTANT**: `launch_subagent` is an async function, you MUST use `await`:\n"
            '  * CORRECT: `await launch_subagent({"item": 1})`\n'
            "  * CORRECT: `await asyncio.gather(launch_subagent(...), launch_subagent(...))`\n"
            '  * WRONG: `launch_subagent({"item": 1})` -- missing await, will error\n'
            "</TIPS>"
        )

        if include_reasoning:
            return (
                base_instructions + "\n\n"
                "You can perform an action by writing a block of code. "
                "You will get multiple steps to complete the task.\n"
                "For your current step, first briefly reason (~1-3 sentences) about your next step "
                "in the <thought> </thought> tags and then output your code action "
                "in <python> </python> tags.\n"
                "Your code cell will be executed inside a jupyter notebook "
                "and the output will be shown to you."
            )
        else:
            return (
                base_instructions + "\n\n"
                "You can perform an action by writing a block of code. "
                "You will get multiple steps to complete the task.\n"
                "Output your code action in <python> </python> tags."
            )


class TextCraftDepthAwareAgent(TextCraftRecursiveAgent):
    """Agent for depth-aware recursive TextCraft training.

    Uses ``TextCraftDepthAwarePromptBuilder`` which omits budget-estimation
    guidance since each agent has a fixed, pre-configured step budget.
    """

    def __init__(
        self,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
        **kwargs,
    ):
        if "prompt_builder" not in kwargs:
            kwargs["prompt_builder"] = TextCraftDepthAwarePromptBuilder(
                prompt_mode=prompt_mode,
                include_reasoning=include_reasoning,
            )
        super().__init__(prompt_mode=prompt_mode, include_reasoning=include_reasoning, **kwargs)

    async def fork(self, task) -> "TextCraftDepthAwareAgent":
        """Fork the agent for a subagent."""
        return TextCraftDepthAwareAgent(
            prompt_mode=self.prompt_builder.prompt_mode,
            include_reasoning=self.include_reasoning,
            prompt_builder=self.prompt_builder,
            llm_client=self.llm_client.fork(),
            inference_params=self.inference_params,
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        )
