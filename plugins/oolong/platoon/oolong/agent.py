"""Oolong agent with recursive spawning support."""
from __future__ import annotations

from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder, PromptMode
from platoon.envs.codeact import CodeActObservation


class OolongPromptBuilder(CodeActPromptBuilder):
    """Prompt builder for Oolong agent.

    Inherits prompt_mode and include_reasoning support from CodeActPromptBuilder:
    - prompt_mode: "sequence_extension" (default) or "no_sequence_extension"
    - include_reasoning: Whether to include <thought> tags (default True)
    """

    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        include_reasoning = context.get("include_reasoning", self.include_reasoning)

        base_instructions = """You are tasked with answering a query that requires analyzing and aggregating information from a large context.

You have access to a REPL environment with the following pre-loaded variable:
- `context` (str): The full text context to analyze (may be very large)

<TIPS>
CONTEXT ANALYSIS:
- If the length on the context is very large (>10K characters), first examine/peek into the structure of the context (what format is the data in?)
- The context often contains structured data that you can programmatically parse or split: messages with timestamps, users, and content
- Use Python string operations to parse, filter, and chunk the data
- For very large contexts, work with chunks rather than the entire context at once
- If you are able to programmatically process chunks (e.g., using regex, list comprehensions, etc.), prefer doing this over printing out the chunk/context to inspect it.
- But if there is no easy rule-based method to analyze the chunk, then you may have to print it out to observe it.

ANSWER SUBMISSION:
- You can submit your answer using the `finish` function in the format requested in the user provided goal.
</TIPS>
"""

        if include_reasoning:
            return base_instructions + """

You can perform actions by writing Python code blocks. You will get multiple steps to complete the task.
For your current step, first briefly reason (~1-3 sentences) about your strategy in <thought> </thought> tags, then output your code in <python> </python> tags.
Your code will be executed in a Jupyter notebook and the output will be shown to you."""
        else:
            return base_instructions + """

You can perform actions by writing Python code blocks. You will get multiple steps to complete the task.
Output your code in <python> </python> tags."""


class OolongAgent(CodeActAgent):
    """Agent for Oolong long-context aggregation environment.

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
            kwargs["prompt_builder"] = OolongPromptBuilder(
                prompt_mode=prompt_mode,
                include_reasoning=include_reasoning,
            )
        super().__init__(
            prompt_mode=prompt_mode,
            include_reasoning=include_reasoning,
            **kwargs
        )


class OolongRecursivePromptBuilder(OolongPromptBuilder):
    """Prompt builder for recursive Oolong agent with delegation strategies."""

    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        include_reasoning = context.get("include_reasoning", self.include_reasoning)

        base_instructions = """You are tasked with answering a query that requires analyzing and aggregating information from a large context.

You have access to a REPL environment with the following pre-loaded variable:
- `context` (str): The full text context to analyze (may be very large)

<TIPS>
CONTEXT ANALYSIS:
- If the length on the context is very large (>10K characters), first examine/peek into the structure of the context (what format is the data in?)
- The context often contains structured data that you can programmatically parse or split: messages with timestamps, users, and content
- Use Python string operations to parse, filter, and chunk the data
- For very large contexts, work with chunks rather than the entire context at once
- If you are able to programmatically process chunks (e.g., using regex, list comprehensions, etc.), prefer doing this over printing out the chunk/context to inspect it.
- But if there is no easy rule-based method to analyze the chunk, then you should use subagents to process the chunk.

RECURSIVE DELEGATION:
- You have the ability to spawn subagents (other instatiations of yourself) with their own `context` and goal.
- Use subagents to process chunks and then aggregate the results to produce a final answer.
- You can use asyncio.gather to process multiple chunks simultaneously.
- Subagents can further spawn subagents to process even smaller chunks allowing you to process context using recursive divide-and-conquer.

ANSWER SUBMISSION:
- You can submit your answer using the `finish` function in the format requested in the user provided goal.
</TIPS>
"""

        if include_reasoning:
            return base_instructions + """

You can perform actions by writing Python code blocks. You will get multiple steps to complete the task.
For your current step, first briefly reason (~1-3 sentences) about your recursive strategy in <thought> </thought> tags, then output your code in <python> </python> tags.
Your code will be executed in a Jupyter notebook and the output will be shown to you."""
        else:
            return base_instructions + """

You can perform actions by writing Python code blocks. You will get multiple steps to complete the task.
Output your code in <python> </python> tags."""


class OolongRecursiveAgent(OolongAgent):
    """Agent for Oolong environment with recursive spawning support.

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
            kwargs["prompt_builder"] = OolongRecursivePromptBuilder(
                prompt_mode=prompt_mode,
                include_reasoning=include_reasoning,
            )
        super().__init__(
            prompt_mode=prompt_mode,
            include_reasoning=include_reasoning,
            **kwargs
        )

    async def fork(self, task) -> OolongRecursiveAgent:
        """Fork the agent for a subagent."""
        return OolongRecursiveAgent(
            prompt_mode=self.prompt_builder.prompt_mode,
            include_reasoning=self.include_reasoning,
            prompt_builder=self.prompt_builder,
            llm_client=self.llm_client.fork(),
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        )
