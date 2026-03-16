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
- If the length on the context is very large (>160K characters), first examine/peek into the structure of the context (what format is the data in?)
- The context often contains structured data that you can programmatically parse or split: messages with timestamps, users, and content
- Use Python string operations to parse, filter, and chunk the data
- For very large contexts (i.e., >160K characters), work with chunks rather than the entire context at once
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
- First check if the length on the context is very large (>32K characters) using `len(context)`.
- For very large contexts (i.e., >32K characters), work with chunks rather than the entire context at once.
- Use subagents to process chunks and then aggregate the results to produce a final answer. Try not to split the context into too many chunks (32K characters per chunk is a good rule of thumb)
- If the context <= 32K characters, prefer to process your context by printing out and reading it rather than using programmatic heuristics.
- **IMPORTANT: DO NOT USE regex, string matching, etc. types of programmatic heuristics. It is important to read the context with `print(context)` to be accurate in your answer.**

SUBAGENT DELEGATION:
- **IMPORTANT: Do not use subagents if the context you need to process is <= 32K characters. Just print out the context to observe it directly and answer the question by reading the context.**
- You have the ability to spawn subagents (other instantiations of yourself), by providing them with their own `context`/chunk to process and a goal/instruction for what result it should return.
- You can use asyncio.gather to process multiple chunks simultaneously.
- Be specific about the format and typein which you expect subagents to return their results.
- Do not provide the context/chunk as part of the goal. Instead, pass it explicitly as the `context` argument to the `launch_subagent` function.

ANSWER SUBMISSION:
- You can submit your answer using the `finish` function in the format requested in the user provided goal.
</TIPS>
"""

        if include_reasoning:
            return base_instructions + """

You can perform printing out or peaking into the context or launching subagents using Python code blocks. You will get multiple steps to complete the task.
For your current step, first briefly reason (~1-3 sentences) about your recursive strategy in <thought> </thought> tags, then output your code in <python> </python> tags.
Your code will be executed in a Jupyter notebook and the output will be shown to you."""
        else:
            return base_instructions + """

You can perform printing out or peaking into the context or launching subagents using Python code blocks. You will get multiple steps to complete the task.
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
            inference_params=self.inference_params,
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        )
