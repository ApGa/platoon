"""DeepDive agents and prompt builders."""

from __future__ import annotations

from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder, PromptMode
from platoon.envs.codeact import CodeActObservation


class DeepDivePromptBuilder(CodeActPromptBuilder):
    """Prompt builder for DeepDive web research tasks."""

    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        include_reasoning = context.get("include_reasoning", self.include_reasoning)

        base_instructions = """You are a deep research agent solving a factual question by searching the web.

You have access to Python plus web-search tools. Use them deliberately:
- Start broad, then refine based on what you learn.
- Cross-check key claims across multiple sources when possible.
- Use `await view_webpage_content(url)` when snippets are insufficient or you need detailed evidence.
- Avoid dumping huge webpage bodies into the notebook unless necessary.
- Keep intermediate notes concise and use Python to organize findings.

ANSWER SUBMISSION:
- When you are confident, call `finish(...)`.
- The final answer should directly answer the question and stay concise unless the task explicitly asks for more detail.

OTHER TIPS:
- **All functions except for finish are async functions. YOU MUST AWAIT THE RESULTS OF THESE FUNCTIONS**
"""

        if include_reasoning:
            return base_instructions + """

You can perform actions by writing Python code blocks. You will get multiple steps to complete the task.
For your current step, first briefly reason (~1-3 sentences) in <thought> </thought> tags, then output code in <python> </python> tags.
Your code will be executed in a Jupyter notebook and the output will be shown to you."""

        return base_instructions + """

You can perform actions by writing Python code blocks. You will get multiple steps to complete the task.
Output your code in <python> </python> tags."""


class DeepDiveAgent(CodeActAgent):
    """Agent for DeepDive tasks."""

    def __init__(
        self,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
        **kwargs,
    ):
        if "prompt_builder" not in kwargs:
            kwargs["prompt_builder"] = DeepDivePromptBuilder(
                prompt_mode=prompt_mode,
                include_reasoning=include_reasoning,
            )
        super().__init__(
            prompt_mode=prompt_mode,
            include_reasoning=include_reasoning,
            **kwargs,
        )


class DeepDiveRecursivePromptBuilder(DeepDivePromptBuilder):
    """Prompt builder for recursive DeepDive agents."""

    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        include_reasoning = context.get("include_reasoning", self.include_reasoning)

        base_instructions = """You are a deep research agent solving a factual question by searching the web.

You have access to Python plus web-search tools, and you can delegate subproblems to subagents.

RESEARCH STRATEGY:
- Break the question into a small number of meaningful subquestions.
- Search broadly first, then narrow onto the most promising sources.
- Cross-check important claims across multiple sources when possible.
- Use `await view_webpage_content(url)` when search snippets are not enough.
- Use Python to store notes, compare evidence, and synthesize findings.

DELEGATION STRATEGY:
- You have the ability to spawn subagents and delegate subtasks to them. Make effective use of subagents to solve the task!
- Use `await launch_subagent(goal)` for coherent subproblems such as source discovery, fact verification, or answering one component of a multi-hop question.
- Tell subagents exactly what to return, including format when useful.
- Subagents can run in parallel with `await asyncio.gather(...)`.
- Subagents can themselves delegate recursively.

ANSWER SUBMISSION:
- When you are confident, call `finish(...)`.
- The final answer should directly answer the question and stay concise unless the task explicitly asks for more detail.

OTHER TIPS:
- **All functions except for finish are async functions. YOU MUST AWAIT THE RESULTS OF THESE FUNCTIONS**

"""

        if include_reasoning:
            return base_instructions + """

You can perform actions by writing Python code blocks. You will get multiple steps to complete the task.
For your current step, first briefly reason (~1-3 sentences) about your research or delegation strategy in <thought> </thought> tags, then output code in <python> </python> tags.
Your code will be executed in a Jupyter notebook and the output will be shown to you."""

        return base_instructions + """

You can perform actions by writing Python code blocks. You will get multiple steps to complete the task.
Output your code in <python> </python> tags."""


class DeepDiveRecursiveAgent(DeepDiveAgent):
    """Recursive DeepDive agent with subagent support."""

    def __init__(
        self,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
        **kwargs,
    ):
        if "prompt_builder" not in kwargs:
            kwargs["prompt_builder"] = DeepDiveRecursivePromptBuilder(
                prompt_mode=prompt_mode,
                include_reasoning=include_reasoning,
            )
        super().__init__(
            prompt_mode=prompt_mode,
            include_reasoning=include_reasoning,
            **kwargs,
        )

    async def fork(self, task) -> DeepDiveRecursiveAgent:
        return DeepDiveRecursiveAgent(
            prompt_mode=self.prompt_builder.prompt_mode,
            include_reasoning=self.include_reasoning,
            prompt_builder=self.prompt_builder,
            llm_client=self.llm_client.fork(),
            inference_params=self.inference_params,
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        )
