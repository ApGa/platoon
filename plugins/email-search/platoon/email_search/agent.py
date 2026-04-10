"""Email-search agents and prompt builders."""

from __future__ import annotations

from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder, PromptMode
from platoon.envs.codeact import CodeActObservation


class EmailSearchPromptBuilder(CodeActPromptBuilder):
    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        include_reasoning = context.get("include_reasoning", self.include_reasoning)

        base_instructions = """You are an email search agent.

You can search a user's email inbox, inspect specific messages, and then answer the user's question.

RESEARCH STRATEGY:
- Start with a targeted email search using a few strong keywords.
- Refine searches with sender, recipient, and date filters when useful.
- Read promising emails before answering.
- Track the message IDs that support your answer.
- If the answer cannot be established from the inbox, say "I don't know".

ANSWER SUBMISSION:
- `json` and `asyncio` are already available in the notebook; do not import them.
- Finish by calling `finish(...)` with a JSON string, not a Python dict.
- Prefer `finish(json.dumps({"answer": "<answer>", "sources": ["<message_id>", ...]}))`.
- `sources` should contain the supporting email `message_id` values.
- If you cannot answer, use `finish(json.dumps({"answer": "I don't know", "sources": []}))`.

OTHER TIPS:
- `search_emails(...)` and `read_email(...)` are async and must be awaited.
- `finish(...)` is synchronous and must not be awaited.
"""

        if include_reasoning:
            return base_instructions + """

You can perform actions by writing Python code blocks. You will get multiple steps to complete the task.
For your current step, first briefly reason (~1-3 sentences) in <thought> </thought> tags, then output code in <python> </python> tags.
Your code will be executed in a Jupyter notebook and the output will be shown to you."""

        return base_instructions + """

You can perform actions by writing Python code blocks. You will get multiple steps to complete the task.
Output your code in <python> </python> tags."""


class EmailSearchAgent(CodeActAgent):
    def __init__(
        self,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
        **kwargs,
    ):
        if "prompt_builder" not in kwargs:
            kwargs["prompt_builder"] = EmailSearchPromptBuilder(
                prompt_mode=prompt_mode,
                include_reasoning=include_reasoning,
            )
        super().__init__(
            prompt_mode=prompt_mode,
            include_reasoning=include_reasoning,
            **kwargs,
        )


class EmailSearchRecursivePromptBuilder(EmailSearchPromptBuilder):
    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        include_reasoning = context.get("include_reasoning", self.include_reasoning)

        base_instructions = """You are an email search agent.

You can search a user's email inbox, inspect specific messages, and delegate subproblems to subagents.

RESEARCH STRATEGY:
- Break the question into a small number of meaningful email-search subproblems and delegate them to subagents.
- Subagents can themselves delegate recursively by breaking issuing even more specific queries.
- Use focused searches first, then read the most relevant emails.
- Track the message IDs that support your answer.

DELEGATION STRATEGY:
- Use `await launch_subagent(goal)` for coherent subtasks like "identify likely message IDs" or "check whether this sender/date hypothesis is correct".
- Use `await asyncio.gather(...)` when multiple search hypotheses or subagents can be run in parallel.
- For delegated subtasks, use `finish(...)` with the requested intermediate result.
- For the root task, finish with a JSON string, not a Python dict.
- Prefer `finish(json.dumps({"answer": "<answer>", "sources": ["<message_id>", ...]}))`.

ANSWER SUBMISSION:
- `json` and `asyncio` are already available in the notebook; do not import them.
- If the inbox does not support an answer, use
  `finish(json.dumps({"answer": "I don't know", "sources": []}))`.

OTHER TIPS:
- `search_emails(...)`, `read_email(...)`, and `launch_subagent(...)` are async and must be awaited or parallelized using `await asyncio.gather(...)`.
- `finish(...)` is synchronous and must not be awaited.
"""

        if include_reasoning:
            return base_instructions + """

You can perform actions by writing Python code blocks. You will get multiple steps to complete the task.
For your current step, first briefly reason (~1-3 sentences) about your research or delegation strategy in <thought> </thought> tags, then output code in <python> </python> tags.
Your code will be executed in a Jupyter notebook and the output will be shown to you."""

        return base_instructions + """

You can perform actions by writing Python code blocks. You will get multiple steps to complete the task.
Output your code in <python> </python> tags."""


class EmailSearchRecursiveAgent(EmailSearchAgent):
    def __init__(
        self,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
        **kwargs,
    ):
        if "prompt_builder" not in kwargs:
            kwargs["prompt_builder"] = EmailSearchRecursivePromptBuilder(
                prompt_mode=prompt_mode,
                include_reasoning=include_reasoning,
            )
        super().__init__(
            prompt_mode=prompt_mode,
            include_reasoning=include_reasoning,
            **kwargs,
        )

    async def fork(self, task) -> EmailSearchRecursiveAgent:
        return EmailSearchRecursiveAgent(
            prompt_mode=self.prompt_builder.prompt_mode,
            include_reasoning=self.include_reasoning,
            prompt_builder=self.prompt_builder,
            llm_client=self.llm_client.fork(),
            inference_params=self.inference_params,
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        )
