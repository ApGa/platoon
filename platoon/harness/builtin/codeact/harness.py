"""The ``platoon-codeact`` function harness: the model acts by writing Python cells that call the env's tools.

How codeact works
-----------------

One agent is one call of :func:`codeact`, and the loop in it is the whole harness. Each step
renders the conversation, samples one reply, runs the reply's ``<python>`` cell and shows the
model what the cell printed. Two threads take part::

    the agent's loop (the rollout's event loop)           the agent's cell thread (its own event loop)
    -------------------------------------------           --------------------------------------------
    messages = builder.build_messages(state)
    reply = await ctx.chat(messages, stop=builder.stop)
    code, thought = builder.parse_reply(reply.text)
    cell = await cells.run(code)  ----- unless refused ---->  IPython runs the cell:
                                                                print(craft({"a": 1}, ("b", 1)))
      craft's invoke(arguments)   <---- LoopBridge.call ----  the stub binds the call, then blocks
        await ctx.call_tool("craft", arguments)                 this thread...
                                  ------ the value ------->   ...until the tool's value comes back
      launch_subagent's invoke    <-- LoopBridge.call_async   await launch_subagent({"b": 1}, 10)
        await ctx.agents.spawn(goal, ...)
                                  ---- the message ------->   ...returns the child's final message
                                  <--- output, error ------   the cell ends
    ctx.annotate_step(**step.annotation())
    state.history.append(step)

A cell function's work (``invoke``) runs on the agent's loop, next to ``ctx``, in a fresh
context, so what the env prints or logs meanwhile stays out of the cell; the cell only binds
the call's arguments and waits. Each subagent is another call of :func:`codeact` with
its own shell, cell thread and state. When the agent ends, leaving its executor cancels the
calls still running and refuses later ones, so a cell left running (one that timed out, say)
can no longer reach the env.

A cell that cancels a launch (``asyncio.wait_for``) cancels the child, as in the original
``CodeActAgent``, the agent class this harness replaced. The launches still running when the
agent ends (an un-awaited ``asyncio.gather``) are cancelled with the other calls, but their
children are kept (``ctx.agents.keep_children``): they finish and are graded while the agent
waits for them, as the original agent let them finish.

The modules answer one question each:

* :mod:`~platoon.harness.builtin.codeact.prompt_builder`: what the model sees and how its reply is read (the
  conversation, ``stop`` and ``parse_reply``). This is the env's variant seam, picked by configuration.
* :mod:`~platoon.harness.builtin.codeact.namespace`: what cells can call: every env tool (``ctx.list_tools()``,
  already filtered by the agent's access), ``finish`` and ``launch_subagent``, each with the coroutine that does
  its work on the agent's loop.
* :mod:`~platoon.harness.builtin.codeact.checks`: what cells may not do: the original agent's refusals and its loop
  check.
* :mod:`~platoon.harness.builtin.codeact.executor`: where cells run: an IPython shell on the agent's cell loop,
  in a thread of its own, with a stub per cell function that binds a call and hands it to the agent's loop.
  Code that blocks (a synchronous tool call, a busy loop) blocks only that thread, never the agent's loop,
  where the gateway, the env session and every other agent keep running.
* :mod:`~platoon.harness.builtin.codeact.ipython_shell`: how cell output is captured: what concurrent cells print
  stays apart, IPython's ``Out[n]:`` prompt is dropped, and ``run_cell_async`` works on IPython 9.14 and 9.17.

The agent ends with ``completed`` when ``finish`` is called or a tool reports the episode finished, with ``limit``
(the original agent's "Exhausted budget" warning) when its budget or ``ctx.max_steps`` runs out, and with
``timeout`` when a cell outlives ``cell_timeout_s``. When the last cells repeat themselves
(``checks.describe_loop``), the agent ends with the original agent's synthetic ``finish`` cell, which takes no
model call and no budget step and is not annotated; if the cells redefined ``finish``, the agent still ends, with
that cell's message. Rewards are not the harness's business: the agent's grade sets ``reward/success`` on its last
step, and ``ctx.agents.spawn`` records ``reward/subagent_launched``/``reward/subagent_succeeded`` on the step that
launched a child.

Options (``harness_options`` in the environment config; unknown keys are an error):

* ``prompt_builder`` (an import path of a ``CodeActPromptBuilder`` subclass), ``prompt_builder_options`` (its
  extra constructor arguments), ``prompt_mode`` and ``include_reasoning``;
* ``sampling``: Chat Completions parameters for every model call, e.g. ``max_completion_tokens``, that override
  the rollout's default sampling (its ``inference_params``);
* ``cell_timeout_s`` (300) and ``max_output_chars`` (no cut);
* ``subagents`` (offer ``launch_subagent``) and ``subagent_budget``: ``argument`` takes the child's step
  budget from the call's ``num_steps``, ``fixed`` drops that argument for the env's ``agents.child_max_steps``;
* ``checks`` (:class:`~platoon.harness.builtin.codeact.checks.Checks`): ``unawaited_async_calls``,
  ``while_loops``, ``interactive_input``, ``stuck_in_loop_threshold`` and ``stuck_in_loop_window`` (0 switches
  the loop check off).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Literal

from platoon.harness.agent import LIMIT_MESSAGE, AgentContext, AgentResult, out_of_steps
from platoon.harness.builtin.codeact.checks import Checks, describe_loop
from platoon.harness.builtin.codeact.executor import CellExecutor
from platoon.harness.builtin.codeact.namespace import Ending, cell_functions
from platoon.harness.builtin.codeact.prompt_builder import CodeActPromptBuilder, CodeActState, CodeActStep, PromptMode
from platoon.harness.function import harness

STUCK_IN_LOOP_MESSAGE = "Stuck in a loop, terminating early."
STUCK_IN_LOOP_CODE = f"finish({STUCK_IN_LOOP_MESSAGE!r})"


@dataclass(frozen=True)
class CodeActOptions:
    """The ``platoon-codeact`` harness options (see the module docstring); unknown keys are an error."""

    prompt_builder: str | None = None
    prompt_builder_options: Mapping[str, Any] = field(default_factory=dict)
    prompt_mode: PromptMode = "sequence_extension"
    include_reasoning: bool = True
    sampling: Mapping[str, Any] = field(default_factory=dict)
    cell_timeout_s: float | None = 300.0
    max_output_chars: int | None = None
    subagents: bool = True
    subagent_budget: Literal["argument", "fixed"] = "argument"
    checks: Checks = Checks()

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any] | None) -> CodeActOptions:
        options = dict(options or {})
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(options) - known)
        if unknown:
            raise ValueError(f"Unknown platoon-codeact harness options {unknown}; known: {sorted(known)}")
        if not isinstance(options.get("checks"), Checks):
            options["checks"] = Checks.from_mapping(options.get("checks"))
        parsed = cls(**options)
        if parsed.subagent_budget not in ("argument", "fixed"):
            raise ValueError(f"subagent_budget must be 'argument' or 'fixed', got {parsed.subagent_budget!r}")
        return parsed

    def make_prompt_builder(self) -> CodeActPromptBuilder:
        builder_class: type[CodeActPromptBuilder] = CodeActPromptBuilder
        if self.prompt_builder is not None:
            from platoon.registry import import_from_string

            builder_class = import_from_string(self.prompt_builder)
            if not (isinstance(builder_class, type) and issubclass(builder_class, CodeActPromptBuilder)):
                raise TypeError(f"prompt_builder {self.prompt_builder!r} is not a CodeActPromptBuilder subclass")
        return builder_class(
            prompt_mode=self.prompt_mode, include_reasoning=self.include_reasoning, **dict(self.prompt_builder_options)
        )


@harness("platoon-codeact", executes_model_code=True)
async def codeact(ctx: AgentContext) -> AgentResult:
    """One CodeAct agent: the model writes a ``<python>`` cell, the cell runs, its output is the next turn."""

    options = CodeActOptions.from_mapping(ctx.options)
    builder = options.make_prompt_builder()  # the env's variant: prompts, stop/parse, subagent goals
    ending = Ending()  # set by finish() or by a tool result that ends the episode
    functions = cell_functions(
        ctx,
        await ctx.list_tools(),
        ending,
        builder,
        subagents=options.subagents,
        fixed_budget=options.subagent_budget == "fixed",
    )
    state = CodeActState(
        task=builder.describe_task(ctx.instruction, ctx.max_steps),
        action_space=builder.describe_action_space(functions),
    )
    async with CellExecutor(functions, checks=options.checks, max_output_chars=options.max_output_chars) as cells:
        try:
            steps = 0
            while not ending.ended:
                if out_of_steps(ctx, steps):
                    return AgentResult("limit", error=LIMIT_MESSAGE)
                loop_report = describe_loop(state.history, options.checks)
                if loop_report is not None:  # a synthetic finish cell: no model call, no budget step
                    code, thought = STUCK_IN_LOOP_CODE, f"{loop_report} It seems that I'm stuck in a loop."
                else:
                    reply = await ctx.chat(builder.build_messages(state), stop=list(builder.stop), **options.sampling)
                    code, thought = builder.parse_reply(reply.text)
                    steps += 1
                cell = await cells.run(code, timeout_s=options.cell_timeout_s)  # the refusal, or output/error
                step = CodeActStep(code=cell.code, thought=thought, output=cell.output, error=cell.error)
                if loop_report is None:  # annotate the model call that wrote this cell
                    ctx.annotate_step(**step.annotation())
                if cell.timed_out:
                    return AgentResult("timeout", error=cell.error)
                state.history.append(step)
                if loop_report is not None and not ending.ended:  # the cells redefined finish(): end as it would have
                    ending.final_message = STUCK_IN_LOOP_MESSAGE
        finally:
            # Leaving the executor cancels the launches still running (an un-awaited asyncio.gather): keep their
            # children, which finish and are graded while the agent waits for them.
            ctx.agents.keep_children()
    return AgentResult("completed", final_message=ending.final_message)


__all__ = ["STUCK_IN_LOOP_CODE", "STUCK_IN_LOOP_MESSAGE", "CodeActOptions", "codeact"]
