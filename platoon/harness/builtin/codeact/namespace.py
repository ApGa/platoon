"""What cells can call: the env's tools, ``finish`` and ``launch_subagent``, as :class:`CellFunction` s.

A cell function has a Python signature and a description, which the action space shows
(``def craft(ingredients: dict, target: list) -> str``), and an ``invoke`` coroutine that
does the work on the agent's loop, where ``ctx`` lives. The executor turns each one into
the Python function cells call (``executor.cell_stub``): Python binds the call's arguments
in the cell, then the cell waits for ``invoke`` on the agent's loop. This module therefore
knows nothing of threads, and :class:`Ending`, the one piece of state the functions share
with the harness loop, is only written on the agent's loop.

* Every env tool (``ctx.list_tools()``) becomes a synchronous function whose signature and
  docstring come from its JSON schema. Positional arguments bind to the schema's
  properties in order (required ones first), and a call becomes
  ``ctx.call_tool(name, arguments)``. It returns the ``structured_content`` of the
  reply's metadata when the env sends one (``get_info`` returns a list of dicts),
  otherwise the reply's text.
* ``finish(message="")`` hands the message to the env's terminal tool, when there is
  one, and ends the agent (``finish(None)`` ends it only when the terminal tool reports
  the episode finished; see :class:`Ending`).
* ``await launch_subagent(targets, num_steps, context="")`` runs a child agent through
  ``ctx.agents.spawn`` and returns its final message (or a failure text). With a fixed
  subagent budget ``num_steps`` goes away and the env's ``agents.child_max_steps`` applies.

The two builtins win over env tools of the same name.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from platoon.harness.agent import terminal_arguments, terminal_tool

if TYPE_CHECKING:
    from platoon.envs.service import ToolResult, ToolSpec
    from platoon.harness.agent import AgentContext
    from platoon.harness.builtin.codeact.prompt_builder import CodeActPromptBuilder

STRUCTURED_CONTENT_KEY = "structured_content"

FINISH_DESCRIPTION = (
    "End the agent trajectory and provide a message to the user.\n"
    "E.g., You can use the message to provide the user an answer to the task."
)

_JSON_TYPES = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "object": "dict",
    "null": "None",
}


def schema_type(schema: Mapping[str, Any] | None) -> str | None:
    """A Python annotation for a JSON schema, or None when the schema says nothing usable."""

    if not schema:
        return None
    for key in ("anyOf", "oneOf"):
        if key in schema:
            options = [schema_type(option) for option in schema[key]]
            names = list(dict.fromkeys(option for option in options if option))
            return " | ".join(names) if names else None
    if "$ref" in schema:
        return "dict"
    kind = schema.get("type")
    if isinstance(kind, list):
        names = list(dict.fromkeys(_JSON_TYPES.get(item, "Any") for item in kind))
        return " | ".join(names)
    if kind == "array":
        if "prefixItems" in schema:
            items = [schema_type(item) or "Any" for item in schema["prefixItems"]]
            return f"tuple[{', '.join(items)}]"
        item = schema_type(schema.get("items"))
        return f"list[{item}]" if item else "list"
    if kind == "object":
        values = schema.get("additionalProperties")
        value = schema_type(values) if isinstance(values, Mapping) else None
        return f"dict[str, {value}]" if value else "dict"
    return _JSON_TYPES.get(kind) if isinstance(kind, str) else None


@dataclass(frozen=True)
class Parameter:
    name: str
    annotation: str | None = None
    required: bool = True
    default: Any = None

    def render(self) -> str:
        text = self.name + (f": {self.annotation}" if self.annotation else "")
        return text if self.required else f"{text} = {self.default!r}"


def schema_parameters(input_schema: Mapping[str, Any] | None) -> list[Parameter]:
    """The parameters of a tool's input schema: required ones first, each in schema order."""

    schema = input_schema or {}
    properties: Mapping[str, Any] = schema.get("properties") or {}
    required = set(schema.get("required") or ())
    parameters = [
        Parameter(
            name=name,
            annotation=schema_type(prop),
            required=name in required,
            default=prop.get("default") if isinstance(prop, Mapping) else None,
        )
        for name, prop in properties.items()
    ]
    return [parameter for parameter in parameters if parameter.required] + [
        parameter for parameter in parameters if not parameter.required
    ]


@dataclass
class CellFunction:
    """A function cells can call: its name, parameters and description, and the coroutine that does the work.

    Required parameters come first, as in a Python signature. ``invoke`` runs on the agent's loop with the
    arguments a call passed, by parameter name; defaults are its own to apply (an env applies its tools'
    defaults). Cells await an ``is_async`` function; any other blocks the cell until ``invoke`` returns.
    """

    name: str
    parameters: list[Parameter]
    description: str
    invoke: Callable[[dict[str, Any]], Awaitable[Any]]
    is_async: bool = False

    def signature(self) -> str:
        prefix = "async def" if self.is_async else "def"
        params = ", ".join(parameter.render() for parameter in self.parameters)
        return f"{prefix} {self.name}({params}) -> str"

    def describe(self) -> str:
        """``def name(...) -> str`` and the description, indented under it."""

        lines = [line.rstrip() for line in self.description.strip().splitlines()]
        body = "\n".join(f"   {line}" if line else "" for line in lines)
        return self.signature() + (f"\n{body}" if body else "")


@dataclass
class Ending:
    """Whether the agent is done: the only state the cell functions share with the harness loop.

    The original ``CodeActAgent``'s halt rule: the agent stops once it has a final message, so ``finish(None)``
    does not end it.
    """

    final_message: str | None = None

    @property
    def ended(self) -> bool:
        return self.final_message is not None

    def observe(self, result: ToolResult) -> None:
        """A tool result that says the episode is finished ends the agent, with the result's text as the final
        message unless ``finish`` gave one."""

        if result.finished and self.final_message is None:
            self.final_message = result_text(result)


def tool_function(ctx: AgentContext, tool: ToolSpec, ending: Ending) -> CellFunction:
    """An env tool as a synchronous function: ``ctx.call_tool`` with the call's arguments."""

    async def invoke(arguments: dict[str, Any]) -> Any:
        result = await ctx.call_tool(tool.name, arguments)
        ending.observe(result)
        return result_value(result)

    return CellFunction(tool.name, schema_parameters(tool.input_schema), tool.description, invoke)


def finish_function(ctx: AgentContext, terminal: ToolSpec | None, ending: Ending) -> CellFunction:
    """``finish(message="")``: hands the message to the env's terminal tool, when there is one, and ends the agent."""

    async def invoke(arguments: dict[str, Any]) -> Any:
        message = arguments.get("message", "")
        if terminal is not None:
            ending.observe(await ctx.call_tool(terminal.name, terminal_arguments(terminal, message)))
        if message is not None:
            ending.final_message = message
        return message

    description = terminal.description if terminal is not None and terminal.description else FINISH_DESCRIPTION
    return CellFunction("finish", [Parameter("message", "str", required=False, default="")], description, invoke)


def launch_subagent_function(ctx: AgentContext, builder: CodeActPromptBuilder, *, fixed_budget: bool) -> CellFunction:
    """``await launch_subagent(targets, num_steps, context="")``: a child agent through ``ctx.agents.spawn``.

    The builder turns the targets into the child's goal and task misc. With ``fixed_budget`` the function has no
    ``num_steps`` and the env's ``agents.child_max_steps`` applies.
    """

    async def invoke(arguments: dict[str, Any]) -> str:
        targets, num_steps, context = arguments["targets"], arguments.get("num_steps"), arguments.get("context", "")
        if num_steps is not None and (isinstance(num_steps, bool) or not isinstance(num_steps, int)):
            raise TypeError(f"num_steps must be an int, got {type(num_steps).__name__}")
        child = await ctx.agents.spawn(
            builder.subagent_goal(targets, context),
            max_steps=num_steps,
            task_misc=builder.subagent_task_misc(targets, ctx.task),
        )
        return child.message

    parameters = [Parameter("targets", "dict | str")]
    if not fixed_budget:
        parameters.append(Parameter("num_steps", "int"))
    parameters.append(Parameter("context", "str", required=False, default=""))
    description = (
        "Launch a subagent for a subtask and return its final message.\n"
        "- targets: the subtask goal\n"
        + ("" if fixed_budget else "- num_steps: the subagent's step budget, taken from yours\n")
        + "- context: optional context for the subagent"
    )
    return CellFunction("launch_subagent", parameters, description, invoke, is_async=True)


def cell_functions(
    ctx: AgentContext,
    tools: Sequence[ToolSpec],
    ending: Ending,
    builder: CodeActPromptBuilder,
    *,
    subagents: bool,
    fixed_budget: bool,
) -> list[CellFunction]:
    """Everything a cell can call: the env's tools, then ``finish`` and (with ``subagents``) ``launch_subagent``."""

    builtins = [finish_function(ctx, terminal_tool(tools), ending)]
    if subagents:
        builtins.append(launch_subagent_function(ctx, builder, fixed_budget=fixed_budget))
    reserved = {function.name for function in builtins}  # builtins win over same-named env tools
    return [tool_function(ctx, tool, ending) for tool in tools if tool.name not in reserved] + builtins


def result_value(result: ToolResult) -> Any:
    """What a tool call returns to cell code: the structured content when sent, else the text."""

    if STRUCTURED_CONTENT_KEY in result.metadata:
        return result.metadata[STRUCTURED_CONTENT_KEY]
    return result_text(result)


def result_text(result: ToolResult) -> str:
    """The text blocks of a tool result, joined; a block without a type counts as text, other blocks are left out.

    Not ``ToolResult.as_text()``, which shows other blocks, and blocks without a type, as placeholders: the cell
    output would change.
    """

    parts = []
    for block in result.content:
        if isinstance(block, Mapping) and block.get("type", "text") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)


__all__ = [
    "CellFunction",
    "Ending",
    "Parameter",
    "cell_functions",
    "finish_function",
    "launch_subagent_function",
    "result_text",
    "result_value",
    "schema_parameters",
    "schema_type",
    "tool_function",
]
