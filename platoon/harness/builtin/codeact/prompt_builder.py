"""What the model sees and how its reply is read: CodeAct's prompts, its cell records and the reply parse.

``CodeActPromptBuilder`` renders the same text as the original ``CodeActAgent`` (the agent
class this harness replaced), so runs stay comparable. Two prompt modes:

- ``sequence_extension`` (default): a multi-turn conversation that grows by one
  assistant turn (the cell) and one user turn (its output) per step, so each
  prompt extends the previous prompt and completion and the trainer merges the
  steps into few datums.
- ``no_sequence_extension``: one user message that embeds the whole cell history,
  rebuilt every step.

A reply is read with the original agent's rule: sampling stops at ``</python>``
(:attr:`~CodeActPromptBuilder.stop`), the cut-off tag is put back, and the original regexes pick
out the ``<thought>`` and the ``<python>`` cell (:meth:`~CodeActPromptBuilder.parse_reply`).

The builder is the environment's variant seam. Environments with their own wording
subclass it and name it in the harness options (``prompt_builder: package.module:Class``):
``build_system_prompt`` for the instructions, ``describe_action_space`` for the functions
the cells can call, ``subagent_goal``/``subagent_task_misc`` for what ``launch_subagent``
hands ``ctx.agents.spawn``, and ``stop``/``parse_reply`` for another reply format.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined

if TYPE_CHECKING:
    from platoon.harness.builtin.codeact.namespace import CellFunction
    from platoon.task import Task

PromptMode = Literal["sequence_extension", "no_sequence_extension"]
Message = dict[str, str]

PROMPTS_DIR = Path(__file__).parent / "prompts"
PYTHON_STOP = "</python>"


# ---------------------------------------------------------------------------- records


@dataclass
class CodeActStep:
    """One cell: the parsed thought and code of a model reply, and the cell's output and error text."""

    code: str | None = None
    thought: str | None = None
    output: str | None = None
    error: str | None = None

    def annotation(self) -> dict[str, str | None]:
        """The step misc the harness puts on the model call that wrote this cell (what viewers show)."""
        return {"thought": self.thought, "code": self.code, "output": self.output, "error": self.error}


@dataclass
class CodeActState:
    """What a prompt builder renders: the task text, the action space and the cells run so far."""

    task: str
    action_space: str = ""
    history: list[CodeActStep] = field(default_factory=list)


def extract_code_and_thought(raw_action: str) -> tuple[str, str]:
    """The ``<python>`` code and ``<thought>`` text of a reply (either may be missing, then "")."""

    match = re.search(r"<thought>(.*?)</thought>\n<python>(.*?)</python>", raw_action, re.DOTALL)
    if match:
        return match.group(2), match.group(1)
    thought_match = re.search(r"<thought>(.*?)</thought>", raw_action, re.DOTALL)
    code_match = re.search(r"<python>(.*?)</python>", raw_action, re.DOTALL)
    return (code_match.group(1) if code_match else ""), (thought_match.group(1) if thought_match else "")


# ---------------------------------------------------------------------------- the builder


class CodeActPromptBuilder:
    """Builds the CodeAct conversation for one agent and reads its replies (see the module docstring).

    Args:
        prompts_dir: Directory of the Jinja templates (default: the built-in ``prompts``). Another directory must
            hold all five: ``system``, ``user``, ``user-initial``, ``user-observation`` and ``user-next-action-str``
            (``.jinja``).
        prompt_mode: ``sequence_extension`` or ``no_sequence_extension``.
        include_reasoning: Ask for ``<thought>`` before each ``<python>`` cell, and keep
            the thoughts in the history.
    """

    stop: tuple[str, ...] = (PYTHON_STOP,)  # stop sequences of every model call

    def __init__(
        self,
        prompts_dir: str | Path | None = None,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
    ) -> None:
        if prompt_mode not in ("sequence_extension", "no_sequence_extension"):
            raise ValueError(
                f"prompt_mode must be 'sequence_extension' or 'no_sequence_extension', got {prompt_mode!r}"
            )
        self.prompts_dir = Path(prompts_dir) if prompts_dir is not None else PROMPTS_DIR
        self.templates = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )
        self.prompt_mode = prompt_mode
        self.include_reasoning = include_reasoning

    def render(self, template: str, **variables: Any) -> str:
        return str(self.templates.get_template(f"{template}.jinja").render(**variables))

    # ------------------------------------------------------------------ replies

    def parse_reply(self, text: str) -> tuple[str, str]:
        """The ``(code, thought)`` of a reply.

        Sampling stops at ``</python>`` and drops it, so, as in the original agent, a reply without the tag gets it
        back first: a cell the model left open still runs.
        """
        if PYTHON_STOP not in text:
            text += PYTHON_STOP
        return extract_code_and_thought(text)

    # ------------------------------------------------------------------ task and action space

    def describe_task(self, instruction: str, max_steps: int | None) -> str:
        """The task section: the goal, and the agent's step budget when it has one."""

        if max_steps:
            return (
                f"Your Goal: {instruction}\nBudget: You have a total budget of {max_steps} steps to complete this task."
            )
        return f"Your Goal: {instruction}"

    def describe_action_space(self, functions: Sequence[CellFunction]) -> str:
        """The functions a cell can call, numbered, each with its signature and description."""

        if not functions:
            return ""
        entries = [f"{index}. {function.describe()}" for index, function in enumerate(functions, start=1)]
        text = "Available Actions (python functions):\n\n" + "\n\n".join(entries) + "\n"
        if any(function.is_async for function in functions):
            text += (
                "\nNote: `asyncio` is already imported. Use `await asyncio.gather(...)` to run async calls in "
                "parallel, or `await` a single call. **Do not forget to await** async functions.\n"
            )
        return text

    # ------------------------------------------------------------------ subagents

    def subagent_goal(self, targets: Any, context: str = "") -> str:
        """The child's goal for ``launch_subagent(targets, ..., context)``: the targets as text, then the
        parent's context in the original agent's words."""

        goal = str(targets)
        if context:
            goal += f"\n\nContext provided from parent agent: {context}"
        return goal

    def subagent_task_misc(self, targets: Any, parent_task: Task) -> dict[str, Any] | None:
        """The child's task misc for ``launch_subagent(targets, ...)``; None keeps the parent's."""

        return None

    # ------------------------------------------------------------------ conversation

    def build_messages(self, state: CodeActState) -> list[Message]:
        """The conversation for the next model call, in the builder's prompt mode."""

        if self.prompt_mode == "sequence_extension":
            return self._build_messages_sequence_extension(state)
        return self._build_messages_no_sequence_extension(state)

    def _build_messages_sequence_extension(self, state: CodeActState) -> list[Message]:
        """[system] instructions, [user] task + action space, then an assistant/user pair per cell."""

        messages: list[Message] = [
            {"role": "system", "content": self.build_system_prompt(state)},
            {"role": "user", "content": self._build_initial_user_message(state)},
        ]
        for index, step in enumerate(state.history):
            messages.append({"role": "assistant", "content": self._format_action_for_history(step)})
            messages.append({"role": "user", "content": self._format_observation_for_history(step, index)})
        return messages

    def _build_messages_no_sequence_extension(self, state: CodeActState) -> list[Message]:
        """[system] instructions and one [user] message holding the task, action space and history."""

        user_prompt = self.build_user_prompt(
            state,
            task=state.task,
            action_space_description=state.action_space,
            action_history_description=self.build_action_history_description(state),
            next_action_str=self.build_next_action_str(state),
        )
        return [
            {"role": "system", "content": self.build_system_prompt(state)},
            {"role": "user", "content": user_prompt},
        ]

    def _build_initial_user_message(self, state: CodeActState) -> str:
        return self.render(
            "user-initial",
            task=state.task,
            action_space_description=state.action_space or "",
            next_action_str=self.build_next_action_str(state),
        )

    def _format_action_for_history(self, step: CodeActStep) -> str:
        """``sequence_extension``: a past cell as the assistant wrote it, ``<thought>`` (with reasoning) and
        ``<python>``."""

        parts = []
        if self.include_reasoning and step.thought:
            parts.append(f"<thought>{step.thought}</thought>")
        if step.code:
            parts.append(f"<python>{step.code}</python>")
        return "\n".join(parts)

    def _format_observation_for_history(self, step: CodeActStep, step_index: int) -> str:
        """``sequence_extension``: the user turn after a past cell, its output and error."""

        return self.render("user-observation", step_index=step_index, output=step.output or "", error=step.error or "")

    def _format_cell_for_history(self, step: CodeActStep) -> str:
        """``no_sequence_extension``: a past cell in the history section, each part stripped and tagged.

        As in the original agent, the thought is shown whatever ``include_reasoning`` says.
        """

        parts = (("thought", step.thought), ("python", step.code), ("output", step.output), ("error", step.error))
        return "".join(_tag(name, text) for name, text in parts if text)

    def build_action_history_description(self, state: CodeActState) -> str:
        if not state.history:
            return "No actions taken yet."
        return "\n".join(
            f"Cell {index}:\n" + self._format_cell_for_history(cell) for index, cell in enumerate(state.history)
        )

    def build_next_action_str(self, state: CodeActState, **context: Any) -> str:
        return self.render("user-next-action-str", include_reasoning=self.include_reasoning, **context)

    def build_system_prompt(self, state: CodeActState, **context: Any) -> str:
        return self.render("system", include_reasoning=self.include_reasoning, **context)

    def build_user_prompt(self, state: CodeActState, **context: Any) -> str:
        return self.render("user", **context)


def _tag(name: str, text: str) -> str:
    return f"<{name}>\n{text.strip()}\n</{name}>\n"


__all__ = [
    "PYTHON_STOP",
    "CodeActPromptBuilder",
    "CodeActState",
    "CodeActStep",
    "Message",
    "PromptMode",
    "extract_code_and_thought",
]
