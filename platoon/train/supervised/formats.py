"""A dataset row as a conversation (:func:`convert`, a :class:`Conversation`), in LLaMA-Factory's two formats, as its
converters align them (``src/llamafactory/data/converter.py``).  Imports no Tinker.

- ``alpaca``: the history's prompt and response pairs come first, each a user turn and an assistant turn; then the
  prompt and the query, joined by a newline (the empty ones left out), as one user turn; then the response, as the
  assistant turn.  A row without a text response trains nothing, and is skipped.
- ``sharegpt``: the messages in order, each role by its tag: the user's turns and the tools' results
  (``observation``) at odd positions, the assistant's turns and its tool calls (``function_call``) at even ones, and an
  even number of them; a first message with the system tag is the system prompt, in place of the system column.  A
  row whose roles are out of that order is skipped.

Either format's system prompt and tools come from their columns.  The tools are a JSON list of functions (``name``,
``description``, ``parameters``), each also as an OpenAI tool (``{"type": "function", "function": ...}``); a tool
call is JSON, one call (``name``, ``arguments``) or a list of them.  A row that is skipped raises :class:`SkippedRow`,
which says why.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from platoon.train.supervised.registry import DatasetEntry

# A turn's speaker, as LLaMA-Factory names it: the user, the assistant, the assistant's tool call (function) and a
# tool's result (observation).
Role = Literal["user", "assistant", "function", "observation"]
# The roles a model's turn has, which train.
MODEL_ROLES: tuple[Role, ...] = ("assistant", "function")


class SkippedRow(ValueError):
    """A row LLaMA-Factory would not train on either, with why."""


@dataclass(frozen=True)
class ToolCall:
    """One call of a tool, as a ``function`` turn makes it: the tool's name and its arguments as JSON text."""

    name: str
    arguments: str


@dataclass(frozen=True)
class Turn:
    """One turn of a conversation: its role and its text, or for a ``function`` turn, its tool calls."""

    role: Role
    text: str = ""
    calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class Conversation:
    """One example to train on: its turns, a user turn or a tool's result, then an assistant turn or a tool call,
    alternating, from the first to the last; its system prompt (empty: none); the tools it may call (each a ``name``,
    ``description`` and ``parameters``); and where it comes from, for what the run says of it: its dataset, and its row
    there, counted from 0 (which two equal conversations need not share)."""

    turns: tuple[Turn, ...]
    system: str = ""
    tools: tuple[dict[str, Any], ...] = ()
    dataset: str = field(default="", compare=False)
    row: int = field(default=0, compare=False)

    @property
    def origin(self) -> str:
        """Where it comes from, as messages name it: ``alpaca_demo row 3``."""

        return f"{self.dataset} row {self.row}"


def convert(entry: DatasetEntry, row: Mapping[str, Any], index: int = 0) -> Conversation:
    """``row`` of ``entry``'s dataset, its row ``index``, as a conversation, in the entry's format; raises
    :class:`SkippedRow`."""

    if entry.formatting == "sharegpt":
        turns, system = _sharegpt(entry, row)
    else:
        turns, system = _alpaca(entry, row), _column(entry.columns.system, row)
    system = "" if system is None else system
    if not isinstance(system, str):
        raise SkippedRow(f"its system prompt is not text: {system!r}")
    tools = _tools(_column(entry.columns.tools, row))
    return Conversation(turns=turns, system=system, tools=tools, dataset=entry.name, row=index)


def _alpaca(entry: DatasetEntry, row: Mapping[str, Any]) -> tuple[Turn, ...]:
    columns = entry.columns
    turns: list[Turn] = []
    history = _column(columns.history, row)
    if isinstance(history, list):
        for pair in history:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(text, str) for text in pair)):
                raise SkippedRow(f"its history holds {pair!r}, not a pair of a prompt and a response")
            turns += [Turn("user", pair[0]), Turn("assistant", pair[1])]
    query = [text for text in (_column(columns.prompt, row), _column(columns.query, row)) if text]
    turns.append(Turn("user", "\n".join(map(str, query))))
    response = _column(columns.response, row)
    if not isinstance(response, str):
        raise SkippedRow(f"its response ({columns.response}) is not text: {response!r}")
    turns.append(Turn("assistant", response))
    return tuple(turns)


def _sharegpt(entry: DatasetEntry, row: Mapping[str, Any]) -> tuple[tuple[Turn, ...], Any]:
    """The row's turns, and its system prompt: the first message's when it has the system tag, else the column's."""

    tags = entry.tags
    assert tags.role_tag and tags.content_tag  # the registry refuses a sharegpt entry without them
    messages = _column(entry.columns.messages, row)
    if not isinstance(messages, list) or not all(isinstance(message, Mapping) for message in messages):
        raise SkippedRow(f"its messages ({entry.columns.messages}) are not a list of objects")
    system = _column(entry.columns.system, row)
    if tags.system_tag is not None and messages and messages[0].get(tags.role_tag) == tags.system_tag:
        system, messages = messages[0].get(tags.content_tag), messages[1:]
    # Each role's tag; a role whose tag the entry leaves out (None) has no message.
    named: dict[str | None, Role] = {
        tags.user_tag: "user",
        tags.observation_tag: "observation",
        tags.assistant_tag: "assistant",
        tags.function_tag: "function",
    }
    roles: dict[str, Role] = {tag: role for tag, role in named.items() if tag is not None}
    # What goes at odd positions (counted from 1), then at even ones.
    expected = tuple(
        " or ".join(tag for tag, role in roles.items() if (role in MODEL_ROLES) == model) for model in (False, True)
    )
    turns: list[Turn] = []
    for index, message in enumerate(messages):
        tag, text = message.get(tags.role_tag), message.get(tags.content_tag)
        role = roles.get(tag) if isinstance(tag, str) else None
        if role is None or (role in MODEL_ROLES) != (index % 2 == 1):
            raise SkippedRow(f"message {index + 1} is {tag!r}, where {expected[index % 2]} goes")
        if not isinstance(text, str):
            raise SkippedRow(f"message {index + 1}'s {tags.content_tag} is not text: {text!r}")
        turns.append(Turn(role, calls=_calls(text)) if role == "function" else Turn(role, text))
    if not turns or len(turns) % 2:
        raise SkippedRow(f"it has {len(turns)} messages, not an even number from 2: it must end with a model's turn")
    return tuple(turns), system


def _calls(text: str) -> tuple[ToolCall, ...]:
    """A ``function`` turn's tool calls, from its JSON: one call (``name``, ``arguments``) or a list of them."""

    try:
        calls = json.loads(text)
    except ValueError as exc:
        raise SkippedRow(f"a tool call is not JSON: {text!r}") from exc
    calls = calls if isinstance(calls, list) else [calls]
    if not calls or not all(isinstance(call, dict) and isinstance(call.get("name"), str) for call in calls):
        raise SkippedRow(f"a tool call is not an object with a name, or a list of them: {text!r}")
    return tuple(ToolCall(call["name"], json.dumps(call.get("arguments", {}), ensure_ascii=False)) for call in calls)


def _tools(value: Any) -> tuple[dict[str, Any], ...]:
    """The tools a row offers, from its tools column: a JSON list (or the list itself) of functions or OpenAI tools."""

    if value in (None, "", []):
        return ()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise SkippedRow(f"its tools are not JSON: {value!r}") from exc
    value = value if isinstance(value, list) else [value]
    tools = [
        tool.get("function") if isinstance(tool, dict) and tool.get("type") == "function" else tool for tool in value
    ]
    if not all(isinstance(tool, dict) and isinstance(tool.get("name"), str) for tool in tools):
        raise SkippedRow(f"its tools are not a list of functions with a name: {value!r}")
    return tuple(
        {"name": tool["name"], "description": tool.get("description") or "", "parameters": tool.get("parameters") or {}}
        for tool in tools
    )


def _column(column: str | None, row: Mapping[str, Any]) -> Any:
    """The row's value in ``column``; None without the column."""

    return row.get(column) if column else None
