"""``pythonic``: tool calls as a Python list of calls, as Llama 4 writes them (vLLM's ``llama4_pythonic``).

::

    [get_weather(city="Paris", days=3), list_files()]

The list starts the response (after the reasoning, between ``<think>`` and ``</think>``), as vLLM reads it, and may
sit between ``<|python_start|>`` and ``<|python_end|>``; text after it is content.  A list anywhere else is content
too, so prose such as "the result is [round(number=2.5)]" is not a call.  Each call is read with Python's parser:
keyword arguments only, each value a Python literal that JSON can hold (``true``, ``false`` and ``null`` are read as
JSON's; a tuple is a list); a name may be dotted.

The list is split into its calls before any is read, so a call that does not parse (a positional argument, a value
that is not a literal) is an unparsed call on its own, and the list's other calls still parse.  A list cut off
before its ``]`` keeps the calls that closed, and the rest is an unparsed call, marked so.  Function names that are
not Python identifiers cannot be written in this format.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from typing import Any

from platoon.train.tinker.renderers.tool_parsers.arguments import ToolSchemas, json_compatible
from platoon.train.tinker.renderers.tool_parsers.base import ParseResult, ResultBuilder, ToolCallParser, skip_whitespace

PYTHON_START, PYTHON_END = "<|python_start|>", "<|python_end|>"
_LIST_START = re.compile(r"\[\s*[A-Za-z_][\w.]*\s*\(")
_JSON_NAMES = {"true": True, "false": False, "null": None}


class PythonicParser(ToolCallParser):
    name = "pythonic"
    aliases = ("llama4_pythonic",)

    def parse_calls(self, text: str, tools: ToolSchemas) -> ParseResult:
        out = ResultBuilder()
        start = skip_whitespace(text, 0)
        if text.startswith(PYTHON_START, start):
            start = skip_whitespace(text, start + len(PYTHON_START))
        if not _LIST_START.match(text, start):
            out.text(_without_markers(text))
            return out.result()
        elements, end, error = _split_list(text, start)
        for element_start, element_end in elements:
            source = text[element_start:element_end]
            try:
                name, arguments = _call(source)
            except ValueError as e:
                out.fail(source.strip(), f"the tool call is not valid: {e}")
            else:
                out.call(name, arguments)
        if error is not None:
            out.fail(text[end:], error)
            return out.result()
        out.text(_without_markers(text[end:]))
        return out.result()


def _split_list(text: str, start: int) -> tuple[list[tuple[int, int]], int, str | None]:
    """The spans of the top-level elements of the list at ``start``, where the list ends, and an error when it is
    cut off (then it ends after its last complete element)."""

    lines = text[start:].splitlines(keepends=True)
    offsets = [start]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def at(position: tuple[int, int]) -> int:
        return offsets[position[0] - 1] + position[1]

    elements: list[tuple[int, int]] = []
    depth, element_start = 0, None
    try:
        for token in tokenize.generate_tokens(io.StringIO(text[start:]).readline):
            if token.type != tokenize.OP:
                continue
            if token.string in "([{":
                depth += 1
                if depth == 1:
                    element_start = at(token.end)
            elif token.string in ")]}":
                depth -= 1
                if depth == 0:
                    if element_start is not None and text[element_start : at(token.start)].strip():
                        elements.append((element_start, at(token.start)))
                    return elements, at(token.end), None
            elif token.string == "," and depth == 1 and element_start is not None:
                elements.append((element_start, at(token.start)))
                element_start = at(token.end)
    except (tokenize.TokenError, SyntaxError) as e:
        reason = e.args[0] if e.args else type(e).__name__
        return elements, element_start if element_start is not None else start, f"the call list is cut off: {reason}"
    return elements, element_start if element_start is not None else start, "the call list is cut off"


def _call(source: str) -> tuple[str, dict[str, Any]]:
    try:
        node = ast.parse(source.strip(), mode="eval").body
    except SyntaxError as e:
        raise ValueError(f"not a Python call: {e.msg}") from None
    if not isinstance(node, ast.Call):
        raise ValueError("not a Python call")
    if node.args:
        raise ValueError("positional arguments are not allowed")
    name = _dotted_name(node.func)
    arguments: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ValueError("**kwargs are not allowed")
        arguments[keyword.arg] = _literal(keyword.value)
    return name, arguments


def _dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr}"
    raise ValueError("the function name is not a (dotted) name")


def _literal(node: ast.expr) -> Any:
    """An argument's value: a Python literal JSON can hold, with ``true``, ``false`` and ``null`` read as JSON's."""

    if isinstance(node, ast.Name) and node.id in _JSON_NAMES:
        return _JSON_NAMES[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            raise ValueError("** in a dict is not a literal")
        keys = [_literal(key) for key in node.keys]  # type: ignore[arg-type]
        if not all(isinstance(key, str) for key in keys):
            raise ValueError(f"a key of {ast.unparse(node)!r} is not a string")
        return dict(zip(keys, (_literal(value) for value in node.values)))
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        raise ValueError(f"argument value {ast.unparse(node)!r} is not a literal") from None
    try:
        return json_compatible(value)
    except ValueError as e:
        raise ValueError(f"argument value {ast.unparse(node)!r} is not a JSON value: {e}") from None


def _without_markers(text: str) -> str:
    return text.replace(PYTHON_START, "").replace(PYTHON_END, "")
