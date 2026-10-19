"""``qwen3_coder``: the XML tool calls of Qwen3-Coder and Qwen3.5 and later.

::

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

A value is text, typed by the request's tool schemas (:mod:`.arguments`): the newline after ``<parameter=KEY>`` and
the one before ``</parameter>`` belong to the format, and everything between them is the value, indentation and
trailing newlines included.  A function with no parameters is ``<function=NAME>`` then ``</function>``.

The format does not escape markup, so a value ends at the first close tag that the structure after it confirms: a
value may hold a literal ``</parameter>``, ``</function>`` or ``<parameter=...>``, whole calls, and whole blocks of
another format (Hermes JSON).  ``</parameter>`` ends a value when the next markup is the call's next ``<parameter=``
or the end of its function.  A function ends when it is followed by ``</tool_call>`` (or that is missing) and then by
nothing that belongs inside a call (``</parameter>``, ``</function>``, ``</tool_call>`` or a ``<parameter=`` line)
before the next call or the end of the response; a ``<tool_call>`` block that starts a line and holds no function is
passed over as a whole.  A call written out inside a value on lines of its own holds the close tags after it until it
closes, so the value does not end inside it: a ``<tool_call>`` closes at its ``</tool_call>``, and a ``<function=``
line without one (as OpenHands' prompts write calls) at its ``</function>``.

Lenient rules, which mark the call recovered:

* a ``</parameter>`` is missing: the value ends where a parameter of the tool that the call does not have yet opens
  on a new line (any such key when the tool's schema is unknown) and does not open again later in the call, or where
  the function ends on a new line, as long as the value holds no call that starts a line;
* ``</function>`` is missing before ``</tool_call>``, or ``</tool_call>`` after ``</function>``;
* ``<tool_call>`` is missing before a ``<function=NAME>`` that starts a line;
* a parameter is given twice: the last value counts.  A well-formed call never repeats one, so the model repeated it
  or a value that holds markup was misread.

The first rule reads text that a well-formed value can also hold: a value with a line ``<parameter=KEY>``, for a
parameter the call has not given yet, looks exactly like a value whose ``</parameter>`` is missing before parameter
``KEY``.  Such a value is split there and the call is marked recovered, unless ``KEY`` opens again later in the call.

A call that still does not parse, such as one cut off inside a value, or a ``<tool_call>`` on a new line that holds
no ``<function=`` (Hermes JSON from an XML model), is an unparsed call that runs from its ``<tool_call>`` to its
``</tool_call>``, else to the next call or the end of the response.  A parse takes about linear time in the
response's length, also on degenerate output that repeats markup.

The format, and typing values by the tool's schema, follow SGLang's ``qwen3_coder`` detector (Apache-2.0; see
``NOTICE`` in this directory); the code is written for platoon.
"""

from __future__ import annotations

import bisect
import functools
import re

from platoon.train.tinker.renderers.tool_parsers.arguments import ToolSchemas
from platoon.train.tinker.renderers.tool_parsers.base import ParseResult, ResultBuilder, ToolCallParser

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
FUNCTION_OPEN = "<function="
FUNCTION_CLOSE = "</function>"
PARAMETER_OPEN = "<parameter="
PARAMETER_CLOSE = "</parameter>"

# A call starts at <tool_call> (a real call when <function= follows) or at a <function= that starts a line.
_STARTS = re.compile(r"<tool_call>|^[ \t]*<function=", re.MULTILINE)
_REAL_START = re.compile(r"<tool_call>\s*<function=")
# A call that holds a function: <tool_call> then <function=, or a <function= that starts a line.
_REAL_CALL = re.compile(r"<tool_call>\s*<function=|^[ \t]*<function=", re.MULTILINE)
_LINE_START_CALL = re.compile(r"^[ \t]*(?:<tool_call>\s*)?<function=", re.MULTILINE)
# A <tool_call> that starts a line and holds no function: another format's call (Hermes JSON, say).
_BARE_BLOCK_START = re.compile(r"^[ \t]*<tool_call>(?!\s*<function=)", re.MULTILINE)
# Where a value may end: its close tag, or (when that is missing) the next markup that starts a line.
_VALUE_ENDS = re.compile(r"</parameter>|^[ \t]*(?:<parameter=|</function>|</tool_call>)", re.MULTILINE)
# Markup that belongs inside a call, which the text between calls never holds.
_CALL_MARKUP = re.compile(r"</parameter>|</function>|</tool_call>|^[ \t]*<parameter=", re.MULTILINE)
_CLOSE_TAGS = re.compile(r"</parameter>|</function>|</tool_call>")
_PARAMETER_LINE = re.compile(r"^[ \t]*<parameter=([^>\n]*)>", re.MULTILINE)
_WHITESPACE = re.compile(r"\s*")


class _Malformed(Exception):
    """A call block that does not parse, and why."""


class Qwen3CoderParser(ToolCallParser):
    name = "qwen3_coder"
    aliases = ("qwen3_xml", "step3p5", "nanbeige")
    call_start = _REAL_START

    def parse_calls(self, text: str, tools: ToolSchemas) -> ParseResult:
        return _Reader(text, tools).parse()


class _Reader:
    """One response being parsed.  What the parse learns about positions in it (where calls that start a line are,
    where close tags are, which block ends were checked) is kept, so that a parse takes about linear time."""

    def __init__(self, text: str, tools: ToolSchemas) -> None:
        self.text = text
        self.tools = tools
        self._block_ends: dict[int, bool] = {}  # by position
        self._closed_ends: dict[int, tuple[int, int, bool] | None] = {}  # by index of a </parameter>
        self._calls_closed: dict[int, int] = {}  # by index of a call that starts a line: where its close tag ends

    def parse(self) -> ParseResult:
        text, out, pos = self.text, ResultBuilder(), 0
        while True:
            start = _next_call(text, pos)
            if start is None:
                out.text(text[pos:])
                return out.result()
            out.text(text[pos:start])
            try:
                name, arguments, end, recovered = self.call(start)
            except _Malformed as e:
                end = _bad_block_end(text, start)
                out.fail(text[start:end], str(e))
            else:
                out.call(name, self.tools.coerce(name, arguments), recovered=recovered)
            pos = end

    def call(self, start: int) -> tuple[str, dict[str, str], int, bool]:
        """``(name, raw arguments, end, recovered)`` of the call block at ``start``; raises :class:`_Malformed`."""

        text = self.text
        pos, recovered = start, False
        has_open = text.startswith(TOOL_CALL_OPEN, pos)
        if has_open:
            pos = _skip(text, pos + len(TOOL_CALL_OPEN))
        else:
            recovered = True  # a <function= line without <tool_call>
        if not text.startswith(FUNCTION_OPEN, pos):
            raise _Malformed("<tool_call> holds no <function=NAME>")
        name, pos = _tag_value(text, pos + len(FUNCTION_OPEN), "function")
        arguments: dict[str, str] = {}
        while True:
            tag = _skip(text, pos)
            if text.startswith(PARAMETER_OPEN, tag):
                key, value_start = _tag_value(text, tag + len(PARAMETER_OPEN), "parameter")
                value_end, pos, relaxed = self.value_end(value_start, name, key, arguments)
                recovered = recovered or relaxed or key in arguments  # a parameter given twice: the last counts
                arguments[key] = _strip_newline(text[value_start:value_end])
            elif text.startswith(FUNCTION_CLOSE, tag):
                pos = tag + len(FUNCTION_CLOSE)
                break
            elif text.startswith(TOOL_CALL_CLOSE, tag) and has_open:
                pos, recovered = tag, True  # </function> is missing
                break
            else:
                what = "is cut off" if tag >= len(text) else f"has unexpected text {text[tag : tag + 30]!r}"
                raise _Malformed(f"function {name!r} {what}")
        tag = _skip(text, pos)
        if text.startswith(TOOL_CALL_CLOSE, tag):
            pos = tag + len(TOOL_CALL_CLOSE)
        elif has_open:
            recovered = True  # </tool_call> is missing
        return name, arguments, pos, recovered

    def value_end(self, start: int, name: str, key: str, arguments: dict[str, str]) -> tuple[int, int, bool]:
        """``(value end, position to go on from, relaxed)`` for the value of ``key`` that starts at ``start``."""

        text = self.text
        for match in _VALUE_ENDS.finditer(text, start):
            at = text.index("<", match.start())
            if self.call_between(start, at):
                # The value holds a call written out on lines of its own: from here on only a confirmed close tag
                # that no open call holds ends it, the same for every value that holds the call.
                found = self.closed_end(at)
                if found is None:
                    break
                return found
            if match.group() == PARAMETER_CLOSE:
                relaxed = self.confirmed(at)
                if relaxed is not None:
                    return at, match.end(), relaxed
            elif text.startswith(PARAMETER_OPEN, at):  # a </parameter> is missing before the next parameter
                try:
                    next_key, _ = _tag_value(text, at + len(PARAMETER_OPEN), "parameter")
                except _Malformed:
                    continue
                if (
                    next_key != key
                    and next_key not in arguments
                    and self.tools.declares(name, next_key) is not False
                    and not self.opens_again(next_key, at)
                ):
                    return at, at, True
            elif text.startswith(FUNCTION_CLOSE, at):  # the call's last </parameter> is missing
                if self.function_ends(at):
                    return at, at, True
            elif self.block_ends(at + len(TOOL_CALL_CLOSE)):  # </parameter> and </function> are missing
                return at, at, True
        raise _Malformed(f"parameter {key!r} of function {name!r} is not closed")

    def confirmed(self, close: int) -> bool | None:
        """Whether what follows the ``</parameter>`` at ``close`` shows that it ends a value: None when it does not,
        else whether the end is relaxed (the call's ``</function>`` is missing)."""

        text = self.text
        following = _skip(text, close + len(PARAMETER_CLOSE))
        if text.startswith(PARAMETER_OPEN, following):
            return False
        if text.startswith(FUNCTION_CLOSE, following) and self.function_ends(following):
            return False
        if text.startswith(TOOL_CALL_CLOSE, following) and self.block_ends(following + len(TOOL_CALL_CLOSE)):
            return True
        return None

    def closed_end(self, pos: int) -> tuple[int, int, bool] | None:
        """The first ``</parameter>`` from ``pos`` on that the structure confirms and that no open call holds, as
        :meth:`value_end` returns a value's end; None when there is none."""

        closes = self._closes[PARAMETER_CLOSE]
        index, passed, found = bisect.bisect_left(closes, pos), [], None
        while index < len(closes):
            if index in self._closed_ends:
                found = self._closed_ends[index]
                break
            passed.append(index)
            at = closes[index]
            relaxed = self.confirmed(at)
            if relaxed is not None and not self.open_at(at):
                found = (at, at + len(PARAMETER_CLOSE), relaxed)
                break
            index += 1
        for seen in passed:
            self._closed_ends[seen] = found
        return found

    def call_between(self, start: int, pos: int) -> bool:
        """Whether a call that starts a line lies between ``start`` and ``pos``."""

        index = bisect.bisect_right(self._line_call_ends, pos) - 1
        return index >= 0 and self._line_calls[index].start() >= start

    def open_at(self, pos: int) -> bool:
        """Whether the last call that starts a line before ``pos`` is still open there: a ``<tool_call>`` without its
        ``</tool_call>``, or a ``<function=`` line without ``<tool_call>`` (as OpenHands' prompts write calls) and
        without its ``</function>``."""

        index = bisect.bisect_right(self._line_call_ends, pos) - 1
        if index < 0:
            return False
        closed = self._calls_closed.get(index)
        if closed is None:
            call = self._line_calls[index]
            close = TOOL_CALL_CLOSE if TOOL_CALL_OPEN in call.group() else FUNCTION_CLOSE
            following = self._next_close(close, call.end())
            closed = self._calls_closed[index] = len(self.text) + 1 if following is None else following + len(close)
        return closed > pos

    def opens_again(self, key: str, at: int) -> bool:
        """Whether parameter ``key`` opens on a new line after the one at ``at``, before the next call: then the one
        at ``at`` is text inside a value, not a parameter whose value's close tag is missing."""

        lines = self._parameter_lines.get(key, [])
        index = bisect.bisect_right(lines, at)
        return index < len(lines) and _REAL_CALL.search(self.text, at + 1, lines[index]) is None

    def function_ends(self, pos: int) -> bool:
        """Whether the ``</function>`` at ``pos`` ends the call: what follows its ``</tool_call>`` (or the lack of
        one) holds no call markup up to the next call or the end."""

        after = _skip(self.text, pos + len(FUNCTION_CLOSE))
        if self.text.startswith(TOOL_CALL_CLOSE, after):
            after += len(TOOL_CALL_CLOSE)
        return self.block_ends(after)

    def block_ends(self, pos: int) -> bool:
        """Whether no call markup follows ``pos`` before the next call that holds a function, or the end.

        A ``<tool_call>`` block that starts a line and holds no function (another format's call, an unparsed call of
        its own) is passed over whole, so a value may hold several; one that holds call markup but no function is a
        broken call of its own, so the next call.
        """

        text, passed = self.text, []
        while (ends := self._block_ends.get(pos)) is None:
            passed.append(pos)
            markup = _CALL_MARKUP.search(text, pos)
            if markup is None or _REAL_CALL.search(text, pos, markup.start()) is not None:
                ends = True
            elif _BARE_BLOCK_START.search(text, pos, markup.start()) is None:
                ends = False
            elif markup.group() != TOOL_CALL_CLOSE:
                ends = True  # a <tool_call> that holds call markup but no function: a broken call of its own
            else:
                pos = markup.end()  # a whole block of another format: pass over it
                continue
            break
        for position in passed:
            self._block_ends[position] = ends
        return ends

    def _next_close(self, close: str, pos: int) -> int | None:
        positions = self._closes[close]
        index = bisect.bisect_left(positions, pos)
        return positions[index] if index < len(positions) else None

    @functools.cached_property
    def _line_calls(self) -> list[re.Match[str]]:
        """The calls that start a line, in order: the only calls a value may hold."""

        return list(_LINE_START_CALL.finditer(self.text))

    @functools.cached_property
    def _line_call_ends(self) -> list[int]:
        return [call.end() for call in self._line_calls]

    @functools.cached_property
    def _closes(self) -> dict[str, list[int]]:
        """Where each close tag occurs, in order."""

        found: dict[str, list[int]] = {PARAMETER_CLOSE: [], FUNCTION_CLOSE: [], TOOL_CALL_CLOSE: []}
        for match in _CLOSE_TAGS.finditer(self.text):
            found[match.group()].append(match.start())
        return found

    @functools.cached_property
    def _parameter_lines(self) -> dict[str, list[int]]:
        """Where each parameter opens on a new line, by key, in order."""

        found: dict[str, list[int]] = {}
        for match in _PARAMETER_LINE.finditer(self.text):
            found.setdefault(match[1].strip(), []).append(match.start())
        return found


def _skip(text: str, pos: int) -> int:
    match = _WHITESPACE.match(text, pos)
    return match.end() if match else pos


def _at_line_start(text: str, pos: int) -> bool:
    line_start = text.rfind("\n", 0, pos) + 1
    return not text[line_start:pos].strip()


def _next_call(text: str, pos: int) -> int | None:
    """Where the next call block starts: a ``<tool_call>`` that holds a function or starts a line, or a
    ``<function=`` that starts a line.  A ``<tool_call>`` in running text that holds no function is just text."""

    for match in _STARTS.finditer(text, pos):
        start = text.index("<", match.start())
        if text.startswith(FUNCTION_OPEN, start) or _REAL_START.match(text, start) or _at_line_start(text, start):
            return start
    return None


def _bad_block_end(text: str, start: int) -> int:
    """Where an unparsed block ends: after its ``</tool_call>``, else at the next call, else at the end."""

    search_from = start + 1
    if text.startswith(TOOL_CALL_OPEN, start):
        inner = _skip(text, start + len(TOOL_CALL_OPEN))
        if text.startswith(FUNCTION_OPEN, inner):
            search_from = inner + 1  # the block's own function is not the next call
    following = _next_call(text, search_from)
    limit = len(text) if following is None else following
    close = text.find(TOOL_CALL_CLOSE, start, limit)
    return limit if close == -1 else close + len(TOOL_CALL_CLOSE)


def _tag_value(text: str, pos: int, what: str) -> tuple[str, int]:
    """The ``NAME`` of ``<function=NAME>`` or ``<parameter=NAME>`` and the position after its ``>``."""

    newline = text.find("\n", pos)
    end = text.find(">", pos, len(text) if newline == -1 else newline)
    if end == -1:
        raise _Malformed(f"a <{what}= tag is not closed")
    value = text[pos:end].strip()
    if not value:
        raise _Malformed(f"a <{what}= tag has no name")
    return value, end + 1


def _strip_newline(value: str) -> str:
    """A value without the newline the format puts after the open tag and the one before the close tag."""

    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value
