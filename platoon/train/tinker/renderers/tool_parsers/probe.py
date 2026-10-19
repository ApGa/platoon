"""Choosing a tool-call parser by a round trip through the model's own chat template.

``tool_call_parser: auto`` renders a probe conversation whose last assistant turn holds reasoning, a short text and
three tool calls, takes the text a model would sample for that turn (the render after the generation prompt, cut at
the first stop string), and keeps the first registered parser that reads the turn back exactly: the same calls in
order, with the same arguments by JSON value (``"3"`` is not ``3``), the text and the reasoning as far as the
template rendered them, and nothing unparsed or recovered.  The calls cover an integer, a multi-line string with
leading indentation, strings that read as a number and a boolean, a list, a boolean, and a call with no arguments.

A template that cannot put all three calls in one sampled turn (it raises on more than one, drops calls, or ends
the turn after the first, as Harmony does) is checked with one probe turn per call.  When no parser reads the probe
back and the template renders tools, ``auto`` refuses (:class:`ToolCallParserError`) with what the template rendered
and what each parser made of it.  When the template renders neither tools nor tool calls, ``auto`` parses none and
warns.  A parser set by name is checked the same way, and a mismatch only warns, since some templates cannot be
probed faithfully.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from platoon.train.tinker.renderers.tool_parsers import (
    AUTO,
    NONE,
    THINK_TAGS,
    ParseResult,
    ReasoningTags,
    SupportsParse,
    canonical_name,
    get_parser,
    parser_names,
)

logger = logging.getLogger(__name__)

# (messages, tools, add_generation_prompt) -> the rendered text.
RenderFn = Callable[[list[dict[str, Any]], "list[dict[str, Any]] | None", bool], str]

PROBE_REASONING = "I will call three tools."
PROBE_CONTENT = "Sure."
PROBE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather forecast.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "overwrite": {"type": "boolean"},
                    "note": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
PROBE_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("get_weather", {"location": "Tokyo, JP", "days": 3, "unit": "celsius"}),
    (
        "write_file",
        {
            "path": "123",
            "content": '    def f():\n        return {"a": [1, 2]}\n',
            "tags": ["x", "y"],
            "overwrite": True,
            "note": "true",
        },
    ),
    ("list_files", {}),
]
# Call ids in the shapes templates demand: Kimi K2 writes the id (which holds the name) into the turn, and Mistral's
# templates raise on any id that is not nine letters and digits.
_ID_STYLES: tuple[Callable[[str, int], str], ...] = (
    lambda name, index: f"functions.{name}:{index}",
    lambda name, index: f"probe{index:04d}",
)
_SHOWN = 1200  # characters of a rendered turn an error shows


class ToolCallParserError(ValueError):
    """No tool-call parser reads the chat template's own tool calls back, or the configured one does not exist."""


@dataclass(frozen=True)
class ParserChoice:
    """Which parser a renderer uses and why, as ``lock.json`` records it.

    ``setting`` is what the config asked for (``auto``, ``none``, a name or ``package.module:attr``) and ``parser``
    the parser in use (its registered name, the ``module:attr``, or ``none``).  ``verified`` is True when the parser
    read the template's own probe turn back exactly, False when it did not (a parser set by name), and None when
    there was nothing to check.  ``reason`` says how the parser was chosen.
    """

    setting: str
    parser: str
    verified: bool | None
    reason: str

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeTurn:
    """One probe turn: the text before it (ending in the generation prompt), what a model would sample, and what a
    parser must read from that."""

    prompt: str
    sampled: str
    calls: list[tuple[str, dict[str, Any]]]
    content: str
    reasoning: str


@dataclass(frozen=True)
class Probe:
    """The probe turns a template rendered (``mode``: one turn with every call, or one turn per call), whether it
    renders tools at all, and why it could not be probed, if it could not."""

    turns: list[ProbeTurn]
    renders_tools: bool
    mode: str = ""
    problem: str | None = None


def choose_parser(
    render: RenderFn,
    stop_strings: Sequence[str],
    *,
    setting: str = AUTO,
    reasoning_tags: ReasoningTags | str = AUTO,
    label: str = "",
) -> tuple[SupportsParse, ParserChoice]:
    """The parser for a chat template and why (see the module docstring).

    ``render`` renders messages with the renderer's own settings, ``stop_strings`` are the decoded stop tokens, and
    ``label`` names the model in messages.  Raises :class:`ToolCallParserError` when ``auto`` finds no parser for a
    template that renders tools, or when ``setting`` names no parser.
    """

    setting = setting or AUTO
    about = f" for {label}" if label else ""
    if setting == NONE:
        return get_parser(NONE, reasoning_tags=reasoning_tags), ParserChoice(NONE, NONE, None, "set to none")
    if setting != AUTO:
        try:
            parser = get_parser(setting, reasoning_tags=reasoning_tags)
        except ValueError as e:
            raise ToolCallParserError(str(e)) from e
        name = canonical_name(setting) or setting
        probe = probe_template(render, stop_strings)
        if probe.problem is not None:
            logger.info("Tool-call parser %s%s is not checked: %s", name, about, probe.problem)
            return parser, ParserChoice(setting, name, None, f"set by name; not checked: {probe.problem}")
        failure = check(parser, probe)
        if failure is not None:
            logger.warning(
                "Tool-call parser %s%s does not read the chat template's own tool calls back: %s", name, about, failure
            )
            return parser, ParserChoice(setting, name, False, f"set by name; the round trip failed: {failure}")
        return parser, ParserChoice(setting, name, True, f"set by name; round trip ({probe.mode})")
    probe = probe_template(render, stop_strings)
    if probe.problem is None:
        failures: dict[str, str] = {}
        for name in parser_names():
            parser = get_parser(name, reasoning_tags=reasoning_tags)
            failure = check(parser, probe)
            if failure is None:
                logger.info("Tool-call parser%s: %s (auto, round trip %s)", about, name, probe.mode)
                return parser, ParserChoice(AUTO, name, True, f"round trip ({probe.mode})")
            failures[name] = failure
        raise ToolCallParserError(_refusal(about, probe, failures))
    if not probe.renders_tools:
        logger.warning("The chat template%s renders no tools, so no tool calls are parsed", about)
        return get_parser(NONE, reasoning_tags=reasoning_tags), ParserChoice(
            AUTO, NONE, None, f"the chat template renders no tools ({probe.problem})"
        )
    raise ToolCallParserError(
        f"No tool-call parser can be checked{about}: the chat template renders tools, but {probe.problem}. {_ADVICE}"
    )


def probe_template(render: RenderFn, stop_strings: Sequence[str]) -> Probe:
    """Render the probe turns (one with every call, else one per call) through ``render``."""

    base = _messages([])[:2]
    try:
        renders_tools = render(base, PROBE_TOOLS, True) != render(base, None, True)
    except Exception as e:  # a template's raise_exception or a Jinja error
        return Probe([], False, problem=f"it does not render a plain prompt ({type(e).__name__}: {e})")
    last_name = PROBE_CALLS[-1][0]
    whole = _turn(render, stop_strings, PROBE_CALLS)
    if isinstance(whole, ProbeTurn) and last_name in whole.sampled:
        return Probe([whole], renders_tools, "one turn with every call")
    singles = [_turn(render, stop_strings, [call]) for call in PROBE_CALLS]
    if all(isinstance(turn, ProbeTurn) and turn.calls[0][0] in turn.sampled for turn in singles):
        return Probe([turn for turn in singles if isinstance(turn, ProbeTurn)], renders_tools, "one turn per call")
    if not isinstance(whole, ProbeTurn):
        problem = whole
    else:
        problem = "it leaves the assistant's tool calls out of the conversation"
    return Probe([], renders_tools, problem=problem)


def check(parser: SupportsParse, probe: Probe) -> str | None:
    """None when ``parser`` reads every probe turn back exactly, else what differed."""

    for turn in probe.turns:
        failure = _round_trip(parser, turn)
        if failure is not None:
            return failure if len(probe.turns) == 1 else f"{turn.calls[0][0]}: {failure}"
    return None


def prompt_ends_in(prompt: str, open_tag: str | None) -> bool:
    """Whether a generation prompt ends in ``open_tag`` (so the sampled text starts inside reasoning)."""

    return bool(open_tag) and prompt.rstrip().endswith(open_tag)  # type: ignore[arg-type]


def reasoning_open(parser: SupportsParse) -> str | None:
    """The tag that opens reasoning for ``parser``: its ``reasoning_open``, else ``<think>``."""

    return getattr(parser, "reasoning_open", THINK_TAGS[0])


def _round_trip(parser: SupportsParse, turn: ProbeTurn) -> str | None:
    in_reasoning = prompt_ends_in(turn.prompt, reasoning_open(parser))
    try:
        result = parser.parse(turn.sampled, tools=PROBE_TOOLS, in_reasoning=in_reasoning)
    except Exception as e:  # a user parser's bug is a failed check, not a crash
        return f"parse raised {type(e).__name__}: {e}"
    if not isinstance(result, ParseResult):
        return f"parse returned a {type(result).__name__}, not a ParseResult"
    problems = []
    try:
        got = [(call.name, _json(call.arguments)) for call in result.tool_calls]
    except (AttributeError, TypeError, ValueError) as e:
        return f"parse returned tool calls that are not ParsedToolCalls with JSON arguments ({type(e).__name__}: {e})"
    want = [(name, _json(arguments)) for name, arguments in turn.calls]
    if got != want:
        problems.append(_calls_differ(got, want))
    if result.unparsed:
        problems.append(f"left {len(result.unparsed)} call(s) unparsed ({result.unparsed[0].error})")
    if result.recovered:
        problems.append(f"needed a lenient rule for {result.recovered} call(s)")
    if result.content != turn.content:
        problems.append(f"read the text {_short(result.content)}, not {_short(turn.content)}")
    if result.reasoning != turn.reasoning:
        problems.append(f"read the reasoning {_short(result.reasoning)}, not {_short(turn.reasoning)}")
    return "; ".join(problems) or None


def _calls_differ(got: list[tuple[str, str]], want: list[tuple[str, str]]) -> str:
    """How the calls a parser read differ from the probe's: none, or the first that differs."""

    if not got:
        return "read no calls"
    index = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    read = f"{got[index][0]}({_short(got[index][1])})" if index < len(got) else "nothing more"
    expected = f"{want[index][0]}({_short(want[index][1])})" if index < len(want) else "nothing more"
    return f"read {len(got)} of {len(want)} calls; call {index + 1} is {read}, not {expected}"


def _turn(render: RenderFn, stop_strings: Sequence[str], calls: list[tuple[str, dict[str, Any]]]) -> ProbeTurn | str:
    """The probe turn for ``calls``, or why the template does not render it."""

    error: Exception | None = None
    for id_style in _ID_STYLES:
        messages = _messages(calls, id_style)
        try:
            prompt = render(messages[:-1], PROBE_TOOLS, True)
            full = render(messages, PROBE_TOOLS, False)
        except Exception as e:  # a template's raise_exception or a Jinja error: try the next id shape
            error = e
            continue
        common = _common_prefix(prompt, full)
        sampled = _cut(full[common:], stop_strings)
        return ProbeTurn(
            prompt=full[:common],
            sampled=sampled,
            calls=[(name, dict(arguments)) for name, arguments in calls],
            content=PROBE_CONTENT if PROBE_CONTENT in sampled else "",
            reasoning=PROBE_REASONING if PROBE_REASONING in sampled else "",
        )
    return f"it raises on an assistant turn with {len(calls)} tool call(s) ({type(error).__name__}: {error})"


def _messages(
    calls: list[tuple[str, dict[str, Any]]], id_style: Callable[[str, int], str] = _ID_STYLES[0]
) -> list[dict[str, Any]]:
    assistant: dict[str, Any] = {
        "role": "assistant",
        "content": PROBE_CONTENT,
        "reasoning_content": PROBE_REASONING,
        "thinking": PROBE_REASONING,
        "tool_calls": [
            {"id": id_style(name, index), "type": "function", "function": {"name": name, "arguments": dict(arguments)}}
            for index, (name, arguments) in enumerate(calls)
        ],
    }
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Check the weather in Tokyo, write the file and list the files."},
        assistant,
    ]


def _common_prefix(a: str, b: str) -> int:
    length = min(len(a), len(b))
    index = 0
    while index < length and a[index] == b[index]:
        index += 1
    return index


def _cut(text: str, stop_strings: Sequence[str]) -> str:
    """``text`` before the first stop string."""

    cuts = [found for stop in stop_strings if stop and (found := text.find(stop)) != -1]
    return text[: min(cuts)] if cuts else text


def _json(arguments: Mapping[str, Any]) -> str:
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _short(text: str, limit: int = 60) -> str:
    return repr(text if len(text) <= limit else text[:limit] + "...")


_ADVICE = (
    "Set renderer_kwargs.tool_call_parser to the parser that matches the model ("
    + ", ".join(parser_names())
    + "), to package.module:attr for a parser of your own, or to none to parse no tool calls; "
    "renderer_kwargs.reasoning_tags sets the reasoning tags when the model's differ from its format's."
)


def _refusal(about: str, probe: Probe, failures: Mapping[str, str]) -> str:
    sampled = "\n\n".join(turn.sampled for turn in probe.turns)
    shown = sampled if len(sampled) <= _SHOWN else sampled[:_SHOWN] + f"... ({len(sampled)} characters)"
    lines = [
        f"No tool-call parser reads the chat template's own tool calls back{about}.",
        f"The template renders the probe ({probe.mode}) as:",
        *("    " + line for line in shown.splitlines()),
        "What each parser made of it:",
        *(f"    {name}: {failure}" for name, failure in failures.items()),
        _ADVICE,
    ]
    return "\n".join(lines)
