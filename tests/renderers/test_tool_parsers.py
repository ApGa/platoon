"""Every built-in tool-call parser on well-formed and malformed output, and the registry that names them.

Each format has a writer that lays calls out as that format's chat templates and models do.  A well-formed response
parses exactly: the same calls in order, with the same arguments by JSON value, and the text and reasoning around
them.  A malformed one parses exactly by a lenient rule (the call is marked recovered), or what does not parse is an
unparsed call with its raw text: never silently dropped, never a false call.  Real chat templates are covered by
``test_tool_parsing_templates.py``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import pytest

from platoon.train.tinker.renderers.tool_parsers import (
    NoToolCalls,
    ParseResult,
    ToolCallParser,
    canonical_name,
    get_parser,
    parser_aliases,
    parser_names,
    reasoning_tags_setting,
)
from platoon.train.tinker.renderers.tool_parsers.arguments import (
    MAX_JSON_DEPTH,
    ToolSchemas,
    coerce_value,
    schema_types,
)
from platoon.train.tinker.renderers.tool_parsers.base import json_value

REASONING = "I will call three tools."
CONTENT = "Sure."
CODE = "    def f():\n        return {\"a\": '</parameter>'}\n"
CALLS: list[tuple[str, dict[str, Any]]] = [
    ("get_weather", {"location": "Tokyo, JP", "days": 3, "unit": "celsius"}),
    ("write_file", {"path": "123", "content": CODE, "tags": ["x", "y"], "overwrite": True, "note": "true"}),
    ("list-dir.v2", {}),
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "overwrite": {"type": "boolean"},
                    "note": {"type": "string"},
                },
            },
        },
    },
    {"type": "function", "function": {"name": "list-dir.v2", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "list_dir_v2", "parameters": {"type": "object", "properties": {}}}},
]


def _text(value: Any) -> str:
    """A value as XML-style templates write it: strings as they are, anything else as JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


# --------------------------------------------------------------------------- writers, one per format


def qwen3_coder(calls, reasoning=REASONING, content=CONTENT) -> str:
    """Qwen3.6's template: think block, content, then the calls."""
    out = f"<think>\n{reasoning}\n</think>\n\n{content}"
    for i, (name, args) in enumerate(calls):
        out += ("\n\n" if content else "") if i == 0 else "\n"
        out += f"<tool_call>\n<function={name}>\n"
        out += "".join(f"<parameter={key}>\n{_text(value)}\n</parameter>\n" for key, value in args.items())
        out += "</function>\n</tool_call>"
    return out


def hermes(calls, reasoning=REASONING, content=CONTENT) -> str:
    """Qwen3's template (the hybrid models: a think block, then Qwen3-2507's call layout)."""
    out = f"<think>\n{reasoning}\n</think>\n\n{content}"
    for i, (name, args) in enumerate(calls):
        out += "\n" if (i == 0 and content) or i else ""
        out += f'<tool_call>\n{{"name": "{name}", "arguments": {json.dumps(args, ensure_ascii=False)}}}\n</tool_call>'
    return out


def glm45(calls, reasoning=REASONING, content=CONTENT) -> str:
    out = f"<think>{reasoning}</think>\n{content}"
    for name, args in calls:
        out += f"\n<tool_call>{name}\n"
        out += "".join(f"<arg_key>{k}</arg_key>\n<arg_value>{_text(v)}</arg_value>\n" for k, v in args.items())
        out += "</tool_call>"
    return out


def glm47(calls, reasoning=REASONING, content=CONTENT) -> str:
    out = f"<think>{reasoning}</think>{content}"
    for name, args in calls:
        out += f"<tool_call>{name}"
        out += "".join(f"<arg_key>{k}</arg_key><arg_value>{_text(v)}</arg_value>" for k, v in args.items())
        out += "</tool_call>"
    return out


def deepseekv3(calls, reasoning=REASONING, content=CONTENT) -> str:
    out = f"<think>{reasoning}</think>{content}<｜tool▁calls▁begin｜>"
    out += "\n".join(
        f"<｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\n```json\n{json.dumps(args)}\n```<｜tool▁call▁end｜>"
        for name, args in calls
    )
    return out + "<｜tool▁calls▁end｜>"


def deepseekv31(calls, reasoning=REASONING, content=CONTENT) -> str:
    out = f"<think>{reasoning}</think>{content}<｜tool▁calls▁begin｜>"
    out += "".join(
        f"<｜tool▁call▁begin｜>{name}<｜tool▁sep｜>{json.dumps(args)}<｜tool▁call▁end｜>" for name, args in calls
    )
    return out + "<｜tool▁calls▁end｜>"


def kimi_k2(calls, reasoning=REASONING, content=CONTENT) -> str:
    out = f"<think>{reasoning}</think>{content}<|tool_calls_section_begin|>"
    out += "".join(
        f"<|tool_call_begin|>functions.{name}:{i}<|tool_call_argument_begin|>{json.dumps(args)}<|tool_call_end|>"
        for i, (name, args) in enumerate(calls)
    )
    return out + "<|tool_calls_section_end|>"


def mistral(calls, reasoning=REASONING, content=CONTENT) -> str:
    """Tokenizer v11 and later, with Magistral's reasoning."""
    out = f"[THINK]{reasoning}[/THINK]{content}"
    return out + "".join(f"[TOOL_CALLS]{name}[ARGS]{json.dumps(args)}" for name, args in calls)


def mistral_v3(calls, reasoning=REASONING, content=CONTENT) -> str:
    """Tokenizer v3: a JSON list with nine-character ids; v3 models do not reason in tags."""
    items = [{"name": name, "arguments": args, "id": f"abcdefgh{i}"} for i, (name, args) in enumerate(calls)]
    return f"{content}[TOOL_CALLS]{json.dumps(items)}"


def llama3(calls, reasoning=REASONING, content=CONTENT) -> str:
    """Llama 3.x: the call is the message (content and reasoning are not written)."""
    return "; ".join(f'{{"name": "{name}", "parameters": {json.dumps(args)}}}' for name, args in calls)


def pythonic(calls, reasoning=REASONING, content=CONTENT) -> str:
    """The call list is the message after the reasoning (content is not written)."""

    def call(name: str, args: dict[str, Any]) -> str:
        return f"{name}(" + ", ".join(f"{key}={value!r}" for key, value in args.items()) + ")"

    return f"<think>{reasoning}</think>\n[" + ", ".join(call(n, a) for n, a in calls) + "]"


def gpt_oss(calls, reasoning=REASONING, content=CONTENT) -> str:
    """One sampled Harmony turn holds one call (the sampler stops at <|call|>)."""
    ((name, args),) = calls
    return (
        f"<|channel|>analysis<|message|>{reasoning}<|end|><|start|>assistant<|channel|>commentary "
        f"to=functions.{name} <|constrain|>json<|message|>{json.dumps(args)}"
    )


Writer = Callable[..., str]
# format name, writer, the calls it can write, the content and reasoning a parse must read back.
FORMATS: list[tuple[str, Writer, list, str, str]] = [
    ("qwen3_coder", qwen3_coder, CALLS, CONTENT, REASONING),
    ("hermes", hermes, CALLS, CONTENT, REASONING),
    ("glm45", glm45, CALLS, CONTENT, REASONING),
    ("glm45", glm47, CALLS, CONTENT, REASONING),
    ("deepseekv3", deepseekv3, CALLS, CONTENT, REASONING),
    ("deepseekv31", deepseekv31, CALLS, CONTENT, REASONING),
    ("kimi_k2", kimi_k2, CALLS, CONTENT, REASONING),
    ("mistral", mistral, CALLS, CONTENT, REASONING),
    ("mistral", mistral_v3, CALLS, CONTENT, ""),
    ("llama3", llama3, CALLS[:1], "", ""),
    ("llama3", llama3, CALLS[:2], "", ""),
    ("pythonic", pythonic, [*CALLS[:2], ("list_dir_v2", {})], "", REASONING),
    ("gpt-oss", gpt_oss, CALLS[1:2], "", REASONING),
]
IDS = [f"{name}-{writer.__name__}-{len(calls)}" for name, writer, calls, *_ in FORMATS]


def _calls(result: ParseResult) -> list[tuple[str, str]]:
    return [(call.name, json.dumps(call.arguments, sort_keys=True)) for call in result.tool_calls]


def _want(calls) -> list[tuple[str, str]]:
    return [(name, json.dumps(args, sort_keys=True)) for name, args in calls]


def parse(name: str, text: str, **kwargs: Any) -> ParseResult:
    return get_parser(name).parse(text, tools=TOOLS, **kwargs)


# --------------------------------------------------------------------------- every format


@pytest.mark.parametrize(("name", "writer", "calls", "content", "reasoning"), FORMATS, ids=IDS)
def test_a_well_formed_response_parses_exactly(name, writer, calls, content, reasoning):
    result = parse(name, writer(calls))
    assert _calls(result) == _want(calls)
    assert (result.content, result.reasoning) == (content, reasoning)
    assert result.unparsed == [] and result.recovered == 0


@pytest.mark.parametrize(("name", "writer", "calls", "content", "reasoning"), FORMATS, ids=IDS)
def test_a_response_cut_inside_its_last_call_keeps_the_others_and_reports_the_rest(
    name, writer, calls, content, reasoning
):
    text = writer(calls)
    last_name = calls[-1][0]
    cut = text.rfind(last_name) + len(last_name) // 2  # inside the last call's name
    result = parse(name, text[:cut])
    assert _calls(result) == _want(calls[:-1])
    assert len(result.unparsed) == 1
    assert text[:cut].endswith(result.unparsed[0].raw_text.rstrip()) or result.unparsed[0].raw_text in text
    assert last_name[: len(last_name) // 2] in result.unparsed[0].raw_text


JSON_FORMATS = [entry for entry in FORMATS if entry[0] not in ("qwen3_coder", "glm45", "pythonic")]


@pytest.mark.parametrize(
    ("name", "writer", "calls", "content", "reasoning"),
    [entry for entry in JSON_FORMATS if len(entry[2]) > 1],
    ids=[f"{e[0]}-{e[1].__name__}" for e in JSON_FORMATS if len(e[2]) > 1],
)
def test_bad_json_in_one_call_is_unparsed_on_its_own(name, writer, calls, content, reasoning):
    text = writer(calls)
    broken = text.replace('"path": "123"', '"path": 123"', 1)
    assert broken != text
    result = parse(name, broken)
    if writer in (mistral_v3, llama3):  # one JSON document: the calls before the broken one parse, the rest does not
        assert _calls(result) == _want(calls[:1]) and len(result.unparsed) == 1
        return
    assert _calls(result) == _want([call for call in calls if call[0] != "write_file"])
    assert len(result.unparsed) == 1 and '"path": 123"' in result.unparsed[0].raw_text


@pytest.mark.parametrize("name", parser_names())
def test_plain_text_is_content_and_a_parser_never_raises(name):
    parser = get_parser(name)
    assert parser.parse("Hello there.").content == "Hello there."
    for text in ("", "<tool_call>", "[TOOL_CALLS]", "<|channel|>", "{", "[f(", "<｜tool▁call▁begin｜>", "\x00"):
        result = parser.parse(text, tools=TOOLS)
        assert isinstance(result, ParseResult) and result.tool_calls == []


# A whole call in each JSON format, with its arguments at %s.
JSON_CALLS = {
    "hermes": '<tool_call>\n{"name": "f", "arguments": %s}\n</tool_call>',
    "kimi_k2": "<|tool_calls_section_begin|><|tool_call_begin|>functions.f:0<|tool_call_argument_begin|>%s"
    "<|tool_call_end|><|tool_calls_section_end|>",
    "deepseekv31": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>f<｜tool▁sep｜>%s"
    "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    "deepseekv3": "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>f\n```json\n%s\n```"
    "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    "mistral": "[TOOL_CALLS]f[ARGS]%s",
    "llama3": '{"name": "f", "parameters": %s}',
    "gpt-oss": "<|channel|>commentary to=functions.f <|constrain|>json<|message|>%s",
}
NOT_JSON_VALUES = {
    "nested 20000 deep": '{"a": ' + "[" * 20000 + "]" * 20000 + "}",
    "nested one level deeper than the limit": '{"a": ' + "[" * MAX_JSON_DEPTH + "]" * MAX_JSON_DEPTH + "}",
    "out of a float's range": '{"a": 1e999}',
    "NaN": '{"a": NaN}',
    "-Infinity": '{"a": [-Infinity]}',
}
NOT_JSON_LITERALS = ["1e999", "{1, 2}", "b'x'", "{(1, 2): 3}"]


@pytest.mark.parametrize("name", list(JSON_CALLS))
@pytest.mark.parametrize("arguments", list(NOT_JSON_VALUES.values()), ids=list(NOT_JSON_VALUES))
def test_arguments_json_cannot_hold_make_the_call_unparsed_not_an_error(name, arguments):
    result = parse(name, JSON_CALLS[name] % arguments)
    assert result.tool_calls == [] and len(result.unparsed) == 1


def test_json_decodes_as_deep_as_the_limit_on_every_python():
    # Python's json module decodes about 10,000 levels on 3.12 and 3.13 and as many as the stack holds on 3.14, so a
    # response would parse differently by Python and by thread; the limit is the same everywhere.
    deepest = "[" * MAX_JSON_DEPTH + "]" * MAX_JSON_DEPTH
    value, end, lenient = json_value(deepest)
    assert end == len(deepest) and not lenient
    with pytest.raises(ValueError, match=f"nests more than {MAX_JSON_DEPTH} levels deep"):
        json_value("[" + deepest + "]")
    assert coerce_value(deepest, {"type": "array"}) == value
    assert coerce_value("[" + deepest + "]", {"type": "array"}) == "[" + deepest + "]"


@pytest.mark.parametrize("value", NOT_JSON_LITERALS, ids=["1e999", "set", "bytes", "tuple key"])
def test_pythonic_values_json_cannot_hold_make_the_call_unparsed_not_an_error(value):
    result = parse("pythonic", f"[f(a={value}), list_dir_v2()]")
    assert _calls(result) == [("list_dir_v2", "{}")] and len(result.unparsed) == 1
    deep = parse("pythonic", "[f(a=" + "[" * 20000 + "]" * 20000 + ")]")  # deeper than Python's tokenizer reads
    assert deep.tool_calls == [] and len(deep.unparsed) == 1


@pytest.mark.parametrize("name", [n for n in parser_names() if n != "gpt-oss"])
def test_reasoning_starts_in_the_prompt_or_with_the_open_tag(name):
    parser = get_parser(name)
    open_tag, close_tag = parser.reasoning_tags
    assert parser.parse(f"thinking{close_tag}answer", in_reasoning=True).reasoning == "thinking"
    assert parser.parse(f"{open_tag}thinking{close_tag}answer").reasoning == "thinking"
    # A re-emitted open tag, and an empty block whose open tag was the prompt's.
    assert parser.parse(f"{open_tag}\nagain{close_tag}x", in_reasoning=True).reasoning == "again"
    assert parser.parse(f"{close_tag}answer").content == "answer"
    # Cut off while reasoning: all of it is reasoning.
    result = parser.parse("still thinking", in_reasoning=True)
    assert (result.reasoning, result.content) == ("still thinking", "")
    # Without the prompt's open tag, a close tag later in the text is just text.
    assert parser.parse(f"answer {close_tag} text").reasoning == ""


def test_reasoning_tags_can_be_overridden_or_turned_off():
    parser = get_parser("hermes", reasoning_tags=["<seed:think>", "</seed:think>"])
    result = parser.parse('<seed:think>plan</seed:think>ok\n<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>')
    assert (result.reasoning, result.content, _calls(result)) == ("plan", "ok", [("a", "{}")])
    assert get_parser("qwen3_coder", reasoning_tags=None).parse("<think>x</think>y").content == "<think>x</think>y"
    assert reasoning_tags_setting("none") is None and reasoning_tags_setting("auto") == "auto"
    with pytest.raises(ValueError, match="reasoning_tags"):
        reasoning_tags_setting(["<think>"])


# --------------------------------------------------------------------------- qwen3_coder


def _qwen(calls=CALLS) -> str:
    return qwen3_coder(calls)


def _replace(text: str, old: str, new: str) -> str:
    assert old in text, old
    return text.replace(old, new, 1)


QWEN_RECOVERED = {
    "missing inner </parameter>": lambda t: _replace(t, "3\n</parameter>\n", "3\n"),
    "missing last </parameter>": lambda t: _replace(t, "celsius\n</parameter>\n", "celsius\n"),
    "missing </function>": lambda t: _replace(t, "celsius\n</parameter>\n</function>", "celsius\n</parameter>"),
    "missing final </tool_call>": lambda t: t[: t.rfind("</tool_call>")],
    "missing inner </tool_call>": lambda t: _replace(
        t, "</function>\n</tool_call>\n<tool_call>", "</function>\n<tool_call>"
    ),
    "<function= without <tool_call>": lambda t: _replace(
        t, "<tool_call>\n<function=list-dir.v2>", "<function=list-dir.v2>"
    ),
    "literal </parameter> value + missing inner </parameter>": lambda t: _replace(t, "3\n</parameter>\n", "3\n"),
}


@pytest.mark.parametrize("variant", list(QWEN_RECOVERED))
def test_qwen3_coder_recovers_missing_close_tags_exactly(variant):
    result = parse("qwen3_coder", QWEN_RECOVERED[variant](_qwen()), in_reasoning=False)
    assert _calls(result) == _want(CALLS)
    assert result.unparsed == [] and result.recovered == 1
    assert (result.content, result.reasoning) == (CONTENT, REASONING)


def test_qwen3_coder_well_formed_hard_cases_are_exact_without_recovery():
    """The four calls the previous parser dropped or cut: a zero-argument call next to others, a literal
    </parameter> inside a value, an unclosed final call's siblings, and calls after an unclosed think block."""
    text = _qwen()
    result = parse("qwen3_coder", text)
    assert _calls(result) == _want(CALLS) and result.recovered == 0  # zero-argument call and the literal close tag
    unclosed_think = text.replace("\n</think>", "", 1)
    result = parse("qwen3_coder", unclosed_think)
    assert _calls(result) == _want(CALLS) and result.reasoning == REASONING + "\n\n" + CONTENT
    prefilled = text.split("<think>\n", 1)[1]  # the generation prompt ended in <think>
    result = parse("qwen3_coder", prefilled, in_reasoning=True)
    assert _calls(result) == _want(CALLS) and (result.content, result.reasoning) == (CONTENT, REASONING)


@pytest.mark.parametrize(
    "value",
    [
        "p = '<parameter=path>'\n",
        "s = '</function>\\n</tool_call>'\n",
        "x = '</think>'\n",
        "line\n</parameter>\nmore\n",
        "<tool_call>\n<function=get_weather>\n<parameter=location>\nX\n</parameter>\n</function>\n</tool_call>\n",
        "\n\n  indented\n\n",
        "",
    ],
    ids=["opener", "function close", "think close", "close on its own line", "whole call", "blank lines", "empty"],
)
def test_qwen3_coder_values_may_hold_literal_markup(value):
    calls = [("write_file", {"path": "a.py", "content": value, "tags": ["x"], "overwrite": False}), ("list-dir.v2", {})]
    result = parse("qwen3_coder", qwen3_coder(calls))
    assert _calls(result) == _want(calls) and result.unparsed == [] and result.recovered == 0


OPENHANDS_EXAMPLE = (
    "Call tools like this:\n<function=get_weather>\n<parameter=location>\nrm -rf build/\n</parameter>\n</function>\n"
)
INLINE_EXAMPLE = (
    "<function=write_file>\n<parameter=path>a.py</parameter>\n<parameter=content>\nx = 1\n</parameter>\n</function>\n"
)
HERMES_BLOCK = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'


@pytest.mark.parametrize(
    "value",
    [
        OPENHANDS_EXAMPLE,
        INLINE_EXAMPLE,
        f"Two examples:\n{HERMES_BLOCK}\n{HERMES_BLOCK.replace('Paris', 'Oslo')}\n",
        "Start with:\n<function=get_weather>\n" + OPENHANDS_EXAMPLE,
        "X\n<parameter=note>\nY\n",
    ],
    ids=[
        "call without <tool_call>",
        "one-line parameters",
        "two Hermes blocks",
        "an open call, then a closed one",
        "a later parameter's line",
    ],
)
def test_qwen3_coder_values_may_hold_calls_written_out(value):
    """A value that documents calls (OpenHands' prompts write them without <tool_call>) or holds a line that opens a
    parameter the call gives later parses exactly: neither the example nor the line is taken for the call's end."""
    calls = [("write_file", {"path": "doc.md", "content": value, "note": "n"}), ("list-dir.v2", {})]
    result = parse("qwen3_coder", qwen3_coder(calls))
    assert _calls(result) == _want(calls) and result.unparsed == [] and result.recovered == 0


def test_qwen3_coder_does_not_run_a_call_written_out_inside_a_broken_value():
    """A value whose close tags are missing, followed by a call without <tool_call>: the broken call is unparsed and
    the next one runs, rather than the first running with the second inside its value."""
    text = (
        "<tool_call>\n<function=get_weather>\n<parameter=location>\nTok\n"
        "<function=write_file>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>"
    )
    result = parse("qwen3_coder", text)
    assert _calls(result) == _want([("write_file", {"path": "a.py"})]) and result.recovered == 1
    assert len(result.unparsed) == 1 and result.unparsed[0].raw_text.startswith("<tool_call>\n<function=get_weather>")
    xml_then_hermes = qwen3_coder(CALLS[:1]) + "\n" + HERMES_BLOCK  # a Hermes call after an XML call: unparsed
    result = parse("qwen3_coder", xml_then_hermes)
    assert _calls(result) == _want(CALLS[:1]) and [u.raw_text for u in result.unparsed] == [HERMES_BLOCK]


def test_qwen3_coder_splits_a_value_at_a_line_that_opens_a_parameter_not_given():
    """The format does not escape markup: a value that holds a line opening a parameter the call has not given yet,
    and does not give later, reads as that parameter with the value's </parameter> missing (as the previous parser
    read it).  The call is marked recovered."""
    calls = [("write_file", {"path": "doc.md", "content": "Format:\n<parameter=note>\nvalue\n"})]
    result = parse("qwen3_coder", qwen3_coder(calls))
    assert result.tool_calls[0].arguments == {"path": "doc.md", "content": "Format:", "note": "value\n"}
    assert result.recovered == 1


def test_qwen3_coder_marks_a_call_that_gives_a_parameter_twice_recovered():
    """A value that holds a close tag and then a line opening a parameter the call already has cannot be told from
    the end of the value; the repeated parameter shows the call was not read as written."""
    calls = [
        ("write_file", {"path": "p.py", "content": "<parameter=note>\n", "note": "</parameter>\n<parameter=content>\n"})
    ]
    result = parse("qwen3_coder", qwen3_coder(calls))
    assert _calls(result) == _want([("write_file", {"path": "p.py", "content": "", "note": ""})])
    assert result.recovered == 1
    repeated = _replace(qwen3_coder(CALLS[:1]), "</function>", "<parameter=days>\n4\n</parameter>\n</function>")
    result = parse("qwen3_coder", repeated)  # a model that gives a parameter twice: the last value counts
    assert result.tool_calls[0].arguments["days"] == 4 and result.recovered == 1


def test_qwen3_coder_reports_what_it_cannot_parse():
    text = _qwen()
    k = text.find("<parameter=content>")
    result = parse("qwen3_coder", text[: k + 40])  # cut inside the second call's value
    assert _calls(result) == _want(CALLS[:1]) and len(result.unparsed) == 1
    assert result.unparsed[0].raw_text.startswith("<tool_call>\n<function=write_file>")
    hermes_json = (
        'ok\n</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"location": "Tokyo"}}\n</tool_call>'
    )
    result = parse("qwen3_coder", hermes_json, in_reasoning=True)
    assert result.tool_calls == [] and result.unparsed[0].raw_text.startswith("<tool_call>\n{")
    assert result.unparsed[0].error == "<tool_call> holds no <function=NAME>"
    # A <tool_call> in running text that holds no function is text.
    result = parse("qwen3_coder", "Use <tool_call> tags to call tools.")
    assert result.content == "Use <tool_call> tags to call tools." and result.unparsed == []


def test_qwen3_coder_types_values_by_the_tool_schema():
    qwen35 = _replace(qwen3_coder(CALLS), "<parameter=overwrite>\ntrue\n", "<parameter=overwrite>\nTrue\n")
    assert _calls(parse("qwen3_coder", qwen35)) == _want(CALLS)  # Qwen3.5 writes Python's str(): True, None
    untyped = get_parser("qwen3_coder").parse(qwen3_coder(CALLS[:1]))  # no tools: every value stays text
    assert untyped.tool_calls[0].arguments == {"location": "Tokyo, JP", "days": "3", "unit": "celsius"}
    unknown = parse("qwen3_coder", qwen3_coder([("get_weather", {"location": "x", "extra": 5})]))
    assert unknown.tool_calls[0].arguments == {"location": "x", "extra": "5"}  # an undeclared parameter stays text


@pytest.mark.parametrize(
    "unit",
    [
        "<tool_call>\n<function=get_weather>\n<parameter=location>\nx\n",
        "\n</parameter>\n</function>\n</tool_call>",
        "\n</function>",
        HERMES_BLOCK + "\n",
        "</parameter>\n<parameter=path>\nq\n",
        "<parameter=note>\nz\n",
        OPENHANDS_EXAMPLE,
        "<tool_call>\n<function=a>\n<parameter=b>\n1\n</parameter>\n</function>\n",
    ],
    ids=[
        "cut calls",
        "close tags",
        "function closes",
        "hermes blocks",
        "close and open",
        "parameter lines",
        "calls without <tool_call>",
        "calls without </tool_call>",
    ],
)
def test_qwen3_coder_is_fast_on_degenerate_output(unit):
    """A policy stuck repeating markup up to the token limit (about 200 kB) parses in tens of milliseconds, whether
    the repeats follow a call or sit inside its value."""
    repeats = unit * (200_000 // len(unit))
    for text in (repeats, qwen3_coder([("write_file", {"path": "a", "content": repeats})])):
        started = time.perf_counter()
        parse("qwen3_coder", text)
        assert time.perf_counter() - started < 2.0


def test_qwen3_coder_is_fast_on_long_responses():
    reasoning = "Let me think about <tool_call> formats and the file layout.\n" * 3000
    code = ("    def f(x):\n        return {'a': '</parameter>'}  # " + " " * 200 + "\n") * 200
    calls = [("write_file", {"path": "a.py", "content": code})] * 4
    text = qwen3_coder(calls, reasoning=reasoning)
    for variant in (text, text.replace("a.py\n</parameter>", "a.py", 1), text.replace("\n</think>", "", 1)):
        started = time.perf_counter()
        result = parse("qwen3_coder", variant)
        assert time.perf_counter() - started < 2.0
        assert [call.arguments["content"] for call in result.tool_calls] == [code] * 4


# --------------------------------------------------------------------------- hermes and the JSON formats


def test_hermes_lenient_rules_mark_calls_recovered():
    raw_newline = hermes([("write_file", {"path": "a", "content": "x"})]).replace('"x"', '"line1\nline2"')
    result = parse("hermes", raw_newline)
    assert result.tool_calls[0].arguments["content"] == "line1\nline2" and result.recovered == 1
    as_string = '<tool_call>\n{"name": "get_weather", "arguments": "{\\"location\\": \\"Oslo\\"}"}\n</tool_call>'
    result = parse("hermes", as_string)
    assert result.tool_calls[0].arguments == {"location": "Oslo"} and result.recovered == 1
    missing_close = '<tool_call>\n{"name": "list-dir.v2", "arguments": {}}'
    assert parse("hermes", missing_close).recovered == 1


def test_hermes_reads_lists_literal_close_tags_and_ignores_prose_mentions():
    listed = '<tool_call>\n[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"x": 1}}]\n</tool_call>'
    assert _calls(parse("hermes", listed)) == [("a", "{}"), ("b", '{"x": 1}')]
    literal = hermes([("write_file", {"path": "a", "content": "</tool_call>\n<tool_call>"})])
    result = parse("hermes", literal)
    assert result.tool_calls[0].arguments["content"] == "</tool_call>\n<tool_call>" and result.unparsed == []
    prose = parse("hermes", "Wrap calls in <tool_call> tags, then stop.")
    assert prose.content == "Wrap calls in <tool_call> tags, then stop." and prose.unparsed == []
    garbage = parse("hermes", "ok\n<tool_call>\nnot json\n</tool_call>")
    assert garbage.content == "ok" and garbage.unparsed[0].raw_text == "<tool_call>\nnot json\n</tool_call>"


def test_hermes_reads_every_call_object_in_a_block():
    two = '<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n{"name": "b", "arguments": {"y": 2}}\n</tool_call>'
    result = parse("hermes", two)
    assert _calls(result) == [("a", '{"x": 1}'), ("b", '{"y": 2}')] and result.recovered == 2
    broken = '<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n{"name": "b", "argu\n</tool_call>\nDone.'
    for text in (broken, broken[: broken.find("\n</tool_call>")]):  # with and without its </tool_call>
        result = parse("hermes", text)
        assert _calls(result) == [("a", '{"x": 1}')] and result.content == ("Done." if text == broken else "")
        assert [u.raw_text.split("\n")[0] for u in result.unparsed] == ['{"name": "b", "argu']


def test_llama3_reads_the_python_tag_and_leaves_json_answers_alone():
    result = parse("llama3", 'Calling.<|python_tag|>{"name": "get_weather", "parameters": {"location": "Oslo"}}')
    assert (result.content, _calls(result)) == ("Calling.", [("get_weather", '{"location": "Oslo"}')])
    for answer in ('{"answer": 42}', '{"name": "John Smith", "age": 42}'):  # calls have parameters (or arguments)
        result = parse("llama3", answer)
        assert (result.content, result.tool_calls, result.unparsed) == (answer, [], [])
    assert _calls(parse("llama3", '{"name": "list-dir.v2", "arguments": {}}')) == [("list-dir.v2", "{}")]


def test_mistral_keeps_or_mints_nine_character_ids():
    listed = parse("mistral", mistral_v3(CALLS))
    assert [call.id for call in listed.tool_calls] == ["abcdefgh0", "abcdefgh1", "abcdefgh2"]
    compact = parse("mistral", '[TOOL_CALLS]get_weather[CALL_ID]a1b2c3d4e[ARGS]{"location": "Oslo"}')
    assert compact.tool_calls[0].id == "a1b2c3d4e"
    minted = parse("mistral", mistral(CALLS[:1])).tool_calls[0].id
    assert minted is not None and len(minted) == 9 and minted.isalnum()
    assert parse("mistral", "<think>plan</think>ok").reasoning == "plan"  # Magistral's earlier tags


def test_kimi_k2_keeps_the_sampled_id_and_refuses_ids_without_a_name():
    result = parse("kimi_k2", kimi_k2(CALLS))
    assert [call.id for call in result.tool_calls] == [f"functions.{name}:{i}" for i, (name, _) in enumerate(CALLS)]
    bare = kimi_k2(CALLS[:1]).replace("functions.get_weather:0", "3")
    result = parse("kimi_k2", bare)
    assert result.tool_calls == [] and "holds no function name" in result.unparsed[0].error


def test_deepseek_keeps_names_with_dots_and_hyphens_and_recovers_missing_ends():
    result = parse("deepseekv3", deepseekv3(CALLS).replace("<｜tool▁call▁end｜><｜tool▁calls▁end｜>", ""))
    assert _calls(result) == _want(CALLS) and result.recovered == 1
    result = parse("deepseekv31", "</think>" + deepseekv31(CALLS, reasoning="").split("</think>", 1)[1])
    assert _calls(result) == _want(CALLS) and result.reasoning == ""


def test_gpt_oss_reads_channels_recipients_and_preambles():
    call = parse("gpt-oss", "<|start|>assistant to=functions.list-dir.v2<|channel|>commentary json<|message|>{}")
    assert _calls(call) == [("list-dir.v2", "{}")]
    final = parse(
        "gpt-oss", "<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>Hi"
    )
    assert (final.reasoning, final.content, final.tool_calls) == ("think", "Hi", [])
    preamble = parse(
        "gpt-oss",
        "<|channel|>commentary<|message|>Checking.<|end|><|start|>assistant<|channel|>commentary "
        'to=functions.get_weather <|constrain|>json<|message|>{"location": "Oslo"}',
    )
    assert (preamble.content, _calls(preamble)) == ("Checking.", [("get_weather", '{"location": "Oslo"}')])
    broken = parse("gpt-oss", "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{oops")
    assert broken.tool_calls == [] and len(broken.unparsed) == 1
    assert parse("gpt-oss", "no markup at all").content == "no markup at all"


def test_glm45_recovers_a_missing_value_close_and_refuses_a_cut_name():
    text = _replace(glm45(CALLS), "Tokyo, JP</arg_value>", "Tokyo, JP")
    result = parse("glm45", text)
    assert _calls(result) == _want(CALLS) and result.recovered == 1
    literal = glm47([("write_file", {"path": "a", "content": "x</arg_value>y"})])
    assert parse("glm45", literal).tool_calls[0].arguments["content"] == "x</arg_value>y"
    result = parse("glm45", glm47(CALLS)[: -len("dir.v2</tool_call>")])
    assert _calls(result) == _want(CALLS[:2]) and "cut off" in result.unparsed[0].error


def test_pythonic_isolates_a_bad_call_and_reads_json_literals():
    result = parse("pythonic", '[get_weather(location="Oslo", days=true), write_file("a.py"), list_dir_v2()]')
    assert _calls(result) == [("get_weather", '{"days": true, "location": "Oslo"}'), ("list_dir_v2", "{}")]
    assert len(result.unparsed) == 1 and "positional" in result.unparsed[0].error
    wrapped = parse("pythonic", "<|python_start|>[list_dir_v2()]<|python_end|>")
    assert (wrapped.content, _calls(wrapped)) == ("", [("list_dir_v2", "{}")])
    literals = parse("pythonic", "[f(t=(1, 2), d={'k': None}), g(s={1, 2}), h(d={1: 2})]")
    assert _calls(literals) == [("f", '{"d": {"k": null}, "t": [1, 2]}')]  # a set or a non-string key is not JSON
    assert [u.raw_text for u in literals.unparsed] == ["g(s={1, 2})", "h(d={1: 2})"]


def test_pythonic_reads_a_list_only_at_the_start_of_the_response():
    prose = parse("pythonic", "The result is [round(number=2.5)].")
    assert (prose.content, prose.tool_calls) == ("The result is [round(number=2.5)].", [])
    assert parse("pythonic", "Let me look.\n[list_dir_v2()]").tool_calls == []
    after = parse("pythonic", "<think>r</think>\n  [list_dir_v2()]\nDone.")
    assert (_calls(after), after.content) == ([("list_dir_v2", "{}")], "Done.")


# --------------------------------------------------------------------------- argument typing


@pytest.mark.parametrize(
    ("text", "schema", "value"),
    [
        ("3", {"type": "integer"}, 3),
        ("3", {"type": "number"}, 3),
        ("3.5", {"type": "number"}, 3.5),
        ("1e3", {"type": "number"}, 1000.0),
        ("3", {"type": "string"}, "3"),
        ("True", {"type": "boolean"}, True),
        ("false", {"type": "boolean"}, False),
        ("None", {"type": ["object", "null"]}, None),
        ("null", {"type": "string"}, "null"),
        ('{"a": 1}', {"type": "object"}, {"a": 1}),
        ("{'a': [1, None]}", {"type": "object"}, {"a": [1, None]}),
        ('["x"]', {"type": "array"}, ["x"]),
        ('["x"]', {"type": "object"}, '["x"]'),
        ("7", {"anyOf": [{"type": "integer"}, {"type": "string"}]}, 7),
        ("seven", {"anyOf": [{"type": "integer"}, {"type": "string"}]}, "seven"),
        ("2", {"enum": [1, 2]}, 2),
        ("x", {"$ref": "#/defs/x"}, "x"),
        ("  padded  ", {"type": "string"}, "  padded  "),
        ("3.0", {"type": "integer"}, "3.0"),
    ],
)
def test_text_values_are_typed_by_the_schema_or_stay_text(text, schema, value):
    assert coerce_value(text, schema) == value and type(coerce_value(text, schema)) is type(value)


@pytest.mark.parametrize(
    ("text", "schema"),
    [
        ("9" * 5000, {"type": "integer"}),
        ("9" * 5000, {"type": "number"}),
        ("1e999", {"type": "number"}),
        ("[1e999]", {"type": "array"}),
        ("[NaN, Infinity]", {"type": "array"}),
        ("{'a': {1, 2}}", {"type": "object"}),
        ("[b'x', 1j]", {"type": "array"}),
        ("{(1, 2): 3}", {"type": "object"}),
        ("[" * 20000 + "]" * 20000, {"type": "array"}),
        ("[" * (MAX_JSON_DEPTH + 1) + "]" * (MAX_JSON_DEPTH + 1), {"type": "array"}),
    ],
    ids=[
        "an integer with more digits than int() converts",
        "a number with more digits than int() converts",
        "a number out of a float's range",
        "an array with a number out of range",
        "an array with NaN",
        "a set",
        "bytes and a complex number",
        "a tuple key",
        "nested 20000 deep",
        "nested one level deeper than the limit",
    ],
)
def test_values_json_cannot_hold_stay_text(text, schema):
    assert coerce_value(text, schema) == text


def test_tool_schemas_read_openai_and_flat_definitions():
    tools = ToolSchemas([*TOOLS[:1], {"name": "flat", "parameters": {"properties": {"q": {"type": "integer"}}}}])
    assert tools.names == ["get_weather", "flat"]
    assert tools.declares("get_weather", "days") and tools.declares("get_weather", "nope") is False
    assert tools.declares("unknown", "x") is None
    assert tools.coerce("flat", {"q": "4", "other": "5"}) == {"q": 4, "other": "5"}
    assert schema_types({"properties": {}}) == {"object"} and schema_types({}) == set()


# --------------------------------------------------------------------------- the registry


def test_names_follow_sglang_and_vllm_spellings_are_aliases():
    assert parser_names() == [
        "qwen3_coder",
        "hermes",
        "glm45",
        "deepseekv31",
        "deepseekv3",
        "kimi_k2",
        "mistral",
        "gpt-oss",
        "llama3",
        "pythonic",
    ]
    vllm = {
        "llama3_json": "llama3",
        "llama4_json": "llama3",
        "deepseek_v3": "deepseekv3",
        "deepseek_v31": "deepseekv31",
        "openai": "gpt-oss",
        "llama4_pythonic": "pythonic",
        "qwen3_xml": "qwen3_coder",
        "glm47": "glm45",
    }
    assert {alias: parser_aliases()[alias] for alias in vllm} == vllm
    for alias, name in parser_aliases().items():
        assert canonical_name(alias) == name and type(get_parser(alias)).name == name
    assert isinstance(get_parser("none"), NoToolCalls)
    with pytest.raises(ValueError, match="Unknown tool_call_parser 'qwen3'.*qwen3_coder"):
        get_parser("qwen3")


class EchoParser(ToolCallParser):
    """A user parser in the format of a made-up model: ``CALL NAME``."""

    name = "echo"

    def parse_calls(self, text, tools):
        result = ParseResult(content=text.split("CALL ")[0].strip())
        if "CALL " in text:
            from platoon.train.tinker.renderers.tool_parsers import ParsedToolCall

            result.tool_calls.append(ParsedToolCall(name=text.split("CALL ")[1].strip(), arguments={}))
        return result


def echo_function(text, *, tools=None, in_reasoning=False):
    return ParseResult(content=text.upper())


ECHO_INSTANCE = EchoParser()


@pytest.mark.parametrize("attr", ["EchoParser", "ECHO_INSTANCE", "echo_function"])
def test_a_user_parser_is_named_by_module_and_attribute(attr):
    parser = get_parser(f"{__name__}:{attr}")
    result = parser.parse("<think>r</think>ok CALL ping" if attr != "echo_function" else "ok", tools=None)
    if attr == "echo_function":
        assert result.content == "OK"
    else:
        assert (result.reasoning, result.content, result.tool_calls[0].name) == ("r", "ok", "ping")


def test_a_user_parser_that_does_not_resolve_is_an_error():
    with pytest.raises(ValueError, match="does not resolve"):
        get_parser(f"{__name__}:NoSuchParser")
    with pytest.raises(ValueError, match="does not resolve"):
        get_parser("no_such_module_anywhere:Parser")
    with pytest.raises(ValueError, match="reasoning_tags applies"):
        get_parser(f"{__name__}:echo_function", reasoning_tags=["<a>", "</a>"])
