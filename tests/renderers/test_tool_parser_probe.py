"""``tool_call_parser: auto`` and its round trip, on small chat templates that write each format's tool calls.

The templates here are written for these tests, one per format, after the shape of the vendors' templates (the
cached models' own templates are ``test_tool_parsing_templates.py``'s).  They render through a Jinja environment
set up as transformers sets up chat templates.
"""

from __future__ import annotations

import json
import logging
import re

import jinja2
import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from platoon.train.tinker.renderers.tool_parsers import NoToolCalls, ParsedToolCall, ParseResult, ToolCallParser
from platoon.train.tinker.renderers.tool_parsers.probe import (
    PROBE_CALLS,
    ToolCallParserError,
    choose_parser,
    probe_template,
)

HEAD = """
{%- for m in messages -%}
{%- if m.role == 'system' -%}
<|system|>{{ m.content }}{% if tools %}<tools>{{ tools | tojson }}</tools>{% endif %}<|end|>
{%- elif m.role == 'user' -%}
<|user|>{{ m.content }}<|end|>
{%- elif m.role == 'assistant' -%}
<|assistant|>"""
TAIL = """<|end|>
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt %}<|assistant|>{% endif -%}"""


def _template(body: str) -> str:
    return HEAD + body + TAIL


VALUE = "{{ v if v is string else v | tojson }}"
TEMPLATES = {
    "qwen3_coder": _template(
        "<think>\n{{ m.reasoning_content }}\n</think>\n\n{{ m.content }}"
        "{% for tc in m.tool_calls %}{% if loop.first %}\n\n{% else %}\n{% endif %}"
        "<tool_call>\n<function={{ tc.function.name }}>\n"
        "{% for k, v in tc.function.arguments.items() %}<parameter={{ k }}>\n" + VALUE + "\n</parameter>\n{% endfor %}"
        "</function>\n</tool_call>{% endfor %}"
    ),
    "hermes": _template(
        "{{ m.content }}{% for tc in m.tool_calls %}\n<tool_call>\n"
        '{"name": "{{ tc.function.name }}", "arguments": {{ tc.function.arguments | tojson }}}\n'
        "</tool_call>{% endfor %}"
    ),
    "glm45": _template(
        "<think>{{ m.reasoning_content }}</think>\n{{ m.content }}{% for tc in m.tool_calls %}\n"
        "<tool_call>{{ tc.function.name }}\n{% for k, v in tc.function.arguments.items() %}"
        "<arg_key>{{ k }}</arg_key>\n<arg_value>" + VALUE + "</arg_value>\n{% endfor %}</tool_call>{% endfor %}"
    ),
    "deepseekv31": _template(
        "</think>{{ m.content }}<｜tool▁calls▁begin｜>{% for tc in m.tool_calls %}<｜tool▁call▁begin｜>"
        "{{ tc.function.name }}<｜tool▁sep｜>{{ tc.function.arguments | tojson }}<｜tool▁call▁end｜>{% endfor %}"
        "<｜tool▁calls▁end｜>"
    ),
    "deepseekv3": _template(
        "{{ m.content }}<｜tool▁calls▁begin｜>{% for tc in m.tool_calls %}<｜tool▁call▁begin｜>function<｜tool▁sep｜>"
        "{{ tc.function.name }}\n```json\n{{ tc.function.arguments | tojson }}\n```<｜tool▁call▁end｜>"
        "{% if not loop.last %}\n{% endif %}{% endfor %}<｜tool▁calls▁end｜>"
    ),
    "kimi_k2": _template(
        "<think>{{ m.reasoning_content }}</think>{{ m.content }}<|tool_calls_section_begin|>"
        "{% for tc in m.tool_calls %}<|tool_call_begin|>{{ tc.id }}<|tool_call_argument_begin|>"
        "{{ tc.function.arguments | tojson }}<|tool_call_end|>{% endfor %}<|tool_calls_section_end|>"
    ),
    "mistral": _template(
        "{{ m.content }}{% for tc in m.tool_calls %}"
        "{% if tc.id | length != 9 %}"
        "{{ raise_exception('Tool call IDs should be alphanumeric strings with length 9!') }}"
        "{% endif %}[TOOL_CALLS]{{ tc.function.name }}[ARGS]{{ tc.function.arguments | tojson }}{% endfor %}"
    ),
    "llama3": _template(
        "{% if m.tool_calls | length != 1 %}"
        "{{ raise_exception('This model only supports single tool-calls at once!') }}"
        '{% endif %}{"name": "{{ m.tool_calls[0].function.name }}", '
        '"parameters": {{ m.tool_calls[0].function.arguments | tojson }}}'
    ),
    "pythonic": _template(
        "[{% for tc in m.tool_calls %}{{ tc.function.name }}("
        "{% for k, v in tc.function.arguments.items() %}{{ k }}={{ v | tojson }}{% if not loop.last %}, {% endif %}"
        "{% endfor %}){% if not loop.last %}, {% endif %}{% endfor %}]"
    ),
}
# Harmony: each call is its own message that ends in <|call|>, a stop token, after the analysis message.
HARMONY = """
{%- for m in messages -%}
{%- if m.role == 'system' -%}
<|start|>system<|message|>{{ m.content }}{% if tools %}{{ tools | tojson }}{% endif %}<|end|>
{%- elif m.role == 'user' -%}
<|start|>user<|message|>{{ m.content }}<|end|>
{%- elif m.role == 'assistant' -%}
<|start|>assistant<|channel|>analysis<|message|>{{ m.thinking }}<|end|>
{%- for tc in m.tool_calls -%}
<|start|>assistant to=functions.{{ tc.function.name }}<|channel|>commentary json<|message|>
{{- tc.function.arguments | tojson }}<|call|>
{%- endfor -%}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt %}<|start|>assistant{% endif -%}"""
STOPS = ["<|end|>"]
HARMONY_STOPS = ["<|return|>", "<|call|>"]


def render_with(source: str):
    """A ``(messages, tools, add_generation_prompt) -> text`` render function, as transformers renders templates."""

    def raise_exception(message: str):
        raise jinja2.exceptions.TemplateError(message)

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.filters["tojson"] = lambda value, **_: json.dumps(value, ensure_ascii=False)
    environment.globals["raise_exception"] = raise_exception
    template = environment.from_string(source)

    def render(messages, tools, add_generation_prompt):
        return template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt)

    return render


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_auto_keeps_the_parser_that_reads_the_template_back(name):
    parser, choice = choose_parser(render_with(TEMPLATES[name]), STOPS, label="test model")
    assert (choice.setting, choice.parser, choice.verified) == ("auto", name, True)
    assert type(parser).name == name
    one_turn_per_call = name == "llama3"  # its template raises on more than one call per turn
    assert choice.reason == f"round trip ({'one turn per call' if one_turn_per_call else 'one turn with every call'})"


def test_a_harmony_template_is_probed_one_call_per_turn():
    parser, choice = choose_parser(render_with(HARMONY), HARMONY_STOPS)
    assert (choice.parser, choice.reason) == ("gpt-oss", "round trip (one turn per call)")
    probe = probe_template(render_with(HARMONY), HARMONY_STOPS)
    assert [turn.calls for turn in probe.turns] == [[call] for call in PROBE_CALLS]
    assert all(turn.sampled.endswith("}") and turn.reasoning for turn in probe.turns)  # cut at <|call|>


def test_only_one_parser_reads_each_template_back():
    from platoon.train.tinker.renderers.tool_parsers import get_parser, parser_names
    from platoon.train.tinker.renderers.tool_parsers.probe import check

    for name, source in TEMPLATES.items():
        probe = probe_template(render_with(source), STOPS)
        assert [other for other in parser_names() if check(get_parser(other), probe) is None] == [name]


def test_a_named_parser_is_checked_and_a_mismatch_only_warns(caplog):
    _, choice = choose_parser(render_with(TEMPLATES["hermes"]), STOPS, setting="qwen25")
    assert (choice.setting, choice.parser, choice.verified) == ("qwen25", "hermes", True)
    with caplog.at_level(logging.WARNING):
        parser, choice = choose_parser(render_with(TEMPLATES["hermes"]), STOPS, setting="qwen3_coder")
    assert (choice.parser, choice.verified) == ("qwen3_coder", False)
    assert "does not read the chat template's own tool calls back" in caplog.text
    assert type(parser).name == "qwen3_coder"  # still the one the config names


def test_auto_refuses_a_template_no_parser_reads_and_says_what_to_do():
    custom = _template(
        "{{ m.content }}{% for tc in m.tool_calls %}<call name='{{ tc.function.name }}'>"
        "{{ tc.function.arguments | tojson }}</call>{% endfor %}"
    )
    with pytest.raises(ToolCallParserError) as error:
        choose_parser(render_with(custom), STOPS, label="my-org/my-model")
    message = str(error.value)
    assert "No tool-call parser reads the chat template's own tool calls back for my-org/my-model" in message
    assert "<call name='get_weather'>" in message  # what the template rendered
    assert "    qwen3_coder: read no calls; read the text" in message and "    pythonic:" in message  # what each read
    assert "Set renderer_kwargs.tool_call_parser" in message and "package.module:attr" in message


def test_a_template_without_tools_parses_no_tool_calls(caplog):
    plain = _template("{{ m.content }}").replace("{% if tools %}<tools>{{ tools | tojson }}</tools>{% endif %}", "")
    with caplog.at_level(logging.WARNING):
        parser, choice = choose_parser(render_with(plain), STOPS)
    assert isinstance(parser, NoToolCalls) and (choice.parser, choice.verified) == ("none", None)
    assert "renders no tools" in caplog.text


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        ("{{ m.content }}", "leaves the assistant's tool calls out"),
        ("{{ raise_exception('no tool calls here') }}", "raises on an assistant turn with 3 tool call(s)"),
    ],
)
def test_a_template_that_renders_tools_but_not_calls_cannot_be_checked(body, problem):
    with pytest.raises(ToolCallParserError, match=re.escape(problem)):
        choose_parser(render_with(_template(body)), STOPS)
    _, choice = choose_parser(render_with(_template(body)), STOPS, setting="hermes")
    assert choice.verified is None and problem in choice.reason


class AngleCallParser(ToolCallParser):
    """A user parser for the ``<call name='NAME'>{...}</call>`` format above."""

    name = "angle"

    def parse_calls(self, text, tools):
        content, *calls = text.split("<call name='")
        result = ParseResult(content=content.strip())
        for call in calls:
            name, rest = call.split("'>", 1)
            result.tool_calls.append(ParsedToolCall(name=name, arguments=json.loads(rest.split("</call>")[0])))
        return result


def test_a_user_parser_is_checked_like_a_built_in_one():
    custom = _template(
        "{{ m.content }}{% for tc in m.tool_calls %}<call name='{{ tc.function.name }}'>"
        "{{ tc.function.arguments | tojson }}</call>{% endfor %}"
    )
    parser, choice = choose_parser(render_with(custom), STOPS, setting=f"{__name__}:AngleCallParser")
    assert isinstance(parser, AngleCallParser)
    assert (choice.parser, choice.verified) == (f"{__name__}:AngleCallParser", True)
    assert choice.record() == {
        "setting": f"{__name__}:AngleCallParser",
        "parser": f"{__name__}:AngleCallParser",
        "verified": True,
        "reason": "set by name; round trip (one turn with every call)",
    }


def test_none_and_unknown_names():
    parser, choice = choose_parser(render_with(TEMPLATES["hermes"]), STOPS, setting="none")
    assert isinstance(parser, NoToolCalls) and choice.record()["parser"] == "none"
    with pytest.raises(ToolCallParserError, match="Unknown tool_call_parser 'qwen3'"):
        choose_parser(render_with(TEMPLATES["hermes"]), STOPS, setting="qwen3")
