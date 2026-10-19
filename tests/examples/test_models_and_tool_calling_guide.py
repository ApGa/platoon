"""The models and tool calling guide stays true: its support matrix lists every registered parser with its aliases,
and its parser of your own works as the guide says."""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

from platoon.train.tinker.renderers.tool_parsers import parser_aliases, parser_names
from platoon.train.tinker.renderers.tool_parsers.probe import choose_parser

GUIDE = Path(__file__).resolve().parents[2] / "docs" / "how-to" / "models-and-tool-calling.md"
TEXT = GUIDE.read_text()
_ROW = re.compile(r"^\| `(?P<name>[^`]+)` \|(?P<aliases>[^|]*)\|", re.M)


def test_the_support_matrix_lists_every_parser_and_alias():
    section = TEXT[TEXT.index("## Supported formats") : TEXT.index("## Set the parser")]
    rows = {
        row["name"]: re.findall(r"`([^`]+)`", row["aliases"])
        for row in _ROW.finditer(section)
        if row["name"] != "tool_call_parser"  # the header
    }
    assert list(rows) == parser_names()
    aliases = {alias: name for name, names in rows.items() for alias in names}
    assert aliases == parser_aliases()


def _block(start: str) -> str:
    return re.search(rf"```python\n({re.escape(start)}.*?)```", TEXT, re.S)[1]


@pytest.fixture
def my_parsers(tmp_path, monkeypatch):
    (tmp_path / "my_parsers.py").write_text(_block("# my_parsers.py"))
    monkeypatch.syspath_prepend(str(tmp_path))
    yield importlib.import_module("my_parsers")
    sys.modules.pop("my_parsers", None)


def _render(messages, tools, add_generation_prompt):
    """A chat template that writes calls as <call name='NAME'>{...}</call>."""
    out = ""
    for message in messages:
        out += f"<|{message['role']}|>{message['content']}"
        for call in message.get("tool_calls") or []:
            out += f"<call name='{call['function']['name']}'>{json.dumps(call['function']['arguments'])}</call>"
        out += "<|end|>"
    return out + ("<|assistant|>" if add_generation_prompt else "")


def test_the_guides_parser_reads_its_format_back(my_parsers):
    parser, choice = choose_parser(_render, ["<|end|>"], setting="my_parsers:AngleCallParser")
    assert isinstance(parser, my_parsers.AngleCallParser)
    assert (choice.parser, choice.verified) == ("my_parsers:AngleCallParser", True)
    result = parser.parse("<think>plan</think>Calling.<call name='lookup'>{\"q\": 1}</call>")
    assert (result.reasoning, result.content) == ("plan", "Calling.")
    assert [(call.name, call.arguments) for call in result.tool_calls] == [("lookup", {"q": 1})]
