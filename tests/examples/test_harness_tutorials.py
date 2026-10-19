"""The harness tutorials stay true to the code: the files and configs they quote are the example files, and the
option tables of the built-in harnesses list exactly the options the harnesses accept."""

from __future__ import annotations

import re
from dataclasses import fields

import pytest
import yaml
from harness_example_fakes import EXAMPLES, REPO

from platoon.harness.builtin.codeact import Checks, CodeActOptions
from platoon.harness.builtin.react import ReactOptions
from platoon.harness.registry import get_harness_spec

TUTORIALS = REPO / "docs" / "tutorials"


def read(tutorial: str) -> str:
    return (TUTORIALS / tutorial).read_text()


def blocks(text: str, language: str) -> list[str]:
    return re.findall(rf"^```{language}\n(.*?)^```$", text, re.S | re.M)


def section(text: str, heading: str) -> str:
    start = text.index(f"\n## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start : end if end != -1 else None]


def table_keys(text: str, header: str) -> list[str]:
    """The first-column names of the table with ``header``."""

    rows = text[text.index(header) :].splitlines()[2:]
    keys = []
    for row in rows:
        if not row.startswith("|"):
            break
        keys.append(re.match(r"\| `(\w+)` \|", row)[1])
    return keys


@pytest.mark.parametrize(
    ("tutorial", "language", "path"),
    [
        ("function-harness.md", "python", EXAMPLES / "function-harness" / "tool_loop.py"),
        ("cli-harness.md", "toml", EXAMPLES / "cli-harness" / "tiny-agent.toml"),
        ("cli-harness.md", "toml", REPO / "platoon" / "harness" / "builtin" / "specs" / "platoon-bash.toml"),
    ],
)
def test_a_tutorial_quotes_the_file_verbatim(tutorial, language, path):
    assert path.read_text() in blocks(read(tutorial), language)


@pytest.mark.parametrize(
    ("tutorial", "path"),
    [
        ("function-harness.md", EXAMPLES / "function-harness" / "config.yaml"),
        ("cli-harness.md", EXAMPLES / "cli-harness" / "config.yaml"),
    ],
)
def test_a_tutorial_quotes_the_example_config(tutorial, path):
    quoted = [yaml.safe_load(block) for block in blocks(read(tutorial), "yaml")]
    assert yaml.safe_load(path.read_text()) in quoted


def test_the_option_tables_list_the_options_the_built_in_harnesses_take():
    text = read("builtin-harnesses.md")
    header = "| Option | Default | Meaning |"

    assert table_keys(section(text, "`platoon-react`"), header) == [item.name for item in fields(ReactOptions)]
    codeact = section(text, "`platoon-codeact`")
    assert table_keys(codeact, header) == [item.name for item in fields(CodeActOptions)]
    assert table_keys(codeact, "| `checks` key |") == [item.name for item in fields(Checks)]


def test_the_openhands_option_table_lists_the_specs_options_with_their_defaults():
    rows = section(read("builtin-harnesses.md"), "`openhands`")
    options = get_harness_spec("openhands").options

    assert table_keys(rows, "| Option | Type | Default | Meaning |") == list(options)
    for name, option in options.items():
        assert f"| `{name}` | " in rows and f" | `{option.render(option.default)}` | " in rows
