"""A TOML spec's ``[options]`` (RFC 0016): declared with a type, a default, a description and the values they take,
checked when the spec loads, checked again against what a config gives, and rendered into the launch as
``{options.NAME}`` and ``{options}``."""

from __future__ import annotations

import dataclasses
import json

import pytest
from test_harness_spec import CONTEXT, minimal

from platoon.harness.spec import HarnessSpecError, OptionSpec, parse_spec

OPTIONS = {
    "replay": {"type": "bool", "default": True, "description": "Send reasoning back"},
    "retries": {"type": "int", "default": 0, "min": 0, "max": 10, "description": "Retries of a failed call"},
    "temperature": {"type": "float", "default": 1, "min": 0.0},
    "mode": {"type": "str", "default": "none", "allowed": ["none", "summarize"]},
}
ARGV = [
    "agent",
    "--replay",
    "{options.replay}",
    "--retries",
    "{options.retries}",
    "--temperature",
    "{options.temperature}",
    "--mode={options.mode}",
]


def spec_with(options=None, **launch):
    return parse_spec(minimal(options=OPTIONS if options is None else options, launch={"argv": ARGV, **launch}))


def with_options(options):
    return dataclasses.replace(CONTEXT, options=options)


def test_options_load_with_their_types_defaults_and_rules_in_the_specs_order():
    spec = spec_with()

    assert list(spec.options) == ["replay", "retries", "temperature", "mode"]
    assert spec.options["retries"] == OptionSpec(
        name="retries", type="int", default=0, description="Retries of a failed call", min=0, max=10
    )
    assert spec.options["temperature"].default == 1.0 and isinstance(spec.options["temperature"].default, float)
    assert spec.options["mode"].allowed == ("none", "summarize")
    assert [option.constraint() for option in spec.options.values()] == [
        "-",
        "0 to 10",
        ">= 0.0",
        "one of none, summarize",
    ]


def test_the_defaults_render_where_the_runs_values_render():
    rendered = spec_with().render(CONTEXT)

    assert rendered.argv == (
        "agent",
        "--replay",
        "true",
        "--retries",
        "0",
        "--temperature",
        "1.0",
        "--mode=none",
    )


def test_given_options_render_as_text_a_program_can_parse():
    options = {"replay": False, "retries": 5, "temperature": 1e-05, "mode": "summarize"}

    rendered = spec_with().render(with_options(options))

    assert rendered.argv[1:] == ("--replay", "false", "--retries", "5", "--temperature", "0.00001", "--mode=summarize")


def test_options_render_into_env_files_mcp_and_the_builtins_knob_and_options_is_one_json_object():
    spec = parse_spec(
        minimal(
            options=OPTIONS,
            launch={
                "argv": ["agent"],
                "env": {"AGENT_RETRIES": "{options.retries}"},
                "files": {"options-{options.mode}.json": "{options}"},
            },
            builtins={"tools": ["bash"], "disable": {"args": ["--retries={options.retries}"]}},
            mcp={"args": ["--mcp", "{mcp_url}", "--mode", "{options.mode}"]},
        )
    )

    rendered = spec.render(
        dataclasses.replace(CONTEXT, mcp_url="http://gw/mcp", options={"retries": 3}),
        disable_builtins=True,
    )

    assert rendered.env["AGENT_RETRIES"] == "3"
    assert json.loads(rendered.files["options-none.json"]) == {
        "mode": "none",
        "replay": True,
        "retries": 3,
        "temperature": 1.0,
    }
    assert rendered.argv == ("agent", "--retries=3", "--mcp", "http://gw/mcp", "--mode", "none")


def test_a_spec_without_options_takes_none():
    spec = parse_spec(minimal())

    assert spec.options == {} and spec.render(CONTEXT).argv == ("toy", CONTEXT.instruction)
    assert spec.option_problems({"retries": 1}, "environments.0.harness_options") == [
        "environments.0.harness_options.retries: unknown key; harness toy@1.0 takes no options"
    ]
    with pytest.raises(HarnessSpecError, match="harness toy@1.0: harness_options.retries: unknown key"):
        spec.render(with_options({"retries": 1}))


def test_what_a_config_gives_is_checked_with_full_dotted_paths_and_a_did_you_mean():
    spec = spec_with()
    where = "environments.0.harness_options"

    assert spec.option_problems({"retries": 3, "temperature": 2, "replay": False, "mode": "summarize"}, where) == []
    assert spec.option_problems(
        {
            "retires": 1,
            "verbose": True,
            "replay": 1,
            "retries": True,
            "temperature": "hot",
            "mode": "fast",
        },
        where,
    ) == [
        f"{where}.retires: unknown key; did you mean {where}.retries?",
        f"{where}.verbose: unknown key; harness toy@1.0 takes replay, retries, temperature, mode",
        f"{where}.replay: 1 is not a bool",
        f"{where}.retries: True is not an int",
        f"{where}.temperature: 'hot' is not a float",
        f"{where}.mode: 'fast' is not one of 'none', 'summarize'",
    ]
    assert spec.option_problems({"retries": 11, "temperature": -0.5}, where) == [
        f"{where}.retries: 11 is above the maximum 10",
        f"{where}.temperature: -0.5 is below the minimum 0.0",
    ]
    assert spec.option_problems({"temperature": float("nan")}, where) == [
        f"{where}.temperature: nan is not a finite number"
    ]
    assert spec.option_problems(["retries"], where) == [f"{where}: expected a mapping of options, got ['retries']"]


def test_resolving_fills_in_the_defaults_and_makes_an_int_a_float():
    spec = spec_with()

    assert spec.resolve_options(None) == {"replay": True, "retries": 0, "temperature": 1.0, "mode": "none"}
    resolved = spec.resolve_options({"temperature": 2, "mode": "summarize"})
    assert resolved == {"replay": True, "retries": 0, "temperature": 2.0, "mode": "summarize"}
    assert isinstance(resolved["temperature"], float)
    with pytest.raises(HarnessSpecError, match=r"harness toy@1.0: harness_options.retries: 11 is above the maximum 10"):
        spec.render(with_options({"retries": 11}))  # a launch checks again: what a worker renders was checked


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"Retries": {"type": "int", "default": 0}}, r"\[options.Retries\]: an option's name must be a lower-case"),
        ({"retries": {"type": "integer", "default": 0}}, r"type 'integer' is not one of"),
        ({"retries": {"type": "int"}}, r"\[options.retries\]: missing required key 'default'"),
        ({"retries": {"type": "int", "default": "0"}}, r"\[options.retries\]: default: '0' is not an int"),
        ({"retries": {"type": "int", "default": 11, "max": 10}}, r"default: 11 is above the maximum 10"),
        ({"mode": {"type": "str", "default": "x", "allowed": ["a", "b"]}}, r"default: 'x' is not one of 'a', 'b'"),
        ({"replay": {"type": "bool", "default": True, "allowed": [True]}}, "allowed lists the values of an int or str"),
        ({"mode": {"type": "str", "default": "a", "allowed": []}}, "allowed must be a non-empty list of str values"),
        ({"mode": {"type": "str", "default": "a", "allowed": ["a", 1]}}, "allowed must be a non-empty list of str"),
        ({"mode": {"type": "str", "default": "a", "allowed": ["a", "a"]}}, "allowed lists a value twice"),
        ({"mode": {"type": "str", "default": "a", "min": 1}}, "min and max bound an int or float option, not a str"),
        ({"retries": {"type": "int", "default": 1, "min": 0.5}}, "the bounds of an int option must be integers"),
        ({"retries": {"type": "int", "default": 1, "min": 2, "max": 1}}, "min 2 is above max 1"),
        ({"retries": {"type": "int", "default": 1, "min": True}}, "min must be a number"),
        ({"retries": {"type": "int", "default": 1, "help": "x"}}, r"unknown key\(s\) \['help'\]"),
        ({"retries": {"type": "int", "default": 1, "description": "two\nlines"}}, "description must be one line"),
        ({"retries": 3}, r"\[options.retries\]: expected a table, got int"),
    ],
)
def test_a_malformed_option_fails_when_the_spec_loads(options, error):
    with pytest.raises(HarnessSpecError, match=error):
        parse_spec(minimal(options=options))


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"launch": {"argv": ["agent", "{options.retires}"]}}, r"unknown placeholder\(s\) \['options.retires'\]"),
        ({"launch": {"argv": ["agent", "{option.retries}"]}}, r"unknown placeholder\(s\) \['option.retries'\]"),
        (
            {"launch": {"argv": ["agent"]}, "final_message": {"file": "{options.mode}.txt"}},
            r"\[final_message\] file: unknown placeholder\(s\) \['options.mode'\]",
        ),
        (
            {"launch": {"argv": ["agent"], "assets": {"{options.mode}.py": "test_harness_spec.py"}}},
            r"assets key: unknown placeholder\(s\) \['options.mode'\]",
        ),
    ],
)
def test_a_placeholder_must_name_a_declared_option_where_options_render(overrides, error, tmp_path):
    with pytest.raises(HarnessSpecError, match=error):
        parse_spec(minimal(options=OPTIONS, **overrides))


def test_other_braces_stay_literal():
    spec = parse_spec(minimal(options=OPTIONS, launch={"argv": ["sh", "-c", 'echo ${HOME} {"a.b": 1}', "{options}"]}))

    assert spec.render(CONTEXT).argv[2] == 'echo ${HOME} {"a.b": 1}'
