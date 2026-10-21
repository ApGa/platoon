"""``openhands``: one OpenHands SDK harness whose settings are options (RFC 0016).  Its defaults are its training
settings; Harbor's adapter's settings are an evaluation's options, rendered from the same spec, runner and toolchain."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
from test_harness_spec import CONTEXT

from platoon import cli
from platoon.harness.registry import available_harnesses, get_harness_spec
from platoon.harness.spec import HarnessSpecError

TOOLCHAIN = "/tc/openhands-1.54.0"
ADAPTER = {"reasoning_replay": False, "request_timeout_s": 300, "num_retries": 5}  # Harbor's openhands-sdk adapter


def render(options=None, max_steps=None):
    context = dataclasses.replace(CONTEXT, toolchain=TOOLCHAIN, max_steps=max_steps, options=options)
    return get_harness_spec("openhands").render(context)


def flags(argv) -> dict[str, str]:
    """The runner's ``--flag value`` pairs."""

    runner = list(argv[3:])
    return {runner[index]: runner[index + 1] for index in range(0, len(runner), 2)}


def test_one_openhands_harness_replaces_the_three_specs():
    names = {ref.partition("@")[0] for ref in available_harnesses()}

    assert "openhands" in names and not names & {"platoon-openhands", "platoon-openhands-eval"}
    assert [ref for ref in available_harnesses() if ref.startswith("openhands@")] == ["openhands@1.54.0.post2"]
    for gone in ("platoon-openhands", "platoon-openhands-eval", "openhands@1.16.0"):
        with pytest.raises(ValueError, match="Unknown harness"):
            get_harness_spec(gone)


def test_openhands_declares_its_options():
    options = get_harness_spec("openhands").options

    assert [(name, option.type, option.default) for name, option in options.items()] == [
        ("reasoning_replay", "bool", True),
        ("request_timeout_s", "int", 3600),
        ("num_retries", "int", 0),
        ("condenser", "str", "none"),
        ("condenser_max_tokens", "int", 0),  # no token trigger: the launch of main
        ("allow_token_count_estimate", "bool", False),
    ]
    assert (options["request_timeout_s"].min, options["request_timeout_s"].max) == (1, None)
    assert (options["num_retries"].min, options["num_retries"].max) == (0, 10)
    assert options["condenser"].allowed == ("none", "llm_summarizing")
    assert (options["condenser_max_tokens"].min, options["condenser_max_tokens"].max) == (0, None)
    assert all(option.description for option in options.values())


def test_a_training_rollout_runs_the_defaults():
    rendered = render()

    assert rendered.argv[:3] == (f"{TOOLCHAIN}/bin/python3", "-I", "/tmp/agent/run_openhands.py")
    assert flags(rendered.argv) == {
        "--instruction-file": "/tmp/agent/instruction.md",
        "--max-iterations": "30",  # the spec's step budget when the run sets none
        "--final-message-file": "/tmp/agent/final_message.txt",
        "--reasoning-replay": "true",
        "--llm-timeout": "3600",
        "--num-retries": "0",
        "--condenser": "none",
        "--condenser-max-tokens": "0",
        "--allow-token-count-estimate": "false",
    }


def test_an_evaluation_with_the_adapters_options_runs_from_the_same_toolchain_and_runner():
    training, evaluation = render(), render(ADAPTER, max_steps=500)

    assert flags(evaluation.argv) == {
        **flags(training.argv),
        "--max-iterations": "500",  # the SDK's own step limit, which the adapter keeps: the run's max_steps
        "--reasoning-replay": "false",
        "--llm-timeout": "300",
        "--num-retries": "5",
    }
    assert evaluation.argv[0] == training.argv[0] == f"{TOOLCHAIN}/bin/python3"  # one toolchain
    assert evaluation.assets == training.assets  # one runner
    assert evaluation.env == training.env and evaluation.files == training.files


@pytest.mark.parametrize(
    ("options", "flag", "value"),
    [
        ({"allow_token_count_estimate": True}, "--allow-token-count-estimate", "true"),
        ({"reasoning_replay": False}, "--reasoning-replay", "false"),
        ({"reasoning_replay": True}, "--reasoning-replay", "true"),
        ({"num_retries": 3}, "--num-retries", "3"),
        ({"request_timeout_s": 120}, "--llm-timeout", "120"),
        ({"condenser": "llm_summarizing"}, "--condenser", "llm_summarizing"),
        ({"condenser_max_tokens": 104857}, "--condenser-max-tokens", "104857"),
    ],
)
def test_each_option_reaches_the_runner(options, flag, value):
    assert flags(render(options).argv)[flag] == value


@pytest.mark.parametrize(
    ("options", "problem"),
    [
        ({"reasoning_replay": "no"}, "harness_options.reasoning_replay: 'no' is not a bool"),
        ({"num_retries": 11}, "harness_options.num_retries: 11 is above the maximum 10"),
        ({"request_timeout_s": 0}, "harness_options.request_timeout_s: 0 is below the minimum 1"),
        ({"condenser": "llm"}, "harness_options.condenser: 'llm' is not one of 'none', 'llm_summarizing'"),
        ({"condenser_max_tokens": -1}, "harness_options.condenser_max_tokens: -1 is below the minimum 0"),
        ({"condenser_max_tokens": 0.8}, "harness_options.condenser_max_tokens: 0.8 is not an int"),
        ({"condenser_max_tokens": True}, "harness_options.condenser_max_tokens: True is not an int"),
        ({"num_retires": 1}, "harness_options.num_retires: unknown key; did you mean harness_options.num_retries?"),
    ],
)
def test_a_launch_refuses_options_the_spec_does_not_take(options, problem):
    with pytest.raises(HarnessSpecError, match=re.escape(problem)):
        render(options)


def test_options_never_change_what_is_installed():
    spec = get_harness_spec("openhands")
    install = spec.install

    assert install.recipe is not None and re.fullmatch(r"3\.\d+\.\d+", install.recipe.python)  # the patch is pinned
    assert install.commands == () and install.toolchain == "openhands-1.54.0"
    for options in ({}, ADAPTER, {"condenser": "llm_summarizing"}, {"condenser_max_tokens": 104857}):
        assert render(options).argv[0] == f"{TOOLCHAIN}/bin/python3"


def test_harness_show_lists_the_options(capsys):
    cli.main(["harness", "show", "openhands"])

    out = capsys.readouterr().out
    assert out.startswith("openhands@1.54.0.post2  (spec)\n")
    assert "toolchain recipe: Python 3.12.14, lock openhands-1.54.0.txt, caches ['tiktoken']" in out
    assert "options (set in an environments entry's harness_options):" in out
    rows = {line.split()[0]: line.split(maxsplit=4)[1:] for line in out.splitlines() if line.startswith("  ")}
    assert rows["option"] == ["type", "default", "values", "description"]
    assert rows["reasoning_replay"][:3] == ["bool", "true", "-"]
    assert rows["request_timeout_s"][:3] == ["int", "3600", ">="]
    assert rows["num_retries"][:2] == ["int", "0"] and "0 to 10" in out
    assert rows["condenser"][:3] == ["str", "none", "one"] and "one of none, llm_summarizing" in out
    assert rows["condenser_max_tokens"][:3] == ["int", "0", ">="] and "as the gateway counts them" in out
    assert "Seconds one model call may take" in out


@pytest.mark.parametrize(
    ("page", "header"),
    [
        ("tutorials/builtin-harnesses.md", "| Option | Type | Default | Meaning |"),
        ("how-to/sandboxes-and-harbor.md", "| Option | Default | What it does |"),
    ],
)
def test_the_docs_list_every_option_with_its_default_as_harness_show_does(page, header):
    options = get_harness_spec("openhands").options
    text = (Path(__file__).resolve().parents[2] / "docs" / page).read_text(encoding="utf-8")
    lines = text[text.index(header) :].splitlines()
    rows = lines[2 : next(index for index, line in enumerate(lines) if not line.startswith("|"))]
    column = header.strip("|").split("|").index(" Default ")
    cells = [[cell.strip() for cell in row.strip("|").split("|")] for row in rows]

    assert [(row[0], row[column]) for row in cells] == [
        (f"`{name}`", f"`{option.render(option.default)}`") for name, option in options.items()
    ]


def test_harness_show_says_a_spec_without_options_takes_none(capsys):
    cli.main(["harness", "show", "platoon-bash"])

    assert "options: - (harness_options must be empty)" in capsys.readouterr().out
