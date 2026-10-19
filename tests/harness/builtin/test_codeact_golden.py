"""The scripted rollouts of ``test_codeact_rollouts.py``, replayed and compared in full with golden fixtures.

The fixtures (``golden/<scenario>.json``, written by ``codeact_golden.py``) were recorded on the CodeAct code that
predates its restructure into the harness's ``prompt_builder``, ``namespace``, ``checks`` and ``executor`` modules:
every decoded prompt, step misc, tool call, reward, outcome, parent link and datum of each rollout.  The
restructure must stay equivalent, not byte-identical, so a difference fails its scenario unless an entry of
``golden_scenarios.ALLOWED_DIFFERENCES`` explains it and says why.  The TextCraft-Synth scenarios and their fixtures
are the TextCraft extension's (``extensions/textcraft/tests/test_textcraft_goldens.py``).

Each scenario runs the policy and settings of its rollout test, plus scenarios for the paths no rollout test
reaches: the loop's edge cases (refusals, unterminated cells, the stuck-in-loop finish), tool calls by keyword and
from tasks and threads a cell starts, a terminal tool whose text is not ``finish``'s message (``submit_env.py``), and
``finish(None)``.  Two pieces of process state
that reached the cell output are pinned, so a replay is the same whatever ran before it: openreward's tool-call log
lines get a deterministic form (``codeact_golden.pin_env_log_lines``; they no longer reach a cell), and IPython's
global shell, whose cells get differently worded tracebacks, is never a scenario's
(``codeact_golden.occupy_global_ipython_shell``).

Record the fixtures, on pre-restructure code only::

    CODEACT_GOLDEN_RECORD=1 pytest tests/harness/builtin/test_codeact_golden.py

A scenario added after the restructure is recorded the same way in an export of that code (``git archive`` of the
last commit before the restructure) with this file, ``golden_scenarios.py``, ``codeact_golden.py`` and
``submit_env.py`` copied in; the scenarios from ``number_search_finish_none`` on were recorded so.
"""

from __future__ import annotations

import re
from typing import Any

import codeact_golden as golden
import golden_scenarios
import pytest
from golden_scenarios import TEXTCRAFT_SCENARIOS, GoldenRun, MakeGateway, Scenario, run_scenario, script
from test_codeact_rollouts import NUMBER_SEARCH, NUMBER_SEARCH_OPTIONS, _bisect

SUBMIT = "python://submit_env:SubmitEnvironment"
# The fixtures hold what Qwen3-4B-Instruct-2507's own tokenizer made of every call, so these replace the toy
# tokenizer and renderer of builtin_fixtures here (golden_scenarios; they skip where its tokenizer is not on disk).
model_name, tokenizer, renderer = golden_scenarios.model_name, golden_scenarios.tokenizer, golden_scenarios.renderer


# --------------------------------------------------------------------------- scenarios


async def number_search_bisection(make_gateway: MakeGateway, tmp_path: Any) -> GoldenRun:
    gateway = make_gateway(lambda prompt: f"<python>guess({_bisect(prompt)})</python>")
    return await run_scenario(gateway, NUMBER_SEARCH, "val:0", max_steps=12, options=NUMBER_SEARCH_OPTIONS)


async def number_search_budget_exhaustion(make_gateway: MakeGateway, tmp_path: Any) -> GoldenRun:
    gateway = make_gateway(lambda prompt: "<python>guess(-1)</python>")
    return await run_scenario(gateway, NUMBER_SEARCH, "val:1", max_steps=3, options=NUMBER_SEARCH_OPTIONS)


async def number_search_cell_exception(make_gateway: MakeGateway, tmp_path: Any) -> GoldenRun:
    from platoon.envs.number_search.env import task_specs

    target = task_specs("val")[2]["misc"]["target"]

    def policy(prompt: str) -> str:
        return "<python>1 / 0</python>" if "[Cell 0 Output]" not in prompt else f"<python>guess({target})</python>"

    return await run_scenario(make_gateway(policy), NUMBER_SEARCH, "val:2", max_steps=5, options=NUMBER_SEARCH_OPTIONS)


async def number_search_cell_timeout(make_gateway: MakeGateway, tmp_path: Any) -> GoldenRun:
    gateway = make_gateway(lambda prompt: "<python>await asyncio.sleep(30)</python>")
    options = {**NUMBER_SEARCH_OPTIONS, "cell_timeout_s": 0.3}
    return await run_scenario(gateway, NUMBER_SEARCH, "val:3", max_steps=5, options=options)


async def number_search_loop_edge_cases(make_gateway: MakeGateway, tmp_path: Any) -> GoldenRun:
    # Not a rollout test: the loop paths the restructure moves that no rollout test reaches, all of them meant to
    # stay byte-identical.  A refused syntax error and empty cell, a reply without a <python> tag and one cut at
    # the stop sequence (as a sampler that drops "</python>" returns it), then one cell four times, so the original
    # stuck-in-loop check ends the agent with a synthetic finish() cell: no model call, step or annotation, and
    # no terminal tool to call.
    replies = [
        "<python>guess(</python>",
        "<python>\n</python>",
        "I will guess next time.",
        "<python>print(sum(range(5)))",
        *["<python>guess(-1)</python>"] * 4,
    ]

    return await run_scenario(
        make_gateway(script(replies)), NUMBER_SEARCH, "val:5", max_steps=12, options=NUMBER_SEARCH_OPTIONS
    )


async def number_search_finish_none(make_gateway: MakeGateway, tmp_path: Any) -> GoldenRun:
    # finish(None) without a terminal tool, then finish with a message.
    replies = ["<python>finish(None)</python>", "<python>finish('given up')</python>"]
    return await run_scenario(
        make_gateway(script(replies)), NUMBER_SEARCH, "val:7", max_steps=5, options=NUMBER_SEARCH_OPTIONS
    )


async def number_search_calls_off_the_cell_loop(make_gateway: MakeGateway, tmp_path: Any) -> GoldenRun:
    # A synchronous tool called from tasks the cell starts, and from a thread it starts.
    replies = [
        "<python>async def g(n):\n    return guess(n)\nprint(await asyncio.gather(g(1), g(2)))</python>",
        "<python>import threading\nout = []\nthread = threading.Thread(target=lambda: out.append(guess(3)))\n"
        "thread.start()\nthread.join()\nprint(out)</python>",
        "<python>finish('x')</python>",
    ]
    return await run_scenario(
        make_gateway(script(replies)), NUMBER_SEARCH, "val:8", max_steps=10, options=NUMBER_SEARCH_OPTIONS
    )


async def submit_env_terminal_tool(make_gateway: MakeGateway, tmp_path: Any) -> GoldenRun:
    # A tool with an optional parameter, called without it and by keyword out of order, then finish through a
    # terminal tool whose text ("Submitted.") is not finish's message.
    replies = [
        "<python>print(echo('a'))</python>",
        "<python>print(echo(times=2, text='b'))</python>",
        "<python>finish('41')</python>",
    ]
    return await run_scenario(
        make_gateway(script(replies)), SUBMIT, "val:0", max_steps=10, options={"subagents": False}
    )


async def submit_env_finish_none(make_gateway: MakeGateway, tmp_path: Any) -> GoldenRun:
    # finish(None) hands None to the terminal tool, which ends the episode.
    replies = ["<python>result = finish(None)\nprint(repr(result))</python>", "<python>finish('42')</python>"]
    return await run_scenario(
        make_gateway(script(replies)), SUBMIT, "val:1", max_steps=10, options={"subagents": False}
    )


async def number_search_react(make_gateway: MakeGateway, tmp_path: Any) -> GoldenRun:
    def policy(prompt: str) -> str:
        low, high = map(int, re.search(r"between (\d+) and (\d+)", prompt).groups())
        answers = re.findall(r'"number": (\d+)\}\}\n</tool_call>.*?<tool_response>\n(Too \w+)', prompt, re.S)
        for number, verdict in answers:
            low, high = (int(number) + 1, high) if verdict == "Too low" else (low, int(number) - 1)
        return f'<tool_call>\n{{"name": "guess", "arguments": {{"number": {(low + high) // 2}}}}}\n</tool_call>'

    options = {"system_prompt": "Find the number with the guess tool."}
    return await run_scenario(
        make_gateway(policy), NUMBER_SEARCH, "val:4", harness="platoon-react", max_steps=12, options=options
    )


SCENARIOS: dict[str, Scenario] = {
    scenario.__name__: scenario
    for scenario in (
        number_search_bisection,
        number_search_budget_exhaustion,
        number_search_cell_exception,
        number_search_cell_timeout,
        number_search_loop_edge_cases,
        number_search_react,
        number_search_finish_none,
        number_search_calls_off_the_cell_loop,
        submit_env_terminal_tool,
        submit_env_finish_none,
    )
}


# --------------------------------------------------------------------------- the comparison


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", list(SCENARIOS))
async def test_rollout_matches_its_golden_fixture(scenario, make_gateway, monkeypatch, tmp_path):
    await golden_scenarios.replay(scenario, SCENARIOS[scenario], make_gateway, monkeypatch, tmp_path)


def test_every_golden_fixture_has_a_scenario():
    assert sorted(path.stem for path in golden.GOLDEN_DIR.glob("*.json")) == sorted(SCENARIOS)


def test_allowed_differences_explain_only_what_they_name():
    recorded = {"trajectories": {"root": {"steps": [{"output": "Too low", "error": ""}], "reward": 1.0}}}
    replayed = {"trajectories": {"root": {"steps": [{"error": "", "output": "too low"}], "reward": 1}}}
    differences = golden.compare(recorded, replayed)
    assert [difference.path for difference in differences] == [
        "/trajectories/root/steps/0/output",
        "/trajectories/root/steps/0@keys",  # the same keys in another order
        "/trajectories/root/reward",  # 1 is not 1.0
    ]
    assert (differences[1].expected, differences[1].actual) == (["output", "error"], ["error", "output"])
    lowercase = golden.AllowedDifference(
        reason="lowercase verdicts",
        path="/trajectories/*/steps/*/output",
        accept=lambda expected, actual: expected.lower() == actual,
    )
    assert golden.unexplained("number_search_bisection", differences, [lowercase]) == differences[1:]
    elsewhere = golden.AllowedDifference(reason="another fixture", path="*", scenarios=("number_search_react",))
    assert golden.unexplained("number_search_bisection", differences, [elsewhere]) == differences
    with pytest.raises(ValueError, match="needs a reason"):
        golden.AllowedDifference(reason=" ", path="*")
    known = {*SCENARIOS, *TEXTCRAFT_SCENARIOS}  # the allow-lists name the TextCraft extension's scenarios too
    for entry in golden_scenarios.ALLOWED_DIFFERENCES:
        assert entry.reason.strip() and (not entry.scenarios or set(entry.scenarios) <= known)
