"""Scripted CodeAct and ReAct rollouts on number search, through the real gateway, env service and inline runtime.

A scripted policy answers each model call from its decoded prompt; the gateway records exact tokens; the
number-search environment runs in process through a ``python://`` service; the agent tree runs on
``platoon.harness.inline`` (``inline_runs.py``); datums come from ``trajectory_to_data``.  The TextCraft extension's
tests run recursive agent trees the same way (``extensions/textcraft/tests/test_textcraft_rollouts.py``).
"""

from __future__ import annotations

import asyncio
import re

import pytest
from inline_runs import open_service_task, run_tree, train_datums

from platoon.harness.builtin.codeact.harness import STUCK_IN_LOOP_CODE

NUMBER_SEARCH = "python://platoon.envs.number_search.env:NumberSearchEnvironment"
NUMBER_SEARCH_OPTIONS = {
    "prompt_builder": "platoon.envs.number_search.prompts:NumberSearchPromptBuilder",
    "include_reasoning": False,
    "subagents": False,
}


def _bisect(prompt: str) -> int:
    low, high = map(int, re.search(r"between (\d+) and (\d+)", prompt).groups())
    turns = re.findall(r"guess\((\d+)\)</python><\|im_end\|>\n<\|im_start\|>user\n.*?'(Too \w+)", prompt, re.S)
    for number, verdict in turns:
        if verdict == "Too low":
            low = int(number) + 1
        else:
            high = int(number) - 1
    return (low + high) // 2


@pytest.mark.asyncio
async def test_codeact_solves_number_search_by_bisection(make_gateway):
    gateway, sampler = make_gateway(lambda prompt: f"<python>guess({_bisect(prompt)})</python>")
    task, session = await open_service_task(NUMBER_SEARCH, "val:0")

    run = await run_tree(gateway, "platoon-codeact", session, task, max_steps=12, options=NUMBER_SEARCH_OPTIONS)

    assert run.result.status == "completed" and "correctly" in run.result.final_message
    assert run.outcome.status == "scored" and run.outcome.reward == 1.0
    steps = run.root.steps
    assert len(steps) == len(sampler.calls) > 1
    assert steps[0].misc["code"].startswith("guess(") and "Too" in steps[0].misc["output"]
    assert steps[-1].misc["reward_misc"]["reward/success"] == 1.0
    # The original system prompt and empty action space: the first prompt names guess() and has no action space.
    first_prompt = sampler.calls[0][0]
    assert "Your answer must call guess(number: int)" in first_prompt and "# Action Space" not in first_prompt
    assert "Your Goal: Guess the correct number between" in first_prompt and "budget of 12 steps" in first_prompt
    # Compact replies re-render exactly, so every step extends the previous one into a single datum.
    data = train_datums(run, task.id)
    assert len(data.datums) == 1 and data.task_reward == 1.0


class _SdkLog:
    """openreward's structlog ``PrintLogger`` as structlog binds it when first imported during a cell's capture: it
    prints to whatever ``sys.stdout`` is at the time.  Keeps what it printed."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def msg(self, message: str) -> None:
        self.lines.append(message)
        print(message, flush=True)

    log = debug = info = warn = warning = error = err = critical = fatal = failure = exception = msg


@pytest.mark.asyncio
async def test_the_env_sdks_info_log_lines_stay_out_of_cell_outputs(make_gateway, monkeypatch):
    # A regression test: openreward logs tool_call_completed at INFO on stdout after every tool call, and a tool
    # call a cell made used to run in a copy of the cell's context, where ShellCapture redirects stdout, so the line
    # reached the cell's output and the model's next prompt.  Here the SDK logs at INFO (platoon leaves an explicit
    # OPENREWARD_LOG_LEVEL alone instead of quieting it) through whatever sys.stdout is; the cells must read what
    # they read when it is quiet.
    import importlib
    import logging

    import structlog

    sdk_log = _SdkLog()
    monkeypatch.setenv("OPENREWARD_LOG_LEVEL", "INFO")
    for name in ("openreward.environments.environment", "openreward.environments.session"):
        module = importlib.import_module(name)
        logger = structlog.wrap_logger(
            sdk_log,
            processors=[lambda _logger, method, event: f"[{method}] {event['event']}"],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        )
        monkeypatch.setattr(module, "logger", logger, raising=False)
    gateway, sampler = make_gateway(lambda prompt: f"<python>guess({_bisect(prompt)})</python>")
    task, session = await open_service_task(NUMBER_SEARCH, "val:0")

    run = await run_tree(gateway, "platoon-codeact", session, task, max_steps=12, options=NUMBER_SEARCH_OPTIONS)

    assert run.result.status == "completed"
    assert "[info] tool_call_completed" in sdk_log.lines  # the SDK did log the calls at INFO, on stdout
    *misses, hit = [step.misc["output"] for step in run.root.steps]
    assert misses and set(misses) <= {" 'Too low, try again.'\n", " 'Too high, try again.'\n"}
    assert re.fullmatch(r" 'You guessed the number \d+ correctly!'\n", hit)
    assert not any("tool_call_completed" in prompt for prompt, *_ in sampler.calls)


@pytest.mark.asyncio
async def test_codeact_budget_exhaustion_ends_the_agent_with_limit(make_gateway):
    gateway, sampler = make_gateway(lambda prompt: "<python>guess(-1)</python>")
    task, session = await open_service_task(NUMBER_SEARCH, "val:1")

    run = await run_tree(gateway, "platoon-codeact", session, task, max_steps=3, options=NUMBER_SEARCH_OPTIONS)

    assert run.result.status == "limit" and run.result.error.startswith("WARNING: Exhausted budget")
    assert len(sampler.calls) == 3
    assert run.outcome.status == "scored" and run.outcome.reward == 0.0 and run.outcome.reason == "no_submit"


@pytest.mark.asyncio
async def test_codeact_cell_exception_is_shown_to_the_model(make_gateway):
    from platoon.envs.number_search.env import task_specs

    task, session = await open_service_task(NUMBER_SEARCH, "val:2")
    target = task_specs("val")[2]["misc"]["target"]

    def policy(prompt: str) -> str:
        return "<python>1 / 0</python>" if "[Cell 0 Output]" not in prompt else f"<python>guess({target})</python>"

    gateway, sampler = make_gateway(policy)
    run = await run_tree(gateway, "platoon-codeact", session, task, max_steps=5, options=NUMBER_SEARCH_OPTIONS)

    first_step = run.root.steps[0]
    assert "ZeroDivisionError" in first_step.misc["output"] + first_step.misc["error"]
    assert "ZeroDivisionError" in sampler.calls[1][0]
    assert run.result.status == "completed" and run.outcome.reward == 1.0


@pytest.mark.asyncio
async def test_codeact_cell_timeout_ends_the_agent_with_timeout(make_gateway):
    gateway, _ = make_gateway(lambda prompt: "<python>await asyncio.sleep(30)</python>")
    task, session = await open_service_task(NUMBER_SEARCH, "val:3")
    options = {**NUMBER_SEARCH_OPTIONS, "cell_timeout_s": 0.3}

    run = await run_tree(gateway, "platoon-codeact", session, task, max_steps=5, options=options)

    assert run.result.status == "timeout" and "0.3 seconds" in run.result.error


@pytest.mark.asyncio
async def test_a_looping_agent_ends_even_when_its_cells_redefined_finish(make_gateway):
    # The synthetic finish cell then runs the cells' own finish, which ends nothing, and repeats the cells it
    # follows: the agent used to loop on synthetic cells forever (no model call, no step, so no limit either).
    redefine = "<python>def finish(message=''):\n    print('not finishing')</python>"
    replies = [redefine] + [f"<python>{STUCK_IN_LOOP_CODE}</python>"] * 4

    def policy(prompt: str) -> str:
        return replies[min(len(re.findall(r"\[Cell \d+ Output\]", prompt)), len(replies) - 1)]

    gateway, sampler = make_gateway(policy)
    task, session = await open_service_task(NUMBER_SEARCH, "val:9")

    run = await asyncio.wait_for(
        run_tree(gateway, "platoon-codeact", session, task, max_steps=10, options=NUMBER_SEARCH_OPTIONS), 30
    )

    assert run.result.status == "completed" and run.result.final_message == "Stuck in a loop, terminating early."
    assert len(sampler.calls) == len(run.root.steps) == 5


@pytest.mark.asyncio
async def test_react_solves_number_search_with_tool_calls(make_gateway):
    def policy(prompt: str) -> str:
        low, high = map(int, re.search(r"between (\d+) and (\d+)", prompt).groups())
        answers = re.findall(r'"number": (\d+)\}\}\n</tool_call>.*?<tool_response>\n(Too \w+)', prompt, re.S)
        for number, verdict in answers:
            low, high = (int(number) + 1, high) if verdict == "Too low" else (low, int(number) - 1)
        return f'<tool_call>\n{{"name": "guess", "arguments": {{"number": {(low + high) // 2}}}}}\n</tool_call>'

    gateway, sampler = make_gateway(policy)
    task, session = await open_service_task(NUMBER_SEARCH, "val:4")
    options = {"system_prompt": "Find the number with the guess tool."}

    run = await run_tree(gateway, "platoon-react", session, task, max_steps=12, options=options)

    assert run.result.status == "completed" and "correctly" in run.result.final_message
    assert run.outcome.reward == 1.0
    assert '"name": "guess"' in sampler.calls[0][0]  # the env tool reached the model as a function tool
