"""What the CodeAct golden tests share: the allow-lists that explain every difference from the fixtures, the
scenarios' scaffolding, the recorded model's tokenizer, and the replay that compares a scenario with its fixture.

``test_codeact_golden.py`` replays core's scenarios (number search and ``submit_env``) against ``golden/``; the
TextCraft extension's ``test_textcraft_goldens.py`` replays the TextCraft-Synth ones against fixtures of its own.
The fixtures were recorded on the CodeAct code that predates its restructure (``codeact_golden.py``), and the
allow-lists name scenarios of both: :data:`TEXTCRAFT_SCENARIOS` are the extension's.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import codeact_golden as golden
import pytest
from inline_runs import open_service_task, run_tree

from platoon.rewards import delegation_capped

# The scenarios whose fixtures the TextCraft extension keeps (extensions/textcraft/tests/golden).
TEXTCRAFT_SCENARIOS = (
    "textcraft_synth_recursive_gather",
    "textcraft_synth_depth_aware",
    "textcraft_synth_recursive_rao_config",
    "textcraft_synth_child_runs_out_of_steps",
    "textcraft_synth_timeout_during_launch",
    "textcraft_synth_keyword_call",
)

# Every difference the restructure makes, with its reason.  The first four are intended; the others are judged
# equivalent.
#
# * Bad calls of env tools raise CPython's TypeError, in its words, and the traceback shows only the cell.  [unit]
# * The unawaited-call check covers only the agent's own async functions, not the original agent's archived
#   names.  [unit]
# * finish(None) follows the original halt rule: without a terminal tool it does not end the agent; with one, the
#   tool's reply ends it with the tool's text.  [number_search_finish_none, submit_env_finish_none, unit]
# * platoon-react and gym.loop end with LIMIT_MESSAGE, and gym.loop stops at ctx.max_steps.  [unit]
# * Only TextCraftPromptBuilder turns {item: n} targets into "Craft the following items: ..." and adds target_items;
#   other builders pass str(targets) and keep the parent's misc.  No shipped config launches subagents through
#   them.  [unit]
# * Arguments passed by keyword reach the env, and the trace, in parameter order: Python binds the call.  JSON
#   object order means nothing to the env or its pydantic models.  [textcraft_synth_keyword_call,
#   submit_env_terminal_tool]
# * An error raised inside a cell function shows the cell, "[... skipping hidden 3 frame]" and the function's own
#   frames instead of the old closures' (+/- a few lines); the error line is the same.  [unit]
# * help() names the executor module; an optional parameter's __defaults__ entry is <not passed>, since the
#   function applies its own default.  [unit]
# * When a cell times out or the agent ends, the call a cell is blocked in is cancelled and later calls raise, so
#   an ended agent no longer calls the env (it used to for up to 10 s, and after the grade).  [unit]
# * SystemExit from a task a cell started reaches the cell, and the cell loop goes on, also when a cell stops it (it
#   used to end, and the agent waited out cell_timeout_s).  [unit]
# * A loop of synthetic finish cells whose finish() the cells redefined ends the agent (it used to spin).  [rollout]
# * The checks option refuses stuck_in_loop_threshold < 1 and a negative window (0 crashed the first step).  [unit]
# * What the env prints or logs during a tool call stays out of the cell: calls run in a fresh context, and
#   IPython's Out[n]: prompt is dropped at the start of every line, as the original CodeActAgent meant to drop it.
#   Explained together with the SDK's quieted logs below (ENV_LOG_LINES).
#   [every scenario whose cells call env tools, rollout regression test, unit]
FINISH_NONE = (
    "finish(None) follows the original halt rule: an agent ends once it has a final message.  None is none, so "
    "without a terminal tool the agent goes on (the pre-restructure code ended it, with no message); with one, the "
    "terminal tool's reply ends it with the tool's text."
)
KEYWORD_ORDER = (
    "A stub is a real function, so Python binds the call and its arguments reach the env in parameter order, not "
    "in the call's keyword order; JSON object order means nothing to the env or its pydantic models."
)


def allow(reason: str, scenario: str, *paths: str, accept: Any = None) -> tuple[golden.AllowedDifference, ...]:
    return tuple(golden.AllowedDifference(reason, path, (scenario,), accept) for path in paths)


def _absent_before(expected: Any, actual: Any) -> bool:
    return expected is golden.ABSENT


ALLOWED_DIFFERENCES: tuple[golden.AllowedDifference, ...] = (
    *allow(
        FINISH_NONE,
        "number_search_finish_none",
        "/result/final_message",
        "/trajectories/root/finish_message",
        accept=lambda expected, actual: (expected, actual) == (None, "given up"),
    ),
    *allow(  # the second call, finish('given up'), which the agent now makes
        FINISH_NONE,
        "number_search_finish_none",
        "/trajectories/root/misc/model_generations/1",
        "/trajectories/root/calls/1",
        "/trajectories/root/steps/1",
        accept=_absent_before,
    ),
    *allow(  # the grade moves to the new last step; the datum and the stats grow by that call
        FINISH_NONE,
        "number_search_finish_none",
        "/trajectories/root/steps/0/reward_misc",
        "/trajectories/root/datums/0/*",
        "/train/trajectories/root/num_*",
    ),
    *allow(
        FINISH_NONE,
        "submit_env_finish_none",
        "/result/final_message",
        "/trajectories/root/finish_message",
        accept=lambda expected, actual: (expected, actual) == (None, "Submitted."),
    ),
    golden.AllowedDifference(
        KEYWORD_ORDER,
        "/trajectories/*/misc/tool_calls/*/arguments@keys",
        accept=lambda expected, actual: sorted(expected) == sorted(actual),
    ),
)

# Differences from changes made on purpose while the restructure was under way, or since; the fixtures predate them,
# so they are explained here too, and nothing else is:
#
# * The env's tool-call log lines (openreward's tool_call_completed, at INFO) no longer reach the cell output:
#   platoon quiets the SDK's INFO logs in process (platoon.envs.ors.sdk.quiet_in_process_logs).  Prompts are
#   shorter by those lines and so are the datums' observation runs; completions do not change.  An output that a
#   log line no longer precedes starts with IPython's Out[n]: prompt, which the original CodeActAgent's rule
#   dropped then, and CodeAct now drops at every line start (above).  [outputs: every scenario whose cells call env
#   tools (ENV_TOOL_CELLS); prompts, token counts and datums: those where a later call's prompt holds such an
#   output (ENV_TOOL_OUTPUTS_PROMPTED)]
# * The example configs' sampling moved from harness_options.sampling to their rollout config's inference_params, so
#   the config-driven scenario's tasks no longer carry the option; its calls sample as before.
#   [textcraft_synth_recursive_rao_config]
# * The harness is named platoon-codeact, not codeact, after platoon-react and platoon-bash.  Only the config-driven
#   scenario records the name: in its tasks' environment misc and its trajectory's harness misc.
#   [textcraft_synth_recursive_rao_config]
# * The root trajectory records the goal its session ran.  Only the config-driven scenario's task is an ORS task
#   whose goal (the env's prompt) comes when the session opens; the trace kept the unopened task's None.  The
#   subagents' tasks, forks of the root's, record that goal among their parent tasks too.
#   [textcraft_synth_recursive_rao_config]
# * The environment route carries on_harness_exit, so the config-driven scenario's tasks record its default.
#   [textcraft_synth_recursive_rao_config]
ENV_LOG_LINES = (
    "openreward's tool_call_completed lines no longer reach cell outputs, prompts or datums: platoon quiets the "
    "SDK's INFO logs in process (quiet_in_process_logs), and CodeAct runs a cell's calls in a fresh context.  "
    "IPython's Out[n]: prompt, which the original CodeActAgent's rule kept after such a line, is dropped at the "
    "start of every line, as that rule meant to drop it."
)
CONFIG_SAMPLING = (
    "The example configs' sampling moved from harness_options.sampling to the rollout config's inference_params, "
    "which the scenario passes on: the tasks no longer carry the harness option."
)
RECORDED_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "max_completion_tokens": 512}  # the config's, then
HARNESS_NAME = (
    "The harness codeact was renamed platoon-codeact (<owner>-<name>, like platoon-react and platoon-bash): the "
    "config names it so, and the tasks and the trajectory's harness record carry that name."
)
RENAMED_HARNESS = (("codeact", "platoon-codeact"), ("codeact@1", "platoon-codeact@1"))  # (recorded, now)
HARNESS_EXIT = (
    "Environment routes carry on_harness_exit, what a harness timeout or crash counts as; the tasks record its "
    "default, which excludes both, as the rollouts did before it existed."
)
ROOT_GOAL = (
    "The root trajectory records the goal its session ran: an ORS task gets its goal, the env's prompt, only when "
    "its session opens, and the trace kept the unopened task's None.  A subagent's task, a fork of the root's, records "
    "that goal among its parent tasks.  Prompts, rewards and datums are unchanged."
)
# The scenarios with cells that call env tools, whose outputs held log lines; the other four have none (react calls
# its tools outside cells).  Of those, the ones where a later call's prompt holds such an output: in the rest only an
# agent's last cell calls one, so their prompts, token counts and datums are as recorded.
ENV_TOOL_OUTPUTS_PROMPTED = (
    "number_search_bisection",
    "number_search_budget_exhaustion",
    "number_search_loop_edge_cases",
    "number_search_calls_off_the_cell_loop",
    "submit_env_terminal_tool",
    "textcraft_synth_recursive_gather",
    "textcraft_synth_recursive_rao_config",
    "textcraft_synth_keyword_call",
)
ENV_TOOL_CELLS = (
    *ENV_TOOL_OUTPUTS_PROMPTED,
    "number_search_cell_exception",
    "submit_env_finish_none",
    "textcraft_synth_depth_aware",
    "textcraft_synth_child_runs_out_of_steps",
)
_ENV_LOG_LINE = re.compile(r"^\[info\] tool_call_completed tool=\S+\n", re.MULTILINE)
_OUT_PROMPT = re.compile(r"^Out\[\d+\]:", re.MULTILINE)


def _as_observed(text: str) -> str:
    """``text`` without the env's tool-call log lines, and without IPython's ``Out[n]:`` prompt at any line start."""
    return _OUT_PROMPT.sub("", _ENV_LOG_LINE.sub("", text))


def _same_without_log_lines(expected: Any, actual: Any) -> bool:
    return isinstance(expected, str) and isinstance(actual, str) and _as_observed(expected) == _as_observed(actual)


def _no_more_tokens(expected: Any, actual: Any) -> bool:
    return type(expected) is int and type(actual) is int and actual <= expected


def _shorter_observations(expected: Any, actual: Any) -> bool:
    """Run-length encoded datum tensors (``0.0*753 1.0*30``) with the same runs, of which only zero runs, the
    observation tokens, may be shorter; a value that is not one (a datum's sha256) is not explained here."""

    if not (isinstance(expected, str) and isinstance(actual, str)):
        return False
    before, after = [run.rpartition("*") for run in expected.split()], [run.rpartition("*") for run in actual.split()]
    try:
        return len(before) == len(after) and all(
            value == new_value and (count == new_count or (float(value) == 0.0 and int(new_count) < int(count)))
            for (value, _, count), (new_value, _, new_count) in zip(before, after)
        )
    except ValueError:  # not run-length encoded
        return False


def _env_log_lines(path: str, scenarios: tuple[str, ...], accept: Any = None) -> golden.AllowedDifference:
    return golden.AllowedDifference(ENV_LOG_LINES, path, scenarios, accept)


INTEGRATION_DIFFERENCES: tuple[golden.AllowedDifference, ...] = (
    _env_log_lines("/trajectories/*/steps/*/output", ENV_TOOL_CELLS, _same_without_log_lines),
    _env_log_lines("/trajectories/*/calls/*/prompt", ENV_TOOL_OUTPUTS_PROMPTED, _same_without_log_lines),
    _env_log_lines("/trajectories/*/steps/*/usage/prompt_tokens", ENV_TOOL_OUTPUTS_PROMPTED, _no_more_tokens),
    _env_log_lines("/trajectories/*/steps/*/usage/total_tokens", ENV_TOOL_OUTPUTS_PROMPTED, _no_more_tokens),
    _env_log_lines("/trajectories/*/datums/*/length", ENV_TOOL_OUTPUTS_PROMPTED, _no_more_tokens),
    _env_log_lines("/trajectories/*/datums/*/sha256", ENV_TOOL_OUTPUTS_PROMPTED),  # of the shorter token ids
    _env_log_lines("/trajectories/*/datums/*", ENV_TOOL_OUTPUTS_PROMPTED, _shorter_observations),
    _env_log_lines("/train/trajectories/*/num_input_tokens", ENV_TOOL_OUTPUTS_PROMPTED, _no_more_tokens),
    *allow(
        CONFIG_SAMPLING,
        "textcraft_synth_recursive_rao_config",
        "/trajectories/*/task/misc/platoon_environment/harness_options/sampling",
        accept=lambda expected, actual: expected == RECORDED_SAMPLING and actual is golden.ABSENT,
    ),
    *allow(
        HARNESS_NAME,
        "textcraft_synth_recursive_rao_config",
        "/trajectories/*/task/misc/platoon_environment/harness",
        "/trajectories/*/misc/harness/spec",
        accept=lambda expected, actual: (expected, actual) in RENAMED_HARNESS,
    ),
    *allow(
        ROOT_GOAL,
        "textcraft_synth_recursive_rao_config",
        "/trajectories/root/task/goal",
        "/trajectories/*/task/parent_tasks/0",
        accept=lambda expected, actual: expected is None and actual == "Craft the following items: 3x a1_i5",
    ),
    *allow(
        HARNESS_EXIT,
        "textcraft_synth_recursive_rao_config",
        "/trajectories/*/task/misc/platoon_environment/on_harness_exit",
        accept=lambda expected, actual: (
            expected is golden.ABSENT and actual == {"timeout": "exclude", "crash": "exclude"}
        ),
    ),
)
ALLOWED_DIFFERENCES += INTEGRATION_DIFFERENCES

# The gateway counts the tool-call blocks of each model call that parsed only by a lenient rule and those that did not
# parse (docs/rfcs/0012-tool-call-parsing.md); the fixtures predate the counts.  CodeAct's turns hold no tool calls,
# so both are 0, and nothing else about the calls changes.  [every scenario]
TOOL_CALL_COUNTS = (
    "The gateway records how many of a call's tool-call blocks parsed only by a lenient rule and how many did not "
    "parse (RFC 0012); the fixtures predate the counts, and CodeAct's turns hold no tool calls, so both are 0."
)
ALLOWED_DIFFERENCES += tuple(
    golden.AllowedDifference(
        TOOL_CALL_COUNTS,
        f"/trajectories/*/misc/model_generations/*/{key}",
        accept=lambda expected, actual: expected is golden.ABSENT and actual == 0,
    )
    for key in ("recovered_tool_calls", "unparsed_tool_calls")
)


@dataclass
class GoldenRun:
    """What a scenario hands the recorder."""

    gateway: Any
    collection: Any
    interactions: dict[str, Any]
    task_id: str
    result: Any = None
    outcome: Any = None


MakeGateway = Callable[[Callable[[str], str]], Any]
Scenario = Callable[[MakeGateway, Any], Awaitable[GoldenRun]]


def cells(prompt: str) -> int:
    """How many cells the agent the prompt is for has run."""
    return len(re.findall(r"\[Cell \d+ Output\]", prompt))


def goal_line(prompt: str) -> str:
    """The first line of the goal of the agent the prompt is for."""
    return re.findall(r"Your Goal: (.*)", prompt)[-1]


def script(replies: list[str]) -> Callable[[str], str]:
    """A policy that answers an agent's n-th call with ``replies[n]`` (the last one from then on)."""
    return lambda prompt: replies[min(cells(prompt), len(replies) - 1)]


async def run_scenario(
    gateway: Any, service: str, task_id: str, harness: str = "platoon-codeact", **kwargs: Any
) -> GoldenRun:
    task, session = await open_service_task(service, task_id)
    run = await run_tree(gateway, harness, session, task, **kwargs)
    return GoldenRun(gateway, run.collection, run.interactions, task.id, run.result, run.outcome)


# --------------------------------------------------------------------------- the recorded model

# The fixtures hold what Qwen3-4B-Instruct-2507's own tokenizer made of every call (token ids, decoded prompts and the
# model name), so the replays render with that tokenizer instead of the toy one the other built-in harness tests use;
# they skip where it is not on disk.


@pytest.fixture(scope="module")
def model_name():
    return golden.MODEL_NAME


@pytest.fixture(scope="module")
def tokenizer():
    if not (golden.SNAPSHOT / "tokenizer.json").exists():
        pytest.skip(f"{golden.MODEL_NAME} tokenizer snapshot not available at {golden.SNAPSHOT}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(golden.SNAPSHOT))


@pytest.fixture(scope="module")
def renderer(tokenizer):
    from tinker_cookbook.renderers import get_renderer

    return get_renderer("qwen3_instruct", tokenizer)


async def replay(
    scenario: str,
    run: Scenario,
    make_gateway: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    directory: Path = golden.GOLDEN_DIR,
    allowed: Sequence[golden.AllowedDifference] = ALLOWED_DIFFERENCES,
) -> None:
    """Run ``scenario`` with a recording gateway and compare what it produced with its fixture in ``directory``: a
    difference fails unless an entry of ``allowed`` explains it.  With ``CODEACT_GOLDEN_RECORD=1``, record it."""

    golden.pin_env_log_lines(monkeypatch)
    golden.occupy_global_ipython_shell()

    def recording_gateway(policy: Callable[[str], str]) -> Any:
        gateway, _ = make_gateway(policy)
        golden.RecordingSampler.install(gateway)
        return gateway

    result = await run(recording_gateway, tmp_path)
    actual = golden.snapshot(
        scenario=scenario,
        collection=result.collection,
        interactions=result.interactions,
        sampler=result.gateway.sampler,
        tokenizer=result.gateway.tokenizer,
        task_id=result.task_id,
        reward_processor=delegation_capped,
        result=result.result,
        outcome=result.outcome,
    )
    if golden.recording():
        pytest.skip(f"recorded {golden.save(scenario, actual, directory)}")
    if not golden.fixture_path(scenario, directory).exists():
        pytest.fail(f"No golden fixture for {scenario}; record it with {golden.RECORD_ENV}=1 on pre-restructure code")

    unexplained = golden.unexplained(scenario, golden.compare(golden.load(scenario, directory), actual), allowed)
    assert not unexplained, golden.report(scenario, unexplained)
