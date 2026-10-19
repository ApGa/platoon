"""Golden fixtures of the CodeAct harness: record what a scripted rollout produced, load it back and compare.

The CodeAct restructure (the agent class split into the harness's ``prompt_builder``, ``namespace``, ``checks``
and ``executor`` modules) must keep behavior equivalent, and the rollout tests check only substrings.  A fixture
(``golden/<scenario>.json``) holds what the model saw and what the trainer gets, so a replay can be compared with
it in full:

* ``trajectories``, one per agent, keyed by tree position (``root``, ``root > <first line of the child's goal>``):
  the parent link and fork step, the task, reward, finish and error messages, the trajectory misc (role, result,
  outcome, flags, tool calls, generation diagnostics), every model call (``calls``: the decoded prompt the sampler
  got, the decoded completion and the sampling parameters), every step misc (``steps``: the ``thought``, ``code``,
  ``output`` and ``error`` annotations, ``reward_misc``, the parsed message and usage) and the datums
  ``trajectory_to_data`` makes of it (token ids as length and hash, the other tensors run-length encoded);
* ``train``: the rewards and per-trajectory stats of ``get_train_data_for_trajectory_collection``;
* ``result`` and ``outcome``: how the root agent's run ended and the episode's grade.

Ids and clocks (call, agent, task, event and tool-call ids, timestamps and spans) are left out; everything else is
kept as recorded, key order included.  :func:`compare` lists the differences between two snapshots by path, and
:class:`AllowedDifference` lets a test accept the ones it can justify.  In the files, strings with newlines are
stored as ``{"$lines": [...]}``, so a re-recorded fixture diffs line by line.
"""

from __future__ import annotations

import atexit
import dataclasses
import difflib
import fnmatch
import hashlib
import importlib
import json
import logging
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub.constants import HF_HUB_CACHE

GOLDEN_DIR = Path(__file__).parent / "golden"
RECORD_ENV = "CODEACT_GOLDEN_RECORD"
FORMAT = 1
# The model the fixtures were recorded with, and its tokenizer files: in the standard Hugging Face cache (HF_HUB_CACHE,
# else HF_HOME/hub, else ~/.cache/huggingface/hub) unless QWEN3_4B_INSTRUCT_SNAPSHOT names another directory.
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SNAPSHOT = Path(
    os.environ.get(
        "QWEN3_4B_INSTRUCT_SNAPSHOT",
        Path(HF_HUB_CACHE) / "models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554",
    )
)

_CLOCK_MISC = ("execution_span", "model_call_spans")  # trajectory misc that holds clock readings
_TOOL_CALL_CLOCKS = ("event_id", "started_at", "ended_at")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_SAMPLING_FIELDS = ("max_tokens", "temperature", "top_k", "top_p", "seed", "stop")
_LINES = "$lines"


# --------------------------------------------------------------------------- recording


@dataclass
class SampledCall:
    prompt: Any  # the tinker.ModelInput the sampler got; held so its id() stays unique
    sampling_params: Any
    tokens: list[int]


class RecordingSampler:
    """Wraps a gateway's sampler and keeps every call, keyed by its prompt object.

    The gateway records that same object as the call's ``TinkerLLMInteraction.obs``, so each trajectory step finds
    the exact prompt and sampling parameters the sampler got, however concurrent agents interleaved.
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: dict[int, SampledCall] = {}

    @classmethod
    def install(cls, gateway: Any) -> RecordingSampler:
        """Wrap ``gateway.sampler``, which the gateway reads at every call."""
        recorder = cls(gateway.sampler)
        gateway.sampler = recorder
        return recorder

    async def sample_async(self, *, prompt: Any, sampling_params: Any, num_samples: int) -> Any:
        response = await self.inner.sample_async(
            prompt=prompt, sampling_params=sampling_params, num_samples=num_samples
        )
        self.calls[id(prompt)] = SampledCall(prompt, sampling_params, list(response.sequences[0].tokens))
        return response

    def sample(self, *, prompt: Any, sampling_params: Any, num_samples: int) -> Any:
        return self.inner.sample(prompt=prompt, sampling_params=sampling_params, num_samples=num_samples)


def snapshot(
    *,
    scenario: str,
    collection: Any,
    interactions: Mapping[str, Any],
    sampler: RecordingSampler,
    tokenizer: Any,
    task_id: str,
    reward_processor: Callable[[dict], tuple[float, dict]],
    result: Any = None,
    outcome: Any = None,
) -> dict[str, Any]:
    """Everything a finished scripted rollout produced, as JSON values (see the module docstring)."""

    from platoon.utils.tinker_data_processing import get_train_data_for_trajectory_collection, trajectory_to_data

    data = collection.to_dict()
    trajectories: dict[str, dict[str, Any]] = data["trajectories"]
    labels, depths = _tree(trajectories)
    unclaimed = dict(sampler.calls)
    agents: dict[str, Any] = {}
    for trajectory_id, trajectory in trajectories.items():
        calls = []
        for step in trajectory["steps"]:
            interaction = interactions[step["misc"]["action_misc"]["completion_id"]]
            calls.append(_call(unclaimed.pop(id(interaction.obs), None), interaction, tokenizer))
        reward, _ = reward_processor(trajectory)
        datums = trajectory_to_data(
            trajectory=trajectory,
            interactions=interactions,
            task_id=task_id,
            trajectory_id=trajectory_id,
            trajectory_advantage=reward,
            checkpoint_version=0,
            traj_depth=depths[trajectory_id],
        ).datums
        parent = trajectory.get("parent_info")
        agents[labels[trajectory_id]] = {
            "parent": labels.get(parent["id"], "<orphan>") if parent else None,
            "fork_step": parent["fork_step"] if parent else None,
            "depth": depths[trajectory_id],
            "task": _task(trajectory.get("task")),
            "reward": trajectory["reward"],
            "finish_message": trajectory["finish_message"],
            "error_message": trajectory["error_message"],
            "misc": _trajectory_misc(trajectory["misc"]),
            "calls": calls,
            "steps": [_step_misc(step["misc"]) for step in trajectory["steps"]],
            "datums": [_datum(datum) for datum in datums],
        }
    train = get_train_data_for_trajectory_collection(
        data,
        interactions,
        task_id,
        checkpoint_version=0,
        reward_processor=reward_processor,
        include_traj_depth=True,
    )
    recorded = {
        "format": FORMAT,
        "scenario": scenario,
        "result": _plain(result),
        "outcome": _plain(outcome),
        "trajectories": agents,
        # Sampler calls no trajectory step claims (none expected), by decoded prompt.
        "unattributed_calls": sorted(tokenizer.decode(call.prompt.to_ints()) for call in unclaimed.values()),
        "train": {
            "task_reward": train.task_reward,
            "task_reward_valid": train.task_reward_valid,
            "root_rewards_dict": train.root_rewards_dict,
            "num_datums": len(train.datums),
            "num_policy_excluded_datums": train.num_policy_excluded_datums,
            "trajectories": {
                labels[stats.trajectory_id]: {
                    "reward": stats.reward,
                    "rewards_dict": stats.rewards_dict,
                    "is_root": stats.is_root,
                    "num_steps": stats.num_steps,
                    "num_datums": stats.num_datums,
                    "num_input_tokens": stats.num_input_tokens,
                    "num_output_tokens": stats.num_output_tokens,
                }
                for stats in train.trajectory_stats
            },
        },
    }
    return json.loads(json.dumps(recorded, default=str))  # JSON values only, exactly as a loaded fixture


def _tree(trajectories: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, str], dict[str, int]]:
    """Each trajectory's label and depth.  The collection lists every parent before its children."""

    labels: dict[str, str] = {}
    depths: dict[str, int] = {}
    for trajectory_id, trajectory in trajectories.items():
        parent = trajectory.get("parent_info")
        if parent is None or parent["id"] not in labels:
            base, depth = "root" if parent is None else "<orphan>", 0
        else:
            goal = str((trajectory.get("task") or {}).get("goal") or "")
            base, depth = f"{labels[parent['id']]} > {goal.splitlines()[0] if goal else ''}", depths[parent["id"]] + 1
        label, copies = base, 1
        while label in labels.values():
            copies += 1
            label = f"{base} #{copies}"
        labels[trajectory_id], depths[trajectory_id] = label, depth
    return labels, depths


def _call(sampled: SampledCall | None, interaction: Any, tokenizer: Any) -> dict[str, Any]:
    if sampled is None:  # the gateway recorded a call the sampler never got
        return {"prompt": tokenizer.decode(interaction.obs.to_ints()), "completion": None, "sampling": None}
    return {
        "prompt": tokenizer.decode(sampled.prompt.to_ints()),
        "completion": tokenizer.decode(sampled.tokens),
        "sampling": {name: getattr(sampled.sampling_params, name, None) for name in _SAMPLING_FIELDS},
    }


def _task(task: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A task without its random id (forked tasks get a uuid); parent tasks by goal (each has its own entry)."""

    if task is None:
        return None
    out = dict(task)
    if isinstance(out.get("id"), str) and _UUID.match(out["id"]):
        out["id"] = "<uuid>"
    if "parent_tasks" in out:
        out["parent_tasks"] = [parent.get("goal") for parent in out["parent_tasks"]]
    return out


def _trajectory_misc(misc: Mapping[str, Any]) -> dict[str, Any]:
    calls = list(misc.get("model_generations") or {})  # the agent's call ids, in call order
    out: dict[str, Any] = {}
    for key, value in misc.items():
        if key in _CLOCK_MISC:
            continue
        if key == "agent":
            value = {name: item for name, item in value.items() if name != "agent_id"}
        elif key == "model_generations":
            value = list(value.values())
        elif key == "tool_calls":
            value = [_tool_call(record, calls) for record in value]
        out[key] = value
    return out


def _tool_call(record: Mapping[str, Any], calls: Sequence[str]) -> dict[str, Any]:
    out = {name: value for name, value in record.items() if name not in _TOOL_CALL_CLOCKS}
    after = out.pop("after_call_id", None)
    out["after_call"] = calls.index(after) if after in calls else after  # the index of the agent's call
    return out


def _step_misc(misc: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(misc)
    action = out.get("action_misc")
    if isinstance(action, Mapping):
        out["action_misc"] = {name: value for name, value in action.items() if name != "completion_id"}
    message = out.get("message")
    if isinstance(message, Mapping) and message.get("tool_calls"):
        out["message"] = {**message, "tool_calls": [{**call, "id": "<id>"} for call in message["tool_calls"]]}
    return out


def _datum(datum: Any) -> dict[str, Any]:
    tokens: list[Any] = []
    for chunk in datum.model_input.chunks:
        chunk_tokens = getattr(chunk, "tokens", None)
        tokens.extend(chunk_tokens if chunk_tokens is not None else [f"<{type(chunk).__name__}:{chunk.length}>"])
    out: dict[str, Any] = {"model_input": _token_ids(tokens)}
    for key, tensor in sorted(datum.loss_fn_inputs.items()):
        values = tensor.to_torch().flatten().tolist()
        out[key] = _token_ids(values) if key == "target_tokens" else _runs(values)
    return out


def _token_ids(tokens: Sequence[Any]) -> dict[str, Any]:
    digest = hashlib.sha256(json.dumps(list(tokens)).encode()).hexdigest()[:16]
    return {"length": len(tokens), "sha256": digest}


def _runs(values: Iterable[Any]) -> str:
    """Consecutive equal values as ``value*count``, space-separated (``0.0*132 -0.5*11``)."""
    runs: list[list[Any]] = []
    for value in values:
        if runs and type(runs[-1][0]) is type(value) and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return " ".join(f"{value!r}*{count}" for value, count in runs)


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


# --------------------------------------------------------------------------- process state that reaches the cells


def occupy_global_ipython_shell() -> None:
    """Make sure the scenario's shells are not IPython's global shell.

    IPython makes the first ``InteractiveShellEmbed`` of a process its global instance, and only that shell's cells
    get tracebacks that read ``Cell In[1], line 1``; every later shell's cells read ``File <ipython-input-1-...>:1``.
    A rollout worker's agents nearly all get the second form, and so does every scenario once a shell created here
    (configured as ``CellExecutor`` configures its own) holds the global slot, whatever ran before.
    """

    from IPython.core.interactiveshell import InteractiveShell

    if InteractiveShell.initialized():
        return
    from IPython.terminal.embed import InteractiveShellEmbed
    from traitlets.config.loader import Config

    config = Config()
    config.HistoryManager.enabled = False
    excepthook = sys.excepthook
    shell = InteractiveShellEmbed(config=config)
    sys.excepthook = excepthook
    atexit.unregister(shell.atexit_operations)


class _StdoutLogger:
    """structlog's ``PrintLogger`` as openreward uses it: it prints to whatever ``sys.stdout`` is at the time."""

    def msg(self, message: str) -> None:
        print(message, flush=True)

    log = debug = info = warn = warning = error = err = critical = fatal = failure = exception = msg


def _render(_logger: Any, method: str, event: dict[str, Any]) -> str:
    return f"[{method}] {event['event']}" + (f" tool={event['tool']}" if "tool" in event else "")


def pin_env_log_lines(monkeypatch: Any) -> None:
    """Make the env SDK's log lines deterministic for one test.

    openreward logs each tool call at INFO to the current ``sys.stdout``.  A tool called from a cell ran in a copy
    of the cell's context, so the line landed in the cell's output and the model's next prompt.  That is how the
    pre-restructure code behaved, and the fixtures record it.  Now the lines stay out of the cell (platoon quiets
    the SDK in process, and a cell's calls run in a fresh context), a difference the allow-list explains.  The
    line's timestamp and timings change every run and its fields vary across SDK versions (0.1.125 has no
    ``lock_wait_ms``), so for the test the line carries only its level, event and tool name.
    """

    import structlog

    for name in ("openreward.environments.environment", "openreward.environments.session"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(module, "logger"):
            logger = structlog.wrap_logger(
                _StdoutLogger(), processors=[_render], wrapper_class=structlog.make_filtering_bound_logger(logging.INFO)
            )
            monkeypatch.setattr(module, "logger", logger)


# --------------------------------------------------------------------------- fixtures


def fixture_path(scenario: str, directory: Path = GOLDEN_DIR) -> Path:
    """Where ``scenario``'s fixture is: ``golden/<scenario>.json`` here, or in a test suite's own ``directory``."""
    return directory / f"{scenario}.json"


def recording() -> bool:
    """Whether this run (re)records the fixtures instead of comparing with them (``CODEACT_GOLDEN_RECORD=1``)."""
    return os.environ.get(RECORD_ENV, "") not in ("", "0")


def save(scenario: str, recorded: Mapping[str, Any], directory: Path = GOLDEN_DIR) -> Path:
    path = fixture_path(scenario, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_encode(recorded), indent=1, ensure_ascii=False) + "\n")
    return path


def load(scenario: str, directory: Path = GOLDEN_DIR) -> dict[str, Any]:
    return _decode(json.loads(fixture_path(scenario, directory).read_text()))


def _encode(value: Any) -> Any:
    if isinstance(value, str) and "\n" in value:
        return {_LINES: value.split("\n")}
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if list(value) == [_LINES]:
            return "\n".join(value[_LINES])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


# --------------------------------------------------------------------------- comparison


class _Absent:
    def __repr__(self) -> str:
        return "<absent>"


ABSENT: Any = _Absent()  # the side of a difference that has no value at its path


@dataclass(frozen=True)
class Difference:
    """A value that differs at ``path`` (``/``-separated keys and indices, JSON-pointer escaped).

    An object whose keys come in another order differs at ``<its path>@keys``, with the shared keys in each order.
    """

    path: str
    expected: Any = ABSENT
    actual: Any = ABSENT


@dataclass(frozen=True)
class AllowedDifference:
    """A difference from the fixtures that a test accepts, and why (``test_codeact_golden.py`` lists them).

    ``path`` is an ``fnmatch`` pattern over difference paths (``/trajectories/*/steps/*/output``); ``scenarios``
    limits the entry to those fixtures (empty: all of them); ``accept``, when given, must also hold for the
    ``(expected, actual)`` values (either may be :data:`ABSENT`).
    """

    reason: str
    path: str
    scenarios: tuple[str, ...] = ()
    accept: Callable[[Any, Any], bool] | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError(f"The allowed difference at {self.path!r} needs a reason")

    def explains(self, scenario: str, difference: Difference) -> bool:
        return (
            (not self.scenarios or scenario in self.scenarios)
            and fnmatch.fnmatchcase(difference.path, self.path)
            and (self.accept is None or bool(self.accept(difference.expected, difference.actual)))
        )


def compare(expected: Any, actual: Any, path: str = "") -> list[Difference]:
    """The differences between two JSON values; scalars must match in type too (``1`` is not ``1.0``), and objects
    in key order (a JSON file keeps the order a recording had)."""

    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[Difference] = []
        for key in [*expected, *(key for key in actual if key not in expected)]:
            at = f"{path}/{str(key).replace('~', '~0').replace('/', '~1')}"
            if key not in actual:
                differences.append(Difference(at, expected=expected[key]))
            elif key not in expected:
                differences.append(Difference(at, actual=actual[key]))
            else:
                differences.extend(compare(expected[key], actual[key], at))
        expected_order = [key for key in expected if key in actual]
        actual_order = [key for key in actual if key in expected]
        if expected_order != actual_order:
            differences.append(Difference(f"{path}@keys", expected_order, actual_order))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        differences = []
        for index in range(max(len(expected), len(actual))):
            at = f"{path}/{index}"
            if index >= len(actual):
                differences.append(Difference(at, expected=expected[index]))
            elif index >= len(expected):
                differences.append(Difference(at, actual=actual[index]))
            else:
                differences.extend(compare(expected[index], actual[index], at))
        return differences
    if type(expected) is type(actual) and expected == actual:
        return []
    return [Difference(path or "/", expected, actual)]


def unexplained(
    scenario: str, differences: Iterable[Difference], allowed: Iterable[AllowedDifference]
) -> list[Difference]:
    """The ``differences`` that no allow-list entry explains."""

    allowed = list(allowed)
    return [
        difference for difference in differences if not any(entry.explains(scenario, difference) for entry in allowed)
    ]


def report(scenario: str, differences: Sequence[Difference], *, limit: int = 20) -> str:
    """A readable account of ``differences``: a line diff for multi-line text, ``expected``/``got`` otherwise."""

    lines = [f"{len(differences)} unexplained difference(s) from {fixture_path(scenario).name}:"]
    for difference in differences[:limit]:
        expected, actual = difference.expected, difference.actual
        body: list[str] = []
        if isinstance(expected, str) and isinstance(actual, str) and ("\n" in expected or "\n" in actual):
            diff = difflib.unified_diff(
                expected.splitlines(), actual.splitlines(), "expected", "actual", n=2, lineterm=""
            )
            body = list(diff)  # empty when only a trailing newline differs
        if body:
            lines.append(f"{difference.path}:")
            lines.extend(f"    {line}" for line in body[:40])
            if len(body) > 40:
                lines.append(f"    ... {len(body) - 40} more diff lines")
        else:
            lines.append(f"{difference.path}: expected {_short(expected)}, got {_short(actual)}")
    if len(differences) > limit:
        lines.append(f"... and {len(differences) - limit} more")
    lines.append(
        "Explain each with an AllowedDifference (and its reason) in golden_scenarios.py, or in the test module that "
        "replays the scenario, or fix the change. "
        f"Re-record ({RECORD_ENV}=1) only on code that predates the CodeAct restructure."
    )
    return "\n".join(lines)


def _short(value: Any, limit: int = 300) -> str:
    text = repr(value)
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text)} chars)"
