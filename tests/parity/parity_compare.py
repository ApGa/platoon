"""Compare two parity runs: everything training sees must be identical, up to a short allowlist.

:func:`canonical` turns a :class:`~parity_runs.GroupRun` into plain data, member by member:

* each trajectory as the rollout returned it, with every step's model call attached (``call``: the prompt tokens,
  the sampled tokens, their logprobs and the routing id) and every tool record's ``observation``, the stripped text
  the probe acted on;
* each model call's sampling parameters, in call order;
* the member's converted data before the group centered it: its rewards, trajectory statistics and datums;

and the group's baselines, its training datums after centering and filtering, its accounting and the statistics it
recorded.

**Identifiers** are random in every run: the rollout id (``<rollout>``), trajectory ids (``<traj K>``), agent ids
(``<agent K>``), model call ids (``<call K.J>``, the J-th step of the K-th trajectory), tool record ids
(``<tool K.I>``), and any other UUID, such as a subagent's task id (``<id N>``, by first appearance).  They are
replaced by these names wherever they appear, so the structure they encode (which step a call is, a child's parent,
the call a tool record followed) is still compared.

**The allowlist** (:data:`ALLOWLIST`) names everything else that may differ, and why: wall-clock times, where the
task came from, and what a mode chooses for the agent.  An allowed field is replaced by ``<allowed>`` in both forms
(and added to a form that lacks it).  What a mode chooses applies only between variants that choose differently:
tool names, arguments and raw texts only between variants that act through different tools (``Variant.surface``),
and grading signals only between variants graded differently (``Variant.grading``).  Between two ORS variants
they are compared like everything else.

:func:`differences` lists what differs, one readable line per field, and :func:`compare` puts the two together.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

ALLOWED = "<allowed>"

When = Literal["always", "surface", "grading"]


@dataclass(frozen=True)
class Allowance:
    """A field that may differ between two variants, why, and between which ones (``when``).

    ``path`` starts at a trajectory, and ``*`` matches any key or index; a ``task.*`` path also applies to every
    task in a subagent task's ``parent_tasks``.
    """

    path: str
    why: str
    when: When = "always"


ALLOWLIST: tuple[Allowance, ...] = (
    # Wall-clock data.
    Allowance("misc.model_call_spans", "model call spans are wall-clock times (gateway/trace.py)"),
    Allowance("misc.execution_span", "an agent's execution span is wall-clock time (harness/inline.py)"),
    Allowance("misc.tool_calls.*.started_at", "tool records are timed in wall-clock time (gateway/mcp.py)"),
    Allowance("misc.tool_calls.*.ended_at", "tool records are timed in wall-clock time (gateway/mcp.py)"),
    # Where the task came from.
    Allowance("task.id", "a task's id in its service: an ORS split:index, a Harbor task directory, a row id"),
    Allowance("task.misc.ors", "where an ORS task lives: its split, index and display name"),
    Allowance("task.misc.harbor", "where a Harbor task lives: its directory"),
    Allowance("task.misc.platoon_environment.service", "the variant's own service"),
    Allowance("task.misc.platoon_environment.service_kwargs", "the variant's own service options"),
    # What the mode chooses for the agent: the tools it acts through, and how it is graded.
    Allowance("task.misc.platoon_environment.harness_options.adapter", "the probe's adapter for the tools", "surface"),
    Allowance("misc.probe.listing", "the tools the variant's environment offers", "surface"),
    Allowance("misc.tool_calls.*.name", "the tool the probe's adapter calls: guess, act or bash", "surface"),
    Allowance("misc.tool_calls.*.arguments", "the arguments of the adapter's tool", "surface"),
    Allowance("misc.tool_calls.*.text", "the tool's raw text; its stripped observation is still compared", "surface"),
    Allowance("misc.tool_calls.*.finished", "ORS ends the episode at the winning call, Harbor after it", "grading"),
    Allowance("misc.tool_calls.*.reward", "ORS rewards the winning call, Harbor's tests the episode", "grading"),
    Allowance("misc.outcome.reason", "ORS calls a failed attempt no_submit, Harbor and gym give no reason", "grading"),
)

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_HEX32 = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")


def compare(reference: Any, candidate: Any, *, limit: int = 40) -> list[str]:
    """What differs between two :class:`~parity_runs.GroupRun` beyond :data:`ALLOWLIST`, one line per field."""

    return differences(canonical(reference, candidate), canonical(candidate, reference), limit=limit)


def allowances(variant: Any, other: Any) -> list[Allowance]:
    """The entries of :data:`ALLOWLIST` that apply between ``variant`` and ``other``."""

    applies = {"always"}
    if variant.surface != other.surface:
        applies.add("surface")
    if variant.grading != other.grading:
        applies.add("grading")
    return [allowance for allowance in ALLOWLIST if allowance.when in applies]


def canonical(run: Any, other: Any) -> dict[str, Any]:
    """``run`` as plain data, with its identifiers named and the fields it may differ in from ``other`` masked."""

    allowed = allowances(run.variant, other.variant)
    return {
        "members": [_member(member, allowed) for member in run.members],
        "group": {"baselines": run.baselines, **run.output},
    }


def _member(member: Any, allowed: Sequence[Allowance]) -> dict[str, Any]:
    collection = member.collection or {"id": None, "trajectories": {}}
    trajectories = [copy.deepcopy(trajectory) for trajectory in collection["trajectories"].values()]
    names = _Names()
    names.add(collection["id"], "rollout")
    for k, trajectory in enumerate(trajectories):
        names.add(trajectory["id"], f"traj {k}")
        names.add((trajectory["misc"].get("agent") or {}).get("agent_id"), f"agent {k}")
        for j, step in enumerate(trajectory["steps"]):
            names.add(_call_id(step), f"call {k}.{j}")
        for i, record in enumerate(trajectory["misc"].get("tool_calls") or ()):
            names.add(record.get("event_id"), f"tool {k}.{i}")
    attached: set[str | None] = set()
    for trajectory in trajectories:
        for step in trajectory["steps"]:
            call_id = _call_id(step)
            attached.add(call_id)
            step["call"] = _interaction(member.interactions.get(call_id))
        for record in trajectory["misc"].get("tool_calls") or ():
            record["observation"] = str(record.get("text", "")).strip()
        _mask_trajectory(trajectory, allowed)
    # Calls no step names (a judge's, say), in the order they were recorded.
    unattached = [_interaction(call) for call_id, call in member.interactions.items() if call_id not in attached]
    return names.apply(
        {
            "id": collection["id"],
            "trajectories": trajectories,
            "calls_without_a_step": unattached,
            "sampling": member.sampling,
            "converted": member.outcome,
        }
    )


def _call_id(step: dict[str, Any]) -> str | None:
    return ((step.get("misc") or {}).get("action_misc") or {}).get("completion_id")


def _interaction(interaction: Any) -> dict[str, Any] | None:
    if interaction is None:
        return None
    return {
        "prompt": interaction.obs.to_ints(),
        "sampled": list(interaction.action.tokens),
        "logprobs": interaction.action.maybe_logprobs,
        "routing_id": interaction.routing_id,
    }


def _mask_trajectory(trajectory: dict[str, Any], allowed: Iterable[Allowance]) -> None:
    for allowance in allowed:
        head, _, rest = allowance.path.partition(".")
        if head != "task":
            _mask(trajectory, allowance.path.split("."))
            continue
        tasks = [trajectory.get("task")]
        while tasks:
            task = tasks.pop()
            if isinstance(task, dict):
                _mask(task, rest.split("."))
                tasks.extend(task.get("parent_tasks") or ())


def _mask(node: Any, parts: list[str]) -> None:
    """Replace what ``parts`` names below ``node`` with :data:`ALLOWED`.  A named last key that ``node`` lacks is added,
    so a field that one variant has and the other lacks (``task.misc.ors`` against ``task.misc.harbor``) compares
    equal."""

    key, rest = parts[0], parts[1:]
    if isinstance(node, dict):
        keys = list(node) if key == "*" else [key] if key in node or not rest else []
    elif isinstance(node, list) and key == "*":
        keys = list(range(len(node)))
    else:
        return
    for name in keys:
        if rest:
            _mask(node[name], rest)
        else:
            node[name] = ALLOWED


class _Names:
    """Names for the random identifiers of one member: the given ones, then any other UUID by first appearance."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}
        self._unnamed = 0

    def add(self, value: Any, name: str) -> None:
        if isinstance(value, str) and value and value not in self.names:
            self.names[value] = f"<{name}>"

    def apply(self, node: Any) -> Any:
        """``node`` with every identifier replaced by its name, in keys and values alike."""

        known = sorted(self.names, key=len, reverse=True)  # a call id before the UUID inside it
        pattern = re.compile("|".join(map(re.escape, known))) if known else None

        def text(value: str) -> str:
            if pattern is not None:
                value = pattern.sub(lambda match: self.names[match.group(0)], value)
            for unnamed in (_UUID, _HEX32):
                value = unnamed.sub(self._name_unnamed, value)
            return value

        def walk(item: Any) -> Any:
            if isinstance(item, dict):
                return {text(key) if isinstance(key, str) else key: walk(value) for key, value in item.items()}
            if isinstance(item, (list, tuple)):
                return [walk(value) for value in item]
            return text(item) if isinstance(item, str) else item

        return walk(node)

    def _name_unnamed(self, match: re.Match[str]) -> str:
        value = match.group(0)
        if value not in self.names:
            self.names[value] = f"<id {self._unnamed}>"
            self._unnamed += 1
        return self.names[value]


# --------------------------------------------------------------------------------------------- differences


def differences(reference: Any, candidate: Any, *, limit: int = 40) -> list[str]:
    """Every field where ``candidate`` differs from ``reference``, as ``path: reference != candidate`` lines (at most
    ``limit``, then a count of the rest)."""

    found: list[str] = []
    _diff(reference, candidate, "", found)
    if len(found) > limit:
        found = [*found[:limit], f"... and {len(found) - limit} more"]
    return found


def _diff(a: Any, b: Any, path: str, found: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in [*a, *(key for key in b if key not in a)]:
            where = f"{path}.{key}" if path else str(key)
            if key not in b:
                found.append(f"{where}: only in the reference: {_short(a[key])}")
            elif key not in a:
                found.append(f"{where}: only in the candidate: {_short(b[key])}")
            else:
                _diff(a[key], b[key], where, found)
    elif isinstance(a, list) and isinstance(b, list):
        if all(not isinstance(item, (dict, list)) for item in [*a, *b]):  # token ids, logprobs, ...: one line
            if a != b:
                first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                window = slice(max(first - 3, 0), first + 5)
                found.append(
                    f"{path}: first differs at [{first}] (lengths {len(a)} and {len(b)}): "
                    f"{_short(a[window])} != {_short(b[window])}"
                )
            return
        for index in range(max(len(a), len(b))):
            where = f"{path}[{index}]"
            if index >= len(b):
                found.append(f"{where}: only in the reference: {_short(a[index])}")
            elif index >= len(a):
                found.append(f"{where}: only in the candidate: {_short(b[index])}")
            else:
                _diff(a[index], b[index], where, found)
    elif a != b or type(a) is not type(b):
        found.append(f"{path}: {_short(a)} != {_short(b)}")


def _short(value: Any, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
