"""The conformance check of a credit estimator (RFC 0021, section 6): :func:`check_credit_estimator`.

It is made of named checks (:data:`CHECKS`), which :func:`check_credit_estimator` runs one after the other; its
``AssertionError`` lists each check that failed, by name, with what it saw:

- ``protocol``: it resolves, it is an estimator class (an async ``assign(group)``, a bool ``needs_agents``, a
  synchronous ``settle`` if any, both or neither of ``state()`` and ``load_state(state)``), ``Class(setup, options)``
  builds it, and its load-time ``check``, given a loaded config of one environment entry (:func:`fixture_config`),
  returns or refuses with a ``ValueError``;
- ``options``: its schema takes the options given, and a dataclass's or a pydantic model's refuses an unknown key;
- ``action_tokens``: over :data:`GROUPS` fixture groups, ``assign`` sets advantages on action tokens only: every row
  keeps its length, its type and its values where its mask is 0, and every advantage is finite;
- ``resumes``: an estimator that credited and settled groups, and a new one that took up its state (``state()`` as
  JSON, then ``load_state``; nothing for an estimator without state), credit the next group alike: a resume of the run
  credits the groups that settle after it as the run would have.  An estimator with ``state()`` also keeps it as it
  was through ``assign`` alone: its state changes in ``settle`` only.  State that an estimator keeps without
  ``state()`` shows only when it changes how the next group is credited.

The fixture's groups are of four members, each a root agent (role ``policy``) that delegated to one subagent (role
``helper``, depth 1), each agent with one or two rows of eight target tokens, the action tokens holding the agent's
reward as the converter loads it; in each group a root was interrupted (no reward, no rows) and a subagent's rows were
left out (its reward counts).  For an estimator that does not read agents (``needs_agents`` False), the members carry
no agents and the rows no tags.

The experimental credit API (``platoon.experimental``) can change in a minor release, and this check changes with it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Mapping
from typing import Any

# The checks, by name, in the order they run.
CHECKS = ("protocol", "options", "action_tokens", "resumes")
# The fixture groups ``action_tokens`` credits, and those ``resumes`` settles before the one both estimators credit.
GROUPS = 8
SETTLED = 3
# The fixture's entry, and a key no options schema takes.
_ENTRY = "fixture"
_UNKNOWN_OPTION = "conformance_check_unknown_option"
# A row's target tokens, and which of them are actions.
_MASKS = ((0, 0, 1, 1, 1, 0, 1, 1), (0, 1, 1, 0, 0, 1, 1, 0))


def check_credit_estimator(estimator: type | str, *, options: Mapping[str, Any] | None = None) -> None:
    """Check a credit estimator against RFC 0021's contract (:data:`CHECKS`); raise ``AssertionError`` naming every
    check it fails, with what it saw.  ``estimator`` is a class, a name a package registers with
    ``@credit_estimator``, or ``module:Class``; ``options`` are checked against its options schema as a config's
    ``credit_options`` would be, and it is built with them.  Call it outside a running event loop."""

    name = estimator if isinstance(estimator, str) else getattr(estimator, "__qualname__", repr(estimator))
    failures = run_checks(estimator, options=options)
    if failures:
        lines = [f"{check}: {what}" for check, found in failures.items() for what in found]
        raise AssertionError(f"credit estimator {name} fails {len(failures)} check(s):\n  " + "\n  ".join(lines))


def run_checks(estimator: type | str, *, options: Mapping[str, Any] | None = None) -> dict[str, list[str]]:
    """Each check's problems, by name; the checks after ``protocol`` or ``options`` do not run when it fails."""

    cls, problems = _resolved(estimator)
    if cls is None:
        return {"protocol": problems}
    parsed, found = check_options(cls, options)
    if found:
        return {"options": found}
    failures = {"protocol": check_protocol(cls, parsed)}
    if not failures["protocol"]:
        failures["action_tokens"] = check_action_tokens(cls, parsed)
        failures["resumes"] = check_resumes(cls, parsed)
    return {check: found for check, found in failures.items() if found}


def _resolved(estimator: type | str) -> tuple[type | None, list[str]]:
    from platoon.extensions import ExtensionError
    from platoon.train.credit.registry import estimator_problem, resolve_credit_estimator

    if isinstance(estimator, str):
        try:
            return resolve_credit_estimator(estimator), []
        except ExtensionError as exc:
            return None, [str(exc)]
    if (problem := estimator_problem(estimator)) is not None:
        return None, [f"{getattr(estimator, '__qualname__', estimator)!r} is not a credit estimator: {problem}"]
    return estimator, []


# --- the checks ------------------------------------------------------------------------------------------------------


def check_options(cls: Any, options: Mapping[str, Any] | None) -> tuple[Any, list[str]]:
    """``options``: its schema takes the options given, and a dataclass's or a pydantic model's refuses an unknown key.
    Returns the options given, parsed."""

    from platoon.options import schema_options
    from platoon.utils.config import _is_pydantic

    schema = getattr(cls, "options_schema", None)
    checker = schema_options(schema, name=cls.__qualname__)
    parsed, problems = checker.check(dict(options or {}), "credit_options")
    typed = schema is not None and (
        (isinstance(schema, type) and dataclasses.is_dataclass(schema)) or _is_pydantic(schema)
    )
    if typed and not checker.option_problems({**dict(options or {}), _UNKNOWN_OPTION: 1}, "credit_options"):
        problems.append(f"its options schema takes an unknown key ({_UNKNOWN_OPTION}); it should refuse it")
    return parsed, problems


def check_protocol(cls: Any, parsed: Any) -> list[str]:
    """``protocol``: ``Class(setup, options)`` builds it, and its load-time ``check(options, ctx)`` returns or refuses
    the config with a ``ValueError``."""

    try:
        _instance(cls, parsed)
    except Exception as exc:
        return [f"Class(setup, options) raised {type(exc).__name__}: {exc}"]
    method = getattr(cls, "check", None)
    if callable(method):
        from platoon.train.credit.api import CreditCheckContext
        from platoon.train.policy.spec import MAIN

        try:
            method(parsed, CreditCheckContext(entry=_ENTRY, policies=(MAIN,), config=fixture_config()))
        except ValueError:
            pass  # a refusal, which says why
        except Exception as exc:
            return [
                f"check(options, ctx) raised {type(exc).__name__}: {exc} (a check is a static method that returns, or "
                "refuses the config with a ValueError)"
            ]
    return []


def check_action_tokens(cls: Any, parsed: Any) -> list[str]:
    """``action_tokens``: over :data:`GROUPS` fixture groups, every row keeps its length, its type and its values where
    its mask is 0, and every advantage is finite."""

    import torch

    estimator = _instance(cls, parsed)
    problems = []
    for seed in range(GROUPS):
        group = fixture_group(seed, tagged=cls.needs_agents)
        before = [sample.advantages.clone() for sample in group.samples()]
        try:
            asyncio.run(estimator.assign(group))
        except Exception as exc:
            problems.append(f"assign() raised on group {seed}: {type(exc).__name__}: {exc}")
            continue
        for index, (sample, old) in enumerate(zip(group.samples(), before, strict=True)):
            new = sample.advantages
            where = f"group {seed}, row {index}"
            if new.shape != old.shape or new.dtype != old.dtype:
                problems.append(
                    f"{where}: advantages of {tuple(new.shape)} {new.dtype}, not {tuple(old.shape)} {old.dtype}"
                )
            elif not torch.equal(new[sample.mask <= 0], old[sample.mask <= 0]):
                problems.append(f"{where}: advantages changed on tokens that are no actions (mask 0)")
            elif not bool(torch.isfinite(new).all()):
                problems.append(f"{where}: advantages that are not finite")
    return problems[:5]


def check_resumes(cls: Any, parsed: Any) -> list[str]:
    """``resumes``: an estimator that credited and settled :data:`SETTLED` groups, and a new one that took up its state
    (none without ``state()``), credit the next group alike, and then give the same state; and each one's ``assign`` of
    that group alone leaves its ``state()`` as it was."""

    stateful = callable(getattr(cls, "state", None))
    running, resumed = _instance(cls, parsed), _instance(cls, parsed)
    try:
        for seed in range(SETTLED):
            _credit(running, fixture_group(seed, tagged=cls.needs_agents))
        if stateful:
            try:
                state = json.loads(json.dumps(running.state()))
            except (TypeError, ValueError) as exc:
                return [f"state() gave no JSON data: {exc}"]
            resumed.load_state(state)
        advantages = []
        for estimator in (running, resumed):
            group = fixture_group(SETTLED, tagged=cls.needs_agents)
            before = json.dumps(estimator.state()) if stateful else None
            asyncio.run(estimator.assign(group))
            if stateful and json.dumps(estimator.state()) != before:  # compared as text: a state may share its lists
                return ["assign(group) changed state(): an estimator changes its state in settle(group) only"]
            _settle(estimator, group)
            advantages.append([sample.advantages.tolist() for sample in group.samples()])
    except Exception as exc:
        return [f"crediting and settling the fixture's groups raised {type(exc).__name__}: {exc}"]
    if advantages[0] != advantages[1]:
        how = "after taking up its state" if stateful else "(it keeps no state())"
        return [f"a new estimator {how} credits the next group otherwise than the one that credited the groups before"]
    if stateful and running.state() != resumed.state():
        return ["after the same group, the resumed estimator's state() differs from the running one's"]
    return []


def _credit(estimator: Any, group: Any) -> None:
    """Credit ``group``, then settle it as the run does."""

    asyncio.run(estimator.assign(group))
    _settle(estimator, group)


def _settle(estimator: Any, group: Any) -> None:
    settle = getattr(estimator, "settle", None)
    if callable(settle):
        settle(group)


def fixture_config() -> Any:
    """A loaded training config of one environment entry, the fixture's, with every training setting at its default:
    the ``config`` of the load-time check's context."""

    from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import build_config

    return build_config(
        PlatoonTinkerRLTrainerConfig,
        {
            "train": {"model_name": "fixture-model", "renderer_name": "fixture-renderer"},
            "environments": [{"name": _ENTRY}],
            "log_path": "unused",
        },
    )


def _instance(cls: Any, parsed: Any) -> Any:
    from platoon.train.config_defs import WorkflowConfig
    from platoon.train.credit.api import CreditSetup
    from platoon.train.policy.spec import MAIN

    return cls(CreditSetup(name=cls.__qualname__, entry=_ENTRY, settings=WorkflowConfig(), policies=(MAIN,)), parsed)


# --- the fixture -----------------------------------------------------------------------------------------------------


def fixture_group(seed: int, *, tagged: bool) -> Any:
    """Fixture group ``seed`` (the module's docstring): its rewards vary with ``seed``; ``tagged`` gives its members
    their agents and its rows their tags."""

    from platoon.train.config_defs import WorkflowConfig
    from platoon.train.credit.api import CreditGroup, PolicyView
    from platoon.train.policy.spec import MAIN
    from platoon.utils.stats_tracker import DeferredStats

    members = [_member(seed, index, tagged=tagged) for index in range(4)]
    return CreditGroup(
        task_id=f"task-{seed}",
        label=None,
        entry=_ENTRY,
        settings=WorkflowConfig(),
        members=members,
        policies={MAIN: PolicyView(MAIN)},
        versions={MAIN: 0},
        stats=DeferredStats(),
    )


def _member(seed: int, index: int, *, tagged: bool) -> Any:
    """Member ``index`` of fixture group ``seed``: a root and one subagent; the group's member ``seed % 4`` has an
    interrupted root, and its member ``(seed + 1) % 4`` a subagent whose rows were left out."""

    from platoon.train.credit.api import AgentRun, MemberRollout, TrainingSample
    from platoon.train.policy.spec import MAIN
    from platoon.utils.tinker_data_processing import TrajectoryCollectionResult

    interrupted, unsampled = index == seed % 4, index == (seed + 1) % 4
    root_reward = None if interrupted else float((seed + index) % 3) / 2
    child_reward = float((seed * index + 1) % 2)
    agents = [
        AgentRun(f"root-{index}", f"t{index}", None, "policy", MAIN, 0, root_reward),
        AgentRun(f"helper-{index}", f"t{index}.1", f"t{index}", "helper", MAIN, 1, child_reward),
    ]
    rows = {agents[0].trajectory_id: 0 if interrupted else 2, agents[1].trajectory_id: 0 if unsampled else 1}
    samples = []
    for agent in agents:
        for row in range(rows[agent.trajectory_id]):
            sample = TrainingSample(_datum(agent.reward or 0.0, _MASKS[row]), MAIN, index)
            if tagged:
                sample.agent_id, sample.role = agent.agent_id, agent.role
                sample.trajectory_id, sample.depth = agent.trajectory_id, agent.depth
                agent.samples.append(sample)
            samples.append(sample)
    result = TrajectoryCollectionResult(
        datums=[sample.datum for sample in samples],
        task_reward=root_reward if root_reward is not None else 0.0,
        trajectory_stats=[],
        root_rewards_dict={},
        task_reward_valid=not interrupted,
    )
    return MemberRollout(index, samples, agents if tagged else [], result.task_reward, not interrupted, result)


def _datum(reward: float, mask: tuple[int, ...]) -> Any:
    """A row of eight target tokens whose action tokens (``mask``) hold ``reward``, as the converter loads it."""

    import tinker
    import torch

    tokens = list(range(10, 10 + len(mask)))
    inputs = {
        "target_tokens": torch.tensor(tokens[1:] + [2]),
        "logprobs": torch.full((len(mask),), -0.5),
        "advantages": torch.tensor([reward if action else 0.0 for action in mask]),
        "mask": torch.tensor([float(action) for action in mask]),
        "checkpoint_version": torch.tensor([0]),
    }
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens),
        loss_fn_inputs={key: tinker.TensorData.from_torch(value) for key, value in inputs.items()},
    )
