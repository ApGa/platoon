"""Component options (RFC 0016): the path that checks the options a config sets for the components its
``environments`` entries select, per entry and phase, before anything runs, and records them with their defaults.

A component that declares options has a schema (:class:`OptionSchema`): a TOML harness spec through its
``[options]`` table (:class:`platoon.harness.spec.OptionSpec`).  Function harnesses, episode programs, named
environments and reward processors take the same path once they declare options (RFC 0016, M2), through
:func:`option_schema`; until then their options go unchecked, as before.

**Phases.**  ``train`` rollouts run with an entry's own options.  Evaluation rollouts, those of every split but
``train`` (the rollouts of ``platoon eval``, and those ``platoon train`` runs when it evaluates), run with the
entry's ``eval_overrides`` merged over them, key by key (:func:`harness_options`).  The routes that carry options to
rollout workers are built per phase (``platoon.train.auto.TaskRouter``), so a worker never has to know the phase.

**Where it runs.** ``platoon train``, ``eval`` and ``rollout``, and ``platoon config validate`` (unless
``--no-resolve``), check every entry's options when they resolve the config's components
(:func:`platoon.extensions.run_components`), and ``lock.json`` records the options each harness ran with, defaults
included, per entry and phase. The trainer and the evaluator check them again when they build the routes, and a
rollout worker parses them once more when it renders a launch, so a schema is the one place that says what an option
means.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from platoon.train.components import EnvironmentConfig

# Which rollouts options apply to: ``train`` rollouts, or evaluation rollouts (every split but ``train``).
Phase = Literal["train", "eval"]
PHASES: tuple[Phase, ...] = ("train", "eval")


def phase_of(split: str) -> Phase:
    """The phase of a split's rollouts: ``train`` for the train split, ``eval`` for every other split."""

    return "train" if split == "train" else "eval"


@runtime_checkable
class OptionSchema(Protocol):
    """What the shared path needs from a component's option schema."""

    def option_problems(self, values: Any, where: str) -> list[str]:
        """Every problem with ``values`` as the component's options, one line each, with its full dotted path under
        ``where``: an unknown key with a did-you-mean, a value of the wrong type, a value it does not allow."""
        ...

    def resolve_options(self, values: Mapping[str, Any] | None, where: str) -> dict[str, Any]:
        """``values`` checked and completed with the defaults, as plain data; raises ``ValueError`` with every
        problem."""
        ...


def option_schema(component: Any) -> OptionSchema | None:
    """The option schema of a resolved component, or None when nothing checks its options.

    A TOML harness spec always has one: a spec without ``[options]`` takes none, so any option set for it is a
    problem.  A function harness has none yet, so its ``harness_options`` reach it unchecked, as ``ctx.options``.
    """

    from platoon.harness.spec import HarnessSpec

    return component if isinstance(component, HarnessSpec) else None


def harness_options(environment: EnvironmentConfig, phase: Phase) -> dict[str, Any]:
    """The ``harness_options`` an entry's rollouts of ``phase`` run with: its own, and for ``eval`` its
    ``eval_overrides.harness_options`` merged over them, key by key."""

    from platoon.utils.config import deep_merge

    own = copy.deepcopy(dict(environment.harness_options))
    if phase == "train":
        return own
    return deep_merge(own, environment.eval_overrides.harness_options)


@dataclass(frozen=True)
class CheckedOptions:
    """One entry's options for one component: what each phase runs with, defaults included (empty when the
    component has no schema, or when there are problems), and every problem, one line each."""

    values: dict[Phase, dict[str, Any]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def check_harness_options(
    environment: EnvironmentConfig, harness: Any, *, where: str, phases: Sequence[Phase] = PHASES
) -> CheckedOptions:
    """Check an entry's options for one of its resolved harnesses against the harness's schema.

    ``where`` is the entry's dotted path (``environments.0``).  The entry's own ``harness_options`` and its
    ``eval_overrides.harness_options`` are each checked where they are written, whatever ``phases`` says, so a
    mistake in either is found before any rollout of either phase; ``phases`` are the phases whose values to give.
    The harnesses of a mixture share one mapping, which must suit each of them, and a problem then says so.
    """

    schema = option_schema(harness)
    if schema is None:
        return CheckedOptions()
    problems = schema.option_problems(environment.harness_options, f"{where}.harness_options")
    overrides = environment.eval_overrides.harness_options
    if overrides:
        problems += schema.option_problems(overrides, f"{where}.eval_overrides.harness_options")
    if problems and isinstance(environment.harness, Mapping) and len(environment.harness) > 1:
        problems.append(
            f"{where}.harness_options: the harnesses of the mixture {sorted(environment.harness)} share one mapping, "
            "which must suit each of them; give a harness whose options the others do not take an environments "
            "entry of its own"
        )
    if problems:
        return CheckedOptions(problems=problems)
    values = {
        phase: schema.resolve_options(harness_options(environment, phase), f"{where}.harness_options")
        for phase in phases
    }
    return CheckedOptions(values=values)


def config_phases(config: Any, split: str | None = None) -> tuple[Phase, ...]:
    """The phases whose rollouts a command runs with a loaded config: with ``split``, that split's phase only (the
    one task ``platoon rollout`` runs); for an evaluation config (``inference``), the phase of its ``split`` (``eval``
    unless it evaluates the ``train`` split); for a training config, ``train``, and ``eval`` too when it evaluates
    (``eval.every`` is at least 1, and the deprecated ``eval.strategy`` is not ``none``); for a config of
    environments alone (:class:`~platoon.train.components.EnvironmentsConfig`), which either kind extends,
    both."""

    from platoon.train.components import EnvironmentsConfig

    if split is not None:
        return (phase_of(split),)
    if isinstance(config, EnvironmentsConfig):
        return PHASES
    if hasattr(config, "inference"):
        return (phase_of(str(getattr(config, "split", "eval"))),)
    evaluation = getattr(config, "eval", None)
    every = getattr(evaluation, "every", 0)
    if evaluation is not None and getattr(evaluation, "strategy", None) != "none" and isinstance(every, int):
        return ("train", "eval") if every > 0 else ("train",)
    return ("train",)


def lock_record(values: Mapping[Phase, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """What ``lock.json`` records of one entry's options for one component: the options of each phase, defaults
    included, with ``eval`` left out when ``eval_overrides`` changes nothing for a config that also trains."""

    record: dict[str, dict[str, Any]] = {str(phase): dict(options) for phase, options in values.items()}
    if "train" in record and record.get("eval") == record["train"]:
        del record["eval"]
    return record


__all__ = [
    "PHASES",
    "CheckedOptions",
    "OptionSchema",
    "Phase",
    "check_harness_options",
    "config_phases",
    "harness_options",
    "lock_record",
    "option_schema",
    "phase_of",
]
