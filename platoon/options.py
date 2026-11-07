"""Component options (RFC 0016): the path that checks the options a config sets for the components its
``environments`` entries select, per entry and phase, before anything runs, and records them with their defaults.

A component that declares options has a schema (:class:`OptionSchema`): a TOML harness spec through its
``[options]`` table (:class:`platoon.harness.spec.OptionSpec`).  Function harnesses, named environments and reward
processors take the same path once they declare options (RFC 0016, M2), through :func:`option_schema`; until then
their options go unchecked, as before.  Task samplers are the first to declare options in Python (RFC 0019, section
5): :func:`schema_options` takes a sampler's ``options_schema``, a dataclass, a pydantic model or a parser of the
options' mapping, and checks ``train.task_sampler_options`` and an entry's ``task_sampler_options`` against it, core's
samplers' when the config loads and the others' where components resolve.  Episode programs declare theirs the same
way (``@pt.program(name, options=..., check=...)``), and an entry sets them in ``program_options``
(:func:`check_program_options`); a program's ``check`` sees the entry, the phase and the phase's rollout config
(:class:`CheckContext`).

**Phases.**  ``train`` rollouts run with an entry's own options.  Evaluation rollouts, those of every split but
``train`` (the rollouts of ``platoon eval``, and those ``platoon train`` runs when it evaluates), run with the
entry's ``eval_overrides`` merged over them, key by key (:func:`harness_options`, :func:`program_options`).  The
routes that carry options to rollout workers are built per phase (``platoon.train.auto.TaskRouter``), so a worker
never has to know the phase.

**Where it runs.** ``platoon train``, ``eval`` and ``rollout``, and ``platoon config validate`` (unless
``--no-resolve``), check every entry's options when they resolve the config's components
(:func:`platoon.extensions.run_components`), run each program's check, and ``lock.json`` records the options each
harness and each program ran with, defaults included, per entry and phase.  A program's check reads the phase's
rollout config, which a training or an evaluation config gives: a config of environments alone, such as the
environments ``platoon prepare`` loads of a config, has its options checked but runs no check.  The trainer and the
evaluator check the options again when they build the routes, and a rollout worker parses them once more when it
renders a launch or runs the program, so a schema is the one place that says what an option means.
"""

from __future__ import annotations

import copy
import enum
import inspect
import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from platoon.config_defs import RolloutConfig
    from platoon.programs import RegisteredProgram
    from platoon.train.components import EnvironmentConfig

logger = logging.getLogger(__name__)

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


class SchemaOptions:
    """The :class:`OptionSchema` of a component whose options are a class or a parser of its own, such as a task
    sampler's (:func:`schema_options`): ``schema`` is a dataclass, a pydantic model, a function that takes the options'
    mapping, returns them parsed and raises ``ValueError``, or None for a component that takes no options (``name``
    says which in that problem)."""

    def __init__(self, schema: Any, *, name: str = "it") -> None:
        self.schema = schema
        self.name = name

    def option_problems(self, values: Any, where: str) -> list[str]:
        return self.check(values, where)[1]

    def resolve_options(self, values: Mapping[str, Any] | None, where: str) -> dict[str, Any]:
        """The options as JSON data, defaults included: a dataclass's fields, a pydantic model's ``model_dump``, or,
        for a parser's result, the options as written."""

        parsed, problems = self.check(values, where)
        if problems:
            raise ValueError("; ".join(problems))
        return self.as_json(parsed, values)

    def parse(self, values: Mapping[str, Any] | None, where: str) -> Any:
        """The options parsed (an instance of the schema, the parser's result, or None without a schema); a
        :class:`~platoon.utils.config.ConfigError` (a ``ValueError``) with every problem, one line each."""

        parsed, problems = self.check(values, where)
        if problems:
            from platoon.utils.config import ConfigError

            raise ConfigError(problems)
        return parsed

    def as_json(self, parsed: Any, values: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """What ``lock.json`` and the sampling fingerprint record of options parsed by this schema."""

        import dataclasses

        if parsed is None:
            return {}
        if dataclasses.is_dataclass(parsed) and not isinstance(parsed, type):
            return _plain(dataclasses.asdict(parsed))
        if hasattr(parsed, "model_dump"):
            return _plain(parsed.model_dump(mode="json"))
        return _plain(dict(values or {}))

    def check(self, values: Any, where: str) -> tuple[Any, list[str]]:
        """``values`` parsed, and every problem with them under ``where``, one line each (then the first is None)."""

        import dataclasses

        from platoon.utils.config import (
            ConfigError,
            _free_form_model,
            _is_pydantic,
            build_config,
            normalize_config,
            unknown_key,
            validate_config,
        )

        values = {} if values is None else values
        if not isinstance(values, Mapping):
            return None, [f"{where}: expected a mapping of options, got {values!r}"]
        schema = self.schema
        if schema is None:
            if values:
                return None, [f"{where}: {self.name} takes no options; leave {where.rsplit('.', 1)[-1]} out"]
            return None, []
        if isinstance(schema, type) and dataclasses.is_dataclass(schema):
            values = normalize_config(dict(values), schema)
            if problems := validate_config(values, schema, path=where):
                return None, problems
            fields = {item.name for item in dataclasses.fields(schema)}
            try:
                return build_config(schema, values), []
            except ConfigError as exc:
                return None, [_field_line(where, problem, fields) for problem in exc.problems]
            except (TypeError, ValueError) as exc:
                return None, [_field_line(where, str(exc), fields)]
        if _is_pydantic(schema):
            # A key the model does not take is refused as a config section's is, with a did-you-mean, unless the
            # model keeps such keys (extra="allow"): by pydantic's default it would vanish without a word.
            names = _pydantic_names(schema)
            free_form = _free_form_model(schema)
            unknown = [] if free_form else [unknown_key(key, names, where) for key in values if key not in names]
            given = {key: item for key, item in values.items() if free_form or key in names}
            try:
                parsed = schema.model_validate(given)
            except Exception as exc:  # pydantic's ValidationError, each error with its location
                errors = getattr(exc, "errors", None)
                if not callable(errors):
                    return None, [*unknown, f"{where}: {exc}"]
                return None, [*unknown, *(_pydantic_line(schema, where, error) for error in errors())]
            return (None, unknown) if unknown else (parsed, [])
        if callable(schema):
            try:
                return schema(dict(values)), []
            except (TypeError, ValueError) as exc:
                return None, [f"{where}: {exc}"]
        return None, [f"{where}: {self.name}'s options_schema is no schema (a dataclass, a pydantic model or a parser)"]


def schema_options(schema: Any, *, name: str = "it") -> SchemaOptions:
    """The option schema (:class:`OptionSchema`) of options whose ``schema`` is a dataclass, a pydantic model, a
    parser (a function of the options' mapping that returns them parsed and raises ``ValueError``), or None for a
    component without options, such as a task sampler's (RFC 0016's path, which task samplers are the first to take
    beside harnesses).  A dataclass's options are checked as config sections are (each unknown key with a
    did-you-mean, each wrong type with its path), and a ``ValueError`` its ``__post_init__`` raises, or each problem
    of a :class:`~platoon.utils.config.ConfigError` it raises, becomes a line under the field it names (a message
    that begins with a field's name and a colon) or under the options; a pydantic model's errors each name their
    location, and a key of the options that a model does not take (neither a field's name nor its alias) is refused
    as a config section's unknown key is, with a did-you-mean, unless the model keeps such keys (``extra="allow"``).
    Nested models keep pydantic's own rules."""

    return SchemaOptions(schema, name=name)


def _field_line(where: str, message: str, fields: set[str]) -> str:
    """A problem a dataclass's checks raised, under ``where``: under the field it names, when it begins with one of
    ``fields`` and a colon (``explore: must be between 0 and 1``)."""

    field_name, colon, rest = message.partition(":")
    if colon and field_name in fields:
        return f"{where}.{field_name}:{rest}"
    return f"{where}: {message}"


def _pydantic_names(model: Any) -> list[str]:
    """The keys a pydantic model takes: its fields' names and aliases."""

    return [alias for name, info in model.model_fields.items() for alias in (name, info.alias) if alias]


def _pydantic_line(model: Any, where: str, error: Mapping[str, Any]) -> str:
    """One of a pydantic model's validation errors under ``where``, at its location: a key of the options that a model
    which forbids extra keys does not take as a config section's unknown key, with a did-you-mean among its fields."""

    from platoon.utils.config import unknown_key

    location = tuple(map(str, error.get("loc", ())))
    if error.get("type") == "extra_forbidden" and len(location) == 1:
        return unknown_key(location[0], _pydantic_names(model), where)
    return f"{'.'.join([where, *location])}: {error.get('msg', 'invalid')}"


def _plain(value: Any) -> Any:
    """``value`` as JSON data: mappings with string keys, lists for tuples, an enum member as its value and a path as
    its text, as a config writes them."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, enum.Enum):
        return _plain(value.value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


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


def program_options(environment: EnvironmentConfig, phase: Phase) -> dict[str, Any]:
    """The ``program_options`` an entry's rollouts of ``phase`` run with: its own, and for ``eval`` its
    ``eval_overrides.program_options`` merged over them, key by key."""

    from platoon.utils.config import deep_merge

    own = copy.deepcopy(dict(environment.program_options))
    if phase == "train":
        return own
    return deep_merge(own, environment.eval_overrides.program_options)


@dataclass(frozen=True)
class CheckContext:
    """What an episode program's load-time check sees beside its options: the entry, the phase and its rollout config.

    A program's ``check`` (``@pt.program(name, options=..., check=...)``, RFC 0016, section 3) is called when a
    training or an evaluation config loads, before anything runs, for each entry that runs the program and once for
    each phase the command runs (``platoon train`` runs ``eval`` too when it evaluates), with that phase's options
    parsed; it refuses the entry by raising ``ValueError``.  A config of environments alone, a file that such configs
    extend, gives no run's rollout config, so its checks wait for the configs that run it; so does ``platoon prepare``,
    which loads only a config's environments.  A check reads the config only: whether a server answers, or a variable
    is set where the run executes, is for the program to find out when it runs.  Each field is a copy, so a check that
    changes one changes nothing."""

    # The environments entry, as the config loaded it: its named environment laid under its own keys, and the command
    # line's overrides applied.
    environment: EnvironmentConfig
    # The phase whose options the check got: ``train``, the train split's rollouts, run with the entry's own options;
    # ``eval``, every other split's, with its ``eval_overrides`` merged over them.
    phase: Phase
    # The rollout config the entry's rollouts of the phase run with: in a training config, the run's (its
    # ``train.workflow_config.rollout_config``, or for ``eval`` its ``eval.workflow_config.rollout_config``) with the
    # entry's own rollout settings laid over it; in an evaluation config, ``inference.workflow.rollout_config`` with the
    # entry's.
    rollout: RolloutConfig


def check_program_options(
    environment: EnvironmentConfig,
    program: RegisteredProgram,
    *,
    where: str,
    phases: Sequence[Phase] = PHASES,
    rollouts: Mapping[Phase, RolloutConfig] | None = None,
) -> CheckedOptions:
    """Check an entry's options for its episode program against the program's schema
    (``@pt.program(name, options=...)``), and with ``rollouts``, the entry's rollout config of each of ``phases``, run
    the program's ``check`` for each (:class:`CheckContext`).

    ``where`` is the entry's dotted path (``environments.0``).  The entry's own ``program_options`` are checked where
    they are written, and so are its ``eval_overrides.program_options``, merged over them as evaluation rollouts run
    them, whatever ``phases`` says, so a mistake in either is found before any rollout of either phase.  A program
    that declares no options takes none.  The values are each phase's options as JSON data, defaults included (empty
    for a program without a schema; for a parser, the mapping it returned, else the options as written), and the
    program's check runs on each phase's options parsed: a ``ValueError`` it raises refuses the config with its
    message, under ``<where>.program``, naming the program and the phases it refused; any other exception is reported
    the same way, with its type, and so is a check that returns an awaitable (an ``async def``), which never ran.
    Without ``rollouts`` the check does not run, as when the trainer checks the options again where it builds a route.
    """

    schema = schema_options(program.options_schema, name=f"program {program.ref}")
    own = f"{where}.program_options"
    problems = schema.option_problems(environment.program_options, own)
    if environment.eval_overrides.program_options:
        written = f"{where}.eval_overrides.program_options"
        merged = schema.option_problems(program_options(environment, "eval"), written)
        problems += [line for line in merged if line.replace(written, own) not in problems]  # not the entry's own
    if problems:
        return CheckedOptions(problems=problems)
    parsed = {phase: schema.parse(program_options(environment, phase), own) for phase in phases}
    values = {phase: _recorded(schema, found, program_options(environment, phase)) for phase, found in parsed.items()}
    if rollouts is None or program.check is None:
        return CheckedOptions(values=values)
    refused: dict[str, list[Phase]] = {}  # each refusal, with the phases whose options it refused
    for phase, options in parsed.items():
        ctx = CheckContext(environment=copy.deepcopy(environment), phase=phase, rollout=copy.deepcopy(rollouts[phase]))
        try:
            returned = program.check(options, ctx)
        except ValueError as exc:
            refused.setdefault(str(exc) or type(exc).__name__, []).append(phase)
        except Exception as exc:  # a check that fails another way still refuses, and says how
            refused.setdefault(f"its check raised {type(exc).__name__}: {exc}", []).append(phase)
        else:
            if inspect.isawaitable(returned):  # an async check: what it would refuse is never known
                close = getattr(returned, "close", None)
                if callable(close):
                    close()  # a coroutine that never runs, closed so that it warns of nothing
                refused.setdefault(
                    "its check is async, so it never ran: a check is a plain function check(options, ctx)", []
                ).append(phase)
    return CheckedOptions(
        values=values,
        problems=[
            f"{where}.program: {program.ref}, for the entry's {' and '.join(refusing)} rollouts: {message}"
            for message, refusing in refused.items()
        ],
    )


def _recorded(schema: SchemaOptions, parsed: Any, written: Mapping[str, Any]) -> dict[str, Any]:
    """What ``lock.json`` records of a program's options, parsed by ``schema`` from ``written``: as JSON data, defaults
    included, so for a parser that returns a mapping, that mapping, with the defaults it filled in.  (``as_json``
    gives a parser's options as written, which task samplers' fingerprints keep reading.)  Options parsed into values
    that JSON cannot hold are recorded as written, with a warning, so that they never stop the run's record."""

    recorded = _plain(dict(parsed)) if isinstance(parsed, Mapping) else schema.as_json(parsed, written)
    try:
        json.dumps(recorded)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "%s's options parse into values that lock.json cannot hold (%s): it records them as written",
            schema.name,
            exc,
        )
        return _plain(dict(written))
    return recorded


def config_phases(config: Any, split: str | None = None) -> tuple[Phase, ...]:
    """The phases whose rollouts a command runs with a loaded config: with ``split``, that split's phase only (the
    one task ``platoon rollout`` runs); for an evaluation config (``inference``), the phase of its ``split`` (``eval``
    unless it evaluates the ``train`` split); for a training config, ``train``, and ``eval`` too when it evaluates
    (``eval.every`` is at least 1); for a config of environments alone
    (:class:`~platoon.train.components.EnvironmentsConfig`), which either kind extends, both."""

    from platoon.train.components import EnvironmentsConfig

    if split is not None:
        return (phase_of(split),)
    if isinstance(config, EnvironmentsConfig):
        return PHASES
    if hasattr(config, "inference"):
        return (phase_of(str(getattr(config, "split", "eval"))),)
    every = getattr(getattr(config, "eval", None), "every", 0)
    return ("train", "eval") if isinstance(every, int) and every > 0 else ("train",)


def lock_record(values: Mapping[Phase, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """What ``lock.json`` records of one entry's options for one component: the options of each phase, defaults
    included, with ``eval`` left out when ``eval_overrides`` changes nothing for a config that also trains."""

    record: dict[str, dict[str, Any]] = {str(phase): dict(options) for phase, options in values.items()}
    if "train" in record and record.get("eval") == record["train"]:
        del record["eval"]
    return record


__all__ = [
    "PHASES",
    "CheckContext",
    "CheckedOptions",
    "OptionSchema",
    "Phase",
    "SchemaOptions",
    "check_harness_options",
    "check_program_options",
    "config_phases",
    "harness_options",
    "lock_record",
    "option_schema",
    "phase_of",
    "program_options",
    "schema_options",
]
