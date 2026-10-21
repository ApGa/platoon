"""Named environments: versioned, partial ``environments`` entries that configs select with ``env:``.

A named environment says once how to run one environment: its service, the options each harness needs on it,
its program and reward processor, its default splits.  An extension registers one in the
``platoon.environments`` entry-point group, under the environment's name::

    # acme_platoon/entries.py
    import platoon as pt

    MAZES = pt.EnvironmentEntry(
        "acme-mazes",
        version="1",
        description="Grid mazes: walk to the exit with move(direction).",
        service="python://acme_platoon.env:MazeEnvironment",
        harness_defaults={"platoon-react": {"system_prompt": "Find the exit."}},
        eval_dataset_kwargs={"split_name": "val"},
    )

    # pyproject.toml
    [project.entry-points."platoon.environments"]
    acme-mazes = "acme_platoon.entries:MAZES"

A config then names it in an ``environments`` entry, and its own keys merge over it::

    environments:
      - env: acme-mazes@1          # the lowest layer of this entry
        harness: platoon-react
        dataset_kwargs: {limit: 500}

**Merging.**  The named entry is the lowest layer: the config's keys (after ``extends:`` and overrides) merge over
it, mappings key by key and anything else, lists included, replacing.  ``harness_defaults`` maps harness names to
options: those of the entry's harness are the lowest layer of its ``harness_options``, so a config that picks
another harness does not inherit options it cannot read.  A harness mixture shares one ``harness_options``
mapping, so a mixture that includes a harness with defaults is a config error unless the entry sets that mapping
itself (then no defaults apply).  The loaded config keeps ``env: name@version`` with the version the name
resolved to, so ``config.resolved.yaml`` pins it.

**Training settings.**  A named environment may ship the rollout settings its tasks were measured with, such as a
calibrated step budget (``workflow_config: {rollout_config: {max_steps: 30}}``), as the lowest layer of its entry's
own, which its evaluations inherit beneath the phase's section, and the reward that counts as a pass
(``success_threshold``); the rest of an entry's ``workflow_config`` (the group size, the baseline, the minimum group
size, the straggler defaults), its ``max_staleness`` and its ``weight`` are training strategy, which a run chooses
(:data:`FORBIDDEN_KEYS`), and it ships no ``eval_overrides.workflow_config``, whose values would win over the run's
evaluation settings.

**Tasks left out.**  An entry that selects a named environment sets ``exclude_tasks`` like any entry: its items
name that environment's tasks, and they leave each list before the named environment's ``limit`` takes its share.
A named environment may not set ``exclude_tasks`` (:data:`FORBIDDEN_KEYS`): an entry's own list would replace the
environment's, not add to it, and bring the tasks it left out back unnoticed.  An environment that should not offer
a task leaves it out of its own task lists, in a new version.

**Versions.**  A change to the tasks, the grading or the defaults bumps the version; older versions stay
registered (several may be at once), so a recipe pinned to ``@1`` keeps meaning the same thing.  A bare name
resolves only while one version is registered.

**Built-ins.**  Core's own named environments (:data:`BUILTIN_ENVIRONMENTS`, such as ``platoon-number-search``)
need nothing installed: a name in that table resolves to core's entry before any entry point is looked at, so no
installed distribution can shadow it.

``env:`` also takes ``module:attr``, an ``EnvironmentEntry`` that no installed distribution registers.
"""

from __future__ import annotations

import copy
import functools
import importlib
import logging
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint
from typing import Any

from platoon import extensions
from platoon.extensions import ExtensionError, Origin
from platoon.train.components import RUN_LEVEL_FIELDS

logger = logging.getLogger(__name__)

# Keys a named environment may not set: what identifies or weighs an entry in a run (``env``, ``name``,
# ``weight``), how a run draws its tasks (``task_sampler``, ``task_sampler_options``) and which of its groups count
# (``group_admission``, ``group_admission_options``), its staleness bound (``max_staleness``, training strategy too),
# the options of one harness (``harness_options``: use ``harness_defaults``, keyed by harness), the tasks a run leaves
# out (``exclude_tasks``: an entry's own list would replace the environment's, not add to it, and bring its tasks back
# unnoticed), and the run-level settings, which are the run's to choose.  Of ``workflow_config`` it may set
# ``rollout_config`` only (:data:`SHIPPED_SETTINGS`).
FORBIDDEN_KEYS = frozenset(
    {
        "env",
        "name",
        "weight",
        "task_sampler",
        "task_sampler_options",
        "group_admission",
        "group_admission_options",
        "max_staleness",
        "harness_options",
        "exclude_tasks",
        *RUN_LEVEL_FIELDS,
    }
)
# The keys of an entry's ``workflow_config`` a named environment may ship: the rollout settings its tasks were measured
# with, which its evaluations inherit beneath the phase's own section. The rest is training strategy, which a run
# chooses, and a named environment's value would beat the run's train.workflow_config for its entry alone. It ships no
# ``eval_overrides.workflow_config``: an entry's evaluation overrides win over the phase's section, which a run's
# eval.workflow_config and an evaluation's inference.workflow must win over.
SHIPPED_SETTINGS = frozenset({"rollout_config"})


@dataclass(frozen=True, init=False)
class EnvironmentEntry:
    """A named environment: ``name@version`` and the partial ``environments`` entry it stands for.

    ``EnvironmentEntry(name, version="1", *, description="", harness_defaults=None, **keys)``, where ``keys`` are
    keys of an ``environments`` entry (``service``, ``service_kwargs``, ``harness``, ``program``,
    ``reward_processor``, ``agents``, ``dataset_kwargs``, ...), checked against the config schema when the entry
    is built.  A run chooses the rest (:data:`FORBIDDEN_KEYS`): ``env``, ``name``, ``weight``, ``task_sampler`` and
    ``task_sampler_options``, ``group_admission`` and ``group_admission_options``, ``max_staleness``, the run-level
    settings, the tasks it leaves out (``exclude_tasks``; an environment that should not offer a task leaves it out of
    its own task lists, in a new version), ``harness_options``, for which ``harness_defaults`` maps a harness name to
    the options the environment needs from that harness, of ``workflow_config`` all but the rollout settings
    (``rollout_config``) its tasks were measured with, and ``eval_overrides.workflow_config``.
    """

    # The environment's name, ``<owner>-<name>``.
    name: str
    # Bumped when the tasks, the grading or the defaults change.
    version: str = "1"
    # One line saying what the environment is, for listings.
    description: str = ""
    # Harness name -> the lowest layer of that harness's ``harness_options`` on this environment.
    harness_defaults: Mapping[str, Mapping[str, Any]] = field(default_factory=dict, repr=False)
    # The partial environments entry: the keyword arguments other than the fields above.
    keys: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __init__(
        self,
        name: str,
        version: str = "1",
        *,
        description: str = "",
        harness_defaults: Mapping[str, Mapping[str, Any]] | None = None,
        **keys: Any,
    ) -> None:
        extensions.check_name(name, what="environment")
        version = extensions.check_version(version, what=f"environment {name!r}")
        what = f"environment {name}@{version}"
        forbidden = sorted(FORBIDDEN_KEYS & set(keys))
        if forbidden:
            raise ValueError(
                f"{what} sets {forbidden}, which a named environment may not set: a run chooses its entry's name, "
                "weight, task sampler, group admission rule, run-level settings and exclude_tasks (leave a task out of "
                "the environment's own task lists instead), and harness options go in harness_defaults, keyed by "
                "harness"
            )
        problems = _schema_problems(keys)
        if problems:
            raise ValueError(f"{what}: " + "; ".join(problems))
        overrides = keys.get("eval_overrides")
        if isinstance(overrides, Mapping) and "harness_options" in overrides:
            raise ValueError(
                f"{what} sets eval_overrides.harness_options, which a named environment may not set: harness options "
                "go in harness_defaults, keyed by harness, and the entry that selects the environment sets its own "
                "eval_overrides"
            )
        settings = keys.get("workflow_config")
        if isinstance(settings, Mapping) and (strategy := sorted(set(settings) - SHIPPED_SETTINGS)):
            raise ValueError(
                f"{what} sets workflow_config {strategy}, which a named environment may not set: they are training "
                "strategy, which a run chooses; a named environment ships the rollout settings its tasks were measured "
                "with only (workflow_config.rollout_config)"
            )
        if isinstance(overrides, Mapping) and "workflow_config" in overrides:
            raise ValueError(
                f"{what} sets eval_overrides.workflow_config, which a named environment may not set: an entry's "
                "evaluation overrides win over the run's eval.workflow_config and an evaluation's inference.workflow; "
                "ship the rollout settings its tasks were measured with in workflow_config.rollout_config, which its "
                "evaluations inherit beneath them"
            )
        defaults: dict[str, dict[str, Any]] = {}
        for harness, options in dict(harness_defaults or {}).items():
            extensions.check_name(harness, what=f"{what}: harness_defaults key (a harness name)")
            if not isinstance(options, Mapping):
                raise ValueError(f"{what}: harness_defaults[{harness!r}] must be a mapping of options")
            defaults[harness] = copy.deepcopy(dict(options))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "description", str(description))
        object.__setattr__(self, "harness_defaults", defaults)
        object.__setattr__(self, "keys", copy.deepcopy(dict(keys)))

    @property
    def ref(self) -> str:
        """``name@version``, what a config pins with ``env:``."""
        return f"{self.name}@{self.version}"

    def entry(self) -> dict[str, Any]:
        """The partial environments entry, as a fresh copy."""
        return copy.deepcopy(dict(self.keys))

    def defaults_for(self, harness: Any) -> dict[str, Any]:
        """The ``harness_defaults`` of the entry's ``harness`` (a reference, or a ``{harness: weight}`` mixture), as a
        fresh copy; empty when it has none.  A ``.toml`` path or ``module:function`` reference matches no name.

        A mixture of several harnesses shares one ``harness_options`` mapping, so a mixture that includes a harness
        with defaults raises ``ValueError``.
        """

        if not self.harness_defaults or harness is None:
            return {}
        refs = [harness] if isinstance(harness, str) else list(harness) if isinstance(harness, Mapping) else []
        found = {
            ref: self.harness_defaults[name]
            for ref in refs
            if isinstance(ref, str)
            and not ref.endswith(".toml")
            and ":" not in ref
            and (name := extensions.split_ref(ref)[0]) in self.harness_defaults
        }
        if not found:
            return {}
        if len(refs) > 1:
            raise ValueError(
                f"named environment {self.ref} gives harness_defaults for {sorted(found)}, but this entry's harness "
                f"is a mixture of {sorted(refs)}, which shares one harness_options mapping; give each harness an "
                "environments entry of its own, or set the mixture's harness_options in the entry"
            )
        return copy.deepcopy(dict(next(iter(found.values()))))


def _schema_problems(keys: Mapping[str, Any]) -> list[str]:
    from platoon.train.components import EnvironmentConfig
    from platoon.utils.config import validate_config

    return validate_config(dict(keys), EnvironmentConfig)


# --- the registry -------------------------------------------------------------------------------------------------

# Core's named environments: name -> where its ``EnvironmentEntry`` (or a list of its versions) is defined.  Imported
# the first time a config or a listing names one.
BUILTIN_ENVIRONMENTS: dict[str, str] = {
    "platoon-number-search": "platoon.envs.number_search.entries:NUMBER_SEARCH",
}

_lock = threading.Lock()
# ref -> (entry, where it is defined: ``module:attr``, or None)
_entries: dict[str, tuple[EnvironmentEntry, str | None]] = {}


def register_environment(entry: EnvironmentEntry, *, source: str | None = None, origin: Origin | None = None) -> None:
    """Add ``entry`` to the process registry; ``source`` says where it is defined (``module:attr``).

    Registering the same entry again is allowed; another entry under a registered ``name@version`` is an error
    that names both (pick one with ``env: module:attr``).
    """

    if not isinstance(entry, EnvironmentEntry):
        raise TypeError(f"a named environment must be a pt.EnvironmentEntry, got {entry!r}")
    with _lock:
        existing = _entries.get(entry.ref)
        if existing is not None and existing[0] is not entry and existing[0] != entry:
            raise ExtensionError(
                f"environment {entry.ref} is defined twice: {existing[1] or '?'} and {source or '?'}; name the one "
                "to use as env: module:attr"
            )
        _entries[entry.ref] = (entry, source if existing is None else existing[1] or source)
    if origin is not None:
        extensions.record_origin("environment", entry.ref, origin)


def accept_entry_point(value: Any, origin: Origin, entry_point: EntryPoint) -> None:
    """Register what a ``platoon.environments`` entry point loads to: an ``EnvironmentEntry`` named as the entry
    point is, or a list of them (several versions of one name)."""

    entries = [value] if isinstance(value, EnvironmentEntry) else value
    if isinstance(entries, (str, bytes, Mapping)) or not isinstance(entries, Iterable):
        raise TypeError(f"a platoon.environments entry point must load to a pt.EnvironmentEntry, got {value!r}")
    entries = list(entries)
    if not entries:
        raise TypeError("a platoon.environments entry point must load to at least one pt.EnvironmentEntry")
    for entry in entries:
        if not isinstance(entry, EnvironmentEntry):
            raise TypeError(f"a platoon.environments entry point must load to a pt.EnvironmentEntry, got {entry!r}")
        if entry.name != entry_point.name:
            raise ValueError(
                f"it loads to environment {entry.ref}, whose name is not the entry point's {entry_point.name!r}; "
                "name the entry point after the environment"
            )
    for entry in entries:
        register_environment(entry, source=entry_point.value, origin=origin)


def builtin_environments() -> list[tuple[str, str]]:
    """Core's named environments (:data:`BUILTIN_ENVIRONMENTS`), registered: ``(name@version, module:attr)`` each."""

    found = []
    for name in sorted(BUILTIN_ENVIRONMENTS):
        found += [(entry.ref, BUILTIN_ENVIRONMENTS[name]) for entry in _register_builtin(name)]
    return found


def _register_builtin(name: str) -> list[EnvironmentEntry]:
    """Import core's named environment ``name`` from where :data:`BUILTIN_ENVIRONMENTS` says, and register it (again,
    after a test restored the registry: registering the same entry twice is allowed)."""

    source = BUILTIN_ENVIRONMENTS[name]
    module_name, _, attribute = source.partition(":")
    value: Any = functools.reduce(getattr, attribute.split("."), importlib.import_module(module_name))
    entries = list(value) if isinstance(value, (list, tuple)) else [value]
    origin = Origin(distribution=extensions.RESERVED_OWNER, version=extensions.platoon_version(), module=module_name)
    for entry in entries:
        if not isinstance(entry, EnvironmentEntry) or entry.name != name:
            raise TypeError(f"{source} must be the pt.EnvironmentEntry {name!r}, or a list of its versions")
        register_environment(entry, source=source, origin=origin)
    return entries


def get_environment(ref: str) -> EnvironmentEntry:
    """The named environment ``ref`` names: ``name[@version]``, or ``module:attr`` (an ``EnvironmentEntry``).

    A name of core's (:data:`BUILTIN_ENVIRONMENTS`) resolves to core's entry, and no entry point is looked at.
    Another name loads the ``platoon.environments`` entry point of that exact name, once per process (after its
    compatibility check).  A bare name must have a single registered version.  Raises
    :class:`~platoon.extensions.ExtensionError` (a ``ValueError``) when nothing or more than one entry matches.
    """

    if not isinstance(ref, str) or not ref:
        raise ExtensionError(f"env must name an environment, name[@version] or module:attr, got {ref!r}")
    if ":" in ref:
        return _import_entry(ref)
    name, _ = extensions.split_ref(ref)
    if name in BUILTIN_ENVIRONMENTS:
        _register_builtin(name)
    else:
        extensions.load_named("environment", name)
    with _lock:
        candidates = [entry for entry, _ in _entries.values()]
    matches = extensions.matching(ref, candidates)
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise extensions.ambiguous("environment", ref, [entry.ref for entry in matches])
    known = [entry.ref for entry in candidates]
    known += [name for name in BUILTIN_ENVIRONMENTS if all(entry.name != name for entry in candidates)]
    raise extensions.unknown("environment", ref, known)


def _import_entry(ref: str) -> EnvironmentEntry:
    module_name, _, attribute = ref.partition(":")
    try:
        value: Any = importlib.import_module(module_name)
        value = functools.reduce(getattr, attribute.split("."), value)
    except (ImportError, AttributeError) as exc:
        raise ExtensionError(f"env {ref!r} does not import: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, EnvironmentEntry):
        raise ExtensionError(f"env {ref!r} is not a pt.EnvironmentEntry, but {value!r}")
    try:
        extensions.check_registration("environment", value.name, module=module_name, what=f"env {ref!r}: environment")
    except ValueError as exc:  # a name of core's, from outside core
        raise ExtensionError(str(exc)) from None
    register_environment(value, source=ref, origin=extensions.origin_of_module(module_name))
    return value


def environment_source(ref: str) -> str | None:
    """Where the registered environment ``name@version`` is defined (``module:attr``), if known."""

    with _lock:
        found = _entries.get(ref)
    return None if found is None else found[1]


# --- expanding configs ----------------------------------------------------------------------------------------------


def expand_environment(entry: Mapping[str, Any]) -> dict[str, Any]:
    """An ``environments`` entry (plain data) with its named environment (``env:``) laid under its own keys.

    The result keeps ``env``, with the version it resolved to (``name@version``; an import string stays as it
    is), so expanding it again changes nothing.  An entry without ``env`` comes back as a copy.
    """

    if entry.get("env") is None:
        return dict(entry)
    ref = entry["env"]
    named = get_environment(ref)
    own = {key: value for key, value in entry.items() if key != "env"}
    from platoon.utils.config import deep_merge

    merged = deep_merge(named.entry(), own)
    harness = merged.get("harness")
    if isinstance(harness, Mapping) and len(harness) > 1 and "harness_options" in own:
        defaults: dict[str, Any] = {}  # the entry sets its mixture's shared options itself
    else:
        defaults = named.defaults_for(harness)
    if defaults:
        options = merged.get("harness_options")
        merged["harness_options"] = deep_merge(defaults, options if isinstance(options, Mapping) else {})
    return {"env": ref if ":" in ref else named.ref, **merged}


def expand_environments(data: Mapping[str, Any]) -> dict[str, Any]:
    """A whole config (plain data) with every ``environments`` entry that names an ``env`` expanded
    (:func:`expand_environment`).  Every problem raises one :class:`~platoon.utils.config.ConfigError`."""

    from platoon.utils.config import ConfigError

    environments = data.get("environments")
    if not isinstance(environments, list):
        return dict(data)
    expanded: list[Any] = []
    problems: list[str] = []
    for index, entry in enumerate(environments):
        if not isinstance(entry, Mapping) or entry.get("env") is None:
            expanded.append(entry)
            continue
        try:
            expanded.append(expand_environment(entry))
        except (ValueError, TypeError) as exc:  # ExtensionError is a ValueError
            problems.append(f"environments.{index}.env: {exc}")
            expanded.append(entry)
    if problems:
        raise ConfigError(problems)
    return {**data, "environments": expanded}


def names_environments(data: Mapping[str, Any]) -> bool:
    """Whether a config (plain data) has an ``environments`` entry with an ``env`` to expand."""

    environments = data.get("environments")
    return isinstance(environments, list) and any(
        isinstance(entry, Mapping) and entry.get("env") is not None for entry in environments
    )


__all__ = [
    "BUILTIN_ENVIRONMENTS",
    "EnvironmentEntry",
    "FORBIDDEN_KEYS",
    "SHIPPED_SETTINGS",
    "accept_entry_point",
    "builtin_environments",
    "environment_source",
    "expand_environment",
    "expand_environments",
    "get_environment",
    "names_environments",
    "register_environment",
]
