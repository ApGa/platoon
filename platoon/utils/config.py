"""Configuration loading: YAML files with ``extends:``, typed ``key=value`` overrides and a strict schema.

A config file may build on others with ``extends:`` (one path or a list, relative to the file that names
them).  Bases load first, in order, then the file merges over them, and bases may extend further files;
a cycle is an error.  Mappings deep-merge; anything else, lists included, replaces.

The input paths a file names are relative to that file too (:func:`resolve_input_paths`): an environment's
harness ``.toml`` spec and the options its service declares as naming files (``dataset_dir``, ``overlays``, a
table's ``path``, ...), made absolute per file before the files merge.  One that names nothing there is a
configuration error, which names the path relative to the working directory when that one exists.  An override's
paths are relative to the working directory, and so are the outputs, ``log_path`` and ``inference.output_dir``.
The loaded config holds the input paths absolute, and a run's record (:func:`write_run_record`) the outputs too, so
the record loads from any directory.

Overrides are one more layer on top.  ``a.b=value`` sets a key, ``a.0.b=value`` reaches into a list
(negative indices count from the end) and ``a.b+=value`` appends one item to a list.  Each value is parsed
against the type of the schema field it targets (a dataclass or pydantic model), so ``stats.trial_name=1``
stays the string ``"1"`` while ``train.batch_size=1`` is the integer 1; lists, mappings and nested models
are written as YAML.

The merged config is checked against the schema before anything is built: an unknown key fails with its
full dotted path and a did-you-mean suggestion, a value of the wrong type fails, and every problem is
reported at once.  Fields typed as free-form mappings (``dict[str, Any]``: ``extra``, ``service_kwargs``,
``harness_options``, ...) accept any keys.  A nested section given in part keeps its field's default for the keys
it leaves out (:func:`build_config`), while a free-form mapping replaces its default whole.  A training config's
``eval.workflow_config`` keeps ``train.workflow_config``'s rollout and worker settings instead (``LAID_OVER``).

An ``environments`` entry that names a named environment (``env: name[@version]``) gets that environment's
partial entry as its lowest layer, after ``extends:`` and the overrides (:mod:`platoon.envs.named`); the
loaded config keeps ``env`` with the version it resolved to.

A key that a config class names as deprecated (its ``DEPRECATED_KEYS``, :mod:`platoon.deprecations`) still loads;
the loader finds each one the config sets, with its dotted path (:func:`find_deprecations`), keeps them on the
:class:`ResolvedConfig` and warns of them as :class:`~platoon.deprecations.PlatoonDeprecationWarning`.

Each run records what it ran: :func:`write_run_record` writes ``config.resolved.yaml`` (the merged config
after overrides, which is also what the run loads) and ``lock.json`` (package versions and the commits of those
installed from git, the platoon git SHA, host, start time and argv, the deprecated keys the config sets,
the images the job pulled when ``$PLATOON_IMAGE_PULLS`` names their record (:func:`image_pulls`), and, when
the run resolved them, the components it names with the distributions and versions that provide them) into the
run directory, and :func:`update_run_lock` adds what the run learns once it starts (the tasks ``exclude_tasks``
left out).
"""

from __future__ import annotations

import collections.abc
import copy
import dataclasses
import difflib
import enum
import importlib
import importlib.metadata
import json
import logging
import os
import re
import socket
import subprocess
import sys
import types
import typing
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeGuard, TypeVar, Union

import yaml

from platoon.deprecations import DEPRECATED_OPTIONS, DeprecatedKey, Deprecation, deprecated_keys
from platoon.deprecations import warn as warn_deprecations

if TYPE_CHECKING:
    from platoon.site import SiteProfile, SiteUse

logger = logging.getLogger(__name__)

T = TypeVar("T")

EXTENDS_KEY = "extends"
RESOLVED_CONFIG_NAME = "config.resolved.yaml"
LOCK_NAME = "lock.json"
# Distributions whose versions ``lock.json`` records (import names; metadata lookup normalizes them), and their commits
# when they were installed from git: the commits of hpc-sandbox between two releases share the earlier one's version.
LOCKED_DISTRIBUTIONS = ("platoon", "tinker", "tinker_cookbook", "openreward", "hpc_sandbox")
# Names a file of the JSON lines `python -m hpc_sandbox pull` printed for the job, which lock.json records
# (:func:`image_pulls`), and the keys hpc-sandbox keeps stable in each line.
IMAGE_PULLS_ENV = "PLATOON_IMAGE_PULLS"
IMAGE_PULL_KEYS = ("reference", "path", "sha256", "manifest_digest")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_MANIFEST_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

_NULLS = frozenset({"null", "Null", "NULL", "none", "None", "~"})
_TRUE = frozenset({"true", "yes", "on", "1"})
_FALSE = frozenset({"false", "no", "off", "0"})


class ConfigError(ValueError):
    """A config that cannot load; ``problems`` lists everything wrong with it, one entry each."""

    def __init__(self, problems: str | Sequence[str], *, source: str | os.PathLike[str] | None = None) -> None:
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        self.source = None if source is None else str(source)
        header = f"invalid config {self.source}" if self.source else "invalid config"
        super().__init__("\n  ".join([f"{header}:", *self.problems]))


# --- files and extends -------------------------------------------------------------------------------------


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """``override`` over ``base``: mappings merge key by key, anything else (lists included) replaces."""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml_config(config_path: str | os.PathLike[str]) -> dict[str, Any]:
    """One config file with its ``extends:`` chain merged in (the ``extends`` key itself is consumed)."""

    return _load_with_bases(Path(config_path), ())[0]


def config_sources(config_path: str | os.PathLike[str]) -> list[Path]:
    """The files a config is built from, in merge order: the deepest base first, the file itself last."""

    return _load_with_bases(Path(config_path), ())[1]


def _load_with_bases(path: Path, chain: tuple[Path, ...]) -> tuple[dict[str, Any], list[Path]]:
    path = path.expanduser().resolve()
    if path in chain:
        cycle = " -> ".join(str(item) for item in (*chain[chain.index(path) :], path))
        raise ConfigError(f"extends cycle: {cycle}")
    if not path.is_file():
        if chain:
            raise ConfigError(f"{chain[-1]}: extends {path}, which does not exist")
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    data = {} if data is None else data
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: a config file must hold a mapping at the top level, not {type(data).__name__}")
    bases = data.pop(EXTENDS_KEY, None)
    if bases is None:
        bases = []
    elif isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or not all(isinstance(base, str) and base for base in bases):
        raise ConfigError(f"{path}: extends must be a path or a list of paths, got {bases!r}")

    if problems := resolve_input_paths(data, path.parent, source=path):  # before the files merge
        raise ConfigError(problems, source=path)

    merged: dict[str, Any] = {}
    sources: list[Path] = []
    for base in bases:
        base_data, base_sources = _load_with_bases(path.parent / Path(base).expanduser(), (*chain, path))
        merged = deep_merge(merged, base_data)
        sources.extend(source for source in base_sources if source not in sources)
    return deep_merge(merged, data), [*sources, path]


# --- input paths ---------------------------------------------------------------------------------------------

# A value that names a URL rather than a local path: ``scheme://...`` (``hf://``, ``harbor://``, ``git+https://``).
_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def resolve_input_paths(data: Any, base: str | os.PathLike[str], *, source: Path | None = None) -> list[str]:
    """Make the relative input paths of ``data``'s ``environments`` entries absolute against the directory ``base``,
    in place, and return the problems: a harness ``.toml`` spec path, and the options each entry's service declares
    as naming files (``path_options``, :class:`platoon.envs.service.PathOption`).

    With ``source``, the config file ``data`` was read from (``base`` is its directory), each such path must name
    something there, unless the run writes it (``PathOption.written``): a problem names each one that does not, with
    the path relative to the working directory when that one exists, which is what such a path named before.  Without
    a ``source`` (overrides, a named environment's entry: relative to the working directory) nothing is checked; the
    service reports a file it cannot read.  Absolute paths, ``~`` expanded, and values that are not local paths are
    left as they are.
    """

    environments = data.get("environments") if isinstance(data, Mapping) else None
    if not isinstance(environments, list):
        return []
    from platoon.envs.service import PathOption, service_path_options

    resolver = _InputPaths(Path(base), source)
    for index, entry in enumerate(environments):
        if not isinstance(entry, dict):
            continue
        where = f"environments.{index}"
        harness = entry.get("harness")
        if isinstance(harness, str):
            entry["harness"] = resolver.harness(harness, f"{where}.harness")
        elif isinstance(harness, dict):
            entry["harness"] = {
                resolver.harness(ref, f"{where}.harness") if isinstance(ref, str) else ref: weight
                for ref, weight in harness.items()
            }
        options = entry.get("service_kwargs")
        service = _entry_service(entry)
        if not isinstance(options, dict) or service is None:
            continue
        rules: Mapping[str, PathOption] = service_path_options(service)
        for option, rule in rules.items():
            name, _, key = option.partition("[].")
            value = options.get(name)
            if value is None or (rule.relative_to is not None and options.get(rule.relative_to) is not None):
                continue
            at = f"{where}.service_kwargs.{name}"
            if key:  # a key of each mapping in a list
                for item_index, item in enumerate(value if isinstance(value, list) else ()):
                    if isinstance(item, dict) and key in item:
                        item[key] = resolver.path(item[key], f"{at}.{item_index}.{key}", rule)
            elif isinstance(value, list):
                options[name] = [
                    resolver.path(item, f"{at}.{item_index}", rule) for item_index, item in enumerate(value)
                ]
            else:
                options[name] = resolver.path(value, at, rule)
    return resolver.problems


def _entry_service(entry: Mapping[str, Any]) -> str | None:
    """The service an ``environments`` entry names, or that the named environment it selects (``env:``) names."""

    service = entry.get("service")
    if isinstance(service, str) and service:
        return service
    reference = entry.get("env")
    if not (isinstance(reference, str) and reference):
        return None
    try:
        from platoon.envs.named import get_environment

        named = get_environment(reference).entry().get("service")
    except Exception:  # loading the config reports a named environment that does not resolve
        return None
    return named if isinstance(named, str) and named else None


class _InputPaths:
    """Relative input paths made absolute against one directory, and the problems with them
    (:func:`resolve_input_paths`)."""

    def __init__(self, base: Path, source: Path | None) -> None:
        self.base = base
        self.source = source
        self.problems: list[str] = []

    def harness(self, ref: str, where: str) -> str:
        """A harness reference: a ``.toml`` spec path resolves; a name or ``module:function`` stays as it is."""

        from platoon.envs.service import PathOption

        return self.path(ref, where, PathOption()) if ref.endswith(".toml") else ref

    def path(self, value: Any, where: str, rule: Any) -> Any:
        if not isinstance(value, str) or not value or _URL.match(value) or value.startswith("git+"):
            return value  # a mapping, a URL or nothing: not a local path
        expanded = os.path.expanduser(value)
        if os.path.isabs(expanded):
            return expanded
        resolved = os.path.normpath(self.base / expanded)
        if self.source is not None and not os.path.exists(resolved):
            here = os.path.normpath(Path.cwd() / expanded)
            elsewhere = here != resolved and os.path.exists(here)
            if elsewhere or not rule.written:
                problem = (
                    f"{where}: {value} names {resolved}, which does not exist: a relative path in a config file is "
                    f"relative to the file's directory, {self.base}"
                )
                if rule.relative_to is not None:  # the file names no such option: the path cannot be relative to it
                    problem += f" (or to {rule.relative_to}, when the same file names one)"
                if elsewhere:
                    problem += (
                        f". {here} exists, relative to the working directory, where relative paths resolved before: "
                        f"write the path relative to {self.source.name}, or absolute"
                    )
                self.problems.append(problem)
        return resolved


def _expand_home_in_outputs(data: dict[str, Any]) -> None:
    """Expand a leading ``~`` in a config's output directories, in place: ``log_path`` and ``inference.output_dir``
    name the home directory with it, both where the command writes the run's record and where the run it starts reads
    that record (:func:`_absolute_outputs`); a relative one stays as written."""

    def expand(value: Any) -> Any:
        return os.path.expanduser(value) if isinstance(value, str) and value.startswith("~") else value

    if "log_path" in data:
        data["log_path"] = expand(data["log_path"])
    inference = data.get("inference")
    if isinstance(inference, dict) and "output_dir" in inference:
        inference["output_dir"] = expand(inference["output_dir"])


def _absolute_outputs(data: dict[str, Any]) -> None:
    """Make a config's output directories absolute against the working directory, in place: ``log_path`` and
    ``inference.output_dir``, which a config file or an override gives relative to it (RFC 0018, section 5), as a run
    writes its record (:func:`write_run_record`), so that the record reloads from anywhere."""

    def absolute(value: Any) -> Any:
        if not isinstance(value, str) or not value:
            return value
        return os.path.normpath(Path.cwd() / os.path.expanduser(value))

    if "log_path" in data:
        data["log_path"] = absolute(data["log_path"])
    inference = data.get("inference")
    if isinstance(inference, dict) and "output_dir" in inference:
        inference["output_dir"] = absolute(inference["output_dir"])


# --- schema introspection ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Field:
    annotation: Any
    required: bool
    deprecated: DeprecatedKey | None = None


def _strip(annotation: Any) -> Any:
    while typing.get_origin(annotation) is typing.Annotated:
        annotation = typing.get_args(annotation)[0]
    return annotation


def _is_union(annotation: Any) -> bool:
    return typing.get_origin(annotation) in (Union, types.UnionType)


def _admits_none(annotation: Any) -> bool:
    annotation = _strip(annotation)
    if annotation in (Any, object, None, type(None)):
        return True
    if _is_union(annotation):
        return any(_admits_none(member) for member in typing.get_args(annotation))
    if typing.get_origin(annotation) is Literal:
        return None in typing.get_args(annotation)
    return False


def _without_none(annotation: Any) -> list[Any]:
    return [member for member in typing.get_args(annotation) if member is not type(None)]


def _optional_choice(members: Sequence[Any]) -> Any | None:
    """The choices of an optional choice (``Literal[...] | None``, whose null is its default) from its union's members
    other than None, which a wrong value is told as the choices alone are: ``'random' is not one of 'auto',
    'weighted'``; None for another union."""

    if len(members) == 1 and typing.get_origin(_strip(members[0])) is Literal:
        return members[0]
    return None


def _is_pydantic(annotation: Any) -> bool:
    return (
        isinstance(annotation, type) and hasattr(annotation, "model_fields") and hasattr(annotation, "model_validate")
    )


def _schema_fields(annotation: Any) -> dict[str, _Field] | None:
    """The settable fields of a dataclass or pydantic model; ``None`` for anything else."""

    annotation = _strip(annotation)
    if isinstance(annotation, type) and is_dataclass(annotation):
        try:
            hints = typing.get_type_hints(annotation, include_extras=True)
        except Exception:  # unresolvable annotations: accept any value for those fields
            hints = {}
        deprecated = deprecated_keys(annotation)
        return {
            item.name: _Field(
                hints.get(item.name, Any),
                item.default is dataclasses.MISSING and item.default_factory is dataclasses.MISSING,
                deprecated.get(item.name),
            )
            for item in dataclasses.fields(annotation)
            if item.init
        }
    if _is_pydantic(annotation):
        found: dict[str, _Field] = {}
        deprecated = deprecated_keys(annotation)
        for name, info in annotation.model_fields.items():
            entry = _Field(info.annotation, info.is_required(), deprecated.get(name))
            found[name] = entry
            if info.alias:
                found[info.alias] = entry
        return found
    return None


def _free_form_model(annotation: Any) -> bool:
    """A pydantic model that keeps keys it does not declare (``extra="allow"``)."""

    return _is_pydantic(annotation) and getattr(annotation, "model_config", {}).get("extra") == "allow"


def _sequence_item(annotation: Any) -> tuple[bool, Any]:
    """``(is a list-like type, its item type)``."""

    annotation = _strip(annotation)
    origin = typing.get_origin(annotation)
    if annotation in (list, tuple, set, frozenset) or origin in (
        list,
        tuple,
        set,
        frozenset,
        collections.abc.Sequence,
        collections.abc.MutableSequence,
    ):
        args = [arg for arg in typing.get_args(annotation) if arg is not Ellipsis]
        return True, args[0] if len(set(args)) == 1 else Any
    return False, None


def _mapping_value(annotation: Any) -> tuple[bool, Any]:
    """``(is a mapping type, its value type)``."""

    annotation = _strip(annotation)
    origin = typing.get_origin(annotation)
    if annotation in (dict, Mapping) or origin in (dict, collections.abc.Mapping, collections.abc.MutableMapping):
        args = typing.get_args(annotation)
        return True, args[1] if len(args) == 2 else Any
    return False, None


def _type_name(annotation: Any) -> str:
    annotation = _strip(annotation)
    if annotation is type(None):
        return "null"
    if _is_union(annotation):
        return " | ".join(_type_name(member) for member in typing.get_args(annotation))
    if typing.get_origin(annotation) is Literal:
        return " | ".join(repr(value) for value in typing.get_args(annotation))
    if isinstance(annotation, type) and not typing.get_args(annotation):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _join(path: str, key: Any) -> str:
    return f"{path}.{key}" if path else str(key)


def _suggest(key: str, candidates: Sequence[str], parent: str) -> str:
    matches = difflib.get_close_matches(key, list(candidates), n=1, cutoff=0.6)
    return f"; did you mean {_join(parent, matches[0])}?" if matches else ""


# A config class may say what to do instead of a key it does not take but that people put in it: a class attribute
# ``UNKNOWN_KEY_HINTS = {key: hint}`` (a ClassVar), which the unknown-key problem quotes.
UNKNOWN_KEY_HINTS = "UNKNOWN_KEY_HINTS"
# And where a required key it misses may come from instead: ``REQUIRED_KEY_HINTS = {key: hint}`` (a ClassVar), which
# the missing-key problem quotes.
REQUIRED_KEY_HINTS = "REQUIRED_KEY_HINTS"
# A config class may check a free-form mapping field against a schema of its own without building it: a class
# attribute ``FIELD_SCHEMAS = {field: schema}`` (a ClassVar).  Validation, the reading of numbers and the parsing of
# overrides take the field's value as a value of ``schema`` (its keys, with its UNKNOWN_KEY_HINTS, and their types),
# while the field keeps the mapping as given, so that a key it leaves out stays told apart from one it sets, null
# included.  An environment entry's ``workflow_config`` is checked so (``platoon.train.components``).
FIELD_SCHEMAS = "FIELD_SCHEMAS"


def _field_annotation(schema: Any, name: str, field: _Field) -> Any:
    """The type a config's value of field ``name`` of ``schema`` is checked as: its ``FIELD_SCHEMAS`` schema, else the
    field's own type."""

    return (getattr(schema, FIELD_SCHEMAS, None) or {}).get(name, field.annotation)


def _unknown_key(key: str, fields: Mapping[str, _Field], parent: str, hints: Mapping[str, str] | None = None) -> str:
    return unknown_key(key, fields, parent, hint=(hints or {}).get(key))


def unknown_key(key: Any, known: Iterable[str], parent: str, *, hint: str | None = None, otherwise: str = "") -> str:
    """The problem line for ``key``, which the mapping at dotted path ``parent`` does not take: the key's full path
    and ``hint`` when given, else a did-you-mean naming the closest of ``known`` (with its full path), else
    ``otherwise`` (such as the keys it does take)."""

    if hint:
        return f"{_join(parent, key)}: unknown key; {hint}"
    suggestion = _suggest(str(key), sorted(known), parent)
    return f"{_join(parent, key)}: unknown key{suggestion or (f'; {otherwise}' if otherwise else '')}"


# --- validation ------------------------------------------------------------------------------------------------


def validate_config(data: Any, schema: Any, *, path: str = "") -> list[str]:
    """Every problem with ``data`` as a value of ``schema``: unknown keys, wrong types, missing required keys, each
    with its dotted path, under ``path`` when ``data`` sits there in a config."""

    return _problems(data, schema, path)


def _problems(value: Any, annotation: Any, path: str) -> list[str]:
    annotation = _strip(annotation)
    where = path or "config"
    if annotation in (Any, object) or isinstance(annotation, TypeVar):
        return []
    if value is None:
        return [] if _admits_none(annotation) else [f"{where}: null is not allowed (expected {_type_name(annotation)})"]
    if _is_union(annotation):
        members = _without_none(annotation)
        if (choice := _optional_choice(members)) is not None:
            return _problems(value, choice, path)
        if any(not _problems(value, member, path) for member in members):
            return []
        # Report the member whose shape matches (a mapping for a model, say); otherwise the union as a whole.
        shaped = [member for member in members if _same_shape(value, member)]
        if len(shaped) == 1:
            return _problems(value, shaped[0], path)
        return [f"{where}: {value!r} is not a {_type_name(annotation)}"]
    if typing.get_origin(annotation) is Literal:
        allowed = typing.get_args(annotation)
        return [] if value in allowed else [f"{where}: {value!r} is not one of {', '.join(map(repr, allowed))}"]

    fields = _schema_fields(annotation)
    if fields is not None:
        if isinstance(value, annotation):
            return []
        if not isinstance(value, Mapping):
            return [f"{where}: expected a mapping ({annotation.__name__}), got {value!r}"]
        problems: list[str] = []
        free_form = _free_form_model(annotation)
        hints = getattr(annotation, UNKNOWN_KEY_HINTS, None)
        for key, item in value.items():
            if key in fields:
                problems.extend(_problems(item, _field_annotation(annotation, key, fields[key]), _join(path, key)))
            elif not free_form:
                problems.append(_unknown_key(str(key), fields, path, hints))
        required = getattr(annotation, REQUIRED_KEY_HINTS, None) or {}
        problems.extend(
            f"{_join(path, name)}: required key is missing" + (f"; {required[name]}" if name in required else "")
            for name, item in fields.items()
            if item.required and name not in value
        )
        return problems

    is_sequence, item_type = _sequence_item(annotation)
    if is_sequence:
        if not isinstance(value, (list, tuple)):
            return [f"{where}: expected a list, got {value!r}"]
        return [
            problem for index, item in enumerate(value) for problem in _problems(item, item_type, _join(path, index))
        ]
    is_mapping, value_type = _mapping_value(annotation)
    if is_mapping:
        if not isinstance(value, Mapping):
            return [f"{where}: expected a mapping, got {value!r}"]
        return [problem for key, item in value.items() for problem in _problems(item, value_type, _join(path, key))]
    return _scalar_problems(value, annotation, where)


def _scalar_problems(value: Any, annotation: Any, where: str) -> list[str]:
    if annotation is bool:
        ok = isinstance(value, bool)
    elif annotation is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif annotation is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not ok and isinstance(value, str) and _float_or_none(value) is not None:
            hint = "YAML reads an exponent without a dot as text; write 1.0e-5"
            return [f"{where}: {value!r} is a string, not a float ({hint})"]
    elif annotation is str:
        ok = isinstance(value, str)
    elif isinstance(annotation, type) and issubclass(annotation, os.PathLike):
        ok = isinstance(value, (str, os.PathLike))
    elif isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        ok = isinstance(value, annotation) or value in {member.value for member in annotation}
    else:  # other classes, protocols and callables: the constructor decides
        return []
    return [] if ok else [f"{where}: {value!r} is not a {_type_name(annotation)}"]


def _same_shape(value: Any, annotation: Any) -> bool:
    annotation = _strip(annotation)
    if isinstance(value, Mapping):
        return _schema_fields(annotation) is not None or _mapping_value(annotation)[0]
    if isinstance(value, (list, tuple)):
        return _sequence_item(annotation)[0]
    return False


def _float_or_none(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def normalize_config(data: Any, schema: Any) -> Any:
    """``data`` with numeric strings in float fields read as floats.

    YAML 1.1 reads an exponent without a dot (``3e-5``) as text; such a value in a float field is a float,
    as pydantic's lax mode would read it.  Everything else is left for :func:`validate_config` to judge.
    """

    annotation = _strip(schema)
    if isinstance(value := data, str):
        members = _without_none(annotation) if _is_union(annotation) else [annotation]
        if float in members and str not in members and (number := _float_or_none(value)) is not None:
            return number
        return value
    if _is_union(annotation):
        shaped = [member for member in _without_none(annotation) if _same_shape(value, member)]
        return normalize_config(value, shaped[0]) if len(shaped) == 1 else value
    fields = _schema_fields(annotation)
    if fields is not None and isinstance(value, Mapping):
        return {
            key: normalize_config(item, _field_annotation(annotation, key, fields[key])) if key in fields else item
            for key, item in value.items()
        }
    is_sequence, item_type = _sequence_item(annotation)
    if is_sequence and isinstance(value, list):
        return [normalize_config(item, item_type) for item in value]
    is_mapping, value_type = _mapping_value(annotation)
    if is_mapping and isinstance(value, Mapping):
        return {key: normalize_config(item, value_type) for key, item in value.items()}
    return value


# --- deprecations --------------------------------------------------------------------------------------------


def find_deprecations(data: Any, schema: Any) -> list[Deprecation]:
    """The deprecated keys that ``data``, a value of ``schema``, sets, in the order they appear, each with its full
    dotted path.  A key counts wherever the config gives it, even as ``null``: the release that removes it refuses
    it as an unknown key.  A renamed key at its former default beside its new name set to another value is ignored,
    and its deprecation says so (:meth:`~platoon.deprecations.DeprecatedKey.ignored_beside`)."""

    return _deprecations(data, schema, "")


def _deprecations(value: Any, annotation: Any, path: str) -> list[Deprecation]:
    annotation = _strip(annotation)
    if value is None or annotation in (Any, object) or isinstance(annotation, TypeVar):
        return []
    if _is_union(annotation):  # the member the value is built as (_build)
        member = next((member for member in _without_none(annotation) if not _problems(value, member, "")), None)
        return [] if member is None else _deprecations(value, member, path)
    fields = _schema_fields(annotation)
    if fields is not None:
        if not isinstance(value, Mapping):
            return []
        found: list[Deprecation] = []
        options = getattr(annotation, DEPRECATED_OPTIONS, None) or {}
        for key, item in value.items():
            entry = fields.get(key)
            if entry is None:
                continue
            where = _join(path, key)
            if entry.deprecated is not None:  # with the value of its new name, which may make it ignored
                renamed = entry.deprecated.renamed_to
                found.append(entry.deprecated.at(where, item, value.get(renamed) if renamed is not None else None))
            found.extend(_deprecations(item, entry.annotation, where))
            if key in options and isinstance(item, Mapping):  # a free-form mapping's keys, by the section's values
                going = options[key](value)
                found.extend(going[name].at(_join(where, name)) for name in item if name in going)
        return found
    is_sequence, item_type = _sequence_item(annotation)
    if is_sequence and isinstance(value, (list, tuple)):
        children = [(_join(path, index), item) for index, item in enumerate(value)]
    else:
        is_mapping, item_type = _mapping_value(annotation)
        if not (is_mapping and isinstance(value, Mapping)):
            return []
        children = [(_join(path, key), item) for key, item in value.items()]
    return [found for where, item in children for found in _deprecations(item, item_type, where)]


# --- building ------------------------------------------------------------------------------------------------


# A config class may lay a field of one of its sections over what another section built to, in place of the field's
# default: a class attribute ``LAID_OVER = {"<section>.<field>": function}`` (a ClassVar).  The function takes the
# sections built so far, by name, and the field's default, and returns the instance the field is laid over; the
# sections it names are built after the others.  A training config lays ``eval.workflow_config`` over
# ``train.workflow_config``'s rollout and worker settings (``platoon.train.tinker.config_defs``).
LAID_OVER = "LAID_OVER"
# The attribute in which a section the loader built keeps the mapping the config gave for it (given_mapping).
_GIVEN = "_platoon_given"


def given_mapping(section: Any) -> dict[str, Any] | None:
    """The mapping a config gave for ``section``, a section (a dataclass) that :func:`build_config` built: the keys
    the config sets for it, nested sections as given, and ``{}`` for a section the config leaves out, whatever
    defaults or other sections it was laid over.  So a training config's ``eval.workflow_config`` tells the keys it
    sets from those it took from ``train.workflow_config``.  None for a section the loader did not build, such as one
    built in Python."""

    given = getattr(section, _GIVEN, None)
    return copy.deepcopy(given) if isinstance(given, dict) else None


def _keep_given(section: Any, given: Mapping[str, Any]) -> None:
    """Keep ``given``, the mapping a config gave for ``section``, on the built section (:func:`given_mapping`), and on
    each of its sections what ``given`` holds for it: ``{}`` for one it leaves out, which holds its default."""

    if not is_dataclass(section) or isinstance(section, type):
        return
    try:
        object.__setattr__(section, _GIVEN, copy.deepcopy(dict(given)))
    except (AttributeError, TypeError):  # a class with slots keeps nothing
        return
    for item in dataclasses.fields(section):
        child = getattr(section, item.name, None)
        if is_dataclass(child) and not isinstance(child, type):
            nested = given.get(item.name)
            _keep_given(child, nested if isinstance(nested, Mapping) else {})


def build_config(schema: type[T], data: Mapping[str, Any]) -> T:
    """Construct ``schema`` from validated plain data, building nested models inside lists, mappings and unions.

    A section given in part is laid over its field's default: the keys it leaves out keep that default's values,
    not the nested class's (``eval.workflow_config`` defaults to ``WorkflowConfig(group_size=1, ...)``, so a
    partial one keeps ``group_size: 1``).  A class's ``LAID_OVER`` names fields laid over what another of its
    sections built to instead, and a section that leaves such a field out takes that whole.  Laying merges a nested
    section key by key, while any other value the section gives, ``null`` included, replaces the default's: free-form
    mappings (``loss_fn_config``, ``rollout_config.extra``) replace their default whole, and a key and the deprecated
    name it was renamed from count as one key.  When such a section's class refuses the values it was built from, the
    ``ValueError`` names the section and the defaults it kept, which the config never states
    (``min_successful_group_size: 4`` with the kept ``group_size: 1``).
    """

    built = _build(schema, data)
    _keep_given(built, data)
    return built


def lay_over(base: T, overlay: Mapping[str, Any]) -> T:
    """``overlay``, plain config data for a section, laid over ``base``, a built section, as :func:`build_config` lays
    a section over its field's default: a nested section merges key by key, any other value ``overlay`` gives,
    ``null`` included, replaces ``base``'s, a free-form mapping (``rollout_config.extra``) replaces ``base``'s whole,
    and a key and the deprecated name it was renamed from count as one key.  A new section of ``base``'s class, which
    may be a subclass with fields of its own; ``ValueError`` when the class refuses the values."""

    return build_config(type(base), _over_default(base, overlay))


def _build(
    annotation: Any,
    value: Any,
    path: str = "",
    kept: Mapping[str, Any] | None = None,
    over: Mapping[str, Any] | None = None,
) -> Any:
    """``value`` built as ``annotation``, at dotted ``path``; ``kept`` holds the values a section laid over its
    field's default took from that default for the keys it leaves out, and ``over`` the instances that fields of the
    section are laid over in place of their defaults (``LAID_OVER``)."""

    annotation = _strip(annotation)
    if value is None or annotation in (Any, object):
        return value
    if _is_union(annotation):
        for member in _without_none(annotation):
            if not _problems(value, member, ""):
                return _build(member, value, path, kept, over)
        return value
    fields = _schema_fields(annotation)
    if fields is not None and isinstance(value, Mapping):
        if _is_pydantic(annotation):
            return annotation.model_validate(dict(value))
        bases: dict[str, dict[str, Any]] = {}
        for target, function in (getattr(annotation, LAID_OVER, None) or {}).items():
            section, _, name = target.partition(".")
            bases.setdefault(section, {})[name] = function
        over = over or {}
        kwargs: dict[str, Any] = {}
        given = dict(value)
        for key in bases:  # a section laid over others that the config leaves out is laid over them all the same
            if key not in given and key in fields and not fields[key].required:
                given[key] = {}
        for key, item in sorted(given.items(), key=lambda pair: pair[0] in bases):  # laid over others: last
            laid = _over_field_default(annotation, key, item, over.get(key))
            if key in bases and laid is not item:  # its fields laid over other sections start from those, not here
                laid = {name: field for name, field in laid.items() if name not in bases[key]}
                laid.update({name: item[name] for name in bases[key] if name in item})
            taken = {name: laid[name] for name in laid if name not in item} if laid is not item else None
            inner = {}
            for name, function in bases.get(key, {}).items():
                base = function(kwargs, _field_default(fields[key].annotation, name))
                if base is not None:
                    inner[name] = base
            kwargs[key] = _build(fields[key].annotation, laid, _join(path, key), taken, inner)
            if isinstance(item, Mapping):  # what the config gave for the section, before it was laid over anything
                _keep_given(kwargs[key], item)
        for key, base in over.items():
            if key not in value:  # a field the section leaves out is what it would be laid over
                kwargs[key] = copy.deepcopy(base)
                _keep_given(kwargs[key], {})
        try:
            built = annotation(**kwargs)
        except (TypeError, ValueError) as exc:
            shown = {name: item for name, item in (kept or {}).items() if _is_scalar(item)}
            if not shown:
                raise
            defaults = ", ".join(f"{name}: {_yaml_scalar(item)}" for name, item in shown.items())
            raise ValueError(f"{path}: {exc} (the keys it leaves out keep their {path} defaults: {defaults})") from exc
        return built
    is_sequence, item_type = _sequence_item(annotation)
    if is_sequence and isinstance(value, (list, tuple)):
        items = [_build(item_type, item, _join(path, index)) for index, item in enumerate(value)]
        for built, item in zip(items, value):
            if isinstance(item, Mapping):
                _keep_given(built, item)
        origin = typing.get_origin(annotation) or annotation
        return tuple(items) if origin is tuple else items
    is_mapping, value_type = _mapping_value(annotation)
    if is_mapping and isinstance(value, Mapping):
        found = {key: _build(value_type, item, _join(path, key)) for key, item in value.items()}
        for key, item in value.items():
            if isinstance(item, Mapping):
                _keep_given(found[key], item)
        return found
    return value


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _yaml_scalar(value: Any) -> str:
    """``value`` as a config file writes it: ``false``, ``null``, ``1``; anything but a scalar as its repr."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return json.dumps(value)
    return repr(value)


def _field_default(cls: Any, name: str) -> Any:
    """The default of field ``name`` of dataclass ``cls``, a fresh one from its factory; None when it has none."""

    cls = _strip(cls)
    if not (isinstance(cls, type) and is_dataclass(cls)):
        return None
    item = next((item for item in dataclasses.fields(cls) if item.name == name), None)
    if item is None:
        return None
    if item.default_factory is not dataclasses.MISSING:
        return item.default_factory()
    return None if item.default is dataclasses.MISSING else item.default


def _over_field_default(cls: type, name: str, value: Any, default: Any = None) -> Any:
    """``value`` for field ``name`` of dataclass ``cls``: a mapping laid over ``default``, else over the field's
    default, when that is a dataclass instance (a nested model's schema default is the lowest layer, as RFC 0008 has
    it)."""

    if not isinstance(value, Mapping):
        return value
    if default is None:
        default = _field_default(cls, name)
    if not is_dataclass(default) or isinstance(default, type):
        return value
    return _over_default(default, value)


def _over_default(default: Any, value: Mapping[str, Any]) -> Mapping[str, Any]:
    """The mapping ``value`` with the values of the dataclass instance ``default`` it leaves out, nested sections
    laid over the default's own.

    Only the values that differ from the class's own defaults are carried; the class fills in the rest as it
    would.  So a default equal to its class's leaves ``value`` as given, and a value ``__post_init__``
    normalizes (``RolloutConfig.propagate_root_success``) never collides with the deprecated key it reads: when
    ``value`` gives a renamed key or its deprecated name, neither is carried from ``default``.
    """

    try:
        plain = type(default)()
    except TypeError:  # required fields, which validation asks of the mapping: nothing to carry
        return value
    merged = {
        item.name: getattr(default, item.name)
        for item in dataclasses.fields(default)
        if item.init and getattr(default, item.name) != getattr(plain, item.name)
    }
    for old, entry in deprecated_keys(type(default)).items():
        if entry.renamed_to is not None and (old in value or entry.renamed_to in value):
            merged.pop(old, None)  # a key and the name it was renamed from are one key: the value given decides
            merged.pop(entry.renamed_to, None)
    for key, item in value.items():
        current = getattr(default, key, None)
        section = is_dataclass(current) and not isinstance(current, type) and isinstance(item, Mapping)
        merged[key] = _over_default(current, item) if section else item
    return merged


# --- overrides -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Override:
    """One ``key=value`` (set) or ``key+=value`` (append) override; ``path`` is the dotted key split up."""

    path: tuple[str, ...]
    value: str
    append: bool = False

    @classmethod
    def parse(cls, text: str) -> Override:
        key, separator, value = text.partition("=")
        append = key.endswith("+")
        key = key[:-1] if append else key
        path = tuple(key.split("."))
        if not separator or not key or not all(path):
            raise ConfigError(f"override {text!r} is not of the form key=value or key+=value")
        return cls(path=path, value=value, append=append)

    @property
    def key(self) -> str:
        return ".".join(self.path)

    def __str__(self) -> str:
        return f"{self.key}{'+=' if self.append else '='}{self.value}"


def apply_overrides(data: Mapping[str, Any], overrides: Sequence[str | Override], schema: Any = None) -> dict[str, Any]:
    """A copy of ``data`` with each override applied in order, values parsed against ``schema``'s field types."""

    result = copy.deepcopy(dict(data))
    problems: list[str] = []
    for override in overrides:
        try:
            _apply(result, override if isinstance(override, Override) else Override.parse(override), schema)
        except ConfigError as exc:
            problems.extend(exc.problems)
    if problems:
        raise ConfigError(problems)
    return result


def _apply(data: dict[str, Any], override: Override, schema: Any) -> None:
    node: Any = data
    annotation: Any = Any if schema is None else schema
    path = ""
    for depth, segment in enumerate(override.path):
        last = depth == len(override.path) - 1
        if isinstance(node, list):
            key: Any = _list_index(node, segment, path, override)
            child_annotation = _child_annotation(annotation, key, path, override)
        elif isinstance(node, dict):
            key = segment
            child_annotation = _child_annotation(annotation, key, path, override)
        else:
            raise ConfigError(f"override {override}: {path} is {node!r}, which has no key {segment!r}")
        path = _join(path, segment)
        if last:
            if override.append:
                current = node[key] if (isinstance(node, list) or key in node) else None
                current = [] if current is None else current
                if not isinstance(current, list):
                    raise ConfigError(f"override {override}: {path} is {current!r}, not a list to append to")
                is_sequence, item_type = _sequence_item(_list_annotation(child_annotation))
                if not is_sequence and child_annotation is not Any:
                    raise ConfigError(f"override {override}: {path} is a {_type_name(child_annotation)}, not a list")
                item_type = item_type if is_sequence else Any
                node[key] = [*current, _parse_value(override.value, item_type, _join(path, len(current)), override)]
            else:
                value = _parse_value(override.value, child_annotation, path, override)
                current = node[key] if (isinstance(node, list) or key in node) else None
                if isinstance(value, Mapping) and isinstance(current, Mapping):
                    value = deep_merge(current, value)
                node[key] = value
            return
        child = node[key] if (isinstance(node, list) or key in node) else None
        if child is None:
            if _sequence_item(_list_annotation(child_annotation))[0]:
                raise ConfigError(f"override {override}: {path} has no entries to index")
            child = {}
            node[key] = child
        node, annotation = child, child_annotation


def _list_annotation(annotation: Any) -> Any:
    """The list-like member of ``annotation`` (``list[str] | None`` -> ``list[str]``), else ``annotation``."""

    annotation = _strip(annotation)
    if _is_union(annotation):
        for member in _without_none(annotation):
            if _sequence_item(member)[0]:
                return member
    return annotation


def _list_index(node: list[Any], segment: str, path: str, override: Override) -> int:
    try:
        index = int(segment)
    except ValueError:
        raise ConfigError(f"override {override}: {path} is a list; {segment!r} is not an index") from None
    if not -len(node) <= index < len(node):
        raise ConfigError(f"override {override}: {path} has {len(node)} entries; index {index} is out of range")
    return index % len(node)


def _child_annotation(annotation: Any, key: Any, path: str, override: Override) -> Any:
    """The type of ``key`` inside a value of type ``annotation``; an unknown key raises with a suggestion."""

    annotation = _strip(annotation)
    if annotation in (Any, object) or isinstance(annotation, TypeVar):
        return Any
    if _is_union(annotation):
        members = _without_none(annotation)
        candidates = [
            member for member in members if _container_kind(member) == ("list" if isinstance(key, int) else "map")
        ]
        if not candidates:
            raise ConfigError(f"override {override}: {path} is a {_type_name(annotation)}, which has no key {key!r}")
        if len(candidates) == 1 or isinstance(key, int):
            return _child_annotation(candidates[0], key, path, override)
        for member in candidates:  # a model whose field this is, else the first mapping
            fields = _schema_fields(member)
            if fields is not None and key in fields:
                return fields[key].annotation
        return _child_annotation(candidates[0], key, path, override)
    if isinstance(key, int):
        is_sequence, item_type = _sequence_item(annotation)
        return item_type if is_sequence else Any
    fields = _schema_fields(annotation)
    if fields is not None:
        if key in fields:
            return _field_annotation(annotation, key, fields[key])
        if _free_form_model(annotation):
            return Any
        hints = getattr(annotation, UNKNOWN_KEY_HINTS, None)
        raise ConfigError(f"override {override}: {_unknown_key(key, fields, path, hints)}")
    is_mapping, value_type = _mapping_value(annotation)
    if is_mapping:
        return value_type
    raise ConfigError(f"override {override}: {path} is a {_type_name(annotation)}, which has no key {key!r}")


def _container_kind(annotation: Any) -> str | None:
    if _schema_fields(annotation) is not None or _mapping_value(annotation)[0]:
        return "map"
    if _sequence_item(annotation)[0]:
        return "list"
    return None


def _parse_value(text: str, annotation: Any, path: str, override: Override) -> Any:
    """``text`` as a value of ``annotation``: scalars by their type, containers and models as YAML."""

    annotation = _strip(annotation)
    if annotation in (Any, object) or isinstance(annotation, TypeVar):
        return _parse_untyped(text)
    if text in _NULLS and _admits_none(annotation):
        return None
    if _is_union(annotation):
        members = _without_none(annotation)
        if (choice := _optional_choice(members)) is not None:
            return _parse_value(text, choice, path, override)
        # Typed members first; a plain string member takes whatever none of them accepts.
        for member in sorted(members, key=lambda member: _strip(member) is str):
            try:
                return _parse_value(text, member, path, override)
            except ConfigError:
                continue
        raise ConfigError(f"override {override}: {text!r} is not a {_type_name(annotation)}")
    if typing.get_origin(annotation) is Literal:
        for allowed in typing.get_args(annotation):
            if text == str(allowed) or (isinstance(allowed, bool) and text.lower() in (_TRUE if allowed else _FALSE)):
                return allowed
        allowed_text = ", ".join(map(repr, typing.get_args(annotation)))
        raise ConfigError(f"override {override}: {path} must be one of {allowed_text}, got {text!r}")
    if annotation is bool:
        if text.lower() in _TRUE:
            return True
        if text.lower() in _FALSE:
            return False
        raise ConfigError(f"override {override}: {path} is a bool; use true or false, got {text!r}")
    if annotation is int:
        try:
            return int(text.replace("_", ""))
        except ValueError:
            raise ConfigError(f"override {override}: {path} is an int, got {text!r}") from None
    if annotation is float:
        number = _float_or_none(text.replace("_", ""))
        if number is None:
            raise ConfigError(f"override {override}: {path} is a float, got {text!r}")
        return number
    if annotation is str or (isinstance(annotation, type) and issubclass(annotation, os.PathLike)):
        return text
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        for member in annotation:
            if text in (str(member.value), member.name):
                return member.value
        raise ConfigError(f"override {override}: {path} must be one of {[member.value for member in annotation]}")
    if _container_kind(annotation) is not None:
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"override {override}: {path} needs YAML for a {_type_name(annotation)}: {exc}") from None
        problems = _problems(value, annotation, path)
        if problems:
            raise ConfigError([f"override {override}: {problem}" for problem in problems])
        return value
    return _parse_untyped(text)


def _parse_untyped(text: str) -> Any:
    """A value for a free-form field: YAML scalars and containers, with exponent floats as numbers."""

    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    if isinstance(value, str) and value == text:
        number = _float_or_none(text)
        if number is not None and text.strip().lower() not in ("inf", "-inf", "+inf", "nan", "infinity"):
            return number
    return value


# --- resolving -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedConfig:
    """A config after ``extends`` and overrides: the plain data, the built object and where they came from, the
    deprecated keys it sets (:func:`find_deprecations`) and the deprecated variables set, and what the site profile
    gave it (``site``, :class:`platoon.site.SiteUse`)."""

    data: dict[str, Any]
    config: Any
    schema: type | None
    sources: tuple[Path, ...] = ()
    overrides: tuple[str, ...] = ()
    deprecations: tuple[Deprecation, ...] = ()
    site: SiteUse | None = None


def resolve_config(
    config_path: str | os.PathLike[str] | None,
    overrides: Sequence[str | Override] = (),
    schema: type | None = None,
    *,
    warn: bool = True,
    site: SiteProfile | None = None,
) -> ResolvedConfig:
    """Load ``config_path`` with its bases, apply ``overrides``, validate against ``schema`` and build it.

    Without a schema the overrides are parsed as YAML and nothing is validated; the config is the plain data.
    A config whose environments name a ``trainer_config`` class that subclasses ``schema`` loads as that class
    (:func:`narrow_schema`).  The class is chosen from the files, before the overrides, which are parsed against
    it; an override that changes the class is an error.  Every problem raises one :class:`ConfigError`.

    The site profile (:mod:`platoon.site`; ``site``, a :class:`~platoon.site.SiteProfile`, else this process's) is laid
    under what the config leaves out once the files and overrides are merged: a training config's ``log_path`` and
    ``stats.experiment_name``, each Harbor entry's provider.  The data holds what it gave, and ``site`` says what that
    was.  (An evaluation config's ``inference.output_dir`` stays unset: ``platoon eval`` makes each start a directory
    of its own under the profile's ``runs``.)

    The deprecated keys the config sets go into ``deprecations``, after the deprecated site variables that are set
    (:func:`platoon.site.variable_deprecations`), and with ``warn`` each is also warned of as a
    :class:`~platoon.deprecations.PlatoonDeprecationWarning`: an ``error`` warning filter
    (``warnings.simplefilter("error", PlatoonDeprecationWarning)``) makes the first one raise.  The command line
    passes ``warn=False`` and reports them itself.
    """

    from platoon.site import apply_profile, site_profile, variable_deprecations, variable_problems

    data: dict[str, Any] = {}
    sources: list[Path] = []
    if config_path is not None:
        data, sources = _load_with_bases(Path(config_path), ())
    parsed = [override if isinstance(override, Override) else Override.parse(override) for override in overrides]
    base_schema = schema
    try:
        if base_schema is not None:
            schema = narrow_schema(base_schema, data)
        named = _trainer_config_references(data)
        data = apply_overrides(data, parsed, schema)
        if base_schema is not None and schema is not None and _trainer_config_references(data) != named:
            _check_trainer_config_override(base_schema, schema, data)
    except ConfigError as exc:
        raise ConfigError(exc.problems, source=config_path) from None
    named_providers = _named_providers(data)
    data = _expand_named_environments(data, config_path)
    # The files' input paths are absolute already; the overrides' and a named environment's are relative to the
    # working directory.  Outputs stay as given, relative to it too (a leading ~ expanded), and the run's record holds
    # them absolute.
    resolve_input_paths(data, Path.cwd())
    _expand_home_in_outputs(data)
    profile = site_profile() if site is None else site
    if problems := variable_problems(profile):
        raise ConfigError(problems, source=config_path)
    fields = _schema_fields(schema) if schema is not None else None
    use = apply_profile(
        data,
        profile,
        config_path=config_path,
        log_path=fields is not None and "log_path" in fields,
        named_providers=named_providers,
    )
    deprecations: tuple[Deprecation, ...] = tuple(variable_deprecations())
    if schema is None:
        config: Any = data
    else:
        data = normalize_config(data, schema)
        problems = validate_config(data, schema)
        if problems:
            raise ConfigError(problems, source=config_path)
        try:
            config = build_config(schema, data)
        except ConfigError as exc:  # a config class's own checks, every problem with its dotted path
            raise ConfigError(exc.problems, source=config_path) from exc
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{type(exc).__name__}: {exc}", source=config_path) from exc
        deprecations += tuple(find_deprecations(data, schema))
    if warn:
        warn_deprecations(deprecations)
    return ResolvedConfig(
        data=data,
        config=config,
        schema=schema,
        sources=tuple(sources),
        overrides=tuple(str(override) for override in parsed),
        deprecations=deprecations,
        site=use,
    )


def _named_providers(data: Mapping[str, Any]) -> list[bool]:
    """For each ``environments`` entry, whether the config's files or overrides name its sandbox provider
    (``service_kwargs.provider``, not ``null``), before a named environment lays its entry under them."""

    environments = data.get("environments")
    return [
        isinstance(entry, Mapping)
        and isinstance(entry.get("service_kwargs"), Mapping)
        and entry["service_kwargs"].get("provider") is not None
        for entry in (environments if isinstance(environments, list) else ())
    ]


def _expand_named_environments(data: dict[str, Any], source: str | os.PathLike[str] | None) -> dict[str, Any]:
    """``data`` with every ``environments`` entry's named environment (``env:``) laid under its own keys."""

    environments = data.get("environments")
    if not isinstance(environments, list) or not any(
        isinstance(entry, Mapping) and entry.get("env") is not None for entry in environments
    ):
        return data
    from platoon.envs.named import expand_environments

    try:
        return expand_environments(data)
    except ConfigError as exc:
        raise ConfigError(exc.problems, source=source) from None


def _trainer_config_references(data: Mapping[str, Any]) -> dict[str, int]:
    """The ``trainer_config`` references a config's ``environments`` entries give (plain data), each with the index
    of the first entry that gives it.  Values that are not strings are left to validation, which refuses them."""

    environments = data.get("environments")
    references: dict[str, int] = {}
    if isinstance(environments, list):
        for index, entry in enumerate(environments):
            reference = entry.get("trainer_config") if isinstance(entry, Mapping) else None
            if isinstance(reference, str) and reference:
                references.setdefault(reference, index)
    return references


def narrow_schema(schema: type, data: Mapping[str, Any]) -> type:
    """``schema``, or the subclass of it that an environment's ``trainer_config`` names (``module:Class``).

    A plugin that adds config sections subclasses the trainer config and names the class on its environment
    entry, so its sections validate; the reference may also be a name registered as a ``trainer_config``
    component by one of the environments' ``package`` modules.  A class that does not subclass ``schema`` leaves
    it as it is: a training config's class means nothing to an evaluation config whose entries name it.  A
    reference that does not resolve, or that names something other than a class, is a :class:`ConfigError`
    (exit 78): the config would otherwise validate against ``schema`` without the class's sections and checks.
    """

    references = _trainer_config_references(data)
    if not references:
        return schema
    if len(references) > 1:
        raise ConfigError(f"environments name different trainer_config classes: {sorted(references)}")
    [(reference, index)] = references.items()
    where = f"environments.{index}.trainer_config"
    try:
        narrowed = _import_trainer_config(reference, data["environments"])
    except Exception as exc:
        raise ConfigError(
            f"{where}: {reference!r} does not resolve ({type(exc).__name__}: {exc}); it names the config class "
            "this config loads as, module:Class, which must import wherever the config loads"
        ) from exc
    if not isinstance(narrowed, type):
        raise ConfigError(f"{where}: {reference!r} is not a class, but {narrowed!r}")
    if not issubclass(narrowed, schema):
        return schema  # a trainer config means nothing to this schema (evaluation, say)
    return narrowed


def _check_trainer_config_override(base_schema: type, schema: type, data: Mapping[str, Any]) -> None:
    """Refuse overrides that change the ``trainer_config`` class the config files chose (``schema``): the
    overrides were parsed against that class, and the config would be built as it."""

    narrowed = narrow_schema(base_schema, data)  # also refuses a reference that does not resolve
    if narrowed is not schema:
        index = min(_trainer_config_references(data).values(), default=None)
        where = "environments[].trainer_config" if index is None else f"environments.{index}.trainer_config"
        raise ConfigError(
            f"{where}: an override changes the config class from {schema.__qualname__} to {narrowed.__qualname__}, "
            "but the class is chosen from the config files, before the overrides; set trainer_config in a file"
        )


def _import_trainer_config(reference: str, environments: list[Any]) -> Any:
    module_name, _, attribute = reference.partition(":")
    if attribute:
        value: Any = importlib.import_module(module_name)
        for part in attribute.split("."):
            value = getattr(value, part)
        return value
    for entry in environments:  # a registered name: the environments' packages register it on import
        if isinstance(entry, Mapping) and entry.get("package"):
            importlib.import_module(entry["package"])
    from platoon.registry import resolve_component

    return resolve_component("trainer_config", reference)


def load_config(
    args: Sequence[str] | None = None,
    config_class: type[T] | None = None,
    default_config_path: str | None = None,
) -> tuple[T | dict[str, Any], dict[str, Any]]:
    """Load a config from command-line style arguments; returns ``(config, resolved data)``.

    ``args`` (default ``sys.argv[1:]``) holds ``--config PATH`` (or ``-c PATH``) and overrides written
    ``key=value``, ``key+=value``, ``--key=value`` or ``--key value``; a bare ``--key`` sets a flag to true.
    See :func:`resolve_config` for what loading does.

        config, data = load_config(["--config", "run.yaml", "train.batch_size=64"], PlatoonTinkerRLTrainerConfig)
    """

    config_path, overrides = _split_args(list(sys.argv[1:] if args is None else args), default_config_path)
    if config_path is not None:
        logger.info("Loading config from %s", config_path)
    resolved = resolve_config(config_path, overrides, config_class)
    if overrides:
        logger.info("Applied %d config overrides", len(overrides))
    return resolved.config, resolved.data


def _split_args(args: list[str], default_config_path: str | None) -> tuple[str | None, list[str]]:
    config_path = default_config_path
    overrides: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        index += 1
        if arg in ("--config", "-c"):
            if index >= len(args):
                raise ConfigError(f"{arg} needs a path")
            config_path = args[index]
            index += 1
        elif arg.startswith("--config="):
            config_path = arg.partition("=")[2]
        elif arg.startswith("--"):
            key = arg[2:]
            if "=" in key:
                overrides.append(key)
            elif index < len(args) and not args[index].startswith("--"):
                overrides.append(f"{key}={args[index]}")
                index += 1
            else:
                overrides.append(f"{key}=true")
        elif "=" in arg:
            overrides.append(arg)
        else:
            raise ConfigError(f"unexpected argument {arg!r}: overrides are key=value")
    return config_path, overrides


# --- run records ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    """The files :func:`write_run_record` wrote."""

    run_dir: Path
    config_path: Path
    lock_path: Path


def write_run_record(
    run_dir: str | os.PathLike[str],
    resolved: ResolvedConfig | Mapping[str, Any],
    *,
    argv: Sequence[str] | None = None,
    components: Sequence[Mapping[str, Any]] | None = None,
) -> RunRecord:
    """Write ``config.resolved.yaml`` and ``lock.json`` into ``run_dir`` (created if needed), replacing old ones.

    The snapshot is the merged config after overrides, loadable as a config file on its own and from any directory:
    its input paths are absolute, as the loader made them, and so are its outputs, ``log_path`` and
    ``inference.output_dir``, which it makes absolute against the working directory.  The values the site profile gave
    are in it, so it loads the same without the profile, and its header names them; so does the provider of each
    Harbor entry that names none without a profile, the service's default, so that a profile written while the run
    goes on cannot change it.  ``components``
    (:func:`platoon.extensions.run_components`) go into the lock with the extensions behind them, and so do a
    :class:`ResolvedConfig`'s deprecated keys (``deprecations``, empty when it sets none), its site profile and cache
    root (``site``: the file and its SHA-256, its values and what it gave the config) and the sandbox provider of each
    Harbor entry with its options and where each came from (``sandbox_providers``).
    """

    run_dir = Path(run_dir)
    deprecations = resolved.deprecations if isinstance(resolved, ResolvedConfig) else None
    # First, so that a record of image pulls that cannot be read writes nothing.
    lock = run_lock(argv, components=components, deprecations=deprecations)
    use = resolved.site if isinstance(resolved, ResolvedConfig) else None
    if use is not None:
        lock["site"] = use.record()
        if use.providers:
            lock["sandbox_providers"] = [choice.record() for choice in use.providers]
    run_dir.mkdir(parents=True, exist_ok=True)
    data = copy.deepcopy(resolved.data if isinstance(resolved, ResolvedConfig) else dict(resolved))
    _absolute_outputs(data)  # the record loads the same from any directory
    if use is not None:
        _record_providers(data, use)  # and under any site profile
    header = ["# Written by platoon: the merged config after extends and overrides. Loadable as a config file."]
    if isinstance(resolved, ResolvedConfig):
        header += [f"# source: {source}" for source in resolved.sources]
        header += [f"# override: {override}" for override in resolved.overrides]
    if use is not None and use.gave:
        header.append(f"# site profile: {use.profile.path} (sha256 {use.profile.sha256})")
        header += [f"# from the site profile: {key}" for key in use.gave]
    config_path = run_dir / RESOLVED_CONFIG_NAME
    body = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)
    config_path.write_text("\n".join(header) + "\n" + body)
    lock_path = run_dir / LOCK_NAME
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    return RunRecord(run_dir=run_dir, config_path=config_path, lock_path=lock_path)


def _record_providers(data: dict[str, Any], use: SiteUse) -> None:
    """Name in a run's snapshot ``data``, in place, the provider of each Harbor entry that names none and that no site
    profile gave one: the service's default, which the run started with (:func:`platoon.site.apply_profile`)."""

    environments = data.get("environments")
    for choice in use.providers:
        if choice.source != "default" or choice.index is None or not isinstance(environments, list):
            continue
        entry = environments[choice.index]
        if not isinstance(entry, dict):
            continue
        kwargs = entry.get("service_kwargs")
        if kwargs is None:
            kwargs = entry["service_kwargs"] = {}
        if isinstance(kwargs, dict) and kwargs.get("provider") is None:
            kwargs["provider"] = choice.provider


def update_run_lock(run_dir: str | os.PathLike[str], values: Mapping[str, Any]) -> bool:
    """Add ``values`` to the ``lock.json`` that :func:`write_run_record` wrote into ``run_dir``, for what a run learns
    only once it starts (the tasks ``exclude_tasks`` left out).  Returns False, writing nothing, when ``run_dir`` has
    no ``lock.json`` (a run that no command started) or one that is not a JSON object."""

    path = Path(run_dir) / LOCK_NAME
    try:
        lock = json.loads(path.read_text())
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as exc:
        logger.warning("Cannot read %s to add %s to it: %s", path, sorted(values), exc)
        return False
    if not isinstance(lock, dict):
        logger.warning("%s holds no JSON object; %s are not added to it", path, sorted(values))
        return False
    lock.update(values)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(lock, indent=2) + "\n")
    partial.replace(path)  # readers never see half a file
    return True


def run_lock(
    argv: Sequence[str] | None = None,
    *,
    components: Sequence[Mapping[str, Any]] | None = None,
    deprecations: Sequence[Deprecation] | None = None,
) -> dict[str, Any]:
    """What ``lock.json`` records: package versions (``versions``) and the commit of each one installed from git
    (``commits``), the platoon checkout's git SHA, host, start time, argv; with ``deprecations``, the deprecated keys
    the config sets (``path``, ``replacement``, ``removed_in``); the images the job pulled (``image_pulls``,
    :func:`image_pulls`) when ``$PLATOON_IMAGE_PULLS`` names their record; and with ``components``, each component
    the run names (``kind``, ``ref``, ``resolved``, ``source``, ``distribution``, ``distribution_version``,
    ``entry_point``, ``environments``) and the extensions behind them (``extensions``: each distribution's version,
    platoon requirements and PEP 610 ``direct_url``).  Raises :class:`ConfigError` for a record of image pulls that
    cannot be read."""

    git_sha, git_dirty = _platoon_git_state()
    lock: dict[str, Any] = {
        "versions": {name: distribution_version(name) for name in LOCKED_DISTRIBUTIONS},
        "commits": {name: commit for name in LOCKED_DISTRIBUTIONS if (commit := distribution_commit(name))},
        "platoon_git_sha": git_sha,
        "platoon_git_dirty": git_dirty,
        "python": sys.version.split()[0],
        "hostname": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": list(sys.argv if argv is None else argv),
    }
    if deprecations is not None:
        lock["deprecations"] = [deprecation.to_dict() for deprecation in deprecations]
    if (pulls := image_pulls()) is not None:
        lock["image_pulls"] = pulls
    if components is not None:
        from platoon.extensions import extension_records

        lock["components"] = [dict(component) for component in components]
        lock["extensions"] = extension_records(components)
    return lock


def image_pulls(path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]] | None:
    """The images the job pulled before the run, as hpc-sandbox's ``pull`` printed them (``python -m hpc_sandbox
    pull [--refresh] IMAGE... > FILE``): one JSON object per line, with the image's ``reference``, the ``path`` of its
    file in the image cache, that file's ``sha256``, and the ``manifest_digest`` of a reference pinned by digest
    (else null).  Read from ``path``, else from the file ``$PLATOON_IMAGE_PULLS`` names; None when neither names
    one.  Raises :class:`ConfigError` for a file that cannot be read, holds no line, or holds a line that is not such
    an object: a job that meant to record its images must not run without the record."""

    if path is None:
        if not (named := os.environ.get(IMAGE_PULLS_ENV)):
            return None
        path, where = named, f"{IMAGE_PULLS_ENV}={named}"
    else:
        where = str(path)
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Cannot read the image pulls in {where}: {exc}") from None
    records: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            record = None
        if not _is_image_pull(record):
            raise ConfigError(
                f"{where}, line {number}: expected a line of `python -m hpc_sandbox pull` (a JSON object with "
                f"{', '.join(IMAGE_PULL_KEYS)}), got {line[:200]!r}"
            )
        records.append(record)
    if not records:
        raise ConfigError(f"{where} holds no image pulls: did `python -m hpc_sandbox pull` fail?")
    return records


def _is_image_pull(record: Any) -> TypeGuard[dict[str, Any]]:
    if not isinstance(record, dict) or not all(key in record for key in IMAGE_PULL_KEYS):
        return False
    reference, path, sha256, manifest = (record[key] for key in IMAGE_PULL_KEYS)
    return (
        isinstance(reference, str)
        and bool(reference)
        and isinstance(path, str)
        and bool(path)
        and isinstance(sha256, str)
        and _SHA256_HEX.fullmatch(sha256) is not None
        and (manifest is None or (isinstance(manifest, str) and _MANIFEST_DIGEST.fullmatch(manifest) is not None))
    )


def _platoon_root() -> Path:
    return Path(__file__).resolve().parents[1]


def distribution_version(name: str) -> str | None:
    """The installed version of a distribution, or ``None``.  For platoon itself it is the running platoon's
    (:func:`platoon.extensions.platoon_version`: a source checkout's ``pyproject.toml`` before any installed
    metadata), so ``platoon --version``, ``platoon.__version__``, ``lock.json`` and the extension checks agree."""

    if name == "platoon":
        from platoon.extensions import platoon_version

        return platoon_version()
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def distribution_commit(name: str) -> str | None:
    """The git commit an installed distribution was built from, as its PEP 610 ``direct_url.json`` records it for an
    install from a git URL (such as an hpc-sandbox commit in place of a release); None for one from an index or a local
    directory, editable checkouts included, and for one that is not installed.  For platoon itself, only when that
    installed copy is the platoon that runs: a source checkout on the path shadows it, and then ``versions`` gives the
    checkout's version (:func:`distribution_version`) and ``platoon_git_sha`` its commit."""

    from platoon.extensions import direct_url

    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    record = direct_url(distribution)
    vcs_info = record.get("vcs_info") if isinstance(record, dict) else None
    commit = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    if not (isinstance(commit, str) and commit):
        return None
    if name == "platoon" and Path(str(distribution.locate_file("platoon"))).resolve() != _platoon_root():
        return None  # the installed copy's commit, but a checkout on the path is the platoon that runs
    return commit


def _platoon_git_state() -> tuple[str | None, bool | None]:
    """``(HEAD sha, has uncommitted changes)`` of the git checkout platoon is imported from, if it is one."""

    package = _platoon_root()

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(package), *args], capture_output=True, text=True, timeout=10, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    toplevel = git("rev-parse", "--show-toplevel")
    if toplevel is None or Path(toplevel).resolve() / "platoon" != package:
        return None, None  # an installed copy inside some other repository is not a platoon checkout
    status = git("status", "--porcelain", "--untracked-files=no")
    return git("rev-parse", "HEAD"), None if status is None else bool(status)


def save_config(config: Any, path: str | os.PathLike[str]) -> None:
    """Save a config (a dataclass or a mapping) as YAML."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dataclasses.asdict(config) if is_dataclass(config) and not isinstance(config, type) else config
    with open(path, "w") as handle:
        yaml.dump(data, handle, default_flow_style=False, sort_keys=False)
    logger.info("Saved config to: %s", path)
