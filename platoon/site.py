"""The site profile: where a site keeps platoon's files, and which isolating runtime starts its sandboxes.

One file per site, written once by whoever runs platoon there and never part of a project (RFC 0018, section 4)::

    # ~/.config/platoon/site.yaml, or the file $PLATOON_SITE names
    cache: /shared/$USER/platoon-cache
    runs: /shared/$USER/platoon-runs
    provider: singularity
    provider_options: {allow_network_none: true, rootfs_layout: fuse-overlay}

**Which file.** ``$PLATOON_SITE``, else ``$XDG_CONFIG_HOME/platoon/site.yaml`` (``~/.config/platoon/site.yaml``),
else there is none.  A ``$PLATOON_SITE`` that names nothing, a key other than these four, or a value of the wrong
kind is a configuration error (exit 78).  ``~`` and ``$VAR`` (or ``${VAR}``) expand in every value, so one file can
serve a group, and a variable that is not set is an error; a relative path is relative to the file.

**The keys,** each optional:

* ``cache``: the cache root (:func:`cache_root`), under which ``platoon prepare`` makes what runs read: the Harbor
  cache, with its locks and image maps, in ``<cache>/harbor``, and harness toolchains in ``<cache>/toolchains``.
  Without it the cache root is ``~/.cache/platoon``.  It replaced ``$PLATOON_HARBOR_CACHE`` and
  ``$PLATOON_TOOLCHAINS``, which platoon 0.3.0 removed: a command that loads a config, or takes the cache from
  the site, refuses to run while one is set (:func:`removed_variable_problems`).
* ``runs``: where the run directories of configs that name none go.  A training config without ``log_path`` writes
  under ``<runs>/<stats.experiment_name>/<stats.trial_name>``, its experiment named after the config file unless it
  names one, so that two configs never share a run directory by default; an evaluation config without
  ``inference.output_dir`` writes each start to a new directory, ``<runs>/eval/<policy>/<UTC time>``
  (:func:`platoon.eval.run.prepare_output_dir`).  An explicit ``log_path`` or ``output_dir`` is never moved: a
  relative one stays relative to the working directory.
* ``provider`` and ``provider_options``: the sandbox runtime of every Harbor entry that names no provider, spelled as
  an entry spells them, and of ``platoon sandbox reap`` for a run whose record names none.  An entry that names its
  own provider uses its own options; one that names none lays its ``provider_options`` over the profile's.
* ``node_services``: declarations of node services by name (:mod:`platoon.envs.harbor.node_services.spec`), such as
  a judge pool whose image and test data live at this site's paths, which a Harbor entry takes by naming one with
  ``site`` (``node_services: {frontier-cs-judge: site}``); each runs on the entry's provider.

**What it may not do.**  The profile says where files live and which isolating
runtime starts sandboxes; it never weakens isolation and never changes what trains.

* It names only a provider that isolates its sandboxes, one whose class says so (``isolates = True``): ``docker``,
  ``singularity``, ``enroot``.  ``process``, which runs the agents' commands on the host, is an error there, and
  ``platoon train`` refuses a Harbor entry on such a provider unless the run config itself names it.
* Its ``provider_options`` are those that do not weaken isolation (:data:`PROVIDER_OPTIONS`); binds and extra runtime
  arguments belong in a run config.
* It holds no dataset, model, reward or recipe setting.

**Visible and recorded.**  The config loader applies it (:func:`apply_profile`), so ``config.resolved.yaml`` holds the
values it gave and a run directory loads again without it, and ``lock.json`` records the file's path and SHA-256 and
each Harbor entry's provider and options with their source.  ``platoon train``, ``eval``, ``rollout`` and
``prepare`` print the profile they used, and refuse a provider from it that this node cannot run.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from platoon.deprecations import removed_hint
from platoon.utils.config import ConfigError

__all__ = [
    "DEFAULT_CACHE_ROOT",
    "KEYS",
    "PROVIDER_OPTIONS",
    "REMOVED_VARIABLES",
    "SITE_ENV",
    "ProviderChoice",
    "SiteProfile",
    "SiteUse",
    "apply_profile",
    "cache_root",
    "check_run_directory",
    "load_profile",
    "profile_path",
    "removed_variable_problems",
    "site_profile",
]

#: The variable that names the site profile's file.
SITE_ENV = "PLATOON_SITE"
#: The keys a site profile takes.
KEYS = ("cache", "runs", "provider", "provider_options", "node_services")
#: The provider options a site profile may set: none of them weakens a sandbox's isolation.  Binds and extra runtime
#: arguments (``extra_start_args``, ``binary``, ...) belong in a run config.
PROVIDER_OPTIONS = ("allow_network_none", "rootfs_layout", "scratch_root", "image_cache", "stage_images", "gpu")
#: Provider options that name directories: relative to the profile's file, as its other paths are.
_PATH_OPTIONS = ("scratch_root", "image_cache")
_BOOLEAN_OPTIONS = ("allow_network_none", "stage_images")
_TEXT_OPTIONS = ("rootfs_layout", "gpu")
#: The cache root without a profile's ``cache``.
DEFAULT_CACHE_ROOT = "~/.cache/platoon"
#: The variables that named a directory under the cache root until platoon 0.3.0 removed them, with that directory's
#: name: the site profile's ``cache`` replaced them.
REMOVED_VARIABLES = {"PLATOON_HARBOR_CACHE": "harbor", "PLATOON_TOOLCHAINS": "toolchains"}
#: The provider of a Harbor entry that names none, without a profile's (``HarborEnvironmentService``'s default).
DEFAULT_PROVIDER = "docker"
_VARIABLE = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))")


class SiteProfileError(ConfigError):
    """A site profile that cannot be used: a configuration error (exit 78) that names the file."""


@dataclass(frozen=True)
class SiteProfile:
    """A site profile as loaded (:func:`load_profile`): its file (None when there is none), the file's SHA-256, and
    the values it gives, ``~`` and variables expanded and paths absolute."""

    path: Path | None = None
    sha256: str | None = None
    cache: Path | None = None
    runs: Path | None = None
    provider: str | None = None
    provider_options: Mapping[str, Any] = field(default_factory=dict)
    node_services: Mapping[str, Any] = field(default_factory=dict)

    @property
    def values(self) -> dict[str, Any]:
        """The keys the profile sets, with their values as it gives them (paths as strings)."""

        values: dict[str, Any] = {}
        if self.cache is not None:
            values["cache"] = str(self.cache)
        if self.runs is not None:
            values["runs"] = str(self.runs)
        if self.provider is not None:
            values["provider"] = self.provider
        if self.provider_options:
            values["provider_options"] = dict(self.provider_options)
        if self.node_services:
            values["node_services"] = dict(self.node_services)
        return values

    def describe(self) -> str:
        """One line: the file and what it gives, or that there is none and where it was looked for."""

        if self.path is None:
            looked = os.environ.get(SITE_ENV) or str(_default_path())
            return f"no site profile ({looked} does not exist): cache root {_shown(cache_root(self))}"
        values = self.values
        if not values:
            return f"site profile {self.path}: sets nothing"
        shown = []
        for key in KEYS:
            if key in values:
                value = values[key]
                if key == "node_services":  # declarations: their names say enough here
                    shown.append(f"{key} {', '.join(value)}")
                else:
                    shown.append(f"{key} {_flow(value) if isinstance(value, Mapping) else value}")
        return f"site profile {self.path}: {', '.join(shown)}"

    def snippet(self) -> str:
        """The profile's provider lines, as the file would hold them, for messages."""

        lines = [f"provider: {self.provider}"]
        if self.provider_options:
            lines.append(f"provider_options: {_flow(self.provider_options)}")
        return "\n".join(lines)

    def record(self) -> dict[str, Any]:
        """What ``lock.json`` records of the profile (``site``): the file and its SHA-256, and its values."""

        return {"path": None if self.path is None else str(self.path), "sha256": self.sha256, **self.values}


def _flow(mapping: Mapping[str, Any]) -> str:
    """A mapping as one line of YAML flow style."""

    import yaml

    return yaml.safe_dump(dict(mapping), default_flow_style=True, sort_keys=False, width=1 << 16).strip()


def _shown(path: Path) -> str:
    home = str(Path.home())
    text = str(path)
    return "~" + text[len(home) :] if text == home or text.startswith(home + os.sep) else text


def _default_path() -> Path:
    """``$XDG_CONFIG_HOME/platoon/site.yaml``, ``~/.config/platoon/site.yaml`` without the variable."""

    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "platoon" / "site.yaml"


def profile_path() -> Path | None:
    """The site profile's file: ``$PLATOON_SITE``, which must name a file, else ``$XDG_CONFIG_HOME/platoon/site.yaml``
    (``~/.config/platoon/site.yaml``) when it exists, else None.  A ``$PLATOON_SITE`` set to the empty string counts
    as unset."""

    if named := os.environ.get(SITE_ENV):
        path = Path(named).expanduser()
        if not path.exists() or path.is_dir():
            raise SiteProfileError(
                f"${SITE_ENV} names {named}, which is not a file: point it at a site profile, or unset it to use "
                f"{_default_path()} when that exists"
            )
        return path.absolute()
    default = _default_path()
    return default.absolute() if default.is_file() else None


def site_profile() -> SiteProfile:
    """This process's site profile (:func:`profile_path`, :func:`load_profile`): an empty one when there is none.  The
    file is read again when it changes."""

    path = profile_path()
    if path is None:
        return SiteProfile()
    try:
        info = path.stat()
    except OSError as exc:
        raise SiteProfileError(f"cannot read the site profile {path}: {exc}") from None
    return _cached_profile(str(path), info.st_mtime_ns, info.st_size, info.st_ino)


@functools.lru_cache(maxsize=8)
def _cached_profile(path: str, mtime_ns: int, size: int, inode: int) -> SiteProfile:
    return load_profile(Path(path))


def load_profile(path: str | os.PathLike[str]) -> SiteProfile:
    """Read and check the site profile at ``path``: its keys, their values (``~`` and variables expanded, paths made
    absolute against the file), its provider (one that isolates) and its provider options (:data:`PROVIDER_OPTIONS`).
    Every problem raises one :class:`SiteProfileError`."""

    import yaml

    path = Path(path).absolute()
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SiteProfileError(f"cannot read the site profile {path}: {exc}") from None
    try:
        data = yaml.safe_load(raw.decode("utf-8")) if raw.strip() else {}
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SiteProfileError(f"{path}: invalid YAML: {exc}", source=path) from None
    data = {} if data is None else data
    if not isinstance(data, Mapping):
        raise SiteProfileError(
            f"a site profile holds a mapping of {', '.join(KEYS)}, not {type(data).__name__}", source=path
        )
    from platoon.utils.config import unknown_key

    takes = (
        "a site profile takes cache, runs, provider, provider_options and node_services: where platoon's files live, "
        "which runtime starts sandboxes and which services a node runs for its rollouts, never what trains"
    )
    problems = [unknown_key(key, KEYS, "", otherwise=takes) for key in data if key not in KEYS]
    values: dict[str, Any] = {}
    for key in ("cache", "runs"):
        if key in data:
            values[key] = _path(data[key], key, path.parent, problems)
    provider = data.get("provider")
    if provider is not None:
        if not isinstance(provider, str) or not provider:
            problems.append(f"provider: {provider!r} is not a provider's name")
            provider = None
        else:
            provider = _expand(provider, "provider", problems)
            if (refusal := _isolation_refusal(provider)) is not None:
                problems.append(f"provider: {refusal}")
    options = data.get("provider_options")
    checked: dict[str, Any] = {}
    if options is not None:
        if not isinstance(options, Mapping):
            problems.append(f"provider_options: {options!r} is not a mapping of options")
        else:
            for name, value in options.items():
                if name not in PROVIDER_OPTIONS:
                    problems.append(
                        f"provider_options.{name}: a site profile sets only the options that do not weaken a "
                        f"sandbox's isolation, {', '.join(PROVIDER_OPTIONS)}; binds, extra runtime arguments and "
                        "the runtime's executable belong in a run config's provider_options"
                    )
                elif name in _PATH_OPTIONS and value is not None:
                    checked[name] = str(_path(value, f"provider_options.{name}", path.parent, problems))
                elif name in _BOOLEAN_OPTIONS and not isinstance(value, bool):
                    problems.append(f"provider_options.{name}: {value!r} is not true or false")
                elif name in _TEXT_OPTIONS and value is not None and not isinstance(value, str):
                    problems.append(f"provider_options.{name}: {value!r} is not text")
                else:
                    checked[name] = _expand_all(value, f"provider_options.{name}", problems)
            if checked and provider is None:
                problems.append("provider_options: options for no provider: name the provider they are for")
    services = data.get("node_services")
    node_services: dict[str, Any] = {}
    if services is not None:
        if not isinstance(services, Mapping) or not all(isinstance(each, Mapping) for each in services.values()):
            problems.append("node_services: declarations of node services by name, each a mapping")
        else:
            node_services = {
                str(name): _expand_all(declared, f"node_services.{name}", problems)
                for name, declared in services.items()
            }
    if problems:
        raise SiteProfileError(problems, source=path)
    return SiteProfile(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        cache=values.get("cache"),
        runs=values.get("runs"),
        provider=provider,
        provider_options=checked,
        node_services=node_services,
    )


def _expand(text: str, key: str, problems: list[str]) -> str:
    """``text`` with ``~`` and ``$VAR``/``${VAR}`` expanded; an unset variable is a problem."""

    def value(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("bare")
        found = os.environ.get(name)
        if found is None:
            problems.append(f"{key}: ${name} is not set")
            return match.group(0)
        return found

    return os.path.expanduser(_VARIABLE.sub(value, text))


def _expand_all(value: Any, key: str, problems: list[str]) -> Any:
    if isinstance(value, str):
        return _expand(value, key, problems)
    if isinstance(value, list):
        return [_expand_all(item, f"{key}.{index}", problems) for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        return {name: _expand_all(item, f"{key}.{name}", problems) for name, item in value.items()}
    return value


def _path(value: Any, key: str, base: Path, problems: list[str]) -> Path:
    if not isinstance(value, str) or not value:
        problems.append(f"{key}: {value!r} is not a path")
        return base
    return Path(os.path.normpath(base / _expand(value, key, problems)))


def _isolation_refusal(provider: str) -> str | None:
    """Why a site profile may not name ``provider``: one whose class does not say it isolates its sandboxes
    (``isolates = True``), or that does not import; None for one that isolates."""

    from platoon.sandbox import provider_class

    try:
        kind = provider_class(provider)
    except Exception as exc:  # ImportError, AttributeError, ValueError: a reference that does not resolve
        return f"{provider!r} does not resolve ({type(exc).__name__}: {exc})"
    if getattr(kind, "isolates", None) is True:
        return None
    if getattr(kind, "isolates", None) is False:
        return (
            f"{provider} runs the agents' commands on this host, without isolation, so a site profile may not choose "
            "it for every config: name it in the run config that means it"
        )
    return (
        f"{provider} does not say that it isolates its sandboxes (isolates = True on its class), so a site profile "
        "may not choose it: name it in the run configs that use it"
    )


def cache_root(profile: SiteProfile | None = None) -> Path:
    """The root of the default locations of what ``platoon prepare`` makes: the site profile's ``cache``, else
    ``~/.cache/platoon``."""

    profile = site_profile() if profile is None else profile
    return profile.cache if profile.cache is not None else Path(DEFAULT_CACHE_ROOT).expanduser()


# --- the removed variables ------------------------------------------------------------------------------------


def removed_variable_problems() -> list[str]:
    """A problem for each removed cache variable that is set (:data:`REMOVED_VARIABLES`): platoon 0.3.0 no longer reads
    it, so a job that sets it would find its files elsewhere than it means to; the site profile's ``cache`` names the
    directory instead.  The config loader, and the commands that take the cache from the site, refuse to run then."""

    return [
        f"${name}: "
        + removed_hint(
            "0.3.0",
            f"the site profile's cache, whose {directory} directory it named (~/.config/platoon/site.yaml, or the "
            f"file ${SITE_ENV} names), and unset ${name}",
        )
        for name, directory in REMOVED_VARIABLES.items()
        if os.environ.get(name)
    ]


# --- applying it to a config ------------------------------------------------------------------------------------

#: Where a Harbor entry's provider came from: the run config (its files or overrides), the site profile, a named
#: environment the entry selects, or the service's default.
ProviderSource = Literal["config", "site", "environment", "default"]


@dataclass(frozen=True)
class ProviderChoice:
    """The sandbox provider one Harbor entry runs, its options, and where each came from."""

    entry: str  # environments.<i>, with its name: environments.0 (harbor-dataset)
    provider: str
    options: Mapping[str, Any]
    source: ProviderSource
    option_sources: Mapping[str, str] = field(default_factory=dict)
    index: int | None = None  # the entry's position in ``environments``

    def record(self) -> dict[str, Any]:
        """What ``lock.json`` records of it (``sandbox_providers``)."""

        return {
            "environment": self.entry,
            "provider": self.provider,
            "options": dict(self.options),
            "source": self.source,
            "option_sources": dict(self.option_sources),
        }


@dataclass(frozen=True)
class SiteUse:
    """What a config took from the site profile when it loaded (:func:`apply_profile`): the profile, the values it gave
    (dotted path -> value), each Harbor entry's provider choice, and the config file whose name it gave the run
    directory (``named_after``: a training run's experiment), when it did.  ``platoon eval`` adds the directory it
    made under the profile's ``runs`` for an evaluation that names none (``inference.output_dir``)."""

    profile: SiteProfile
    gave: Mapping[str, Any] = field(default_factory=dict)
    providers: Sequence[ProviderChoice] = ()
    named_after: str | None = None

    def lines(self) -> list[str]:
        """What the commands print: the profile, and what it gave the config."""

        lines = [self.profile.describe()]
        if self.gave:
            given = ", ".join(
                f"{key}={_flow(value) if isinstance(value, Mapping) else value}" for key, value in self.gave.items()
            )
            lines.append(f"from the site profile: {given}")
        return lines

    def record(self) -> dict[str, Any]:
        """What ``lock.json`` records (``site``): the profile, the cache root, the values it gave the config, and the
        config file the run directory is named after, when the profile named it (:func:`check_run_directory`)."""

        root = cache_root(self.profile)
        record = {
            **self.profile.record(),
            "cache_root": str(root),
            "cache_root_source": "site" if self.profile.cache is not None else "default",
            "gave": dict(self.gave),
        }
        if self.named_after is not None:
            record["named_after"] = self.named_after
        return record

    def unisolated(self) -> list[ProviderChoice]:
        """The Harbor entries on a provider that does not isolate its sandboxes which the run config does not name."""

        from platoon.sandbox import provider_class

        found = []
        for choice in self.providers:
            if choice.source == "config":
                continue
            try:
                kind = provider_class(choice.provider)
            except Exception:  # building the provider reports it
                continue
            if getattr(kind, "isolates", None) is False:
                found.append(choice)
        return found


def apply_profile(
    data: dict[str, Any],
    profile: SiteProfile,
    *,
    config_path: str | os.PathLike[str] | None,
    log_path: bool,
    named_providers: Sequence[bool] = (),
) -> SiteUse:
    """Lay the site profile under a loaded config's plain ``data``, in place, and return what it gave.

    * With ``log_path`` (a training config's schema) and no ``log_path`` in ``data``, the profile's ``runs`` is the
      ``log_path``, and then ``stats.experiment_name``, unless ``data`` names one, is the config file's name.  An
      evaluation config's ``inference.output_dir`` stays as it is: without one, each start of the evaluation gets a
      new directory under the profile's ``runs`` when it starts (:func:`platoon.eval.run.prepare_output_dir`).
    * Each Harbor entry that names no ``provider`` (or ``null``) gets the profile's, and its ``provider_options`` laid
      over the profile's.  ``named_providers`` says, per entry, whether the run config's files or overrides named its
      provider, before a named environment (``env:``) laid its entry under them.

    A run directory named after the config file's name is named after the file too (``SiteUse.named_after``), which
    the run records and :func:`check_run_directory` compares.
    """

    gave: dict[str, Any] = {}
    named_after: str | None = None
    stem = Path(config_path).stem if config_path is not None else None
    if profile.runs is not None and stem:
        config_file = str(Path(config_path).expanduser().resolve()) if config_path is not None else None
        if log_path and data.get("log_path") is None:
            data["log_path"] = str(profile.runs)
            gave["log_path"] = str(profile.runs)
            stats = data.get("stats")
            if not isinstance(stats, dict):
                stats = data["stats"] = {} if stats is None else stats
            if isinstance(stats, dict) and stats.get("experiment_name") is None:
                stats["experiment_name"] = stem
                gave["stats.experiment_name"] = stem
                named_after = config_file
    providers: list[ProviderChoice] = []
    environments = data.get("environments")
    for index, entry in enumerate(environments if isinstance(environments, list) else ()):
        if not isinstance(entry, dict) or not _is_harbor(entry.get("service")):
            continue
        where = f"environments.{index}"
        label = f"{where} ({entry['name']})" if isinstance(entry.get("name"), str) else where
        kwargs = entry.get("service_kwargs")
        if kwargs is None:
            kwargs = {}  # added to the entry only when the profile gives it something
        if not isinstance(kwargs, dict):
            continue  # validation refuses it
        own = kwargs.get("provider_options") if isinstance(kwargs.get("provider_options"), Mapping) else {}
        if kwargs.get("provider") is not None:
            named = index < len(named_providers) and named_providers[index]
            source: ProviderSource = "config" if named or not named_providers else "environment"
            if not isinstance(kwargs["provider"], str):
                continue  # a provider object passed in Python
            providers.append(
                ProviderChoice(label, kwargs["provider"], dict(own), source, {name: source for name in own}, index)
            )
        elif profile.provider is not None:
            options = {**profile.provider_options, **own}
            entry["service_kwargs"] = kwargs
            kwargs["provider"] = profile.provider
            gave[f"{where}.service_kwargs.provider"] = profile.provider
            if options:
                kwargs["provider_options"] = options
                if profile.provider_options:
                    gave[f"{where}.service_kwargs.provider_options"] = dict(profile.provider_options)
            sources = {name: ("config" if name in own else "site") for name in options}
            providers.append(ProviderChoice(label, profile.provider, options, "site", sources, index))
        else:
            kwargs.pop("provider", None)  # a null provider names none: the service's default starts the sandboxes
            default = _default_provider(entry["service"])
            providers.append(ProviderChoice(label, default, dict(own), "default", dict.fromkeys(own, "config"), index))
    return SiteUse(profile=profile, gave=gave, providers=tuple(providers), named_after=named_after)


def check_run_directory(run_dir: str | os.PathLike[str], use: SiteUse | None) -> None:
    """Refuse (:class:`ConfigError`, exit 78) a training run directory that the site profile named after this config's
    file name when it holds the run of another config file of that name.

    The profile names a training run's directory ``<runs>/<config file name>/<trial>``, so two projects'
    ``train.yaml`` would share one: the second would resume the first's checkpoints as its own.  A run records the
    file its directory is named after (``lock.json``: ``site.named_after``), and a run directory named so belongs to
    that file alone.  A run directory the config names itself (``log_path`` with ``stats.experiment_name``) is never
    checked: sharing it is the config's choice.  An evaluation's directory is never named after its file: each start
    of one that names none gets a new one (:func:`platoon.eval.run.prepare_output_dir`).
    """

    if use is None or use.named_after is None:
        return
    import json

    from platoon.utils.config import LOCK_NAME

    try:
        lock = json.loads((Path(run_dir) / LOCK_NAME).read_text())
    except (OSError, ValueError):
        return  # no record yet, or none to compare with
    site = lock.get("site") if isinstance(lock, dict) else None
    other = site.get("named_after") if isinstance(site, dict) else None
    if not isinstance(other, str) or other == use.named_after:
        return
    raise ConfigError(
        f"{run_dir} holds the training run of {other}: the site profile names a training run's directory after its "
        f"config file's name, and {use.named_after} has the same name, so it would take that run for its own; name "
        "this config's experiment (stats.experiment_name=...), or give it a log_path"
    )


def _is_harbor(service: Any) -> bool:
    """Whether a service reference builds the Harbor service (or a subclass of it)."""

    if not isinstance(service, str) or not service:
        return False
    return _builds_harbor(service)


@functools.lru_cache(maxsize=64)
def _builds_harbor(service: str) -> bool:
    from platoon.envs.service import service_target
    from platoon.registry import import_from_string

    target = service_target(service)
    if target is None or service.startswith("python://"):
        return False
    try:
        factory = import_from_string(target)
        from platoon.envs.harbor.service import HarborEnvironmentService
    except Exception:  # a reference that does not import is reported where the service resolves
        return False
    return isinstance(factory, type) and issubclass(factory, HarborEnvironmentService)


@functools.lru_cache(maxsize=64)
def _default_provider(service: str) -> str:
    """The provider a Harbor service class starts sandboxes with when its entry names none: its ``provider``
    parameter's default, ``docker`` for :class:`~platoon.envs.harbor.service.HarborEnvironmentService`."""

    import inspect

    from platoon.envs.service import service_target
    from platoon.registry import import_from_string

    target = service_target(service)
    if target is None:
        return DEFAULT_PROVIDER
    try:
        default = inspect.signature(import_from_string(target)).parameters["provider"].default
    except Exception:  # a factory without the parameter starts what Harbor's does
        return DEFAULT_PROVIDER
    return default if isinstance(default, str) and default else DEFAULT_PROVIDER
