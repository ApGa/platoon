"""Harnesses as data: a ``HarnessSpec`` describes how to install a CLI harness,
launch it headlessly against an OpenAI-compatible endpoint, turn its built-in
tools off, and read its final message.

Specs are TOML files. String values in ``[launch]``, ``[builtins.disable]``,
``[mcp]`` and ``[final_message]`` are templates: ``{name}`` is replaced for each
placeholder in :data:`PLACEHOLDERS`, any other ``{lowercase_identifier}`` is
rejected when the spec loads, and all other text (shell ``${VAR}``, JSON braces)
is literal. The agent key ``{api_key}`` may only be rendered into
``[launch.env]`` and ``[mcp.env]``, so it never appears in a process listing or a
file. Install commands are plain shell.

``[options]`` declares the spec's options (RFC 0016), one table each::

    [options.num_retries]
    type = "int"            # bool, int, float or str
    default = 0             # required: an option always has a value
    min = 0                 # bounds of an int or float option, inclusive
    max = 10
    description = "Most attempts at a failed model call"
    # allowed = [...]       # the values an int or str option may take

A config sets them in an ``environments`` entry's ``harness_options`` (and
``eval_overrides.harness_options``), checked against the table when the config
loads (:meth:`HarnessSpec.option_problems`) and again when a launch renders them
(:meth:`HarnessSpec.resolve_options`).  ``{options.NAME}`` renders one option's
value (a boolean as ``true`` or ``false``, a number in decimal, a string as it is)
and ``{options}`` all of them, defaults included, as one JSON object, in
``[launch]`` argv, env and files, ``[mcp]`` and ``[builtins.disable]``; never in
``[install]``, so one toolchain serves every setting.  A spec without ``[options]``
takes none: any key set for it is an error.

``[mcp]`` says how the harness takes the env's tools over MCP. When it is
enabled, the rollout serves the env session's tools at the agent's MCP endpoint
and the table applies: ``args`` are appended to argv, ``env`` is added, and
``config_template`` is written to ``config_file`` for harnesses that read an MCP
config file. ``{mcp_url}`` (the endpoint; clients send the agent key as a
Bearer token, e.g. from ``PLATOON_API_KEY``) and ``{mcp_config_path}`` render
only there.

``[install] toolchain`` is an absolute path, or the name of a toolchain under the
environment's toolchain root (``toolchains`` in the site profile's cache, else
``~/.cache/platoon/toolchains``), which a rollout stages and mounts read-only at
its build path (:mod:`platoon.harness.toolchains`) and ``{toolchain}`` renders as.
``[install.recipe]`` says what ``platoon prepare`` (or ``platoon toolchain build``)
installs there: a
uv-managed CPython, a hash-locked package closure shipped next to the spec,
caches and generated files. ``[resources]
memory_mb`` is the memory the harness's own processes need, which a task
sandbox's memory limit grows by. ``[budget] enforce = "gateway"`` makes the model
gateway end the agent after its step budget (``{max_steps}``), for harnesses that
have no step limit of their own.
"""

from __future__ import annotations

import decimal
import json
import math
import posixpath
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, get_args

from platoon.extensions import NAME_PATTERN, NAME_RULE, VERSION_PATTERN
from platoon.harness.toolchains import check_name as check_toolchain_name
from platoon.harness.toolchains import is_named, toolchain_root

Dialect = Literal["openai_chat", "openai_responses", "anthropic_messages"]
ExitStatus = Literal["completed", "error"]
# Who stops an agent at its step budget: the harness itself (it renders {max_steps}), or the model gateway.
BudgetEnforcement = Literal["harness", "gateway"]
# The types an option of ``[options]`` may have.
OptionType = Literal["bool", "int", "float", "str"]
# Caches ``platoon toolchain build`` fills inside a toolchain (``[install.recipe] cache``).
TOOLCHAIN_CACHES = ("tiktoken",)

PLACEHOLDERS = frozenset(
    {"instruction", "base_url", "api_key", "model", "workdir", "state_dir", "toolchain", "max_steps"}
)
SECRET_PLACEHOLDER = "api_key"
# Defined only when the agent serves tools over MCP, so they render only into [mcp].
MCP_PLACEHOLDERS = frozenset({"mcp_url", "mcp_config_path"})
DEFAULT_MCP_CONFIG_FILE = "mcp.json"
# ``{options}``, every option as one JSON object, and ``{options.NAME}``, one option's value.
OPTIONS_PLACEHOLDER = "options"

_IDENTIFIER = r"[a-z_][a-z0-9_]*"
_TEMPLATE_FIELD = re.compile(rf"\{{({_IDENTIFIER}(?:\.{_IDENTIFIER})?)\}}")
_OPTION_NAME = re.compile(_IDENTIFIER)
_PYTHON_VERSION = re.compile(r"^3\.\d+(\.\d+)?$")


class HarnessSpecError(ValueError):
    """A harness spec is malformed, or cannot launch with a run's settings."""


class _MissingValue(HarnessSpecError):
    """A template references a placeholder that has no value in this launch."""

    def __init__(self, name: str) -> None:
        super().__init__(f"no value for the placeholder {{{name}}}")
        self.name = name


def template_fields(template: str) -> set[str]:
    """Placeholder names a template references."""

    return set(_TEMPLATE_FIELD.findall(template))


def render_template(template: str, values: Mapping[str, str]) -> str:
    """Substitute placeholders in one pass, so substituted text is never re-scanned."""

    def value(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise _MissingValue(name)
        return values[name]

    return _TEMPLATE_FIELD.sub(value, template)


_TYPE_NAMES: dict[str, str] = {"bool": "a bool", "int": "an int", "float": "a float", "str": "a str"}


@dataclass(frozen=True)
class OptionSpec:
    """One option a spec declares (``[options.NAME]``): its type, its default, what it does and the values it takes.

    A value has the option's type exactly: a boolean is not an int, while an int is a valid float (and becomes one).
    ``allowed`` lists the values an int or str option may take; ``min`` and ``max`` bound an int or float option,
    inclusive.
    """

    name: str
    type: OptionType
    default: bool | int | float | str
    description: str = ""
    allowed: tuple[int | str, ...] | None = None
    min: int | float | None = None
    max: int | float | None = None

    def problems(self, value: Any, where: str) -> list[str]:
        """What is wrong with ``value`` as this option's value, as one line each; ``where`` is its dotted path."""

        if not _has_type(value, self.type):
            return [f"{where}: {value!r} is not {_TYPE_NAMES[self.type]}"]
        if self.type == "float" and not math.isfinite(value):
            return [f"{where}: {value!r} is not a finite number"]
        if self.allowed is not None and value not in self.allowed:
            return [f"{where}: {value!r} is not one of {', '.join(map(repr, self.allowed))}"]
        if self.min is not None and value < self.min:
            return [f"{where}: {value!r} is below the minimum {self.min!r}"]
        if self.max is not None and value > self.max:
            return [f"{where}: {value!r} is above the maximum {self.max!r}"]
        return []

    def coerce(self, value: Any) -> bool | int | float | str:
        """A checked value as the option holds it: an int given for a float option becomes a float."""

        return float(value) if self.type == "float" else value

    def render(self, value: Any) -> str:
        """The value as ``{options.NAME}`` renders it: ``true``/``false``, a number in decimal, a string as it is."""

        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return format(decimal.Decimal(repr(value)), "f")
        return str(value)

    def constraint(self) -> str:
        """The values it takes beyond its type, for listings: ``one of ...``, a range, or ``-``."""

        if self.allowed is not None:
            return "one of " + ", ".join(self.render(item) for item in self.allowed)
        if self.min is not None and self.max is not None:
            return f"{self.render(self.min)} to {self.render(self.max)}"
        if self.min is not None:
            return f">= {self.render(self.min)}"
        if self.max is not None:
            return f"<= {self.render(self.max)}"
        return "-"


def _has_type(value: Any, kind: str) -> bool:
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, str)


@dataclass(frozen=True)
class Knob:
    """Arguments appended to argv plus environment variables, applied together."""

    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolchainRecipe:
    """What ``platoon toolchain build`` installs into the spec's toolchain (``[install.recipe]``).

    ``python`` is the CPython version uv installs (python-build-standalone, so images need glibc); its
    ``bin/python3`` becomes the toolchain's interpreter.  ``lock`` is a hash-locked requirements file
    (``uv pip compile --generate-hashes``) shipped next to the spec, installed into that interpreter without a
    virtual environment.  ``cache`` names caches filled at build time (:data:`TOOLCHAIN_CACHES`: ``tiktoken``
    goes to ``share/tiktoken``), since a task sandbox may have no network.  ``generate`` maps a path inside the
    toolchain to a script shipped next to the spec, run with the toolchain's own interpreter and the path as its
    argument, to write that file (a harness's settings, made with the harness's own library).  Paths resolve
    relative to the spec file when it loads.
    """

    python: str
    lock: Path
    cache: tuple[str, ...] = ()
    # Not hashed: the runner keys installs by InstallSpec, and a mapping is not hashable.
    generate: Mapping[str, Path] = field(default_factory=dict, hash=False)


@dataclass(frozen=True)
class InstallSpec:
    """Shell commands run where the agent runs before each launch, and an optional toolchain.

    ``toolchain`` is a directory that must exist where the harness runs: an
    absolute path, or the name of a toolchain under the environment's toolchain
    root, which the rollout stages and mounts read-only at its build path
    (:mod:`platoon.harness.toolchains`; :func:`resolve_toolchain` for a launch
    outside a rollout).  Templates reach it as ``{toolchain}``.  ``recipe`` says
    how ``platoon toolchain build`` builds it.
    """

    commands: tuple[str, ...] = ()
    toolchain: str | None = None
    timeout_s: float = 1200.0
    recipe: ToolchainRecipe | None = None


def resolve_toolchain(toolchain: str, root: str | None = None) -> str:
    """The directory of ``[install] toolchain``: an absolute path as written, a name under ``root`` (default: the
    toolchain root, :func:`platoon.harness.toolchains.toolchain_root`: ``toolchains`` in the site profile's cache,
    else ``~/.cache/platoon/toolchains``)."""

    if not is_named(toolchain):
        return toolchain
    return posixpath.join(str(toolchain_root(root or None)), toolchain)


@dataclass(frozen=True)
class ResourcesSpec:
    """What the harness's own processes need where it runs (``[resources]``).

    ``memory_mb`` is added to a task sandbox's memory limit, so a watchdog that kills the largest process does
    not kill the harness for memory the task's own limit never counted.
    """

    memory_mb: int | None = None


@dataclass(frozen=True)
class BudgetSpec:
    """Who ends the agent at its step budget (``[budget]``).

    ``harness`` (default): the harness itself, from ``{max_steps}``.  ``gateway``: the model gateway answers the
    agent's policy calls past its step budget (the run's ``max_steps``, else the spec's) with a final assistant
    message without tool calls, which is not recorded; the episode is then graded with reason ``budget``.  For
    harnesses with no step limit of their own (RFC 0009, option A).
    """

    enforce: BudgetEnforcement = "harness"


@dataclass(frozen=True)
class LaunchSpec:
    """How to start the harness.

    ``files`` maps path templates to content templates written before launch;
    ``assets`` maps path templates to files shipped next to the spec (resolved
    at load time), copied before launch. Relative target paths are relative to
    the agent's state directory.
    """

    argv: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)
    files: Mapping[str, str] = field(default_factory=dict)
    assets: Mapping[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class BuiltinsSpec:
    """The harness's own tools, and the knob that turns them off (absent: they cannot be)."""

    tools: tuple[str, ...] = ()
    disable: Knob | None = None


@dataclass(frozen=True)
class McpSpec:
    """How the harness takes the agent's env tools over MCP (streamable HTTP, Bearer agent key).

    Applied only when the agent serves tools: ``args`` are appended to argv after any
    built-ins knob, ``env`` is added to the environment, and ``config_template`` is
    rendered to ``config_file`` (relative paths are in the state directory), whose
    path is ``{mcp_config_path}``.
    """

    enabled: bool = False
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    config_template: str | None = None
    config_file: str = DEFAULT_MCP_CONFIG_FILE


@dataclass(frozen=True)
class FinalMessageSpec:
    """Where the final message is: a file (optionally a dotted key of its JSON) or a stdout regex."""

    file: str | None = None
    json_key: str | None = None
    stdout_regex: str | None = None


@dataclass(frozen=True)
class RenderContext:
    """Values for one launch of one agent."""

    instruction: str
    base_url: str
    api_key: str = field(repr=False)
    model: str
    workdir: str
    state_dir: str
    toolchain: str | None = None
    max_steps: int | None = None
    mcp_url: str | None = None  # set when the agent serves its env tools over MCP
    # The entry's harness_options, as the route carries them: checked against the spec's [options] and completed
    # with their defaults when the launch renders.
    options: Mapping[str, Any] | None = None

    def values(self) -> dict[str, str]:
        values = {
            "instruction": self.instruction,
            "base_url": self.base_url.rstrip("/"),
            "api_key": self.api_key,
            "model": self.model,
            "workdir": self.workdir,
            "state_dir": self.state_dir,
            "toolchain": self.toolchain or "",
        }
        if self.max_steps is not None:
            values["max_steps"] = str(self.max_steps)
        if self.mcp_url is not None:
            values["mcp_url"] = self.mcp_url
        return values


@dataclass(frozen=True)
class RenderedLaunch:
    """A launch with every template filled in; file paths are as the spec wrote them."""

    argv: tuple[str, ...]
    env: dict[str, str] = field(repr=False)
    files: dict[str, str] = field(repr=False)
    assets: dict[str, Path]
    final_message_file: str | None


@dataclass(frozen=True)
class HarnessSpec:
    name: str
    version: str
    launch: LaunchSpec
    dialect: Dialect = "openai_chat"
    timeout_s: float | None = None
    install: InstallSpec = field(default_factory=InstallSpec)
    builtins: BuiltinsSpec = field(default_factory=BuiltinsSpec)
    mcp: McpSpec = field(default_factory=McpSpec)
    final_message: FinalMessageSpec | None = None
    exit_codes: Mapping[int, ExitStatus] = field(default_factory=dict)
    max_steps: int | None = None  # the step budget {max_steps} gets when the run gives none
    resources: ResourcesSpec = field(default_factory=ResourcesSpec)
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    # The options it declares (``[options]``), by name, in the spec's order.
    options: Mapping[str, OptionSpec] = field(default_factory=dict)
    source: str | None = field(default=None, compare=False)

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def option_problems(self, values: Any, where: str = "harness_options") -> list[str]:
        """Every problem with ``values`` as this spec's options, one line each with its dotted path under
        ``where``: an unknown key (with a did-you-mean), a value of the wrong type, one outside ``allowed`` or the
        range.  A spec without ``[options]`` takes none, so any key is a problem."""

        from platoon.utils.config import unknown_key

        if values is None:
            return []
        if not isinstance(values, Mapping):
            return [f"{where}: expected a mapping of options, got {values!r}"]
        problems: list[str] = []
        for key, value in values.items():
            option = self.options.get(key) if isinstance(key, str) else None
            if option is None:
                takes = f"harness {self.ref} takes " + (", ".join(self.options) if self.options else "no options")
                problems.append(unknown_key(key, self.options, where, otherwise=takes))
            else:
                problems.extend(option.problems(value, f"{where}.{key}"))
        return problems

    def resolve_options(self, values: Mapping[str, Any] | None, where: str = "harness_options") -> dict[str, Any]:
        """``values`` checked (:meth:`option_problems`) and completed with the defaults: every option the spec
        declares, in its order.  Raises ``HarnessSpecError`` with every problem."""

        problems = self.option_problems(values, where)
        if problems:
            raise HarnessSpecError(f"harness {self.ref}: " + "; ".join(problems))
        given = dict(values or {})
        return {name: option.coerce(given.get(name, option.default)) for name, option in self.options.items()}

    def exit_status(self, exit_code: int | None) -> ExitStatus:
        """``completed`` for exit 0 unless remapped; anything unmapped is an ``error``."""

        if exit_code is None:
            return "error"
        return self.exit_codes.get(exit_code, "completed" if exit_code == 0 else "error")

    def render(self, context: RenderContext, *, disable_builtins: bool = False) -> RenderedLaunch:
        """Fill in every template; ``disable_builtins`` applies ``[builtins.disable]``, and
        ``[mcp]`` applies when it is enabled and ``context`` carries an MCP URL.

        ``{max_steps}`` is the run's step budget (``context.max_steps``), else the spec's ``max_steps``; a launch
        that renders it with neither is a ``HarnessSpecError`` that names the missing settings.  ``{options}`` and
        ``{options.NAME}`` render ``context.options`` completed with the defaults (:meth:`resolve_options`); options
        the spec does not take, or that break its rules, are a ``HarnessSpecError``.
        """

        knob = self.builtins.disable if disable_builtins else None
        if disable_builtins and knob is None:
            raise HarnessSpecError(f"harness {self.ref} has no knob to disable its built-in tools")
        values = context.values()
        if "max_steps" not in values and self.max_steps is not None:
            values["max_steps"] = str(self.max_steps)  # the spec's default when the run sets no budget
        options = self.resolve_options(context.options)
        values[OPTIONS_PLACEHOLDER] = json.dumps(options, sort_keys=True)
        values.update(
            {f"{OPTIONS_PLACEHOLDER}.{name}": self.options[name].render(value) for name, value in options.items()}
        )
        try:
            return self._render(context, values, knob)
        except _MissingValue as exc:
            if exc.name != "max_steps":
                raise
            raise HarnessSpecError(
                f"harness {self.ref} renders {{max_steps}}, but the run sets no step budget (rollout_config.max_steps "
                "or the task's max_steps) and the spec no max_steps default"
            ) from None

    def _render(self, context: RenderContext, values: dict[str, str], knob: Knob | None) -> RenderedLaunch:
        argv = [render_template(arg, values) for arg in self.launch.argv]
        env = {key: render_template(value, values) for key, value in self.launch.env.items()}
        if knob is not None:
            argv.extend(render_template(arg, values) for arg in knob.args)
            env.update({key: render_template(value, values) for key, value in knob.env.items()})
        files = {
            render_template(path, values): render_template(content, values)
            for path, content in self.launch.files.items()
        }
        if self.mcp.enabled and context.mcp_url is not None:
            if self.mcp.config_template is not None:
                config_file = render_template(self.mcp.config_file, values)
                values["mcp_config_path"] = posixpath.join(context.state_dir, config_file)
                files[config_file] = render_template(self.mcp.config_template, values)
            argv.extend(render_template(arg, values) for arg in self.mcp.args)
            env.update({key: render_template(value, values) for key, value in self.mcp.env.items()})
        final_message_file = self.final_message.file if self.final_message is not None else None
        return RenderedLaunch(
            argv=tuple(argv),
            env=env,
            files=files,
            assets={render_template(path, values): source for path, source in self.launch.assets.items()},
            final_message_file=render_template(final_message_file, values) if final_message_file else None,
        )


def load_spec(path: str | Path) -> HarnessSpec:
    """Load and validate one TOML spec; asset paths resolve relative to the file."""

    path = Path(path)
    with path.open("rb") as handle:
        try:
            data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise HarnessSpecError(f"{path}: invalid TOML: {exc}") from exc
    return parse_spec(data, base_dir=path.parent, source=str(path))


def parse_spec(data: Mapping[str, Any], *, base_dir: Path | None = None, source: str = "<spec>") -> HarnessSpec:
    """Validate a spec mapping (the parsed TOML document)."""

    table = _Table(data, source)
    name = table.string("name")
    version = table.string("version")
    if not NAME_PATTERN.fullmatch(name) or name.endswith(".toml"):
        raise HarnessSpecError(f"{source}: name {name!r} must match the naming rule: {NAME_RULE}")
    if not VERSION_PATTERN.fullmatch(version):
        raise HarnessSpecError(f"{source}: version {version!r} must match {VERSION_PATTERN.pattern}")
    dialect = table.string("dialect", "openai_chat")
    if dialect not in get_args(Dialect):
        raise HarnessSpecError(f"{source}: dialect {dialect!r} is not one of {get_args(Dialect)}")
    timeout_s = table.positive_number("timeout_s", None)
    max_steps = table.positive_number("max_steps", None)
    if max_steps is not None and max_steps != int(max_steps):
        raise HarnessSpecError(f"{source}: max_steps must be a whole number, got {max_steps!r}")

    options = _parse_options(table.optional_table("options"), source)
    install = _parse_install(table.table("install"), f"{source}: [install]", base_dir)
    allowed = PLACEHOLDERS if install.toolchain is not None else PLACEHOLDERS - {"toolchain"}
    public = allowed - {SECRET_PLACEHOLDER}
    # Options render where the run's own values do, but never into [install] or where a file is found.
    with_options = frozenset({OPTIONS_PLACEHOLDER, *(f"{OPTIONS_PLACEHOLDER}.{name}" for name in options)})
    launch = _parse_launch(
        table.table("launch", required=True), f"{source}: [launch]", allowed, public, base_dir, with_options
    )
    builtins = _parse_builtins(table.table("builtins"), f"{source}: [builtins]", public | with_options)
    mcp = _parse_mcp(table.optional_table("mcp"), f"{source}: [mcp]", allowed | with_options, public | with_options)
    final_message = _parse_final_message(table.optional_table("final_message"), f"{source}: [final_message]", public)
    exit_codes = _parse_exit_codes(table.table("exit_codes"), f"{source}: [exit_codes]")
    resources = _parse_resources(table.table("resources"), f"{source}: [resources]")
    budget = _parse_budget(table.table("budget"), f"{source}: [budget]")
    table.finish()
    return HarnessSpec(
        name=name,
        version=version,
        dialect=dialect,
        timeout_s=timeout_s,
        install=install,
        launch=launch,
        builtins=builtins,
        mcp=mcp,
        final_message=final_message,
        exit_codes=exit_codes,
        max_steps=None if max_steps is None else int(max_steps),
        resources=resources,
        budget=budget,
        options=options,
        source=source,
    )


def _parse_options(table: _Table | None, source: str) -> dict[str, OptionSpec]:
    """``[options]``: one table per option, ``[options.NAME]``."""

    if table is None:
        return {}
    options: dict[str, OptionSpec] = {}
    for name in table.keys():
        where = f"{source}: [options.{name}]"
        if not _OPTION_NAME.fullmatch(name):
            raise HarnessSpecError(f"{where}: an option's name must be a lower-case identifier, as a placeholder's is")
        options[name] = _parse_option(_Table(table.value(name), where), name, where)
    table.finish()
    return options


def _parse_option(table: _Table, name: str, where: str) -> OptionSpec:
    kind = table.string("type")
    default = table.value("default")
    description = table.string("description", "")
    allowed = table.value("allowed", None)
    minimum = table.number("min", None)
    maximum = table.number("max", None)
    table.finish()
    if kind not in get_args(OptionType):
        raise HarnessSpecError(f"{where}: type {kind!r} is not one of {get_args(OptionType)}")
    if "\n" in description:
        raise HarnessSpecError(f"{where}: description must be one line")
    if allowed is not None:
        if kind not in ("int", "str"):
            raise HarnessSpecError(f"{where}: allowed lists the values of an int or str option, not of a {kind}")
        if not isinstance(allowed, list) or not allowed or not all(_has_type(item, kind) for item in allowed):
            raise HarnessSpecError(f"{where}: allowed must be a non-empty list of {kind} values")
        if len(set(allowed)) != len(allowed):
            raise HarnessSpecError(f"{where}: allowed lists a value twice")
    if (minimum is not None or maximum is not None) and kind not in ("int", "float"):
        raise HarnessSpecError(f"{where}: min and max bound an int or float option, not a {kind}")
    if kind == "int" and any(bound is not None and not isinstance(bound, int) for bound in (minimum, maximum)):
        raise HarnessSpecError(f"{where}: the bounds of an int option must be integers")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise HarnessSpecError(f"{where}: min {minimum!r} is above max {maximum!r}")
    option = OptionSpec(
        name=name,
        type=kind,
        default=default,
        description=description,
        allowed=None if allowed is None else tuple(allowed),
        min=minimum,
        max=maximum,
    )
    if problems := option.problems(default, "default"):
        raise HarnessSpecError(f"{where}: " + "; ".join(problems))
    return replace(option, default=option.coerce(default))


def _parse_install(table: _Table, where: str, base_dir: Path | None) -> InstallSpec:
    recipe_table = table.optional_table("recipe")
    install = InstallSpec(
        commands=table.strings("commands"),
        toolchain=table.string("toolchain", None),
        timeout_s=table.positive_number("timeout_s", InstallSpec.timeout_s),
    )
    table.finish()
    toolchain = install.toolchain
    if toolchain is not None:
        try:
            check_toolchain_name(toolchain)
        except ValueError as exc:
            raise HarnessSpecError(f"{where}: {exc}") from None
    if recipe_table is None:
        return install
    if toolchain is None:
        raise HarnessSpecError(f"{where}: a recipe builds the spec's toolchain, so it needs [install] toolchain")
    recipe = _parse_recipe(recipe_table, f"{where} [recipe]", base_dir)
    return InstallSpec(commands=install.commands, toolchain=toolchain, timeout_s=install.timeout_s, recipe=recipe)


def _parse_recipe(table: _Table, where: str, base_dir: Path | None) -> ToolchainRecipe:
    python = table.string("python")
    lock = table.string("lock")
    cache = table.strings("cache")
    generate = table.string_map("generate")
    table.finish()
    if not _PYTHON_VERSION.match(python):
        raise HarnessSpecError(f"{where}: python {python!r} must be a CPython version such as '3.12'")
    if unknown := sorted(set(cache) - set(TOOLCHAIN_CACHES)):
        raise HarnessSpecError(f"{where}: unknown cache(s) {unknown}; known: {list(TOOLCHAIN_CACHES)}")
    generated: dict[str, Path] = {}
    for target, script in generate.items():
        parts = target.split("/")
        if target.startswith("/") or not target or any(part in ("", ".", "..") for part in parts):
            raise HarnessSpecError(f"{where}: generate key {target!r} must be a relative path inside the toolchain")
        generated[target] = _spec_file(script, f"{where} generate.{target!r}", base_dir)
    return ToolchainRecipe(
        python=python,
        lock=_spec_file(lock, f"{where} lock", base_dir),
        cache=cache,
        generate=generated,
    )


def _spec_file(path: str, where: str, base_dir: Path | None) -> Path:
    """A file shipped next to the spec, resolved when the spec loads."""

    source = (base_dir or Path.cwd()) / path
    if not source.is_file():
        raise HarnessSpecError(f"{where}: {path!r} not found at {source}")
    return source.resolve()


def _parse_resources(table: _Table, where: str) -> ResourcesSpec:
    memory_mb = table.positive_number("memory_mb", None)
    table.finish()
    if memory_mb is not None and memory_mb != int(memory_mb):
        raise HarnessSpecError(f"{where}: memory_mb must be a whole number of megabytes, got {memory_mb!r}")
    return ResourcesSpec(memory_mb=None if memory_mb is None else int(memory_mb))


def _parse_budget(table: _Table, where: str) -> BudgetSpec:
    enforce = table.string("enforce", BudgetSpec.enforce)
    table.finish()
    if enforce not in get_args(BudgetEnforcement):
        raise HarnessSpecError(f"{where}: enforce {enforce!r} is not one of {get_args(BudgetEnforcement)}")
    return BudgetSpec(enforce=enforce)


def _parse_launch(
    table: _Table,
    where: str,
    allowed: frozenset[str],
    public: frozenset[str],
    base_dir: Path | None,
    options: frozenset[str] = frozenset(),
) -> LaunchSpec:
    """``[launch]``; ``options`` are the option placeholders, which render in argv, env and files."""

    argv = table.strings("argv")
    if not argv:
        raise HarnessSpecError(f"{where}: argv must not be empty")
    env = table.string_map("env")
    files = table.string_map("files")
    assets = table.string_map("assets")
    table.finish()
    for arg in argv:
        _check_template(arg, f"{where} argv", public | options)
    for key, value in env.items():
        _check_template(value, f"{where} env.{key}", allowed | options)
    for path, content in files.items():
        _check_template(path, f"{where} files key", public | options)
        _check_template(content, f"{where} files.{path!r}", public | options)
    resolved_assets: dict[str, Path] = {}
    for path, asset in assets.items():
        _check_template(path, f"{where} assets key", public)
        source = (base_dir or Path.cwd()) / asset
        if not source.is_file():
            raise HarnessSpecError(f"{where}: asset {asset!r} not found at {source}")
        resolved_assets[path] = source.resolve()
    return LaunchSpec(argv=argv, env=env, files=files, assets=resolved_assets)


def _parse_builtins(table: _Table, where: str, public: frozenset[str]) -> BuiltinsSpec:
    tools = table.strings("tools")
    disable_table = table.optional_table("disable")
    table.finish()
    if disable_table is None:
        return BuiltinsSpec(tools=tools)
    disable = Knob(args=disable_table.strings("args"), env=disable_table.string_map("env"))
    disable_table.finish()
    if not disable.args and not disable.env:
        raise HarnessSpecError(f"{where}: disable must set args or env")
    for arg in disable.args:
        _check_template(arg, f"{where} disable.args", public)
    for key, value in disable.env.items():
        _check_template(value, f"{where} disable.env.{key}", public)
    return BuiltinsSpec(tools=tools, disable=disable)


def _parse_mcp(table: _Table | None, where: str, allowed: frozenset[str], public: frozenset[str]) -> McpSpec:
    if table is None:
        return McpSpec()
    spec = McpSpec(
        enabled=table.boolean("enabled", True),
        args=table.strings("args"),
        env=table.string_map("env"),
        config_template=table.string("config_template", None),
        config_file=table.string("config_file", DEFAULT_MCP_CONFIG_FILE),
    )
    table.finish()
    mcp = MCP_PLACEHOLDERS if spec.config_template is not None else MCP_PLACEHOLDERS - {"mcp_config_path"}
    for arg in spec.args:
        _check_template(arg, f"{where} args", public | mcp)
    for key, value in spec.env.items():
        _check_template(value, f"{where} env.{key}", allowed | mcp)
    if spec.config_template is not None:
        _check_template(spec.config_template, f"{where} config_template", public | {"mcp_url"})
    _check_template(spec.config_file, f"{where} config_file", public)
    return spec


def _parse_final_message(table: _Table | None, where: str, public: frozenset[str]) -> FinalMessageSpec | None:
    if table is None:
        return None
    spec = FinalMessageSpec(
        file=table.string("file", None),
        json_key=table.string("json_key", None),
        stdout_regex=table.string("stdout_regex", None),
    )
    table.finish()
    if (spec.file is None) == (spec.stdout_regex is None):
        raise HarnessSpecError(f"{where}: set exactly one of file or stdout_regex")
    if spec.json_key is not None and spec.file is None:
        raise HarnessSpecError(f"{where}: json_key requires file")
    if spec.file is not None:
        _check_template(spec.file, f"{where} file", public)
    if spec.stdout_regex is not None:
        try:
            re.compile(spec.stdout_regex)
        except re.error as exc:
            raise HarnessSpecError(f"{where}: invalid stdout_regex: {exc}") from exc
    return spec


def _parse_exit_codes(table: _Table, where: str) -> dict[int, ExitStatus]:
    exit_codes: dict[int, ExitStatus] = {}
    for key in table.keys():
        status = table.string(key)
        try:
            code = int(key)
        except ValueError:
            raise HarnessSpecError(f"{where}: key {key!r} is not an integer exit code") from None
        if status not in get_args(ExitStatus):
            raise HarnessSpecError(f"{where}: exit code {code} maps to {status!r}, not one of {get_args(ExitStatus)}")
        exit_codes[code] = status
    return exit_codes


def _check_template(template: str, where: str, allowed: frozenset[str]) -> None:
    fields = template_fields(template)
    if SECRET_PLACEHOLDER in fields and SECRET_PLACEHOLDER not in allowed:
        raise HarnessSpecError(f"{where}: {{{SECRET_PLACEHOLDER}}} may only be rendered into [launch.env]")
    unknown = fields - allowed
    if unknown:
        raise HarnessSpecError(f"{where}: unknown placeholder(s) {sorted(unknown)}; allowed: {sorted(allowed)}")


_REQUIRED = object()


class _Table:
    """Typed, strict access to one TOML table: unread keys are an error at ``finish``."""

    def __init__(self, data: Mapping[str, Any], where: str) -> None:
        if not isinstance(data, Mapping):
            raise HarnessSpecError(f"{where}: expected a table, got {type(data).__name__}")
        self._data = dict(data)
        self._where = where

    def keys(self) -> list[str]:
        return list(self._data)

    def _take(self, key: str, default: Any) -> Any:
        if key in self._data:
            return self._data.pop(key)
        if default is _REQUIRED:
            raise HarnessSpecError(f"{self._where}: missing required key {key!r}")
        return default

    def value(self, key: str, default: Any = _REQUIRED) -> Any:
        """The value of ``key``, whatever its type."""
        return self._take(key, default)

    def number(self, key: str, default: Any) -> Any:
        """A finite int or float (a boolean is neither)."""
        value = self._take(key, default)
        if value is default:
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise HarnessSpecError(f"{self._where}: {key} must be a number")
        return value

    def string(self, key: str, default: Any = _REQUIRED) -> Any:
        value = self._take(key, default)
        if value is not default and not isinstance(value, str):
            raise HarnessSpecError(f"{self._where}: {key} must be a string")
        return value

    def boolean(self, key: str, default: Any = _REQUIRED) -> Any:
        value = self._take(key, default)
        if value is not default and not isinstance(value, bool):
            raise HarnessSpecError(f"{self._where}: {key} must be true or false")
        return value

    def strings(self, key: str) -> tuple[str, ...]:
        value = self._take(key, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise HarnessSpecError(f"{self._where}: {key} must be a list of strings")
        return tuple(value)

    def string_map(self, key: str) -> dict[str, str]:
        value = self._take(key, {})
        if not isinstance(value, Mapping) or not all(isinstance(item, str) for item in value.values()):
            raise HarnessSpecError(f"{self._where}: {key} must be a table of strings")
        return dict(value)

    def positive_number(self, key: str, default: Any) -> Any:
        value = self._take(key, default)
        if value is default:
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise HarnessSpecError(f"{self._where}: {key} must be a positive number")
        return float(value)

    def table(self, key: str, *, required: bool = False) -> _Table:
        return _Table(self._take(key, _REQUIRED if required else {}), f"{self._where} [{key}]")

    def optional_table(self, key: str) -> _Table | None:
        value = self._take(key, None)
        return None if value is None else _Table(value, f"{self._where} [{key}]")

    def finish(self) -> None:
        if self._data:
            raise HarnessSpecError(f"{self._where}: unknown key(s) {sorted(self._data)}")
