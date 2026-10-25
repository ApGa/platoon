"""Build plans for Harbor task Dockerfiles: a shared build prefix and a suffix applied when a session opens.

Many Harbor tasks differ only in their last Dockerfile steps: a data-analysis pool copies each task's inputs
(``COPY input/ /data/input/``) into one common image, a math pool its instance file. A Dockerfile splits into

- the **build prefix**: every instruction up to the last one that cannot be deferred (a ``RUN``, a ``FROM``, a
  ``COPY --from``, ...). It is what gets built;
- the **deferred suffix**: the trailing ``COPY`` and ``ADD`` of local files, ``WORKDIR``, ``ENV``, ``ARG`` and
  metadata (``LABEL``, ``EXPOSE``, ``CMD``, ...). Nothing executes after them at build time, so a session applies
  them to a sandbox of the prefix's image (uploads, ``mkdir -p``, environment) with the same result.

The **build key** is a SHA-256 over the normalized prefix and the hashes of the build-context files it reads, so
tasks whose prefixes match share one image however their suffixes differ.

**Remote sources** are told apart from build-context paths as BuildKit's Dockerfile frontend tells them apart
(:func:`remote_source`), after the step's variables are expanded (:func:`variables_in_scope`): an ``ADD`` of a URL
or of a git repository, written out or through an ``ARG`` or ``ENV`` value, always stays in the prefix. The builder
fetches it as Docker does: it verifies an ``--checksum`` and never extracts a remote archive. Planning reads no
network: the key covers such a step as written, the URL (or the ``ARG`` that holds it) and its checksum. A ``COPY``
of one fails the plan, as it fails a Docker build.

Deferred steps are JSON lists: ``["WORKDIR", path]``, ``["ENV", name, value]``, ``["ARG", name, value]`` and
``["COPY", source, ..., destination]`` with sources relative to the build context. Variables in a destination,
a ``WORKDIR`` or an ``ENV`` value are expanded when the session opens, against the image's environment, as a build
would; a source must be literal. ``USER`` in the suffix is reported (``suffix_user``), never applied: rootless
sandboxes run as one uid. Anything the planner cannot defer safely stays in the prefix.

**Image configuration.** The metadata of the suffix (``CMD``, ``ENTRYPOINT``, ``HEALTHCHECK``, ``USER``, ``EXPOSE``,
``STOPSIGNAL``, ``VOLUME``) is never built, so the plan records what it sets (``config``,
:func:`image_config`) for the session to lay over the prefix image's own: Docker's rules, such as an
``ENTRYPOINT`` resetting a ``CMD`` that the stage did not set itself, and a shell form running under the stage's
``SHELL``.  A **whole** plan (``whole=True``, a compose service's build context) defers nothing: its key also hashes
the files the suffix copies, since nothing can be uploaded into a service before its command starts, and its
``config`` is what the last stage sets over its base image.  Build arguments (``build_args``, a compose service's
``build.args``) override the ``ARG`` defaults and enter the key.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import posixpath
import re
import urllib.parse
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, overload

from platoon.envs.harbor.hashing import file_sha256

BUILD_KEY_RULE = "platoon-buildkey-v1"
DOCKERFILE = "Dockerfile"
DOCKERIGNORE = ".dockerignore"
_DIRECTIVE = re.compile(r"^#\s*([a-zA-Z][a-zA-Z0-9]*)\s*=\s*(.+?)\s*$")
_HEREDOC = re.compile(r"(?<!<)<<(-?)\s*([\"']?)([A-Za-z_][A-Za-z0-9_]*)\2(?![A-Za-z0-9_<])")
_FLAG = re.compile(r"--([a-zA-Z][a-zA-Z0-9-]*)(?:=(\S*))?(?:\s+|$)")
_VARIABLE = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?[-+])([^}]*))?\}|([A-Za-z_][A-Za-z0-9_]*))")
METADATA = frozenset(
    {"LABEL", "EXPOSE", "CMD", "ENTRYPOINT", "HEALTHCHECK", "STOPSIGNAL", "VOLUME", "SHELL", "MAINTAINER", "ONBUILD"}
)
_SHELL_ESCAPES = {'"': '\\"', "`": "\\`", "\\": "\\\\", "$": "\\$"}  # literal in a double-quoted shell word
_COPY_FLAGS_DEFERRABLE = frozenset({"chown", "link"})  # same files in a single-uid sandbox
_ARCHIVE_MAGIC = (b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00", b"\x28\xb5\x2f\xfd")
# BuildKit's git transports: a URL scheme (http and https only with a path ending in .git), or ssh's user@host:path.
_URL_SCHEME = re.compile(r"^([A-Za-z0-9]+)://")
_SCP_LIKE = re.compile(r"^[A-Za-z0-9_-]+@[A-Za-z0-9.-]+:")
_REMOTE = {"git": "a git repository", "url": "a URL"}


class DockerfileError(ValueError):
    """A Dockerfile platoon cannot plan: no ``FROM``, an unterminated heredoc, a missing ``COPY`` source, ..."""


@dataclass(frozen=True)
class Instruction:
    keyword: str  # upper case: FROM, RUN, COPY, ...
    arguments: str  # the instruction's text after the keyword, continuation lines joined
    line: int  # 1-based line of the keyword
    heredocs: tuple[str, ...] = ()  # bodies of the instruction's here-documents, in order
    text: str = ""  # the instruction as written (comments dropped), for rebuilding a Dockerfile

    @property
    def normalized(self) -> str:
        """The instruction as it enters a build key: keyword, arguments and here-document bodies."""

        return "\n".join([f"{self.keyword} {self.arguments}".strip(), *self.heredocs])

    def flags_and_operands(self) -> tuple[dict[str, str], list[str]]:
        """Leading ``--name[=value]`` flags, and the operands after them (a JSON array or whitespace split)."""

        rest = self.arguments.strip()
        flags: dict[str, str] = {}
        while match := _FLAG.match(rest):
            flags[match.group(1).lower()] = match.group(2) or ""
            rest = rest[match.end() :]
        if rest.startswith("["):
            try:
                operands = json.loads(rest)
            except json.JSONDecodeError:
                operands = None
            if isinstance(operands, list) and all(isinstance(item, str) for item in operands):
                return flags, operands
        return flags, rest.split()


@dataclass(frozen=True)
class Dockerfile:
    directives: tuple[str, ...]  # parser directive lines (``# syntax=...``, ``# escape=...``), as written
    instructions: tuple[Instruction, ...]


@dataclass(frozen=True)
class DockerfilePlan:
    """How one Dockerfile builds: the prefix that is built, and the steps a session applies."""

    key: str  # "sha256:<hex>" over the normalized prefix and the context files it reads
    directives: tuple[str, ...]
    prefix: tuple[Instruction, ...]
    deferred: tuple[tuple[str, ...], ...]
    base_images: tuple[str, ...]  # the FROM references of the prefix, ARG defaults expanded
    suffix_user: str | None = None  # a USER in the deferred suffix: rootless sandboxes cannot honor it
    context_files: tuple[str, ...] = ()  # context paths the prefix reads ("." for the whole context)
    notes: tuple[str, ...] = field(default=())  # why steps stayed in the prefix, for `images plan`
    # What the suffix's metadata sets in the image configuration (a whole plan: what the last stage sets over its
    # base), as image_config gives it; empty when it sets nothing.
    config: Mapping[str, Any] = field(default_factory=dict)
    whole: bool = False  # nothing deferred: a compose service's image

    @property
    def copied(self) -> tuple[str, ...]:
        """The build-context paths the image takes, each once: what the prefix reads (``context_files``, built into
        the image; ``.`` for the whole context), then the sources of the deferred ``COPY`` steps, which a session
        uploads.  Relative to the context, without a trailing slash."""

        sources = [source.rstrip("/") or "." for step in self.deferred if step[0] == "COPY" for source in step[1:-1]]
        return tuple(dict.fromkeys([*self.context_files, *sources]))

    def prefix_dockerfile(self) -> str:
        """A Dockerfile holding only the prefix: what a builder builds."""

        return "\n".join([*self.directives, *(instruction.text for instruction in self.prefix)]) + "\n"

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "deferred": [list(step) for step in self.deferred],
            "base": list(self.base_images),
            **({"suffix_user": self.suffix_user} if self.suffix_user else {}),
            **({"config": dict(self.config)} if self.config else {}),
            **({"whole": True} if self.whole else {}),
        }


def parse_dockerfile(text: str) -> Dockerfile:
    """A Dockerfile's parser directives and instructions, with BuildKit's line rules: directives (``# escape=``)
    first, comment lines dropped (also inside continued instructions), line continuations and here-documents."""

    lines = text.splitlines()
    escape = "\\"
    directives: list[str] = []
    index = 0
    while index < len(lines):  # parser directives come first, before any comment, blank line or instruction
        match = _DIRECTIVE.match(lines[index].strip())
        if not match:
            break
        directives.append(lines[index].strip())
        if match.group(1).lower() == "escape":
            escape = match.group(2)
            if escape not in ("\\", "`"):
                raise DockerfileError(f"invalid escape directive {escape!r}")
        index += 1
    continuation = re.compile(re.escape(escape) + r"[ \t]*$")
    instructions: list[Instruction] = []
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        start = index
        parts: list[str] = []
        raw: list[str] = []
        line = lines[index]
        while True:
            raw.append(line)
            if continuation.search(line):
                parts.append(continuation.sub("", line))
                index += 1
                while index < len(lines) and (not lines[index].strip() or lines[index].lstrip().startswith("#")):
                    index += 1  # comment and empty lines inside a continued instruction are dropped
                if index >= len(lines):
                    break
                line = lines[index]
                continue
            parts.append(line)
            index += 1
            break
        joined = "".join(parts).strip()
        keyword, arguments = (re.split(r"\s+", joined, maxsplit=1) + [""])[:2]
        keyword = keyword.upper()
        heredocs: list[str] = []
        if keyword in ("RUN", "COPY", "ADD"):
            for match in _HEREDOC.finditer(arguments):
                strip_tabs, delimiter = match.group(1) == "-", match.group(3)
                body: list[str] = []
                while True:
                    if index >= len(lines):
                        raise DockerfileError(f"line {start + 1}: here-document {delimiter!r} is not terminated")
                    current = lines[index]
                    raw.append(current)
                    index += 1
                    if (current.lstrip("\t") if strip_tabs else current) == delimiter:
                        break
                    body.append(current.lstrip("\t") if strip_tabs else current)
                heredocs.append("\n".join(body))
        instructions.append(
            Instruction(
                keyword=keyword,
                arguments=arguments.strip(),
                line=start + 1,
                heredocs=tuple(heredocs),
                text="\n".join(raw),
            )
        )
    return Dockerfile(directives=tuple(directives), instructions=tuple(instructions))


def plan_dockerfile(
    dockerfile: str | Path,
    context: str | Path | None = None,
    *,
    with_key: bool = True,
    whole: bool = False,
    build_args: Mapping[str, str] | None = None,
) -> DockerfilePlan:
    """The build plan of ``dockerfile`` with build context ``context`` (default: the Dockerfile's directory).

    ``with_key=False`` leaves the build key empty, and so reads no file of the context: for what the plan copies
    (:attr:`DockerfilePlan.copied`) alone.  ``whole=True`` defers nothing (a compose service's image), and
    ``build_args`` override the ``ARG`` defaults, as ``docker build --build-arg`` does; both enter the key.
    """

    dockerfile = Path(dockerfile)
    context = Path(context) if context is not None else dockerfile.parent
    parsed = parse_dockerfile(dockerfile.read_text())
    instructions = list(parsed.instructions)
    if not instructions or not any(instruction.keyword == "FROM" for instruction in instructions):
        raise DockerfileError(f"{dockerfile} has no FROM instruction")
    if instructions[0].keyword not in ("FROM", "ARG"):
        raise DockerfileError(
            f"{dockerfile}: the first instruction must be FROM (or ARG), got {instructions[0].keyword}"
        )
    args = dict(build_args or {})
    has_ignore = (context / DOCKERIGNORE).is_file()
    scoped = list(variables_in_scope(instructions, build_args=args))
    split = len(instructions)
    notes: list[str] = []
    while split > 0 and not whole:
        why = _not_deferrable(*scoped[split - 1], context, has_ignore)
        if why is not None:
            if why:
                notes.append(f"line {instructions[split - 1].line}: {why}")
            break
        split -= 1
    prefix, suffix = instructions[:split], instructions[split:]
    scope = _Scope.of(prefix, build_args=args)
    deferred: list[tuple[str, ...]] = [("ARG", name, value) for name, value in scope.stage_args.items()]
    suffix_user = None
    for instruction in suffix:
        steps, user = _deferred_steps(instruction, scope)
        deferred += steps
        suffix_user = user or suffix_user
    context_files = _prefix_context(scoped[:split], context, has_ignore)
    key = _build_key(parsed.directives, prefix, context, context_files, has_ignore, args) if with_key else ""
    return DockerfilePlan(
        key=key,
        directives=parsed.directives,
        prefix=tuple(prefix),
        deferred=tuple(deferred) if not whole else (),
        base_images=tuple(scope.bases),
        suffix_user=suffix_user,
        context_files=tuple(context_files),
        notes=tuple(notes),
        config=image_config(scoped, start=0 if whole else split),
        whole=whole,
    )


#: The shell of a shell-form ``CMD``, ``ENTRYPOINT`` or ``RUN`` in a Linux image without a ``SHELL`` instruction.
DEFAULT_SHELL = ("/bin/sh", "-c")


def image_config(scoped: Sequence[tuple[Instruction, Mapping[str, str]]], *, start: int = 0) -> dict[str, Any]:
    """What the metadata instructions of ``scoped[start:]`` (each with the variables in scope at it,
    :func:`variables_in_scope`) set in the image configuration of the last stage, by Docker's rules.

    The keys are those set: ``entrypoint`` and ``cmd`` (argv lists; a shell form under the stage's ``SHELL``, else
    ``/bin/sh -c``), ``healthcheck`` (``{"test": [...], "interval": ...}`` with Docker's option names, ``test``
    ``["NONE"]`` for ``HEALTHCHECK NONE``), ``user``, ``expose`` (ports added, as ``"8000/tcp"``),
    ``stop_signal`` and ``volumes`` (added).  An ``ENTRYPOINT`` empties the ``CMD`` the stage inherits unless the stage
    sets one before it (Docker's ``cmdSet``), which ``cmd`` records as ``[]``.  ``start`` is where a deferred suffix
    starts: what the stage set before it is the built prefix's own configuration, but a ``CMD`` there still keeps a
    later ``ENTRYPOINT`` from emptying it, and a ``SHELL`` there still holds."""

    configs: dict[str, dict[str, Any]] = {}  # stage name -> what it set, for a stage built FROM it
    shells: dict[str, tuple[str, ...]] = {}  # stage name -> its SHELL at its end
    config: dict[str, Any] = {}
    shell = DEFAULT_SHELL
    stage: str | None = None
    cmd_set = False  # the stage set a CMD itself (Docker's cmdSet)
    for index, (instruction, variables) in enumerate(scoped):
        keyword = instruction.keyword
        applies = index >= start
        if keyword == "FROM":
            if stage is not None:
                configs[stage], shells[stage] = config, shell
            _, operands = instruction.flags_and_operands()
            parent = operands[0].lower() if operands else ""
            # A stage built FROM an earlier one inherits its configuration; a suffix never holds a FROM, so a plan
            # with a suffix records the suffix's own settings alone.
            config = json.loads(json.dumps(configs.get(parent, {}))) if applies else {}
            shell = shells.get(parent, DEFAULT_SHELL)
            stage = operands[2].lower() if len(operands) >= 3 and operands[1].lower() == "as" else None
            cmd_set = False
        elif keyword == "SHELL":
            shell = _shell_operands(instruction)
        elif keyword == "CMD":
            cmd_set = True
            if applies:
                config["cmd"] = _command_form(instruction.arguments, shell)
        elif not applies:
            continue
        elif keyword == "ENTRYPOINT":
            config["entrypoint"] = _command_form(instruction.arguments, shell)
            if not cmd_set:
                config["cmd"] = []
        elif keyword == "HEALTHCHECK":
            config["healthcheck"] = _healthcheck(instruction)
        elif keyword == "USER":
            config["user"] = expand(instruction.arguments.strip(), variables)
        elif keyword == "EXPOSE":
            ports = [expand(word, variables) for word in instruction.arguments.split()]
            config.setdefault("expose", []).extend(port if "/" in port else f"{port}/tcp" for port in ports)
        elif keyword == "STOPSIGNAL":
            config["stop_signal"] = expand(instruction.arguments.strip(), variables)
        elif keyword == "VOLUME":
            _, operands = instruction.flags_and_operands()
            config.setdefault("volumes", []).extend(expand(path, variables) for path in operands)
    return config


def _exec_form(text: str) -> list[str] | None:
    """An instruction's exec form, a JSON array of strings; None for its shell form."""

    text = text.strip()
    if not text.startswith("["):
        return None
    try:
        words = json.loads(text)
    except json.JSONDecodeError:
        return None
    return words if isinstance(words, list) and all(isinstance(word, str) for word in words) else None


def _command_form(text: str, shell: Sequence[str]) -> list[str]:
    """A ``CMD`` or ``ENTRYPOINT``: its exec form, else its text under ``shell``.  Docker expands no variable in
    either: a shell form's shell does, when it runs."""

    words = _exec_form(text)
    return words if words is not None else [*shell, text.strip()]


def _shell_operands(instruction: Instruction) -> tuple[str, ...]:
    words = _exec_form(instruction.arguments)
    if not words:
        raise DockerfileError(f"line {instruction.line}: SHELL takes a JSON array of strings")
    return tuple(words)


_HEALTHCHECK_OPTIONS = frozenset({"interval", "timeout", "start-period", "start-interval", "retries"})


def _healthcheck(instruction: Instruction) -> dict[str, Any]:
    """A ``HEALTHCHECK [OPTIONS] CMD command`` (or ``NONE``) as compose writes a healthcheck: ``test`` (``["CMD",
    ...]``, ``["CMD-SHELL", text]`` or ``["NONE"]``) and the options given, in compose's spelling (``start_period``)."""

    rest = instruction.arguments.strip()
    flags: dict[str, str] = {}
    while match := _FLAG.match(rest):
        flags[match.group(1).lower()] = match.group(2) or ""
        rest = rest[match.end() :]
    if unknown := sorted(set(flags) - _HEALTHCHECK_OPTIONS):
        raise DockerfileError(f"line {instruction.line}: HEALTHCHECK has no option --{unknown[0]}")
    word, _, command = rest.partition(" ")
    if word.upper() == "NONE":
        return {"test": ["NONE"]}
    if word.upper() != "CMD" or not command.strip():
        raise DockerfileError(f"line {instruction.line}: HEALTHCHECK takes NONE, or CMD and a command")
    words = _exec_form(command)
    healthcheck: dict[str, Any] = {"test": ["CMD", *words] if words is not None else ["CMD-SHELL", command.strip()]}
    for name, value in flags.items():
        healthcheck[name.replace("-", "_")] = int(value) if name == "retries" and value.isdigit() else value
    return healthcheck


def expand(value: str, variables: Mapping[str, str]) -> str:
    """Expand ``$NAME``, ``${NAME}``, ``${NAME:-word}`` and ``${NAME:+word}`` as a Dockerfile build does; an
    unset variable expands to nothing, and ``\\$`` is a literal dollar."""

    out: list[str] = []
    index = 0
    while index < len(value):
        if value.startswith("\\$", index):
            out.append("$")
            index += 2
            continue
        match = _VARIABLE.match(value, index)
        if match is None:
            out.append(value[index])
            index += 1
            continue
        name = match.group(1) or match.group(4)
        operator, word = match.group(2), match.group(3)
        current = variables.get(name)
        if operator in (":-", "-"):
            out.append(current if current not in (None, "") or (operator == "-" and current is not None) else word)
        elif operator in (":+", "+"):
            out.append(word if current not in (None, "") or (operator == "+" and current is not None) else "")
        else:
            out.append(current or "")
        index = match.end()
    return "".join(out)


def shell_word(value: str, known: Mapping[str, str]) -> str:
    """``value`` (an ``ENV`` value as :func:`parse_assignments` returns it) as a double-quoted POSIX shell word
    that expands to what a build would: references to ``known`` variables are substituted now, other variable
    references are left for the shell to expand (they may name the base image's environment, as ``$PATH``
    does), and everything else is literal, quotes, backticks, ``$(...)`` and backslashes included."""

    return f'"{_shell_text(value, known)}"'


def has_variables(value: str) -> bool:
    return bool(_VARIABLE.search(value.replace("\\$", "")))


def variables_defined(value: str, variables: Mapping[str, str]) -> bool:
    """Whether every variable ``value`` names (outside ``${NAME:-default}`` forms) is in ``variables``."""

    return all(
        (match.group(1) or match.group(4)) in variables or match.group(2) in (":-", "-")
        for match in _VARIABLE.finditer(value.replace("\\$", ""))
    )


def remote_source(source: str, variables: Mapping[str, str] | None = None) -> Literal["git", "url"] | None:
    """What a ``COPY`` or ``ADD`` source fetches, told apart as BuildKit's Dockerfile frontend tells it: ``"git"``
    for a git repository (``git://``, ``ssh://``, ``user@host:path``, or an ``http(s)://`` URL whose path ends in
    ``.git``, each with an optional ``#ref:subdir``), ``"url"`` for any other ``http://`` or ``https://`` URL, and
    None for a path in the build context, which any other text is (``ftp://host/x`` and ``github.com/org/repo``
    too).

    With ``variables`` (:func:`variables_in_scope`), the source is told apart once they are expanded in it, as a
    build expands them; a source that names a variable they do not hold is told apart as written."""

    if variables is not None and variables_defined(source, variables):
        source = expand(source, variables)
    scheme = _URL_SCHEME.match(source)
    if scheme is not None:
        protocol = scheme.group(1).lower()
        if protocol in ("git", "ssh"):
            return "git"
        if protocol in ("http", "https"):
            try:
                path = urllib.parse.urlsplit(source).path
            except ValueError:
                path = ""
            if path.endswith(".git"):
                return "git"
    elif _SCP_LIKE.match(source):
        return "git"
    return "url" if source.startswith(("http://", "https://")) else None


def variables_in_scope(
    instructions: Sequence[Instruction], *, build_args: Mapping[str, str] | None = None
) -> Iterator[tuple[Instruction, dict[str, str]]]:
    """Each instruction, with the values the Dockerfile gives the variables in scope after it, scoped as BuildKit
    scopes them: an ``ARG`` before the first ``FROM`` holds in a stage only once the stage declares it again (without
    a value); a stage's ``ARG`` and ``ENV`` values hold in it and in the stages built ``FROM`` it; an ``ENV`` value
    wins over an ``ARG`` of the same name, and the values of one ``ENV`` step see the variables as they were before
    it.  A variable whose value only the base image knows is left out: one the Dockerfile never sets, one an ``ARG``
    declares without a value, and one set to a value that names such a variable.  ``build_args`` give declared
    ``ARG``\\ s their values instead of their defaults, as ``--build-arg`` does."""

    overrides = dict(build_args or {})

    global_args: dict[str, str | None] = {}
    stages: dict[str, tuple[dict[str, str | None], dict[str, str | None]]] = {}
    args: dict[str, str | None] = {}
    env: dict[str, str | None] = {}
    seen_from = False

    def known(*layers: Mapping[str, str | None]) -> dict[str, str]:
        merged: dict[str, str | None] = {}
        for layer in layers:
            merged.update(layer)
        return {name: value for name, value in merged.items() if value is not None}

    def resolved(value: str, variables: Mapping[str, str]) -> str | None:
        return expand(value, variables) if variables_defined(value, variables) else None

    for instruction in instructions:
        keyword = instruction.keyword
        if keyword == "FROM":
            seen_from = True
            _, operands = instruction.flags_and_operands()
            base = resolved(operands[0], known(global_args)) if operands else None
            parent = stages.get(base.lower()) if base is not None else None
            args, env = (dict(parent[0]), dict(parent[1])) if parent is not None else ({}, {})
            if len(operands) >= 3 and operands[1].lower() == "as":
                stages[operands[2].lower()] = (args, env)
        elif keyword == "ARG":
            for name, value in parse_assignments(instruction.arguments, "ARG"):
                if name in overrides:
                    (args if seen_from else global_args)[name] = overrides[name]
                elif not seen_from:
                    global_args[name] = None if value is None else resolved(value, known(global_args))
                elif value is not None:
                    args[name] = resolved(value, known(args, env))
                elif global_args.get(name) is not None:
                    args[name] = global_args[name]
        elif keyword == "ENV" and seen_from:
            before = known(args, env)
            for name, value in parse_assignments(instruction.arguments, "ENV"):
                env[name] = resolved(value, before)
        yield instruction, known(args, env)


@overload
def parse_assignments(arguments: str, keyword: Literal["ARG"]) -> list[tuple[str, str | None]]: ...


@overload
def parse_assignments(arguments: str, keyword: Literal["ENV", "LABEL"]) -> list[tuple[str, str]]: ...


def parse_assignments(arguments: str, keyword: str) -> list[tuple[str, str | None]] | list[tuple[str, str]]:
    """``ENV``/``LABEL``/``ARG`` arguments as ``(name, value)`` pairs: ``a=b c="d e"``, or the legacy
    ``ENV name value with spaces``. Quotes and backslashes are processed as BuildKit does for these forms.  Only an
    ``ARG`` without a default has the value None."""

    words = _shell_words(arguments)
    if not words:
        raise DockerfileError(f"{keyword} needs arguments")
    if keyword in ("ENV", "LABEL") and "=" not in words[0]:
        name, _, rest = arguments.strip().partition(" ")
        return [(name, rest.strip())]
    pairs: list[tuple[str, str | None]] = []
    for word in words:
        name, sep, value = word.partition("=")
        if keyword != "ARG" and not sep:
            raise DockerfileError(f"{keyword} {arguments!r}: expected name=value")
        pairs.append((name, value if sep else None))
    return pairs


# --- internals ------------------------------------------------------------------------------------------------


@dataclass
class _Scope:
    """ARG defaults and ENV values the Dockerfile itself defines, and its stages' bases."""

    global_args: dict[str, str] = field(default_factory=dict)  # ARGs before the first FROM
    stage_args: dict[str, str] = field(default_factory=dict)  # ARGs of the last stage, in scope at its end
    env: dict[str, str] = field(default_factory=dict)  # ENVs of the last stage the file sets
    bases: list[str] = field(default_factory=list)
    stages: set[str] = field(default_factory=set)

    @classmethod
    def of(cls, instructions: Sequence[Instruction], *, build_args: Mapping[str, str] | None = None) -> _Scope:
        scope = cls()
        overrides = dict(build_args or {})
        seen_from = False
        for instruction in instructions:
            if instruction.keyword == "FROM":
                seen_from = True
                scope.stage_args, scope.env = {}, {}
                _, operands = instruction.flags_and_operands()
                if operands:
                    base = expand(operands[0], scope.global_args)
                    scope.bases.append(base)
                    if len(operands) >= 3 and operands[1].lower() == "as":
                        scope.stages.add(operands[2].lower())
            elif instruction.keyword == "ARG":
                for name, value in parse_assignments(instruction.arguments, "ARG"):
                    if not seen_from:
                        scope.global_args[name] = overrides.get(name, value or "")
                    elif name in overrides:
                        scope.stage_args[name] = overrides[name]
                    else:
                        default = scope.global_args.get(name, "") if value is None else value
                        scope.stage_args[name] = expand(default, {**scope.stage_args, **scope.env})
            elif instruction.keyword == "ENV" and seen_from:
                for name, value in parse_assignments(instruction.arguments, "ENV"):
                    scope.env[name] = value
        return scope


def _not_deferrable(
    instruction: Instruction, variables: Mapping[str, str], context: Path, has_ignore: bool
) -> str | None:
    """None when ``instruction`` can be applied at session open; else why not ("" for a plain RUN or FROM).
    ``variables`` are those in scope at it (:func:`variables_in_scope`)."""

    keyword = instruction.keyword
    if keyword in ("RUN", "FROM"):
        return ""
    if keyword in METADATA or keyword in ("WORKDIR", "ENV", "ARG", "USER"):
        return None
    if keyword not in ("COPY", "ADD"):
        return f"{keyword} is not deferred"
    if instruction.heredocs:
        return f"{keyword} with a here-document stays in the build"
    flags, operands = instruction.flags_and_operands()
    if unknown := sorted(set(flags) - _COPY_FLAGS_DEFERRABLE):
        return f"{keyword} --{unknown[0]} stays in the build"
    if len(operands) < 2:
        return f"{keyword} needs a source and a destination"
    if has_ignore:
        return f"{keyword} from a context with a {DOCKERIGNORE} stays in the build"
    for source in operands[:-1]:
        if remote := remote_source(source, variables):
            return f"{keyword} of {_REMOTE[remote]} stays in the build"
        if "$" in source:
            return f"{keyword} source {source!r} has a variable"
        if any(char in source for char in "*?["):
            return f"{keyword} source {source!r} has a wildcard"
        try:
            path = _context_path(context, source)
        except DockerfileError as exc:
            return str(exc)
        if not path.exists():
            return f"{keyword} source {source!r} is missing from the build context"
        if keyword == "ADD" and _is_archive(path):
            return f"ADD of archive {source!r} extracts it, so it stays in the build"
    return None


def _deferred_steps(instruction: Instruction, scope: _Scope) -> tuple[list[tuple[str, ...]], str | None]:
    keyword = instruction.keyword
    if keyword == "WORKDIR":
        return [("WORKDIR", instruction.arguments.strip())], None
    if keyword == "ENV":
        return [("ENV", name, value) for name, value in parse_assignments(instruction.arguments, "ENV")], None
    if keyword == "ARG":
        steps = []
        for name, value in parse_assignments(instruction.arguments, "ARG"):
            steps.append(("ARG", name, scope.global_args.get(name, "") if value is None else value))
        return steps, None
    if keyword == "USER":
        return [], instruction.arguments.strip()
    if keyword in ("COPY", "ADD"):
        _, operands = instruction.flags_and_operands()
        sources = [_normalize_source(source) for source in operands[:-1]]
        return [("COPY", *sources, operands[-1])], None
    return [], None  # metadata: no effect on files or the environment


def _prefix_context(
    prefix: Sequence[tuple[Instruction, Mapping[str, str]]], context: Path, has_ignore: bool
) -> list[str]:
    """Context paths the prefix reads: COPY/ADD sources ("." when a wildcard, a variable whose value only the base
    image knows or a bind mount reads the whole context, or a .dockerignore filters it). A remote source reads none:
    the builder fetches it, and the key covers the step as written. ``prefix`` pairs each instruction with the
    variables in scope at it (:func:`variables_in_scope`)."""

    paths: set[str] = set()
    whole = False  # the key covers the whole context: later steps are read only for a COPY of remote content
    for instruction, variables in prefix:
        flags, operands = instruction.flags_and_operands()
        if instruction.keyword == "RUN":
            for match in re.finditer(r"--mount=(\S+)", instruction.arguments):
                options = dict(item.partition("=")[::2] for item in match.group(1).split(","))
                if options.get("type", "bind") == "bind" and "from" not in options:
                    whole = True
            continue
        if instruction.keyword not in ("COPY", "ADD") or "from" in flags or instruction.heredocs:
            continue
        for source in operands[:-1]:
            known = variables_defined(source, variables)
            if remote := remote_source(source, variables):
                if instruction.keyword == "COPY":
                    shown = expand(source, variables) if known else source
                    raise DockerfileError(
                        f"line {instruction.line}: COPY cannot fetch {_REMOTE[remote]} ({shown}); only ADD can"
                    )
                continue  # remote content: the key covers the step as written
            if whole or has_ignore or not known:
                whole = True
                continue
            source = expand(source, variables)
            if any(char in source for char in "*?["):
                whole = True
                continue
            path = _context_path(context, source)
            if not path.exists():
                raise DockerfileError(f"line {instruction.line}: {instruction.keyword} source {source!r} is missing")
            relative = path.relative_to(context.resolve()).as_posix()
            paths.add(relative)
    return ["."] if whole else sorted(paths)


def _build_key(
    directives: Sequence[str],
    prefix: Sequence[Instruction],
    context: Path,
    context_files: Sequence[str],
    has_ignore: bool,
    build_args: Mapping[str, str] | None = None,
) -> str:
    files = []
    for relative in context_files:
        for path in _walk(context.resolve() / relative if relative != "." else context.resolve()):
            name = path.relative_to(context.resolve()).as_posix()
            files.append([name, file_sha256(path), bool(path.stat().st_mode & 0o111)])
    payload: dict[str, Any] = {
        "rule": BUILD_KEY_RULE,
        "directives": list(directives),
        "prefix": [instruction.normalized for instruction in prefix],
        "context": sorted(files),
        "dockerignore": file_sha256(context / DOCKERIGNORE) if has_ignore else None,
    }
    if build_args:  # only then, so that every key planned before build arguments existed stays as it was
        payload["build_args"] = dict(sorted(build_args.items()))
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _walk(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
        return
    for child in sorted(path.rglob("*")):
        if child.is_file():
            yield child


def _context_path(context: Path, source: str) -> Path:
    root = context.resolve()
    normalized = posixpath.normpath("/" + source.lstrip("/"))
    path = (root / normalized.lstrip("/")).resolve() if normalized != "/" else root
    if path != root and root not in path.parents:
        raise DockerfileError(f"source {source!r} is outside the build context")
    return path


def _normalize_source(source: str) -> str:
    normalized = posixpath.normpath("/" + source.lstrip("/")).lstrip("/") or "."
    return normalized + "/" if source.endswith("/") and normalized != "." else normalized


def _is_archive(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        head = handle.read(512)
    return head.startswith(_ARCHIVE_MAGIC) or head[257:262] == b"ustar"


def _shell_text(value: str, known: Mapping[str, str]) -> str:
    """The inside of :func:`shell_word`'s double quotes, scanned as :func:`expand` scans."""

    out: list[str] = []
    index = 0
    while index < len(value):
        if value.startswith("\\$", index):
            out.append("\\$")
            index += 2
            continue
        match = _VARIABLE.match(value, index)
        if match is None:
            out.append(_SHELL_ESCAPES.get(value[index], value[index]))
            index += 1
            continue
        name, operator, word = match.group(1) or match.group(4), match.group(2), match.group(3)
        if name in known:
            out.append("".join(_SHELL_ESCAPES.get(char, char) for char in expand(match.group(0), known)))
        elif operator:
            out.append(f"${{{name}{operator}{_shell_text(word, known)}}}")
        else:
            out.append(f"${{{name}}}")
        index = match.end()
    return "".join(out)


def _shell_words(text: str) -> list[str]:
    """Split on unquoted whitespace, removing quotes and backslash escapes (BuildKit's word rules for ENV, LABEL
    and ARG; variables are kept for expansion later)."""

    words: list[str] = []
    current: list[str] = []
    quote: str | None = None
    started = False
    index = 0
    while index < len(text):
        char = text[index]
        if quote is None and char.isspace():
            if started:
                words.append("".join(current))
                current, started = [], False
            index += 1
            continue
        started = True
        if char == "\\" and index + 1 < len(text) and quote != "'":
            following = text[index + 1]
            if quote == '"' and following not in '"\\$':
                current.append(char)
            elif following == "$":
                current.append("\\")  # keep \$ escaped for expansion
            current.append(following)
            index += 2
            continue
        if char in "\"'" and (quote is None or quote == char):
            quote = None if quote == char else char
            index += 1
            continue
        if char == "$" and quote == "'":
            current.append("\\")  # single quotes keep a dollar literal: escape it for expansion
        current.append(char)
        index += 1
    if quote is not None:
        raise DockerfileError(f"unterminated quote in {text!r}")
    if started:
        words.append("".join(current))
    return words


def matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
