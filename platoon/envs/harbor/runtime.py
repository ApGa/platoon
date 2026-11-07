"""Runtime Harbor tasks: task directories written while a run runs, made into checked, content-addressed tasks of an
environment entry, which its sessions run and platoon's launcher grades.

A producer of tasks, such as an environment generator (a paper-to-task pipeline, a bug injector), writes a Harbor task
directory while the run runs.  An environment entry that sets ``service_kwargs.runtime_tasks``
(:class:`RuntimeTasksConfig`) takes it in through :meth:`HarborEnvironmentService.materialize
<platoon.envs.harbor.service.HarborEnvironmentService.materialize>` (:func:`materialize_task`), which refuses a
directory that breaks the entry's rules and never rewrites one, so that a producer learns the rules rather than has
them patched:

* it copies the directory's regular files and directories, and refuses links, special files, more than ``max_bytes``
  of files and more than :data:`MAX_ENTRIES` entries;
* ``tests/`` holds only ``gen/``, the task's grader, which ``tests/gen/grader.json`` describes
  (:func:`read_grader_manifest`): ``tests/test.sh`` is platoon's launcher (:mod:`platoon.envs.harbor.launcher`), which
  ``materialize`` installs;
* there is no ``environment/Dockerfile`` (a runtime task runs a prebuilt image and builds none) and no
  ``environment/docker-compose.yaml`` (it runs in one sandbox, never a stack);
* its ``task.toml`` nests tables and arrays at most :data:`MAX_TOML_DEPTH` levels deep (:func:`load_toml`), and is the
  ``template`` but for the ``[task] name`` and the ``artifacts``, which are the task's own
  (:func:`template_differences`); the template itself gives no network, grades in a separate verifier on the task's own
  image and names an image of ``images`` (:func:`template_problems`);
* each of its ``artifacts`` is a path relative to the working directory, below it (:func:`artifact_problems`), so that
  what a solver submits lands there in the verifier and nowhere near ``/tests`` or ``/logs/verifier``;
* ``HarborTask.load`` reads it, with the entry's user policies but for ``verifier_user``, which
  ``runtime_tasks.verifier_user`` replaces.

It then writes the launcher's settings (``tests/platoon/launch.json``: who wrote the grader, the reward's name, the
verifier's time limit, the artifacts, the grader's command and the replay, the template's variables the replay keeps
(:func:`replay_variables`), and the fingerprint of the ``runtime_tasks`` it was checked against,
:func:`config_fingerprint`), computes the task's digest (:func:`task_digest`) and renames the copy into
``<store>/<digest>``, so that two producers that write the same task share it.  The task it returns
(:class:`MaterializedTask`) has the digest as its id and the stored directory in ``misc["harbor"]["task_dir"]``.

A runtime task's directory says it is one: its launcher's settings, which only ``materialize`` writes
(:func:`launch_settings`).  An environment runs one only from its own store (:func:`in_store`) and only if it was
checked against the environment's own ``runtime_tasks`` (the same fingerprint); any other is refused, and never runs as
an ordinary task.  Its sessions run the image ``images`` binds its ``docker_image`` to and upload its ``environment/``
into the working directory, as Harbor does for a prebuilt image (:func:`runtime_image`: the plan a fetched dataset's
lock would hold); its separate verifier starts with ``runtime_tasks.verifier_provider_options`` laid over the entry's;
and its grade is what the launcher's report says (:func:`launcher_outcome`): once the grader runs, a non-score is named
by the grader's owner, as is a launcher that did not finish (:func:`unfinished`).  The entry's other tasks are fetched,
listed, opened and graded as before.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import shutil
import stat
import tomllib
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from platoon.envs.harbor import launcher
from platoon.envs.harbor.hashing import file_sha256
from platoon.envs.harbor.images import SessionImage, session_image
from platoon.envs.harbor.lock import default_cache
from platoon.envs.harbor.task import (
    CONFIG_FILENAME,
    ENVIRONMENT_DIR,
    HARBOR_MISC_KEY,
    TESTS_DIR,
    VERIFIER_USER_POLICIES,
    HarborTask,
    HarborTaskError,
    _check_linux,
    _environment_settings,
    _phase_network,
    _verifier_environment,
)
from platoon.envs.harbor.verifier import VerifierFile
from platoon.envs.service import Outcome
from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.task import Task

#: Who wrote a runtime task's grader, which names the failure kinds of its non-scores (``verifier_generated_*`` for one
#: the generator wrote, ``verifier_trusted_*`` for trusted code's, such as a task producer's own grader).
GRADER_OWNERS = launcher.OWNERS
DEFAULT_MAX_BYTES = 256 * 2**20
#: The most files and directories a runtime task holds.
MAX_ENTRIES = 10_000
#: The most artifacts a runtime task declares, and the longest path of one, in characters.
MAX_ARTIFACTS = 256
MAX_ARTIFACT_CHARS = 1024
#: The largest output file of a submission the launcher hands its grader.
MAX_OUTPUT_BYTES = 64 * 2**20
#: The address space each of a replay's processes may reserve (``RLIMIT_AS``), ``materialize``'s default: None, no
#: per-process cap, since a numeric replay legitimately reserves a large virtual address space and a cap too low would
#: score an honest submission 0.  The launcher always makes the replay the most killable to the OOM killer, so memory
#: the replay runs the sandbox out of ends the replay, not the launcher; a producer that wants a hard per-process
#: ceiling as well passes ``replay_address_space_bytes``.
DEFAULT_REPLAY_ADDRESS_SPACE_BYTES: int | None = None
#: The largest ``tests/gen/grader.json``.
MAX_MANIFEST_BYTES = 64 * 1024
#: How deeply a runtime task's ``task.toml``, and the template, may nest tables and arrays, the document's own table
#: included: deeper TOML is refused, before it is parsed where its text shows it (:func:`load_toml`).  Python's TOML
#: parser gives up at depths of its own, all deeper than this: where the stack it was called with reaches the
#: recursion limit (about 330 inline tables deep from a shallow stack), and, in recent 3.13 and 3.14 releases, at a
#: dotted key of more parts than that limit, which 3.12 parses in time quadratic in its parts.  So what is refused is
#: the same on every Python.
MAX_TOML_DEPTH = 64
#: The most problems a refusal lists.
MAX_PROBLEMS = 20
#: The store's directory when ``store`` is None: in the run's directory, else in the Harbor cache.
STORE_DIR = "runtime-tasks"
STAGING_DIR = ".staging"  # in the store: copies being checked, renamed to their digest once they pass
GEN_DIR = "gen"  # tests/gen: the task's grader
GRADER_MANIFEST = "grader.json"
LAUNCHER_DIR = "platoon"  # tests/platoon: the launcher and its settings
LAUNCHER_FILE = "launcher.py"
TEST_SCRIPT = "test.sh"
DIGEST_RULE = "platoon-runtime-task-v1"
FINGERPRINT_RULE = "platoon-runtime-tasks-config-v1"
#: The largest launcher's settings a task's directory may hold, for :func:`launch_settings` to read.
MAX_SETTINGS_BYTES = 1 << 20
#: ``misc["harbor"]`` of a runtime task marks it so: the service refuses such a task whose directory is none.
RUNTIME_MISC_KEY = "runtime"
# The longest reason a runtime task's outcome takes from the launcher's detail, in characters.
_REASON_CHARS = 2000
# The reasons of a score of 0 the launcher gives, the solver's faults.
_SOLVER_FAULTS = (launcher.OUTPUT_CONTRACT, launcher.NONREPRODUCIBLE)
# The keys of a runtime task's task.toml that are its own; everything else is the template's.
_OWN_KEYS = (("task", "name"), ("artifacts",))
_DIGEST = re.compile(r"[0-9a-f]{64}")
_VARIABLE = launcher._VARIABLE  # a name the launcher keeps in a replay's environment
_REFUSED_ENVIRONMENT_FILES = (
    ("Dockerfile", "a runtime task runs a prebuilt image of runtime_tasks.images, and builds none"),
    ("docker-compose.yaml", "a runtime task runs in one sandbox, never a multi-container stack"),
)
_CONFIG_KEYS = ("images", "template", "verifier_user", "verifier_provider_options", "max_bytes", "store")
# What toml_text_depth reads a TOML text by.  First, in one pass, each string (tried as a multi-line one first; one that
# never ends runs to the end of its line, or a multi-line one to the end of the text) and each comment, which each
# become one character of a bare key; then the brackets left, and the dotted keys left (bare keys joined by dots, a
# string among them now a bare key too), each tried from its first character only.  An alternative for a string or a
# comment whose opening matches goes on to match, and no repetition gives back what it took, so each pass is linear in
# the text's length.
_TOML_STRING_OR_COMMENT = re.compile(
    r'"""(?:[^"\\]++|\\[\s\S]|"(?!""))*+(?:"{3,5}|\\?\Z)'
    r"|'''(?:[^']++|'(?!''))*+(?:'{3,5}|\Z)"
    r'|"(?:[^"\\\n]++|\\[^\n])*+"?'
    r"|'[^'\n]*+'?"
    r"|#[^\n]*+"
)
_TOML_NOT_BRACKET = re.compile(r"[^\[\]{}]+")
_TOML_DOTTED_KEY = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]++(?:[ \t]*+\.[ \t]*+[A-Za-z0-9_-]++)++")
_BRACKET_STEP = {"[": 1, "{": 1, "]": -1, "}": -1}
# The launcher's entry point, which a verifier runs as `bash /tests/test.sh`: the image's python3 runs the launcher
# beside it, and without one the grade is a fault on the trusted side, never a score.
TEST_SH = """#!/bin/sh
# platoon's launcher of a runtime task, installed by materialize: it replays the submission, then runs the task's
# grader in tests/gen, and writes the only reward file (tests/platoon/launcher.py says how).
here=$(dirname "$0")
for python in python3 /usr/bin/python3 /usr/local/bin/python3; do
  if command -v "$python" >/dev/null 2>&1; then
    exec "$python" -I "$here/platoon/launcher.py"
  fi
done
logs="${PLATOON_SANDBOX_ROOT:-}/logs/verifier"
echo '{"failure_kind": "verifier_trusted_launcher", "invalid": 1}' > "$logs/reward.json"
echo '{"schema": "platoon.launcher/v1", "failure_kind": "verifier_trusted_launcher", "detail": "the verifier image has no python3"}' > "$logs/launcher.json"
echo "platoon launcher: the verifier image has no python3, which the launcher runs on"
"""  # noqa: E501


@dataclass(frozen=True)
class RuntimeTasksConfig:
    """How an environment entry's runtime tasks run: ``service_kwargs.runtime_tasks`` of a Harbor environment.

    Built from a mapping of its fields (:meth:`coerce`), where ``template`` may name a TOML file.  The template nests
    tables and arrays at most 64 levels deep (:data:`MAX_TOML_DEPTH`), as a runtime task's ``task.toml`` must; it must
    give the agents and the verifier no network (``[environment] network_mode = "no-network"``), grade in a separate
    verifier (``[verifier] environment_mode = "separate"``) that runs the task's own image (no
    ``[verifier.environment] docker_image``, whose image would bring its own ``/tests``), and name one of ``images`` as
    its ``[environment] docker_image``; and it gives neither ``[task] name`` nor ``artifacts``, which are each task's
    own.  The launcher runs as root, so the template's ``[verifier] user`` is root, with ``verifier_user: honor``, on
    verifiers' sandboxes that give root.
    """

    # The docker_image names a runtime task may give, each with the image its sandboxes run, written as a task's own
    # docker_image is: a .sif or .sqsh file or an absolute path as written, a registry reference without a scheme with
    # docker://.
    images: Mapping[str, str]
    # The task.toml every runtime task has, but for its [task] name and its artifacts: a mapping, or a TOML file.
    template: Mapping[str, Any]
    # What a runtime task's [verifier].user gets (refuse, ignore or honor), in place of service_kwargs.verifier_user.
    verifier_user: Literal["refuse", "ignore", "honor"] = "honor"
    # The provider options of runtime tasks' verifiers' sandboxes, laid over the entry's verifier_provider_options
    # (or provider_options), such as {rootfs_layout: copy, fakeroot: true} for verifiers that run as root on
    # Singularity; the entry's other tasks keep theirs.
    verifier_provider_options: Mapping[str, Any] = field(default_factory=dict)
    # The most a runtime task's files may hold, in bytes.
    max_bytes: int = DEFAULT_MAX_BYTES
    # The directory that keeps the runtime tasks, each in <store>/<digest>. None: runtime-tasks in the run's directory
    # (platoon train's, platoon eval's), or outside a run, in the Harbor cache.
    store: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.images, Mapping) or not self.images:
            raise ValueError("runtime_tasks.images must map the docker_image names runtime tasks may give to images")
        for name, image in self.images.items():
            if not (isinstance(name, str) and name and isinstance(image, str) and image):
                raise ValueError(f"runtime_tasks.images must map names to images, got {name!r}: {image!r}")
        object.__setattr__(self, "images", dict(self.images))
        if not isinstance(self.template, Mapping):
            raise ValueError(
                f"runtime_tasks.template must be a task.toml's tables, or a TOML file, got {self.template!r}"
            )
        if toml_depth(self.template) > MAX_TOML_DEPTH:
            raise ValueError(
                f"runtime_tasks.template is nested too deeply: more than {MAX_TOML_DEPTH} levels of tables and arrays"
            )
        object.__setattr__(self, "template", _plain(self.template))
        if problems := template_problems(self.template, self.images):
            raise ValueError("runtime_tasks.template cannot be runtime tasks' task.toml: " + "; ".join(problems))
        if self.verifier_user not in VERIFIER_USER_POLICIES:
            raise ValueError(
                f"runtime_tasks.verifier_user must be one of {VERIFIER_USER_POLICIES}, got {self.verifier_user!r}"
            )
        if not isinstance(self.verifier_provider_options, Mapping):
            raise ValueError(
                f"runtime_tasks.verifier_provider_options must be a mapping, got {self.verifier_provider_options!r}"
            )
        object.__setattr__(self, "verifier_provider_options", dict(self.verifier_provider_options))
        if isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, int) or self.max_bytes < 1:
            raise ValueError(f"runtime_tasks.max_bytes must be a positive number of bytes, got {self.max_bytes!r}")
        if self.store is not None and not isinstance(self.store, str | os.PathLike):
            raise ValueError(f"runtime_tasks.store must be a directory, got {self.store!r}")

    @classmethod
    def coerce(cls, value: RuntimeTasksConfig | Mapping[str, Any]) -> RuntimeTasksConfig:
        """The config ``value`` gives: itself, or a mapping of its fields whose ``template`` may name a TOML file."""

        if isinstance(value, RuntimeTasksConfig):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(f"runtime_tasks must be a mapping of {', '.join(_CONFIG_KEYS)}, got {value!r}")
        if unknown := sorted(set(value) - set(_CONFIG_KEYS)):
            raise ValueError(f"runtime_tasks takes {', '.join(_CONFIG_KEYS)}, not {', '.join(map(str, unknown))}")
        options = dict(value)
        template = options.get("template")
        if isinstance(template, str | os.PathLike):
            path = Path(template).expanduser()
            try:
                options["template"] = load_toml(path.read_text())
            # A TOMLDecodeError, a bad encoding, and an integer past Python's digit limit are all ValueErrors.
            except (OSError, ValueError) as exc:
                raise ValueError(f"runtime_tasks.template: {path}: {exc}") from exc
        if "images" not in options or "template" not in options:
            raise ValueError("runtime_tasks needs images and a template")
        return cls(**options)

    def store_path(self) -> Path:
        """The store, as an absolute path: ``store``, else ``runtime-tasks`` in the run's directory (which every process
        of a ``platoon train`` or ``platoon eval`` run labels its sandboxes with), else in the Harbor cache."""

        if self.store is not None:
            return Path(self.store).expanduser().resolve()
        run = sandbox_cleanup.labels().get(sandbox_cleanup.RUN_LABEL)
        return ((Path(run) if run else default_cache()) / STORE_DIR).resolve()


@dataclass(frozen=True)
class MaterializedTask:
    """What ``materialize`` made of a directory: a runtime task, or why there is none."""

    # The runtime task: its goal is its instruction, its id its digest, and misc["harbor"]["task_dir"] its directory
    # in the store, <store>/<digest>. None when the directory was refused.
    task: Task | None
    # The task's digest, a SHA-256 of all its files: its id and its name in the store. None when it was refused.
    digest: str | None
    # Why the directory was refused, each one sentence, for the producer to learn from; empty when it was not.
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the directory became a runtime task."""

        return not self.problems


def _refused(problems: list[str]) -> MaterializedTask:
    return MaterializedTask(task=None, digest=None, problems=tuple(problems[:MAX_PROBLEMS]))


def materialize_task(
    source: str | os.PathLike[str],
    config: RuntimeTasksConfig,
    *,
    store: Path,
    load: Callable[[Path], HarborTask],
    reward_key: str,
    timeout_multiplier: float = 1.0,
    grader_owner: str = "generated",
    replay_address_space_bytes: int | None = DEFAULT_REPLAY_ADDRESS_SPACE_BYTES,
) -> MaterializedTask:
    """Make the task directory ``source`` a runtime task in ``store`` under ``config`` (the module docstring), or say
    why it cannot be one.  ``load`` reads a task directory as the environment reads its runtime tasks; ``reward_key``
    is the name of the reward the launcher writes (the reward contract's first), and ``timeout_multiplier`` scales the
    verifier's time limit, as the environment's ``verifier_timeout_multiplier``.  ``grader_owner`` says who wrote the
    task's grader: ``generated`` (by the producer) or ``trusted`` (trusted code), which names the failure kinds of its
    non-scores.  ``replay_address_space_bytes`` caps the address space each of the task's replay processes may reserve
    (``RLIMIT_AS``; :data:`DEFAULT_REPLAY_ADDRESS_SPACE_BYTES`, None, sets no per-process cap).  Problems of ``source``
    come back as problems; a ``ValueError`` for a replay address space that is not a positive integer, and an
    ``OSError`` writing the store is raised."""

    if grader_owner not in GRADER_OWNERS:
        raise ValueError(f"grader_owner must be one of {GRADER_OWNERS}, got {grader_owner!r}")
    space = replay_address_space_bytes
    if space is not None and (isinstance(space, bool) or not isinstance(space, int) or space <= 0):
        raise ValueError(f"replay_address_space_bytes must be a positive integer or None, got {space!r}")
    source = Path(source)
    try:
        if not stat.S_ISDIR(source.lstat().st_mode):
            return _refused([f"{source} is not a directory"])
    except OSError as exc:
        return _refused([f"{source} cannot be read: {exc.strerror}"])
    staging = store / STAGING_DIR / uuid.uuid4().hex
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        problems = _copy_checked(source, staging, config.max_bytes)
        manifest = None
        if not problems:
            manifest, problems = _layout(staging)
            toml = _task_toml(staging, problems)
            problems += template_differences(toml, config.template)
            problems += artifact_problems(toml)
        if problems:
            return _refused(problems)
        assert manifest is not None
        _install_launcher(staging)
        try:
            harbor_task = load(staging)
        except (HarborTaskError, OSError, ValueError) as exc:  # what the task holds: refused, as a fetch refuses it
            return _refused([_without(str(exc), staging)])
        settings = {
            "schema": launcher.SCHEMA,
            "fingerprint": config_fingerprint(config, reward_key=reward_key, timeout_multiplier=timeout_multiplier),
            "grader_owner": grader_owner,
            "reward_key": reward_key,
            "timeout_s": harbor_task.verifier_timeout_s * timeout_multiplier,
            "max_output_bytes": MAX_OUTPUT_BYTES,
            # The task's environment/ (at most max_bytes) is in the working directory too, beside the submission.
            "max_copy_bytes": config.max_bytes + launcher.MAX_SUBMISSION_BYTES,
            "artifacts": [artifact.source for artifact in harbor_task.artifacts],
            "grader": {"command": manifest["command"]},
            "replay": manifest["replay"],
            "replay_env": replay_variables(harbor_task),
            "max_address_space_bytes": replay_address_space_bytes,
        }
        _write(staging / TESTS_DIR / LAUNCHER_DIR / launcher.CONFIG, json.dumps(settings, indent=1, sort_keys=True))
        digest = task_digest(staging)
        stored = store / digest
        try:
            os.rename(staging, stored)
        except OSError:
            if not stored.is_dir():
                raise
            # Another producer wrote the same task first: the digest names the same files.
    finally:
        shutil.rmtree(staging, ignore_errors=True)  # gone already once it was renamed
    return MaterializedTask(task=runtime_task(load(stored), digest), digest=digest)


def runtime_task(harbor_task: HarborTask, digest: str) -> Task:
    """The platoon task of a runtime task: its digest as its id, and in ``misc["harbor"]`` its directory, its digest
    and that it is a runtime task."""

    task = harbor_task.to_task(digest)
    task.misc[HARBOR_MISC_KEY].update({"digest": digest, RUNTIME_MISC_KEY: True})
    return task


def template_problems(template: Mapping[str, Any], images: Mapping[str, str]) -> list[str]:
    """Why ``template`` cannot be every runtime task's ``task.toml`` (:class:`RuntimeTasksConfig`); none when it can."""

    problems = []
    if "artifacts" in template:
        problems.append("it gives artifacts, which each runtime task declares itself")
    tables = {}
    for name in ("task", "environment", "agent", "verifier"):
        table = template.get(name) or {}
        if not isinstance(table, Mapping):
            return [*problems, f"its [{name}] is not a table"]
        tables[name] = table
    if "name" in tables["task"]:
        problems.append("it gives [task] name, which each runtime task gives itself")
    if template.get("steps"):
        problems.append("it has steps, and multi-step tasks are not supported")
    try:
        _check_linux(dict(tables["environment"]), "its [environment]")
        settings = _environment_settings(dict(tables["environment"]), "[environment]")
        agents = _phase_network(dict(tables["agent"]), "[agent]") or settings["network"]
        verifier = _verifier_environment(Path(os.devnull), dict(tables["environment"]), dict(tables["verifier"]))
    except HarborTaskError as exc:
        return [*problems, str(exc)]
    image = tables["environment"].get("docker_image")
    if image not in images:
        problems.append(
            f"its [environment] docker_image ({image!r}) is none of runtime_tasks.images ({sorted(images)})"
        )
    if agents != "none":
        problems.append('it gives the agents a network: set [environment] network_mode = "no-network"')
    if verifier is None:
        problems.append(
            "it grades in the agents' own sandbox (Harbor's shared verifier mode): set [verifier] environment_mode = "
            '"separate", so that what the agents leave never runs beside the launcher'
        )
    else:
        if verifier.network != "none":
            problems.append('it gives the verifier a network: set its network_mode = "no-network"')
        if verifier.image_source != "task":
            problems.append(
                "its verifier runs [verifier.environment] docker_image, whose image brings its own /tests and so never "
                "runs the launcher: leave it out, and the verifier runs the task's own image"
            )
    return problems


def template_differences(config: Mapping[str, Any] | None, template: Mapping[str, Any]) -> list[str]:
    """How a runtime task's ``task.toml`` (``config``; None: already said why there is none) differs from the
    template, but for the task's own keys, its ``[task] name`` and its ``artifacts``; each difference one sentence."""

    if config is None:
        return []
    found: list[str] = []
    _compare(_without_own(config), _without_own(template), (), found)
    return [f"task.toml {difference}" for difference in found]


def artifact_problems(config: Mapping[str, Any] | None) -> list[str]:
    """Why a runtime task's ``artifacts`` (``config``, its ``task.toml``; None: already said why there is none) would
    put what its solver submits anywhere but below the working directory, each one sentence: an artifact is a file or
    directory the solver leaves there, named as a normalized path relative to it (no leading ``/``, no ``..``), at
    most :data:`MAX_ARTIFACT_CHARS` long; the verifier puts it back at the same path, so an absolute one could land in
    ``/tests`` or ``/logs/verifier``.  Entries ``HarborTask.load`` refuses (no source, a compose service) are left to
    it."""

    entries = (config or {}).get("artifacts")
    if not isinstance(entries, list):
        return []
    if len(entries) > MAX_ARTIFACTS:
        return [f"task.toml declares {len(entries)} artifacts, more than a runtime task may ({MAX_ARTIFACTS})"]
    problems = []
    for entry in entries:
        source = entry.get("source") if isinstance(entry, Mapping) else entry
        if not isinstance(source, str) or not source:
            continue
        if not launcher.relative_path(source) or len(source) > MAX_ARTIFACT_CHARS:
            problems.append(
                f"task.toml declares the artifact {_shown(source)}, which is no path below the working directory: "
                "name each as a normalized path relative to it, without a leading / or .., of at most "
                f"{MAX_ARTIFACT_CHARS} characters"
            )
    return problems


def read_grader_manifest(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """``tests/gen/grader.json``: ``{"command": [...], "replay": {"command": [...], "outputs": [...]}}``, the grader's
    argv, run in ``/tests/gen``, and the replay's, run in a copy of the working directory, with the outputs it writes
    there (relative paths); ``replay`` may be left out, and nothing else may be there.  The manifest and None, or None
    and why it breaks that."""

    where = f"{TESTS_DIR}/{GEN_DIR}/{GRADER_MANIFEST}"
    try:
        if path.lstat().st_size > MAX_MANIFEST_BYTES:
            return None, [f"{where} holds more than {MAX_MANIFEST_BYTES} bytes"]
        manifest = launcher.load_json(path.read_bytes().decode("utf-8"))
    except (FileNotFoundError, NotADirectoryError):
        return None, [f"{where} is missing: it names the grader's command, and the task's replay if it has one"]
    except launcher.NestedTooDeeply:
        return None, [f"{where} is nested too deeply to read as JSON"]
    except (OSError, ValueError) as exc:
        return None, [f"{where} is not JSON: {exc}"]
    if not isinstance(manifest, dict):
        return None, [f"{where} is not an object of command and replay"]
    problems = []
    if unknown := sorted(set(manifest) - {"command", "replay"}):
        problems.append(f"{where} holds {', '.join(unknown)}, beyond command and replay")
    if not _argv(manifest.get("command")):
        problems.append(f"{where}: command must be the grader's argv, a list of strings, run in /{TESTS_DIR}/{GEN_DIR}")
    replay = manifest.get("replay")
    if replay is not None:
        if not isinstance(replay, dict):
            return None, [*problems, f"{where}: replay must be an object of command and outputs"]
        if unknown := sorted(set(replay) - {"command", "outputs"}):
            problems.append(f"{where}: replay holds {', '.join(unknown)}, beyond command and outputs")
        if not _argv(replay.get("command")):
            problems.append(f"{where}: replay.command must be the replay's argv, a list of strings")
        if not _outputs(replay.get("outputs")):
            problems.append(
                f"{where}: replay.outputs must list the files the replay writes, as distinct paths relative to the "
                "working directory, none below another"
            )
    if problems:
        return None, problems
    plain_replay = None if replay is None else {"command": list(replay["command"]), "outputs": list(replay["outputs"])}
    return {"command": list(manifest["command"]), "replay": plain_replay}, []


class TomlNestedTooDeeply(ValueError):
    """TOML that nests tables and arrays more than :data:`MAX_TOML_DEPTH` levels deep (:func:`load_toml`)."""


def load_toml(text: str) -> dict[str, Any]:
    """The TOML document ``text``: a :class:`TomlNestedTooDeeply` for one that nests more than
    :data:`MAX_TOML_DEPTH` levels deep, refused before it is parsed where its text shows that (:func:`toml_text_depth`),
    else once it is parsed (:func:`toml_depth`), and a ``tomllib.TOMLDecodeError`` for one that is not TOML.  Python's
    parser then never reaches the depths at which it gives up, which differ between Pythons."""

    if toml_text_depth(text) <= MAX_TOML_DEPTH:
        document = tomllib.loads(text)
        if toml_depth(document) <= MAX_TOML_DEPTH:
            return document
    raise TomlNestedTooDeeply(f"nested too deeply: more than {MAX_TOML_DEPTH} levels of tables and arrays")


def toml_text_depth(text: str) -> int:
    """How deeply the TOML ``text`` nests, read from its text, never by parsing it: one more (its own table) than the
    most arrays and inline tables (or brackets of a table's header) that any start of it opens and leaves open outside
    its strings and comments, or the most parts of a dotted key in it, if that is more.  A parser that stops at an
    error was never deeper, and a TOML document nests at least as deeply (:func:`toml_depth`) unless this is 2, the
    parts of a float.  It takes time linear in the text's length."""

    text = _TOML_STRING_OR_COMMENT.sub("_", text)
    brackets = _TOML_NOT_BRACKET.sub("", text)
    opened = max(itertools.accumulate(map(_BRACKET_STEP.__getitem__, brackets), initial=0))
    parts = max((key.count(".") for key in _TOML_DOTTED_KEY.findall(text)), default=0) + 1
    return max(opened + 1, parts)


def toml_depth(value: Any) -> int:
    """How many levels of tables and arrays (mappings, lists and tuples) ``value``, TOML as Python holds it, nests, its
    own included, counted without recursion and only up to one past :data:`MAX_TOML_DEPTH`."""

    deepest, pending = 0, [(value, 1)]
    while pending:
        item, depth = pending.pop()
        if isinstance(item, Mapping):
            items = item.values()
        elif isinstance(item, list | tuple):
            items = item
        else:
            continue
        deepest = max(deepest, depth)
        if depth <= MAX_TOML_DEPTH:
            pending.extend((inner, depth + 1) for inner in items)
    return deepest


def task_digest(task_dir: Path) -> str:
    """The SHA-256 of a runtime task: of every directory's path and every file's path, executable bit and SHA-256,
    sorted, with :data:`DIGEST_RULE` first."""

    lines = []
    for directory, directories, files in os.walk(task_dir):
        relative = Path(directory).relative_to(task_dir).as_posix()
        prefix = "" if relative == "." else f"{relative}/"
        lines += [f"{prefix}{name}/\n" for name in directories]
        for name in files:
            path = Path(directory) / name
            executable = "x" if path.lstat().st_mode & stat.S_IXUSR else "-"
            lines.append(f"{prefix}{name}\0{executable}\0{file_sha256(path)}\n")
    return hashlib.sha256((DIGEST_RULE + "\n" + "".join(sorted(lines))).encode()).hexdigest()


def config_fingerprint(config: RuntimeTasksConfig, *, reward_key: str, timeout_multiplier: float) -> str:
    """A SHA-256 of what a runtime task was checked and set up against: the config but for its store, the name of the
    reward the launcher writes and the verifier's time multiplier.  ``materialize`` writes it into the task's launcher
    settings, and an environment runs only the runtime tasks that carry its own (another entry's ``runtime_tasks``, or
    this one's before they changed, never checked them)."""

    fields = {key: getattr(config, key) for key in _CONFIG_KEYS if key != "store"}
    text = json.dumps(
        {"config": fields, "reward_key": reward_key, "timeout_multiplier": float(timeout_multiplier)},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(f"{FINGERPRINT_RULE}\n{text}".encode()).hexdigest()


def replay_variables(harbor_task: HarborTask) -> list[str]:
    """The variables a runtime task's replay keeps besides the launcher's own (``launcher.REPLAY_VARIABLES``), which
    ``materialize`` writes into the launcher's settings as ``replay_env``: those of the template's ``[environment]
    env`` that the verifier's sandbox gets too, with the same value, and that the tests' own ``[verifier] env`` does
    not set, so that the replay runs as the solver's code ran and never sees what only the grader gets.  Names a
    process environment cannot hold portably are left out."""

    verifier = harbor_task.verifier_environment
    shared = verifier.env if verifier is not None else {}
    return sorted(
        name
        for name, value in harbor_task.env.items()
        if _VARIABLE.fullmatch(name) and shared.get(name) == value and name not in harbor_task.verifier_env
    )


def launch_settings(task_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The launcher's settings in a task's directory (``tests/platoon/launch.json``), the mark of a runtime task: only
    ``materialize`` writes them, since it refuses a directory whose ``tests/`` holds more than ``gen/``.  None for a
    directory without that file; an empty mapping for one whose file holds no settings the launcher could use, which is
    still a runtime task's (it never runs as an ordinary task)."""

    path = Path(task_dir) / TESTS_DIR / LAUNCHER_DIR / launcher.CONFIG
    try:
        path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError:
        return {}
    try:
        with open(path, "rb") as stream:
            settings = launcher.load_json(stream.read(MAX_SETTINGS_BYTES + 1).decode("utf-8"))
    except (OSError, ValueError):
        return {}
    return settings if isinstance(settings, dict) and settings.get("schema") == launcher.SCHEMA else {}


def in_store(task_dir: str | os.PathLike[str], store: Path) -> bool:
    """Whether ``task_dir`` is a runtime task of ``store``: a directory in it named by its digest."""

    path = Path(task_dir)
    return _DIGEST.fullmatch(path.name) is not None and path.resolve().parent == store


def stored(task_id: str, store: Path) -> Path | None:
    """The directory of the runtime task whose id is ``task_id`` (its digest) in ``store``, or None."""

    if _DIGEST.fullmatch(task_id) is None or not (store / task_id).is_dir():
        return None
    return store / task_id


def runtime_plan(harbor_task: HarborTask) -> dict[str, Any]:
    """The image plan of a runtime task, as a fetched dataset's lock holds one for a task with a ``docker_image``: a
    prebuilt image, whose sessions get the task's ``environment/`` uploaded into their working directory when it holds
    anything."""

    plan: dict[str, Any] = {"kind": "prebuilt", "ref": harbor_task.docker_image}
    environment = harbor_task.task_dir / ENVIRONMENT_DIR
    if environment.is_dir() and any(environment.iterdir()):
        plan["upload_environment"] = True
    return plan


def runtime_image(harbor_task: HarborTask, config: RuntimeTasksConfig) -> SessionImage:
    """What a runtime task's sandboxes run: the image ``images`` binds its ``docker_image`` to, with its
    ``environment/`` to upload (:func:`runtime_plan`).  ``ValueError`` for an image ``images`` does not list."""

    image = config.images.get(harbor_task.docker_image or "")
    if image is None:
        raise ValueError(
            f"Harbor task {harbor_task.name!r} runs {harbor_task.docker_image!r}, which runtime_tasks.images does not "
            "list"
        )
    return session_image(harbor_task.task_dir, plan=runtime_plan(harbor_task), image_map=None, docker_image=image)


def launcher_outcome(
    files: Mapping[str, VerifierFile],
    *,
    settings: Mapping[str, Any],
    exit_code: int | None,
    timed_out: bool,
    timeout_s: float,
) -> Outcome:
    """A runtime task's outcome: what its launcher concluded, from its report (``launcher.json`` of ``files``, what
    grading collected, which goes into the metadata as ``launcher``), with the grader's owner and the reward's name
    from the task's launcher settings (``settings``, read on the host).  The report is the whole grade: the reward file
    is the launcher's too, and the entry's verifier classifiers, its reading of reward files that are not regular files
    and its reward check apply to its other tasks only, since the grader, root beside the launcher, could write what
    they read (``test-stdout.txt`` among it).  A score of 0 the launcher gave keeps the solver's fault it names
    (``reason``: ``solver_output_contract``, ``solver_nonreproducible``); a non-score keeps its failure kind, with the
    launcher's detail as its reason.  A launcher that outlived the verifier's time, exited nonzero or left no report
    it writes did not finish (:func:`unfinished`), whatever the reward file says."""

    report = _report(files.get(launcher.REPORT))
    if timed_out:
        return unfinished(settings, f"the verifier outlived its {timeout_s:g} s", report, {"timeout_s": timeout_s})
    if exit_code != 0:
        return unfinished(settings, f"it exited {exit_code}", report)
    if report is None:
        return unfinished(settings, "it left no report", None)
    failure_kind, reward, reason = report.get("failure_kind"), report.get("reward"), report.get("reason")
    if failure_kind is not None:
        if failure_kind not in _failure_kinds(settings):
            return unfinished(settings, f"its report names the failure kind {_shown(failure_kind)}", report)
        detail = report.get("detail")
        return Outcome.infra(
            failure_kind,
            status="invalid",
            reason=detail[:_REASON_CHARS] if isinstance(detail, str) else None,
            metadata={"launcher": report},
        )
    scored = isinstance(reward, int) and not isinstance(reward, bool) and reward in (0, 1)
    if not scored or (reason is not None and (reward != 0 or reason not in _SOLVER_FAULTS)):
        return unfinished(settings, "its report holds no reward the launcher writes", report)
    key = settings["reward_key"]
    return Outcome(
        status="scored", rewards={key: float(reward)}, primary=key, reason=reason, metadata={"launcher": report}
    )


def unfinished(
    settings: Mapping[str, Any],
    why: str,
    report: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Outcome:
    """The outcome of a runtime task whose launcher did not finish its grade once it started, ``why`` saying how (it
    outlived the verifier's time, exited nonzero or left no report it writes, or grading could not kill what was left
    in the verifier or read its output): ``invalid``, ``<grader>_unfinished``, named by the grader's owner in the
    task's launcher settings (``verifier_generated_unfinished`` for a generated grader, ``verifier_trusted_unfinished``
    for a trusted one).  The grader runs as root beside the launcher, so it may have stopped it; so may a fault of the
    sandbox's, which a known-good task on the same environment tells apart.  ``report``, the launcher's, goes into the
    metadata
    as ``launcher`` when there is one."""

    return Outcome.infra(
        f"{grader_kind(settings)}_{launcher.UNFINISHED}",
        status="invalid",
        reason=f"the runtime task's launcher did not finish: {why}; no reward of its stands",
        metadata={**(metadata or {}), **({"launcher": dict(report)} if report is not None else {})},
    )


def grader_kind(settings: Mapping[str, Any]) -> str:
    """The prefix of the failure kinds of a runtime task's grader, by its owner in the task's launcher settings:
    ``verifier_generated`` for a generated grader (also where the settings name no owner) and ``verifier_trusted`` for
    a trusted one."""

    owner = settings.get("grader_owner")
    return launcher.GRADER_KINDS[owner if owner in launcher.OWNERS else "generated"]


def _failure_kinds(settings: Mapping[str, Any]) -> tuple[str, ...]:
    """The failure kinds a runtime task's launcher writes, by its grader's owner."""

    return (launcher.LAUNCHER_FAULT, *(f"{grader_kind(settings)}_{fault}" for fault in launcher.GRADER_FAULTS))


def _report(file: VerifierFile | None) -> dict[str, Any] | None:
    if file is None or file.kind != "file":
        return None
    try:
        report = launcher.load_json(file.text)
    except ValueError:
        return None
    return report if isinstance(report, dict) else None


def _copy_checked(source: Path, target: Path, max_bytes: int) -> list[str]:
    """Copy ``source``'s regular files and directories into the new directory ``target``, never following a link: why
    it is no runtime task's directory (a link, a special file, too many entries or bytes, an unreadable file), each
    one sentence; nothing is copied once one is found."""

    problems: list[str] = []
    total = entries = 0
    _directory(target)
    pending = [(source, target, "")]
    while pending:
        directory, into, prefix = pending.pop()
        try:
            listing = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            problems.append(f"{prefix or './'} cannot be read: {exc.strerror}")
            continue
        for entry in listing:
            name = f"{prefix}{entry.name}"
            entries += 1
            if entries > MAX_ENTRIES:
                return [*problems, f"the task holds more than {MAX_ENTRIES} files and directories"]
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as exc:
                problems.append(f"{name} cannot be read: {exc.strerror}")
                continue
            if stat.S_ISDIR(info.st_mode):
                if not problems:
                    _directory(into / entry.name)
                pending.append((Path(entry.path), into / entry.name, f"{name}/"))
            elif stat.S_ISREG(info.st_mode):
                total += info.st_size
                if total > max_bytes:
                    return [*problems, f"the task's files hold more than {max_bytes} bytes (runtime_tasks.max_bytes)"]
                if not problems and (problem := _copy_file(Path(entry.path), into / entry.name, info)) is not None:
                    problems.append(f"{name} {problem}")
            else:
                kind = "a symbolic link" if stat.S_ISLNK(info.st_mode) else "a special file"
                problems.append(f"{name} is {kind}: a runtime task holds regular files and directories only")
    return problems


def _copy_file(source: Path, target: Path, info: os.stat_result) -> str | None:
    """Copy the regular file ``source`` to the new file ``target``, executable where ``source`` is, never following a
    link; why it cannot be read, or None.  An error writing ``target`` is raised."""

    try:
        reading = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        return f"cannot be read: {exc.strerror}"
    with os.fdopen(reading, "rb") as source_file:
        if not stat.S_ISREG(os.fstat(source_file.fileno()).st_mode):
            return "changed while it was copied"
        with open(target, "xb") as target_file:
            while True:
                try:
                    chunk = source_file.read(1 << 20)
                except OSError as exc:
                    return f"cannot be read: {exc.strerror}"
                if not chunk:
                    break
                target_file.write(chunk)
            os.fchmod(target_file.fileno(), 0o755 if info.st_mode & stat.S_IXUSR else 0o644)
    return None


def _layout(task_dir: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """The task's grader manifest, and why its layout is no runtime task's: ``tests/`` that holds more than ``gen/``,
    no or a broken ``tests/gen/grader.json``, an ``environment/Dockerfile`` or compose file."""

    problems = []
    tests = task_dir / TESTS_DIR
    if tests.is_dir():
        for entry in sorted(tests.iterdir()):
            if entry.name != GEN_DIR:
                problems.append(
                    f"{TESTS_DIR}/{entry.name}: a runtime task's tests/ holds only gen/, its grader; tests/test.sh is "
                    "platoon's launcher, which materialize installs"
                )
    manifest, found = read_grader_manifest(tests / GEN_DIR / GRADER_MANIFEST)
    for name, why in _REFUSED_ENVIRONMENT_FILES:
        if (task_dir / ENVIRONMENT_DIR / name).exists():
            problems.append(f"{ENVIRONMENT_DIR}/{name}: {why}")
    return manifest, [*problems, *found]


def _task_toml(task_dir: Path, problems: list[str]) -> dict[str, Any] | None:
    """The task's ``task.toml`` (:func:`load_toml`), or None with why not added to ``problems``."""

    try:
        return load_toml((task_dir / CONFIG_FILENAME).read_text())
    except FileNotFoundError:
        problems.append(f"{CONFIG_FILENAME} is missing")
    except TomlNestedTooDeeply:
        problems.append(
            f"{CONFIG_FILENAME} is nested too deeply to read as TOML: more than {MAX_TOML_DEPTH} levels of tables and "
            "arrays"
        )
    except (OSError, ValueError) as exc:  # TOMLDecodeError, a bad encoding, and an integer past Python's digit limit
        problems.append(f"{CONFIG_FILENAME}: {exc}")
    return None


def _install_launcher(task_dir: Path) -> None:
    """Put platoon's launcher in the task's ``tests/``: ``test.sh``, and ``platoon/launcher.py`` (its settings come
    once the task loads)."""

    _directory(task_dir / TESTS_DIR / LAUNCHER_DIR)
    _write(task_dir / TESTS_DIR / TEST_SCRIPT, TEST_SH, mode=0o755)
    _write(task_dir / TESTS_DIR / LAUNCHER_DIR / LAUNCHER_FILE, Path(launcher.__file__).read_text())


def _compare(actual: Mapping[str, Any], expected: Mapping[str, Any], path: tuple[str, ...], found: list[str]) -> None:
    for key in sorted(set(actual) | set(expected), key=str):
        where = (*path, str(key))
        if key not in expected:
            found.append(f"sets {_key(where, actual[key])}, which the template does not")
        elif key not in actual:
            found.append(f"lacks {_key(where, expected[key])}, which the template sets to {_shown(expected[key])}")
        elif isinstance(actual[key], Mapping) and isinstance(expected[key], Mapping):
            _compare(actual[key], expected[key], where, found)
        elif not _same(actual[key], expected[key]):
            found.append(
                f"sets {_key(where, actual[key])} to {_shown(actual[key])}, and the template to {_shown(expected[key])}"
            )


def _same(value: Any, other: Any) -> bool:
    """Whether two TOML values are the same: numbers by value, but never a boolean for a number."""

    if isinstance(value, bool) or isinstance(other, bool):
        return type(value) is type(other) and value == other
    if isinstance(value, int | float) and isinstance(other, int | float):
        return value == other
    if isinstance(value, Mapping) and isinstance(other, Mapping):
        return value.keys() == other.keys() and all(_same(value[key], other[key]) for key in value)
    if isinstance(value, list | tuple) and isinstance(other, list | tuple):
        return len(value) == len(other) and all(_same(item, theirs) for item, theirs in zip(value, other))
    return type(value) is type(other) and value == other


def _key(path: tuple[str, ...], value: Any) -> str:
    """A key as TOML names it: ``[environment] network_mode``, a table as ``[verifier.environment]``."""

    if isinstance(value, Mapping):
        return f"[{'.'.join(path)}]"
    return f"[{'.'.join(path[:-1])}] {path[-1]}" if len(path) > 1 else path[0]


def _shown(value: Any) -> str:
    text = json.dumps(value, default=str, sort_keys=True)
    return text if len(text) <= 80 else f"{text[:77]}..."


def _without_own(config: Mapping[str, Any]) -> dict[str, Any]:
    plain: dict[str, Any] = _plain(config)
    for path in _OWN_KEYS:
        table = plain
        for key in path[:-1]:
            nested = table.get(key)
            table = nested if isinstance(nested, dict) else {}
        table.pop(path[-1], None)
    if plain.get("task") == {}:
        del plain["task"]
    return plain


def _plain(value: Any) -> Any:
    """``value`` with every mapping a dict and every sequence a list, copied."""

    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def _argv(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item and "\0" not in item for item in value)
    )


def _outputs(outputs: Any) -> bool:
    """Whether ``outputs`` lists a replay's outputs: distinct relative paths (``launcher.relative_path``), none below
    another."""

    if not (isinstance(outputs, list) and outputs and all(launcher.relative_path(output) for output in outputs)):
        return False
    return not any(
        other == output or other.startswith(f"{output}/") or output.startswith(f"{other}/")
        for index, output in enumerate(outputs)
        for other in outputs[index + 1 :]
    )


def _without(message: str, task_dir: Path) -> str:
    """``message`` with the staging directory's path left out: its files relative to the task."""

    for root in sorted({str(task_dir), str(task_dir.resolve())}, key=len, reverse=True):
        message = message.replace(f"{root}/", "").replace(root, "the task")
    return message


def _directory(path: Path) -> None:
    path.mkdir()
    os.chmod(path, 0o755)


def _write(path: Path, text: str, *, mode: int = 0o644) -> None:
    with open(path, "x") as stream:
        stream.write(text)
        os.fchmod(stream.fileno(), mode)
