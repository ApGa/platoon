"""Runtime Harbor tasks (``service_kwargs.runtime_tasks``, :mod:`platoon.envs.harbor.runtime`): the config and the
templates it refuses; ``materialize``, which refuses what breaks the entry's rules and never rewrites it, installs the
launcher and keeps the task under its digest; the sessions of a runtime task, which run the image ``runtime_tasks``
binds it to with the task's ``environment/`` uploaded and grade in a verifier of the runtime tasks' options; what a
grade makes of the launcher's report and exit; and the service's other tasks, which are served and graded as before.

The sandboxes are the process provider's, standing in for one whose verifiers run as root (:class:`RootfulProvider`).
Its commands run as this user, so the launcher itself refuses to grade (it needs root), which these tests check end to
end; ``tests/envs/test_harbor_runtime_launcher.py`` checks the launcher's grading."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import platoon.sandbox
from platoon.envs.harbor import HarborEnvironmentService, launcher, runtime
from platoon.envs.harbor.runtime import MaterializedTask, RuntimeTasksConfig, task_digest
from platoon.envs.harbor.task import HarborTask
from platoon.envs.harbor.verifier import VerifierFile
from platoon.envs.service import EnvSetupError
from platoon.sandbox import ExecResult, SandboxSpec
from platoon.sandbox.process import ProcessProvider, ProcessSandbox
from platoon.task import Task
from platoon.testing.runtime_tasks import to_toml

IMAGES = {"science": "/images/science.sif"}
TEMPLATE: dict[str, Any] = {
    "version": "1.0",
    "environment": {"docker_image": "science", "network_mode": "no-network", "cpus": 2, "memory_mb": 4096},
    "agent": {"timeout_sec": 300},
    "verifier": {"environment_mode": "separate", "user": "root", "timeout_sec": 120},
}
GRADER_JSON = {
    "command": ["python3", "grade.py"],
    "replay": {"command": ["python3", "solve.py"], "outputs": ["a.json"]},
}
ROOT = "${PLATOON_SANDBOX_ROOT:-}"


class RootfulProvider(ProcessProvider):
    """The process provider, standing in for one whose sandboxes run commands as root with other users mapped (as
    Singularity's --fakeroot does): asked who its commands run as, a sandbox answers root, with every id up to
    nobody's mapped, and that nobody may run its shell.  Its options are recorded (``tag`` tells two providers built
    with different options apart), and so is each spec it starts.  ``refusal``, ``network_refusal`` and
    ``layout_refusal`` are what its class says of root, of sandboxes without a network and of their layout."""

    def __init__(
        self,
        root: str,
        tag: str = "agents",
        refusal: str | None = None,
        network_refusal: str | None = None,
        layout_refusal: str | None = None,
    ) -> None:
        super().__init__(root)
        self.options = {"root": root, "tag": tag, "refusal": refusal}
        self.started: list[SandboxSpec] = []

    @classmethod
    def root_refusal(cls, options: Mapping[str, Any]) -> str | None:
        return options.get("refusal")

    @classmethod
    def network_none_refusal(cls, options: Mapping[str, Any]) -> str | None:
        return options.get("network_refusal")

    @classmethod
    def rootfs_layout_refusal(cls, options: Mapping[str, Any]) -> str | None:
        return options.get("layout_refusal")

    async def start(self, spec: SandboxSpec) -> ProcessSandbox:
        sandbox = await super().start(spec)
        real = sandbox.exec

        async def exec(cmd: str | list[str], **options: Any) -> ExecResult:
            if isinstance(cmd, list) and "platoon-root-check" in cmd:
                return ExecResult(
                    0, "uid 0\n" + "".join(f"{kind}_map 0 1\n{kind}_map 1 65536\n" for kind in ("uid", "gid")), ""
                )
            if isinstance(cmd, list) and "platoon-nobody-check" in cmd:
                return ExecResult(0, "ran\n", "")
            return await real(cmd, **options)

        sandbox.exec = exec  # type: ignore[method-assign]
        self.started.append(spec)
        return sandbox


@pytest.fixture
def rootful(monkeypatch) -> str:
    monkeypatch.setitem(platoon.sandbox._BUILTIN_PROVIDERS, "rootful-process", f"{__name__}:RootfulProvider")
    return "rootful-process"


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return tmp_path / "store"


def make_service(tmp_path: Path, provider: str, store: Path, **options: Any) -> HarborEnvironmentService:
    config = {"images": IMAGES, "template": TEMPLATE, "store": str(store), **options.pop("runtime_tasks", {})}
    return HarborEnvironmentService(
        tmp_path / "curated",
        provider=provider,
        provider_options={"root": str(tmp_path / "sandboxes"), **options.pop("provider_options", {})},
        agent_user="ignore",
        runtime_tasks=config,
        **options,
    )


def write_task(root: Path, name: str = "generated", *, edit: Any = None, files: dict[str, str] | None = None) -> Path:
    """An honest runtime task written from the template: an instruction, data in ``environment/``, a solution, and a
    grader with its manifest in ``tests/gen``; ``edit`` changes its ``task.toml``'s tables, ``files`` adds files (or,
    with None, takes them away)."""

    config = json.loads(json.dumps(TEMPLATE))
    config["task"] = {"name": name}
    config["artifacts"] = ["solve.py", "a.json"]
    if edit is not None:
        edit(config)
    task_dir = root / name
    contents = {
        "task.toml": to_toml(config),
        "instruction.md": "Compute it.\n",
        "environment/data.txt": "21\n",
        "environment/tools/run.sh": "#!/bin/sh\necho run\n",
        "solution/solve.sh": "echo solved\n",
        "tests/gen/grade.py": "print('grading')\n",
        "tests/gen/grader.json": json.dumps(GRADER_JSON),
        **(files or {}),
    }
    for relative, text in contents.items():
        path = task_dir / relative
        if text is None:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (task_dir / "environment" / "tools" / "run.sh").chmod(0o755)
    return task_dir


def edit(table: str, key: str, value: Any):
    def change(config: dict[str, Any]) -> None:
        config.setdefault(table, {})[key] = value

    return change


# ---------------------------------------------------------------------------------------------------------- the config


def test_a_config_reads_its_template_from_a_toml_file(tmp_path: Path):
    template = tmp_path / "template.toml"
    template.write_text(to_toml(TEMPLATE))

    config = RuntimeTasksConfig.coerce({"images": IMAGES, "template": str(template)})

    assert config.template == TEMPLATE
    assert (config.verifier_user, config.verifier_provider_options, config.max_bytes) == ("honor", {}, 256 * 2**20)


def _without(table: str, key: str) -> dict[str, Any]:
    changed = json.loads(json.dumps(TEMPLATE))
    del changed[table][key]
    return changed


def _with(table: str, key: str, value: Any) -> dict[str, Any]:
    changed = json.loads(json.dumps(TEMPLATE))
    changed.setdefault(table, {})[key] = value
    return changed


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"images": {}}, "runtime_tasks.images must map"),
        ({"template": _without("environment", "network_mode")}, "it gives the agents a network"),
        ({"template": _with("agent", "network_mode", "public")}, "it gives the agents a network"),
        ({"template": _without("verifier", "environment_mode")}, "it grades in the agents' own sandbox"),
        (
            {"template": _with("verifier", "environment", {"docker_image": "x", "network_mode": "no-network"})},
            "whose image brings its own /tests",
        ),
        ({"template": _with("verifier", "environment", {"cpus": 1})}, "it gives the verifier a network"),
        ({"template": _with("environment", "docker_image", "other")}, "is none of runtime_tasks.images"),
        ({"template": {**TEMPLATE, "artifacts": ["a"]}}, "it gives artifacts"),
        ({"template": _with("task", "name", "fixed")}, r"it gives \[task\] name"),
        ({"template": _with("environment", "os", "windows")}, "only Linux tasks"),
        ({"verifier_user": "obey"}, "runtime_tasks.verifier_user must be one of"),
        ({"max_bytes": 0}, "runtime_tasks.max_bytes must be a positive"),
        ({"verifier_provider_options": ["fakeroot"]}, "verifier_provider_options must be a mapping"),
        ({"regrade": 2}, "runtime_tasks takes images, template, .* not regrade"),
    ],
)
def test_a_config_refuses_what_runtime_tasks_cannot_run_as(options: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=message):
        RuntimeTasksConfig.coerce({"images": IMAGES, "template": TEMPLATE, **options})


def test_runtime_verifier_options_need_a_provider_named_by_its_name(tmp_path: Path):
    with pytest.raises(ValueError, match="runtime_tasks.verifier_provider_options need a provider named"):
        HarborEnvironmentService(
            tmp_path,
            provider=ProcessProvider(tmp_path),
            runtime_tasks={"images": IMAGES, "template": TEMPLATE, "verifier_provider_options": {"fakeroot": True}},
        )


def test_a_config_refuses_a_template_nested_deeper_than_a_runtime_task_may_be(tmp_path: Path):
    """A template that nests deeper than a runtime task's ``task.toml`` may (``runtime.MAX_TOML_DEPTH``), which no task
    could then match, is refused, as a mapping and as a TOML file, and the file before it is parsed where its text
    shows it, the same way on every Python."""

    deep: dict[str, Any] = {}
    for _ in range(100_000):
        deep = {"d": deep}
    with pytest.raises(ValueError, match="runtime_tasks.template is nested too deeply: more than 64 levels"):
        RuntimeTasksConfig.coerce({"images": IMAGES, "template": {**TEMPLATE, "metadata": deep}})
    template = tmp_path / "template.toml"
    for deeply in ("deep = " + "[" * 100_000, ".".join(["d"] * 100_000) + " = 1"):
        template.write_text(f"{deeply}\n{to_toml(TEMPLATE)}")
        with pytest.raises(ValueError, match=re.escape(f"runtime_tasks.template: {template}: nested too deeply")):
            RuntimeTasksConfig.coerce({"images": IMAGES, "template": str(template)})


def test_the_store_is_in_the_runs_directory_else_in_the_harbor_cache(tmp_path: Path, monkeypatch):
    from platoon.sandbox import cleanup

    config = RuntimeTasksConfig.coerce({"images": IMAGES, "template": TEMPLATE})
    monkeypatch.setattr(runtime, "default_cache", lambda: tmp_path / "cache" / "harbor")
    assert config.store_path() == (tmp_path / "cache" / "harbor" / "runtime-tasks").resolve()
    with cleanup.labelled(cleanup.run_label(tmp_path / "run")):
        assert config.store_path() == (tmp_path / "run" / "runtime-tasks").resolve()
    named = RuntimeTasksConfig.coerce({"images": IMAGES, "template": TEMPLATE, "store": str(tmp_path / "mine")})
    assert named.store_path() == (tmp_path / "mine").resolve()


# --------------------------------------------------------------------------------------------------------- materialize


@pytest.mark.asyncio
async def test_materialize_keeps_a_checked_copy_under_its_digest_with_the_launcher(tmp_path, rootful, store):
    service = make_service(
        tmp_path, rootful, store, reward={"primary": ["score", "reward"]}, verifier_timeout_multiplier=2
    )
    source = write_task(tmp_path / "sources")

    made = await service.materialize(source)

    assert made.ok and made.problems == () and len(made.digest) == 64
    stored = store / made.digest
    assert service.runtime_store == store.resolve()
    assert made.task.id == made.digest and made.task.goal == "Compute it.\n"
    assert made.task.misc["harbor"] == {
        "task_dir": str(stored.resolve()),
        "name": "generated",
        "agent_timeout_s": 300,
        "digest": made.digest,
        "runtime": True,
    }
    assert (stored / "environment" / "data.txt").read_text() == "21\n"
    assert stat.S_IMODE((stored / "environment" / "tools" / "run.sh").stat().st_mode) == 0o755
    assert stat.S_IMODE((stored / "environment" / "data.txt").stat().st_mode) == 0o644
    assert stat.S_IMODE((stored / "tests" / "test.sh").stat().st_mode) == 0o755
    assert (stored / "tests" / "platoon" / "launcher.py").read_text() == Path(runtime.launcher.__file__).read_text()
    assert json.loads((stored / "tests" / "platoon" / "launch.json").read_text()) == {
        "schema": "platoon.launcher/v1",
        "fingerprint": runtime.config_fingerprint(service.runtime_tasks, reward_key="score", timeout_multiplier=2),
        "grader_owner": "generated",
        "reward_key": "score",  # the name the reward contract trains first
        "timeout_s": 240.0,  # [verifier] timeout_sec times verifier_timeout_multiplier
        "max_output_bytes": runtime.MAX_OUTPUT_BYTES,
        "max_copy_bytes": runtime.DEFAULT_MAX_BYTES + runtime.launcher.MAX_SUBMISSION_BYTES,
        "artifacts": ["solve.py", "a.json"],
        "grader": {"command": ["python3", "grade.py"]},
        "replay": {"command": ["python3", "solve.py"], "outputs": ["a.json"]},
        "replay_env": [],
        "max_address_space_bytes": None,  # no per-process RLIMIT_AS by default
    }
    assert sorted(path.name for path in (stored / "tests").iterdir()) == ["gen", "platoon", "test.sh"]
    assert not (source / "tests" / "test.sh").exists()  # the source is left as it was
    assert list((store / ".staging").iterdir()) == []
    assert task_digest(stored) == made.digest
    assert HarborTask.load(stored, agent_user="ignore", verifier_user="honor").verifier_root


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verifier_environment", "kept"),
    [
        (None, ["DATA_DIR"]),
        ({"network_mode": "no-network"}, []),  # a verifier's own table: its sandbox gets none of [environment] env
        ({"network_mode": "no-network", "env": {"DATA_DIR": "/data"}}, ["DATA_DIR"]),
    ],
    ids=["shared", "own-table", "own-table-alike"],
)
async def test_a_replay_keeps_the_templates_variables_the_verifier_shares_and_its_tests_leave_alone(
    tmp_path, rootful, store, verifier_environment, kept
):
    """A replay runs as the solver's code ran, with the variables of the template's ``[environment] env`` that the
    verifier's sandbox gets too with the same value, and never one the tests' ``[verifier] env`` sets, which may be
    the grader's own."""

    def with_env(config: dict[str, Any]) -> None:
        config["environment"]["env"] = {"DATA_DIR": "/data", "TOKEN": "agents", "NOT A NAME": "x"}
        config["verifier"]["env"] = {"TOKEN": "${GRADER_TOKEN:-}"}
        if verifier_environment is not None:
            config["verifier"]["environment"] = dict(verifier_environment)

    template = json.loads(json.dumps(TEMPLATE))
    with_env(template)
    service = make_service(tmp_path, rootful, store, runtime_tasks={"template": template})

    made = await service.materialize(write_task(tmp_path / "sources", edit=with_env))

    assert made.ok, made.problems
    launch = json.loads((store / made.digest / "tests" / "platoon" / "launch.json").read_text())
    assert launch["replay_env"] == kept


@pytest.mark.asyncio
async def test_the_same_task_twice_is_one_directory_and_its_graders_owner_another(tmp_path, rootful, store):
    service = make_service(tmp_path, rootful, store)
    source = write_task(tmp_path / "sources")

    first, again = await service.materialize(source), await service.materialize(source)
    trusted = await service.materialize(source, grader_owner="trusted")

    assert first.digest == again.digest != trusted.digest
    assert sorted(path.name for path in store.iterdir()) == sorted([".staging", first.digest, trusted.digest])
    launch = json.loads((store / trusted.digest / "tests" / "platoon" / "launch.json").read_text())
    assert launch["grader_owner"] == "trusted"


@pytest.mark.asyncio
async def test_materialize_needs_runtime_tasks_and_a_known_graders_owner(tmp_path):
    plain = HarborEnvironmentService(tmp_path, provider="process")
    with pytest.raises(ValueError, match="takes no runtime tasks"):
        await plain.materialize(tmp_path)
    service = make_service(tmp_path, "process", tmp_path / "store")
    with pytest.raises(ValueError, match="grader_owner must be one of"):
        await service.materialize(tmp_path, grader_owner="nobody")


def _fifo(task_dir: Path) -> None:
    os.mkfifo(task_dir / "environment" / "pipe")


def _symlink(task_dir: Path) -> None:
    (task_dir / "environment" / "passwd").symlink_to("/etc/passwd")


def _oversized(task_dir: Path) -> None:
    with open(task_dir / "environment" / "big.bin", "wb") as stream:
        stream.truncate(4097)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"edit": edit("environment", "network_mode", "public")}, 'sets [environment] network_mode to "public", and'),
        ({"edit": edit("environment", "gpus", 1)}, "task.toml sets [environment] gpus, which the template does not"),
        ({"edit": edit("environment", "docker_image", "other")}, 'sets [environment] docker_image to "other"'),
        ({"edit": edit("verifier", "environment_mode", "shared")}, '[verifier] environment_mode to "shared"'),
        (
            {"edit": lambda config: config["agent"].pop("timeout_sec")},
            "lacks [agent] timeout_sec, which the template sets",
        ),
        ({"edit": edit("environment", "cpus", True)}, "sets [environment] cpus to true, and the template to 2"),
        ({"edit": edit("verifier", "environment", {})}, "sets [verifier.environment], which the template does not"),
        (
            {"files": {"environment/Dockerfile": "FROM scratch\n"}},
            "environment/Dockerfile: a runtime task runs a prebuilt",
        ),
        ({"files": {"environment/docker-compose.yaml": "services: {}\n"}}, "never a multi-container stack"),
        ({"files": {"tests/test.sh": "echo 1\n"}}, "tests/test.sh: a runtime task's tests/ holds only gen/"),
        (
            {"files": {"tests/Dockerfile": "FROM scratch\n"}},
            "tests/Dockerfile: a runtime task's tests/ holds only gen/",
        ),
        ({"files": {"tests/gen/grader.json": None}}, "tests/gen/grader.json is missing"),
        ({"files": {"tests/gen/grader.json": "{"}}, "tests/gen/grader.json is not JSON"),
        ({"files": {"tests/gen/grader.json": '{"command": "python3 grade.py"}'}}, "command must be the grader's argv"),
        ({"files": {"tests/gen/grader.json": '{"command": ["a"], "timeout_sec": 5}'}}, "holds timeout_sec, beyond"),
        (
            {
                "files": {
                    "tests/gen/grader.json": '{"command": ["a"], "replay": {"command": ["b"], "outputs": ["../x"]}}'
                }
            },
            "replay.outputs must list the files the replay writes",
        ),
        (
            {
                "files": {
                    "tests/gen/grader.json": '{"command": ["a"], "replay": {"command": ["b"], "outputs": ["o", "o/a"]}}'
                }
            },
            "replay.outputs must list",
        ),
        ({"files": {"task.toml": None}}, "task.toml is missing"),
        ({"files": {"task.toml": "[environment\n"}}, "task.toml: "),
        ({"files": {"instruction.md": None}}, "the task has no instruction.md"),
        (
            {"edit": lambda config: config["artifacts"].append("/tests/gen")},
            'declares the artifact "/tests/gen", which is no path below the working directory',
        ),
        ({"edit": lambda config: config["artifacts"].append("/logs/verifier")}, 'the artifact "/logs/verifier", which'),
        ({"edit": lambda config: config["artifacts"].append({"source": "/solution"})}, 'the artifact "/solution"'),
        ({"edit": lambda config: config["artifacts"].append("./a.json")}, 'the artifact "./a.json", which is no path'),
        ({"edit": lambda config: config["artifacts"].append("out//a.json")}, 'the artifact "out//a.json"'),
        ({"edit": lambda config: config["artifacts"].append("x" * 1025)}, "of at most 1024 characters"),
        (
            {"edit": lambda config: config.update(artifacts=[f"a{n}" for n in range(257)])},
            "task.toml declares 257 artifacts, more than a runtime task may (256)",
        ),
        ({"setup": _symlink}, "environment/passwd is a symbolic link: a runtime task holds regular files"),
        ({"setup": _fifo}, "environment/pipe is a special file"),
        ({"setup": _oversized}, "the task's files hold more than 4096 bytes (runtime_tasks.max_bytes)"),
    ],
)
async def test_materialize_refuses_what_breaks_the_rules_and_rewrites_nothing(
    tmp_path, rootful, store, change, problem
):
    service = make_service(tmp_path, rootful, store, runtime_tasks={"max_bytes": 4096})
    source = write_task(tmp_path / "sources", edit=change.get("edit"), files=change.get("files"))
    if (setup := change.get("setup")) is not None:
        setup(source)
    before = sorted(str(path.relative_to(source)) for path in source.rglob("*"))

    made = await service.materialize(source)

    assert made == MaterializedTask(task=None, digest=None, problems=made.problems) and not made.ok
    assert any(problem in found for found in made.problems), made.problems
    assert not any(".staging" in found for found in made.problems)  # nothing names where it was checked
    assert [path.name for path in store.iterdir()] == [".staging"] and list((store / ".staging").iterdir()) == []
    assert sorted(str(path.relative_to(source)) for path in source.rglob("*")) == before


@pytest.mark.asyncio
async def test_a_refusal_lists_every_problem_it_found(tmp_path, rootful, store):
    service = make_service(tmp_path, rootful, store)
    source = write_task(
        tmp_path / "sources",
        edit=edit("environment", "network_mode", "public"),
        files={"tests/test.sh": "x", "environment/Dockerfile": "FROM x\n"},
    )

    made = await service.materialize(source)

    assert len(made.problems) == 3, made.problems


@pytest.mark.asyncio
async def test_materialize_refuses_more_entries_than_a_runtime_task_holds(tmp_path, rootful, store, monkeypatch):
    monkeypatch.setattr(runtime, "MAX_ENTRIES", 12)
    service = make_service(tmp_path, rootful, store)

    made = await service.materialize(
        write_task(tmp_path / "sources", files={f"environment/f{n}": "" for n in range(9)})
    )

    assert made.problems == ("the task holds more than 12 files and directories",)


@pytest.mark.asyncio
async def test_a_source_that_is_not_a_directory_is_refused(tmp_path, rootful, store):
    service = make_service(tmp_path, rootful, store)
    (tmp_path / "file").write_text("x")

    assert (await service.materialize(tmp_path / "file")).problems == (f"{tmp_path / 'file'} is not a directory",)
    assert "cannot be read" in (await service.materialize(tmp_path / "nothing")).problems[0]


@pytest.mark.parametrize(
    "deep",
    ["[" * (launcher.MAX_JSON_DEPTH + 1) + "]" * (launcher.MAX_JSON_DEPTH + 1), "[" * 60_000],
    ids=["parsable", "beyond-some-parsers"],
)
def test_platoon_reads_a_runtime_tasks_json_no_deeper_than_its_launcher_does(tmp_path, deep):
    """The grader manifest, the launcher's settings and its report nested deeper than the launcher reads
    (``launcher.MAX_JSON_DEPTH``) are refused before they are parsed, the same way on every Python, whether its parser
    would take them or give up (a ``RecursionError``)."""

    manifest = tmp_path / "grader.json"
    manifest.write_text(deep)
    refused = "tests/gen/grader.json is nested too deeply to read as JSON"
    assert runtime.read_grader_manifest(manifest) == (None, [refused])
    settings = tmp_path / "tests" / "platoon" / launcher.CONFIG
    settings.parent.mkdir(parents=True)
    settings.write_text(f'{{"schema": "{launcher.SCHEMA}", "deep": {deep}}}')
    assert runtime.launch_settings(tmp_path) == {}
    report = VerifierFile("file", f'{{"schema": "{launcher.SCHEMA}", "reward": 1, "deep": {deep}}}')
    outcome = runtime.launcher_outcome(
        {launcher.REPORT: report},
        settings={"grader_owner": "generated", "reward_key": "reward"},
        exit_code=0,
        timed_out=False,
        timeout_s=60.0,
    )
    assert (outcome.status, outcome.failure_kind) == ("invalid", "verifier_generated_unfinished")
    assert outcome.reason is not None and "it left no report" in outcome.reason


# A task.toml nested deeper than runtime tasks may be (runtime.MAX_TOML_DEPTH, 64 levels with its own table), each
# written around the honest one: arrays and inline tables that every Python's parser takes, arrays that every Python's
# parser gives up on (a RecursionError), a dotted key that recent 3.13 and 3.14 releases give up on while 3.12 parses
# it, and a header and a key, each shallow enough, that nest too deeply together.
DEEP_TOML = {
    "arrays": lambda toml: "deep = " + "[" * 64 + "]" * 64 + "\n" + toml,
    "inline-tables": lambda toml: "deep = " + "{d = " * 64 + "1" + "}" * 64 + "\n" + toml,
    "beyond-every-parser": lambda toml: "deep = " + "[" * 100_000 + "\n" + toml,
    "dotted-key": lambda toml: ".".join(["d"] * 2_000) + " = 1\n" + toml,
    "header-and-key": lambda toml: toml + "[" + ".".join(["h"] * 40) + "]\n" + ".".join(["k"] * 40) + " = 1\n",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("deepen", DEEP_TOML.values(), ids=DEEP_TOML.keys())
async def test_a_task_toml_nested_too_deeply_is_refused_the_same_way_on_every_python(tmp_path, rootful, store, deepen):
    service = make_service(tmp_path, rootful, store)
    source = write_task(tmp_path / "sources")
    (source / "task.toml").write_text(deepen((source / "task.toml").read_text()))

    made = await service.materialize(source)

    assert made.problems == (
        "task.toml is nested too deeply to read as TOML: more than 64 levels of tables and arrays",
    )
    assert [path.name for path in store.iterdir()] == [".staging"] and list((store / ".staging").iterdir()) == []


@pytest.mark.asyncio
async def test_a_task_toml_whose_integer_python_cannot_read_is_refused_not_raised(tmp_path, rootful, store):
    """``tomllib`` reads an integer with ``int()``, which raises a plain ``ValueError`` (no ``TOMLDecodeError``) past
    Python's limit on the digits of an integer: a refusal, as any other unreadable ``task.toml`` is."""

    service = make_service(tmp_path, rootful, store)
    source = write_task(tmp_path / "sources")
    (source / "task.toml").write_text("big = " + "1" * 4_301 + "\n" + (source / "task.toml").read_text())

    made = await service.materialize(source)

    assert len(made.problems) == 1 and made.problems[0].startswith("task.toml: ")
    assert "4300 digits" in made.problems[0]


def test_a_template_whose_integer_python_cannot_read_is_refused_with_its_path(tmp_path: Path):
    template = tmp_path / "template.toml"
    template.write_text("big = " + "1" * 4_301 + "\n" + to_toml(TEMPLATE))
    with pytest.raises(ValueError, match=re.escape(f"runtime_tasks.template: {template}: ")):
        RuntimeTasksConfig.coerce({"images": IMAGES, "template": str(template)})


@pytest.mark.parametrize(
    ("text", "depth"),
    [
        ("", 1),  # the document's own table
        ("a = 1", 1),
        ("a = [1, [2, {b = 3}]]", 4),
        ("a.b.c = 1", 3),  # a dotted key's parts
        ("a . 'b.c' . \"d.e\" = 1", 3),  # a quoted key's dots are its text
        ("[a.b]\nc = 1", 2),  # a header's brackets and its key's parts
        ("[[a]]", 3),
        ("a = '[[[' # [[[", 1),  # brackets in strings and comments are text
        ('a = "\\"[[[\\\\" # ]', 1),  # escapes: a quote, a backslash
        ('a = """\n"[[""\\"""[{"""', 1),  # quotes in a multi-line string, which ends at three
        ("a = '''\n[[[''''\nb = [1]", 2),  # a multi-line literal string ending in a quote
        ("a = " + "[" * 3, 4),  # what a text opens counts, closed or not
        ('a = "[[\nb = [[', 3),  # a string that never ends runs to the end of its line
        ("a = 1.5", 2),  # a float's two parts, which never make a document too deep alone
        ("]]][", 1),  # a parser stops at the first ]
    ],
)
def test_toml_text_depth_reads_how_deeply_a_text_nests_without_parsing_it(text, depth):
    assert runtime.toml_text_depth(text) == depth


def test_load_toml_refuses_toml_nested_deeper_than_runtime_tasks_may_be():
    """TOML at ``MAX_TOML_DEPTH`` parses, on every Python; a level deeper is refused, before it is parsed where its text
    shows it (so never where a Python's parser would give up, or take a key's parts in quadratic time), else once it is
    parsed, as a ``ValueError``, as text that is not TOML is."""

    limit = runtime.MAX_TOML_DEPTH
    for text in (
        "a = " + "[" * (limit - 1) + "]" * (limit - 1),
        "a = " + "{b = " * (limit - 1) + "1" + "}" * (limit - 1),
        ".".join(["a"] * limit) + " = 1",
        "[" + ".".join(["h"] * 32) + "]\n" + ".".join(["k"] * 32) + " = 1",
    ):
        assert runtime.toml_depth(runtime.load_toml(text)) == limit
    assert runtime.load_toml('a = "' + "[" * 100_000 + '"') == {"a": "[" * 100_000}
    for text in (
        "a = " + "[" * limit + "]" * limit,
        "a = " + "{b = " * limit + "1" + "}" * limit,
        "a = " + "[" * 100_000,
        ".".join(["a"] * (limit + 1)) + " = 1",
        ".".join(["a"] * 100_000),  # no TOML: a key without a value
        "[" + ".".join(["h"] * 33) + "]\n" + ".".join(["k"] * 33) + " = 1",
    ):
        with pytest.raises(runtime.TomlNestedTooDeeply, match="nested too deeply: more than 64 levels") as refused:
            runtime.load_toml(text)
        assert isinstance(refused.value, ValueError)


# ----------------------------------------------------------------------------------------------------------- sessions


@pytest.mark.asyncio
async def test_a_runtime_task_runs_its_listed_image_with_its_environment_uploaded(tmp_path, rootful, store):
    service = make_service(tmp_path, rootful, store, provider_options={"tag": "agents"})
    made = await service.materialize(write_task(tmp_path / "sources"))

    session = await service.open(made.task, rollout_id="r")
    try:
        listed = await session.call_tool("bash", {"command": "cat data.txt; ls tools"})
    finally:
        await session.close()

    assert listed.content[0]["text"].split() == ["21", "run.sh"]
    (spec,) = service.provider.started
    assert spec.image == "/images/science.sif" and spec.network == "none" and spec.workdir == "/app"
    assert (spec.cpus, spec.memory_mb) == (2, 4096)


@pytest.mark.asyncio
async def test_a_runtime_tasks_verifier_starts_with_its_own_options_and_the_others_with_the_entrys(
    tmp_path, rootful, store
):
    curated = tmp_path / "curated" / "plain"
    curated.mkdir(parents=True)
    (curated / "task.toml").write_text('artifacts = ["/app/a.txt"]\n[verifier]\nenvironment_mode = "separate"\n')
    (curated / "instruction.md").write_text("Do it.\n")
    (curated / "tests").mkdir()
    (curated / "tests" / "test.sh").write_text(f"echo 1 > {ROOT}/logs/verifier/reward.txt\n")
    service = make_service(
        tmp_path,
        rootful,
        store,
        default_image="local",
        verifier_provider_options={"tag": "verifiers"},
        runtime_tasks={"verifier_provider_options": {"tag": "runtime verifiers"}},
    )
    made = await service.materialize(write_task(tmp_path / "sources"))

    for task in (made.task, await service.get_task("plain")):
        session = await service.open(task, rollout_id="r")
        try:
            await session.grade()
        finally:
            await session.close()

    assert service.provider.options["tag"] == "agents"
    assert service.verifier_provider.options["tag"] == "verifiers"
    assert service.runtime_verifier_provider.options["tag"] == "runtime verifiers"
    assert service.runtime_verifier_provider.options["root"] == service.provider.options["root"]  # laid over
    ((runtime_spec,), (curated_spec,)) = service.runtime_verifier_provider.started, service.verifier_provider.started
    assert runtime_spec.labels["platoon.task"] == "generated" and curated_spec.labels["platoon.task"] == "plain"
    assert runtime_spec.image == "/images/science.sif" and curated_spec.image == "local"


@pytest.mark.asyncio
async def test_runtime_tasks_get_their_own_verifier_user_and_the_others_the_entrys(tmp_path, rootful, store):
    """The entry refuses tasks that name users; its runtime tasks' root verifiers are honored."""

    service = HarborEnvironmentService(
        tmp_path / "curated",
        provider=rootful,
        provider_options={"root": str(tmp_path / "sandboxes")},
        runtime_tasks={"images": IMAGES, "template": TEMPLATE, "store": str(store)},
    )
    made = await service.materialize(write_task(tmp_path / "sources"))
    assert made.ok, made.problems

    assert service.harbor_task(made.task).verifier_root
    copied = tmp_path / "curated" / "copy"
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(store / made.digest, copied)  # the same files outside the store, which the task's own directory
    (copied / "tests" / "platoon" / "launch.json").unlink()  # marks a runtime task's, without it: an ordinary task
    with pytest.raises(ValueError, match=r"\[verifier\].user = 'root', and the environment refuses tasks that name"):
        service.harbor_task(HarborTask.load(copied, agent_user="ignore").to_task("copy"))


@pytest.mark.asyncio
async def test_a_runtime_task_of_another_store_is_refused_when_it_opens(tmp_path, rootful, store):
    made = await make_service(tmp_path, rootful, store).materialize(write_task(tmp_path / "sources"))
    other = make_service(tmp_path, rootful, tmp_path / "other-store")

    with pytest.raises(EnvSetupError, match="is a runtime task of another store") as refused:
        await other.open(made.task, rollout_id="r")

    assert refused.value.failure_kind == "env_config"


def _wide(template: dict[str, Any]) -> dict[str, Any]:
    """The template with more of everything: what one entry gives its runtime tasks, and another does not."""

    wider = json.loads(json.dumps(template))
    wider["environment"]["cpus"] = 16
    wider["agent"]["timeout_sec"] = 7200
    return wider


@pytest.mark.asyncio
async def test_a_runtime_task_checked_against_other_runtime_tasks_is_refused_where_the_store_is_shared(
    tmp_path, rootful, store
):
    """Two entries that share a store (as entries that leave ``store`` unset do: the run's) each run only the runtime
    tasks checked against their own ``runtime_tasks``; so does an entry whose ``runtime_tasks`` changed since, as a
    resumed run's may."""

    wide = make_service(tmp_path, rootful, store, runtime_tasks={"template": _wide(TEMPLATE)})
    made = await wide.materialize(write_task(tmp_path / "sources", edit=lambda config: config.update(_wide(config))))
    assert made.ok, made.problems
    narrow = make_service(tmp_path, rootful, store)
    assert not (await narrow.materialize(Path(made.task.misc["harbor"]["task_dir"]))).ok  # its template refuses it

    for refuse in (narrow.open(made.task, rollout_id="r"), narrow.get_task(made.digest)):
        with pytest.raises(EnvSetupError, match="checked against other runtime_tasks") as refused:
            await refuse
        assert refused.value.failure_kind == "env_config"
    with pytest.raises(ValueError, match="checked against other runtime_tasks"):
        narrow.agent_timeout_s(made.digest)
    same = make_service(tmp_path, rootful, store, runtime_tasks={"template": _wide(TEMPLATE)})  # another, alike
    assert same.harbor_task(made.task).name == "generated"
    rewarded = make_service(tmp_path, rootful, store, runtime_tasks={"template": _wide(TEMPLATE)}, reward="score")
    with pytest.raises(ValueError, match="checked against other runtime_tasks"):
        rewarded.harbor_task(made.task)  # its launcher writes another reward than this entry trains


@pytest.mark.asyncio
async def test_an_environment_without_runtime_tasks_refuses_a_runtime_task_never_running_it_as_its_own(
    tmp_path, rootful, store
):
    made = await make_service(tmp_path, rootful, store).materialize(write_task(tmp_path / "sources"))
    plain = HarborEnvironmentService(tmp_path / "curated", provider=rootful, agent_user="ignore")
    unmarked = Task(
        id="copied", goal="Compute it.", misc={"harbor": {"task_dir": made.task.misc["harbor"]["task_dir"]}}
    )

    for task in (made.task, unmarked):  # marked a runtime task, or not: its directory says what it is
        with pytest.raises(EnvSetupError, match="is a runtime task, and the Harbor environment .* takes none"):
            await plain.open(task, rollout_id="r")
    listed = HarborEnvironmentService(  # a curated task's directory, which honors its root verifier, as one would
        task_dirs=[made.task.misc["harbor"]["task_dir"]], provider=rootful, agent_user="ignore", verifier_user="honor"
    )
    with pytest.raises(EnvSetupError, match="takes none"):
        await listed.open(await listed.get_task(made.digest), rollout_id="r")


@pytest.mark.asyncio
async def test_a_task_marked_a_runtime_task_whose_directory_is_none_is_refused(tmp_path, rootful, store):
    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources"))
    (Path(made.task.misc["harbor"]["task_dir"]) / "tests" / "platoon" / "launch.json").unlink()

    with pytest.raises(EnvSetupError, match="is marked a runtime task, and .* holds none") as refused:
        await service.open(made.task, rollout_id="r")

    assert refused.value.failure_kind == "env_config"


@pytest.mark.asyncio
async def test_a_runtime_task_is_found_by_its_digest(tmp_path, rootful, store):
    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources"))

    task = await service.get_task(made.digest)

    assert (task.id, task.misc["harbor"]) == (made.digest, made.task.misc["harbor"])
    assert service.agent_timeout_s(made.digest) == 300
    assert service.rollout_overhead_s(made.digest) > 120  # its verifier's start and time limit
    with pytest.raises(KeyError, match="Unknown Harbor task"):
        await service.get_task("0" * 64)  # no such runtime task: a task of the dataset directory, which has none


# --------------------------------------------------------------------------------------------------------- grading


async def _graded(service: HarborEnvironmentService, task) -> Any:
    session = await service.open(task, rollout_id="r")
    try:
        assert not (await session.call_tool("bash", {"command": "echo 'x' > solve.py; echo '{}' > a.json"})).is_error
        return await session.grade()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_launcher_that_does_not_run_as_root_fails_closed_as_a_trusted_fault(tmp_path, rootful, store):
    """The process provider runs the verifier's commands as this user: the launcher grades nothing, and says why."""

    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources"))

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.failure_kind) == ("invalid", "verifier_trusted_launcher")
    assert outcome.metadata["verifier_exit_code"] == 0 and outcome.metadata["verifier_user"] == "root"
    report = outcome.metadata["launcher"]
    assert report["failure_kind"] == "verifier_trusted_launcher" and "not root" in report["detail"]


def _launched(report: dict[str, Any] | None, reward: str = '{"reward": 1}', tail: str = "") -> str:
    """A test.sh that stands for the launcher: it writes ``reward`` as the reward file and ``report`` as its report."""

    script = f"echo '{reward}' > {ROOT}/logs/verifier/reward.json\n"
    if report is not None:
        script += f"echo '{json.dumps(report)}' > {ROOT}/logs/verifier/launcher.json\n"
    return script + tail


_SCORED_ZERO = {"schema": "platoon.launcher/v1", "reward": 0, "failure_kind": None, "reason": "solver_nonreproducible"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("owner", "prefix"), [("generated", "verifier_generated"), ("trusted", "verifier_trusted")])
@pytest.mark.parametrize(
    ("script", "expected"),
    [
        (_launched(_SCORED_ZERO, '{"reward": 0}'), ("scored", 0.0, None, "solver_nonreproducible")),
        (
            _launched(
                {**_SCORED_ZERO, "reward": None, "failure_kind": "verifier_trusted_launcher", "detail": "no root"}
            ),
            ("invalid", None, "verifier_trusted_launcher", "no root"),
        ),
        (
            _launched({}, tail="exit 3\n"),
            ("invalid", None, "{prefix}_unfinished", "the runtime task's launcher did not"),
        ),
        (
            _launched(None),
            ("invalid", None, "{prefix}_unfinished", "the runtime task's launcher did not finish: it left no report"),
        ),
        (
            _launched({**_SCORED_ZERO, "reward": None, "failure_kind": "verifier_subprocess_timeout"}),
            ("invalid", None, "{prefix}_unfinished", "the runtime task's launcher did not finish: its report names"),
        ),
        (
            _launched({**_SCORED_ZERO, "reward": 1}),
            ("invalid", None, "{prefix}_unfinished", "the runtime task's launcher did not finish: its report holds"),
        ),
    ],
    ids=["scored", "trusted-fault", "killed", "no-report", "foreign-kind", "solver-fault-with-a-reward-of-1"],
)
async def test_a_grade_is_the_launchers_report_once_it_finished(
    tmp_path, rootful, store, monkeypatch, owner, prefix, script, expected
):
    """A finished launcher's report is the grade, and goes into the outcome; one that exits nonzero, or leaves no report
    it writes, did not finish: what the reward file says does not stand, and the outcome is named by the grader's
    owner, which the task's settings on the host say, as only the grader, root beside the launcher, or the sandbox could
    have stopped it."""

    monkeypatch.setattr(runtime, "TEST_SH", script)
    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources"), grader_owner=owner)

    outcome = await _graded(service, made.task)

    status, reward, failure_kind, reason = expected
    failure_kind = failure_kind and failure_kind.format(prefix=prefix)
    assert (outcome.status, outcome.reward, outcome.failure_kind) == (status, reward, failure_kind), outcome.metadata
    assert outcome.reason is not None and outcome.reason.startswith(reason), outcome.reason
    if status == "scored":
        assert outcome.metadata["launcher"] == _SCORED_ZERO and outcome.rewards == {"reward": 0.0}


@pytest.mark.asyncio
async def test_a_runtime_grade_that_outlives_the_verifiers_time_is_the_graders(tmp_path, rootful, store, monkeypatch):
    """A grader, root beside the launcher, can stop it (SIGSTOP) until the verifier's time runs out."""

    monkeypatch.setattr(runtime, "TEST_SH", "sleep 30\n")
    template = json.loads(json.dumps(TEMPLATE))
    template["verifier"]["timeout_sec"] = 1
    service = make_service(tmp_path, rootful, store, runtime_tasks={"template": template})
    made = await service.materialize(write_task(tmp_path / "sources", edit=edit("verifier", "timeout_sec", 1)))

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.failure_kind) == ("invalid", "verifier_generated_unfinished")
    assert "the verifier outlived its 1 s" in outcome.reason and outcome.metadata["timeout_s"] == 1


@pytest.mark.asyncio
async def test_a_runtime_grade_whose_output_cannot_be_read_is_the_graders(tmp_path, rootful, store, monkeypatch):
    """The grader, root in the verifier, can take away what grading reads its output with (its shell, its tools)."""

    from platoon.envs.harbor import session as session_module

    async def unreadable(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Could not read the verifier's output in /logs/verifier: no /bin/sh")

    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources"), grader_owner="trusted")
    monkeypatch.setattr(session_module, "collect", unreadable)

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.failure_kind) == ("invalid", "verifier_trusted_unfinished")
    assert "grading could not read the verifier's output: Could not read" in outcome.reason


@pytest.mark.asyncio
async def test_an_entrys_classifiers_never_read_a_runtime_tasks_grade(tmp_path, rootful, store, monkeypatch):
    """The grader writes into the verifier's log (it is root there): a line the entry's classifiers would match must
    not make the launcher's score an invalid measurement, which would let a generated grader censor one."""

    report = {**_SCORED_ZERO, "reason": None}
    line = "subprocess.TimeoutExpired: Command 'grade' timed out after 1 seconds"
    monkeypatch.setattr(runtime, "TEST_SH", _launched(report, '{"reward": 0}', f'echo "{line}"\n'))
    service = make_service(tmp_path, rootful, store, verifier_classifiers="harbor-common")
    made = await service.materialize(write_task(tmp_path / "sources"))

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.reward, outcome.failure_kind) == ("scored", 0.0, None)
    assert line in outcome.metadata["test_output"]


@pytest.mark.asyncio
async def test_the_services_other_tasks_grade_as_they_did_without_runtime_tasks(tmp_path, rootful, store):
    curated = tmp_path / "curated" / "plain"
    (curated / "tests").mkdir(parents=True)
    (curated / "environment").mkdir()
    (curated / "environment" / "data.txt").write_text("not uploaded: the task names no docker_image\n")
    (curated / "task.toml").write_text('artifacts = ["/app/a.txt"]\n[verifier]\nenvironment_mode = "separate"\n')
    (curated / "instruction.md").write_text("Do it.\n")
    (curated / "tests" / "test.sh").write_text(
        f"ls; cat {ROOT}/app/a.txt; echo 1 > {ROOT}/logs/verifier/reward.txt; echo '{{}}' > {ROOT}/logs/verifier/x\n"
    )
    outcomes = []
    for with_runtime in (False, True):
        options: dict[str, Any] = {"runtime_tasks": {"images": IMAGES, "template": TEMPLATE, "store": str(store)}}
        service = HarborEnvironmentService(
            tmp_path / "curated",
            provider=rootful,
            provider_options={"root": str(tmp_path / f"sandboxes-{with_runtime}")},
            default_image="local",
            **(options if with_runtime else {}),
        )
        session = await service.open(await service.get_task("plain"), rollout_id="r")
        try:
            await session.call_tool("bash", {"command": "echo made > a.txt"})
            outcome = await session.grade()
        finally:
            await session.close()
        outcome.metadata.pop("verifier_seconds")
        outcomes.append(outcome)

    assert outcomes[0] == outcomes[1]
    assert outcomes[0].rewards == {"reward": 1.0} and "launcher" not in outcomes[0].metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {"runtime_tasks": {"verifier_provider_options": {"refusal": "no root here"}}},
            "runtime tasks' verifiers run as root, and their sandboxes cannot give it: no root here",
        ),
        (
            {"provider_options": {"network_refusal": "no network namespaces"}},
            "runtime tasks run without a network, which their agents' sandboxes cannot: no network namespaces",
        ),
        (
            {"runtime_tasks": {"verifier_provider_options": {"network_refusal": "not on this site"}}},
            "runtime tasks run without a network, which their verifiers' sandboxes cannot: not on this site",
        ),
        (
            {"runtime_tasks": {"verifier_provider_options": {"layout_refusal": "no fakeroot here"}}},
            "the rootful-process provider would start no runtime task's verifier: no fakeroot here",
        ),
    ],
    ids=["root", "agents-network", "verifiers-network", "verifiers-layout"],
)
async def test_check_tasks_refuses_an_environment_whose_sandboxes_cannot_run_its_runtime_tasks(
    tmp_path, rootful, store, options, message
):
    service = make_service(tmp_path, rootful, store, **options)

    with pytest.raises(EnvSetupError, match=message) as refused:
        await service.check_tasks("train", [])

    assert refused.value.failure_kind == "env_config"
    await make_service(tmp_path, rootful, store).check_tasks("train", [])
