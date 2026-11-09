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
from platoon.envs.harbor.runtime import (
    ForbiddenLiterals,
    MaterializedTask,
    ReplayChecks,
    ReplayVariant,
    RuntimeTasksConfig,
    task_digest,
)
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
    assert config.regrade == 1


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
        ({"regrade": -1}, "runtime_tasks.regrade must be how many times a generated grader's score is graded again"),
        ({"regrade": True}, "runtime_tasks.regrade must be how many times"),
        ({"regrade": 1.5}, "runtime_tasks.regrade must be how many times"),
        ({"regrades": 2}, "runtime_tasks takes images, template, .*, regrade, .* not regrades"),
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
        "launcher_digest": runtime.launcher_digest(),  # of the launcher it installed, this platoon's
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


def _upgrade(monkeypatch) -> str:
    """Make this process's platoon one whose launcher differs from the one it had, as an upgrade's does: the launcher's
    source with one more line, which ``materialize`` installs from then on.  That source."""

    source = Path(launcher.__file__).read_text() + "# another version's\n"
    monkeypatch.setattr(runtime, "_launcher_source", lambda: source)
    return source


def _unrecorded(task_dir: Path) -> None:
    """Leave the launcher's digest out of a stored task's settings, as a platoon that recorded none wrote them."""

    path = task_dir / "tests" / "platoon" / "launch.json"
    settings = json.loads(path.read_text())
    del settings["launcher_digest"]
    path.write_text(json.dumps(settings))


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_by", ["upgraded", "unrecorded"])
async def test_a_runtime_task_set_up_with_another_launcher_is_refused_never_graded_by_it(
    tmp_path, rootful, store, monkeypatch, stored_by
):
    """A stored task keeps the launcher it was given, which would grade it: an environment of the same
    ``runtime_tasks`` runs only a task its own platoon's launcher was installed in, and refuses one set up by another,
    by its directory and by its digest, before anything starts: a task the platoon before an upgrade stored
    (``upgraded``), and one whose settings record no launcher (``unrecorded``)."""

    made = await make_service(tmp_path, rootful, store).materialize(write_task(tmp_path / "sources"))
    assert made.ok, made.problems
    if stored_by == "upgraded":
        _upgrade(monkeypatch)
    else:
        _unrecorded(Path(made.task.misc["harbor"]["task_dir"]))
    service = make_service(tmp_path, rootful, store)

    for refuse in (service.open(made.task, rollout_id="r"), service.get_task(made.digest)):
        with pytest.raises(EnvSetupError, match="set up with another launcher than this platoon's") as refused:
            await refuse
        assert refused.value.failure_kind == "env_config"
        assert "materialize it again with this platoon" in str(refused.value)
    with pytest.raises(ValueError, match="set up with another launcher than this platoon's"):
        service.agent_timeout_s(made.digest)
    assert service.provider.started == []


@pytest.mark.asyncio
async def test_a_task_refused_for_its_launcher_runs_once_materialized_again_under_a_new_digest(
    tmp_path, rootful, store, monkeypatch
):
    """Materialized again with this platoon, the same directory gets this platoon's launcher, and with it another
    digest, whose task opens and is graded by that launcher; the task of the old digest stays refused."""

    source = write_task(tmp_path / "sources")
    old = await make_service(tmp_path, rootful, store).materialize(source)
    upgraded = _upgrade(monkeypatch)
    service = make_service(tmp_path, rootful, store)

    again = await service.materialize(source)

    assert again.ok and again.digest != old.digest
    assert (store / again.digest / "tests" / "platoon" / "launcher.py").read_text() == upgraded
    outcome = await _graded(service, again.task)  # as this user, the launcher refuses to grade: it ran
    assert (outcome.status, outcome.failure_kind) == ("invalid", "verifier_trusted_launcher")
    assert "not root" in outcome.metadata["launcher"]["detail"]
    with pytest.raises(EnvSetupError, match="set up with another launcher than this platoon's"):
        await service.open(old.task, rollout_id="r")


@pytest.mark.asyncio
async def test_the_same_platoon_and_runtime_tasks_reuse_a_stored_task_wherever_platoon_is_installed(
    tmp_path, rootful, store, monkeypatch
):
    """Another process of the same platoon, installed elsewhere, with the same ``runtime_tasks`` (a resumed run's, say)
    gives the same directory the same digest and runs the stored task: the launcher's digest is its files', never where
    they are."""

    made = await make_service(tmp_path, rootful, store).materialize(write_task(tmp_path / "sources"))
    elsewhere = tmp_path / "elsewhere" / "launcher.py"
    elsewhere.parent.mkdir()
    shutil.copyfile(launcher.__file__, elsewhere)
    monkeypatch.setattr(launcher, "__file__", str(elsewhere))
    runtime._launcher_source.cache_clear()  # read again, as another process reads it
    service = make_service(tmp_path, rootful, store)

    again = await service.materialize(write_task(tmp_path / "written-again"))
    task = await service.get_task(made.digest)
    session = await service.open(task, rollout_id="r")
    await session.close()

    assert again.digest == made.digest and task.misc["harbor"] == made.task.misc["harbor"]
    assert sorted(path.name for path in store.iterdir()) == [".staging", made.digest]


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


# -------------------------------------------------------------------------------------------------------- regrades

_PASSED = {"schema": "platoon.launcher/v1", "reward": 1, "failure_kind": None, "reason": None, "detail": "passed"}
_FAILED = {**_PASSED, "reward": 0, "detail": "failed"}


def _in_turn(tmp_path: Path, *grades: str) -> str:
    """A test.sh that runs the n-th of ``grades`` (bodies as :func:`_launched` writes them) on its n-th run, and the
    last from then on.  Each run notes the root of its sandbox in ``tmp_path/verifiers``, and in
    ``tmp_path/overlapped`` the root of an earlier run's sandbox that is still there (not stopped)."""

    count, ran, overlapped = tmp_path / "grades", tmp_path / "verifiers", tmp_path / "overlapped"
    arms = "".join(f"{number})\n{body};;\n" for number, body in enumerate(grades[:-1], start=1))
    return (
        f'n=$(( $(cat "{count}" 2>/dev/null || echo 0) + 1 )); echo "$n" > "{count}"\n'
        f'if [ -f "{ran}" ]; then\n'
        f'  while read -r earlier; do [ -e "$earlier" ] && echo "$earlier" >> "{overlapped}"; done < "{ran}"\n'
        "fi\n"
        f'echo "{ROOT}" >> "{ran}"\n'
        f"case $n in\n{arms}*)\n{grades[-1]};;\nesac\n"
    )


def _verifiers(tmp_path: Path) -> list[str]:
    """The roots of the sandboxes a task's tests ran in, in order: each a sandbox of its own, which stopped before the
    next one's tests ran."""

    ran = (tmp_path / "verifiers").read_text().split() if (tmp_path / "verifiers").exists() else []
    assert len(set(ran)) == len(ran), ran
    assert not (tmp_path / "overlapped").exists(), "a regrade ran while an earlier verifier was still there"
    return ran


def _verifier_starts(service: HarborEnvironmentService) -> int:
    return sum(spec.labels.get("platoon.role") == "verifier" for spec in service.provider.started)


def test_a_config_takes_any_number_of_regrades_from_0():
    for regrade in (0, 1, 3):
        config = RuntimeTasksConfig.coerce({"images": IMAGES, "template": TEMPLATE, "regrade": regrade})
        assert config.regrade == regrade


@pytest.mark.asyncio
async def test_a_configs_fingerprint_leaves_out_how_often_it_regrades(tmp_path, rootful, store):
    """How often a generated grader's score is graded again is no part of what a task was checked against: an entry
    that regrades more, or not at all, runs the tasks another materialized into a shared store, as a resumed run that
    changed it does."""

    made = await make_service(tmp_path, rootful, store).materialize(write_task(tmp_path / "sources"))

    for regrade in (0, 3):
        other = make_service(tmp_path, rootful, store, runtime_tasks={"regrade": regrade})
        assert other.harbor_task(made.task).name == "generated"
        assert other.runtime_regrades(other.harbor_task(made.task)) == regrade
    fingerprints = {
        runtime.config_fingerprint(
            RuntimeTasksConfig.coerce({"images": IMAGES, "template": TEMPLATE, "regrade": regrade}),
            reward_key="reward",
            timeout_multiplier=1.0,
        )
        for regrade in (0, 1, 3)
    }
    assert len(fingerprints) == 1


@pytest.mark.asyncio
async def test_a_generated_graders_score_is_graded_again_in_a_fresh_verifier_and_stands_when_it_repeats(
    tmp_path, rootful, store, monkeypatch
):
    monkeypatch.setattr(runtime, "TEST_SH", _in_turn(tmp_path, _launched(_PASSED)))
    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources"))

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.reward, outcome.failure_kind, outcome.reason) == ("scored", 1.0, None, None)
    assert len(_verifiers(tmp_path)) == _verifier_starts(service) == 2
    assert outcome.metadata["launcher"] == _PASSED and outcome.metadata["artifacts"] == ["/app/a.json", "/app/solve.py"]
    (regrade,) = outcome.metadata["regrades"]
    assert {key: regrade[key] for key in ("status", "reward", "failure_kind", "reason")} == {
        "status": "scored",
        "reward": 1.0,
        "failure_kind": None,
        "reason": None,
    }
    assert regrade["metadata"]["launcher"] == _PASSED and regrade["metadata"]["verifier_exit_code"] == 0
    assert "artifacts" not in regrade["metadata"]  # the first grade's: every grade gets the same ones
    json.dumps(outcome.metadata)  # recorded as it is


@pytest.mark.asyncio
async def test_a_regrade_gets_the_artifacts_as_they_were_collected(tmp_path, rootful, store, monkeypatch):
    """Nothing a grader does in its verifier, root there, reaches the next: the regrade's verifier gets the artifacts
    as they were collected from the agents' sandbox, though the first grade's grader rewrote them."""

    seen = tmp_path / "seen"
    tamper = f'cat {ROOT}/app/a.json >> "{seen}"; echo tampered > {ROOT}/app/a.json\n'
    monkeypatch.setattr(runtime, "TEST_SH", _in_turn(tmp_path, tamper + _launched(_PASSED)))
    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources"))

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.reward, len(outcome.metadata["regrades"])) == ("scored", 1.0, 1)
    assert seen.read_text().split() == ["{}", "{}"]  # what the agents left, both times


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "then"), [(_PASSED, _FAILED), (_FAILED, _PASSED)], ids=["1-then-0", "0-then-1"])
async def test_a_regrade_that_scores_otherwise_makes_the_grade_flaky(
    tmp_path, rootful, store, monkeypatch, first, then
):
    """A grader that gives the same artifacts another score in another fresh verifier measures nothing: neither score
    stands, and the grade is the generated grader's non-score (``verifier_flaky``)."""

    monkeypatch.setattr(runtime, "TEST_SH", _in_turn(tmp_path, _launched(first), _launched(then)))
    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources"))

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.reward, outcome.failure_kind) == ("invalid", None, "verifier_flaky")
    assert outcome.rewards == {}
    assert f"scored the same artifacts {first['reward']}, then {then['reward']} in a fresh verifier" in outcome.reason
    assert len(_verifiers(tmp_path)) == 2
    assert outcome.metadata["launcher"] == first  # the first grade's report; the regrade's is its own
    (regrade,) = outcome.metadata["regrades"]
    assert (regrade["status"], regrade["reward"], regrade["metadata"]["launcher"]) == ("scored", then["reward"], then)


@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "then"), [(_PASSED, _FAILED), (_FAILED, _PASSED)], ids=["1-then-0", "0-then-1"])
@pytest.mark.parametrize(
    ("replayed", "expected"),
    [
        (("a" * 64, "a" * 64), ("invalid", None, "verifier_flaky")),
        (("a" * 64, "b" * 64), ("scored", 0.0, None)),
    ],
    ids=["the-same-outputs", "other-outputs"],
)
async def test_a_disagreement_over_outputs_the_replays_wrote_otherwise_is_the_solvers_0(
    tmp_path, rootful, store, monkeypatch, first, then, replayed, expected
):
    """Each grade replays the submission, and its launcher's report holds the SHA-256 of the outputs the replay wrote.
    A grader that scores the same outputs otherwise is flaky.  Scores of outputs the replays wrote otherwise judged a
    submission that computes something else each time it runs, which does not reproduce: the solver's 0, whichever
    score came first, and never the grader's fault."""

    reports = [{**report, "replayed_sha256": digest} for report, digest in zip((first, then), replayed, strict=True)]
    monkeypatch.setattr(runtime, "TEST_SH", _in_turn(tmp_path, *map(_launched, reports)))
    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources"))

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.reward, outcome.failure_kind) == expected
    if outcome.trainable:
        assert outcome.rewards == {"reward": 0.0} and outcome.reason == "solver_nonreproducible"
    else:
        assert outcome.reason.startswith(
            f"the task's generated grader scored the same artifacts and replayed outputs {first['reward']}, then "
            f"{then['reward']} in a fresh verifier (regrade 1)"
        )
    assert outcome.metadata["launcher"] == reports[0] and "first_grade" not in outcome.metadata
    (regrade,) = outcome.metadata["regrades"]
    assert (regrade["reward"], regrade["metadata"]["launcher"]) == (then["reward"], reports[1])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("grades", "expected", "runs"),
    [
        ((_PASSED,), ("scored", 1.0, None), 4),
        ((_PASSED, _PASSED, _FAILED, _PASSED), ("invalid", None, "verifier_flaky"), 3),
    ],
    ids=["all-agree", "the-second-regrade-disagrees"],
)
async def test_regrades_run_one_after_the_other_until_one_does_not_agree(
    tmp_path, rootful, store, monkeypatch, grades, expected, runs
):
    monkeypatch.setattr(runtime, "TEST_SH", _in_turn(tmp_path, *map(_launched, grades)))
    service = make_service(tmp_path, rootful, store, runtime_tasks={"regrade": 3})
    made = await service.materialize(write_task(tmp_path / "sources"))

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.reward, outcome.failure_kind) == expected
    assert len(_verifiers(tmp_path)) == _verifier_starts(service) == runs
    assert len(outcome.metadata["regrades"]) == runs - 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("regrade", "expected"),
    [
        (_launched(_SCORED_ZERO, '{"reward": 0}'), ("scored", 0.0, None, "solver_nonreproducible")),
        (
            _launched({**_SCORED_ZERO, "reward": None, "failure_kind": "verifier_generated_timeout", "detail": "slow"}),
            ("invalid", None, "verifier_generated_timeout", "slow"),
        ),
        (
            _launched({}, tail="exit 3\n"),
            ("invalid", None, "verifier_generated_unfinished", "the runtime task's launcher"),
        ),
        (
            _launched(
                {**_SCORED_ZERO, "reward": None, "failure_kind": "verifier_trusted_launcher", "detail": "no root"}
            ),
            ("invalid", None, "verifier_trusted_launcher", "no root"),
        ),
    ],
    ids=["solvers-zero", "graders-timeout", "unfinished", "trusted-fault"],
)
async def test_a_regrade_that_gives_no_score_of_the_graders_makes_the_grade_its_own(
    tmp_path, rootful, store, monkeypatch, regrade, expected
):
    """A score stands only once every regrade gave it again.  A regrade that gives none is attributed as any grade:
    a replay that does not reproduce this time is the solver's 0, a grader's non-score its own kind (which a program
    tells from the sandbox's faults as it does a first grade's), and a fault on the trusted side the trusted side's."""

    monkeypatch.setattr(runtime, "TEST_SH", _in_turn(tmp_path, _launched(_PASSED), regrade))
    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources"))

    outcome = await _graded(service, made.task)

    status, reward, failure_kind, reason = expected
    assert (outcome.status, outcome.reward, outcome.failure_kind) == (status, reward, failure_kind), outcome.reason
    assert outcome.reason.startswith(reason), outcome.reason
    (entry,) = outcome.metadata["regrades"]
    assert (entry["status"], entry["failure_kind"], entry["reason"]) == (status, failure_kind, outcome.reason)
    # The outcome is the regrade's, and so is its metadata (its launcher report, its verifier's exit status, the tail of
    # its output); the first grade's is beside it, and the artifacts, which every grade gets alike, are listed once.
    assert {key: outcome.metadata[key] for key in entry["metadata"]} == entry["metadata"]
    assert outcome.metadata["launcher"] != _PASSED and "verifier_exit_code" in entry["metadata"]
    first = outcome.metadata["first_grade"]
    assert (first["status"], first["reward"], first["metadata"]["launcher"]) == ("scored", 1.0, _PASSED)
    assert outcome.metadata["artifacts"] == ["/app/a.json", "/app/solve.py"] and "artifacts" not in first["metadata"]


@pytest.mark.asyncio
async def test_a_regrade_whose_verifier_cannot_start_makes_the_grade_an_infrastructure_fault(
    tmp_path, rootful, store, monkeypatch
):
    monkeypatch.setattr(runtime, "TEST_SH", _in_turn(tmp_path, _launched(_PASSED)))
    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources"))
    opened = service.open_verifier
    calls = []

    async def second_fails(harbor_task: HarborTask, *, rollout_id: str) -> Any:
        calls.append(rollout_id)
        if len(calls) == 2:
            raise EnvSetupError("env_setup", "the node ran out of sandboxes")
        return await opened(harbor_task, rollout_id=rollout_id)

    monkeypatch.setattr(service, "open_verifier", second_fails)

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.failure_kind) == ("error", "verifier_start")
    assert "ran out of sandboxes" in outcome.reason and len(calls) == 2 and len(_verifiers(tmp_path)) == 1
    assert outcome.metadata["regrades"][0]["metadata"] == {"setup_failure_kind": "env_setup"}
    assert outcome.metadata["setup_failure_kind"] == "env_setup" and "launcher" not in outcome.metadata
    assert outcome.metadata["first_grade"]["metadata"]["launcher"] == _PASSED


@pytest.mark.asyncio
async def test_a_regrade_that_outlives_the_verifiers_time_gives_the_grade_its_time_limit(
    tmp_path, rootful, store, monkeypatch
):
    """The regrade's grader stopped the launcher until the verifier's time ran out: the grade is that regrade's, with
    the time limit it outlived."""

    monkeypatch.setattr(runtime, "TEST_SH", _in_turn(tmp_path, _launched(_PASSED), "sleep 30"))
    template = json.loads(json.dumps(TEMPLATE))
    template["verifier"]["timeout_sec"] = 2
    service = make_service(tmp_path, rootful, store, runtime_tasks={"template": template})
    made = await service.materialize(write_task(tmp_path / "sources", edit=edit("verifier", "timeout_sec", 2)))

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.failure_kind) == ("invalid", "verifier_generated_unfinished")
    assert "the verifier outlived its 2 s" in outcome.reason and outcome.metadata["timeout_s"] == 2
    assert "timeout_s" not in outcome.metadata["first_grade"]["metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("owner", "options", "first", "expected"),
    [
        ("trusted", {}, _launched(_PASSED), ("scored", 1.0, None)),
        ("generated", {"regrade": 0}, _launched(_PASSED), ("scored", 1.0, None)),
        ("generated", {}, _launched(_SCORED_ZERO, '{"reward": 0}'), ("scored", 0.0, None)),
        (
            "generated",
            {},
            _launched({**_SCORED_ZERO, "reward": None, "failure_kind": "verifier_generated_exit"}),
            ("invalid", None, "verifier_generated_exit"),
        ),
        (
            "generated",
            {},
            _launched({**_SCORED_ZERO, "reward": None, "failure_kind": "verifier_trusted_launcher"}),
            ("invalid", None, "verifier_trusted_launcher"),
        ),
    ],
    ids=["trusted-grader", "regrade-0", "solvers-zero", "graders-non-score", "trusted-fault"],
)
async def test_only_a_generated_graders_scores_are_regraded(
    tmp_path, rootful, store, monkeypatch, owner, options, first, expected
):
    """A trusted grader answers for its own grades, and there is nothing to regrade in a grade that is no score of the
    grader's: the solver's 0, which the launcher gives before the grader runs, a non-score, a fault on the trusted
    side."""

    monkeypatch.setattr(runtime, "TEST_SH", _in_turn(tmp_path, first, _launched(_FAILED)))
    service = make_service(tmp_path, rootful, store, runtime_tasks=options)
    made = await service.materialize(write_task(tmp_path / "sources"), grader_owner=owner)

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.reward, outcome.failure_kind) == expected
    assert len(_verifiers(tmp_path)) == _verifier_starts(service) == 1 and "regrades" not in outcome.metadata


@pytest.mark.asyncio
async def test_curated_tasks_are_never_regraded(tmp_path, rootful, store):
    """Regrading a curated task would change its grading: a dataset's own tests, of the same outcome as before."""

    curated = tmp_path / "curated" / "plain"
    (curated / "tests").mkdir(parents=True)
    (curated / "task.toml").write_text('artifacts = ["/app/a.txt"]\n[verifier]\nenvironment_mode = "separate"\n')
    (curated / "instruction.md").write_text("Do it.\n")
    (curated / "tests" / "test.sh").write_text(f"echo 1 > {ROOT}/logs/verifier/reward.txt\n")
    service = make_service(tmp_path, rootful, store, default_image="local", runtime_tasks={"regrade": 3})
    task = await service.get_task("plain")

    session = await service.open(task, rollout_id="r")
    try:
        await session.call_tool("bash", {"command": "echo made > a.txt"})
        outcome = await session.grade()
    finally:
        await session.close()

    assert outcome.rewards == {"reward": 1.0} and "regrades" not in outcome.metadata
    assert _verifier_starts(service) == 1 and service.runtime_regrades(service.harbor_task(task)) == 0


@pytest.mark.asyncio
async def test_grading_a_session_again_regrades_the_artifacts_it_kept(tmp_path, rootful, store, monkeypatch):
    """Once the separate verifier released ``main``, a grade again grades the kept artifacts, regrades included."""

    grades = (_PASSED, _PASSED, _PASSED, _FAILED)
    monkeypatch.setattr(runtime, "TEST_SH", _in_turn(tmp_path, *map(_launched, grades)))
    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources"))

    session = await service.open(made.task, rollout_id="r")
    try:
        await session.call_tool("bash", {"command": "echo 'x' > solve.py; echo '{}' > a.json"})
        first, second = await session.grade(), await session.grade()
    finally:
        await session.close()

    assert (first.status, first.reward) == ("scored", 1.0)
    assert (second.status, second.failure_kind) == ("invalid", "verifier_flaky")
    assert len(_verifiers(tmp_path)) == 4


@pytest.mark.asyncio
async def test_a_rollouts_deadline_counts_each_regrade_of_a_generated_grader(tmp_path, rootful, store):
    """Each regrade starts another verifier and runs its tests for up to their time limit (120 s here), with the
    grading commands around them: a deadline on the whole rollout leaves room for them, and only for them."""

    source = write_task(tmp_path / "sources")
    once = make_service(tmp_path, rootful, store, runtime_tasks={"regrade": 0})
    trusted = (await once.materialize(source, grader_owner="trusted")).digest
    generated = (await once.materialize(source)).digest
    regrade = 120 + once.sandbox_start_budget_s + 120  # the tests' time limit, a start budget, the grading commands

    assert once.rollout_overhead_s(generated) == once.rollout_overhead_s(trusted)
    for count in (1, 3):
        service = make_service(tmp_path, rootful, store, runtime_tasks={"regrade": count})
        assert service.rollout_overhead_s(trusted) == once.rollout_overhead_s(trusted)
        assert service.rollout_overhead_s(generated) == once.rollout_overhead_s(generated) + count * regrade


# --------------------------------------------------------------------------------------------------- replay checks

CHECKS: dict[str, Any] = {
    "variants": [
        {
            "name": "fifty",
            "inputs": "tests/gen/variants/fifty/inputs",
            "expected": "tests/gen/variants/fifty/expected.json",
            "tolerance": {"default": 0.5, "keys": {"a.json#/answer": 1}},
            "meta": {"covers": [[0, 1]]},
        }
    ],
    "forbidden_literals": {"path": "tests/gen/forbidden.json", "sources": ["solve.py"]},
}
# The data of CHECKS, in the task's tests/gen: the variant's data.txt of 50, which gives an answer of 100, and the
# answer as a forbidden literal.
CHECK_FILES = {
    "tests/gen/variants/fifty/inputs/data.txt": "50\n",
    "tests/gen/variants/fifty/expected.json": json.dumps({"a.json#/answer": 100}),
    "tests/gen/forbidden.json": json.dumps([{"value": 42, "tolerance": 0.5}]),
}


def _variant(**changes: Any) -> dict[str, Any]:
    return {**CHECKS["variants"][0], **changes}


def _untolerant() -> dict[str, Any]:
    return {key: value for key, value in _variant().items() if key != "tolerance"}


def _nested(depth: int) -> Any:
    """A list ``depth`` levels deep."""

    value: Any = 0
    for _ in range(depth):
        value = [value]
    return value


def test_replay_checks_are_built_from_mappings_and_record_what_the_launcher_runs():
    checks = ReplayChecks.coerce(CHECKS)

    assert isinstance(checks.variants[0], ReplayVariant) and isinstance(checks.forbidden_literals, ForbiddenLiterals)
    assert checks.settings() == {
        "variants": [CHECKS["variants"][0]],
        "forbidden_literals": {**CHECKS["forbidden_literals"], "scan": "python-numeric-literals/v1"},
    }
    assert ReplayChecks.coerce(checks) is checks
    one = ReplayVariant(name="v", inputs="tests/gen/v/inputs", expected="tests/gen/v/expected.json", tolerance=0.25)
    assert (one.tolerance, one.meta) == ({"default": 0.25, "keys": {}}, None)
    assert ReplayChecks.coerce({"variants": [_variant(tolerance={"default": 0.25})]}).variants[0].tolerance == {
        "default": 0.25,
        "keys": {},
    }
    assert ReplayChecks.coerce({"variants": [_variant(tolerance=0)]}).variants[0].tolerance == {
        "default": 0,
        "keys": {},
    }


@pytest.mark.parametrize(
    ("checks", "message"),
    [
        ({"variants": [_variant(name="fifty two")]}, "a replay variant's name must be letters"),
        ({"variants": [_variant(inputs="tests/platoon/inputs")]}, "inputs must be a path of the task below tests/gen/"),
        ({"variants": [_variant(inputs="environment/data")]}, "inputs must be a path of the task below tests/gen/"),
        ({"variants": [_variant(inputs="tests/gen/../gen/x")]}, "inputs must be a path of the task below"),
        ({"variants": [_variant(expected="/tests/gen/e.json")]}, "expected must be a path of the task below"),
        (
            {"variants": [_variant(expected="tests/gen/variants/fifty/inputs/e.json")]},
            "its expected outputs lie in its inputs, which its replay would read",
        ),
        ({"variants": [_variant(), _variant()]}, "replay_checks name two variants alike"),
        (
            {"variants": [_variant(), _variant(name="nested", inputs="tests/gen/variants/fifty/inputs/more")]},
            "replay variant nested's inputs lie in variant fifty's",
        ),
        (
            {
                "variants": [
                    _variant(),
                    _variant(name="other", inputs="tests/gen/o", expected="tests/gen/variants/fifty/inputs/o.json"),
                ]
            },
            "replay variant other's expected outputs lie in variant fifty's inputs",
        ),
        (
            {
                "variants": [_variant()],
                "forbidden_literals": {"path": "tests/gen/variants/fifty/inputs/f.json", "sources": ["s.py"]},
            },
            "the forbidden literals lie in replay variant fifty's inputs",
        ),
        ({"variants": [_variant(tolerance=-1)]}, "tolerance must be a number of 0 or more, or a mapping"),
        ({"variants": [_variant(tolerance=True)]}, "tolerance must be a number of 0 or more"),
        ({"variants": [_variant(tolerance={"default": 0, "margin": 1})]}, "or a mapping of a default and keys"),
        ({"variants": [_variant(tolerance={"keys": {"a": 1}})]}, "or a mapping of a default and keys"),  # no default
        ({"variants": [_untolerant()]}, "missing 1 required positional argument: 'tolerance'"),
        ({"variants": [_variant(tolerance={"default": float("nan")})]}, "its tolerance's default must be a finite"),
        ({"variants": [_variant(tolerance={"default": 0, "keys": {"a": -1}})]}, "its tolerance's keys must map output"),
        ({"variants": [_variant(meta={1, 2})]}, "meta must be JSON"),
        ({"variants": [_variant(meta="x" * 9000)]}, "more than a variant records (8192)"),
        ({"variants": [_variant(meta=_nested(17))]}, "meta nests more than 16 levels deep"),
        ({"variants": [_variant(meta=_nested(1200))]}, "meta nests more than 16 levels deep"),  # within 8 KiB
        # Deeper than some Pythons encode, and some do: counted before it is encoded, it is refused alike on each.
        ({"variants": [_variant(meta=_nested(100_000))]}, "meta nests more than 16 levels deep"),
        (  # 8 KiB of meta each, which launch.json holds one number a line: less than its cap compact, more indented
            {"variants": [_variant(name=f"v{n}", inputs=f"tests/gen/{n}", meta=[0] * 2700) for n in range(16)]},
            "bytes of settings, more than a task's launcher settings take (262144)",
        ),
        ({"variants": [_variant(unknown=1)]}, "a replay variant: "),
        ({"variants": ["fifty"]}, "a replay variant must be a ReplayVariant or a mapping"),
        ({"variants": "fifty"}, "replay_checks.variants must be a list"),
        ({"variants": [_variant(name=f"v{n}", inputs=f"tests/gen/{n}") for n in range(33)]}, "33 variants, more than"),
        ({"forbidden_literals": {"path": "tests/gen/f.json", "sources": "solve.py"}}, "sources must list 1 to 64"),
        ({"forbidden_literals": {"path": "tests/gen/f.json", "sources": []}}, "sources must list 1 to 64"),
        ({"forbidden_literals": {"path": "tests/gen/f.json", "sources": ["../solve.py"]}}, "sources must list"),
        ({"forbidden_literals": {"path": "tests/gen/f.json", "sources": ["a.py", "a.py"]}}, "sources must list"),
        (
            {"forbidden_literals": {"path": "tests/gen/f.json", "sources": ["analysis.R"]}},
            "sources must be Python files (.py), which python-numeric-literals/v1 reads",
        ),
        ({"forbidden_literals": {"path": "tests/f.json", "sources": ["a.py"]}}, "path must be a JSON file of the task"),
        (
            {"forbidden_literals": {"path": "tests/gen/f.json", "sources": ["a.py"], "scan": "regex/v1"}},
            "scan must be python-numeric-literals/v1",
        ),
        ({"variants": [], "tolerance": 1}, "replay_checks take variants and forbidden_literals, not tolerance"),
    ],
)
def test_replay_checks_refuse_what_no_launcher_could_run(checks: dict[str, Any], message: str):
    with pytest.raises(ValueError, match=re.escape(message)):
        ReplayChecks.coerce(checks)


@pytest.mark.asyncio
async def test_materialize_records_the_replay_checks_in_the_tasks_launcher_settings(tmp_path, rootful, store):
    """The checks are the caller's word, kept in the task's settings, so that the same directory with other checks is
    another task; checks that check nothing are none."""

    service = make_service(tmp_path, rootful, store)
    source = write_task(tmp_path / "sources", files=CHECK_FILES)

    checked, again = (
        await service.materialize(source, replay_checks=CHECKS),
        await service.materialize(source, replay_checks=ReplayChecks.coerce(CHECKS)),
    )
    plain, empty = await service.materialize(source), await service.materialize(source, replay_checks={})

    assert checked.ok and checked.digest == again.digest != plain.digest == empty.digest
    launch = json.loads((store / checked.digest / "tests" / "platoon" / "launch.json").read_text())
    assert launch["replay_checks"] == ReplayChecks.coerce(CHECKS).settings()
    assert "replay_checks" not in json.loads((store / plain.digest / "tests" / "platoon" / "launch.json").read_text())
    assert (
        store / checked.digest / "tests" / "gen" / "variants" / "fifty" / "inputs" / "data.txt"
    ).read_text() == "50\n"
    with pytest.raises(ValueError, match="a replay variant's name must be"):
        await service.materialize(source, replay_checks={"variants": [_variant(name="")]})


@pytest.mark.asyncio
async def test_materialize_records_a_time_for_each_replay_that_leaves_the_grader_some(tmp_path, rootful, store):
    """``replay_timeout_s`` is the caller's word too: the task's settings record it, so that the same directory with
    another is another task.  The template's verifier gives the launcher 108 s of its 120: the replay and one variant
    may each take up to 53 s, and the grader the rest, but not 54 s each."""

    service = make_service(tmp_path, rootful, store)
    source = write_task(tmp_path / "sources", files=CHECK_FILES)
    one = {"variants": CHECKS["variants"], "forbidden_literals": None}

    timed = await service.materialize(source, replay_checks=one, replay_timeout_s=53)
    plain = await service.materialize(source, replay_checks=one)

    assert timed.ok and timed.digest != plain.digest
    settings = {
        made.digest: json.loads((store / made.digest / "tests" / "platoon" / "launch.json").read_text())
        for made in (timed, plain)
    }
    assert settings[timed.digest]["replay_timeout_s"] == 53 and "replay_timeout_s" not in settings[plain.digest]
    with pytest.raises(ValueError, match=re.escape("2 replays of 54 s leave the grader none of the 108 s")):
        await service.materialize(source, replay_checks=one, replay_timeout_s=54)
    wrongs: list[Any] = [0, -1, float("nan"), True, "300"]
    for wrong in wrongs:
        with pytest.raises(ValueError, match="replay_timeout_s must be a finite number of seconds above 0"):
            await service.materialize(source, replay_timeout_s=wrong)
    untimed = write_task(tmp_path / "plain", files={"tests/gen/grader.json": json.dumps({"command": ["python3", "g"]})})
    made = await service.materialize(untimed, replay_timeout_s=10)
    assert made.problems == ("replay_timeout_s needs a replay, and tests/gen/grader.json gives none",)
    assert sorted(path.name for path in store.iterdir()) == sorted([".staging", str(timed.digest), str(plain.digest)])


@pytest.mark.asyncio
async def test_materialize_refuses_launcher_settings_larger_than_the_launcher_reads(
    tmp_path, rootful, store, monkeypatch
):
    """Settings the launcher would not read whole are refused, never stored: a task without checks fits, the same
    task with them would not."""

    service = make_service(tmp_path, rootful, store)
    source = write_task(tmp_path / "sources", files=CHECK_FILES)
    plain = await service.materialize(source)
    size = len((store / plain.digest / "tests" / "platoon" / "launch.json").read_bytes())
    monkeypatch.setattr(runtime.launcher, "MAX_CONFIG_BYTES", size + 10)

    made = await service.materialize(source, replay_checks=CHECKS)

    assert len(made.problems) == 1, made.problems
    assert made.problems[0].startswith("the task's launcher settings would hold ")
    assert f"more than its launcher reads ({size + 10})" in made.problems[0]
    assert sorted(path.name for path in store.iterdir()) == [".staging", plain.digest]


def _empty_inputs(task_dir: Path) -> None:
    (task_dir / "tests" / "gen" / "variants" / "fifty" / "inputs" / "data.txt").unlink()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("files", "setup", "problem"),
    [
        (
            {"tests/gen/grader.json": json.dumps({"command": ["python3", "grade.py"]})},
            None,
            "replay_checks need a replay, and tests/gen/grader.json gives none",
        ),
        (
            {"tests/gen/variants/fifty/inputs/data.txt": None},
            None,
            "variant fifty's inputs, tests/gen/variants/fifty/inputs, cannot be read:",
        ),
        ({}, _empty_inputs, "variant fifty's inputs, tests/gen/variants/fifty/inputs, hold no file to lay over"),
        (
            {"tests/gen/variants/fifty/inputs/solve.py": "print(1)\n"},
            None,
            "variant fifty's inputs cannot be laid over the working inputs: solve.py would change solve.py, which the "
            "submission holds or the replay writes",
        ),
        ({"tests/gen/variants/fifty/inputs/a.json": "{}"}, None, "a.json would change a.json"),
        (
            {"tests/gen/variants/fifty/inputs/tools": "a file\n"},
            None,
            "tools is a file of the variant's and a directory of environment/'s",
        ),
        (
            {"tests/gen/variants/fifty/inputs/data.txt": None, "tests/gen/variants/fifty/inputs/data.txt/more": "1\n"},
            None,
            "data.txt is a directory of the variant's and a file of environment/'s",
        ),
        (
            {"tests/gen/variants/fifty/expected.json": "{"},
            None,
            "variant fifty's expected outputs, tests/gen/variants/fifty/expected.json, cannot be read as JSON",
        ),
        (
            {"tests/gen/variants/fifty/expected.json": json.dumps({"answer.json#/answer": 100})},
            None,
            'hold the key "answer.json#/answer", which names no output of the replay',
        ),
        (
            {"tests/gen/variants/fifty/expected.json": json.dumps({"a.json#/other": 100})},
            None,
            'lack "a.json#/answer", which the variant\'s tolerance names',
        ),
        (
            {"tests/gen/forbidden.json": None},
            None,
            "the forbidden literals, tests/gen/forbidden.json, cannot be read as JSON",
        ),
        ({"tests/gen/forbidden.json": json.dumps({"value": 42})}, None, "are no list of at most 10000 objects"),
    ],
    ids=[
        "no-replay",
        "no-inputs",
        "empty-inputs",
        "inputs-replace-an-artifact",
        "inputs-replace-an-output",
        "inputs-a-file-for-a-directory",
        "inputs-a-directory-for-a-file",
        "expected-not-json",
        "expected-no-output",
        "tolerance-of-a-key-not-expected",
        "no-forbidden-literals",
        "forbidden-literals-no-list",
    ],
)
async def test_materialize_refuses_replay_checks_whose_data_the_launcher_could_not_use(
    tmp_path, rootful, store, files, setup, problem
):
    service = make_service(tmp_path, rootful, store)
    source = write_task(tmp_path / "sources", files={**CHECK_FILES, **files})
    if setup is not None:
        setup(source)
    before = sorted(str(path.relative_to(source)) for path in source.rglob("*"))

    made = await service.materialize(source, replay_checks=CHECKS)

    assert not made.ok and any(problem in found for found in made.problems), made.problems
    assert not any(".staging" in found or str(store) in found for found in made.problems)
    assert [path.name for path in store.iterdir()] == [".staging"] and list((store / ".staging").iterdir()) == []
    assert sorted(str(path.relative_to(source)) for path in source.rglob("*")) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sources", "problem"),
    [
        (["notes.py"], "the forbidden literals' source notes.py is no artifact of the task, nor below one"),
        (["made.py"], "the forbidden literals' source made.py is what the replay writes again"),
    ],
)
async def test_materialize_refuses_a_source_to_scan_that_the_submission_does_not_hold(
    tmp_path, rootful, store, sources, problem
):
    """The scan reads the submission's own code, never what the replay writes again (here made.py, an artifact too)."""

    service = make_service(tmp_path, rootful, store)
    checks = {**CHECKS, "forbidden_literals": {**CHECKS["forbidden_literals"], "sources": sources}}
    manifest = {"command": ["python3", "grade.py"], "replay": {"command": ["python3", "solve.py"], "outputs": []}}
    manifest["replay"]["outputs"] = ["a.json", "made.py"]
    source = write_task(
        tmp_path / "sources",
        edit=lambda config: config["artifacts"].append("made.py"),
        files={**CHECK_FILES, "tests/gen/grader.json": json.dumps(manifest)},
    )

    made = await service.materialize(source, replay_checks=checks)

    assert made.problems == (problem + (": the scan reads what the submission holds" if "artifact" in problem else ""),)


@pytest.mark.asyncio
async def test_a_refusal_of_replay_checks_lists_each_variants_problem_and_the_tasks(tmp_path, rootful, store):
    service = make_service(tmp_path, rootful, store)
    checks = {
        "variants": [_variant(), _variant(name="ten", inputs="tests/gen/ten", expected="tests/gen/ten.json")],
        "forbidden_literals": None,
    }
    source = write_task(
        tmp_path / "sources",
        edit=edit("environment", "network_mode", "public"),
        files={**CHECK_FILES, "tests/gen/variants/fifty/expected.json": "[]"},
    )

    made = await service.materialize(source, replay_checks=checks)

    assert len(made.problems) == 3, made.problems
    assert "variant fifty's expected outputs" in made.problems[1] and "variant ten's inputs" in made.problems[2]


_CHECK_ZERO = {**_SCORED_ZERO, "detail": "the replay of variant fifty gave a.json#/answer outside the tolerance"}


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["solver_variant_failed", "solver_hardcoded"])
async def test_a_replay_checks_zero_is_the_solvers_and_never_regraded(tmp_path, rootful, store, monkeypatch, reason):
    """It is no score of the grader's, which the launcher gives before the grader runs."""

    report = {**_CHECK_ZERO, "reason": reason}
    monkeypatch.setattr(runtime, "TEST_SH", _in_turn(tmp_path, _launched(report, '{"reward": 0}'), _launched(_PASSED)))
    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources"))

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.reward, outcome.reason, outcome.failure_kind) == ("scored", 0.0, reason, None)
    assert outcome.metadata["launcher"] == report and "regrades" not in outcome.metadata
    assert len(_verifiers(tmp_path)) == 1


@pytest.mark.asyncio
async def test_a_regrade_whose_replay_fails_a_variant_makes_the_grade_the_solvers_zero(
    tmp_path, rootful, store, monkeypatch
):
    """Each regrade runs the task's replay checks again: a submission that follows its inputs only once fails the
    variant in a regrade, and the grade is that regrade's own, the solver's 0."""

    report = {**_CHECK_ZERO, "reason": "solver_variant_failed"}
    monkeypatch.setattr(runtime, "TEST_SH", _in_turn(tmp_path, _launched(_PASSED), _launched(report, '{"reward": 0}')))
    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources"))

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.reward, outcome.reason) == ("scored", 0.0, "solver_variant_failed")
    assert outcome.metadata["launcher"] == report and outcome.metadata["first_grade"]["metadata"]["launcher"] == _PASSED
    assert [entry["reason"] for entry in outcome.metadata["regrades"]] == ["solver_variant_failed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["solver_variant_failed", "solver_hardcoded"])
async def test_a_replay_checks_reason_with_a_reward_of_1_is_no_report_the_launcher_writes(
    tmp_path, rootful, store, monkeypatch, reason
):
    monkeypatch.setattr(runtime, "TEST_SH", _launched({**_CHECK_ZERO, "reason": reason, "reward": 1}))
    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources"))

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.failure_kind) == ("invalid", "verifier_generated_unfinished")
    assert "its report holds no reward the launcher writes" in outcome.reason


# ----------------------------------------------------------------------------------------- hidden evaluation conditions

# A grader manifest whose replay has two hidden conditions, with their inputs in tests/gen/conditions and the grader's
# hidden targets beside them, outside every condition's inputs.
CONDITIONED_JSON = {
    **GRADER_JSON,
    "replay": {
        **GRADER_JSON["replay"],
        "conditions": [{"id": "E1", "inputs": "conditions/E1"}, {"id": "E2", "inputs": "conditions/E2"}],
    },
}
CONDITION_FILES = {
    "tests/gen/grader.json": json.dumps(CONDITIONED_JSON),
    "tests/gen/conditions/E1/data.txt": "50\n",
    "tests/gen/conditions/E2/data.txt": "60\n",
    "tests/gen/targets.json": json.dumps({"E1": 100, "E2": 120}),
}


def _manifest(conditions: Any) -> str:
    return json.dumps({**GRADER_JSON, "replay": {**GRADER_JSON["replay"], "conditions": conditions}})


def test_the_grader_manifest_gives_the_replays_conditions_with_their_inputs_as_paths_of_the_task(tmp_path):
    manifest = tmp_path / "grader.json"
    manifest.write_text(json.dumps(CONDITIONED_JSON))

    read, problems = runtime.read_grader_manifest(manifest)

    assert problems == [] and read is not None
    assert read["replay"] == {
        "command": ["python3", "solve.py"],
        "outputs": ["a.json"],
        "conditions": [
            {"id": "E1", "inputs": "tests/gen/conditions/E1"},
            {"id": "E2", "inputs": "tests/gen/conditions/E2"},
        ],
    }
    manifest.write_text(json.dumps(GRADER_JSON))
    assert "conditions" not in runtime.read_grader_manifest(manifest)[0]["replay"]  # none: the launcher's as before


def test_the_grader_manifest_holds_the_most_conditions_with_their_longest_ids_and_long_paths(tmp_path):
    """32 conditions, each of a 128-character id and inputs 2,000 characters long, take more than 64 KiB."""

    conditions = [
        {"id": f"{number:02d}" + "x" * 126, "inputs": f"conditions/{number:02d}/" + "d/" * 993 + "in"}
        for number in range(32)
    ]
    manifest = tmp_path / "grader.json"
    manifest.write_text(_manifest(conditions))
    assert manifest.stat().st_size > 64 * 1024

    read, problems = runtime.read_grader_manifest(manifest)

    assert problems == [] and read is not None and len(read["replay"]["conditions"]) == 32


@pytest.mark.parametrize(
    ("conditions", "problem"),
    [
        ([], "replay.conditions must list 1 to 32 conditions, each an object of id and inputs (leave conditions out"),
        ({"id": "E1", "inputs": "c"}, "replay.conditions must list 1 to 32 conditions"),
        ([{"id": f"E{n}", "inputs": f"c/{n}"} for n in range(33)], "replay.conditions must list 1 to 32 conditions"),
        (["conditions/E1"], "replay.conditions[0] must be an object of id and inputs, and nothing else"),
        ([{"id": "E1"}], "replay.conditions[0] must be an object of id and inputs, and nothing else"),
        (
            [{"id": "E1", "inputs": "c", "outputs": ["a.json"]}],
            "replay.conditions[0] must be an object of id and inputs, and nothing else",
        ),
        ([{"id": "E/1", "inputs": "c"}], "replay.conditions[0].id must be letters, digits, _, . and -, at most 128"),
        ([{"id": ".E1", "inputs": "c"}], 'starting with a letter or digit, got ".E1"'),
        ([{"id": 1, "inputs": "c"}], "starting with a letter or digit, got 1"),
        ([{"id": "E1", "inputs": "c"}, {"id": "E1", "inputs": "d"}], "replay.conditions name the condition E1 twice"),
        ([{"id": "E1", "inputs": "../c"}], "replay.conditions[0].inputs must be a directory of the task's tests/gen"),
        ([{"id": "E1", "inputs": "/tests/gen/c"}], 'relative to it (conditions/E1), got "/tests/gen/c"'),
        ([{"id": "E1", "inputs": "c//d"}], 'got "c//d"'),
        ([{"id": "E1", "inputs": "."}], 'got "."'),
        ([{"id": "E1", "inputs": "c" * 2100}], "replay.conditions[0].inputs must be a directory"),
    ],
    ids=[
        "empty",
        "no-list",
        "too-many",
        "no-object",
        "no-inputs",
        "an-outputs-key",
        "an-id-with-a-slash",
        "a-hidden-id",
        "an-id-no-string",
        "an-id-twice",
        "inputs-above-tests-gen",
        "inputs-absolute",
        "inputs-not-normalized",
        "inputs-tests-gen-itself",
        "inputs-too-long",
    ],
)
def test_the_grader_manifest_refuses_conditions_no_launcher_runs(tmp_path, conditions, problem):
    manifest = tmp_path / "grader.json"
    manifest.write_text(_manifest(conditions))

    read, problems = runtime.read_grader_manifest(manifest)

    assert read is None and any(problem in found for found in problems), problems


@pytest.mark.asyncio
async def test_materialize_records_the_conditions_in_the_tasks_launcher_settings(tmp_path, rootful, store):
    """The conditions are the task's own word, in its grader manifest: its launcher settings record them, with their
    inputs as paths of the task, and a task without conditions has the settings it had."""

    service = make_service(tmp_path, rootful, store)
    conditioned = await service.materialize(write_task(tmp_path / "sources", files=CONDITION_FILES))
    plain = await service.materialize(write_task(tmp_path / "plain"))

    assert conditioned.ok and plain.ok, (conditioned.problems, plain.problems)
    settings = json.loads((store / conditioned.digest / "tests" / "platoon" / "launch.json").read_text())
    assert settings["replay"]["conditions"] == [
        {"id": "E1", "inputs": "tests/gen/conditions/E1"},
        {"id": "E2", "inputs": "tests/gen/conditions/E2"},
    ]
    assert launcher.read_config(str(store / conditioned.digest / "tests" / "platoon" / "launch.json")) == settings
    plain_settings = json.loads((store / plain.digest / "tests" / "platoon" / "launch.json").read_text())
    assert plain_settings["replay"] == GRADER_JSON["replay"]


@pytest.mark.asyncio
async def test_a_time_for_each_replay_counts_the_conditions(tmp_path, rootful, store):
    """The template's verifier gives the launcher 108 s: the replay and two conditions may take 35 s each, not 36."""

    service = make_service(tmp_path, rootful, store)
    source = write_task(tmp_path / "sources", files=CONDITION_FILES)

    assert (await service.materialize(source, replay_timeout_s=35)).ok
    made = await service.materialize(source, replay_timeout_s=36)  # the replay alone fits: the task's conditions not
    assert not made.ok and made.problems == (
        "tests/gen/grader.json lists 2 conditions, whose replays leave the grader no time: 3 replays of 36 s leave the "
        "grader none of the 108 s the launcher's steps have (the verifier's time limit, scaled); list fewer conditions",
    )
    with pytest.raises(ValueError, match=re.escape("3 replays of 108 s leave the grader none of the 108 s")):
        await service.materialize(source, replay_timeout_s=108)  # the caller's time, which not even the replay fits


def _conditions_with_checks() -> dict[str, Any]:
    return {**CHECKS, "variants": [_variant(inputs="tests/gen/conditions/E1/v", expected="tests/gen/v.json")]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("files", "checks", "problem"),
    [
        (
            {"tests/gen/conditions/E2/data.txt": None},
            None,
            "condition E2's inputs, tests/gen/conditions/E2, cannot be read:",
        ),
        (
            {"tests/gen/conditions/E2/data.txt": None, "tests/gen/conditions/E2": "a file\n"},
            None,
            "condition E2's inputs, tests/gen/conditions/E2, are a file, not a directory",
        ),
        (
            {"tests/gen/conditions/E2/data.txt": None, "tests/gen/conditions/E2/empty/.keep": None},
            None,
            "condition E2's inputs, tests/gen/conditions/E2, hold no file to lay over the working inputs",
        ),
        (
            {"tests/gen/conditions/E1/solve.py": "print(1)\n"},
            None,
            "condition E1's inputs cannot be laid over the working inputs: solve.py would change solve.py, which the "
            "submission holds or the replay writes",
        ),
        ({"tests/gen/conditions/E1/a.json": "{}"}, None, "a.json would change a.json"),
        (
            {"tests/gen/conditions/E1/tools": "a file\n"},
            None,
            "tools is a file of the condition's and a directory of environment/'s",
        ),
        (
            {"tests/gen/conditions/E1/data.txt": None, "tests/gen/conditions/E1/data.txt/x": "1\n"},
            None,
            "data.txt is a directory of the condition's and a file of environment/'s",
        ),
        (
            {"tests/gen/conditions/E1/hidden-format/query.tsv": "k\tu\n"},
            None,
            "condition E1's inputs hold hidden-format/query.tsv, which environment/ does not hold as a file: a "
            "condition replaces public inputs, and adds none",
        ),
        (
            {
                "tests/gen/grader.json": _manifest(
                    [{"id": "E1", "inputs": "conditions/E1"}, {"id": "E2", "inputs": "conditions/E1/E2"}]
                ),
                "tests/gen/conditions/E1/E2/data.txt": "60\n",
            },
            None,
            "condition E2's inputs lie in condition E1's",
        ),
        (
            {
                "tests/gen/grader.json": _manifest(
                    [{"id": "E1", "inputs": "conditions/E1"}, {"id": "E2", "inputs": "conditions/E1"}]
                )
            },
            None,
            "conditions E1 and E2 name the same inputs",
        ),
        (
            {**CHECK_FILES, "tests/gen/conditions/E1/v/data.txt": "7\n", "tests/gen/v.json": '{"a.json#/answer": 14}'},
            _conditions_with_checks(),
            "condition E1's inputs and replay variant fifty's lie one in the other",
        ),
        (
            {**CHECK_FILES, "tests/gen/conditions/E1/expected.json": '{"a.json#/answer": 100}'},
            {**CHECKS, "variants": [_variant(expected="tests/gen/conditions/E1/expected.json")]},
            "replay variant fifty's expected outputs lie in condition E1's inputs",
        ),
        (
            {**CHECK_FILES, "tests/gen/conditions/E1/forbidden.json": '[{"value": 42, "tolerance": 0.5}]'},
            {
                **CHECKS,
                "forbidden_literals": {
                    **CHECKS["forbidden_literals"],
                    "path": "tests/gen/conditions/E1/forbidden.json",
                },
            },
            "the forbidden literals lie in condition E1's inputs",
        ),
    ],
    ids=[
        "no-inputs",
        "inputs-a-file",
        "empty-inputs",
        "inputs-replace-an-artifact",
        "inputs-replace-an-output",
        "inputs-a-file-for-a-directory",
        "inputs-a-directory-for-a-file",
        "inputs-a-file-the-environment-lacks",
        "nested",
        "the-same-inputs",
        "a-variants-inputs-inside",
        "a-variants-expected-outputs-inside",
        "the-forbidden-literals-inside",
    ],
)
async def test_materialize_refuses_conditions_whose_inputs_the_launcher_could_not_use(
    tmp_path, rootful, store, files, checks, problem
):
    service = make_service(tmp_path, rootful, store)
    source = write_task(tmp_path / "sources", files={**CONDITION_FILES, **files})
    if "tests/gen/conditions/E2/empty/.keep" in files:
        (source / "tests" / "gen" / "conditions" / "E2" / "empty").mkdir(parents=True)
    before = sorted(str(path.relative_to(source)) for path in source.rglob("*"))

    made = await service.materialize(source, **({"replay_checks": checks} if checks is not None else {}))

    assert not made.ok and any(problem in found for found in made.problems), made.problems
    assert not any(".staging" in found or str(store) in found for found in made.problems)
    assert [path.name for path in store.iterdir()] == [".staging"] and list((store / ".staging").iterdir()) == []
    assert sorted(str(path.relative_to(source)) for path in source.rglob("*")) == before


_CONDITION_ZERO = {
    **_SCORED_ZERO,
    "reason": "solver_condition_failed",
    "detail": "the replay of condition E1 exited 3",
}


@pytest.mark.asyncio
async def test_a_failed_conditions_zero_is_the_solvers_and_never_regraded(tmp_path, rootful, store, monkeypatch):
    monkeypatch.setattr(
        runtime, "TEST_SH", _in_turn(tmp_path, _launched(_CONDITION_ZERO, '{"reward": 0}'), _launched(_PASSED))
    )
    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources", files=CONDITION_FILES))

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.reward, outcome.reason, outcome.failure_kind) == (
        "scored",
        0.0,
        "solver_condition_failed",
        None,
    )
    assert "regrades" not in outcome.metadata and len(_verifiers(tmp_path)) == 1
    assert not runtime.grader_scored(outcome)


@pytest.mark.asyncio
async def test_a_failed_conditions_reason_with_a_reward_of_1_is_no_report_the_launcher_writes(
    tmp_path, rootful, store, monkeypatch
):
    monkeypatch.setattr(runtime, "TEST_SH", _launched({**_CONDITION_ZERO, "reward": 1}))
    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources", files=CONDITION_FILES))

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.failure_kind) == ("invalid", "verifier_generated_unfinished")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("conditions", "expected"),
    [
        (("c" * 64, "c" * 64), ("invalid", None, "verifier_flaky")),
        (("c" * 64, "d" * 64), ("scored", 0.0, None)),
    ],
    ids=["the-same-condition-outputs", "other-condition-outputs"],
)
async def test_a_disagreement_over_what_the_conditions_replays_wrote_otherwise_is_the_solvers_0(
    tmp_path, rootful, store, monkeypatch, conditions, expected
):
    """A submission deterministic on the working inputs and random on the hidden conditions writes the same replayed
    outputs each time, and other condition outputs: a grader that scores those otherwise is not flaky, and the
    solver's 0 is the grade.  The same bytes everywhere, scored otherwise, are the grader's flakiness."""

    reports = [
        {**report, "replayed_sha256": "a" * 64, "conditions_sha256": digest}
        for report, digest in zip((_PASSED, _FAILED), conditions, strict=True)
    ]
    monkeypatch.setattr(runtime, "TEST_SH", _in_turn(tmp_path, *map(_launched, reports)))
    service = make_service(tmp_path, rootful, store)
    made = await service.materialize(write_task(tmp_path / "sources", files=CONDITION_FILES))

    outcome = await _graded(service, made.task)

    assert (outcome.status, outcome.reward, outcome.failure_kind) == expected
    if outcome.trainable:
        assert outcome.reason == "solver_nonreproducible"
    else:
        assert outcome.reason.startswith(
            "the task's generated grader scored the same artifacts, replayed outputs and condition outputs 1, then 0"
        )
