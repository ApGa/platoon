from __future__ import annotations

from pathlib import Path

import pytest

from platoon.envs.harbor.task import (
    HARBOR_MISC_KEY,
    Artifact,
    HarborTask,
    HarborTaskError,
    Healthcheck,
    McpServer,
    VerifierEnvironment,
    resolve_env,
    strip_canary,
)


def test_loads_a_task_directory(harbor_fixtures: Path):
    task = HarborTask.load(harbor_fixtures / "hello-file")

    assert task.name == "platoon/hello-file"
    assert task.instruction == "Create the file `/app/hello.txt` containing the single line `hello`.\n"
    assert (task.docker_image, task.workdir, task.network) == ("ubuntu:24.04", "/app", "host")
    assert (task.cpus, task.memory_mb) == (1, 512)
    assert (task.agent_timeout_s, task.verifier_timeout_s) == (120.0, 30.0)
    assert task.verifier_env == {"EXPECTED": "hello"}
    assert task.healthcheck == Healthcheck(command="test -d /app", interval_s=0.1, retries=2)
    assert task.metadata == {"difficulty": "easy"}
    assert task.tests_dir == harbor_fixtures.resolve() / "hello-file" / "tests"


def test_reads_mcp_servers_network_mode_and_defaults(harbor_fixtures: Path):
    task = HarborTask.load(harbor_fixtures / "failing-check")

    assert task.name == "failing-check"
    assert task.network == "none"
    assert task.agent_timeout_s is None
    assert task.mcp_servers == (
        McpServer(name="calculator", transport="stdio", command="python", args=("-m", "calculator_server")),
        McpServer(name="docs", transport="streamable-http", url="http://localhost:8000/mcp"),
    )


def test_to_task_locates_the_directory(harbor_fixtures: Path):
    harbor_task = HarborTask.load(harbor_fixtures / "hello-file")

    task = harbor_task.to_task("hello-file")

    assert (task.id, task.goal) == ("hello-file", harbor_task.instruction)
    assert task.misc[HARBOR_MISC_KEY] == {
        "task_dir": str(harbor_task.task_dir),
        "name": "platoon/hello-file",
        "agent_timeout_s": 120.0,
    }


def test_reads_legacy_environment_fields(make_task):
    task = HarborTask.load(make_task("legacy", task_toml='[environment]\nmemory = "2G"\nallow_internet = false\n'))

    assert (task.memory_mb, task.network, task.verifier_timeout_s) == (2048, "none", 600.0)


@pytest.mark.parametrize(
    ("task_toml", "message"),
    [
        ('[[steps]]\nname = "one"\n', "multi-step"),
        ('[environment]\nos = "windows"\n', "only Linux"),
        ('[verifier]\nenvironment_mode = "parallel"\n', "environment_mode must be shared or separate"),
        ('[verifier]\ntimeout_sec = "slow"\n', r"\[verifier\].timeout_sec must be int or float"),
        ("[agent]\ntimeout_sec = true\n", r"\[agent\].timeout_sec must be int or float"),
        ('[environment]\nnetwork_mode = "vpn"\n', "network_mode"),
        ('[environment]\nmemory = "lots"\n', "memory must look like"),
        ("[environment.env]\nTOKEN = 3\n", r"env.TOKEN must be str"),
        ('[[environment.mcp_servers]]\nname = "x"\ntransport = "stdio"\n', "stdio needs a command"),
        ('[[environment.mcp_servers]]\nname = "x"\ntransport = "ws"\nurl = "ws://x"\n', "transport must be"),
        ("[environment.healthcheck]\nretries = 1\n", "command is required"),
        ("not toml [", "task.toml"),
    ],
)
def test_rejects_tasks_platoon_cannot_run(make_task, task_toml: str, message: str):
    with pytest.raises(HarborTaskError, match=message):
        HarborTask.load(make_task("bad", task_toml=task_toml))


def test_requires_the_task_files(make_task, tmp_path: Path):
    with pytest.raises(HarborTaskError, match="no task.toml"):
        HarborTask.load(tmp_path)
    task_dir = make_task("no-tests")
    (task_dir / "tests" / "test.sh").unlink()
    with pytest.raises(HarborTaskError, match="tests/test.sh"):
        HarborTask.load(task_dir)
    (task_dir / "instruction.md").unlink()
    with pytest.raises(HarborTaskError, match="instruction.md"):
        HarborTask.load(task_dir)


def test_strip_canary_drops_only_leading_markers():
    text = "# BENCHMARK DATA canary GUID 1234\n<!-- canary -->\n\nDo it.\n# canary in body stays\n"

    assert strip_canary(text) == "Do it.\n# canary in body stays\n"


def test_resolve_env_reads_host_templates(monkeypatch):
    monkeypatch.setenv("PLATOON_TEST_TOKEN", "secret")
    monkeypatch.delenv("PLATOON_TEST_UNSET", raising=False)

    resolved = resolve_env({"A": "${PLATOON_TEST_TOKEN}", "B": "${PLATOON_TEST_UNSET:-fallback}", "C": "literal"})

    assert resolved == {"A": "secret", "B": "fallback", "C": "literal"}
    with pytest.raises(KeyError, match="PLATOON_TEST_UNSET"):
        resolve_env({"A": "${PLATOON_TEST_UNSET}"})


COMPOSE = """services:
  main:
    environment:
      - JUDGE_URL=http://judge:8082
  judge:
    image: frontier-cs-judge:latest
"""


@pytest.mark.parametrize("compose", ["docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml"])
@pytest.mark.parametrize("task_toml", ["", '[environment]\ndocker_image = "ubuntu:24.04"\n'], ids=["default", "image"])
def test_refuses_multi_container_tasks_whatever_their_image_source(make_task, compose: str, task_toml: str):
    task_dir = make_task("stack", task_toml=task_toml)
    (task_dir / "environment" / compose).write_text(COMPOSE)

    with pytest.raises(HarborTaskError, match=rf"environment/{compose} makes the environment a multi-container"):
        HarborTask.load(task_dir)


# ------------------------------------------------------------------------------------------- separate verifiers


def _verifier_files(task_dir: Path, *files: str) -> Path:
    for name in files:
        (task_dir / "tests" / name).write_text("FROM python:3.12-slim\n")
    return task_dir


def test_a_separate_verifier_without_its_own_table_copies_the_environment(make_task):
    task = HarborTask.load(
        make_task(
            "smoldata-like",
            task_toml="""artifacts = ["/workdir/answer.txt"]
[verifier]
environment_mode = "separate"
[environment]
docker_image = "python:3.12-slim"
workdir = "/workdir"
cpus = 2
memory_mb = 4096
network_mode = "no-network"
env = {MODE = "train"}
""",
        )
    )

    assert task.verifier_mode == "separate"
    assert task.artifacts == (Artifact(source="/workdir/answer.txt"),)
    assert task.verifier_environment == VerifierEnvironment(
        image_source="task",
        docker_image="python:3.12-slim",
        workdir="/workdir",
        env={"MODE": "train"},
        network="none",
        cpus=2,
        memory_mb=4096,
    )
    assert not task.verifier_environment.bundled_tests  # the task's own image: tests/ is uploaded


def test_a_verifier_table_implies_separate_mode_and_brings_its_own_settings(make_task):
    task = HarborTask.load(
        make_task(
            "tbs-like",
            task_toml="""artifacts = ["/root/results/out.csv", {source = "/root/work", exclude = ["*.pyc", "cache"]}]
[agent]
network_mode = "no-network"
[verifier]
network_mode = "public"
[verifier.environment]
cpus = 1
memory = "2G"
network_mode = "no-network"
[environment]
network_mode = "public"
cpus = 4
""",
        )
    )

    assert task.network == "none"  # the agent phase's own policy wins over the environment's
    assert task.artifacts == (
        Artifact(source="/root/results/out.csv"),
        Artifact(source="/root/work", exclude=("*.pyc", "cache")),
    )
    venv = task.verifier_environment
    assert (venv.image_source, venv.docker_image, venv.cpus, venv.memory_mb) == ("task", None, 1, 2048)
    assert venv.network == "host"  # [verifier].network_mode is the phase override


def test_the_verifier_image_follows_harbors_precedence(make_task):
    image = HarborTask.load(
        make_task("image", task_toml='[verifier.environment]\ndocker_image = "grader:1"\n')
    ).verifier_environment
    assert (image.image_source, image.docker_image, image.bundled_tests) == ("verifier_image", "grader:1", True)

    built = HarborTask.load(
        _verifier_files(make_task("built", task_toml='[verifier]\nenvironment_mode = "separate"\n'), "Dockerfile")
    ).verifier_environment
    assert (built.image_source, built.bundled_tests) == ("tests", True)

    # The table's image comes first, before tests/Dockerfile.
    both = _verifier_files(make_task("both", task_toml='[verifier.environment]\ndocker_image = "g:1"\n'), "Dockerfile")
    assert HarborTask.load(both).verifier_environment.image_source == "verifier_image"


def test_a_verifier_with_its_own_image_needs_no_test_script_in_the_task(make_task):
    bundled = make_task("bundled", task_toml='[verifier.environment]\ndocker_image = "grader:1"\n')
    (bundled / "tests" / "test.sh").unlink()
    assert HarborTask.load(bundled).verifier_environment.bundled_tests

    uploaded = make_task("uploaded", task_toml='[verifier]\nenvironment_mode = "separate"\n')
    (uploaded / "tests" / "test.sh").unlink()
    with pytest.raises(HarborTaskError, match="tests/test.sh"):
        HarborTask.load(uploaded)  # its verifier runs the task's own image, which gets the task's tests


@pytest.mark.parametrize(
    ("task_toml", "message"),
    [
        ('[verifier]\nenvironment_mode = "shared"\n[verifier.environment]\ncpus = 1\n', "conflicts with"),
        ('[verifier]\ncollect = [{command = "dump-db"}]\n', r"\[\[verifier.collect\]\] hooks"),
        ('artifacts = [{source = "/data", service = "db"}]\n', "compose service 'db'"),
        ('artifacts = ["../etc/passwd"]\n', "must not contain '..'"),
        ("artifacts = [3]\n", "artifacts entries must be paths"),
        ('artifacts = [{source = "/x", exclude = "*.pyc"}]\n', r"artifacts.exclude must be list"),
        ('[verifier.environment]\nos = "windows"\n', "only Linux"),
        ('[verifier.environment]\nnetwork_mode = "vpn"\n', r"\[verifier.environment\].network_mode must be"),
        ('[agent]\nnetwork_mode = "vpn"\n', r"\[agent\].network_mode must be"),
        ("[environment]\ngpus = 1.5\n", r"gpus must be int"),
    ],
)
def test_refuses_verifier_settings_platoon_cannot_honor(make_task, task_toml: str, message: str):
    with pytest.raises(HarborTaskError, match=message):
        HarborTask.load(make_task("bad", task_toml=task_toml))


def test_a_compose_file_in_tests_makes_the_verifier_a_stack(make_task):
    task_dir = _verifier_files(make_task("stack", task_toml='[verifier]\nenvironment_mode = "separate"\n'))
    (task_dir / "tests" / "docker-compose.yaml").write_text("services:\n  main: {}\n  judge: {image: j}\n")

    with pytest.raises(HarborTaskError, match="tests/docker-compose.yaml makes the separate verifier"):
        HarborTask.load(task_dir)


@pytest.mark.parametrize(
    "compose",
    ["services:\n  main: {}\n", "services:\n  main:\n", 'version: "3"\nservices:\n  main: {}\n'],
    ids=["empty-mapping", "null", "with-version"],
)
def test_a_compose_file_that_only_names_main_is_one_container(make_task, compose: str):
    task_dir = make_task("trivial", task_toml='[environment]\ndocker_image = "ubuntu:24.04"\n')
    (task_dir / "environment" / "docker-compose.yaml").write_text(compose)

    assert HarborTask.load(task_dir).docker_image == "ubuntu:24.04"


@pytest.mark.parametrize(
    "compose",
    [
        "services:\n  main:\n    environment: [A=1]\n",
        "services:\n  main: {}\nnetworks:\n  private: {}\n",
        "services:\n  app: {}\n",
        "services: [main]\n",
        "{not: yaml: at all",
    ],
    ids=["main-with-keys", "networks", "other-service", "list", "unparsable"],
)
def test_any_other_compose_file_stays_refused(make_task, compose: str):
    task_dir = make_task("stack")
    (task_dir / "environment" / "compose.yaml").write_text(compose)

    with pytest.raises(HarborTaskError, match="makes the environment a multi-container stack"):
        HarborTask.load(task_dir)


def test_users_are_refused_unless_ignored(make_task):
    task_dir = make_task("users", task_toml='[agent]\nuser = "agent"\n[verifier]\nuser = 0\n')

    with pytest.raises(HarborTaskError, match=r"\[agent\].user = 'agent', \[verifier\].user = '0'.*agent_user: ignore"):
        HarborTask.load(task_dir)
    task = HarborTask.load(task_dir, agent_user="ignore")
    assert task.ignored_users == {"agent": "agent", "verifier": "0"}
    with pytest.raises(ValueError, match="agent_user must be one of"):
        HarborTask.load(task_dir, agent_user="obey")  # type: ignore[arg-type]


def test_gpus_and_the_index_row_are_kept(make_task):
    task = HarborTask.load(make_task("gpu", task_toml="[environment]\ngpus = 2\n"), index={"source_dataset": "x/y"})

    assert (task.gpus, task.index) == (2, {"source_dataset": "x/y"})
    assert task.to_task("gpu").misc[HARBOR_MISC_KEY]["index"] == {"source_dataset": "x/y"}
    assert "index" not in HarborTask.load(make_task("plain")).to_task("plain").misc[HARBOR_MISC_KEY]
