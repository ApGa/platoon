from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from platoon.envs.harbor import task as task_module
from platoon.envs.harbor.compose import plan_compose
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
    command: ["judge", "--port", "8082"]
"""


@pytest.mark.parametrize("task_toml", ["", '[environment]\ndocker_image = "ubuntu:24.04"\n'], ids=["default", "image"])
def test_refuses_multi_container_tasks_where_no_stack_runs_whatever_their_image_source(make_task, task_toml: str):
    task_dir = make_task("stack", task_toml=task_toml)
    (task_dir / "environment" / "docker-compose.yaml").write_text(COMPOSE)

    with pytest.raises(
        HarborTaskError,
        match=r"environment/docker-compose.yaml makes the environment a multi-container stack \(sidecar services\), "
        "and the docker provider runs none; the task is refused, never started without them",
    ):
        HarborTask.load(task_dir, stack_refusal="the docker provider runs none")
    task = HarborTask.load(task_dir)  # where stacks run, it is one
    assert task.is_stack and task.main is not None and task.main.environment == {"JUDGE_URL": "http://judge:8082"}


@pytest.mark.parametrize("compose", ["docker-compose.yml", "compose.yaml", "compose.yml"])
def test_compose_names_that_harbor_does_not_read_are_plain_files(make_task, compose: str, caplog):
    """Harbor reads only environment/docker-compose.yaml: a task with another compose name runs as one container in
    Harbor, and so in platoon, which warns of it."""

    task_dir = make_task(f"plain-{compose}", task_toml='[environment]\ndocker_image = "ubuntu:24.04"\n')
    (task_dir / "environment" / compose).write_text(COMPOSE)
    task_module._WARNED_UNREAD.discard(f"environment/{compose}")

    with caplog.at_level("WARNING", logger="platoon.envs.harbor.task"):
        task = HarborTask.load(task_dir)

    assert task.compose is None and task.docker_image == "ubuntu:24.04"
    assert f"has environment/{compose}, which Harbor does not read" in caplog.text


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
        (
            '[verifier]\ncollect = [{command = "dump-db", service = "db"}]\n',
            r"\[\[verifier.collect\]\] runs a hook in the compose service 'db', which the task does not start",
        ),
        ('[verifier]\ncollect = [{service = "main"}]\n', r"\[\[verifier.collect\]\] needs a command"),
        ('[verifier]\ncollect = [{command = "x", cwd = "/"}]\n', r"has unknown keys \['cwd'\]"),
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


STACK = "services:\n  main: {}\n  db:\n    image: postgres:16\n    command: [postgres]\n"


def test_collect_hooks_run_main_s_first_and_their_users_follow_the_policy(make_task):
    task_dir = make_task(
        "hooks",
        task_toml=(
            '[[verifier.collect]]\nservice = "db"\ncommand = "pg_dump > /tmp/db.sql"\ntimeout_sec = 120\n'
            '[[verifier.collect]]\ncommand = "sync"\n'
        ),
    )
    (task_dir / "environment" / "docker-compose.yaml").write_text(STACK)
    task = HarborTask.load(task_dir)
    assert [(hook.service, hook.command, hook.timeout_s) for hook in task.collect] == [
        ("main", "sync", 60.0),  # main's hooks run first, as in Harbor
        ("db", "pg_dump > /tmp/db.sql", 120.0),
    ]

    as_user = make_task("as-user", task_toml='[[verifier.collect]]\ncommand = "sync"\nuser = "postgres"\n')
    with pytest.raises(HarborTaskError, match="runs a hook as user 'postgres'.*sets agent_user: honor or ignore"):
        HarborTask.load(as_user)
    honored = HarborTask.load(as_user, verifier_user="honor")  # the hook runs as its user, in main
    assert honored.collect[0].user == "postgres" and honored.honors_users_in("agents")
    ignored = HarborTask.load(as_user, agent_user="ignore")
    assert ignored.ignored_users == {"collect": "postgres"} and ignored.collect[0].service == "main"
    assert ignored.collect[0].user is None and not ignored.honors_users_in("agents")


def test_artifacts_of_a_service_name_an_absolute_path_of_one_the_stack_starts(make_task):
    toml = 'artifacts = ["out.txt", {source = "/var/lib/db/dump.sql", service = "db"}, '
    toml += '{source = "x", service = "main"}]\n'
    task_dir = make_task("artifacts", task_toml=toml)
    (task_dir / "environment" / "docker-compose.yaml").write_text(STACK)
    assert [(artifact.source, artifact.service) for artifact in HarborTask.load(task_dir).artifacts] == [
        ("out.txt", None),
        ("/var/lib/db/dump.sql", "db"),
        ("x", None),
    ]
    relative = make_task("relative", task_toml='artifacts = [{source = "dump.sql", service = "db"}]\n')
    (relative / "environment" / "docker-compose.yaml").write_text(STACK)
    with pytest.raises(HarborTaskError, match="must be an absolute path in that service"):
        HarborTask.load(relative)


def test_a_stack_whose_service_a_node_service_serves_loads_with_it(make_task):
    # A node service serves db (an overlay's external): the task loads, and the service that serves it is the
    # environment's to declare (HarborEnvironmentService.check_tasks refuses a task whose node service it lacks).
    task_dir = make_task("external")
    (task_dir / "environment" / "docker-compose.yaml").write_text(STACK)
    plan = HarborTask.load(task_dir).compose
    assert plan is not None
    scope = {"problem_id": "${PROBLEM_ID}"}
    marked = {**plan.to_json(), "services": {**plan.services, "db": {"external": "judge-pool", "scope": scope}}}
    loaded = HarborTask.load(task_dir, compose=marked).compose
    assert loaded is not None
    assert loaded.external == {"db": "judge-pool"}
    assert loaded.external_scope("db") == scope


@pytest.mark.parametrize(
    ("compose", "message"),
    [
        (
            "services:\n  main: {}\n  db:\n    image: postgres:16\n",
            "whose service db runs the command of its registry image postgres:16, which platoon does not know yet",
        ),
        (
            "services:\n  main: {}\n  api:\n    build: ./api\n",
            "whose service api builds its image from environment/api/Dockerfile, and a dataset directory's task has",
        ),
    ],
    ids=["registry-image-without-a-command", "build-context"],
)
def test_a_dataset_directory_s_stack_whose_sidecar_no_rollout_could_start_is_refused(
    make_task, compose: str, message: str
):
    """In a dataset directory, whose tasks have no image map: a fetched dataset's lock keeps them, and a run refuses
    them when it lists them (where the image map lacks a sidecar's image), as a later platoon may run them."""

    task_dir = make_task("unrunnable")
    (task_dir / "environment" / "api").mkdir()
    (task_dir / "environment" / "api" / "Dockerfile").write_text('FROM python:3.12-slim\nCMD ["python"]\n')
    (task_dir / "environment" / "docker-compose.yaml").write_text(compose)
    with pytest.raises(HarborTaskError, match=message):
        HarborTask.load(task_dir)
    plan = plan_compose(task_dir)  # as a fetch plans it, with the build keys
    assert plan is not None and HarborTask.load(task_dir, compose=plan).is_stack


@pytest.mark.parametrize(
    ("compose", "message"),
    [
        (
            "services:\n  main: {}\n  judge:\n    image: j\n    command: [j]\n    volumes:\n"
            "      - ${HOST_AGENT_LOGS_PATH}:${ENV_AGENT_LOGS_PATH}:ro\n",
            "whose service judge binds the agent log directory read-only, to read what main writes there",
        ),
        (
            "services:\n  main:\n    volumes: [./data:/srv/data]\n  web:\n    image: w\n    command: [w]\n"
            "    volumes: [./data/sub:/sub:ro]\n",
            "whose services main and web bind the task's environment/data and environment/data/sub",
        ),
    ],
    ids=["read-only-log-directory", "a-written-bind-inside-another"],
)
def test_a_stack_whose_services_would_not_share_what_compose_shares_is_refused(make_task, compose: str, message: str):
    task_dir = make_task("apart")
    (task_dir / "environment" / "data" / "sub").mkdir(parents=True)
    (task_dir / "environment" / "docker-compose.yaml").write_text(compose)
    with pytest.raises(HarborTaskError, match=message):
        HarborTask.load(task_dir)


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
    ("compose", "message"),
    [
        ("services:\n  app: {}\n", "app: has neither an image nor a build"),
        ("services: [main]\n", "services: must map service names"),
        ("{not: yaml: at all", "the file is not YAML"),
        ("services:\n  main:\n    privileged: true\n", "main.privileged: gives the service every privilege"),
    ],
    ids=["service-without-image", "list", "unparsable", "privileged"],
)
def test_other_compose_files_stay_refused_with_the_reason(make_task, compose: str, message: str):
    task_dir = make_task("stack")
    (task_dir / "environment" / "docker-compose.yaml").write_text(compose)

    with pytest.raises(HarborTaskError, match=message):
        HarborTask.load(task_dir)


@pytest.mark.parametrize(
    ("compose", "kind"),
    [
        ("services:\n  main:\n    environment: [A=1]\n", "main_only"),
        ("services:\n  main: {}\nnetworks:\n  private: {}\n", "trivial"),
        ("services:\n  main:\n    stdin_open: true\n    tty: true\n", "trivial"),
    ],
    ids=["main-with-keys", "unused-network", "ignored-keys"],
)
def test_compose_files_that_name_only_main_run_as_one_sandbox(make_task, compose: str, kind: str):
    task_dir = make_task("main-only", task_toml='[environment]\ndocker_image = "ubuntu:24.04"\n')
    (task_dir / "environment" / "docker-compose.yaml").write_text(compose)

    task = HarborTask.load(task_dir)

    assert task.compose is not None and task.compose.kind == kind
    assert (task.main is not None) == (kind == "main_only")


def test_users_are_refused_unless_ignored(make_task):
    task_dir = make_task("users", task_toml='[agent]\nuser = "agent"\n[verifier]\nuser = 0\n')

    with pytest.raises(HarborTaskError, match=r"\[agent\].user = 'agent', \[verifier\].user = '0'.*honor, .*or ignore"):
        HarborTask.load(task_dir)
    task = HarborTask.load(task_dir, agent_user="ignore")
    assert task.ignored_users == {"agent": "agent", "verifier": "0"}
    honored = HarborTask.load(task_dir, agent_user="honor")  # a root verifier in the sandbox of agents run as theirs
    assert (honored.honored_users, honored.verifier_root, honored.ignored_users) == (
        {"agent": "agent", "verifier": "0"},
        True,
        {},
    )
    unknown: Any = "obey"
    with pytest.raises(ValueError, match="agent_user must be one of"):
        HarborTask.load(task_dir, agent_user=unknown)


def test_gpus_and_the_index_row_are_kept(make_task):
    task = HarborTask.load(make_task("gpu", task_toml="[environment]\ngpus = 2\n"), index={"source_dataset": "x/y"})

    assert (task.gpus, task.index) == (2, {"source_dataset": "x/y"})
    assert task.to_task("gpu").misc[HARBOR_MISC_KEY]["index"] == {"source_dataset": "x/y"}
    assert "index" not in HarborTask.load(make_task("plain")).to_task("plain").misc[HARBOR_MISC_KEY]
