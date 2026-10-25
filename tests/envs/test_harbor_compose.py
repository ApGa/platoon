"""Harbor compose files (platoon.envs.harbor.compose): the subset platoon runs, Harbor's layering and interpolation,
the stack plan a fetch locks, and main-only compose files run on the process provider.

The compose files of Harbor's adapters and examples are copies of Harbor's (``fixtures/harbor-compose/upstream``,
Apache-2.0, see its NOTICE); the tasks below ``fixtures/harbor-compose/tasks`` and the Hub-shaped files here are
written for platoon.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

import pytest

from platoon.envs.harbor import HarborEnvironmentService, HarborSession, compose
from platoon.envs.harbor.compose import (
    COMPOSE_RULE,
    MAX_DOCUMENT_VALUES,
    UNSET,
    ComposeEdit,
    ComposeError,
    ComposePlan,
    MainService,
    ServiceHealthcheck,
    interpolate,
    is_idle,
    plan_compose,
    resolve_argv,
    resolve_deferred,
)
from platoon.envs.harbor.dockerfile import image_config, parse_dockerfile, plan_dockerfile, variables_in_scope
from platoon.envs.harbor.fetch import FetchError, Overlay, fetch, fetch_config
from platoon.envs.harbor.holdout import HoldoutRow, HoldoutRule, hold_out
from platoon.envs.harbor.inputs import input_files
from platoon.envs.harbor.lock import DatasetLock, LockedTask, LockError, read_lock, write_lock
from platoon.envs.harbor.sidecars import bind_mounts
from platoon.envs.harbor.task import HarborTask, HarborTaskError
from platoon.sandbox.process import ProcessProvider, ProcessSandbox

FIXTURES = Path(__file__).parent / "fixtures" / "harbor-compose"
UPSTREAM = FIXTURES / "upstream"
TASKS = FIXTURES / "tasks"
STUB_DOCKERFILE = 'FROM python:3.12-slim\nWORKDIR /srv\nCOPY . /srv\nCMD ["python", "server.py"]\n'


def _task(root: Path, name: str, compose: str | None, *, toml: str = "", files: dict[str, str] | None = None) -> Path:
    task_dir = root / name
    contents = {
        "task.toml": toml,
        "instruction.md": "Do the task.\n",
        "tests/test.sh": "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n",
        "environment/Dockerfile": "FROM python:3.12-slim\nWORKDIR /app\n",
        **({"environment/docker-compose.yaml": compose} if compose is not None else {}),
        **(files or {}),
    }
    for relative, text in contents.items():
        path = task_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return task_dir


def _plan(task_dir: Path, **options) -> ComposePlan:
    plan = plan_compose(task_dir, **options)
    assert plan is not None
    return plan


# --- interpolation -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain", "plain"),
        ("$$HOME and $${HOME}", "$HOME and ${HOME}"),  # $$ is a literal $
        ("${SET}/$SET", "value/value"),
        ("${UNSET}x", "x"),  # unset without a default: blank
        ("${UNSET:-default}|${UNSET-default}", "default|default"),
        ("${EMPTY:-default}|${EMPTY-default}", "default|"),  # :- also replaces an empty value, - only an unset one
        ("${SET:+alt}|${EMPTY:+alt}|${EMPTY+alt}|${UNSET+alt}", "alt||alt|"),
        ("${UNSET:-${SET}}|${UNSET:-${ALSO_UNSET:-deep}}", "value|deep"),  # defaults nest
        ("${UNSET:-}", ""),
        ("${SET:?must be set}", "value"),
        ('{"a": "${SET}"}', '{"a": "value"}'),
        ("echo $1 $? $(date) $", "echo $1 $? $(date) $"),  # a $ that starts no name stays, as in compose
    ],
)
def test_interpolation_follows_compose(text: str, expected: str):
    variables = {"SET": "value", "EMPTY": "", "UNSETTLED": UNSET}

    assert interpolate(text, variables) == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("${UNSET:?give it}", "required variable UNSET is missing a value: give it"),
        ("${EMPTY:?give it}", "required variable EMPTY"),
        ("${UNSET?give it}", "required variable UNSET"),
        ("${unterminated", "unterminated"),
        ("${A B}", "invalid interpolation format"),
        ("${}", "invalid interpolation format"),
    ],
)
def test_interpolation_errors_name_the_form(text: str, message: str):
    with pytest.raises(ComposeError, match=message):
        interpolate(text, {"EMPTY": ""}, where="main.command")


def test_unset_variables_are_noted_once():
    notes: list[str] = []

    interpolate("${A} ${A} $B ${C:-c}", {}, notes=notes)

    assert notes == ["A is not set: blank", "B is not set: blank"]


def test_harbor_variables_take_platoon_meanings(tmp_path: Path):
    """CONTEXT_DIR is the task's environment/, HOST_*_PATH the session's log directories, CPUS and MEMORY the task's
    resources; host variables count only when an overlay lists them."""

    task_dir = _task(
        tmp_path,
        "infra",
        """services:
  main:
    environment:
      - LOGS=${ENV_VERIFIER_LOGS_PATH}
      - CPU_COUNT=${CPUS}
      - MEM=${MEMORY}
      - TASK_VALUE=${FROM_TASK}
      - HOST_HOME=${HOME:-no host variable}
      - TOKEN=${API_TOKEN}
      - PASSED
    volumes:
      - ${CONTEXT_DIR}/data:/data:ro
      - ${HOST_AGENT_LOGS_PATH}:${ENV_AGENT_LOGS_PATH}
""",
        files={"environment/data/x.txt": "x"},
    )

    plan = _plan(
        task_dir, env={"FROM_TASK": "literal", "API_TOKEN": "${API_TOKEN}"}, cpus=2, memory_mb=4096, host_env=["PASSED"]
    )
    main = plan.services["main"]

    assert main["environment"]["LOGS"] == "/logs/verifier"
    assert (main["environment"]["CPU_COUNT"], main["environment"]["MEM"]) == ("2", "4096M")
    assert main["environment"]["TASK_VALUE"] == "literal"
    assert main["environment"]["HOST_HOME"] == "no host variable"  # the host's HOME never counts
    assert main["volumes"] == [
        {"type": "bind", "target": "/data", "read_only": True, "task": "environment/data"},
        {"type": "bind", "target": "/logs/agent", "read_only": False, "logs": "agent"},
    ]
    # The task's host template and the listed host variable stay unresolved: no host value enters a plan.
    assert plan.deferred == ("host:PASSED", "task:API_TOKEN")
    assert "\ue000" in main["environment"]["TOKEN"] and "\ue000" in main["environment"]["PASSED"]
    host = {"API_TOKEN": "s3cret", "PASSED": "passed"}
    env = {"FROM_TASK": "literal", "API_TOKEN": "${API_TOKEN}"}
    assert resolve_deferred(main["environment"]["TOKEN"], env, host) == "s3cret"
    assert resolve_deferred(main["environment"]["PASSED"], env, host) == "passed"
    with pytest.raises(KeyError, match="PASSED"):
        resolve_deferred(main["environment"]["PASSED"], env, {})


@pytest.mark.parametrize(
    ("compose", "message"),
    [
        ("services:\n  main:\n    environment: ['WHERE=${CONTEXT_DIR}']\n", "main.environment: names a host path"),
        ("services:\n  main:\n    command: [cat, '${HOST_VERIFIER_LOGS_PATH}']\n", "main.command: names a host path"),
        ("services:\n  main:\n    command: [echo, '${LISTED}']\n", "main.command: names a host variable"),
        ("services:\n  main:\n    volumes: ['/var/run/docker.sock:/var/run/docker.sock']\n", "binds the host path"),
        ("services:\n  main:\n    volumes: ['${LISTED}:/data']\n", "binds a host path writable"),
        ("services:\n  main:\n    volumes: ['../../outside:/data:ro']\n", "climbs out of the task's directory"),
        ("services:\n  main:\n    volumes: ['./x:/tests:ro']\n", "binds /tests, where platoon puts the tests"),
        # A host path only through a listed host variable, as it names it or below it: never a task's host template.
        ("services:\n  main:\n    volumes: ['${TEMPLATED}:/data:ro']\n", "through the task variable TEMPLATED"),
        ("services:\n  main:\n    volumes: ['${LISTED}/../etc:/data:ro']\n", "other than the one a listed host"),
        ("services:\n  main:\n    volumes: ['${LISTED}2:/data:ro']\n", "other than the one a listed host"),
        ("services:\n  main:\n    volumes: ['${LISTED:-/srv}:/data:ro']\n", "other than the one a listed host"),
        ("services:\n  main:\n    volumes: ['/srv/${LISTED}:/data:ro']\n", "other than the one a listed host"),
        ("services:\n  main:\n    volumes: ['${LISTED}/${TEMPLATED}:/data:ro']\n", "other than the one a listed"),
        ("services:\n  main:\n    volumes: ['./x:/srv/${LISTED}']\n", "mounts at a path that a host variable"),
        ("services:\n  main:\n    environment: {A: '${LISTED:-${TEMPLATED}}'}\n", "a session resolves too"),
    ],
)
def test_host_paths_and_variables_stay_where_they_may(tmp_path: Path, compose: str, message: str):
    task_dir = _task(tmp_path, "paths", compose)

    with pytest.raises(ComposeError, match=message):
        plan_compose(task_dir, host_env=["LISTED"], env={"TEMPLATED": "${HOME}"})


def test_a_listed_host_variable_binds_its_path_or_one_below_it_read_only(tmp_path: Path, monkeypatch):
    task_dir = _task(
        tmp_path,
        "host-bind",
        "services:\n  main:\n    volumes: ['${PLATOON_TEST_DATA}/sub:/data:ro', '${PLATOON_TEST_DATA}:/all:ro']\n",
        toml='[environment]\ndocker_image = "ubuntu:24.04"\n',
    )
    plan = _plan(task_dir, host_env=["PLATOON_TEST_DATA"])
    task = HarborTask.load(task_dir, compose=plan)
    data = tmp_path / "host-data"
    (data / "sub").mkdir(parents=True)
    monkeypatch.setenv("PLATOON_TEST_DATA", str(data))

    assert task.main is not None and plan.deferred == ("host:PLATOON_TEST_DATA",)
    mounts = bind_mounts(task, task.main.binds)

    assert [(mount.source, mount.target, mount.read_only) for mount in mounts] == [
        (str(data / "sub"), "/data", True),
        (str(data), "/all", True),
    ]


@pytest.mark.parametrize(
    ("where", "compose_text", "options", "files"),
    [
        ("main.environment.A", "services:\n  main:\n    environment: {A: '\ue000host.SECRET\ue001'}\n", {}, {}),
        ("main.volumes", 'services:\n  main:\n    volumes: ["\ue000host.SECRET\ue001:/x:ro"]\n', {}, {}),
        ("main.environment", "services:\n  main:\n    environment: {'\ue000': x}\n", {}, {}),
        ("x-anchors", "x-anchors: '\ue002'\nservices:\n  main: {command: [x]}\n", {}, {}),
        (
            "main.env_file: environment/vars.env",
            "services:\n  main:\n    env_file: [vars.env]\n",
            {},
            {"environment/vars.env": "A=\ue000host.SECRET\ue001\n"},
        ),
        (
            "[environment].env.A",
            "services:\n  main:\n    environment: {B: '${A}'}\n",
            {"env": {"A": "\ue000host.SECRET\ue001"}},
            {},
        ),
        (
            "[environment].docker_image",
            "services:\n  main:\n    environment: {B: '${PREBUILT_IMAGE_NAME}'}\n",
            {"docker_image": "\ue000host.SECRET\ue001"},
            {},
        ),
    ],
)
def test_the_private_use_characters_that_mark_deferred_variables_are_refused_wherever_a_task_writes(
    tmp_path: Path, where: str, compose_text: str, options: dict, files: dict
):
    """Written into a task, platoon's markers would name host variables that no overlay lists: refused at fetch."""

    task_dir = _task(tmp_path, "forged", compose_text, files=files)

    with pytest.raises(ComposeError, match="private-use character") as caught:
        plan_compose(task_dir, host_env=["LISTED"], **options)
    assert caught.value.where == where


def test_a_plan_defers_only_the_variables_the_planner_deferred(tmp_path: Path, monkeypatch):
    """Even past the check of what a task writes, a marker the planner did not make is refused, never resolved."""

    monkeypatch.setattr(compose, "_check_reserved", lambda value, where: None)
    task_dir = _task(tmp_path, "forged", "services:\n  main:\n    environment: {A: '\ue000host.SECRET\ue001'}\n")

    with pytest.raises(ComposeError, match="names host:SECRET, which no overlay lists"):
        plan_compose(task_dir, host_env=["LISTED"])


FORMS = ["${HV}", "${HV:-dflt}", "${HV-dflt}", "${HV:+alt}", "${HV+alt}", "${HV:?need it}", "${HV?need it}"]


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("host", [{"HV": "value"}, {"HV": ""}, {}], ids=["set", "empty", "unset"])
def test_a_deferred_variable_takes_its_compose_form_when_the_session_resolves_it(tmp_path: Path, form: str, host):
    """The plan keeps a listed host variable's form (a default, an alternative, a requirement) for the session, which
    applies it as compose would have; one named without a default, or required, must be set then."""

    task_dir = _task(tmp_path, "forms", f"services:\n  main:\n    environment: {{A: 'x{form}y'}}\n")
    value = _plan(task_dir, host_env=["HV"]).services["main"]["environment"]["A"]
    compose_value = None
    try:
        compose_value = interpolate(f"x{form}y", {"HV": host.get("HV", UNSET)})
    except ComposeError:
        pass  # a requirement that compose would fail on

    if compose_value is None or ("HV" not in host and form == "${HV}"):
        with pytest.raises(KeyError, match="HV"):
            resolve_deferred(value, {}, host)
    else:
        assert resolve_deferred(value, {}, host) == compose_value


def test_a_task_variable_that_is_a_host_template_takes_its_compose_form_too(tmp_path: Path):
    task_dir = _task(
        tmp_path, "task-forms", "services:\n  main:\n    environment: {A: '${T:-fallback}', B: '${T:+on}'}\n"
    )
    env = {"T": "${HOST_T:-}"}
    environment = _plan(task_dir, env=env).services["main"]["environment"]

    assert [resolve_deferred(environment[name], env, {"HOST_T": "v"}) for name in "AB"] == ["v", "on"]
    assert [resolve_deferred(environment[name], env, {}) for name in "AB"] == ["fallback", ""]


# --- commands: compose's and Docker's rules (RFC 0010 b.3) -------------------------------------------------------


@pytest.mark.parametrize(
    ("entrypoint", "command", "expected"),
    [
        (None, None, ["/entry.sh", "serve"]),  # the image's ENTRYPOINT and CMD
        (None, ["run", "fast"], ["/entry.sh", "run", "fast"]),  # command replaces the CMD
        (["/other"], None, ["/other"]),  # entrypoint replaces the ENTRYPOINT and drops the CMD
        (["/other"], ["x"], ["/other", "x"]),
        ([], None, []),  # an empty entrypoint clears both
        ([], ["python", "app.py"], ["python", "app.py"]),
    ],
)
def test_a_service_command_follows_compose(entrypoint, command, expected):
    assert resolve_argv(
        image_entrypoint=["/entry.sh"], image_cmd=["serve"], entrypoint=entrypoint, command=command
    ) == (expected)


def test_compose_splits_a_string_command_into_shell_words_without_a_shell(tmp_path: Path):
    task_dir = _task(
        tmp_path,
        "split",
        """services:
  main:
    command: python -c "print('a b')" --flag
    entrypoint: ""
""",
    )

    main = _plan(task_dir).main()

    assert main is not None
    assert main.command == ("python", "-c", "print('a b')", "--flag")
    assert main.entrypoint == ()
    assert main.argv(["/image-entrypoint"]) == ["python", "-c", "print('a b')", "--flag"]


@pytest.mark.parametrize(
    ("dockerfile", "config"),
    [
        # Metadata after the last RUN is the deferred suffix: the plan records what it sets.
        ('FROM a\nRUN x\nCMD ["python", "s.py"]\nEXPOSE 8000\n', {"cmd": ["python", "s.py"], "expose": ["8000/tcp"]}),
        # A shell form runs under /bin/sh -c (a shell-form ENTRYPOINT then ignores CMD's words: they are its $0...).
        (
            "FROM a\nRUN x\nENTRYPOINT exec serve --port 1\n",
            {"entrypoint": ["/bin/sh", "-c", "exec serve --port 1"], "cmd": []},
        ),
        # An ENTRYPOINT empties the CMD the stage inherits, but not one the stage set before it, even in the prefix.
        ('FROM a\nRUN x\nENTRYPOINT ["/e"]\n', {"entrypoint": ["/e"], "cmd": []}),
        ('FROM a\nCMD ["c"]\nRUN x\nENTRYPOINT ["/e"]\n', {"entrypoint": ["/e"]}),
        # SHELL holds for the shell forms after it.
        ('FROM a\nSHELL ["/bin/bash", "-lc"]\nRUN x\nCMD serve\n', {"cmd": ["/bin/bash", "-lc", "serve"]}),
        (
            "FROM a\nRUN x\nHEALTHCHECK --interval=5s --retries=4 CMD curl -f http://localhost/ || exit 1\nUSER 65534\n"
            "STOPSIGNAL SIGINT\nVOLUME /data\n",
            {
                "healthcheck": {
                    "test": ["CMD-SHELL", "curl -f http://localhost/ || exit 1"],
                    "interval": "5s",
                    "retries": 4,
                },
                "user": "65534",
                "stop_signal": "SIGINT",
                "volumes": ["/data"],
            },
        ),
        ("FROM a\nRUN x\nHEALTHCHECK NONE\n", {"healthcheck": {"test": ["NONE"]}}),
        ("FROM a\nRUN x\nCOPY f /f\n", {}),
    ],
)
def test_the_plan_records_what_the_suffix_sets_in_the_image_config(dockerfile: str, config: dict, tmp_path: Path):
    (tmp_path / "f").write_text("f")
    path = tmp_path / "Dockerfile"
    path.write_text(dockerfile)

    assert dict(plan_dockerfile(path, tmp_path).config) == config


def test_a_whole_plan_records_what_the_last_stage_sets_over_its_base(tmp_path: Path):
    """A compose service's image is built whole: its config is what its last stage sets over its base image,
    stages it is built FROM included."""

    (tmp_path / "server.py").write_text("print(1)\n")
    path = tmp_path / "Dockerfile"
    path.write_text(
        'FROM python:3.12 AS base\nCMD ["python"]\nEXPOSE 1\n'
        'FROM base\nCOPY server.py .\nENTRYPOINT ["python", "server.py"]\n'
    )

    whole = plan_dockerfile(path, tmp_path, whole=True)

    assert whole.deferred == () and whole.whole
    assert whole.config == {"cmd": [], "expose": ["1/tcp"], "entrypoint": ["python", "server.py"]}
    # Over a base image's ENTRYPOINT and CMD, what the plan sets wins: the service runs its server.
    entrypoint = whole.config.get("entrypoint", ["/base-entrypoint"])
    assert resolve_argv(image_entrypoint=entrypoint, image_cmd=whole.config.get("cmd", ["base-cmd"])) == [
        "python",
        "server.py",
    ]


def test_image_config_reads_variables_in_user_and_expose():
    parsed = parse_dockerfile("FROM a\nARG PORT=8080\nENV ME=worker\nUSER $ME\nEXPOSE ${PORT}/udp\n")

    assert image_config(list(variables_in_scope(parsed.instructions))) == {"user": "worker", "expose": ["8080/udp"]}


@pytest.mark.parametrize(
    ("argv", "idle"),
    [
        (["sh", "-c", "sleep infinity"], True),
        (["sleep", "infinity"], True),
        (["/bin/bash", "-c", "exec sleep infinity"], True),
        (["tail", "-f", "/dev/null"], True),
        (["sh", "-c", "tail -f /dev/null"], True),
        (["/entry.sh", "sh", "-c", "sleep infinity"], False),  # the image's ENTRYPOINT runs first
        (["sh", "-c", "setup && sleep infinity"], False),
        (["python3", "serve.py"], False),
    ],
)
def test_main_idles_only_on_a_keep_alive_command(argv: list[str], idle: bool):
    assert is_idle(argv) is idle


def test_mains_init_is_compose_command_under_the_images_entrypoint():
    assert MainService().init(None) is None  # Harbor's base command: main idles
    assert MainService().init(["/entry.sh"]) == ["/entry.sh", "sh", "-c", "sleep infinity"]
    assert MainService(entrypoint=()).init(["/entry.sh"]) is None  # entrypoint: [] clears the image's
    assert MainService(command=("/app/run.sh",)).init(None) == ["/app/run.sh"]


# --- Harbor's adapters and examples (copies of Harbor's files) ---------------------------------------------------

RESOURCES = "[environment]\ncpus = 2\nmemory_mb = 4096\n"
# adapter -> (kind as written, None when the subset runs it else the refusal, the files its contexts and binds need)
ADAPTERS: dict[str, tuple[str, str | None, dict[str, str]]] = {
    "clbench": ("main_only", None, {"environment/messages/0.json": "{}"}),
    "cooperbench": (
        "sidecars",
        None,
        {"environment/agent1/Dockerfile": STUB_DOCKERFILE, "environment/agent2/Dockerfile": STUB_DOCKERFILE},
    ),
    "cybergym": (
        "sidecars",
        "main.cap_add: adds NET_ADMIN, which a service's root does not hold",
        {"environment/task-server/Dockerfile": STUB_DOCKERFILE},
    ),
    "financeagent": ("main_only", None, {}),
    "frontier-cs-algorithm": ("sidecars", "judge.privileged: gives the service every privilege of the host", {}),
    "gaia2": (
        "sidecars",
        None,
        {"environment/are-sidecar/Dockerfile": STUB_DOCKERFILE, "environment/scenario.json": "{}"},
    ),
    "kumo": ("sidecars", None, {"environment/Dockerfile_verifier": STUB_DOCKERFILE}),
    "ml_dev_bench": ("main_only", None, {}),
    "osworld": ("main_only", "volumes.osworld-cache.external: names a volume outside the stack", {}),
    "swelancer": ("main_only", None, {}),
    "tau3-bench": ("sidecars", None, {"environment/runtime-server/Dockerfile": STUB_DOCKERFILE}),
    "textarena": ("sidecars", None, {"environment/Dockerfile.game": STUB_DOCKERFILE}),
}


def _adapter_task(root: Path, adapter: str) -> Path:
    _, _, files = ADAPTERS[adapter]
    return _task(root, adapter, (UPSTREAM / "adapters" / f"{adapter}.yaml").read_text(), toml=RESOURCES, files=files)


@pytest.mark.parametrize("adapter", sorted(ADAPTERS))
def test_harbor_adapter_templates_are_planned_or_refused_by_name(tmp_path: Path, adapter: str):
    kind, refusal, _ = ADAPTERS[adapter]
    task_dir = _adapter_task(tmp_path, adapter)

    if refusal is not None:
        with pytest.raises(ComposeError, match=refusal.replace("(", r"\(").replace(")", r"\)")):
            plan_compose(task_dir, cpus=2, memory_mb=4096)
        return
    plan = _plan(task_dir, cpus=2, memory_mb=4096)  # the task's [environment] resources: CPUS and MEMORY
    assert (plan.kind, plan.data["rule"]) == (kind, COMPOSE_RULE)
    assert json.loads(json.dumps(plan.to_json())) == plan.to_json()  # what a lock records


def test_ml_dev_bench_binds_the_task_with_its_solution_and_tests_masked(tmp_path: Path):
    plan = _plan(_adapter_task(tmp_path, "ml_dev_bench"), cpus=2, memory_mb=4096)
    main = plan.main()

    assert main is not None and not main.unsupported
    task_root, setup, verifier_logs, agent_logs = main.binds
    assert (task_root.task, task_root.target, task_root.read_only) == (".", "/opt/task-src", True)
    assert task_root.masks == ("solution", "tests")
    assert (setup.task, setup.missing) == ("setup_workspace", True)  # not in the task: an empty directory
    assert (verifier_logs.logs, agent_logs.logs) == ("verifier", "agent")
    assert main.volumes == ()  # the two anonymous volumes mask the bind: no directory of main's own
    assert main.working_dir == "/app"
    assert (main.cpus, main.memory_mb) == (2.0, 4096)  # ${CPUS:-4} and ${MEMORY:-16G} are the task's
    # Compose's $$ is the shell's $: the init's script reads its own variable.
    assert '"$item"' in main.command[2] and main.command[:2] == ("bash", "-c")
    assert set(plan.data["ignored"]) == {"main.stdin_open", "main.tty"}
    assert "TEST_DIR is not set: blank" in plan.data["notes"]


def test_hello_mcp_plans_its_mcp_server_with_a_whole_build_key(tmp_path: Path):
    task_dir = tmp_path / "hello-mcp"
    shutil.copytree(UPSTREAM / "examples" / "hello-mcp", task_dir)

    plan = _plan(task_dir)
    server = plan.services["mcp-server"]

    assert plan.kind == "sidecars"
    assert plan.services["main"]["depends_on"] == {"mcp-server": {"condition": "service_healthy"}}
    assert server["build"]["context"] == "environment/mcp-server"
    assert server["build"]["key"].startswith("sha256:")
    assert server["build"]["config"] == {"expose": ["8000/tcp"], "cmd": ["python", "server.py"]}
    assert server["expose"] == ["8000/tcp"]
    assert server["healthcheck"] == {
        "test": [
            "CMD",
            "python",
            "-c",
            "import socket; s=socket.create_connection(('localhost',8000),timeout=2); s.close()",
        ],
        "interval_s": 2.0,
        "timeout_s": 5.0,
        "start_period_s": 5.0,
        "start_interval_s": 5.0,
        "retries": 15,
    }
    # A fetched dataset's task, whose lock holds this plan, is a stack where a provider runs stacks (whose session
    # finds the sidecar's image in the image map); refused where none does, and in a dataset directory, which has no
    # image map for a sidecar's image.
    task = HarborTask.load(task_dir, compose=plan)
    assert task.is_stack and task.main is not None and task.main.unsupported == ()
    with pytest.raises(HarborTaskError, match="makes the environment a multi-container stack"):
        HarborTask.load(task_dir, compose=plan, stack_refusal="the docker provider runs no multi-container stacks")
    with pytest.raises(HarborTaskError, match="builds its image from environment/mcp-server/Dockerfile, and a dataset"):
        HarborTask.load(task_dir)


@pytest.mark.parametrize(
    ("example", "check"),
    [
        ("sidecar-artifacts", lambda plan: plan.services["api"]["build"]["context"] == "environment/api-server"),
        (
            "environment-env-multi",
            lambda plan: (
                plan.services["sidecar"]["command"][2].startswith("printf '%s' \"${HARBOR_STARTUP_ENV_TEST-unset}\"")
                and plan.data["volumes"] == {"startup-evidence": {}}
            ),
        ),
        (
            "compose-sidecar-controlled",
            lambda plan: (
                plan.services["helper"]["volumes"]
                == [{"type": "bind", "target": "/logs/artifacts", "read_only": False, "logs": "artifacts"}]
            ),
        ),
        ("compose-sidecar-uncontrolled-network", lambda plan: plan.services["helper"]["networks"] == {"default": {}}),
    ],
)
def test_harbor_example_stacks_are_within_the_subset(tmp_path: Path, example: str, check):
    task_dir = tmp_path / example
    shutil.copytree(UPSTREAM / "examples" / example, task_dir)

    plan = _plan(task_dir)

    assert plan.kind == "sidecars" and check(plan)


# --- the subset on Hub-shaped files (written for platoon, after published tasks) ---------------------------------

TB4_LIKE = """services:
  seeder:
    image: registry.example/tasks:seeder@sha256:0000000000000000000000000000000000000000000000000000000000000001
    depends_on:
      kafka: {condition: service_healthy}
    environment:
      KAFKA_BROKERS: kafka:9092
  main:
    depends_on:
      seeder: {condition: service_completed_successfully}
      kafka: {condition: service_healthy}
    healthcheck:
      test: ["CMD-SHELL", "test -S /tmp/supervisor.sock || exit 1"]
      interval: 2s
      timeout: 2s
      retries: 60
      start_period: 30s
    cap_add: [SYS_PTRACE]
  kafka:
    image: apache/kafka-native:4.3.1
    expose: ["9092"]
    healthcheck:
      test: ["CMD-SHELL", "nc -z localhost 9092 || exit 1"]
      interval: 3s
  probe:
    image: busybox:1.36
    network_mode: service:kafka
    command: sleep infinity
"""
LSCB_LIKE = """services:
  main:
    build:
      context: .
      dockerfile: Dockerfile
    command: ["sh", "-c", "sleep infinity"]
    environment:
      - DB_HOST=postgres
      - DB_PORT=5432
    depends_on:
      postgres:
        condition: service_healthy
  postgres:
    image: postgres:11
    environment:
      - POSTGRES_PASSWORD=password
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
"""
SETA_LIKE = """services:
  main:
    networks:
      lan: {ipv4_address: 172.30.0.10}
  db-internal:
    image: mysql:8
    hostname: db-internal
    networks:
      lan: {ipv4_address: 172.30.0.20}
networks:
  lan:
    ipam:
      config: [{subnet: 172.30.0.0/24}]
"""


def test_terminal_bench_style_stacks_are_within_the_subset(tmp_path: Path):
    plan = _plan(_task(tmp_path, "tb4", TB4_LIKE))
    services = plan.services

    assert plan.kind == "sidecars"
    assert services["main"]["depends_on"]["seeder"] == {"condition": "service_completed_successfully"}
    assert services["main"]["cap_add"] == ["SYS_PTRACE"]  # a service's root holds it over its own processes
    assert services["main"]["healthcheck"]["start_period_s"] == 30.0
    assert services["probe"]["network_mode"] == "service:kafka"
    assert services["probe"]["command"] == ["sleep", "infinity"]
    assert services["seeder"]["image"].endswith("@sha256:" + "0" * 63 + "1")


def test_database_sidecar_stacks_are_within_the_subset(tmp_path: Path):
    plan = _plan(_task(tmp_path, "lscb", LSCB_LIKE))

    assert plan.kind == "sidecars" and plan.services["postgres"]["image"] == "postgres:11"
    assert plan.services["main"]["environment"] == {"DB_HOST": "postgres", "DB_PORT": "5432"}


def test_healthcheck_durations_of_0_are_docker_s_defaults(tmp_path: Path):
    """Docker reads 0 as its default for a healthcheck's interval, timeout, start interval and retries (a start period
    of 0 is its default), so a check of 0s neither times out at once nor runs without a pause."""

    compose = (
        "services:\n  main:\n    depends_on:\n      db:\n        condition: service_healthy\n"
        "  db:\n    image: postgres:16\n    command: [postgres]\n    healthcheck:\n"
        "      test: [CMD, pg_isready]\n      interval: 0s\n      timeout: 0s\n      start_interval: 0s\n"
        "      start_period: 0s\n      retries: 0\n"
    )
    check = _plan(_task(tmp_path, "zeros", compose)).services["db"]["healthcheck"]
    assert check == {
        "test": ["CMD", "pg_isready"],
        "interval_s": 30.0,
        "timeout_s": 30.0,
        "start_period_s": 0.0,
        "start_interval_s": 5.0,
        "retries": 3,
    }
    recorded = {**check, "interval_s": 0.0, "timeout_s": 0.0, "start_interval_s": 0.0, "retries": 0}  # an older lock's
    assert ServiceHealthcheck.from_json(recorded) == ServiceHealthcheck.from_json(check)


def test_static_addresses_are_refused(tmp_path: Path):
    with pytest.raises(ComposeError, match=r"networks.lan.ipam: is outside the compose subset"):
        plan_compose(_task(tmp_path, "seta", SETA_LIKE))


@pytest.mark.parametrize(
    ("compose", "message"),
    [
        ("services:\n  main:\n    user: '1000'\n", "main.user: runs the service as another user"),
        ("services:\n  main:\n    sysctls: [net.ipv4.ip_forward=1]\n", "main.sysctls"),
        ("services:\n  main:\n    extra_hosts: ['pypi.org:0.0.0.0']\n", "main.extra_hosts: adds host names"),
        ("services:\n  main:\n    platform: linux/s390x\n", "main.platform: asks for 'linux/s390x'"),
        ("services:\n  main:\n    weird_key: 1\n", "main.weird_key: is outside the compose subset"),
        ("services:\n  main:\n    network_mode: host\n", "main.network_mode: 'host'"),
        ("services:\n  main:\n    cap_add: [NET_RAW]\n", "main.cap_add: adds NET_RAW"),
        (
            "services:\n  main:\n    deploy: {resources: {reservations: {devices: [{capabilities: [gpu]}]}}}\n",
            "asks for GPUs",
        ),
        ("services:\n  main:\n    profiles: [debug]\n", "main.profiles: would keep main from starting"),
        ("services:\n  main:\n    depends_on: [db]\n", "depends on 'db', which is not started"),
        ("services:\n  main:\n    volumes: [data:/data]\n", "names the volume 'data', which the file does not declare"),
        ("services:\n  main: {}\nsecrets: {token: {file: ./t}}\n", "secrets: is a top-level key outside"),
        (
            "services:\n  main:\n    networks: [front]\n  db:\n    image: db\n    networks: [back]\n"
            "networks: {front: {}, back: {}}\n",
            "db.networks: keeps the service from main",
        ),
        (
            "services:\n  main:\n    networks: [front]\n  db:\n    image: db\n    networks: [front, egress]\n"
            "networks: {front: {internal: true}, egress: {}}\n",
            "mixes internal and other networks",
        ),
        ("services:\n  main:\n    healthcheck: {test: [CMD], interval: 1x}\n", "main.healthcheck"),
        (
            "services:\n  main:\n    expose: [8000]\n  api:\n    image: api\n    ports: ['18000:8000']\n",
            "api.expose: declares 8000/tcp, which main declares too",
        ),
    ],
)
def test_keys_outside_the_subset_are_refused_by_service_and_key(tmp_path: Path, compose: str, message: str):
    with pytest.raises(ComposeError, match=message):
        plan_compose(_task(tmp_path, "refused", compose))


def test_yaml_scalars_are_read_as_compose_reads_them(tmp_path: Path):
    """Compose reads YAML 1.2: ``no`` and ``on`` are strings, ``22:22`` a port pair (YAML 1.1's base-60 number)."""

    task_dir = _task(
        tmp_path,
        "yaml12",
        "services:\n  main:\n    environment: {FLAG: no, ON_SWITCH: on, RATIO: 0.5, REAL: true}\n"
        "    ports: [22:22]\n    restart: no\n",
    )

    main = _plan(task_dir).services["main"]

    assert main["environment"] == {"FLAG": "no", "ON_SWITCH": "on", "RATIO": "0.5", "REAL": "true"}
    assert main["expose"] == ["22/tcp"]


def test_aliases_that_expand_a_file_beyond_what_planning_walks_are_refused(tmp_path: Path):
    """A few anchors can stand for billions of values ("billion laughs"): refused when the file is read, before
    planning walks them; an anchor that holds an alias of itself, and nesting too deep to read, too."""

    lines = ["x-a: &a [x, x, x, x, x, x, x, x, x, x]"]
    lines += [f"x-{level}: &{level} [{', '.join([f'*{previous}'] * 10)}]" for previous, level in zip("abcde", "bcdef")]
    bomb = _task(tmp_path, "bomb", "\n".join(lines) + "\nservices:\n  main:\n    command: [echo, *f]\n")
    cycle = _task(tmp_path, "cycle", "x-loop: &loop [*loop]\nservices:\n  main: {}\n")
    deep = _task(tmp_path, "deep", "services:\n  main:\n    labels: " + "[" * 2000 + "]" * 2000 + "\n")
    shared = _task(  # merges and aliases a real file uses stay far below the bound
        tmp_path, "shared", "x-env: &env {A: '1', B: '2'}\nservices:\n  main:\n    environment: *env\n"
    )

    with pytest.raises(ComposeError, match=f"expand it to more than {MAX_DOCUMENT_VALUES:,} values"):
        plan_compose(bomb)
    with pytest.raises(ComposeError, match="an anchor that holds an alias of itself"):
        plan_compose(cycle)
    with pytest.raises(ComposeError, match="nests its values too deeply"):
        plan_compose(deep)
    assert _plan(shared).services["main"]["environment"] == {"A": "1", "B": "2"}


def test_env_files_are_read_as_compose_reads_them(tmp_path: Path):
    """compose-go's dotenv: comments, quotes (single ones literal, double ones with escapes), export, a name alone,
    interpolation with compose's variables and the file's own."""

    text = (
        "\ufeff# a comment\r\n"
        "PLAIN=three # a comment\n"
        "HASH=a#b\n"
        "export EXPORTED = spaced  \n"
        "SINGLE='${PLAIN} \\n stays'\n"
        'DOUBLE="${PLAIN}\\tthen\\nnext \\"quoted\\" \\$PLAIN"\n'
        'MULTI="one\ntwo" # after the quote\n'
        "YAML: style\n"
        "FROM_TASK\n"
        "UNKNOWN\n"
        "EMPTY=\n"
        "LATER=${PLAIN:-unset}-${FROM_TASK}\n"
    )
    task_dir = _task(
        tmp_path, "dotenv", "services:\n  main:\n    env_file: [vars.env]\n", files={"environment/vars.env": text}
    )

    environment = _plan(task_dir, env={"FROM_TASK": "task"}).services["main"]["environment"]

    assert environment == {
        "PLAIN": "three",
        "HASH": "a#b",
        "EXPORTED": "spaced",
        "SINGLE": "${PLAIN} \\n stays",
        "DOUBLE": 'three\tthen\nnext "quoted" $PLAIN',
        "MULTI": "one\ntwo",
        "YAML": "style",
        "FROM_TASK": "task",
        "EMPTY": "",
        "LATER": "three-task",
    }


@pytest.mark.parametrize(
    ("text", "message"),
    [("A='unterminated\n", "its quoted value has no closing '"), ("A B=1\n", "cannot read the line 'A B=1'")],
)
def test_env_files_compose_cannot_read_are_refused(tmp_path: Path, text: str, message: str):
    task_dir = _task(
        tmp_path, "dotenv", "services:\n  main:\n    env_file: [vars.env]\n", files={"environment/vars.env": text}
    )

    with pytest.raises(ComposeError, match=message):
        plan_compose(task_dir)


def test_a_lock_whose_plans_follow_another_compose_rule_is_refused(tmp_path: Path):
    """A change of Harbor's layering is a rule bump: plans made by another rule are not used, while a lock without
    plans (as platoon 0.4.0 wrote them) reads as before."""

    plan = {"rule": "harbor-older", "kind": "main_only", "services": {"main": {"command": ["serve"]}}}
    task = LockedTask(name="t", digest="sha256:0", path="tasks/0", flags={"stack": "main_only"}, compose=plan)
    for rule, tasks, readable in (
        ("harbor-older", (task,), False),
        (COMPOSE_RULE, (task,), True),
        ("", (LockedTask(name="t", digest="sha256:0", path="tasks/0"),), True),
    ):
        path = tmp_path / f"lock-{rule or 'none'}.json"
        write_lock(DatasetLock("dir://x", "x", "local", tasks, compose_rule=rule), path)
        if readable:
            assert read_lock(path).tasks == tasks
        else:
            with pytest.raises(LockError, match="by rule 'harbor-older'; this platoon plans them by .*fetch the"):
                read_lock(path)


def test_ignored_keys_profiles_and_published_ports_are_recorded(tmp_path: Path):
    task_dir = _task(
        tmp_path,
        "recorded",
        """services:
  main:
    container_name: fixed
    restart: always
    pull_policy: missing
    ports: ["18806:8765", "127.0.0.1:9000:9000/udp"]
    expose: [8000]
  debug:
    image: busybox
    profiles: [debug]
""",
    )

    plan = _plan(task_dir)

    assert plan.kind == "main_only"  # the profiled service never starts
    assert set(plan.services) == {"main"}
    assert plan.services["main"]["expose"] == ["8000/tcp", "8765/tcp", "9000/udp"]
    assert plan.data["ignored"] == ["main.container_name", "main.restart", "main.pull_policy"]
    assert any("debug has profiles" in note for note in plan.data["notes"])
    assert any("kept as expose" in note for note in plan.data["notes"])


def test_a_main_only_file_that_one_sandbox_cannot_apply_is_refused_when_the_task_loads(tmp_path: Path):
    task_dir = _task(tmp_path, "shm", "services:\n  main:\n    shm_size: 1gb\n    cap_add: [SYS_PTRACE]\n")

    assert _plan(task_dir).kind == "main_only"  # within the subset: stacks will apply them
    with pytest.raises(HarborTaskError, match=r"gives main cap_add, shm_size, which platoon does not apply"):
        HarborTask.load(task_dir)


# --- overlays edit compose services ---------------------------------------------------------------------------------


def _overlay(tmp_path: Path, text: str, name: str = "fix") -> Path:
    directory = tmp_path / "overlays" / name
    directory.mkdir(parents=True)
    (directory / "overlay.toml").write_text(text)
    return directory


def test_an_overlay_drops_a_capability_and_sets_keys(tmp_path: Path):
    edits = (
        ComposeEdit.parse("fix", {"service": "main", "unset": ["cap_add"], "set": {"environment": {"MODE": "eval"}}}),
        ComposeEdit.parse("fix", {"service": "task-server", "replace": {"command": ["serve", "--fast"]}}),
    )
    task_dir = _adapter_task(tmp_path, "cybergym")

    plan = _plan(task_dir, edits=edits)

    assert "cap_add" not in plan.services["main"]
    assert plan.services["task-server"]["command"] == ["serve", "--fast"]
    assert plan.data["edits"] == ["fix: main: set environment; unset cap_add", "fix: task-server: replace command"]


def test_an_overlay_marks_a_service_external(tmp_path: Path):
    edit = ComposeEdit.parse("pool", {"service": "judge", "external": "judge-pool"})
    task_dir = _adapter_task(tmp_path, "frontier-cs-algorithm")

    plan = _plan(task_dir, edits=[edit])

    assert plan.services["judge"] == {"external": "judge-pool"}  # served by a node service, never started here


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"service": "main"}, "give set, replace, unset or external"),
        ({"service": "main", "unset": "cap_add"}, "unset must list keys"),
        ({"service": "bad name!", "unset": ["x"]}, "needs a service name"),
        ({"service": "main", "rename": {}}, "unknown keys"),
    ],
)
def test_compose_edits_are_checked_when_an_overlay_loads(entry, message: str):
    with pytest.raises(ValueError, match=message):
        ComposeEdit.parse("fix", entry)


def test_a_fetch_applies_compose_overlays_and_records_them(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _adapter_task(dataset, "cybergym")
    _task(dataset, "plain", None)
    overlay = _overlay(
        tmp_path,
        """compose_host_env = ["CYBERGYM_TOKEN"]
[match]
name = "cybergym"
[[compose]]
service = "main"
unset = ["cap_add"]
[[compose]]
service = "task-server"
set = {environment = {TOKEN = "${CYBERGYM_TOKEN}"}}
""",
    )
    assert Overlay.load(overlay).compose_host_env == ("CYBERGYM_TOKEN",)

    plain = fetch(fetch_config(str(dataset), cache=tmp_path / "cache", lock=tmp_path / "plain.json"))
    fixed = fetch(
        fetch_config(str(dataset), cache=tmp_path / "cache", lock=tmp_path / "fixed.json", overlays=[overlay])
    )

    assert (plain.get("cybergym").refused or "").startswith("compose: main.cap_add: adds NET_ADMIN")
    task = fixed.get("cybergym")
    assert (task.refused or "").startswith("stack: ")  # within the subset now; sidecar stacks are refused by default
    assert task.compose is not None
    assert task.overlays == ("fix",) and task.compose["edits"] == [
        "fix: main: unset cap_add",
        "fix: task-server: set environment",
    ]
    assert task.compose["deferred"] == ["host:CYBERGYM_TOKEN"]
    assert fixed.get("plain").compose is None and fixed.get("plain").refused is None


def test_an_overlay_that_edits_a_missing_service_refuses_the_task(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _task(dataset, "lonely", "services:\n  main:\n    environment: [A=1]\n")
    overlay = _overlay(tmp_path, '[[compose]]\nservice = "judge"\nunset = ["privileged"]\n')

    lock = fetch(fetch_config(str(dataset), cache=tmp_path / "cache", lock=tmp_path / "l.json", overlays=[overlay]))

    assert lock.get("lonely").refused == "overlay: judge: overlay fix edits a service the compose file lacks"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('compose_host_env = ["BAD-NAME"]\n', "compose_host_env must list variable names"),
        ("[compose]\nservice = 'main'\n", r"\[\[compose\]\] must be an array of tables"),
        ('[[compose]]\nservice = "main"\n', "give set, replace, unset or external"),
    ],
)
def test_bad_compose_overlays_fail_the_fetch(tmp_path: Path, text: str, message: str):
    with pytest.raises(FetchError, match=message):
        Overlay.load(_overlay(tmp_path, text))


# --- build keys of compose services --------------------------------------------------------------------------------


def _context(root: Path, server: str, dockerfile: str = STUB_DOCKERFILE) -> Path:
    root.mkdir(parents=True)
    (root / "Dockerfile").write_text(dockerfile)
    (root / "server.py").write_text(server)
    return root


def test_services_that_differ_only_in_files_their_suffix_copies_never_share_an_image(tmp_path: Path):
    """A prefix key hashes the prefix only, so two contexts whose Dockerfiles end with ``COPY server.py`` share one;
    a service's whole key hashes what the suffix copies too."""

    one = _context(tmp_path / "one", "print(1)\n", "FROM python:3.12-slim\nRUN true\nCOPY server.py .\n")
    two = _context(tmp_path / "two", "print(2)\n", "FROM python:3.12-slim\nRUN true\nCOPY server.py .\n")

    assert plan_dockerfile(one / "Dockerfile", one).key == plan_dockerfile(two / "Dockerfile", two).key
    assert (
        plan_dockerfile(one / "Dockerfile", one, whole=True).key
        != plan_dockerfile(two / "Dockerfile", two, whole=True).key
    )


def test_build_arguments_override_arg_defaults_and_enter_the_key(tmp_path: Path):
    context = _context(
        tmp_path / "ctx", "print(1)\n", "FROM python:3.12-slim\nARG ROLE=val\nRUN echo $ROLE\nCOPY server.py .\n"
    )
    dockerfile = context / "Dockerfile"

    default = plan_dockerfile(dockerfile, context, whole=True)
    same = plan_dockerfile(dockerfile, context, whole=True, build_args={"ROLE": "val"})
    other = plan_dockerfile(dockerfile, context, whole=True, build_args={"ROLE": "test"})

    assert len({default.key, same.key, other.key}) == 3  # given arguments are part of the key
    assert plan_dockerfile(dockerfile, context).key == plan_dockerfile(dockerfile, context).key


def test_a_service_build_context_is_planned_from_the_compose_file(tmp_path: Path):
    task_dir = _task(
        tmp_path,
        "grader",
        """services:
  main:
    depends_on: [grader]
  grader:
    build:
      context: ./grader-ctx
      dockerfile: Dockerfile.grader
      args: {ROLE: test}
""",
        files={"environment/grader-ctx/Dockerfile.grader": "FROM python:3.12\nARG ROLE=val\nRUN echo $ROLE\n"},
    )

    grader = _plan(task_dir).services["grader"]["build"]

    assert (grader["context"], grader["dockerfile"], grader["args"]) == (
        "environment/grader-ctx",
        "Dockerfile.grader",
        {"ROLE": "test"},
    )
    assert grader["base"] == ["python:3.12"]
    assert _plan(task_dir, with_keys=False).services["grader"]["build"].get("key") is None


# --- Harbor's file-name rule, the lock, input files --------------------------------------------------------------


def test_a_fetch_locks_each_kind_with_its_plan_and_warns_of_compose_files_harbor_does_not_read(tmp_path: Path, caplog):
    dataset = tmp_path / "dataset"
    _task(dataset, "single", None)
    _task(dataset, "trivial", "services:\n  main: {}\n")
    _task(dataset, "main-only", "services:\n  main:\n    environment: [A=1]\n    command: [sleep, '1000']\n")
    _task(dataset, "sidecars", LSCB_LIKE)
    _task(dataset, "unread", None, files={"environment/compose.yaml": LSCB_LIKE})

    with caplog.at_level(logging.WARNING, logger="platoon.envs.harbor.fetch"):
        lock = fetch(fetch_config(str(dataset), cache=tmp_path / "cache", lock=tmp_path / "l.json"))

    kinds = {task.name: task.flags["stack"] for task in lock.tasks}
    assert kinds == {
        "main-only": "main_only",
        "sidecars": "sidecars",
        "single": "single",
        "trivial": "trivial",
        "unread": "single",
    }
    refused = {task.name: (task.refused or "").split(":")[0] for task in lock.tasks}
    assert refused == {"main-only": "", "sidecars": "stack", "single": "", "trivial": "", "unread": ""}
    assert lock.compose_rule == COMPOSE_RULE
    assert lock.get("unread").flags["unread_compose"] == ["environment/compose.yaml"]
    assert (
        "1 Harbor task(s) of" in caplog.text and "environment/compose.yaml, which Harbor does not read" in caplog.text
    )
    # The lock round-trips the plans, and a task loads from its lock's plan as from its files.
    reread = read_lock(tmp_path / "l.json")
    assert reread.get("main-only").compose == lock.get("main-only").compose
    assert reread.get("single").compose is None and (reread.get("trivial").compose or {}).get("kind") == "trivial"
    cached = tmp_path / "cache" / lock.get("main-only").path
    from_lock = HarborTask.load(cached, compose=reread.get("main-only").compose)
    from_files = HarborTask.load(cached)
    assert from_lock.main is not None and from_lock.main == from_files.main
    assert from_lock.main.command == ("sleep", "1000")


def test_files_that_main_binds_are_input_files_but_what_its_volumes_mask(tmp_path: Path):
    task_dir = _task(
        tmp_path,
        "binds",
        """services:
  main:
    volumes:
      - ${CONTEXT_DIR}/..:/opt/task-src:ro
      - /opt/task-src/solution
      - ./more:/more:ro
""",
        toml='[environment]\ndocker_image = "ubuntu:24.04"\n',
        files={"environment/more/m.txt": "m", "solution/solve.sh": "secret", "environment/data/d.txt": "d"},
    )

    files = input_files(task_dir, compose=HarborTask.load(task_dir).compose)

    assert "solution/solve.sh" not in files  # masked
    assert set(files) == {"environment/more/m.txt", "environment/data/d.txt"}  # never the task's definition or build
    assert input_files(task_dir) == {}  # without its binds a prebuilt image with a compose file takes nothing


def test_tasks_of_a_template_that_bind_themselves_whole_stay_apart_in_a_hold_out(tmp_path: Path):
    """ml_dev_bench's tasks bind themselves whole: their task.toml, instruction, Dockerfile, compose file and tests
    are one template's, byte for byte, and must not join every task into one group."""

    rows = []
    for name in ("first", "second"):
        task_dir = _adapter_task(tmp_path / name, "ml_dev_bench")
        (task_dir / "environment" / "data.csv").write_text(f"{name}\n")
        files = input_files(task_dir, compose=HarborTask.load(task_dir).compose)
        assert set(files) == {"environment/data.csv"}
        rows.append(HoldoutRow(name, {"group": name}, input_files=files))

    split = hold_out(rows, HoldoutRule(fraction=0.5, group_by=("group",)))

    assert split.groups is not None and (split.groups.groups, split.groups.joined_groups) == (2, 0)
    assert len(split.eval) == 1


# --- main-only tasks on the process provider -----------------------------------------------------------------------


@pytest.fixture
def provider(tmp_path: Path) -> ProcessProvider:
    return ProcessProvider(root=tmp_path / "sandboxes")


@pytest.fixture
def service(provider: ProcessProvider) -> HarborEnvironmentService:
    return HarborEnvironmentService(TASKS, provider=provider)


async def _open(service: HarborEnvironmentService, task_id: str) -> HarborSession:
    return await service.open(await service.get_task(task_id), rollout_id="rollout-1")


@pytest.mark.asyncio
async def test_main_only_init_serves_through_grading_so_the_oracle_scores_and_a_no_op_does_not(
    service: HarborEnvironmentService, provider: ProcessProvider
):
    session = await _open(service, "main-only-serve")
    try:
        assert isinstance(session.sandbox, ProcessSandbox)
        assert session.sandbox.spec.env["GREETING"] == "hello from compose"
        unsolved = await session.grade()
        assert (unsolved.status, unsolved.reward) == ("scored", 0.0)
        assert "the service answered: hello from compose 42" in unsolved.metadata["test_output"]

        solved = await session.run_solution()
        assert solved.exit_code == 0, solved.stderr
        # Something an agent left running dies before the tests; the init does not.
        await session.sandbox.exec("sleep 300 >/dev/null 2>&1 &")
        outcome = await session.grade()
        assert (outcome.status, outcome.reward, outcome.metadata["stack"]) == ("scored", 1.0, "main_only")
        with pytest.raises(NotImplementedError, match="copy fork restores files, not processes"):
            await session.fork("copy")
    finally:
        await session.close()
    assert list(provider.root.iterdir()) == []


@pytest.mark.asyncio
async def test_an_init_that_exits_before_grading_is_an_infrastructure_error(service: HarborEnvironmentService):
    session = await _open(service, "main-only-serve")
    try:
        init = session._lease.init
        assert init is not None and session.sandbox is not None
        await session.sandbox.exec(["kill", "-9", init.split(":")[0]])
        await asyncio.sleep(0.2)

        outcome = await session.grade()

        assert (outcome.status, outcome.failure_kind) == ("error", "main_exited")
        assert not outcome.trainable
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_masked_binds_hide_the_solution_and_tests_and_writable_binds_are_copies(
    service: HarborEnvironmentService, provider: ProcessProvider
):
    session = await _open(service, "main-only-masks")
    try:
        assert isinstance(session.sandbox, ProcessSandbox)
        spec = session.sandbox.spec
        assert (spec.workdir, spec.cpus, spec.memory_mb) == ("/app", 2.0, 1024)
        assert spec.env["INPUTS"] == "/opt/task-src/environment/data"
        noop = await session.grade()
        assert (noop.status, noop.reward) == ("scored", 0.0)
        assert "output.txt is wrong" in noop.metadata["test_output"]

        assert (await session.run_solution()).exit_code == 0
        outcome = await session.grade()
        assert (outcome.status, outcome.reward) == ("scored", 1.0), outcome.metadata.get("test_output")
        copies = list(session._lease.scratch)
        assert copies and all(path.is_dir() for path in copies)
    finally:
        await session.close()
    assert not any(path.exists() for path in copies)  # the bind copies go with the sandbox
    assert not (TASKS / "main-only-masks" / "environment" / "scratch" / "note.txt").exists()  # never the task's files
    assert list(provider.root.iterdir()) == []


@pytest.mark.asyncio
async def test_a_dataset_of_main_only_tasks_lists_them_both(service: HarborEnvironmentService):
    assert sorted(await service.list_tasks("train")) == ["main-only-masks", "main-only-serve"]
    task = service.harbor_task(await service.get_task("main-only-serve"))
    assert task.compose is not None and task.compose.kind == "main_only"
    assert task.main is not None and task.main.init(None) == ["python3", "/opt/data/serve.py"]
    assert service.rollout_overhead_s("main-only-serve") > service.sandbox_start_budget_s
