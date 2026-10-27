"""Harbor stacks as hpc-sandbox pods on a real runtime, end to end (RFC 0010 M7's checks): the stack fixtures and
Harbor's hello-mcp score their oracles and their no-ops, names resolve, the relay reaches a host socket, the host
reaches a service's port, the tool plane serves the tasks' MCP servers through the sockets the pods expose (a function
harness solves hello-mcp, ``mcp-http`` and ``mcp-sse`` through it, and an agent's access level hides what it should),
a CLI harness in ``main`` solves them at the URLs that ``{task_mcp_servers}`` gives it,
one stack cannot reach another's services, a kill of every process in main grades as in a
single sandbox, database images switch users in their sidecars, a judge's record that only its own user reads reaches
the tests, a published task's main reaches its judge sidecar and shares its logs, and nothing is left behind, also
after stacks cancelled while they start.  With an hpc-sandbox older than 0.5, stacks are refused instead, naming 0.5.

Skipped unless ``PLATOON_TEST_PODS=1`` on a compute node with the runtime of ``PLATOON_TEST_PODS_PROVIDER``
(``singularity``, the default, with SingularityCE or Apptainer on ``PATH``, or ``enroot``; ``PLATOON_TEST_PODS_BINARY``
names the executable, such as ``apptainer``) and hpc-sandbox; the rollouts that solve the MCP tasks go through platoon's
model gateway, which needs the ``tinker`` extra.  Never run it on a login node: it starts containers.  The images come
from the network: the nodes pull ``docker://`` references, the registry gives the configurations that pins record,
hello-mcp's and kumo's images are built (the oci builder, or def on Apptainer; enroot, which builds none, skips them),
hello-mcp's tests install their dependencies, and kumo's task is downloaded from GitHub.  ``HPC_SANDBOX_SCRATCH``
(node-local disk) and ``HPC_SANDBOX_IMAGE_CACHE`` say where sandboxes and images go; ``PLATOON_TEST_PODS_STACKS`` and
``PLATOON_TEST_PODS_CONCURRENCY`` size the test of many stacks (20 and 8).  Sidecars run with your subordinate ids where
your user has them (``getsubids $USER``), else as your own uid: the checks of their users follow the run mode each got.
Apptainer and enroot pods are experimental: this module is what a site runs to check them.
"""

from __future__ import annotations

import asyncio
import os
import random
import shutil
import statistics
import tempfile
import time
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from mcp_solver import DIRECT, ROTATE_TOOL, SECRET_TOOL, SOLVER, hidden_from_read_only, solved, solved_directly
from no_model import no_model_gateway
from site_cache import use_site_cache

from platoon.config_defs import RolloutConfig
from platoon.envs.harbor import HarborEnvironmentService, HarborSession
from platoon.envs.harbor.builders import build_images, pin_images, provider_builder
from platoon.envs.harbor.fetch import fetch, fetch_config
from platoon.envs.harbor.images import default_image_dir, default_image_map
from platoon.envs.service import EnvSetupError, SessionNeeds
from platoon.harness.rollout import run_harness_rollout
from platoon.sandbox import Sandbox, cleanup, stack_refusal
from platoon.task import Task
from platoon.train.components import EnvironmentRoute

PROVIDER = os.environ.get("PLATOON_TEST_PODS_PROVIDER", "singularity")
BINARY = os.environ.get("PLATOON_TEST_PODS_BINARY")
RUNTIMES = {"singularity": ("singularity", "apptainer"), "enroot": ("enroot",)}
FIXTURES = Path(__file__).resolve().parents[1] / "envs" / "fixtures"
STACKS = FIXTURES / "harbor-stacks" / "tasks"
IMAGES = FIXTURES / "harbor-stacks-images" / "tasks"
HELLO_MCP = FIXTURES / "harbor-compose" / "upstream" / "examples" / "hello-mcp"
# A task of kumo's in laude-institute/harbor-datasets, file by file at a commit.
KUMO = (
    "https://raw.githubusercontent.com/laude-institute/harbor-datasets/eff0de1ca438bcf6e958fbd6e5f87b00814c8118/"
    "datasets/kumo"
)
KUMO_TASK = "chemicalenv-t12-a16-v1-s0"
KUMO_FILES = (
    "task.toml",
    "instruction.md",
    "environment/docker-compose.yaml",
    "environment/Dockerfile",
    "environment/Dockerfile_verifier",
    "environment/kumo_cli.py",
    "environment/kumo_verifier.py",
    "environment/secret.json",
    "environment/knowledge_book.txt",
    "environment/system_prompt.txt",
    "solution/solve.sh",
    "solution/oracle_truth.txt",
    "solution/oracle_action_count.txt",
    "tests/test.sh",
    "tests/score.py",
    "tests/expected.json",
)
# hello-mcp's tests' own uv, installed in main's image at its build (the overlay of the hello-mcp test).
UV_IN_MAIN = r"""
description = "Install the uv that hello-mcp's tests run in main's image"
[[edit]]
file = "environment/Dockerfile"
pattern = '^WORKDIR /app$'
replace = 'RUN curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh\nWORKDIR /app'
count = 1
"""
COUNT = int(os.environ.get("PLATOON_TEST_PODS_STACKS", "20"))
CONCURRENCY = int(os.environ.get("PLATOON_TEST_PODS_CONCURRENCY", "8"))
LOCK_STACKS: dict[str, bool] = {}  # a fetch's default, which locks stacks as runnable
FETCH = (
    "import sys, urllib.request\n"
    "try:\n"
    "    print(urllib.request.urlopen(sys.argv[1], timeout=10).read().decode())\n"
    "except Exception as exc:\n"
    "    print(f'nothing ({exc})')\n"
)
PUT = (
    "import sys, urllib.request\n"
    "urllib.request.urlopen(urllib.request.Request(sys.argv[1], sys.argv[2].encode(), method='PUT'), timeout=10)\n"
)
# Waits up to 60 s for a URL to answer (a service without a healthcheck starts its dependents before it listens).
AWAIT = (
    "import sys, time, urllib.request\n"
    "deadline = time.monotonic() + 60\n"
    "while True:\n"
    "    try:\n"
    "        urllib.request.urlopen(sys.argv[1], timeout=5)\n"
    "        break\n"
    "    except OSError:\n"
    "        if time.monotonic() > deadline:\n"
    "            raise\n"
    "        time.sleep(0.5)\n"
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(os.environ.get("PLATOON_TEST_PODS") != "1", reason="set PLATOON_TEST_PODS=1 to run pods"),
    pytest.mark.skipif(
        not any(shutil.which(binary) for binary in ((BINARY,) if BINARY else RUNTIMES.get(PROVIDER, ()))),
        reason=f"no runtime of the {PROVIDER} provider on PATH",
    ),
]


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The test's own hpc-sandbox scratch root (on node-local disk where ``HPC_SANDBOX_SCRATCH`` says), and node
    scratch for platoon's own copies."""

    root = Path(os.environ.get("HPC_SANDBOX_SCRATCH") or tmp_path) / f"platoon-pods-{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "node"))
    (tmp_path / "node").mkdir()
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def registered() -> set[int]:
    """What this process registered for cleanup before the test, attached or not."""

    return set(cleanup._live)


def _options(scratch: Path) -> dict[str, Any]:
    options: dict[str, Any] = {"scratch_root": str(scratch)}
    if PROVIDER == "singularity":
        options["allow_network_none"] = True  # a separate verifier's sandbox runs without a network
    if os.environ.get("HPC_SANDBOX_IMAGE_CACHE"):
        options["image_cache"] = os.environ["HPC_SANDBOX_IMAGE_CACHE"]
    if BINARY:
        options["binary"] = BINARY
    return options


def _refusal(scratch: Path) -> str | None:
    return stack_refusal(PROVIDER, _options(scratch))


def _skip_without_pods(scratch: Path) -> None:
    pytest.importorskip("hpc_sandbox")
    if (refusal := _refusal(scratch)) is not None:
        pytest.skip(f"no stacks here: {refusal}")


def _service(scratch: Path, **settings: Any) -> HarborEnvironmentService:
    _skip_without_pods(scratch)
    return HarborEnvironmentService(provider=PROVIDER, provider_options=_options(scratch), **settings)


async def _open(service: HarborEnvironmentService, task_id: str, *, needs: SessionNeeds | None = None) -> HarborSession:
    return await service.open(await service.get_task(task_id), rollout_id=f"pods-{uuid.uuid4().hex[:8]}", needs=needs)


def _main(session: HarborSession) -> Sandbox:
    assert session.sandbox is not None
    return session.sandbox


def _service_sandbox(session: HarborSession, name: str) -> Sandbox:
    assert session.stack is not None
    return session.stack.services[name]


async def _oracle_and_no_op(service: HarborEnvironmentService, task_id: str) -> dict[str, Any]:
    """The task's no-op grade and, in a session of its own, its oracle's; the oracle outcome's metadata."""

    session = await _open(service, task_id)
    try:
        noop = await session.grade()
    finally:
        await session.close()
    assert (noop.status, noop.reward) == ("scored", 0.0), (task_id, noop.status, noop.reason, noop.metadata)
    session = await _open(service, task_id)
    try:
        solved = await session.run_solution()
        assert solved.exit_code == 0, (task_id, solved.stdout[-2000:], solved.stderr[-2000:])
        oracle = await session.grade()
    finally:
        await session.close()
    assert (oracle.status, oracle.reward) == ("scored", 1.0), (task_id, oracle.status, oracle.reason, oracle.metadata)
    return oracle.metadata


def _pod_processes() -> list[str]:
    """The processes of this user that hpc-sandbox started for this process's pods: their holders and the launchers of
    their services, which name this process's pid after their script."""

    found = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        for index, word in enumerate(argv[:-1]):
            if word.endswith((b"/holder.py", b"/anchor.py")) and argv[index + 1] == str(os.getpid()).encode():
                found.append(f"{pid}: {b' '.join(argv[:8]).decode(errors='replace')}")
    return found


async def _nothing_left(service: HarborEnvironmentService, scratch: Path, registered: set[int]) -> None:
    """No pod, sandbox or file of another owner in the scratch root once its deletions are done, no process of the
    pods, nothing registered for cleanup and nothing in node scratch but the record of the task entries it checked."""

    inner = cast(Any, service.provider).inner
    assert await inner.wait_for_deletions(300)
    sessions = [
        f"{root.name}/{entry.name}"
        for root in (inner.pod_root, inner.session_root)
        if root.is_dir()
        for entry in root.iterdir()
        if entry.name.startswith("hpcsb-") or (entry.name == ".trash" and any(entry.iterdir()))
    ]
    foreign = [str(path) for path in _walk(scratch) if path.lstat().st_uid != os.getuid()]
    verified = f"platoon-harbor-{os.getuid()}"  # where a node records the cache entries it checked: kept on purpose
    node = [path for path in Path(os.environ["PLATOON_NODE_SCRATCH"]).iterdir() if path.name != verified]
    left = sorted(set(cleanup._live) - registered)
    assert (sessions, foreign, _pod_processes(), node, left) == ([], [], [], [], [])


def _walk(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for directory, names, files in os.walk(root):
        for name in [*names, *files]:
            yield Path(directory) / name


# --- the stack fixtures -----------------------------------------------------------------------------------------


async def test_the_stack_fixtures_score_their_oracles_and_their_no_ops(scratch: Path, registered: set[int]):
    service = _service(scratch, dataset_dir=STACKS)
    for task_id in ("app-kv", "app-kv-seeded", "sidecar-artifacts"):
        metadata = await _oracle_and_no_op(service, task_id)
        assert metadata["services"]["main"]["run_as"] == "root"
    await _nothing_left(service, scratch, registered)


async def test_a_sidecar_that_exits_or_never_turns_healthy_is_never_scored(scratch: Path, registered: set[int]):
    service = _service(scratch, dataset_dir=STACKS)
    session = await _open(service, "app-kv-crash")
    try:
        await session.run_solution()  # the store exits after its first write
        crashed = await session.grade()
    finally:
        await session.close()
    assert (crashed.status, crashed.failure_kind) == ("invalid", "sidecar_unhealthy") and not crashed.trainable
    with pytest.raises(EnvSetupError) as error:
        await _open(service, "app-kv-unhealthy")
    assert error.value.failure_kind == "sidecar_start"
    await _nothing_left(service, scratch, registered)


async def test_names_resolve_and_the_relay_reaches_a_host_socket(scratch: Path, registered: set[int]):
    service = _service(scratch, dataset_dir=STACKS)
    directory = Path(tempfile.mkdtemp(prefix="platoon-relay-", dir="/tmp"))  # a short path: a socket's must fit

    async def answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.0 200 OK\r\nContent-Length: 19\r\n\r\nhello from the host")
        await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(answer, path=str(directory / "gw.sock"))
    needs = SessionNeeds(upstreams={"gateway": f"unix://{directory / 'gw.sock'}"})
    try:
        session = await _open(service, "app-kv", needs=needs)
        try:
            route = session.routes["gateway"]
            assert route.startswith("http://127.0.0.1:") and session.host_loopback is False
            relayed = await _main(session).exec(["python3", "-c", FETCH, route])
            assert relayed.stdout.strip() == "hello from the host", relayed
            names = "import socket; print(socket.gethostbyname('kv'), socket.gethostbyname('main'))"
            resolved = await _main(session).exec(["python3", "-c", names])
            assert resolved.stdout.split() == ["127.0.0.1", "127.0.0.1"], resolved
        finally:
            await session.close()
    finally:
        server.close()
        await server.wait_closed()
        shutil.rmtree(directory, ignore_errors=True)
    await _nothing_left(service, scratch, registered)


async def test_one_stack_cannot_reach_another_s_services(scratch: Path, registered: set[int]):
    """Two stacks of one task, whose stores listen on one port: each main reaches its own store, never the other's."""

    service = _service(scratch, dataset_dir=STACKS)
    first, second = await asyncio.gather(_open(service, "app-kv"), _open(service, "app-kv"))
    try:
        for session, value in ((first, "one"), (second, "two")):
            put = await _main(session).exec(["python3", "-c", PUT, f"http://kv:6379/{value}", value])
            assert put.exit_code == 0, put.stderr
        for session, own, other in ((first, "one", "two"), (second, "two", "one")):
            mine = await _main(session).exec(["python3", "-c", FETCH, f"http://kv:6379/{own}"])
            theirs = await _main(session).exec(["python3", "-c", FETCH, f"http://kv:6379/{other}"])
            assert mine.stdout.strip() == own and theirs.stdout.strip().startswith("nothing"), (mine, theirs)
    finally:
        await asyncio.gather(first.close(), second.close())
    await _nothing_left(service, scratch, registered)


async def test_the_host_reaches_a_service_s_port_through_connect(scratch: Path, registered: set[int]):
    """``connect`` gives a socket of this host that reaches the pod's port, also where the pod's own path is too long
    for a socket's address (a deep scratch root): then a link to it in node scratch."""

    deep = scratch / ("d" * 60)
    service = _service(deep, dataset_dir=STACKS)
    session = await _open(service, "app-kv")
    try:
        assert session.stack is not None
        put = await _main(session).exec(["python3", "-c", PUT, "http://kv:6379/answer", "forty-two"])
        assert put.exit_code == 0, put.stderr
        path = (await session.stack.connect("kv", 6379)).removeprefix("unix://")
        assert len(os.fsencode(path)) < 108, path
        reader, writer = await asyncio.open_unix_connection(path)
        try:
            writer.write(b"GET /answer HTTP/1.0\r\nHost: kv\r\n\r\n")
            await writer.drain()
            reply = await asyncio.wait_for(reader.read(), 30)
        finally:
            writer.close()
    finally:
        await session.close()
    assert reply.startswith(b"HTTP/1.") and reply.endswith(b"forty-two"), reply
    await _nothing_left(service, deep, registered)


async def test_a_kill_of_every_process_in_main_grades_as_in_a_single_sandbox(scratch: Path, registered: set[int]):
    """The agents' ``kill -9 -1`` kills what they ran, never main itself, whose idle loop is its PID 1 there, as a
    single sandbox's and a Docker container's are: the attempt is scored (0.0: nothing was stored)."""

    service = _service(scratch, dataset_dir=STACKS)
    session = await _open(service, "app-kv")
    try:
        await session.call_tool("bash", {"command": "kill -9 -1"})
        outcome = await session.grade()
    finally:
        await session.close()
    assert (outcome.status, outcome.reward) == ("scored", 0.0), (outcome.status, outcome.failure_kind, outcome.reason)
    await _nothing_left(service, scratch, registered)


# --- sidecars that switch users ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("task_id", "sidecar", "data", "uid"),
    [("app-redis", "redis", ".", 999), ("app-postgres", "db", "/var/lib/postgresql/data", 70)],
)
async def test_database_sidecars_switch_users_as_in_docker(
    task_id: str,
    sidecar: str,
    data: str,
    uid: int,
    scratch: Path,
    registered: set[int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The image's entrypoint, as root with your subordinate ids, gives the data directory to the database's own user
    and runs the server as it, as Docker runs it; as your own uid, where your user has none, it skips both."""

    _skip_without_pods(scratch)
    cache = use_site_cache(monkeypatch, tmp_path / "cache")
    dataset = tmp_path / "dataset"
    shutil.copytree(IMAGES / task_id, dataset / task_id)
    lock = fetch(fetch_config(str(dataset), lock=tmp_path / "lock.json", refuse=LOCK_STACKS))
    pin_images(lock, image_map=default_image_map(cache, PROVIDER))  # the registry's configuration of the image
    service = _service(scratch, dataset=str(dataset), lock=str(tmp_path / "lock.json"), refuse=LOCK_STACKS)
    await service.check_tasks("train", [task_id])
    metadata = await _oracle_and_no_op(service, task_id)
    mode = metadata["services"][sidecar]["run_as"]
    assert mode in ("subids", "caller") and metadata["services"]["main"]["run_as"] == "root"
    session = await _open(service, task_id)
    try:
        owner = await _service_sandbox(session, sidecar).exec(["stat", "-c", "%u", data])
    finally:
        await session.close()
    assert owner.stdout.strip() == str(uid if mode == "subids" else os.getuid()), (mode, owner)
    await _nothing_left(service, scratch, registered)


async def test_a_judge_s_record_that_only_its_own_user_reads_reaches_the_tests(scratch: Path, registered: set[int]):
    """The judge gives the verifier log directory it binds to its image's user and records there as that user, in a
    file only that user reads, which belongs to one of your subordinate ids on this host: grading reads it through the
    judge, so the tests find it."""

    service = _service(scratch, dataset_dir=IMAGES)
    session = await _open(service, "judge-logs")
    try:
        assert session.stack is not None
        mode = session.stack.run_modes["judge"]
        owner = session.stack.volume_path("platoon:logs-verifier").stat().st_uid
        outcome = await session.grade()
    finally:
        await session.close()
    assert (owner != os.getuid()) == (mode == "subids"), (mode, owner)
    assert (outcome.status, outcome.reward) == ("scored", 1.0), (outcome.status, outcome.reason, outcome.metadata)
    await _nothing_left(service, scratch, registered)


# --- hello-mcp --------------------------------------------------------------------------------------------------


async def test_hello_mcp_scores_its_oracle_and_its_no_op(
    scratch: Path, registered: set[int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Harbor's hello-mcp, its images built as platoon prepare builds them (its MCP server's context whole); its
    solution and tests install packages through the pod's egress.  Its tests install uv with uv's installer, whose
    ``tar``, run as root, gives the files their archive's owners, which ``main``, root of a namespace that maps one id,
    cannot: an overlay installs it in ``main``'s image instead (as a dataset's overlay would), so that the tests run
    after the no-op too, not only after the solution, which installs uv in a way that works there."""

    _skip_without_pods(scratch)
    builder = provider_builder(PROVIDER)
    if builder is None:
        pytest.skip(f"the {PROVIDER} provider runs registry images only: hello-mcp's would be built and pushed")
    cache = use_site_cache(monkeypatch, tmp_path / "cache")
    overlay = tmp_path / "overlays" / "install-uv"
    overlay.mkdir(parents=True)
    (overlay / "overlay.toml").write_text(UV_IN_MAIN)
    settings: dict[str, Any] = {"lock": str(tmp_path / "lock.json"), "refuse": LOCK_STACKS, "overlays": [str(overlay)]}
    lock = fetch(fetch_config(str(HELLO_MCP), **settings))
    build_images(
        lock,
        cache=cache,
        image_map=default_image_map(cache, PROVIDER),
        out_dir=default_image_dir(cache, PROVIDER),
        builder=builder,
        scratch=tmp_path / "build",
    )
    service = _service(scratch, dataset=str(HELLO_MCP), **settings)
    [task_id] = await service.list_tasks("train")
    await service.check_tasks("train", [task_id])
    metadata = await _oracle_and_no_op(service, task_id)
    assert set(metadata["services"]) == {"main", "mcp-server"}
    # Through the tool plane, which reaches the server through the socket the pod exposes: hello-mcp's get_secret
    # says nothing of being read-only, so a read-only agent cannot call it.  And from main, at the server's own URL.
    assert solved(await _rollout_of(SOLVER, service, task_id, tmp_path / "plane"))
    assert solved_directly(await _rollout_of(DIRECT, service, task_id, tmp_path / "direct"))
    session = await _open(service, task_id)
    try:
        assert await hidden_from_read_only(session, SECRET_TOOL)
    finally:
        await session.close()
    await _nothing_left(service, scratch, registered)


# --- task MCP servers, through the tool plane ---------------------------------------------------------------------

_BUILT: dict[str, HarborEnvironmentService] = {}


def built_service(*, key: str) -> HarborEnvironmentService:
    """A service that a test built, for its rollouts' route (``service: <this module>:built_service``), so that the
    test's checks of what is left see what their sessions left."""

    return _BUILT[key]


async def _rollout_of(harness: str, service: HarborEnvironmentService, task_id: str, tmp_path: Path) -> dict[str, Any]:
    """A rollout of one of the harnesses that solve the task (``tests/mcp_solver.py``: through the tool plane, or from
    main), on the rollout path: its root trajectory."""

    _BUILT[task_id] = service
    route = EnvironmentRoute(
        name="pods", service=f"{__name__}:built_service", service_kwargs={"key": task_id}, harness=harness
    )
    config = RolloutConfig(output_dir=str(tmp_path / "out"), return_dict=True, verbose=False)
    task = route.attach(await service.get_task(task_id))
    collection = await run_harness_rollout(task, config, gateway=no_model_gateway())
    assert isinstance(collection, dict)
    (root,) = collection["trajectories"].values()
    return root


async def test_the_mcp_fixtures_are_solved_through_the_tool_plane(scratch: Path, registered: set[int], tmp_path: Path):
    """mcp-http and mcp-sse, an MCP sidecar over each transport: a function harness solves each through the tool plane,
    which reaches the sidecar through the socket the pod exposes, and main's own clients (the oracle's, and a CLI
    harness's at the URL it is given) reach it by name; a read-only agent cannot call the tool that changes the
    secret."""

    service = _service(scratch, dataset_dir=STACKS)
    for task_id in ("mcp-http", "mcp-sse"):
        assert solved(await _rollout_of(SOLVER, service, task_id, tmp_path / task_id / "plane"))
        assert solved_directly(await _rollout_of(DIRECT, service, task_id, tmp_path / task_id / "direct"))
        await _oracle_and_no_op(service, task_id)
        session = await _open(service, task_id)
        try:
            assert await hidden_from_read_only(session, ROTATE_TOOL)
        finally:
            await session.close()
    await _nothing_left(service, scratch, registered)


# --- a judge sidecar of a published dataset ---------------------------------------------------------------------


async def test_a_kumo_task_reaches_its_judge_by_name_and_grades_with_the_judge_s_log(
    scratch: Path, registered: set[int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """kumo's chemicalenv-t12-a16-v1-s0 from harbor-datasets, at a commit: ``main`` runs the task's client against its
    judge, a sidecar built from the task's ``Dockerfile_verifier``; the judge's record of the action, in the verifier
    logs it binds, reaches ``main``'s only when grading copies it there for the tests; the no-op scores 0.0.  Its
    tests write strings into ``reward.json``, which platoon, as Harbor, refuses as malformed: its oracle is no
    control."""

    _skip_without_pods(scratch)
    builder = provider_builder(PROVIDER)
    if builder is None:
        pytest.skip(f"the {PROVIDER} provider runs registry images only: kumo's would be built and pushed")
    cache = use_site_cache(monkeypatch, tmp_path / "cache")
    task_dir = tmp_path / "dataset" / KUMO_TASK
    for name in KUMO_FILES:
        (task_dir / name).parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{KUMO}/{KUMO_TASK}/{name}", timeout=60) as response:
            (task_dir / name).write_bytes(response.read())
    lock = fetch(fetch_config(str(task_dir.parent), lock=tmp_path / "lock.json", refuse=LOCK_STACKS))
    build_images(
        lock,
        cache=cache,
        image_map=default_image_map(cache, PROVIDER),
        out_dir=default_image_dir(cache, PROVIDER),
        builder=builder,
        scratch=tmp_path / "build",
    )
    service = _service(scratch, dataset=str(task_dir.parent), lock=str(tmp_path / "lock.json"), refuse=LOCK_STACKS)
    await service.check_tasks("train", [KUMO_TASK])
    session = await _open(service, KUMO_TASK)
    try:
        assert session.stack is not None
        judge = await _main(session).exec(["python", "-c", AWAIT, "http://verifier:8080/health"])
        assert judge.exit_code == 0, (judge.stderr[-2000:], await session.stack.logs("verifier"))
        acted = await _main(session).exec(["python", "/app/kumo_cli.py", "act", "Odor test"])
        assert acted.exit_code == 0 and '"steps_used": 1' in acted.stdout, acted
        unseen = await _main(session).exec(["test", "-e", "/logs/verifier/actions.jsonl"])
        assert unseen.exit_code != 0, unseen
        noop = await session.grade()
        logged = await _main(session).exec(["cat", "/logs/verifier/actions.jsonl"])
        assert '"action": "Odor test"' in logged.stdout, logged
    finally:
        await session.close()
    assert (noop.status, noop.reward) == ("scored", 0.0), (noop.status, noop.reason)
    assert noop.metadata["services"]["verifier"]["run_as"] in ("subids", "caller")
    await _nothing_left(service, scratch, registered)


# --- many stacks ------------------------------------------------------------------------------------------------


async def test_many_stacks_some_cancelled_while_they_start_leave_nothing(scratch: Path, registered: set[int]):
    service = _service(scratch, dataset_dir=STACKS)
    rng = random.Random(20261026)
    gate = asyncio.Semaphore(CONCURRENCY)
    starts: list[float] = []
    cancelled = 0

    async def one(index: int) -> None:
        nonlocal cancelled
        async with gate:
            began = time.monotonic()
            opening = asyncio.ensure_future(_open(service, "app-kv"))
            if index % 4 == 0:
                await asyncio.sleep(rng.uniform(0.0, 2.0))
                opening.cancel()
            try:
                session = await opening
            except asyncio.CancelledError:
                cancelled += 1
                return
            starts.append(time.monotonic() - began)
            try:
                command = "python3 -c 'import socket; socket.gethostbyname(\"kv\")'"
                result = await session.call_tool("bash", {"command": command})
                assert not result.is_error
            finally:
                await session.close()

    await asyncio.gather(*(one(index) for index in range(COUNT)))
    await _nothing_left(service, scratch, registered)
    assert cancelled > 0 or COUNT < 4
    quantiles = statistics.quantiles(starts, n=100) if len(starts) >= 2 else [0.0] * 99
    print(  # noqa: T201 - the timings the exit check records
        f"pods: {COUNT} stacks of app-kv, {CONCURRENCY} at once, {cancelled} cancelled while they started: setup "
        f"p50 {statistics.median(starts):.2f} s, p99 {quantiles[98]:.2f} s"
    )


# --- without hpc-sandbox 0.5 --------------------------------------------------------------------------------------


async def test_an_hpc_sandbox_without_0_5_s_pods_refuses_stacks_naming_0_5(scratch: Path, registered: set[int]):
    pytest.importorskip("hpc_sandbox")
    refusal = _refusal(scratch)
    if refusal is None or "hpc-sandbox 0.5" not in refusal:
        pytest.skip("the installed hpc-sandbox runs stacks")
    service = HarborEnvironmentService(dataset_dir=STACKS, provider=PROVIDER, provider_options=_options(scratch))
    with pytest.raises(ValueError, match="No runnable Harbor tasks .* install hpc-sandbox 0.5"):
        await service.list_tasks("train")  # a directory's stacks are skipped, with a warning
    task = Task(id="app-kv", misc={"harbor": {"task_dir": str(STACKS / "app-kv")}})
    with pytest.raises(EnvSetupError, match="install hpc-sandbox 0.5") as error:
        await service.open(task, rollout_id="pods-refused")
    assert error.value.failure_kind == "env_config"
    assert set(cleanup._live) == registered
