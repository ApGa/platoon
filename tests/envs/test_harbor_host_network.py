"""Harbor stacks on the host-network fallback (``stacks: host_network``) over the process provider: the fixture below
``fixtures/harbor-host-network`` scores its oracle and its no-op with an authenticated sidecar, whose healthcheck names
its own port on its loopback, and a sidecar found exposed at grading makes the outcome invalid; an MCP server in
``main`` is served through the tool plane; a sidecar that no overlay authenticates, a port without a listen rule, a
task whose files name a service where the fallback cannot rewrite it (until an overlay edits it), an MCP server that
is a sidecar and a stack without a network are refused; and nothing is left after many stacks, some cancelled while
they start."""

from __future__ import annotations

import asyncio
import os
import random
import shutil
import socket
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from mcp_solver import SECRET
from site_cache import use_site_cache

from platoon.envs.harbor import HarborEnvironmentService, HarborSession
from platoon.envs.harbor.dataset import HarborDataset
from platoon.envs.harbor.fetch import fetch, fetch_config
from platoon.envs.service import EnvSetupError
from platoon.sandbox import cleanup, host_network_stacks
from platoon.sandbox.host_network_stacks import HostNetworkStackProvider, beyond_loopback
from platoon.sandbox.process import ProcessProvider
from platoon.sandbox.stacks import DrivenStack

FIXTURES = Path(__file__).parent / "fixtures" / "harbor-host-network"
# The stack fixtures' MCP server, which listens on PLATOON_PORT_8000 where that is set, and its healthcheck.
MCP_SERVER = Path(__file__).parent / "fixtures" / "harbor-stacks" / "tasks" / "mcp-http" / "environment" / "mcp"
TASK = "app-kv-auth"
AUTHENTICATE = FIXTURES / "overlays" / "authenticate"
LOCK_STACKS: dict[str, bool] = {}  # a fetch's default, which locks stacks as runnable
# Answers the store's status for a request to the URL given, with the token in KV_TOKEN when the second argument is 1.
ASK = (
    "import os, sys, urllib.error, urllib.request\n"
    "request = urllib.request.Request(sys.argv[1])\n"
    "if sys.argv[2] == '1':\n"
    "    request.add_header('Authorization', 'Bearer ' + os.environ['KV_TOKEN'])\n"
    "try:\n"
    "    print(urllib.request.urlopen(request, timeout=5).status)\n"
    "except urllib.error.HTTPError as exc:\n"
    "    print(exc.code)\n"
)


@pytest.fixture(autouse=True)
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Node scratch of each test's own: the sessions' copies, and the fallback's stack directories."""

    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "scratch"))
    return tmp_path / "scratch"


@pytest.fixture
def before() -> set[int]:
    return {id(sandbox) for sandbox in cleanup.live_sandboxes()}


@pytest.fixture
def provider(tmp_path: Path) -> ProcessProvider:
    return ProcessProvider(root=tmp_path / "sandboxes")


def _dataset(root: Path, *names: str) -> Path:
    """A dataset directory holding the fixture task under each of ``names``."""

    for name in names:
        task = root / name
        shutil.copytree(FIXTURES / "tasks" / TASK, task)
        toml = task / "task.toml"
        toml.write_text(toml.read_text().replace(f'name = "platoon/{TASK}"', f'name = "platoon/{name}"'))
    return root


def _overlay(root: Path, name: str, text: str) -> Path:
    directory = root / "overlays" / name
    directory.mkdir(parents=True)
    (directory / "overlay.toml").write_text(text)
    return directory


def _service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: ProcessProvider,
    dataset: Path,
    overlays: list[Path],
    **options: Any,
) -> HarborEnvironmentService:
    """A host-network service of ``dataset``, fetched with ``overlays``."""

    use_site_cache(monkeypatch, tmp_path / "cache")
    settings: dict[str, Any] = {
        "dataset": str(dataset),
        "lock": str(tmp_path / "lock.json"),
        "refuse": LOCK_STACKS,
        "overlays": [str(path) for path in overlays],
    }
    fetch(fetch_config(settings["dataset"], lock=settings["lock"], refuse=LOCK_STACKS, overlays=settings["overlays"]))
    return HarborEnvironmentService(
        provider=provider, stacks=options.pop("stacks", "host_network"), **settings, **options
    )


async def _open(service: HarborEnvironmentService, task_id: str = TASK) -> HarborSession:
    return await service.open(await service.get_task(task_id), rollout_id=f"rollout-{task_id}")


def _stack(session: HarborSession) -> DrivenStack:
    assert isinstance(session.stack, DrivenStack)
    return session.stack


def _left(service: HarborEnvironmentService, provider: ProcessProvider, before: set[int]) -> list[Any]:
    fallback = cast(HostNetworkStackProvider, service.stack_provider)
    scratch = Path(os.environ["PLATOON_NODE_SCRATCH"])
    paths = [
        path
        for root in (provider.root, scratch)
        if root.exists()
        for path in root.iterdir()
        # The fallback's root, whose stacks are listed below, and where a node records the cache entries it checked.
        if path not in (fallback.root, scratch / f"platoon-harbor-{os.getuid()}")
    ]
    stacks = list(fallback.root.iterdir()) if fallback.root.exists() else []
    return [*paths, *stacks, *(sandbox for sandbox in cleanup.live_sandboxes() if id(sandbox) not in before)]


async def _eventually(condition: Callable[[], bool], timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "the condition never held"
        await asyncio.sleep(0.02)


# --- the fixture's outcomes ------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_app_kv_scores_its_oracle_and_its_no_op_with_an_authenticated_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: ProcessProvider, before: set[int]
):
    service = _service(tmp_path, monkeypatch, provider, _dataset(tmp_path / "dataset", TASK), [AUTHENTICATE])
    await service.check_tasks("train", [TASK])
    session = await _open(service)
    try:
        noop = await session.grade()
    finally:
        await session.close()
    assert (noop.status, noop.reward) == ("scored", 0.0), noop.reason
    session = await _open(service)
    try:
        solved = await session.run_solution()
        assert solved.exit_code == 0, solved.stderr
        oracle = await session.grade()  # its tests reach the store through [verifier].env, rewritten
    finally:
        await session.close()
    assert (oracle.status, oracle.reward) == ("scored", 1.0), (oracle.reason, oracle.metadata)
    assert oracle.metadata["services"]["kv"]["run_as"] == "caller"
    await _eventually(lambda: _left(service, provider, before) == [])


@pytest.mark.asyncio
async def test_the_sidecar_listens_on_the_loopback_at_its_port_and_answers_its_own_stack_s_token_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: ProcessProvider, before: set[int]
):
    service = _service(tmp_path, monkeypatch, provider, _dataset(tmp_path / "dataset", TASK), [AUTHENTICATE])
    first, second = await asyncio.gather(_open(service), _open(service))
    try:
        stack = _stack(first)
        port = int((await stack.connect("kv", 6379)).rsplit(":", 1)[1])
        kv = stack.services["kv"]
        main_env, kv_env = cast(Any, first.sandbox).spec.env, cast(Any, kv).spec.env
        assert (kv_env["KV_PORT"], kv_env["KV_HOST"]) == (str(port), "127.0.0.1")
        assert main_env["KV_URL"] == f"http://127.0.0.1:{port}"
        assert main_env["KV_TOKEN"] == kv_env["KV_TOKEN"] != cast(Any, second.sandbox).spec.env["KV_TOKEN"]
        check = stack._checks["kv"].command  # its healthcheck named its own port on its loopback
        assert check[-1] == f"http://127.0.0.1:{port}/"
        assert beyond_loopback({port}) == []
        asked = {}
        main = first.sandbox
        assert main is not None
        for token in ("0", "1"):
            asked[token] = (await main.exec(["python3", "-c", ASK, main_env["KV_URL"] + "/", token])).stdout.strip()
        assert asked == {"0": "401", "1": "200"}  # another stack's agents, and anything else on the node, lack it
        assert stack.rewrite("http://kv:6379", service="main") == f"http://127.0.0.1:{port}"
    finally:
        await asyncio.gather(first.close(), second.close())
    await _eventually(lambda: _left(service, provider, before) == [])


@pytest.mark.asyncio
async def test_a_sidecar_found_exposed_at_grading_makes_the_outcome_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: ProcessProvider
):
    """The probes before and after the tests look at what listens at each sidecar's ports: one listening beyond the
    node's loopback by then, where other hosts reach it, makes the outcome invalid, never scored."""

    service = _service(tmp_path, monkeypatch, provider, _dataset(tmp_path / "dataset", TASK), [AUTHENTICATE])
    session = await _open(service)
    try:
        port = int((await _stack(session).connect("kv", 6379)).rsplit(":", 1)[1])
        table = tmp_path / "tcp"  # the store, as /proc/net/tcp would list it on 0.0.0.0
        table.write_text(
            "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
            f"   0: 00000000:{port:04X} 00000000:0000 0A 00000000:00000000 00:00000000 00000000 {os.getuid():5d}"
            "        0 4242 1 0000000000000000 100 0 0 10 0\n"
        )
        monkeypatch.setattr(host_network_stacks, "_TABLES", ((str(table), 4),))
        outcome = await session.grade()
    finally:
        await session.close()
    assert (outcome.status, outcome.failure_kind) == ("invalid", "sidecar_unhealthy"), outcome.reason
    assert f"listens on 0.0.0.0 at port {port} (its port 6379), beyond this host's loopback" in (outcome.reason or "")


@pytest.mark.asyncio
async def test_main_s_mcp_server_is_served_through_the_port_its_stack_holds_and_one_that_is_a_sidecar_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: ProcessProvider
):
    """A server in ``main`` (at a port main declares, which its overlay names) is at its rewritten URL, for harnesses in
    ``main``, and the tool plane serves its tools through the port its stack holds; one that is a sidecar is refused
    when the tasks are listed and when a session opens: an MCP client sends no token, the plane's no more than a
    harness's, so the sidecar would answer the node."""

    dataset = _dataset(tmp_path / "dataset", TASK, "app-kv-mcp")
    compose = dataset / TASK / "environment" / "docker-compose.yaml"
    url = "      KV_URL: http://kv:6379\n"
    # main serves the stack fixtures' MCP server at its port 8000, and is healthy once it listens. Its check reads the
    # port from its listen variable, since main's init waits for the check as written, before the stack rewrites it.
    health = f"{sys.executable} {MCP_SERVER / 'health.py'} http://localhost:$${{TOOLS_PORT:-8000}}/"
    main = (
        "    expose: ['8000']\n"
        f"    command: ['{sys.executable}', '{MCP_SERVER / 'server.py'}', '--transport', 'streamable-http']\n"
        "    healthcheck:\n"
        f"      test: ['CMD-SHELL', '{health}']\n"
        "      interval: 100ms\n"
        "      retries: 100\n"
    )
    compose.write_text(compose.read_text().replace(url, url + main))
    server = '\n[[environment.mcp_servers]]\nname = "{name}"\ntransport = "streamable-http"\nurl = "{url}"\n'
    toml = dataset / TASK / "task.toml"
    toml.write_text(toml.read_text() + server.format(name="tools", url="http://main:8000/mcp"))
    toml = dataset / "app-kv-mcp" / "task.toml"
    toml.write_text(toml.read_text() + server.format(name="store", url="http://kv:6379/mcp"))
    main_port = _overlay(
        tmp_path,
        "main-port",
        f'[match]\nname = "{TASK}"\n[[compose]]\nservice = "main"\n'
        'host_network = { listen = { 8000 = "TOOLS_PORT" } }\n',
    )
    service = _service(tmp_path, monkeypatch, provider, dataset, [AUTHENTICATE, main_port])
    session = await _open(service)
    try:
        port = int((await _stack(session).connect("main", 8000)).rsplit(":", 1)[1])
        assert [server.url for server in session.mcp_servers] == [f"http://127.0.0.1:{port}/mcp"]
        assert [server["url"] for server in session.task_mcp_servers] == [f"http://127.0.0.1:{port}/mcp"]
        assert "tools__get_secret" in [tool.name for tool in await session.tools()]
        assert (await session.call_tool("tools__get_secret", {})).as_text() == SECRET
    finally:
        await session.close()
    message = "whose MCP server store is its sidecar kv .http://kv:6379/mcp.: an MCP client sends no token"
    with pytest.raises(EnvSetupError, match=message) as listed:
        await service.check_tasks("train", [TASK, "app-kv-mcp"])
    assert listed.value.failure_kind == "env_config"
    with pytest.raises(EnvSetupError, match=message):
        await _open(service, "app-kv-mcp")


# --- what the fallback refuses ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_sidecar_that_no_overlay_authenticates_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: ProcessProvider
):
    service = _service(tmp_path, monkeypatch, provider, _dataset(tmp_path / "dataset", TASK), [])
    with pytest.raises(EnvSetupError, match="whose sidecar kv no overlay authenticates") as listed:
        await service.check_tasks("train", [TASK])
    assert listed.value.failure_kind == "env_config"
    assert "1 of the 1 tasks of split 'train' are stacks that the host-network fallback" in str(listed.value)
    with pytest.raises(EnvSetupError, match="whose sidecar kv no overlay authenticates") as opened:
        await _open(service)
    assert opened.value.failure_kind == "env_config"
    assert not provider.root.exists() or list(provider.root.iterdir()) == []  # nothing started


@pytest.mark.asyncio
async def test_a_declared_port_without_a_listen_rule_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: ProcessProvider
):
    token_only = _overlay(
        tmp_path, "token-only", '[[compose]]\nservice = "kv"\nhost_network = { token = "KV_TOKEN", bind = "KV_HOST" }\n'
    )
    service = _service(tmp_path, monkeypatch, provider, _dataset(tmp_path / "dataset", TASK), [token_only])
    with pytest.raises(EnvSetupError, match="whose service kv declares port 6379 and no listen rule"):
        await service.check_tasks("train", [TASK])


@pytest.mark.asyncio
async def test_a_task_whose_instruction_names_a_service_is_refused_until_an_overlay_edits_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: ProcessProvider
):
    dataset = _dataset(tmp_path / "dataset", "app-kv-named")
    instruction = dataset / "app-kv-named" / "instruction.md"
    instruction.write_text(
        instruction.read_text().replace("runs at `$KV_URL`", "runs at `http://kv:6379` (in `$KV_URL`)")
    )
    named = _service(tmp_path / "named", monkeypatch, provider, dataset, [AUTHENTICATE])
    assert named._locked is not None
    plan = named._locked.dataset(fetch_allowed=False).lock.get("app-kv-named").compose
    assert plan is not None
    assert plan["unrewritable_references"] == {"first": ["instruction.md: http://kv:6379"], "count": 1}
    with pytest.raises(EnvSetupError, match=r"cannot rewrite them \(instruction.md: http://kv:6379\)"):
        await named.check_tasks("train", ["app-kv-named"])

    unname = _overlay(
        tmp_path,
        "unname-kv",
        '[match]\nname = "app-kv-named"\n[[edit]]\nfile = "instruction.md"\n'
        "pattern = 'runs at `http://kv:6379` \\(in `\\$KV_URL`\\)'\nreplace = 'runs at `$KV_URL`'\n",
    )
    fixed = _service(tmp_path / "fixed", monkeypatch, provider, dataset, [AUTHENTICATE, unname])
    await fixed.check_tasks("train", ["app-kv-named"])
    session = await _open(fixed, "app-kv-named")
    try:
        assert "kv:6379" not in session.instruction
        await session.run_solution()
        oracle = await session.grade()
    finally:
        await session.close()
    assert (oracle.status, oracle.reward) == ("scored", 1.0), oracle.reason


@pytest.mark.asyncio
async def test_a_stack_without_a_network_is_refused_on_the_host_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: ProcessProvider
):
    service = _service(
        tmp_path, monkeypatch, provider, _dataset(tmp_path / "dataset", TASK), [AUTHENTICATE], network_policy="none"
    )
    with pytest.raises(EnvSetupError, match="runs its stack without a network for its agents") as listed:
        await service.check_tasks("train", [TASK])
    assert "cannot take away" in str(listed.value)
    with pytest.raises(EnvSetupError, match="cannot take away"):
        await _open(service)


def test_a_dataset_directory_s_stacks_are_skipped_on_the_host_network(tmp_path: Path):
    """A dataset directory has no overlays, so no sidecar of its is authenticated: its stacks are skipped, the message
    saying to fetch the dataset with an overlay."""

    dataset = _dataset(tmp_path / "dataset", TASK)
    with pytest.raises(ValueError, match="a dataset directory has none of: fetch the dataset with such an overlay"):
        HarborDataset(dataset, on_host_network=True)
    assert HarborDataset(dataset).names("train") == [TASK]


@pytest.mark.asyncio
async def test_the_same_lock_runs_its_stacks_elsewhere_as_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: ProcessProvider
):
    """The overlay's rules act on the fallback alone: on the process provider's own stacks the store listens on its
    listen variable, without a token."""

    service = _service(
        tmp_path, monkeypatch, provider, _dataset(tmp_path / "dataset", TASK), [AUTHENTICATE], stacks="auto"
    )
    assert service.stack_provider is service.provider
    session = await _open(service)
    try:
        env = cast(Any, _stack(session).services["kv"]).spec.env
        assert "KV_TOKEN" not in env and "KV_HOST" not in env and "KV_PORT" not in env
        await session.run_solution()
        oracle = await session.grade()
    finally:
        await session.close()
    assert (oracle.status, oracle.reward) == ("scored", 1.0), oracle.reason


# --- leaks -----------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stacks_started_and_cancelled_mid_start_leave_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: ProcessProvider, before: set[int]
):
    """The exit check's leak test, smaller: stacks open and close, a few at a time, and some are cancelled while they
    start; afterwards no sandbox, stack directory, port claim, process or registration is left."""

    service = _service(tmp_path, monkeypatch, provider, _dataset(tmp_path / "dataset", TASK), [AUTHENTICATE])
    rng = random.Random(0)
    ports: set[int] = set()

    async def one(index: int) -> str:
        opening = asyncio.ensure_future(_open(service))
        if index % 3 == 0:
            await asyncio.sleep(rng.uniform(0.0, 0.3))
            opening.cancel()
            try:
                session = await opening
            except asyncio.CancelledError:
                return "cancelled"
        else:
            session = await opening
        ports.add(int((await _stack(session).connect("kv", 6379)).rsplit(":", 1)[1]))
        await session.close()
        return "closed"

    results = []
    for batch in range(4):
        results += await asyncio.gather(*(one(batch * 6 + index) for index in range(6)))
    assert "closed" in results and "cancelled" in results
    await _eventually(lambda: _left(service, provider, before) == [])
    for port in ports:  # released with their stacks: another stack may claim each again
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as claim:
            claim.bind(f"\0platoon-stack-port-{port}")
    root = str(provider.root)
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            environ = Path(f"/proc/{pid}/environ").read_bytes()
        except OSError:
            continue
        assert f"PLATOON_SANDBOX_ROOT={root}".encode() not in environ, f"process {pid} was left running"
