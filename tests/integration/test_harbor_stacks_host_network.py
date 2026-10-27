"""Harbor stacks on the host-network fallback over real Singularity sandboxes, end to end (RFC 0010 M8's checks): with
``stacks: host_network``, each service a single sandbox on the node's network, the host-network fixture scores its
oracle and its no-op through its authenticated sidecar, which listens on the loopback alone and answers its own stack's
token alone; a sidecar that no overlay authenticates is refused before anything starts; and nothing is left after many
stacks, some cancelled while they start.  The fallback needs no pods, so any hpc-sandbox runs it, 0.4 included.

Skipped unless ``PLATOON_TEST_HOST_NETWORK=1`` on a compute node with SingularityCE or Apptainer on ``PATH`` and
hpc-sandbox installed.  Never run it on a login node: it starts containers.  The images come from the network: the
nodes pull ``docker://python:3.12-slim``, whose configuration the registry gives the pin.  ``HPC_SANDBOX_SCRATCH``
(node-local disk) and ``HPC_SANDBOX_IMAGE_CACHE`` say where sandboxes and images go;
``PLATOON_TEST_HOST_NETWORK_STACKS`` and ``PLATOON_TEST_HOST_NETWORK_CONCURRENCY`` size the test of many stacks (20 and
8).
"""

from __future__ import annotations

import asyncio
import os
import random
import shutil
import socket
import statistics
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from site_cache import use_site_cache

from platoon.envs.harbor import HarborEnvironmentService, HarborSession
from platoon.envs.harbor.builders import pin_images
from platoon.envs.harbor.fetch import fetch, fetch_config
from platoon.envs.harbor.images import default_image_map
from platoon.envs.service import EnvSetupError
from platoon.sandbox import cleanup
from platoon.sandbox.host_network_stacks import HostNetworkStackProvider, beyond_loopback

FIXTURES = Path(__file__).resolve().parents[1] / "envs" / "fixtures" / "harbor-host-network"
TASK = "app-kv-auth"
AUTHENTICATE = FIXTURES / "overlays" / "authenticate"
LOCK_STACKS: dict[str, bool] = {}  # a fetch's default, which locks stacks as runnable
COUNT = int(os.environ.get("PLATOON_TEST_HOST_NETWORK_STACKS", "20"))
CONCURRENCY = int(os.environ.get("PLATOON_TEST_HOST_NETWORK_CONCURRENCY", "8"))
ASK = (
    "import os, sys, urllib.error, urllib.request\n"
    "request = urllib.request.Request(sys.argv[1])\n"
    "if sys.argv[2] == '1':\n"
    "    request.add_header('Authorization', 'Bearer ' + os.environ['KV_TOKEN'])\n"
    "try:\n"
    "    print(urllib.request.urlopen(request, timeout=10).status)\n"
    "except urllib.error.HTTPError as exc:\n"
    "    print(exc.code)\n"
)

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.environ.get("PLATOON_TEST_HOST_NETWORK") != "1", reason="set PLATOON_TEST_HOST_NETWORK=1 to run it"
    ),
    pytest.mark.skipif(
        not (shutil.which("singularity") or shutil.which("apptainer")), reason="no singularity or apptainer on PATH"
    ),
]


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The test's own hpc-sandbox scratch root (on node-local disk where ``HPC_SANDBOX_SCRATCH`` says), and node
    scratch for platoon's own copies and the fallback's stack directories."""

    pytest.importorskip("hpc_sandbox")
    root = Path(os.environ.get("HPC_SANDBOX_SCRATCH") or tmp_path) / f"platoon-host-{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "node"))
    (tmp_path / "node").mkdir()
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def registered() -> set[int]:
    return set(cleanup._live)


def _service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scratch: Path, overlays: list[Path]) -> Any:
    """A host-network service of the fixture, fetched with ``overlays``, its sidecar's image pinned."""

    cache = use_site_cache(monkeypatch, tmp_path / "cache")
    dataset = tmp_path / "dataset"
    shutil.copytree(FIXTURES / "tasks" / TASK, dataset / TASK)
    settings: dict[str, Any] = {
        "dataset": str(dataset),
        "lock": str(tmp_path / "lock.json"),
        "refuse": LOCK_STACKS,
        "overlays": [str(path) for path in overlays],
    }
    config = fetch_config(settings["dataset"], lock=settings["lock"], refuse=LOCK_STACKS, overlays=settings["overlays"])
    lock = fetch(config)
    pin_images(lock, image_map=default_image_map(cache, "singularity"))  # the registry's configuration of the image
    options: dict[str, Any] = {"scratch_root": str(scratch)}
    if os.environ.get("HPC_SANDBOX_IMAGE_CACHE"):
        options["image_cache"] = os.environ["HPC_SANDBOX_IMAGE_CACHE"]
    return HarborEnvironmentService(provider="singularity", provider_options=options, stacks="host_network", **settings)


async def _open(service: HarborEnvironmentService) -> HarborSession:
    return await service.open(await service.get_task(TASK), rollout_id=f"host-{uuid.uuid4().hex[:8]}")


def _processes() -> list[str]:
    """The fixture's stores still running on the host network: processes of this user that run it with a token, which
    only the fallback gives them."""

    found = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes()
            environ = Path(f"/proc/{pid}/environ").read_bytes() if b"/srv/kv/kv.py" in argv else b""
        except OSError:
            continue
        if b"\0KV_TOKEN=" in b"\0" + environ:
            words = argv.replace(b"\0", b" ").decode(errors="replace")
            found.append(f"{pid}: {words[:200]}")
    return found


async def _nothing_left(service: HarborEnvironmentService, scratch: Path, registered: set[int]) -> None:
    """No sandbox in the scratch root once its deletions are done, no stack directory, no process of the fixture's
    store, nothing registered for cleanup, and nothing in node scratch but the record of the task entries it checked."""

    inner = cast(Any, service.provider).inner
    assert await inner.wait_for_deletions(300)
    root = inner.session_root
    entries = list(root.iterdir()) if root.is_dir() else []
    sessions = [
        entry.name
        for entry in entries
        if entry.name.startswith("hpcsb-") or (entry.name == ".trash" and any(entry.iterdir()))
    ]
    fallback = cast(HostNetworkStackProvider, service.stack_provider)
    stacks = list(fallback.root.iterdir()) if fallback.root.exists() else []
    kept = {f"platoon-harbor-{os.getuid()}", fallback.root.name}  # the node's record of checked cache entries
    node = [path for path in Path(os.environ["PLATOON_NODE_SCRATCH"]).iterdir() if path.name not in kept]
    left = sorted(set(cleanup._live) - registered)
    assert (sessions, stacks, _processes(), node, left) == ([], [], [], [], [])


async def test_app_kv_scores_its_oracle_and_its_no_op_through_its_authenticated_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scratch: Path, registered: set[int]
):
    service = _service(tmp_path, monkeypatch, scratch, [AUTHENTICATE])
    await service.check_tasks("train", [TASK])
    session = await _open(service)
    try:
        noop = await session.grade()
    finally:
        await session.close()
    assert (noop.status, noop.reward) == ("scored", 0.0), (noop.reason, noop.metadata)
    session = await _open(service)
    try:
        assert session.stack is not None and session.host_loopback  # its main shares the node's network
        port = int((await session.stack.connect("kv", 6379)).rsplit(":", 1)[1])
        assert beyond_loopback({port}) == []  # the store listens on 127.0.0.1 alone
        url = f"http://127.0.0.1:{port}/"
        main = session.sandbox
        assert main is not None
        asked = {token: (await main.exec(["python3", "-c", ASK, url, token])).stdout.strip() for token in ("0", "1")}
        assert asked == {"0": "401", "1": "200"}, asked  # only its own stack's services hold the token
        solved = await session.run_solution()
        assert solved.exit_code == 0, (solved.stdout[-2000:], solved.stderr[-2000:])
        oracle = await session.grade()
    finally:
        await session.close()
    assert (oracle.status, oracle.reward) == ("scored", 1.0), (oracle.reason, oracle.metadata)
    assert oracle.metadata["services"]["kv"]["run_as"] == "caller"
    await _nothing_left(service, scratch, registered)


async def test_a_sidecar_that_no_overlay_authenticates_is_refused_before_anything_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scratch: Path, registered: set[int]
):
    service = _service(tmp_path, monkeypatch, scratch, [])
    with pytest.raises(EnvSetupError, match="whose sidecar kv no overlay authenticates") as listed:
        await service.check_tasks("train", [TASK])
    assert listed.value.failure_kind == "env_config"
    with pytest.raises(EnvSetupError, match="whose sidecar kv no overlay authenticates") as opened:
        await _open(service)
    assert opened.value.failure_kind == "env_config"
    assert not (scratch / "sessions").exists() or not any(
        entry.name.startswith("hpcsb-") for entry in (scratch / "sessions").iterdir()
    )
    assert sorted(set(cleanup._live) - registered) == []


async def test_many_stacks_some_cancelled_while_they_start_leave_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scratch: Path, registered: set[int]
):
    """``COUNT`` stacks, ``CONCURRENCY`` at a time, one in five cancelled while it starts; the others put a value and
    close.  Their ports are released, and nothing of them is left."""

    service = _service(tmp_path, monkeypatch, scratch, [AUTHENTICATE])
    rng = random.Random(0)
    gate = asyncio.Semaphore(CONCURRENCY)
    opened: list[float] = []
    ports: set[int] = set()

    async def one(index: int) -> str:
        async with gate:
            began = time.monotonic()
            opening = asyncio.ensure_future(_open(service))
            if index % 5 == 0:
                await asyncio.sleep(rng.uniform(0.0, 2.0))
                opening.cancel()
                try:
                    session = await opening
                except asyncio.CancelledError:
                    return "cancelled"
            else:
                session = await opening
            opened.append(time.monotonic() - began)
            try:
                assert session.stack is not None
                ports.add(int((await session.stack.connect("kv", 6379)).rsplit(":", 1)[1]))
                solved = await session.run_solution()
                assert solved.exit_code == 0, solved.stderr[-2000:]
            finally:
                await session.close()
            return "closed"

    results = await asyncio.gather(*(one(index) for index in range(COUNT)))
    assert "closed" in results and "cancelled" in results
    print(
        f"{COUNT} stacks, {CONCURRENCY} at a time: {results.count('closed')} closed, "
        f"{results.count('cancelled')} cancelled; open p50 {statistics.median(opened):.2f} s, "
        f"p99 {sorted(opened)[min(len(opened) - 1, int(0.99 * len(opened)))]:.2f} s, {len(ports)} ports"
    )
    deadline = time.monotonic() + 120
    while True:  # a cancelled start's threads delete what they made
        try:
            await _nothing_left(service, scratch, registered)
            break
        except AssertionError:
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(1.0)
    for port in ports:  # released with their stacks: another stack may claim each again
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as claim:
            claim.bind(f"\0platoon-stack-port-{port}")
