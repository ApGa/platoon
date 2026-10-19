"""What a rollout's harness needs from a Harbor session's sandbox (``SessionNeeds``): the gateway's upstream and its
route, toolchain mounts and the harness's memory; and services that predate ``needs``."""

from __future__ import annotations

from pathlib import Path

import pytest

from platoon.envs.harbor import HarborEnvironmentService
from platoon.envs.service import EnvSetupError, SessionNeeds, open_session, takes_needs
from platoon.sandbox import Mount, SandboxSpec
from platoon.sandbox.process import ProcessProvider
from platoon.task import Task


def _offline_task(make_task) -> Path:
    return make_task(
        "offline",
        task_toml='[environment]\ndocker_image = "ubuntu:24.04"\nnetwork_mode = "no-network"\nmemory_mb = 1024\n',
    )


@pytest.mark.asyncio
async def test_needs_reach_the_sandbox_and_its_route_the_session(tmp_path, make_task, provider: ProcessProvider):
    task_dir = _offline_task(make_task)
    service = HarborEnvironmentService(tmp_path / "tasks", provider=provider)
    toolchain = tmp_path / "toolchain"
    toolchain.mkdir()
    needs = SessionNeeds(
        upstreams={"gateway": f"unix://{tmp_path}/gw/gw.sock"},
        mounts=(Mount(str(toolchain), "/opt/tc"),),
        extra_memory_mb=2048,
    )
    (tmp_path / "gw").mkdir()

    session = await service.open(Task(id=task_dir.name), rollout_id="r1", needs=needs)
    try:
        spec = session.sandbox.spec
        assert spec.network == "none" and spec.memory_mb == 1024 + 2048
        assert spec.upstreams == {"gateway": f"unix://{tmp_path}/gw/gw.sock"}
        assert Mount(str(toolchain), "/opt/tc") in spec.mounts
        assert session.routes["gateway"].startswith("http://127.0.0.1:")

        copy = await session.fork("copy")  # a restored sandbox gets its own route
        try:
            assert copy.routes["gateway"].startswith("http://127.0.0.1:")
            assert copy.routes["gateway"] != session.routes["gateway"]
        finally:
            await copy.close()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_without_needs_a_session_opens_as_before(make_task, tmp_path, provider: ProcessProvider):
    task_dir = _offline_task(make_task)
    service = HarborEnvironmentService(tmp_path / "tasks", provider=provider)

    session = await service.open(Task(id=task_dir.name), rollout_id="r1")
    try:
        assert session.sandbox.spec.upstreams == {} and session.routes == {}
        assert session.sandbox.spec.memory_mb == 1024
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_sandbox_without_python_for_its_route_is_a_harness_setup_error(make_task, tmp_path, provider):
    task_dir = _offline_task(make_task)
    bin_dir = tmp_path / "bin"  # a shell, but no Python
    bin_dir.mkdir()
    for tool in ("sh", "mkdir", "rm", "cat", "sleep"):
        found = next(Path(directory, tool) for directory in ("/bin", "/usr/bin") if Path(directory, tool).exists())
        (bin_dir / tool).symlink_to(found)
    # Mounted at a sandbox path: the process provider translates env values under /tmp, where tmp_path usually is.
    service = HarborEnvironmentService(
        tmp_path / "tasks",
        provider=provider,
        env={"PATH": "/opt/no-python/bin"},
        mounts=[Mount(str(bin_dir), "/opt/no-python/bin")],
    )
    (tmp_path / "gw").mkdir()

    with pytest.raises(EnvSetupError) as raised:
        await service.open(
            Task(id=task_dir.name),
            rollout_id="r1",
            needs=SessionNeeds(upstreams={"gateway": f"unix://{tmp_path}/gw/s"}),
        )

    assert raised.value.failure_kind == "harness_setup" and "no Python 3" in str(raised.value)
    assert list(provider.root.iterdir()) == []


def test_needs_keep_the_specs_own_mounts_and_an_unlimited_memory():
    spec = SandboxSpec(image="x", mounts=[Mount("/site/tc", "/opt/tc")], upstreams={"judge": "http://h:1"})
    needs = SessionNeeds(upstreams={"gateway": "unix:///s/gw.sock"}, mounts=(Mount("/stage/tc", "/opt/tc"),))

    applied = needs.apply(SessionNeeds(extra_memory_mb=512).apply(spec))

    assert applied.mounts == [Mount("/site/tc", "/opt/tc")]  # the configured mount wins
    assert applied.upstreams == {"judge": "http://h:1", "gateway": "unix:///s/gw.sock"}
    assert applied.memory_mb is None  # no limit to grow
    assert spec.mounts == [Mount("/site/tc", "/opt/tc")] and spec.upstreams == {"judge": "http://h:1"}


class OldService:
    """A service written before ``needs``."""

    name = "old"

    def __init__(self) -> None:
        self.opened: list[str] = []

    async def open(self, task: Task, *, rollout_id: str):
        self.opened.append(rollout_id)
        return "session"


class KeywordService(OldService):
    async def open(self, task: Task, *, rollout_id: str, **options):
        self.opened.append(f"{rollout_id}:{sorted(options)}")
        return "session"


@pytest.mark.asyncio
async def test_services_that_predate_needs_open_without_them():
    old, keyword = OldService(), KeywordService()
    needs = SessionNeeds(extra_memory_mb=1)

    assert not takes_needs(old) and takes_needs(keyword)
    assert await open_session(old, Task(id="t"), rollout_id="r", needs=needs) == "session"
    assert await open_session(keyword, Task(id="t"), rollout_id="r", needs=needs) == "session"
    assert await open_session(keyword, Task(id="t"), rollout_id="s") == "session"
    assert old.opened == ["r"] and keyword.opened == ["r:['needs']", "s:[]"]
