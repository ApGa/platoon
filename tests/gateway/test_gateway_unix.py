"""The gateway's Unix socket: a private directory, the same API as the TCP port, and keys bound to the socket's
rollout."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path

import aiohttp
import pytest

from platoon.gateway.server import SOCKET_NAME, GatewayServer, socket_directory


@pytest.fixture
def short_base() -> Iterator[Path]:
    """A base directory short enough for socket paths wherever the test's temporary directory is."""

    base = Path(tempfile.mkdtemp(prefix="pgwt-", dir="/tmp"))
    yield base
    shutil.rmtree(base, ignore_errors=True)


def open_root(gateway, rollout_id: str):
    return gateway.open_agent(rollout_id=rollout_id, trajectory_id=f"{rollout_id}-root", interactions={})


async def get(url: str, *, socket_path: str | None = None, api_key: str | None = None) -> tuple[int, dict]:
    connector = aiohttp.UnixConnector(path=socket_path) if socket_path else None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.get(url, headers=headers) as response:
            return response.status, await response.json(content_type=None)


def test_a_socket_directory_is_private_and_short(short_base: Path):
    directory = socket_directory(short_base)

    assert directory.parent == short_base / f"pgw-{os.getuid()}"
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700 and stat.S_IMODE(directory.parent.stat().st_mode) == 0o700
    assert len(str(directory / SOCKET_NAME)) < 108
    assert socket_directory(short_base) != directory  # one per rollout


def test_a_base_too_long_for_a_socket_falls_back_to_tmp(tmp_path: Path):
    long_base = tmp_path / ("x" * 120)

    directory = socket_directory(long_base)

    try:
        assert directory.parent == Path("/tmp") / f"pgw-{os.getuid()}"
    finally:
        directory.rmdir()


def test_an_open_parent_directory_is_made_private(short_base: Path):
    parent = short_base / f"pgw-{os.getuid()}"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)

    socket_directory(short_base)

    assert stat.S_IMODE(parent.stat().st_mode) == 0o700


def test_a_parent_directory_that_is_a_symlink_is_refused(short_base: Path):
    elsewhere = short_base / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    (short_base / f"pgw-{os.getuid()}").symlink_to(elsewhere)

    with pytest.raises(PermissionError, match="must be a directory of this user's"):
        socket_directory(short_base)

    assert list(elsewhere.iterdir()) == []  # nothing is created through the link


def test_a_parent_directory_of_another_user_is_refused(short_base: Path, monkeypatch: pytest.MonkeyPatch):
    uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: uid + 1)  # the parent the call finds belongs to another user than it

    with pytest.raises(PermissionError, match="must be a directory of this user's"):
        socket_directory(short_base)

    assert list((short_base / f"pgw-{uid + 1}").iterdir()) == []


@pytest.mark.asyncio
async def test_the_socket_serves_the_api_and_takes_only_its_rollouts_keys(gateway, short_base: Path):
    server = GatewayServer(gateway)
    base_url = await server.start(host="127.0.0.1")
    path = str(socket_directory(short_base) / SOCKET_NAME)
    ours, other = open_root(gateway, "rollout-a"), open_root(gateway, "rollout-b")
    try:
        assert await server.start_unix(path, rollout_id="rollout-a") == path == server.socket_path

        status, health = await get("http://localhost/healthz", socket_path=path)
        assert status == 200 and health["status"] == "ok"
        status, models = await get("http://localhost/v1/models", socket_path=path, api_key=ours.api_key)
        assert status == 200 and models["data"][0]["object"] == "model"

        # Another rollout's key is refused on this rollout's socket, but still works on the loopback port.
        status, refused = await get("http://localhost/v1/models", socket_path=path, api_key=other.api_key)
        assert status == 401 and "another rollout" in str(refused)
        status, _ = await get(f"{base_url}/v1/models", api_key=other.api_key)
        assert status == 200

        with pytest.raises(RuntimeError, match="already listens"):
            await server.start_unix(path)
    finally:
        await server.stop()
    assert not os.path.exists(path) and server.socket_path is None


@pytest.mark.asyncio
async def test_a_socket_needs_a_started_server(gateway, tmp_path: Path):
    with pytest.raises(RuntimeError, match="before its Unix socket"):
        await GatewayServer(gateway).start_unix(str(tmp_path / SOCKET_NAME))
