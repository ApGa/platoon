"""An ``ors+http://.../<EnvName>`` name that is a non-default environment of a multi-environment server is refused.

An env server answers every request that names no variant with its default (first) environment; the URI's name
travels only as the deployment.  So ``.../CraftEnv`` on a server whose default is CounterEnv would run CounterEnv
under CraftEnv's name.  ``?variant=`` selects an environment.
"""

from __future__ import annotations

import asyncio

import ors_toys
import pytest
from test_ors_remote import _serve

from platoon.envs.ors import service as service_module
from platoon.envs.service import EnvSetupError, get_service

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _fresh_listings(monkeypatch):
    monkeypatch.setattr(service_module, "_SERVER_ENVIRONMENTS", {}, raising=False)
    monkeypatch.setattr(service_module, "_LISTINGS_IN_FLIGHT", {}, raising=False)


async def test_a_non_default_environment_without_a_variant_is_refused():
    with _serve([ors_toys.CounterEnv, ors_toys.CraftEnv]) as served:
        craft = get_service(f"ors+http://{served.hostport}/CraftEnv")
        for listing in (craft.list_splits, lambda: craft.list_tasks("train")):
            with pytest.raises(EnvSetupError, match=r"select it with \?variant=CraftEnv") as refused:
                await listing()
            assert refused.value.failure_kind == "env_config"
        task = await get_service(f"ors+http://{served.hostport}/srv?variant=CraftEnv").get_task("train:0")
        with pytest.raises(EnvSetupError, match=r"answers requests without a variant with 'counterenv'") as refused:
            await craft.open(task, rollout_id="r")
        assert refused.value.failure_kind == "env_config"
        assert served.ors._active_envs == {}  # refused before any session started


async def test_the_default_environment_a_variant_or_a_deployment_name_still_open():
    with _serve([ors_toys.CounterEnv, ors_toys.CraftEnv]) as served:
        for uri, tasks in (
            (f"ors+http://{served.hostport}/CounterEnv", 3),  # the default environment
            (f"ors+http://{served.hostport}/counterenv", 3),  # names match as the server matches them
            (f"ors+http://{served.hostport}/srv?variant=CraftEnv", 1),  # named by its variant
            (f"ors+http://{served.hostport}/my-deployment", 3),  # a deployment name the server does not list
        ):
            service = get_service(uri, terminal_tools=["submit"])
            assert len(await service.list_tasks("train")) == tasks
            session = await service.open(await service.get_task("train:0"), rollout_id="r")
            await session.close()


async def test_servers_without_a_listing_are_not_checked_and_asked_once(monkeypatch):
    web = pytest.importorskip("aiohttp.web")
    requests = []

    async def no_listing(request):
        requests.append(request.path)
        return web.Response(status=200, text="<html>not a listing</html>")

    app = web.Application()
    app.router.add_get("/list_environments", no_listing)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        url = f"http://127.0.0.1:{runner.addresses[0][1]}"
        service = get_service("ors+http:///CraftEnv")
        for _ in range(3):
            await service.check_server_environment(url)
        assert requests == ["/list_environments"]  # one listing per server and process
        await service.check_server_environment("http://127.0.0.1:9")  # nothing listens: the session reports it
    finally:
        await runner.cleanup()


async def _slow_listing_server(requests: list[str]):
    """An env server whose listing takes a while: still answering while the other sessions of a step open."""

    web = pytest.importorskip("aiohttp.web")

    async def listing(request):
        requests.append(request.path)
        await asyncio.sleep(0.3)
        return web.json_response(["counterenv", "craftenv"])

    app = web.Application()
    app.router.add_get("/list_environments", listing)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    return runner, f"http://127.0.0.1:{runner.addresses[0][1]}"


async def test_sessions_that_open_at_once_share_one_listing():
    requests: list[str] = []
    runner, url = await _slow_listing_server(requests)
    try:
        counter, craft = get_service("ors+http:///CounterEnv"), get_service("ors+http:///CraftEnv")
        await asyncio.gather(*(counter.check_server_environment(url) for _ in range(64)))
        refused = await asyncio.gather(*(craft.check_server_environment(url) for _ in range(8)), return_exceptions=True)
    finally:
        await runner.cleanup()

    assert requests == ["/list_environments"]  # a training step opens a session per rollout, all at once
    assert all(isinstance(error, EnvSetupError) for error in refused)


async def test_a_cancelled_open_leaves_the_listing_to_the_others():
    requests: list[str] = []
    runner, url = await _slow_listing_server(requests)
    try:
        craft = get_service("ors+http:///CraftEnv")
        first = asyncio.ensure_future(craft.check_server_environment(url))
        second = asyncio.ensure_future(craft.check_server_environment(url))
        await asyncio.sleep(0.1)  # both wait on the listing
        first.cancel()
        with pytest.raises(EnvSetupError, match=r"select it with \?variant=CraftEnv"):
            await second
    finally:
        await runner.cleanup()

    assert first.cancelled() and requests == ["/list_environments"]
