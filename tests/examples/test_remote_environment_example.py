"""``docs/tutorials/remote-environments.md``: the order-packing environment on a local ORS env server
(``examples/custom-environment/serve.py``), reached through ``remote.yaml`` and an
``OPENREWARD_SESSION_URLS_<LABEL>`` pool."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time
from collections.abc import Iterator

import pytest
from example_runs import load_config, rollout

from platoon.envs import get_service
from platoon.envs.ors import pool_env_var


@contextlib.contextmanager
def serving(environments: list[type]) -> Iterator[str]:
    """The openreward SDK's ``Server`` for ``environments`` on a free local port; yields its URL."""

    uvicorn = pytest.importorskip("uvicorn")
    from openreward.environments import Server

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(Server(environments).app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("the ORS server did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


def tool_call(name: str, **arguments) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


@pytest.fixture
def server(example) -> Iterator[str]:
    example("custom-environment")
    from order_packing import OrderPackingEnv

    with serving([OrderPackingEnv]) as url:
        yield url


def test_the_pool_variable_is_named_after_the_label():
    assert pool_env_var("OrderPackingEnv") == "OPENREWARD_SESSION_URLS_ORDERPACKINGENV"
    assert pool_env_var("order-packing") == "OPENREWARD_SESSION_URLS_ORDER_PACKING"


def test_remote_hints_come_from_the_service_options(server):
    async def tools(**options) -> dict[str, tuple[bool, bool]]:
        service = get_service(f"ors+http://{server.removeprefix('http://')}/OrderPackingEnv", **options)
        session = await service.open(await service.get_task("train:0"), rollout_id="hints")
        try:
            return {spec.name: (spec.terminal, spec.read_only) for spec in await session.tools()}
        finally:
            await session.close()

    # The class's platoon_terminal_tools and platoon_read_only_tools do not cross the wire...
    assert asyncio.run(tools()) == {"done": (False, False), "look": (False, False), "pack": (False, False)}
    # ...so the service options name them.
    hinted = asyncio.run(tools(terminal_tools=["done"], read_only_tools=["look"]))
    assert hinted == {"done": (True, False), "look": (False, True), "pack": (False, False)}
    # ...or env_class names the class, whose hints then apply as in process.
    assert asyncio.run(tools(env_class="order_packing:OrderPackingEnv")) == hinted


def test_remote_yaml_rolls_out_against_the_pool(server, policy_gateway, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENREWARD_SESSION_URLS_ORDERPACKINGENV", server)

    def policy(message: str) -> str:
        if message.startswith("Pack this order"):
            return tool_call("pack", item="apple", quantity=2)
        if "Packed 2 apple" in message:
            return tool_call("pack", item="bread", quantity=1)
        return "Packed 2 apple and 1 bread."

    policy_gateway(policy)
    collection = asyncio.run(rollout(load_config("remote.yaml"), "train:0", tmp_path))

    (root,) = collection.trajectories.values()
    assert (root.reward, root.misc["outcome"]["status"]) == (1.0, "scored")
    assert [call["name"] for call in root.misc["tool_calls"]] == ["pack", "pack", "done"]


def test_max_tool_calls_ends_the_episode_with_the_last_reward(server):
    async def play():
        service = get_service(f"ors+http://{server.removeprefix('http://')}/OrderPackingEnv", max_tool_calls=2)
        session = await service.open(await service.get_task("train:0"), rollout_id="cap")
        try:
            await session.call_tool("look", {})
            await session.call_tool("look", {})
            capped = await session.call_tool("pack", {"item": "apple"})
            return capped, await session.grade()
        finally:
            await session.close()

    capped, outcome = asyncio.run(play())
    assert capped.finished and capped.is_error
    assert json.loads(capped.as_text()) == {"bridge_error": "max_tool_calls_exceeded", "finished": True, "reward": None}
    assert (outcome.status, outcome.reward, outcome.reason) == ("scored", 0.0, "max_tool_calls")


def test_bridge_json_results_are_one_json_text_block(server):
    async def play():
        service = get_service(
            f"ors+http://{server.removeprefix('http://')}/OrderPackingEnv", result_format="bridge_json"
        )
        session = await service.open(await service.get_task("train:0"), rollout_id="json")
        try:
            return await session.call_tool("pack", {"item": "apple", "quantity": 2})
        finally:
            await session.close()

    result = asyncio.run(play())
    assert not result.is_error and len(result.content) == 1
    assert json.loads(result.as_text()) == {
        "finished": False,
        "reward": None,
        "text": "Packed 2 apple. The box holds 2 apple.",
        "blocks": ["Packed 2 apple. The box holds 2 apple."],
    }
