"""PlatoonAuthority's own rules, on a gateway of the test's that answers as each test says: every URL becomes an origin
and a bearer, a minted agent is at its parent's origin, a harness that comes back keeps its access and is never minted,
nothing is retried, nothing raised names a URL or a body, an agent's end reads as a run by the finish table, one the
root's close cancelled with no turn of its under way as retired, and a spawned agent's finish returns only once
platoon's spawn of it has."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from platoon_harness.config import Secret
from platoon_harness.drivers import DriverError
from platoon_harness.runtime import AgentOutcome, AgentRef, Gateway, SpawnRequest

import platoon as pt
from platoon.harness.builtin.platoon_harness.authority import HANDOFF, PlatoonAuthority, normalize, result_of

KEY = "root-key-c2VjcmV0"
CHILD = "child-key-dG9rZW4"
BODY = "a body nobody reads"


class Platoon:
    """A gateway that mints by ``mint`` (a status and a JSON body) and lists its models by ``models``, counting what
    it was asked."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.mint: tuple[int, Any] = (201, {})
        self.models: tuple[int, Any] = (200, {"object": "list", "data": [{"id": "toy"}]})
        self.asked: list[tuple[str, str]] = []
        self.origin = ""

    async def answer(self, request: web.Request) -> web.Response:
        self.asked.append((request.method, request.headers.get("Authorization", "")))
        status, body = self.mint if request.method == "POST" else self.models
        return web.json_response(body, status=status)


@pytest.fixture
def platoon() -> Iterator[Platoon]:
    loop = asyncio.new_event_loop()
    gateway = Platoon(loop)
    app = web.Application()
    app.router.add_post("/v1/platoon/agents", gateway.answer)
    app.router.add_get("/v1/models", gateway.answer)
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", 0).start())
    gateway.origin = "http://{}:{}".format(*runner.addresses[0])
    yield gateway
    loop.run_until_complete(runner.cleanup())
    loop.close()


def context(platoon: Platoon, key: str, agent_id: str, spawn: Any = None) -> Any:
    """A platoon context of the agent ``agent_id`` that names the gateway under ``key`` the way platoon's do, and whose
    spawns ``spawn`` runs."""

    return SimpleNamespace(
        agent_id=agent_id,
        endpoint=pt.ModelEndpoint(base_url=f"{platoon.origin}/s/{key}/v1", api_key=key, model="toy"),
        mcp=pt.McpEndpoint(url=f"{platoon.origin}/s/{key}/mcp"),
        agents=SimpleNamespace(spawn=spawn),
    )


def authority(platoon: Platoon) -> PlatoonAuthority:
    """An authority under a root's context, for a team that starts several agents, so that each is minted."""

    return PlatoonAuthority(context(platoon, KEY, "root-id"), None, entry=None)  # type: ignore[arg-type]


def admit(platoon: Platoon, made: PlatoonAuthority, agent: str = "alpha", **spawn: Any) -> Any:
    request = SpawnRequest(agent=agent, role="worker", goal="unlock it", **spawn)
    return platoon.loop.run_until_complete(made.admit(request))


def minted(platoon: Platoon, base_url: str | None = None) -> dict[str, Any]:
    return {"agent_id": "child-id", "api_key": CHILD, "base_url": base_url or f"{platoon.origin}/v1"}


def test_a_minted_agent_is_a_gateway_at_its_parents_origin_under_its_own_key(platoon):
    platoon.mint = (201, minted(platoon))

    admission = admit(platoon, authority(platoon))

    gateway = admission.access
    assert isinstance(gateway, Gateway)
    assert (gateway.origin, gateway.bearer, gateway.model) == (platoon.origin, Secret(CHILD), "toy")
    assert gateway.tools == f"{platoon.origin}/mcp"
    assert dict(gateway.compaction_headers) == {"X-Platoon-Call-Kind": "aux"}
    assert dict(admission.backend) == {"agent_id": "child-id"}
    assert platoon.asked == [("POST", f"Bearer {KEY}"), ("GET", f"Bearer {CHILD}")]


def test_a_minted_origin_must_equal_its_parents(platoon):
    platoon.mint = (201, minted(platoon, f"http://127.0.0.2:1/s/{CHILD}/v1"))

    with pytest.raises(DriverError, match="the mint's base_url is at another origin than its parent's"):
        admit(platoon, authority(platoon))


def test_a_child_of_a_minted_agent_is_minted_on_its_parents_key(platoon):
    platoon.mint = (201, minted(platoon))
    made = authority(platoon)
    admit(platoon, made)

    admit(platoon, made, "w1", parent="agent:alpha")

    assert [key for method, key in platoon.asked if method == "POST"] == [f"Bearer {KEY}", f"Bearer {CHILD}"]


def test_a_mint_is_never_retried(platoon):
    platoon.mint = (500, {"error": {"message": BODY}})

    with pytest.raises(DriverError, match="the mint answered 500"):
        admit(platoon, authority(platoon))
    assert len(platoon.asked) == 1


def test_a_harness_that_comes_back_keeps_its_access_and_is_never_minted(platoon):
    platoon.mint = (201, minted(platoon))
    made = authority(platoon)
    first = admit(platoon, made)

    again = admit(platoon, made, resumed=True)
    with pytest.raises(DriverError, match="beta was never admitted here"):
        admit(platoon, made, "beta", resumed=True)
    platoon.models = (410, {"error": {"message": "This agent is closed"}})
    with pytest.raises(DriverError, match="the gateway's models answered 410"):
        admit(platoon, made, resumed=True)

    assert (again.access, again.backend) == (first.access, first.backend)
    assert again.token != first.token
    assert [method for method, _ in platoon.asked].count("POST") == 1


@pytest.mark.parametrize(
    "url",
    [
        f"http://127.0.0.1:9/s/{KEY}/v1",
        "http://127.0.0.1:9/v1",
        f"http://127.0.0.1:9/s/{KEY}/mcp",
        "https://gateway.example:443/mcp",
    ],
)
def test_every_url_becomes_an_origin_and_a_bearer(url):
    origin, bearer = normalize(url, KEY, "an endpoint")

    assert origin == url.partition("/s/")[0].removesuffix("/v1").removesuffix("/mcp")
    assert bearer == Secret(KEY)


@pytest.mark.parametrize(
    ("url", "mint", "said"),
    [
        pytest.param(f"http://user:{KEY}@HOST/v1", None, "an endpoint holds a user", id="user"),
        pytest.param(f"http://HOST/v1?key={KEY}", None, "an endpoint holds a query or a fragment", id="query"),
        pytest.param(f"http://HOST/v1#{KEY}", None, "an endpoint holds a query or a fragment", id="fragment"),
        pytest.param(f"http://HOST/s/{KEY}/v1/chat", None, "an endpoint has a path other than /v1 or /mcp", id="path"),
        pytest.param(
            f"http://HOST/s/{CHILD}/v1", None, "an endpoint holds another key than its agent's", id="another-key"
        ),
        pytest.param(f"ftp://HOST/s/{KEY}/v1", None, "an endpoint has no http origin", id="scheme"),
        pytest.param(None, (400, {"error": {"message": BODY}}), "the mint answered 400", id="non-201"),
        pytest.param(None, (201, {"api_key": CHILD, "note": BODY}), "the mint's 201 has no agent_id", id="no-agent"),
        pytest.param(None, (201, {"agent_id": "child-id", "note": BODY}), "the mint's 201 has no api_key", id="no-key"),
        pytest.param(None, (201, [BODY]), "the mint answered 201 with no JSON object", id="no-object"),
        pytest.param(
            None,
            (
                201,
                {"agent_id": "child-id", "api_key": CHILD, "base_url": f"http://HOST/s/{CHILD}/v1/chat", "note": BODY},
            ),
            "the mint's base_url has a path other than /v1 or /mcp",
            id="minted-url",
        ),
    ],
)
def test_an_error_the_bridge_raises_names_no_url_or_body(platoon, url, mint, said):
    """A refused URL, a mint answered with another status than 201 and a malformed 201 each raise an error that names a
    field or a status, and never a URL, a key or a body."""

    def at_the_gateway(value: Any) -> Any:
        return json.loads(json.dumps(value).replace("HOST", platoon.origin.removeprefix("http://")))

    with pytest.raises((ValueError, DriverError)) as refused:
        if url is not None:
            normalize(at_the_gateway(url), KEY, "an endpoint")
        else:
            platoon.mint = at_the_gateway(mint)
            admit(platoon, authority(platoon))

    assert str(refused.value).startswith(said)
    assert not [text for text in (KEY, CHILD, BODY, platoon.origin, "127.0.0.1") if text in str(refused.value)]


def test_an_agents_end_reads_as_a_run_by_the_finish_table():
    def run(status: str) -> pt.AgentResult:
        return result_of("alpha", AgentOutcome(status=status, report="it ends\n\nits last words"))

    assert [run(status).status for status in ("finished", "retired", "cancelled", "lost", "failed")] == [
        "completed",
        "completed",
        "cancelled",
        "error",
        "error",
    ]
    assert run("lost").error == "alpha ended lost" and run("finished").error is None
    assert run("retired").final_message == "it ends\n\nits last words"


def test_an_agent_the_close_cancelled_with_no_turn_under_way_is_told_as_retired(platoon):
    made = authority(platoon)
    cancelled = AgentOutcome(status="cancelled", report="alpha cancelled: cancelled by program:platoon")

    async def tell() -> dict[str, AgentOutcome]:
        for agent, under_way in (("alpha", False), ("beta", True)):
            close = made.closing(agent)
            await made.finish(AgentRef(agent=agent, role="worker"), cancelled)
            close.set_result(under_way)
        await made.finish(AgentRef(agent="gamma", role="worker"), cancelled)  # cancelled, and not by the close
        return await made.ended(["alpha", "beta", "gamma"])

    told = platoon.loop.run_until_complete(tell())

    assert {agent: outcome.status for agent, outcome in told.items()} == {
        "alpha": "retired",
        "beta": "cancelled",
        "gamma": "cancelled",
    }
    assert told["alpha"].report == cancelled.report


def test_a_spawned_agents_finish_returns_once_platoons_spawn_of_it_has(platoon):
    """Platoon's spawn of an agent returns some time after its member harness has reported the agent's end, once
    platoon has graded it, closed its session and waited for its children: ``finish`` returns only then, so that the
    session tells the agent's parent nothing of the end while the agent's reservation is still the parent's."""

    told, released = asyncio.Event(), asyncio.Event()
    returned: list[str] = []

    async def spawn(goal: str, *, harness: str, role: str, options: dict[str, Any]) -> Any:
        await made.serve(options[HANDOFF], context(platoon, CHILD, "child-id"))
        told.set()
        await released.wait()  # platoon grades the agent, closes its session and waits for its children
        returned.append(options[HANDOFF])
        return SimpleNamespace(outcome=None)

    made = PlatoonAuthority(context(platoon, KEY, "root-id", spawn), None, entry="lead")  # type: ignore[arg-type]
    ended = AgentOutcome(status="finished", report="w1 finished: unlocked")

    async def tell() -> tuple[bool, list[str]]:
        await made.admit(SpawnRequest(agent="lead", role="lead", goal="unlock it"))
        await made.admit(SpawnRequest(agent="w1", role="worker", goal="unlock it", parent="agent:lead"))
        finishing = asyncio.create_task(made.finish(AgentRef(agent="w1", role="worker"), ended))
        await told.wait()
        for _ in range(16):  # turns of the loop in which a finish that waited on nothing would return
            await asyncio.sleep(0)
        waiting = not finishing.done()
        released.set()
        await finishing
        return waiting, list(returned)

    assert platoon.loop.run_until_complete(tell()) == (True, ["w1"])


def test_finish_is_idempotent(platoon):
    made = authority(platoon)
    told = [AgentOutcome(status="retired", report="first"), AgentOutcome(status="lost", report="again")]

    async def tell() -> dict[str, AgentOutcome]:
        for outcome in told:
            await made.finish(AgentRef(agent="alpha", role="worker"), outcome)
        return await made.ended(["alpha"])

    assert platoon.loop.run_until_complete(tell()) == {"alpha": told[0]}
