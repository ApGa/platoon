"""How each agent of a team's session reaches platoon's model gateway: :class:`PlatoonAuthority` admits it as a platoon
agent of its own, and hears how it ended.

The route follows one fact, whether the agent's parent holds a platoon context:

* **The entry agent**, the one agent a team that starts one starts, holds the root's context.
* **A spawned agent**, whose parent holds a context, is spawned through it (``ctx.agents.spawn`` with the
  ``platoon-harness`` harness), so that platoon forks, reserves, limits and grades it as any subagent; its member
  harness hands its own context back here, which its own children are spawned through.
* **A minted agent**, any other, is minted over HTTP (``POST /v1/platoon/agents``) with its parent's key, or the root's
  when its parent is no agent.

Each admission is a :class:`~platoon_harness.runtime.Gateway` at the gateway's origin under the agent's own key, with
the gateway's MCP tools and ``aux`` as the compaction headers, and names the platoon ``agent_id`` as its backend id.
Nothing is retried: a refusal raises, so the spawn fails, in platoon's words for a spawn and naming a status or a
field for a mint.  Every URL platoon gives becomes an origin and a bearer (:func:`normalize`) before a driver sees it,
and nothing raised here names a URL or a body.
"""

from __future__ import annotations

import asyncio
import re
import secrets
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import aiohttp
from platoon_harness.config import Secret
from platoon_harness.drivers import DriverError
from platoon_harness.runtime import Admission, AgentOutcome, AgentRef, Gateway, SpawnRequest

import platoon as pt

from . import NAME

if TYPE_CHECKING:
    from .bridge import Bridge

# The options the harness reads: the bridge its program gave, and for a member, the agent of ours it runs.
BRIDGE = "bridge"
HANDOFF = "handoff"

# The finish table: how an agent's end reads as the run of a harness.  An agent the root's close cancelled with no turn
# of its under way reads as retired (:meth:`PlatoonAuthority.closing`).
RUN_STATUS: Mapping[str, pt.AgentStatus] = {
    "finished": "completed",
    "retired": "completed",
    "cancelled": "cancelled",
    "lost": "error",
    "failed": "error",
}

# The path of a URL platoon gives an agent, its models (``/v1``) or its tools (``/mcp``), bare or after the agent's key.
_PATH = re.compile(r"(?:/s/(?P<key>[^/]+))?/(?:v1|mcp)")


def normalize(url: str, key: str, field: str) -> tuple[str, Secret]:
    """The origin and the bearer of a URL platoon gives with the agent key ``key``: its path is ``/v1`` or ``/mcp``,
    bare or after ``/s/<key>``.  Refuses, naming ``field`` and never the URL, one with a user, a query, a fragment, any
    other path, or another key in its path."""

    try:
        parts = urlsplit(url)
        user = parts.username is not None or parts.password is not None
    except ValueError:
        raise ValueError(f"{field} is no URL") from None
    path = _PATH.fullmatch(parts.path)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"{field} has no http origin")
    if user:
        raise ValueError(f"{field} holds a user, and a gateway's credential is its bearer alone")
    if parts.query or parts.fragment:
        raise ValueError(f"{field} holds a query or a fragment, which no gateway URL has")
    if path is None:
        raise ValueError(f"{field} has a path other than /v1 or /mcp")
    if path["key"] is not None and not secrets.compare_digest(path["key"], key):
        raise ValueError(f"{field} holds another key than its agent's")
    return f"{parts.scheme}://{parts.netloc}", Secret(key)


def result_of(agent: str, outcome: AgentOutcome) -> pt.AgentResult:
    """An agent's end as the run of the harness that ran it, by the finish table, with its report as the final
    message."""

    status = RUN_STATUS[outcome.status]
    error = None if status == "completed" else f"{agent} ended {outcome.status}"
    return pt.AgentResult(status, final_message=outcome.report, error=error)


class PlatoonAuthority:
    """Admits each agent of one rollout's session as a platoon agent of its own (see the module docstring), and hears
    how each ended: a member harness waits for its agent's end (:meth:`serve`), and the root for every agent's
    (:meth:`ended`), each as it trains.  ``entry`` is the one agent the team starts, which holds the root's context, or
    None for a team that starts several."""

    def __init__(self, root: pt.AgentContext, bridge: Bridge, *, entry: str | None) -> None:
        self._root = root
        self._bridge = bridge
        self._entry = entry
        self._origin, self._key = normalize(root.endpoint.base_url, root.endpoint.api_key, "the root's endpoint")
        self._headers = root.endpoint.headers("aux")
        self._contexts: dict[str, pt.AgentContext] = {}  # each agent that holds a platoon context
        self._admitted: dict[str, Admission] = {}
        self._handoffs: dict[str, asyncio.Future[pt.AgentContext]] = {}
        self._spawns: dict[str, asyncio.Task[pt.SubagentResult]] = {}
        self._ends: dict[str, asyncio.Future[AgentOutcome]] = {}
        self._closes: dict[str, asyncio.Future[bool]] = {}  # each agent the close cancels: was a turn of its under way

    async def admit(self, spawn: SpawnRequest) -> Admission:
        """Admit an agent by its route, or again, for a harness that comes back, on the access it had, while its platoon
        agent is open: a resume never mints."""

        if spawn.resumed:
            return await self._resumed(spawn.agent)
        parent = spawn.parent.removeprefix("agent:") if spawn.parent and spawn.parent.startswith("agent:") else None
        if spawn.parent is None and spawn.agent == self._entry:
            gateway, agent_id = self._held(spawn.agent, self._root)
        elif parent in self._contexts:
            gateway, agent_id = self._held(spawn.agent, await self._spawned(spawn, self._contexts[parent]))
        else:
            minting = self._key if parent not in self._admitted else self._bearer(parent)
            gateway, agent_id = await self._minted(spawn, minting)
        admission = Admission(token=Secret(secrets.token_urlsafe(32)), access=gateway, backend={"agent_id": agent_id})
        self._admitted[spawn.agent] = admission
        return admission

    async def finish(self, agent: AgentRef, outcome: AgentOutcome) -> None:
        """Hear how an agent ended, and return once platoon's spawn of it, if it was spawned, has returned, which is
        after platoon has graded it and its own children have ended, so that its reservation is off its parent: the
        session hands the parent nothing of the end before then.  A second telling is the first."""

        ended = self._end(agent.agent)
        if not ended.done():
            ended.set_result(outcome)
        if (spawned := self._spawns.get(agent.agent)) is not None:
            await asyncio.wait({spawned})

    async def serve(self, agent: str, ctx: pt.AgentContext) -> AgentOutcome:
        """Hand a member harness's context to the admission of ``agent``, which waits for it, and wait for that agent's
        end.  A handoff is served once: a second run under it refuses, as that agent's end is not its own."""

        handoff = self._handoffs[agent]
        if handoff.done():
            raise DriverError(f"{agent}'s handoff was served, and a run under it again would report its end as its own")
        handoff.set_result(ctx)
        return await self._trained(agent)

    async def ended(self, agents: Iterable[str]) -> dict[str, AgentOutcome]:
        """How each of ``agents`` ended, once each has, as it trains."""

        return {agent: await self._trained(agent) for agent in agents}

    def closing(self, agent: str) -> asyncio.Future[bool]:
        """What the root's close sets, once it has cancelled ``agent``, to whether a turn of its was under way when it
        ended.  Until then the agent's end is told to nobody, and with none under way it is told as retired: every turn
        it took is whole, and platoon drops a cancelled run's tokens.  The session's record keeps its cancel."""

        close = self._closes[agent] = asyncio.get_running_loop().create_future()
        return close

    async def grades(self) -> dict[str, pt.Outcome]:
        """The grade platoon gave each spawned agent it graded, once every spawn has returned."""

        await asyncio.gather(*self._spawns.values(), return_exceptions=True)
        graded = {}
        for agent, spawned in self._spawns.items():
            if not spawned.cancelled() and spawned.exception() is None and (grade := spawned.result().outcome):
                graded[agent] = grade
        return graded

    async def _trained(self, agent: str) -> AgentOutcome:
        """How ``agent`` ended, but retired where the close cancelled it with no turn of its under way."""

        outcome = await self._end(agent)
        close = self._closes.get(agent)
        if outcome.status == "cancelled" and close is not None and not await close:
            return outcome.model_copy(update={"status": "retired"})
        return outcome

    def _end(self, agent: str) -> asyncio.Future[AgentOutcome]:
        if agent not in self._ends:
            self._ends[agent] = asyncio.get_running_loop().create_future()
        return self._ends[agent]

    def _bearer(self, agent: str) -> Secret:
        access = self._admitted[agent].access
        assert isinstance(access, Gateway)
        return access.bearer

    def _held(self, agent: str, ctx: pt.AgentContext) -> tuple[Gateway, str]:
        """The gateway of an agent that holds the platoon context ``ctx``: its endpoint and its tools, normalized."""

        endpoint = ctx.endpoint
        origin, bearer = normalize(endpoint.base_url, endpoint.api_key, "an agent's endpoint")
        if ctx.mcp is not None and normalize(ctx.mcp.url, endpoint.api_key, "an agent's tools")[0] != origin:
            raise ValueError("an agent's tools are at another origin than its endpoint")
        self._contexts[agent] = ctx
        return self._gateway(origin, bearer, endpoint.model), ctx.agent_id

    async def _spawned(self, spawn: SpawnRequest, parent: pt.AgentContext) -> pt.AgentContext:
        """Spawn an agent through its parent's platoon context, as a task kept here, and give the context its member
        harness hands back; a launch platoon refuses raises with platoon's words."""

        handoff = self._handoffs[spawn.agent] = asyncio.get_running_loop().create_future()
        options = {BRIDGE: self._bridge, HANDOFF: spawn.agent}
        spawned = parent.agents.spawn(spawn.goal, harness=NAME, role=spawn.role, options=options)
        task = self._spawns[spawn.agent] = asyncio.create_task(spawned)
        await asyncio.wait({handoff, task}, return_when=asyncio.FIRST_COMPLETED)
        if handoff.done():
            return handoff.result()
        raise DriverError(task.result().message)

    async def _minted(self, spawn: SpawnRequest, key: Secret) -> tuple[Gateway, str]:
        """Mint an agent with ``key``.  Its key is a secret before anything else reads the reply, its URL must be at the
        gateway's origin, and its model is the one the gateway names for it."""

        body = {"role": spawn.role, "goal": spawn.goal}
        reply = await self._ask("POST", "/v1/platoon/agents", key, body, expect=201, what="the mint")
        minted = reply.pop("api_key", None)
        if not isinstance(minted, str) or not minted:
            raise DriverError("the mint's 201 has no api_key")
        bearer, agent_id, base_url = Secret(minted), reply.get("agent_id"), reply.get("base_url")
        if not isinstance(agent_id, str) or not agent_id:
            raise DriverError("the mint's 201 has no agent_id")
        if not isinstance(base_url, str):
            raise DriverError("the mint's 201 has no base_url")
        if normalize(base_url, bearer.reveal(), "the mint's base_url")[0] != self._origin:
            raise DriverError("the mint's base_url is at another origin than its parent's")
        return self._gateway(self._origin, bearer, await self._model(bearer)), agent_id

    async def _resumed(self, agent: str) -> Admission:
        """The access an agent was admitted on, with a token of its own, while its platoon agent is open."""

        admitted = self._admitted.get(agent)
        if admitted is None:
            raise DriverError(f"{agent} was never admitted here, and a harness that comes back is never minted")
        assert isinstance(admitted.access, Gateway)
        await self._model(admitted.access.bearer)
        return admitted.model_copy(update={"token": Secret(secrets.token_urlsafe(32))})

    async def _model(self, bearer: Secret) -> str:
        """The model the gateway names for the agent ``bearer`` is the key of: it answers 410 once that agent closes."""

        listed = await self._ask("GET", "/v1/models", bearer, None, expect=200, what="the gateway's models")
        models = listed.get("data")
        if not isinstance(models, list) or not models or not isinstance(models[0], dict):
            raise DriverError("the gateway's models name no model")
        model = models[0].get("id")
        if not isinstance(model, str) or not model:
            raise DriverError("the gateway's models name no model")
        return model

    async def _ask(self, method: str, path: str, key: Secret, body: Any, *, expect: int, what: str) -> dict[str, Any]:
        """One request to the gateway under ``key``, never retried: the JSON object it answers with status
        ``expect``, else a refusal that names ``what`` and the status, and never the URL or the body."""

        headers = {"Authorization": f"Bearer {key.reveal()}"}
        try:
            async with aiohttp.ClientSession() as http:
                async with http.request(method, f"{self._origin}{path}", json=body, headers=headers) as answer:
                    if answer.status != expect:
                        raise DriverError(f"{what} answered {answer.status}")
                    reply = await answer.json(content_type=None)
        except aiohttp.ClientError as error:
            raise DriverError(f"{what} gave no answer ({type(error).__name__})") from None
        except ValueError:
            raise DriverError(f"{what} answered {expect} with no JSON") from None
        if not isinstance(reply, dict):
            raise DriverError(f"{what} answered {expect} with no JSON object")
        return reply

    def _gateway(self, origin: str, bearer: Secret, model: str) -> Gateway:
        return Gateway(
            origin=origin, bearer=bearer, model=model, tools=f"{origin}/mcp", compaction_headers=self._headers
        )
