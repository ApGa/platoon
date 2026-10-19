"""ORS environments on env servers (``ors+http://``), against a real SDK ``Server`` on a local port.

The server runs the toy environments in a uvicorn thread; the service talks to
it through the openreward client exactly as it would to a production pool.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import socket
import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace

import ors_toys
import pytest

from platoon.envs.ors import OrsEnvironmentService, OrsSession, select_session_url
from platoon.envs.ors import service as service_module
from platoon.envs.service import EnvSetupError, get_service
from platoon.task import Task

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def server() -> Iterator[SimpleNamespace]:
    with _serve([ors_toys.CounterEnv, ors_toys.CraftEnv]) as served:
        yield served


@contextlib.contextmanager
def _serve(environments: list[type]) -> Iterator[SimpleNamespace]:
    """An SDK ``Server`` for ``environments`` on a free local port, in a uvicorn thread."""

    uvicorn = pytest.importorskip("uvicorn")
    from openreward.environments import Server

    ors = Server(environments)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    uv = uvicorn.Server(uvicorn.Config(ors.app, host="127.0.0.1", port=port, log_level="warning"))
    service_module._SERVER_ENVIRONMENTS.pop(f"http://127.0.0.1:{port}", None)  # an earlier server's, on this port
    thread = threading.Thread(target=uv.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not uv.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("The ORS test server did not start")
        time.sleep(0.05)
    try:
        yield SimpleNamespace(url=f"http://127.0.0.1:{port}", hostport=f"127.0.0.1:{port}", ors=ors)
    finally:
        uv.should_exit = True
        thread.join(timeout=10)
        sock.close()


def _counter(server: SimpleNamespace, **options) -> OrsEnvironmentService:
    options.setdefault("terminal_tools", ["submit"])
    options.setdefault("read_only_tools", ["peek"])
    return get_service(f"ors+http://{server.hostport}/CounterEnv", **options)


async def _open(service: OrsEnvironmentService, task_id: str = "train:0", rollout_id: str = "rollout-1") -> OrsSession:
    return await service.open(await service.get_task(task_id), rollout_id=rollout_id)


async def test_remote_tasks_list_by_split_and_resolve_by_name(server):
    service = _counter(server)

    assert service.is_remote and service.name == "CounterEnv"
    assert await service.list_splits() == ["train", "test"]
    assert await service.list_tasks("test") == ["test:0", "test:1", "test:2"]
    assert (await service.get_task("train-2")).misc["ors"] == {"split": "train", "index": 2, "name": "train-2"}
    assert (await service.get_task("test:1")).misc["ors"] == {"split": "test", "index": 1, "name": None}
    assert await _counter(server, task_names=["train-1"]).list_tasks("train") == ["train:1"]


async def test_remote_task_names_let_an_entry_leave_tasks_out_by_name(server):
    from platoon.train.auto import AutoDataset
    from platoon.train.components import EnvironmentConfig

    names = {"test:0": "test-0", "test:1": "test-1", "test:2": "test-2"}
    assert await _counter(server).list_task_names("test") == names
    entry = EnvironmentConfig(
        service=f"ors+http://{server.hostport}/CounterEnv",
        exclude_tasks=["train-1", "test-2"],  # test-2: another split
    )
    config = SimpleNamespace(environments=[entry], train=SimpleNamespace(batch_size=None, task_sampler="auto"))
    assert list(AutoDataset.from_config(config, "train")["task_id"]) == ["train:0", "train:2"]


async def test_a_remote_session_plays_and_grades_like_an_in_process_one(server):
    service = _counter(server)
    session = await _open(service, "train:1")
    try:
        assert session.instruction == "Reach 3." and session.url == server.url
        tools = {spec.name: spec for spec in await session.tools()}
        assert tools["add"].input_schema["properties"]["amount"]["type"] == "integer"
        assert tools["submit"].terminal and tools["peek"].read_only
        assert tools["search"].meta == {"openhands.dev/tool-routing": {"lane": "search"}}  # from the schema extension

        assert (await session.call_tool("add", {"amount": 3})).metadata == {"total": 3}
        peek = await session.call_tool("peek", {})
        assert peek.content[1] == {"type": "image", "data": "aGk=", "mimeType": "image/png"}
        invalid = await session.call_tool("add", {"amount": "many"})
        assert invalid.is_error and invalid.metadata["error_reason"] in (
            "input_validation",
            "invalid_call",
        )  # 0.1.125 sends no reason
        result = await session.call_tool("submit", {"answer": 3})
        assert (result.finished, result.reward) == (True, 1.0)
        outcome = await session.grade()
    finally:
        await session.close()
    assert (outcome.status, outcome.reward) == ("scored", 1.0)
    assert server.ors._active_envs == {}  # the server tore the session down


async def test_a_remote_tool_that_raised_is_the_policys_failed_call_on_either_client(server):
    """As in process: the model sees the error, and an episode that never submits scores 0.  0.1.163's client ends
    a session whose tool raised (ToolFailed), so later calls are refused; 0.1.125's keeps it.  The grade is the same."""

    session = await _open(_counter(server))
    try:
        raised = await session.call_tool("explode", {})
        assert raised.is_error and "boom" in raised.as_text()
        assert raised.metadata["error_reason"] == "tool_raised"
        again = await session.call_tool("add", {"amount": 1})
        dead = session._lease.dead is not None  # 0.1.163 kills a session whose tool raised; 0.1.125 keeps it
        if dead:
            assert again.is_error and again.metadata == {"error_reason": "dead"}
        else:
            assert again.as_text() == "total=1"
        outcome = await session.grade()
    finally:
        await session.close()
    assert (outcome.status, outcome.reward, outcome.reason) == ("scored", 0.0, "no_submit")
    assert outcome.metadata["errors"][0].startswith("explode: ")


@pytest.mark.parametrize("tool_name", ["wait_inner", "check"])
async def test_a_tool_that_raised_with_an_empty_message_is_still_a_tool_that_raised(tool_name):
    """An inner ``TimeoutError()`` or a bare ``assert`` reaches the client as "Unknown SSE error": a tool that raised,
    graded like one that raised with a message, never as a transport failure."""

    with _serve([ors_toys.SilentRaiseEnv]) as served:
        session = await _open(get_service(f"ors+http://{served.hostport}/SilentRaiseEnv", terminal_tools=["submit"]))
        try:
            raised = await session.call_tool(tool_name, {})
            again = await session.call_tool("add", {"amount": 1})
            dead = session._lease.dead is not None  # 0.1.163's ToolFailed ends the session, as for any raise
            outcome = await session.grade()
        finally:
            await session.close()
    assert raised.is_error and raised.metadata["error_reason"] == "tool_raised"
    if dead:
        assert again.metadata == {"error_reason": "dead"}
    else:
        assert again.as_text() == "total=1"
    assert (outcome.status, outcome.reward, outcome.reason) == ("scored", 0.0, "no_submit")


async def _fail_and_grade(service, tool_name: str) -> tuple[str, tuple]:
    """The ``error_reason`` of calling ``tool_name``, and the grade of the session that never submitted."""

    session = await _open(service)
    try:
        raised = await session.call_tool(tool_name, {})
        assert raised.is_error
        outcome = await session.grade()
    finally:
        await session.close()
    reason = outcome.reason if outcome.status == "scored" else None  # an infra reason quotes the mode's exception
    return raised.metadata["error_reason"], (outcome.status, outcome.reward, reason, outcome.failure_kind)


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("explode", ("tool_raised", ("scored", 0.0, "no_submit", None))),
        ("lose_sandbox", ("env_infra", ("error", None, None, "env_error"))),
        ("exhaust_retries", ("env_infra", ("error", None, None, "env_error"))),
        # httpx's ConnectError is a TransportError, but not the SDK's: an env server sends its text, which no SDK
        # error has, so it is the tool's own failure in process too.
        ("fetch_page", ("tool_raised", ("scored", 0.0, "no_submit", None))),
    ],
)
async def test_a_tool_failure_is_classified_the_same_in_process_and_on_an_env_server(tool_name, expected):
    """The same env class, in process and served, on either openreward client: a tool that raised is the policy's
    failed call (``tool_raised``), and one that raised the SDK's own session or transport error, which the env's
    backing service gave it, is infrastructure (``env_infra``), never a score."""

    with _serve([ors_toys.BackendFailureEnv]) as served:
        remote = get_service(f"ors+http://{served.hostport}/BackendFailureEnv", terminal_tools=["submit"])
        in_process = get_service("python://ors_toys:BackendFailureEnv")
        assert await _fail_and_grade(in_process, tool_name) == expected
        assert await _fail_and_grade(remote, tool_name) == expected


async def test_a_call_whose_task_the_server_lost_is_a_transport_failure(server, monkeypatch):
    """A resumed call whose task the server no longer knows lost its reply in transport: an infra error, though
    the session lives on."""

    session = await _open(_counter(server))
    remote = session._lease.backend.session  # type: ignore[union-attr]

    async def lost(name, arguments):
        raise RuntimeError(f"(environment session sid={remote.sid}) unknown task_id")  # as 0.1.125 raises it

    try:
        with monkeypatch.context() as patch:
            patch.setattr(remote, "call_tool", lost)
            failed = await session.call_tool("add", {"amount": 1})
        assert (await session.call_tool("add", {"amount": 1})).as_text() == "total=1"
        outcome = await session.grade()
    finally:
        await session.close()
    assert failed.is_error and failed.metadata["error_reason"] == "transport"
    assert (outcome.status, outcome.failure_kind) == ("error", "env_error")


async def _delete_on_the_server(session: OrsSession) -> None:
    """Delete the session's env server-side, as a server reaper or an operator would."""

    import aiohttp

    remote = session._lease.backend.session  # type: ignore[union-attr]
    async with aiohttp.ClientSession() as client:
        async with client.post(f"{session.url}/delete", headers={"X-Session-ID": remote.sid}) as response:
            response.raise_for_status()


async def test_a_session_deleted_on_the_server_is_dead_and_grades_as_an_infra_error(server):
    session = await _open(_counter(server))
    try:
        assert (await session.call_tool("add", {"amount": 1})).as_text() == "total=1"
        await _delete_on_the_server(session)
        gone = await session.call_tool("add", {"amount": 1})
        assert gone.is_error and gone.metadata["error_reason"] == "dead"
        assert session._lease.dead is not None
        refused = await session.call_tool("add", {"amount": 1})
        assert refused.metadata == {"error_reason": "dead"}
        outcome = await session.grade()
    finally:
        await session.close()
    assert (outcome.status, outcome.failure_kind) == ("error", "env_error")


async def test_remote_bridge_json_errors_show_the_plain_text(server):
    session = await _open(_counter(server, result_format="bridge_json"))
    try:
        raised = (await session.call_tool("explode", {})).as_text()
        assert raised.startswith("Error executing tool explode: ") and "boom" in raised
        assert "RuntimeError" not in raised and "ToolFailed" not in raised
        await _delete_on_the_server(session)  # dead on either SDK from here
        first = (await session.call_tool("add", {"amount": 1})).as_text()
        again = (await session.call_tool("add", {"amount": 1})).as_text()
    finally:
        await session.close()
    assert first.startswith("Error executing tool add: ")
    plain = first.removeprefix("Error executing tool add: ")
    assert not plain.startswith(("ClientResponseError", "RuntimeError", "SessionTerminatedError", "ToolFailed"))
    assert again == first  # a dead session repeats the plain text


async def test_remote_max_tool_calls_and_bridge_records(server, tmp_path):
    service = _counter(server, max_tool_calls=1, result_format="bridge_json", bridge_tools=True, log_dir=tmp_path)
    session = await _open(service, rollout_id="capped")
    try:
        added = json.loads((await session.call_tool("add", {"amount": 1})).as_text())
        assert added == {
            "finished": False,
            "reward": None,
            "text": "total=1",
            "blocks": ["total=1"],
            "metadata": {"total": 1},
        }
        capped = await session.call_tool("add", {"amount": 1})
        assert not capped.is_error and json.loads(capped.as_text()) == {
            "bridge_error": "max_tool_calls_exceeded",
            "finished": True,
            "reward": None,
        }
        status = json.loads((await session.call_tool("get_status", {})).as_text())
        assert (status["turn"], status["finished"]) == (1, True)
        outcome = await session.grade()
    finally:
        await session.close()
    assert (outcome.status, outcome.reward, outcome.reason) == ("scored", 0.0, "max_tool_calls")
    types = [
        json.loads(line)["type"] for line in (tmp_path / "capped" / "bridge_events.jsonl").read_text().splitlines()
    ]
    assert types == ["session_started", "tool_call", "tool_result", "max_tool_calls_exceeded", "session_closing"]
    session_started = json.loads((tmp_path / "capped" / "bridge_events.jsonl").read_text().splitlines()[0])
    assert session_started["session_url"] == server.url


async def test_remote_bridge_tools_report_bridge_json(tmp_path):
    from openreward.api.environments.client import convert_tool_response

    with _serve([ors_toys.StalePromptEnv]) as served:
        service = get_service(
            f"ors+http://{served.hostport}/StalePromptEnv",
            label="toolathlongym",  # the pool label, which is not the env's name
            prompt_filter="ors_toys:drop_stale_completion_line",
            bridge_tools=True,
            result_format="bridge_json",
            log_dir=tmp_path,
        )
        session = await _open(service, "train:1", rollout_id="named")
        try:
            assert session.ref.name == "train-1" and session.task.misc["ors"]["name"] == "train-1"
            assert session.instruction == "Reach 3.\nThen submit."
            task = json.loads((await session.call_tool("get_task", {})).as_text())
            status = json.loads((await session.call_tool("get_status", {})).as_text())
        finally:
            await session.close()
    assert (task["task_name"], task["task_index"]) == ("train-1", 1)
    assert task["prompt"] == "Reach 3.\nThen submit." and ors_toys.STALE_COMPLETION_LINE not in task["prompt"]
    assert task["environment_tools"] == convert_tool_response(
        ors_toys.StalePromptEnv.list_tools().model_dump(), "openai"
    )
    assert (status["env"], status["task_name"]) == ("StalePromptEnv", "train-1")
    records = [json.loads(line) for line in (tmp_path / "named" / "bridge_events.jsonl").read_text().splitlines()]
    assert (records[0]["env"], records[0]["task_name"]) == ("StalePromptEnv", "train-1")
    state = json.loads((tmp_path / "named" / "bridge_state.json").read_text())
    assert (state["env"], state["task_name"]) == ("StalePromptEnv", "train-1")


async def test_remote_forks_share_or_reopen_and_refuse_copies(server):
    service = _counter(server)
    parent = await _open(service)
    try:
        shared = await parent.fork("shared", task=Task(id="child", goal="Help."))
        fresh = await parent.fork("fresh")
        try:
            await shared.call_tool("add", {"amount": 2})
            assert (await parent.call_tool("add", {"amount": 0})).as_text() == "total=2"  # one remote session
            assert (await fresh.call_tool("add", {"amount": 0})).as_text() == "total=0"  # a new one
            assert len(server.ors._active_envs) == 2
            with pytest.raises(NotImplementedError, match="cannot be copied"):
                await parent.fork("copy")
        finally:
            await shared.close()
            await fresh.close()
        assert len(server.ors._active_envs) == 1  # the parent still holds the shared session
    finally:
        await parent.close()
    assert server.ors._active_envs == {}


async def test_variants_select_an_environment_on_a_multi_env_server(server):
    service = get_service(f"ors+http://{server.hostport}/craft?variant=craftenv")
    session = await _open(service)
    try:
        assert session.instruction == "craft 3 plank"
        for _ in range(3):
            await session.call_tool("craft", {"item": "plank"})
        await session.call_tool("done", {})
        assert (await session.grade()).reward == 1.0
    finally:
        await session.close()


async def test_pools_shard_sessions_by_rollout_id(server, monkeypatch):
    dead = "http://127.0.0.1:9"  # nothing listens on the discard port
    pool = [dead, server.url]
    live_id = next(f"r{index}" for index in range(100) if select_session_url(pool, f"r{index}") == server.url)
    dead_id = next(f"r{index}" for index in range(100) if select_session_url(pool, f"r{index}") == dead)
    service = get_service(f"ors+http://127.0.0.1:9,{server.hostport}/CounterEnv", close_timeout_s=5)
    session = await _open(service, rollout_id=live_id)
    try:
        assert session.url == server.url
    finally:
        await session.close()
    with pytest.raises(EnvSetupError) as refused:
        await _open(service, rollout_id=dead_id)
    assert refused.value.failure_kind == "env_connect"

    monkeypatch.setenv("OPENREWARD_SESSION_URLS_COUNTER_POOL", server.hostport)
    from_env = get_service("ors+http:///CounterEnv", label="counter-pool")
    assert from_env.session_urls() == [server.url]
    session = await _open(from_env, rollout_id="any")
    try:
        assert session.url == server.url
    finally:
        await session.close()


async def test_tool_meta_from_a_server_passes_through_with_the_terminal_tool(monkeypatch):
    """A server that sends ``_meta`` on its tools (keys the SDK's ToolSpec rejects) still lists, calls and grades."""

    environments = pytest.importorskip("openreward.environments")
    server_module = pytest.importorskip("openreward.environments.server")
    has_terminal = hasattr(environments, "terminal")

    class MetaQuizEnv(ors_toys.CounterEnv):
        if has_terminal:

            @environments.terminal
            @environments.tool
            async def respond(self, params: ors_toys.QuizAnswer) -> ors_toys.ToolOutput:
                """Answer the quiz."""
                reward = float(params.answer == "4")
                return environments.ToolOutput(blocks=[environments.TextBlock(text="ok")], reward=reward, finished=True)

    listed = server_module.list_session_tools

    class _WithMeta:
        def __init__(self, listing):
            self.listing = listing

        def model_dump(self, *args, **kwargs):
            dump = self.listing.model_dump(*args, **kwargs)
            for tool in dump["tools"]:
                tool["_meta"] = {"provider.dev/source": tool["name"]}
            return dump

    async def list_with_meta(env, toolset):
        return _WithMeta(await listed(env, toolset))

    monkeypatch.setattr(server_module, "list_session_tools", list_with_meta)
    with _serve([MetaQuizEnv]) as served:
        service = get_service(f"ors+http://{served.hostport}/MetaQuizEnv", terminal_tools=["submit"])
        session = await _open(service)
        try:
            tools = {spec.name: spec for spec in await session.tools()}
            assert tools["add"].meta == {"provider.dev/source": "add"} and tools["submit"].terminal
            assert (await session.call_tool("add", {"amount": 2})).metadata == {"total": 2}
            if has_terminal:
                assert tools["respond"].terminal
                assert tools["respond"].meta == {"openreward.dev/terminal-tool": {"arg": "answer"}}
                assert (await session.call_tool("respond", {"answer": "4"})).finished
            else:
                assert (await session.call_tool("submit", {"answer": 2})).finished
            outcome = await session.grade()
        finally:
            await session.close()
    assert (outcome.status, outcome.reward) == ("scored", 1.0)


async def test_a_remote_env_without_a_terminal_tool_is_reported_and_terminal_tools_fixes_it(monkeypatch, caplog):
    """Over ors+http only @terminal is on the wire: exclusive tools and platoon_terminal_tools are not."""

    environments = pytest.importorskip("openreward.environments")
    from platoon.gateway.mcp import filter_tools

    monkeypatch.setattr(service_module, "_WARNED_NO_TERMINAL_TOOL", set())
    if "exclusive" in inspect.signature(environments.tool).parameters:

        class ExamEnv(ors_toys.CounterEnv):
            @environments.tool(exclusive=True)
            async def hand_in(self, params: ors_toys.HandIn) -> ors_toys.ToolOutput:
                """Hand in the final total; this ends the episode."""
                reward = float(params.answer == self.total == self.task_spec["target"])
                return environments.ToolOutput(blocks=[environments.TextBlock(text="ok")], reward=reward, finished=True)

        environment, submit = ExamEnv, "hand_in"
    else:  # 0.1.125: no exclusive tools; CounterEnv's platoon_terminal_tools hint is just as invisible remotely
        environment, submit = ors_toys.CounterEnv, "submit"

    with _serve([environment]) as served:
        uri = f"ors+http://{served.hostport}/{environment.__name__}"
        with caplog.at_level(logging.WARNING, logger="platoon.envs.ors.service"):
            for _ in range(2):
                session = await _open(get_service(uri))
                try:
                    listed = {spec.name: spec for spec in await session.tools()}
                finally:
                    await session.close()
        assert not listed[submit].terminal
        warnings = [record for record in caplog.records if "declares no terminal tool" in record.getMessage()]
        assert len(warnings) == 1 and "service_kwargs.terminal_tools" in warnings[0].getMessage()

        session = await _open(get_service(uri, terminal_tools=[submit]))
        try:
            visible = {spec.name for spec in filter_tools(await session.tools(), "no_terminal")}
        finally:
            await session.close()
    assert submit not in visible and "add" in visible


async def _listing(service: OrsEnvironmentService) -> list[tuple]:
    """Every tool a session of ``service`` lists, as an agent and the gate see it: name, description, schema,
    terminal and read-only flags, meta, and the turn its call takes on the live env."""

    session = await _open(service)
    try:
        lease = session._lease
        return [
            (
                spec.name,
                spec.description,
                spec.input_schema,
                spec.terminal,
                spec.read_only,
                spec.meta,
                await service.call_class(spec.name, lease),
            )
            for spec in await session.tools()
        ]
    finally:
        await session.close()


@pytest.mark.parametrize("env_name", ["CounterEnv", "FlaggedEnv"])
async def test_a_served_class_named_by_env_class_lists_its_tools_as_in_process(env_name):
    """An env server sends tool names, schemas and @terminal's name and argument.  With ``env_class`` the hints and
    tool descriptors come from the class, as in process: platoon_* hints, _meta, exclusive and concurrent tools,
    and @terminal's own parameter schema."""

    environment = getattr(ors_toys, env_name, None)
    if environment is None:
        pytest.skip("this openreward SDK has no @terminal, exclusive or concurrent tools")
    with _serve([environment]) as served:
        uri = f"ors+http://{served.hostport}/{env_name}"
        in_process = await _listing(get_service(f"python://ors_toys:{env_name}"))
        served_class = await _listing(get_service(uri, env_class=f"ors_toys:{env_name}"))
        bare = await _listing(get_service(uri))

    assert served_class == in_process
    assert bare != in_process  # without env_class only the service options carry the hints


async def test_an_env_class_that_the_server_does_not_serve_is_refused(server, monkeypatch):
    """The server's listing of the class must match the class's: otherwise the hints read from the class could
    describe tools that the server does not serve.  A session open and the catalog listing both refuse it."""

    monkeypatch.setattr(service_module, "_SERVED_CLASS_CHECKS", {})
    uri = f"ors+http://{server.hostport}/CounterEnv"
    refusals = {
        "ors_toys:CraftEnv": "the server lists no craft",
        "ors_toys:RecountEnv": "add has another description",
        "ors_toys:NoSuchEnv": "env_class 'ors_toys:NoSuchEnv' does not import",
    }
    for env_class, expected in refusals.items():
        service = get_service(uri, env_class=env_class)
        for attempt in (lambda: _open(service), lambda: service.list_tasks("train")):
            with pytest.raises(EnvSetupError) as raised:
                await attempt()
            assert raised.value.failure_kind == "env_config" and expected in str(raised.value), str(raised.value)
    session = await _open(get_service(uri, env_class=ors_toys.CounterEnv))  # a class, as well as a reference
    await session.close()


async def test_hooks_of_a_served_class_are_reported_once(monkeypatch, caplog):
    """platoon_grade, platoon_fork and platoon_copy need the live env, so a served class's hooks never run."""

    monkeypatch.setattr(service_module, "_WARNED_SERVED_HOOKS", set())
    with _serve([ors_toys.GradedCounterEnv]) as served:
        service = get_service(f"ors+http://{served.hostport}/GradedCounterEnv", env_class=ors_toys.GradedCounterEnv)
        with caplog.at_level(logging.WARNING, logger="platoon.envs.ors.service"):
            for _ in range(2):
                session = await _open(service)
                try:
                    outcome = await session.grade()
                finally:
                    await session.close()

    assert (outcome.status, outcome.reward, outcome.reason) == ("scored", 0.0, "no_submit")  # no platoon_grade
    warnings = [record.getMessage() for record in caplog.records if "run only in process" in record.getMessage()]
    assert len(warnings) == 1 and "GradedCounterEnv, whose hooks (platoon_grade) run only in process" in warnings[0]


async def test_class_listings_compare_names_descriptions_and_schemas_and_tolerate_terminal_tools():
    """A server on another SDK version may list a @terminal tool as an ordinary tool, or describe it only by its
    name and argument; only its name is compared."""

    from platoon.envs.ors.sdk import class_listing_differences

    def spec(name: str, description: str = "Does it.", schema: dict | None = None) -> SimpleNamespace:
        return SimpleNamespace(name=name, description=description, input_schema=schema)

    answer = {"type": "object", "properties": {"verdict": {"type": "string", "description": "In one word."}}}
    terminal = SimpleNamespace(name="respond", arg="verdict", description="Answer.")

    class Listed:
        @staticmethod
        def list_tools():
            return SimpleNamespace(
                tools=[spec("add", schema={"title": "Add", "type": "object"})], terminal_tool=terminal
            )

    assert class_listing_differences(Listed, [spec("add", schema={"type": "object"})], terminal) == []
    assert (
        class_listing_differences(
            Listed, [spec("add", schema={"type": "object"}), spec("respond", "Answer.", answer)], None
        )
        == []
    )
    assert class_listing_differences(Listed, [spec("add", "Adds."), spec("peek")], terminal) == [
        "the class has no peek",
        "add has another description",
        "add takes other parameters",
    ]
    assert class_listing_differences(Listed, [], None) == ["the server lists no add, respond"]


async def test_open_remote_sessions_are_listed_mirrored_and_deletable(server, tmp_path):
    from platoon.envs.ors import cleanup

    mirror = tmp_path / "sessions.json"
    cleanup.mirror_to(str(mirror))
    try:
        session = await _open(_counter(server), rollout_id="listed")
        sid = session._lease.backend.session.sid  # type: ignore[union-attr]
        listed = [record for record in cleanup.live_sessions() if record.sid == sid]
        assert listed and listed[0].url == server.url
        assert [record.sid for record in cleanup.read_mirror(str(mirror))] == [sid]
        assert mirror.stat().st_mode & 0o777 == 0o600  # it holds API keys

        assert await cleanup.delete_orphaned_sessions(listed, timeout_s=10) == 1  # what a killed worker leaves
        assert server.ors._active_envs == {}
        await session.close()  # closing a deleted session is harmless
        assert all(record.sid != sid for record in cleanup.live_sessions())
        assert cleanup.read_mirror(str(mirror)) == []
    finally:
        cleanup.mirror_to(None)
    unreachable = cleanup.RemoteSessionRecord(url="http://127.0.0.1:9", sid="gone")
    assert await cleanup.delete_orphaned_sessions([unreachable], timeout_s=2) == 0


async def test_a_process_that_shut_down_remote_sessions_opens_and_calls_none(server, monkeypatch):
    """What a stopping rollout worker relies on (``cleanup.shut_down``): however its rollout tries, nothing reaches
    an env server any more, while closing (deleting) a session still works."""

    from platoon.envs.ors import cleanup

    monkeypatch.setattr(cleanup, "_shut_down", False)  # the switch is process-wide: put it back afterwards
    service = _counter(server)
    session = await _open(service, rollout_id="stopping")
    backend = session._lease.backend  # type: ignore[union-attr]
    sent: list[str] = []
    original = backend.session.call_tool

    async def recording(name, arguments):
        sent.append(name)
        return await original(name, arguments)

    monkeypatch.setattr(backend.session, "call_tool", recording)
    assert (await session.call_tool("add", {"amount": 1})).content[0]["text"] == "total=1"
    active = len(server.ors._active_envs)

    listed = cleanup.shut_down()

    assert backend.session.sid in [record.sid for record in listed]
    refused = await session.call_tool("add", {"amount": 1})
    assert refused.is_error and "RemoteSessionsShutDown" in refused.content[0]["text"]
    with pytest.raises(EnvSetupError, match="RemoteSessionsShutDown"):
        await _open(service, rollout_id="after")
    assert sent == ["add"] and len(server.ors._active_envs) == active
    await session.close()
    assert len(server.ors._active_envs) == active - 1
