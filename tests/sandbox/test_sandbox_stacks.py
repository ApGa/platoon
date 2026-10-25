"""The stack driver (platoon.sandbox.stacks): services start in dependency order under compose's three conditions,
a failure or a cancellation stops what started, and probes tell each service's state; over a fake backend."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import pytest

from platoon.sandbox.base import ExecResult, SandboxSpec
from platoon.sandbox.stacks import (
    Dependency,
    DrivenStack,
    Healthcheck,
    ServiceSpec,
    StackSpec,
    StackStartError,
    dependency_cycle,
    drive_stack,
    longest_path,
)
from platoon.utils.stopping import CleanupScope, cleanups, let_finish

CHECK = Healthcheck(command=("probe",), interval_s=0.01, timeout_s=1.0, start_interval_s=0.01, retries=3)


@dataclass
class Behavior:
    """How a fake service behaves: how long its start takes, after how many checks it is healthy (None: never), and
    after how long its command exits, with which status."""

    start_s: float = 0.0
    healthy_after: int | None = 0
    exits_after_s: float | None = None
    exit_code: int = 0
    fails_to_start: bool = False


class FakeSandbox:
    def __init__(self, name: str, behavior: Behavior) -> None:
        self.id = name
        self.behavior = behavior
        self.checks = 0
        self.commands: list[Any] = []

    @property
    def routes(self) -> dict[str, str]:
        return {}

    async def exec(self, cmd: Any, **options: Any) -> ExecResult:
        self.commands.append(cmd)
        self.checks += 1
        healthy = self.behavior.healthy_after is not None and self.checks > self.behavior.healthy_after
        return ExecResult(0 if healthy else 1, "", "" if healthy else "not yet")

    async def upload(self, src: Any, dst: str) -> None: ...

    async def download(self, src: str, dst: Any) -> None: ...

    async def snapshot(self) -> Any: ...

    async def kill_processes(self) -> None: ...

    async def stop(self) -> None: ...


class FakeService:
    run_as = "caller"

    def __init__(self, name: str, sandbox: FakeSandbox, has_command: bool) -> None:
        self.name = name
        self.sandbox = sandbox
        self.running = True
        self.code: int | None = None
        self.stopped = False
        self.has_command = has_command

    def exit(self, code: int) -> None:
        self.running, self.code = False, code

    async def status(self) -> tuple[bool, int | None]:
        return (self.running if self.has_command else not self.stopped), self.code

    async def logs(self, max_bytes: int) -> str:
        return f"{self.name}'s log"

    async def stop(self) -> None:
        self.stopped = True
        self.running = False


@dataclass
class FakeBackend:
    behaviors: dict[str, Behavior] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    services: dict[str, FakeService] = field(default_factory=dict)
    specs: dict[str, ServiceSpec] = field(default_factory=dict)
    networks: list[str] = field(default_factory=list)
    failing_switches: int = 0  # how many set_network calls fail midway
    closed: bool = False

    async def start_service(self, name: str, spec: ServiceSpec) -> FakeService:
        behavior = self.behaviors.get(name, Behavior())
        self.events.append(f"start {name}")
        await asyncio.sleep(behavior.start_s)
        if behavior.fails_to_start:
            raise OSError("no capacity")
        sandbox = FakeSandbox(name, behavior)
        if spec.setup is not None:
            await spec.setup(sandbox)  # type: ignore[arg-type]
        service = FakeService(name, sandbox, spec.command is not None)
        self.services[name] = service
        self.specs[name] = spec
        self.events.append(f"started {name}")
        if behavior.exits_after_s is not None:
            asyncio.get_running_loop().call_later(behavior.exits_after_s, service.exit, behavior.exit_code)
        return service

    async def set_network(self, network: str) -> None:
        self.networks.append(network)
        if self.failing_switches:
            self.failing_switches -= 1
            raise RuntimeError("the switch failed midway")

    async def connect(self, service: str, port: int) -> str:
        return f"tcp://fake-{service}:{port}"

    def rewrite(self, value: str, *, service: str | None = None) -> str:
        return value.replace("://db:5432", "://127.0.0.1:40000")

    def volume_path(self, name: str) -> Path:
        return Path("/fake-volumes") / name

    async def close(self) -> None:
        self.closed = True
        self.events.append("closed")


def _service(command: list[str] | None = None, **fields: Any) -> ServiceSpec:
    return ServiceSpec(sandbox=SandboxSpec(image="local"), command=command, **fields)


def _spec(**services: ServiceSpec) -> StackSpec:
    return StackSpec(services={"main": services.pop("main", _service()), **services})


def _all_stopped(backend: FakeBackend) -> bool:
    return backend.closed and all(service.stopped for service in backend.services.values())


@pytest.mark.asyncio
async def test_services_start_once_their_dependencies_reach_their_conditions():
    backend = FakeBackend(
        {"db": Behavior(healthy_after=2), "seed": Behavior(exits_after_s=0.05, exit_code=0), "cache": Behavior()}
    )
    spec = _spec(
        main=_service(
            depends_on={
                "seed": Dependency("service_completed_successfully"),
                "cache": Dependency("service_started"),
            }
        ),
        db=_service(["db"], healthcheck=CHECK),
        seed=_service(["seed"], depends_on={"db": Dependency("service_healthy")}),
        cache=_service(["cache"]),
    )

    stack = await drive_stack(backend, spec, id="stack-1")

    events = backend.events
    assert events.index("started db") < events.index("start seed")  # seed waited for db to be healthy
    # Two failed checks, then a pass (one fewer than its retries), and the start's last probe, before the agents.
    assert backend.services["db"].sandbox.checks == 4
    assert events.index("start main") > events.index("started cache")
    assert backend.services["seed"].code == 0 and events[-1] == "started main"  # main waited for seed to complete
    assert stack.main is backend.services["main"].sandbox and set(stack.services) == {"main", "db", "seed", "cache"}
    assert stack.run_modes == {"main": "caller", "db": "caller", "seed": "caller", "cache": "caller"}
    states = await stack.health(["db", "seed", "cache"])
    assert {name: state.status for name, state in states.items()} == {
        "db": "healthy",
        "seed": "completed",
        "cache": "running",
    }
    await stack.stop()
    await stack.stop()  # idempotent
    assert _all_stopped(backend)


@pytest.mark.asyncio
async def test_services_that_wait_for_nothing_start_at_once():
    backend = FakeBackend({name: Behavior(start_s=0.3) for name in ("a", "b", "c")})
    started = time.monotonic()
    stack = await drive_stack(backend, _spec(a=_service(["a"]), b=_service(["b"]), c=_service(["c"])), id="s")
    assert time.monotonic() - started < 0.8  # four starts of 0.3 s side by side: main waits for none of them
    await stack.stop()


@pytest.mark.parametrize(
    ("behaviors", "services", "failed", "message"),
    [
        (
            {"db": Behavior(healthy_after=None)},
            {"db": _service(["db"], healthcheck=CHECK)},
            "db",
            "never turned healthy: its healthcheck failed 3 times: exit 1: not yet",
        ),
        (
            {"db": Behavior(healthy_after=None, exits_after_s=0.0, exit_code=3)},
            {"db": _service(["db"], healthcheck=CHECK)},
            "db",
            "exited with status 3 before its healthcheck passed",
        ),
        (
            {"seed": Behavior(exits_after_s=0.0, exit_code=1)},
            {
                "seed": _service(["seed"]),
                "main": _service(depends_on={"seed": Dependency("service_completed_successfully")}),
            },
            "seed",
            "exited with status 1, and others wait for it to complete successfully",
        ),
        (
            {"web": Behavior(exits_after_s=0.0, exit_code=0)},
            {"web": _service(["web"]), "main": _service(depends_on={"web": Dependency()})},
            "web",
            "exited with status 0 as the stack started",
        ),
        ({"db": Behavior(fails_to_start=True)}, {"db": _service(["db"])}, "db", "did not start: OSError: no capacity"),
    ],
    ids=["never-healthy", "exits-before-healthy", "init-job-fails", "exits-at-start", "start-fails"],
)
@pytest.mark.asyncio
async def test_a_service_that_fails_to_start_fails_the_stack_and_everything_started_stops(
    behaviors, services, failed: str, message: str
):
    backend = FakeBackend(behaviors)
    with pytest.raises(StackStartError) as error:
        await drive_stack(backend, _spec(**services), id="s")
    assert error.value.service == failed and message in str(error.value)
    assert error.value.log == ("" if failed == "db" and "did not start" in message else f"{failed}'s log")
    assert _all_stopped(backend)
    if "did not start" in message:
        assert isinstance(error.value.__cause__, OSError)


@pytest.mark.asyncio
async def test_a_start_cancelled_midway_stops_what_started_and_starts_nothing_more():
    backend = FakeBackend({"db": Behavior(healthy_after=None), "slow": Behavior(start_s=30.0)})
    spec = _spec(
        main=_service(depends_on={"db": Dependency("service_healthy")}),
        db=_service(["db"], healthcheck=Healthcheck(command=("probe",), interval_s=0.01, retries=10**6)),
        slow=_service(["slow"]),
    )
    task = asyncio.ensure_future(drive_stack(backend, spec, id="s"))
    while "started db" not in backend.events or "start slow" not in backend.events:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert _all_stopped(backend)
    assert "start main" not in backend.events and "slow" not in backend.services


@pytest.mark.asyncio
async def test_a_setup_error_passes_through_as_it_is_and_rolls_back():
    class Unprepared(Exception):
        pass

    async def setup(sandbox: Any) -> None:
        raise Unprepared("main's files cannot be copied")

    backend = FakeBackend()
    with pytest.raises(Unprepared, match="cannot be copied"):
        await drive_stack(backend, _spec(main=_service(setup=setup), db=_service(["db"])), id="s")
    assert _all_stopped(backend)


@pytest.mark.asyncio
async def test_the_setup_runs_with_the_service_s_sandbox_before_it_counts_as_started():
    seen: list[str] = []

    async def setup(sandbox: Any) -> None:
        await asyncio.sleep(0.05)
        seen.append(sandbox.id)

    backend = FakeBackend()
    spec = _spec(
        main=_service(setup=setup, healthcheck=CHECK),
        judge=_service(["judge"], depends_on={"main": Dependency("service_healthy")}),
    )
    stack = await drive_stack(backend, spec, id="s")
    assert seen == ["main"] and backend.events.index("started main") < backend.events.index("start judge")
    await stack.stop()


@pytest.mark.asyncio
async def test_probes_tell_each_service_s_state():
    backend = FakeBackend({"flaky": Behavior(healthy_after=0)})
    spec = _spec(
        db=_service(["db"]),
        flaky=_service(["flaky"], healthcheck=CHECK),
        main=_service(depends_on={"db": Dependency()}),
    )
    stack = await drive_stack(backend, spec, id="s")
    backend.services["db"].exit(137)
    backend.behaviors["flaky"].healthy_after = None  # from now on it fails its checks
    states = await stack.health()

    assert states["db"].status == "exited" and states["db"].exit_code == 137 and not states["db"].up
    assert states["db"].describe() == "exited with status 137"
    assert states["flaky"].status == "unhealthy" and states["flaky"].describe() == "is unhealthy: exit 1: not yet"
    assert states["main"].status == "running" and states["main"].up
    assert backend.services["flaky"].sandbox.checks == 2 + 3  # the start's pass and its last probe, then three tries
    assert (await stack.health(["nope"]))["nope"].status == "exited"
    assert await stack.logs("db") == "db's log"
    await stack.stop()


@pytest.mark.asyncio
async def test_the_driver_rewrites_environment_values_and_healthchecks_for_the_stack_s_addresses():
    backend = FakeBackend()
    check = Healthcheck(command=("pg_isready", "-d", "postgres://db:5432/app"), interval_s=0.01)
    main = ServiceSpec(sandbox=SandboxSpec(image="local", env={"DB": "postgres://db:5432/app", "HOME": "/root"}))
    stack = await drive_stack(backend, _spec(main=main, db=_service(["db"], healthcheck=check)), id="s")

    assert backend.specs["main"].sandbox.env == {"DB": "postgres://127.0.0.1:40000/app", "HOME": "/root"}
    assert backend.services["db"].sandbox.commands[0] == ["pg_isready", "-d", "postgres://127.0.0.1:40000/app"]
    assert stack.rewrite("x://db:5432") == "x://127.0.0.1:40000"
    assert await stack.connect("db", 5432) == "tcp://fake-db:5432"
    with pytest.raises(KeyError):
        await stack.connect("cache", 1)
    await stack.set_network("none")
    await stack.set_network("none")  # already: nothing to change
    assert backend.networks == ["none"] and stack.network == "none"
    with pytest.raises(ValueError, match="egress or none"):
        await stack.set_network(cast(Any, "host"))
    await stack.stop()


@pytest.mark.asyncio
async def test_a_network_switch_that_failed_midway_is_made_again_whichever_way_it_goes():
    """As for the verifier's egress, then the agents' next phase without: the switch back reaches the backend, since
    the failed one may have moved some services already."""

    backend = FakeBackend(failing_switches=1)
    stack = await drive_stack(backend, replace(_spec(), network="none"), id="s")

    with pytest.raises(RuntimeError, match="midway"):
        await stack.set_network("egress")
    await stack.set_network("none")
    await stack.set_network("none")  # settled now: nothing to change
    assert backend.networks == ["egress", "none"] and stack.network == "none"
    await stack.stop()


@pytest.mark.parametrize(
    ("services", "message"),
    [
        ({"db": _service(["db"])}, "needs a 'main' service"),
        ({"main": _service(depends_on={"db": Dependency()})}, "depends on 'db', which the stack does not have"),
        ({"main": _service(depends_on={"main": Dependency()})}, "depends on itself"),
        (
            {
                "main": _service(depends_on={"a": Dependency()}),
                "a": _service(["a"], depends_on={"b": Dependency()}),
                "b": _service(["b"], depends_on={"a": Dependency()}),
            },
            "in a cycle: a -> b -> a",
        ),
        (
            {"main": _service(depends_on={"db": Dependency("service_healthy")}), "db": _service(["db"])},
            "waits for 'db' to be healthy, and 'db' has no healthcheck",
        ),
        ({"main": _service(network_of="vpn")}, "shares the network of 'vpn'"),
    ],
    ids=["no-main", "unknown", "itself", "cycle", "no-healthcheck", "network"],
)
@pytest.mark.asyncio
async def test_specs_no_stack_can_start_are_refused_before_anything_starts(services, message: str):
    backend = FakeBackend()
    with pytest.raises(ValueError, match=message):
        await drive_stack(backend, StackSpec(services=services), id="s")
    assert backend.events == ["closed"]  # nothing started, and what the backend holds is released


def test_dependency_cycles_and_the_longest_path():
    assert dependency_cycle({"a": {"b"}, "b": {"c"}, "c": set()}) is None
    assert dependency_cycle({"a": {"b"}, "b": {"a"}}) == ["a", "b", "a"]
    graph = {"main": {"seed", "cache"}, "seed": {"db"}, "db": set(), "cache": set()}
    weights = {"main": 1.0, "seed": 2.0, "db": 5.0, "cache": 7.0}
    assert longest_path(graph, weights) == 8.0  # db, seed, main; cache, main is as long


@pytest.mark.asyncio
async def test_a_stack_stopped_while_a_service_is_still_stopping_stops_it_once():
    """stop() from two places at once (a lease's teardown and a stopping worker) stops each service once."""

    stops: list[str] = []

    class Counting(FakeService):
        async def stop(self) -> None:
            stops.append(self.name)
            await asyncio.sleep(0.05)
            await super().stop()

    class Backend(FakeBackend):
        async def start_service(self, name: str, spec: ServiceSpec) -> FakeService:
            started = await super().start_service(name, spec)
            service = Counting(name, started.sandbox, spec.command is not None)
            self.services[name] = service
            return service

    backend = Backend()
    stack = await drive_stack(backend, _spec(db=_service(["db"])), id="s")
    assert isinstance(stack, DrivenStack)
    await asyncio.gather(stack.stop(), stack.stop())
    assert sorted(stops) == ["db", "main"] and backend.closed


@pytest.mark.parametrize("recancel", [False, True], ids=["cancelled-once", "cancelled-again-and-again"])
@pytest.mark.asyncio
async def test_a_cancelled_start_lets_a_service_caught_mid_start_roll_back_before_it_counts_as_stopped(recancel: bool):
    """A service whose start is cancelled stops what it started itself, shielded and slowly here (as the process
    provider's backend does): the stack's stop waits for that rollback and never cancels it again, so nothing of the
    service is left once the cancelled start raises; and when a stopping worker cancels every task but the cleanups,
    again and again, the rollback still runs to its end inside the rollout's cleanup scope."""

    class RollingBack(FakeBackend):
        async def start_service(self, name: str, spec: ServiceSpec) -> FakeService:
            if name != "main":
                return await super().start_service(name, spec)
            self.events.append("start main")
            try:
                await asyncio.sleep(30)  # main's setup, where the cancellation lands
            except BaseException:
                await asyncio.shield(let_finish(asyncio.ensure_future(self._roll_back(name))))
                raise
            raise AssertionError("never reached")

        async def _roll_back(self, name: str) -> None:
            await asyncio.sleep(0.3)
            self.events.append(f"rolled back {name}")

    backend = RollingBack()
    spec = _spec(main=_service(depends_on={"db": Dependency()}), db=_service(["db"]))
    scope = CleanupScope()

    async def rollout() -> None:
        with scope:  # as a rollout's code runs, its permit waiting for the scope's cleanups
            await drive_stack(backend, spec, id="s")

    task = asyncio.ensure_future(rollout())
    while "start main" not in backend.events:
        await asyncio.sleep(0.01)
    task.cancel()
    if recancel:  # as a stopping rollout worker does
        loop = asyncio.get_running_loop()
        for _ in range(3):
            await asyncio.sleep(0.02)
            spared = {asyncio.current_task(), *cleanups(loop)}
            for other in asyncio.all_tasks():
                if other not in spared and not other.done():
                    other.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    if not recancel:
        assert "rolled back main" in backend.events and backend.events[-1] == "closed"  # waited for, then closed
    assert await scope.finish(10.0) == 0
    assert "rolled back main" in backend.events and _all_stopped(backend)


@pytest.mark.asyncio
async def test_an_init_job_that_runs_past_its_timeout_fails_the_start_with_its_log():
    backend = FakeBackend({"seed": Behavior()})  # its command never exits
    spec = _spec(
        main=_service(depends_on={"seed": Dependency("service_completed_successfully")}),
        seed=_service(["seed"], timeout_s=0.2),
    )
    started = time.monotonic()
    with pytest.raises(StackStartError, match="the service seed did not complete within 0.2s") as error:
        await drive_stack(backend, spec, id="s")
    assert time.monotonic() - started < 5.0 and error.value.service == "seed" and error.value.log == "seed's log"
    assert "start main" not in backend.events and _all_stopped(backend)


@pytest.mark.asyncio
async def test_a_service_that_is_down_once_every_service_started_fails_the_start():
    """The start's last probe: a sidecar that passed its healthcheck and went unhealthy while main was prepared."""

    backend = FakeBackend({"db": Behavior(healthy_after=0)})

    async def setup(sandbox: Any) -> None:
        backend.behaviors["db"].healthy_after = None  # from now on its checks fail

    spec = _spec(
        main=_service(setup=setup, depends_on={"db": Dependency("service_healthy")}),
        db=_service(["db"], healthcheck=CHECK),
    )
    with pytest.raises(StackStartError, match="the service db is unhealthy: exit 1: not yet as the stack started"):
        await drive_stack(backend, spec, id="s")
    assert _all_stopped(backend)
