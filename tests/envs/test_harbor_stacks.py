"""Harbor tasks that are multi-container stacks (platoon.envs.harbor.sidecars), run on the process provider: the
fixtures below ``fixtures/harbor-stacks`` (written for platoon) score their oracles and no-ops, a sidecar that is
down at grading makes the outcome invalid, one that never starts an error, and nothing is left behind."""

from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from site_cache import use_site_cache

from platoon.envs.harbor import HarborEnvironmentService, HarborSession, sidecars
from platoon.envs.harbor.fetch import fetch, fetch_config
from platoon.envs.harbor.images import SessionImage
from platoon.envs.harbor.task import HarborTask, HarborTaskError
from platoon.envs.service import EnvSetupError
from platoon.sandbox import cleanup
from platoon.sandbox import compose as docker_compose
from platoon.sandbox.process import ProcessProvider
from platoon.sandbox.stacks import DrivenStack

STACKS = Path(__file__).parent / "fixtures" / "harbor-stacks" / "tasks"
ROOT = 'root="${PLATOON_SANDBOX_ROOT:-}"\n'  # what a test script writes below on the process provider


@pytest.fixture(autouse=True)
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Node scratch of each test's own, where the sessions keep their bind copies and sidecars' log snapshots."""

    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "scratch"))
    return tmp_path / "scratch"


@pytest.fixture(autouse=True)
def registered() -> set[int]:
    """What is registered for cleanup before a test: other tests of the process may leave registrations."""

    global _BEFORE
    _BEFORE = {id(sandbox) for sandbox in cleanup.live_sandboxes()}
    return _BEFORE


_BEFORE: set[int] = set()


@pytest.fixture
def provider(tmp_path: Path) -> ProcessProvider:
    return ProcessProvider(root=tmp_path / "sandboxes")


@pytest.fixture
def service(provider: ProcessProvider) -> HarborEnvironmentService:
    return HarborEnvironmentService(STACKS, provider=provider)


async def _open(service: HarborEnvironmentService, task_id: str) -> HarborSession:
    return await service.open(await service.get_task(task_id), rollout_id=f"rollout-{task_id}")


def _stack(session: HarborSession) -> DrivenStack:
    assert isinstance(session.stack, DrivenStack)
    return session.stack


def _service_pid(session: HarborSession, name: str) -> int:
    """The pid of the program a service of the session's stack runs: the child of the shell that waits for it."""

    shell = int(cast(Any, _stack(session)._handles[name])._entry.split(":")[0])
    [child] = Path(f"/proc/{shell}/task/{shell}/children").read_text().split()
    return int(child)


async def _kill(session: HarborSession, name: str) -> None:
    """SIGKILL the program a service runs, and wait until its stack sees it exited."""

    os.kill(_service_pid(session, name), signal.SIGKILL)
    handle = cast(Any, _stack(session)._handles[name])
    deadline = time.monotonic() + 10.0
    while (await handle.status())[0]:
        assert time.monotonic() < deadline, f"the service {name} still runs"
        await asyncio.sleep(0.02)


async def _eventually(condition: Callable[[], bool], timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "the condition never held"
        await asyncio.sleep(0.02)


def _leftovers(provider: ProcessProvider) -> list[Any]:
    scratch = Path(os.environ["PLATOON_NODE_SCRATCH"])
    registered = [sandbox for sandbox in cleanup.live_sandboxes() if id(sandbox) not in _BEFORE]
    return [
        *(provider.root.iterdir() if provider.root.exists() else ()),
        *(scratch.iterdir() if scratch.exists() else ()),
        *registered,
    ]


def _nothing_left(provider: ProcessProvider) -> bool:
    """No sandbox or stack directory, no registration for cleanup, and nothing in node scratch is left."""

    assert (leftovers := _leftovers(provider)) == [], leftovers
    return True


def _variant(name: str, into: Path, source: str = "app-kv") -> Path:
    """A copy of a fixture task under another name."""

    task = into / name
    shutil.copytree(STACKS / source, task)
    toml = task / "task.toml"
    toml.write_text(toml.read_text().replace(f'name = "platoon/{source}"', f'name = "platoon/{name}"'))
    return task


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, (old, path)
    path.write_text(text.replace(old, new))


# --- the fixtures' outcomes ----------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_app_kv_scores_its_oracle_and_its_no_op_through_its_sidecar(
    service: HarborEnvironmentService, provider: ProcessProvider
):
    session = await _open(service, "app-kv")
    try:
        assert session.stack is not None and set(session.stack.services) == {"main", "kv"}
        noop = await session.grade()
        assert (noop.status, noop.reward) == ("scored", 0.0), noop.metadata
        assert noop.metadata["stack"] == "sidecars"
        assert noop.metadata["services"] == {
            "main": {"run_as": "caller"},
            "kv": {"run_as": "caller", "state": "healthy"},
        }

        solved = await session.run_solution()
        assert solved.exit_code == 0, solved.stderr
        outcome = await session.grade()
        assert (outcome.status, outcome.reward) == ("scored", 1.0), outcome.metadata.get("test_output")
        assert "records its write" in outcome.metadata["test_output"]
        with pytest.raises(NotImplementedError, match="multi-container stack, and a copy fork restores"):
            await session.fork("copy")
        shared = await session.fork("shared")
        assert shared.stack is session.stack
        await shared.close()
        # The sidecar's verifier log directory is the stack's own volume, which reaping deletes with the stack; the
        # session keeps nothing of its own in node scratch.
        logs = session.stack.volume_path("platoon:logs-verifier")
        assert logs.is_dir() and logs.is_relative_to(provider.root / session.stack.id)
        assert not Path(os.environ["PLATOON_NODE_SCRATCH"]).exists() or not any(
            Path(os.environ["PLATOON_NODE_SCRATCH"]).iterdir()
        )
    finally:
        await session.close()
    assert not logs.exists() and _nothing_left(provider)


@pytest.mark.asyncio
async def test_a_sidecar_s_own_record_reaches_the_tests_and_the_agents_cannot_forge_it(
    service: HarborEnvironmentService, provider: ProcessProvider
):
    session = await _open(service, "app-kv")
    try:
        assert session.sandbox is not None and session.stack is not None
        # A value that never went through the sidecar (written where only the sidecar could write it, in a real
        # stack), and a forged record of its write in main's log directory, which grading makes afresh before the
        # sidecar's own record is put there: the tests find no record.
        [volume] = provider.root.glob(f"{session.stack.id}/volumes/kv-data")
        (volume / "answer").write_text("forty-two")
        await session.sandbox.exec('echo answer > "$PLATOON_SANDBOX_ROOT/logs/verifier/kv-writes.log"')
        forged = await session.grade()
        assert (forged.status, forged.reward) == ("scored", 0.0), forged.metadata.get("test_output")
        assert (await session.run_solution()).exit_code == 0  # through the sidecar, which records it
        solved = await session.grade()
        assert (solved.status, solved.reward) == ("scored", 1.0), solved.metadata.get("test_output")
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_killed_sidecar_makes_the_outcome_invalid_unless_the_environment_grades_anyway(
    provider: ProcessProvider,
):
    for on_down, expected in (("invalid", ("invalid", "sidecar_unhealthy")), ("grade", ("scored", None))):
        service = HarborEnvironmentService(STACKS, provider=provider, on_sidecar_down=on_down)
        session = await _open(service, "app-kv")
        try:
            assert (await session.run_solution()).exit_code == 0
            await _kill(session, "kv")
            outcome = await session.grade()
            assert (outcome.status, outcome.failure_kind) == expected, outcome
            if on_down == "invalid":
                assert not outcome.trainable and outcome.metadata["sidecar"] == "kv"
                assert outcome.reason == "the sidecar kv exited with status 137 before the tests"
                assert outcome.metadata["sidecar_status"] == "exited" and "sidecar_log" in outcome.metadata
                assert outcome.metadata["services"]["kv"] == {"run_as": "caller", "state": "exited"}
            else:
                assert outcome.reward == 0.0  # the store is gone: the tests cannot read the answer back
                assert outcome.metadata["sidecars_down"] == [
                    "the sidecar kv exited with status 137 before the tests",
                    "the sidecar kv exited with status 137 after the tests",
                ]
        finally:
            await session.close()
    assert _nothing_left(provider)


@pytest.mark.asyncio
async def test_a_sidecar_that_the_tests_kill_makes_the_outcome_invalid_after_them(make_task, provider):
    marker = f"sleep {4000 + os.getpid() % 1000}.5"  # this test's own sidecar, which its tests find and kill
    test_sh = f'pkill -f "^{marker}$"; {ROOT}echo 1 > "$root/logs/verifier/reward.txt"\n'
    task_dir = make_task("killer", task_toml='[environment]\ndocker_image = "python:3.12-slim"\n', test_sh=test_sh)
    (task_dir / "environment" / "docker-compose.yaml").write_text(
        f"services:\n  main:\n    depends_on: [side]\n  side:\n    image: python:3.12-slim\n"
        f'    command: ["sleep", "{marker.split()[1]}"]\n'
    )
    service = HarborEnvironmentService(task_dirs=[task_dir], provider=provider)
    session = await _open(service, "killer")
    try:
        outcome = await session.grade()
        assert (outcome.status, outcome.failure_kind) == ("invalid", "sidecar_unhealthy"), outcome
        assert outcome.reason == "the sidecar side exited with status 143 after the tests"
        assert outcome.metadata["verifier_exit_code"] == 0  # the tests' own metadata is kept
    finally:
        await session.close()
    assert _nothing_left(provider)


@pytest.mark.asyncio
async def test_a_sidecar_that_crashes_in_the_agent_phase_makes_the_outcome_invalid(
    service: HarborEnvironmentService, provider: ProcessProvider
):
    session = await _open(service, "app-kv-crash")
    try:
        noop = await session.grade()
        assert (noop.status, noop.reward) == ("scored", 0.0)  # it crashes on a write, which no one made
        assert (await session.run_solution()).exit_code == 0
        outcome = await session.grade()
        assert (outcome.status, outcome.failure_kind) == ("invalid", "sidecar_unhealthy")
        assert outcome.metadata["sidecar_exit_code"] == 1
    finally:
        await session.close()
    assert _nothing_left(provider)


@pytest.mark.asyncio
async def test_a_sidecar_that_never_turns_healthy_fails_the_session_s_start(
    service: HarborEnvironmentService, provider: ProcessProvider
):
    with pytest.raises(EnvSetupError, match="the service kv never turned healthy") as error:
        await _open(service, "app-kv-unhealthy")
    assert error.value.failure_kind == "sidecar_start"
    assert _nothing_left(provider)


@pytest.mark.asyncio
async def test_an_init_job_runs_to_its_end_before_main_starts(
    service: HarborEnvironmentService, provider: ProcessProvider
):
    session = await _open(service, "app-kv-seeded")
    try:
        assert session.stack is not None
        assert (await session.stack.health(["seed"]))["seed"].status == "completed"
        noop = await session.grade()
        assert (noop.status, noop.reward) == ("scored", 0.0)
        assert noop.metadata["services"]["seed"]["state"] == "completed"
        assert (await session.run_solution()).exit_code == 0
        outcome = await session.grade()
        assert (outcome.status, outcome.reward) == ("scored", 1.0), outcome.metadata.get("test_output")
    finally:
        await session.close()
    assert _nothing_left(provider)


@pytest.mark.asyncio
async def test_an_init_job_that_never_ends_fails_the_start_within_its_budget(make_task, provider):
    task_dir = make_task("hang", task_toml='[environment]\ndocker_image = "python:3.12-slim"\n')
    (task_dir / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    depends_on:\n      seed:\n        condition: service_completed_successfully\n"
        '  seed:\n    image: python:3.12-slim\n    command: ["sh", "-c", "echo waiting for a database; sleep 3600"]\n'
    )
    service = HarborEnvironmentService(task_dirs=[task_dir], provider=provider, sandbox_start_budget_s=1.0)
    started = time.monotonic()
    with pytest.raises(EnvSetupError, match="the service seed did not complete within 1s") as error:
        await _open(service, "hang")
    assert error.value.failure_kind == "sidecar_start" and "waiting for a database" in str(error.value)
    assert time.monotonic() - started < 20.0 and _nothing_left(provider)


# --- grading -------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_separate_verifier_grades_what_a_collect_hook_and_the_sidecar_s_artifacts_hold(
    service: HarborEnvironmentService, provider: ProcessProvider
):
    task = service.harbor_task(await service.get_task("sidecar-artifacts"))
    assert [(hook.service, hook.timeout_s) for hook in task.collect] == [("api", 30.0)]
    assert [(artifact.source, artifact.service) for artifact in task.artifacts] == [
        ("/tmp/api/orders.log", "api"),
        ("/tmp/stats.json", "api"),
    ]
    for solve, reward in ((False, 0.0), (True, 1.0)):
        session = await _open(service, "sidecar-artifacts")
        try:
            if solve:
                assert (await session.run_solution()).exit_code == 0
            outcome = await session.grade()
            assert (outcome.status, outcome.reward) == ("scored", reward), outcome.metadata.get("test_output")
            assert outcome.metadata["verifier_mode"] == "separate"
            if solve:
                assert outcome.metadata["artifacts"] == ["/tmp/api/orders.log", "/tmp/stats.json"]
            else:  # no order: the api's log was never written, its snapshot was
                assert outcome.metadata["artifacts"] == ["/tmp/stats.json"]
            assert "collect_failed" not in outcome.metadata
        finally:
            await session.close()
    assert _nothing_left(provider)


@pytest.mark.asyncio
async def test_a_separate_verifier_gets_the_sidecars_log_directories_from_a_hook_that_names_its_service(
    tmp_path: Path, provider: ProcessProvider
):
    """A collect hook whose command reaches the api by name (``http://api:8000``, rewritten on this provider) writes
    into the api's verifier log directory, which the separate verifier's /logs/verifier receives."""

    task = _variant("logged", tmp_path / "dataset", source="sidecar-artifacts")
    _edit(
        task / "environment" / "docker-compose.yaml",
        "      - ./api:/srv/api:ro\n",
        "      - ./api:/srv/api:ro\n      - ${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}\n",
    )
    _edit(
        task / "task.toml",
        'command = "python3 /srv/api/fetch.py \\"$API_URL/stats\\" > /tmp/stats.json"',
        'command = "python3 /srv/api/fetch.py http://api:8000/stats > /logs/verifier/api-stats.json"',
    )
    (task / "tests" / "test.sh").write_text(
        ROOT + 'if grep -q \'"orders": 0\' "$root/logs/verifier/api-stats.json"; then r=1; else r=0; fi\n'
        'echo $r > "$root/logs/verifier/reward.txt"\n'
    )
    service = HarborEnvironmentService(tmp_path / "dataset", provider=provider)
    session = await _open(service, "logged")
    try:
        outcome = await session.grade()
        assert (outcome.status, outcome.reward) == ("scored", 1.0), outcome.metadata
        assert "collect_failed" not in outcome.metadata
    finally:
        await session.close()
    assert _nothing_left(provider)


@pytest.mark.asyncio
async def test_a_sidecar_artifact_at_a_path_main_s_artifact_claims_is_left_out(tmp_path: Path, provider):
    task = _variant("collide", tmp_path / "dataset", source="sidecar-artifacts")
    _edit(task / "task.toml", "artifacts = [\n", 'artifacts = [\n    { source = "/tmp/stats.json" },\n')
    service = HarborEnvironmentService(tmp_path / "dataset", provider=provider)
    session = await _open(service, "collide")
    try:
        assert session.sandbox is not None
        await session.sandbox.exec('echo "{\\"orders\\": 99}" > /tmp/stats.json')
        outcome = await session.grade()
        assert outcome.metadata["artifacts_skipped"] == {"api:/tmp/stats.json": "collision"}  # as Harbor keeps main's
        assert outcome.metadata["artifacts"] == ["/tmp/stats.json"] and outcome.reward == 0.0
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_separate_verifier_grades_anyway_where_the_environment_says_so(provider: ProcessProvider):
    service = HarborEnvironmentService(STACKS, provider=provider, on_sidecar_down="grade")
    session = await _open(service, "sidecar-artifacts")
    try:
        assert (await session.run_solution()).exit_code == 0
        await _kill(session, "api")
        outcome = await session.grade()
        assert outcome.failure_kind != "sidecar_unhealthy"
        assert outcome.metadata["sidecars_down"] == [
            "the sidecar api exited with status 137 before the artifacts were collected",
            "the sidecar api exited with status 137 after the artifacts were collected",
        ]
        hook = 'api: python3 /srv/api/fetch.py "$API_URL/stats" > /tmp/stats.json'
        assert outcome.metadata["collect_failed"] == [hook]  # the api is gone
    finally:
        await session.close()
    assert _nothing_left(provider)


@pytest.mark.asyncio
async def test_grading_in_main_gives_the_stack_the_verifier_s_network_and_the_agents_theirs_again(
    tmp_path: Path, provider: ProcessProvider, monkeypatch: pytest.MonkeyPatch
):
    task = _variant("offline-tests", tmp_path / "dataset")
    _edit(task / "task.toml", "[verifier]\n", '[verifier]\nnetwork_mode = "no-network"\n')
    switched: list[str] = []
    original = DrivenStack.set_network

    async def spy(self: DrivenStack, network: Any) -> None:
        switched.append(network)
        await original(self, network)

    monkeypatch.setattr(DrivenStack, "set_network", spy)
    service = HarborEnvironmentService(tmp_path / "dataset", provider=provider)
    session = await _open(service, "offline-tests")
    try:
        outcome = await session.grade()
        assert outcome.status == "scored" and switched == ["none", "egress"]
        assert session.stack is not None and session.stack.network == "egress"
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_probe_that_cannot_run_is_an_infrastructure_error(service: HarborEnvironmentService):
    session = await _open(service, "app-kv")
    try:

        async def broken(services: Any = None) -> Any:
            raise RuntimeError("the runtime is gone")

        cast(Any, session.stack).health = broken
        outcome = await session.grade()
        assert (outcome.status, outcome.failure_kind) == ("error", "sidecar_probe")
        assert "cannot probe the stack's sidecars before the tests: the runtime is gone" in (outcome.reason or "")
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_an_outcome_records_only_the_states_its_own_grade_probed(service: HarborEnvironmentService):
    session = await _open(service, "app-kv")
    try:
        first = await session.grade()
        assert first.metadata["services"]["kv"] == {"run_as": "caller", "state": "healthy"}
        await _kill(session, "kv")
        assert session.sandbox is not None
        await session.sandbox.exec('rm -rf "$PLATOON_SANDBOX_ROOT/app"', cwd="/")  # the agents remove their workdir
        second = await session.grade()  # it ends before any probe
        assert (second.status, second.reason) == ("scored", "workdir_missing")
        assert second.metadata["services"]["kv"] == {"run_as": "caller"}  # no state: this grade probed none
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_collect_hooks_of_a_single_container_task_run_before_its_tests(make_task, provider):
    toml = (
        '[environment]\ndocker_image = "python:3.12-slim"\n'
        '[[verifier.collect]]\ncommand = "echo collected > /logs/verifier/hooked.txt"\n'
        '[[verifier.collect]]\ncommand = "exit 7"\n'
    )
    test_sh = (
        ROOT + 'if grep -qx collected "$root/logs/verifier/hooked.txt"; then r=1; else r=0; fi\n'
        'echo $r > "$root/logs/verifier/reward.txt"\n'
    )
    task_dir = make_task("hooked", task_toml=toml, test_sh=test_sh)
    service = HarborEnvironmentService(task_dirs=[task_dir], provider=provider)
    session = await _open(service, "hooked")
    try:
        outcome = await session.grade()
        assert (outcome.status, outcome.reward) == ("scored", 1.0), outcome.metadata
        assert outcome.metadata["collect_failed"] == ["main: exit 7"]  # best effort, as Harbor runs them
    finally:
        await session.close()


# --- binds ---------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_services_that_bind_one_path_of_the_task_writable_share_one_copy_of_it(make_task, provider):
    task_dir = make_task("shared-bind", task_toml='[environment]\ndocker_image = "python:3.12-slim"\n')
    (task_dir / "environment" / "shared").mkdir()
    (task_dir / "environment" / "shared" / "seed.txt").write_text("seed\n")
    (task_dir / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    volumes:\n      - ./shared:/shared\n    depends_on: [web]\n"
        '  web:\n    image: python:3.12-slim\n    command: ["sleep", "600"]\n    volumes:\n'
        "      - ./shared:/shared:ro\n"
    )
    service = HarborEnvironmentService(task_dirs=[task_dir], provider=provider)
    session = await _open(service, "shared-bind")
    try:
        assert session.sandbox is not None and session.stack is not None
        assert (await session.sandbox.exec("echo from-main > /shared/from-main.txt")).exit_code == 0
        web = session.stack.services["web"]
        assert (await web.exec("cat /shared/from-main.txt")).stdout == "from-main\n"  # as compose shares it
        assert (await web.exec("cat /shared/seed.txt")).stdout == "seed\n"
        assert not (task_dir / "environment" / "shared" / "from-main.txt").exists()  # the cache is never written
        copy = session.stack.volume_path("platoon:copy-0")
        assert copy.is_relative_to(provider.root / session.stack.id)  # the stack's: reaping deletes it with the stack
    finally:
        await session.close()
    assert _nothing_left(provider)


@pytest.mark.asyncio
async def test_a_sidecar_binds_a_readable_copy_of_task_files_that_other_users_cannot_read(make_task, provider):
    """A task fetched under a umask that keeps other users out (027) has files only you and your group read, where a
    Docker host's checkout has them 0644, which a sidecar that reads them as a user of its own (the official postgres
    image runs its init scripts as postgres) needs.  Its read-only binds of them bind a copy, the stack's own, that
    every user reads (a+rX: an executable stays one), and the cache keeps its modes; what every user reads already
    binds the cache, and so does what only main binds, which reads it as you."""

    task_dir = make_task("umask-027", task_toml='[environment]\ndocker_image = "python:3.12-slim"\n')
    environment = task_dir / "environment"
    (environment / "conf").mkdir()
    files = {"init.sql": 0o640, "conf/app.conf": 0o640, "conf/start.sh": 0o750, "open.txt": 0o644, "mine.txt": 0o600}
    for name, mode in files.items():
        (environment / name).write_text(f"{name}\n")
        (environment / name).chmod(mode)
    (environment / "conf").chmod(0o750)
    (environment / "docker-compose.yaml").write_text(
        "services:\n  main:\n    volumes:\n      - ./mine.txt:/srv/mine.txt:ro\n    depends_on: [db]\n"
        '  db:\n    image: python:3.12-slim\n    command: ["sleep", "600"]\n    volumes:\n'
        "      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro\n      - ./conf:/etc/app:ro\n"
        "      - ./open.txt:/srv/open.txt:ro\n"
    )

    def mode(path: Path) -> int:
        return path.stat().st_mode & 0o7777

    service = HarborEnvironmentService(task_dirs=[task_dir], provider=provider)
    session = await _open(service, "umask-027")
    try:
        stack = _stack(session)
        db = cast(Any, stack.services["db"])
        init = db.host_path("/docker-entrypoint-initdb.d/init.sql").resolve()
        conf = db.host_path("/etc/app").resolve()
        assert init.is_relative_to(provider.root / stack.id) and conf.is_relative_to(provider.root / stack.id)
        modes = (mode(init), mode(conf), mode(conf / "app.conf"), mode(conf / "start.sh"))
        assert modes == (0o644, 0o755, 0o644, 0o755)
        assert init.read_text() == "init.sql\n" and (conf / "start.sh").read_text() == "conf/start.sh\n"
        assert db.host_path("/srv/open.txt").resolve() == (environment / "open.txt").resolve()
        assert cast(Any, session.sandbox).host_path("/srv/mine.txt").resolve() == (environment / "mine.txt").resolve()
        assert {name: mode(environment / name) for name in files} == files and mode(environment / "conf") == 0o750
    finally:
        await session.close()
    assert _nothing_left(provider)


@pytest.mark.asyncio
async def test_a_sidecar_s_masked_copy_of_readable_files_links_them_and_opens_its_directories(make_task, provider):
    """A masked bind's copy hard-links what every user reads, which keeps the cache's modes (an owner's executable
    stays 0744: the cache's inode is never changed), and makes its directories as Docker's host does (0755), under a
    umask of 027 too."""

    task_dir = make_task("masked-027", task_toml='[environment]\ndocker_image = "python:3.12-slim"\n')
    data = task_dir / "environment" / "data"
    (data / "hidden").mkdir(parents=True)
    (data / "run.sh").write_text("#!/bin/sh\n")
    (data / "run.sh").chmod(0o744)
    (data / "hidden" / "answer.txt").write_text("42\n")
    for path, mode in ((data, 0o755), (data / "hidden", 0o755), (data / "hidden" / "answer.txt", 0o644)):
        path.chmod(mode)  # every user reads them, whatever the umask that wrote them
    (task_dir / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    depends_on: [db]\n"
        '  db:\n    image: python:3.12-slim\n    command: ["sleep", "600"]\n    volumes:\n'
        "      - ./data:/srv/data:ro\n      - /srv/data/hidden\n"
    )
    service = HarborEnvironmentService(task_dirs=[task_dir], provider=provider)
    previous = os.umask(0o027)
    try:
        session = await _open(service, "masked-027")
    finally:
        os.umask(previous)
    try:
        copy = cast(Any, _stack(session).services["db"]).host_path("/srv/data").resolve()
        assert copy.is_relative_to(provider.root / _stack(session).id)
        assert (copy / "run.sh").stat().st_ino == (data / "run.sh").stat().st_ino
        assert (data / "run.sh").stat().st_mode & 0o7777 == 0o744
        assert list((copy / "hidden").iterdir()) == []  # masked
        assert [path.stat().st_mode & 0o7777 for path in (copy, copy / "hidden")] == [0o755, 0o755]
    finally:
        await session.close()
    assert _nothing_left(provider)


@pytest.mark.asyncio
async def test_reaping_a_stack_s_label_deletes_what_its_session_copied(make_task, provider):
    """What a stack's session copies of the task (a writable bind) and what its sidecars write in their log
    directories are the stack's volumes: reaping by label, as after a killed worker, deletes them with the stack, and
    nothing of the session is in node scratch."""

    task_dir = make_task("reaped", task_toml='[environment]\ndocker_image = "python:3.12-slim"\n')
    (task_dir / "environment" / "data").mkdir()
    (task_dir / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    depends_on: [web]\n"
        '  web:\n    image: python:3.12-slim\n    command: ["sleep", "600"]\n    volumes:\n'
        "      - ./data:/srv/data\n      - ${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}\n"
    )
    service = HarborEnvironmentService(task_dirs=[task_dir], provider=provider)
    session = await _open(service, "reaped")
    try:
        scratch = Path(os.environ["PLATOON_NODE_SCRATCH"])
        assert not scratch.exists() or list(scratch.iterdir()) == []
        assert await provider.reap({"platoon.rollout_id": "rollout-reaped"}) == 3  # main, web and the stack
        assert list(provider.root.iterdir()) == []
    finally:
        await session.close()  # what reaping took is gone already: closing is quiet
    assert _nothing_left(provider)


# --- what is refused, and when -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stacks_are_listed_where_they_run_and_skipped_where_they_do_not(provider: ProcessProvider, caplog):
    tasks = ["app-kv", "app-kv-crash", "app-kv-seeded", "app-kv-unhealthy", "mcp-http", "mcp-sse", "sidecar-artifacts"]
    assert sorted(await HarborEnvironmentService(STACKS, provider=provider).list_tasks("train")) == tasks

    class NoStacks(ProcessProvider):
        capabilities = ProcessProvider.capabilities - {"stacks"}

    for refusing in (
        HarborEnvironmentService(STACKS, provider=NoStacks(root=provider.root)),
        HarborEnvironmentService(STACKS, provider=provider, stacks="refuse"),
    ):
        with pytest.raises(ValueError, match="No runnable Harbor tasks .*multi-container stack"):
            await refusing.list_tasks("train")
    service = HarborEnvironmentService(STACKS, provider=provider, stacks="refuse")
    task = await HarborEnvironmentService(STACKS, provider=provider).get_task("app-kv")
    with pytest.raises(EnvSetupError, match="the environment sets stacks: refuse") as error:
        await service.open(task, rollout_id="r")
    assert error.value.failure_kind == "env_config" and _nothing_left(provider)


@pytest.mark.asyncio
async def test_a_fetched_stack_is_refused_when_the_tasks_are_listed_where_no_stack_runs(
    tmp_path: Path, provider: ProcessProvider, monkeypatch: pytest.MonkeyPatch
):
    use_site_cache(monkeypatch, tmp_path / "cache")
    dataset = tmp_path / "dataset"
    shutil.copytree(STACKS / "app-kv", dataset / "app-kv")
    settings: dict[str, Any] = {
        "dataset": str(dataset),
        "lock": str(tmp_path / "lock.json"),
        "refuse": {"sidecars": False},
    }
    lock = fetch(fetch_config(str(dataset), lock=tmp_path / "lock.json", refuse={"sidecars": False}))
    assert lock.get("app-kv").flags["stack"] == "sidecars" and lock.get("app-kv").refused is None

    runs = HarborEnvironmentService(provider=provider, **settings)
    await runs.check_tasks("train", ["app-kv"])  # the process provider runs stacks
    refuses = HarborEnvironmentService(provider=provider, stacks="refuse", **settings)
    with pytest.raises(EnvSetupError, match="1 of the 1 tasks of split 'train' are multi-container stacks") as error:
        await refuses.check_tasks("train", ["app-kv"])
    assert error.value.failure_kind == "env_config"


@pytest.mark.asyncio
async def test_fetched_stacks_that_no_rollout_could_start_are_refused_when_the_tasks_are_listed(
    tmp_path: Path, provider: ProcessProvider, monkeypatch: pytest.MonkeyPatch
):
    """A registry sidecar without a command, one that builds its image from a context the image map lacks (which
    platoon prepare builds: until then the run is not prepared), and a dependency on the health of a sidecar without a
    healthcheck: locked as stacks, refused when a run lists them, and each session open is a configuration error,
    never an infra one."""

    use_site_cache(monkeypatch, tmp_path / "cache")
    dataset = tmp_path / "dataset"
    registry = _variant("registry", dataset)
    _edit(registry / "environment" / "docker-compose.yaml", '    command: ["python3", "/srv/kv/kv.py"]\n', "")
    built = _variant("built", dataset)
    _edit(built / "environment" / "docker-compose.yaml", "    image: python:3.12-slim\n", "    build: ./kv\n")
    (built / "environment" / "kv" / "Dockerfile").write_text("FROM python:3.12-slim\nCOPY kv.py /srv/kv/kv.py\n")
    unchecked = _variant("unchecked", dataset)
    compose = unchecked / "environment" / "docker-compose.yaml"
    text = compose.read_text()
    compose.write_text(text[: text.index("    healthcheck:")] + "volumes:\n  kv-data:\n")  # service_healthy stays
    lock = fetch(fetch_config(str(dataset), lock=tmp_path / "lock.json", refuse={"sidecars": False}))
    assert {task.name: (task.flags["stack"], task.refused) for task in lock.tasks} == {
        name: ("sidecars", None) for name in ("built", "registry", "unchecked")
    }
    settings: dict[str, Any] = {
        "dataset": str(dataset),
        "lock": str(tmp_path / "lock.json"),
        "refuse": {"sidecars": False},
    }
    service = HarborEnvironmentService(provider=provider, **settings)
    unrunnable = "are multi-container stacks that no rollout can start"
    for name, listed, message in (
        (
            "registry",
            unrunnable,
            "the service kv runs the command of its image docker://python:3.12-slim, which platoon does",
        ),
        (
            "built",
            "build their images from Dockerfiles, which no image map holds",
            "Harbor task 'platoon/built': the service kv builds its image from environment/kv/Dockerfile .*: platoon "
            "prepare builds it",
        ),
        ("unchecked", unrunnable, "The service 'main' waits for 'kv' to be healthy, and 'kv' has no healthcheck"),
    ):
        with pytest.raises(EnvSetupError, match=f"1 of the 1 tasks of split 'train' {listed}"):
            await service.check_tasks("train", [name])
        with pytest.raises(EnvSetupError, match=message) as error:
            await _open(service, name)
        assert error.value.failure_kind == "env_config"
    assert not provider.root.exists() or list(provider.root.iterdir()) == []  # nothing started (the fetch has scratch)


@pytest.mark.parametrize(
    ("compose", "message"),
    [
        (
            "services:\n  main:\n    depends_on: [a]\n  a:\n    image: x\n    command: [x]\n    depends_on: [b]\n"
            "  b:\n    image: x\n    command: [x]\n    depends_on: [a]\n",
            "depend on one another in a cycle",
        ),
        ("services:\n  cache:\n    image: redis:7-alpine\n", "the command of its registry image redis:7-alpine"),
    ],
    ids=["cycle", "registry-image-without-a-command"],
)
def test_stacks_platoon_cannot_start_are_refused_when_the_task_loads(make_task, compose: str, message: str):
    task_dir = make_task("bad")
    (task_dir / "environment" / "docker-compose.yaml").write_text(compose)
    with pytest.raises(HarborTaskError, match=message):
        HarborTask.load(task_dir)


@pytest.mark.asyncio
async def test_network_policy_none_gives_a_stack_and_a_single_sandbox_no_network(provider: ProcessProvider, make_task):
    service = HarborEnvironmentService(STACKS, provider=provider, network_policy="none")
    session = await _open(service, "app-kv")
    try:
        assert session.stack is not None and session.stack.network == "none"
        assert session._lease.spec.network == "none"
    finally:
        await session.close()
    single = make_task("single", task_toml='[environment]\ndocker_image = "ubuntu:24.04"\n')
    spec = HarborEnvironmentService(task_dirs=[single], provider=provider, network_policy="none")._sandbox_spec(
        HarborTask.load(single), "r"
    )
    assert spec.network == "none"
    with pytest.raises(ValueError, match="network_policy must be one of"):
        HarborEnvironmentService(STACKS, provider=provider, network_policy="public")


@pytest.mark.asyncio
async def test_network_policy_none_is_named_where_a_provider_cannot_run_a_sandbox_without_a_network(
    make_task, tmp_path: Path
):
    class NoOffline(ProcessProvider):
        @classmethod
        def network_none_refusal(cls, options: Any) -> str:
            return "the provider cannot start a sandbox without a network"

    online = make_task("online", task_toml='[environment]\ndocker_image = "python:3.12-slim"\n')
    service = HarborEnvironmentService(
        task_dirs=[online], provider=NoOffline(root=tmp_path / "sb"), network_policy="none"
    )
    with pytest.raises(EnvSetupError, match="as the environment's network_policy: none makes every sandbox.*task"):
        await service.check_tasks("train", ["online"])
    with pytest.raises(EnvSetupError, match="runs without a network \\(the environment's network_policy: none\\)"):
        await _open(service, "online")


# --- deadlines -----------------------------------------------------------------------------------------------------


def test_a_stack_s_deadline_counts_its_services_along_the_longest_path(provider: ProcessProvider):
    service = HarborEnvironmentService(STACKS, provider=provider, sandbox_start_budget_s=100.0)
    single = 100.0 + 30.0 + 120.0 + service.close_timeout_s  # a start, the tests, grading, the close
    kv = 100.0 + (100 + 1) * (5.0 + 0.1)  # kv's start and its healthcheck's longest wait
    probes = 3 * 100 * (5.0 + 0.1)  # the start's last, one before the tests, one after
    assert service.rollout_overhead_s("app-kv") == pytest.approx(single + kv + probes)
    # app-kv-seeded's path is kv, then seed (its start, and its run, bounded by a start budget), then main.
    assert service.rollout_overhead_s("app-kv-seeded") == pytest.approx(single + kv + 2 * 100.0 + probes)
    # A separate verifier's start, and the collect hook's timeout.
    api = 100.0 + (100 + 1) * (5.0 + 0.1)
    assert service.rollout_overhead_s("sidecar-artifacts") == pytest.approx(single + api + probes + 100.0 + 30.0)


def test_a_deadline_counts_the_healthchecks_of_the_sidecars_images(make_task):
    """A sidecar on main's image (``${MAIN_IMAGE_NAME}``) waits for that image's HEALTHCHECK, which the start's and
    the probes' bounds count as they count compose's."""

    task_dir = make_task("image-check", task_toml='[environment]\ndocker_image = "python:3.12-slim"\n')
    (task_dir / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    depends_on:\n      api:\n        condition: service_healthy\n"
        '  api:\n    image: ${MAIN_IMAGE_NAME}\n    command: ["python3", "-m", "http.server"]\n'
    )
    task = HarborTask.load(task_dir)
    image = SessionImage(
        image="docker://python:3.12-slim",
        env={},
        workdir=None,
        healthcheck={"test": ["CMD", "true"], "interval": 30.0, "timeout": 30.0, "retries": 3, "start_period": 60.0},
    )

    def no_build(name: str, build: Any) -> SessionImage:
        raise AssertionError("nothing builds here")

    checks = sidecars.healthchecks(task, main_image=image, service_image=no_build)
    api = checks["api"]
    assert api is not None and api.worst_case_s == 60.0 + 4 * 60.0
    assert checks["main"] == api  # main runs the same image, so the same HEALTHCHECK, which _limits adds to main's
    start = sidecars.start_bound_s(task, checks, start_s=100.0, main_health_s=0.0, job_s=100.0)
    assert start == pytest.approx(100.0 + api.worst_case_s + 100.0)  # api's start and wait, then main's start
    assert sidecars.probe_bound_s(checks) == 3 * 60.0  # retries times (timeout and interval)


# --- leaks ---------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stacks_started_and_cancelled_mid_start_leave_nothing(service: HarborEnvironmentService, provider):
    """The exit check's leak test, smaller: stacks open and close, a few at a time, and some are cancelled while
    they start; afterwards no sandbox, stack directory, process or registration is left."""

    rng = random.Random(0)

    async def one(index: int) -> str:
        opening = asyncio.ensure_future(_open(service, "app-kv"))
        if index % 3 == 0:
            await asyncio.sleep(rng.uniform(0.0, 0.3))
            opening.cancel()
            try:
                session = await opening
            except asyncio.CancelledError:
                return "cancelled"
        else:
            session = await opening
        await session.close()
        return "closed"

    results = []
    for batch in range(4):
        results += await asyncio.gather(*(one(batch * 6 + index) for index in range(6)))
    assert "closed" in results and "cancelled" in results
    await _eventually(lambda: _leftovers(provider) == [])  # a cancelled start's threads delete what they made
    assert _nothing_left(provider)
    root = str(provider.root)
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            environ = Path(f"/proc/{pid}/environ").read_bytes()
        except OSError:
            continue
        assert f"PLATOON_SANDBOX_ROOT={root}".encode() not in environ, f"process {pid} was left running"


@pytest.mark.parametrize("task_id", ["app-kv", "logged"], ids=["shared", "separate"])
@pytest.mark.asyncio
async def test_a_grade_cancelled_while_the_sidecars_logs_are_copied_leaves_nothing(
    task_id: str, tmp_path: Path, provider: ProcessProvider, monkeypatch: pytest.MonkeyPatch
):
    """The copy runs on in its thread after the cancellation, and what it made is deleted once it is done."""

    dataset = tmp_path / "dataset"
    shutil.copytree(STACKS / "app-kv", dataset / "app-kv")
    logged = _variant("logged", dataset, source="sidecar-artifacts")
    _edit(
        logged / "environment" / "docker-compose.yaml",
        "      - ./api:/srv/api:ro\n",
        "      - ./api:/srv/api:ro\n      - ${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}\n",
    )
    copying, release = threading.Event(), threading.Event()
    snapshot = sidecars.snapshot_logs

    def slow_snapshot(directories: Any, into: Path) -> Any:
        copying.set()
        release.wait(10)
        return snapshot(directories, into)

    monkeypatch.setattr(sidecars, "snapshot_logs", slow_snapshot)
    service = HarborEnvironmentService(dataset, provider=provider)
    session = await _open(service, task_id)
    stack = session.stack
    assert stack is not None
    (stack.volume_path("platoon:logs-verifier") / "evidence.log").write_text("what a sidecar wrote\n")
    try:
        grading = asyncio.ensure_future(session.grade())
        await _eventually(copying.is_set)
        grading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await grading
        release.set()
    finally:
        release.set()
        await session.close()
    await _eventually(lambda: _leftovers(provider) == [])


def test_a_snapshot_of_the_sidecars_logs_copies_regular_files_and_never_through_a_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A link at any depth, a FIFO, and a file swapped for a link right after the copy looked at it (as a sidecar
    racing the copy might) are left out; what is outside is never read."""

    logs, outside = tmp_path / "logs", tmp_path / "outside"
    (logs / "deep").mkdir(parents=True)
    outside.mkdir()
    (outside / "secret").write_text("never copied\n")
    (logs / "kept.log").write_text("kept\n")
    (logs / "deep" / "kept-too.log").write_text("kept too\n")
    (logs / "link").symlink_to(outside / "secret")
    (logs / "linked-dir").symlink_to(outside)
    os.mkfifo(logs / "fifo")
    (logs / "swapped.log").write_text("a regular file, when looked at\n")
    real_stat, real_lstat, swapped = os.stat, os.lstat, []

    def swap_once(result: Any, path: Any) -> Any:  # once the copy has seen a regular file there
        named = isinstance(path, str | os.PathLike) and os.path.basename(os.fspath(path)) == "swapped.log"
        if named and not swapped:  # (a path; fwalk also looks at directories by their descriptors)
            swapped.append(path)
            os.unlink(logs / "swapped.log")
            os.symlink(outside / "secret", logs / "swapped.log")
        return result

    with monkeypatch.context() as patched:
        patched.setattr(os, "stat", lambda path, *args, **options: swap_once(real_stat(path, *args, **options), path))
        patched.setattr(os, "lstat", lambda path, *args, **options: swap_once(real_lstat(path, *args, **options), path))
        copied = sidecars.snapshot_logs({"verifier": logs}, tmp_path / "into")

    target = tmp_path / "into" / "verifier"
    assert copied == {"verifier": target} and swapped
    found = sorted(str(path.relative_to(target)) for path in target.rglob("*"))
    assert found == ["deep", "deep/kept-too.log", "kept.log"]
    assert not any(path.is_symlink() for path in target.rglob("*"))


@pytest.mark.parametrize("unreadable", ["file", "directory"])
def test_a_snapshot_of_the_sidecars_logs_raises_for_what_it_cannot_read(tmp_path: Path, unreadable: str):
    """What a sidecar's own user keeps from this host's user (in a pod, a user of its subordinate ids) is never left
    out of the copy the tests get; what is gone since it was listed is."""

    if os.geteuid() == 0:
        pytest.skip("root reads every file")
    logs = tmp_path / "logs"
    (logs / "judge").mkdir(parents=True)
    (logs / "judge" / "actions.jsonl").write_text("a record\n")
    kept = logs / "judge" if unreadable == "directory" else logs / "judge" / "actions.jsonl"
    kept.chmod(0)
    try:
        with pytest.raises(PermissionError):
            sidecars.snapshot_logs({"verifier": logs}, tmp_path / "into")
    finally:
        kept.chmod(0o700)


class _Downloads:
    """A sidecar's sandbox that downloads its log directory, or fails to (one that is down)."""

    def __init__(self, *, fails: bool = False) -> None:
        self.fails = fails
        self.asked: list[str] = []

    async def download(self, src: str, dst: Path) -> None:
        self.asked.append(src)
        if self.fails:
            raise RuntimeError("the sidecar is down")
        Path(dst).mkdir(parents=True)
        (Path(dst) / "actions.jsonl").write_text("a record\n")


class _Stack:
    def __init__(self, run_as: str, sandbox: _Downloads, volume: Path) -> None:
        self.services = {"main": object(), "judge": sandbox}
        self.run_modes = {"main": "root", "judge": run_as}
        self.volume = volume

    def volume_path(self, name: str) -> Path:
        assert name == "platoon:logs-verifier"
        return self.volume


@pytest.mark.asyncio
async def test_grading_reads_a_log_directory_that_a_subids_sidecar_binds_through_it(tmp_path: Path):
    """A sidecar with your subordinate ids may write its log as a user of its own, whose files this host's user may not
    read: it downloads them, as its root; another sidecar's, or one that is down, are read from the stack's volume."""

    logs = sidecars.SidecarLogs(logs="verifier", volume="platoon:logs-verifier", services={"judge": "/logs/verifier"})
    volume, downloads = tmp_path / "volume", tmp_path / "downloads"
    judge = _Downloads()
    source = await sidecars.log_source(cast(Any, _Stack("subids", judge, volume)), logs, downloads)
    assert source == downloads / "verifier" / "judge" and (source / "actions.jsonl").is_file()
    assert judge.asked == ["/logs/verifier"]
    for run_as, sandbox in (("caller", _Downloads()), ("subids", _Downloads(fails=True))):
        assert await sidecars.log_source(cast(Any, _Stack(run_as, sandbox, volume)), logs, tmp_path / run_as) == volume


@pytest.mark.parametrize("task_id", ["app-kv", "logged"], ids=["shared", "separate"])
@pytest.mark.asyncio
async def test_what_grading_cannot_read_of_the_sidecars_logs_fails_the_grade_rather_than_go_missing(
    task_id: str, tmp_path: Path, provider: ProcessProvider
):
    """A file of a sidecar's log directory that grading cannot read makes the outcome an infrastructure error, never a
    grade whose tests went without it."""

    if os.geteuid() == 0:
        pytest.skip("root reads every file")
    dataset = tmp_path / "dataset"
    shutil.copytree(STACKS / "app-kv", dataset / "app-kv")
    logged = _variant("logged", dataset, source="sidecar-artifacts")
    _edit(
        logged / "environment" / "docker-compose.yaml",
        "      - ./api:/srv/api:ro\n",
        "      - ./api:/srv/api:ro\n      - ${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}\n",
    )
    service = HarborEnvironmentService(dataset, provider=provider)
    session = await _open(service, task_id)
    assert session.stack is not None
    kept = session.stack.volume_path("platoon:logs-verifier") / "judge"
    kept.mkdir()
    (kept / "actions.jsonl").write_text("a record\n")
    kept.chmod(0)
    try:
        outcome = await session.grade()
    finally:
        kept.chmod(0o700)
        await session.close()
    kind = "verifier_setup" if task_id == "app-kv" else "artifact_collect"
    assert (outcome.status, outcome.failure_kind) == ("error", kind), outcome.reason
    assert "cannot copy what the sidecars wrote in their log directories" in (outcome.reason or "")


def test_the_fixtures_lock_as_stacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    use_site_cache(monkeypatch, tmp_path / "cache")
    lock = fetch(fetch_config(str(STACKS), lock=tmp_path / "lock.json", refuse={"sidecars": False}))
    assert {task.name: (task.flags["stack"], task.refused) for task in lock.tasks} == {
        name: ("sidecars", None)
        for name in (
            "app-kv",
            "app-kv-crash",
            "app-kv-seeded",
            "app-kv-unhealthy",
            "mcp-http",
            "mcp-sse",
            "sidecar-artifacts",
        )
    }
    plan = lock.get("sidecar-artifacts").compose
    assert plan is not None and json.dumps(plan["services"]["api"]["expose"]) == '["8000/tcp"]'


# --- privileges, for the docker provider alone ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stack_fetched_with_its_privileges_runs_on_docker_alone(
    tmp_path: Path, provider: ProcessProvider, monkeypatch: pytest.MonkeyPatch
):
    """A fetch with ``refuse: {privileges: false}`` locks a stack whose services ask for privileges; a run on a
    provider that gives none refuses it when it lists the tasks, and as it opens, and the docker provider's spec of it
    gives each service what it asks for."""

    use_site_cache(monkeypatch, tmp_path / "cache")
    dataset = tmp_path / "dataset"
    judged = _variant("judged", dataset)
    compose = judged / "environment" / "docker-compose.yaml"
    _edit(
        compose,
        "\n  kv:\n    image: python:3.12-slim\n",
        "\n    cap_add: [NET_ADMIN]\n  kv:\n    privileged: true\n    image: python:3.12-slim\n",
    )
    settings: dict[str, Any] = {
        "dataset": str(dataset),
        "lock": str(tmp_path / "lock.json"),
        "refuse": {"sidecars": False, "privileges": False},
    }
    lock = fetch(fetch_config(settings["dataset"], lock=settings["lock"], refuse=settings["refuse"]))
    judged_row = lock.get("judged")
    assert judged_row.refused is None and judged_row.compose is not None
    assert judged_row.compose["privileges"] == ["main.cap_add NET_ADMIN", "kv.privileged"]

    on_process = HarborEnvironmentService(provider=provider, **settings)
    with pytest.raises(EnvSetupError, match=r"stacks whose services ask for privileges \(such as judged: main.cap_"):
        await on_process.check_tasks("train", ["judged"])
    with pytest.raises(EnvSetupError, match="the 'process' sandbox provider gives no service such privileges") as error:
        await _open(on_process, "judged")
    assert error.value.failure_kind == "env_config"

    monkeypatch.setattr(docker_compose, "compose_refusal", lambda binary: None)  # a docker CLI with compose, if any
    on_docker = HarborEnvironmentService(provider="docker", **settings)  # never built: no stack starts here
    harbor_task = on_docker._lookup("judged")
    spec = on_docker._stack_plan(harbor_task, on_docker._sandbox_spec(harbor_task, "check"), setup=_nothing).spec
    assert (spec.services["main"].cap_add, spec.services["main"].privileged) == (("NET_ADMIN",), False)
    assert (spec.services["kv"].cap_add, spec.services["kv"].privileged) == ((), True)


async def _nothing(sandbox) -> None:
    pass
