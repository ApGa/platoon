"""What a run readies before its first rollout (:mod:`platoon.train.startup`): the fork server's preload derived from
the environment entries, the task lists an in-process ORS environment builds once, and the toolchains staged ahead;
and the node services a start stops as it ends."""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from platoon.envs.ors.service import TaskRef
from platoon.envs.service import get_service
from platoon.train import startup
from platoon.train.auto import harness_toolchains
from platoon.train.components import HARNESS_ROLLOUT, normalize_environment_configs
from platoon.train.startup import HARNESS_ROLLOUT_MODULES, RolloutStartup, WorkerPreload, worker_preload

ENVS_DIR = str(Path(__file__).resolve().parent.parent / "envs")  # ors_toys
STARTUP = "platoon.train.startup"


@pytest.fixture
def toys(monkeypatch):
    monkeypatch.syspath_prepend(ENVS_DIR)


def _environments(*entries: dict):
    return normalize_environment_configs(list(entries))


ORS_ENTRY = {"service": "python://ors_toys:CounterEnv", "harness": "platoon-codeact", "agents": {"max_depth": 0}}


# ------------------------------------------------------------------------------------- the derived preload


def test_an_ors_environment_played_by_a_function_harness_preloads_what_its_rollouts_open_with(toys):
    preload = worker_preload(_environments(ORS_ENTRY), tasks=[{"task_id": "train:0"}, {"task_id": "train:2"}, "test:1"])

    assert preload.modules[: len(HARNESS_ROLLOUT_MODULES)] == HARNESS_ROLLOUT_MODULES  # harness_rollout's, first
    assert HARNESS_ROLLOUT.partition(":")[0] in preload.modules
    for module in ("platoon.envs.ors", "openreward", "ors_toys", "platoon.harness.builtin.codeact.harness"):
        assert module in preload.modules, module
    assert len(set(preload.modules)) == len(preload.modules)
    assert preload.prepare == (
        {"ref": f"{STARTUP}:warm_harness", "kwargs": {"ref": "platoon-codeact"}},
        {"ref": f"{STARTUP}:warm_program", "kwargs": {"name": "single"}},
        {"ref": f"{STARTUP}:warm_task_lists", "kwargs": {"service": ORS_ENTRY["service"], "splits": ["test", "train"]}},
    )


def test_a_mixture_builds_each_in_process_environment_s_task_lists_for_its_own_splits(toys):
    environments = _environments(
        {**ORS_ENTRY, "name": "counter"},
        {"service": "python://ors_toys:GradedCounterEnv", "harness": "platoon-codeact", "name": "graded"},
    )
    tasks = [{"task_id": "counter::train:0", "_platoon_environment": "counter"}, {"task_id": "graded::test:1"}]

    preload = worker_preload(environments, tasks=tasks)

    warmed = {
        step["kwargs"]["service"]: step["kwargs"]["splits"] for step in preload.prepare if "service" in step["kwargs"]
    }
    assert warmed == {"python://ors_toys:CounterEnv": ["train"], "python://ors_toys:GradedCounterEnv": ["test"]}
    assert [step["kwargs"] for step in preload.prepare].count({"ref": "platoon-codeact"}) == 1  # each build once


def test_a_harbor_environment_preloads_its_service_and_sandbox_provider_and_a_spec_runs_no_program(tmp_path):
    entry = {
        "service": "harbor",
        "service_kwargs": {"dataset_dir": str(tmp_path), "provider": "singularity"},
        "harness": "openhands",
    }

    preload = worker_preload(_environments(entry), tasks=[{"task_id": "task-a"}])

    for module in ("platoon.envs.harbor", "platoon.sandbox.singularity", "hpc_sandbox", "platoon.harness.runner"):
        assert module in preload.modules, module
    assert "openreward" not in preload.modules
    assert preload.prepare == ({"ref": f"{STARTUP}:warm_harness", "kwargs": {"ref": "openhands"}},)


def test_an_entry_s_own_rollouts_and_the_run_s_extra_modules_are_preloaded(toys):
    own = {"rollout": "ors_toys:CounterEnv", "task_loader": "platoon.task:Task", "name": "own"}
    evaluated = {**ORS_ENTRY, "name": "ors", "eval_rollout": "platoon.programs:get_program"}
    environments = _environments(own, evaluated)

    train = worker_preload(environments, phase="train", extra=["my_env.extras", "openreward"])
    evaluation = worker_preload(environments, phase="eval")

    assert "ors_toys" in train.modules  # the entry's own rollout's module
    assert train.modules[-1] == "my_env.extras"  # the run's own, after the derived ones, each once
    assert "platoon.programs" in evaluation.modules and HARNESS_ROLLOUT_MODULES[0] in train.modules
    assert WorkerPreload() == WorkerPreload(modules=(), prepare=())


# -------------------------------------------------------------------------- task lists built once, in the server


def test_warm_task_lists_lists_each_split_once_and_the_lookups_read_what_a_fresh_process_would(toys):
    from ors_toys import CounterEnv

    listed: list[str] = []

    class CountedEnv(CounterEnv):
        @classmethod
        def list_tasks(cls, split: str) -> list[dict]:
            listed.append(split)
            return super().list_tasks(split)

    sys.modules["ors_toys"].CountedEnv = CountedEnv  # importable by reference, as a service names it
    try:
        fresh = get_service("python://ors_toys:CountedEnv")
        expected = asyncio.run(fresh.get_task("train:1"))
        listed.clear()
        CountedEnv.__dict__["_tasks_cache"].clear()  # a new process: nothing listed yet

        warmed = get_service("python://ors_toys:CountedEnv")
        warmed.warm_task_lists(["train", "nonsuch", "train"])
        assert listed == ["train"]
        task = asyncio.run(get_service("python://ors_toys:CountedEnv").get_task("train:1"))
        spec = asyncio.run(warmed._local_spec(TaskRef.parse("train:1")))
    finally:
        del sys.modules["ors_toys"].CountedEnv
    assert listed == ["train"]  # the lookups read the cache: no listing in the "worker"
    assert task == expected and spec == CounterEnv.list_tasks("train")[1]


# ------------------------------------------------------------------------------ toolchains staged ahead of rollouts


def test_a_run_stages_the_named_toolchains_of_its_toml_harness_specs_each_once(tmp_path):
    entry = {"service": "harbor", "service_kwargs": {"dataset_dir": str(tmp_path)}}
    environments = _environments(
        {**entry, "name": "a", "harness": "openhands"},
        {**entry, "name": "b", "harness": {"openhands": 1.0, "platoon-bash": 1.0}},
        {**entry, "name": "c", "harness": "platoon-codeact"},
    )

    assert harness_toolchains(environments) == ["openhands-1.54.0"]  # a function harness and platoon-bash need none


class _Workflow:
    def __init__(self, events: list[str], name: str, fail: bool = False) -> None:
        self.events, self.name, self.fail = events, name, fail

    async def warm_up(self) -> None:
        self.events.append(f"warm_up {self.name}")
        if self.fail:
            raise RuntimeError("cannot start")


def test_the_startup_starts_the_workers_and_stages_on_a_thread_of_its_own_until_ready(monkeypatch, caplog):
    events: list[str] = []
    release = threading.Event()

    def prestage(names, log):
        events.append(f"staging {list(names)} on {threading.current_thread().name}")
        release.wait(30)
        events.append("staged")
        return {}

    monkeypatch.setattr("platoon.train.auto.harness_toolchains", lambda environments: ["tc-1.0", "tc-1.0"])
    monkeypatch.setattr("platoon.harness.toolchains.prestage", prestage)

    async def scenario() -> None:
        workflows = [_Workflow(events, "train"), None, _Workflow(events, "eval", fail=True), object()]
        begun = await RolloutStartup.begin([], workflows)
        events.append("begun")  # begin returns while the staging runs
        await asyncio.sleep(0.05)
        release.set()
        await begun.ready()
        events.append("ready")

    with caplog.at_level(logging.INFO, logger=STARTUP):
        asyncio.run(scenario())

    staging = "staging ['tc-1.0'] on platoon-toolchain-staging"
    assert events[:2] == ["warm_up train", "warm_up eval"]  # the workers first, then the staging, on its own thread
    assert sorted(events[2:4]) == sorted(["begun", staging])  # begin returned while the staging ran
    assert events[4:] == ["staged", "ready"]
    assert "could not start the rollout workers of" in caplog.text  # logged; its first rollouts start them
    assert "Staged the toolchains in" in caplog.text


def test_a_staging_that_fails_is_logged_and_the_rollouts_go_on(monkeypatch, caplog):
    def prestage(names, log):
        raise OSError("disk full")

    monkeypatch.setattr("platoon.train.auto.harness_toolchains", lambda environments: ["tc-1.0"])
    monkeypatch.setattr("platoon.harness.toolchains.prestage", prestage)

    async def scenario() -> None:
        await (await RolloutStartup.begin([])).ready()

    with caplog.at_level(logging.WARNING, logger=STARTUP):
        asyncio.run(scenario())
    assert "could not stage the toolchains ['tc-1.0'] ahead of the rollouts" in caplog.text


def test_the_rollouts_wait_for_a_staging_only_so_long_and_not_past_a_shutdown(monkeypatch, caplog):
    """A staging that hangs (a stuck read of the shared filesystem, a lock never released) holds the first rollouts
    for ``STAGING_WAIT_S`` at most, and not past a shutdown request: the rollouts then start and stage the toolchain
    themselves, as before; the staging goes on, and how it ended is logged."""

    release = threading.Event()

    def prestage(names, log):
        release.wait(30)
        return {}

    monkeypatch.setattr("platoon.train.auto.harness_toolchains", lambda environments: ["tc-1.0"])
    monkeypatch.setattr("platoon.harness.toolchains.prestage", prestage)
    monkeypatch.setattr(startup, "STAGING_WAIT_S", 0.3)

    async def scenario() -> list[float]:
        begun = await RolloutStartup.begin([])
        waits = []
        began = time.monotonic()
        await begun.ready()  # STAGING_WAIT_S
        waits.append(time.monotonic() - began)
        stop = asyncio.Event()
        asyncio.get_running_loop().call_later(0.1, stop.set)
        began = time.monotonic()
        await begun.ready(timeout=30.0, stop=stop)
        waits.append(time.monotonic() - began)
        release.set()
        for _ in range(100):  # the staging ends after the rollouts started
            if begun._staging.done():
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0)
        return waits

    with caplog.at_level(logging.INFO, logger=STARTUP):
        waits = asyncio.run(scenario())

    assert 0.25 < waits[0] < 10 and 0.05 < waits[1] < 10
    assert "Starting the rollouts before the toolchains tc-1.0 are staged (after 0s)" in caplog.text
    assert "Starting the rollouts before the toolchains tc-1.0 are staged (a shutdown was asked for)" in caplog.text
    assert "Staged the toolchains in" in caplog.text and "after the rollouts started" in caplog.text


def test_without_toolchains_or_workflows_the_startup_is_ready_at_once():
    async def scenario() -> None:
        begun = await RolloutStartup.begin([])
        assert begun.toolchains == [] and begun._staging is None
        await begun.ready()

    asyncio.run(scenario())
    assert startup.__all__  # the module's public names


# ------------------------------------------------------------------------------------- node services as the start ends


class _Startup:
    """A rollout startup whose node services failed (``failure``), or not."""

    def __init__(self, failure: str | None) -> None:
        self.failure = failure
        self.closed = 0

    async def close(self) -> str | None:
        self.closed += 1
        return self.failure


class _Run:
    """A start's lifecycle that ends, or fails with ``error``."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error

    async def execute(self) -> None:
        if self.error is not None:
            raise self.error


def test_a_start_stops_its_node_services_as_it_ends_and_fails_once_they_failed():
    from platoon.envs.harbor.node_services import NodeServiceError
    from platoon.train.trainer.rl import PlatoonTinkerRLTrainer

    trainer: Any = object.__new__(PlatoonTinkerRLTrainer)  # only what _execute uses
    asyncio.run(trainer._execute(_Run()))  # no startup yet: nothing to stop
    trainer._startup = served = _Startup(None)
    asyncio.run(trainer._execute(_Run()))
    trainer._startup = failed = _Startup("the judge could not be started again")
    with pytest.raises(NodeServiceError, match="node services failed: the judge could not be started again"):
        asyncio.run(trainer._execute(_Run()))
    with pytest.raises(KeyError):  # the start's own failure is its error
        asyncio.run(trainer._execute(_Run(KeyError("checkpoint"))))
    assert (served.closed, failed.closed) == (1, 2)


def test_without_node_services_the_startup_hears_of_no_failure_and_closes_at_once():
    begun = asyncio.run(RolloutStartup.begin([]))
    heard: list[str] = []
    begun.on_node_service_failure(heard.append)
    assert asyncio.run(begun.close()) is None and heard == []
