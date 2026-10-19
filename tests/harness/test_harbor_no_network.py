"""A Harbor task without a network: a harness placed in its sandbox could not reach the model gateway on loopback, so
the rollout is refused before the harness starts (a harness_setup error that says why) instead of launched to crash.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from harness_fakes import FakeChatServer, FakeGateway, answer
from test_harness_rollout import rollout_config, runner

from platoon.harness.rollout import HARNESS_MISC_KEY, run_harness_rollout
from platoon.sandbox.process import ProcessProvider
from platoon.task import Task
from platoon.train.components import EnvironmentRoute

FIXTURE = Path(__file__).resolve().parent.parent / "envs" / "fixtures" / "harbor" / "hello-file"


class IsolatingProvider(ProcessProvider):
    """A process provider that declares it enforces ``network: none``, as the docker, enroot and (with
    ``allow_network_none``) singularity providers do."""

    capabilities = frozenset({"snapshot", "network_none", "host_loopback"})


class ElsewhereProvider(ProcessProvider):
    """A provider whose sandboxes do not share this host's network (as a remote or cloud one's would not) and that
    gives them no gateway route: it declares neither ``host_loopback`` nor ``routes``."""

    capabilities = frozenset({"snapshot"})


def _task_dir(tmp_path: Path, network_mode: str | None) -> Path:
    task_dir = tmp_path / "dataset" / "offline"
    shutil.copytree(FIXTURE, task_dir)
    if network_mode is not None:
        toml = task_dir / "task.toml"
        setting = f'[environment]\nnetwork_mode = "{network_mode}"\n'
        toml.write_text(toml.read_text().replace("[environment]\n", setting))
    return task_dir


async def _run(
    tmp_path: Path, task_dir: Path, *, provider: type = IsolatingProvider, placement: str = "auto"
) -> tuple[FakeGateway, FakeChatServer, dict]:
    server = FakeChatServer([answer("done")])
    gateway = FakeGateway(server)
    route = EnvironmentRoute(
        name="harbor",
        service="harbor",
        service_kwargs={
            "dataset_dir": str(task_dir.parent),
            "provider": f"{provider.__module__}:{provider.__qualname__}",
            "provider_options": {"root": str(tmp_path / "sandboxes")},
        },
        harness="platoon-bash",
        placement=placement,
        resolve_task=True,
    )
    collection = await run_harness_rollout(
        route.attach(Task(id=task_dir.name)), rollout_config(tmp_path), gateway=gateway, runner=runner(tmp_path)
    )
    return gateway, server, collection


@pytest.mark.asyncio
async def test_a_harness_is_not_launched_into_a_sandbox_without_a_network(tmp_path):
    gateway, server, collection = await _run(tmp_path, _task_dir(tmp_path, "no-network"))

    ((_, outcome),) = gateway.collected
    assert (outcome.status, outcome.failure_kind) == ("error", "harness_setup")
    assert "has no network" in outcome.reason and "placement: local" in outcome.reason
    assert server.requests == []  # never started
    (root,) = collection["trajectories"].values()
    assert root["misc"][HARNESS_MISC_KEY]["placement"] == "task_sandbox"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("network_mode", "provider", "placement"),
    [
        (None, IsolatingProvider, "auto"),  # the task has a network
        ("no-network", ProcessProvider, "auto"),  # the provider cannot isolate it: the sandbox shares the host's
        ("no-network", IsolatingProvider, "local"),  # the harness runs on the host, beside the gateway
    ],
)
async def test_a_harness_that_can_reach_the_gateway_still_runs(tmp_path, network_mode, provider, placement):
    gateway, server, _ = await _run(tmp_path, _task_dir(tmp_path, network_mode), provider=provider, placement=placement)

    ((_, outcome),) = gateway.collected
    assert outcome.status == "scored"
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_a_harness_is_not_launched_into_a_sandbox_off_this_hosts_network_without_a_route(tmp_path):
    gateway, server, collection = await _run(tmp_path, _task_dir(tmp_path, None), provider=ElsewhereProvider)

    ((_, outcome),) = gateway.collected
    assert (outcome.status, outcome.failure_kind) == ("error", "harness_setup")
    assert "does not share this host's network" in outcome.reason and "host_loopback" in outcome.reason
    assert server.requests == []  # never started
    (root,) = collection["trajectories"].values()
    assert root["misc"][HARNESS_MISC_KEY]["placement"] == "task_sandbox"


@pytest.mark.asyncio
async def test_off_this_hosts_network_a_harness_still_runs_beside_the_gateway(tmp_path):
    gateway, server, _ = await _run(tmp_path, _task_dir(tmp_path, None), provider=ElsewhereProvider, placement="local")

    ((_, outcome),) = gateway.collected
    assert outcome.status == "scored" and len(server.requests) == 1
