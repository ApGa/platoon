"""Stacks on the docker provider (``platoon.sandbox.compose``), against a fake docker CLI (``fake_docker_cli.py``):
the compose project a stack becomes and every command it runs, its networks per phase, its volumes, starts that fail
or are cancelled and leave nothing, the teardown of what services wrote as other users, reaping by label, and the
providers' refusals.  Each Docker fixture (``tests/envs/fixtures/harbor-stacks-docker``) and stack fixture
(``tests/envs/fixtures/harbor-stacks``, and Harbor's hello-mcp) opens and closes as a Harbor session, and the project
it becomes and the commands each of its services runs match ``compose_projects/<fixture>.yaml``
(``PLATOON_RECORD_COMPOSE_PROJECTS=1`` records them again)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from fake_docker_cli import command_name
from site_cache import use_site_cache

from platoon.envs.harbor import HarborEnvironmentService
from platoon.envs.harbor.builders import plan_images
from platoon.envs.harbor.fetch import fetch, fetch_config
from platoon.envs.harbor.images import ImageEntry, update_image_map
from platoon.envs.service import EnvSetupError
from platoon.sandbox import StackStartError, cleanup, stack_refusal
from platoon.sandbox.base import Mount, SandboxSpec
from platoon.sandbox.compose import (
    PROJECT_FILE,
    STACK_RECORD,
    compose_refusal,
    remove_directory,
    start_compose_stack,
)
from platoon.sandbox.docker import DockerProvider
from platoon.sandbox.stacks import (
    Dependency,
    DrivenStack,
    Healthcheck,
    ServiceSpec,
    StackNetwork,
    StackSpec,
    Volume,
)
from platoon.utils.stopping import cleanups, process_start_time

FAKE = Path(__file__).with_name("fake_docker_cli.py")
PROJECTS = Path(__file__).with_name("compose_projects")
FIXTURES = Path(__file__).resolve().parents[1] / "envs" / "fixtures"
WORKER = cleanup.WORKER_LABEL
FAKE_VARIABLES = (
    "FAKE_DOCKER_FAIL",
    "FAKE_DOCKER_FAIL_ONCE",
    "FAKE_DOCKER_ERROR",
    "FAKE_DOCKER_SLOW",
    "FAKE_DOCKER_LOST",
    "FAKE_DOCKER_EXITED",
    "FAKE_DOCKER_UNHEALTHY",
    "FAKE_DOCKER_NO_COMPOSE",
    "FAKE_DOCKER_SERVER_VERSION",
)


class FakeDocker:
    """What the fake docker CLI was asked, and what it holds now."""

    def __init__(self, root: Path) -> None:
        self.log = root / "docker.log"
        self.state_file = root / "docker-state.json"

    def calls(self) -> list[list[str]]:
        """The calls since the last look."""

        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        self.log.write_text("")
        return calls

    def state(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return {"containers": {}, "networks": {}, "projects": {}}
        return json.loads(self.state_file.read_text())

    def called(self, command: str) -> bool:
        """Whether a call of ``command`` (``start``, ``create``, ``network create``: as the fake names its calls) was
        made since the last look."""

        lines = self.log.read_text().splitlines() if self.log.exists() else []
        return any(command_name(call) == command for call in map(json.loads, lines))

    def daemon_done(self) -> bool:
        """Whether the daemon carried out the call that ``$FAKE_DOCKER_SLOW`` names."""

        return self.log.with_suffix(".done").exists()

    def run(self, *args: str) -> None:
        """Run the fake CLI by hand, as something outside platoon would."""

        subprocess.run(["docker", *args], check=True, capture_output=True)


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cleanup, "_live", {})
    monkeypatch.setattr(cleanup, "_labels", {})
    monkeypatch.setattr(cleanup, "_mirror_path", None)
    monkeypatch.setattr(cleanup, "_shut_down", False)
    compose_refusal.cache_clear()


@pytest.fixture
def docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    binary = tmp_path / "bin" / "docker"
    binary.parent.mkdir()
    binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    binary.chmod(0o755)
    fake = FakeDocker(tmp_path)
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(fake.log))
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(fake.state_file))
    monkeypatch.setenv("PATH", f"{binary.parent}:{Path(sys.executable).parent}:/usr/bin:/bin")
    for name in FAKE_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return fake


@pytest.fixture
def provider(docker: FakeDocker, tmp_path: Path) -> DockerProvider:
    return DockerProvider(stack_root=tmp_path / "stacks")


def _spec(
    tmp_path: Path, *, network: StackNetwork = "egress", setup: Any = None, filled: list[Path] | None = None
) -> StackSpec:
    """main, which waits for kv to be healthy and reaches the model gateway's socket; kv, with a healthcheck, an alias,
    volumes of every kind and what only some runtimes apply; probe, which shares kv's network."""

    gateway = tmp_path / "gateway"
    gateway.mkdir(exist_ok=True)
    data = tmp_path / "task" / "data"
    data.mkdir(parents=True, exist_ok=True)

    def fill(directory: Path) -> None:
        (filled if filled is not None else []).append(directory)
        (directory / "tree").mkdir()
        (directory / "tree" / "seed.txt").write_text("seed")

    copy = "platoon:copy-0"
    main = ServiceSpec(
        sandbox=SandboxSpec(
            image="docker://python:3.12-slim",
            workdir="/app",
            env={"KV_URL": "http://kv:6379", "PRICE": "$5 or ${PRICE}"},
            mounts=[Mount(source=str(data), target="/data")],
            cpus=1.5,
            memory_mb=512,
            labels={"platoon.rollout_id": "r1"},
            upstreams={"gateway": f"unix://{gateway}/gateway.sock"},
        ),
        depends_on={"kv": Dependency("service_healthy")},
        volumes=[Volume(target="/app/shared", name=copy, subpath="tree", fill=fill)],
        setup=setup,
    )
    kv = ServiceSpec(
        sandbox=SandboxSpec(
            image="python:3.12-slim", workdir="/srv", env={"MODE": "kv"}, labels={"platoon.rollout_id": "r1"}
        ),
        command=["python3", "-m", "http.server", "6379"],
        healthcheck=Healthcheck(command=("python3", "-c", "print('ok')"), interval_s=0.01, retries=3),
        volumes=[
            Volume(target="/var/lib/kv"),
            Volume(target="/run/kv", tmpfs=True),
            Volume(target="/shared", name=copy, subpath="tree", read_only=True),
            Volume(target="/logs/verifier", name="platoon:logs-verifier"),
        ],
        ports=(6379,),
        aliases=("store",),
        shm_mb=64,
        ulimits={"nofile": (1024, 2048)},
        cap_add=("SYS_PTRACE",),
        stop_signal="SIGINT",
    )
    probe = ServiceSpec(
        sandbox=SandboxSpec(image="busybox:1.36", workdir="/"), command=["sleep", "1000"], network_of="kv"
    )
    return StackSpec(
        services={"main": main, "kv": kv, "probe": probe}, network=network, labels={"platoon.rollout_id": "r1"}
    )


def _project(calls: list[list[str]]) -> str:
    """The compose project of a stack's calls."""

    return next(call[call.index("--project-name") + 1] for call in calls if "--project-name" in call)


def _replaced(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _normalized(value: Any, replacements: list[tuple[str, str]]) -> Any:
    """``value`` (a project file, commands) with the test's paths and the stack's name replaced, and each command's
    marker and each script (an argument of several lines) left out, so that it reads the same in every run."""

    if isinstance(value, str):
        if "\n" in value:
            return "<script>"
        if value.startswith("--env=PLATOON_EXEC="):
            return "--env=PLATOON_EXEC=<marker>"
        return _replaced(value, replacements)
    if isinstance(value, list):
        return [_normalized(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _normalized(item, replacements) for key, item in value.items()}
    return value


def _by_service(calls: list[list[str]], project: str) -> dict[str, list[list[str]]]:
    """The calls by the service they concern (``<project>`` for the stack's own), each service's in its order: the
    services start at once where they wait for nothing, so only each one's own calls come in a fixed order."""

    def service(call: list[str]) -> str:
        if call[0] == "compose" and "create" in call:
            return call[-1]
        for arg in call[1:]:
            name = arg.split(":", 1)[0]  # docker cp <container>:<path>
            if name.startswith(f"{project}-") and name not in (f"{project}-internal", f"{project}-egress"):
                return name.removeprefix(f"{project}-")
        return "<project>"

    grouped: dict[str, list[list[str]]] = {}
    for call in calls:
        grouped.setdefault(service(call), []).append(call)
    return grouped


async def _settled() -> None:
    """Wait for the cleanups a cancelled start left running (rollbacks, the stack's stop)."""

    while pending := [task for task in cleanups() if not task.done()]:
        await asyncio.wait(pending)


def _nothing_left(docker: FakeDocker, provider: DockerProvider) -> bool:
    """No container, network, project directory or registration for cleanup is left."""

    state = docker.state()
    left = {
        "containers": sorted(state["containers"]),
        "networks": sorted(state["networks"]),
        "directories": sorted(provider.stack_root.iterdir()) if provider.stack_root.exists() else [],
        "registered": len(cleanup._live),
    }
    assert left == {"containers": [], "networks": [], "directories": [], "registered": 0}, left
    return True


# --- a stack as a compose project ----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stack_runs_as_a_compose_project_of_its_own(docker: FakeDocker, provider: DockerProvider, tmp_path):
    prepared: list[str] = []

    async def setup(sandbox) -> None:
        prepared.append(sandbox.id)
        await sandbox.exec(["touch", "/app/ready"])

    filled: list[Path] = []
    stack = await start_compose_stack(provider, _spec(tmp_path, setup=setup, filled=filled))
    calls = docker.calls()
    project = stack.id
    directory = tmp_path / "stacks" / project
    replacements = [(str(tmp_path), "<tmp>"), (project, "<project>")]

    assert project.startswith("platoon-stack-") and _project(calls) == project
    # JSON, which compose reads as the YAML it also is: no value changes its type on the way.
    assert _normalized(json.loads((directory / PROJECT_FILE).read_text()), replacements) == {
        "name": "<project>",
        "services": {
            "main": {
                "image": "python:3.12-slim",
                "container_name": "<project>-main",
                "entrypoint": ["sleep", "infinity"],  # main idles: its commands are exec'd into it
                "working_dir": "/app",
                "labels": {"platoon.rollout_id": "r1", "platoon.stack": "<project>", "platoon.service": "main"},
                "environment": {"KV_URL": "http://kv:6379", "PRICE": "$$5 or $${PRICE}"},  # compose interpolates none
                "volumes": [
                    {"type": "bind", "source": "<tmp>/task/data", "target": "/data", "read_only": True},
                    {"type": "bind", "source": "<tmp>/gateway", "target": "/run/platoon/gateway", "read_only": True},
                    {
                        "type": "bind",
                        "source": "<tmp>/stacks/<project>/volumes/platoon%3Acopy-0/tree",
                        "target": "/app/shared",
                        "read_only": False,
                    },
                ],
                "networks": {"internal": {}},
                "cpus": 1.5,
                "mem_limit": "512m",
            },
            "kv": {
                "image": "python:3.12-slim",
                "container_name": "<project>-kv",
                "entrypoint": ["python3", "-m", "http.server", "6379"],  # the final argv, the image's CMD dropped
                "working_dir": "/srv",
                "labels": {"platoon.rollout_id": "r1", "platoon.stack": "<project>", "platoon.service": "kv"},
                "environment": {"MODE": "kv"},
                "volumes": [
                    {
                        "type": "bind",
                        "source": "<tmp>/stacks/<project>/services/kv/0",
                        "target": "/var/lib/kv",
                        "read_only": False,
                    },
                    {"type": "tmpfs", "target": "/run/kv"},
                    {
                        "type": "bind",
                        "source": "<tmp>/stacks/<project>/volumes/platoon%3Acopy-0/tree",
                        "target": "/shared",
                        "read_only": True,
                    },
                    {
                        "type": "bind",
                        "source": "<tmp>/stacks/<project>/volumes/platoon%3Alogs-verifier",
                        "target": "/logs/verifier",
                        "read_only": False,
                    },
                ],
                "networks": {"internal": {"aliases": ["store"]}},
                "shm_size": "64m",
                "ulimits": {"nofile": {"soft": 1024, "hard": 2048}},
                "cap_add": ["SYS_PTRACE"],
                "stop_signal": "SIGINT",
            },
            "probe": {
                "image": "busybox:1.36",
                "container_name": "<project>-probe",
                "entrypoint": ["sleep", "1000"],
                "working_dir": "/",
                "labels": {"platoon.rollout_id": "r1", "platoon.stack": "<project>", "platoon.service": "probe"},
                "network_mode": "service:kv",
            },
        },
        "networks": {
            "internal": {
                "name": "<project>-internal",
                "internal": True,
                "driver_opts": {"com.docker.network.bridge.inhibit_ipv4": "true"},  # no address on the host
                "labels": {"platoon.rollout_id": "r1", "platoon.stack": "<project>"},
            }
        },
    }
    record = json.loads((directory / STACK_RECORD).read_text())
    assert record == {
        "labels": {"platoon.rollout_id": "r1", "platoon.stack": project},
        "image": "python:3.12-slim",  # what deletes what the services wrote, at the end
        "owner_pid": os.getpid(),
        "owner_start_time": process_start_time(os.getpid()),
    }

    # Every service: created by compose, connected to the egress network but probe (which shares kv's), started.
    services = _by_service(_normalized(calls, replacements), "<project>")
    create = ["compose", "--project-name", "<project>", "--file", "<tmp>/stacks/<project>/compose.yaml", "create"]
    assert services["<project>"] == [
        [
            "network",
            "create",
            "--driver=bridge",
            "--label=platoon.rollout_id=r1",
            "--label=platoon.stack=<project>",
            "<project>-egress",
        ]
    ]
    assert services["kv"][:3] == [
        [*create, "--no-recreate", "kv"],
        ["network", "connect", "<project>-egress", "<project>-kv"],
        ["start", "<project>-kv"],
    ]
    assert services["probe"][:2] == [[*create, "--no-recreate", "probe"], ["start", "<project>-probe"]]
    assert services["main"][:3] == [
        [*create, "--no-recreate", "main"],
        ["network", "connect", "<project>-egress", "<project>-main"],
        ["start", "<project>-main"],
    ]
    # kv's healthcheck ran in its container until it passed, then once more as the start's last probe.
    check = ["exec", "--workdir", "/srv", "--env=PLATOON_EXEC=<marker>", "<project>-kv", "python3", "-c", "print('ok')"]
    assert services["kv"][3:] == [
        check,
        ["inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}", "<project>-kv"],
        check,
    ]
    # main lists its own processes, starts the gateway route's forwarder, then its setup runs, before the last probe.
    main = [call for call in services["main"] if call[0] == "exec"]
    assert main[0][-1] == "platoon-list" and "platoon-route" in main[-2] and main[-1][-2:] == ["touch", "/app/ready"]
    assert prepared == [f"{project}-main"]
    assert stack.main.routes == {"gateway": "http://127.0.0.1:18080"}
    assert stack.host_loopback is False  # its services share the project's network, not this host's
    assert set(stack.services) == {"main", "kv", "probe"} and stack.run_modes == {
        "kv": "subids",
        "probe": "subids",
        "main": "subids",
    }

    # Volumes: directories any service's user may write, made and filled (once, for the services that share it).
    volumes = directory / "volumes"
    assert filled == [volumes / "platoon%3Acopy-0"]
    assert (volumes / "platoon%3Acopy-0" / "tree" / "seed.txt").read_text() == "seed"
    for made in (
        volumes / "platoon%3Acopy-0",
        volumes / "platoon%3Alogs-verifier",
        directory / "services" / "kv" / "0",
    ):
        assert made.stat().st_mode & 0o777 == 0o777
    assert stack.volume_path("platoon:logs-verifier") == volumes / "platoon%3Alogs-verifier"
    with pytest.raises(KeyError):
        stack.volume_path("platoon:nothing")

    await stack.stop()
    services = _by_service(_normalized(docker.calls(), replacements), "<project>")
    for name in ("main", "kv", "probe"):
        assert services[name] == [["rm", "--force", "--volumes", f"<project>-{name}"]]
    assert services["<project>"] == [  # what carries its label, whatever it knows it made
        ["ps", "--all", "--quiet", "--filter=label=platoon.stack=<project>"],
        ["network", "ls", "--quiet", "--filter=label=platoon.stack=<project>"],
        ["network", "rm", "<project>-egress", "<project>-internal"],
    ]
    assert _nothing_left(docker, provider)
    await stack.stop()  # idempotent
    assert docker.calls() == []


@pytest.mark.asyncio
async def test_the_egress_network_follows_the_stacks_network(docker: FakeDocker, provider: DockerProvider, tmp_path):
    stack = await start_compose_stack(provider, _spec(tmp_path, network="none"))
    calls = docker.calls()
    project = stack.id
    egress = f"{project}-egress"
    assert not any(call[:2] in (["network", "create"], ["network", "connect"]) for call in calls)
    assert docker.state()["containers"][f"{project}-main"]["networks"] == [f"{project}-internal"]

    await stack.set_network("egress")  # made when first needed, for every service that has a network of its own
    calls = docker.calls()
    assert calls[0][:2] == ["network", "create"] and calls[0][-1] == egress
    assert sorted(call[2:] for call in calls[1:]) == [[egress, f"{project}-kv"], [egress, f"{project}-main"]]
    assert all(call[:2] == ["network", "connect"] for call in calls[1:])

    await stack.set_network("none")
    calls = docker.calls()
    assert sorted(calls) == [
        ["network", "disconnect", egress, f"{project}-kv"],
        ["network", "disconnect", egress, f"{project}-main"],
    ]
    assert docker.state()["containers"][f"{project}-kv"]["networks"] == [f"{project}-internal"]
    await stack.set_network("egress")  # the network made before
    assert sorted(call[1] for call in docker.calls()) == ["connect", "connect"]

    await stack.stop()
    assert ["network", "rm", egress, f"{project}-internal"] in docker.calls()
    assert _nothing_left(docker, provider)


@pytest.mark.asyncio
async def test_a_network_switch_that_fails_midway_completes_when_tried_again(
    docker: FakeDocker, provider: DockerProvider, tmp_path, monkeypatch
):
    stack = await start_compose_stack(provider, _spec(tmp_path, network="none"))
    egress = f"{stack.id}-egress"

    def on_egress() -> list[str]:
        containers = docker.state()["containers"]
        return sorted(
            name.removeprefix(f"{stack.id}-") for name, each in containers.items() if egress in each["networks"]
        )

    monkeypatch.setenv("FAKE_DOCKER_FAIL_ONCE", "network connect")
    with pytest.raises(RuntimeError, match=f"Could not connect the service (kv|main) to {egress}: simulated failure"):
        await stack.set_network("egress")
    assert len(on_egress()) == 1  # the other one was connected
    await stack.set_network("none")  # as for the agents' next phase: the one that was connected is disconnected
    assert on_egress() == [] and stack.network == "none"
    await stack.set_network("egress")
    assert on_egress() == ["kv", "main"]

    monkeypatch.setenv("FAKE_DOCKER_FAIL_ONCE", "network disconnect")
    with pytest.raises(RuntimeError, match=f"Could not disconnect the service (kv|main) from {egress}"):
        await stack.set_network("none")
    assert len(on_egress()) == 1
    await stack.set_network("none")  # Docker refuses to disconnect the other one again: it counts as switched
    assert on_egress() == []

    await stack.stop()
    assert _nothing_left(docker, provider)


@pytest.mark.asyncio
async def test_services_start_in_dependency_order_and_after_the_service_whose_network_they_share(
    docker: FakeDocker, provider: DockerProvider, tmp_path
):
    spec = _spec(tmp_path)
    stack = await start_compose_stack(provider, spec)
    calls = docker.calls()
    project = stack.id

    def first(*argv: str) -> int:
        return next(index for index, call in enumerate(calls) if call[-len(argv) :] == list(argv))

    started = first("start", f"{project}-kv")
    passed = first(f"{project}-kv", "python3", "-c", "print('ok')")  # kv's healthcheck, which passes at once
    assert first("create", "--no-recreate", "kv") < started < first("create", "--no-recreate", "probe")
    assert passed < first("create", "--no-recreate", "main")
    assert "kv" not in spec.services["probe"].depends_on  # the caller's spec is left as it was
    await stack.stop()
    assert _nothing_left(docker, provider)


@pytest.mark.asyncio
async def test_reaching_a_service_from_this_host_and_its_state_and_log(
    docker: FakeDocker, provider: DockerProvider, tmp_path
):
    stack = await start_compose_stack(provider, _spec(tmp_path))
    project = stack.id

    assert await stack.connect("kv", 6379) == "tcp://172.30.0.3:6379"  # kv's address on the egress network
    assert await stack.connect("probe", 6379) == "tcp://172.30.0.3:6379"  # probe is reached on kv's network
    assert stack.rewrite("http://kv:6379/x") == "http://kv:6379/x"  # services reach one another by name
    states = await stack.health()
    assert {name: state.status for name, state in states.items()} == {
        "main": "running",
        "kv": "healthy",
        "probe": "running",
    }
    assert await stack.logs("kv") == f"the output of {project}-kv\nthe errors of {project}-kv\n"

    await stack.set_network("none")
    with pytest.raises(RuntimeError, match="Nothing on this host reaches the service 'kv' of a stack without egress"):
        await stack.connect("kv", 6379)  # the internal network has no address on this host

    docker.run("kill", f"{project}-kv")  # killed outside platoon
    (state,) = (await stack.health(["kv"])).values()
    assert (state.status, state.exit_code, state.up) == ("exited", 137, False)
    docker.run("rm", f"{project}-kv")  # and removed
    (state,) = (await stack.health(["kv"])).values()
    assert (state.status, state.exit_code) == ("exited", None)
    await stack.stop()
    assert _nothing_left(docker, provider)


@pytest.mark.asyncio
async def test_a_probe_docker_cannot_answer_fails_rather_than_find_the_service_down(
    docker: FakeDocker, provider: DockerProvider, tmp_path, monkeypatch
):
    stack = await start_compose_stack(provider, _spec(tmp_path))
    monkeypatch.setenv("FAKE_DOCKER_FAIL", "inspect")
    monkeypatch.setenv("FAKE_DOCKER_ERROR", "Cannot connect to the Docker daemon at unix:///var/run/docker.sock.")

    with pytest.raises(RuntimeError, match=r"Could not inspect .*-kv: Cannot connect to the Docker daemon"):
        await stack.health(["kv"])  # the session's probe makes that infrastructure's (sidecar_probe), not invalid

    monkeypatch.delenv("FAKE_DOCKER_FAIL")
    await stack.stop()
    assert _nothing_left(docker, provider)


@pytest.mark.asyncio
async def test_privileges_a_spec_asks_for_reach_compose(docker: FakeDocker, provider: DockerProvider, tmp_path):
    spec = _spec(tmp_path)
    spec.services["kv"].privileged = True
    spec.services["main"].cap_add = ("NET_ADMIN",)

    stack = await start_compose_stack(provider, spec)

    services = docker.state()["projects"][stack.id]["services"]
    assert services["kv"]["privileged"] is True and services["kv"]["cap_add"] == ["SYS_PTRACE"]
    assert services["main"]["cap_add"] == ["NET_ADMIN"] and "privileged" not in services["main"]
    assert {"stacks", "privileged"} <= provider.capabilities
    await stack.stop()
    assert _nothing_left(docker, provider)


@pytest.mark.asyncio
async def test_values_that_yaml_1_2_reads_as_numbers_reach_compose_as_strings(
    docker: FakeDocker, provider: DockerProvider, tmp_path
):
    """compose reads its file as YAML 1.2, where a plain 1e3 or 0o17 is a number, which an entrypoint must not hold
    and an environment value would become, and where U+2028 and U+2029 break a line, dropping the spaces around them."""

    spec = _spec(tmp_path)
    spec.services["kv"].command = ["sleep", "1e3"]
    values = {"OCT": "0o17", "EXP": "1e3", "DOT": ".5e3", "LR": "1e-4", "MASK": "0x1F", "DAY": "2026-10-26", "ON": "on"}
    odd = "\x7f\x85\ufeff \u2028 \u2029 , and \U0001f600"  # what YAML takes only escaped, and what it takes as it is
    spec.services["kv"].sandbox.env = {**values, "ODD": odd}

    stack = await start_compose_stack(provider, spec)

    kv = docker.state()["projects"][stack.id]["services"]["kv"]  # as the fake reads it, as compose does
    assert kv["entrypoint"] == ["sleep", "1e3"]
    assert kv["environment"] == {**values, "ODD": odd}
    text = (provider.stack_root / stack.id / PROJECT_FILE).read_text(encoding="utf-8")
    assert '"ODD": "\\u007f\\u0085\\ufeff \\u2028 \\u2029 , and \U0001f600"' in text
    await stack.stop()
    assert _nothing_left(docker, provider)


@pytest.mark.asyncio
async def test_a_named_volume_called_services_holds_none_of_the_services_own(
    docker: FakeDocker, provider: DockerProvider, tmp_path
):
    spec = _spec(tmp_path)
    spec.services["main"].volumes.append(Volume(target="/srv/shared", name="services"))

    stack = await start_compose_stack(provider, spec)

    assert list(stack.volume_path("services").iterdir()) == []  # kv's own volume is not in it
    await stack.stop()
    assert _nothing_left(docker, provider)


# --- starts that fail or are cancelled -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        (
            {"FAKE_DOCKER_FAIL": "create", "FAKE_DOCKER_ERROR": "pull access denied for python"},
            "the service kv did not start: RuntimeError: Could not create the service kv: pull access denied",
        ),
        (
            {"FAKE_DOCKER_FAIL": "start"},
            "the service kv did not start: RuntimeError: Could not start .*-kv: simulated failure",
        ),
        (
            {"FAKE_DOCKER_FAIL": "network connect"},
            "the service kv did not start: RuntimeError: Could not connect .*-kv to .*-egress",
        ),
        (
            {"FAKE_DOCKER_EXITED": "kv:3"},
            "the service kv exited with status 3 before its healthcheck passed; its log ends: the",
        ),
        (
            {"FAKE_DOCKER_UNHEALTHY": "kv"},
            "the service kv never turned healthy: its healthcheck failed 3 times: exit 1",
        ),
    ],
    ids=["image", "start", "network", "exits", "unhealthy"],
)
@pytest.mark.asyncio
async def test_a_service_that_does_not_start_leaves_nothing(
    docker: FakeDocker, provider: DockerProvider, tmp_path, monkeypatch, environment: dict[str, str], message: str
):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(StackStartError, match=message) as error:
        await start_compose_stack(provider, _spec(tmp_path))

    assert error.value.service == "kv"
    calls = docker.calls()
    project = _project(calls)
    assert ["rm", "--force", "--volumes", f"{project}-kv"] in calls  # its rollback
    assert not any(call[-1] == "main" for call in calls if "create" in call)  # main waited for it, and never started
    assert ["ps", "--all", "--quiet", f"--filter=label=platoon.stack={project}"] in calls  # the stack's teardown
    assert _nothing_left(docker, provider)


@pytest.mark.asyncio
async def test_a_setups_own_exception_passes_through_and_the_stack_is_gone(docker, provider: DockerProvider, tmp_path):
    async def setup(sandbox) -> None:
        raise OSError("the setup failed")

    with pytest.raises(OSError, match="the setup failed"):
        await start_compose_stack(provider, _spec(tmp_path, setup=setup))
    assert _nothing_left(docker, provider)


@pytest.mark.parametrize("command", ["start", "create", "network create"])
@pytest.mark.asyncio
async def test_a_start_cancelled_while_docker_carries_out_a_command_leaves_nothing(
    docker: FakeDocker, provider: DockerProvider, tmp_path, monkeypatch, command: str
):
    """The daemon carries a request out whatever becomes of its CLI: a start given up waits for its docker commands
    rather than kill them, then removes what they made (kv's container, the egress network)."""

    monkeypatch.setenv("FAKE_DOCKER_SLOW", command)  # kv's, the first: the daemon takes a second
    starting = asyncio.ensure_future(start_compose_stack(provider, _spec(tmp_path)))
    while not docker.called(command):
        await asyncio.sleep(0.01)

    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    await _settled()

    assert docker.daemon_done()
    calls = docker.calls()
    project = _project(calls)
    assert ["rm", "--force", "--volumes", f"{project}-kv"] in calls  # kv, caught in its start, rolled back
    assert not any(call[-1] == "main" for call in calls if "create" in call)
    assert _nothing_left(docker, provider)


@pytest.mark.asyncio
async def test_what_docker_made_for_a_command_whose_cli_failed_is_removed(
    docker: FakeDocker, provider: DockerProvider, tmp_path, monkeypatch
):
    monkeypatch.setenv("FAKE_DOCKER_LOST", "network create")  # the daemon made it; its CLI never heard back

    with pytest.raises(StackStartError, match=r"Could not create the network .*-egress: error during connect"):
        await start_compose_stack(provider, _spec(tmp_path))

    assert _nothing_left(docker, provider)  # the egress network too, which the stack never knew it had


@pytest.mark.asyncio
async def test_a_project_file_compose_cannot_read_is_still_taken_down(
    docker: FakeDocker, provider: DockerProvider, tmp_path
):
    stack = await start_compose_stack(provider, _spec(tmp_path))
    (provider.stack_root / stack.id / PROJECT_FILE).write_text("services: [")

    await stack.stop()

    assert _nothing_left(docker, provider)  # the teardown reads no file: its containers and networks carry its label


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("setup", "has a setup and a command: the docker provider runs a setup only in a service that idles"),
        ("image", "The docker provider cannot run image 'local'"),
        ("upstream", "A sandbox without a network cannot reach the TCP upstream"),
    ],
)
@pytest.mark.asyncio
async def test_specs_the_provider_cannot_run_are_refused_before_anything_is_made(
    docker: FakeDocker, provider: DockerProvider, tmp_path, change: str, message: str
):
    spec = _spec(tmp_path)
    if change == "setup":
        spec.services["kv"].setup = _no_setup
    elif change == "image":
        spec.services["probe"].sandbox.image = "local"
    else:
        spec.services["main"].sandbox.upstreams["inference"] = "http://127.0.0.1:8000"

    with pytest.raises(ValueError, match=message):
        await start_compose_stack(provider, spec)
    assert docker.calls() == [] and not provider.stack_root.exists() and cleanup._live == {}


async def _no_setup(sandbox) -> None:
    pass


# --- teardown and reaping ------------------------------------------------------------------------------------------


@pytest.mark.skipif(os.geteuid() == 0, reason="root deletes what other users own by itself")
@pytest.mark.asyncio
async def test_the_teardown_deletes_what_services_wrote_as_other_users(docker: FakeDocker, provider, tmp_path):
    stack = await start_compose_stack(provider, _spec(tmp_path))
    project = stack.id
    # A directory a database's user made in its volume, which this user can list but not empty (root's, in Docker).
    owned = stack.volume_path("platoon:logs-verifier") / "pgdata"
    owned.mkdir()
    (owned / "PG_VERSION").write_text("16")
    owned.chmod(0o500)
    docker.calls()

    await stack.stop()

    calls = docker.calls()
    assert calls[-1] == [
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--user=0:0",
        "--entrypoint=rm",
        f"--volume={tmp_path / 'stacks' / project}:/platoon-remains",
        "python:3.12-slim",  # main's image, which the stack ran
        "-rf",
        "--",
        "/platoon-remains/volumes",  # what is left once this process deleted what it could
    ]
    assert _nothing_left(docker, provider)


@pytest.mark.skipif(os.geteuid() == 0, reason="root deletes what other users own by itself")
@pytest.mark.asyncio
async def test_a_directory_the_teardown_cannot_empty_keeps_its_record_for_a_reap(
    docker: FakeDocker, provider: DockerProvider, tmp_path, monkeypatch
):
    stack = await start_compose_stack(provider, _spec(tmp_path))
    owned = stack.volume_path("platoon:logs-verifier") / "pgdata"
    owned.mkdir()
    (owned / "PG_VERSION").write_text("16")
    owned.chmod(0o500)
    monkeypatch.setenv("FAKE_DOCKER_FAIL", "run")  # the container that deletes as root does not run

    await stack.stop()

    assert sorted(os.listdir(provider.stack_root / stack.id)) == ["stack.json", "volumes"]  # its record stays
    (entry,) = cleanup._live.values()
    assert entry.record is not None
    monkeypatch.delenv("FAKE_DOCKER_FAIL")
    assert await DockerProvider(**entry.record.options).reap(entry.record.labels) == 1
    cleanup._live.clear()
    assert _nothing_left(docker, provider)


# A worker that starts a stack whose volume it is filling when it is killed: $1 its stack root, $2 its label.
FILLING_WORKER = """
import asyncio, sys, time
from platoon.sandbox import cleanup
from platoon.sandbox.base import SandboxSpec
from platoon.sandbox.compose import start_compose_stack
from platoon.sandbox.docker import DockerProvider
from platoon.sandbox.stacks import ServiceSpec, StackSpec, Volume

def fill(directory):
    print("filling", flush=True)
    time.sleep(3600)

cleanup.set_labels({"platoon.worker": sys.argv[2]})
main = ServiceSpec(SandboxSpec(image="python:3.12-slim"), volumes=[Volume(target="/data", name="data", fill=fill)])
asyncio.run(start_compose_stack(DockerProvider(stack_root=sys.argv[1]), StackSpec(services={"main": main})))
"""


@pytest.mark.asyncio
async def test_a_worker_killed_while_it_fills_a_stacks_volumes_leaves_what_a_reap_removes(
    docker: FakeDocker, provider: DockerProvider
):
    worker = subprocess.Popen(
        [sys.executable, "-c", FILLING_WORKER, str(provider.stack_root), "w3"], stdout=subprocess.PIPE, text=True
    )
    try:
        assert worker.stdout is not None and worker.stdout.readline().strip() == "filling"
    finally:
        worker.kill()
        worker.wait()
    assert len(list(provider.stack_root.iterdir())) == 1  # its project directory, which no container holds yet

    assert await provider.reap({WORKER: "w3"}) == 1
    assert _nothing_left(docker, provider)


@pytest.mark.asyncio
async def test_a_teardown_that_fails_stays_registered_until_a_reap_removes_the_stack(
    docker: FakeDocker, provider: DockerProvider, tmp_path, monkeypatch, caplog
):
    stack = await start_compose_stack(provider, _spec(tmp_path))
    monkeypatch.setenv("FAKE_DOCKER_FAIL", "rm,network rm")  # the daemon refuses to remove anything

    await stack.stop()

    assert "is not gone: a reap by its labels removes it" in caplog.text
    (entry,) = cleanup._live.values()
    assert entry.record is not None and entry.record.labels == {"platoon.rollout_id": "r1", "platoon.stack": stack.id}
    assert len(docker.state()["containers"]) == 3 and f"{stack.id}-internal" in docker.state()["networks"]

    monkeypatch.delenv("FAKE_DOCKER_FAIL")
    rebuilt = DockerProvider(**entry.record.options)  # as the supervisor rebuilds it from the record
    assert await rebuilt.reap(entry.record.labels) == 1
    cleanup._live.clear()
    assert _nothing_left(docker, provider)


@pytest.mark.asyncio
async def test_reaping_removes_the_stacks_of_a_process_that_is_gone(
    docker: FakeDocker, provider: DockerProvider, tmp_path
):
    cleanup.set_labels({WORKER: "w1"})
    ours = await start_compose_stack(provider, _spec(tmp_path))
    theirs = await start_compose_stack(provider, _spec(tmp_path))  # held by another live process, below
    cleanup.set_labels({})
    record = provider.stack_root / theirs.id / STACK_RECORD
    held = json.loads(record.read_text())
    held.update(owner_pid=os.getppid(), owner_start_time=process_start_time(os.getppid()))
    record.write_text(json.dumps(held))
    docker.calls()

    assert await DockerProvider(stack_root=provider.stack_root).reap({WORKER: "w1"}) == 2

    (removed,) = [call[3:] for call in docker.calls() if call[:3] == ["rm", "--force", "--volumes"]]
    # Their containers carry the label too: Docker cannot tell whose they are.
    assert sorted(removed) == sorted(
        f"{stack.id}-{name}" for stack in (ours, theirs) for name in ("main", "kv", "probe")
    )
    assert sorted(docker.state()["containers"]) == [] and docker.state()["networks"] == {}
    assert not (provider.stack_root / ours.id).exists()
    assert (provider.stack_root / theirs.id).exists()  # its directory is its owner's to delete
    shutil.rmtree(provider.stack_root / theirs.id)
    cleanup._live.clear()
    assert await DockerProvider(stack_root=provider.stack_root).reap({WORKER: "w1"}) == 0
    with pytest.raises(ValueError, match="needs labels"):
        await provider.reap({})


@pytest.mark.asyncio
async def test_reaping_finds_a_stack_whose_project_directory_is_gone(docker: FakeDocker, provider, tmp_path):
    cleanup.set_labels({WORKER: "w2"})
    stack = await start_compose_stack(provider, _spec(tmp_path))
    cleanup.set_labels({})
    shutil.rmtree(provider.stack_root / stack.id)
    docker.calls()

    assert await provider.reap({WORKER: "w2"}) == 1  # by its containers' labels

    cleanup._live.clear()
    assert _nothing_left(docker, provider)


# --- refusals ------------------------------------------------------------------------------------------------------


def test_stacks_need_the_docker_clis_compose_plugin(docker: FakeDocker, monkeypatch, tmp_path):
    assert stack_refusal("docker", {}) is None
    assert stack_refusal(DockerProvider(stack_root=tmp_path / "stacks")) is None
    assert stack_refusal("docker", {"binary": "no-such-docker"}) is None  # building the provider says why

    monkeypatch.setenv("FAKE_DOCKER_NO_COMPOSE", "1")
    compose_refusal.cache_clear()
    refusal = stack_refusal("docker", {})
    assert (
        refusal is not None and "has no compose plugin: `" in refusal and "'compose' is not a docker command" in refusal
    )
    assert stack_refusal(DockerProvider(stack_root=tmp_path / "stacks")) == refusal
    assert sum(call == ["compose", "version"] for call in docker.calls()) == 2  # asked once per process (and clear)


def test_a_daemon_whose_internal_networks_let_dns_queries_out_is_warned_of(docker: FakeDocker, monkeypatch, caplog):
    monkeypatch.setenv("FAKE_DOCKER_SERVER_VERSION", "24.0.7")
    assert stack_refusal("docker", {}) is None  # it still runs stacks
    assert "The Docker daemon is 24.0.7: before 25.0.5, an internal network passes its containers' DNS" in caplog.text

    caplog.clear()
    compose_refusal.cache_clear()
    monkeypatch.setenv("FAKE_DOCKER_SERVER_VERSION", "25.0.5")
    assert stack_refusal("docker", {}) is None and "DNS" not in caplog.text


@pytest.mark.asyncio
async def test_a_stack_root_that_is_no_directory_of_this_users_own_is_refused(docker: FakeDocker, tmp_path, caplog):
    elsewhere = tmp_path / "elsewhere"  # another user's, say, who made the link on a node's shared scratch
    (elsewhere / "platoon-stack-0").mkdir(parents=True)
    (elsewhere / "platoon-stack-0" / STACK_RECORD).write_text(json.dumps({"labels": {WORKER: "w4"}}))
    root = tmp_path / "stacks"
    root.symlink_to(elsewhere)
    provider = DockerProvider(stack_root=root)

    with pytest.raises(RuntimeError, match="stacks is not a directory of this user's, so no stack's compose project"):
        await start_compose_stack(provider, _spec(tmp_path))
    assert docker.calls() == [] and cleanup._live == {} and os.listdir(elsewhere) == ["platoon-stack-0"]
    assert await provider.reap({WORKER: "w4"}) == 0  # nothing there is read
    assert "Not reaping the docker compose projects there" in caplog.text and (elsewhere / "platoon-stack-0").exists()

    root.unlink()
    root.mkdir(mode=0o777)
    root.chmod(0o777)  # this user's own, which anyone could enter
    stack = await start_compose_stack(provider, _spec(tmp_path))
    assert root.stat().st_mode & 0o777 == 0o700
    await stack.stop()
    assert _nothing_left(docker, provider)


@pytest.mark.asyncio
async def test_a_stack_root_whose_path_holds_a_colon_is_refused(docker: FakeDocker, tmp_path, caplog):
    """Docker reads a bind's source up to its first ':', and compose passes the file's binds on so: below such a root,
    every service with a volume, a directory of the project, would fail to create, and the teardown's rm too."""

    root = tmp_path / "a:b" / "stacks"
    provider = DockerProvider(stack_root=root)

    with pytest.raises(RuntimeError, match="stacks has a ':', which Docker reads as the end of a bind's source, so no"):
        await start_compose_stack(provider, _spec(tmp_path))
    assert docker.calls() == [] and cleanup._live == {} and not (tmp_path / "a:b").exists()  # nothing made
    root.mkdir(parents=True)
    assert await provider.reap({WORKER: "w4"}) == 0
    assert "Not reaping the docker compose projects there" in caplog.text


@pytest.mark.asyncio
async def test_the_teardown_deletes_nothing_through_a_link(docker: FakeDocker, provider: DockerProvider, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "volumes").mkdir(parents=True)
    link = provider.stack_root / "platoon-stack-0"
    link.parent.mkdir()
    link.symlink_to(elsewhere)

    assert await remove_directory(provider, link, image="python:3.12-slim") is False
    assert (elsewhere / "volumes").exists() and docker.calls() == []  # no rm as root was pointed there


# --- each fixture as a Harbor session ------------------------------------------------------------------------------


# fixture -> (where it is, the fake CLI's settings, how its session's start ends)
SESSIONS = {
    "app-redis": ("harbor-stacks-docker", {}, None),
    "app-redis-crash": ("harbor-stacks-docker", {}, None),
    "app-redis-unhealthy": ("harbor-stacks-docker", {"FAKE_DOCKER_UNHEALTHY": "cache"}, "sidecar_start"),
    "app-privileges": ("harbor-stacks-docker", {}, None),
    "app-kv": ("harbor-stacks", {}, None),
    "app-kv-crash": ("harbor-stacks", {}, None),
    "app-kv-unhealthy": ("harbor-stacks", {"FAKE_DOCKER_UNHEALTHY": "kv"}, "sidecar_start"),
    "app-kv-seeded": ("harbor-stacks", {"FAKE_DOCKER_EXITED": "seed:0"}, None),
    "sidecar-artifacts": ("harbor-stacks", {}, None),
    "hello-mcp": ("harbor-compose/upstream/examples", {}, None),
}


def _service(name: str, tmp_path: Path, provider: DockerProvider, monkeypatch) -> HarborEnvironmentService:
    """The Harbor service of one fixture on ``provider``: a dataset directory of a copy of it, or a fetched copy (for
    hello-mcp, whose sidecar builds its image, and app-privileges, whose fetch keeps its privileges) whose image map
    holds the images it builds, as platoon prepare records them."""

    where, _, _ = SESSIONS[name]
    dataset = tmp_path / "dataset"
    source = FIXTURES / where / (name if where.endswith("examples") else f"tasks/{name}")
    shutil.copytree(source, dataset / name)
    _readable_by_all(dataset)
    if name not in ("hello-mcp", "app-privileges"):
        return HarborEnvironmentService(dataset, provider=provider)
    use_site_cache(monkeypatch, tmp_path / "cache")
    settings: dict[str, Any] = {
        "dataset": str(dataset),
        "lock": str(tmp_path / "lock.json"),
        "refuse": {"sidecars": False, "privileges": False},
    }
    lock = fetch(fetch_config(settings["dataset"], lock=settings["lock"], refuse=settings["refuse"]))
    service = HarborEnvironmentService(provider=provider, **settings)
    entries = {}
    for image in plan_images(lock).images:  # main's and the sidecar's, as the docker builder records them
        tag = f"platoon-harbor:{image.key.removeprefix('sha256:')[:32]}"
        if image.role == "service":  # mcp-server/Dockerfile's WORKDIR and CMD, which docker image inspect gives
            entries[image.key] = ImageEntry(image=tag, builder="docker", workdir="/app", cmd=("python", "server.py"))
        else:
            entries[image.key] = ImageEntry(image=tag, builder="docker", workdir="/")
    assert service.task_images.path is not None
    update_image_map(service.task_images.path, entries)
    return service


def _readable_by_all(root: Path) -> None:
    """Give ``root``'s files the modes a checkout has under a umask of 022, which every user reads, whatever this
    process's umask: a sidecar's read-only bind of task files that not every user may read binds a copy instead."""

    for path in [root, *root.rglob("*")]:
        mode = path.stat().st_mode
        path.chmod(mode | (0o555 if path.is_dir() or mode & 0o100 else 0o444))


@pytest.mark.parametrize("name", sorted(SESSIONS))
@pytest.mark.asyncio
async def test_each_fixture_becomes_its_compose_project(
    name: str, docker: FakeDocker, provider: DockerProvider, tmp_path, monkeypatch
):
    _, environment, failure = SESSIONS[name]
    for variable, value in environment.items():
        monkeypatch.setenv(variable, value)
    service = _service(name, tmp_path, provider, monkeypatch)
    task = await service.get_task(name)
    outcome = "opened and closed"
    try:
        session = await service.open(task, rollout_id=f"rollout-{name}")
    except EnvSetupError as exc:
        outcome = f"{exc.failure_kind}: {exc}"
        assert exc.failure_kind == failure
    else:
        assert failure is None
        stack = session.stack
        assert isinstance(stack, DrivenStack) and set(stack.run_modes.values()) == {"subids"}
        await session.close()
    calls = docker.calls()
    project = _project(calls)
    replacements = [(str(tmp_path), "<tmp>"), (project, "<project>")]
    actual = {
        "outcome": _replaced(outcome, replacements),
        "project": _normalized(docker.state()["projects"][project], replacements),
        "commands": _by_service(_normalized(calls, replacements), "<project>"),
    }
    expected = PROJECTS / f"{name}.yaml"
    if os.environ.get("PLATOON_RECORD_COMPOSE_PROJECTS"):
        expected.parent.mkdir(exist_ok=True)
        expected.write_text(yaml.safe_dump(actual, sort_keys=False, width=1 << 30, default_flow_style=None))
    assert actual == yaml.safe_load(expected.read_text())
    assert _nothing_left(docker, provider)


def test_every_recorded_project_has_its_fixture():
    assert sorted(path.stem for path in PROJECTS.glob("*.yaml")) == sorted(SESSIONS)
