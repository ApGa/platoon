"""Harbor stacks on the singularity provider, as hpc-sandbox pods (platoon.sandbox.hpc_stacks), against the fake
``hpc_sandbox`` package of ``tests/train/fake_hpc_sandbox`` (0.5's API): what a stack task's session asks of the pod,
registry sidecars run as the image map pinned them, and what a run refuses before any rollout: stacks that an
hpc-sandbox without 0.5's pods cannot run, and images that ``platoon prepare`` has not built or pinned.  Real pods run
in ``tests/integration/test_harbor_stacks_hpc.py``."""

from __future__ import annotations

import dataclasses
import importlib.util
import shutil
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest
from site_cache import use_site_cache

from platoon.envs.harbor import HarborEnvironmentService, HarborSession
from platoon.envs.harbor.fetch import fetch, fetch_config
from platoon.envs.harbor.images import ImageEntry, default_image_map, pulled_key, update_image_map
from platoon.envs.service import EnvNotPrepared, EnvSetupError, SessionNeeds
from platoon.sandbox import hpc_stacks
from platoon.sandbox.stacks import DrivenStack

STACKS = Path(__file__).parent / "fixtures" / "harbor-stacks" / "tasks"
FAKE_HPC_SANDBOX = Path(__file__).resolve().parents[1] / "train" / "fake_hpc_sandbox" / "hpc_sandbox" / "__init__.py"
GATEWAY = "unix:///srv/gateway/gw.sock"
REDIS = (
    "services:\n"
    "  main:\n"
    "    environment:\n"
    "      KV_URL: redis://kv:6379\n"
    "    depends_on: [kv]\n"
    "  kv:\n"
    "    image: redis:7-alpine\n"
    "    expose: ['6379']\n"
)
LOCK_STACKS = {"sidecars": False}  # a fetch refuses stacks unless told to lock them


@pytest.fixture
def hpc_sandbox(monkeypatch) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("hpc_sandbox", FAKE_HPC_SANDBOX)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "hpc_sandbox", module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(hpc_stacks, "nesting_refusal", lambda: None)
    return module


def _service(tmp_path: Path, **settings: Any) -> HarborEnvironmentService:
    options = {"scratch_root": str(tmp_path / "scratch"), "allow_network_none": True}
    return HarborEnvironmentService(provider="singularity", provider_options=options, **settings)


async def _open(service: HarborEnvironmentService, task_id: str) -> HarborSession:
    task = await service.get_task(task_id)
    return await service.open(task, rollout_id=f"rollout-{task_id}", needs=SessionNeeds(upstreams={"gateway": GATEWAY}))


def _pod(service: HarborEnvironmentService) -> Any:
    [pod] = cast(Any, service.provider).inner.started_pods
    return pod


def _redis_dataset(tmp_path: Path) -> Path:
    """app-kv with a registry sidecar that its compose file gives no command, fetched into a lock."""

    dataset = tmp_path / "dataset"
    task = dataset / "app-redis"
    shutil.copytree(STACKS / "app-kv", task)
    (task / "environment" / "docker-compose.yaml").write_text(REDIS)
    (task / "task.toml").write_text((task / "task.toml").read_text().replace("platoon/app-kv", "platoon/app-redis"))
    fetch(fetch_config(str(dataset), lock=tmp_path / "lock.json", refuse=LOCK_STACKS))
    return dataset


def _pin(cache: Path, *, user: str | None = None) -> None:
    entry = ImageEntry(
        image="docker://redis@sha256:" + "a" * 64,
        builder="pulled",
        env={"REDIS_VERSION": "7.4"},
        workdir="/data",
        user=user,
        entrypoint=("docker-entrypoint.sh",),
        cmd=("redis-server",),
    )
    update_image_map(default_image_map(cache, "singularity"), {pulled_key("redis:7-alpine"): entry})


@pytest.mark.asyncio
async def test_a_stack_opens_as_one_pod_whose_main_reaches_the_gateway_through_its_relay(hpc_sandbox, tmp_path: Path):
    service = _service(tmp_path, dataset_dir=STACKS)
    session = await _open(service, "app-kv")
    pod = _pod(service)
    try:
        assert isinstance(session.stack, DrivenStack) and session.stack.main is session.sandbox
        assert session.routes == {"gateway": "http://127.0.0.1:18080"}
        assert session.host_loopback is False  # main is in the pod's network, not this host's
        assert pod.spec.upstreams == {"gateway": GATEWAY} and pod.spec.users == "subids"
        started = {spec.name: spec for spec in pod.started}
        assert list(started) == ["kv", "main"]  # main waits for kv to be healthy
        assert {name: spec.run_as for name, spec in started.items()} == {"kv": "subids", "main": "root"}
        kv = started["kv"]
        assert kv.command == ["python3", "/srv/kv/kv.py"] and kv.sandbox.image == "docker://python:3.12-slim"
        targets = {mount.target: mount.read_only for mount in kv.sandbox.mounts}
        assert targets == {"/srv/kv": True, "/var/lib/kv": False, "/logs/verifier": False}
        assert started["main"].sandbox.env["KV_URL"] == "http://kv:6379"  # names resolve in a pod: no rewriting
    finally:
        await session.close()
    assert pod.stops == 1 and not pod.path.exists()


@pytest.mark.asyncio
async def test_a_registry_sidecar_runs_as_the_image_map_pinned_it(hpc_sandbox, tmp_path: Path, monkeypatch):
    cache = use_site_cache(monkeypatch, tmp_path / "cache")
    dataset = _redis_dataset(tmp_path)
    service = _service(tmp_path, dataset=str(dataset), lock=str(tmp_path / "lock.json"), refuse=LOCK_STACKS)

    with pytest.raises(EnvNotPrepared, match="have sidecars that run registry images the image map .* has not pinned"):
        await service.check_tasks("train", ["app-redis"])
    _pin(cache)
    service = _service(tmp_path, dataset=str(dataset), lock=str(tmp_path / "lock.json"), refuse=LOCK_STACKS)
    await service.check_tasks("train", ["app-redis"])
    session = await _open(service, "app-redis")
    try:
        kv = next(spec for spec in _pod(service).started if spec.name == "kv")
        assert kv.sandbox.image == "docker://redis@sha256:" + "a" * 64
        assert kv.command == ["docker-entrypoint.sh", "redis-server"]  # the image's ENTRYPOINT and CMD
        assert kv.sandbox.workdir == "/data" and kv.sandbox.env["REDIS_VERSION"] == "7.4"
        assert kv.run_as == "subids"  # its entrypoint switches to its own user, as Docker's root can
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_sidecar_whose_image_runs_as_another_user_runs_as_your_own_uid(
    hpc_sandbox, tmp_path: Path, monkeypatch
):
    cache = use_site_cache(monkeypatch, tmp_path / "cache")
    dataset = _redis_dataset(tmp_path)
    _pin(cache, user="redis")
    service = _service(tmp_path, dataset=str(dataset), lock=str(tmp_path / "lock.json"), refuse=LOCK_STACKS)
    session = await _open(service, "app-redis")
    try:
        assert session.stack is not None and session.stack.run_modes == {"kv": "caller", "main": "root"}
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_a_sidecar_whose_build_context_the_image_map_lacks_is_not_prepared(
    hpc_sandbox, tmp_path: Path, monkeypatch
):
    use_site_cache(monkeypatch, tmp_path / "cache")
    dataset = tmp_path / "dataset"
    task = dataset / "built"
    shutil.copytree(STACKS / "app-kv", task)
    compose = task / "environment" / "docker-compose.yaml"
    compose.write_text(compose.read_text().replace("    image: python:3.12-slim\n", "    build: ./kv\n"))
    (task / "environment" / "kv" / "Dockerfile").write_text("FROM python:3.12-slim\nCOPY kv.py /srv/kv/kv.py\n")
    fetch(fetch_config(str(dataset), lock=tmp_path / "lock.json", refuse=LOCK_STACKS))
    service = _service(tmp_path, dataset=str(dataset), lock=str(tmp_path / "lock.json"), refuse=LOCK_STACKS)

    with pytest.raises(EnvNotPrepared, match="build their images from Dockerfiles"):
        await service.check_tasks("train", ["built"])


def _hpc_sandbox_0_4(fake: types.ModuleType) -> types.ModuleType:
    """The fake as hpc-sandbox 0.4 has it: pods whose services all run as root, without host aliases or exposed
    ports."""

    module = types.ModuleType("hpc_sandbox")
    module.__dict__.update({name: value for name, value in vars(fake).items() if not name.startswith("__")})
    module.__dict__.update(
        __version__="0.4.0",
        PodSpec=dataclasses.make_dataclass("PodSpec", ["network", "upstreams", "upstream_ports", "labels"]),
        ServiceSpec=dataclasses.make_dataclass("ServiceSpec", ["name", "sandbox", "command", "healthcheck"]),
    )
    return module


@pytest.mark.asyncio
async def test_with_hpc_sandbox_0_4_stacks_are_refused_before_any_rollout_naming_0_5(
    hpc_sandbox, tmp_path: Path, monkeypatch, caplog
):
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _hpc_sandbox_0_4(hpc_sandbox))
    cache = use_site_cache(monkeypatch, tmp_path / "cache")
    dataset = _redis_dataset(tmp_path)
    _pin(cache)

    directory = _service(tmp_path, dataset_dir=dataset)
    with pytest.raises(ValueError, match="No runnable Harbor tasks .* install hpc-sandbox 0.5"):
        await directory.list_tasks("train")  # a dataset directory skips its stacks, with a warning
    assert "Skipping Harbor task" in caplog.text
    locked = _service(tmp_path, dataset=str(dataset), lock=str(tmp_path / "lock.json"), refuse=LOCK_STACKS)
    with pytest.raises(EnvSetupError, match="are multi-container stacks .* install hpc-sandbox 0.5") as error:
        await locked.check_tasks("train", ["app-redis"])
    assert error.value.failure_kind == "env_config"


@pytest.mark.asyncio
@pytest.mark.parametrize("lost", ["main", "pod"])
async def test_a_stack_whose_main_or_pod_is_gone_by_grading_says_which(hpc_sandbox, tmp_path: Path, monkeypatch, lost):
    """Grading cannot kill what is left in a main that is gone: main's container exited (``main_exited``, as Docker
    would have ended it), or the whole pod is gone, its holder killed, say (``sidecar_probe``); never scored."""

    service = _service(tmp_path, dataset_dir=STACKS)
    session = await _open(service, "app-kv")
    pod = _pod(service)
    try:
        if lost == "main":
            pod.services["main"].exit_status = 137
        else:

            async def gone(self, name: str) -> str:
                raise hpc_sandbox.SandboxLostError(f"pod {self.id} lost its holder")

            monkeypatch.setattr(hpc_sandbox.SessionPod, "health", gone)
            pod.services["main"].stopped = True
        outcome = await session.grade()
    finally:
        await session.close()
    kind = "main_exited" if lost == "main" else "sidecar_probe"
    assert (outcome.status, outcome.failure_kind, outcome.trainable) == ("error", kind, False), outcome.reason
