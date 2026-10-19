"""Harbor tasks run the same way on every sandbox provider, or are refused before any rollout starts: a provider that
would refuse every sandbox without a network (``singularity`` without ``allow_network_none``) refuses the
``no-network`` tasks when the tasks are listed, which stops ``platoon train`` and ``eval`` at startup."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import sys
import types
from pathlib import Path

import pytest
import yaml

from platoon.envs.harbor import HarborEnvironmentService
from platoon.envs.service import EnvSetupError
from platoon.eval.run import arun_eval
from platoon.sandbox import get_provider, network_none_refusal, rootfs_layout_refusal
from platoon.sandbox.enroot import enroot_options
from platoon.train.auto import _service_task_ids
from platoon.utils.config import ConfigError

OFFLINE = '[environment]\nnetwork_mode = "no-network"\n'
OFFLINE_VERIFIER = '[verifier.environment]\nnetwork_mode = "no-network"\n'


async def rollout(task, config):  # the entry's rollout; never run here
    raise AssertionError("not called")


@pytest.mark.parametrize(
    ("provider", "options", "refused"),
    [
        ("singularity", {}, True),
        ("singularity", {"allow_network_none": False}, True),
        ("singularity", {"allow_network_none": True}, False),
        ("enroot", {}, False),  # enroot makes its own network namespace
        ("enroot", {"allow_network_none": True}, False),
        # A block shared with a Singularity site: hpc-sandbox's enroot provider then refuses network none too.
        ("enroot", {"allow_network_none": False}, True),
        ("docker", {}, False),
        ("process", {}, False),  # runs on the host network, as documented
        ("no.such.module:Provider", {}, False),  # building it reports the error
    ],
)
def test_providers_say_from_their_options_whether_they_refuse_sandboxes_without_a_network(provider, options, refused):
    refusal = network_none_refusal(provider, options)
    assert (refusal is not None) is refused
    if refused:
        assert ("allow_network_none: true" if provider == "singularity" else "allow_network_none: false") in refusal


def _fake_hpc_sandbox() -> types.ModuleType:
    module = types.ModuleType("hpc_sandbox")

    class SingularityProvider:
        def __init__(self, **options):
            self.capabilities = frozenset({"snapshot", "isolation"} | ({"network_none"} if options else set()))

    module.SingularityProvider = SingularityProvider
    module.Mount = dataclasses.make_dataclass("Mount", ["source", "target", "read_only"])
    return module


@pytest.mark.parametrize(("options", "refused"), [({}, True), ({"allow_network_none": True}, False)])
def test_a_built_provider_says_it_too(monkeypatch, options, refused):
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _fake_hpc_sandbox())
    provider = get_provider("singularity", **options)
    assert (network_none_refusal(provider) is not None) is refused
    assert ("network_none" in provider.capabilities) is not refused


def _hpc_sandbox_with(*, layouts: bool) -> types.ModuleType:
    """An ``hpc_sandbox`` whose Singularity provider takes ``rootfs_layout`` (``layouts``), or, as 47b73aa's, only
    named options of its own, so that building it with ``rootfs_layout`` raises ``TypeError``."""

    module = types.ModuleType("hpc_sandbox")
    if layouts:

        class SingularityProvider:
            def __init__(self, *, rootfs_layout=None, **options):
                self.capabilities = frozenset({"snapshot", "isolation"})

    else:

        class SingularityProvider:  # type: ignore[no-redef]
            def __init__(self, *, allow_network_none=False, gpu=None, cgroup_limits=False, extra_start_args=()):
                self.capabilities = frozenset({"snapshot", "isolation"})

    module.SingularityProvider = SingularityProvider
    module.Mount = dataclasses.make_dataclass("Mount", ["source", "target", "read_only"])
    return module


@pytest.fixture
def fuse_tools(tmp_path, monkeypatch):
    """PATH with the fuse-overlay layout's tools on it (``fuse_tools(False)``: without them)."""

    def put(present: bool) -> None:
        directory = tmp_path / ("tools" if present else "no-tools")
        directory.mkdir(exist_ok=True)
        if present:
            for tool in ("fuse-overlayfs", "fusermount3"):
                (directory / tool).write_text("#!/bin/sh\n")
                (directory / tool).chmod(0o755)
        monkeypatch.setenv("PATH", str(directory))

    return put


@pytest.mark.parametrize(
    ("layouts", "tools", "options", "job_default", "refusal"),
    [
        (True, True, {}, None, None),  # the copy layout, as always
        (True, True, {"rootfs_layout": "copy"}, None, None),
        (True, True, {"rootfs_layout": None}, None, None),
        (True, True, {"rootfs_layout": "overlay"}, None, "must be one of copy, fuse-overlay, not 'overlay'"),
        (True, True, {"rootfs_layout": "fuse-overlay"}, None, None),
        (True, True, {}, "fuse-overlay", None),
        (True, False, {"rootfs_layout": "fuse-overlay"}, None, "needs fuse-overlayfs and fusermount3 on PATH"),
        (True, False, {}, "fuse-overlay", "needs fuse-overlayfs and fusermount3 on PATH"),
        (True, True, {"rootfs_layout": "fuse-overlay", "gpu": "nv"}, None, "need copy"),
        (True, True, {"gpu": "nv"}, "fuse-overlay", None),  # hpc-sandbox gives such a provider copies
        (True, True, {"rootfs_layout": "", "gpu": "nv"}, "fuse-overlay", "need copy"),  # given, though empty
        (True, True, {"rootfs_layout": "fuse-overlay", "rootfs_templates": False}, None, "leave rootfs_templates on"),
        (True, False, {"rootfs_templates": False}, "fuse-overlay", None),  # copies, which need no fuse tools
        (False, True, {"rootfs_layout": "fuse-overlay"}, None, "has no rootfs layouts"),
        # Without layouts hpc-sandbox takes no rootfs_layout at all: copy, or null, would fail every provider build.
        (False, True, {"rootfs_layout": "copy"}, None, "has no rootfs layouts"),
        (False, True, {"rootfs_layout": None}, None, "has no rootfs layouts"),
        (False, True, {}, "fuse-overlay", None),  # it ignores the job's default
        (False, True, {}, None, None),
    ],
)
def test_the_singularity_provider_says_from_its_options_whether_a_rootfs_layout_can_run(
    monkeypatch, fuse_tools, layouts, tools, options, job_default, refusal
):
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _hpc_sandbox_with(layouts=layouts))
    if job_default:
        monkeypatch.setenv("HPC_SANDBOX_ROOTFS_LAYOUT", job_default)
    else:
        monkeypatch.delenv("HPC_SANDBOX_ROOTFS_LAYOUT", raising=False)
    fuse_tools(tools)
    said = rootfs_layout_refusal("singularity", options)
    assert (said is None) is (refusal is None), said
    if refusal is not None:
        assert refusal in said


def test_enroot_takes_no_rootfs_layout_and_ignores_one(monkeypatch, fuse_tools):
    fuse_tools(False)
    assert rootfs_layout_refusal("enroot", {"rootfs_layout": "fuse-overlay"}) is None
    assert "rootfs_layout" not in enroot_options({"rootfs_layout": "fuse-overlay", "gpu": "nv"})


@pytest.mark.parametrize("tools", [True, False], ids=["fuse-tools", "no-fuse-tools"])
@pytest.mark.parametrize("job_default", [None, "copy", "fuse-overlay", "overlay"])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"rootfs_layout": "copy"},
        {"rootfs_layout": "fuse-overlay"},
        {"rootfs_layout": "fuse-overlay", "gpu": "nv"},
        {"rootfs_layout": "fuse-overlay", "cgroup_limits": True},
        {"rootfs_layout": "fuse-overlay", "extra_start_args": ["--fakeroot"]},
        {"rootfs_layout": "fuse-overlay", "rootfs_templates": False},
        {"rootfs_layout": "fuse-overlay", "rootfs_templates": True},
        {"rootfs_layout": "copy", "rootfs_templates": True},
        {"gpu": "nv"},
        {"rootfs_templates": False},
        {"rootfs_layout": "overlay"},
    ],
    ids=str,
)
def test_the_rootfs_layout_refusal_matches_the_published_singularity_provider(
    monkeypatch, fuse_tools, tmp_path, options, job_default, tools
):
    """Against the installed hpc-sandbox: a run refuses a layout when it lists its tasks exactly when every rollout's
    provider would fail to build with those options, or would mount the fuse-overlay layout without its tools."""

    hpc_sandbox = pytest.importorskip("hpc_sandbox")
    if "rootfs_layout" not in inspect.signature(hpc_sandbox.SingularityProvider).parameters:
        pytest.skip("this hpc-sandbox has no rootfs layouts (they came with 0.3)")
    if job_default:
        monkeypatch.setenv("HPC_SANDBOX_ROOTFS_LAYOUT", job_default)
    else:
        monkeypatch.delenv("HPC_SANDBOX_ROOTFS_LAYOUT", raising=False)
    fuse_tools(tools)
    try:
        built = hpc_sandbox.SingularityProvider(scratch_root=str(tmp_path), image_cache=str(tmp_path), **options)
        layout = getattr(built, "rootfs_layout", "copy")
    except (TypeError, ValueError):
        layout = None

    refusal = rootfs_layout_refusal("singularity", options)

    assert (refusal is None) is (layout == "copy" or (layout == "fuse-overlay" and tools)), (refusal, layout)


@pytest.mark.asyncio
async def test_a_rootfs_layout_that_cannot_run_is_refused_when_the_tasks_are_listed(make_task, monkeypatch, fuse_tools):
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _hpc_sandbox_with(layouts=True))
    monkeypatch.delenv("HPC_SANDBOX_ROOTFS_LAYOUT", raising=False)
    fuse_tools(False)
    service = HarborEnvironmentService(
        task_dirs=[make_task("one")],
        provider="singularity",
        provider_options={"rootfs_layout": "fuse-overlay", "allow_network_none": True},
        default_image="x.sif",
    )
    listed = await service.list_tasks("train")

    with pytest.raises(
        EnvSetupError, match="the singularity provider would start no sandbox: the fuse-overlay"
    ) as refused:
        await service.check_tasks("train", listed)

    assert refused.value.failure_kind == "env_config"
    assert "provider" not in vars(service)  # nothing was started, nor the provider built
    fuse_tools(True)
    await service.check_tasks("train", listed)


@pytest.mark.asyncio
async def test_any_rootfs_layout_is_refused_when_listed_on_an_hpc_sandbox_without_layouts(make_task, monkeypatch):
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _hpc_sandbox_with(layouts=False))
    monkeypatch.delenv("HPC_SANDBOX_ROOTFS_LAYOUT", raising=False)
    options = {"rootfs_layout": "copy", "allow_network_none": True}
    with pytest.raises(TypeError, match="rootfs_layout"):  # what every rollout's sandbox start would meet
        get_provider("singularity", **options)
    service = HarborEnvironmentService(
        task_dirs=[make_task("one")], provider="singularity", provider_options=options, default_image="x.sif"
    )
    listed = await service.list_tasks("train")

    with pytest.raises(EnvSetupError, match="would start no sandbox: the installed hpc-sandbox has no rootfs layouts"):
        await service.check_tasks("train", listed)

    del options["rootfs_layout"]
    service = HarborEnvironmentService(
        task_dirs=[make_task("two")], provider="singularity", provider_options=options, default_image="x.sif"
    )
    await service.check_tasks("train", await service.list_tasks("train"))
    assert {"snapshot", "isolation"} <= service.provider.capabilities  # and without it the provider builds


@pytest.mark.asyncio
@pytest.mark.parametrize("task_toml", [OFFLINE, OFFLINE_VERIFIER], ids=["agents", "separate-verifier"])
async def test_no_network_tasks_are_refused_when_listed_on_a_provider_that_would_refuse_them(make_task, task_toml):
    offline, online = make_task("offline", task_toml=task_toml), make_task("online")
    service = HarborEnvironmentService(task_dirs=[offline, online], provider="singularity", default_image="x.sif")
    listed = await service.list_tasks("train")

    with pytest.raises(
        EnvSetupError, match=r"1 of the 2 tasks of split 'train' run without a network \(offline\)"
    ) as refused:
        await service.check_tasks("train", listed)

    assert refused.value.failure_kind == "env_config" and "allow_network_none: true" in str(refused.value)
    assert "provider" not in vars(service)  # nothing was started, nor the provider built
    await service.check_tasks("train", ["online"])  # the run left the offline task out: nothing refused

    allowed = HarborEnvironmentService(
        task_dirs=[offline, online],
        provider="singularity",
        provider_options={"allow_network_none": True},
        default_image="x.sif",
    )
    assert sorted(await allowed.list_tasks("train")) == ["offline", "online"]
    await allowed.check_tasks("train", ["offline", "online"])
    on_process = HarborEnvironmentService(task_dirs=[offline, online], provider="process", default_image="local")
    assert sorted(await on_process.list_tasks("train")) == ["offline", "online"]
    await on_process.check_tasks("train", ["offline", "online"])


def test_exclude_tasks_leaves_out_the_tasks_a_provider_would_refuse(make_task):
    """A run lists its tasks, leaves out those its entry's exclude_tasks names, and only then has the service refuse
    what it would not run: leaving the no-network tasks out is one of the two fixes the refusal offers."""

    offline, online = make_task("offline", task_toml=OFFLINE), make_task("online")
    service_kwargs = {"task_dirs": [str(offline), str(online)], "provider": "singularity", "default_image": "x.sif"}

    with pytest.raises(EnvSetupError, match=r"1 of the 2 tasks of split 'train' run without a network"):
        _service_task_ids("harbor", service_kwargs, "train")
    listing = _service_task_ids("harbor", service_kwargs, "train", ["offline"])

    assert (listing.task_ids, listing.listed, listing.excluded) == (["online"], 2, ["offline"])


@pytest.mark.asyncio
async def test_a_no_network_task_opened_directly_is_refused_before_its_sandbox_starts(make_task):
    service = HarborEnvironmentService(
        task_dirs=[make_task("offline", task_toml=OFFLINE)], provider="singularity", default_image="x.sif"
    )

    with pytest.raises(
        EnvSetupError, match="'offline' runs without a network, and the singularity provider"
    ) as refused:
        await service.open(await service.get_task("offline"), rollout_id="r")

    assert refused.value.failure_kind == "env_config"  # a configuration error, not a sandbox_start failure
    assert "provider" not in vars(service)


def test_evaluation_stops_at_startup(make_task, tmp_path: Path):
    environment = {
        "name": "science",
        "service": "harbor",
        "service_kwargs": {"task_dirs": [str(make_task("offline", task_toml=OFFLINE))], "provider": "singularity"},
        "rollout": f"{__name__}:rollout",
    }
    config = tmp_path / "eval.yaml"
    config.write_text(
        yaml.safe_dump(
            {"inference": {"model_name": "m", "output_dir": str(tmp_path / "results")}, "environments": [environment]}
        )
    )

    with pytest.raises(ConfigError, match="run without a network.*allow_network_none: true"):
        asyncio.run(arun_eval(["--config", str(config)]))


def test_the_built_in_providers_run_sandboxes_on_this_hosts_network(monkeypatch):
    from platoon.sandbox.docker import DockerProvider
    from platoon.sandbox.process import ProcessProvider

    monkeypatch.setitem(sys.modules, "hpc_sandbox", _fake_hpc_sandbox())
    for capabilities in (
        ProcessProvider.capabilities,
        DockerProvider.capabilities,
        get_provider("singularity").capabilities,
    ):
        assert "host_loopback" in capabilities


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capabilities", "task_toml", "isolated", "loopback"),
    [
        ({"routes", "host_loopback"}, "", False, True),  # the process provider
        ({"routes", "host_loopback"}, OFFLINE, False, True),  # it cannot isolate: the sandbox stays on the host's
        ({"network_none", "routes", "host_loopback"}, OFFLINE, True, False),  # isolated: only a route reaches out
        ({"network_none", "routes"}, "", False, False),  # a provider whose sandboxes run elsewhere
    ],
)
async def test_a_session_says_whether_its_sandbox_shares_this_hosts_network(
    make_task, tmp_path, capabilities, task_toml, isolated, loopback
):
    from platoon.sandbox.process import ProcessProvider

    class Provider(ProcessProvider):
        pass

    Provider.capabilities = frozenset({"snapshot", *capabilities})
    service = HarborEnvironmentService(
        task_dirs=[make_task("t", task_toml=task_toml)],
        provider=Provider(root=tmp_path / "sandboxes"),
        default_image="local",
    )
    session = await service.open(await service.get_task("t"), rollout_id="r")
    try:
        assert (session.network_isolated, session.host_loopback) == (isolated, loopback)
    finally:
        await session.close()
