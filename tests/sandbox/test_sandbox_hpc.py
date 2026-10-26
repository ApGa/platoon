"""The hpc-sandbox adapters (``singularity`` and ``enroot``) against a fake ``hpc_sandbox`` package, and against the
hpc-sandbox a checkout locks where it is installed."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import platoon.sandbox
from platoon.sandbox import (
    CWD_ERROR_EXIT_CODE,
    CWD_ERROR_PREFIX,
    CWD_ERROR_REASONS,
    DownloadReport,
    DroppedLink,
    ExecResult,
    Mount,
    SandboxSnapshot,
    SandboxSpec,
    base,
    cleanup,
    cwd_error,
    get_provider,
    network_none_refusal,
)
from platoon.sandbox.enroot import EnrootProvider, enroot_options
from platoon.sandbox.hpc import HpcProvider
from platoon.sandbox.hpc_stacks import pods_refusal
from platoon.sandbox.singularity import SingularityProvider
from platoon.sandbox.stacks import STACKS

PROVIDERS = {"singularity": SingularityProvider, "enroot": EnrootProvider}
IMAGES = {"singularity": "x.sif", "enroot": "x.sqsh"}


def _fake_hpc_sandbox(calls: list[tuple], *, can_delete: bool = True, file_owners: bool = False) -> types.ModuleType:
    """An ``hpc_sandbox`` stand-in whose types mirror the field names of platoon's, with a fake provider per
    runtime that records its calls; with ``file_owners``, its spec takes an image's owner manifest, as 0.5's does."""

    module = types.ModuleType("hpc_sandbox")
    module.Mount = dataclasses.make_dataclass("Mount", ["source", "target", "read_only"])
    fields: list[Any] = ["image", "workdir", "env", "mounts", "network", "cpus", "memory_mb", "labels"]
    if file_owners:
        fields.append(("file_owners", str | None, dataclasses.field(default=None)))
    module.SandboxSpec = dataclasses.make_dataclass("SandboxSpec", fields)
    module.SandboxSnapshot = dataclasses.make_dataclass(
        "SandboxSnapshot", ["id", "provider", "path", "image", "metadata"]
    )
    module.ExecResult = dataclasses.make_dataclass("ExecResult", ["exit_code", "stdout", "stderr", "timed_out"])
    module.DroppedLink = dataclasses.make_dataclass("DroppedLink", ["path", "target", "reason"])
    module.DownloadReport = dataclasses.make_dataclass("DownloadReport", ["dropped_links", "dropped_total"])

    def provider_class(name: str) -> type:
        class Sandbox:
            id = f"{name}-1"

            def __init__(self, image: str) -> None:
                self.image = image

            async def exec(self, cmd, **kwargs):
                calls.append(("exec", cmd, kwargs))
                if "platoon-list" in cmd:  # the adapter lists the sandbox's own processes
                    return module.ExecResult(0, "1:10\n14:12\n", "", False)
                if "platoon-kill" in cmd:
                    return module.ExecResult(0, "", "", False)
                return module.ExecResult(124, "out", "err", True)

            async def upload(self, src, dst):
                calls.append(("upload", src, dst))

            async def download(self, src, dst):
                calls.append(("download", src, dst))
                if src == "/old":  # what hpc-sandbox before 0.2.0 returned
                    return None
                return module.DownloadReport([module.DroppedLink("a/b", "/etc/passwd", "absolute")], 3)

            async def snapshot(self):
                return module.SandboxSnapshot("snap-1", name, "/scratch/snapshots/snap-1", self.image, {"format": 1})

            async def stop(self):
                calls.append(("stop",))

        class Provider:
            def __init__(self, **options):
                calls.append(("init", name, options))
                self.capabilities = frozenset({"snapshot", "isolation", *options.get("extra_capabilities", ())})

            async def start(self, spec):
                calls.append(("start", spec))
                return Sandbox(spec.image)

            async def restore(self, snapshot, spec):
                calls.append(("restore", snapshot, spec))
                return Sandbox(spec.image)

        if can_delete:

            async def delete_snapshot(self, snapshot):
                calls.append(("delete_snapshot", snapshot.id))

            Provider.delete_snapshot = delete_snapshot
        return Provider

    module.SingularityProvider = provider_class("singularity")
    module.EnrootProvider = provider_class("enroot")
    return module


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(PROVIDERS))
async def test_adapts_hpc_sandbox_types_and_calls(monkeypatch, tmp_path: Path, name: str):
    calls: list[tuple] = []
    fake = _fake_hpc_sandbox(calls)
    monkeypatch.setitem(sys.modules, "hpc_sandbox", fake)
    image = IMAGES[name]

    provider = get_provider(name, scratch_root=str(tmp_path))
    assert isinstance(provider, PROVIDERS[name]) and provider.name == name
    # routes and host_loopback: the adapter's own
    assert provider.capabilities == frozenset({"snapshot", "isolation", "routes", "host_loopback"})
    spec = SandboxSpec(image=image, env={"A": "1"}, mounts=[Mount("/tools", "/opt/tools")], labels={"k": "v"})
    sandbox = await provider.start(spec)

    assert calls[0] == ("init", name, {"scratch_root": str(tmp_path)})
    assert calls[1] == (
        "start",
        fake.SandboxSpec(
            image, "/app", {"A": "1"}, [fake.Mount("/tools", "/opt/tools", True)], "host", None, None, {"k": "v"}
        ),
    )
    assert sandbox.id == f"{name}-1"
    # Before any command, the adapter lists the sandbox's own processes from inside it, in "/".
    assert calls[2][0] == "exec" and calls[2][1][-1] == "platoon-list"
    assert calls[2][2] == {"timeout_s": 60.0, "env": None, "cwd": "/", "stdin": None}
    result = await sandbox.exec("sleep 9", timeout_s=1.0)
    assert result == ExecResult(exit_code=124, stdout="out", stderr="err", timed_out=True)
    # A string runs through bash, as platoon's sandbox contract says, not hpc-sandbox's /bin/sh.
    assert calls[3] == ("exec", ["bash", "-c", "sleep 9"], {"timeout_s": 1.0, "env": None, "cwd": None, "stdin": None})
    await sandbox.upload("local", "/tests")
    # A download returns what it left out, as platoon's own types.
    report = await sandbox.download("/logs", "out")
    assert report == DownloadReport([DroppedLink("a/b", "/etc/passwd", "absolute")], 3)
    assert type(report) is DownloadReport and type(report.dropped_links[0]) is DroppedLink
    assert calls[4:6] == [("upload", "local", "/tests"), ("download", "/logs", "out")]
    await sandbox.kill_processes()
    assert calls[6][1][3:] == ["platoon-kill", "100", "0.05", "", "14:12", "1:10"]  # every process but these dies
    del calls[6]
    sandbox.keep_processes(["30:40"])  # a task's init: it lives on, with its descendants
    await sandbox.kill_processes()
    assert calls[6][1][3:] == ["platoon-kill", "100", "0.05", "30:40", "14:12", "1:10"]
    del calls[6]

    snapshot = await sandbox.snapshot()
    assert snapshot == SandboxSnapshot("snap-1", name, "/scratch/snapshots/snap-1", image, {"format": 1})
    restored = await provider.restore(snapshot, spec)
    assert calls[6][1] == fake.SandboxSnapshot("snap-1", name, "/scratch/snapshots/snap-1", image, {"format": 1})
    assert calls[7][1][-1] == "platoon-list"  # so does a restored sandbox
    assert restored.id == f"{name}-1"

    await sandbox.stop()
    await sandbox.stop()
    assert calls[8:] == [("stop",), ("delete_snapshot", "snap-1"), ("stop",)]


@pytest.mark.asyncio
async def test_a_sandbox_that_cannot_list_its_processes_refuses_to_kill(monkeypatch):
    calls: list[tuple] = []
    fake = _fake_hpc_sandbox(calls)
    monkeypatch.setitem(sys.modules, "hpc_sandbox", fake)
    provider = get_provider("singularity")
    original = provider.inner.start

    async def start_without_proc(spec):
        sandbox = await original(spec)

        async def failing_exec(cmd, **kwargs):
            calls.append(("exec", cmd, kwargs))
            return fake.ExecResult(3, "", "platoon: /proc in this sandbox does not list its processes", False)

        sandbox.exec = failing_exec
        return sandbox

    provider.inner.start = start_without_proc
    sandbox = await provider.start(SandboxSpec(image="x.sif"))  # starts all the same

    with pytest.raises(RuntimeError, match="could not list its own processes when it started"):
        await sandbox.kill_processes()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(PROVIDERS))
async def test_restore_refuses_another_providers_snapshot(monkeypatch, name: str):
    calls: list[tuple] = []
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _fake_hpc_sandbox(calls))
    provider = get_provider(name)
    other = next(other for other in PROVIDERS if other != name)

    with pytest.raises(ValueError, match=f"cannot restore snapshot snap-1, which the '{other}' provider took"):
        await provider.restore(SandboxSnapshot("snap-1", other, "/s/snap-1", "x", {}), SandboxSpec(image="x"))
    assert [call[0] for call in calls] == ["init"]


@pytest.mark.asyncio
async def test_snapshots_stay_when_the_inner_provider_cannot_delete_them(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _fake_hpc_sandbox(calls, can_delete=False))
    sandbox = await get_provider("enroot").start(SandboxSpec(image="x.sqsh"))

    await sandbox.snapshot()
    await sandbox.stop()

    assert calls[-1] == ("stop",) and not any(call[0] == "delete_snapshot" for call in calls)


@pytest.mark.parametrize("name", sorted(PROVIDERS))
def test_options_and_capabilities_come_from_the_inner_provider(monkeypatch, name: str):
    calls: list[tuple] = []
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _fake_hpc_sandbox(calls))

    provider = get_provider(name, image_cache="/cache", extra_capabilities=("gpu", "network_none"))

    assert provider.capabilities == frozenset(
        {"snapshot", "isolation", "gpu", "network_none", "routes", "host_loopback"}
    )
    assert calls == [("init", name, {"image_cache": "/cache", "extra_capabilities": ("gpu", "network_none")})]


@pytest.mark.parametrize("name", sorted(PROVIDERS))
def test_capabilities_outside_platoons_contract_are_left_out(monkeypatch, name: str):
    """A capability says what platoon may ask of a provider (``base.CAPABILITIES``): hpc-sandbox's providers also
    report ``pods``, which is not one of them (an adapter declares ``stacks`` where it runs stacks as pods:
    ``test_sandbox_hpc_stacks.py``), so the adapter leaves it out, and any capability hpc-sandbox adds later until
    platoon has a use for it."""

    monkeypatch.setitem(sys.modules, "hpc_sandbox", _fake_hpc_sandbox([]))

    provider = get_provider(name, extra_capabilities=("pods", "gpu", "a-later-capability"))

    assert provider.capabilities == frozenset({"snapshot", "isolation", "gpu", "routes", "host_loopback"})
    assert provider.capabilities <= base.CAPABILITIES
    assert "pods" in provider.inner.capabilities  # the inner provider keeps its own


@pytest.mark.parametrize("name", sorted(PROVIDERS))
def test_needs_the_hpc_sandbox_package(monkeypatch, name: str):
    monkeypatch.setitem(sys.modules, "hpc_sandbox", None)

    expected = (
        rf"The {name} provider needs the hpc-sandbox package: install the hpc extra, e\.g\. pip install "
        r"'platoon\[hpc\]', or in a platoon checkout uv sync --extra tinker --extra ors --extra hpc \(uv sync removes "
        r"the extras it is not given\)$"
    )
    with pytest.raises(ImportError, match=expected):
        PROVIDERS[name]()


def test_both_providers_share_one_adapter():
    assert all(issubclass(provider, HpcProvider) for provider in PROVIDERS.values())
    assert {provider.inner_class for provider in PROVIDERS.values()} == {"SingularityProvider", "EnrootProvider"}


# Fields the adapter implements itself rather than passing to hpc-sandbox: upstreams become binds and forwarders.
ADAPTER_FIELDS = {"SandboxSpec": {"upstreams"}}
# Fields of a later hpc-sandbox, which the adapter passes only to one that has them: an image's owner manifest (0.5).
LATER_FIELDS = {"SandboxSpec": {"file_owners"}}


@pytest.mark.parametrize(
    "name", ["Mount", "SandboxSpec", "ExecResult", "SandboxSnapshot", "DownloadReport", "DroppedLink"]
)
def test_hpc_sandbox_types_match_platoon(name: str):
    hpc_sandbox = pytest.importorskip("hpc_sandbox")

    theirs = [field.name for field in dataclasses.fields(getattr(hpc_sandbox, name))]
    ours = [field.name for field in dataclasses.fields(getattr(base, name))]
    later = LATER_FIELDS.get(name, set()) - set(theirs)
    assert theirs == [field for field in ours if field not in ADAPTER_FIELDS.get(name, set()) | later]


def test_enroot_passes_the_singularity_option_names_through(monkeypatch):
    """hpc-sandbox's EnrootProvider takes the singularity provider's option names itself, so a provider_options
    block written for a Singularity site reaches it as it is; root, another name for remap_root, is the one alias
    platoon's enroot provider renames."""

    calls: list[tuple] = []
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _fake_hpc_sandbox(calls))
    singularity_block = {
        "image_cache": "/cache",
        "gpu": "nv",
        "allow_network_none": False,
        "stop_grace_s": 5.0,
        "cgroup_limits": False,
        "bind_resolv_conf": False,
    }

    get_provider("enroot", **singularity_block)
    get_provider("enroot", root=False)
    get_provider("enroot", gpu=None, gpus="0,1", remap_root=True)

    assert calls == [
        ("init", "enroot", singularity_block),
        ("init", "enroot", {"remap_root": False}),
        ("init", "enroot", {"gpu": None, "gpus": "0,1", "remap_root": True}),
    ]


def test_enroot_refuses_root_and_remap_root_together():
    with pytest.raises(ValueError, match="Set remap_root or root for the enroot provider, not both"):
        enroot_options({"root": True, "remap_root": True})


def _with_stacks(provider: object, *capabilities: str) -> frozenset[str]:
    """``capabilities``, and ``stacks`` where the installed hpc-sandbox's pods run them here (0.5 on a node that nests
    user namespaces)."""

    assert isinstance(provider, HpcProvider)
    return frozenset({*capabilities, *([STACKS] if pods_refusal(provider.name, provider.inner_class) is None else [])})


def _published_hpc_sandbox():
    hpc_sandbox = pytest.importorskip("hpc_sandbox")
    if not hasattr(hpc_sandbox, "EnrootProvider"):
        pytest.skip("this hpc-sandbox has no enroot provider")
    return hpc_sandbox


def test_one_block_builds_both_published_providers(tmp_path: Path):
    """Against the installed hpc-sandbox (the lock's 0.5.1): its EnrootProvider takes every option its
    SingularityProvider takes, so one provider_options block builds both providers and means the same on both, with
    platoon's capabilities only (not its pods), and ``stacks`` where its pods run them here; root, which the
    SingularityProvider refuses, builds the EnrootProvider too."""

    hpc_sandbox = _published_hpc_sandbox()
    singularity_names = set(inspect.signature(hpc_sandbox.SingularityProvider).parameters)
    assert singularity_names <= set(inspect.signature(hpc_sandbox.EnrootProvider).parameters)
    block = {
        "scratch_root": str(tmp_path),
        "image_cache": str(tmp_path / "cache"),
        "gpu": "nv",
        "allow_network_none": False,
        "bind_resolv_conf": False,
        "stop_grace_s": 5.0,
        "cgroup_limits": False,
        "fakeroot": False,
    }

    singularity, enroot = get_provider("singularity", **block), get_provider("enroot", **block)

    for provider in (singularity, enroot):
        assert provider.capabilities == _with_stacks(
            provider, "snapshot", "isolation", "gpu", "routes", "host_loopback"
        )
    assert (singularity.inner.bind_resolv_conf, enroot.inner.bind_resolv_conf) == (False, False)
    assert enroot.inner.gpus == "all"
    with pytest.raises(TypeError, match="root"):
        get_provider("singularity", **block, root=True)
    enroot = get_provider("enroot", scratch_root=str(tmp_path), gpu="nv", gpus=None, root=False)
    assert (enroot.inner.gpus, enroot.inner.remap_root) == ("all", False)
    assert enroot.capabilities == _with_stacks(
        enroot, "snapshot", "isolation", "network_none", "gpu", "routes", "host_loopback"
    )


@pytest.mark.parametrize("provider", ["singularity", "enroot"])
@pytest.mark.parametrize("options", [{}, {"allow_network_none": True}, {"allow_network_none": False}])
def test_the_refusal_said_before_any_sandbox_starts_matches_the_published_providers(
    tmp_path: Path, provider: str, options: dict
):
    """Harbor refuses no-network tasks when it lists them on a provider that would refuse every sandbox without a
    network (network_none_refusal, read from the options alone): it must say so exactly when the built provider lacks
    network_none, which an enroot block shared with a Singularity site (allow_network_none: false) also takes away."""

    _published_hpc_sandbox()
    built = get_provider(provider, scratch_root=str(tmp_path), **options)

    refused = network_none_refusal(provider, options)
    assert (refused is None) == ("network_none" in built.capabilities), (provider, options, built.capabilities)
    assert network_none_refusal(built) == refused


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"gpu": "rocm"}, "NVIDIA GPUs only"),
        ({"gpu": "nv", "gpus": "all"}, "not both"),
        ({"cgroup_limits": True}, "cannot apply cgroup limits"),
        ({"fakeroot": True}, "maps only your uid"),  # 0.4 on: root there could not switch users
    ],
)
def test_the_published_enroot_provider_refuses_what_enroot_cannot_honor(tmp_path: Path, options: dict, message: str):
    _published_hpc_sandbox()

    with pytest.raises(ValueError, match=message):
        get_provider("enroot", scratch_root=str(tmp_path), **options)


def test_the_published_enroot_provider_refuses_the_fuse_overlay_layout_platoon_leaves_out(tmp_path: Path):
    """hpc-sandbox's EnrootProvider takes rootfs_layout (0.3 on) but refuses fuse-overlay, which a provider_options
    block shared with a Singularity site may set: platoon's enroot provider leaves the option out, so that such a block
    builds it."""

    hpc_sandbox = _published_hpc_sandbox()
    with pytest.raises(ValueError, match="not rootfs_layout='fuse-overlay'"):
        hpc_sandbox.EnrootProvider(scratch_root=str(tmp_path), rootfs_layout="fuse-overlay")

    provider = get_provider("enroot", scratch_root=str(tmp_path), rootfs_layout="fuse-overlay")
    assert "rootfs_layout" not in provider.options


# ------------------------------------------------------------------------------------------- hpc-sandbox 0.2's contract


@pytest.mark.asyncio
async def test_a_download_from_an_older_hpc_sandbox_reports_nothing(monkeypatch):
    """hpc-sandbox before 0.2.0 returned None from a download: the adapter says it does not know."""

    calls: list[tuple] = []
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _fake_hpc_sandbox(calls))
    sandbox = await get_provider("singularity").start(SandboxSpec(image="x.sif"))

    assert await sandbox.download("/old", "out") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", CWD_ERROR_REASONS)
async def test_a_command_that_cannot_enter_its_working_directory_passes_through(monkeypatch, reason: str):
    """hpc-sandbox 0.2 answers such a command as `docker exec` does, as the command's failure: exit status 126, empty
    stdout and on stderr only its line, which the adapter passes through unchanged and cwd_error reads."""

    calls: list[tuple] = []
    fake = _fake_hpc_sandbox(calls)
    monkeypatch.setitem(sys.modules, "hpc_sandbox", fake)
    provider = get_provider("enroot")
    original = provider.inner.start
    line = f"{CWD_ERROR_PREFIX}/app: {reason}\n"

    async def start(spec):
        sandbox = await original(spec)
        listing = sandbox.exec

        async def cannot_enter(cmd, **kwargs):
            if "platoon-list" in cmd:
                return await listing(cmd, **kwargs)
            return fake.ExecResult(CWD_ERROR_EXIT_CODE, "", line, False)

        sandbox.exec = cannot_enter
        return sandbox

    provider.inner.start = start
    sandbox = await provider.start(SandboxSpec(image="x.sqsh"))

    result = await sandbox.exec("true")
    assert result == ExecResult(exit_code=126, stdout="", stderr=line, timed_out=False)
    assert cwd_error(result) == line.removesuffix("\n")


@pytest.mark.parametrize(
    "result",
    [
        ExecResult(126, "", ""),  # docker's, or a command's own 126: it says nothing
        ExecResult(126, "partial output\n", f"{CWD_ERROR_PREFIX}/app: not a directory\n"),  # the command ran
        ExecResult(1, "", f"{CWD_ERROR_PREFIX}/app: not a directory\n"),
        ExecResult(126, "", f"{CWD_ERROR_PREFIX}/app: not a directory\n", timed_out=True),
        ExecResult(126, "", f"noise\n{CWD_ERROR_PREFIX}/app: not a directory\n"),  # not the whole of stderr
        ExecResult(126, "", f"{CWD_ERROR_PREFIX}/app: no such directory\n"),  # not one of the reasons
    ],
    ids=["says-nothing", "with-stdout", "other-status", "timed-out", "not-alone", "unknown-reason"],
)
def test_cwd_error_reads_only_the_line_hpc_sandbox_gives(result: ExecResult):
    assert cwd_error(result) is None


def test_cwd_error_takes_a_directory_with_line_breaks_whole():
    """The directory is part of the line as it is, line breaks included: the whole of stderr is the line."""

    line = f"{CWD_ERROR_PREFIX}/work/a b\nc: permission denied"
    assert cwd_error(ExecResult(CWD_ERROR_EXIT_CODE, "", line + "\n")) == line
    assert cwd_error(ExecResult(CWD_ERROR_EXIT_CODE, "", line)) == line


def test_the_working_directory_error_is_the_published_hpc_sandbox_s():
    """platoon's constants mirror hpc-sandbox's (0.2 on), and cwd_error reads every line hpc-sandbox makes of what
    its prologue printed, whatever the runtime printed before it."""

    hpc_sandbox = pytest.importorskip("hpc_sandbox")
    assert hpc_sandbox.CWD_ERROR_EXIT_CODE == CWD_ERROR_EXIT_CODE
    assert hpc_sandbox.CWD_ERROR_PREFIX == CWD_ERROR_PREFIX
    assert tuple(hpc_sandbox.CWD_ERROR_REASONS) == CWD_ERROR_REASONS
    from hpc_sandbox import sandbox as published

    for reason in CWD_ERROR_REASONS:
        for directory in ("/app", "/work/a b", "/x\ny"):
            printed = f"runtime noise\n{CWD_ERROR_PREFIX}{directory}: {reason}\n"
            line = published._cwd_error(CWD_ERROR_EXIT_CODE, printed)
            assert line is not None
            assert cwd_error(ExecResult(CWD_ERROR_EXIT_CODE, "", line)) == f"{CWD_ERROR_PREFIX}{directory}: {reason}"


def test_the_published_enroot_provider_takes_digest_pins(tmp_path: Path):
    """`platoon harbor images build --builder docker --push` records images as repo@sha256:<hex>, which enroot sites
    import: hpc-sandbox 0.2 imports such a pin as repo:sha256:<hex> (a form enroot 3.x reads), where 0.1 refused it."""

    _published_hpc_sandbox()
    digest = "sha256:" + "0123456789abcdef" * 4
    images = get_provider("enroot", scratch_root=str(tmp_path)).inner.images

    for reference in (f"docker://ghcr.io/org/task@{digest}", f"docker://ghcr.io/org/task:v1@{digest}"):
        images.format.check(reference)  # what a sandbox start checks before it creates anything
        uri = images.format.pull_uri(reference)
        assert uri.endswith(f"task:{digest}") and "@" not in uri, uri


@pytest.mark.asyncio
async def test_an_owner_manifest_reaches_an_hpc_sandbox_that_applies_one(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _fake_hpc_sandbox(calls, file_owners=True))
    provider = get_provider("singularity", fakeroot=True)

    await provider.start(SandboxSpec(image="x.sif", file_owners="/srv/images/x.owners"))
    await provider.start(SandboxSpec(image="x.sif"))

    owners = [call[1].file_owners for call in calls if call[0] == "start"]
    assert owners == ["/srv/images/x.owners", None]
    assert platoon.sandbox.file_owners_refusal("singularity", {"fakeroot": True}) is None
    assert platoon.sandbox.loses_file_owners("singularity") and platoon.sandbox.loses_file_owners(provider)
    assert not platoon.sandbox.loses_file_owners("docker")  # its images keep their owners


@pytest.mark.asyncio
async def test_an_hpc_sandbox_before_0_5_refuses_an_owner_manifest_and_runs_without_one(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _fake_hpc_sandbox(calls))
    provider = get_provider("singularity", fakeroot=True)

    with pytest.raises(ValueError, match="SandboxSpec.file_owners.* Install hpc-sandbox 0.5"):
        await provider.start(SandboxSpec(image="x.sif", file_owners="/srv/images/x.owners"))
    await provider.start(SandboxSpec(image="x.sif"))

    assert [call[0] for call in calls].count("start") == 1  # the refused one never started
    assert "hpc-sandbox 0.5" in (platoon.sandbox.file_owners_refusal("singularity", {}) or "")
    assert platoon.sandbox.file_owners_refusal("docker", {}) is None  # it needs none


def test_the_owner_manifest_refusal_matches_the_installed_hpc_sandbox():
    """Against the installed hpc-sandbox: a manifest is refused exactly when its SandboxSpec has no file_owners."""

    hpc_sandbox = pytest.importorskip("hpc_sandbox")
    takes = "file_owners" in {field.name for field in dataclasses.fields(hpc_sandbox.SandboxSpec)}

    assert (platoon.sandbox.file_owners_refusal("singularity", {"fakeroot": True}) is None) is takes


@pytest.mark.asyncio
async def test_a_start_given_up_runs_to_its_end_and_its_sandbox_is_stopped(monkeypatch):
    """Cancelled while hpc-sandbox makes the container, a start raises at once, while hpc-sandbox's start runs to its
    end uncancelled (killing the runtime's command midway can leave a container that hpc-sandbox's own rollback does
    not stop); the sandbox it made is stopped then, and its registration for cleanup goes with it."""

    calls: list[tuple] = []
    fake = _fake_hpc_sandbox(calls)
    started, release = asyncio.Event(), asyncio.Event()
    start = fake.SingularityProvider.start

    async def slow_start(self, spec):
        started.set()
        await release.wait()  # the runtime making the container: a cancellation here would kill it midway
        return await start(self, spec)

    monkeypatch.setattr(fake.SingularityProvider, "start", slow_start)
    monkeypatch.setitem(sys.modules, "hpc_sandbox", fake)
    provider = get_provider("singularity")
    registered = len(cleanup._live)
    opening = asyncio.ensure_future(provider.start(SandboxSpec(image="x.sif", labels={"k": "v"})))
    await asyncio.wait_for(started.wait(), timeout=10)
    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening
    assert len(cleanup._live) == registered + 1  # until what it starts is stopped
    release.set()
    for _ in range(500):
        if ("stop",) in calls:
            break
        await asyncio.sleep(0.01)
    assert [call[0] for call in calls] == ["init", "start", "stop"]  # never cancelled, then stopped
    assert len(cleanup._live) == registered


@pytest.mark.asyncio
async def test_a_start_given_up_in_the_step_in_which_it_returns_its_sandbox_stops_that_sandbox(monkeypatch):
    """The caller is cancelled in the very step of the loop in which hpc-sandbox's start returns the sandbox (as a
    deadline or a rollout's stop can): the start raises, and the sandbox it made is stopped, never left running with
    its registration for cleanup dropped."""

    calls: list[tuple] = []
    fake = _fake_hpc_sandbox(calls)
    start = fake.SingularityProvider.start
    opening: list[asyncio.Future[Any]] = []

    async def returning_as_its_caller_is_cancelled(self, spec):
        sandbox = await start(self, spec)
        asyncio.get_running_loop().call_soon(opening[0].cancel)  # before the start's own completion reaches its caller
        return sandbox

    monkeypatch.setattr(fake.SingularityProvider, "start", returning_as_its_caller_is_cancelled)
    monkeypatch.setitem(sys.modules, "hpc_sandbox", fake)
    provider = get_provider("singularity")
    registered = len(cleanup._live)
    opening.append(asyncio.ensure_future(provider.start(SandboxSpec(image="x.sif", labels={"k": "v"}))))
    with pytest.raises(asyncio.CancelledError):
        await opening[0]
    for _ in range(500):
        if ("stop",) in calls:
            break
        await asyncio.sleep(0.01)
    assert [call[0] for call in calls] == ["init", "start", "stop"]
    assert len(cleanup._live) == registered
