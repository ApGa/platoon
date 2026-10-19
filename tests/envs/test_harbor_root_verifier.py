"""Harbor verifiers that run as root (``verifier_user: honor``): which tasks may ask for it, which sandbox providers can
give it and which are refused before any rollout, the separate verifiers' own provider options, the checks of a started
verifier's sandbox (root with other users mapped, an image whose shell nobody may run), how a root verifier grades
(Docker's umask, artifacts readable by every user, the image's own directories left as they are), the images of a
fetched dataset that the oci builder of platoon before 0.3.0 built, and the tasks a fetch left out."""

from __future__ import annotations

import dataclasses
import inspect
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from site_cache import use_site_cache

import platoon.sandbox
from platoon.envs.harbor import HarborEnvironmentService
from platoon.envs.harbor.images import FileOp, ImageEntry, SessionImage, apply_image_files, update_image_map
from platoon.envs.harbor.lock import read_lock
from platoon.envs.harbor.task import HarborTask, HarborTaskError, is_root
from platoon.envs.harbor.users import NOBODY, nobody_problem, root_problem
from platoon.envs.service import EnvSetupError
from platoon.sandbox import ExecResult, SandboxSpec, get_provider, root_refusal, rootfs_layout_refusal
from platoon.sandbox.process import ProcessProvider, ProcessSandbox

ROOT = "${PLATOON_SANDBOX_ROOT:-}"
# A task whose agent runs as nobody and whose separate verifier runs as root, as tasks do whose tests run the agent's
# code as another user.
ROOT_VERIFIER = """artifacts = ["/app/answer.txt", "/app/out"]
[agent]
user = "65534"
[verifier]
environment_mode = "separate"
user = "root"
timeout_sec = 30
"""
# Passes only where the tests run with Docker's umask, and the workdir and the artifacts are open to every user.
CHECK_AS_DOCKER = f"""
mode() {{ stat -c %a "{ROOT}$1"; }}
found="umask=$(umask) app=$(mode /app) answer=$(mode /app/answer.txt) out=$(mode /app/out)"
found="$found result=$(mode /app/out/result.txt)"
echo "$found"
if [ "$found" = "umask=0022 app=755 answer=644 out=755 result=644" ]; then echo 1; else echo 0; fi \\
  > {ROOT}/logs/verifier/reward.txt
"""
# The agent leaves its files private to itself, as an agent does under a umask of 077.
LEAVE_PRIVATE_FILES = (
    "mkdir -p out && echo 42 > answer.txt && echo r > out/result.txt && chmod 700 out && chmod 600 answer.txt "
    "out/result.txt"
)
SEPARATE_PLAIN = 'artifacts = ["/app/a"]\n[verifier]\nenvironment_mode = "separate"\n'


# ----------------------------------------------------------------------------------------------- which tasks ask for it


@pytest.mark.parametrize(
    ("user", "root"),
    [
        ("root", True),
        ("0", True),
        ("0:0", True),
        ("root:root", True),
        ("root:0", True),
        ("nobody", False),
        ("65534", False),
        ("root:1000", False),
        ("1000:0", False),
        ("rootless", False),
    ],
)
def test_root_is_named_as_docker_names_it(user: str, root: bool):
    assert is_root(user) is root


def test_honor_runs_a_separate_verifier_whose_user_is_root_as_root(make_task):
    task_dir = make_task("root", task_toml=ROOT_VERIFIER)

    task = HarborTask.load(task_dir, agent_user="ignore", verifier_user="honor")

    assert task.verifier_root
    assert task.ignored_users == {"agent": "65534"}  # the agent still runs as the sandbox's uid
    for user in ("0", "0:0"):
        root = make_task(f"root-{user}", task_toml=ROOT_VERIFIER.replace('user = "root"', f'user = "{user}"'))
        assert HarborTask.load(root, agent_user="ignore", verifier_user="honor").verifier_root
    numeric = make_task("root-int", task_toml=ROOT_VERIFIER.replace('user = "root"', "user = 0"))
    assert HarborTask.load(numeric, agent_user="ignore", verifier_user="honor").verifier_root


def test_without_verifier_user_agent_user_decides_for_both_phases_as_before(make_task):
    task_dir = make_task("root", task_toml=ROOT_VERIFIER)

    with pytest.raises(HarborTaskError, match=r"agent_user: ignore$"):  # the agent's user is refused too
        HarborTask.load(task_dir)
    task = HarborTask.load(task_dir, agent_user="ignore")
    assert not task.verifier_root
    assert task.ignored_users == {"agent": "65534", "verifier": "root"}

    verifier_only = make_task("verifier-only", task_toml=ROOT_VERIFIER.replace('[agent]\nuser = "65534"\n', ""))
    with pytest.raises(HarborTaskError, match="agent_user: ignore, or verifier_user: honor to run its tests as root"):
        HarborTask.load(verifier_only)


def test_verifier_user_decides_for_the_verifier_and_agent_user_for_the_agent(make_task):
    task_dir = make_task("root", task_toml=ROOT_VERIFIER)

    with pytest.raises(HarborTaskError, match=r"\[agent\].user = '65534', but .* sets agent_user"):
        HarborTask.load(task_dir, verifier_user="honor")
    task = HarborTask.load(task_dir, agent_user="ignore", verifier_user="ignore")
    assert (task.verifier_root, task.ignored_users) == (False, {"agent": "65534", "verifier": "root"})
    with pytest.raises(HarborTaskError, match=r"\[verifier\].user = 'root'.*verifier_user: ignore, or honor"):
        HarborTask.load(task_dir, agent_user="ignore", verifier_user="refuse")
    with pytest.raises(ValueError, match="verifier_user must be one of"):
        HarborTask.load(task_dir, verifier_user="obey")  # type: ignore[arg-type]


def test_honor_takes_only_root(make_task):
    nobody = make_task("nobody", task_toml=ROOT_VERIFIER.replace('user = "root"', 'user = "nobody"'))

    with pytest.raises(HarborTaskError, match="honors only root; .* unless the environment sets verifier_user: ignore"):
        HarborTask.load(nobody, agent_user="ignore", verifier_user="honor")
    with pytest.raises(HarborTaskError, match=r"verifier_user: ignore$"):  # nor a root verifier in another group
        HarborTask.load(
            make_task("group", task_toml=ROOT_VERIFIER.replace('user = "root"', 'user = "root:1000"')),
            agent_user="ignore",
            verifier_user="honor",
        )
    assert not HarborTask.load(nobody, agent_user="ignore", verifier_user="ignore").verifier_root


def test_honor_refuses_a_root_verifier_in_the_agents_own_sandbox(make_task):
    shared = make_task("shared", task_toml='[verifier]\nuser = "root"\n')

    with pytest.raises(HarborTaskError, match="graded in its agents' sandbox .* its agents would run as root too"):
        HarborTask.load(shared, verifier_user="honor")
    assert HarborTask.load(shared, verifier_user="ignore").ignored_users == {"verifier": "root"}


def test_a_task_without_a_verifier_user_is_not_a_root_verifier(make_task):
    task_dir = make_task("plain", task_toml='artifacts = ["/app/a"]\n[verifier]\nenvironment_mode = "separate"\n')

    task = HarborTask.load(task_dir, verifier_user="honor")

    assert (task.verifier_root, task.ignored_users) == (False, {})


# ---------------------------------------------------------------------------------------------- which providers give it


def _hpc_sandbox(*, fakeroot: bool) -> types.ModuleType:
    """An ``hpc_sandbox`` whose Singularity provider takes ``fakeroot`` (and hands back what a --fakeroot sandbox's
    other users write), or, as 0.3.0's, does not."""

    module = types.ModuleType("hpc_sandbox")
    if fakeroot:

        class SingularityProvider:
            def __init__(self, *, rootfs_layout=None, fakeroot=False, extra_start_args=(), allow_network_none=False):
                self.capabilities = frozenset({"snapshot", "isolation"})

    else:

        class SingularityProvider:  # type: ignore[no-redef]
            def __init__(self, *, rootfs_layout=None, extra_start_args=(), allow_network_none=False):
                self.capabilities = frozenset({"snapshot", "isolation"})

    module.SingularityProvider = SingularityProvider
    module.Mount = dataclasses.make_dataclass("Mount", ["source", "target", "read_only"])
    return module


@pytest.mark.parametrize(
    ("provider", "options", "refusal"),
    [
        ("process", {}, "runs commands on the host as this process's user"),
        ("enroot", {}, "maps only your own uid into a sandbox"),
        ("enroot", {"remap_root": True}, "maps only your own uid into a sandbox"),
        ("docker", {}, None),  # the image's own user decides: the started sandbox is checked
        ("no.such.module:Provider", {}, None),  # building it reports the error
    ],
)
def test_providers_say_from_their_options_whether_they_can_run_a_verifier_as_root(provider, options, refusal):
    said = root_refusal(provider, options)

    assert (said is None) is (refusal is None), said
    if refusal is not None:
        assert refusal in said


@pytest.mark.parametrize(
    ("fakeroot", "options", "job_default", "refusal"),
    [
        (True, {}, None, "unless its options set fakeroot: true"),
        (True, {"fakeroot": False}, None, "unless its options set fakeroot: true"),
        (True, {"fakeroot": True}, None, None),
        (True, {"fakeroot": True, "rootfs_layout": "copy"}, None, None),
        (True, {"extra_start_args": ["--fakeroot"]}, None, None),
        (True, {"extra_start_args": ["-f"]}, None, None),
        (True, {"fakeroot": True, "rootfs_layout": "fuse-overlay"}, None, "--fakeroot needs rootfs_layout: copy"),
        (True, {"fakeroot": True}, "fuse-overlay", None),  # hpc-sandbox gives such a provider copies
        # An hpc-sandbox that cannot delete what the other users of a --fakeroot sandbox write, whichever the spelling.
        (False, {"fakeroot": True}, None, "cannot delete what the other users of a --fakeroot sandbox write"),
        (False, {"extra_start_args": ["--fakeroot"]}, None, "cannot delete what the other users"),
        (False, {}, None, "unless its options set fakeroot: true"),
    ],
)
def test_the_singularity_provider_gives_root_only_through_fakeroot_on_an_hpc_sandbox_that_cleans_up(
    monkeypatch, fakeroot, options, job_default, refusal
):
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _hpc_sandbox(fakeroot=fakeroot))
    if job_default:
        monkeypatch.setenv("HPC_SANDBOX_ROOTFS_LAYOUT", job_default)
    else:
        monkeypatch.delenv("HPC_SANDBOX_ROOTFS_LAYOUT", raising=False)

    said = root_refusal("singularity", options)

    assert (said is None) is (refusal is None), said
    if refusal is not None:
        assert refusal in said


def test_a_fakeroot_option_the_installed_hpc_sandbox_lacks_would_start_no_sandbox(monkeypatch):
    monkeypatch.delenv("HPC_SANDBOX_ROOTFS_LAYOUT", raising=False)
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _hpc_sandbox(fakeroot=False))

    assert "takes no fakeroot option" in rootfs_layout_refusal("singularity", {"fakeroot": True})
    assert "takes no fakeroot option" in rootfs_layout_refusal("singularity", {"fakeroot": False})
    assert rootfs_layout_refusal("singularity", {"extra_start_args": ["--fakeroot"]}) is None

    monkeypatch.setitem(sys.modules, "hpc_sandbox", _hpc_sandbox(fakeroot=True))
    assert rootfs_layout_refusal("singularity", {"fakeroot": True}) is None
    assert "need copy" in rootfs_layout_refusal("singularity", {"fakeroot": True, "rootfs_layout": "fuse-overlay"})


def test_a_built_provider_says_it_too(monkeypatch, tmp_path: Path):
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _hpc_sandbox(fakeroot=True))

    assert root_refusal(get_provider("singularity", fakeroot=True)) is None
    assert "fakeroot: true" in root_refusal(get_provider("singularity"))
    assert "this process's user" in root_refusal(ProcessProvider(tmp_path))


def test_the_root_refusal_matches_the_installed_hpc_sandbox():
    """Against the installed hpc-sandbox: a fakeroot sandbox is refused exactly when its Singularity provider does not
    take ``fakeroot``, the option that comes with handing back what a --fakeroot sandbox's other users write."""

    hpc_sandbox = pytest.importorskip("hpc_sandbox")
    takes = "fakeroot" in inspect.signature(hpc_sandbox.SingularityProvider).parameters

    assert (root_refusal("singularity", {"fakeroot": True}) is None) is takes
    assert (root_refusal("singularity", {"extra_start_args": ["--fakeroot"]}) is None) is takes


# ------------------------------------------------------------------------------------------- a provider that gives root


class RootProcessProvider(ProcessProvider):
    """The process provider, standing in for one that runs a sandbox's commands as root with other users mapped (as
    Singularity's --fakeroot does): asked which user its commands run as, a sandbox answers uid ``uid`` with the ids
    ``0`` to ``mapped`` in its user namespace, and asked to run the image's shell as nobody, it answers ``nobody`` (by
    default, that the shell ran).  Its class says whether its options give root (``refusal``).  Each sandbox's workdir
    starts with mode ``workdir_mode``: 755, as a --fakeroot sandbox's start makes a workdir the image lacks (Docker's
    modes), or the mode of one the image has, such as 700 for a /root kept private."""

    def __init__(
        self,
        root: str,
        uid: int = 0,
        mapped: int = 65536,
        refusal: str | None = None,
        nobody: tuple[int, str, str] = (0, "ran\n", ""),
        workdir_mode: int = 0o755,
    ) -> None:
        super().__init__(root)
        self.options = {
            "root": root,
            "uid": uid,
            "mapped": mapped,
            "refusal": refusal,
            "nobody": nobody,
            "workdir_mode": workdir_mode,
        }
        self.started: list[tuple[SandboxSpec, ProcessSandbox]] = []

    @classmethod
    def root_refusal(cls, options: dict[str, Any]) -> str | None:
        return options.get("refusal")

    async def start(self, spec: SandboxSpec) -> ProcessSandbox:
        sandbox = await super().start(spec)
        workdir = sandbox.host_path(spec.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        workdir.chmod(self.options["workdir_mode"])
        real = sandbox.exec
        uid, mapped = self.options["uid"], self.options["mapped"]

        async def exec(cmd: str | list[str], **options: Any) -> ExecResult:
            if isinstance(cmd, list) and cmd[3:4] == ["platoon-root-check"]:
                maps = "".join(
                    f"{kind}_map 0 1\n" + (f"{kind}_map 1 {mapped}\n" if mapped else "") for kind in ("uid", "gid")
                )
                return ExecResult(0, f"uid {uid}\n{maps}", "")
            if isinstance(cmd, list) and cmd[3:4] == ["platoon-nobody-check"]:
                return ExecResult(*self.options["nobody"])
            return await real(cmd, **options)

        sandbox.exec = exec  # type: ignore[method-assign]
        self.started.append((spec, sandbox))
        return sandbox


@pytest.fixture
def rootful(monkeypatch) -> str:
    """The name ``rootful-process``, which builds a :class:`RootProcessProvider` from its options."""

    monkeypatch.setitem(platoon.sandbox._BUILTIN_PROVIDERS, "rootful-process", f"{__name__}:RootProcessProvider")
    return "rootful-process"


def _service(
    task_dirs: list[Path], provider: str, root: Path, provider_options: dict | None = None, **options: Any
) -> HarborEnvironmentService:
    return HarborEnvironmentService(
        task_dirs=task_dirs,
        provider=provider,
        provider_options={"root": str(root), **(provider_options or {})},
        default_image="local",
        agent_user="ignore",
        **options,
    )


# ---------------------------------------------------------------------------------------------- refused before rollouts


@pytest.mark.asyncio
async def test_root_verifiers_are_refused_when_listed_on_a_provider_that_cannot_give_root(make_task, tmp_path):
    tasks = [make_task(f"root-{n}", task_toml=ROOT_VERIFIER) for n in range(4)] + [make_task("plain")]
    service = _service(tasks, "process", tmp_path / "sandboxes", verifier_user="honor")
    listed = await service.list_tasks("train")

    with pytest.raises(EnvSetupError, match="4 of the 5 tasks of split 'train' cannot run their verifier") as refused:
        await service.check_tasks("train", listed)

    assert refused.value.failure_kind == "env_config"
    message = str(refused.value)
    assert "(root-0, root-1, root-2 and 1 more)" in message
    assert "its verifier runs as root, and the verifiers' sandboxes cannot give it root: the process" in message
    assert "verifier_provider_options" in message and "exclude_tasks" in message
    assert "provider" not in vars(service) and "verifier_provider" not in vars(service)  # nothing was built
    await service.check_tasks("train", ["plain"])  # what the run keeps once it leaves them out
    ignored = _service(tasks, "process", tmp_path / "sandboxes", verifier_user="ignore")
    await ignored.check_tasks("train", await ignored.list_tasks("train"))  # run as the sandbox's own user, as before


@pytest.mark.asyncio
async def test_a_root_verifier_opened_directly_is_refused_before_the_agent_runs(make_task, tmp_path):
    task = make_task("root", task_toml=ROOT_VERIFIER)
    service = _service([task], "process", tmp_path / "sandboxes", verifier_user="honor")

    with pytest.raises(EnvSetupError, match="runs its verifier as root, and the verifiers' sandboxes") as refused:
        await service.open(await service.get_task("root"), rollout_id="r")

    assert refused.value.failure_kind == "env_config"
    assert not (tmp_path / "sandboxes").exists()  # no sandbox started


@pytest.mark.asyncio
async def test_tasks_whose_verifier_user_honor_refuses_at_load_are_named_when_listed(make_task, tmp_path, monkeypatch):
    """A fetched dataset lists the tasks its lock lets through, users and all; with verifier_user: honor the service
    names those whose [verifier].user it cannot honor when the run lists them (here, as such a listing would)."""

    nobody = make_task("nobody", task_toml=ROOT_VERIFIER.replace('user = "root"', 'user = "nobody"'))
    service = _service([nobody], "process", tmp_path / "sandboxes", verifier_user="honor")
    monkeypatch.setattr(
        service, "_peek", lambda name: HarborTask.load(nobody, agent_user="ignore", verifier_user="honor")
    )

    with pytest.raises(EnvSetupError, match="1 of the 1 tasks .* nobody: .*honors only root"):
        await service.check_tasks("train", ["nobody"])


@pytest.mark.asyncio
@pytest.mark.parametrize("fakeroot", [True, False], ids=["cleans-up", "0.3.0"])
async def test_singularity_root_verifiers_need_fakeroot_on_an_hpc_sandbox_that_cleans_up(
    make_task, monkeypatch, fakeroot: bool
):
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _hpc_sandbox(fakeroot=fakeroot))
    monkeypatch.delenv("HPC_SANDBOX_ROOTFS_LAYOUT", raising=False)
    tasks = [make_task("root", task_toml=ROOT_VERIFIER)]

    def service(**verifier_options: Any) -> HarborEnvironmentService:
        return HarborEnvironmentService(
            task_dirs=tasks,
            provider="singularity",
            provider_options={"allow_network_none": True, "rootfs_layout": "copy"},
            verifier_provider_options=verifier_options or None,
            default_image="x.sif",
            agent_user="ignore",
            verifier_user="honor",
        )

    with pytest.raises(EnvSetupError, match="unless its options set fakeroot: true"):
        await service().check_tasks("train", ["root"])  # the agents' options, which give no root
    spelled = service(extra_start_args=["--fakeroot"])
    if fakeroot:
        await spelled.check_tasks("train", ["root"])
        await service(fakeroot=True).check_tasks("train", ["root"])
    else:
        with pytest.raises(EnvSetupError, match="cannot delete what the other users of a --fakeroot sandbox write"):
            await spelled.check_tasks("train", ["root"])
        with pytest.raises(EnvSetupError, match="separate verifier's sandbox .* takes no fakeroot option"):
            await service(fakeroot=True).check_tasks("train", ["root"])


@pytest.mark.asyncio
async def test_each_sandbox_is_checked_for_a_network_with_its_own_options(make_task, monkeypatch):
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _hpc_sandbox(fakeroot=True))
    verifier_offline = make_task("verifier-offline", task_toml=ROOT_VERIFIER + 'network_mode = "no-network"\n')
    agents_offline = make_task(
        "agents-offline", task_toml=ROOT_VERIFIER + '[environment]\nnetwork_mode = "no-network"\n'
    )

    def service(task: Path, provider_options: dict, verifier_options: dict) -> HarborEnvironmentService:
        return HarborEnvironmentService(
            task_dirs=[task],
            provider="singularity",
            provider_options=provider_options,
            verifier_provider_options=verifier_options,
            default_image="x.sif",
            agent_user="ignore",
            verifier_user="honor",
        )

    await service(verifier_offline, {}, {"fakeroot": True, "allow_network_none": True}).check_tasks(
        "train", ["verifier-offline"]
    )
    with pytest.raises(EnvSetupError, match="1 of the 1 tasks of split 'train' run without a network"):
        await service(
            verifier_offline, {"allow_network_none": True}, {"fakeroot": True, "allow_network_none": False}
        ).check_tasks("train", ["verifier-offline"])
    with pytest.raises(EnvSetupError, match="run without a network .*allow_network_none: true"):
        await service(agents_offline, {}, {"fakeroot": True, "allow_network_none": True}).check_tasks(
            "train", ["agents-offline"]
        )


@pytest.mark.asyncio
async def test_verifier_options_that_would_start_no_sandbox_are_refused_only_for_separate_verifiers(
    make_task, monkeypatch
):
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _hpc_sandbox(fakeroot=True))
    monkeypatch.delenv("HPC_SANDBOX_ROOTFS_LAYOUT", raising=False)

    def service(task: Path) -> HarborEnvironmentService:
        return HarborEnvironmentService(
            task_dirs=[task],
            provider="singularity",
            verifier_provider_options={"rootfs_layout": "overlay"},
            default_image="x.sif",
        )

    with pytest.raises(EnvSetupError, match="would start no separate verifier's sandbox .*not 'overlay'"):
        await service(make_task("separate", task_toml=SEPARATE_PLAIN)).check_tasks("train", ["separate"])
    await service(make_task("shared")).check_tasks("train", ["shared"])  # its tests run in the agents' sandbox


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"verifier_user": "obey"}, "verifier_user must be one of"),
        ({"verifier_provider_options": ["fakeroot"]}, "verifier_provider_options must be a mapping"),
        ({"verifier_provider_options": {"fakeroot": True}, "provider": ProcessProvider()}, "need a provider named by"),
    ],
)
def test_service_options_are_checked(make_task, options, message):
    with pytest.raises(ValueError, match=message):
        HarborEnvironmentService(task_dirs=[make_task("t")], **options)


# ------------------------------------------------------------------------------------------------------ grading as root


@pytest.mark.asyncio
async def test_a_root_verifier_grades_in_a_sandbox_of_the_verifier_options_as_docker_would(
    make_task, tmp_path, rootful
):
    task = make_task("root", task_toml=ROOT_VERIFIER, test_sh=CHECK_AS_DOCKER)
    service = _service(
        [task], rootful, tmp_path / "sandboxes", verifier_user="honor", verifier_provider_options={"mapped": 70000}
    )
    await service.check_tasks("train", ["root"])
    session = await service.open(await service.get_task("root"), rollout_id="r")
    try:
        assert not (await session.call_tool("bash", {"command": LEAVE_PRIVATE_FILES})).is_error

        outcome = await session.grade()
    finally:
        await session.close()

    assert (outcome.status, outcome.rewards) == ("scored", {"reward": 1.0}), outcome.metadata.get("test_output")
    assert outcome.metadata["verifier_user"] == "root"
    assert outcome.metadata["ignored_users"] == {"agent": "65534"}
    agents, verifiers = service.provider, service.verifier_provider
    assert agents is not verifiers
    assert agents.options["mapped"] == 65536 and verifiers.options["mapped"] == 70000
    assert verifiers.options["root"] == agents.options["root"]  # laid over the agents' options
    assert [spec.labels.get("platoon.role") for spec, _ in agents.started] == [None]
    ((spec, verifier),) = verifiers.started
    assert spec.labels["platoon.role"] == "verifier" and not verifier.root.exists()


@pytest.mark.asyncio
async def test_a_verifier_that_does_not_run_as_root_grades_as_before(make_task, tmp_path, rootful):
    task = make_task("root", task_toml=ROOT_VERIFIER, test_sh=CHECK_AS_DOCKER)
    # Its sandboxes' start makes the workdir under a host umask of 027, as a rootless one does.
    service = _service(
        [task], rootful, tmp_path / "sandboxes", provider_options={"workdir_mode": 0o750}, verifier_user="ignore"
    )
    session = await service.open(await service.get_task("root"), rollout_id="r")
    try:
        assert not (await session.call_tool("bash", {"command": LEAVE_PRIVATE_FILES})).is_error
        outcome = await session.grade()
    finally:
        await session.close()

    assert outcome.rewards == {"reward": 0.0}
    output = outcome.metadata["test_output"]
    assert "app=750 answer=600 " in output and "result=600" in output  # as the agent and the runtime left them
    assert "verifier_user" not in outcome.metadata
    assert service.verifier_provider is service.provider  # no verifier options: the agents' provider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verifier_options", "problem"),
    [
        ({"uid": 1000}, "its sandbox runs its commands as uid 1000, not as root"),
        ({"mapped": 0}, "maps only the uids 0 into its user namespace, so they could not run anything as another"),
        ({"mapped": NOBODY - 1}, f"maps only the uids 0, 1-{NOBODY - 1} into its user namespace"),
    ],
)
async def test_a_verifier_sandbox_that_cannot_run_the_tests_as_root_is_an_error(
    make_task, tmp_path, rootful, verifier_options, problem
):
    task = make_task("root", task_toml=ROOT_VERIFIER, test_sh=CHECK_AS_DOCKER)
    service = _service(
        [task], rootful, tmp_path / "sandboxes", verifier_user="honor", verifier_provider_options=verifier_options
    )
    session = await service.open(await service.get_task("root"), rollout_id="r")
    try:
        outcome = await session.grade()
    finally:
        await session.close()

    assert (outcome.status, outcome.failure_kind) == ("error", "verifier_start")
    assert outcome.metadata["setup_failure_kind"] == "env_config"
    assert problem in outcome.reason and "runs its verifier as root" in outcome.reason
    ((_, verifier),) = service.verifier_provider.started
    assert not verifier.root.exists()  # stopped


# Passes only where the image's private workdir stayed private, and what platoon made has Docker's modes.
CHECK_PRIVATE_WORKDIR = f"""
mode() {{ stat -c %a "{ROOT}$1"; }}
found="app=$(mode /app) answer=$(mode /app/answer.txt) logs=$(mode /logs) verifier=$(mode /logs/verifier)"
echo "$found"
if [ "$found" = "app=700 answer=644 logs=755 verifier=755" ]; then echo 1; else echo 0; fi \\
  > {ROOT}/logs/verifier/reward.txt
"""


@pytest.mark.asyncio
async def test_a_root_verifier_leaves_the_images_own_workdir_as_it_is(make_task, tmp_path, rootful):
    """A workdir the image has keeps its mode, as Docker keeps it: an image that keeps /root to root (700) keeps the
    files the tests write there from the users they switch to.  What platoon makes gets Docker's modes."""

    task = make_task("root", task_toml=ROOT_VERIFIER, test_sh=CHECK_PRIVATE_WORKDIR)
    service = _service(
        [task],
        rootful,
        tmp_path / "sandboxes",
        verifier_user="honor",
        verifier_provider_options={"workdir_mode": 0o700},
    )
    previous = os.umask(0o027)  # a host umask that would keep other users out of what platoon makes
    try:
        session = await service.open(await service.get_task("root"), rollout_id="r")
        try:
            assert not (await session.call_tool("bash", {"command": LEAVE_PRIVATE_FILES})).is_error
            outcome = await session.grade()
        finally:
            await session.close()
    finally:
        os.umask(previous)

    assert (outcome.status, outcome.rewards) == ("scored", {"reward": 1.0}), outcome.metadata.get("test_output")


@pytest.mark.asyncio
async def test_a_verifier_whose_image_keeps_nobody_from_its_shell_is_an_error(make_task, tmp_path, rootful):
    task = make_task("root", task_toml=ROOT_VERIFIER, test_sh=CHECK_AS_DOCKER)
    denied = (127, "", "setpriv: failed to execute /bin/sh: Permission denied\n")
    service = _service(
        [task], rootful, tmp_path / "sandboxes", verifier_user="honor", verifier_provider_options={"nobody": denied}
    )
    session = await service.open(await service.get_task("root"), rollout_id="r")
    try:
        outcome = await session.grade()
    finally:
        await session.close()

    assert (outcome.status, outcome.failure_kind) == ("error", "verifier_start")
    assert "cannot run the image's /bin/sh as nobody (65534): setpriv: failed to execute /bin/sh" in outcome.reason
    assert "give the verifier an image whose programs other users may run" in outcome.reason  # not one of the map
    ((_, verifier),) = service.verifier_provider.started
    assert not verifier.root.exists()  # stopped


@pytest.mark.asyncio
async def test_deferred_workdirs_get_the_umask_they_are_given(tmp_path):
    """A root verifier makes its image's deferred WORKDIR directories with Docker's umask, as a build makes them;
    others keep the host's."""

    sandbox = await ProcessProvider(tmp_path / "sandboxes").start(SandboxSpec(image="local", workdir="/app"))
    previous = os.umask(0o027)
    try:
        for umask, target, mode in (("022", "/app/docker/deep", 0o755), (None, "/app/host/deep", 0o750)):
            image = SessionImage(image="local", env={}, workdir="/app", files=(FileOp("mkdir", target),))
            await apply_image_files(sandbox, image, workdir="/app", umask=umask)
            for path in (target, target.rsplit("/", 1)[0]):
                assert sandbox.host_path(path).stat().st_mode & 0o777 == mode, (path, umask)
    finally:
        os.umask(previous)
        await sandbox.stop()


# ---------------------------------------------------------------------------- the images and tasks of a fetched dataset

TASK_DOCKERFILE = "FROM python:3.12-slim\nRUN echo built > /built.txt\nWORKDIR /app\n"
VERIFIER_DOCKERFILE = "FROM python:3.12-slim\nRUN echo grader > /grader.txt\n"


def _fetched(tmp_path: Path, rootful: str, verifier_user: str = "honor", **options: Any) -> HarborEnvironmentService:
    """A service on a fetched dataset of two tasks: ``root``, whose separate verifier, built from tests/Dockerfile,
    runs as root, and ``plain``; both build their image from environment/Dockerfile.  The Harbor cache, with the
    dataset's default image map, is the site's (``use_site_cache``)."""

    dataset = tmp_path / "dataset"
    for name, toml in (("root", ROOT_VERIFIER), ("plain", SEPARATE_PLAIN)):
        files = {
            "task.toml": toml,
            "instruction.md": "Do the task.\n",
            "tests/test.sh": "echo 1 > /logs/verifier/reward.txt\n",
            "environment/Dockerfile": TASK_DOCKERFILE,
        }
        if name == "root":
            files["tests/Dockerfile"] = VERIFIER_DOCKERFILE
        for relative, text in files.items():
            (dataset / name / relative).parent.mkdir(parents=True, exist_ok=True)
            (dataset / name / relative).write_text(text)
    return HarborEnvironmentService(
        dataset=str(dataset),
        lock=str(tmp_path / "dataset.lock.json"),
        provider=rootful,
        provider_options={"root": str(tmp_path / "sandboxes")},
        agent_user="ignore",
        verifier_user=verifier_user,
        **options,
    )


@pytest.mark.asyncio
async def test_root_verifiers_on_images_of_the_oci_builder_before_0_3_0_are_refused_when_listed(
    tmp_path, rootful, monkeypatch
):
    """Their receipts record no builder revision: on a host whose umask is 027 their files keep out the users the
    tests switch to, so the run stops, naming the build that makes them again; the task image, which the agents run,
    does not matter, nor does an image another builder made."""

    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "node"))
    use_site_cache(monkeypatch, tmp_path / "cache")
    service = _fetched(tmp_path, rootful, refuse={"agent_user": False})
    image_map = service.task_images.path
    assert image_map is not None
    listed = await service.list_tasks("train")
    lock = read_lock(tmp_path / "dataset.lock.json")
    verifier_key, task_key = lock.get("root").verifier_image["key"], lock.get("root").image["key"]
    assert sorted(listed) == ["plain", "root"] and verifier_key != task_key

    def mapped(verifier: ImageEntry) -> None:
        update_image_map(image_map, {verifier_key: verifier, task_key: ImageEntry(image="local", builder="oci")})

    mapped(ImageEntry(image="local", builder="oci", receipt={"builder_version": "4.4.1"}))
    assert service.root_images_closed() == {verifier_key: ["root"]}
    with pytest.raises(EnvSetupError, match="1 of the 2 tasks of split 'train' run their verifier as root") as refused:
        await service.check_tasks("train", listed)
    assert refused.value.failure_kind == "env_config"
    assert f"`platoon harbor images build CONFIG --key {verifier_key}`" in str(refused.value)
    assert task_key not in str(refused.value)
    await service.check_tasks("train", ["plain"])  # what the run keeps once it leaves it out

    mapped(ImageEntry(image="local", builder="oci", receipt={"builder_revision": 1}))  # built again
    await service.check_tasks("train", listed)
    mapped(ImageEntry(image="local", builder="def"))
    await service.check_tasks("train", listed)
    mapped(ImageEntry(image="local", builder="oci"))
    ignored = _fetched(tmp_path, rootful, "ignore", refuse={"agent_user": False})
    assert ignored.root_images_closed() == {}
    await ignored.check_tasks("train", await ignored.list_tasks("train"))


@pytest.mark.asyncio
async def test_honor_warns_of_root_verifiers_the_fetch_left_out(tmp_path, rootful, monkeypatch, caplog):
    """The fetch's refuse.agent_user, on by default, refuses every task that names a user: honor never sees those
    tasks, and says so."""

    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "node"))
    use_site_cache(monkeypatch, tmp_path / "cache")
    service = _fetched(tmp_path, rootful)
    image_map = service.task_images.path
    assert image_map is not None
    assert await service.list_tasks("train") == ["plain"]
    lock = read_lock(tmp_path / "dataset.lock.json")
    assert lock.get("root").refused.startswith("agent_user:")
    update_image_map(image_map, {lock.get("plain").image["key"]: ImageEntry(image="local", builder="def")})

    with caplog.at_level(logging.WARNING, logger="platoon.envs.harbor.service"):
        await service.check_tasks("train", ["plain"])
        await service.check_tasks("eval", ["plain"])

    (warning,) = [record.getMessage() for record in caplog.records if "verifier_user: honor" in record.getMessage()]
    assert "runs none of the 1 task(s)" in warning and "(root)" in warning
    assert "refuse: {agent_user: false}" in warning


# ----------------------------------------------------------------------------------------------------- the check itself


class _Answering:
    """A sandbox that answers every command with one result."""

    id = "answering"

    def __init__(self, result: ExecResult) -> None:
        self.result = result
        self.commands: list[Any] = []

    async def exec(self, cmd: Any, **options: Any) -> ExecResult:
        self.commands.append((cmd, options))
        return self.result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stdout", "problem"),
    [
        ("uid 0\nuid_map 0 4294967295\ngid_map 0 4294967295\n", None),  # the host's own namespace
        ("uid 0\nuid_map 0 1\nuid_map 1 65536\ngid_map 0 1\ngid_map 1 65536\n", None),  # --fakeroot
        ("uid 0\nuid_map 1 65536\nuid_map 0 1\ngid_map 0 65535\n", None),  # in any order
        ("uid 0\nuid_map 0 65535\ngid_map 0 65535\n", None),
        ("uid 0\nuid_map 0 65534\ngid_map 0 65535\n", "maps only the uids 0-65533"),
        ("uid 0\nuid_map 0 1\ngid_map 0 1\n", "maps only the uids 0 "),
        ("uid 0\nuid_map 0 1\nuid_map 2 65536\ngid_map 0 70000\n", "maps only the uids 0, 2-65537"),
        ("uid 0\nuid_map 0 70000\ngid_map 0 1\n", "maps only the gids 0 "),
        ("uid 0\n", "maps only the uids none"),
        ("uid 1000\nuid_map 1000 1\n", "runs its commands as uid 1000, not as root"),
        ("", "cannot tell which user its commands run as"),
    ],
)
async def test_the_check_wants_root_with_every_user_up_to_nobody(stdout: str, problem: str | None):
    sandbox = _Answering(ExecResult(0, stdout, ""))

    said = await root_problem(sandbox)

    assert (said is None) is (problem is None), said
    if problem is not None:
        assert problem in said
    ((cmd, options),) = sandbox.commands
    assert cmd[:2] == ["/bin/sh", "-c"] and options["cwd"] == "/"


@pytest.mark.asyncio
async def test_the_check_reads_the_users_inside_a_real_shell(tmp_path):
    """The check's script on this host, through the process provider: the effective uid and the user namespace's
    maps of whoever runs the tests."""

    sandbox = await ProcessProvider(tmp_path).start(SandboxSpec(image="local"))
    try:
        said = await root_problem(sandbox)
    finally:
        await sandbox.stop()

    if os.geteuid() != 0:
        assert said == f"runs its commands as uid {os.geteuid()}, not as root"
    else:
        assert said is None or "maps only" in said


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "problem"),
    [
        (ExecResult(0, "ran\n", ""), None),
        (ExecResult(127, "", "setpriv: failed to execute /bin/sh: Permission denied\n"), "setpriv: failed to execute"),
        (ExecResult(1, "", "su: can't execute '/bin/sh': Permission denied\n"), "su: can't execute '/bin/sh'"),
        # The shell of an image of the oci builder before 0.3.0, run as nobody: its libraries' directory is closed.
        (
            ExecResult(
                127,
                "",
                "/bin/sh: error while loading shared libraries: libc.so.6: cannot open shared object file: No "
                "such file or directory\n",
            ),
            "/bin/sh: error while loading shared libraries: libc.so.6",
        ),
        (ExecResult(0, "", "platoon: the image has neither setpriv nor su\n"), None),  # nothing tells
        (ExecResult(1, "", "setpriv: setresuid failed: Operation not permitted\n"), None),  # not the image's files
        (ExecResult(-1, "", "Permission denied", timed_out=True), None),
    ],
)
async def test_the_image_must_let_nobody_run_its_shell(result: ExecResult, problem: str | None):
    sandbox = _Answering(result)

    said = await nobody_problem(sandbox)

    assert (said is None) is (problem is None), said
    if problem is not None:
        assert said.startswith(f"cannot run the image's /bin/sh as nobody ({NOBODY}): {problem}")
    ((cmd, options),) = sandbox.commands
    assert cmd[:2] == ["/bin/sh", "-c"] and cmd[3] == "platoon-nobody-check" and options["cwd"] == "/"
    assert f"setpriv --reuid={NOBODY} --regid={NOBODY} --clear-groups /bin/sh" in cmd[2]
