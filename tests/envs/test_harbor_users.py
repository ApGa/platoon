"""Harbor tasks whose users are honored (``agent_user: honor``, ``verifier_user: honor``): who each phase runs as
(Harbor's rule), how a sandbox's view as a user runs its commands (the switch's command forms, its environment, its
uploads), the session that hands the agents their view while grading keeps root's, and what is refused before any
rollout.

The provider here is the ``process`` provider standing in for one that gives root and other users: it answers the
checks of a started sandbox and the image's accounts, and runs a switched command as the host's user, recording whom it
switched to (:class:`SwitchingProvider`).  Real switches run in ``test_harbor_users_singularity.py`` and in
``tests/integration/test_harbor_users_docker.py``."""

from __future__ import annotations

import json
import logging
import os
import stat
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from site_cache import use_site_cache

import platoon.sandbox
from platoon.envs.harbor import HarborEnvironmentService
from platoon.envs.harbor.images import ImageEntry, update_image_map
from platoon.envs.harbor.lock import read_lock
from platoon.envs.harbor.owners import PulledImage, record_pulled
from platoon.envs.harbor.task import HarborTask, HarborTaskError
from platoon.envs.harbor.users import ROOT, copy_owner, hook_users, parse_user, phase_users
from platoon.envs.harbor.views import SandboxView, root_view, uploaded_paths, user_views
from platoon.envs.service import EnvNotPrepared, EnvSetupError
from platoon.sandbox import ExecResult, SandboxSpec
from platoon.sandbox.base import DownloadReport, SandboxSnapshot
from platoon.sandbox.process import ProcessProvider, ProcessSandbox

ROOT_DIR = "${PLATOON_SANDBOX_ROOT:-}"
PASSWD = (
    "root:x:0:0:root:/root:/bin/bash\n"
    "nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n"
    "agent:x:1000:1000::/srv/agent:/bin/sh\n"
)
GROUP = "root:x:0:\nstaff:x:50:agent\nagent:x:1000:\nnogroup:x:65534:\n"
UMASK = ["/bin/sh", "-c", 'umask 022 && exec "$@"']


# ---------------------------------------------------------------------------------------------- the users tasks name


@pytest.mark.parametrize(
    ("spec", "uid", "gid", "name", "home", "groups", "login"),
    [
        ("agent", 1000, 1000, "agent", "/srv/agent", (50,), "agent"),
        ("1000", 1000, 1000, "agent", "/srv/agent", (50,), "agent"),  # a uid /etc/passwd lists is that user
        ("agent:staff", 1000, 50, "agent", "/srv/agent", (), None),  # a group named: none of /etc/group's, as Docker
        ("1000:50", 1000, 50, "agent", "/srv/agent", (), None),
        ("agent:agent", 1000, 1000, "agent", "/srv/agent", (), None),  # su would add them
        ("4242", 4242, 0, None, "/", (), None),  # as Docker: a uid it does not list, with group 0 and home /
        ("4242:4242", 4242, 4242, None, "/", (), None),
        ("root", 0, 0, "root", "/root", (), "root"),
        ("nobody", 65534, 65534, "nobody", "/nonexistent", (), "nobody"),
    ],
)
def test_users_resolve_as_docker_resolves_its_user_option(spec, uid, gid, name, home, groups, login):
    user = parse_user(spec, PASSWD, GROUP)

    assert (user.uid, user.gid, user.name, user.home, user.groups, user.login) == (uid, gid, name, home, groups, login)


@pytest.mark.parametrize(
    ("spec", "message"),
    [("ghost", "has no user 'ghost'"), ("agent:ghosts", "has no group 'ghosts'"), (":50", "names no user")],
)
def test_a_name_the_image_lacks_is_refused(spec, message):
    with pytest.raises(ValueError, match=message):
        parse_user(spec, PASSWD, GROUP)


@pytest.mark.parametrize(
    ("spec", "owner"),
    [
        ("agent", "1000:1000"),
        ("worker", "1001:50"),  # a name: the group /etc/passwd gives it
        ("1001", "1001:1001"),  # a uid: the gid of its own number, listed or not
        ("4242", "4242:4242"),
        ("worker:agent", "1001:1000"),
        ("4242:staff", "4242:50"),
    ],
)
def test_a_copys_owner_resolves_as_buildkit_resolves_chown(spec, owner):
    assert copy_owner(spec, PASSWD + "worker:x:1001:50::/srv/worker:/bin/sh\n", GROUP) == owner


def test_each_phase_runs_as_its_user_else_the_images_else_the_sandboxs(make_task):
    """Harbor's rule: the agents as [agent].user, else the image's USER; the tests as [verifier].user, else the image's
    USER; None, the sandbox's own user, where a phase is not honored."""

    named = make_task("named", task_toml='[agent]\nuser = "agent"\n[verifier]\nuser = "root"\n')
    plain = make_task("plain")
    hooks = make_task("hooks", task_toml='[[verifier.collect]]\ncommand = "sync"\nuser = "tests"\n')
    separate = make_task("separate", task_toml='[verifier]\nenvironment_mode = "separate"\nuser = "root"\n')

    task = HarborTask.load(named, agent_user="honor")
    assert phase_users(task, image_user="65534") == {"agents": "agent", "tests": "root", "default": "65534"}
    task = HarborTask.load(plain, agent_user="honor")
    assert phase_users(task, image_user="65534") == {"agents": "65534", "tests": "65534", "default": "65534"}
    assert phase_users(task, image_user=None) == {"agents": None, "tests": None, "default": None}
    task = HarborTask.load(named, agent_user="honor", verifier_user="ignore")
    assert phase_users(task, image_user=None) == {"agents": "agent", "tests": None, "default": None}
    task = HarborTask.load(hooks, verifier_user="honor")  # a hook's user, even one named as a phase, is its own
    assert phase_users(task, image_user=None) == {"agents": None, "tests": None, "default": None}
    assert (hook_users(task), hook_users(task, verifier=True)) == (["tests"], [])
    task = HarborTask.load(separate, agent_user="ignore", verifier_user="honor")
    # Nothing runs as the image's USER in a separate verifier's sandbox whose tests run as root.
    assert phase_users(task, image_user="65534", verifier=True) == {"agents": None, "tests": "root", "default": None}


def test_users_on_a_stack_are_refused_until_stacks_honor_them(make_task):
    """Users that would run in the stack itself are refused: under agent_user: honor every stack's agents, which run as
    [agent].user, else main's USER.  A separate verifier's sandbox is a single one, which honors its user."""

    compose = "services:\n  main: {}\n  db:\n    image: redis:7-alpine\n    command: [redis-server]\n"
    stacks = {}
    for name, toml in (
        ("named", '[agent]\nuser = "agent"\n'),
        ("plain", ""),
        ("separate", 'artifacts = ["/app/a"]\n[verifier]\nenvironment_mode = "separate"\nuser = "root"\n'),
    ):
        stacks[name] = make_task(name, task_toml=toml)
        (stacks[name] / "environment" / "docker-compose.yaml").write_text(compose)

    for name in ("named", "plain"):
        with pytest.raises(HarborTaskError, match="multi-container stack .*stacks that honor users follow"):
            HarborTask.load(stacks[name], agent_user="honor")
    assert HarborTask.load(stacks["named"], agent_user="ignore").ignored_users == {"agent": "agent"}
    separate = HarborTask.load(stacks["separate"], verifier_user="honor")
    assert separate.verifier_root and separate.honors_users_in("verifier") and not separate.honors_users_in("agents")
    with pytest.raises(HarborTaskError, match="multi-container stack"):
        HarborTask.load(stacks["separate"], agent_user="honor")


# ------------------------------------------------------------------------------------- a sandbox's view as one user


class Recording:
    """A sandbox that records the commands it is given, and answers each by a word in it (``answers``)."""

    id = "recording"
    routes: Mapping[str, str] = {}

    def __init__(self, answers: dict[str, ExecResult] | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.uploads: list[tuple[Path, str]] = []
        self.answers = {
            "platoon-accounts": ExecResult(0, f"{PASSWD}\nplatoon-group-file\n{GROUP}", ""),
            **(answers or {}),
        }

    async def exec(
        self,
        cmd: str | list[str],
        *,
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> ExecResult:
        argv = ["bash", "-c", cmd] if isinstance(cmd, str) else list(cmd)
        self.calls.append((argv, {"env": env, "cwd": cwd}))
        return next((result for word, result in self.answers.items() if word in argv), ExecResult(0, "", ""))

    async def upload(self, src: str | Path, dst: str) -> None:
        self.uploads.append((Path(src), dst))

    async def download(self, src: str, dst: str | Path) -> DownloadReport | None:
        return None

    async def snapshot(self) -> SandboxSnapshot:
        raise NotImplementedError

    async def kill_processes(self) -> None:
        pass

    async def stop(self) -> None:
        pass


class Native(Recording):
    """A sandbox that runs a command as a user itself, as a docker sandbox does (``exec_as``)."""

    async def exec_as(self, user: str, cmd: str | list[str], **options: Any) -> ExecResult:
        result = await self.exec(cmd, **options)
        self.calls[-1][1]["user"] = user
        return result


def _probe(tool: str) -> dict[str, ExecResult]:
    return {"platoon-switch": ExecResult(0, f"{tool}\n", "")}


@pytest.mark.asyncio
async def test_root_runs_platoons_commands_under_dockers_umask():
    sandbox, native = Recording(), Native()

    await root_view(sandbox).exec(["id"])
    await root_view(native).exec("echo $HOME", env={"A": "1"})

    assert sandbox.calls == [([*UMASK, "platoon-root", "id"], {"env": None, "cwd": None})]
    assert native.calls == [
        ([*UMASK, "platoon-root", "bash", "-c", "echo $HOME"], {"env": {"A": "1"}, "cwd": None, "user": "0"})
    ]
    assert root_view(root_view(sandbox)).sandbox is sandbox  # a view of a view is a view of the sandbox


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "prefix"),
    [
        ("setpriv", ["setpriv", "--reuid=1000", "--regid=1000", "--init-groups", "--"]),
        ("su", ["su", "-m", "-s", "/bin/sh", "-c", 'exec "$@"', "agent", "--", "platoon-user"]),
        ("python /opt/tools/bin/python3", ["/opt/tools/bin/python3", "-I", "-c"]),
    ],
)
async def test_a_users_commands_switch_with_the_first_tool_the_image_has(tool, prefix):
    sandbox = Recording(_probe(tool))

    views = await user_views(sandbox, ["agent"], pythons=["/opt/tools/bin/python3"])
    await views["agent"].exec(["id", "-u"], cwd="/app")

    probe = next(argv for argv, _ in sandbox.calls if "platoon-switch" in argv)
    assert probe[probe.index("platoon-switch") + 1 :] == ["1000", "1000", "agent", "python3", "/opt/tools/bin/python3"]
    argv, options = sandbox.calls[-1]
    assert argv[: len(UMASK) + 1] == [*UMASK, "platoon-user"] and argv[len(UMASK) + 1 :][: len(prefix)] == prefix
    assert argv[-2:] == ["id", "-u"] and options == {
        "env": {"HOME": "/srv/agent", "USER": "agent", "LOGNAME": "agent"},
        "cwd": "/app",
    }
    if tool.startswith("python"):  # uid, gid and the groups /etc/group adds the user to, then the command
        assert argv[-5:-2] == ["1000", "1000", "50"]


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", ["agent:staff", "agent:agent", "1000:1000"])
async def test_a_user_named_with_a_group_switches_without_the_groups_etc_group_adds(spec: str):
    """Docker gives a --user that names a group that group alone, none of those /etc/group adds the user to, which
    setpriv's --init-groups and su would give."""

    setpriv, python = Recording(_probe("setpriv")), Recording(_probe("python /usr/bin/python3"))
    for sandbox in (setpriv, python):
        await (await user_views(sandbox, [spec]))[spec].exec(["id"])

    probe = next(argv for argv, _ in setpriv.calls if "platoon-switch" in argv)
    assert probe[probe.index("platoon-switch") + 3] == ""  # no name for su
    assert "--clear-groups" in setpriv.calls[-1][0] and "--init-groups" not in setpriv.calls[-1][0]
    assert python.calls[-1][0][-2] == ""  # the groups the switch sets


@pytest.mark.asyncio
async def test_a_uid_the_image_does_not_list_switches_without_its_groups_or_su():
    sandbox = Recording(_probe("setpriv"))

    view = (await user_views(sandbox, ["4242"], spec_env={"HOME": "/work"}))["4242"]
    await view.exec(["true"])

    probe = next(argv for argv, _ in sandbox.calls if "platoon-switch" in argv)
    assert probe[probe.index("platoon-switch") + 1 : probe.index("platoon-switch") + 4] == ["4242", "0", ""]
    argv, options = sandbox.calls[-1]
    assert "--clear-groups" in argv and options["env"] is None  # the spec's HOME stays, and no name to set


@pytest.mark.asyncio
async def test_an_image_without_a_way_to_switch_users_is_refused():
    with pytest.raises(ValueError, match="no way to switch to the user agent .* no setpriv .* no su, and no Python"):
        await user_views(Recording(), ["agent"])


@pytest.mark.asyncio
async def test_a_sandbox_that_runs_commands_as_users_switches_with_its_own_exec_as():
    sandbox = Native({"platoon-own-user": ExecResult(0, "1000:1000\n", "")})

    views = await user_views(sandbox, ["agent", None, "root"])
    await views["agent"].exec(["id"])
    await views[None].exec(["id"])

    assert not any("platoon-switch" in argv for argv, _ in sandbox.calls)  # no tool of the image needed
    (agent_call, own_call) = sandbox.calls[-2:]
    assert agent_call[1]["user"] == "agent" and agent_call[1]["env"] == {
        "HOME": "/srv/agent",
        "USER": "agent",
        "LOGNAME": "agent",
    }
    # The container's own user runs as a plain exec runs, with its supplementary groups, which an explicit
    # --user uid:gid would drop; what it uploads is its uid's and gid's.
    assert "user" not in own_call[1] and views[None].user.uid == 1000
    assert views["root"].user is ROOT


def test_an_upload_writes_its_destination_and_the_entries_of_its_source(tmp_path: Path):
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "sub" / "file").write_text("x")
    (tree / "link").symlink_to("sub")

    assert uploaded_paths(tree / "sub" / "file", "/srv/f") == [("/srv/f", tree / "sub" / "file")]
    assert uploaded_paths(tree, "/srv/d") == [  # a directory's copy gives its destination the source's mode too
        ("/srv/d", tree),
        ("/srv/d/link", tree / "link"),
        ("/srv/d/sub", tree / "sub"),
        ("/srv/d/sub/file", tree / "sub" / "file"),
    ]


@pytest.mark.asyncio
async def test_what_a_user_uploads_is_the_users_and_what_root_uploads_opens_as_its_group_is(tmp_path: Path):
    tree = tmp_path / "tree"
    tree.mkdir()
    tree.chmod(0o750)
    (tree / "file").write_text("x")
    (tree / "file").chmod(0o640)
    (tree / "private").write_text("x")
    (tree / "private").chmod(0o600)
    (tree / "link").symlink_to("file")
    sandbox = Recording(_probe("setpriv"))
    agent = (await user_views(sandbox, ["agent"]))["agent"]

    await agent.upload(tree, "/srv/state")
    await root_view(sandbox).upload(tree / "file", "/tests/sub/file")

    assert sandbox.uploads == [(tree, "/srv/state"), (tree / "file", "/tests/sub/file")]
    commands = [argv[len(UMASK) + 1 :] for argv, _ in sandbox.calls if argv[: len(UMASK)] == UMASK]
    made = [command for command in commands if command[:1] == ["mkdir"]]  # as root, under Docker's umask
    assert made == [["mkdir", "-p", "--", "/srv/state"], ["mkdir", "-p", "--", "/tests/sub"]]
    owned = ["/srv/state", "/srv/state/file", "/srv/state/link", "/srv/state/private"]
    assert ["chown", "-h", "1000:1000", "--", *owned] in commands
    assert ["chmod", "644", "--", "/tests/sub/file"] in commands
    before = len(sandbox.calls)
    await root_view(sandbox).upload(tree, "/tests")
    # Every user gets what the group has, as under Docker's umask; a file the host keeps from its group stays so, and
    # a link is left alone.
    chmods = [argv[len(UMASK) + 1 :] for argv, _ in sandbox.calls[before:] if "chmod" in argv]
    assert chmods == [
        ["chmod", "755", "--", "/tests"],
        ["chmod", "644", "--", "/tests/file"],
        ["chmod", "600", "--", "/tests/private"],
    ]
    with pytest.raises(PermissionError, match="does not download"):
        await agent.download("/srv/state/file", tmp_path / "out")


@pytest.mark.asyncio
async def test_a_root_view_on_a_real_sandbox_writes_with_dockers_umask(tmp_path: Path):
    sandbox = await ProcessProvider(tmp_path / "sandboxes").start(SandboxSpec(image="local"))
    previous = os.umask(0o077)
    try:
        assert (await root_view(sandbox).exec(["/bin/sh", "-c", "umask"])).stdout.strip() == "0022"
        tree = tmp_path / "tests"
        tree.mkdir()
        tree.chmod(0o750)
        for name, mode in (("checked-out", 0o640), ("private", 0o600), ("program", 0o750)):
            (tree / name).write_text("x")
            (tree / name).chmod(mode)
        await root_view(sandbox).upload(tree, "/tests")
        uploaded = [sandbox.host_path("/tests"), *sandbox.host_path("/tests").iterdir()]
        modes = {path.name: stat.S_IMODE(path.stat().st_mode) for path in uploaded}
        # As a checkout under Docker's umask has them, but for the file that the host keeps from its group.
        assert modes == {"tests": 0o755, "checked-out": 0o644, "private": 0o600, "program": 0o755}
    finally:
        os.umask(previous)
        await sandbox.stop()


# --------------------------------------------------------------------------------------- sessions that honor users


class SwitchingProvider(ProcessProvider):
    """The process provider, standing in for one whose sandboxes give root and other users (as Docker does): a
    sandbox answers the checks of root and of nobody, and the image's accounts, and runs a switched command as the
    host's user, recording whom it switched to (``switched``) and the owners it was asked to give (``given``)."""

    loses_file_owners = False

    def __init__(self, root: str, tool: str = "setpriv") -> None:
        super().__init__(root)
        self.options = {"root": root, "tool": tool}
        self.started: list[tuple[SandboxSpec, ProcessSandbox]] = []
        self.switched: list[tuple[str, list[str], dict[str, str]]] = []
        self.given: list[list[str]] = []

    @classmethod
    def root_refusal(cls, options: Mapping[str, object]) -> str | None:
        return None

    async def start(self, spec: SandboxSpec) -> ProcessSandbox:
        # The process provider maps no "/": a sandbox that starts there starts in /tmp, which it maps, making nothing.
        sandbox = await super().start(replace(spec, workdir="/tmp") if spec.workdir == "/" else spec)
        real = sandbox.exec

        async def exec(cmd, **options):
            argv = ["bash", "-c", cmd] if isinstance(cmd, str) else list(cmd)
            if "platoon-root-check" in argv:
                return ExecResult(0, "uid 0\nuid_map 0 1\nuid_map 1 65536\ngid_map 0 1\ngid_map 1 65536\n", "")
            if "platoon-nobody-check" in argv:
                return ExecResult(0, "ran\n", "")
            if "platoon-accounts" in argv:
                return ExecResult(0, f"{PASSWD}\nplatoon-group-file\n{GROUP}", "")
            if "platoon-switch" in argv:
                return ExecResult(0, f"{self.options['tool']}\n", "")
            if "chown" in argv:
                self.given.append(argv[argv.index("chown") :])
                return ExecResult(0, "", "")
            if "setpriv" in argv:
                start = argv.index("setpriv")
                rest = argv[start + 1 :]
                rest = rest[rest.index("--") + 1 :]
                self.switched.append((argv[start + 1], rest, dict(options.get("env") or {})))
                argv = [*argv[:start], *rest]
            return await real(argv, **options)

        setattr(sandbox, "exec", exec)  # its own exec, wrapped
        self.started.append((spec, sandbox))
        return sandbox


@pytest.fixture
def switching(monkeypatch) -> str:
    """The name ``switching-process``, which builds a :class:`SwitchingProvider`."""

    monkeypatch.setitem(platoon.sandbox._BUILTIN_PROVIDERS, "switching-process", f"{__name__}:SwitchingProvider")
    return "switching-process"


def _service(task_dirs: list[Path], provider: str, root: Path, **options: Any) -> HarborEnvironmentService:
    return HarborEnvironmentService(
        task_dirs=task_dirs, provider=provider, provider_options={"root": str(root)}, default_image="local", **options
    )


AGENT_TASK = '[agent]\nuser = "agent"\n'
_REVISED = {"builder_revision": 1}  # an image the oci builder of this platoon built
# 1 when the agents wrote out.txt (the oracle does), else 0.
CHECK_WRITTEN = f"[ -f {ROOT_DIR}/app/out.txt ] && echo 1 > {ROOT_DIR}/logs/verifier/reward.txt || \\\n"
CHECK_WRITTEN += f"echo 0 > {ROOT_DIR}/logs/verifier/reward.txt"


@pytest.mark.asyncio
async def test_the_agents_act_as_their_user_while_grading_acts_as_root(make_task, tmp_path, switching):
    """The env tools, the harness's sandbox and the oracle are the agents' view; setup, the kills and grading keep
    root's; the directories the agents publish in are every user's; the outcome records whom the phases ran as."""

    task = make_task("agent", task_toml=AGENT_TASK, test_sh=CHECK_WRITTEN)
    (task / "solution").mkdir()
    (task / "solution" / "solve.sh").write_text(f"echo solved > {ROOT_DIR}/app/out.txt\n")
    service = _service([task], switching, tmp_path / "sandboxes", agent_user="honor")
    await service.check_tasks("train", ["agent"])
    session = await service.open(await service.get_task("agent"), rollout_id="r")
    provider = service.provider
    assert isinstance(provider, SwitchingProvider)
    try:
        assert isinstance(session.sandbox, SandboxView) and session.sandbox.user.uid == 1000
        ((_, sandbox),) = provider.started
        for published in ("/logs/agent", "/logs/artifacts"):
            assert stat.S_IMODE(sandbox.host_path(published).stat().st_mode) == 0o777
        assert not (await session.call_tool("bash", {"command": "echo hi > note.txt"})).is_error
        edited = await session.call_tool("edit", {"path": "note.txt", "old_str": "hi", "new_str": "ho"})
        assert not edited.is_error and sandbox.host_path("/app/note.txt").read_text() == "ho\n"
        assert (await session.run_solution()).exit_code == 0
        switched = [(uid, argv) for uid, argv, _ in provider.switched]
        assert ("--reuid=1000", ["bash", "-c", "echo hi > note.txt"]) in switched  # the bash tool
        assert sum(argv[:2] == ["bash", "-c"] and "read" in argv for _, argv in switched) == 1  # the edit reads as it
        assert ("--reuid=1000", ["bash", "-c", "bash /solution/solve.sh"]) in switched  # the oracle
        assert all(env.get("HOME") == "/srv/agent" for _, _, env in provider.switched)
        assert ["chown", "-h", "1000:1000", "--", "/solution", "/solution/solve.sh"] in provider.given
        before = len(provider.switched)
        outcome = await session.grade()
    finally:
        await session.close()

    assert (outcome.status, outcome.rewards) == ("scored", {"reward": 1.0}), outcome.metadata.get("test_output")
    assert len(provider.switched) == before  # the tests ran as root: the task names no user for them, nor its image
    assert outcome.metadata["users"] == {"agent": "agent", "verifier": None}


@pytest.mark.asyncio
async def test_a_root_verifier_grades_in_the_sandbox_of_agents_that_run_as_their_user(make_task, tmp_path, switching):
    """Docker's umask; and the tests' files and log directory root's, the tests' files with the host's modes, as
    Harbor copies them, so that the agent's code the tests run as another user can neither read a test file that the
    host keeps from other users (as a checkout under a umask of 027 does) nor write a reward."""

    task = make_task(
        "shared-root",
        task_toml=AGENT_TASK + '[verifier]\nuser = "root"\n',
        test_sh=f'[ "$(umask)" = 0022 ] && [ "$(stat -c %a {ROOT_DIR}/logs/verifier)" = 755 ] && '
        f'[ "$(stat -c %a {ROOT_DIR}/tests/held-out.txt)" = 640 ] && echo 1 > {ROOT_DIR}/logs/verifier/reward.txt',
    )
    (task / "tests" / "held-out.txt").write_text("42\n")
    (task / "tests" / "held-out.txt").chmod(0o640)
    service = _service([task], switching, tmp_path / "sandboxes", agent_user="honor")

    session = await service.open(await service.get_task("shared-root"), rollout_id="r")
    try:
        outcome = await session.grade()
    finally:
        await session.close()

    assert (outcome.status, outcome.rewards) == ("scored", {"reward": 1.0}), outcome.metadata.get("test_output")
    assert outcome.metadata["verifier_user"] == "root"


@pytest.mark.asyncio
async def test_tests_that_run_as_a_user_get_their_files_and_write_their_log_directory(make_task, tmp_path, switching):
    task = make_task(
        "tests-as-nobody",
        task_toml=AGENT_TASK + '[verifier]\nuser = "nobody"\n',
        test_sh=f'[ "$(stat -c %a {ROOT_DIR}/logs/verifier)" = 777 ] && echo 1 > {ROOT_DIR}/logs/verifier/reward.txt',
    )
    service = _service([task], switching, tmp_path / "sandboxes", agent_user="honor")

    session = await service.open(await service.get_task("tests-as-nobody"), rollout_id="r")
    try:
        outcome = await session.grade()
    finally:
        await session.close()

    assert (outcome.status, outcome.rewards) == ("scored", {"reward": 1.0}), outcome.metadata.get("test_output")
    provider = service.provider
    assert isinstance(provider, SwitchingProvider)
    assert ["chown", "-h", "65534:65534", "--", "/tests", "/tests/test.sh"] in provider.given
    assert ("--reuid=65534", ["bash", "-c", "bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1"]) in [
        (uid, argv) for uid, argv, _ in provider.switched
    ]


@pytest.mark.asyncio
async def test_a_dataset_directory_cannot_tell_its_images_user_where_sandboxes_lose_the_owners(
    make_task, tmp_path, switching, monkeypatch
):
    """Where a provider loses the owners an image gives its files (hpc-sandbox's), only the image map's record tells
    the image's USER: a dataset directory, which has none, is refused rather than run its agents as root."""

    monkeypatch.setattr(SwitchingProvider, "loses_file_owners", True)
    task = make_task("img", task_toml='[environment]\ndocker_image = "example.org/team/app:1"\n')
    service = _service([task], switching, tmp_path / "sandboxes", agent_user="honor")

    with pytest.raises(EnvSetupError, match="its USER, and a dataset directory has no image map") as refused:
        await service.check_tasks("train", ["img"])
    assert refused.value.failure_kind == "env_config"
    with pytest.raises(EnvSetupError, match="a dataset directory has no image map"):
        await service.open(await service.get_task("img"), rollout_id="r")


@pytest.mark.asyncio
async def test_a_docker_images_user_runs_once_prepare_records_it_where_sandboxes_lose_the_owners(
    tmp_path, switching, monkeypatch
):
    """A fetched task's docker_image, whose USER such a sandbox cannot read, is an owner gap that platoon prepare fills
    with its pulled entry, until which the run stops; its agents then run as that USER, on the pinned image."""

    monkeypatch.setattr(SwitchingProvider, "loses_file_owners", True)
    use_site_cache(monkeypatch, tmp_path / "cache")
    dataset = tmp_path / "dataset"
    files = {"task.toml": '[environment]\ndocker_image = "example.org/team/app:1"\n', "instruction.md": "Do it.\n"}
    for relative, text in {**files, "tests/test.sh": "true\n"}.items():
        (dataset / "img" / relative).parent.mkdir(parents=True, exist_ok=True)
        (dataset / "img" / relative).write_text(text)

    def fetched() -> HarborEnvironmentService:
        return HarborEnvironmentService(
            dataset=str(dataset),
            lock=str(tmp_path / "dataset.lock.json"),
            provider=switching,
            provider_options={"root": str(tmp_path / "sandboxes")},
            agent_user="honor",
        )

    service = fetched()
    assert await service.list_tasks("train") == ["img"]
    reference, pinned = "docker://example.org/team/app:1", "docker://example.org/team/app@sha256:" + "e" * 64
    assert service.owner_gaps() == ({}, {reference: ["img"]})
    with pytest.raises(EnvNotPrepared, match=f"no pulled entry for its image {reference}"):
        await service.check_tasks("train", ["img"])
    assert service.task_images.path is not None
    record_pulled(
        [reference],
        image_map=service.task_images.path,
        out_dir=tmp_path / "images",
        pull=lambda _: PulledImage(pinned, "sha256:" + "e" * 64, "1000", []),
    )

    service = fetched()
    await service.check_tasks("train", await service.list_tasks("train"))
    session = await service.open(await service.get_task("img"), rollout_id="r")
    try:
        assert isinstance(session.sandbox, SandboxView) and session.sandbox.user.uid == 1000
        provider = service.provider
        assert isinstance(provider, SwitchingProvider)
        ((spec, _),) = provider.started
        assert spec.image == pinned
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_honor_warns_of_copies_whose_owner_a_lock_before_0_5_dropped(tmp_path, switching, monkeypatch, caplog):
    use_site_cache(monkeypatch, tmp_path / "cache")
    dataset = tmp_path / "dataset"
    files = {
        "task.toml": AGENT_TASK,
        "instruction.md": "Do it.\n",
        "tests/test.sh": "true\n",
        "environment/Dockerfile": "FROM python:3.12-slim\nRUN true\nCOPY --chown=agent:agent a.txt /app/a.txt\n",
        "environment/a.txt": "a\n",
    }
    dockerfiles = {
        "copy": files["environment/Dockerfile"],
        "workdir": "FROM python:3.12-slim\nRUN true\nUSER agent\nWORKDIR /app/work\n",  # where USER falls is lost
        "neither": "FROM python:3.12-slim\nRUN true\nWORKDIR /app/work\n",
    }
    for name, dockerfile in dockerfiles.items():
        for relative, text in {**files, "environment/Dockerfile": dockerfile}.items():
            (dataset / name / relative).parent.mkdir(parents=True, exist_ok=True)
            (dataset / name / relative).write_text(text)
    lock_path = tmp_path / "dataset.lock.json"

    def service(**options: Any) -> HarborEnvironmentService:
        root = {"root": str(tmp_path / "sandboxes")}
        return HarborEnvironmentService(
            lock=str(lock_path), provider=switching, provider_options=root, agent_user="honor", **options
        )

    fetched = service(dataset=str(dataset), refuse={"agent_user": False})  # the task names a user
    await fetched.list_tasks("train")
    old = json.loads(lock_path.read_text())
    lock_path.write_text(json.dumps({**old, "compose_rule": ""}))  # as platoon 0.4 wrote it
    lock = read_lock(lock_path)
    built = ImageEntry(image="local", builder="oci", receipt=_REVISED)
    assert fetched.task_images.path is not None
    update_image_map(fetched.task_images.path, {lock.get(name).image["key"]: built for name in dockerfiles})

    with caplog.at_level(logging.WARNING):
        await service().check_tasks("train", sorted(dockerfiles))

    assert "2 task(s) of" in caplog.text and "(copy, workdir)" in caplog.text
    assert "keeps no COPY --chown and no USER among the deferred steps" in caplog.text


@pytest.mark.asyncio
async def test_a_workdir_after_a_user_is_that_users_and_the_session_runs_there(tmp_path, switching, monkeypatch):
    """A deferred WORKDIR after a USER makes its directories as that USER, as BuildKit does: the sandbox starts in /,
    so that its start makes no working directory as root, and the session's commands run in the WORKDIR."""

    use_site_cache(monkeypatch, tmp_path / "cache")
    dataset = tmp_path / "dataset"
    files = {
        "task.toml": 'version = "1.0"\n',
        "instruction.md": "Do it.\n",
        # In the WORKDIR: out.txt there, as the oracle wrote it where it ran.
        "tests/test.sh": f"[ -f {ROOT_DIR}/app/work/out.txt ] && echo 1 > {ROOT_DIR}/logs/verifier/reward.txt\n",
        "solution/solve.sh": "echo solved > out.txt\n",
        "environment/Dockerfile": "FROM python:3.12-slim\nRUN true\nUSER agent\nWORKDIR /app/work\n",
    }
    for relative, text in files.items():
        (dataset / "workdir" / relative).parent.mkdir(parents=True, exist_ok=True)
        (dataset / "workdir" / relative).write_text(text)
    service = HarborEnvironmentService(
        dataset=str(dataset),
        lock=str(tmp_path / "dataset.lock.json"),
        refuse={"agent_user": False},  # the Dockerfile ends with USER
        provider=switching,
        provider_options={"root": str(tmp_path / "sandboxes")},
        agent_user="honor",
    )
    await service.list_tasks("train")
    key = read_lock(tmp_path / "dataset.lock.json").get("workdir").image["key"]
    assert service.task_images.path is not None
    update_image_map(service.task_images.path, {key: ImageEntry(image="local", builder="oci", receipt=_REVISED)})

    session = await service.open(await service.get_task("workdir"), rollout_id="r")
    provider = service.provider
    assert isinstance(provider, SwitchingProvider)
    try:
        ((spec, sandbox),) = provider.started
        assert spec.workdir == "/"
        assert ["chown", "-h", "1000:1000", "--", "/app", "/app/work"] in provider.given  # the image had neither
        where = await session.call_tool("bash", {"command": "pwd -P"})
        assert where.content[0]["text"].strip() == os.path.realpath(sandbox.host_path("/app/work"))
        assert (await session.run_solution()).exit_code == 0
        outcome = await session.grade()
    finally:
        await session.close()

    assert (outcome.status, outcome.rewards) == ("scored", {"reward": 1.0}), outcome.metadata.get("test_output")
    assert ("--reuid=1000", ["bash", "-c", "bash /solution/solve.sh"]) in [
        (uid, argv) for uid, argv, _ in provider.switched
    ]


@pytest.mark.asyncio
async def test_a_user_the_image_lacks_fails_the_sessions_start(make_task, tmp_path, switching):
    task = make_task("ghost", task_toml='[agent]\nuser = "ghost"\n')
    service = _service([task], switching, tmp_path / "sandboxes", agent_user="honor")

    with pytest.raises(EnvSetupError, match="has no user 'ghost'") as refused:
        await service.open(await service.get_task("ghost"), rollout_id="r")

    assert refused.value.failure_kind == "env_config"
    provider = service.provider
    assert isinstance(provider, SwitchingProvider)
    ((_, sandbox),) = provider.started
    assert not sandbox.root.exists()  # stopped


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "options", "refusal"),
    [
        ("process", {}, "runs commands on the host as this process's user"),
        ("enroot", {}, "maps only your own uid into a sandbox"),
    ],
)
async def test_providers_that_cannot_give_root_refuse_honored_users_before_any_rollout(
    make_task, tmp_path, provider, options, refusal
):
    task = make_task("agent", task_toml=AGENT_TASK)
    service = HarborEnvironmentService(
        task_dirs=[task], provider=provider, provider_options=options, default_image="local", agent_user="honor"
    )

    with pytest.raises(EnvSetupError, match="its agents' sandbox runs commands as the users it names") as refused:
        await service.check_tasks("train", ["agent"])
    assert refusal in str(refused.value) and "Give those sandboxes root" in str(refused.value)
    with pytest.raises(EnvSetupError, match=refusal):
        await service.open(await service.get_task("agent"), rollout_id="r")
    await HarborEnvironmentService(task_dirs=[task], provider=provider, agent_user="ignore").check_tasks(
        "train", ["agent"]
    )


@pytest.mark.asyncio
async def test_a_copy_fork_is_refused_where_snapshots_lose_the_files_owners(
    make_task, tmp_path, switching, monkeypatch
):
    task = make_task("plain")
    service = _service([task], switching, tmp_path / "sandboxes", agent_user="honor")
    session = await service.open(await service.get_task("plain"), rollout_id="r")
    try:
        fork = await session.fork("copy")  # where snapshots keep owners (Docker's), a copy fork has the same users
        assert isinstance(fork.sandbox, SandboxView)
        await fork.close()
        monkeypatch.setattr(SwitchingProvider, "loses_file_owners", True)
        with pytest.raises(NotImplementedError, match="a snapshot of a 'process' sandbox keeps no file owners"):
            await session.fork("copy")
    finally:
        await session.close()


def test_the_honor_policy_is_an_option(make_task):
    unknown: Any = "obey"
    with pytest.raises(ValueError, match="agent_user must be one of"):
        HarborEnvironmentService(task_dirs=[make_task("t")], agent_user=unknown)
    assert HarborEnvironmentService(task_dirs=[make_task("u")], agent_user="honor").agent_user == "honor"
