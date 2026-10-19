"""Harbor tasks with a separate verifier, on the process provider: the artifacts reach a verifier sandbox of their
own, the agents never touch the grader, main is released early, and every verifier sandbox is stopped."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from platoon.envs.harbor import HarborEnvironmentService, HarborSession
from platoon.envs.harbor.artifacts import collect_artifacts
from platoon.envs.service import PLATOON_ZERO_KEY, EnvSetupError
from platoon.sandbox import (
    CWD_ERROR_EXIT_CODE,
    CWD_ERROR_PREFIX,
    DownloadReport,
    DroppedLink,
    ExecResult,
    Mount,
    SandboxSpec,
)
from platoon.sandbox.process import ProcessProvider, ProcessSandbox
from platoon.task import Task

pytestmark = pytest.mark.asyncio

ROOT = "${PLATOON_SANDBOX_ROOT:-}"
SEPARATE = """artifacts = ["/app/answer.txt"]
[verifier]
environment_mode = "separate"
timeout_sec = 30
[verifier.env]
EXPECTED = "42"
"""
# Scores the answer the agent left in main, as the verifier sandbox received it.
CHECK_ANSWER = f"""
if [ "$(cat {ROOT}/app/answer.txt 2>/dev/null)" = "$EXPECTED" ]; then echo 1; else echo 0; fi \\
  > {ROOT}/logs/verifier/reward.txt
"""


class RecordingProvider(ProcessProvider):
    """A process provider that keeps every spec it started a sandbox for, and the sandboxes."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.started: list[tuple[SandboxSpec, ProcessSandbox]] = []

    async def start(self, spec: SandboxSpec) -> ProcessSandbox:
        sandbox = await super().start(spec)
        self.started.append((spec, sandbox))
        return sandbox

    def verifiers(self) -> list[ProcessSandbox]:
        return [sandbox for spec, sandbox in self.started if spec.labels.get("platoon.role") == "verifier"]


@pytest.fixture
def recording(tmp_path: Path) -> RecordingProvider:
    return RecordingProvider(tmp_path / "sandboxes")


def _service(provider, *task_dirs: Path, **options) -> HarborEnvironmentService:
    return HarborEnvironmentService(task_dirs=list(task_dirs), provider=provider, default_image="local", **options)


async def _open(service: HarborEnvironmentService, name: str) -> HarborSession:
    return await service.open(await service.get_task(name), rollout_id=f"rollout-{name}")


async def test_the_verifier_grades_the_artifacts_in_a_sandbox_of_its_own(recording, make_task):
    service = _service(recording, make_task("answer", task_toml=SEPARATE, test_sh=CHECK_ANSWER))
    session = await _open(service, "answer")
    try:
        main = session.sandbox
        await session.call_tool("bash", {"command": "echo 42 > answer.txt"})
        await session.call_tool("bash", {"command": f"echo 1 > {ROOT}/logs/verifier/reward.txt"})  # a forgery

        outcome = await session.grade()

        assert (outcome.status, outcome.reward) == ("scored", 1.0)
        assert outcome.metadata["verifier_mode"] == "separate"
        assert outcome.metadata["artifacts"] == ["/app/answer.txt"]
        (verifier,) = recording.verifiers()
        assert verifier is not main and not verifier.root.exists()  # stopped once graded
        assert not (main.root / "tests").exists()  # the tests never reached the agents' sandbox
    finally:
        await session.close()


async def test_a_no_op_scores_zero_even_when_it_forges_a_reward(recording, make_task):
    service = _service(recording, make_task("answer", task_toml=SEPARATE, test_sh=CHECK_ANSWER))
    session = await _open(service, "answer")
    try:
        await session.call_tool("bash", {"command": f"echo 1 > {ROOT}/logs/verifier/reward.txt"})
        outcome = await session.grade()
        assert (outcome.status, outcome.reward) == (
            "scored",
            0.0,
        )  # the missing artifact was skipped; the tests decided
    finally:
        await session.close()


async def test_grading_releases_main_and_regrades_from_the_kept_artifacts(recording, make_task):
    service = _service(recording, make_task("answer", task_toml=SEPARATE, test_sh=CHECK_ANSWER))
    session = await _open(service, "answer")
    main = session.sandbox
    await session.call_tool("bash", {"command": "echo 42 > answer.txt"})

    first = await session.grade()

    assert session.sandbox is None  # nothing else held main: released once the artifacts were out
    assert (await session.call_tool("bash", {"command": "true"})).is_error
    with pytest.raises(RuntimeError, match="fork or act before grading"):
        await session.fork("copy")
    second = await session.grade()
    assert (first.reward, second.reward) == (1.0, 1.0)
    assert len(recording.verifiers()) == 2
    kept = session._kept_artifacts[0]
    await session.close()
    assert not main.root.exists() and not kept.exists()
    await session.close()  # idempotent


async def test_a_shared_fork_keeps_main_alive_through_grading(recording, make_task):
    service = _service(recording, make_task("answer", task_toml=SEPARATE, test_sh=CHECK_ANSWER))
    parent = await _open(service, "answer")
    child = await parent.fork()
    try:
        await child.call_tool("bash", {"command": "echo 41 > answer.txt"})
        await child.call_tool("bash", {"command": f"echo 1 > {ROOT}/logs/verifier/reward.txt"})  # a forgery in main
        assert (await parent.grade()).reward == 0.0
        assert parent.sandbox is not None  # the child still holds main, whose reward files grading never reads
        await child.call_tool("bash", {"command": "echo 42 > answer.txt"})
        assert (await parent.grade()).reward == 1.0  # collected again: main changed
        assert (await child.grade()).reward == 1.0
    finally:
        await child.close()
        await parent.close()
    assert all(not sandbox.root.exists() for _, sandbox in recording.started)


async def test_directory_artifacts_honor_excludes_and_logs_artifacts_is_handed_over(recording, make_task):
    task_toml = """artifacts = [{source = "/app/out", exclude = ["*.pyc"]}, "/app/missing.txt"]
[verifier]
environment_mode = "separate"
"""
    test_sh = f"""
out={ROOT}/app/out
[ -f $out/keep.txt ] && [ -f $out/sub/deep.txt ] && [ ! -e $out/skip.pyc ] \\
  && [ "$(cat {ROOT}/logs/artifacts/notes.md)" = "published" ] && [ ! -e {ROOT}/app/missing.txt ] \\
  && echo 1 > {ROOT}/logs/verifier/reward.txt || echo 0 > {ROOT}/logs/verifier/reward.txt
"""
    service = _service(recording, make_task("dirs", task_toml=task_toml, test_sh=test_sh))
    session = await _open(service, "dirs")
    try:
        await session.call_tool(
            "bash",
            {
                "command": "mkdir -p out/sub && echo k > out/keep.txt && echo d > out/sub/deep.txt && "
                f"echo x > out/skip.pyc && echo published > {ROOT}/logs/artifacts/notes.md"
            },
        )
        outcome = await session.grade()
        assert (outcome.status, outcome.reward) == ("scored", 1.0), outcome.metadata.get("test_output")
        assert outcome.metadata["artifacts"] == ["/app/out", "/logs/artifacts"]
    finally:
        await session.close()


async def test_the_verifier_sandbox_takes_its_own_settings(recording, make_task, tmp_path: Path):
    task_toml = """artifacts = ["/app/answer.txt"]
[verifier.environment]
cpus = 1
memory_mb = 256
network_mode = "no-network"
env = {GRADER = "on"}
[environment]
cpus = 4
memory_mb = 8192
"""
    shared, private = tmp_path / "shared", tmp_path / "private"
    shared.mkdir(), private.mkdir()
    mounts = [Mount(str(shared), "/data", read_only=True), Mount(str(private), "/scratch", read_only=False)]
    service = _service(recording, make_task("own", task_toml=task_toml, test_sh=CHECK_ANSWER), mounts=mounts)
    session = await _open(service, "own")
    try:
        await session.grade()
    finally:
        await session.close()

    (main_spec, _), (verifier_spec, _) = recording.started
    assert (main_spec.cpus, main_spec.memory_mb, main_spec.network) == (4, 8192, "host")
    assert (verifier_spec.cpus, verifier_spec.memory_mb, verifier_spec.network) == (1, 256, "none")
    assert verifier_spec.env == {"GRADER": "on"} and verifier_spec.image == "local"
    assert verifier_spec.labels == {
        "platoon.env": "harbor",
        "platoon.rollout_id": "rollout-own",
        "platoon.task": "own",
        "platoon.role": "verifier",
    }
    assert [mount.target for mount in main_spec.mounts] == ["/data", "/scratch"]
    assert [mount.target for mount in verifier_spec.mounts] == ["/data"]  # nothing the agents can write


async def test_a_verifier_image_holds_its_own_tests(recording, make_task):
    task_dir = make_task("image", task_toml='[verifier.environment]\ndocker_image = "grader:1"\n', test_sh="exit 9")
    service = _service(recording, task_dir)
    session = await _open(service, "image")
    try:
        outcome = await session.grade()
    finally:
        await session.close()

    (_, _), (verifier_spec, _) = recording.started
    assert verifier_spec.image == "docker://grader:1"
    # The tests were not uploaded: the process provider's "image" has no /tests/test.sh, so no reward file.
    assert (outcome.status, outcome.failure_kind, outcome.reason) == (
        "error",
        "verifier_no_reward",
        "reward_file_missing",
    )
    assert outcome.metadata["verifier_exit_code"] == 127


async def test_a_verifier_built_from_tests_is_refused_before_the_agent_runs(recording, make_task):
    task_dir = make_task("built", task_toml='[verifier]\nenvironment_mode = "separate"\n')
    (task_dir / "tests" / "Dockerfile").write_text("FROM python:3.12-slim\nCOPY . /tests/\n")
    service = _service(recording, task_dir)

    with pytest.raises(EnvSetupError, match=r"builds from tests/Dockerfile .*no image map") as refused:
        await _open(service, "built")
    assert refused.value.failure_kind == "env_config" and recording.started == []


async def test_a_verifier_that_cannot_start_is_an_error(make_task, tmp_path: Path):
    class NoVerifiers(RecordingProvider):
        async def start(self, spec: SandboxSpec) -> ProcessSandbox:
            if spec.labels.get("platoon.role") == "verifier":
                raise OSError("no capacity for the verifier")
            return await super().start(spec)

    provider = NoVerifiers(tmp_path / "sandboxes")
    service = _service(provider, make_task("answer", task_toml=SEPARATE, test_sh=CHECK_ANSWER))
    session = await _open(service, "answer")
    try:
        outcome = await session.grade()
        assert (outcome.status, outcome.failure_kind) == ("error", "verifier_start")
        assert "no capacity for the verifier" in outcome.reason
        assert outcome.metadata["setup_failure_kind"] == "sandbox_start"
    finally:
        await session.close()


async def test_a_cancelled_grade_still_stops_the_verifier(recording, make_task):
    service = _service(recording, make_task("slow", task_toml=SEPARATE, test_sh="sleep 30"))
    session = await _open(service, "slow")
    try:
        grading = asyncio.ensure_future(session.grade())
        for _ in range(100):
            if recording.verifiers() and (recording.verifiers()[0].root / "tests" / "test.sh").exists():
                break
            await asyncio.sleep(0.05)
        (verifier,) = recording.verifiers()
        grading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await grading
        for _ in range(100):
            if not verifier.root.exists():
                break
            await asyncio.sleep(0.05)
        assert not verifier.root.exists()
    finally:
        await session.close()


async def test_an_artifact_that_cannot_be_collected_is_an_error(recording, make_task):
    service = _service(recording, make_task("answer", task_toml=SEPARATE, test_sh=CHECK_ANSWER))
    session = await _open(service, "answer")
    try:
        await session.call_tool("bash", {"command": "echo 42 > answer.txt"})

        async def broken_download(src, dst):
            raise PermissionError(f"cannot read {src}")

        session.sandbox.download = broken_download  # type: ignore[method-assign]
        outcome = await session.grade()
        assert (outcome.status, outcome.failure_kind) == ("error", "artifact_collect")
        assert "cannot read" in outcome.reason and recording.verifiers() == []
    finally:
        await session.close()


@pytest.mark.parametrize(
    ("task_toml", "who"),
    [("[environment]\ngpus = 1\n", "it needs 1"), ("[verifier.environment]\ngpus = 2\n", "its verifier needs 2")],
)
async def test_tasks_that_need_gpus_are_refused_on_a_provider_without_them(recording, make_task, task_toml, who):
    service = _service(recording, make_task("gpu", task_toml=task_toml))

    with pytest.raises(EnvSetupError, match=rf"{who} GPU\(s\).*cannot give a sandbox GPUs") as refused:
        await _open(service, "gpu")
    assert refused.value.failure_kind == "env_config" and recording.started == []


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        ("ubuntu:24.04", "docker://ubuntu:24.04"),
        ("ghcr.io/org/img@sha256:abc", "docker://ghcr.io/org/img@sha256:abc"),
        ("docker://ubuntu:24.04", "docker://ubuntu:24.04"),
        ("oras://registry/img:1", "oras://registry/img:1"),
        ("/images/task.sif", "/images/task.sif"),
        ("images/task.sif", "images/task.sif"),
        ("task.sqsh", "task.sqsh"),
    ],
)
async def test_local_image_files_get_no_docker_prefix(recording, make_task, image: str, expected: str):
    service = _service(recording, make_task("image", task_toml=f'[environment]\ndocker_image = "{image}"\n'))
    session = await _open(service, "image")
    await session.close()

    assert recording.started[0][0].image == expected


async def test_workers_find_registered_tasks_without_indexing_the_dataset(recording, make_task, tmp_path: Path):
    task_dir = make_task("smol--1", task_toml=SEPARATE, test_sh=CHECK_ANSWER)
    root = tmp_path / "dataset"
    (root / "tasks" / "smol").mkdir(parents=True)
    task_dir.rename(root / "tasks" / "smol" / "smol--1")
    registry = [
        {"name": "d", "version": "1", "description": "", "tasks": [{"name": "smol--1", "path": "tasks/smol/smol--1"}]}
    ]
    (root / "registry.json").write_text(json.dumps(registry))
    (root / "task-index.json").write_text(json.dumps({"smol--1": {"source_dataset": "FineEnvs/SmolData"}}))
    service = HarborEnvironmentService(root, provider=recording, default_image="local")

    task = await service.get_task("smol--1")
    session = await service.open(task, rollout_id="r")
    try:
        assert session.harbor_task.index == {"source_dataset": "FineEnvs/SmolData"}
        assert task.misc["harbor"]["index"] == {"source_dataset": "FineEnvs/SmolData"}
        assert "dataset" not in vars(service)
    finally:
        await session.close()
    with pytest.raises(KeyError, match="Unknown Harbor task 'tasks/smol/smol--1'"):
        await service.get_task("tasks/smol/smol--1")  # a registered task is named by its registry name only


async def test_the_task_agent_timeout_scales_and_ignored_users_are_recorded(recording, make_task):
    task_dir = make_task("user", task_toml='[agent]\ntimeout_sec = 120\nuser = "agent"\n')
    made_elsewhere = Task(id="user", goal="Do the task.", misc={"harbor": {"task_dir": str(task_dir)}})
    with pytest.raises(EnvSetupError, match="agent_user: ignore") as refused:
        await _service(recording, task_dir).open(made_elsewhere, rollout_id="r")
    assert refused.value.failure_kind == "env_config" and recording.started == []

    service = _service(recording, task_dir, agent_timeout_multiplier=2.5, agent_user="ignore")
    session = await _open(service, "user")
    try:
        assert session.agent_timeout_s == service.agent_timeout_s("user") == 300.0
        outcome = await session.grade()
        assert outcome.metadata["ignored_users"] == {"agent": "agent"}
        assert outcome.metadata["verifier_mode"] == "shared"
    finally:
        await session.close()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"agent_timeout_multiplier": 0}, "agent_timeout_multiplier must be a positive number"),
        ({"agent_timeout_multiplier": True}, "agent_timeout_multiplier must be a positive number"),
        ({"verifier_timeout_multiplier": 0}, "verifier_timeout_multiplier must be a positive number"),
        ({"verifier_timeout_multiplier": -1.5}, "verifier_timeout_multiplier must be a positive number"),
        ({"tool_timeout_s": 0}, "tool_timeout_s must be a positive number"),
        ({"tool_timeout_s": "60"}, "tool_timeout_s must be a positive number"),
        ({"agent_user": "obey"}, "agent_user must be one of"),
        ({"reward": {"primary": "x", "weights": {}}}, "reward has unknown keys"),
        ({"verifier_classifiers": ["no-such-set"]}, "Unknown verifier classifier set"),
        ({"max_artifact_bytes": 0}, "max_artifact_bytes must be a positive number of bytes"),
        ({"max_artifact_bytes": 1.5}, "max_artifact_bytes must be a positive number of bytes"),
    ],
)
async def test_service_options_are_checked(make_task, options, message):
    with pytest.raises(ValueError, match=message):
        HarborEnvironmentService(task_dirs=[make_task("t")], **options)


# ------------------------------------------------------------------------------------- what the tests leave behind
# The tests run in the verifier sandbox, and may run code of the agent's that the artifacts hold (here /app/hook.sh).
RUNS_HOOK = """artifacts = ["/app/hook.sh"]
[verifier]
environment_mode = "separate"
"""
RUN_HOOK = f"echo 0 > {ROOT}/logs/verifier/reward.txt; [ -f {ROOT}/app/hook.sh ] && bash {ROOT}/app/hook.sh; true"


@pytest.mark.parametrize("launch", ["", "setsid"], ids=["in-its-group", "own-session"])
async def test_what_the_tests_left_running_in_the_verifier_cannot_rewrite_their_reward(
    recording, make_task, monkeypatch, launch: str
):
    """Code of the agent's that the tests run leaves a process that writes 1 into reward.txt once the tests are over;
    grading kills it before it reads their output (which here waits until the writer would have had its say)."""

    import platoon.envs.harbor.session as session_module

    real_collect = session_module.collect

    async def collect_later(sandbox, directory, **kwargs):
        sandbox.host_path("/tmp/go").write_text("")  # the writer's cue: the tests are over
        await asyncio.sleep(0.3)
        return await real_collect(sandbox, directory, **kwargs)

    monkeypatch.setattr(session_module, "collect", collect_later)
    writer = (
        'while [ ! -e "$PLATOON_SANDBOX_ROOT/tmp/go" ]; do sleep 0.005; done; '
        'while :; do echo 1 > "$PLATOON_SANDBOX_ROOT/logs/verifier/reward.txt"; sleep 0.005; done'
    )
    service = _service(recording, make_task("hook", task_toml=RUNS_HOOK, test_sh=RUN_HOOK))
    session = await _open(service, "hook")
    try:
        hook = f"{launch} sh -c '{writer}' </dev/null >/dev/null 2>&1 &\nsleep 0.2\n"
        assert not (await session.call_tool("write", {"path": "hook.sh", "content": hook})).is_error

        outcome = await session.grade()

        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": 0.0}, None), outcome
        (verifier,) = recording.verifiers()
        assert not verifier.root.exists()
    finally:
        await session.close()


async def test_a_reward_symlink_from_the_agents_code_in_the_verifier_scores_zero(recording, make_task, tmp_path):
    one = tmp_path / "host-file"
    one.write_text("1\n")
    service = _service(recording, make_task("hook", task_toml=RUNS_HOOK, test_sh=RUN_HOOK))
    session = await _open(service, "hook")
    try:
        hook = f'ln -sfn {one} "$PLATOON_SANDBOX_ROOT/logs/verifier/reward.txt"\n'
        assert not (await session.call_tool("write", {"path": "hook.sh", "content": hook})).is_error

        outcome = await session.grade()

        assert (outcome.status, outcome.reward, outcome.reason) == ("scored", 0.0, "reward_file_not_regular")
        assert outcome.trainable and outcome.metadata["reward_files"] == {"reward.txt": "symlink"}
    finally:
        await session.close()


async def test_a_verifier_that_cannot_kill_what_the_tests_left_is_an_error(recording, make_task):
    class Unkillable(RecordingProvider):
        async def start(self, spec: SandboxSpec) -> ProcessSandbox:
            sandbox = await super().start(spec)
            if spec.labels.get("platoon.role") == "verifier":

                async def cannot_kill() -> None:
                    raise RuntimeError("platoon: processes still alive after SIGKILL: 42")

                sandbox.kill_processes = cannot_kill  # type: ignore[method-assign]
            return sandbox

    provider = Unkillable(recording.root)
    service = _service(provider, make_task("answer", task_toml=SEPARATE, test_sh=CHECK_ANSWER))
    session = await _open(service, "answer")
    try:
        await session.call_tool("bash", {"command": "echo 42 > answer.txt"})
        outcome = await session.grade()
        assert (outcome.status, outcome.failure_kind, outcome.trainable) == ("error", "verifier_isolation", False)
        assert "after the tests" in outcome.reason and "verifier_exit_code" in outcome.metadata
    finally:
        await session.close()


# ------------------------------------------------------------------------------------- artifacts as regular files
OUT = """artifacts = ["/app/out"]
[verifier]
environment_mode = "separate"
"""
# Scores 1 when the answer the agent handed over matches the expected one, which only the tests hold.
MATCH_EXPECTED = f"""
answer=$(cat {ROOT}/app/out/answer.txt 2>/dev/null)
if [ "$answer" = "$(cat {ROOT}/tests/expected.txt)" ]; then echo 1; else echo 0; fi > {ROOT}/logs/verifier/reward.txt
"""


def _with_expected(task_dir: Path, expected: str = "the secret answer") -> Path:
    (task_dir / "tests" / "expected.txt").write_text(expected + "\n")
    return task_dir


async def test_an_artifact_symlink_never_reaches_the_verifier(recording, make_task):
    """A symlink in the artifacts would resolve in the verifier's sandbox, here to the expected answer that only the
    tests hold; it is left out, so the verifier sees no answer."""

    task_dir = _with_expected(make_task("link", task_toml=OUT, test_sh=MATCH_EXPECTED))
    service = _service(recording, task_dir)
    session = await _open(service, "link")
    try:
        made = await session.call_tool(
            "bash", {"command": "mkdir -p out && ln -s ../../tests/expected.txt out/answer.txt"}
        )
        assert not made.is_error, made

        outcome = await session.grade()

        assert (outcome.status, outcome.reward, outcome.trainable) == ("scored", 0.0, True), outcome
        assert outcome.metadata["artifacts_skipped"] == {"/app/out/answer.txt": "symlink"}
        assert outcome.metadata["artifacts_skipped_count"] == 1
    finally:
        await session.close()


async def test_an_artifact_that_is_a_symlink_to_a_host_file_is_never_read(recording, make_task, tmp_path: Path):
    """The process provider's sandbox is a host directory: a declared artifact that is a symlink would be copied from
    whatever host file it points to."""

    secret = tmp_path / "host-secret"
    secret.write_text("42\n")
    service = _service(recording, make_task("host", task_toml=SEPARATE, test_sh=CHECK_ANSWER))
    session = await _open(service, "host")
    try:
        assert not (await session.call_tool("bash", {"command": f"ln -s {secret} answer.txt"})).is_error

        outcome = await session.grade()

        assert (outcome.status, outcome.reward) == ("scored", 0.0), outcome
        assert outcome.metadata["artifacts_skipped"] == {"/app/answer.txt": "symlink"}
        assert outcome.metadata["artifacts"] == []
    finally:
        await session.close()


async def test_regular_files_hard_links_and_locked_directories_reach_the_verifier(recording, make_task):
    test_sh = f"""
out={ROOT}/app/out
ok=1
[ "$(cat $out/answer.txt)" = "the secret answer" ] || ok=0
[ "$(cat $out/locked/inner.txt)" = inner ] || ok=0
[ "$(cat $out/twin.txt)" = "the secret answer" ] && [ -f $out/twin.txt ] && [ ! -L $out/twin.txt ] || ok=0
[ -x $out/run.sh ] || ok=0
[ -e $out/fifo ] && ok=0
[ -e $out/link ] && ok=0
echo $ok > {ROOT}/logs/verifier/reward.txt
"""
    service = _service(recording, make_task("files", task_toml=OUT, test_sh=test_sh))
    session = await _open(service, "files")
    try:
        made = await session.call_tool(
            "bash",
            {
                "command": "mkdir -p out/locked && echo 'the secret answer' > out/answer.txt && ln out/answer.txt "
                "out/twin.txt && echo inner > out/locked/inner.txt && printf 'echo hi\\n' > out/run.sh && "
                "chmod +x out/run.sh && mkfifo out/fifo && ln -s /etc/hostname out/link && "
                "chmod 000 out/locked/inner.txt out/locked"
            },
        )
        assert not made.is_error, made

        outcome = await session.grade()

        assert (outcome.status, outcome.reward) == ("scored", 1.0), outcome.metadata.get("test_output")
        assert outcome.metadata["artifacts_skipped"] == {"/app/out/fifo": "special", "/app/out/link": "symlink"}
    finally:
        await session.close()


async def test_an_artifact_over_the_size_cap_is_left_out(recording, make_task):
    """An artifact whose archive would exceed max_artifact_bytes never reaches the host or the verifier: it is left
    out, measured in the agents' sandbox before anything is written, and the tests decide what its absence means."""

    task_toml = """artifacts = ["/app/big", "/app/answer.txt"]
[verifier]
environment_mode = "separate"
"""
    test_sh = f"""
[ ! -e {ROOT}/app/big ] && [ "$(cat {ROOT}/app/answer.txt)" = 42 ] && echo 1 > {ROOT}/logs/verifier/reward.txt \\
  || echo 0 > {ROOT}/logs/verifier/reward.txt
"""
    service = _service(recording, make_task("cap", task_toml=task_toml, test_sh=test_sh), max_artifact_bytes=50_000)
    session = await _open(service, "cap")
    try:
        made = await session.call_tool(
            "bash", {"command": "mkdir -p big && head -c 200000 /dev/zero > big/blob && echo 42 > answer.txt"}
        )
        assert not made.is_error, made

        outcome = await session.grade()

        assert (outcome.status, outcome.reward) == ("scored", 1.0), outcome.metadata.get("test_output")
        assert outcome.metadata["artifacts_skipped"] == {"/app/big": "too_large"}
        assert outcome.metadata["artifacts"] == ["/app/answer.txt"]
    finally:
        await session.close()


async def test_agent_processes_are_killed_before_the_artifacts_are_collected(recording, make_task):
    """A loop the agent left running swaps the answer for a symlink to the expected one again and again; grading kills
    it before the artifacts are packed (a shared fork keeps main up to show it), and neither form scores."""

    task_dir = _with_expected(make_task("loop", task_toml=OUT, test_sh=MATCH_EXPECTED))
    service = _service(recording, task_dir)
    parent = await _open(service, "loop")
    child = await parent.fork()
    try:
        loop = (
            "echo $$ > /tmp/loop.pid; mkdir -p out; while :; do echo wrong > out/answer.txt.new; "
            "mv -f out/answer.txt.new out/answer.txt; ln -sfn ../../tests/expected.txt out/answer.txt; "
            "sleep 0.001; done"
        )
        started = await child.call_tool(
            "bash", {"command": f"setsid sh -c '{loop}' </dev/null >/dev/null 2>&1 & sleep 0.2"}
        )
        assert not started.is_error, started
        pid = int(parent.sandbox.host_path("/tmp/loop.pid").read_text())
        assert _running(pid)

        outcome = await parent.grade()

        assert (outcome.status, outcome.reward) == ("scored", 0.0), outcome
        assert parent.sandbox is not None and not _running(pid)  # killed by grading, main still up
    finally:
        await child.close()
        await parent.close()


def _running(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except FileNotFoundError:
        return False


# --------------------------------------------------------------------------- downloads (hpc-sandbox 0.2's reports)


def _spy_downloads(sandbox: Any, seen: list[tuple[Path, list[str], bool]]) -> None:
    """Record each download's destination, what its directory held when it started, and whether it existed."""

    download = sandbox.download

    async def recorded(src, dst):
        dst = Path(dst)
        seen.append((dst, sorted(entry.name for entry in dst.parent.iterdir()), os.path.lexists(dst)))
        return await download(src, dst)

    sandbox.download = recorded  # type: ignore[method-assign]


async def test_each_grade_downloads_the_artifacts_into_a_new_directory(recording, make_task):
    """A download keeps the links that stay inside its copy, and those can reach whatever the destination held
    before (hpc-sandbox 0.2): every grade collects into a directory of its own that held nothing, other rollouts'
    grades and a second grade of the same session (main still held by a fork) included."""

    service = _service(recording, make_task("answer", task_toml=SEPARATE, test_sh=CHECK_ANSWER))
    first = await service.open(await service.get_task("answer"), rollout_id="rollout-1")
    second = await service.open(await service.get_task("answer"), rollout_id="rollout-2")
    fork = await first.fork("shared")  # holds main, so grading collects again rather than reuse the artifacts
    seen: list[tuple[Path, list[str], bool]] = []
    try:
        for session in (first, second):
            await session.call_tool("bash", {"command": "echo 42 > answer.txt"})
            _spy_downloads(session.sandbox, seen)
        grades = []
        for session in (first, first, second):
            before = len(seen)
            grades.append(await session.grade())
            assert len(seen) > before  # this grade downloaded
            destination, held, existed = seen[before]
            assert held == [] and not existed  # the grade's first download went into a directory that held nothing
        assert [grade.reward for grade in grades] == [1.0, 1.0, 1.0]
    finally:
        await fork.close()
        await first.close()
        await second.close()
    directories = {destination.parent for destination, _, _ in seen}
    assert len(directories) == 3  # one per grade
    assert not any(directory.exists() for directory in directories)  # removed with the grade or the session


async def test_artifacts_are_collected_only_into_a_new_or_empty_directory(tmp_path: Path):
    into = tmp_path / "into"
    into.mkdir()
    (into / "earlier").write_text("what an earlier download left")

    with pytest.raises(ValueError, match="new or empty directory"):
        await collect_artifacts(cast(Any, None), [], workdir="/app", into=into)  # never reaches the sandbox


async def test_an_archive_download_that_left_links_out_is_an_artifact_collect_error(recording, make_task):
    """An artifact's archive is one regular file: a download that reports links it left out (hpc-sandbox's
    DownloadReport) brought back something the sandbox put in its place.  The outcome names the links, and no
    verifier starts."""

    service = _service(recording, make_task("answer", task_toml=SEPARATE, test_sh=CHECK_ANSWER))
    session = await _open(service, "answer")
    try:
        await session.call_tool("bash", {"command": "echo 42 > answer.txt"})
        download = session.sandbox.download

        async def leaves_links_out(src, dst):
            await download(src, dst)
            return DownloadReport([DroppedLink("lib/x", "/etc/passwd", "absolute")], 7)

        session.sandbox.download = leaves_links_out  # type: ignore[method-assign]
        outcome = await session.grade()

        assert (outcome.status, outcome.failure_kind) == ("error", "artifact_collect")
        assert "left out 7 symlink(s) (lib/x -> /etc/passwd (absolute))" in (outcome.reason or "")
        assert recording.verifiers() == []
    finally:
        await session.close()


async def test_an_archive_that_comes_back_as_a_directory_is_an_artifact_collect_error(recording, make_task):
    service = _service(recording, make_task("answer", task_toml=SEPARATE, test_sh=CHECK_ANSWER))
    session = await _open(service, "answer")
    try:
        await session.call_tool("bash", {"command": "echo 42 > answer.txt"})

        async def a_directory(src, dst):
            Path(dst).mkdir()
            return DownloadReport()  # nothing left out, and still not the archive

        session.sandbox.download = a_directory  # type: ignore[method-assign]
        outcome = await session.grade()

        assert (outcome.status, outcome.failure_kind) == ("error", "artifact_collect")
        assert "did not come back as a regular file" in (outcome.reason or "") and recording.verifiers() == []
    finally:
        await session.close()


async def test_a_verifier_whose_tests_could_not_enter_its_workdir_records_why(recording, make_task):
    """Where hpc-sandbox says the tests' command never entered the verifier's workdir, the outcome records its line;
    the tests never wrote a reward, so the grade is an error, as for any missing reward."""

    service = _service(recording, make_task("answer", task_toml=SEPARATE, test_sh=CHECK_ANSWER))
    session = await _open(service, "answer")
    line = f"{CWD_ERROR_PREFIX}/app: permission denied"
    original_start = recording.start

    async def start(spec):
        sandbox = await original_start(spec)
        if spec.labels.get("platoon.role") == "verifier":
            exec_ = sandbox.exec

            async def cannot_enter(cmd, **kwargs):
                if "test.sh" in str(cmd):
                    return ExecResult(CWD_ERROR_EXIT_CODE, "", line + "\n")
                return await exec_(cmd, **kwargs)

            sandbox.exec = cannot_enter  # type: ignore[method-assign]
        return sandbox

    recording.start = start  # type: ignore[method-assign]
    try:
        await session.call_tool("bash", {"command": "echo 42 > answer.txt"})
        outcome = await session.grade()

        assert outcome.status == "error" and not outcome.trainable, outcome
        assert outcome.metadata["verifier_cwd_error"] == line and outcome.metadata["verifier_exit_code"] == 126
    finally:
        await session.close()


async def test_a_pack_failure_is_an_artifact_collect_error(recording, make_task):
    service = _service(recording, make_task("answer", task_toml=SEPARATE, test_sh=CHECK_ANSWER))
    session = await _open(service, "answer")
    try:
        await session.call_tool("bash", {"command": "echo 42 > answer.txt"})
        real_exec = session.sandbox.exec

        async def no_tar(cmd, **kwargs):
            if isinstance(cmd, list) and "platoon-artifact" in cmd:
                kwargs["env"] = {**(kwargs.get("env") or {}), "PATH": "/nonexistent"}
            return await real_exec(cmd, **kwargs)

        session.sandbox.exec = no_tar  # type: ignore[method-assign]
        outcome = await session.grade()
        assert (outcome.status, outcome.failure_kind) == ("error", "artifact_collect")
        assert "no tar" in outcome.reason
    finally:
        await session.close()


async def test_a_verifier_s_tests_that_exit_126_are_graded_by_their_reward(recording, make_task):
    """In the agents' sandbox, a 126 from the tests' command has grading check the workdir again before it reads any
    reward file (a process the kill missed may have removed it, and the tests never ran).  A verifier's sandbox never
    held the agents' processes, so a 126 there is the tests' own exit status, and their reward counts, even when
    they removed the verifier's workdir themselves."""

    test_sh = CHECK_ANSWER + f'rm -rf "{ROOT}/app"\nexit 126\n'
    service = _service(recording, make_task("answer", task_toml=SEPARATE, test_sh=test_sh))
    session = await _open(service, "answer")
    try:
        await session.call_tool("bash", {"command": "echo 42 > answer.txt"})
        outcome = await session.grade()
        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": 1.0}, None), outcome
        assert outcome.metadata["verifier_exit_code"] == 126
    finally:
        await session.close()


# --------------------------------------------------------------------------------------------- an unusable workdir
class _Raising(RecordingProvider):
    """Fails as hpc-sandbox did before 0.2.0 where a sandbox's workdir cannot be entered: commands run there, and
    transfers, which copied through commands run there, raise instead of starting."""

    async def start(self, spec: SandboxSpec) -> ProcessSandbox:
        sandbox = await super().start(spec)
        original_exec, original_download = sandbox.exec, sandbox.download

        def enter(path: str) -> None:
            directory = sandbox.host_path(path)
            if not (directory.is_dir() and os.access(directory, os.X_OK)):
                raise RuntimeError(f"sandbox {sandbox.id}: the command did not start (exit 2): can't cd to {path}")

        async def exec_(cmd, **kwargs):
            enter(kwargs.get("cwd") or spec.workdir)
            return await original_exec(cmd, **kwargs)

        async def download(src, dst):
            enter(spec.workdir)
            return await original_download(src, dst)

        sandbox.exec, sandbox.download = exec_, download  # type: ignore[method-assign]
        return sandbox


@pytest.mark.parametrize("raises", [False, True], ids=["process", "raises"])
@pytest.mark.parametrize(
    ("command", "state"),
    [("cd / && rm -rf /app", "missing"), ("cd / && rm -rf /app && echo 42 > /app", "not_a_directory")],
    ids=["removed", "a-file"],
)
async def test_an_agent_that_left_no_workdir_scores_zero_without_a_verifier(
    make_task, tmp_path: Path, raises: bool, command: str, state: str
):
    """The artifacts are collected from the agents' workdir; one they removed, or left a file in place of, scores 0
    before any verifier starts, whichever way the provider fails there."""

    provider = (_Raising if raises else RecordingProvider)(tmp_path / "sandboxes")
    service = _service(provider, make_task("answer", task_toml=SEPARATE, test_sh=CHECK_ANSWER))
    session = await _open(service, "answer")
    try:
        done = await session.call_tool("bash", {"command": f"echo 42 > answer.txt && {command}"})
        assert not done.is_error, done

        outcome = await session.grade()

        assert (outcome.status, outcome.rewards, outcome.reason) == ("scored", {"reward": 0.0}, "workdir_missing")
        assert outcome.metadata == {
            "workdir": "/app",
            "workdir_state": state,
            "verifier_mode": "separate",
            PLATOON_ZERO_KEY: True,
        }
        assert provider.verifiers() == [] and session.sandbox is not None
    finally:
        await session.close()


@pytest.mark.skipif(os.geteuid() == 0, reason="root enters a directory without its permissions")
async def test_an_agent_that_locked_its_workdir_scores_zero_without_a_verifier(recording, make_task):
    service = _service(recording, make_task("answer", task_toml=SEPARATE, test_sh=CHECK_ANSWER))
    session = await _open(service, "answer")
    workdir = session.sandbox.host_path("/app")
    try:
        assert not (await session.call_tool("bash", {"command": "echo 42 > answer.txt && chmod 000 /app"})).is_error
        outcome = await session.grade()
        assert (outcome.reason, outcome.metadata["workdir_state"]) == ("workdir_missing", "not_enterable")
        assert recording.verifiers() == []
    finally:
        workdir.chmod(0o755)
        await session.close()
