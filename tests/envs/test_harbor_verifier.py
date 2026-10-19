from __future__ import annotations

import os
from pathlib import Path

import pytest
import pytest_asyncio

from platoon.envs.harbor.verifier import (
    LOG_TAIL_BYTES,
    MAX_REWARD_BYTES,
    TEST_OUTPUT_TAIL_BYTES,
    VerifierFile,
    collect,
    outcome_from_reward_json,
    outcome_from_reward_text,
    output_tail,
    read_outcome,
    suspect_outcome,
    write_local,
)
from platoon.envs.service import PLATOON_ZERO_KEY
from platoon.sandbox import SandboxSpec
from platoon.sandbox.process import ProcessProvider


def test_reward_text_is_the_primary_reward():
    outcome = outcome_from_reward_text(" 0.75\n")

    assert (outcome.status, outcome.rewards, outcome.reward) == ("scored", {"reward": 0.75}, 0.75)


def test_reward_json_keeps_named_components():
    outcome = outcome_from_reward_json('{"reward": 1, "tests_passed": 3, "lint": true, "invalid": 0}')

    assert outcome.status == "scored"
    assert outcome.rewards == {"reward": 1.0, "tests_passed": 3.0, "lint": 1.0}


@pytest.mark.parametrize(
    ("raw", "failure_kind"),
    [
        ('{"reward": 0, "invalid": 1, "failure_kind": "judge_unavailable"}', "judge_unavailable"),
        ('{"reward": 0, "invalid": true}', "verifier_declared"),
        ('{"invalid": 0.5}', "verifier_declared"),
    ],
)
def test_an_invalid_key_declares_the_grade_invalid(raw: str, failure_kind: str):
    outcome = outcome_from_reward_json(raw)

    assert (outcome.status, outcome.failure_kind, outcome.reward, outcome.rewards) == (
        "invalid",
        failure_kind,
        None,
        {},
    )


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("", "reward_file_empty"),
        ("{not json", "reward_file_malformed"),
        ("[1, 2]", "reward_file_malformed"),
        ('{"reward": "high"}', "reward_file_malformed"),
        ('{"reward": NaN}', "reward_file_malformed"),
        ('{"reward": 1, "invalid": "yes"}', "reward_file_malformed"),
        ('{"reward": 1, "invalid": 1, "failure_kind": 7}', "reward_file_malformed"),
    ],
)
def test_bad_reward_json_is_an_infra_error_not_zero(raw: str, reason: str):
    outcome = outcome_from_reward_json(raw)

    assert (outcome.status, outcome.failure_kind, outcome.reason, outcome.reward) == (
        "error",
        "verifier_no_reward",
        reason,
        None,
    )


@pytest.mark.parametrize(("raw", "reason"), [("\n", "reward_file_empty"), ("pass", "reward_file_malformed")])
def test_bad_reward_text_is_an_infra_error(raw: str, reason: str):
    outcome = outcome_from_reward_text(raw)

    assert (outcome.status, outcome.reason) == ("error", reason)
    assert outcome_from_reward_text("inf").reason == "reward_file_malformed"


def test_reward_json_without_the_primary_reward_is_an_error():
    outcome = outcome_from_reward_json('{"accuracy": 0.5}')

    assert (outcome.status, outcome.failure_kind) == ("error", "verifier_no_primary")


def test_read_outcome_prefers_json_and_reports_missing_files(tmp_path: Path):
    assert read_outcome(tmp_path).reason == "reward_file_missing"
    (tmp_path / "reward.txt").write_text("0")
    assert read_outcome(tmp_path).reward == 0.0
    (tmp_path / "reward.json").write_text('{"reward": 0.5}')
    assert read_outcome(tmp_path).reward == 0.5


@pytest.mark.parametrize(
    ("files", "reason"),
    [
        ({"reward.txt": VerifierFile("symlink")}, "reward_file_not_regular"),
        ({"reward.json": VerifierFile("special"), "reward.txt": VerifierFile("file", "1")}, "reward_file_not_regular"),
        (
            {"reward.json": VerifierFile("file", '{"reward": 1}'), "reward.txt": VerifierFile("symlink")},
            "reward_file_not_regular",
        ),
        ({"reward.txt": VerifierFile("unreadable")}, "reward_file_not_regular"),
        ({"reward.json": VerifierFile("mount"), "reward.txt": VerifierFile("mount")}, "reward_file_not_regular"),
        ({"reward.json": VerifierFile("too_large")}, "reward_file_too_large"),
    ],
)
def test_a_reward_file_that_is_not_a_regular_file_scores_zero(files: dict[str, VerifierFile], reason: str):
    """Grading made the directory afresh and killed the agents' processes, so only code the tests ran (the agent's)
    could have put it there: a trainable 0, whatever another reward file says."""

    outcome = suspect_outcome(files)

    assert outcome is not None
    assert (outcome.status, outcome.reward, outcome.trainable) == ("scored", 0.0, True)
    assert outcome.reason == reason
    assert outcome.metadata["reward_files"] == {
        name: file.kind for name, file in files.items() if file.kind not in ("file", "missing")
    }
    assert outcome.metadata[PLATOON_ZERO_KEY] is True  # platoon's own: it trained no reward key


def test_regular_or_missing_reward_files_are_read_from_a_local_copy(tmp_path: Path):
    files = {
        "reward.json": VerifierFile("missing"),
        "reward.txt": VerifierFile("file", "0.5\n"),
        "test-stdout.txt": VerifierFile("file", "x" * 5000 + "end"),
        "stderr.txt": VerifierFile("symlink"),
    }
    assert suspect_outcome(files) is None and suspect_outcome({}) is None

    local = write_local(files, tmp_path / "verifier")

    assert sorted(path.name for path in local.iterdir()) == ["reward.txt", "test-stdout.txt"]
    assert read_outcome(local).reward == 0.5
    assert output_tail(files) == ("x" * 5000 + "end")[-TEST_OUTPUT_TAIL_BYTES:]
    assert output_tail({"test-stdout.txt": VerifierFile("symlink")}) is None


@pytest_asyncio.fixture
async def sandbox(tmp_path: Path):
    box = await ProcessProvider(root=tmp_path / "sandboxes").start(SandboxSpec(image="local"))
    try:
        yield box
    finally:
        await box.stop()


@pytest.mark.asyncio
async def test_collect_reads_only_regular_files_reached_through_no_symlink(sandbox, tmp_path: Path):
    secret = tmp_path / "host-secret.txt"
    secret.write_text("1\n")
    verifier = sandbox.host_path("/logs/verifier")
    verifier.mkdir(parents=True)
    (verifier / "reward.txt").symlink_to(secret)
    os.mkfifo(verifier / "reward.json")  # reading it would block
    (verifier / "big.json").write_text("x" * (MAX_REWARD_BYTES + 1))
    (verifier / "ok.txt").write_text("0.25\n")
    (verifier / "dir.txt").mkdir()
    (verifier / "test-stdout.txt").write_text("a" * 10_000 + "the end\n")
    (verifier / "linked-stdout.txt").symlink_to(secret)  # a log's tail is not read through a symlink either

    files = await collect(
        sandbox,
        "/logs/verifier",
        whole=("reward.txt", "reward.json", "big.json", "ok.txt", "dir.txt", "absent.txt"),
        tail=("test-stdout.txt", "linked-stdout.txt"),
        tail_bytes=TEST_OUTPUT_TAIL_BYTES,
    )

    assert {name: file.kind for name, file in files.items()} == {
        "reward.txt": "symlink",
        "reward.json": "special",
        "big.json": "too_large",
        "ok.txt": "file",
        "dir.txt": "special",
        "absent.txt": "missing",
        "test-stdout.txt": "file",
        "linked-stdout.txt": "symlink",
    }
    assert files["ok.txt"].text == "0.25\n" and files["reward.txt"].text == ""
    assert len(files["test-stdout.txt"].text) == TEST_OUTPUT_TAIL_BYTES
    assert files["test-stdout.txt"].text.endswith("the end\n")


@pytest.mark.asyncio
async def test_collect_reads_the_reward_files_and_the_last_mib_of_each_log_by_default(sandbox):
    verifier = sandbox.host_path("/logs/verifier")
    verifier.mkdir(parents=True)
    (verifier / "reward.json").write_text('{"reward": 1}')
    (verifier / "test-stdout.txt").write_text("a" * LOG_TAIL_BYTES + "the end\n")
    (verifier / "stderr.txt").write_text("subprocess.TimeoutExpired: Command\n")

    files = await collect(sandbox, "/logs/verifier")

    assert {name: file.kind for name, file in files.items()} == {
        "reward.json": "file",
        "reward.txt": "missing",
        "test-stdout.txt": "file",
        "test-stderr.txt": "missing",
        "stdout.txt": "missing",
        "stderr.txt": "file",
    }
    assert len(files["test-stdout.txt"].text) == LOG_TAIL_BYTES and files["test-stdout.txt"].text.endswith("end\n")


@pytest.mark.asyncio
@pytest.mark.parametrize(("make", "kind"), [("symlink", "symlink"), ("file", "special"), (None, "missing")])
async def test_collect_never_reads_through_a_replaced_directory(sandbox, tmp_path: Path, make: str | None, kind: str):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "reward.txt").write_text("1")
    logs = sandbox.host_path("/logs")
    logs.mkdir(parents=True)
    if make == "symlink":
        (logs / "verifier").symlink_to(elsewhere)
    elif make == "file":
        (logs / "verifier").write_text("1")

    files = await collect(sandbox, "/logs/verifier", tail=())

    assert files == {"reward.json": VerifierFile(kind), "reward.txt": VerifierFile(kind)}
