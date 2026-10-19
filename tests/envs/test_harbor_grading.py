"""Reward contracts and verifier-log classifiers: which reported reward trains, and when a reading is invalid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from platoon.envs.harbor import HarborEnvironmentService
from platoon.envs.harbor.grading import CLASSIFIER_SETS, LogClassifier, RewardContract, classifier_list, classify
from platoon.envs.harbor.task import HarborTask
from platoon.envs.service import PLATOON_ZERO_KEY, Outcome
from platoon.sandbox.process import ProcessProvider

ROOT = "${PLATOON_SANDBOX_ROOT:-}"
CHECKS: list[tuple[dict[str, float], Path, Any]] = []


def record_and_keep(rewards: dict[str, float], verifier_dir: Path, task: Any) -> Outcome | None:
    CHECKS.append((rewards, verifier_dir, task))
    return None


def smoldata_rule(rewards: dict[str, float], verifier_dir: Path, task: Any) -> Outcome | None:
    """A dataset's own rule: correctness counts only with a submission."""

    if rewards.get("correctness") and not rewards.get("submission"):
        return Outcome.infra("verifier_contract", status="invalid", reason="correctness without a submission")
    return None


def raising(rewards: dict[str, float], verifier_dir: Path, task: Any) -> Outcome | None:
    raise RuntimeError("the dataset's rule broke")


def not_an_outcome(rewards: dict[str, float], verifier_dir: Path, task: Any) -> Any:
    return 1.0


def rescored(rewards: dict[str, float], verifier_dir: Path, task: Any) -> Outcome | None:
    return Outcome(status="scored", rewards={"reward": rewards["correctness"] * 0.5})


def _verifier_dir(tmp_path: Path, rewards: dict[str, Any] | str, *, stdout: str = "", name: str = "verifier") -> Path:
    verifier_dir = tmp_path / name / "verifier"
    verifier_dir.mkdir(parents=True)
    if isinstance(rewards, str):
        (verifier_dir / "reward.txt").write_text(rewards)
    else:
        (verifier_dir / "reward.json").write_text(json.dumps(rewards))
    if stdout:
        (verifier_dir / "test-stdout.txt").write_text(stdout)
    return verifier_dir


def test_the_default_contract_trains_reward(tmp_path: Path):
    outcome = RewardContract().read(_verifier_dir(tmp_path, "0.5\n"), task=None)

    assert (outcome.status, outcome.primary, outcome.reward) == ("scored", "reward", 0.5)


def test_a_primary_key_trains_from_a_multi_key_reward_file(tmp_path: Path):
    rewards = {"correctness": 1, "submission": 1, "tool_efficiency": 0.25}
    outcome = RewardContract.coerce("correctness").read(_verifier_dir(tmp_path, rewards), task=None)

    assert (outcome.status, outcome.primary, outcome.reward) == ("scored", "correctness", 1.0)
    assert outcome.rewards == {"correctness": 1.0, "submission": 1.0, "tool_efficiency": 0.25}


def test_a_reward_file_without_the_primary_is_an_error_never_zero(tmp_path: Path):
    outcome = RewardContract(primary="correctness").read(_verifier_dir(tmp_path, "1.0"), task=None)

    assert (outcome.status, outcome.failure_kind) == ("error", "verifier_no_primary")
    assert outcome.reason == "the verifier reported ['reward'], not the primary reward 'correctness'"


@pytest.mark.parametrize(
    ("contract", "rewards", "failure_kind"),
    [
        ({"primary": "reward", "components": ["reward"]}, {"reward": 1, "bonus": 1}, "verifier_unexpected_keys"),
        ({"primary": "reward", "range": [0, 1]}, {"reward": 100}, "verifier_out_of_range"),
        ({"primary": "reward", "range": [0, 1]}, {"reward": -0.5}, "verifier_out_of_range"),
    ],
)
def test_components_and_range_make_a_reading_invalid(tmp_path: Path, contract, rewards, failure_kind):
    outcome = RewardContract.coerce(contract).read(_verifier_dir(tmp_path, rewards), task=None)

    assert (outcome.status, outcome.failure_kind, outcome.reward) == ("invalid", failure_kind, None)
    assert outcome.metadata["reported"] == {key: float(value) for key, value in rewards.items()}


def test_the_range_bounds_only_the_primary(tmp_path: Path):
    contract = RewardContract.coerce({"primary": "reward", "range": [0, 1]})

    outcome = contract.read(_verifier_dir(tmp_path, {"reward": 1, "objective": 731.5}), task=None)
    assert (outcome.status, outcome.reward) == ("scored", 1.0)


def test_a_check_hook_sees_the_rewards_the_verifier_dir_and_the_task(tmp_path: Path):
    CHECKS.clear()
    task = HarborTask(task_dir=tmp_path, name="t", instruction="", index={"source_dataset": "FineEnvs/SmolData"})
    verifier_dir = _verifier_dir(tmp_path, {"reward": 1}, stdout="ok\n")

    outcome = RewardContract(check=f"{__name__}:record_and_keep").read(verifier_dir, task)

    assert (outcome.status, outcome.reward) == ("scored", 1.0)
    ((rewards, seen_dir, seen_task),) = CHECKS
    assert rewards == {"reward": 1.0} and seen_task is task
    assert seen_dir.name == "verifier" and (seen_dir / "test-stdout.txt").read_text() == "ok\n"


@pytest.mark.parametrize(
    ("hook", "rewards", "expected"),
    [
        ("smoldata_rule", {"correctness": 1, "submission": 1}, ("scored", None)),
        ("smoldata_rule", {"correctness": 1, "submission": 0}, ("invalid", "verifier_contract")),
        ("raising", {"correctness": 1}, ("error", "unexpected:RuntimeError")),
        ("not_an_outcome", {"correctness": 1}, ("error", "verifier_contract")),
        ("rescored", {"correctness": 1}, ("scored", None)),
    ],
)
def test_a_check_hook_applies_the_datasets_own_rules(tmp_path: Path, hook, rewards, expected):
    contract = RewardContract(primary="correctness", check=f"{__name__}:{hook}")

    outcome = contract.read(_verifier_dir(tmp_path, rewards), task=None)

    assert (outcome.status, outcome.failure_kind) == expected
    if hook == "rescored":
        assert outcome.reward == 0.5


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"primary": ""}, "reward.primary must name a reward"),
        ({"primary": "a", "components": ["b"]}, "must include the primary 'a'"),
        ({"range": [1, 0]}, "reward.range must be"),
        ({"range": [0]}, "reward.range must be"),
        ({"range": [0, float("inf")]}, "reward.range must be"),
        ({"check": "no_colon"}, "reward.check must be 'module:function'"),
        ({"check": "platoon.no_such_module:fn"}, "cannot be imported"),
        ({"check": "platoon.envs.harbor.grading:CLASSIFIED_LOGS"}, "is not callable"),
        ({"weights": {}}, "reward has unknown keys"),
        (3, "reward must be a mapping, a reward name or a list of names"),
        ({"primary": []}, "reward.primary must list reward names"),
        ({"primary": ["reward", ""]}, "reward.primary must list reward names"),
        ({"primary": ["reward", 1]}, "reward.primary must list reward names"),
        (["reward", "reward"], "reward.primary lists a reward twice"),
        (
            {"primary": ["reward", "correctness"], "components": ["reward"]},
            r"must include every primary \['correctness'\]",
        ),
    ],
)
def test_contracts_are_checked_when_the_service_is_built(value, message):
    with pytest.raises(ValueError, match=message):
        RewardContract.coerce(value)


@pytest.mark.parametrize(
    ("log", "kind"),
    [
        (
            "Traceback (most recent call last):\nsubprocess.TimeoutExpired: Command 'x' timed out\n",
            "verifier_subprocess_timeout",
        ),
        ("/tests/test.sh: line 4: pytest: command not found\n", "verifier_missing_test_runner"),
        ("/usr/bin/python3: No module named pytest\n", "verifier_missing_test_runner"),
    ],
)
def test_the_harbor_common_classifiers(tmp_path: Path, log: str, kind: str):
    verifier_dir = _verifier_dir(tmp_path, "0", stdout=log)

    outcome = classify(verifier_dir, classifier_list("harbor-common"))

    assert outcome is not None and (outcome.status, outcome.failure_kind) == ("invalid", kind)


def test_classifiers_match_whole_lines_of_any_verifier_log(tmp_path: Path):
    verifier_dir = _verifier_dir(tmp_path, "0", stdout="the agent printed subprocess.TimeoutExpired: in a message\n")
    common = classifier_list(["harbor-common"])

    assert classify(verifier_dir, common) is None  # not at a line start: the agent's own output
    (verifier_dir / "test-stderr.txt").write_text("subprocess.TimeoutExpired: Command 'grader' timed out\n")
    assert classify(verifier_dir, common).failure_kind == "verifier_subprocess_timeout"
    assert classify(verifier_dir, ()) is None


def test_classifier_lists_mix_named_sets_and_patterns():
    custom = {"pattern": r"^judge unreachable$", "kind": "judge_unavailable"}

    assert classifier_list(None) == ()
    assert classifier_list(custom) == (LogClassifier(r"^judge unreachable$", "judge_unavailable"),)
    assert classifier_list(["harbor-common", custom]) == (*CLASSIFIER_SETS["harbor-common"], classifier_list(custom)[0])
    for bad, message in [
        (["no-such-set"], "Unknown verifier classifier set"),
        ([{"pattern": "("}], "needs a kind"),
        ([{"pattern": "(", "kind": "k"}], "not a valid regular expression"),
        ([{"pattern": "x", "kind": "k", "files": []}], "takes pattern and kind"),
        ([3], "set names or"),
        (3, "must be a list"),
    ]:
        with pytest.raises(ValueError, match=message):
            classifier_list(bad)


@pytest.mark.asyncio
async def test_the_service_grades_under_the_contract_after_the_classifiers(tmp_path: Path, make_task):
    multi = make_task(
        "multi", test_sh=f'echo \'{{"correctness": 1, "submission": 1}}\' > {ROOT}/logs/verifier/reward.json'
    )
    crashed = make_task(
        "crashed",
        test_sh=f"echo 0 > {ROOT}/logs/verifier/reward.txt; echo 'subprocess.TimeoutExpired: Command' >&2; exit 1",
    )
    provider = ProcessProvider(tmp_path / "sandboxes")
    service = HarborEnvironmentService(
        task_dirs=[multi, crashed],
        provider=provider,
        default_image="local",
        reward={"primary": "correctness", "check": f"{__name__}:smoldata_rule"},
        verifier_classifiers=["harbor-common"],
    )
    outcomes = {}
    for name in ("multi", "crashed"):
        session = await service.open(await service.get_task(name), rollout_id=name)
        try:
            outcomes[name] = await session.grade()
        finally:
            await session.close()

    assert (outcomes["multi"].status, outcomes["multi"].primary, outcomes["multi"].reward) == (
        "scored",
        "correctness",
        1.0,
    )
    crashed_outcome = outcomes["crashed"]
    assert (crashed_outcome.status, crashed_outcome.failure_kind) == ("invalid", "verifier_subprocess_timeout")
    assert "subprocess.TimeoutExpired" in crashed_outcome.metadata["test_output"]  # the pre-written 0 does not count


# --- a list of primary names: one environment for tasks that report their score under different names ----------

LISTED = ["reward", "correctness"]


def test_one_contract_scores_a_reward_txt_task_and_a_reward_json_task(tmp_path: Path):
    """SmolData's tasks write reward.txt (its number is ``reward``) or reward.json with ``correctness`` and
    ``submission``: one contract trains each under its own name, and the outcome says which."""

    contract = RewardContract.coerce({"primary": LISTED, "range": [0, 1]})

    text = contract.read(_verifier_dir(tmp_path, "0.5\n", name="txt"), task=None)
    as_json = contract.read(_verifier_dir(tmp_path, {"correctness": 1, "submission": 1}, name="json"), task=None)

    assert (text.status, text.primary, text.reward, text.rewards) == ("scored", "reward", 0.5, {"reward": 0.5})
    assert (as_json.status, as_json.primary, as_json.reward) == ("scored", "correctness", 1.0)
    assert as_json.rewards == {"correctness": 1.0, "submission": 1.0}


@pytest.mark.parametrize("order", [LISTED, list(reversed(LISTED))])
def test_a_reading_with_two_listed_names_is_invalid_whatever_the_order(tmp_path: Path, order: list[str]):
    """A policy whose code the tests run could add a second listed key to choose what trains: never scored."""

    outcome = RewardContract.coerce(order).read(_verifier_dir(tmp_path, {"reward": 1, "correctness": 0}), task=None)

    assert (outcome.status, outcome.failure_kind, outcome.reward) == ("invalid", "verifier_ambiguous_primary", None)
    assert outcome.reason == (
        f"the verifier reported ['correctness', 'reward'], more than one of the primary rewards {order}"
    )
    assert outcome.metadata["reported"] == {"reward": 1.0, "correctness": 0.0}


def test_a_reading_with_no_listed_name_is_an_error_never_zero(tmp_path: Path):
    outcome = RewardContract.coerce(LISTED).read(_verifier_dir(tmp_path, {"submission": 1}), task=None)

    assert (outcome.status, outcome.failure_kind) == ("error", "verifier_no_primary")
    assert (
        outcome.reason == "the verifier reported ['submission'], none of the primary rewards ['reward', 'correctness']"
    )


def test_both_reward_files_are_invalid_under_a_list_and_read_as_before_under_one_name(tmp_path: Path):
    """With a list, a second reward file could change which name trains: the reading is invalid, also with a list of
    one name.  With one name, reward.json is read and reward.txt ignored, as before."""

    verifier_dir = _verifier_dir(tmp_path, {"correctness": 1, "submission": 1})
    (verifier_dir / "reward.txt").write_text("0\n")

    listed = RewardContract.coerce(LISTED).read(verifier_dir, task=None)
    assert (listed.status, listed.failure_kind) == ("invalid", "verifier_ambiguous_reward_files")
    assert listed.reason == (
        "the verifier left both reward.json and reward.txt; with a list of primary rewards a reading comes from one "
        "reward file"
    )
    one_listed = RewardContract.coerce({"primary": ["correctness"]}).read(verifier_dir, task=None)
    assert (one_listed.status, one_listed.failure_kind) == ("invalid", "verifier_ambiguous_reward_files")
    single = RewardContract.coerce("correctness").read(verifier_dir, task=None)
    assert (single.status, single.primary, single.reward) == ("scored", "correctness", 1.0)


def test_the_range_bounds_the_name_that_trained(tmp_path: Path):
    contract = RewardContract.coerce({"primary": LISTED, "range": [0, 1]})

    over = contract.read(_verifier_dir(tmp_path, {"correctness": 5, "submission": 1}, name="over"), task=None)
    other = contract.read(_verifier_dir(tmp_path, {"correctness": 1, "objective": 731.5}, name="other"), task=None)

    assert (over.status, over.failure_kind, over.reason) == (
        "invalid",
        "verifier_out_of_range",
        "correctness = 5 is outside [0.0, 1.0]",
    )
    assert (other.status, other.primary, other.reward) == ("scored", "correctness", 1.0)  # objective is not bounded


def test_components_take_every_listed_name(tmp_path: Path):
    contract = RewardContract.coerce({"primary": LISTED, "components": ["reward", "correctness", "submission"]})

    assert contract.components == ("reward", "correctness", "submission")
    outcome = contract.read(_verifier_dir(tmp_path, {"correctness": 1, "bonus": 1}), task=None)
    assert (outcome.status, outcome.failure_kind) == ("invalid", "verifier_unexpected_keys")


def test_a_list_is_short_for_a_list_primary_and_one_name_is_unchanged():
    assert RewardContract.coerce(LISTED) == RewardContract(primary=("reward", "correctness"))
    assert RewardContract.coerce({"primary": LISTED}) == RewardContract.coerce(tuple(LISTED))
    assert RewardContract.coerce(LISTED).names == ("reward", "correctness")
    assert RewardContract.coerce("correctness").primary == "correctness"
    assert RewardContract.coerce("correctness").names == ("correctness",)
    assert RewardContract().names == ("reward",)


def test_a_check_hook_sees_the_rewards_of_the_name_that_trained(tmp_path: Path):
    contract = RewardContract.coerce({"primary": LISTED, "check": f"{__name__}:smoldata_rule"})

    kept = contract.read(_verifier_dir(tmp_path, {"correctness": 1, "submission": 1}, name="kept"), task=None)
    refused = contract.read(_verifier_dir(tmp_path, {"correctness": 1, "submission": 0}, name="refused"), task=None)

    assert (kept.status, kept.primary) == ("scored", "correctness")
    assert (refused.status, refused.failure_kind) == ("invalid", "verifier_contract")


@pytest.mark.parametrize(
    ("primary", "key"),
    [
        ("reward", "reward"),
        ("correctness", "correctness"),
        (["correctness", "reward"], "correctness"),
        (LISTED, "reward"),
    ],
)
def test_platoons_own_zeros_score_under_the_contracts_first_name(primary, key):
    outcome = RewardContract.coerce(primary).zero("workdir_missing", {"workdir": "/app"})

    assert (outcome.status, outcome.primary, outcome.rewards) == ("scored", key, {key: 0.0})
    assert (outcome.reason, outcome.trainable) == ("workdir_missing", True)
    # Marked as platoon's own, so that it trained no reward key (platoon.utils.trajectory_status.trained_reward_key).
    assert outcome.metadata == {"workdir": "/app", PLATOON_ZERO_KEY: True}


@pytest.mark.asyncio
async def test_one_environment_grades_tasks_that_write_either_reward_file(tmp_path: Path, make_task):
    txt = make_task("writes-txt", test_sh=f"echo 0.25 > {ROOT}/logs/verifier/reward.txt")
    as_json = make_task(
        "writes-json", test_sh=f'echo \'{{"correctness": 1, "submission": 1}}\' > {ROOT}/logs/verifier/reward.json'
    )
    both = make_task(
        "writes-both",
        test_sh=(
            f"echo 1 > {ROOT}/logs/verifier/reward.txt; "
            f"echo '{{\"correctness\": 1}}' > {ROOT}/logs/verifier/reward.json"
        ),
    )
    service = HarborEnvironmentService(
        task_dirs=[txt, as_json, both],
        provider=ProcessProvider(tmp_path / "sandboxes"),
        default_image="local",
        reward={"primary": LISTED, "range": [0, 1]},
    )
    outcomes = {}
    for name in ("writes-txt", "writes-json", "writes-both"):
        session = await service.open(await service.get_task(name), rollout_id=name)
        try:
            outcomes[name] = await session.grade()
        finally:
            await session.close()

    assert (outcomes["writes-txt"].status, outcomes["writes-txt"].primary, outcomes["writes-txt"].reward) == (
        "scored",
        "reward",
        0.25,
    )
    assert (outcomes["writes-json"].status, outcomes["writes-json"].primary, outcomes["writes-json"].reward) == (
        "scored",
        "correctness",
        1.0,
    )
    assert (outcomes["writes-both"].status, outcomes["writes-both"].failure_kind) == (
        "invalid",
        "verifier_ambiguous_reward_files",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("planted", "failure_kind"),
    [
        ('{"correctness": 1}', "verifier_ambiguous_reward_files"),  # beside the tests' reward.txt
        ('{"reward": 1, "correctness": 0}', "verifier_ambiguous_primary"),  # over the tests' reward.json
    ],
)
async def test_code_the_tests_run_cannot_choose_the_name_that_trains(tmp_path: Path, make_task, planted, failure_kind):
    """The tests run the agent's program (hook.sh), which writes a reward file of its own: under a list of names
    the reading is invalid, never the planted score."""

    writes = "reward.txt" if failure_kind == "verifier_ambiguous_reward_files" else "reward.json"
    score = "0" if writes == "reward.txt" else '{"correctness": 0}'
    test_sh = f"echo '{score}' > {ROOT}/logs/verifier/{writes}; [ -f hook.sh ] && bash hook.sh; true"
    service = HarborEnvironmentService(
        task_dirs=[make_task("runs-agent-code", test_sh=test_sh)],
        provider=ProcessProvider(tmp_path / "sandboxes"),
        default_image="local",
        reward=LISTED,
    )
    session = await service.open(await service.get_task("runs-agent-code"), rollout_id="r")
    try:
        hook = f"echo '{planted}' > \"$PLATOON_SANDBOX_ROOT/logs/verifier/reward.json\"\n"
        assert not (await session.call_tool("write", {"path": "hook.sh", "content": hook})).is_error
        outcome = await session.grade()
    finally:
        await session.close()

    assert (outcome.status, outcome.failure_kind, outcome.reward) == ("invalid", failure_kind, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("what", ["workdir_missing", "reward_file_not_regular"])
async def test_a_zero_platoon_gives_records_the_contracts_first_name(tmp_path: Path, make_task, what: str):
    """An agent that removed its workdir, or code the tests ran that replaced a reward file with a symlink, scores 0
    without a reading: under ``[correctness, reward]`` that 0 is ``correctness``, the name the contract trains first."""

    one = tmp_path / "host-file"
    one.write_text("1\n")
    test_sh = f"echo 0 > {ROOT}/logs/verifier/reward.txt; [ -f hook.sh ] && bash hook.sh; true"
    service = HarborEnvironmentService(
        task_dirs=[make_task("zeroed", test_sh=test_sh)],
        provider=ProcessProvider(tmp_path / "sandboxes"),
        default_image="local",
        reward=["correctness", "reward"],
    )
    session = await service.open(await service.get_task("zeroed"), rollout_id="r")
    try:
        if what == "workdir_missing":
            done = await session.call_tool("bash", {"command": "cd / && rm -rf /app"})
        else:
            hook = f'ln -sfn {one} "$PLATOON_SANDBOX_ROOT/logs/verifier/reward.txt"\n'
            done = await session.call_tool("write", {"path": "hook.sh", "content": hook})
        assert not done.is_error
        outcome = await session.grade()
    finally:
        await session.close()

    assert (outcome.status, outcome.primary, outcome.rewards, outcome.reason) == (
        "scored",
        "correctness",
        {"correctness": 0.0},
        what,
    )
