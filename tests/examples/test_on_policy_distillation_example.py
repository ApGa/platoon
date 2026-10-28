"""The on-policy distillation example (``examples/on-policy-distillation``) trains as its README says, on CPU: both
configs validate, and ``platoon train`` runs with each in process on ``platoon.testing``'s fake Tinker service.  The
student is the quickstart's scripted bisection policy, sampled through the real gateway; the teacher is the fake
service's sampling client, which scores every token at ``SCORED_LOGPROB``.  The teacher opens a session of its own
(on its own URL in the second config), scores exactly the rows of the training groups, each action token's advantage
is the negative reverse KL, and the evaluation runs without calling it.  A teacher whose scoring call fails, runs out
of time or gives a response of the wrong length fails that group alone, and the run trains on; one of another
tokenizer, several sampled policies beside a workflow that binds one, or a teacher's sample timeout that is not
positive, stop ``platoon train`` and ``platoon config validate`` before any Tinker session opens, and a teacher whose
tokenizer could not be compared as the config loaded stops ``platoon train`` as it builds the estimator."""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OPENREWARD_DISABLE_UPDATE_CHECK", "1")
pytest.importorskip("openreward")

from test_quickstart import bisect  # noqa: E402

from platoon import cli  # noqa: E402
from platoon.testing import serve_policy  # noqa: E402
from platoon.testing import trainers as checks  # noqa: E402
from platoon.testing.policy import DEFAULT_LOGPROB  # noqa: E402
from platoon.train.credit.reverse_kl import ReverseKL  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
EXAMPLE = EXAMPLES / "on-policy-distillation"
SHARED, OWN = EXAMPLE / "train.yaml", EXAMPLE / "train-teacher-service.yaml"


@pytest.fixture(autouse=True)
def _as_the_readme_runs_it(monkeypatch):
    """With the quickstart's directory on the path, as the README's PYTHONPATH puts it."""

    monkeypatch.syspath_prepend(str(EXAMPLES / "quickstart"))


STUDENT_MODEL, TEACHER_MODEL = "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-32B"


@pytest.fixture(autouse=True)
def _tokenizers() -> Iterator[None]:
    """The student's and the teacher's tokenizers as tinker-cookbook loads them by model name, which loading compares:
    one tokenizer, as Qwen3 models share theirs (the toy one, so nothing is downloaded)."""

    from tinker_cookbook.tokenizer_utils import register_tokenizer, unregister_tokenizer

    from platoon.testing.tokenizer import toy_tokenizer

    register_tokenizer(STUDENT_MODEL, toy_tokenizer)
    register_tokenizer(TEACHER_MODEL, toy_tokenizer)
    yield
    unregister_tokenizer(STUDENT_MODEL)
    unregister_tokenizer(TEACHER_MODEL)


@pytest.fixture
def credited(monkeypatch) -> list[str]:
    """The task of every group the estimator credited, in order."""

    tasks: list[str] = []
    assign = ReverseKL.assign

    async def recorded(self: ReverseKL, group: Any) -> None:
        tasks.append(group.task_id)
        await assign(self, group)

    monkeypatch.setattr(ReverseKL, "assign", recorded)
    return tasks


def train(config: Path, log_path: Path, *overrides: str) -> tuple[checks.FakePolicyService, Path]:
    """``platoon train`` on ``config`` on the fake service: 2 steps of 2 groups of 2 rollouts, each step evaluated on
    two test tasks."""

    service = checks.FakePolicyService()
    settings = (
        f"log_path={log_path}",
        "train.batch_size=2",
        "train.max_training_steps=2",
        "train.workflow_config.group_size=2",
        "eval.every=1",
        "environments.0.eval_dataset_kwargs.limit=2",
        "watchdog.enabled=false",
        *overrides,
    )
    with serve_policy(bisect), checks._faked(service), checks._restored_signals():
        cli.main(["train", str(config), *settings])
    return service, log_path / "quickstart" / config.stem.replace("train", "guess-distilled")


def exit_code(*args: str) -> Any:
    """``platoon`` with ``args``: its exit code."""

    try:
        return cli.main(list(args))
    except SystemExit as exc:
        return exc.code


def _scored(datum: Any) -> list[int]:
    """The sequence the teacher scores for a sent datum: its prompt and its last target token."""

    return [*datum.model_input.to_ints(), int(datum.loss_fn_inputs["target_tokens"].to_torch()[-1])]


def test_the_example_configs_validate(capsys):
    cli.main(["config", "validate", str(SHARED), str(OWN)])
    assert capsys.readouterr().out.count("ok") == 2


@pytest.mark.parametrize(("config", "teacher_url"), [(SHARED, None), (OWN, "http://localhost:8001")])
def test_the_student_trains_on_the_teacher_s_reverse_kl(config, teacher_url, credited, tmp_path):
    service, run = train(config, tmp_path)

    rows = [json.loads(line) for line in (run / "checkpoints.jsonl").read_text().splitlines()]
    assert [(row["name"], row["batch"]) for row in rows] == [("recovery-000000", 0), ("final", 2)]
    # A session for the teacher, on its URL, as the trainer is built; then the student's training session.
    assert service.sessions == [teacher_url, None]
    assert {model for model, _ in service.scored} == {"Qwen/Qwen3-32B"}

    # The teacher scored every row the step sent, and each action token's advantage is the negative reverse KL
    # (the same at every token, the student's and the teacher's logprobs being constant), scaled as the step scales.
    scored = [tokens for _, tokens in service.scored]
    assert service.datums and all(_scored(datum) in scored for datum in service.datums)
    assert DEFAULT_LOGPROB > checks.SCORED_LOGPROB  # the student is surer than the teacher: every advantage < 0
    for datum in service.datums:
        advantages = {value for value in datum.loss_fn_inputs["advantages"].to_torch().tolist() if value != 0}
        assert len(advantages) == 1 and advantages.pop() < 0

    # Evaluation ran, and never called the teacher: only training groups were credited.
    assert len((run / "evaluations.jsonl").read_text().splitlines()) == 2
    assert credited and all(task.startswith("train:") for task in credited)
    identity = json.loads((run / "identity.json").read_text())
    assert identity["policies"]["teacher"] == {
        "model_name": "Qwen/Qwen3-32B",
        "renderer_name": "qwen3",
        "trainable": False,
        "load_checkpoint_path": None,
    }
    lock = json.loads((run / "lock.json").read_text())
    [record] = [record for record in lock["components"] if record["kind"] == "credit"]
    assert (record["ref"], record["options"]["guess"]["teacher"]) == ("reverse_kl", "teacher")


@pytest.fixture
def metrics(monkeypatch) -> list[dict[str, Any]]:
    """The metrics of every step the run logged, in order."""

    from platoon.utils.stats_logger import StatsLogger

    logged: list[dict[str, Any]] = []
    log = StatsLogger.log

    def recorded(self: StatsLogger, step: int, stats: dict[str, float], *args: Any, **kwargs: Any) -> None:
        logged.append(dict(stats))
        log(self, step, stats, *args, **kwargs)

    monkeypatch.setattr(StatsLogger, "log", recorded)
    return logged


def _failures(metrics: list[dict[str, Any]]) -> list[float]:
    return [value for stats in metrics for key, value in stats.items() if key.endswith("credit/failures__count")]


def test_a_failed_scoring_call_fails_its_group_and_the_run_trains_on(tmp_path, monkeypatch, metrics):
    calls = {"n": 0}
    score = checks._Scorer.compute_logprobs_async

    async def flaky(self: Any, prompt: Any) -> list[float | None]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("teacher service: 503 Service Unavailable")
        return await score(self, prompt)

    monkeypatch.setattr(checks._Scorer, "compute_logprobs_async", flaky)
    service, run = train(SHARED, tmp_path)

    rows = [json.loads(line)["name"] for line in (run / "checkpoints.jsonl").read_text().splitlines()]
    assert rows == ["recovery-000000", "final"] and service.optimizer_steps > 0
    assert _failures(metrics) == [1]  # one group failed, in one step


def test_a_teacher_that_does_not_answer_in_time_fails_its_group_and_the_run_trains_on(tmp_path, monkeypatch, metrics):
    calls = {"n": 0}
    score = checks._Scorer.compute_logprobs_async

    async def slow_once(self: Any, prompt: Any) -> list[float | None]:
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.sleep(30)
        return await score(self, prompt)

    monkeypatch.setattr(checks._Scorer, "compute_logprobs_async", slow_once)
    service, run = train(SHARED, tmp_path, "policies.teacher.sample_timeout_seconds=0.5")

    rows = [json.loads(line)["name"] for line in (run / "checkpoints.jsonl").read_text().splitlines()]
    assert rows == ["recovery-000000", "final"] and service.optimizer_steps > 0
    assert _failures(metrics) == [1]


@pytest.mark.parametrize("kind", ["two entries", "one short", "one long"])
def test_a_teacher_response_of_the_wrong_length_fails_its_group_and_the_run_trains_on(
    kind, tmp_path, monkeypatch, metrics
):
    calls = {"n": 0}

    async def malformed(self: Any, prompt: Any) -> list[float | None]:
        calls["n"] += 1
        length = len(prompt.to_ints())  # a logprob per token of the sequence scored, the first None
        if calls["n"] == 1:
            length = {"two entries": 2, "one short": length - 1, "one long": length + 1}[kind]
            return [None] + [-7.0] * (length - 1)
        return [None] + [checks.SCORED_LOGPROB] * (length - 1)

    monkeypatch.setattr(checks._Scorer, "compute_logprobs_async", malformed)
    service, run = train(SHARED, tmp_path)

    rows = [json.loads(line)["name"] for line in (run / "checkpoints.jsonl").read_text().splitlines()]
    assert rows == ["recovery-000000", "final"] and service.optimizer_steps > 0
    assert _failures(metrics) == [1]  # that group failed, never trained on a misaligned (or broadcast) score


@pytest.mark.parametrize("seconds", ["-1", "0"])
def test_a_teacher_s_sample_timeout_that_is_not_positive_stops_validate_and_train(seconds, tmp_path, capsys):
    key = f"policies.teacher.sample_timeout_seconds={seconds}"
    assert exit_code("config", "validate", str(SHARED), key) == 78
    assert "the sample_timeout_seconds of the policy of Qwen/Qwen3-32B must be a positive number of seconds" in (
        capsys.readouterr().out
    )

    service = checks.FakePolicyService()
    with serve_policy(bisect), checks._faked(service), checks._restored_signals():
        code = exit_code("train", str(SHARED), f"log_path={tmp_path}", "watchdog.enabled=false", key)
    assert code == 78 and service.sessions == []  # as train.sample_timeout_seconds: before any Tinker session


@pytest.fixture
def other_teacher_tokenizer() -> None:
    """The teacher's model with a tokenizer of one token more than the student's."""

    from tinker_cookbook.tokenizer_utils import register_tokenizer
    from transformers import PreTrainedTokenizerFast

    from platoon.testing.tokenizer import EOS_TOKEN, TOY_TOKENIZER_DIR

    other: Any = PreTrainedTokenizerFast(tokenizer_file=str(TOY_TOKENIZER_DIR / "tokenizer.json"), eos_token=EOS_TOKEN)
    other.add_tokens(["distill"])
    register_tokenizer(TEACHER_MODEL, lambda: other)  # _tokenizers unregisters it


def test_a_teacher_of_another_tokenizer_stops_the_run_before_any_session_opens(
    tmp_path, capsys, other_teacher_tokenizer
):
    assert exit_code("config", "validate", str(SHARED)) == 78
    assert (
        "teacher: policies.teacher's tokenizer (Qwen/Qwen3-32B) differs from the student's" in capsys.readouterr().out
    )

    service = checks.FakePolicyService()
    with serve_policy(bisect), checks._faked(service), checks._restored_signals():
        code = exit_code("train", str(SHARED), f"log_path={tmp_path}", "watchdog.enabled=false")
    assert code == 78 and service.sessions == []  # no Tinker session opened
    assert not (tmp_path / "quickstart" / "guess-distilled" / "identity.json").exists()


def test_a_teacher_whose_tokenizer_was_not_compared_at_load_stops_the_run_as_the_estimator_is_built(
    tmp_path, capsys, monkeypatch, other_teacher_tokenizer
):
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from platoon.train.credit import reverse_kl

    # As when a tokenizer could not be loaded as the config loaded (no network): loading compares nothing.
    monkeypatch.setattr(reverse_kl, "tokenizer_problem", lambda student_model, teacher_model: None)
    # Each policy's fake model gateway, main's and the teacher's, has the tokenizer the run loads for its model, as a
    # real one has (the fake's own is the toy tokenizer).
    monkeypatch.setattr(checks._FakeLLM, "tokenizer", property(lambda llm: get_tokenizer(llm.model_name)))
    assert exit_code("config", "validate", str(SHARED)) in (0, None)
    capsys.readouterr()

    service = checks.FakePolicyService()
    with serve_policy(bisect), checks._faked(service), checks._restored_signals():
        code = exit_code("train", str(SHARED), f"log_path={tmp_path}", "watchdog.enabled=false")
    assert code == 78
    assert "policies.teacher's tokenizer, as the run loaded it, differs from the student's" in capsys.readouterr().err
    assert service.optimizer_steps == 0 and service.datums == []  # nothing trained


def test_validate_refuses_several_sampled_policies_beside_a_workflow_that_binds_one(tmp_path, capsys, monkeypatch):
    (tmp_path / "one_policy_workflows.py").write_text(
        textwrap.dedent(
            """
            from platoon.train.workflow.group import GroupRolloutWorkflow

            class OnePolicy(GroupRolloutWorkflow):
                binds_policies = False
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    judge = [
        "policies.judge.model_name=Qwen/Qwen3-32B",
        "policies.judge.renderer_name=qwen3",
        "policies.judge.trainable=false",
    ]
    bound = [*judge, "agents.roles.judge.policy=judge"]

    assert (
        exit_code("config", "validate", str(SHARED), *bound, "environments.0.workflow=one_policy_workflows:OnePolicy")
        == 78
    )
    assert "agents: the agents sample several policies (main, judge)" in capsys.readouterr().out
    # A frozen policy that nothing uses is only warned of.
    assert exit_code("config", "validate", str(SHARED), *judge) in (0, None)
    assert "warning: policies.judge: no agent samples it" in capsys.readouterr().out
