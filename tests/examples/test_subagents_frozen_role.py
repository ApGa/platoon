"""A role bound to a frozen policy, end to end in process: ``platoon train`` on the subagents example, on the fake
Tinker service, with its ``subagent`` role bound to a frozen policy ``helper`` (``agents.roles.subagent.policy``).  The
trainer's own ``PolicySet.build`` serves both policies through the process's gateway: the helper samples every
subagent call, and no row it sampled is converted to a datum or reaches ``forward_backward`` (its sampled logprob is
distinctive), while main's rows train.  With ``credit: reverse_kl`` and a frozen teacher too, the teacher scores the
student's multi-turn tool-call rows, and each action token's advantage is the negative reverse KL at that token."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_subagents_example import tool_call, vault_policy

from platoon import cli
from platoon.gateway import core
from platoon.testing import ScriptedPolicy, make_gateway
from platoon.testing import trainers as checks
from platoon.train.credit.reverse_kl import ReverseKL
from platoon.train.policy import set as set_module
from platoon.train.workers import run as worker_run

CONFIG = Path(__file__).resolve().parents[2] / "examples" / "subagents" / "subagents.yaml"
MAIN_LOGPROB, HELPER_LOGPROB = -0.25, -0.875
_FIRST_USER = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def teacher_logprob(position: int, token: int) -> float:
    """The teacher's logprob of ``token`` at ``position`` of a scored sequence, varied so that misalignment shows."""

    return -0.001 * ((token * 7 + position * 13) % 997) - 0.01


class _Teacher:
    """The teacher's sampling client: it scores each sequence it is asked for with :func:`teacher_logprob`."""

    def __init__(self) -> None:
        self.asked: list[list[int]] = []

    async def compute_logprobs_async(self, prompt: Any) -> list[float | None]:
        tokens = prompt.to_ints()
        self.asked.append(tokens)
        return [None] + [teacher_logprob(position, tokens[position]) for position in range(1, len(tokens))]


def one_right_root_a_group():
    """The vault policy, whose every other root answers one too many, so that each group's rewards differ."""

    answers: Counter[str] = Counter()

    def policy(prompt: str) -> str:
        reply = vault_policy(prompt)
        goal = _FIRST_USER.search(prompt)
        if '"name": "answer"' in reply and "<tool_response>\nAnswered" in prompt and goal is not None:
            answers[goal.group(1)] += 1
            if answers[goal.group(1)] % 2:
                return tool_call("answer", total=json.loads(reply.split("\n")[1])["arguments"]["total"] + 1)
        return reply

    return policy


class _LLM:
    """A policy's ``TinkerLLM`` over a scripted gateway, with its version, as the fake service's trainer reads it."""

    def __init__(self, gateway: Any, model_name: str, sampling_client: Any = None) -> None:
        self.gateway, self.model_name, self.version = gateway, model_name, 0
        self._sampling_client = sampling_client

    @property
    def sampling_client(self) -> Any:
        return self._sampling_client if self._sampling_client is not None else self.gateway.sampler

    @property
    def renderer(self) -> Any:
        return self.gateway.renderer

    @property
    def tokenizer(self) -> Any:
        return self.gateway.tokenizer

    def set_version(self, version: int) -> None:
        self.version = version

    def update_sampling_client(self, sampling_client: Any, increment_version: bool = True) -> None:
        if increment_version:
            self.version += 1


def _info(llm: _LLM, model_name: str, renderer_name: str, **kwargs: Any) -> Any:
    renderer_kwargs = kwargs.get("renderer_kwargs") or {}
    return SimpleNamespace(llm=llm, model_name=model_name, renderer_name=renderer_name, renderer_kwargs=renderer_kwargs)


@pytest.fixture
def tokenizers() -> Iterator[None]:
    """The models' tokenizers as tinker-cookbook loads them by model name: the toy one, so nothing is downloaded."""

    from tinker_cookbook.tokenizer_utils import register_tokenizer, unregister_tokenizer

    from platoon.testing.tokenizer import toy_tokenizer

    names = ("Qwen/Qwen3-4B-Instruct-2507", "teacher-model")
    for name in names:
        register_tokenizer(name, toy_tokenizer)
    yield
    for name in names:
        unregister_tokenizer(name)


@pytest.mark.parametrize("credit", ["group", "reverse_kl"])
def test_a_frozen_role_samples_and_trains_nothing(serve_policy, tmp_path, monkeypatch, credit, tokenizers):
    main_policy = serve_policy(ScriptedPolicy(one_right_root_a_group(), logprob=MAIN_LOGPROB))
    helper_policy = ScriptedPolicy(vault_policy, logprob=HELPER_LOGPROB, tokenizer=main_policy.tokenizer)
    teacher = _Teacher()

    def registered(model_name: str, renderer_name: str, **kwargs: Any) -> Any:
        return _info(_LLM(core.default_gateway(), model_name), model_name, renderer_name, **kwargs)

    def opened(model_name: str, renderer_name: str, **kwargs: Any) -> Any:
        if model_name == "teacher-model":
            gateway = make_gateway(ScriptedPolicy(vault_policy, tokenizer=main_policy.tokenizer), model_name=model_name)
            return _info(_LLM(gateway, model_name, teacher), model_name, renderer_name, **kwargs)
        return _info(_LLM(make_gateway(helper_policy, model_name=model_name), model_name), model_name, renderer_name)

    conversions: list[Any] = []
    convert = worker_run.get_train_data_for_trajectory_collection

    def recorded(*args: Any, **kwargs: Any) -> Any:
        result = convert(*args, **kwargs)
        conversions.append((kwargs.get("policies"), result))
        return result

    monkeypatch.setattr(worker_run, "get_train_data_for_trajectory_collection", recorded)
    credited: list[tuple[list[Any], list[Any]]] = []
    assign = ReverseKL.assign

    async def assigned(self: ReverseKL, group: Any) -> None:
        before = [copy.deepcopy(sample.datum) for sample in group.samples()]
        await assign(self, group)
        credited.append((before, [copy.deepcopy(sample.datum) for sample in group.samples()]))

    monkeypatch.setattr(ReverseKL, "assign", assigned)
    service = checks.FakePolicyService()
    settings = [
        f"log_path={tmp_path}",
        "train.batch_size=2",
        "train.max_training_steps=2",
        "train.workflow_config.group_size=2",
        "eval.every=0",
        "watchdog.enabled=false",
        "policies.helper.model_name=helper-model",
        "policies.helper.renderer_name=qwen3_instruct",
        "policies.helper.trainable=false",
        "agents.roles.subagent.policy=helper",
    ]
    if credit == "reverse_kl":
        settings += [
            "policies.teacher.model_name=teacher-model",
            "policies.teacher.renderer_name=qwen3_instruct",
            "policies.teacher.trainable=false",
            "environments.0.credit=reverse_kl",
            "environments.0.credit_options.teacher=teacher",
        ]
    with checks._faked(service), checks._restored_signals():
        monkeypatch.setattr(set_module, "register_tinker_llm", registered)
        monkeypatch.setattr(set_module, "open_tinker_llm", opened)
        cli.main(["train", str(CONFIG), *settings])

    run = tmp_path / "subagents" / "delegator"
    rows = [json.loads(line) for line in (run / "checkpoints.jsonl").read_text().splitlines()]
    assert [(row["name"], row["batch"]) for row in rows] == [("recovery-000000", 0), ("final", 2)]
    assert helper_policy.calls, "the frozen helper sampled nothing: the role binding did not take"
    # Only main's trajectories make datums; the helper's are excluded, and none reaches forward_backward.
    converted = [datum for _, result in conversions for datum in result.datums]
    assert converted and not any(
        bool((d.loss_fn_inputs["logprobs"].to_torch() == HELPER_LOGPROB).any()) for d in converted
    )
    assert sum(result.num_policy_excluded_datums for _, result in conversions) > 0
    assert all(policies is not None and policies.trained == frozenset({"main"}) for policies, _ in conversions)
    sent = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in service.datums]
    assert sent and not any(bool((logprobs == HELPER_LOGPROB).any()) for logprobs in sent)
    assert json.loads((run / "identity.json").read_text())["agents"] == {
        "policy": "main",
        "roles": {"subagent": "helper"},
    }
    if credit != "reverse_kl":
        return
    # The teacher scored each credited row's own sequence, and each action token's advantage is the negative reverse
    # KL at that token; every other token keeps 0.
    assert credited
    for before, after in credited:
        for row, scored in zip(before, after, strict=True):
            targets = row.loss_fn_inputs["target_tokens"].to_torch().tolist()
            assert [*row.model_input.to_ints(), targets[-1]] in teacher.asked
            student, mask = row.loss_fn_inputs["logprobs"].to_torch(), row.loss_fn_inputs["mask"].to_torch()
            advantages = scored.loss_fn_inputs["advantages"].to_torch()
            for position, target in enumerate(targets):
                expected = teacher_logprob(position + 1, target) - float(student[position]) if mask[position] > 0 else 0
                assert float(advantages[position]) == pytest.approx(expected, abs=1e-5)
