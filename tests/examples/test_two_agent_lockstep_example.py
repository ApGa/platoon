"""The two-agent lockstep example (``examples/two-agent-lockstep``) trains as its README says, on CPU: both configs
validate, and ``platoon train`` runs with each in process, on ``platoon.testing``'s fake Tinker service (one per policy
for the second config), the subagents example's scripted policy sampling for both roles through the real gateway, each
role's rows at a logprob of its policy's own.  Both policies step in each step, each on its own rows only; each
checkpoint saves both and the run's row names the worker's; and a second ``platoon train`` resumes both from the last
checkpoint, while one that freezes the worker stops with exit code 78 before any Tinker session opens."""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import tinker
from test_subagents_example import tool_call, vault_policy
from test_subagents_frozen_role import _LLM, _info
from vault_env import goal_for, parse_keys

from platoon import cli
from platoon.gateway import core
from platoon.testing import ScriptedPolicy, make_gateway, serve_policy
from platoon.testing import trainers as checks
from platoon.testing.trainers import FakePolicyService
from platoon.train.policy import set as set_module

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "two-agent-lockstep"
SHARED, OWN = EXAMPLE / "train.yaml", EXAMPLE / "train-worker-service.yaml"
TRIALS = {SHARED: "delegator-and-worker", OWN: "delegator-and-worker-service"}
WORKER_URL = "http://localhost:8001"
MAIN_LOGPROB, WORKER_LOGPROB = -0.25, -0.875
_FIRST_USER = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
# How a delegator marks the subagents' goals of the first and the second rollout of a task in its group.
FIRST, SECOND = "(first rollout)", "(second rollout)"


def _goal(prompt: str) -> str:
    """An agent's goal: the first user message of its prompt."""

    found = _FIRST_USER.search(prompt)
    assert found is not None
    return found.group(1)


def delegator_policy():
    """The delegator's policy (main): the subagents example's, whose subagents' goals say which of the group's two
    rollouts of a task delegated them, so that the worker can answer the first rollout's wrong."""

    delegated: Counter[str] = Counter()

    def policy(prompt: str) -> str:
        reply = vault_policy(prompt)
        if '"name": "delegate"' not in reply:
            return reply
        goal = _goal(prompt)
        delegated[goal] += 1
        mark = FIRST if delegated[goal] % 2 else SECOND
        keys = parse_keys(goal)
        half = len(keys) // 2
        return "\n".join(tool_call("delegate", goal=f"{goal_for(part)} {mark}") for part in (keys[:half], keys[half:]))

    return policy


def worker_policy(prompt: str) -> str:
    """The worker's policy: the subagents example's, which answers one too many for the first rollout of a group, so
    that the agents of each role score unlike within each group, and both policies have something to learn."""

    reply = vault_policy(prompt)
    if FIRST in _goal(prompt) and '"name": "answer"' in reply:
        return tool_call("answer", total=json.loads(reply.split("\n")[1])["arguments"]["total"] + 1)
    return reply


# Numbers each start's services, so that each start's training runs have names of their own, as on a real service.
_STARTS = itertools.count(1)


class Services:
    """Fake Tinker services by URL, in ``tinker.ServiceClient``'s place: the run's (``None``) and, for the second
    config, the worker's own.  Each start's are new, their training runs named apart from any earlier start's."""

    def __init__(self, *urls: str | None) -> None:
        start = next(_STARTS)
        self.by_url = {url: FakePolicyService(name=f"{'worker' if url else 'main'}-{start}") for url in (None, *urls)}

    def service_client(self, **kwargs: Any) -> FakePolicyService:
        return self.by_url[kwargs.get("base_url")].service_client(**kwargs)

    def clients(self) -> list[Any]:
        return [client for service in self.by_url.values() for client in service.training_clients]


# Steps of 2 groups of 2 rollouts, with no evaluation and no watchdog.
SETTINGS = ("train.batch_size=2", "train.workflow_config.group_size=2", "eval.every=0", "watchdog.enabled=false")


def train(config: Path, log_path: Path, services: Services, *overrides: str) -> Path:
    """``platoon train`` on ``config`` on ``services``: steps of 2 groups of 2 rollouts, with no evaluation."""

    main = ScriptedPolicy(delegator_policy(), logprob=MAIN_LOGPROB)
    worker = ScriptedPolicy(worker_policy, logprob=WORKER_LOGPROB, tokenizer=main.tokenizer)

    def registered(model_name: str, renderer_name: str, **kwargs: Any) -> Any:
        return _info(_LLM(core.default_gateway(), model_name), model_name, renderer_name, **kwargs)

    def opened(model_name: str, renderer_name: str, **kwargs: Any) -> Any:
        tinker.ServiceClient(base_url=kwargs.get("base_url"))  # a session of its own, on its service
        gateway = make_gateway(worker, model_name=model_name)
        return _info(_LLM(gateway, model_name), model_name, renderer_name, **kwargs)

    settings = (*SETTINGS, f"log_path={log_path}", *overrides)
    with (
        serve_policy(main),
        mock.patch.object(tinker, "ServiceClient", services.service_client),
        mock.patch.object(set_module, "register_tinker_llm", registered),
        mock.patch.object(set_module, "open_tinker_llm", opened),
        checks._restored_signals(),
    ):
        cli.main(["train", str(config), *settings])
    return log_path / "subagents" / TRIALS[config]


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (path / "checkpoints.jsonl").read_text().splitlines()]


def sampled_at(client: Any) -> set[float]:
    """The logprobs that a training client's rows were sampled at, on their action tokens."""

    return {
        round(float(logprob), 3)
        for datum in client.datums
        for logprob, advantage in zip(
            datum.loss_fn_inputs["logprobs"].to_torch(), datum.loss_fn_inputs["advantages"].to_torch(), strict=True
        )
        if advantage != 0
    }


@pytest.fixture(autouse=True)
def _tokenizers():
    """Both policies' tokenizers as tinker-cookbook loads them by model name: the toy one, so nothing is downloaded."""

    from tinker_cookbook.tokenizer_utils import register_tokenizer, unregister_tokenizer

    from platoon.testing.tokenizer import toy_tokenizer

    register_tokenizer("Qwen/Qwen3-4B-Instruct-2507", toy_tokenizer)
    yield
    unregister_tokenizer("Qwen/Qwen3-4B-Instruct-2507")


def exit_code(*args: str) -> Any:
    try:
        return cli.main(list(args))
    except SystemExit as exc:
        return exc.code


def test_the_example_configs_validate(capsys):
    cli.main(["config", "validate", str(SHARED), str(OWN)])
    assert capsys.readouterr().out.count("ok") == 2


@pytest.mark.parametrize(("config", "worker_url"), [(SHARED, None), (OWN, WORKER_URL)])
def test_both_policies_step_each_step_save_together_and_resume(config, worker_url, tmp_path):
    services = Services(*([worker_url] if worker_url else []))
    run = train(config, tmp_path, services, "train.max_training_steps=2")

    # Two training clients, one per policy, each stepping in each step, on its own policy's rows only.
    main_client, worker_client = services.clients()
    assert (main_client.optimizer_steps, worker_client.optimizer_steps) == (2, 2)
    assert sampled_at(main_client) == {MAIN_LOGPROB} and sampled_at(worker_client) == {WORKER_LOGPROB}
    assert services.by_url[worker_url].training_clients[-1] is worker_client  # on the worker's service

    leader, worker = rows(run), rows(run / "policies" / "worker")
    assert [(row["name"], row["batch"]) for row in leader] == [("recovery-000000", 0), ("final", 2)]
    assert [row["policies"]["worker"]["state_path"] for row in leader] == [row["state_path"] for row in worker]
    identity = json.loads((run / "identity.json").read_text())
    assert identity["policies"]["worker"]["trainable"] is True
    assert identity["agents"] == {"policy": "main", "roles": {"subagent": "worker"}}

    # A second start resumes both from the last checkpoint, each from its own row, and steps both once more.
    again = Services(*([worker_url] if worker_url else []))
    train(config, tmp_path, again, "train.max_training_steps=3")
    opened = [call for service in again.by_url.values() for call in service.calls if "create_training" in call]
    assert sorted(opened) == sorted(
        f"create_training_client_from_state_with_optimizer {row['state_path']}" for row in (leader[-1], worker[-1])
    )
    assert [client.optimizer_steps for client in again.clients()] == [1, 1]
    assert [row["batch"] for row in rows(run)] == [0, 2, 2, 3]


def test_a_start_that_freezes_a_trained_policy_stops_before_any_session_opens(tmp_path, capsys):
    services = Services()
    train(SHARED, tmp_path, services, "train.max_training_steps=1")

    again = Services()
    frozen = (
        "policies.worker.trainable=false",
        *(f"policies.worker.{key}=null" for key in ("lora_rank", "optimizer", "max_staleness")),
    )
    with mock.patch.object(tinker, "ServiceClient", again.service_client):
        code = exit_code("train", str(SHARED), *SETTINGS, f"log_path={tmp_path}", *frozen)
    assert code == 78 and all(not service.sessions for service in again.by_url.values())
    assert "policies.worker.trainable: the run's is True, and this start's False" in capsys.readouterr().err
