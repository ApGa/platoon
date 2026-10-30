"""The two-agent independent example (``examples/two-agent-independent``) trains as its README says, on CPU: its config
validates, and ``platoon train`` runs it in process on ``platoon.testing``'s fake Tinker service, the subagents
example's scripted policy sampling for both roles through the real gateway.  The two policies step each on its own
clock, the worker about four times for each of main's steps, each saving its own rows, each full save a commit; a run
killed in the middle resumes from its newest commit, and no policy's weights train its share of a group twice; and a
start that steps the run's policies in lockstep instead stops with exit code 78 before any Tinker session opens."""

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
from platoon.testing.trainers import FakePolicyService, _Done, _FakeTrainingClient
from platoon.train.checkpoints import commits as commits_module
from platoon.train.policy import set as set_module

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "two-agent-independent"
CONFIG = EXAMPLE / "train.yaml"
MAIN_LOGPROB, WORKER_LOGPROB = -0.25, -0.875
# Steps of 4 groups of 2 rollouts for main, of 1 group for the worker, with no evaluation and no watchdog.
SETTINGS = (
    "train.batch_size=4",
    "train.workflow_config.group_size=2",
    "policies.worker.batch_size=1",
    "policies.worker.checkpoint.recovery_every=2",
    "eval.every=0",
    "watchdog.enabled=false",
)


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


class _Lineage(_FakeTrainingClient):
    """A training client that remembers what it trained (``trained``: each row's tokens), going on with the lineage of
    the state it was opened from, which each of its saves remembers."""

    def __init__(self, service: _Service, run: str, trained: list[tuple[int, ...]]) -> None:
        super().__init__(service, run)
        self.lineages = service.lineages
        self.trained: list[tuple[int, ...]] = list(trained)
        self._pending: list[tuple[int, ...]] = []

    async def forward_backward_async(self, datums: list[Any], loss_fn: str, loss_fn_config: Any = None) -> _Done:
        self._pending.extend(tuple(datum.model_input.to_ints()) for datum in datums)
        return await super().forward_backward_async(datums, loss_fn, loss_fn_config)

    async def optim_step_async(self, params: Any) -> _Done:
        self.trained += self._pending
        self._pending = []
        return await super().optim_step_async(params)

    async def save_state_async(self, name: str, ttl_seconds: int | None = None) -> _Done:
        done = await super().save_state_async(name, ttl_seconds)
        self.lineages[done.value.path] = list(self.trained)
        return done


class _Service(FakePolicyService):
    """A fake service whose training clients remember their lineage, across starts (``lineages``)."""

    def __init__(self, lineages: dict[str, list[tuple[int, ...]]], name: str) -> None:
        super().__init__(name=name)
        self.lineages = lineages

    def _open(self, call: str) -> _FakeTrainingClient:
        self.calls.append(call)
        trained = self.lineages.get(call.split()[-1], []) if "from_state" in call else []
        client = _Lineage(self, f"{self.name}-{len(self.training_clients) + 1}", trained)
        self.training_clients.append(client)
        return client


_STARTS = itertools.count(1)


def train(log_path: Path, lineages: dict[str, list[tuple[int, ...]]], *overrides: str) -> _Service:
    """``platoon train`` on the example on one fake service (each policy in a session of its own), whose clients
    remember their lineage."""

    service = _Service(lineages, f"run{next(_STARTS)}")
    main = ScriptedPolicy(delegator_policy(), logprob=MAIN_LOGPROB)
    worker = ScriptedPolicy(worker_policy, logprob=WORKER_LOGPROB, tokenizer=main.tokenizer)

    def registered(model_name: str, renderer_name: str, **kwargs: Any) -> Any:
        return _info(_LLM(core.default_gateway(), model_name), model_name, renderer_name, **kwargs)

    def opened(model_name: str, renderer_name: str, **kwargs: Any) -> Any:
        tinker.ServiceClient(base_url=kwargs.get("base_url"))  # a session of its own
        return _info(_LLM(make_gateway(worker, model_name=model_name), model_name), model_name, renderer_name, **kwargs)

    with (
        serve_policy(main),
        mock.patch.object(tinker, "ServiceClient", service.service_client),
        mock.patch.object(set_module, "register_tinker_llm", registered),
        mock.patch.object(set_module, "open_tinker_llm", opened),
        checks._restored_signals(),
    ):
        cli.main(["train", str(CONFIG), *SETTINGS, f"log_path={log_path}", *overrides])
    return service


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (path / "checkpoints.jsonl").read_text().splitlines()]


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


def test_the_example_config_validates(capsys):
    cli.main(["config", "validate", str(CONFIG)])
    assert "ok" in capsys.readouterr().out


class Killed(BaseException):
    """The process died."""


def test_each_policy_steps_on_its_own_clock_and_a_killed_run_resumes_training_no_share_twice(tmp_path, monkeypatch):
    lineages: dict[str, list[tuple[int, ...]]] = {}
    run = tmp_path / "subagents" / "delegator-and-worker-independent"
    appended = commits_module.Commits._appended

    def kill_at_the_worker_s_second_commit(self: Any, policy: str, own: Any, others: Any) -> None:
        appended(self, policy, own, others)
        if policy == "worker" and own["step"] == 4:
            raise Killed()

    with monkeypatch.context() as patched:
        patched.setattr(commits_module.Commits, "_appended", kill_at_the_worker_s_second_commit)
        with pytest.raises(BaseException):
            train(tmp_path, lineages, "train.max_training_steps=3")
    commits = [row for row in rows(run / "policies" / "worker") if "commit_seq" in row]
    assert [row["policies"]["worker"]["step"] for row in commits] == [2, 4]

    service = train(tmp_path, lineages, "train.max_training_steps=3")
    main, worker = service.training_clients  # main's client opened first, then the worker's
    # main took its 4 groups a step; the worker steps on each of its rows' groups, on its own clock.
    leader = rows(run)
    assert leader[-1]["name"] == "final" and leader[-1]["policies"]["main"]["step"] == 3
    assert leader[-1]["policies"]["worker"]["step"] == 12  # 12 groups, one of the worker's steps each
    own = [row for row in rows(run / "policies" / "worker") if "commit_seq" in row]
    assert [row["policies"]["worker"]["step"] for row in own][-1] == 12
    # No policy's weights trained a row twice, killed and resumed.
    for path in (run, run / "policies" / "worker"):
        trained = lineages[rows(path)[-1]["state_path"]]
        assert trained and all(count == 1 for count in Counter(trained).values()), path
    assert main.optimizer_steps >= 1 and worker.optimizer_steps >= 1


def test_a_resume_that_steps_the_run_s_policies_in_lockstep_stops_before_any_session_opens(tmp_path, capsys):
    train(tmp_path, {}, "train.max_training_steps=1")
    again = _Service({}, "again")
    lockstep = ("train.stepping=lockstep", "policies.worker.batch_size=null", "policies.worker.checkpoint=null")
    settings = (*SETTINGS[:2], *SETTINGS[4:])
    with mock.patch.object(tinker, "ServiceClient", again.service_client):
        code = exit_code("train", str(CONFIG), *settings, f"log_path={tmp_path}", *lockstep)
    assert code == 78 and not again.sessions
    assert "train.stepping: the run's is 'independent', and this start's 'lockstep'" in capsys.readouterr().err
