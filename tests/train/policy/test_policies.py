"""Several policies in one run (``policies:``, ``agents:``; RFC 0021): what loading reads, refuses (a policy's
``sample_timeout_seconds`` as ``train:``'s, and both keys under ``trainer: sft``) and warns of, how the run's policies
are built (each policy on a Tinker session of its own, a frozen one sampling the sampler it serves, the agents' default
registered as the process's gateway and serving the others to the roles bound to them, an error opening one naming it
and its service), a frozen policy's scores of other rows (bounded per policy and per call, a response of the wrong
length or a missing logprob refused, a failed call cancelling the rows' others), the identity a resume checks, and the
bindings rollout workers follow."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tinker
import torch

from platoon.train import identity
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.credit.api import ScoringFailed
from platoon.train.credit.registry import policy_views
from platoon.train.policy import policy as policy_module
from platoon.train.policy import set as set_module
from platoon.train.policy.policy import Policy
from platoon.train.policy.set import PolicySet
from platoon.train.policy.spec import (
    MAIN,
    AgentBindings,
    PolicySpec,
    agent_bindings,
    policy_specs,
    policy_warnings,
    workflow_problems,
)
from platoon.train.workers import protocol
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.utils.config import ConfigError, build_config

SETTINGS: dict[str, Any] = {
    "train": {
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "renderer_name": "qwen3_instruct",
        "optimizer": {"learning_rate": 1e-5},
        "loss_fn": "importance_sampling",
    },
    "tinker_base_url": "http://student:8000",
    "eval": {"every": 0},
    "log_path": "logs",
}
TEACHER = {"model_name": "Qwen/Qwen3-32B", "renderer_name": "qwen3", "trainable": False}


def config(**sections: Any) -> PlatoonTinkerRLTrainerConfig:
    return build_config(PlatoonTinkerRLTrainerConfig, {**copy.deepcopy(SETTINGS), **sections})


def problems(**sections: Any) -> list[str]:
    with pytest.raises(ConfigError) as raised:
        config(**sections)
    return raised.value.problems


# --- loading ----------------------------------------------------------------------------------------------------------


def test_without_policies_or_agents_the_run_has_main_alone():
    loaded = config()
    assert list(policy_specs(loaded)) == [MAIN]
    assert agent_bindings(loaded) == AgentBindings(default=MAIN, roles={})
    assert identity.policy_table(loaded) == {}  # identity.json as it was


def test_a_policy_of_policies_uses_train_s_names_and_the_run_s_service_and_settings_by_default():
    specs = policy_specs(config(policies={"teacher": TEACHER, "judge": {**TEACHER, "tinker_base_url": "http://j:9"}}))

    assert list(specs) == [MAIN, "teacher", "judge"]
    teacher = specs["teacher"]
    assert (teacher.model_name, teacher.renderer_name, teacher.trainable) == ("Qwen/Qwen3-32B", "qwen3", False)
    assert teacher.tinker_base_url == "http://student:8000"  # the run's service, in a session of its own
    assert specs["judge"].tinker_base_url == "http://j:9"


def test_agents_bind_roles_to_policies_and_the_root_samples_main():
    bindings = agent_bindings(config(policies={"judge": TEACHER}, agents={"roles": {"judge": {"policy": "judge"}}}))

    assert bindings == AgentBindings(default=MAIN, roles={"judge": "judge"})
    assert bindings.sampled() == (MAIN, "judge")


@pytest.mark.parametrize(
    ("sections", "start"),
    [
        ({"policies": {"main": TEACHER}}, "policies.main: main is the policy train: declares"),
        ({"policies": {"Teacher": TEACHER}}, "policies.Teacher: a policy id is a lowercase letter"),
        ({"agents": {"policy": "nobody"}}, "agents.policy: 'nobody' is not a policy of the run"),
        ({"policies": {"teacher": TEACHER}, "agents": {"policy": "teacher"}}, "agents.policy: 'teacher' is frozen"),
        ({"agents": {"roles": {"judge": {"policy": "judge"}}}}, "agents.roles.judge.policy: 'judge' is not a policy"),
    ],
)
def test_loading_refuses_policies_and_agents_it_cannot_run(sections, start):
    found = problems(**sections)
    assert any(problem.startswith(start) for problem in found), found


SFT_POLICIES = "policies: only the rl trainer samples or scores with other policies, and this run trains with sft: "
SFT_AGENTS = "agents: only the rl trainer's agents sample policies, and this run trains with sft: leave it out"


@pytest.mark.parametrize(
    ("sections", "refused"),
    [
        ({"policies": {"judge": TEACHER}}, [SFT_POLICIES + "leave it out"]),
        ({"agents": {"policy": MAIN}}, [SFT_AGENTS]),
        (
            {"policies": {"judge": TEACHER}, "agents": {"roles": {"judge": {"policy": "judge"}}}},
            [SFT_POLICIES + "leave it out", SFT_AGENTS],
        ),
    ],
)
def test_the_sft_trainer_refuses_policies_and_agents_which_only_rl_samples_or_scores_with(sections, refused):
    settings = copy.deepcopy(SETTINGS)
    del settings["train"]["loss_fn"]  # a key only rl reads
    settings |= {"trainer": "sft", "trainer_options": {"dataset": "rows", "num_epochs": 1}, **sections}

    with pytest.raises(ConfigError) as raised:
        build_config(PlatoonTinkerRLTrainerConfig, settings)

    assert raised.value.problems == refused


def test_several_sampled_policies_beside_a_workflow_that_binds_one_stop_the_run():
    loaded = config(policies={"judge": TEACHER}, agents={"roles": {"judge": {"policy": "judge"}}})

    class OnePolicy:
        async def arun_episode(self, data: dict) -> list:
            return []

    assert workflow_problems(loaded, GroupRolloutWorkflow, "default") == []
    [problem] = workflow_problems(loaded, OnePolicy, "acme:OnePolicy")
    assert problem.startswith("agents: the agents sample several policies (main, judge), and the run's workflow")
    assert workflow_problems(config(policies={"teacher": TEACHER}), OnePolicy, "acme:OnePolicy") == []


@pytest.mark.parametrize("seconds", [-1, 0, float("nan"), float("inf"), True, "30"])
def test_a_policy_s_sample_timeout_is_refused_as_train_s_is(seconds):
    with pytest.raises(ValueError, match=r"train.sample_timeout_seconds must be a positive number of seconds or null"):
        config(train={**SETTINGS["train"], "sample_timeout_seconds": seconds})
    with pytest.raises(
        ValueError,
        match=r"^policies\.teacher: sample_timeout_seconds must be a positive number of seconds or null, got ",
    ):
        config(policies={"teacher": {**TEACHER, "sample_timeout_seconds": seconds}})


def test_a_policy_s_sample_timeout_may_be_any_positive_number_of_seconds():
    for seconds in (0.5, 30, None):
        loaded = config(policies={"teacher": {**TEACHER, "sample_timeout_seconds": seconds}})
        assert policy_specs(loaded)["teacher"].sample_timeout_seconds == seconds


def test_a_frozen_policy_nothing_uses_is_warned_of():
    loaded = config(policies={"teacher": TEACHER, "judge": TEACHER}, agents={"roles": {"judge": {"policy": "judge"}}})
    [line] = policy_warnings(loaded)
    assert line.startswith("policies.teacher: no agent samples it")


def test_a_policy_bound_to_the_root_agent_s_role_is_warned_of():
    loaded = config(policies={"teacher": TEACHER}, agents={"roles": {"policy": {"policy": "teacher"}}})
    [line] = policy_warnings(loaded)
    assert line.startswith("agents.roles.policy.policy: the root agent's role is policy, and the root samples")
    assert policy_warnings(config(agents={"roles": {"policy": {"policy": MAIN}}})) == []


# --- building the run's policies --------------------------------------------------------------------------------------


class _Gateway:
    def __init__(self) -> None:
        self.served: list[Any] = []

    def serve_policies(self, policy: str, *, runtimes: dict, roles: dict) -> None:
        self.served.append((policy, runtimes, roles))


def _served(policies: PolicySet) -> list[Any]:
    """What main's (fake) gateway was asked to serve."""

    gateway: Any = policies[MAIN].info.llm.gateway
    return gateway.served


def _info(model: str) -> Any:
    llm = SimpleNamespace(version=0, gateway=_Gateway(), sampling_client=f"sampler of {model}", renderer=None)
    return SimpleNamespace(llm=llm, model_name=model)


@pytest.fixture
def opened(monkeypatch) -> dict[str, list]:
    """The policies' model infos, faked: the calls that registered the default one and opened the others."""

    calls: dict[str, list] = {"registered": [], "opened": []}

    def registered(*args: Any, **kwargs: Any) -> Any:
        calls["registered"].append(args)
        return _info(args[0])

    def opened(*args: Any, **kwargs: Any) -> Any:
        calls["opened"].append((args, kwargs))
        return _info(args[0])

    monkeypatch.setattr(set_module, "register_tinker_llm", registered)
    monkeypatch.setattr(set_module, "open_tinker_llm", opened)
    return calls


def test_a_frozen_teacher_opens_its_own_session_and_serves_no_agent(opened):
    teacher = {**TEACHER, "load_checkpoint_path": "tinker://teacher/sampler_weights/final"}
    loaded = config(policies={"teacher": teacher})
    policies = PolicySet.build(policy_specs(loaded), agent_bindings(loaded))

    assert opened["registered"] == [("Qwen/Qwen3-4B-Instruct-2507", "qwen3_instruct")]  # main: today's call
    [(args, kwargs)] = opened["opened"]
    assert args == ("Qwen/Qwen3-32B", "qwen3")
    assert (kwargs["base_url"], kwargs["model_path"]) == (
        "http://student:8000",
        "tinker://teacher/sampler_weights/final",
    )
    assert policies.leader == MAIN and [policy.id for policy in policies.trainable()] == [MAIN]
    assert [policy.id for policy in policies.sampled()] == [MAIN]
    assert policies["teacher"].sampling_client == "sampler of Qwen/Qwen3-32B"
    assert _served(policies) == []  # a teacher only scores: no agent samples it


def test_a_role_s_frozen_policy_is_served_by_the_default_gateway(opened):
    loaded = config(policies={"judge": TEACHER}, agents={"roles": {"judge": {"policy": "judge"}}})
    policies = PolicySet.build(policy_specs(loaded), agent_bindings(loaded))

    [(policy, runtimes, roles)] = _served(policies)
    assert (policy, roles) == (MAIN, {"judge": "judge"})
    assert runtimes == {"judge": policies["judge"].info.llm.gateway}
    assert [p.id for p in policies.sampled()] == [MAIN, "judge"]


def test_only_the_trained_policy_connects_and_opens(opened, monkeypatch):
    loaded = config(policies={"teacher": TEACHER})
    policies = PolicySet.build(policy_specs(loaded), agent_bindings(loaded))
    sessions: list[Any] = []

    def session(**kwargs: Any) -> str:
        sessions.append(kwargs)
        return "session"

    monkeypatch.setattr(tinker, "ServiceClient", session)

    policies.connect()
    assert sessions == [{"base_url": "http://student:8000"}]
    opening: list[str] = []

    async def open_policy(self: Policy, start: Any, *, stop: Any = None) -> bool:
        opening.append(self.id)
        return True

    monkeypatch.setattr(Policy, "open", open_policy)
    start: Any = SimpleNamespace(policies={MAIN: None})
    assert asyncio.run(policies.open(start)) and opening == [MAIN]


def test_an_error_opening_a_policy_names_it_and_its_service(monkeypatch):
    from platoon.eval.config_defs import TinkerKeyRefusedError

    def unreachable(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("Connection error.")

    monkeypatch.setattr(set_module, "register_tinker_llm", lambda *args, **kwargs: _info(args[0]))
    monkeypatch.setattr(set_module, "open_tinker_llm", unreachable)
    teacher = {**TEACHER, "tinker_base_url": "http://teacher:8001"}
    loaded = config(policies={"teacher": teacher})
    with pytest.raises(
        RuntimeError,
        match=r"^policy teacher: could not open a sampling client of Qwen/Qwen3-32B on http://teacher:8001: "
        r"ConnectionError: Connection error\.$",
    ):
        PolicySet.build(policy_specs(loaded), agent_bindings(loaded))

    def refused(*args: Any, **kwargs: Any) -> Any:
        raise TinkerKeyRefusedError("the key was refused")

    monkeypatch.setattr(set_module, "open_tinker_llm", refused)
    with pytest.raises(TinkerKeyRefusedError):  # a configuration error stays one (exit 78)
        PolicySet.build(policy_specs(loaded), agent_bindings(loaded))


# --- a frozen policy's scores -----------------------------------------------------------------------------------------


class _Scorer:
    def __init__(self) -> None:
        self.asked: list[list[int]] = []

    async def compute_logprobs_async(self, sequence: tinker.ModelInput) -> list[float | None]:
        tokens = sequence.to_ints()
        self.asked.append(tokens)
        return [None, *(-0.1 * token for token in tokens[1:])]


def _datum(prompt: list[int], targets: list[int]) -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(prompt),
        loss_fn_inputs={"target_tokens": tinker.TensorData.from_torch(torch.tensor(targets))},
    )


def _teacher(client: Any, *, sample_timeout_seconds: float | None = None) -> Policy:
    spec = PolicySpec(
        "teacher", model_name="t", renderer_name="r", trainable=False, sample_timeout_seconds=sample_timeout_seconds
    )
    info: Any = SimpleNamespace(llm=SimpleNamespace(sampling_client=client, renderer="r", tokenizer="t", version=0))
    return Policy(spec, info)


def test_a_frozen_policy_scores_each_target_token_of_another_policy_s_rows():
    teacher = _teacher(scorer := _Scorer())

    scores = asyncio.run(teacher.score([_datum([1, 2, 3], [2, 3, 4]), _datum([5, 6], [6, 7])]))

    assert scorer.asked == [[1, 2, 3, 4], [5, 6, 7]]  # the prompt and its last target
    assert [score.tolist() for score in scores] == [
        pytest.approx([-0.2, -0.3, -0.4]),  # one per target token
        pytest.approx([-0.6, -0.7]),
    ]
    # As an estimator reads it: its tokenizer and renderer, its version as it moves on, and its scores.
    view = policy_views({"teacher": teacher})["teacher"]
    assert (view.id, view.renderer, view.tokenizer, view.version) == ("teacher", "r", "t", 0)
    llm: Any = teacher.info.llm
    llm.version = 3  # as if it published
    assert view.version == 3
    scores = asyncio.run(view.score([_datum([1, 2], [2, 3])]))
    assert [score.tolist() for score in scores] == [pytest.approx([-0.2, -0.3])]


def test_a_target_token_the_service_gives_no_logprob_for_fails_the_scoring():
    class Holes:
        async def compute_logprobs_async(self, sequence: tinker.ModelInput) -> list[float | None]:
            return [None, -1.0, None, -2.0]

    teacher = _teacher(Holes())
    with pytest.raises(ValueError, match="^the service gave no logprob of target token 1 of a row$"):
        asyncio.run(teacher.score([_datum([1, 2, 3], [2, 3, 4])]))
    view = policy_views({"teacher": teacher})["teacher"]
    with pytest.raises(
        ScoringFailed, match=r"^policy teacher could not score 1 row\(s\): ValueError: the service gave"
    ):
        asyncio.run(view.score([_datum([1, 2, 3], [2, 3, 4])]))


@pytest.mark.parametrize("returned", [[None, -7.0], [None, -1.0, -1.0], [None, -1.0, -1.0, -1.0, -1.0], []])
def test_a_response_of_another_length_than_the_sequence_fails_the_scoring(returned):
    class Fixed:
        async def compute_logprobs_async(self, sequence: tinker.ModelInput) -> list[float | None]:
            return list(returned)

    teacher = _teacher(Fixed())
    row = _datum([1, 2, 3], [2, 3, 4])  # 3 targets: the sequence scored has 4 tokens, so the response 4 entries
    message = rf"the service gave {len(returned)} logprobs for a sequence of 4 tokens \(a row's prompt and its last"
    with pytest.raises(ValueError, match="^" + message):
        asyncio.run(teacher.score([row]))
    view = policy_views({"teacher": teacher})["teacher"]
    with pytest.raises(ScoringFailed, match=r"^policy teacher could not score 1 row\(s\): ValueError: " + message):
        asyncio.run(view.score([row]))


def test_the_first_failed_call_cancels_the_other_calls_of_its_rows():
    state = {"started": 0, "cancelled": 0}

    class OneFails:
        async def compute_logprobs_async(self, sequence: tinker.ModelInput) -> list[float | None]:
            state["started"] += 1
            if sequence.to_ints()[0] == 0:
                await asyncio.sleep(0.01)
                raise RuntimeError("503 Service Unavailable")
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                state["cancelled"] += 1
                raise
            return [None, -1.0]

    teacher = _teacher(OneFails())

    async def scored() -> dict[str, int]:
        with pytest.raises(RuntimeError, match="^503 Service Unavailable$"):  # the call's own error
            await teacher.score([_datum([1], [2]), _datum([0], [1]), _datum([2], [3])])
        return dict(state)  # as score returned: its other calls already ended

    assert asyncio.run(scored()) == {"started": 3, "cancelled": 2}
    assert teacher._scoring is not None and teacher._scoring._value == policy_module.SCORE_CONCURRENCY


def test_each_scoring_call_is_bounded_by_the_policy_s_sample_timeout():
    class Slow:
        async def compute_logprobs_async(self, sequence: tinker.ModelInput) -> list[float | None]:
            await asyncio.sleep(5)
            return []

    teacher = _teacher(Slow(), sample_timeout_seconds=0.05)
    with pytest.raises(TimeoutError, match=r"^a scoring call did not return within 0.05 s$"):
        asyncio.run(teacher.score([_datum([1, 2], [2, 3])]))


def test_a_policy_keeps_at_most_its_bound_of_scoring_calls_in_flight_over_every_group(monkeypatch):
    monkeypatch.setattr(policy_module, "SCORE_CONCURRENCY", 4)
    state = {"now": 0, "most": 0}

    class Counting:
        async def compute_logprobs_async(self, sequence: tinker.ModelInput) -> list[float | None]:
            state["now"] += 1
            state["most"] = max(state["most"], state["now"])
            await asyncio.sleep(0.01)
            state["now"] -= 1
            return [None, -1.0]

    teacher = _teacher(Counting())

    async def groups() -> None:  # four groups of ten rows, credited at once
        await asyncio.gather(*(teacher.score([_datum([1], [2])] * 10) for _ in range(4)))

    asyncio.run(groups())
    assert state["most"] == 4


# --- identity ---------------------------------------------------------------------------------------------------------


def test_the_identity_records_the_policy_table_and_a_resume_may_not_change_it(tmp_path: Path, monkeypatch):
    loaded = config(policies={"teacher": TEACHER})
    table = identity.policy_table(loaded)
    assert table == {
        "policies": {
            MAIN: {"model_name": "Qwen/Qwen3-4B-Instruct-2507", "renderer_name": "qwen3_instruct", "trainable": True},
            "teacher": {
                "model_name": "Qwen/Qwen3-32B",
                "renderer_name": "qwen3",
                "trainable": False,
                "load_checkpoint_path": None,  # the sampler a frozen policy serves: its base model's
            },
        },
        "agents": {"policy": MAIN, "roles": {}},
    }
    monkeypatch.setattr(identity, "resumes", lambda run_dir: False)
    identity.check_run_identity(tmp_path, identity.run_identity(loaded, []), {})
    written = json.loads((tmp_path / identity.IDENTITY_NAME).read_text())
    assert written["policies"] == table["policies"]  # no service URL: a resume may reach its services elsewhere

    monkeypatch.setattr(identity, "resumes", lambda run_dir: True)
    identity.check_run_identity(tmp_path, identity.run_identity(loaded, []), {})  # the same table resumes
    moved = config(policies={"teacher": {**TEACHER, "tinker_base_url": "http://elsewhere:9000"}})
    identity.check_run_identity(tmp_path, identity.run_identity(moved, []), {})  # so does a new URL


@pytest.mark.parametrize(
    ("sections", "problems"),
    [
        (
            {"policies": {"teacher": {**TEACHER, "model_name": "Qwen/Qwen3-8B"}}},
            ["policies.teacher.model_name: the run's is 'Qwen/Qwen3-32B', and this start's 'Qwen/Qwen3-8B'"],
        ),
        (
            {"policies": {"teacher": {**TEACHER, "load_checkpoint_path": "tinker://other/sampler_weights/000100"}}},
            [
                "policies.teacher.load_checkpoint_path: the run's is None, and this start's "
                "'tinker://other/sampler_weights/000100'"
            ],
        ),
        ({}, ["policies.teacher: the run has it, and this start does not declare it"]),
        (
            {"policies": {"teacher": TEACHER, "judge": TEACHER}, "agents": {"roles": {"judge": {"policy": "judge"}}}},
            [
                "policies.judge: this start declares it, and the run does not have it",
                "agents.roles.judge.policy: the run's is None, and this start's 'judge'",
            ],
        ),
    ],
)
def test_a_resume_that_changes_the_policy_table_names_each_change(sections, problems):
    previous = identity.run_identity(config(policies={"teacher": TEACHER}), [])
    assert identity.policy_changes(previous, identity.run_identity(config(**sections), [])) == problems


def test_a_resume_s_changes_are_named_in_the_run_s_order_then_this_start_s():
    assert identity._in_order({"b": 1, "a": 2}, {"c": 3, "a": 4, "d": 5}) == ["b", "a", "c", "d"]
    assert identity._in_order({}, {}) == []


# --- bindings for rollout workers -------------------------------------------------------------------------------------


def test_each_policy_s_binding_file_is_its_own():
    assert protocol.binding_path("/srv/ipc") == "/srv/ipc/model_binding.pkl"
    assert protocol.binding_path("/srv/ipc", "judge") == "/srv/ipc/model_binding.judge.pkl"
    assert protocol.is_binding_file("model_binding.judge.pkl") and protocol.is_binding_file("model_binding.pkl")
    assert not protocol.is_binding_file("model_binding.pkl.123.tmp")
