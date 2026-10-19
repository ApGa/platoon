"""``on_harness_exit``: a harness that ran out of time or crashed is graded or excluded as the environment says, a
crash after a gateway failure never trains, and the task's own agent timeout bounds the harness."""

from __future__ import annotations

import time

import pytest
from harness_fakes import FakeChatServer, FakeGateway, FakeService, answer
from test_harness_rollout import SERVICE, _with_key, run, spec_file

from platoon.harness.agent import AgentResult
from platoon.harness.inline import agent_timeout, failure_outcome
from platoon.harness.rollout import HARNESS_MISC_KEY
from platoon.task import Task
from platoon.train.components import EnvironmentConfig, EnvironmentRoute, HarnessExitPolicy


@pytest.fixture(autouse=True)
def service_root(tmp_path):
    FakeService.root = tmp_path / "sandboxes"
    FakeService.sessions.clear()
    yield
    FakeService.sessions.clear()


def exit_routed_task(harness: str, on_harness_exit, **service_kwargs) -> Task:
    route = EnvironmentRoute(
        name="fake",
        service=SERVICE,
        service_kwargs=service_kwargs,
        harness=harness,
        on_harness_exit=on_harness_exit,
        resolve_task=True,
    )
    return route.attach(Task(id="train-0", max_steps=7))


def _with_gateway_errors(open_agent, count: int):
    def open_failed(**kwargs):
        agent = open_agent(**kwargs)
        agent.gateway_errors = count
        return agent

    return open_failed


@pytest.mark.asyncio
async def test_a_timeout_is_graded_when_the_environment_says_so(tmp_path):
    harness = spec_file(tmp_path, '[launch]\nargv = ["sleep", "60"]\n')
    gateway = FakeGateway(FakeChatServer([]))

    await run(tmp_path, exit_routed_task(harness, {"timeout": "grade"}), gateway, timeout=1)

    ((_, outcome),) = gateway.collected
    assert (outcome.status, outcome.reason, outcome.reward) == ("scored", "agent_timeout", 0.0)
    assert FakeService.sessions[0].graded == 1


@pytest.mark.parametrize(
    ("crash", "gateway_errors", "expected"),
    [
        ("exclude", 0, ("error", "harness_crash", 0)),
        ("grade", 3, ("scored", "harness_crash", 1)),
        ("grade_if_gateway_clean", 0, ("scored", "harness_crash", 1)),
        ("grade_if_gateway_clean", 1, ("error", "harness_crash_gateway", 0)),
    ],
)
@pytest.mark.asyncio
async def test_a_crash_follows_the_crash_rule(tmp_path, crash, gateway_errors, expected):
    gateway = FakeGateway(FakeChatServer([answer("never")], api_key="another-agent"))
    gateway.open_agent = _with_gateway_errors(_with_key(gateway.open_agent, "wrong-secret"), gateway_errors)

    collection = await run(tmp_path, exit_routed_task("platoon-bash", {"crash": crash}), gateway)

    ((_, outcome),) = gateway.collected
    status, kind, graded = expected
    assert outcome.status == status and (outcome.reason if status == "scored" else outcome.failure_kind) == kind
    assert FakeService.sessions[0].graded == graded
    if kind == "harness_crash_gateway":
        assert "after the model gateway failed 1 of the agent's calls" in outcome.reason
    (root,) = collection["trajectories"].values()
    assert root["misc"][HARNESS_MISC_KEY]["exit_code"] == 2


@pytest.mark.asyncio
async def test_the_tasks_agent_timeout_bounds_the_harness(tmp_path):
    harness = spec_file(tmp_path, '[launch]\nargv = ["sleep", "60"]\n')
    gateway = FakeGateway(FakeChatServer([]))

    started = time.monotonic()
    await run(tmp_path, exit_routed_task(harness, None, agent_timeout_s=1.0), gateway, timeout=600)

    assert time.monotonic() - started < 15
    ((_, outcome),) = gateway.collected
    assert (outcome.status, outcome.reason) == ("timeout", "agent_timeout")


@pytest.mark.asyncio
async def test_the_default_policy_excludes_both(tmp_path):
    harness = spec_file(tmp_path, '[launch]\nargv = ["sleep", "60"]\n')
    gateway = FakeGateway(FakeChatServer([]))

    await run(tmp_path, exit_routed_task(harness, None), gateway, timeout=1)

    ((_, outcome),) = gateway.collected
    assert outcome.status == "timeout" and FakeService.sessions[0].graded == 0


# --------------------------------------------------------------------------- config and the inline taxonomy


def test_the_policy_is_configured_per_environment_and_travels_with_the_route():
    assert EnvironmentConfig(service="harbor").on_harness_exit == HarnessExitPolicy("exclude", "exclude")
    assert EnvironmentConfig(service="harbor", on_harness_exit="grade").on_harness_exit == HarnessExitPolicy(
        "grade", "grade"
    )
    configured = EnvironmentConfig(service="harbor", on_harness_exit={"crash": "grade_if_gateway_clean"})
    assert configured.on_harness_exit == HarnessExitPolicy(timeout="exclude", crash="grade_if_gateway_clean")

    route = EnvironmentRoute(name="r", on_harness_exit={"timeout": "grade"})
    assert EnvironmentRoute.of_task(route.attach(Task(id="t"))).on_harness_exit == HarnessExitPolicy("grade")
    for bad, message in [
        ({"timeout": "retry"}, "on_harness_exit.timeout must be one of"),
        ({"crash": "grade_if_clean"}, "on_harness_exit.crash must be one of"),
        ({"crashes": "grade"}, "takes timeout and crash"),
        ("grade_if_gateway_clean", "must be grade, exclude or"),
        (3, "must be a mapping"),
    ]:
        with pytest.raises(ValueError, match=message):
            EnvironmentConfig(name="e", service="harbor", on_harness_exit=bad)


@pytest.mark.parametrize(
    ("result", "policy", "gateway_errors", "expected"),
    [
        (AgentResult(status="timeout"), None, 0, ("timeout", None)),
        (AgentResult(status="timeout"), {"timeout": "grade"}, 0, (None, "agent_timeout")),
        (AgentResult(status="cancelled"), "grade", 0, ("cancelled", None)),
        (AgentResult(status="error", error="boom"), None, 0, ("error", None)),
        (AgentResult(status="error", error="boom"), {"crash": "grade"}, 2, (None, "harness_crash")),
        (AgentResult(status="error", error="boom"), {"crash": "grade_if_gateway_clean"}, 1, ("error", None)),
        (AgentResult(status="error", error="boom"), {"crash": "grade_if_gateway_clean"}, 0, (None, "harness_crash")),
    ],
)
def test_function_harnesses_follow_the_same_rules(result, policy, gateway_errors, expected):
    outcome, reason = failure_outcome(
        result,
        context_overflows=0,
        on_context_overflow="grade",
        harness_ref="h",
        on_harness_exit=policy,
        gateway_errors=gateway_errors,
    )

    assert ((outcome.status if outcome else None), reason) == expected
    if outcome is not None and outcome.status == "error":
        assert outcome.failure_kind == ("harness_crash_gateway" if gateway_errors else "harness_crash")


def test_a_crash_after_a_context_overflow_keeps_its_own_rule():
    crashed = AgentResult(status="error", error="boom")
    for policy in ("exclude", "grade"):
        outcome, reason = failure_outcome(
            crashed, context_overflows=1, on_context_overflow="grade", harness_ref="h", on_harness_exit=policy
        )
        assert (outcome, reason) == (None, "context_overflow")


def test_agent_timeout_is_the_tightest_limit():
    class Session:
        agent_timeout_s = 90.0

    assert agent_timeout(rollout_config_like(600), Session()) == 90.0
    assert agent_timeout(rollout_config_like(30), Session()) == 30
    assert agent_timeout(rollout_config_like(None), Session()) == 90.0
    assert agent_timeout(rollout_config_like(None), object()) is None


def rollout_config_like(timeout):
    class Config:
        pass

    config = Config()
    config.timeout = timeout
    return config
