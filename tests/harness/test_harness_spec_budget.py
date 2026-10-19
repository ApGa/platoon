"""A spec that renders ``{max_steps}`` needs a step budget: from the task, the run (``rollout_config.max_steps``) or
the spec's own ``max_steps`` default.  Without one, the rollout is a ``harness_config`` error that says which
setting is missing, not an ``unexpected:HarnessSpecError`` crash."""

from __future__ import annotations

import dataclasses

import pytest
from harness_fakes import FakeChatServer, FakeGateway, FakeService
from test_harness_rollout import rollout_config, runner, spec_file
from test_harness_spec import CONTEXT, minimal

from platoon.harness.rollout import run_harness_rollout
from platoon.harness.spec import HarnessSpecError, parse_spec
from platoon.task import Task
from platoon.train.components import EnvironmentRoute

NO_BUDGET = "renders {max_steps}, but the run sets no step budget (rollout_config.max_steps or the task's max_steps)"


@pytest.fixture(autouse=True)
def service_root(tmp_path):
    FakeService.root = tmp_path / "sandboxes"
    FakeService.sessions.clear()
    yield
    FakeService.sessions.clear()


async def _rollout(tmp_path, spec: str, **config):
    route = EnvironmentRoute(
        name="fake", service="harness_fakes:FakeService", service_kwargs={}, harness=spec, resolve_task=True
    )
    gateway = FakeGateway(FakeChatServer([]))
    await run_harness_rollout(
        route.attach(Task(id="train-0")), rollout_config(tmp_path, **config), gateway=gateway, runner=runner(tmp_path)
    )
    ((_, outcome),) = gateway.collected
    return outcome


@pytest.mark.asyncio
async def test_a_max_steps_placeholder_without_any_budget_is_a_harness_config_error(tmp_path):
    spec = spec_file(tmp_path, '[launch]\nargv = ["sh", "-c", "echo {max_steps}"]\n')

    outcome = await _rollout(tmp_path, spec)

    assert (outcome.status, outcome.failure_kind) == ("error", "harness_config")
    assert outcome.reason == f"harness custom@1 {NO_BUDGET} and the spec no max_steps default"
    (session,) = FakeService.sessions
    assert session.graded == 0 and session.closed


@pytest.mark.asyncio
async def test_the_runs_budget_fills_the_placeholder(tmp_path):
    spec = spec_file(tmp_path, '[launch]\nargv = ["sh", "-c", "test {max_steps} = 5"]\n')

    outcome = await _rollout(tmp_path, spec, max_steps=5)

    assert outcome.status == "scored"


def test_rendering_names_the_spec_and_the_missing_settings():
    spec = parse_spec(minimal(launch={"argv": ["toy", "--steps", "{max_steps}"]}))

    with pytest.raises(HarnessSpecError) as raised:
        spec.render(CONTEXT)

    assert str(raised.value) == f"harness toy@1.0 {NO_BUDGET} and the spec no max_steps default"
    assert spec.render(dataclasses.replace(CONTEXT, max_steps=3)).argv == ("toy", "--steps", "3")
    assert dataclasses.replace(spec, max_steps=4).render(CONTEXT).argv == ("toy", "--steps", "4")


def test_a_placeholder_in_a_knob_that_is_not_applied_needs_no_budget():
    spec = parse_spec(minimal(builtins={"tools": ["bash"], "disable": {"args": ["--no-bash", "--steps={max_steps}"]}}))

    assert spec.render(CONTEXT).argv == ("toy", CONTEXT.instruction)
    with pytest.raises(HarnessSpecError, match=r"renders \{max_steps\}, but the run sets no step budget"):
        spec.render(CONTEXT, disable_builtins=True)
