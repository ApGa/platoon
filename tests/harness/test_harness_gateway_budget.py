"""A spec with ``[budget] enforce = "gateway"``: the gateway ends a harness that has no step limit of its own at the
run's step budget, the episode is graded with reason ``budget``, and only the sampled calls train.  End to end on
CPU through the rollout worker's own body, the real gateway (Qwen3-4B-Instruct tokenizer, scripted sampler) and
platoon-bash told to take far more turns than the budget.  A harness in the task sandbox may make only the policy
calls a budget counts; one on the rollout host keeps its aux and judge calls."""

from __future__ import annotations

from pathlib import Path

import pytest
from harness_fakes import FakeChatServer, FakeGateway, FakeService
from test_harness_gateway_integration import BASH_CALL, MODEL_NAME, SNAPSHOT, ScriptedSampler, worker_request
from test_harness_rollout import rollout_config, runner

from platoon.harness.rollout import run_harness_rollout
from platoon.task import Task
from platoon.train.components import EnvironmentRoute

PLATOON_BASH = Path(__file__).resolve().parents[2] / "platoon" / "harness" / "builtin" / "platoon_bash.py"
# A harness that makes an aux, a judge and a policy call, and writes each call's HTTP status as its final message.
CALL_KIND_PROBE = """\
import json, os, sys, urllib.error, urllib.request

statuses = []
for kind in ("aux", "judge", "policy"):
    request = urllib.request.Request(
        sys.argv[1] + "/v1/chat/completions",
        data=json.dumps({"model": "policy", "messages": [{"role": "user", "content": "Summarize."}]}).encode(),
        headers={
            "Authorization": "Bearer " + os.environ["PLATOON_API_KEY"],
            "Content-Type": "application/json",
            "X-Platoon-Call-Kind": kind,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            statuses.append(f"{kind}={response.status}")
    except urllib.error.HTTPError as error:
        statuses.append(f"{kind}={error.code}")
with open(sys.argv[2], "w") as out:
    out.write(" ".join(statuses))
"""


def unbounded_bash(tmp_path: Path, *, max_steps: int | None = None) -> str:
    """platoon-bash with a turn limit far above any budget, so only the gateway can stop it."""

    default = f"max_steps = {max_steps}\n" if max_steps is not None else ""
    path = tmp_path / "unbounded-bash.toml"
    path.write_text(
        f"""name = "unbounded-bash"
version = "1"
{default}
[budget]
enforce = "gateway"

[launch]
argv = ["python3", "{{state_dir}}/platoon_bash.py", "--base-url", "{{base_url}}/v1", "--model", "{{model}}",
        "--instruction-file", "{{state_dir}}/instruction.md", "--final-message-file", "{{state_dir}}/final.txt",
        "--max-turns", "50"]
env = {{ PLATOON_API_KEY = "{{api_key}}" }}
files = {{ "instruction.md" = "{{instruction}}" }}
assets = {{ "platoon_bash.py" = "{PLATOON_BASH}" }}

[final_message]
file = "{{state_dir}}/final.txt"
"""
    )
    return str(path)


def call_kind_probe(tmp_path: Path) -> str:
    """The spec of a harness that runs :data:`CALL_KIND_PROBE`."""

    script = tmp_path / "probe.py"
    script.write_text(CALL_KIND_PROBE)
    path = tmp_path / "call-kind-probe.toml"
    path.write_text(
        f"""name = "call-kind-probe"
version = "1"

[launch]
argv = ["python3", "{{state_dir}}/probe.py", "{{base_url}}", "{{state_dir}}/final.txt"]
env = {{ PLATOON_API_KEY = "{{api_key}}" }}
assets = {{ "probe.py" = "{script}" }}

[final_message]
file = "{{state_dir}}/final.txt"
"""
    )
    return str(path)


@pytest.fixture(scope="module")
def qwen_tokenizer():
    if not (SNAPSHOT / "tokenizer.json").exists():
        pytest.skip(f"{MODEL_NAME} tokenizer snapshot not available at {SNAPSHOT}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(SNAPSHOT))


@pytest.fixture
def budget_gateway(qwen_tokenizer):
    """This process's policy gateway (the one ``harness_rollout`` serves), over a scripted sampler."""

    from tinker_cookbook.renderers import get_renderer

    from platoon.gateway import ModelGateway, set_default_gateway

    sampler = ScriptedSampler(qwen_tokenizer, [])
    renderer = get_renderer("qwen3_instruct", qwen_tokenizer)
    gateway = ModelGateway(sampler=sampler, renderer=renderer, tokenizer=qwen_tokenizer, model_name=MODEL_NAME)
    set_default_gateway(gateway)
    yield gateway, sampler
    set_default_gateway(None)


def routed(harness: str, *, service_kwargs: dict | None = None, **task) -> Task:
    route = EnvironmentRoute(
        name="fake",
        service="harness_fakes:FakeService",
        service_kwargs=service_kwargs or {},
        harness=harness,
        resolve_task=True,
    )
    return route.attach(Task(id="train-0", **task))


@pytest.mark.asyncio
async def test_the_gateway_ends_the_run_at_the_step_budget_and_the_episode_is_graded(tmp_path, budget_gateway):
    from platoon.harness.rollout import harness_rollout
    from platoon.train.tinker.workers.run import run_rollout_request

    FakeService.root = tmp_path / "sandboxes"
    FakeService.sessions.clear()
    gateway, sampler = budget_gateway
    sampler.script[:] = [BASH_CALL, BASH_CALL]  # the policy would go on calling tools: only the budget stops it

    response = await run_rollout_request(
        worker_request(tmp_path, routed(unbounded_bash(tmp_path)), max_steps=2), rollout_fn=harness_rollout
    )

    assert response.status == "completed" and response.model_calls == 2 and len(sampler.prompts) == 2
    (session,) = FakeService.sessions
    assert (session.sandbox.workdir / "answer.txt").read_text().strip() == "42"
    result = response.result
    assert result is not None and result.task_reward == 1.0 and result.task_reward_valid
    # The stop message was never sampled, so exactly the two sampled calls train.
    assert sum(int(datum.loss_fn_inputs["mask"].to_torch().sum()) for datum in result.datums) == sum(
        len(sampler.tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)) for text in [BASH_CALL] * 2
    )


@pytest.mark.asyncio
async def test_a_budget_stop_is_graded_with_reason_budget(tmp_path, budget_gateway):
    from platoon.harness.rollout import EmbeddedGateway

    FakeService.root = tmp_path / "sandboxes"
    FakeService.sessions.clear()
    gateway, sampler = budget_gateway
    sampler.script[:] = [BASH_CALL]

    collection = await run_harness_rollout(
        routed(unbounded_bash(tmp_path, max_steps=1)),
        rollout_config(tmp_path),
        gateway=EmbeddedGateway(gateway),
        runner=runner(tmp_path),
    )

    (root,) = collection["trajectories"].values()
    assert root["misc"]["outcome"]["reason"] == "budget" and root["reward"] == 1.0
    assert root["misc"]["budget"] == {"enforce": "gateway", "max_steps": 1, "stops": 1}
    assert len(root["steps"]) == 1 and "step budget of 1 model calls is used up" in root["finish_message"]


@pytest.mark.asyncio
async def test_a_gateway_budget_needs_a_step_budget(tmp_path):
    FakeService.root = tmp_path / "sandboxes"
    gateway = FakeGateway(FakeChatServer([]))

    await run_harness_rollout(
        routed(unbounded_bash(tmp_path)), rollout_config(tmp_path), gateway=gateway, runner=runner(tmp_path)
    )

    ((_, outcome),) = gateway.collected
    assert (outcome.status, outcome.failure_kind) == ("error", "harness_config")
    assert "has the gateway enforce its step budget, but the run sets no step budget" in outcome.reason
    assert gateway.server.requests == []


@pytest.mark.asyncio
async def test_a_gateway_that_cannot_cap_calls_refuses_the_spec(tmp_path):
    FakeService.root = tmp_path / "sandboxes"
    gateway = FakeGateway(FakeChatServer([]))  # its agents have no call cap

    await run_harness_rollout(
        routed(unbounded_bash(tmp_path)),
        rollout_config(tmp_path, max_steps=3),
        gateway=gateway,
        runner=runner(tmp_path),
    )

    ((_, outcome),) = gateway.collected
    assert (outcome.status, outcome.failure_kind) == ("error", "harness_setup")
    assert "needs a gateway that enforces step budgets" in outcome.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sandbox", "placement", "statuses", "sampled"),
    [
        (True, "task_sandbox", "aux=403 judge=403 policy=200", 1),
        (False, "local", "aux=200 judge=200 policy=200", 3),
    ],
    ids=["task-sandbox", "rollout-host"],
)
async def test_a_harness_in_the_task_sandbox_may_make_only_policy_calls(
    tmp_path, budget_gateway, sandbox, placement, statuses, sampled
):
    from platoon.harness.rollout import EmbeddedGateway

    FakeService.root = tmp_path / "sandboxes"
    FakeService.sessions.clear()
    gateway, sampler = budget_gateway
    sampler.script[:] = ["A summary.", "A verdict.", "An answer."][-sampled:]

    collection = await run_harness_rollout(
        routed(call_kind_probe(tmp_path), service_kwargs={"sandbox": sandbox}),
        rollout_config(tmp_path),
        gateway=EmbeddedGateway(gateway),
        runner=runner(tmp_path),
    )

    (root,) = collection["trajectories"].values()
    assert root["misc"]["harness"]["placement"] == placement
    assert root["finish_message"] == statuses and len(sampler.prompts) == sampled and sampler.script == []
