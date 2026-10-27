"""End to end on CPU: ``harness_rollout`` through the rollout worker's own body, the real model gateway
(``platoon.testing``'s toy tokenizer under the ``qwen3_instruct`` renderer, scripted sampler), platoon-bash in a
process-backed task sandbox, and the real datum conversion. Skipped until ``platoon.gateway`` is present."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pytest
from harness_fakes import FakeService

platoon_gateway = pytest.importorskip("platoon.gateway")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import tinker  # noqa: E402

from platoon.config_defs import RolloutConfig  # noqa: E402
from platoon.task import Task  # noqa: E402
from platoon.testing import TOY_MODEL_NAME, TOY_TOKENIZER_DIR, toy_tokenizer  # noqa: E402
from platoon.train.components import EnvironmentRoute  # noqa: E402
from platoon.train.workers.protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    ConversionPolicy,
    ModelBinding,
    RolloutWorkerRequest,
)
from platoon.train.workers.run import run_rollout_request  # noqa: E402

MODEL_NAME = TOY_MODEL_NAME
# The test model's tokenizer files (tokenizer.json and tokenizer_config.json, which AutoTokenizer loads).
SNAPSHOT = TOY_TOKENIZER_DIR
BASH_CALL = '<tool_call>\n{"name": "bash", "arguments": {"command": "echo 42 > answer.txt"}}\n</tool_call>'
HELLO_CALL = '<tool_call>\n{"name": "bash", "arguments": {"command": "echo hello > hello.txt"}}\n</tool_call>'
HARBOR_TASKS = Path(__file__).parent / "fixtures" / "harbor"


@dataclass
class ScriptedSampler:
    """Answers each ``sample_async`` with the next scripted completion text."""

    tokenizer: Any
    script: list[str]
    prompts: list[Any] = field(default_factory=list)
    sampling_params: list[Any] = field(default_factory=list)

    async def sample_async(self, *, prompt, sampling_params, num_samples):
        self.prompts.append(prompt)
        self.sampling_params.append(sampling_params)
        tokens = self.tokenizer.encode(self.script.pop(0) + "<|im_end|>", add_special_tokens=False)
        sequence = {"stop_reason": "stop", "tokens": tokens, "logprobs": [-0.5] * len(tokens)}
        return tinker.types.SampleResponse.model_validate({"sequences": [sequence], "type": "sample"})

    def sample(self, **kwargs):
        raise AssertionError("the gateway samples asynchronously")


@pytest.fixture(scope="module")
def tokenizer():
    return toy_tokenizer()


@pytest.fixture
def policy_gateway(tokenizer):
    from tinker_cookbook.renderers import get_renderer

    sampler = ScriptedSampler(tokenizer, [BASH_CALL, "The answer is in answer.txt."])
    gateway = platoon_gateway.ModelGateway(
        sampler=sampler, renderer=get_renderer("qwen3_instruct", tokenizer), tokenizer=tokenizer, model_name=MODEL_NAME
    )
    platoon_gateway.set_default_gateway(gateway)
    yield gateway, sampler
    platoon_gateway.set_default_gateway(None)


def worker_request(tmp_path: Path, task: Task, **config_overrides: Any) -> RolloutWorkerRequest:
    config = RolloutConfig(
        output_dir=str(tmp_path / "out"), return_dict=True, train=True, verbose=False, timeout=120, **config_overrides
    )
    return RolloutWorkerRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="harness-e2e",
        task_id=task.id,
        rollout_number=0,
        task=task,
        rollout_config=asdict(config),
        rollout_fn_ref="",
        model=ModelBinding(
            binding_path="", bootstrap_blob=b"", follow_poll_seconds=5.0, model_name="", renderer_name=""
        ),
        conversion=ConversionPolicy(),
        response_path="",
        ready_path="",
    )


def routed_task(**service_kwargs) -> Task:
    route = EnvironmentRoute(
        name="fake",
        service="harness_fakes:FakeService",
        service_kwargs=service_kwargs,
        harness="platoon-bash",
        resolve_task=True,
    )
    return route.attach(Task(id="train-0"))


@pytest.mark.asyncio
@pytest.mark.parametrize(("grade_status", "trainable"), [("scored", True), ("invalid", False)])
async def test_harness_rollout_trains_exact_gateway_tokens_and_excludes_untrainable_outcomes(
    tmp_path, policy_gateway, grade_status, trainable
):
    from platoon.harness.rollout import harness_rollout

    FakeService.root = tmp_path / "sandboxes"
    FakeService.sessions.clear()
    gateway, sampler = policy_gateway

    response = await run_rollout_request(
        worker_request(tmp_path, routed_task(grade_status=grade_status)), rollout_fn=harness_rollout
    )

    (session,) = FakeService.sessions
    assert (session.sandbox.workdir / "answer.txt").read_text().strip() == "42"
    assert session.closed and response.model_calls == 2 and len(sampler.prompts) == 2
    assert not gateway._agents_by_id  # every agent of the rollout was closed
    result = response.result
    assert result is not None and response.status == "completed"
    assert result.task_reward_valid is trainable
    if trainable:
        assert result.task_reward == 1.0 and result.datums
        # Exactly the sampled tokens are trained, whether or not the two calls prefix-merge into one datum.
        assert sum(int(datum.loss_fn_inputs["mask"].to_torch().sum()) for datum in result.datums) == sum(
            len(interaction_tokens) for interaction_tokens in _sampled_tokens(sampler)
        )
    else:
        assert not result.datums and result.num_policy_excluded_datums > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("inference_params", "max_tokens"),
    [({}, 4096), ({"max_completion_tokens": 77}, 77)],
    ids=["unset-gateway-default", "rollout-budget"],
)
async def test_an_http_harness_that_sends_no_max_tokens_gets_the_rollouts_budget_or_the_gateways(
    tmp_path, policy_gateway, inference_params, max_tokens
):
    # platoon-bash sends no max_tokens: its calls get inference_params.max_completion_tokens when the rollout sets
    # it, else the gateway's own default (4096).
    from platoon.config_defs import InferenceParams
    from platoon.harness.rollout import harness_rollout

    FakeService.root = tmp_path / "sandboxes"
    FakeService.sessions.clear()
    gateway, sampler = policy_gateway
    request = worker_request(tmp_path, routed_task(), inference_params=InferenceParams(**inference_params))

    response = await run_rollout_request(request, rollout_fn=harness_rollout)

    assert response.status == "completed" and response.model_calls == 2
    assert gateway.max_tokens == 4096
    assert [params.max_tokens for params in sampler.sampling_params] == [max_tokens, max_tokens]


@pytest.mark.asyncio
async def test_harness_rollout_on_a_harbor_task_in_a_process_sandbox(tmp_path, policy_gateway):
    pytest.importorskip("platoon.envs.harbor")
    from platoon.harness.rollout import harness_rollout

    gateway, sampler = policy_gateway
    sampler.script[:] = [HELLO_CALL, "Created hello.txt."]
    route = EnvironmentRoute(
        name="harbor",
        service="harbor",
        service_kwargs={
            "task_dirs": [str(HARBOR_TASKS / "write-hello")],
            "provider": "process",
            "provider_options": {"root": str(tmp_path / "sandboxes")},
        },
        harness="platoon-bash",
        resolve_task=True,
    )

    response = await run_rollout_request(
        worker_request(tmp_path, route.attach(Task(id="write-hello"))), rollout_fn=harness_rollout
    )

    result = response.result
    assert response.status == "completed" and response.model_calls == 2
    assert result is not None and result.task_reward_valid and result.task_reward == 1.0 and result.datums
    assert not gateway._agents_by_id


def _sampled_tokens(sampler: ScriptedSampler) -> list[list[int]]:
    tokenizer = sampler.tokenizer
    return [
        tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)
        for text in (BASH_CALL, "The answer is in answer.txt.")
    ]
