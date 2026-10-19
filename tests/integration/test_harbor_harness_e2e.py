"""End to end on CPU: one Harbor task through every layer.

A toy Harbor task runs in a process-provider sandbox. The platoon-bash harness runs in that sandbox and talks to
the real embedded model gateway (``platoon.testing``'s toy tokenizer under the ``qwen3_instruct`` renderer), where a
scripted sampler stands in for Tinker: it writes a bash tool call, then a final answer. The task's own
``tests/test.sh`` grades the sandbox, and ``trajectory_to_data`` turns the gateway's call records into datums that
train exactly the sampled tokens. A grade the verifier fails to produce is an error that never trains, not a zero.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import tinker  # noqa: E402

from platoon.config_defs import RolloutConfig  # noqa: E402
from platoon.gateway import ModelGateway  # noqa: E402
from platoon.harness.rollout import EmbeddedGateway, run_harness_rollout  # noqa: E402
from platoon.task import Task  # noqa: E402
from platoon.testing import TOY_MODEL_NAME, toy_tokenizer  # noqa: E402
from platoon.train.components import EnvironmentRoute  # noqa: E402
from platoon.train.tinker.proxy import TinkerLLMProxySession  # noqa: E402
from platoon.utils.tinker_data_processing import (  # noqa: E402
    get_train_data_for_trajectory_collection,
    trajectory_to_data,
)
from platoon.utils.trajectory_status import trajectory_was_interrupted, trajectory_was_invalid  # noqa: E402

TASK_FILES = {
    "task.toml": """\
version = "1.0"

[task]
name = "platoon/sum-to-file"

[agent]
timeout_sec = 120.0

[verifier]
timeout_sec = 30.0

[environment]
docker_image = "ubuntu:24.04"
workdir = "/app"
""",
    "instruction.md": "Write the sum of 19 and 23 to the file `answer.txt` in the working directory.\n",
    "tests/test.sh": """\
#!/bin/bash
# Runs in the workdir. PLATOON_SANDBOX_ROOT is set only by the process provider.
reward_file="${PLATOON_SANDBOX_ROOT:-}/logs/verifier/reward.txt"
if [ "$(cat answer.txt 2>/dev/null)" = "42" ]; then echo 1 > "$reward_file"; else echo 0 > "$reward_file"; fi
""",
}
SOLVE = '<tool_call>\n{"name": "bash", "arguments": {"command": "echo $((19 + 23)) > answer.txt"}}\n</tool_call>'
FINAL_ANSWER = "answer.txt now holds 42."


@dataclass
class ScriptedSampler:
    """Stands in for a Tinker sampling client: each ``sample_async`` returns the next scripted completion."""

    tokenizer: Any
    script: list[str]
    sampled: list[list[int]] = field(default_factory=list)

    async def sample_async(self, *, prompt, sampling_params, num_samples):
        tokens = self.tokenizer.encode(self.script.pop(0) + "<|im_end|>", add_special_tokens=False)
        self.sampled.append(tokens)
        sequence = {"stop_reason": "stop", "tokens": tokens, "logprobs": [-0.5] * len(tokens)}
        return tinker.types.SampleResponse.model_validate({"sequences": [sequence], "type": "sample"})


@pytest.fixture(scope="module")
def tokenizer():
    return toy_tokenizer()


@pytest.fixture
def harbor_task(tmp_path) -> Path:
    task_dir = tmp_path / "dataset" / "sum-to-file"
    for name, content in TASK_FILES.items():
        (task_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (task_dir / name).write_text(content)
    return task_dir


async def run_rollout(tmp_path: Path, task_dir: Path, sampler: ScriptedSampler) -> tuple[dict, dict, ModelGateway]:
    """One harness rollout as a rollout worker runs it: inside a proxy session that collects the gateway's records."""

    from tinker_cookbook.renderers import get_renderer

    gateway = ModelGateway(
        sampler=sampler,
        renderer=get_renderer("qwen3_instruct", sampler.tokenizer),
        tokenizer=sampler.tokenizer,
        model_name=TOY_MODEL_NAME,
    )
    route = EnvironmentRoute(
        name="harbor",
        service="harbor",
        service_kwargs={
            "dataset_dir": str(task_dir.parent),
            "provider": "process",
            "provider_options": {"root": str(tmp_path / "sandboxes")},
        },
        harness="platoon-bash",
        resolve_task=True,
    )
    config = RolloutConfig(output_dir=str(tmp_path / "out"), return_dict=True, train=True, verbose=False, timeout=120)
    with TinkerLLMProxySession() as proxy:
        collection = await run_harness_rollout(
            route.attach(Task(id=task_dir.name)), config, gateway=EmbeddedGateway(gateway)
        )
        interactions = dict(proxy.interactions)
    return collection, interactions, gateway


def root_trajectory(collection: dict) -> dict:
    (trajectory,) = collection["trajectories"].values()
    return trajectory


def masked(datum: tinker.Datum, key: str) -> list[float]:
    """The values of ``loss_fn_inputs[key]`` at the positions the loss trains."""

    mask = datum.loss_fn_inputs["mask"].to_torch().tolist()
    return [value for value, trained in zip(datum.loss_fn_inputs[key].to_torch().tolist(), mask) if trained]


@pytest.mark.asyncio
async def test_solved_harbor_task_trains_exactly_the_sampled_tokens(tmp_path, tokenizer, harbor_task):
    sampler = ScriptedSampler(tokenizer, [SOLVE, FINAL_ANSWER])

    collection, interactions, gateway = await run_rollout(tmp_path, harbor_task, sampler)

    trajectory = root_trajectory(collection)
    assert trajectory["reward"] == 1.0 and not trajectory_was_interrupted(trajectory)
    assert trajectory["misc"]["harness"]["spec"] == "platoon-bash@1"
    assert trajectory["misc"]["harness"]["placement"] == "task_sandbox"
    assert trajectory["misc"]["harness"]["exit_code"] == 0
    assert trajectory["finish_message"] == FINAL_ANSWER
    assert not gateway.agents(collection["id"])  # the rollout closed every agent it opened
    assert not any((tmp_path / "sandboxes").iterdir())  # and stopped its sandbox

    call_ids = [step["misc"]["action_misc"]["completion_id"] for step in trajectory["steps"]]
    assert len(call_ids) == 2 and set(call_ids) == set(interactions)
    # The harness ran the first call's command in the sandbox and sent its result back with the second call.
    second_prompt = tokenizer.decode(list(interactions[call_ids[1]].obs.to_ints()))
    assert "exit code 0" in second_prompt

    result = trajectory_to_data(
        trajectory,
        interactions,
        task_id=collection["id"],
        trajectory_id=trajectory["id"],
        trajectory_advantage=0.5,
        checkpoint_version=0,
    )

    trained_tokens = [token for datum in result.datums for token in masked(datum, "target_tokens")]
    assert trained_tokens == [token for tokens in sampler.sampled for token in tokens]
    assert {advantage for datum in result.datums for advantage in masked(datum, "advantages")} == {0.5}


@pytest.mark.asyncio
async def test_unsolved_harbor_task_is_a_scored_zero_that_still_trains(tmp_path, tokenizer, harbor_task):
    sampler = ScriptedSampler(tokenizer, ["I cannot do that."])

    collection, interactions, _ = await run_rollout(tmp_path, harbor_task, sampler)

    trajectory = root_trajectory(collection)
    assert trajectory["reward"] == 0.0 and not trajectory_was_interrupted(trajectory)
    result = trajectory_to_data(
        trajectory,
        interactions,
        task_id=collection["id"],
        trajectory_id=trajectory["id"],
        trajectory_advantage=-0.5,
        checkpoint_version=0,
    )
    (datum,) = result.datums
    assert masked(datum, "target_tokens") == sampler.sampled[0]


@pytest.mark.asyncio
async def test_harbor_verifier_failure_is_an_error_not_a_zero(tmp_path, tokenizer, harbor_task):
    (harbor_task / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n")  # writes no reward file
    sampler = ScriptedSampler(tokenizer, [SOLVE, FINAL_ANSWER])

    collection, interactions, _ = await run_rollout(tmp_path, harbor_task, sampler)

    trajectory = root_trajectory(collection)
    assert trajectory["misc"]["harness"]["exit_code"] == 0
    assert trajectory["misc"]["outcome"]["status"] == "error" and trajectory_was_invalid(trajectory)
    result = get_train_data_for_trajectory_collection(
        trajectory_collection=collection, interactions=interactions, task_id=collection["id"], checkpoint_version=0
    )
    assert not result.task_reward_valid  # out of the group baseline
    assert not result.datums and result.num_policy_excluded_datums > 0  # and out of training
