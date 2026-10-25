"""examples/harbor-local: ``platoon eval`` runs the example's Harbor tasks as docs/how-to/sandboxes-and-harbor.md
says: the runnable task is solved in its sandbox by platoon-bash and graded by its own tests, and the task with a
compose file is refused, never run.

CPU only: the sandbox provider is overridden to ``process`` (host directories, no container), and the policy is a
script behind the real model gateway (``platoon.testing``'s toy tokenizer under the config's renderer), installed
where ``EvalPolicy`` would start a Tinker sampler.  platoon-bash runs as its own process and
reaches that gateway over HTTP on loopback, as it would from a real task sandbox.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gateway"))

from gateway_fakes import sample_response  # noqa: E402

from platoon import cli  # noqa: E402
from platoon.testing import toy_tokenizer  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "harbor-local"
TASKS = EXAMPLE / "tasks"
COMMAND = "echo 'hello from platoon' > greeting.txt"
WRITE = f'<tool_call>\n{{"name": "bash", "arguments": {{"command": "{COMMAND}"}}}}\n</tool_call>'


@dataclass
class ScriptedBash:
    """Writes the greeting with one bash call, then answers once the tool result is in the prompt."""

    tokenizer: Any
    prompts: list[str] = field(default_factory=list)

    async def sample_async(self, *, prompt, sampling_params, num_samples):
        text = self.tokenizer.decode(prompt.to_ints())
        self.prompts.append(text)
        reply = "I wrote greeting.txt." if "<tool_response>" in text else WRITE
        return sample_response(self.tokenizer.encode(reply + "<|im_end|>", add_special_tokens=False))

    def sample(self, **kwargs):
        raise AssertionError("the gateway samples asynchronously")


SAMPLERS: list[ScriptedBash] = []


@dataclass(frozen=True)
class ScriptedPolicy:
    """Stands in for ``platoon.eval.EvalPolicy``: serves :class:`ScriptedBash` instead of a Tinker sampler."""

    model_name: str
    renderer_name: str
    model_path: str | None = None
    renderer_kwargs: dict = field(default_factory=dict)
    context_window_length: int | None = None

    def install(self) -> None:
        from tinker_cookbook.renderers import get_renderer

        from platoon.gateway import ModelGateway, set_default_gateway

        tokenizer = toy_tokenizer()
        sampler = ScriptedBash(tokenizer)
        renderer = get_renderer(self.renderer_name, tokenizer)
        set_default_gateway(
            ModelGateway(sampler=sampler, renderer=renderer, tokenizer=tokenizer, model_name=self.model_name)
        )
        SAMPLERS.append(sampler)


@pytest.fixture
def scripted_policy(monkeypatch):
    from platoon.gateway import set_default_gateway

    SAMPLERS.clear()
    # ``platoon eval`` builds its policy in ``EvalConfig.policy``; ``platoon rollout`` takes the package's class.
    monkeypatch.setattr("platoon.eval.config_defs.EvalPolicy", ScriptedPolicy)
    monkeypatch.setattr(sys.modules[cli._evaluation().__package__], "EvalPolicy", ScriptedPolicy, raising=False)
    yield SAMPLERS
    set_default_gateway(None)


def test_the_example_config_validates(capsys):
    cli.main(["config", "validate", str(EXAMPLE / "eval.yaml")])

    assert capsys.readouterr().out == f"ok   {EXAMPLE / 'eval.yaml'} (PlatoonEvalConfig)\n"


def test_eval_solves_the_runnable_task_in_a_process_sandbox(tmp_path, capsys, scripted_policy):
    results = tmp_path / "results"

    cli.main(
        [
            "eval",
            str(EXAMPLE / "eval.yaml"),
            f"environments.0.service_kwargs.dataset_dir={TASKS}",
            "environments.0.service_kwargs.provider=process",
            # The process provider runs stacks, unlike the example's providers: refuse them, as those do.
            "environments.0.service_kwargs.stacks=refuse",
            f"environments.0.service_kwargs.provider_options={{root: {tmp_path / 'sandboxes'}}}",
            f"inference.output_dir={results}",
            "inference.workflow.rollout_config.timeout=120",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["total_tasks"] == 1  # needs-a-database is refused, not run
    assert summary["valid_rollouts"] == 1 and summary["errored_rollouts"] == 0
    assert summary["success_rate"] == 1.0
    (sampler,) = scripted_policy
    assert len(sampler.prompts) == 2
    assert "Create a file named `greeting.txt`" in sampler.prompts[0]

    report = json.loads((results / "reports" / "final_report.json").read_text())
    assert report["summary"]["outcome_status_counts"] == {"scored": 1}
    assert (results / "config.resolved.yaml").is_file() and (results / "lock.json").is_file()
    assert (
        yaml.safe_load((results / "config.resolved.yaml").read_text())["environments"][0]["service_kwargs"]["provider"]
        == "process"
    )


def test_a_task_with_a_compose_file_is_refused_with_the_reason():
    from platoon.envs.harbor import HarborEnvironmentService

    service = HarborEnvironmentService(dataset_dir=TASKS, provider="docker")  # never built: listing needs no Docker

    assert asyncio.run(service.list_tasks("eval")) == ["write-greeting"]
    with pytest.raises(
        ValueError,
        match=r"multi-container stack \(sidecar services\), and the 'docker' sandbox provider runs no multi-container",
    ):
        asyncio.run(service.get_task("needs-a-database"))


def test_the_reference_solution_passes_the_tasks_own_tests(tmp_path):
    """The task itself is sound: its solution scores 1 and doing nothing scores 0 (checked on the host)."""

    import subprocess

    def grade(solve: bool) -> str:
        root = tmp_path / ("solved" if solve else "untouched")
        (root / "logs" / "verifier").mkdir(parents=True)
        (root / "app").mkdir()
        env = {**os.environ, "PLATOON_SANDBOX_ROOT": str(root)}
        if solve:
            subprocess.run(["bash", str(TASKS / "write-greeting/solution/solve.sh")], cwd=root / "app", check=True)
        subprocess.run(["bash", str(TASKS / "write-greeting/tests/test.sh")], cwd=root / "app", env=env, check=True)
        return (root / "logs" / "verifier" / "reward.txt").read_text().strip()

    assert grade(solve=True) == "1"
    assert grade(solve=False) == "0"
