"""Separate verifiers of fetched tasks (process provider): the verifier runs the image the site's image map gives it,
with that image's deferred Dockerfile steps, and the task's index row from the lock reaches the reward contract's
check hook."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from platoon.envs.harbor import HarborEnvironmentService
from platoon.envs.harbor.images import ImageEntry, update_image_map
from platoon.envs.harbor.lock import read_lock
from platoon.envs.service import EnvSetupError, Outcome
from platoon.sandbox import SandboxSpec
from platoon.sandbox.process import ProcessProvider, ProcessSandbox

pytestmark = pytest.mark.asyncio

ROOT = "${PLATOON_SANDBOX_ROOT:-}"
# The verifier sees the answer it was handed, the input the deferred COPY put into its own sandbox, and the
# environment of the image the map gave it.
TEST_SH = f"""#!/bin/bash
mkdir -p {ROOT}/logs/verifier
if [ "$(cat {ROOT}/app/answer.txt 2>/dev/null)" = 42 ] && [ -s {ROOT}/app/input/data.csv ] \\
   && [ "$FROM_MAP" = "$EXPECTED_IMAGE" ]; then
  echo 1 > {ROOT}/logs/verifier/reward.txt
else
  echo 0 > {ROOT}/logs/verifier/reward.txt
fi
"""
SEEN: list[dict] = []


def record_index(rewards, verifier_dir, task) -> Outcome | None:
    """A check hook that keeps the index row it was given."""

    SEEN.append(dict(task.index))
    return None


class RecordingProvider(ProcessProvider):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.specs: list[SandboxSpec] = []

    async def start(self, spec: SandboxSpec) -> ProcessSandbox:
        self.specs.append(spec)
        return await super().start(spec)

    def verifier_specs(self) -> list[SandboxSpec]:
        return [spec for spec in self.specs if spec.labels.get("platoon.role") == "verifier"]


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _dataset(root: Path, *, verifier_dockerfile: bool) -> Path:
    files = {
        "task.toml": 'artifacts = ["/app/answer.txt"]\n[verifier]\nenvironment_mode = "separate"\ntimeout_sec = 30\n',
        "instruction.md": "Write 42 to answer.txt.\n",
        "environment/Dockerfile": "FROM python:3.12-slim\nRUN true\nWORKDIR /app\nCOPY input/ /app/input/\n",
        "environment/input/data.csv": "x\n1\n",
    }
    if verifier_dockerfile:
        # The verifier's own image holds its tests: test.sh arrives through the deferred COPY, never an upload.
        files["tests/Dockerfile"] = (
            "FROM python:3.12-slim\nRUN pip install pytest\nCOPY test.sh /tests/test.sh\nWORKDIR /app\n"
            "COPY data.csv /app/input/data.csv\n"
        )
        files["tests/data.csv"] = "x\n2\n"
    files["tests/test.sh"] = TEST_SH
    _write(root / "tasks" / "answer", files)
    (root / "registry.json").write_text(
        json.dumps([{"name": "d", "tasks": [{"name": "d--answer", "path": "tasks/answer"}]}])
    )
    (root / "task-index.json").write_text(json.dumps({"d--answer": {"source_dataset": "d/answers"}}))
    return root


def _service(tmp_path: Path, provider, **kwargs) -> HarborEnvironmentService:
    return HarborEnvironmentService(
        dataset=str(tmp_path / "dataset"),
        cache=str(tmp_path / "cache"),
        lock=str(tmp_path / "dataset.lock.json"),
        provider=provider,
        **kwargs,
    )


async def _grade_an_answer(service: HarborEnvironmentService) -> Outcome:
    session = await service.open(await service.get_task("d--answer"), rollout_id="r1")
    try:
        await session.call_tool("bash", {"command": "echo 42 > answer.txt"})
        return await session.grade()
    finally:
        await session.close()


async def test_a_verifier_on_the_tasks_image_gets_its_mapped_image_deferred_steps_and_index(tmp_path: Path):
    _dataset(tmp_path / "dataset", verifier_dockerfile=False)
    provider = RecordingProvider(tmp_path / "sandboxes")
    await _service(tmp_path, provider).list_tasks("train")  # the trainer fetches and locks
    key = read_lock(tmp_path / "dataset.lock.json").get("d--answer").image["key"]
    image_map = tmp_path / "images.json"
    update_image_map(image_map, {key: ImageEntry(image="local", env={"FROM_MAP": "task"})})
    SEEN.clear()
    service = _service(
        tmp_path,
        provider,
        image_map=str(image_map),
        env={"EXPECTED_IMAGE": "task"},
        reward={"primary": "reward", "check": f"{__name__}:record_index"},
    )

    outcome = await _grade_an_answer(service)

    assert (outcome.status, outcome.reward) == ("scored", 1.0), outcome
    (verifier,) = provider.verifier_specs()
    assert (verifier.image, verifier.workdir, verifier.env["FROM_MAP"]) == ("local", "/app", "task")
    assert [row.get("source_dataset") for row in SEEN] == ["d/answers"]  # the lock's index row reached the hook


async def test_a_verifier_built_from_tests_dockerfile_runs_its_own_mapped_image(tmp_path: Path):
    _dataset(tmp_path / "dataset", verifier_dockerfile=True)
    provider = RecordingProvider(tmp_path / "sandboxes")
    await _service(tmp_path, provider).list_tasks("train")
    row = read_lock(tmp_path / "dataset.lock.json").get("d--answer")
    task_key, verifier_key = row.image["key"], row.verifier_image["key"]
    assert task_key != verifier_key
    image_map = tmp_path / "images.json"
    update_image_map(image_map, {task_key: ImageEntry(image="local", env={"FROM_MAP": "task"})})
    service = _service(tmp_path, provider, image_map=str(image_map), env={"EXPECTED_IMAGE": "tests"})
    with pytest.raises(EnvSetupError, match="builds from tests/Dockerfile") as refused:
        await service.open(await service.get_task("d--answer"), rollout_id="r0")  # before the agent runs
    assert refused.value.failure_kind == "env_config"

    update_image_map(image_map, {verifier_key: ImageEntry(image="local", env={"FROM_MAP": "tests"})})
    outcome = await _grade_an_answer(
        _service(tmp_path, provider, image_map=str(image_map), env={"EXPECTED_IMAGE": "tests"})
    )

    assert (outcome.status, outcome.reward) == ("scored", 1.0), outcome
    verifier = provider.verifier_specs()[-1]
    assert (verifier.image, verifier.env["FROM_MAP"]) == ("local", "tests")
