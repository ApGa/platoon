"""The user fixtures (``tests/envs/fixtures/harbor-users``) end to end on a runtime that honors users: fetched as a
dataset, their images built with the runtime's builder, then each graded after its reference solution (1.0) and
untouched (0.0), and the agent's own sandbox looked at through its tools.  ``test_harbor_users_singularity.py`` runs
them on Singularity with ``fakeroot``, ``tests/integration/test_harbor_users_docker.py`` on Docker."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from site_cache import use_site_cache

from platoon.envs.harbor import HarborEnvironmentService
from platoon.envs.harbor.builders import build_images
from platoon.envs.service import Outcome

FIXTURES = Path(__file__).parent / "envs" / "fixtures" / "harbor-users" / "tasks"
TASKS = ("agent-user", "image-user-65534", "shared-root-verifier", "workdir-after-user")


def fetched_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, provider: str, provider_options: dict[str, Any]
) -> HarborEnvironmentService:
    """A service that honors the fixtures' users, on a copy of them as a dataset to fetch, in a site cache of the
    test's own."""

    dataset = tmp_path / "dataset"
    shutil.copytree(FIXTURES, dataset)
    use_site_cache(monkeypatch, tmp_path / "cache")
    return HarborEnvironmentService(
        dataset=str(dataset),
        lock=str(tmp_path / "dataset.lock.json"),
        refuse={"agent_user": False},  # image-user-65534's and workdir-after-user's Dockerfiles defer USER
        agent_user="honor",
        provider=provider,
        provider_options=provider_options,
    )


async def fetch_and_build(service: HarborEnvironmentService, *, builder: str, scratch: Path) -> list[str]:
    """Fetch the dataset and build its images into its image map, as ``platoon prepare`` does; the task ids."""

    names = await service.list_tasks("train")
    locked = service.locked
    assert locked is not None and service.task_images.path is not None
    lock = locked.dataset(fetch_allowed=False).lock
    build_images(lock, cache=locked.cache, image_map=service.task_images.path, builder=builder, scratch=scratch)
    return sorted(names)


async def oracle_and_noop(service: HarborEnvironmentService, name: str) -> tuple[Outcome, Outcome]:
    """The task graded after its reference solution ran, and graded untouched, each in a session of its own."""

    outcomes = []
    for solve in (True, False):
        session = await service.open(await service.get_task(name), rollout_id=f"users-{name}-{solve}")
        try:
            if solve:
                solved = await session.run_solution()
                assert solved.exit_code == 0, solved.stdout + solved.stderr
            outcomes.append(await session.grade())
        finally:
            await session.close()
    return outcomes[0], outcomes[1]


async def the_agents_view(service: HarborEnvironmentService) -> dict[str, str]:
    """What the agent of ``agent-user`` finds through its tools: its uid and ``HOME``, and what ``edit`` says of the
    answer that only root may read."""

    session = await service.open(await service.get_task("agent-user"), rollout_id="users-view")
    try:
        who = await session.call_tool("bash", {"command": "id -u; echo $HOME"})
        edit = await session.call_tool("edit", {"path": "/srv/secret/answer.txt", "old_str": "42", "new_str": "0"})
        return {
            "who": who.content[0]["text"].strip(),
            "edit": edit.content[0]["text"],
            "edit_error": str(edit.is_error),
        }
    finally:
        await session.close()
