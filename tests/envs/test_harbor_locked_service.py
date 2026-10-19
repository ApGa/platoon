"""The Harbor service on a fetched dataset: ids from the lock, auto-fetch in the trainer only, and session images
from the image map with deferred Dockerfile steps applied at open (process provider)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from platoon.envs.harbor import HarborEnvironmentService
from platoon.envs.harbor.images import ImageEntry, update_image_map
from platoon.envs.harbor.lock import read_lock
from platoon.envs.service import EnvSetupError
from platoon.sandbox.process import ProcessProvider
from platoon.task import Task

pytestmark = pytest.mark.asyncio

VERIFIER_DIR = '"${PLATOON_SANDBOX_ROOT:-}/logs/verifier"'
DOCKERFILE = """FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.4.6
WORKDIR /app
ENV GREETING=hello
COPY input/ /app/input/
COPY instance.json /app/instance.json
"""
TEST_SH = f"""#!/bin/bash
mkdir -p {VERIFIER_DIR}
if [ -s "${{PLATOON_SANDBOX_ROOT:-}}/app/input/data.csv" ] && [ "$GREETING" = hello ]; then
  echo 1 > {VERIFIER_DIR}/reward.txt
else
  echo 0 > {VERIFIER_DIR}/reward.txt
fi
"""


def _dataset(root: Path) -> Path:
    for name in ("alpha", "beta"):
        files = {
            "task.toml": "[verifier]\ntimeout_sec = 30.0\n",
            "instruction.md": f"Task {name}.\n",
            "environment/Dockerfile": DOCKERFILE,
            "environment/input/data.csv": f"x\n{name}\n",
            "environment/instance.json": json.dumps({"name": name}),
            "tests/test.sh": TEST_SH,
        }
        for relative, text in files.items():
            path = root / name / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    (root / "splits.json").write_text(json.dumps({"train": ["alpha"], "eval": ["beta"]}))
    return root


def _service(tmp_path: Path, **kwargs) -> HarborEnvironmentService:
    options = {
        "dataset": str(tmp_path / "dataset"),
        "cache": str(tmp_path / "cache"),
        "lock": str(tmp_path / "dataset.lock.json"),
        "provider": ProcessProvider(root=tmp_path / "sandboxes"),
        **kwargs,
    }
    return HarborEnvironmentService(**options)


async def test_trainer_fetches_and_workers_read_the_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "node"))
    _dataset(tmp_path / "dataset")
    worker = _service(tmp_path)
    with pytest.raises(EnvSetupError, match="does not exist") as refused:
        await worker.get_task("alpha")  # a worker never fetches
    assert refused.value.failure_kind == "env_config"
    trainer = _service(tmp_path, splits=str(tmp_path / "dataset" / "splits.json"))
    assert await trainer.list_tasks("train") == ["alpha"] and await trainer.list_tasks("eval") == ["beta"]
    lock = read_lock(tmp_path / "dataset.lock.json")
    assert [task.name for task in lock.tasks] == ["alpha", "beta"]
    task = await _service(tmp_path).get_task("beta")
    misc = task.misc["harbor"]
    assert misc["digest"] == lock.get("beta").digest and misc["image_key"] == lock.get("beta").image["key"]
    assert Path(misc["task_dir"]) == (tmp_path / "cache" / lock.get("beta").path).resolve()
    assert list((tmp_path / "node").rglob("verified/*"))  # checked once on this node


async def test_exclude_tasks_finds_a_locked_task_in_another_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An exclude_tasks item that the run's splits do not hold is looked for in the service's other splits
    (``list_splits``), which a lock-backed service reads from its manifest as a directory-backed one does."""

    from platoon.train.exclusion import UnknownTasks, excluded_service_tasks

    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "node"))
    _dataset(tmp_path / "dataset")
    service = _service(tmp_path, splits=str(tmp_path / "dataset" / "splits.json"))
    assert await service.list_splits() == ["train", "eval"]
    train = await service.list_tasks("train")

    assert await excluded_service_tasks(service, ["train"], train, ["beta"]) == set()  # an eval task, known
    with pytest.raises(UnknownTasks, match="gamma"):
        await excluded_service_tasks(service, ["train"], train, ["gamma"])
    assert await _service(tmp_path).list_splits() == []  # no manifest: every split holds every task


async def test_sessions_run_the_mapped_image_with_the_deferred_steps(tmp_path: Path):
    _dataset(tmp_path / "dataset")
    service = _service(tmp_path)
    await service.list_tasks("train")
    lock = read_lock(tmp_path / "dataset.lock.json")
    key = lock.get("alpha").image["key"]
    assert lock.get("alpha").image["kind"] == "shared" and lock.get("beta").image["key"] == key

    task = await service.get_task("alpha")
    with pytest.raises(EnvSetupError, match="no image map") as refused:
        await service.open(task, rollout_id="r0")  # never run on a default image
    assert refused.value.failure_kind == "env_config"

    image_map = tmp_path / "site" / "images.json"
    update_image_map(image_map, {key: ImageEntry(image="local", env={"PATH": "/usr/bin:/bin", "BASE": "1"})})
    service = _service(tmp_path, image_map=str(image_map), env={"EXTRA": "x"})
    session = await service.open(await service.get_task("alpha"), rollout_id="r1")
    try:
        spec = session.sandbox.spec
        assert spec.image == "local" and spec.workdir == "/app"
        assert spec.env == {"PATH": "/usr/bin:/bin", "BASE": "1", "GREETING": "hello", "EXTRA": "x"}
        assert session.sandbox.host_path("/app/input/data.csv").read_text() == "x\nalpha\n"
        assert json.loads(session.sandbox.host_path("/app/instance.json").read_text()) == {"name": "alpha"}
        outcome = await session.grade()
        assert (outcome.status, outcome.reward) == ("scored", 1.0)
    finally:
        await session.close()


async def test_a_fetched_dataset_runs_its_default_image_map_once_it_exists(tmp_path: Path):
    """A fetched dataset whose entry names no image map uses images/<provider>.json in its cache, where platoon
    prepare builds its images; until that file exists there is no map, and a task built from a Dockerfile is refused
    as before."""

    _dataset(tmp_path / "dataset")
    service = _service(tmp_path)
    await service.list_tasks("train")
    default = tmp_path / "cache" / "images" / "process.json"
    assert service.task_images.path == default and service.task_images.is_default
    with pytest.raises(EnvSetupError, match="no image map"):
        await service.open(await service.get_task("alpha"), rollout_id="r0")

    key = read_lock(tmp_path / "dataset.lock.json").get("alpha").image["key"]
    update_image_map(default, {key: ImageEntry(image="local", env={"PATH": "/usr/bin:/bin"})})
    service = _service(tmp_path)
    session = await service.open(await service.get_task("alpha"), rollout_id="r1")
    try:
        assert session.sandbox.spec.image == "local"
        assert session.sandbox.host_path("/app/input/data.csv").read_text() == "x\nalpha\n"  # the deferred COPY
    finally:
        await session.close()

    named = tmp_path / "site" / "images.json"
    assert _service(tmp_path, image_map=str(named)).task_images.path == named  # an entry's own map wins
    assert not _service(tmp_path, image_map=str(named)).task_images.is_default
    assert _service(tmp_path, image_map={"images": {}}).task_images.path is None  # an inline map has no file
    other = HarborEnvironmentService(dataset=str(tmp_path / "dataset"), cache=str(tmp_path / "c"), provider="docker")
    assert other.task_images.path == tmp_path / "c" / "images" / "docker.json"  # one map per provider


async def test_a_dataset_directory_never_takes_a_default_image_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An entry that serves a directory keeps running its tasks' docker_image or the service's default_image unless it
    names an image map, even when the cache's default map holds an image for its tasks' build key (I-R10)."""

    from platoon.envs.harbor.dockerfile import plan_dockerfile

    root = _dataset(tmp_path / "dataset")
    monkeypatch.setenv("PLATOON_HARBOR_CACHE", str(tmp_path / "cache"))
    key = plan_dockerfile(root / "alpha" / "environment" / "Dockerfile").key
    update_image_map(tmp_path / "cache" / "images" / "process.json", {key: ImageEntry(image="built-for-the-lock")})
    service = HarborEnvironmentService(
        str(root), provider=ProcessProvider(root=tmp_path / "sandboxes"), default_image="local"
    )
    assert service.task_images.path is None and not service.task_images.is_default
    session = await service.open(await service.get_task("alpha"), rollout_id="r3")
    try:
        assert session.sandbox.spec.image == "local"  # the default image, as before
    finally:
        await session.close()


async def test_listing_refuses_tasks_whose_images_the_map_lacks(tmp_path: Path):
    """A run never builds an image: the tasks of a fetched dataset whose images the map lacks (or whose image file is
    gone) are refused when they are listed, as not prepared; a dataset directory's are not checked (I-R10)."""

    from platoon.envs.service import EnvNotPrepared

    _dataset(tmp_path / "dataset")
    service = _service(tmp_path)
    names = await service.list_tasks("train")
    default = tmp_path / "cache" / "images" / "process.json"
    key = read_lock(tmp_path / "dataset.lock.json").get("alpha").image["key"]
    with pytest.raises(EnvNotPrepared) as refused:
        await service.check_tasks("train", names)
    assert refused.value.failure_kind == "env_config"
    assert str(refused.value) == (
        "2 of the 2 tasks of split 'train' build their images from Dockerfiles, which no image map holds (the default "
        f"image map {default} does not exist): 1 image(s), such as {key} for 2 task(s) (alpha); a run never builds them"
    )

    update_image_map(default, {key: ImageEntry(image="local")})
    await _service(tmp_path).check_tasks("train", names)  # prepared
    update_image_map(default, {key: ImageEntry(image=str(tmp_path / "gone.sif"))})
    with pytest.raises(EnvNotPrepared, match=f"which the image map {default} lacks"):
        await _service(tmp_path).check_tasks("train", names)

    directory = HarborEnvironmentService(
        str(tmp_path / "dataset"), provider=ProcessProvider(root=tmp_path / "sandboxes"), default_image="local"
    )
    await directory.check_tasks("train", ["alpha", "beta"])  # planned when a session opens, as before


async def test_a_lock_alone_serves_its_tasks(tmp_path: Path):
    _dataset(tmp_path / "dataset")
    await _service(tmp_path).list_tasks("train")
    service = HarborEnvironmentService(
        lock=str(tmp_path / "dataset.lock.json"), cache=str(tmp_path / "cache"), provider="process"
    )
    assert await service.list_tasks("anything") == ["alpha", "beta"]
    with pytest.raises(ValueError, match="not both"):
        HarborEnvironmentService(str(tmp_path / "dataset"), lock=str(tmp_path / "dataset.lock.json"))
    with pytest.raises(ValueError, match="choose what a fetch"):
        HarborEnvironmentService(lock=str(tmp_path / "dataset.lock.json"), tasks=["x.json"])


async def test_an_image_map_also_serves_tasks_outside_a_lock(tmp_path: Path):
    root = _dataset(tmp_path / "dataset")
    from platoon.envs.harbor.dockerfile import plan_dockerfile

    key = plan_dockerfile(root / "alpha" / "environment" / "Dockerfile").key
    service = HarborEnvironmentService(
        str(root),
        provider=ProcessProvider(root=tmp_path / "sandboxes"),
        image_map={"images": {key: {"image": "local"}}},
    )
    session = await service.open(await service.get_task("beta"), rollout_id="r2")
    try:
        assert session.sandbox.host_path("/app/input/data.csv").read_text() == "x\nbeta\n"
    finally:
        await session.close()


async def test_a_locked_task_found_by_its_id_alone_keeps_its_index_row(tmp_path: Path):
    root = _dataset(tmp_path / "dataset")
    rows = {name: {"task_path": name, "source_dataset": f"source-{name}"} for name in ("alpha", "beta")}
    (root / "task-index.json").write_text(json.dumps(rows))
    await _service(tmp_path).list_tasks("train")  # the trainer fetches
    row = read_lock(tmp_path / "dataset.lock.json").get("beta")
    assert row.index == {"source_dataset": "source-beta"}
    harbor_task = _service(tmp_path).harbor_task(Task(id="beta"))  # no misc: looked up in the lock by its id
    assert harbor_task.task_dir == (tmp_path / "cache" / row.path).resolve()
    assert harbor_task.index == {"source_dataset": "source-beta"}  # what reward-contract hooks read


async def test_verifier_images_from_tests_dockerfiles(tmp_path: Path):
    from platoon.envs.harbor.locked import LockedTasks, TaskImages

    root = _dataset(tmp_path / "dataset")
    for name in ("alpha", "beta"):
        (root / name / "task.toml").write_text('[verifier]\nenvironment_mode = "separate"\n')
        (root / name / "tests" / "Dockerfile").write_text(
            "FROM python:3.12-slim\nRUN pip install pytest\nCOPY expected.json /grader/\n"
        )
        (root / name / "tests" / "expected.json").write_text(json.dumps({"name": name}))
    locked = LockedTasks(dataset=str(root), cache=str(tmp_path / "cache"), lock=str(tmp_path / "l.json"))
    lock = locked.dataset(fetch_allowed=True).lock
    row = lock.get("beta")
    assert row.flags["verifier"] == "separate" and row.verifier_image["kind"] == "shared"
    assert row.verifier_image["deferred"] == [["COPY", "expected.json", "/grader/"]]
    task_dir = tmp_path / "cache" / row.path
    images = TaskImages({"images": {row.verifier_image["key"]: {"image": "/v.sif", "workdir": "/grader"}}}, locked)
    verifier = images.for_verifier(task_dir)
    assert verifier.image == "/v.sif" and verifier.context == task_dir / "tests"
    assert [(op.kind, op.target, op.source) for op in verifier.files] == [
        ("copy", "/grader/expected.json", "expected.json")
    ]
    assert TaskImages(None, None).for_verifier(task_dir.parent / "elsewhere", docker_image="python:3.12") is None


async def test_a_trainers_network_check_reads_locked_tasks_without_hashing_them(tmp_path, monkeypatch):
    """Checking the listed tasks on a provider that refuses sandboxes without a network reads each task's task.toml,
    never its cache entry's digest, which opening a session checks (hashing every file of the task)."""

    import platoon.envs.harbor.lock as lock_module

    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "node"))
    _dataset(tmp_path / "dataset")
    assert await _service(tmp_path).list_tasks("train") == ["alpha", "beta"]  # fetches, and locks

    def hashed(directory, digest):
        raise AssertionError(f"hashed {directory}")

    monkeypatch.setattr(lock_module, "check_entry", hashed)
    key = read_lock(tmp_path / "dataset.lock.json").get("alpha").image["key"]
    update_image_map(tmp_path / "cache" / "images" / "singularity.json", {key: ImageEntry(image="docker://built")})
    service = HarborEnvironmentService(
        lock=str(tmp_path / "dataset.lock.json"), cache=str(tmp_path / "cache"), provider="singularity"
    )
    assert await service.list_tasks("train") == ["alpha", "beta"]
    await service.check_tasks("train", ["alpha", "beta"])  # online tasks, their image built: nothing refused


async def test_a_locked_tasks_overhead_is_read_without_hashing_it(tmp_path, monkeypatch):
    import platoon.envs.harbor.lock as lock_module

    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "node"))
    _dataset(tmp_path / "dataset")
    await _service(tmp_path).list_tasks("train")

    def hashed(directory, digest):
        raise AssertionError(f"hashed {directory}")

    monkeypatch.setattr(lock_module, "check_entry", hashed)
    service = HarborEnvironmentService(lock=str(tmp_path / "dataset.lock.json"), cache=str(tmp_path / "cache"))
    assert service.rollout_overhead_s("alpha") == 600 + 30 + 120 + 30  # a start, the 30 s verifier, grading, close
