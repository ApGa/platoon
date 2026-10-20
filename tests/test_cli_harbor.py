"""``platoon harbor``: fetch, verify, and plan and alias images, on a local dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from site_cache import use_site_cache

from platoon import cli

DOCKERFILE = "FROM python:3.12-slim\nRUN pip install numpy\nWORKDIR /app\nCOPY instance.json /app/instance.json\n"


def _exit_code(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    return exit_info.value.code


@pytest.fixture
def harbor_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The Harbor cache, in the site cache of the test's own (``tests/site_cache.py``)."""

    return use_site_cache(monkeypatch, tmp_path / "cache")


@pytest.fixture
def local_config(tmp_path: Path, harbor_cache: Path) -> Path:
    """A training config whose entry names a local dataset of two tasks that build one image, and the lock to write,
    relative to the config file."""

    for name in ("alpha", "beta"):
        files = {
            "task.toml": "",
            "instruction.md": f"Task {name}.\n",
            "environment/Dockerfile": DOCKERFILE,
            "environment/instance.json": json.dumps({"name": name}),
            "tests/test.sh": "echo 1\n",
        }
        for relative, text in files.items():
            path = tmp_path / "dataset" / name / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    config = tmp_path / "configs" / "train.yaml"
    config.parent.mkdir()
    config.write_text(
        "environments:\n"
        "  - service: harbor\n"
        "    service_kwargs: {dataset: ../dataset, lock: ../locks/dataset.lock.json}\n"
        "    harness: platoon-bash\n"
        "train: {model_name: m, renderer_name: r}\n"
    )
    return config


def test_fetch_verify_plan_and_alias(local_config: Path, harbor_cache: Path, tmp_path: Path, capsys):
    cli.main(["harbor", "fetch", str(local_config)])
    printed = json.loads(capsys.readouterr().out)
    lock = tmp_path / "locks" / "dataset.lock.json"
    assert printed == {"environment": "default", "lock": str(lock), "tasks": 2, "runnable": 2, "fetched": True}

    cli.main(["harbor", "verify", str(local_config)])
    assert "environments[default]: 2 of 2 task(s) match their digests" in capsys.readouterr().err

    cli.main(["harbor", "images", "plan", str(lock), "--json-output"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["missing"] == 1 and plan["images"][0]["kind"] == "shared" and plan["images"][0]["tasks"] == 2

    image_map = tmp_path / "site" / "images.json"
    cli.main(["harbor", "images", "alias", str(lock), "--image-map", str(image_map), "--image", "local"])
    capsys.readouterr()
    cli.main(["harbor", "images", "plan", str(lock), "--image-map", str(image_map)])
    table = capsys.readouterr().out
    assert "shared" in table and "1 image(s), 0 missing from the image map" in table

    entry = next((harbor_cache / "tasks").iterdir())
    (entry / "instruction.md").write_text("tampered\n")
    assert _exit_code(["harbor", "verify", str(lock), "--cache", str(harbor_cache)]) == 1
    assert "the lock says" in capsys.readouterr().out
    assert _exit_code(["harbor", "verify", str(lock)]) == 1  # a lock file's tasks are in the Harbor cache by default


def test_images_verify_checks_the_image_files_against_their_sha256(local_config: Path, tmp_path: Path, capsys):
    from platoon.envs.harbor.hashing import file_sha256
    from platoon.envs.harbor.images import ImageEntry, update_image_map

    cli.main(["harbor", "fetch", str(local_config)])
    lock = tmp_path / "locks" / "dataset.lock.json"
    image_map = tmp_path / "site" / "images.json"
    capsys.readouterr()
    assert _exit_code(["harbor", "images", "verify", str(lock), "--image-map", str(image_map)]) == 78
    assert "does not exist" in capsys.readouterr().err

    cli.main(["harbor", "images", "plan", str(lock), "--json-output"])
    key = json.loads(capsys.readouterr().out)["images"][0]["key"]
    sif = (tmp_path / "site" / "images" / "shared.sif").resolve()
    sif.parent.mkdir(parents=True)
    sif.write_bytes(b"image")
    recorded = file_sha256(sif)
    update_image_map(image_map, {key: ImageEntry(image="images/shared.sif", sha256=recorded, builder="oci")})
    verify = ["harbor", "images", "verify", str(lock), "--image-map", str(image_map)]
    cli.main(verify)
    checked = capsys.readouterr()
    assert checked.out == ""
    assert "1 image(s): 1 match their SHA-256, 0 missing or mismatched, 0 not checked" in checked.err

    sif.write_bytes(b"another image")
    assert _exit_code(verify) == 1
    mismatched = capsys.readouterr()
    problem = f"{sif} has SHA-256 {file_sha256(sif)}, the image map says {recorded}"
    assert mismatched.out == f"{key} (environment, 2 task(s)): {problem}\n"
    assert "0 match their SHA-256, 1 missing or mismatched" in mismatched.err

    sif.unlink()
    assert _exit_code(verify) == 1
    assert capsys.readouterr().out == f"{key} (environment, 2 task(s)): {sif} does not exist\n"
    cli.main(["harbor", "images", "plan", str(lock), "--image-map", str(image_map)])
    assert "1 image(s), 1 missing from the image map" in capsys.readouterr().out  # the entry has no file

    update_image_map(image_map, {key: ImageEntry(image=f"docker://registry.example/shared@sha256:{'0' * 64}")})
    cli.main(verify)
    unchecked = capsys.readouterr()
    assert unchecked.out == "" and "is not a local image file, so it is not checked" in unchecked.err
    assert "0 match their SHA-256, 0 missing or mismatched, 1 not checked" in unchecked.err


def test_a_local_directory_is_fetched_again_with_its_changes(
    local_config: Path, harbor_cache: Path, tmp_path: Path, capsys
):
    """A local directory's files may change under the same path, so every fetch reads them again: a changed task gets
    its new digest, and a damaged cache entry is replaced, as before."""

    cli.main(["harbor", "fetch", str(local_config)])
    lock = tmp_path / "locks" / "dataset.lock.json"
    before = json.loads(lock.read_text())["tasks"]
    capsys.readouterr()
    (tmp_path / "dataset" / "alpha" / "instruction.md").write_text("Task alpha, revised.\n")
    damaged = harbor_cache / next(row["path"] for row in before if row["name"] == "beta")
    (damaged / "instruction.md").write_text("damaged\n")

    cli.main(["harbor", "fetch", str(local_config)])

    assert json.loads(capsys.readouterr().out)["fetched"] is True
    after = {row["name"]: row["digest"] for row in json.loads(lock.read_text())["tasks"]}
    assert after["alpha"] != next(row["digest"] for row in before if row["name"] == "alpha")
    assert (damaged / "instruction.md").read_text() == "Task beta.\n"


def test_configuration_problems_exit_78(tmp_path: Path, capsys):
    (tmp_path / "bad.yaml").write_text("train: {model_name: m, renderer_name: r}\n")
    assert _exit_code(["harbor", "fetch", str(tmp_path / "bad.yaml")]) == 78
    assert "is not a training or evaluation config: it lists no environments and extends no config" in (
        capsys.readouterr().err
    )
    assert _exit_code(["harbor", "fetch", str(tmp_path / "missing.lock.json")]) == 78
    assert "platoon harbor fetch takes a training or evaluation config, not a lock file" in capsys.readouterr().err
    assert _exit_code(["harbor", "verify", str(tmp_path / "missing.lock.json")]) == 78
    assert _exit_code(["harbor", "images", "build", str(tmp_path / "missing.lock.json"), "--image-map", "m.json"]) == 78


def test_harbor_help_lists_its_commands(capsys):
    cli.main(["harbor", "--help"])
    out = capsys.readouterr().out
    assert "fetch" in out and "verify" in out and "images" in out


# --- training and evaluation configs ---------------------------------------------------------------------------


@pytest.fixture
def run_config(tmp_path: Path, monkeypatch) -> Path:
    """A training config whose Harbor entry names a local dataset of ten tasks in five groups, with a hold-out, relative
    to the config file, as every input path is; the site profile's cache, site/ in the working directory, holds the
    Harbor cache."""

    rows = {}
    for n in range(10):
        name = f"task-{n}"
        for relative, text in {
            "task.toml": "",
            "instruction.md": f"Task {name}.\n",
            "environment/Dockerfile": DOCKERFILE,
            "environment/instance.json": json.dumps({"name": name}),
            "tests/test.sh": "echo 1\n",
        }.items():
            path = tmp_path / "dataset" / name / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        rows[name] = {"task_path": name, "group": f"g{n // 2}"}
    (tmp_path / "dataset" / "task-index.json").write_text(json.dumps(rows))
    config = tmp_path / "configs" / "train.yaml"
    config.parent.mkdir()
    config.write_text(
        "environments:\n"
        "  - name: science\n"
        "    service: harbor\n"
        "    service_kwargs:\n"
        "      dataset: ../dataset\n"
        "      holdout: {fraction: 0.2, group_by: group}\n"
        "    harness: platoon-bash\n"
        "train: {model_name: m, renderer_name: r}\n"
    )
    (tmp_path / "site.yaml").write_text(f"cache: {tmp_path / 'site'}\n")
    monkeypatch.setenv("PLATOON_SITE", str(tmp_path / "site.yaml"))
    monkeypatch.chdir(tmp_path)
    return config


def test_fetch_takes_a_training_config_and_prints_its_hold_out(run_config: Path, tmp_path: Path, capsys):
    cli.main(["harbor", "fetch", str(run_config)])

    out, err = capsys.readouterr()
    printed = json.loads(out)
    lock = tmp_path / "site" / "harbor" / "locks"
    assert printed["environment"] == "science" and printed["fetched"] is True
    assert (printed["tasks"], printed["runnable"]) == (10, 10)
    assert Path(printed["lock"]).parent == lock  # the entry's default lock, which platoon train reads
    assert printed["holdout"]["eval"] == 2 and printed["holdout"]["train"] == 8  # one group of two
    assert (
        "environments[science]: hold-out: 20% of the dataset, grouped by group, joined by shared input files, seed 0"
        in err
    )
    assert "  groups: 5 by group; no two of them share an input file; largest g0: 2 task(s)" in err
    assert f"eval digest {printed['holdout']['eval_digest'][:12]}" in err


def test_fetch_takes_overrides_like_platoon_train(run_config: Path, tmp_path: Path, capsys):
    cli.main(["harbor", "fetch", str(run_config), "environments.0.service_kwargs.lock=locks/mine.lock.json"])

    assert Path(json.loads(capsys.readouterr().out)["lock"]) == tmp_path / "locks" / "mine.lock.json"


def test_verify_takes_a_training_config(run_config: Path, tmp_path: Path, capsys):
    assert _exit_code(["harbor", "verify", str(run_config)]) == 78  # nothing fetched: no lock answers it
    assert "no lock answers its dataset settings" in capsys.readouterr().err

    cli.main(["harbor", "fetch", str(run_config)])
    capsys.readouterr()
    cli.main(["harbor", "verify", str(run_config)])
    assert "environments[science]: 10 of 10 task(s) match their digests" in capsys.readouterr().err

    entry = next((tmp_path / "site" / "harbor" / "tasks").iterdir())
    (entry / "instruction.md").write_text("tampered\n")
    assert _exit_code(["harbor", "verify", str(run_config)]) == 1
    assert "the lock says" in capsys.readouterr().out
    assert _exit_code(["harbor", "verify", str(run_config), "--task", "task-0", "--task", "nope"]) == 78


def test_verify_refuses_a_hold_out_whose_lock_records_no_input_files(run_config: Path, tmp_path: Path, capsys):
    """A lock written before platoon recorded input files cannot serve a hold-out, which keeps the tasks that share
    one on one side: verify exits 78 naming the fetch, as before any fetch, and a fetch records them."""

    from dataclasses import replace

    from platoon.envs.harbor.lock import read_lock, write_lock

    cli.main(["harbor", "fetch", str(run_config)])
    lock_path = Path(json.loads(capsys.readouterr().out)["lock"])
    lock = read_lock(lock_path)
    write_lock(replace(lock, tasks=tuple(replace(task, input_files=None) for task in lock.tasks)), lock_path)

    assert _exit_code(["harbor", "verify", str(run_config)]) == 78
    assert "records no input files, which its hold-out needs" in capsys.readouterr().err
    cli.main(["harbor", "verify", str(run_config), "environments.0.service_kwargs.holdout=null"])  # no hold-out: ok
    assert "10 of 10 task(s) match their digests" in capsys.readouterr().err

    cli.main(["harbor", "fetch", str(run_config)])  # a local directory is fetched again in any case
    assert read_lock(lock_path).records_input_files
    capsys.readouterr()
    cli.main(["harbor", "verify", str(run_config)])
    assert "10 of 10 task(s) match their digests" in capsys.readouterr().err


def test_run_config_problems_exit_78(run_config: Path, tmp_path: Path, capsys):
    assert _exit_code(["harbor", "verify", str(run_config), "--cache", str(tmp_path / "c")]) == 78
    assert "--cache applies to a lock file; a training or evaluation config's entries keep their tasks" in (
        capsys.readouterr().err
    )
    assert _exit_code(["harbor", "fetch", str(run_config), "environments.0.service_kwargs.holdout.fraction=2"]) == 78
    assert "environments[science].service_kwargs: holdout.fraction must be a number between 0 and 1" in (
        capsys.readouterr().err
    )
    assert _exit_code(["harbor", "verify", str(tmp_path / "missing.lock.json"), "workers=2"]) == 78
    assert "overrides apply to training and evaluation configs, not to a lock file" in capsys.readouterr().err
    (tmp_path / "configs" / "tables.yaml").write_text("environments: [{service: table, service_kwargs: {}}]\n")
    assert _exit_code(["harbor", "fetch", str(tmp_path / "configs" / "tables.yaml")]) == 78
    assert "has no Harbor environment entry" in capsys.readouterr().err


def _fake_builds(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Builds that write an image file instead of running a builder; returns (key, builder) of each build."""

    from platoon.envs.harbor import builders
    from platoon.envs.harbor.hashing import file_sha256
    from platoon.envs.harbor.images import ImageEntry

    calls: list[tuple[str, str]] = []

    def build_image(task_dir: Path, key: str, *, builder: str, out_dir: Path, **_: object) -> ImageEntry:
        calls.append((key, builder))
        out_dir.mkdir(parents=True, exist_ok=True)
        image = out_dir / f"{key.removeprefix('sha256:')}.sif"
        image.write_bytes(b"an image")
        return ImageEntry(image=str(image), sha256=file_sha256(image), builder=builder)

    monkeypatch.setattr(builders, "build_image", build_image)
    return calls


def test_images_take_a_training_config_and_its_entrys_default_image_map(
    run_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """The images commands act on each entry's lock with its image map: images/<provider>.json in the cache when the
    entry names none, built with its provider's builder into images/<provider>/."""

    calls = _fake_builds(monkeypatch)
    singularity = "environments.0.service_kwargs.provider=singularity"
    assert _exit_code(["harbor", "images", "plan", str(run_config)]) == 78  # nothing fetched yet
    assert "no lock answers its dataset settings" in capsys.readouterr().err
    cli.main(["harbor", "fetch", str(run_config)])
    capsys.readouterr()

    cli.main(["harbor", "images", "plan", str(run_config), singularity, "--json-output"])
    plan = json.loads(capsys.readouterr().out)["science"]
    default = tmp_path / "site" / "harbor" / "images" / "singularity.json"
    assert plan["image_map"] == str(default) and plan["missing"] == 1

    cli.main(["harbor", "images", "build", str(run_config), singularity])
    built = json.loads(capsys.readouterr().out)
    (key,) = built["built"]
    assert calls == [(key, "oci")] and built["image_map"] == str(default)
    assert built["built"][key] == str(tmp_path / "site" / "harbor" / "images" / "singularity" / f"{key[7:]}.sif")
    cli.main(["harbor", "images", "verify", str(run_config), singularity])
    assert "environments[science]: 1 image(s): 1 match their SHA-256" in capsys.readouterr().err
    cli.main(["harbor", "images", "build", str(run_config), singularity])
    assert json.loads(capsys.readouterr().out)["built"] == {} and len(calls) == 1  # nothing missing: nothing built

    cli.main(["harbor", "images", "alias", str(run_config), "--image", "local"])  # the docker provider's map
    assert json.loads(capsys.readouterr().out) == [key]
    harbor = tmp_path / "site" / "harbor"
    assert json.loads((harbor / "images" / "docker.json").read_text())["images"][key]["image"] == "local"

    enroot = ["harbor", "images", "build", str(run_config), "environments.0.service_kwargs.provider=enroot"]
    assert _exit_code(enroot) == 78
    assert "the enroot provider has no builder of its own" in capsys.readouterr().err
    cli.main([*enroot, "--builder", "docker", "--push", "registry.example/tasks"])
    assert calls[-1] == (key, "docker")
    assert _exit_code(["harbor", "images", "plan", str(run_config), "--image-map", "m.json"]) == 78
    assert "--image-map and --cache apply to lock files" in capsys.readouterr().err


# The fetch config files that platoon 0.3.0 removed, and the options of an entry it removed, are refused in
# tests/test_removed_forms.py.
