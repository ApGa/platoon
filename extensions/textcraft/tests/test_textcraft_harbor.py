"""TextCraft-Synth as a Harbor dataset (``platoon_textcraft.harbor``): the export is deterministic, reproduces the
tasks published on the Hub and lays out Harbor tasks that share one image, the in-container game plays as the ORS
environment does, and the verifier scores only legal episodes that reach the goal, whatever the agent leaves
behind."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import shutil
import stat
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from platoon_textcraft import env as textcraft_env
from platoon_textcraft.env import CraftParams, FinishParams, GetInfoParams, TextCraftSynthEnvironment
from platoon_textcraft.harbor import export, game

REPO = Path(__file__).resolve().parents[3]
LIMIT = 3  # tasks per split
LOG = "app/textcraft/actions.jsonl"
# What the dataset published on the Hub (apurvaga/textcraft-synth, exported at platoon a924801) holds, which every
# export must keep reproducing: the content hashes of the first LIMIT tasks of each split (its task-index.json), the
# game file every task carries, and the task files the card names, which the tasks are built from.
PUBLISHED_DIGESTS = {
    "textcraft-synth-train-medium-00000": "sha256:9f199db57d2bd003aacf07c740f99906f47be7cdef0e734504c53d88ab2cb1b7",
    "textcraft-synth-train-medium-00001": "sha256:d99b1338b2e8f56d08ff9fe8445cea4157a6d36ca4ad5fc1e0bac6e7467ac8c5",
    "textcraft-synth-train-hard-00002": "sha256:269376baf845240ac4f18d0d9092ea9bf44a140e98762d1ff7cb923126ce5b54",
    "textcraft-synth-val-easy-00000": "sha256:485b7f16dc5eb44efb5d1fd0bd10a2c2b380ca6bca0a304bf017a2427bbb3627",
    "textcraft-synth-val-medium-00001": "sha256:8e5d36243212f7639c386fa46d5a2139c9b0fbddd6d88e1fd50442c6ca4ed047",
    "textcraft-synth-val-medium-00002": "sha256:9dd00deba177a5bb71a97a207220a076c42300f1692ca4a7e11b6b800e877e8b",
}
PUBLISHED_GAME_SHA256 = "61e5883d941e1b9f98aec1c118e53154853d273690268110f0abeb89867e6de7"
PUBLISHED_TASK_FILES_SHA256 = {
    "train": "685823fbca90aa89e061e61fef58d0122def4221eed14cf2c8c730341363a00f",
    "val": "84a123ee46e29e65f4d7b95f4943aa56907fc3fd5992268faa749602579dfc0c",
}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("export") / "textcraft-synth"
    export.export_dataset(out, limit=LIMIT, source_revision="0123abc")
    return out


def _tasks(dataset: Path) -> list[Path]:
    return sorted(path for path in (dataset / "tasks").iterdir())


def _files(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ---------------------------------------------------------------------------------------------------- the export


def test_the_same_inputs_export_the_same_bytes(dataset: Path, tmp_path: Path):
    again = tmp_path / "again"
    export.export_dataset(again, limit=LIMIT, source_revision="0123abc")
    assert _files(again) == _files(dataset)
    with pytest.raises(FileExistsError):
        export.export_dataset(again, limit=LIMIT)


def test_the_export_reproduces_the_tasks_published_on_the_hub(dataset: Path):
    index = json.loads((dataset / "task-index.json").read_text())

    assert {name: row["digest"] for name, row in index.items()} == PUBLISHED_DIGESTS
    assert hashlib.sha256((export.HERE / "game.py").read_bytes()).hexdigest() == PUBLISHED_GAME_SHA256
    for split, digest in PUBLISHED_TASK_FILES_SHA256.items():
        path = textcraft_env.DATA_DIR / TextCraftSynthEnvironment.split_files[split]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path


def test_registry_index_and_splits_describe_every_task(dataset: Path):
    names = [path.name for path in _tasks(dataset)]
    assert len(names) == 2 * LIMIT
    registry = json.loads((dataset / "registry.json").read_text())
    assert [(entry["name"], entry["version"]) for entry in registry] == [
        ("textcraft-synth-train", "1.0"),
        ("textcraft-synth-val", "1.0"),
    ]
    listed = [task["name"] for entry in registry for task in entry["tasks"]]
    assert sorted(listed) == names and len(set(listed)) == len(listed)
    assert all(task["path"] == f"tasks/{task['name']}" for entry in registry for task in entry["tasks"])

    index = json.loads((dataset / "task-index.json").read_text())
    assert sorted(index) == names
    row = index["textcraft-synth-train-medium-00000"]
    assert (
        row["split"] == "train" and row["difficulty"] == "medium" and row["source_task_id"] == "textcraft_synth.train.0"
    )
    assert row["env_task_id"] == "train:0" and row["env_split_task_id"] == "train-medium:0"
    spec = TextCraftSynthEnvironment.list_tasks("train-medium")[0]
    assert row["goal"] == spec["goal"] and row["target_items"] == spec["misc"]["target_items"]
    for name, row in index.items():
        assert row["digest"] == _harbor_content_hash(dataset / row["task_path"])

    splits = json.loads((dataset / "splits.json").read_text())
    assert splits["train"] == [task["name"] for task in registry[0]["tasks"]]
    assert set(splits["train-medium"]) <= set(splits["train"])
    assert all(index[name]["difficulty"] == "medium" for name in splits["train-medium"])


def _harbor_content_hash(task_dir: Path) -> str:
    """harbor-packager-v1, from the files on disk."""

    files = [name for name in ("task.toml", "instruction.md", "README.md") if (task_dir / name).is_file()]
    for directory in ("environment", "tests", "solution", "steps"):
        files += [p.relative_to(task_dir).as_posix() for p in (task_dir / directory).rglob("*") if p.is_file()]
    outer = hashlib.sha256()
    for relative in sorted(files):
        outer.update(f"{relative}\0{hashlib.sha256((task_dir / relative).read_bytes()).hexdigest()}\n".encode())
    return "sha256:" + outer.hexdigest()


def test_task_toml_runs_networkless_with_a_separate_verifier(dataset: Path):
    for task_dir in _tasks(dataset):
        config = tomllib.loads((task_dir / "task.toml").read_text())
        assert config["schema_version"] == "1.4" and config["artifacts"] == ["/app/textcraft/actions.jsonl"]
        assert config["task"]["name"] == "textcraft-synth/" + task_dir.name.removeprefix("textcraft-synth-")
        assert config["verifier"] == {"environment_mode": "separate", "timeout_sec": 180.0}
        assert config["environment"]["network_mode"] == "no-network" and config["environment"]["gpus"] == 0
        metadata = config["metadata"]
        assert metadata["environment"] == "textcraft-synth@1" and metadata["split"] in ("train", "val")
        assert config["agent"]["timeout_sec"] == export.agent_timeout_s(metadata["num_craft_steps"])
        assert set(os.listdir(task_dir)) == {"task.toml", "instruction.md", "environment", "tests", "solution"}
        assert stat.S_IMODE((task_dir / "tests" / "test.sh").stat().st_mode) == 0o755
        assert stat.S_IMODE((task_dir / "solution" / "solve.sh").stat().st_mode) == 0o755
        instruction = (task_dir / "instruction.md").read_text()
        assert metadata["goal"] in instruction and "<TIPS>" not in instruction


def _instructions(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line and not line.startswith("#")]


def test_every_dockerfile_builds_one_image_and_copies_only_files_after_it(dataset: Path):
    prefixes, environment_suffixes, tests_suffixes = set(), set(), set()
    for task_dir in _tasks(dataset):
        for kind, suffixes in (("environment", environment_suffixes), ("tests", tests_suffixes)):
            lines = _instructions(task_dir / kind / "Dockerfile")
            run_end = max(i for i, line in enumerate(lines) if not line.startswith(("COPY", "WORKDIR", "ENV")))
            prefixes.add(tuple(lines[: run_end + 1]))
            suffixes.add(tuple(lines[run_end + 1 :]))
            assert all(line.split()[0] in ("COPY", "WORKDIR", "ENV") for line in lines[run_end + 1 :])
        first = (task_dir / "environment" / "Dockerfile").read_text().splitlines()[0]
        task_sha = hashlib.sha256((task_dir / "environment" / "task.json").read_bytes()).hexdigest()
        assert first == f"# {task_dir.name}: task.json sha256:{task_sha}"
    assert len(prefixes) == 1  # FROM and the one RUN, the same for the agent and the verifier
    assert len(environment_suffixes) == 1 and len(tests_suffixes) == 1
    (prefix,) = prefixes
    assert prefix[0] == f"FROM {export.BASE_IMAGE}"


def test_shared_files_are_the_game_and_the_environments_recipes(dataset: Path):
    task_dir = _tasks(dataset)[0]
    assert (task_dir / "environment" / "textcraft_game.py").read_bytes() == (export.HERE / "game.py").read_bytes()
    assert (task_dir / "tests" / "task.json").read_bytes() == (task_dir / "environment" / "task.json").read_bytes()
    assert (dataset / "LICENSE").read_bytes() == (REPO / "LICENSE").read_bytes()
    book = game.RecipeBook.load(task_dir / "environment" / "recipes.json")
    export.check_recipes(book, TextCraftSynthEnvironment.recipe_db())
    task = game.load_task(task_dir / "environment" / "task.json")
    assert task["recipes_sha256"] == book.sha256


def test_the_card_declares_an_rl_environment_for_harbor(dataset: Path):
    card = (dataset / "README.md").read_text()
    front, _, body = card.removeprefix("---\n").partition("\n---\n")
    metadata = yaml.safe_load(front)
    assert metadata["pretty_name"] == "TextCraft-Synth" and metadata["license"] == "mit"
    assert {"rl-environment", "harbor"} <= set(metadata["tags"]) and metadata["viewer"] is False
    assert metadata["size_categories"] == ["n<1K"]  # LIMIT tasks per split
    assert "--ak record_terminal_session=false" in body and "@misc{gandhi2025platoon," in body
    assert "[ADaPT][adapt] (Prasad et al., Findings of NAACL 2024) introduced TextCraft" in body
    assert "@inproceedings{prasad2024adapt," in body and "[adapt]: https://arxiv.org/abs/2311.05772" in body
    assert "run untrusted agents, and train, only on an environment whose verifier runs as root" in body
    assert "the Singularity caveat above does not apply to it" in body
    assert "harbor run --repo https://huggingface.co/datasets/apurvaga/textcraft-synth" in body
    assert "--dataset textcraft-synth-val@1.0" in body and "hf://apurvaga/textcraft-synth@<commit>" in body
    assert "commit `0123abc`" in body
    # The license section of the card published on the Hub, word for word.
    assert (
        "\n## License\n\nMIT, the license of platoon, whose code generated the recipes and the tasks and whose game "
        "file every task\ncarries (`LICENSE`). The images are built from the `python:3.12-slim` base image at run "
        "time; it is not part of this\nrepository.\n"
    ) in body


def test_the_original_variant_exports_too(tmp_path: Path):
    tasks = export.export_dataset(tmp_path / "textcraft", variant="original", limit=2, repo_id="someone/textcraft")
    assert [task.name for task in tasks] == [
        "textcraft-train-00000",
        "textcraft-train-00001",
        "textcraft-val-00000",
        "textcraft-val-00001",
    ]
    _assert_plays(tmp_path / "textcraft" / "tasks" / tasks[0].name, tmp_path / "sandbox", solve=True, reward=1.0)
    # MIT too, and the card says whose the recipes are: Minecraft's, not generated by platoon.
    card = (tmp_path / "textcraft" / "README.md").read_text()
    front, _, body = card.removeprefix("---\n").partition("\n---\n")
    assert yaml.safe_load(front)["license"] == "mit"
    assert (tmp_path / "textcraft" / "LICENSE").read_bytes() == (REPO / "LICENSE").read_bytes()
    assert "The recipes are the crafting recipes of Minecraft: Java Edition 1.16.5" in body
    assert "whose code generated the recipes" not in body


# ---------------------------------------------------------------------------------------------------- one sandbox


def _sandbox(task_dir: Path, root: Path) -> Path:
    """A sandbox root as platoon's process provider lays one out after the deferred Dockerfile steps."""

    game_dir = root / "app" / "textcraft"
    game_dir.mkdir(parents=True)
    for name in ("textcraft_game.py", "recipes.json", "task.json"):
        shutil.copy2(task_dir / "environment" / name, game_dir / name)
    return root


def _env(root: Path) -> dict[str, str]:
    path = ":".join(
        p for p in os.environ.get("PATH", "/usr/bin:/bin").split(":") if not (Path(p) / "textcraft").exists()
    )
    return {"PATH": path, "PLATOON_SANDBOX_ROOT": str(root), "TEXTCRAFT_DIR": str(root / "app" / "textcraft")}


def _grade(task_dir: Path, root: Path) -> dict[str, Any]:
    tests = root / "tests"
    if tests.exists():
        shutil.rmtree(tests)
    shutil.copytree(task_dir / "tests", tests)
    result = subprocess.run(["bash", str(tests / "test.sh")], env=_env(root), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    verifier = root / "logs" / "verifier"
    reward = json.loads((verifier / "reward.json").read_text())
    verdict = json.loads((verifier / "textcraft-verdict.json").read_text())
    assert reward == {"reward": verdict["reward"]}
    return verdict


def _assert_plays(task_dir: Path, root: Path, *, solve: bool, reward: float) -> dict[str, Any]:
    _sandbox(task_dir, root)
    if solve:
        result = subprocess.run(
            ["bash", str(task_dir / "solution" / "solve.sh")], env=_env(root), capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
    verdict = _grade(task_dir, root)
    assert verdict["reward"] == reward, verdict
    return verdict


def test_the_oracle_scores_one_and_the_noop_zero(dataset: Path, tmp_path: Path):
    for number, task_dir in enumerate(_tasks(dataset)):
        verdict = _assert_plays(task_dir, tmp_path / f"oracle{number}", solve=True, reward=1.0)
        assert verdict["finished"] and verdict["crafts"] == verdict["records"] - 1
        assert _assert_plays(task_dir, tmp_path / f"noop{number}", solve=False, reward=0.0)["reason"].startswith(
            "no action log"
        )


def _play(root: Path, *argv: str) -> game.Result:
    return game.run(list(argv), directory=str(root / "app" / "textcraft"))


def _oracle_lines(task_dir: Path, root: Path) -> list[bytes]:
    _sandbox(task_dir, root)
    subprocess.run(["bash", str(task_dir / "solution" / "solve.sh")], env=_env(root), check=True, capture_output=True)
    return (root / LOG).read_bytes().splitlines(keepends=True)


def test_logs_the_game_could_not_have_written_score_zero(dataset: Path, tmp_path: Path):
    task_dir = next(path for path in _tasks(dataset) if "medium" in path.name)
    lines = _oracle_lines(task_dir, tmp_path / "oracle")
    records = [json.loads(line) for line in lines]
    crafts = [record for record in records if record["tool"] == "craft"]
    assert len(crafts) >= 3 and records[-1]["tool"] == "finish"

    def graded(name: str, data: bytes) -> dict[str, Any]:
        root = _sandbox(task_dir, tmp_path / name)
        (root / LOG).write_bytes(data)
        return _grade(task_dir, root)

    def renumbered(kept: list[dict[str, Any]]) -> bytes:
        return b"".join(game.encode_record(i, r["tool"], r["args"], r.get("ok")) for i, r in enumerate(kept, 1))

    skipped = graded("skips", renumbered([records[0], *records[-2:]]))  # the last craft without its inputs
    assert skipped["reward"] == 0.0 and "the rules give: Error" in skipped["reason"]
    claims = dict(records[0], ok=False)  # a craft that succeeds, logged as rejected
    assert graded("claims", renumbered([claims, *records[1:]]))["reward"] == 0.0
    late = graded("late", renumbered([*records, records[0]]))
    assert late["reward"] == 0.0 and "follows finish" in late["reason"]
    gap = b"".join(lines[:1] + lines[2:])  # numbering
    assert graded("gap", gap)["reward"] == 0.0
    duplicate = lines[0].replace(b'"seq":1,', b'"seq":1,"seq":1,')
    assert graded("duplicate", duplicate + b"".join(lines[1:]))["reward"] == 0.0
    assert graded("torn", b"".join(lines)[:-1])["reward"] == 0.0  # no final newline
    assert graded("garbage", b"\x00\xff\n")["reward"] == 0.0
    empty = renumbered([*records[:-1], {"tool": "finish", "args": {"message": ""}}])
    assert graded("empty-message", empty)["reason"] == "finished with an empty message"
    unfinished = graded("unfinished", b"".join(lines[:-1]))
    assert unfinished["reward"] == 0.0 and unfinished["reason"] == "the episode did not end with textcraft finish"

    # A craft the rules reject changes nothing, as in the environment: the episode still wins.
    rejected = {"tool": "craft", "args": {"ingredients": {"raw_m0": 1}, "target": [crafts[-1]["args"]["target"][0], 1]}}
    fumble = graded("fumble", renumbered([dict(rejected, ok=False), *records]))
    assert fumble["reward"] == 1.0 and fumble["crafts"] == len(crafts)
    # The same rejected craft claimed as a success is an illegal move.
    assert graded("illegal", renumbered([dict(rejected, ok=True), *records]))["reward"] == 0.0


def test_replay_checks_but_never_answers_the_tools_that_change_nothing(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # A get_info record may name millions of items (the log may hold 32 MiB): building its answer again on every
    # replay would cost the verifier gigabytes and minutes, and a verifier that dies writes no reward at all.
    task_dir = next(path for path in _tasks(dataset) if "medium" in path.name)
    records = [json.loads(line) for line in _oracle_lines(task_dir, tmp_path / "oracle")]
    item = records[-2]["args"]["target"][0]
    noisy = [{"tool": "get_info", "args": {"items": [item] * 100_000}}, {"tool": "view_inventory", "args": {}}]
    noisy += records
    data = b"".join(game.encode_record(i, r["tool"], r["args"], r.get("ok")) for i, r in enumerate(noisy, 1))
    called: list[str] = []
    original = game.Game.call
    monkeypatch.setattr(game.Game, "call", lambda self, tool, args: called.append(tool) or original(self, tool, args))
    book = game.RecipeBook.load(task_dir / "tests" / "recipes.json")
    task = game.load_task(task_dir / "tests" / "task.json")
    play = game.Game(book, task)
    assert game.replay(play, data) == len(noisy) and play.finished and play.score == 1.0
    assert set(called) == {"craft", "finish"}
    for bad in ({"items": item}, {"items": [1]}, {"items": [], "extra": 1}):  # their arguments are still checked
        with pytest.raises(game.LogError, match="record 1"):
            game.replay(game.Game(book, task), game.encode_record(1, "get_info", bad))
    with pytest.raises(game.LogError, match="record 1"):
        game.replay(game.Game(book, task), game.encode_record(1, "view_inventory", {"items": []}))
    root = _sandbox(task_dir, tmp_path / "noisy")
    (root / LOG).write_bytes(data)
    verdict = _grade(task_dir, root)
    assert verdict["reward"] == 1.0 and verdict["records"] == len(noisy)


def test_the_verifier_trusts_no_file_the_agent_can_edit(dataset: Path, tmp_path: Path):
    task_dir = next(path for path in _tasks(dataset) if "medium" in path.name)
    target, count = next(iter(json.loads((task_dir / "environment" / "task.json").read_text())["target_items"].items()))

    # A forged state: the agent's task.json grants the target's ingredients, so its game accepts crafts that the
    # verifier's copy of the task does not.
    root = _sandbox(task_dir, tmp_path / "forged")
    book = game.RecipeBook.load(root / "app/textcraft/recipes.json")
    recipe = book.get_recipes_for_item(target)[0]
    task_path = root / "app/textcraft/task.json"
    forged = json.loads(task_path.read_text())
    forged["initial_inventory"] = {item: 1000 for item in recipe.ingredients}
    task_path.write_text(json.dumps(forged))
    crafted = recipe.result_count * count
    played = _play(root, "craft", target, str(crafted), *(f"{i}={n * count}" for i, n in recipe.ingredients.items()))
    assert played.code == 0 and played.out.startswith("Successfully crafted")
    assert _play(root, "finish", "done").code == 0
    verdict = _grade(task_dir, root)
    assert verdict["reward"] == 0.0 and "the rules give: Error" in verdict["reason"]

    # Edited recipes: the agent's game refuses to play them.
    root = _sandbox(task_dir, tmp_path / "recipes")
    (root / "app/textcraft/recipes.json").write_text((root / "app/textcraft/recipes.json").read_text() + " ")
    assert _play(root, "inventory").code == 3

    # A planted reward and no log.
    root = _sandbox(task_dir, tmp_path / "planted")
    (root / "logs/verifier").mkdir(parents=True)
    (root / "logs/verifier/reward.json").write_text('{"reward": 1.0}')
    (root / "logs/verifier/reward.txt").write_text("1")
    assert _grade(task_dir, root)["reward"] == 0.0 and not (root / "logs/verifier/reward.txt").exists()

    # The log as a symbolic link to a winning log, or below a linked directory.
    winning = b"".join(_oracle_lines(task_dir, tmp_path / "winner"))
    (tmp_path / "winning.jsonl").write_bytes(winning)
    root = _sandbox(task_dir, tmp_path / "link")
    (root / LOG).symlink_to(tmp_path / "winning.jsonl")
    verdict = _grade(task_dir, root)
    assert verdict["reward"] == 0.0 and "symbolic link" in verdict["reason"]
    root = tmp_path / "dirlink"
    (root / "app").mkdir(parents=True)
    (root / "app" / "textcraft").symlink_to(tmp_path / "winner" / "app" / "textcraft")
    assert _grade(task_dir, root)["reward"] == 0.0

    # A huge log (sparse), a directory, and a FIFO, which must not block the verifier.
    root = _sandbox(task_dir, tmp_path / "huge")
    with (root / LOG).open("wb") as handle:
        handle.write(winning)
        handle.truncate(game.LOG_CAP_BYTES + 1)
    assert _grade(task_dir, root)["reward"] == 0.0
    root = _sandbox(task_dir, tmp_path / "directory")
    (root / LOG).mkdir()
    assert _grade(task_dir, root)["reward"] == 0.0
    root = _sandbox(task_dir, tmp_path / "fifo")
    os.mkfifo(root / LOG)
    assert _grade(task_dir, root)["reward"] == 0.0


def test_the_command_logs_only_what_it_runs(dataset: Path, tmp_path: Path):
    task_dir = _tasks(dataset)[0]
    root = _sandbox(task_dir, tmp_path / "cli")
    assert _play(root, "craft", "x").code == 2  # usage errors log nothing
    assert _play(root, "craft", "x", "two", "a=1").code == 2
    assert _play(root, "craft", "x", "1", "a=1", "a=2").code == 2
    assert _play(root, "craft", "--json", '{"ingredients": {"a": 1e999}, "target": ["x", 1]}').code == 2
    assert _play(root, "info", "--json", '"a1_i5"').code == 2
    assert _play(root, "finish").code == 2
    assert not (root / LOG).exists()
    inventory = _play(root, "inventory")
    assert (
        inventory.code == 0
        and json.loads(inventory.out) == json.loads((root / "app/textcraft/task.json").read_text())["initial_inventory"]
    )
    assert _play(root, "craft", "no_such_item", "1").out == "Error: No recipe found for no_such_item"
    assert _play(root, "finish", "giving", "up").out == "giving up"
    assert _play(root, "inventory").code == 1  # after finish: refused, not logged
    records = [json.loads(line) for line in (root / LOG).read_text().splitlines()]
    assert [record["tool"] for record in records] == ["view_inventory", "craft", "finish"]
    assert records[1]["ok"] is False and records[2]["args"] == {"message": "giving up"}


# ---------------------------------------------------------------------------------------------------- parity


def _env_call(environment: TextCraftSynthEnvironment, tool: str, args: dict[str, Any]) -> tuple[str, Any]:
    if tool == "craft":
        output = asyncio.run(environment.craft(CraftParams(**args)))
    elif tool == "get_info":
        output = asyncio.run(environment.get_info(GetInfoParams(**args)))
    elif tool == "view_inventory":
        output = asyncio.run(environment.view_inventory())
    else:
        output = asyncio.run(environment.finish(FinishParams(**args)))
    return output.blocks[0].text, output.reward


def random_actions(spec: dict[str, Any], book: game.RecipeBook, rng: random.Random, length: int) -> list[tuple]:
    """A random episode: crafts of the task's own items (right, scaled wrong, short of ingredients or extra ones) and
    of other items, info and inventory calls, and sometimes a finish."""

    gold = spec["misc"]["gold_trajectory"]
    near = [step["target"][0] for step in gold] + list(spec["misc"]["target_items"])
    items = sorted(book.recipes)
    names = near + items[:50] + sorted(book.base_items)[:10] + ["no_such_item"]
    actions: list[tuple] = []
    for _ in range(length):
        roll = rng.random()
        if roll < 0.65:
            item = rng.choice(near) if rng.random() < 0.8 else rng.choice(items)
            recipe = book.get_recipes_for_item(item)[0]
            times = rng.randint(1, 3)
            ingredients = {name: n * times for name, n in recipe.ingredients.items()}
            count = recipe.result_count * times
            fault = rng.random()
            if fault < 0.1:
                count += 1
            elif fault < 0.2 and ingredients:
                ingredients.pop(next(iter(ingredients)))
            elif fault < 0.3:
                ingredients[rng.choice(names)] = 1
            elif fault < 0.35:
                count = rng.choice([0, -2, 2.5, "3"])
            actions.append(("craft", {"ingredients": ingredients, "target": [item, count]}))
        elif roll < 0.85:
            actions.append(("get_info", {"items": rng.sample(names, rng.randint(0, 3))}))
        elif roll < 0.97:
            actions.append(("view_inventory", {}))
        else:
            actions.append(("finish", {"message": rng.choice(["done", ""])}))
            break
    return actions


def _assert_parity(spec: dict[str, Any], actions: list[tuple], root: Path, task_dir: Path) -> float:
    environment = TextCraftSynthEnvironment(spec)
    book = game.RecipeBook.load(task_dir / "environment" / "recipes.json")
    play = game.Game(book, game.load_task(task_dir / "environment" / "task.json"))
    _sandbox(task_dir, root)
    reward = 0.0
    for tool, args in actions:
        expected, env_reward = _env_call(environment, tool, args)
        assert play.call(tool, json.loads(json.dumps(args))) == expected
        argv = {
            "craft": ["craft", "--json", json.dumps(args)],
            "get_info": ["info", "--json", json.dumps(args.get("items"))],
            "view_inventory": ["inventory"],
            "finish": ["finish", args.get("message", "")],
        }[tool]
        assert _play(root, *argv).out == expected
        assert play.inventory == environment.inventory
        assert list(play.inventory) == list(environment.inventory)  # the order view_inventory prints
        if tool == "finish":
            reward = env_reward
            assert play.score == env_reward
    verdict = _grade(task_dir, root)
    assert verdict["reward"] == (reward if actions and actions[-1][0] == "finish" else 0.0)
    return verdict["reward"]


def test_the_game_plays_as_the_environment(dataset: Path, tmp_path: Path):
    rng = random.Random(7)
    index = json.loads((dataset / "task-index.json").read_text())
    for number, task_dir in enumerate(_tasks(dataset)):
        row = index[task_dir.name]
        spec = TextCraftSynthEnvironment.list_tasks(row["split"])[int(row["env_task_id"].split(":")[1])]
        book = game.RecipeBook.load(task_dir / "environment" / "recipes.json")
        gold = [
            ("craft", {"ingredients": step["ingredients"], "target": [step["target"][0], step["result_count"]]})
            for step in spec["misc"]["gold_trajectory"]
        ]
        assert _assert_parity(spec, [*gold, ("finish", {"message": "done"})], tmp_path / f"gold{number}", task_dir) == 1
        for attempt in range(3):
            actions = random_actions(spec, book, rng, 25)
            _assert_parity(spec, actions, tmp_path / f"random{number}-{attempt}", task_dir)
