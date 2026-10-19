"""Export TextCraft as a Harbor dataset: one task directory per task of platoon's TextCraft environment.

    python -m platoon_textcraft.harbor export --variant synth --out DIR --version 1.0

The tasks are the environment's own (``TextCraftSynthEnvironment.list_tasks``: the ``train`` and ``val`` splits of
``textcraft-synth@1``; ``--variant original`` exports ``TextCraftEnvironment``, ``textcraft@1``), with its recipes.
Each task holds:

- ``instruction.md``: the game description platoon's TextCraft agents get, the goal, and the ``textcraft``
  commands;
- ``task.toml``: timeouts from the task's reference step count, metadata (split, difficulty, crafting depth, the
  source task's ids), no network, and a separate verifier that receives the action log as an artifact;
- ``environment/``: a Dockerfile whose build is the same for every task (a Python base image, tmux and the
  ``textcraft`` command), then the game and the recipes, which are the same in every task, and the task's
  ``task.json``, the only per-task step;
- ``tests/``: ``test.sh``, which replays the action log with trusted copies of the game, the recipes and the task,
  and the separate verifier's Dockerfile (the same image, with those copies in ``/tests``);
- ``solution/solve.sh``: the task's gold trajectory as ``textcraft`` commands, then ``textcraft finish``.  The
  export plays it with the game first and fails if it does not reach the goal.

The dataset's root holds ``registry.json`` (one Harbor dataset per split, ``<world>-<split>@<version>``),
``task-index.json`` (each task's fields and Harbor content hash), ``splits.json`` (the environment's split names,
``train-medium`` included, to task names), ``LICENSE`` (platoon's, since every task carries its game) and the dataset
card ``README.md``.  The same inputs give the same bytes: nothing depends on the time, the machine or the order of a
directory listing.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shlex
import shutil
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platoon_textcraft.harbor import game

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
# python:3.12-slim (Debian trixie), pinned by its multi-architecture index digest.
BASE_IMAGE = (
    "docker.io/library/python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1"
)
# The one RUN step of every Dockerfile in the dataset, so they all build one image: the `textcraft` command, and tmux
# for terminal agents (Harbor's terminus-2 installs it at run time otherwise, which a task without network cannot).
# /usr/bin/python3 stays absent: Harbor's Singularity environment installs Debian's python3 there for its own server,
# which a link to the image's Python would break as soon as an apt package pulls Debian's python3 in (asciinema does).
IMAGE_RUN = (
    "set -eux; \\\n"
    "    printf '#!/bin/sh\\nexec python3 -I /app/textcraft/textcraft_game.py \"$@\"\\n' \\\n"
    "        > /usr/local/bin/textcraft; \\\n"
    "    chmod 0755 /usr/local/bin/textcraft; \\\n"
    "    apt-get -o APT::Sandbox::User=root update; \\\n"
    "    apt-get -o APT::Sandbox::User=root install -y --no-install-recommends tmux; \\\n"
    "    rm -rf /var/lib/apt/lists/*"
)
GAME_DIR = "/app/textcraft"
LOG_PATH = f"{GAME_DIR}/{game.LOG_FILE}"
GAME_FILE = "textcraft_game.py"
VERIFIER_TIMEOUT_S = 180
AGENT_TIMEOUT_BASE_S = 600
AGENT_TIMEOUT_PER_CRAFT_S = 10
CPUS = 1
MEMORY_MB = 2048
BUILD_TIMEOUT_S = 900
DIFFICULTIES = ("easy", "medium", "hard", "extreme")
# Harbor's content hash covers these root files and every file below these directories (harbor-packager-v1).
HASHED_FILES = ("task.toml", "instruction.md", "README.md", "trajectory.json")
HASHED_DIRS = ("environment", "tests", "solution", "steps")
# Shared files at least this large are written once and hard-linked into every task that holds them.
LINK_MIN_BYTES = 4096


@dataclass(frozen=True)
class Variant:
    """One TextCraft world as an exported dataset."""

    world: str  # dataset and task-name prefix
    pretty: str
    environment: str  # module:Class of the ORS environment
    named: str  # the named environment it reproduces
    option: str  # its --variant value
    craft_notes: str  # instruction lines after the craft command's arguments
    info_notes: str  # instruction lines after the info command's output


VARIANTS = {
    "synth": Variant(
        world="textcraft-synth",
        pretty="TextCraft-Synth",
        environment="platoon_textcraft.env:TextCraftSynthEnvironment",
        named="textcraft-synth@1",
        option="synth",
        craft_notes="   - Example: `textcraft craft m2_i2 2 m0_i1=2 m1_i1=1`\n",
        info_notes="   - Example: `textcraft info m2_i2 raw_m0`\n",
    ),
    "original": Variant(
        world="textcraft",
        pretty="TextCraft",
        environment="platoon_textcraft.env:TextCraftEnvironment",
        named="textcraft@1",
        option="original",
        craft_notes=(
            '   - For tag-based ingredients (e.g., "tag:planks"), provide a CONCRETE item from that category\n'
            "   - Example: `textcraft craft wooden_pickaxe 1 stick=2 oak_planks=3`\n"
            "   - Example: `textcraft craft oak_planks 16 oak_log=4`\n"
        ),
        info_notes=(
            '   - Tag-based ingredients show {"tag:X": {"count": N, "use_one_of": [concrete items]}}\n'
            "   - Example: `textcraft info yellow_dye yellow_terracotta`\n"
        ),
    ),
}


@dataclass(frozen=True)
class ExportedTask:
    name: str
    split: str
    index: int
    path: str  # relative to the dataset root
    fields: dict[str, Any]  # the task index row (without the digest)
    digest: str  # Harbor's content hash, "sha256:<hex>"


def environment_class(variant: Variant) -> Any:
    module, _, attribute = variant.environment.partition(":")
    return getattr(importlib.import_module(module), attribute)


# ---------------------------------------------------------------------------------------------------- recipes


def recipes_document(recipe_db: Any, world: str) -> dict[str, Any]:
    """The ``textcraft-recipes/1`` document of a platoon recipe database: every recipe in the database's order,
    the base items, the nonzero crafting depths and the items of every tag a recipe names."""

    items = set(recipe_db.recipes) | set(recipe_db.base_items) | set(getattr(recipe_db, "all_items", ()))
    tags = set(getattr(recipe_db, "all_tags", ()))
    recipes: dict[str, list[dict[str, Any]]] = {}
    for item in sorted(recipe_db.recipes):
        entries = []
        for recipe in recipe_db.get_recipes_for_item(item):
            entries.append(
                {
                    "result_count": recipe.result_count,
                    "ingredients": [[name, count] for name, count in recipe.ingredients.items()],
                }
            )
            tags |= {name[4:] for name in recipe.ingredients if name.startswith("tag:")}
        recipes[item] = entries
    depths = {item: depth for item in sorted(items) if (depth := recipe_db.get_crafting_depth(item))}
    return {
        "format": game.RECIPES_FORMAT,
        "name": world,
        "base_items": sorted(recipe_db.base_items),
        "depths": depths,
        "tags": {tag: list(recipe_db.get_items_for_tag(tag)) for tag in sorted(tags)},
        "recipes": recipes,
    }


def encode_recipes(document: Mapping[str, Any]) -> bytes:
    """Compact JSON with one line per item's recipes, so the file reads and diffs line by line."""

    def dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))

    lines = ["{"]
    for key in ("format", "name", "base_items", "depths", "tags"):
        lines.append(f"{dump(key)}:{dump(document[key])},")
    lines.append('"recipes":{')
    entries = list(document["recipes"].items())
    for number, (item, recipes) in enumerate(entries):
        lines.append(f"{dump(item)}:{dump(recipes)}" + ("," if number < len(entries) - 1 else ""))
    lines.append("}}")
    return ("\n".join(lines) + "\n").encode("ascii")


def check_recipes(book: game.RecipeBook, recipe_db: Any) -> None:
    """The exported recipes answer every lookup the environment makes as its database does."""

    names = set(recipe_db.recipes) | set(recipe_db.base_items) | set(getattr(recipe_db, "all_items", ()))
    for item in sorted(names) + ["no_such_item"]:
        if (
            book.can_craft(item) != recipe_db.can_craft(item)
            or book.is_base_item(item) != recipe_db.is_base_item(item)
            or book.get_crafting_depth(item) != recipe_db.get_crafting_depth(item)
        ):
            raise ValueError(f"the exported recipes describe {item!r} unlike the environment's")
        mine = [(r.result_count, list(r.ingredients.items())) for r in book.get_recipes_for_item(item)]
        theirs = [(r.result_count, list(r.ingredients.items())) for r in recipe_db.get_recipes_for_item(item)]
        if mine != theirs:
            raise ValueError(f"the exported recipes of {item!r} differ from the environment's")
    for tag in book.tags:
        if book.get_items_for_tag(tag) != list(recipe_db.get_items_for_tag(tag)):
            raise ValueError(f"the exported items of tag {tag!r} differ from the environment's")


# ---------------------------------------------------------------------------------------------------- one task


def task_name(variant: Variant, split: str, index: int, difficulty: str | None) -> str:
    """``<world>-<split>-<difficulty>-<index>``: the source index in its split, zero-padded; the difficulty when
    there is one."""

    return "-".join([variant.world, split, *([difficulty] if difficulty else []), f"{index:05d}"])


def agent_timeout_s(craft_steps: int) -> int:
    """Ten minutes, plus ten seconds per craft of the reference solution, rounded up to a minute."""

    seconds = AGENT_TIMEOUT_BASE_S + AGENT_TIMEOUT_PER_CRAFT_S * craft_steps
    return -(-seconds // 60) * 60


def reference_solution(
    book: game.RecipeBook, document: Mapping[str, Any], gold: Sequence[Mapping[str, Any]], name: str
) -> tuple[list[tuple[str, int, dict[str, int]]], str]:
    """The gold trajectory as crafts the game accepts, played from the task's starting inventory, and the finish
    message; ValueError when a craft fails or the crafts miss a target.  A gold step's ``tag:`` ingredient becomes
    the first item of the tag that the inventory holds enough of (original TextCraft)."""

    play = game.Game(book, document)
    steps: list[tuple[str, int, dict[str, int]]] = []
    for number, step in enumerate(gold, 1):
        item = step["target"][0]
        count = step["result_count"]
        ingredients: dict[str, int] = {}
        for ingredient, amount in step["ingredients"].items():
            if ingredient.startswith("tag:"):
                choices = [
                    candidate
                    for candidate in book.get_items_for_tag(ingredient[4:])
                    if play.inventory.get(candidate, 0) >= amount and candidate not in ingredients
                ]
                if not choices:
                    raise ValueError(f"{name}: gold step {number} needs {amount} of {ingredient}, which no item covers")
                ingredient = choices[0]
            ingredients[ingredient] = amount
        text = play.call("craft", {"ingredients": ingredients, "target": [item, count]})
        if not text.startswith(game.CRAFTED):
            raise ValueError(f"{name}: gold step {number} fails: {text}")
        steps.append((item, count, ingredients))
    message = "Crafted " + ", ".join(f"{count}x {item}" for item, count in document["target_items"].items())
    play.call("finish", {"message": message})
    if play.score != 1.0:
        raise ValueError(f"{name}: the gold trajectory misses {play.reward_misc.get('missing_items')}")
    return steps, message


def solve_script(name: str, steps: Sequence[tuple[str, int, Mapping[str, int]]], message: str) -> str:
    lines = [
        "#!/bin/bash",
        f"# The reference solution of {name}: the source task's gold trajectory ({len(steps)} crafts), then finish.",
        "set -euo pipefail",
        "if ! command -v textcraft >/dev/null 2>&1; then",
        "  # platoon's process provider runs no image, so there is no textcraft command: run the game file itself.",
        f'  textcraft() {{ python3 -I "${{PLATOON_SANDBOX_ROOT:-}}{GAME_DIR}/{GAME_FILE}" "$@"; }}',
        "fi",
    ]
    for item, count, ingredients in steps:
        words = [
            "textcraft",
            "craft",
            item,
            str(count),
            *(f"{ingredient}={n}" for ingredient, n in ingredients.items()),
        ]
        lines.append(" ".join(shlex.quote(word) for word in words))
    lines.append(f"textcraft finish {shlex.quote(message)}")
    return "\n".join(lines) + "\n"


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)  # a JSON string is a TOML basic string


def task_toml(variant: Variant, name: str, fields: Mapping[str, Any]) -> str:
    difficulty = fields.get("difficulty")
    goal = fields["goal"]
    level = f"{difficulty}, " if difficulty else ""
    description = f"{variant.pretty}: {goal[0].lower()}{goal[1:]} ({level}crafting depth {fields['max_depth']})"
    metadata = [
        ("environment", _toml_string(variant.named)),
        ("split", _toml_string(fields["split"])),
        *([("difficulty", _toml_string(difficulty))] if difficulty else []),
        ("max_depth", str(fields["max_depth"])),
        ("num_craft_steps", str(fields["num_craft_steps"])),
        ("goal", _toml_string(goal)),
        ("source_task_id", _toml_string(fields["source_task_id"])),
        ("env_task_id", _toml_string(fields["env_task_id"])),
        *([("env_split_task_id", _toml_string(fields["env_split_task_id"]))] if difficulty else []),
    ]
    keywords = ", ".join(_toml_string(k) for k in ("textcraft", variant.world, "crafting", "planning"))
    lines = [
        'schema_version = "1.4"',
        "# The game's action log: a separate verifier receives it at the same path.",
        f"artifacts = [{_toml_string(LOG_PATH)}]",
        "",
        "[task]",
        f"name = {_toml_string(variant.world + '/' + name[len(variant.world) + 1 :])}",
        f"description = {_toml_string(description)}",
        f"keywords = [{keywords}]",
        "",
        "[metadata]",
        *(f"{key} = {value}" for key, value in metadata),
        "",
        "[verifier]",
        'environment_mode = "separate"',
        f"timeout_sec = {float(VERIFIER_TIMEOUT_S)}",
        "",
        "[agent]",
        f"timeout_sec = {float(fields['agent_timeout_sec'])}",
        "",
        "[environment]",
        'network_mode = "no-network"',
        f"cpus = {CPUS}",
        f"memory_mb = {MEMORY_MB}",
        "gpus = 0",
        f"build_timeout_sec = {float(BUILD_TIMEOUT_S)}",
    ]
    return "\n".join(lines) + "\n"


def instruction(variant: Variant, goal: str) -> str:
    text = (TEMPLATES / "instruction.md").read_text()
    text = text.replace("@@GOAL@@", goal)
    text = text.replace("@@CRAFT_NOTES@@\n", variant.craft_notes + "\n")
    return text.replace("@@INFO_NOTES@@\n", variant.info_notes + "\n")


def dockerfile(template: str, header: str) -> str:
    text = (TEMPLATES / template).read_text()
    return header + text.replace("@@BASE_IMAGE@@", BASE_IMAGE).replace("@@IMAGE_RUN@@", IMAGE_RUN)


def task_files(
    variant: Variant,
    spec: Mapping[str, Any],
    environment: Any,
    book: game.RecipeBook,
    *,
    split: str,
    index: int,
    split_index: int | None,
    shared: Mapping[str, bytes],
) -> tuple[str, dict[str, Any], dict[str, tuple[bytes, bool]]]:
    """A task's name, index fields and files (relative path -> (bytes, executable))."""

    misc = dict(spec.get("misc") or {})
    difficulty = misc.get("difficulty")
    gold = list(misc.get("gold_trajectory") or [])
    name = task_name(variant, split, index, difficulty)
    goal = environment.goal
    targets = dict(environment.target_items)
    document = {
        "format": game.TASK_FORMAT,
        "name": name,
        "world": variant.world,
        "goal": goal,
        "target_items": targets,
        "initial_inventory": dict(environment.initial_inventory),
        "recipes_sha256": hashlib.sha256(shared["recipes.json"]).hexdigest(),
    }
    task_json = (json.dumps(document, indent=1, ensure_ascii=True) + "\n").encode("ascii")
    steps, message = reference_solution(book, document, gold, name)
    max_depth = misc.get("max_depth")
    if max_depth is None:
        max_depth = max(book.get_crafting_depth(item) for item in targets)
    fields: dict[str, Any] = {
        "task_path": f"tasks/{name}",
        "split": split,
        **({"difficulty": difficulty} if difficulty else {}),
        "max_depth": max_depth,
        "num_craft_steps": len(steps),
        "agent_timeout_sec": agent_timeout_s(len(steps)),
        "goal": goal,
        "target_items": targets,
        "source_task_id": spec.get("id"),
        "env_task_id": f"{split}:{index}",
        **({"env_split_task_id": f"{split}-{difficulty}:{split_index}"} if difficulty else {}),
    }
    # Builders that cache an image by its Dockerfile's text (harbor-singularity-hpc) need the text to differ between
    # tasks whose task.json differs; Docker, BuildKit and platoon's planner ignore comments.
    header = f"# {name}: task.json sha256:{hashlib.sha256(task_json).hexdigest()}\n"
    files: dict[str, tuple[bytes, bool]] = {
        "task.toml": (task_toml(variant, name, fields).encode("ascii"), False),
        "instruction.md": (instruction(variant, goal).encode("utf-8"), False),
        "environment/Dockerfile": (dockerfile("environment.Dockerfile", header).encode("ascii"), False),
        f"environment/{GAME_FILE}": (shared[GAME_FILE], False),
        "environment/recipes.json": (shared["recipes.json"], False),
        "environment/task.json": (task_json, False),
        "tests/Dockerfile": (dockerfile("tests.Dockerfile", header).encode("ascii"), False),
        "tests/test.sh": (shared["test.sh"], True),
        f"tests/{GAME_FILE}": (shared[GAME_FILE], False),
        "tests/recipes.json": (shared["recipes.json"], False),
        "tests/task.json": (task_json, False),
        "solution/solve.sh": (solve_script(name, steps, message).encode("ascii"), True),
    }
    return name, fields, files


def content_hash(files: Mapping[str, tuple[bytes, bool]]) -> str:
    """Harbor's task content hash (``Packager.compute_content_hash``, harbor-packager-v1): SHA-256 over
    ``relpath\\0sha256(file)\\n`` lines, in path order, of the hashed root files and every file below the hashed
    directories."""

    outer = hashlib.sha256()
    for relative in sorted(files):
        if relative in HASHED_FILES or relative.split("/", 1)[0] in HASHED_DIRS:
            outer.update(f"{relative}\0{hashlib.sha256(files[relative][0]).hexdigest()}\n".encode())
    return "sha256:" + outer.hexdigest()


# ---------------------------------------------------------------------------------------------------- the dataset


def export_dataset(
    out: str | Path,
    *,
    variant: str = "synth",
    version: str = "1.0",
    splits: Sequence[str] | None = None,
    limit: int | None = None,
    repo_id: str = "apurvaga/textcraft-synth",
    source_revision: str | None = None,
    force: bool = False,
    log: Callable[[str], None] | None = None,
) -> list[ExportedTask]:
    """Write the dataset to ``out``: built in a new directory next to it, then renamed into place."""

    chosen = VARIANTS[variant]
    say = log or (lambda message: None)
    out = Path(out)
    if out.exists() and not force:
        raise FileExistsError(f"{out} exists; pass force=True (--force) to replace it")
    env_class = environment_class(chosen)
    recipe_db = env_class.recipe_db()
    document = recipes_document(recipe_db, chosen.world)
    recipes = encode_recipes(document)
    book = game.RecipeBook(json.loads(recipes), hashlib.sha256(recipes).hexdigest())
    check_recipes(book, recipe_db)
    shared = {
        GAME_FILE: (HERE / "game.py").read_bytes(),
        "recipes.json": recipes,
        "test.sh": (TEMPLATES / "test.sh").read_bytes(),
    }
    splits = list(splits or env_class.split_files)
    unknown = sorted(set(splits) - set(env_class.split_files))
    if unknown:
        raise ValueError(f"unknown splits {unknown}; {chosen.named} has {list(env_class.split_files)}")
    sources = {split: _source_file(env_class, split) for split in splits}
    staging = out.parent / f".{out.name}.{uuid.uuid4().hex[:8]}.tmp"
    staging.mkdir(parents=True)
    links: dict[str, Path] = {}  # identical shared files are hard links of one copy
    exported: list[ExportedTask] = []
    try:
        for split in splits:
            specs = list(env_class.list_tasks(split))
            if limit is not None:
                specs = specs[:limit]
            seen: Counter[str] = Counter()  # each difficulty's tasks so far: the index in <split>-<difficulty>
            for index, spec in enumerate(specs):
                difficulty = (spec.get("misc") or {}).get("difficulty")
                split_index = seen[difficulty] if difficulty else None
                seen[difficulty] += 1
                environment = env_class(spec)
                name, fields, files = task_files(
                    chosen, spec, environment, book, split=split, index=index, split_index=split_index, shared=shared
                )
                for relative, (data, executable) in files.items():
                    _write(staging / fields["task_path"] / relative, data, executable, links)
                exported.append(ExportedTask(name, split, index, fields["task_path"], fields, content_hash(files)))
            say(f"{split}: {len(specs)} tasks")
        names = [task.name for task in exported]
        if len(set(names)) != len(names):
            raise ValueError("two tasks have the same name")
        _write(staging / "registry.json", registry(chosen, exported, version, splits), False, None)
        _write(staging / "task-index.json", task_index(exported), False, None)
        _write(staging / "splits.json", split_manifest(exported, splits), False, None)
        _write(staging / "LICENSE", (TEMPLATES / "LICENSE").read_bytes(), False, None)
        card = dataset_card(chosen, exported, version, splits, repo_id, source_revision, recipes, document, sources)
        _write(staging / "README.md", card.encode("utf-8"), False, None)
        if out.exists():
            shutil.rmtree(out)
        os.rename(staging, out)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return exported


def registry(variant: Variant, tasks: Sequence[ExportedTask], version: str, splits: Sequence[str]) -> bytes:
    """Harbor's registry: one dataset per split, ``<world>-<split>`` at ``version``."""

    datasets = []
    for split in splits:
        members = [task for task in tasks if task.split == split]
        counts = Counter(task.fields.get("difficulty") for task in members)
        detail = ", ".join(f"{counts[level]:,} {level}" for level in DIFFICULTIES if counts[level])
        datasets.append(
            {
                "name": f"{variant.world}-{split}",
                "version": version,
                "description": (
                    f"{variant.pretty} {split} split: {len(members):,} crafting tasks"
                    + (f" ({detail})" if detail else "")
                ),
                "tasks": [{"name": task.name, "path": task.path} for task in members],
            }
        )
    return (json.dumps(datasets, indent=1, ensure_ascii=True) + "\n").encode("ascii")


def task_index(tasks: Sequence[ExportedTask]) -> bytes:
    """``{task name: {task_path, split, difficulty, ..., digest}}``, the fields platoon's fetch can select on."""

    rows = {task.name: {**task.fields, "digest": task.digest} for task in sorted(tasks, key=lambda task: task.name)}
    return (json.dumps(rows, indent=1, ensure_ascii=True, sort_keys=True) + "\n").encode("ascii")


def split_manifest(tasks: Sequence[ExportedTask], splits: Sequence[str]) -> bytes:
    """The environment's split names (``train``, ``train-medium``, ...) to task names, in the environment's order."""

    manifest: dict[str, list[str]] = {}
    for split in splits:
        members = [task for task in tasks if task.split == split]
        manifest[split] = [task.name for task in members]
        for level in DIFFICULTIES:
            chosen = [task.name for task in members if task.fields.get("difficulty") == level]
            if chosen:
                manifest[f"{split}-{level}"] = chosen
    return (json.dumps(manifest, indent=1, ensure_ascii=True) + "\n").encode("ascii")


def _write(path: Path, data: bytes, executable: bool, links: dict[str, Path] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(data).hexdigest() + ("x" if executable else "")
    if links is not None and len(data) >= LINK_MIN_BYTES and key in links:
        os.link(links[key], path)
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
    finally:
        os.close(fd)
    os.chmod(path, 0o755 if executable else 0o644)
    if links is not None and len(data) >= LINK_MIN_BYTES:
        links[key] = path


def dataset_card(
    variant: Variant,
    tasks: Sequence[ExportedTask],
    version: str,
    splits: Sequence[str],
    repo_id: str,
    source_revision: str | None,
    recipes: bytes,
    document: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> str:
    from platoon_textcraft.harbor.card import render_card

    return render_card(
        variant=variant,
        tasks=tasks,
        version=version,
        splits=splits,
        repo_id=repo_id,
        source_revision=source_revision,
        recipes=recipes,
        document=document,
        sources=sources,
    )


def _source_file(env_class: Any, split: str) -> dict[str, Any]:
    """The environment's task file of a split: its path in platoon, task count and SHA-256."""

    from platoon_textcraft import env as textcraft_env

    path = textcraft_env.DATA_DIR / env_class.split_files[split]
    data = path.read_bytes()
    return {
        "file": f"extensions/textcraft/src/platoon_textcraft/data/{path.name}",
        "tasks": sum(1 for line in data.splitlines() if line.strip()),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
