"""The dataset card (``README.md``) of an exported TextCraft Harbor dataset."""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from platoon_textcraft.harbor import export

DIFFICULTY_DEPTHS = {"easy": "2-3", "medium": "4-6", "hard": "7-9", "extreme": "10-12"}
# The commands and what they print, as a Markdown table.
COMMANDS = "\n".join(
    [
        "| Command | platoon tool | Prints |",
        "|---|---|---|",
        "| `textcraft craft TARGET COUNT INGREDIENT=N ...` | `craft` "
        "| `Successfully crafted N TARGET(s)`, or `Error: ...` and nothing changes |",
        "| `textcraft info ITEM ...` | `get_info` "
        "| each item's recipes, crafting depth, whether it is a base item, and how many the inventory holds, as JSON |",
        "| `textcraft inventory` | `view_inventory` | the inventory, as JSON |",
        "| `textcraft finish MESSAGE` | `finish` | the message; the episode ends |",
    ]
)
PLATOON = "https://github.com/ApGa/platoon"
HARBOR = "https://github.com/harbor-framework/harbor"
ADAPT = "https://arxiv.org/abs/2311.05772"  # ADaPT (Prasad et al., Findings of NAACL 2024), which introduced TextCraft


def _levels(tasks: Sequence[Any]) -> list[str]:
    return [level for level in DIFFICULTY_DEPTHS if any(task.fields.get("difficulty") == level for task in tasks)]


def _split_table(world: str, version: str, tasks: Sequence[Any], splits: Sequence[str]) -> str:
    levels = _levels(tasks)
    rows = ["| Split | Harbor dataset | Tasks |" + "".join(f" {level} |" for level in levels)]
    rows.append("|---|---|---:|" + "---:|" * len(levels))
    for split in splits:
        members = [task for task in tasks if task.split == split]
        counts = [sum(task.fields.get("difficulty") == level for task in members) for level in levels]
        rows.append(
            f"| `{split}` | `{world}-{split}@{version}` | {len(members):,} |" + "".join(f" {n:,} |" for n in counts)
        )
    return "\n".join(rows)


def _difficulty_table(tasks: Sequence[Any]) -> str:
    rows = [
        "| Difficulty | Crafting depth | Reference crafts (min / median / max) | Agent timeout (s) |",
        "|---|---|---|---|",
    ]
    for level in _levels(tasks):
        members = [task for task in tasks if task.fields.get("difficulty") == level]
        steps = sorted(task.fields["num_craft_steps"] for task in members)
        timeouts = sorted(task.fields["agent_timeout_sec"] for task in members)
        rows.append(
            f"| {level} | {DIFFICULTY_DEPTHS[level]} | {steps[0]} / {statistics.median(steps):g} / {steps[-1]} | "
            f"{timeouts[0]:,}-{timeouts[-1]:,} |"
        )
    return "\n".join(rows)


def _source_table(sources: Mapping[str, Mapping[str, Any]]) -> str:
    rows = ["| Split | platoon task file | Tasks | SHA-256 |", "|---|---|---:|---|"]
    rows += [
        f"| `{split}` | `{info['file']}` | {info['tasks']:,} | `{info['sha256']}` |" for split, info in sources.items()
    ]
    return "\n".join(rows)


def render_card(
    *,
    variant: Any,
    tasks: Sequence[Any],
    version: str,
    splits: Sequence[str],
    repo_id: str,
    source_revision: str | None,
    recipes: bytes,
    document: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, Any]],
) -> str:
    world, pretty = variant.world, variant.pretty
    count = len(tasks)
    size = "n<1K" if count < 1000 else "1K<n<10K" if count < 10000 else "10K<n<100K"
    items = len(set(document["recipes"]) | set(document["base_items"]))
    max_depth = max(document["depths"].values(), default=0)
    first, last = splits[0], splits[-1]
    medium = sum(task.split == first and task.fields.get("difficulty") == "medium" for task in tasks)
    revision = f"commit `{source_revision}`" if source_revision else "the commit this card does not name"
    repo_url = f"https://huggingface.co/datasets/{repo_id}"
    image = export.BASE_IMAGE.split("@")[0].removeprefix("docker.io/library/")
    synth = variant.option == "synth"
    world_text = (
        f"the world that `platoon_textcraft.synth_recipe_generator` builds with seed 42 and 25 items per domain and "
        f"tier: {items:,} items, {len(document['base_items'])} of them base materials, and recipes up to depth "
        f"{max_depth}. The environment builds this world in process when it loads"
        if synth
        else f"the crafting recipes of Minecraft: Java Edition 1.16.5, which the TextCraft extension ships unmodified "
        f"in `data/recipes/`: {items:,} items, recipes up to depth {max_depth}"
    )
    generator = "platoon_textcraft.synth_tasks" if synth else "platoon_textcraft.tasks"
    # platoon generated TextCraft-Synth's recipes; original TextCraft's are Minecraft's.
    license_text = (
        "MIT, the license of platoon, whose code generated the recipes and the tasks and whose game file every task\n"
        "carries (`LICENSE`)."
        if synth
        else "MIT, the license of platoon, whose code generated the tasks and whose game file every task carries\n"
        "(`LICENSE`). The recipes are the crafting recipes of Minecraft: Java Edition 1.16.5 (see\n"
        "[Provenance](#provenance)); Minecraft is a trademark of Mojang, and this dataset is not affiliated with\n"
        "Mojang or Microsoft."
    )
    return f"""---
pretty_name: {pretty}
license: mit
language:
- en
tags:
- rl-environment
- harbor
- textcraft
- crafting
- planning
size_categories:
- {size}
viewer: false
---

# {pretty}

{pretty} is a text crafting game for training and evaluating LLM agents on long-horizon planning: craft the
target items from a starting inventory of base materials, through recipes up to {max_depth} levels deep. This
repository holds it as an RL environment in the [Harbor][harbor] task format: {count:,} tasks, in each of which the
agent plays through a `textcraft` command in a container, and a verifier replays the agent's moves to grade them.

The tasks, rules, observations and success rule are those of `{variant.named}`, the {pretty} environment of the
TextCraft extension of [platoon][platoon], exported task for task.
[ADaPT][adapt] (Prasad et al., Findings of NAACL 2024) introduced TextCraft, a text game of crafting Minecraft items;
TextCraft-Synth is platoon's synthetic-recipe variant of it.

## Use it

### With Harbor

[Harbor](https://pypi.org/project/harbor/) (0.21.0 or later) loads the tasks from this repository by the
datasets in `registry.json`, one per split:

```bash
harbor run --repo {repo_url} --dataset {world}-{last}@{version} --agent oracle
harbor run --repo {repo_url} --dataset {world}-{first}@{version} \\
  --include-task-name '{world}-{first}-medium-*' --agent terminus-2 --model <model> \\
  --ak record_terminal_session=false
```

The oracle runs each task's reference solution and scores 1 on every task; the `nop` agent scores 0. Task names
carry the split and the difficulty, so `--include-task-name` picks difficulties: `{world}-{first}-medium-*` is
the {medium:,}-task medium training set that platoon's RAO configurations train on.

The tasks have no network (`network_mode = "no-network"`): this repository holds every reference solution, which
an agent with internet access could fetch. Harbor's Docker environment enforces it. Terminal agents that drive
the container from outside, such as terminus-2, need nothing more: the image has tmux, and
`--ak record_terminal_session=false` (above) stops terminus-2 from spending up to four minutes trying to install
its session recorder. An agent installed in the container needs its hosts allowed with `--allow-environment-host`
(or `--allow-agent-host` for its run alone). Harbor's Singularity environment cannot enforce `no-network` and
refuses the tasks. To run them there, edit `network_mode` to `"public"` in a copy: its in-container server
installs its own Python packages over the network when a container starts.

With Harbor, run untrusted agents, and train, only on an environment whose verifier runs as root, such as Docker.
Under Singularity the agent and the verifier run as the same unprivileged user and share `/logs/verifier`: an agent
that leaves a reward file there and makes the directory read-only stops Harbor from clearing it and from running
the tests, and Harbor then reads the agent's reward (seen with Harbor 0.21.0 and harbor-singularity-hpc 0.1.0).

### With platoon

platoon's Harbor support fetches the dataset at a commit, selecting tasks on the fields of `task-index.json`:

```yaml
environments:
  - name: {world}
    service: harbor
    service_kwargs:
      dataset: hf://{repo_id}@<commit>
      select: {{split: {first}, difficulty: medium}}   # the RAO training set
      image_map: images/site.json   # `platoon harbor images build` builds the one image every task uses
      provider: singularity
      provider_options: {{allow_network_none: true}}   # the tasks run without a network
    harness: <harness>
```

platoon stops the agent's processes and grades each task in a verifier sandbox of its own, which gets the action log
but not the agent's `/logs/verifier`, so the Singularity caveat above does not apply to it: a planted reward scores 0.

To train on one split and evaluate on another from one entry, fetch every task (no `select`) and pass a local copy
of this repository's `splits.json` as `splits`:

```bash
hf download {repo_id} splits.json --repo-type dataset --revision <commit> --local-dir .
```

It maps the environment's split names (`{first}`, `{last}`, `{first}-medium`, `{last}-hard`, ...) to task names,
for `dataset_kwargs: {{split_name: {first}-medium}}` and `eval_dataset_kwargs: {{split_name: {last}}}`.

platoon also runs the same game in process (`env: {variant.named}`), which is what its recursive agents need:
subagents that share the parent's inventory and are graded on their own targets. This dataset serves runners
that take Harbor tasks.

## The game

The goal names the targets, for example `Craft the following items: 3x a1_i5`. The starting inventory holds the
base materials they need, with some to spare and a few unrelated ones. Item names carry no meaning, so the agent
looks recipes up.

{COMMANDS}

A craft must name exactly the recipe's ingredients, each in the recipe's amount times `COUNT / result_count`,
and `COUNT` must be divisible by the recipe's `result_count`. The episode succeeds when it ends with `finish` and
a non-empty message, and the inventory then holds every target count on top of what it started with; there is
no partial credit. The commands print what platoon's tools return, character for character; `craft --json` and
`info --json` take the tools' JSON arguments.

## Task format

```
registry.json        Harbor datasets: {world}-<split>@{version}
task-index.json      every task's split, difficulty, crafting depth, reference crafts, source ids and content hash
splits.json          the environment's split names to task names
LICENSE              MIT, platoon's license, which covers the game file every task carries
tasks/<name>/
  instruction.md     the game description of platoon's TextCraft agents, the goal, and the commands
  task.toml          timeouts, metadata, no network, a separate verifier, the action log as an artifact
  environment/       Dockerfile, textcraft_game.py, recipes.json, task.json
  tests/             test.sh, Dockerfile, and the verifier's copies of the game, the recipes and the task
  solution/solve.sh  the task's gold trajectory as textcraft commands, then finish
```

- **One image.** Every Dockerfile, the environment's and the verifier's, starts with the same two steps:
  `FROM {image}`, pinned by digest, and one `RUN` that adds tmux and the `textcraft` command. The rest only
  copies files: the game and the recipes, which are the same in every task, then the task's `task.json`. Docker
  reuses the cached layers across tasks, and platoon builds the image once and copies the files when a session
  starts. A comment on each Dockerfile's first line names its task, so that builders that cache by a
  Dockerfile's text (harbor-singularity-hpc) never reuse one task's image for another.
- **State.** `/app/textcraft` holds the game, the recipes, `task.json` and the action log `actions.jsonl`. Every
  command replays the log under a lock, runs, and appends one record, so the log is the state.
- **Resources.** {export.CPUS} CPU, {export.MEMORY_MB // 1024} GiB of memory, no GPU, no network. The agent's
  timeout is {export.AGENT_TIMEOUT_BASE_S} s plus {export.AGENT_TIMEOUT_PER_CRAFT_S} s per craft of the
  reference solution, rounded up to a minute; the verifier's is {export.VERIFIER_TIMEOUT_S} s.

{_difficulty_table(tasks)}

## The verifier

`tests/test.sh` replays `/app/textcraft/actions.jsonl` from the task's starting inventory with its own copies of
the game, the recipes and the task, and writes `reward.json` (`{{"reward": 1.0}}` or `{{"reward": 0.0}}`) and its
reasons (`textcraft-verdict.json`) to `/logs/verifier`:

- The reward is 1 only when the log is an episode the rules allow that ends with `finish` with every target
  crafted. A log without `finish`, or with an empty finish message, scores 0, as in platoon.
- The log must be a regular file of at most 32 MiB, reached without symbolic links. A record that does not
  parse, breaks the record format or the numbering, follows `finish`, or claims a craft outcome the rules do not
  give ends the replay with reward 0: a hand-written log scores only if the moves it records are legal and reach
  the goal. A craft the rules reject, recorded as rejected, changes nothing, as in the environment.
- The log is the agent's to edit, so an agent can take back a move by deleting its record. The reward still
  needs every move the final log records to be legal from the task's starting inventory, and the goal reached:
  editing the log never earns a reward that its moves do not.
- The verifier runs in a container of its own (`environment_mode = "separate"`) that receives the log as an
  artifact; reward files already in `/logs/verifier` are removed first, where the runner can clear that directory
  (not under Singularity: see [With Harbor](#with-harbor)). The same `test.sh` also runs in the agent's
  container when a runner uses a shared verifier, but there the agent, which is root in that container, could
  have replaced Python or the grader itself: use a shared verifier only for agents you trust.

## Splits

{_split_table(world, version, tasks, splits)}

Difficulties are crafting depths: easy 2-3, medium 4-6, hard 7-9 and extreme 10-12. Train and validation targets
are disjoint: the generator held out a fifth of each tier's craftable items as validation targets, and each split
holds every pair of one of its items and a count, one task each.

## Provenance

The tasks are platoon's {pretty} task files, exported by `python -m platoon_textcraft.harbor export
--variant {variant.option} --version {version}` (the TextCraft extension, `extensions/textcraft`) at platoon
{revision}. The export writes the same bytes from the same inputs.

{_source_table(sources)}

platoon generated the task files with `{generator}` (seed 42) on {world_text}. The world's recipes are
`recipes.json`, `sha256:{hashlib.sha256(recipes).hexdigest()}`; the export checks that it answers every
recipe lookup as the environment's database does. The generator draws each task's unrelated base materials from
a Python set, so running it again gives the same goals, but in another order and with other starting inventories,
unless it runs under the same string-hash seed: the task files above are the source. Each reference solution is
the task's gold trajectory from the same files, which the export plays with the game, failing unless it reaches
the goal.

## License

{license_text} The images are built from the `{image}` base image at run time; it is not part of this
repository.

## Citation

platoon was built for [Recursive Agent Optimization](https://arxiv.org/abs/2605.06639) (RAO), whose experiments
train on TextCraft-Synth. Please cite RAO, platoon, which generated and exported the tasks, and ADaPT, which
introduced TextCraft:

```bibtex
@article{{gandhi2026rao,
  title   = {{Recursive Agent Optimization}},
  author  = {{Gandhi, Apurva and Chakraborty, Satyaki and Wang, Xiangjun and Kumar, Aviral and Neubig, Graham}},
  journal = {{arXiv preprint arXiv:2605.06639}},
  year    = {{2026}}
}}

@misc{{gandhi2025platoon,
  author       = {{Gandhi, Apurva}},
  title        = {{{{Platoon}}: Build and Train Systems of Agents}},
  howpublished = {{\\url{{https://github.com/ApGa/platoon}}}},
  year         = {{2025}}
}}

@inproceedings{{prasad2024adapt,
  title     = {{{{ADaPT}}: As-Needed Decomposition and Planning with Language Models}},
  author    = {{Prasad, Archiki and Koller, Alexander and Hartmann, Mareike and Clark, Peter and Sabharwal, Ashish
               and Bansal, Mohit and Khot, Tushar}},
  booktitle = {{Findings of the Association for Computational Linguistics: NAACL 2024}},
  pages     = {{4226--4252}},
  year      = {{2024}},
  url       = {{https://aclanthology.org/2024.findings-naacl.264/}}
}}
```

[adapt]: {ADAPT}
[harbor]: {HARBOR}
[platoon]: {PLATOON}
"""
