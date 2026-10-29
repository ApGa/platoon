# TextCraft for platoon

`platoon-textcraft` is platoon's first-party TextCraft extension: crafting environments for training tool-using and
recursive LLM agents, served as [OpenReward](https://openreward.ai) (ORS) environments, with platoon's named
environments, the CodeAct prompts of platoon's TextCraft agents, and an export of TextCraft-Synth as a
[Harbor](https://github.com/harbor-framework/harbor) dataset.

- **TextCraft** comes from [ADaPT](https://arxiv.org/abs/2311.05772) (Prasad et al., Findings of NAACL 2024): craft
  Minecraft items by their recipes, with shallow crafting trees.
- **TextCraft-Synth** is platoon's synthetic variant: generated recipes with deeper crafting trees (up to 12
  levels), abstract item names and difficulty tags. The [RAO recipe](../../recipes/rao/README.md) trains on it.

## Install

```bash
pip install 'platoon-textcraft[platoon]'
```

The `platoon` extra installs platoon 0.5 with its `ors` extra (`platoon[ors]>=0.5,<0.6`), for what imports platoon: the
prompt builders, the named environments and the task generators. With platoon 0.4, install `platoon-textcraft` 0.3.0,
the release for it (`pip install 'platoon-textcraft[platoon]>=0.3,<0.4'`); with platoon 0.3, 0.2.0 (`>=0.2,<0.3`); and
with platoon 0.2, 0.1.0 (`>=0.1,<0.2`). To train or evaluate with platoon, add its `tinker` extra:
`pip install 'platoon[tinker,ors]' 'platoon-textcraft[platoon]'`. The environments themselves need only openreward and
pydantic: `platoon_textcraft.env` imports neither platoon nor IPython, so any ORS client can serve them after
`pip install platoon-textcraft`.

The configs below and the tests are in platoon's repository, not in the package. From a checkout of platoon, in its
uv workspace:

```bash
uv sync --package platoon-textcraft --extra platoon     # the extension, platoon and the test tools
```

or into an environment of your own: `uv pip install -e "extensions/textcraft[platoon]"`.

## Environments

| Service | Recipes | Splits |
|---|---|---|
| `python://platoon_textcraft.env:TextCraftEnvironment` | Minecraft | `train` (1,000), `val` (100) |
| `python://platoon_textcraft.env:TextCraftSynthEnvironment` | synthetic | `train` (2,522), `val` (632), and `<split>-<difficulty>`: `train-medium` holds the 852 RAO training tasks |

Task ids are `<split>:<index>`. The tools are `craft(ingredients, target)`, `get_info(items)`, `view_inventory()` and
the terminal `finish(message)`, which grades the episode: success (reward 1) when every target was crafted on top
of the starting inventory and the message is not empty.

Subagents share their parent's inventory: the environment's `platoon_fork(task_spec)` gives a child its own targets
(parsed from its goal, `Craft the following items: 2x stick, ...`) on the same inventory, and the child's `finish`
grades its own targets (`agents.child_grading: env`).

## Named environments

Installed, the extension registers `textcraft@1` and `textcraft-synth@1` in the `platoon.environments` entry-point
group (`entries.py`). Each stands for the environment, the TextCraft prompts for `platoon-codeact`
(`platoon_textcraft.prompts:TextCraftPromptBuilder`) and the `val` split for evaluation; `textcraft-synth@1` also
scores with platoon's RAO reward processor, `platoon-delegation`. A config names one with `env:`, picks the arm of
the prompts with `prompt_builder_options.variant` (`linear`, `recursive` or `depth_aware`) and adds the rest:

```yaml
environments:
  - env: textcraft-synth@1
    harness: platoon-codeact
    harness_options:
      prompt_builder_options: {variant: recursive}
    agents: {max_depth: null, budget: linear, child_grading: env}
    dataset_kwargs: {split_name: train-medium}
```

`platoon env show textcraft-synth` prints what it stands for, and `platoon extensions check platoon-textcraft`
checks the extension's metadata. The default episode program, `single`, applies the rollout config's
`skip_subagent_reward_computation` and `propagate_root_success`, in that order.

## Configs

| Config | What it runs |
|---|---|
| `configs/textcraft_train.yaml` | Trains the recursive agent on original TextCraft: the first 1,000 training tasks, 50 steps per episode |
| `configs/textcraft_eval.yaml` | Evaluates the recursive agent on the first 100 validation tasks of original TextCraft |
| `configs/textcraft_synth_eval.yaml` | Evaluates the linear agent on TextCraft-Synth, greedy, up to 800 steps: the easy, medium and hard validation tasks, shuffled with seed 42, first 100 |

```bash
platoon train extensions/textcraft/configs/textcraft_train.yaml
platoon eval extensions/textcraft/configs/textcraft_synth_eval.yaml inference.model_path=tinker://...
```

The RAO paper's TextCraft-Synth experiments, its single agent against its recursive agent and its ablation of the
subagents' rewards, are a recipe of their own: [`recipes/rao/`](../../recipes/rao/README.md).

## Data

The task files ship in the package, under `src/platoon_textcraft/data/`, and `data/SHA256SUMS` pins them:

| File | Tasks |
|---|---|
| `textcraft_train.jsonl`, `textcraft_val.jsonl` | 1,000 and 100 original TextCraft tasks |
| `textcraft_synth_train.jsonl`, `textcraft_synth_val.jsonl` | 2,522 and 632 TextCraft-Synth tasks, the source of the published Harbor dataset |

`data/recipes/` holds the 860 recipes of original TextCraft: the recipe files of Minecraft: Java Edition 1.16.5 (the
game's `data/minecraft/recipes/`), unmodified, as
[InventivetalentDev/minecraft-assets](https://github.com/InventivetalentDev/minecraft-assets/tree/1b7a1d48b0343c0b8da80604598707df428dc32a/data/minecraft/recipes)
extracted them from the game. ADaPT's README sends TextCraft's users there for them; ADaPT's repository does not
include them. TextCraft-Synth needs no recipe files: the environment builds its world in memory (seed 42, 25 items
per domain and tier, 1,452 items).

Every task keeps its gold trajectory (`misc.gold_trajectory`), a sequence of crafts that reaches the goal. The
environments never read it; the Harbor export builds each task's reference solution from it.

The task generators are not deterministic across runs: they draw each task's unrelated base materials from a Python
set, so running them again gives the same goals in another order and with other starting inventories, unless they
run under the same string-hash seed. The shipped files are the source of truth. To generate others, write them
outside the package:

```bash
python -m platoon_textcraft.tasks --num_samples 10000 --eval_size 1000 --out /tmp/textcraft-tasks
python -m platoon_textcraft.synth_tasks --num_samples 10000 --eval_size 1000 --seed 42 --out /tmp/textcraft-synth-tasks
python -m platoon_textcraft.synth_recipe_generator --output-dir /tmp/textcraft-synth-recipes --seed 42 --items-per-tier 25
```

The last command writes the TextCraft-Synth world the environment builds as JSON files, to read.

## Export as a Harbor dataset

`platoon_textcraft.harbor` writes TextCraft-Synth as a Harbor dataset: one task directory per task of the `train`
and `val` splits, which runners that take Harbor tasks (Harbor itself, platoon's `service: harbor`) play in a
container through a `textcraft` command:

```bash
python -m platoon_textcraft.harbor export --variant synth --out /data/textcraft-synth --version 1.0 \
  --source-revision "$(git rev-parse HEAD)"
```

| Option | Default | Meaning |
|---|---|---|
| `--variant` | `synth` | `synth` (TextCraft-Synth) or `original` (Minecraft recipes) |
| `--out` | | a new directory (`--force` replaces one) |
| `--version` | `1.0` | the version of the registry's datasets, `<world>-<split>@<version>` |
| `--splits`, `--limit` | every split, every task | export only these splits, or the first N tasks of each |
| `--repo-id` | `apurvaga/textcraft-synth` | the Hugging Face repository the dataset card's commands name |
| `--source-revision` | none | the platoon commit the card names as the dataset's source |

The same inputs give the same bytes. The dataset published as
[`apurvaga/textcraft-synth`](https://huggingface.co/datasets/apurvaga/textcraft-synth) was exported before
TextCraft moved here, so its card names the earlier command, `python -m platoon.textcraft.harbor export`; an export
from this package gives the same tasks, byte for byte, and a card that differs only in where the tasks came from.
The extension's tests check the first tasks of each split against the published content hashes.

Each task holds:

- `instruction.md`: the game description the linear agent's prompt opens with (without its strategy tips), the
  goal, and the four commands, which print what the environment's tools return;
- `environment/`: a Dockerfile whose build (`python:3.12-slim`, tmux and the `textcraft` command) is the same in
  every task, then copies of the game (`harbor/game.py`, standard library only), the recipes and the task's
  `task.json`;
- `tests/test.sh`: the verifier. It replays the game's action log (`/app/textcraft/actions.jsonl`) with its own
  copies of the game, the recipes and the task, and scores 1 only for an episode the rules allow that ends with
  `finish` with every target crafted. It reads the log only as a regular file of at most 32 MiB, reached without
  symbolic links. `task.toml` runs it in a separate verifier container (`environment_mode = "separate"`), which
  receives the log as an artifact; it also runs in the agent's container.
- `solution/solve.sh`: the gold trajectory as commands. The export plays every one with the game and fails unless
  it reaches the goal.

The dataset's root holds `registry.json` (`textcraft-synth-train@1.0`, `textcraft-synth-val@1.0`), `task-index.json`
(each task's split, difficulty, crafting depth, source ids and Harbor content hash, which platoon's `select:`
filters on), `splits.json` (the environment's split names, such as `train-medium`, to task names, for platoon's
`splits:`), `LICENSE` and the dataset card `README.md`, which says how to run the tasks with Harbor and with
platoon. The tasks have no network, so an agent cannot fetch the published solutions.

## Modules

- `env.py`: the ORS environments (`TextCraftEnvironment`, `TextCraftSynthEnvironment`); `recipe_loader.py`,
  `synth_recipe_generator.py`, `synth_recipe_loader.py`: their recipes.
- `prompts.py`: the system prompts and action spaces of the linear, recursive and depth-aware agents.
- `entries.py`: the named environments.
- `tasks.py`, `synth_tasks.py`: task generation and loading.
- `harbor/`: the Harbor export (`export.py`, `card.py`, `templates/`) and the in-container game and verifier
  (`game.py`).

## Test

```bash
uv sync --package platoon-textcraft --extra platoon
uv run --package platoon-textcraft pytest extensions/textcraft/tests -n auto --dist loadfile
```

The tests run on CPU and share platoon's test helpers (`tests/`). The golden replays of recursive CodeAct rollouts
need the `Qwen/Qwen3-4B-Instruct-2507` tokenizer in the Hugging Face cache, and skip without it.

## License

MIT (`LICENSE`), platoon's license, for the code and the data: the task files, which platoon generated, and the
recipe files in `data/recipes/`, which are Minecraft's (see [Data](#data)). Minecraft is a trademark of Mojang;
platoon is not affiliated with Mojang or Microsoft.

## Citation

TextCraft comes from ADaPT; platoon generated TextCraft-Synth for
[Recursive Agent Optimization](https://arxiv.org/abs/2605.06639) (RAO):

```bibtex
@inproceedings{prasad2024adapt,
  title     = {{ADaPT}: As-Needed Decomposition and Planning with Language Models},
  author    = {Prasad, Archiki and Koller, Alexander and Hartmann, Mareike and Clark, Peter and Sabharwal, Ashish
               and Bansal, Mohit and Khot, Tushar},
  booktitle = {Findings of the Association for Computational Linguistics: NAACL 2024},
  pages     = {4226--4252},
  year      = {2024},
  url       = {https://aclanthology.org/2024.findings-naacl.264/}
}

@article{gandhi2026rao,
  title   = {Recursive Agent Optimization},
  author  = {Gandhi, Apurva and Chakraborty, Satyaki and Wang, Xiangjun and Kumar, Aviral and Neubig, Graham},
  journal = {arXiv preprint arXiv:2605.06639},
  year    = {2026}
}
```
