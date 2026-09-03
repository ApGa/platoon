# Plugin catalog

Every plugin that ships in `plugins/`, what it needs to run, and the commands that run it. Use this
page to pick a starting point: which plugin is closest to your task, and whether you can run it on a
laptop, on a GPU node, or only against an external service.

If you want to know how a plugin is *built* rather than what exists, read
[plugin anatomy](../walkthroughs/plugin-anatomy.md) and
[build a plugin](../tutorials/build-a-plugin.md).

## At a glance

| Plugin | Task domain | Backends | Inference | External setup | Recursive | Good template for |
|---|---|---|---|---|---|---|
| [number-search](#number-search) | Guess a hidden number from "too low" / "too high" | <span class="pl-tag pl-tag--both">Both</span> | no | none | no | the smallest complete plugin |
| [textcraft](#textcraft) | Minecraft-style crafting, plus a procedurally generated deep-recipe benchmark | <span class="pl-tag pl-tag--both">Both</span> | yes | none | yes | registry wiring, generated datasets, three rollout styles |
| [codegrep](#codegrep) | Predict which files an issue touches | <span class="pl-tag pl-tag--both">Both</span> | no | clones repos from GitHub at rollout time | no | a plugin built on another plugin |
| [appworld](#appworld) | API tasks over a simulated multi-app world | <span class="pl-tag pl-tag--areal">AReaL</span> | yes | `appworld` package, downloaded data, `APPWORLD_ROOT` | yes | wrapping a stateful third-party simulator |
| [oolong](#oolong) | Long-context aggregation over logs and D&D transcripts | <span class="pl-tag pl-tag--both">Both</span> | yes | HF datasets (`oolongbench/*`) | yes | map-reduce delegation with no tools at all |
| [email-search](#email-search) | Answer questions about an Enron mailbox | <span class="pl-tag pl-tag--both">Both</span> | yes | local SQLite/FTS5 index built from HF; LLM judge | yes | a locally built retrieval index and a structured final answer |
| [deepdive](#deepdive) | Multi-hop research over the live web | <span class="pl-tag pl-tag--both">Both</span> | yes | `TAVILY_API_KEY`, HF `zai-org/DeepDive`, LLM judge | yes | an external API tool with credentials and rate limiting |
| [openreward](#openreward) | OpenHands agents against containerized gyms — Toolathlon, TMax, SWE-rebench | <span class="pl-tag pl-tag--both">Both</span> | yes | a running OpenReward env server; multi-node Slurm for the real recipes | yes | environment mixtures, curricula, judged subagents |
| [openhands](#openhands) | none — library plugin | — | — | OpenHands SDK packages | provides the machinery | adapting an external agent SDK to Platoon |

Only **textcraft** is wired into the registry today. It is the one plugin with a `registry.py`, the
one with a `[project.entry-points."platoon.plugins"]` declaration, and the only one shipping a config
with a top-level `environments:` list. Every other plugin has its own `train_*.py` that imports its
rollout and task loader directly. [The registry](../architecture/registry.md) explains what the
migration buys you.

## Infrastructure bar

Three tiers, and they matter more than the task domain when you are deciding what to try first.

| Tier | What it needs | Which plugins |
|---|---|---|
| Inference / eval only | An OpenAI-compatible endpoint and a key. No local GPU. | textcraft, appworld, oolong, email-search, deepdive, openreward — every plugin with an `inference_scripts/` directory |
| <span class="pl-tag pl-tag--tinker">Tinker</span> training | A Tinker service account. No local GPU. | number-search, textcraft, codegrep, oolong, email-search, deepdive, openreward |
| <span class="pl-tag pl-tag--areal">AReaL</span> training | Linux/CUDA, and in practice eight GPUs on a node. | all of the above, plus appworld |

Every AReaL config in the repository sets `cluster.n_gpus_per_node: 8`; there is no single-GPU AReaL
config to copy. The openreward recipes go further, to 16 and 32 nodes under Slurm. With one GPU or
none, start from Tinker or from an inference run.

Each plugin's own service requirements stack on top of the tier. An inference run of deepdive still
needs a Tavily key, and an inference run of openreward still needs a gym server.

!!! warning "Shipped AReaL configs are site-specific"
    `cluster.fileroot` is `/mnt/efs/tmp/areal/experiments` in most configs and a `/lustre/fsw/...`
    path in the `nv_*` variants, and `stats_logger.wandb.mode` is `online`. Change both before
    running anywhere but the author's cluster.

!!! warning "Two override syntaxes"
    AReaL entrypoints parse overrides through `areal.api.cli_args.load_expr_config`, so overrides are
    bare `key=value` with **no** leading dashes. Tinker and inference entrypoints use `load_config`
    in <span class="pl-src">platoon/utils/config.py</span>, which only accepts
    `--dotted.key value`. Getting this backwards silently drops the override. See
    [the CLI reference](cli.md).

---

## number-search

Binary search as an RL task. Each task carries a range and a hidden target; the only action is
`guess(n)`, which answers "Too low" / "Too high" or sets the finish message on a correct guess.
Reward is 1.0 for a correct guess, 0.0 otherwise.

**Data.** Committed next to the package: 50,000 training tasks and 1,000 validation tasks as JSONL.
Nothing to download, no service.

```bash
cd plugins/number-search
uv sync --extra areal          # or --extra tinker

# optional: regenerate the datasets
uv run python -m platoon.number_search.tasks --num_samples 50000 --eval_size 1000

# AReaL
uv run python3 platoon/number_search/train.py \
  --config platoon/number_search/nv_number_search_cispo_areal.yaml

# Tinker
uv run python -m platoon.number_search.train_tinker \
  --config platoon/number_search/number_search_tinker.yaml
```

Other configs: `number_search_areal.yaml` and `nv_number_search_areal.yaml` use GRPO;
`nv_number_search_cispo_areal.yaml` and `nv_number_search_cispo_areal-1.yaml` use CISPO and differ
from each other only in learning rate and trial name.

**Worth copying.** The whole plugin is five short modules, and
<span class="pl-src">plugins/number-search/platoon/number_search/env.py</span> is the clearest
statement of what an environment is: a factory that closes over the answer, a tuple of callables
handed to `IPythonCodeExecutor`, and an `evaluate()` returning `(reward, misc)`. `train.py` and
`train_tinker.py` are the reference for the explicit-wiring path — copy them and change the imports.

Do not copy the reward check literally. `NumberSearchEnv.evaluate` tests for the substring
`"correctly"` in the finish message, which is fine for a smoke test and fragile for anything else.

## textcraft

Two benchmarks in one plugin. *Original TextCraft* uses the 860 vanilla Minecraft recipe files and
targets items at crafting depth 2–5. *TextCraft-Synth* procedurally generates a 13-tier crafting
world with deliberately meaningless item names (`m0_i1`, `c3_i7`) so the model has to call
`get_info()` instead of recalling what an oak plank is. Its trees reach depth 12, which is where one
agent's step budget stops being enough and delegation starts to pay.

**Data.** Everything is committed: the recipe JSONs, `textcraft_train.jsonl` / `textcraft_val.jsonl`
(1,000 / 100), and `textcraft_synth_train.jsonl` / `textcraft_synth_val.jsonl` (2,522 / 632). No
external service.

```bash
cd plugins/textcraft
uv sync --extra areal          # or --extra tinker

# original TextCraft
uv run python3 platoon/textcraft/train_scripts/areal/train_areal.py \
  --config platoon/textcraft/configs/areal/textcraft_areal.yaml
uv run python -m platoon.textcraft.train_scripts.tinker.train_tinker \
  --config platoon/textcraft/configs/tinker/textcraft_tinker.yaml

# TextCraft-Synth, AReaL: one script, rollout style chosen by the config flags
uv run python3 platoon/textcraft/train_scripts/areal/train_areal_synth.py \
  --config platoon/textcraft/configs/areal/textcraft_synth_ctx8192_depth_aware_medium_areal.yaml

# TextCraft-Synth, Tinker: one script per rollout style
uv run python -m platoon.textcraft.train_scripts.tinker.train_tinker_synth_depth_aware \
  --config platoon/textcraft/configs/tinker/textcraft_synth_depth_aware_tinker.yaml

# the registry route: shared trainer, no plugin train script
uv run python -m platoon.train.tinker.train \
  --config platoon/textcraft/configs/tinker/textcraft_synth_depth_aware_tinker.yaml

# inference against an OpenAI-compatible endpoint
uv run python platoon/textcraft/inference_scripts/run_inference.py \
  --config platoon/textcraft/configs/inference/textcraft_inference.yaml
uv run python platoon/textcraft/inference_scripts/run_synth_inference.py \
  --config platoon/textcraft/configs/inference/textcraft_synth_inference.yaml
```

The AReaL synth configs form a matrix: context length (`ctx4096` / `ctx8192` / `ctx40000`) × rollout
style (`linear` / `recursive` / `depth_aware`) × difficulty (`medium` / `hard`), with `nv_*` variants
for a lustre fileroot and `*_prealloc_2node` variants for Slurm. The `recursive:` and `depth_aware:`
top-level keys come from `TextCraftSynthArealTrainerConfig` in
<span class="pl-src">plugins/textcraft/platoon/textcraft/areal_config.py</span>, and
`train_areal_synth.py` checks `depth_aware` first.

!!! warning "Pass `--config` explicitly to the Tinker scripts"
    Three of the four compute a `default_config_path` next to themselves (`textcraft_tinker.yaml`,
    `textcraft_synth_tinker.yaml`) instead of under `configs/tinker/`, so the default resolves to a
    file that does not exist. Only `train_tinker_synth_depth_aware.py` points at the right path.

**Worth copying.** Two things. <span class="pl-src">plugins/textcraft/platoon/textcraft/registry.py</span>
is the only worked example of registering a task loader, dataset loader, rollouts, and a reward
processor so the shared trainers can be driven entirely from YAML. And the linear → recursive →
depth-aware progression in <span class="pl-src">plugins/textcraft/platoon/textcraft/env.py</span> is
the cleanest illustration of the two delegation budget policies: the recursive variant lets the model
choose `num_steps` out of a shared budget, while the depth-aware variant fixes the child budget at
construction and caps only tree depth. [Recursive agents](../recipes/recursive.md) covers the
trade-off.

The generated `synth_recipes/` directory is an inspection dump, not the runtime database — both the
environment and the dataset regenerate recipes in memory from `seed=42` with
`items_per_domain_tier=25`. Regenerate the dataset with a different value and you must pass the same
value to the environment factory.

## codegrep

Given a SWE-bench-style issue description and a repository, predict the files that need to change.
Reward is the F1 between the predicted file list and the ground-truth patch files.

**Data and services.** A committed 42 MB `train.parquet` of 2,438 rows, split 80/20 at load time. At
rollout time the plugin clones `https://github.com/{repo}.git` into
`{rollout_config.output_dir}/testbed/` and checks out `base_commit`, so it needs network access and
disk. Existing clones are reused across runs.

```bash
cd plugins/codegrep
uv sync --extra tinker         # or --extra areal; also installs plugins/openhands editable

uv run python -m platoon.codegrep.train_tinker \
  --config platoon/codegrep/codegrep_tinker.yaml
uv run python3 platoon/codegrep/train.py \
  --config platoon/codegrep/codegrep_areal.yaml
```

**Worth copying.** It is one of the two plugins built on another plugin — openreward is the other,
and far larger. `pyproject.toml` lists both `platoon` and `platoon-openhands` as local editable path
sources, and `CodeGreEnv` subclasses `OpenHandsEnv` rather than `CodeActEnv`. If your task needs a
real software agent but not a full gym, this is the smallest example of that shape.

!!! bug "codegrep's reward path raises `KeyError: 'repo_dir'`"
    `reward_function` in <span class="pl-src">plugins/codegrep/platoon/codegrep/env.py</span> reads
    `instance["repo_dir"]` outside its `try` block, but `create_task_from_instance` in
    <span class="pl-src">plugins/codegrep/platoon/codegrep/tasks.py</span> populates only
    `instance_id`, `repo`, `base_commit`, `problem_statement` and `target`, and the rollout does not
    add `repo_dir` either. Treat this plugin as a structural example, not a working recipe, until
    that is fixed.

## appworld

AppWorld tasks: day-to-day instructions ("cancel my subscription and pay my roommate back") executed
against a stateful in-process simulation of several apps. Root tasks are scored by AppWorld's own
success oracle rather than an LLM judge.

**Data and services.** The `appworld` package is a git dependency carrying LFS data, and the
benchmark data has to be installed once. `APPWORLD_ROOT` is read at runtime and by the SFT exporter.
Recursive and depth-aware runs additionally need an LLM for rubric judging of subagent goals.

```bash
cd plugins/appworld
uv sync --extra areal

export APPWORLD_ROOT=/path/to/appworld
uv run appworld install
uv run appworld download data

uv run python3 platoon/appworld/train_scripts/areal/train_areal.py \
  --config platoon/appworld/configs/areal/appworld_ctx40000_4b-linear.yaml

uv run python -m platoon.appworld.inference_scripts.run_inference \
  --config platoon/appworld/configs/inference/appworld_inference.yaml
```

There is **no Tinker train script for appworld** — `train_scripts/` holds only `areal/` and the SFT
exporter. The eight AReaL configs cover 4B and 14B models across linear, recursive, and depth-aware
modes; `AppWorldArealTrainerConfig` supplies the `recursive` and `depth_aware` flags, and
`depth_aware` wins when both are set.

!!! warning "The README's inference command is wrong"
    `plugins/appworld/README.md` says `python -m platoon.appworld.run_inference`. No such module
    exists — use `platoon.appworld.inference_scripts.run_inference`, as above.

**Worth copying.** Three patterns. Prompts live in `prompts/*.jinja` and are rendered through a
`PromptRetriever`, so a prompt variant is a template file rather than a Python subclass. The
depth-aware executor rebinds `launch_subagent` in the shell namespace to a closure with `max_steps`
already fixed, which is how you take a knob out of the model's action space. And
`collections_to_sft_data.py` converts finished rollouts into an SFT dataset filtered by reward — the
only rollouts-to-SFT path in the repo.

## oolong

Long-context aggregation from the Oolong benchmark: counting, per-user, and timeline questions over
synthetic logs and real D&D campaign transcripts. There are no tools. The whole context is a string
preloaded into the agent's Python namespace, and the agent's only move is to write code that reads it.

**Data and services.** Hugging Face datasets `oolongbench/oolong-synth` and `oolongbench/oolong-real`,
downloaded lazily on first use. Root answers are graded by a verbatim port of the benchmark's own
scorers, so linear runs need no judge; recursive runs call an LLM judge for subagent trajectories.

```bash
cd plugins/oolong
uv sync --extra areal          # or --extra tinker

# optional: materialize task JSONLs locally
uv run python -m platoon.oolong.tasks --generate --dataset both

uv run python platoon/oolong/train_scripts/areal/train_areal.py \
  --config platoon/oolong/configs/train/areal/oolong_recursive_areal.yaml
uv run python -m platoon.oolong.train_scripts.tinker.train_tinker \
  --config platoon/oolong/configs/train/tinker/oolong_linear_tinker.yaml

uv run python -m platoon.oolong.inference_scripts.run_inference \
  --config platoon/oolong/configs/inference/oolong_inference.yaml
```

Oolong has no train split: `train_areal.py` trains on the validation split and evaluates on the test
split. `OolongArealTrainerConfig` exposes dataset filters as top-level config keys — `oolong_dataset`,
`task_group`, `answer_type`, `min_context_len`, `max_context_len` — where the context-length bounds
count characters, not tokens.

**Worth copying.** The delegation style. Oolong's bound `launch_subagent(goal, context)` deep-copies
`task.misc`, swaps in a *chunk* of the context, and launches the child on that chunk: map-reduce over
a context window rather than decomposition of a goal. It is also the plugin to read for
partial-credit scoring — `eval_helpers.py` ports the upstream scorers, including `0.75 ** |gold -
pred|` for numeric answers.

## email-search

The ART-E task: answer a natural-language question about an Enron mailbox by searching and reading
emails, then return JSON carrying both an answer and the source message ids. Both halves are graded.

**Data and services.** A local SQLite database with an FTS5 index, built once from the HF corpus
`corbt/enron-emails`; the questions come from `corbt/enron_emails_sample_questions`. The environment
opens the database read-only and caches one connection per process. Root answers go through an LLM
judge after an exact normalized-string short circuit, so this plugin needs judge credentials even for
linear runs.

```bash
cd plugins/email-search
uv sync --extra tinker         # or --extra areal

uv run python -m platoon.email_search.data.local_email_db --overwrite
# or on faster local disk:
uv run python -m platoon.email_search.data.local_email_db --db-path /tmp/enron_emails.db --overwrite
export PLATOON_EMAIL_SEARCH_DB_PATH=/tmp/enron_emails.db

uv run python -m platoon.email_search.train_scripts.tinker.train_tinker \
  --config platoon/email_search/configs/tinker/email_search_tinker.yaml
uv run python platoon/email_search/train_scripts/areal/train_areal.py \
  --config platoon/email_search/configs/areal/email_search_areal_recursive.yaml

uv run python -m platoon.email_search.inference_scripts.run_inference \
  --config platoon/email_search/configs/inference/email_search_inference.yaml
```

The README names `configs/areal/email_search_areal.yaml`, which does not exist; the two real files
are `email_search_areal_linear.yaml` and `email_search_areal_recursive.yaml`.
`EmailSearchArealTrainerConfig` defaults `recursive` to `True`, unlike the other plugins' configs.

**Worth copying.** `data/local_email_db.py` is a complete data-preparation CLI — schema, indexes, FTS
triggers, corpus filters, dedup — and is the model to follow when your task needs a locally built
index rather than a downloaded dataset. The environment also enforces a *structured* final answer
(`finish(json.dumps({"answer": ..., "sources": [...]}))`) and grades sources separately from the
answer. And `RootTaskMetrics` records nine process facts per episode — did the agent ever find the
right email, ever read it, ever try to read an invalid id — which land in `reward_misc` as cheap
diagnosis alongside the reward. See [rewards](../customization/rewards.md).

## deepdive

Multi-hop research questions from `zai-org/DeepDive`, answered against the live public web through
Tavily. Root answers are compared with a `ground_truth` field by an LLM judge; subagent goals are
graded by a rubric checklist.

**Data and services.** A Tavily account. `TAVILY_API_KEY` is read at *import* time in
<span class="pl-src">plugins/deepdive/platoon/deepdive/search_tools.py</span>, so set it before
anything imports `platoon.deepdive`. This is the only plugin whose rollouts hit the public internet,
and whose per-run cost scales with rollout count.

```bash
cd plugins/deepdive
uv sync --extra areal          # or --extra tinker
export TAVILY_API_KEY=tvly-...

uv run python platoon/deepdive/train_scripts/areal/train_areal.py \
  --config platoon/deepdive/configs/areal/deepdive_areal.yaml
uv run python -m platoon.deepdive.train_scripts.tinker.train_tinker \
  --config platoon/deepdive/configs/tinker/deepdive_tinker.yaml

uv run python -m platoon.deepdive.inference_scripts.run_inference \
  --config platoon/deepdive/configs/inference/deepdive_inference.yaml
```

The plugin ships no README. Note that the Tinker config inverts the splits relative to the dataclass
defaults: `train_split: qa_sft`, `eval_split: qa_rl`.

**Worth copying.** The tool-registration form is the simplest in the repo — module-level coroutines
(`search_web`, `view_webpage_content`) passed straight into the executor's `actions` tuple with all
three AST guards enabled. And `search_tools.py` is the reference for wrapping a rate-limited external
API: a sliding-window limiter plus a semaphore, opt-in through `PLATOON_TAVILY_RATE_LIMIT_ENABLED`,
tuned by `PLATOON_TAVILY_MAX_REQUESTS_PER_MINUTE` (default 200) and `PLATOON_TAVILY_MAX_CONCURRENCY`
(default 1000), and rebuilt after a fork by comparing the pid.

## openreward

The frontier end of the repo: OpenHands SDK agents trained against containerized OpenReward gyms
(Toolathlon, TMax, SWE-rebench), with environment mixtures, curriculum staging, and two-stage
subagent reward judging. [OpenReward integration](../integrations/openreward.md) is the full page;
this is the catalog entry.

**Data and services.** A running OpenReward environment server, an MCP bridge subprocess per rollout,
and — for the recipes that were actually run — a 16- or 32-node Slurm allocation with per-node
environment servers. The single-node `toolathlon_openhands_areal.yaml` against a local Docker gym is
the smallest entry point.

```bash
docker run --rm -e OPENREWARD_PORT=8080 -p 8080:8080 \
  ghcr.io/apga/openreward-toolathlon-gym:latest

cd plugins/openreward
uv sync --extra areal          # or --extra tinker

# AReaL: bare key=value overrides
uv run python -m platoon.openreward.train_scripts.areal.train_areal \
  --config platoon/openreward/configs/areal/toolathlon_openhands_areal.yaml \
  openreward.session_url=http://localhost:8080

# Tinker: --dotted.key overrides
uv run python -m platoon.openreward.train_scripts.tinker.train_tinker \
  --config platoon/openreward/configs/tinker/toolathlon_openhands_tinker.yaml \
  --openreward.session_url=http://localhost:8080

uv run python -m platoon.openreward.inference_scripts.run_inference \
  --config platoon/openreward/configs/inference/toolathlon_openhands_inference.yaml \
  --openreward.session_url=http://localhost:8080
```

!!! danger "`openreward.environments` is not the registry's `environments`"
    The mixture list in openreward's configs is nested under the `openreward:` key, and its entries
    are `OpenRewardEnvironmentConfig`: `label`, `env_name`, `session_url`, `session_urls_env_var`,
    `sampling_weight`. That is environment-mixture and curriculum control. It is unrelated to the
    top-level `environments:` list of `EnvironmentConfig`, which wires registered components into the
    shared trainers. Openreward does not use the registry — it ships its own train scripts.

**Worth copying.** The mixture and curriculum machinery: `BalancedEnvironmentSampler` and the
accepted-batch coordinator stop a fast environment from crowding out a slow one, and
`sampling_start_step` stages an environment in partway through a run. See
[curriculum](../recipes/curriculum.md) and [scale](../recipes/scale.md).

## openhands

A library plugin. No tasks, no rollouts, no train scripts. It wraps an OpenHands SDK `Conversation`
as a Platoon `Env`, adds a reasoning-safe context condenser, and supplies the pieces that make
recursion work: a `launch_subagent` OpenHands tool bound to the live episode loop, a parallel task
tracker, and programmatic tool calling. Both `codegrep` and `openreward` depend on it.

It is also the only plugin with no `tinker` / `areal` extras, because it is never installed on its
own — it arrives as a path dependency of whatever plugin needs it.

Details are on the [OpenHands integration](../integrations/openhands.md) page.

## See also

- [Build a plugin](../tutorials/build-a-plugin.md) — the scaffold, step by step.
- [Plugin anatomy](../walkthroughs/plugin-anatomy.md) — what each module is responsible for.
- [Packaging](../customization/packaging.md) — namespace layout, extras, and the uv override block
  every plugin re-declares.
- [Installation](../get-started/installation.md) — backends, extras, and why they conflict.
- [CLI reference](cli.md) — entrypoints and the two override syntaxes.
