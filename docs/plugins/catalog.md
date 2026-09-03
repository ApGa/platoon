# Plugin catalog

The plugins that ship in `plugins/`. Seven are **task plugins** — a task or environment plus the
rollout program that runs it. Two are **capability plugins**, framework machinery other plugins build
on: `openhands` supplies an agent harness, and `openreward` an environment-server integration whose
task suites come from the server rather than the package. Your own plugin does not have to live here;
see [write a plugin](../guides/first-plugin.md).

## At a glance

| Plugin | Task domain | Backends | Evaluation | Needs to run | Multi-agent |
|---|---|---|---|---|---|
| [number-search](#number-search) | Guess a hidden number from "too low" / "too high" | <span class="pl-tag pl-tag--both">Both</span> | — | nothing | no |
| [textcraft](#textcraft) | Minecraft-style crafting, plus a generated deep-recipe benchmark | <span class="pl-tag pl-tag--both">Both</span> | yes | nothing | yes |
| [codegrep](#codegrep) | Predict which files an issue touches | <span class="pl-tag pl-tag--both">Both</span> | — | network access to clone repositories | no |
| [appworld](#appworld) | Everyday API tasks over a simulated multi-app world | <span class="pl-tag pl-tag--areal">AReaL</span> | yes | `appworld` package and its data | yes |
| [oolong](#oolong) | Long-context aggregation over logs and transcripts | <span class="pl-tag pl-tag--both">Both</span> | yes | Hugging Face datasets | yes |
| [email-search](#email-search) | Answer questions about an Enron mailbox | <span class="pl-tag pl-tag--both">Both</span> | yes | a locally built SQLite index; judge endpoint | yes |
| [deepdive](#deepdive) | Multi-hop research over the live web | <span class="pl-tag pl-tag--both">Both</span> | yes | `TAVILY_API_KEY`; judge endpoint | yes |
| [openreward](#openreward) | Software and tool-use gyms — Toolathlon, TMax, SWE-rebench | <span class="pl-tag pl-tag--both">Both</span> | yes | a running OpenReward environment server | yes |
| [openhands](#openhands) | capability plugin — no tasks | — | — | OpenHands SDK packages | provides the machinery |

**Where to start.** Read **number-search** first: it is the smallest complete plugin, and it fits in
one sitting. Then **textcraft**, which is the worked example of registry wiring and of the different
delegation styles. Both run with no external service and no downloads.

Each plugin's own requirements stack on top of whichever backend you use. Training on AReaL needs a
Linux/CUDA node; the Tinker path and evaluation runs need only an endpoint and a key. See
[installation](../get-started/installation.md) and [backends](../architecture/backends.md).

!!! warning "Two override syntaxes"
    AReaL entrypoints take bare `key=value` overrides. Tinker and evaluation entrypoints take
    `--dotted.key value`. See the [CLI reference](../reference/cli.md).

---

## number-search

Binary search as an RL task. Each task carries a range and a hidden target, the only action is
`guess(n)`, and reward is 1.0 for a correct guess. Its datasets ship alongside the package, so there
is nothing to download.

Copy this one when you want the shape of a plugin without any surrounding complexity. Its `env.py` is
the clearest statement of what an environment is — a factory closing over the answer, a tuple of
callables handed to the code executor, and an `evaluate()` returning a reward and a metrics dict.

```bash
cd plugins/number-search
uv sync --extra tinker
uv run python -m platoon.number_search.train_tinker \
  --config platoon/number_search/number_search_tinker.yaml
```

## textcraft

Two benchmarks in one plugin. *Original TextCraft* uses the vanilla Minecraft recipe files and targets
items at crafting depth 2–5. *TextCraft-Synth* generates a 13-tier crafting world with meaningless
item names, so the agent has to look recipes up instead of recalling them; its trees reach depth 12,
deep enough that one agent's step budget runs out and delegation starts to pay.

This is the reference plugin. It is the worked example of a `registry.py` that registers tasks,
datasets, rollouts and reward processing so the shared trainers run straight from YAML, and its
linear, recursive and depth-aware rollouts are the clearest side-by-side of the delegation styles in
[multi-agent workflows](../guides/multi-agent.md). Everything it needs ships with the plugin.

```bash
cd plugins/textcraft
uv sync --extra tinker
uv run python -m platoon.train.tinker.train \
  --config platoon/textcraft/configs/tinker/textcraft_synth_depth_aware_tinker.yaml
```

## codegrep

Given a SWE-bench-style issue description and a repository, predict the files that need to change;
reward is the F1 between the predicted list and the files in the ground-truth patch. The dataset ships
with the plugin; at rollout time it clones the repository under test and checks out the base commit,
so it needs network access and disk.

It is the smallest example of a plugin built on another plugin: it depends on `openhands` and its
environment subclasses `OpenHandsEnv`. Reach for this shape when your task needs a real software agent
but not a full containerized gym.

```bash
cd plugins/codegrep
uv sync --extra tinker
uv run python -m platoon.codegrep.train_tinker \
  --config platoon/codegrep/codegrep_tinker.yaml
```

## appworld

AppWorld tasks — instructions like "cancel my subscription and pay my roommate back", executed against
a stateful simulation of several apps and scored by AppWorld's own success oracle. Install the
`appworld` package and its data once, and point `APPWORLD_ROOT` at it. Training is
<span class="pl-tag pl-tag--areal">AReaL</span> only.

Good template for wrapping a stateful third-party simulator. Two of its patterns travel well: prompts
live as `prompts/*.jinja` templates rendered through a retriever, so a prompt variant is a file rather
than a subclass; and `collections_to_sft_data.py` turns finished rollouts into a reward-filtered SFT
dataset.

```bash
cd plugins/appworld
uv sync --extra areal
export APPWORLD_ROOT=/path/to/appworld
uv run appworld install && uv run appworld download data
uv run python platoon/appworld/train_scripts/areal/train_areal.py \
  --config platoon/appworld/configs/areal/appworld_ctx40000_4b-linear.yaml
```

## oolong

Long-context aggregation from the Oolong benchmark: counting, per-user and timeline questions over
synthetic logs and D&D campaign transcripts, pulled from Hugging Face on first use. There are no tools
at all — the context is a string in the agent's Python namespace, and the agent's only move is to write
code that reads it. Root answers are graded by a port of the benchmark's own scorers.

The delegation style is what to copy. A subagent launch here swaps in a *chunk* of the context and runs
the child on that: map-reduce over a context window, rather than decomposition of a goal.

```bash
cd plugins/oolong
uv sync --extra tinker
uv run python -m platoon.oolong.train_scripts.tinker.train_tinker \
  --config platoon/oolong/configs/train/tinker/oolong_linear_tinker.yaml
```

## email-search

The ART-E task: answer a question about an Enron mailbox by searching and reading emails, then return
JSON carrying both an answer and the source message ids. Both halves are graded, the answer through an
LLM judge. Build the local SQLite/FTS5 index once from the public corpus before running.

Two things are worth copying. `data/local_email_db.py` is a complete data-preparation CLI — schema,
indexes, FTS triggers, dedup — and the model to follow when your task needs an index you build rather
than a dataset you download. And the environment records process facts per episode (did the agent find
the right email, did it read it) into `reward_misc` as cheap diagnosis alongside the reward.

```bash
cd plugins/email-search
uv sync --extra tinker
uv run python -m platoon.email_search.data.local_email_db --overwrite
uv run python -m platoon.email_search.train_scripts.tinker.train_tinker \
  --config platoon/email_search/configs/tinker/email_search_tinker.yaml
```

## deepdive

Multi-hop research questions answered against the live public web through Tavily. Root answers are
judged against a ground-truth field; subagent goals are graded by a rubric checklist. Set
`TAVILY_API_KEY` before anything imports the plugin. This is the only plugin whose rollouts leave your
machine for the open internet, so cost scales with rollout count.

Its tool registration is the simplest in the repository — module-level coroutines passed straight into
the executor's action tuple. `search_tools.py` is the reference for wrapping a rate-limited external
API behind a sliding-window limiter and a semaphore.

```bash
cd plugins/deepdive
uv sync --extra tinker
export TAVILY_API_KEY=...
uv run python -m platoon.deepdive.train_scripts.tinker.train_tinker \
  --config platoon/deepdive/configs/tinker/deepdive_tinker.yaml
```

## openreward

A capability plugin: an integration with an external environment server, plus the mixture and reward
machinery around it. OpenHands agents train against containerized OpenReward gyms — Toolathlon, TMax,
SWE-rebench — with environment mixtures, curriculum staging and two-stage subagent reward judging.
Environments run as a service: point the plugin at a gym server's URL. A single-node Toolathlon gym in
Docker is the smallest entry point; the larger recipes spread per-node servers across a Slurm
allocation.

Copy this for the mixture and curriculum machinery: a balanced sampler keeps a fast environment from
crowding out a slow one, and an environment can be staged in partway through a run. Full details are on
the [OpenReward integration](openreward.md) page, and [scaling out](../guides/scale.md) covers the
multi-node shape.

## openhands

A capability plugin. No tasks, no rollouts, no train scripts — it wraps an OpenHands SDK conversation
as a Platoon environment, adds a reasoning-safe context condenser, and supplies the pieces that make
delegation work: a `launch_subagent` tool bound to the live episode loop, a parallel task tracker, and
programmatic tool calling. Both `codegrep` and `openreward` depend on it.

It is never installed on its own; it arrives as a path dependency of whichever plugin needs it. See the
[OpenHands integration](openhands.md) page.

## See also

- [Write a plugin](../guides/first-plugin.md) — the scaffold, step by step.
- [Extend the framework](../guides/extend.md) — capability plugins and registry entry points.
- [Configuration](../reference/configuration.md) — the config keys these plugins share.
