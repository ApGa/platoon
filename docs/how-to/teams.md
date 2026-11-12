# Train a team of agents

The `platoon-harness` harness and episode program run and train a team of agents through
[platoon-harness](https://github.com/ApGa/platoon-harness). A team file names the roles, the harness each runs, such as
pi or OpenHands, and how they talk. Each rollout runs the team as one platoon-harness session whose agents call the
policy through platoon's model gateway, each on a platoon agent of its own, so every agent's calls land on a
trajectory of its own and train as any agent's do. It is experimental: a minor release may change it.

platoon-harness's own documentation covers the rest of a team:
[installing it](https://github.com/ApGa/platoon-harness/blob/main/docs/getting-started/install.md),
[the team file](https://github.com/ApGa/platoon-harness/blob/main/docs/reference/team-file.md) and
[training on sessions](https://github.com/ApGa/platoon-harness/blob/main/docs/extending/training.md).

## Install

```bash
pip install 'platoon[tinker,ors,platoon-harness]'
```

The `platoon-harness` extra installs platoon-harness's SDK, 0.1 (`platoon-harness>=0.1,<0.2`). Without it, the
harness and the program are still listed, and naming either fails with the install command.

Each host that runs rollouts also needs:

- `git`: each rollout's session works in a fresh repository with one empty commit.
- For a role on OpenHands, platoon-harness's `openhands` extra in the same environment as platoon, as each OpenHands
  agent runs on the rollout worker's own interpreter. Its pins are outside platoon's lock, so in a checkout build an
  environment of its own, which `uv sync` leaves alone, here with the TextCraft extension that the example below
  names, and run `platoon train` with its `platoon`:

  ```bash
  uv venv .venv-teams --python 3.12
  uv pip install --python .venv-teams -e '.[tinker,ors,platoon-harness]' -e 'extensions/textcraft[platoon]' \
    'platoon-harness[openhands]'
  ```

  platoon-harness's install guide says what else the extra brings.
- For a role on pi, pi 1.5.0 and its Node on `PATH`.

## Before you train

A rollout's session checks its host when it starts, which is after `platoon train` has started the trainer. So first
run platoon-harness's checks on each host that runs rollouts. Run its command line from a tool environment of its own,
never from platoon's, at the platoon-harness version that platoon's environment holds: a check made under another
version speaks for none of your rollouts.

```bash
version=$(python -c "import platoon_harness; print(platoon_harness.__version__)")   # platoon's environment's python
export PLATOON_HARNESS_HOME=<a local directory>   # as the rollouts will have it
uvx --from "platoon-harness[cli,openhands]==$version" platoon-harness doctor --check
uvx --from "platoon-harness[cli,openhands]==$version" platoon-harness team check --trainable crew.toml
```

`doctor --check` keeps the result that earns each harness trainable on the host. Run `team check` from the team
file's folder, and add `--trust` for a team that names a command or a `module:attr`. A rollout whose session cannot
start trainable is the rollout's `error`, never a score.

Each rollout's session, its record, stays as a folder under `PLATOON_HARNESS_HOME`, and nothing removes it. Keep that
root on a local directory of each rollout host.

## Write the config

A tree of OpenHands crafters on TextCraft-Synth: the team starts one crafter, which may hand parts of its task to
crafters of its own. A role's instructions name the environment's tools as its harness shows them,
`environment_<tool>` on OpenHands and `mcp__environment__<tool>` on pi.

```toml
# crew.toml
[roles.crafter]
harness = "openhands"
approvals = "allow"            # a trainable team leaves no approval to a person
may_spawn = ["crafter"]
stays = false                  # each crafter finishes after its task, rather than waiting for the next message
instructions = """\
Craft the items your task names with the environment's tools: environment_get_info gives an item's recipe, \
environment_view_inventory what you hold, and environment_craft makes an item from what you hold. You may hand a part \
of the crafting to a crafter of your own: spawn it with a goal of the form `Craft the following items: 2x stick, 1x \
oak_planks`, and it works on the same inventory and reports back when it is done. Call environment_finish once every \
item of your task is crafted."""

[topology.tree]
role = "crafter"
max_depth = 6
```

The entry of a training config beside `crew.toml`:

```yaml
environments:
  - env: textcraft-synth@1
    harness: platoon-harness
    program: platoon-harness
    harness_options:
      team: crew.toml              # or a trusted module:attr naming a Team
    agents: {max_depth: 6, budget: depth_aware, child_grading: env}
```

`platoon config validate` runs the program's load check, which names each problem it finds in the entry and its team.
Each crafter that another spawns runs as a platoon subagent, graded on its own targets.
[`recipes/teams`](../../recipes/teams/README.md) has whole configs.

## Options

- **`harness_options.team`** is a team file or a trusted `module:attr` naming a `Team`. A relative path is read beside
  the config. A file loads over the host's user config and the `platoon-harness.toml` beside it, with
  `PLATOON_HARNESS_CONFIG` over it, so keep those alike on every host.
- **`harness_options.rewards`** (optional) maps a role to a producer, `module:attr`, with what it runs with, such as
  `worker: {producer: "platoon_harness.credit:propagate", retained: 0.5}`. The program calls it once per rollout whose
  episode closed, as `produce(session, "task", *, judge, **params)`. It writes one row on each agent's first start and
  on each span, under `producer=` the exact `module:attr` the entry names. An agent of the role takes its reward from
  the row on its first start, and the role's agents share a baseline. Its calls train only inside a span with a scored
  row of its own, so a producer that writes rows only on agents' starts trains none of the role's calls
  (`credit/keyed/unscored_tokens` counts them). Such an entry trains under `credit: keyed`, or another credit that
  reads spans; a run that trains several policies credits an entry that names none with keyed. In a run of one
  policy, an entry that names none passes `platoon config validate` but stops the run at its first training group,
  so name `credit: keyed` there.
- **The root.** A team that starts one agent runs it on the root's own endpoint, and it trains on the environment's
  grade. A team that starts several runs under a root that makes no model call, and mints every agent.
- **The step budget** is the gateway's: each agent's `policy` calls past its `max_steps` get a stop, are never
  sampled, and grade its run `limit`, with reason `budget`. A run with no `max_steps` is refused. Agents minted under
  the root share its `max_steps`, so set it for the whole team. A compaction is an `aux` call, outside the budget,
  when its harness sends the compaction headers; without them the gateway counts it as a `policy` call.

The entry's other settings apply as for any harness: `reward_processor` (the default or `platoon-delegation`, which
read a reward where a producer's row lands), `agents`, `on_harness_exit` and `on_context_overflow`, and the rollout
config's `skip_subagent_reward_computation` and `propagate_root_success`.

## What trains

A step trains only when its call is one its agent's harness reported once, and an `aux` step only when that call was
a compaction, so no call of unknown purpose trains. platoon-harness says which agents and calls may train. Excluded
are an agent whose trainable level was not met, one that shared its key, one whose harness started an agent of its
own on its key, one that never ended, and a call a person made or held. Each trajectory's `misc["team"]` names its
session's folder, its agent and the row it trains on.

## Read a rollout

`platoon viz` shows each step's tool results and the inputs its harness took in. platoon-harness's command line opens
the session itself, where `misc["team"]["session"]` is `<root>/sessions/<name>`:

```bash
export PLATOON_HARNESS_HOME=<root>
uvx --from "platoon-harness[cli]==$version" platoon-harness attach <name>   # the terminal UI
uvx --from "platoon-harness[cli]==$version" platoon-harness web <name>      # in the browser
```

The session holds no key, and its view link acts on nothing.

## Limits

- A role trains only on a harness that launches sealed and has a conformance result kept on the host: pi, OpenHands,
  or a wire program that declares a sealed launch. Codex, Claude Code, OpenCode, an ACP agent and one-shot commands
  are refused when the session starts.
- A trainable agent's harness runs with none of its own tools: only platoon-harness's team tools and the
  environment's tools, such as Harbor's `bash`.
- A role on `harness = "openhands"` runs OpenHands through platoon-harness, not through platoon's own `openhands`
  harness.
- `harness: platoon-harness` runs only under `program: platoon-harness`: under another program, every rollout ends
  `error`.
- Roles work in `shared`, or in `isolated` where the team names a confining extension, such as
  `extensions = ["bwrap"]`, which needs bubblewrap and unprivileged user namespaces on each rollout host.
- A tree whose agents wait for each child they spawn is credited and budgeted as platoon's own recursive agent is
  under the same settings, though its prompts differ. No training result with it is published yet.
