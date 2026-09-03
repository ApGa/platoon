# Plugins

Everything Platoon runs comes from a plugin. The core supplies the episode loop, the trajectory
tree, the registry and the training backends; a plugin supplies the thing being trained on and, when
you need it, new machinery for the framework itself.

A plugin is an ordinary Python package. It installs one subpackage into the `platoon` namespace and
advertises its components by name.

## Two kinds

**A task plugin** packages a task or environment together with the rollout program that runs it: how
a problem instance is generated, what actions the agent may take, how the outcome is scored, and how
one episode is played out. This is the common case, and the one
[Build your first task](../guides/first-plugin.md) walks through.

<span class="pl-src">plugins/textcraft</span> is a compact example — a crafting environment, an
agent, a rollout, a `registry.py` naming all of it, and configs for both backends.

**A capability plugin** adds framework functionality rather than a task. An agent harness, an
integration with an environment service, a new workflow — things other plugins then build on.

<span class="pl-src">plugins/openhands</span> is one: it wraps the OpenHands agent SDK as a Platoon
agent, with its own condenser and a recursive variant that lets an agent delegate to sub-agents. It
ships no tasks at all. Other plugins depend on it and supply the work.
<span class="pl-src">plugins/openreward</span> is another: it speaks to an environment server, so
the tasks arrive from a service rather than from the package.

The kinds are not exclusive — a package can register a task *and* a capability — but knowing which
one you are writing tells you what to build first.

## Your plugin can live anywhere

Nothing about the plugin mechanism assumes the Platoon repository. A plugin is a normal distribution
with a normal `pyproject.toml`. Keep your research or training project in your own repo, depend on
`platoon`, install the two side by side, and Platoon finds your components.

No fork. No branch to rebase. No upstreaming. If you later want your work in the tree,
[contributions](../contributing.md) are welcome — but the framework never requires it, and the
plugins that ship here are built exactly the way yours will be.

The `plugins/` directory in this repository is just the set of plugins that happen to live here.

## Layout

```text
mytask/
├── pyproject.toml
└── platoon/                  # namespace shim — no __init__.py here
    └── mytask/
        ├── __init__.py
        ├── tasks.py          # task ids and instances
        ├── env.py            # actions and reward
        ├── agent.py          # the agent that plays it
        ├── rollout.py        # one episode
        ├── registry.py       # the names your config uses
        └── configs/
            ├── areal/mytask_areal.yaml
            └── tinker/mytask_tinker.yaml
```

Three names are in play and they may differ: the directory (`mytask`, hyphens allowed), the
distribution (`platoon-mytask`), and the import path (`platoon.mytask`, underscores only).

Core Platoon and every plugin ship a top-level `platoon/` directory, and they merge into one
importable package because the core `__init__.py` calls `extend_path`. That is why
`platoon.registry` and `platoon.mytask` resolve together while living in different checkouts.

!!! warning "No `__init__.py` in the plugin's `platoon/` directory"
    The plugin's `platoon/` contributes contents, not an identity. An `__init__.py` at that level
    stops the merge. The one a level deeper — `platoon/mytask/__init__.py` — is required, and may
    be empty.

File names are a convention, not a rule: the registry resolves dotted import paths, so it does not
care where a function lives. Following the layout above makes your plugin legible to anyone who has
read another one.

## Discovery

Your components become available when a module containing `@register_*` calls is imported.
Conventionally that module is `platoon.<name>.registry`. There are two ways to get it imported.

**Name it in the config.** One environment, one import:

```yaml
environments:
  - package: platoon.mytask.registry
    dataset_loader: mytask/default
    task_loader: mytask/default
    rollout: mytask/default
    reward_processor: mytask/success
    workflow: group_rollout
```

**Or advertise an entry point** and let Platoon find every installed plugin at once:

```toml
[project.entry-points."platoon.plugins"]
mytask = "platoon.mytask.registry"
```

`discover_entry_points` in <span class="pl-src">platoon/registry.py</span> loads every entry point in
the `platoon.plugins` group. It runs when a config sets `discover_entry_points: true` on the first
environment. Write the value as a bare module path with no `:attr` suffix — the registrations happen
as a side effect of the import. The key on the left is only a label.

Use `package` for a single-environment run; turn on discovery when several installed plugins must
register together. Registered names are global, so namespace yours as `<plugin>/<variant>` to avoid
collisions. The full set of `environments:` keys is in the
[configuration reference](../reference/configuration.md).

## A minimal `pyproject.toml`

```toml title="mytask/pyproject.toml"
[project]
name = "platoon-mytask"
version = "0.1.0"
requires-python = "~=3.12.0"
dependencies = ["platoon >= 0.1.0"]

[project.entry-points."platoon.plugins"]
mytask = "platoon.mytask.registry"

[project.optional-dependencies]
tinker = ["platoon[tinker]"]
areal = ["platoon[areal]"]

[tool.uv]
conflicts = [[{ extra = "tinker" }, { extra = "areal" }]]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["platoon"]
```

Two lines carry weight. `packages = ["platoon"]` is what makes the namespace layout build —
hatchling would otherwise look for a `platoon_mytask` directory and find nothing. The `conflicts`
block lets one lockfile hold both backend resolutions, since AReaL and Tinker pull incompatible
torch builds and cannot be installed together.

`requires-python = "~=3.12.0"` matches the root project and every plugin here. Add your own runtime
dependencies to `dependencies`, and pin git or local sources under `[tool.uv.sources]` rather than
in `[project]`.

Then install and check that both halves of the namespace resolve:

```bash
uv sync --extra tinker
uv run python -c "import platoon.registry, platoon.mytask.registry; print('ok')"
```

[Build your first task](../guides/first-plugin.md) does this end to end, with the code for each
file. [Extend Platoon](../guides/extend.md) covers the extension points themselves.

## Next

<div class="pl-cards" markdown>

<div class="pl-card" markdown>
<span class="pl-card__kicker">What ships here</span>
### [Catalog](catalog.md)
Every plugin in the repository, what it trains on, and which backend it is configured for.
</div>

<div class="pl-card" markdown>
<span class="pl-card__kicker">Capability plugin</span>
### [OpenHands](openhands.md)
Run the OpenHands software-engineering agent inside a Platoon episode, delegation included, and
train it.
</div>

<div class="pl-card" markdown>
<span class="pl-card__kicker">Environments as a service</span>
### [OpenReward](openreward.md)
Train against hosted task environments with an outcome verifier, a behavior judge and weighted task
mixtures.
</div>

</div>
