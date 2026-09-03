# Installation

Platoon is a repository of [`uv`](https://docs.astral.sh/uv/) projects: a root project holding the
framework, and one independent project per plugin under `plugins/`. Install the framework, pick a
training backend, then install the plugin you want to work with.

## Prerequisites

| | Requirement |
| --- | --- |
| **Python** | 3.12.x — every project declares `requires-python = "~=3.12.0"` |
| **Package manager** | `uv`. The AReaL extra resolves from a git revision, and the conflict rules and dependency overrides are `[tool.uv]` keys that pip ignores |
| **Hardware for AReaL training** | Linux x86_64 with NVIDIA GPUs |
| **Hardware for Tinker training** | None locally — training runs on a remote Tinker-compatible backend |

`uv python install 3.12` gets you a suitable interpreter.

!!! note "You do not need a GPU to develop"

    Environments, agents, rollout functions, reward processors and the trajectory viewer all run
    against an OpenAI-compatible endpoint. Build and debug your task that way first — see
    [Evaluate a model](../guides/evaluate.md).

## Install the framework

From the repository root:

```bash
uv sync
```

This installs the core dependencies plus the `dev` group (`mypy`, `pre-commit`, `pytest`,
`pytest-asyncio`, `ruff`). It installs no training backend and no torch; the test suite runs
without either.

## Choose one training backend

The framework ships two backends as optional extras.

=== "AReaL"

    ```bash
    uv sync --extra areal
    ```

    Pulls `areal[cuda]` from a pinned git revision, plus classic FlashAttention-2 on Linux x86_64.
    Training runs locally on your GPUs. FlashAttention is built from source against the venv's
    torch, so expect the first sync to take a while.

=== "Tinker"

    ```bash
    uv sync --extra tinker
    ```

    Pulls the `tinker` SDK and `tinker-cookbook`. Training runs on a Tinker-compatible backend
    reached over the network; your process only builds batches and submits them.

The two extras are declared a `uv` conflict group because their dependency graphs resolve
incompatible versions of torch and of the surrounding CUDA stack. So `uv sync --extra areal --extra
tinker` is rejected, and switching backends means re-syncing rather than adding. The choice has
consequences beyond installation — see [Backends](../architecture/backends.md).

!!! note "Use uv, not pip"

    AReaL is not on PyPI; it comes from a `[tool.uv.sources]` git pin that pip does not read.
    `uv pip install -e ".[areal]"` works if you prefer the pip-style interface.

## Install a plugin

Each plugin is its own uv project, with its own lockfile and its own virtual environment. Install
one by changing into it:

```bash
cd plugins/number-search
uv sync --extra tinker        # or: uv sync --extra areal
```

Every command for that plugin then runs from that directory with `uv run`. The plugin depends on
the framework through an editable path source, and `platoon` is a namespace package, so
`import platoon.number_search` and `import platoon.train.tinker` both resolve in the same
interpreter even though they live in separate projects on disk.

One project per plugin means plugins with incompatible dependency stacks stay out of each other's
way — and it means **a plugin does not have to live in this repository**. Both kinds of plugin (a
task plugin, or a capability plugin adding framework functionality) are ordinary Python packages:
keep your research project in your own repo, depend on `platoon`, and make it discoverable without
a fork. See [Write your first plugin](../guides/first-plugin.md).

The exception is `plugins/openhands`, a library wrapper consumed by other plugins rather than
something you train from; it declares no backend extras. Two plugins need a data step after
`uv sync`: `appworld` and `email-search`. The [plugin catalog](../plugins/catalog.md) has the
details for each.

## Credentials

Platoon itself reads few environment variables.

| Variable | Needed for |
| --- | --- |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL` | Any inference, LLM-judge or rubric path. `LLMClient` raises if given neither these nor explicit arguments |
| `WANDB_API_KEY`, `WANDB_BASE_URL` | Weights & Biases logging. Platoon sets them from the `stats_logger.wandb` config keys when those are provided |

The Tinker SDK reads its own service credentials from the environment; the Platoon-side knob is the
`tinker_base_url` config key. `HF_TOKEN` is consumed directly by `huggingface_hub` for gated models
and datasets, and litellm reads its own variables.

Plugins add their own — `TAVILY_API_KEY` for `deepdive`, `APPWORLD_ROOT` for `appworld`, the
`OPENREWARD_*` variables that point at an [OpenReward](../plugins/openreward.md) environment
server. Each plugin page lists what it needs.

!!! warning "Never commit credentials"

    Supply keys from the submitting environment rather than writing them into a config file or a
    launcher script.

??? note "Megatron and Transformer Engine"

    Skip this unless you train with the Megatron actor backend. The FSDP backend and every
    inference workflow work immediately after `uv sync --extra areal`.

    Transformer Engine is excluded from the lockfiles: its torch bindings are source-only and need
    a CUDA toolkit that compute nodes often lack, and forcing it into the resolution graph would
    break `uv sync` for every backend. Platoon imports the Megatron actor lazily, so FSDP runs are
    unaffected.

    For Megatron runs, provide Transformer Engine through your training container or base image, or
    build it into the plugin venv on a machine with `nvcc`, following
    [NVIDIA's installation guide](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html).
    Megatron's fused gradient accumulation additionally wants APEX, built the same way. Then select
    the backend per run with `actor.backend: megatron`. [Run at scale](../guides/scale.md) covers
    the multi-node picture.

## Verify

From the repository root:

```bash
# The framework imports and the namespace package resolves.
uv run python -c "import platoon; print(platoon.__path__)"

# The test suite — what CI runs, with no backend extra installed.
uv run pytest tests/ -v
```

Then the backend:

=== "AReaL"

    ```bash
    uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
    uv run python -c "import areal, platoon.train.areal; print('areal backend ok')"
    ```

=== "Tinker"

    ```bash
    uv run python -c "import tinker, platoon.train.tinker; print('tinker backend ok')"
    ```

And inside a plugin:

```bash
cd plugins/number-search
uv run python -c "import platoon.number_search.env; print('plugin ok')"
```

!!! tip "One venv per project"

    With an editable root and a venv per plugin, a `VIRTUAL_ENV` or `UV_PROJECT_ENVIRONMENT`
    inherited from another project silently wins. Unset both if you shell-hop between plugin
    directories.

## Next

- [Quickstart](quickstart.md) — run `number-search` end to end.
- [Core concepts](concepts.md) — tasks, environments, agents and the rollout loop.
- [Backends](../architecture/backends.md) — AReaL versus Tinker, and what changes if you switch.
- [Common questions](../reference/faq.md).
