# Your first custom task

This page builds a complete, working Platoon plugin from nothing — a tiny word-unscrambling task
with a checkable reward — and trains on it. It is the fast route: seven short files, each one
mirroring the shape of the real `number-search` plugin in the repository. If you want the same
journey against a harder task — multi-step episodes, hidden information, partial credit — with the
reasoning spelled out at length, read [build a plugin](../tutorials/build-a-plugin.md) instead; this
page assumes you would rather read code than prose.

You need [Platoon installed](installation.md) and enough familiarity with
[the core concepts](concepts.md) to know what a task, an environment, an agent and a rollout are.

## The task

The agent is shown a scrambled word and must call `answer("...")` with the unscrambled form. The
reward is `1.0` when the answer is right and `0.0` otherwise. This is deliberately trivial: it is
the smallest thing that still exercises every extension point, so you can get a run moving quickly
and then swap the task for the one you actually care about.

Seven files, in dependency order:

| File | What it supplies |
| --- | --- |
| `pyproject.toml` | The distribution, so `platoon.unscramble` becomes importable |
| `tasks.py` | Task ids and `get_task(task_id) -> Task` |
| `env.py` | The action space and `evaluate() -> tuple[float, dict]` |
| `agent.py` | A system prompt that tells the model what actions exist |
| `rollout.py` | `async run_rollout(task, config)` — one episode, recorded |
| `registry.py` | Names for the components so YAML can select them |
| `unscramble_tinker.yaml` | The `environments:` block plus a trainer config |

## Layout

```text
plugins/unscramble/
├── pyproject.toml
└── platoon/                        # NO __init__.py in this directory
    └── unscramble/
        ├── __init__.py             # may be empty
        ├── tasks.py
        ├── env.py
        ├── agent.py
        ├── rollout.py
        ├── registry.py
        └── unscramble_tinker.yaml
```

!!! warning "Do not add `plugins/unscramble/platoon/__init__.py`"
    Core Platoon makes `platoon` a namespace package with
    `__path__ = extend_path(__path__, __name__)`
    (<span class="pl-src">platoon/__init__.py</span>). That is what lets your
    `platoon/unscramble/` directory merge into the same package as the core `platoon/` directory
    living in a different repository. An `__init__.py` in the plugin's `platoon/` directory fights
    that mechanism. No plugin in the repository has one.

Note the three different names: the directory is `plugins/unscramble` (hyphens are allowed), the
importable module is `platoon.unscramble` (underscores only), and the distribution is
`platoon-unscramble`.

## 1. `pyproject.toml`

Copy `plugins/number-search/pyproject.toml` and change the name and description. The trimmed
version below shows the load-bearing parts.

```toml title="plugins/unscramble/pyproject.toml"
[project]
name = "platoon-unscramble"
version = "0.1.0"
description = "Platoon plugin for the unscramble task."
requires-python = "~=3.12.0"
dependencies = [
    "platoon >= 0.1.0",
]

[project.optional-dependencies]
tinker = ["platoon[tinker]"]
areal = ["platoon[areal]"]

[tool.uv]
conflicts = [[{ extra = "tinker" }, { extra = "areal" }]]
override-dependencies = [ ... ]   # copy verbatim from plugins/number-search/pyproject.toml
no-build-isolation-package = ["flash-attn", "causal-conv1d", "mamba-ssm"]

[tool.uv.sources]
platoon = { path = "../..", editable = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["platoon"]
```

The `override-dependencies` list must be copied verbatim, not elided: `uv` only honors overrides
declared by the *root* project, and every plugin is its own root. The comment at
<span class="pl-src">plugins/number-search/pyproject.toml</span> says exactly this. The
`tinker` and `areal` extras are declared as conflicting, so a given checkout is installed against
one backend or the other, never both.

## 2. `tasks.py` — task ids and `get_task`

The contract the trainer needs is one function: `(task_id: str) -> Task`
(<span class="pl-src">platoon/train/components.py</span>). `Task` is a plain dataclass with
`goal`, `id`, `max_steps`, `misc` and `fork_strategy`
(<span class="pl-src">platoon/envs/base.py</span>); everything task-specific goes in `misc`,
which the environment reads.

Real plugins generate a dataset offline and ship it as JSONL next to the code. This task is cheap
enough to synthesize on demand from the id, which keeps the example to one file.

```python title="plugins/unscramble/platoon/unscramble/tasks.py"
import random
from typing import Literal

from platoon.envs.base import Task

TRAIN_WORDS = ["anchor", "basket", "candle", "dolphin", "engine", "forest", "granite", "harbor"]
VAL_WORDS = ["island", "jungle", "kettle", "lantern"]

TASKS: dict[str, Task] = {}


def get_task_ids(
    split: Literal["train", "val"],
    num_samples_train: int = 2000,
    num_samples_val: int = 200,
) -> list[str]:
    if split == "train":
        return [f"unscramble.train.{i}" for i in range(num_samples_train)]
    if split == "val":
        return [f"unscramble.val.{i}" for i in range(num_samples_val)]
    raise ValueError(f"Invalid split: {split}")


def _scramble(word: str, rng: random.Random) -> str:
    letters = list(word)
    for _ in range(20):
        rng.shuffle(letters)
        if "".join(letters) != word:
            break
    return "".join(letters)


def get_task(task_id: str) -> Task:
    if task_id in TASKS:
        return TASKS[task_id]
    _, split, index = task_id.split(".")
    words = TRAIN_WORDS if split == "train" else VAL_WORDS
    word = words[int(index) % len(words)]
    scrambled = _scramble(word, random.Random(task_id))
    task = Task(
        goal=f"Unscramble the letters '{scrambled}' into a single English word.",
        id=task_id,
        max_steps=3,
        misc={"word": word, "scrambled": scrambled},
    )
    TASKS[task_id] = task
    return task
```

Seeding `random.Random(task_id)` with the id makes the scramble reproducible across processes,
which matters because both backends re-import and re-run your task loader in every rollout worker.
Train and validation words are disjoint so the eval split measures generalization rather than
recall — `number-search` achieves the same thing by hashing the `(low, target, high)` triplet
(<span class="pl-src">plugins/number-search/platoon/number_search/tasks.py</span>).

!!! warning "The workflow mutates the task you return"
    Both backends overwrite `task.max_steps` with `rollout_config.max_steps` when that key is set
    (<span class="pl-src">platoon/train/areal/workflows/group_rollout_workflow.py</span>,
    <span class="pl-src">platoon/train/tinker/workflows/group_rollout_workflow.py</span>).
    Since `TASKS` hands back the same object every time, the `max_steps=3` above is a default the
    config wins over. Do not cache anything the environment mutates during an episode.

## 3. `env.py` — the action space and the reward

A Platoon environment is anything satisfying the `Env` protocol
(<span class="pl-src">platoon/envs/base.py</span>), but for a task the model can solve by
writing Python you subclass `CodeActEnv` and inject callables into an IPython namespace. Its
constructor is `(task, code_executor, return_obs_copy=True, parent_state=None, **kwargs)`
(<span class="pl-src">platoon/envs/codeact/env.py</span>) and `IPythonCodeExecutor` is
`(task, actions=(finish, safe_asyncio), detect_unawaited_async_calls=True,
detect_while_loops=False, detect_interactive_input=False)`
(<span class="pl-src">platoon/envs/codeact/env.py</span>). Each callable in `actions` is
bound into the shell under its own `__name__`.

```python title="plugins/unscramble/platoon/unscramble/env.py"
from platoon.agents.actions.common import finish
from platoon.envs.base import Task
from platoon.envs.codeact import CodeActEnv, IPythonCodeExecutor
from platoon.episode.context import finish_message


def answer_factory(word: str):
    def answer(guess: str) -> str:
        """Submit a guess for the unscrambled word."""
        if guess.strip().lower() == word:
            finish_message.set(f"You unscrambled {word} correctly!")
            return "Correct."
        return f"'{guess}' is not the word. Try again."

    return answer


class UnscrambleEnv(CodeActEnv):
    def __init__(self, task: Task):
        super().__init__(task, IPythonCodeExecutor(task, actions=(finish, answer_factory(task.misc["word"]))))

    async def evaluate(self) -> tuple[float, dict]:
        if self._state.finished:
            message = finish_message.get(None)
            if message is not None and "correctly" in message:
                return 1.0, {"reward/success": 1.0}
            return 0.0, {"reward/success": 0.0}
        return 0.0, {}
```

This is `plugins/number-search/platoon/number_search/env.py` with the action swapped. Four things
about it are load-bearing:

- **The closure carries the answer.** `answer_factory` captures `word`, so the correct answer never
  appears in the shell namespace where model-authored code could read it. `guess_factory` in
  number-search does the same thing for the same reason.
- **`finish_message` is what ends the episode.** Setting that contextvar makes `CodeActEnv.step`
  mark the state finished (<span class="pl-src">platoon/envs/codeact/env.py</span>), which
  makes the episode loop halt. The built-in `finish` action sets it too, which is why an agent that
  gives up still terminates cleanly. The reward check keys off the *content* of that message, so
  `finish("I give up")` scores zero.
- **`evaluate()` runs on every step**, not only the last
  (<span class="pl-src">platoon/envs/codeact/env.py</span>). Its float is attached to the step
  and accumulated onto the trajectory, so a per-step reward must be a genuine increment, not a
  running total. Its dict lands in `step.misc["reward_misc"]`.
- **Keys prefixed `reward/` are the aggregation convention.** Nothing enforces it, but reward
  processors in the repository sum exactly those keys across steps
  (<span class="pl-src">plugins/textcraft/platoon/textcraft/registry.py</span>). Emit them
  even if you do not write a reward processor yet.

!!! note "The model only knows the actions your prompt describes"
    `IPythonCodeExecutor.describe_action_space()` returns the empty string
    (<span class="pl-src">platoon/envs/codeact/env.py</span>). Injecting `answer` into the
    shell does not tell the model it exists — the system prompt in the next file is the only
    channel. If you add an action and forget the prompt, the model will never call it.

## 4. `agent.py` — a system prompt

For a CodeAct task the agent is usually only a prompt. Subclass `CodeActPromptBuilder` to override
`build_system_prompt`, then subclass `CodeActAgent` to install your builder by default.
`CodeActAgent.__init__` is `(prompt_builder=None, prompt_mode="sequence_extension",
include_reasoning=True, llm_client=None, inference_params=None, stuck_in_loop_threshold=4,
stuck_in_loop_window=3)` (<span class="pl-src">platoon/agents/codeact/agent.py</span>).

```python title="plugins/unscramble/platoon/unscramble/agent.py"
from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder, PromptMode
from platoon.envs.codeact import CodeActObservation


class UnscramblePromptBuilder(CodeActPromptBuilder):
    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        return """Solve step by step. Put thoughts in <thought> </thought> and code in <python> </python>.
Your answer must call answer(guess: str) with a single lowercase English word.

Example:
<thought>
thought process here
</thought>
<python>
answer("anchor")
</python>
"""


class UnscrambleAgent(CodeActAgent):
    def __init__(
        self,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
        **kwargs,
    ):
        if "prompt_builder" not in kwargs:
            kwargs["prompt_builder"] = UnscramblePromptBuilder(
                prompt_mode=prompt_mode,
                include_reasoning=include_reasoning,
            )
        super().__init__(prompt_mode=prompt_mode, include_reasoning=include_reasoning, **kwargs)
```

Overriding `build_system_prompt` replaces the default Jinja template at
`platoon/agents/codeact/prompts/system.jinja` entirely, so your string must carry the tag-format
instructions itself — the parser looks for `<python>` blocks and, when `include_reasoning` is true,
`<thought>` blocks (<span class="pl-src">platoon/agents/codeact/agent.py</span>). The
number-search builder branches on `include_reasoning` and returns a prompt without the `<thought>`
instructions when it is false; the version above ignores the flag, which is fine as long as you
leave the default alone. `build_user_prompt`, `build_next_action_str` and
`build_action_history_description` are the other override points
(<span class="pl-src">platoon/agents/codeact/prompt_builder.py</span>).

## 5. `rollout.py` — one episode, recorded

This is the function both trainers call. It receives exactly two positional arguments, a `Task` and
a `RolloutConfig` (<span class="pl-src">platoon/config_defs.py</span>), and returns the
serialized trajectory collection. Copy the structure below exactly; every line of it is doing
something.

```python title="plugins/unscramble/platoon/unscramble/rollout.py"
import asyncio
import os
from contextlib import suppress
from logging import getLogger

from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.episode.context import current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import TrajectoryCollection
from platoon.utils.llm_client import LiteLLMClient
from platoon.visualization.event_sinks import JsonlFileSink

from .agent import UnscrambleAgent
from .env import UnscrambleEnv

logger = getLogger("platoon.unscramble.rollout")


async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = None
    try:
        llm_client = LiteLLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
        )
        env = UnscrambleEnv(task)
        agent = UnscrambleAgent(llm_client=llm_client, inference_params=config.inference_params)

        traj_collection = TrajectoryCollection()
        current_trajectory_collection.set(traj_collection)

        events_path = os.path.join(config.output_dir, "events", f"events_{task.id}_{traj_collection.id}.jsonl")
        traj_collection.register_event_handlers(
            JsonlFileSink(events_path, collection_id=traj_collection.id, process_id=os.getpid())
        )

        rollout_task = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))
        try:
            _ = await asyncio.wait_for(rollout_task, timeout=config.timeout)
        except asyncio.TimeoutError:
            rollout_task.cancel()
            with suppress(asyncio.CancelledError):
                await rollout_task
            raise

        if config.return_dict:
            return current_trajectory_collection.get().to_dict()
        return current_trajectory_collection.get()
    finally:
        if agent is not None:
            await agent.close()
        if env is not None:
            await env.close()
```

The invariants, all visible in
`plugins/number-search/platoon/number_search/rollout.py`:

- **Fresh `TrajectoryCollection` per rollout, set on the contextvar before `run_episode`.**
  `CodeActEnv.reset` reads `current_trajectory_collection` to register the task
  (<span class="pl-src">platoon/envs/codeact/env.py</span>); without it the rollout raises.
- **`run_episode` goes inside `asyncio.create_task`.** The comment at
  <span class="pl-src">platoon/episode/loop.py</span> explains why: it stops the episode's
  contextvar writes from leaking into the calling context, which is what keeps concurrent rollouts
  and nested subagents from stepping on each other.
- **Two different timeouts.** `run_episode(..., timeout=...)` is the *per-step* deadline applied to
  each `agent.act` and `env.step` (<span class="pl-src">platoon/episode/loop.py</span>); the
  outer `asyncio.wait_for` is the whole-trajectory deadline. `RolloutConfig.step_timeout` defaults
  to `300` and `timeout` defaults to `None`.
- **Return the dict when asked.** Both workflows force `config.return_dict = True` before calling
  you, so the dict branch is the one that runs during training.
- **Close both resources in `finally`.**

!!! warning "The rollout must be an importable module-level `async` function"
    Both workflows `await` its result, so a synchronous function fails at rollout time. On the
    AReaL path the workflow is shipped to worker processes as a dotted *import path*, not a pickle,
    and raises `ValueError("GroupRolloutWorkflow requires importable rollout_fn/get_task_fn")` when
    the callable has none
    (<span class="pl-src">platoon/train/areal/workflows/group_rollout_workflow.py</span>).
    Do not register a lambda, a closure, or a `functools.partial`. The same restriction applies to
    the task loader, which is additionally called *synchronously* — do not make it `async`.

## 6. `registry.py` — names for the config

The registry is a process-local name-to-object map with one decorator per component kind
(<span class="pl-src">platoon/registry.py</span>). Importing this module is what populates
it; the config names the module in `environments[0].package`, and Platoon imports it for the side
effects (<span class="pl-src">platoon/train/auto.py</span>).

```python title="plugins/unscramble/platoon/unscramble/registry.py"
"""Registered unscramble components for the shared Platoon trainers."""

from __future__ import annotations

from typing import Any

from platoon.registry import register_dataset_loader, register_rollout, register_task_loader

from platoon.unscramble.rollout import run_rollout
from platoon.unscramble.tasks import get_task, get_task_ids


@register_task_loader("unscramble/default")
def load_unscramble_task(task_id: str):
    return get_task(task_id)


@register_dataset_loader("unscramble/default")
def load_unscramble_dataset(
    config: Any,
    split: str,
    limit: int | None = None,
    num_samples_train: int = 2000,
    num_samples_val: int = 200,
):
    split_name = "val" if split == "eval" else split
    task_ids = get_task_ids(split_name, num_samples_train, num_samples_val)
    if limit is not None:
        task_ids = task_ids[:limit]
    return task_ids


register_rollout("unscramble/default", run_rollout)
```

Three details worth knowing:

- **`split` is the literal string `"train"` or `"eval"`**, never `"val"`
  (<span class="pl-src">platoon/train/tinker/train.py</span>). Translate it yourself, as the
  first line of the loader does; TextCraft does exactly the same
  (<span class="pl-src">plugins/textcraft/platoon/textcraft/registry.py</span>).
- **The dataset loader ignores `config`.** The whole trainer config object is passed as the first
  argument, but taking everything from `dataset_kwargs` instead keeps the loader independent of
  which backend config class is in play. Returning a plain `list[str]` is enough — `AutoDataset`
  converts it into `Dataset.from_list([{"task_id": ...}, ...])`
  (<span class="pl-src">platoon/train/components.py</span>).
- **`register_rollout` is called directly, not as a decorator.** Passing a value as the second
  argument registers immediately instead of returning a decorator
  (<span class="pl-src">platoon/registry.py</span>); that is the natural form when the
  function is defined in another module.

!!! tip "Namespace your names"
    Registering a name that already exists in the same kind raises `ValueError`
    (<span class="pl-src">platoon/registry.py</span>). The `"<plugin>/<variant>"` convention
    that textcraft uses is not enforced, but it is what keeps two installed plugins from colliding
    at import time.

You can skip this file entirely. Any spec string that is not a registered name is treated as a
dotted import path and imported (<span class="pl-src">platoon/registry.py</span>), so
`rollout: platoon.unscramble.rollout.run_rollout` works with no registry module at all.
Registering buys you short stable names, decoupled from your module layout, and a list of the valid
alternatives in the error message when you typo one.

## 7. The config

The top-level `environments:` key is a list of exactly one `EnvironmentConfig`
(<span class="pl-src">platoon/train/components.py</span>) — more than one entry raises
`NotImplementedError`. It is what makes the shared trainers environment-agnostic: a new task is a
YAML block, not a new training script.

!!! warning "Two unrelated keys named `environments`"
    This is the *top-level* `environments:`. The `openreward` plugin has its own, entirely separate
    `environments:` list nested under its `openreward:` config section, describing a mixture of
    task sources with sampling weights. They share nothing but the name.

=== "Tinker"

    ```yaml title="plugins/unscramble/platoon/unscramble/unscramble_tinker.yaml"
    environments:
      - package: platoon.unscramble.registry
        dataset_loader: unscramble/default
        eval_dataset_loader: unscramble/default
        task_loader: unscramble/default
        rollout: unscramble/default
        workflow: group_rollout
        dataset_kwargs:
          num_samples_train: 2000
        eval_dataset_kwargs:
          num_samples_val: 200
          limit: 100

    train:
      model_name: Qwen/Qwen3-4B-Instruct-2507
      renderer_name: qwen3_instruct
      batch_size: 32
      num_epochs: 1
      lora_rank: 32
      loss_fn: cispo
      loss_fn_config:
        clip_low_threshold: 0.0
        clip_high_threshold: 5.0
      workflow_config:
        group_size: 8
        rollout_config:
          max_steps: 3
          output_dir: ./rollout_results
          verbose: true
          timeout: 300

    eval:
      strategy: step
      every: 10
      workflow_config:
        group_size: 1
        rollout_config:
          max_steps: 3
          output_dir: ./eval_results
          verbose: false
          timeout: 300

    log_path: ./logs
    ```

    `train`, `eval` and `log_path` have no defaults on `PlatoonTinkerRLTrainerConfig`
    (<span class="pl-src">platoon/train/tinker/config_defs.py</span>), and `model_name` and
    `renderer_name` have none on `TrainConfig`
    (<span class="pl-src">platoon/train/tinker/config_defs.py</span>), so all five are
    mandatory. Everything else above is either a default restated for clarity or a value borrowed
    from `plugins/number-search/platoon/number_search/number_search_tinker.yaml`.

=== "AReaL"

    The `environments:` block is identical — `PlatoonArealRLTrainerConfig` carries the same field
    (<span class="pl-src">platoon/train/areal/config_defs.py</span>) and
    `python -m platoon.train.areal.train` consumes it through the same `Auto*` factories.

    ```yaml
    environments:
      - package: platoon.unscramble.registry
        dataset_loader: unscramble/default
        eval_dataset_loader: unscramble/default
        task_loader: unscramble/default
        rollout: unscramble/default
        workflow: group_rollout
        dataset_kwargs:
          num_samples_train: 2000
        eval_dataset_kwargs:
          num_samples_val: 200
          limit: 100
    ```

    The rest of an AReaL config is not something you can shorten: `rollout.backend` and
    `actor.backend` both raise when unset
    (<span class="pl-src">platoon/train/areal/config_defs.py</span>), and the cluster,
    scheduler and SGLang blocks all need real values for your hardware. Start by copying
    `plugins/number-search/platoon/number_search/nv_number_search_cispo_areal.yaml`, fix
    `cluster.fileroot` and `experiment_name`, and paste the block above at the top. AReaL configs
    also support `${...}` interpolation, which the Tinker loader does not.

    !!! warning "The registry path is exercised end to end only on Tinker today"
        Every AReaL config in the repository still runs through a per-plugin `train_*.py` script;
        the one AReaL `environments:` block that exists is commented out
        (<span class="pl-src">plugins/textcraft/platoon/textcraft/configs/areal/textcraft_synth_ctx40000_depth_aware_medium_areal.yaml</span>).
        `python -m platoon.train.areal.train` is real code that reads the block, but you would be
        the early adopter. The per-plugin script route documented in
        [build a plugin](../tutorials/build-a-plugin.md) is the well-trodden AReaL path.

The keys you did not set matter too:

| Key | Default | What happens |
| --- | --- | --- |
| `reward_processor` | `None` | Falls back to `lambda traj: (traj["reward"], {})` (<span class="pl-src">platoon/train/auto.py</span>), which is exactly what this task needs |
| `eval_rollout` | `None` | Falls back to `rollout` |
| `workflow` | `"group_rollout"` | A sentinel, not a registry entry — it selects the backend's own `GroupRolloutWorkflow` (<span class="pl-src">platoon/train/auto.py</span>) |
| `discover_entry_points` | `false` | Only needed when several installed plugins must register at once; naming `package` is enough for one |
| `trainer_config` | `None` | Registered but not yet read by any code; the entrypoint module picks the config class |

!!! warning "`eval_dataset_kwargs` does not inherit from `dataset_kwargs`"
    The train/eval *loader* falls back, but the kwargs do not — `eval_dataset_kwargs` defaults to
    `{}` and is used as-is (<span class="pl-src">platoon/train/auto.py</span>). Anything the
    eval split needs must be repeated in its own block. That is why the live textcraft config
    repeats `num_samples_train` and `num_samples_val` twice.

There is also **no `rollout_kwargs`**. A rollout function is called with exactly two positional
arguments, so extra parameters are reachable only through their defaults. To vary them, register a
second name bound to a differently-parameterized function, which is what textcraft's three rollout
registrations do.

## 8. Run it

Install the plugin against one backend:

```bash
cd plugins/unscramble
uv sync --extra tinker
```

Before spending GPUs, run one rollout against any OpenAI-compatible endpoint. This exercises the
task loader, the environment, the prompt and the reward without a trainer:

```python title="plugins/unscramble/smoke_test.py"
import asyncio

from platoon.config_defs import InferenceParams, RolloutConfig
from platoon.unscramble.rollout import run_rollout
from platoon.unscramble.tasks import get_task


async def main() -> None:
    config = RolloutConfig(
        model_name="openai/Qwen/Qwen3-4B-Instruct-2507",
        model_endpoint="http://127.0.0.1:30000/v1",
        model_api_key="dummy",
        max_steps=3,
        output_dir="./smoke_results",
        return_dict=True,
        inference_params=InferenceParams(temperature=0.0, max_completion_tokens=256),
    )
    task = get_task("unscramble.val.0")
    task.max_steps = config.max_steps
    collection = await run_rollout(task, config)
    root = next(iter(collection["trajectories"].values()))
    print("reward:", root["reward"], "| finish:", root["finish_message"])


asyncio.run(main())
```

```bash
uv run python smoke_test.py
```

The `openai/` prefix on `model_name` is LiteLLM's provider selector for an OpenAI-compatible
server; the AReaL workflow adds the same prefix itself
(<span class="pl-src">platoon/train/areal/workflows/group_rollout_workflow.py</span>).
Note that `max_steps` on `RolloutConfig` is not applied to the task for you outside a workflow —
the two lines that set it here are the same two the workflows run. A result of `reward: 1.0` means
every contract on this page is satisfied.

The rollout also wrote JSONL events under `./smoke_results/events/`, which you can watch in the
terminal UI:

```bash
uv run python -m platoon.visualization.cli tail --rdir ./smoke_results
```

Then train:

=== "Tinker"

    ```bash
    uv run python -m platoon.train.tinker.train \
      --config platoon/unscramble/unscramble_tinker.yaml
    ```

    Overrides on this path go through `platoon.utils.config.load_config`, which is argparse-based
    and **requires the leading dashes**:

    ```bash
    uv run python -m platoon.train.tinker.train \
      --config platoon/unscramble/unscramble_tinker.yaml \
      --train.batch_size 64
    ```

=== "AReaL"

    This page only builds `unscramble_tinker.yaml`. `unscramble_areal.yaml` is a file you still
    have to write: an AReaL config is a much larger document — `actor.path`, `actor.backend`,
    `rollout.backend` and the `cluster` block all need real values — and it needs the same
    top-level `environments:` block, copied verbatim, for the shared entrypoint to resolve
    components. Start from
    <span class="pl-src">plugins/number-search/platoon/number_search/number_search_areal.yaml</span>.
    Nothing in the repository exercises `python -m platoon.train.areal.train` end to end today, so
    expect to debug the config rather than the command.

    ```bash
    uv run python -m platoon.train.areal.train \
      --config platoon/unscramble/unscramble_areal.yaml
    ```

    Overrides on this path go through `areal.api.cli_args.load_expr_config`, which is OmegaConf and
    takes bare `key=value` with **no leading dashes**:

    ```bash
    uv run python -m platoon.train.areal.train \
      --config platoon/unscramble/unscramble_areal.yaml \
      trial_name=debug-run train_dataset.batch_size=16
    ```

!!! danger "The two override syntaxes are not interchangeable"
    Tinker's parser only looks at tokens starting with `--`
    (<span class="pl-src">platoon/utils/config.py</span>), so `train.batch_size=64` without
    dashes is silently dropped and the run quietly uses the YAML value. It also coerces eagerly:
    `"1"`/`"true"`/`"yes"` become `True` and `"0"`/`"false"`/`"no"` become `False`
    (<span class="pl-src">platoon/utils/config.py</span>), so `--train.batch_size 0` yields
    `False`, not `0`. AReaL's OmegaConf path has neither behavior.

## What to change for a real task

The scaffolding above is finished; the task-specific work is a short list.

- [ ] **Generate a real dataset.** Write a `python -m platoon.<name>.tasks` CLI that emits
      `<name>_train.jsonl` / `<name>_val.jsonl` and load them with `Task.from_dict`, the way
      `plugins/number-search/platoon/number_search/tasks.py` does. Make the train/eval
      split disjoint by construction, not by chance. See
      [datasets and task loaders](../customization/dataset.md).
- [ ] **Design the action space.** Every action is a callable injected into the IPython shell. Give
      each one a docstring, keep return values short and informative, and never expose the answer
      through the namespace.
- [ ] **Make `evaluate()` cheap and honest.** It runs on every step. Prefer verifying an effect the
      agent produced over checking that a state exists — TextCraft, for instance, requires target
      items to have been *crafted*, comparing against the initial inventory rather than only the
      final one.
- [ ] **Describe the action space in the prompt.** The executor contributes nothing here.
- [ ] **Emit `reward/*` keys** from `evaluate()` and add a
      [reward processor](../customization/rewards.md) once you have more than one signal to
      combine.
- [ ] **Set a realistic `max_steps`** in `rollout_config`, remembering that it overwrites whatever
      the task carries.
- [ ] **Decide whether subagents help.** If the task decomposes, make your executor forkable and
      add `launch_subagent` to the action space — see
      [recursive agents](../tutorials/recursive-agents.md) and
      [subagents](../architecture/subagents.md).
- [ ] **Tune the training knobs** — `group_size`, the loss function, LoRA rank — in
      [algorithm recipes](../recipes/algorithms.md).

## See also

- [Build a plugin](../tutorials/build-a-plugin.md) — the same journey at full length, including the
  per-plugin `train_*.py` script route as well as the registry route.
- [Plugin anatomy](../walkthroughs/plugin-anatomy.md) — a line-by-line read of a real plugin.
- [Custom environment](../customization/environment.md) and
  [custom agent](../customization/agent.md) — the extension points in depth.
- [The registry](../architecture/registry.md) — how `environments:` resolves to callables.
- [Configuration reference](../reference/configuration.md) — every key in both trainer configs.
- [Troubleshooting](../reference/troubleshooting.md) — for when a run fails in a way this page did
  not warn you about.
