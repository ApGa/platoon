# Build your first task

By the end of this page you have a **task plugin** of your own — a task, an environment, an agent
and a rollout program — installed, registered, run against a model and wired to a training config.
Each stage ends in a command that proves it worked; only the last needs hardware.

!!! tip "Your plugin does not have to live in the Platoon repository"
    A plugin is an ordinary Python package: keep it in your own repo, install it alongside
    Platoon, and advertise it through the `platoon.plugins` entry point — no fork, no upstreaming.
    That holds for both kinds — a task plugin like this one, and a capability plugin that adds
    framework functionality such as an agent harness or an environment-server integration. The
    paths below assume a standalone project directory, `unscramble/`.

You need [Platoon installed](../get-started/installation.md) and the vocabulary from
[core concepts](../get-started/concepts.md). The task: the agent sees a scrambled word, calls
`answer("...")` with the unscrambled form, and scores `1.0` when right.

```text
unscramble/
├── pyproject.toml
└── platoon/                     # NO __init__.py in this directory
    └── unscramble/
        ├── __init__.py          # may be empty
        ├── tasks.py             # task ids and get_task
        ├── env.py               # actions and reward
        ├── agent.py             # the system prompt
        ├── rollout.py           # one episode, recorded
        ├── registry.py          # names the config resolves
        └── unscramble_tinker.yaml
```

## 1. Scaffold and install

`pyproject.toml` is an ordinary hatchling package with `packages = ["platoon"]`, a dependency on
`platoon >= 0.1.0`, and conflicting `tinker` / `areal` extras. Only the entry point is specific to
Platoon — it is how Platoon finds a plugin it was never told about:

```toml title="unscramble/pyproject.toml"
[project.entry-points."platoon.plugins"]
unscramble = "platoon.unscramble.registry"
```

Do not put an `__init__.py` in the plugin's `platoon/` directory: core Platoon makes `platoon` a
namespace package so your directory merges with the core one living in another repository.

```bash
mkdir -p unscramble/platoon/unscramble && touch unscramble/platoon/unscramble/__init__.py
cd unscramble && uv sync --extra tinker
uv run python -c "import platoon, platoon.unscramble; print(platoon.__path__)"
```

Two entries in `__path__` — core Platoon and yours — means the merge worked.

## 2. Tasks

The trainer needs one function: `(task_id: str) -> Task`. `Task` carries `goal`, `id`, `max_steps`,
`misc` and `fork_strategy`; task-specific state goes in `misc`, which the environment reads.

```python title="platoon/unscramble/tasks.py"
import random

from platoon.envs.base import Task

WORDS = {"train": ["anchor", "basket", "candle", "dolphin"], "val": ["island", "jungle"]}
TASKS: dict[str, Task] = {}


def get_task_ids(split: str, num_train: int = 2000, num_val: int = 200) -> list[str]:
    count = {"train": num_train, "val": num_val}[split]
    return [f"unscramble.{split}.{i}" for i in range(count)]

def get_task(task_id: str) -> Task:
    if task_id not in TASKS:
        _, split, index = task_id.split(".")
        word = WORDS[split][int(index) % len(WORDS[split])]
        scrambled = "".join(random.Random(task_id).sample(word, len(word)))
        goal = f"Unscramble the letters '{scrambled}' into one English word."
        TASKS[task_id] = Task(goal=goal, id=task_id, max_steps=3, misc={"word": word})
    return TASKS[task_id]
```

Seed on the task id: every rollout worker re-runs your loader and the scramble has to come out
identical in all of them. Both backends overwrite `task.max_steps` with `rollout_config.max_steps`,
so that is a default. Larger tasks ship a generated dataset as JSONL — see [extend](extend.md).

```bash
uv run python -c "from platoon.unscramble.tasks import get_task; print(get_task('unscramble.val.0'))"
```

## 3. The environment

An environment is anything satisfying the `Env` protocol. For a task the model solves by writing
Python, subclass `CodeActEnv`, give it an executor whose actions are plain callables, and override
`evaluate()`.

```python title="platoon/unscramble/env.py"
from platoon.agents.actions.common import finish
from platoon.envs.base import Task
from platoon.envs.codeact import CodeActEnv, IPythonCodeExecutor
from platoon.episode.context import finish_message

class UnscrambleExecutor(IPythonCodeExecutor):
    def __init__(self, task: Task):
        self._word: str = task.misc["word"]
        super().__init__(task, actions=(finish, self.answer))

    def answer(self, guess: str) -> str:
        if guess.strip().lower() == self._word:
            finish_message.set(f"You unscrambled {self._word} correctly!")
            return "Correct."
        return f"'{guess}' is not the word. Try again."

    async def describe_action_space(self) -> str:
        return "def answer(guess: str) -> str  # one lowercase word; finish(msg) ends the episode"

class UnscrambleEnv(CodeActEnv):
    def __init__(self, task: Task):
        super().__init__(task, UnscrambleExecutor(task))

    async def evaluate(self) -> tuple[float, dict]:
        if not self._state.finished:
            return 0.0, {}
        success = 1.0 if "correctly" in (finish_message.get(None) or "") else 0.0
        return success, {"reward/success": success}
```

Four details are load-bearing. Only the bound `answer` method reaches the IPython namespace, so
model-authored code cannot read `self._word`. Setting `finish_message` marks the state finished and
halts the episode loop — `finish` sets it too, so an agent that gives up terminates cleanly.
`evaluate()` runs on *every* step and accumulates, so return an increment. And the inherited
`describe_action_space()` is empty: override it or the model never learns your actions exist.

## 4. The agent

For a CodeAct task the agent is usually only a prompt. Extend the built-in system template through
its `env_specific_system_context` slot rather than replacing it, so the `<thought>`/`<python>` rules
the parser depends on stay correct.

```python title="platoon/unscramble/agent.py"
from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder, PromptMode
from platoon.envs.codeact import CodeActObservation

class UnscramblePromptBuilder(CodeActPromptBuilder):
    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        context.setdefault("env_specific_system_context",
                           "Rearrange the scrambled letters into one common English word and "
                           "submit it with answer(...). Wrong guesses cost a step.")
        return super().build_system_prompt(obs, **context)

class UnscrambleAgent(CodeActAgent):
    def __init__(self, prompt_mode: PromptMode = "sequence_extension",
                 include_reasoning: bool = True, **kwargs):
        kwargs.setdefault("prompt_builder", UnscramblePromptBuilder(
            prompt_mode=prompt_mode, include_reasoning=include_reasoning))
        super().__init__(prompt_mode=prompt_mode, include_reasoning=include_reasoning, **kwargs)
```

Both flags must reach both objects: the builder uses `include_reasoning` to ask for `<thought>`
blocks, the agent to parse them. The action list comes from the executor's action-space text.

## 5. The rollout function

The function both trainers call: two positional arguments, a `Task` and a `RolloutConfig`
(<span class="pl-src">platoon/config_defs.py</span>), returning the trajectory collection.

```python title="platoon/unscramble/rollout.py"
import asyncio

from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.episode.context import current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import TrajectoryCollection
from platoon.utils.llm_client import LiteLLMClient
from platoon.visualization.event_sinks import JsonlFileSink
from .agent import UnscrambleAgent
from .env import UnscrambleEnv

async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = None
    try:
        client = LiteLLMClient(config.model_name, config.model_endpoint, config.model_api_key)
        env = UnscrambleEnv(task)
        agent = UnscrambleAgent(llm_client=client, inference_params=config.inference_params)
        collection = TrajectoryCollection()
        current_trajectory_collection.set(collection)
        path = f"{config.output_dir}/events/events_{task.id}_{collection.id}.jsonl"
        collection.register_event_handlers(JsonlFileSink(path, collection_id=collection.id))
        episode = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))
        await asyncio.wait_for(episode, timeout=config.timeout)
        collection = current_trajectory_collection.get()
        return collection.to_dict() if config.return_dict else collection
    finally:
        for handle in (agent, env):
            if handle is not None:
                await handle.close()
```

Three things are not optional: a fresh `TrajectoryCollection` set on the context variable *before*
`run_episode`, which `CodeActEnv.reset` reads to register the task; `run_episode` wrapped in
`asyncio.create_task`, which keeps its context-variable writes out of the caller and is what lets
concurrent rollouts and nested subagents coexist; and `close()` on both handles in `finally`.

It must be an importable, module-level `async` function: on the AReaL path it reaches workers as a
dotted import path, so a lambda or closure is rejected. `run_episode(timeout=...)` bounds each step
and the outer `wait_for` the whole trajectory. The task loader, by contrast, must not be `async`.

## 6. Run one rollout against a real model

One API call per step exercises the whole chain, against any OpenAI-compatible endpoint — a hosted
API or a local vLLM/SGLang server.

```python title="unscramble/smoke_test.py"
import asyncio
from platoon.config_defs import RolloutConfig
from platoon.unscramble.rollout import run_rollout
from platoon.unscramble.tasks import get_task

config = RolloutConfig(model_name="openai/Qwen/Qwen3-4B-Instruct-2507", max_steps=3,
                       model_endpoint="http://127.0.0.1:30000/v1", model_api_key="dummy",
                       output_dir="./smoke_results", return_dict=True)
task = get_task("unscramble.val.0")
task.max_steps = config.max_steps  # the workflows do this for you during training
out = asyncio.run(run_rollout(task, config))
print(next(iter(out["trajectories"].values()))["reward"])
```

```bash
uv run python smoke_test.py
```

The `openai/` prefix is LiteLLM's provider selector for an OpenAI-compatible server; against a
hosted API, drop `model_endpoint` and set a real key. A reward of `1.0` means every contract here is
satisfied; replay the event log left in `./smoke_results/events/` in the [TUI](inspect-rollouts.md).

## 7. Register the components

A registry maps names to objects, one registration function per component kind
(<span class="pl-src">platoon/registry.py</span>); importing this module populates them.

```python title="platoon/unscramble/registry.py"
from platoon.registry import register_dataset_loader, register_rollout, register_task_loader
from platoon.unscramble.rollout import run_rollout
from platoon.unscramble.tasks import get_task, get_task_ids

def load_dataset(config, split: str, limit: int | None = None,
                 num_train: int = 2000, num_val: int = 200):
    ids = get_task_ids("val" if split == "eval" else split, num_train, num_val)
    return ids[:limit] if limit is not None else ids

register_task_loader("unscramble/default", get_task)
register_rollout("unscramble/default", run_rollout)
register_dataset_loader("unscramble/default", load_dataset)
```

`split` arrives as `"train"` or `"eval"`, never `"val"`; namespace names as `"<plugin>/<variant>"`.

```bash
uv run python -c "import platoon.unscramble.registry; from platoon.registry import get_registry
print([get_registry(k).names() for k in ['task_loader', 'dataset_loader', 'rollout']])"
```

## 8. Train

The top-level `environments:` key is a list of exactly one `EnvironmentConfig`
(<span class="pl-src">platoon/train/components.py</span>) — registry wiring, and what keeps the
trainers task-agnostic: a new task is a YAML block, not a training script.

```yaml title="platoon/unscramble/unscramble_tinker.yaml"
environments:
  - discover_entry_points: true
    dataset_loader: unscramble/default
    task_loader: unscramble/default
    rollout: unscramble/default
    dataset_kwargs: {num_train: 2000}
    eval_dataset_kwargs: {num_val: 200, limit: 100}

train:
  model_name: Qwen/Qwen3-4B-Instruct-2507
  renderer_name: qwen3_instruct
  batch_size: 32
  workflow_config:
    group_size: 8
    rollout_config: {max_steps: 3, output_dir: ./rollout_results, timeout: 300}

eval: {strategy: step, every: 10, workflow_config: {group_size: 1, rollout_config: {max_steps: 3}}}
log_path: ./logs
```

`discover_entry_points: true` loads every installed plugin's `platoon.plugins` entry point — the
line that makes an out-of-repo package resolvable by name. `train`, `eval`, `log_path`, `model_name`
and `renderer_name` are mandatory. Rest: [configuration](../reference/configuration.md).

```bash
uv run python -m platoon.train.tinker.train --config platoon/unscramble/unscramble_tinker.yaml
```

That trains against any Tinker-compatible backend. Overrides here are argparse-style and need the
leading dashes (`--train.batch_size 64`); AReaL reads the same block but takes bare `key=value`.

Next: [extend Platoon](extend.md) for custom datasets, rewards and workflows, or
[multi-agent workflows](multi-agent.md) to add `launch_subagent` and train a delegation tree.
