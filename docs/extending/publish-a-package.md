# Publish an extension package

**Goal.** Ship your components as a Python package that anyone can install and then name in a config: a harness,
an episode program, a reward processor, a set of tasks and a named environment that bundles them, declared with
entry points so that platoon finds them by name, tested on CPU with `platoon.testing`, and built into a wheel. The
finished package is [`examples/extension-package/`](../../examples/extension-package/).

**Prerequisites.**

- Platoon installed from the repository root with the `tinker` and `ors` extras: `uv sync --extra tinker --extra ors`.
- [Extending platoon](index.md), and the guides of the components you package: [function
  harnesses](../tutorials/function-harness.md), [episode programs](../tutorials/programs.md) and [reward
  processors](rewards.md).
- For the run in step 9, a [Tinker](https://thinkingmachines.ai/tinker/) API key in `TINKER_API_KEY`. The tests in
  step 8 need none.

Run the commands from the repository root.

## What you will build

| Component | Name in configs | Module | Entry-point group |
|---|---|---|---|
| A function harness that answers, then checks its answer | `example-self-check` | `harness.py` | `platoon.harnesses` |
| An episode program that grades the final answer against the task's answer | `example-exact-answer` | `program.py` | `platoon.programs` |
| A reward processor that gives near misses part of the reward | `example-partial-credit` | `rewards.py` | `platoon.reward_processors` |
| Arithmetic questions, as a task table shipped in the package | `example_extension.tasks:arithmetic` | `tasks.py`, `data/arithmetic.jsonl` | none: a `module:function` reference |
| A named environment that bundles them, with the harness's options and the splits | `example-arithmetic@1` | `entries.py` | `platoon.environments` |

Since 0.2.0, a package provides programs, named environments and reward processors by name, as it
provides harnesses ([RFC 0013](../rfcs/0013-extension-model.md)).

## Step 1: Lay out the package

```text
examples/extension-package/
├── pyproject.toml              the distribution and its entry points
├── README.md
├── eval.yaml, train.yaml       configs that name the components
├── named.yaml                  a config that names the named environment
├── src/example_extension/
│   ├── __init__.py             empty
│   ├── harness.py
│   ├── program.py
│   ├── rewards.py
│   ├── tasks.py
│   ├── entries.py              the named environment
│   └── data/arithmetic.jsonl
└── tests/test_example_extension.py
```

- **One owner per package.** Every component name starts with the package's owner, `example-` here: `<owner>-<name>`,
  in lower-case words of letters and digits joined by `-`, `_` or `.`. Names that start with `platoon-`, and
  `single`, `gym`, `mini-swe-agent` and `openhands`, belong to platoon itself. Since 0.2.0, platoon refuses
  a name that breaks the rule, and `platoon extensions check` warns about a package with more than one owner.
- **Keep `__init__.py` empty.** Platoon imports each component's module when a config names it, in the trainer and
  in every rollout worker; an `__init__.py` that imports everything would make every run that names one component
  import them all.
- **Ship small data in the package.** The tasks are read with `importlib.resources`, so they work wherever the
  package is installed. Keep large datasets out of the wheel and download them on first use.

## Step 2: Write the harness

[`harness.py`](../../examples/extension-package/src/example_extension/harness.py):

```python
"""``example-self-check``: a function harness that answers, then checks its answer before giving it.

It suits tasks without tools, such as the rows of a task table: the first model call answers the instruction,
each check asks the model to look over its answer, and the last reply is the agent's final answer. Every model
call is one step of the agent's budget.

Options (``harness_options`` in the environment config):

* ``checks``: how many times the model checks its answer (default 1; 0 answers once).
* ``check_prompt``: the message that asks for a check.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import platoon as pt
from platoon.harness.agent import LIMIT_MESSAGE

CHECK_PROMPT = "Check your answer step by step. Then reply with the final answer only."


@dataclass(frozen=True)
class SelfCheckOptions:
    checks: int = 1
    check_prompt: str = CHECK_PROMPT

    @classmethod
    def from_mapping(cls, options: Mapping[str, Any]) -> SelfCheckOptions:
        """The options of ``harness_options``; an unknown key or a bad value is a ``ValueError``."""

        unknown = sorted(set(options) - {field.name for field in fields(cls)})
        if unknown:
            raise ValueError(f"example-self-check has no options {unknown}")
        parsed = cls(**options)
        if isinstance(parsed.checks, bool) or not isinstance(parsed.checks, int) or parsed.checks < 0:
            raise ValueError(f"checks must be a non-negative integer, got {parsed.checks!r}")
        return parsed


@pt.harness("example-self-check")
async def self_check(ctx: pt.AgentContext) -> pt.AgentResult:
    """Answer the instruction, check the answer `checks` times, and reply with the final answer."""

    options = SelfCheckOptions.from_mapping(ctx.options)  # raising here makes the rollout an error, never a score
    messages: list[dict[str, Any]] = [{"role": "user", "content": ctx.instruction}]
    for turn in range(options.checks + 1):
        if turn:
            messages.append({"role": "user", "content": options.check_prompt})
        if ctx.budget.remaining() <= 0:  # one step per model call; platoon does not stop the loop for you
            return pt.AgentResult("limit", error=LIMIT_MESSAGE)
        reply = await ctx.chat(messages)
        messages.append({"role": "assistant", "content": reply.text})
        ctx.annotate_step(turn="answer" if turn == 0 else "check")  # shown by the trajectory viewer
    return pt.AgentResult("completed", final_message=reply.text)
```

The harness checks its options itself, so a misspelled key in `harness_options` fails the rollout as an error
instead of being ignored. Each turn's prompt extends the previous one and its reply, so the gateway merges the
turns into one training sequence.

## Step 3: Write the episode program

A task table has no grader of its own, so the program grades the root agent's final answer.
[`program.py`](../../examples/extension-package/src/example_extension/program.py):

```python
"""``example-exact-answer``: an episode program that grades the root agent's final answer.

A task table has no grader of its own (its sessions grade ``invalid``, reason ``self_grade``), so this program sets
the outcome: the task's ``answer`` field (a column of the table) is the expected answer. The outcome's primary
reward is 1.0 for an exact match, after normalizing case, spaces and a final period, and 0.0 otherwise. A second
reward, ``close``, says how near a numeric answer came to the expected number, for a reward processor that wants
it (``example-partial-credit``).
"""

from __future__ import annotations

import re

import platoon as pt

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    """Lower case, single spaces, no final period or exclamation mark."""

    return " ".join(text.lower().strip().rstrip(".!").split())


def closeness(given: str, expected: str) -> float:
    """1.0 for the expected number, falling linearly to 0.0 as the answer's last number moves away from it."""

    numbers = _NUMBER.findall(given)
    try:
        value, target = float(numbers[-1]), float(expected)
    except (IndexError, ValueError):
        return 0.0
    return max(0.0, 1.0 - abs(value - target) / max(1.0, abs(target)))


@pt.program("example-exact-answer")
async def exact_answer(ep: pt.EpisodeContext) -> pt.Outcome:
    """Run the root agent, then compare its final answer with the task's ``answer``."""

    result = await ep.run()  # the root agent, with the environment's harness and options
    if result.status not in ("completed", "limit"):
        return await ep.grade()  # a crash, timeout or cancellation keeps its untrainable outcome
    expected = ep.task.misc.get("answer")
    if expected is None:  # a row without an answer cannot be measured: never train on it as reward 0
        return pt.Outcome.infra("no_answer", status="invalid", reason="the task has no answer to compare with")
    given = normalize(result.final_message or "")
    rewards = {"reward": float(given == normalize(str(expected))), "close": closeness(given, str(expected))}
    return pt.Outcome(status="scored", rewards=rewards, metadata={"expected": str(expected), "given": given})
```

The outcome's `reward` is what the episode scores and what evaluation counts as success; `close` travels with it
in the root trajectory's `misc["outcome"]["rewards"]`, for the reward processor.

## Step 4: Write the reward processor and the tasks

[`rewards.py`](../../examples/extension-package/src/example_extension/rewards.py) trains on exact answers and
gives near misses half of their `close` reward:

```python
"""``example-partial-credit``: a reward processor that gives near misses part of the reward.

A reward processor turns each finished trajectory, as a dict, into the reward it trains on and the components that
training reports as metrics. This one keeps an exact answer at 1.0 and gives a wrong numeric answer half of the
``close`` reward that ``example-exact-answer`` graded, so near misses train better than far ones. Evaluation still
counts only exact answers as successes: its success rate reads the graded reward, not this one.
"""

from __future__ import annotations

from typing import Any


def partial_credit(trajectory: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """The training reward of one trajectory, and its components (keys that start with ``reward/``)."""

    graded = ((trajectory.get("misc") or {}).get("outcome") or {}).get("rewards") or {}
    exact = float(trajectory.get("reward") or 0.0)
    close = float(graded.get("close", 0.0))
    reward = exact if exact >= 1.0 else 0.5 * close
    return reward, {"reward/exact": exact, "reward/close": close}
```

[`tasks.py`](../../examples/extension-package/src/example_extension/tasks.py) serves the questions from the
package's own data file:

```python
"""The arithmetic tasks: a task table that ships inside the package.

A config builds it with ``service: example_extension.tasks:arithmetic``. The rows are read from the installed
package, so the config needs no path, and every process that runs rollouts finds the same file.
"""

from __future__ import annotations

from importlib.resources import files

from platoon.envs import TaskTableService


def arithmetic() -> TaskTableService:
    """The arithmetic questions: splits ``train`` and ``test``, each row with its ``answer``."""

    return TaskTableService(path=str(files(__package__) / "data" / "arithmetic.jsonl"), name="example-arithmetic")
```

Each row of `data/arithmetic.jsonl` is a task: its `id`, `split` and `goal`, and an `answer` column that the task
gets in its `misc`:

```json
{"id": "test-0", "split": "test", "goal": "What is 22 + 80? Reply with the number only.", "answer": "102"}
```

## Step 5: Declare the entry points

[`pyproject.toml`](../../examples/extension-package/pyproject.toml) declares one entry point per component, in the
group of the component's kind, named exactly as configs name it:

```toml
# An example platoon extension, packaged so that any platoon install finds its components by name.
#
#   uv pip install -e examples/extension-package
#   uv run platoon harness show example-self-check

[project]
name = "example-extension"
version = "0.1.0"
description = "An example platoon extension: a self-checking harness, an exact-answer program and arithmetic tasks."
requires-python = ">=3.12"
dependencies = ["platoon >= 0.4, < 0.5"]

# One group per kind of component. Each entry's name is the component's name and its value is the object that
# defines it, so platoon imports a module only when a config names one of its components.
[project.entry-points."platoon.harnesses"]
example-self-check = "example_extension.harness:self_check"

[project.entry-points."platoon.programs"]
example-exact-answer = "example_extension.program:exact_answer"

[project.entry-points."platoon.reward_processors"]
example-partial-credit = "example_extension.rewards:partial_credit"

[project.entry-points."platoon.environments"]
example-arithmetic = "example_extension.entries:ARITHMETIC"

[tool.uv.sources]
platoon = { path = "../..", editable = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/example_extension"]

[tool.ruff]
line-length = 120
src = ["src"]   # example_extension sorts as this package's own imports, platoon as a dependency's

[tool.ruff.lint]
select = ["E", "F", "I"]
```

| Group | Entry name | Value |
|---|---|---|
| `platoon.harnesses` | the harness's name | the `@pt.harness` function, or the path of a TOML spec file |
| `platoon.programs` | the program's name | the `@pt.program` function |
| `platoon.reward_processors` | the reward processor's name | the function |
| `platoon.environments` | the named environment's name | the `pt.EnvironmentEntry`, or a list of entries that share the name, one per version |

Since 0.2.0, platoon reads the groups from the installed metadata and imports an entry point only when a
config names its component, in each process that needs it, so installing a package never changes a run that does not
name one of its components. The requirement on platoon, `platoon >= 0.4, < 0.5`, says which versions of platoon the
package works with (step 10).

`[tool.uv.sources]` points platoon at this repository while you develop here; a package of your own leaves it out,
or points it at your checkout of platoon. `[tool.ruff]` holds the package's lint settings: with `src = ["src"]`,
imports sort with platoon as a dependency and `example_extension` as the package's own.

## Step 6: Bundle a named environment

*Since 0.2.0.* A named environment says once how to run your tasks: the service, the harness and the
options it needs, the program, the reward processor and the splits. A config then needs one line for them.
[`entries.py`](../../examples/extension-package/src/example_extension/entries.py):

```python
"""``example-arithmetic@1``: the package's named environment.

It says once how to run the arithmetic tasks: the task table, the harness that plays them and its options, the
program that grades them, the reward processor that trains on them, and the splits. A config selects it with
``env: example-arithmetic@1`` and may override any of its keys.
"""

from __future__ import annotations

import platoon as pt

ARITHMETIC = pt.EnvironmentEntry(
    "example-arithmetic",
    version="1",
    description="Arithmetic questions with numeric answers, graded by exact match (a task table)",
    service="example_extension.tasks:arithmetic",
    harness="example-self-check",
    harness_defaults={"example-self-check": {"checks": 1}},
    program="example-exact-answer",
    reward_processor="example-partial-credit",
    dataset_kwargs={"split_name": "train"},
    eval_dataset_kwargs={"split_name": "test"},
)
```

`pt.EnvironmentEntry(name, version="1", *, description="", harness_defaults=None, **keys)` takes the keys of an
`environments` entry, except the ones each run decides: `name`, `weight`, `env`, `harness_options`, `exclude_tasks`,
`max_staleness`, the run-level settings, and of the entry's training settings all but the rollout settings its tasks
were measured with (`workflow_config.rollout_config`, with no `eval_overrides.workflow_config`) and the reward that
counts as a pass (`success_threshold`). `harness_defaults` maps a harness to the options the environment needs from
it, so a config that picks another harness does not inherit options it cannot read. Leave a task the environment
should not offer out of its own task lists, in a new version, rather than in `exclude_tasks`: a config's own
`exclude_tasks` would replace the entry's list instead of adding to it. The entry is checked against the config schema
when it is built.

[`named.yaml`](../../examples/extension-package/named.yaml) evaluates through it:

```yaml
# eval.yaml through the package's named environment: env: example-arithmetic@1 brings the task table, the harness
# and its options, the program, the reward processor and the splits. With the package installed:
#
#   uv run platoon rollout examples/extension-package/named.yaml --task test-0

extends: eval.yaml

environments:
  - name: arithmetic
    env: example-arithmetic@1
```

- **Merging.** The named environment is the lowest layer of the entry. The config's own keys merge over it,
  mappings key by key and anything else, lists included, replacing: `harness: platoon-react` on the entry plays the
  same tasks with another harness, and `eval_dataset_kwargs: {limit: 2}` keeps the split and adds a limit.
- **Versions.** Bump the version when the tasks, the grading or the defaults change, and keep the old version
  registered, so that a config pinned to `@1` keeps its meaning. A bare name resolves only while one version is
  registered, and the resolved config (`config.resolved.yaml`) records the version it resolved to.
- **Harbor datasets.** A package that ships Harbor task directories registers them the same way:
  `service="harbor"` with `service_kwargs={"dataset_dir": str(files(__package__) / "tasks")}`. Leave site facts out
  of the entry: the sandbox provider and its options belong in each site's config, which merges over the entry.

## Step 7: Install and check

Install the package into the environment platoon runs in. An editable install picks up code changes at once; a
change to the entry points needs the install to run again.

```bash
uv pip install -e examples/extension-package
uv run platoon harness show example-self-check
```

```text
example-self-check@1  (function)
source: example_extension.harness:self_check
runtime: inline
executes_model_code: False

Answer the instruction, check the answer `checks` times, and reply with the final answer.
```

An entry point that fails to load (its module does not import, or it names something that is not a component of
its group) fails the lookups of its own name, with an error that names its package, and leaves every other component
available; a run that names it exits with code 78 and the reason.

Since 0.2.0, more commands show what a package provides. `platoon extensions list` reads the installed
metadata, so it imports nothing of an extension:

```bash
uv run platoon extensions list --kind program
```

```text
kind     name                  distribution       version  status  source
program  gym@1                 platoon            0.4.0    ok      platoon.gym:gym_program
program  single@1              platoon            0.4.0    ok      platoon.programs:single
program  example-exact-answer  example-extension  0.1.0    ok      example_extension.program:exact_answer
```

`platoon extensions check` checks a package's names, its single owner and its requirement on platoon from the
metadata; `--load` also imports every component as a config naming it would:

```bash
uv run platoon extensions check example-extension --load
```

```text
ok: no problems found
```

`platoon extensions show NAME` prints one component as the list shows it, with the entry point that provides it,
the package's requirement on platoon and where configs use it; `--load` also imports it, to show the versions it
registers (`uv run platoon extensions show example-exact-answer --load`).

`platoon env show` prints a named environment with the entry and the harness options it stands for:

```bash
uv run platoon env show example-arithmetic
```

```text
example-arithmetic@1
source: example_extension.entries:ARITHMETIC
distribution: example-extension 0.1.0
description: Arithmetic questions with numeric answers, graded by exact match (a task table)
entry:
  service: example_extension.tasks:arithmetic
  harness: example-self-check
  program: example-exact-answer
  reward_processor: example-partial-credit
  dataset_kwargs:
    split_name: train
  eval_dataset_kwargs:
    split_name: test
harness_defaults:
  example-self-check:
    checks: 1
```

`platoon harness list`, `platoon program list` and `platoon env list` list the components of each kind with the
distribution that provides each.

## Step 8: Test on CPU

`platoon.testing` runs whole episodes with a scripted policy in place of the model: the environment session, the
episode program, the real model gateway and the trajectories, on CPU, with no model download.
[`tests/test_example_extension.py`](../../examples/extension-package/tests/test_example_extension.py):

```python
"""CPU tests of the example extension: its task table, whole episodes with a scripted policy, and the named
environment through ``platoon rollout``.

    uv pip install -e examples/extension-package
    uv run pytest examples/extension-package/tests -q

``platoon.testing.run_episode`` runs one task the way training and evaluation do: the environment session, the
episode program, the real model gateway and the trajectories, with a scripted policy in place of the model. Those
tests name the components by import string, so they run whether or not the package is installed. The last test
serves a scripted policy to ``platoon rollout`` with ``platoon.testing.serve_policy`` and runs ``named.yaml``, whose
``env: example-arithmetic@1`` finds every component by the name its entry point gives: it needs the package
installed.
"""

from __future__ import annotations

import asyncio
import json
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import pytest
from platoon.cli import main as platoon_cli
from platoon.testing import ScriptedPolicy, run_episode, serve_policy

from example_extension.program import closeness
from example_extension.rewards import partial_credit
from example_extension.tasks import arithmetic

NAMED = Path(__file__).parents[1] / "named.yaml"
INSTALLED = bool(entry_points(group="platoon.environments", name="example-arithmetic"))


def episode(task_id: str, *replies: str, **harness_options: Any) -> tuple[Any, ScriptedPolicy]:
    """One episode of the arithmetic task ``task_id``, in which the model writes ``replies`` in order."""

    policy = ScriptedPolicy(list(replies))
    run = asyncio.run(
        run_episode(
            task_id,
            service="example_extension.tasks:arithmetic",
            harness="example_extension.harness:self_check",
            program="example_extension.program:exact_answer",
            harness_options=harness_options,
            policy=policy,
        )
    )
    return run, policy


def test_the_tasks_ship_with_the_package():
    async def read() -> tuple[list[str], list[str], Any]:
        table = arithmetic()
        return await table.list_tasks("train"), await table.list_tasks("test"), await table.get_task("test-0")

    train, test, task = asyncio.run(read())

    assert (len(train), len(test)) == (24, 8)
    assert (task.goal, task.misc["answer"]) == ("What is 22 + 80? Reply with the number only.", "102")


def test_the_answer_after_the_check_is_graded():
    run, policy = episode("test-0", "100", "102")  # a wrong first answer, corrected by the check

    assert run.outcome["status"] == "scored" and run.reward == 1.0
    assert run.outcome["rewards"]["close"] == 1.0
    assert len(policy.calls) == 2
    assert policy.calls[1].last_user_message.startswith("Check your answer step by step.")


def test_a_near_miss_earns_partial_credit_in_training_only():
    run, _ = episode("test-0", "100", "101")

    assert run.reward == 0.0  # the grade: 101 is not the answer
    trajectory = run.collection.to_dict()["trajectories"][run.root.id]
    assert partial_credit(trajectory) == (
        0.5 * closeness("101", "102"),
        {"reward/exact": 0.0, "reward/close": closeness("101", "102")},
    )


def test_checks_sets_how_many_times_the_model_is_asked():
    run, policy = episode("test-2", "36", checks=0)

    assert run.reward == 1.0 and len(policy.calls) == 1


def test_an_unknown_option_is_an_error_not_a_score():
    run, policy = episode("test-2", chekcs=1)

    assert run.outcome["status"] == "error" and not policy.calls


@pytest.mark.skipif(not INSTALLED, reason="needs the package installed: uv pip install -e examples/extension-package")
def test_the_named_environment_finds_every_component_by_name(tmp_path, capsys):
    with serve_policy(["100", "102"]) as policy:  # platoon rollout samples this policy instead of a Tinker model
        platoon_cli(["rollout", str(NAMED), "--task", "test-0", "--output-dir", str(tmp_path), "--json-output"])

    summary = json.loads(capsys.readouterr().out)
    assert (summary["reward"], summary["total_steps"]) == (1.0, 2) and len(policy.calls) == 2
    lock = json.loads((tmp_path / "lock.json").read_text())
    assert {component["kind"]: component["resolved"] for component in lock["components"]} == {
        "environment": "example-arithmetic@1",
        "harness": "example-self-check@1",
        "program": "example-exact-answer@1",
        "service": "example_extension.tasks:arithmetic",
        "reward_processor": "example-partial-credit",
    }
```

```bash
uv run pytest examples/extension-package/tests -q
```

```text
......                                                                   [100%]
6 passed in ...s
```

`ScriptedPolicy` answers each model call with the next completion of its script (or with a function of the prompt),
and records the call; `run_episode` returns the episode's trajectories, its outcome and its reward. `serve_policy`
serves a scripted policy to `platoon rollout` and `platoon eval` called in the test's process, in place of Tinker.
The last test runs `named.yaml` with it, so it checks what only an installed package can get wrong: that each entry
point names its component and loads it. `lock.json` shows what each name resolved to. The test skips until the
package is installed, as step 7 installed it. [Test an environment, a harness or a program on
CPU](../how-to/test-extensions.md) covers `platoon.testing` in full: tool calls, policies that read the prompt,
subagents, and the pytest plugin.

## Step 9: Run it on a model

[`eval.yaml`](../../examples/extension-package/eval.yaml) names every component by the name its entry point gives:

```yaml
environments:
  - name: arithmetic
    service: example_extension.tasks:arithmetic   # a function that returns the package's task table
    harness: example-self-check                   # the components, by the names their entry points give
    program: example-exact-answer
    reward_processor: example-partial-credit
    eval_dataset_kwargs: {split_name: test}
```

```bash
uv run platoon rollout examples/extension-package/eval.yaml --task test-0
```

A rollout in which the model answers right the second time prints the outcome with both of its rewards:

```text
rollout test-0-<id>  task test-0
agents 1  steps 2  max depth 0  reward 1.0  outcome scored reward=1.0, close=1.0
└─ policy <id>  steps 2  reward 1.0  outcome scored reward=1.0, close=1.0
```

`uv run platoon rollout examples/extension-package/named.yaml --task test-0` runs the same task through the named
environment. `uv run platoon eval examples/extension-package/eval.yaml` evaluates the eight test tasks, and
`uv run platoon train examples/extension-package/train.yaml` trains a LoRA for five steps.

Since 0.2.0, a run's `lock.json` records every component the run named: what it resolved to, the
distribution and version that provided it, and the entry point that named it.

## Step 10: Publish

- **Version.** Give the distribution a version and bump it on every release.
- **Require a range of platoon.** `dependencies = ["platoon >= X.Y, < N"]`, from the release you tested against up
  to the first release that may break it: the next minor release while platoon is below 1.0 (`platoon >= 0.4, < 0.5`),
  the next major release from 1.0 on. Since 0.2.0, platoon checks an extension's requirement against the
  running platoon before it imports any of its components, and refuses one that excludes it; set
  `PLATOON_IGNORE_EXTENSION_REQUIREMENTS=1` to load it anyway while you develop against a checkout.
- **Check it.** `uv run platoon extensions check example-extension --load` checks the names, the owner and the
  requirement from the metadata, then imports every component as a config naming it would.
- **Build and try the wheel.** `uv build` in the package directory writes the wheel and the source distribution
  into `dist/`. Install the wheel into a fresh environment with platoon and run your tests there: it checks that the
  data files and the entry points made it into the wheel.
- **Upload.** `uv publish` uploads to PyPI, or to your own index with `--publish-url`.
- **Document it.** The README says what each component does, what it needs (extras, environment variables, a
  sandbox runtime), and how a config names it.

## Reference

### How platoon resolves a name

Since 0.2.0, a reference with `:` is an import string and is imported as written. Any other reference
is `name` or `name@version`, and resolves in this order:

1. Platoon's own components. Their names are reserved: `platoon-*` names, and `single`, `gym`, `mini-swe-agent` and
   `openhands`.
2. The installed entry point of the component's group that has exactly that name. It is imported the first time a
   config or a program names it, after its distribution's platoon requirement is checked.
3. Components that the environment entry's `package:` modules registered.

Several versions of one name may be installed at once: `name@version` picks one, a version prefix matches at dots
(`@2` matches `2.4.6`), and a bare name resolves only while exactly one version is registered.

### Troubleshooting

| Problem | Fix |
|---|---|
| `Unknown harness 'example-self-check'` | Install the package into the environment platoon runs in, and check `platoon harness list` for an entry point that failed to load |
| A new or renamed entry point is not found | Entry points are recorded when the package installs: install it again |
| `Harness example-self-check@1 is defined twice` | Two installed packages, or a package and a `package:` module, register the same `name@version` |
| `ModuleNotFoundError` in rollout workers | The package is installed only where the trainer runs; install it wherever rollouts run |
| `example-extension 0.1.0 requires platoon >= ..., but platoon ... is running` | Install a version of the package that supports this platoon, or set `PLATOON_IGNORE_EXTENSION_REQUIREMENTS=1` while you develop |
| `... reference 'example-arithmetic' is ambiguous` | Several versions are installed: name one, `example-arithmetic@1` |
| A warning that an entry point is named after its module | Name each `platoon.harnesses` entry point after the harness it registers: platoon looks entry points up by name, and loads the others only when a name matches nothing |

## Next steps

- [Extending platoon](index.md): every extension point, and the rules every extension follows.
- [Test an environment, a harness or a program on CPU](../how-to/test-extensions.md): everything `platoon.testing`
  offers.
- [Reward processors and finalizers](rewards.md) and [Judges and verifiers](judges-and-verifiers.md): more
  components to package.
- [Write an environment service](environment-services.md) and [Write a sandbox provider](sandbox-providers.md):
  environments and sandboxes of your own.
