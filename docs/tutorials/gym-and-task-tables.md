# Gym adapters and task tables

**Goal.** Train on code you already have without writing an ORS environment class: a reset/step environment,
adapted with `pt.gym.loop` or `pt.gym.as_environment`, and a plain list of tasks (a *task table*) graded by your
own episode program.

You will build the examples in [`examples/gym-env`](../../examples/gym-env/): a hi-lo number game played two
ways, and a question table graded by exact match.

For how harnesses, episode programs and environments fit together, see
[the architecture overview](../architecture.md).

## Prerequisites

- Python 3.12 or later and [uv](https://docs.astral.sh/uv/).
- Platoon installed from the repository root: `uv sync --extra tinker --extra ors` (the `ors` extra is needed for
  `pt.gym.as_environment`).
- To run a rollout or train: `TINKER_API_KEY`.

Run every command from the example directory, with `PYTHONPATH=.` so platoon can import the example's modules:

```bash
cd examples/gym-env
```

## Which adapter to use

| | `pt.gym.loop` | `pt.gym.as_environment` |
|---|---|---|
| Makes | a function harness | an ORS `Environment` class with one `act` tool |
| The model acts by | replying: each reply's text is the action | calling `act(action)`, through any harness |
| Tasks come from | any environment service, usually a task table | the `tasks` mapping you pass |
| The episode is scored by | the `gym` program, with the return | the `act` call that ends the episode, with the return |
| An episode that runs out of budget scores | the return so far (`max_turns`, or the agent's step budget) | the return so far at the session's `max_tool_calls`; 0 (`no_submit`) when the agent's own step budget runs out first |
| Pick it when | the game is the whole agent: a plain text loop | any harness should play it through a tool, or it should run on an env server |

The *return* is the sum of the step rewards, the usual gym convention.

## Step 1: Write the step environment

A step environment has two async methods, and optionally `close()` (sync or async), which runs when the episode
ends:

```python
class StepEnv(Protocol):
    async def reset(self, task: pt.Task) -> str: ...  # the first observation
    async def step(self, action: str) -> tuple[str, float, bool]: ...  # observation, reward, done
```

[`guess_game.py`](../../examples/gym-env/guess_game.py) defines hi-lo. It reads its target from the task's
`misc`, which holds the task's extra fields (step 3):

```python
class HiLoGame:
    async def reset(self, task: pt.Task) -> str:
        self.target = int(task.misc["target"])
        return (
            f"I am thinking of a number from 1 to 100. Guess it in at most {MAX_GUESSES} tries; "
            "reply with just the number."
        )

    async def step(self, action: str) -> tuple[str, float, bool]:
        try:
            guess = int(action.strip())
        except ValueError:
            return "Reply with a single whole number.", 0.0, False
        if guess == self.target:
            return f"Correct! The number was {self.target}.", 1.0, True
        return ("Too low." if guess < self.target else "Too high."), 0.0, False
```

Test it as plain Python:

```bash
PYTHONPATH=. uv run python -c '
import asyncio, platoon as pt
from guess_game import HiLoGame
async def main():
    game = HiLoGame()
    print(await game.reset(pt.Task(misc={"target": 37})))
    for guess in ("50", "25", "37"):
        print(guess, "->", await game.step(guess))
asyncio.run(main())'
```

```text
I am thinking of a number from 1 to 100. Guess it in at most 7 tries; reply with just the number.
50 -> ('Too high.', 0.0, False)
25 -> ('Too low.', 0.0, False)
37 -> ('Correct! The number was 37.', 1.0, True)
```

## Step 2: Play it with `pt.gym.loop`

`pt.gym.loop(env_factory, *, max_turns=None, system=None)` returns a harness function; register it with
`pt.harness`:

```python
pt.harness("example-hilo-loop")(
    pt.gym.loop(HiLoGame, max_turns=MAX_GUESSES, system="You are playing a guessing game. Reply with one number.")
)
```

What the loop does, per episode:

1. Calls `env_factory()` and `reset(task)`. The observation is the first user message, after the optional
   `system` message. The task's goal is not shown, so put the instructions in the observation.
2. Samples a reply, passes its text to `step`, and sends the observation back as the next user message.
3. Ends `completed` when `step` says `done`, or `limit` after `max_turns` turns or when the agent's step budget
   runs out. Then calls `close()`.

Each step's action, observation and reward are recorded in the step's `misc["gym"]`, and the running return in
the trajectory's `misc["gym_return"]`.

The game lives inside the harness, so the environment service only supplies tasks, and the **`gym` program**
scores the episode with the recorded return. A run that crashed or timed out keeps its untrainable outcome, and
a run that recorded no return is graded by the session.

[`train_loop.yaml`](../../examples/gym-env/train_loop.yaml) takes its tasks from the task table in
[`tasks.jsonl`](../../examples/gym-env/tasks.jsonl):

```yaml
environments:
  - name: hilo
    service: table
    service_kwargs: {path: tasks.jsonl}   # relative to this file
    harness: example-hilo-loop
    program: gym
    package: guess_game                   # registers @pt.harness("example-hilo-loop")
    dataset_kwargs: {split_name: train}
    eval_dataset_kwargs: {split_name: test}
```

`program: gym` is required here: under the default program (`single`) a task table's session grades `invalid`,
and nothing trains.

```bash
PYTHONPATH=. uv run platoon config validate train_loop.yaml
PYTHONPATH=. uv run platoon rollout train_loop.yaml --task train-0     # needs TINKER_API_KEY
PYTHONPATH=. uv run platoon train train_loop.yaml
```

A rollout that finds the number in two guesses prints:

```text
rollout train-0-<id>  task train-0
agents 1  steps 2  max depth 0  reward 1.0  outcome scored reward=1.0
└─ policy <id>  steps 2  reward 1.0  outcome scored reward=1.0
```

Task-table task ids are the rows' `id` values (`train-0`), not `<split>:<index>`.

## Step 3: Serve it as an environment with `pt.gym.as_environment`

`pt.gym.as_environment(env_factory, *, name, tasks=None)` returns an openreward `Environment` subclass named
`name`. Assign it to a module-level name so `python://` can import it:

```python
HiLoEnvironment = pt.gym.as_environment(HiLoGame, name="HiLoEnvironment", tasks=tasks_by_split())
```

- `tasks` maps each split to its task specs (default: one `train` task, `{"id": "0"}`). A spec's `goal` and `id`
  become the `pt.Task` that `reset` gets, and the whole spec becomes its `misc`. `tasks_by_split()` reads the
  same `tasks.jsonl`, grouped by each row's `split`.
- The prompt is the `reset` observation, and the one tool is `act(action: str)`. Each call steps the game; its
  result is the observation, with the return so far as its reward and `metadata={"step_reward": ...,
  "return": ...}`. The call that ends the episode returns `finished=True`, which grades the session with the
  return.
- An episode that the agent leaves unfinished is graded like any ORS episode without a finished result: at the
  session's `max_tool_calls` (a service option) with the last reward, which is the return so far, as `pt.gym.loop`
  scores one that runs out of turns; otherwise, say when the agent's own step budget runs out first, 0
  (`no_submit`). To give partial credit, cap the calls with `max_tool_calls` below the agent's step budget.
- Teardown calls the game's `close()`. A `copy` fork deep-copies the game, and each copy closes its own.
- It works with openreward 0.1.125 and later.

Check it through platoon's environment service:

```bash
PYTHONPATH=. uv run python -c '
import asyncio
from platoon.envs import get_service
async def main():
    service = get_service("python://guess_game:HiLoEnvironment")
    session = await service.open(await service.get_task("train:0"), rollout_id="demo")
    print([tool.name for tool in await session.tools()])
    for guess in ("50", "37"):
        result = await session.call_tool("act", {"action": guess})
        print(result.as_text(), result.finished, result.reward)
    outcome = await session.grade()
    print(outcome.status, outcome.reward)
    await session.close()
asyncio.run(main())'
```

```text
['act']
Too high. False 0.0
Correct! The number was 37. True 1.0
scored 1.0
```

[`train_tool.yaml`](../../examples/gym-env/train_tool.yaml) plays it with the built-in `platoon-react` harness,
which calls `act` as a native tool:

```yaml
environments:
  - name: hilo
    service: python://guess_game:HiLoEnvironment
    harness: platoon-react
```

ORS task ids are `<split>:<index>`:

```bash
PYTHONPATH=. uv run platoon rollout train_tool.yaml --task train:0
```

## Step 4: Train on a task table

A task table serves tasks that have no tools and no grader. A session's instruction is the row's goal, it lists
no tools, and its grade is `invalid` (reason `self_grade`), so the episode program must set the outcome. The
`gym` program is one such program; step 5 writes another.

A row is a goal string or a mapping:

| Key | Meaning |
|---|---|
| `id` (or `task_id`) | The task id. Default: `task-<row number>`. |
| `goal` (or `instruction`, `prompt`) | The instruction. |
| `split` | The row's split. Default: `train`. |
| `max_steps` | The task's step limit, which applies when the run sets no `rollout_config.max_steps`. Only table rows carry one: an ORS task spec's `max_steps` is not read, and Harbor tasks have none. |
| any other key | Goes to the task's `misc`, for the harness and the program. |

Splits come from the `splits={split: [ids]}` argument when it is given, else from the rows' `split`, else every
row is in `train`.

There are three ways to serve one:

1. **A JSONL file**, one row per line: `service: table` with `service_kwargs: {path: tasks.jsonl}`. Each process
   reads the file once. Use this for anything beyond a handful of rows. A relative path is relative to the config
   file that names it (an override's to the working directory).
2. **Inline rows**: `service_kwargs: {rows: [...]}`. The rows travel with every task of the entry, so keep them
   few (platoon warns above 20).
3. **Python**: `task_table(rows, *, splits=None, name="table")` returns the service. Point `service:` at a function
   that builds it, for example `service: my_tasks:make_table` for `def make_table(): return task_table([...])`.

```bash
PYTHONPATH=. uv run python -c '
import asyncio
from platoon.envs import task_table
table = task_table([
    {"id": "add", "goal": "What is 2 + 3?", "answer": "5"},
    {"id": "mul", "goal": "What is 2 * 3?", "answer": "6", "split": "test"},
    "Name a prime number.",
])
async def main():
    print(await table.list_splits(), await table.list_tasks("train"))
    task = await table.get_task("add")
    print(task.id, "|", task.goal, "|", task.misc)
    session = await table.open(task, rollout_id="demo")
    outcome = await session.grade()
    print(await session.tools(), outcome.status, outcome.reason)
asyncio.run(main())'
```

```text
['train', 'test'] ['add', 'task-2']
add | What is 2 + 3? | {'answer': '5'}
[] invalid self_grade
```

## Step 5: Grade a task table with an episode program

An episode program runs the agents of one rollout and returns its outcome. It is
`async def program(ep: pt.EpisodeContext) -> pt.Outcome`, registered with `@pt.program(name)`.
[`exact_match.py`](../../examples/gym-env/exact_match.py) compares the agent's final answer with the row's
`answer` column:

```python
@pt.program("example-exact-match")
async def exact_match(ep: pt.EpisodeContext) -> pt.Outcome:
    result = await ep.run()  # the root agent, with the environment's harness
    if result.status not in ("completed", "limit"):
        return await ep.grade()  # a crash, timeout or cancellation keeps its untrainable outcome
    expected = normalize(str(ep.task.misc["answer"]))
    correct = normalize(result.final_message) == expected
    return pt.Outcome(status="scored", rewards={"reward": float(correct)}, metadata={"expected": expected})
```

- `ep.run()` runs the root agent and returns its `pt.AgentResult`.
- `ep.task.misc` holds the row's extra columns.
- `ep.grade()` grades the session, and turns a failed run into its untrainable outcome.
- A program that raises is an `error` outcome, never a score.
- For a model-graded answer, `ep.judge_endpoint()` gives an OpenAI-compatible endpoint whose calls are recorded
  for audit and never trained.

[`train_qa.yaml`](../../examples/gym-env/train_qa.yaml) pairs the questions in
[`questions.jsonl`](../../examples/gym-env/questions.jsonl) with `platoon-react`, which, with no tools to show,
answers in one reply:

```yaml
environments:
  - name: qa
    service: table
    service_kwargs: {path: questions.jsonl}
    harness: platoon-react
    harness_options:
      system_prompt: Answer with one word or number and nothing else.
    program: example-exact-match
    package: exact_match                  # registers @pt.program("example-exact-match")
```

```bash
PYTHONPATH=. uv run platoon rollout train_qa.yaml --task q0
PYTHONPATH=. uv run platoon train train_qa.yaml
```

## Reference

| API | What it does |
|---|---|
| `pt.gym.loop(env_factory, *, max_turns=None, system=None)` | A harness function that plays `env_factory()` turn by turn (step 2). |
| `pt.gym.as_environment(env_factory, *, name, tasks=None)` | An ORS `Environment` class with one `act` tool (step 3). |
| `program: gym` | Runs the root agent and scores it with the return its `pt.gym.loop` harness recorded. |
| `platoon.envs.task_table(rows, *, splits=None, name="table")` | A task-table environment service (step 4). |
| `service: table` with `service_kwargs: {path: ...}` or `{rows: [...]}` | The same, from a config. |

## Troubleshooting

- **Every task-table rollout is `invalid` with reason `self_grade`.** The program does not set the outcome. Use
  `program: gym` with `pt.gym.loop`, or your own program.
- **`... is a TOML harness spec, which runs no episode program, and a task table has no grader`.** A TOML harness
  spec runs no program, so it cannot play a task table; `platoon train`, `eval` and `rollout` refuse it. Use a
  function harness with `program: gym` or your own program.
- **`Unknown episode program 'example-exact-match'`.** Add the module that defines it to the entry's `package`.
- **`... tasks.jsonl names ..., which does not exist`.** A table's `path` in a config file is relative to that
  file's directory, and one given as an override to the working directory; move the file there, or use an absolute
  path.

## Next steps

- [Write a custom environment](custom-environment.md): an ORS environment class with several tools, grading and
  shared state for subagents.
- [Remote environments](remote-environments.md): serve environments, including `pt.gym.as_environment`
  classes, from env servers.
- [The architecture overview](../architecture.md): programs, harnesses and outcomes.
