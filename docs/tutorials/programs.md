# Episode programs: compose an episode and shape its rewards

**Goal.** Write an episode program (`@pt.program`) that runs the root agent, grades it, and post-processes the
rollout's trajectories with a finalizer: here, one that shapes number-search rewards toward fewer guesses and asks
a judge for a critique, by a penalty that the config sets as the program's option. Then run it, check it on CPU and
train with it.

**Prerequisites.**

- Platoon installed with the `tinker` and `ors` extras. Number search, the environment here, ships with platoon.
- The example used throughout: [`examples/program-reward-shaping/`](../../examples/program-reward-shaping/).
- For the run in step 4, a Tinker API key (`TINKER_API_KEY`). Step 5 runs the example on CPU without one.

[`docs/architecture.md`](../architecture.md) explains where programs sit between environments, harnesses and the
trainer.

## What a program is

An environment's `program:` decides how one task's rollout runs its agents. The default, `single`, runs the
environment's harness on the root agent and grades the session:

```python
@pt.program("single")
async def single(ep: pt.EpisodeContext) -> pt.Outcome:
    await ep.run()
    if ep.config.skip_subagent_reward_computation:
        ep.add_finalizer(zero_subagent_rewards)
    if ep.config.propagate_root_success:
        ep.add_finalizer(propagate_root_success)
    return await ep.grade()
```

A program is an `async` function that takes an `EpisodeContext` and returns the episode's `pt.Outcome`. It runs in
the rollout worker, next to the model gateway, as trusted code, and it fails closed: a program that raises gives
the episode an `error` outcome, which never trains as reward 0.

| `EpisodeContext` member | What it is |
|---|---|
| `task`, `session` | The task and its environment session (`session.instruction` is the goal) |
| `options` | The program's options, as the environment entry sets them in `program_options`, parsed against the schema the program declares (`@pt.program(name, options=...)`), defaults included; None for a program that declares none |
| `config` | The rollout config (in training, `train.workflow_config.rollout_config`) |
| `route` | The environment entry as the rollout sees it: harness, `harness_options`, `agents`, `program`, `program_options` |
| `root` | The root agent's record: its `misc`, `result` and calls |
| `run(harness=None, *, goal=None, options=None)` | Run the root agent once (default: the route's harness, on the instruction) and return its `AgentResult` |
| `grade()` | Grade the session, failing closed (a run that crashed or timed out gets its untrainable outcome) |
| `add_finalizer(fn)` | Run `fn(collection)` (sync or async) on the rollout's `TrajectoryCollection` once it is built |
| `judge_endpoint()` | An OpenAI-compatible endpoint on a `judge` agent under the root: recorded for audit, never trained |
| `snapshot()` | The agent tree recorded so far, as a read-only `TrajectoryCollection`, for example for a judge that reads a subagent's work mid-episode |
| `behavior_judge` | Set before `run()` to audit verified subagents ([Subagents](subagents.md#verifiers-and-judges)) |
| `tool_transform` | Set before `run()` to rewrite the tool listing every agent sees |

## Steps

### 1. Write the program

The example's program runs and grades the root like `single`, and registers a finalizer. It declares one option,
the penalty per extra guess, with `options=`:

```python
# examples/program-reward-shaping/efficient_search.py
@dataclass(frozen=True)
class SearchOptions:
    """The program's options, which an environments entry sets in ``program_options``."""

    guess_penalty: float = 0.1  # reward lost per guess beyond binary search's

    def __post_init__(self) -> None:
        if not 0 <= self.guess_penalty <= 1:
            raise ValueError(f"guess_penalty: must be between 0 and 1, got {self.guess_penalty}")


@pt.program("example-efficient-search", options=SearchOptions)
async def efficient_search(ep: pt.EpisodeContext[SearchOptions]) -> pt.Outcome:
    await ep.run()
    ep.add_finalizer(
        EfficiencyShaping(
            goal=ep.session.instruction,
            judge=ep.judge_endpoint(),
            penalty=ep.options.guess_penalty,  # program_options in the config, checked when it loads
        )
    )
    return await ep.grade()
```

The schema's fields are the keys a config may set, and their defaults apply to the keys it leaves out; the program reads
them, parsed, as `ep.options`, and `pt.EpisodeContext[SearchOptions]` types them. The returned outcome is the episode's
grade: the root trajectory's outcome, and the reward of every subagent that has no grade of its own. A program may also
build its own `pt.Outcome`, as the built-in `gym` program does with the return its harness recorded.

### 2. Write the finalizer

Finalizers run after the trajectories are built and before they are returned to the trainer, in the order they
were added (adding the same finalizer twice runs it once). This one lowers the root's reward by `penalty` for every
guess beyond what a binary search needs, and stores a judge's one-line critique:

```python
@dataclass
class EfficiencyShaping:
    goal: str
    judge: pt.ModelEndpoint
    penalty: float = 0.1

    async def __call__(self, collection: pt.TrajectoryCollection) -> None:
        root = next(iter(collection.trajectories.values()))  # the root's trajectory comes first
        if trajectory_was_interrupted(root):
            return  # an untrainable outcome (timeout, cancellation, infra error) stays out of training as it is
        guesses = [call for call in root.misc.get("tool_calls", []) if call.get("name") == "guess"]
        extra = max(0, len(guesses) - binary_search_guesses(self.goal))
        root.reward = root.reward * max(0.0, 1.0 - self.penalty * extra)
        if root.steps:
            reward_misc = root.steps[-1].misc.setdefault("reward_misc", {})
            reward_misc["reward/extra_guesses"] = float(extra)
            reward_misc["reward/shaped"] = root.reward
        root.misc["judge_critique"] = await self.critique(guesses)
```

Four rules keep shaping safe:

1. **Leave untrainable trajectories alone.** `trajectory_was_interrupted` is true for timeouts, cancellations and
   invalid or failed measurements; they train nothing whatever their reward says.
2. **Change `trajectory.reward`, keep `reward/success`.** Training reads the trajectory's reward (through the
   environment's `reward_processor`, which by default returns it unchanged); `platoon eval` counts success from
   the last step's `reward/success`, which the trace set from the grade.
3. **Shaping the root moves the baseline.** Group baselines are computed from root rewards
   ([Credit assignment](credit-assignment.md)), so a shaped root reward changes every advantage in its group.
4. **Order matters.** `propagate_root_success` sets every trajectory's reward, the root's included, to the root's
   `reward/success`: added after a shaping finalizer it undoes the shaping. A custom program adds the finalizers
   it wants in the order it needs them; `single` adds only the propagation.

The critique goes through `ep.judge_endpoint()`, an OpenAI-compatible endpoint any client can call:

```python
async def critique(self, guesses: list[dict]) -> str:
    moves = "; ".join(f"{call.get('arguments', {}).get('number')}: {call.get('text', '')}" for call in guesses)
    client = openai.AsyncOpenAI(
        base_url=self.judge.base_url, api_key=self.judge.api_key, default_headers=self.judge.headers("judge")
    )
    try:
        reply = await client.chat.completions.create(
            model=self.judge.model,
            messages=[
                {
                    "role": "user",
                    "content": f"Task: {self.goal}\nGuesses and feedback: {moves}\n"
                    "In one sentence, how could this search have used fewer guesses?",
                }
            ],
            max_tokens=64,
        )
    finally:
        await client.close()
    return reply.choices[0].message.content or ""
```

The judge samples the policy being trained, through the gateway. Its calls are kept under the root trajectory's
`judge_calls` for audit and are never trained, and they cost no agent steps. The endpoint stays served while
finalizers run. Because the judge is the policy itself, the example keeps its verdict out of the reward; a reward
from a judge you trust (another model, a rubric checker) can be added in the same finalizer.

### 3. Point the environment at the program

```yaml
# examples/program-reward-shaping/efficient_search.yaml (abridged)
environments:
  - name: number-search
    service: python://platoon.envs.number_search.env:NumberSearchEnvironment
    harness: platoon-react
    program: example-efficient-search
    program_options:
      guess_penalty: 0.1          # the program's option: reward lost per guess beyond binary search's
    package: efficient_search     # imported first, so @pt.program("example-efficient-search") registers
    reward_processor: efficient_search:reward_parts   # train on the shaped reward, report its parts
    agents:
      max_depth: 0                # one agent per episode

train:
  workflow_config:
    rollout_config:
      max_steps: 12
```

`program:` takes a registered name or a `module:function` reference, and `program_options:` sets its options, next
to it. They are checked when the config loads, by `platoon config validate` and before `platoon rollout`, `eval` or
`train` runs anything: a key the program does not take fails with exit code 78 and a did-you-mean
(`environments.0.program_options.guess_penalt: unknown key; did you mean environments.0.program_options.guess_penalty?`),
and so does a value the schema refuses (`guess_penalty: 2`). `platoon program show example-efficient-search` lists
the options with their types, defaults, the values they take and what they do, a command-line override sets one as
it sets any key (`environments.0.program_options.guess_penalty=0`), and the run's `lock.json` records the options
each entry ran with, defaults included.

### 4. Run one task

```bash
export PYTHONPATH=$PWD/examples/program-reward-shaping
uv run platoon rollout examples/program-reward-shaping/efficient_search.yaml --task train:0
```

For a search that took 11 guesses where binary search needs 10 (task `train:0` asks for a number between 6 and
988), the root's trained reward is 0.9 while its graded outcome stays 1.0:

```text
rollout train-0-7f22b5df5b56  task train:0
agents 1  steps 11  max depth 0  reward 0.9  outcome scored reward=1.0
└─ policy 2b29db57  steps 11  reward 0.9  outcome scored reward=1.0
```

In the run directory's `trajectory_collection.json`, the root's last step carries the parts of the reward:

```json
{
  "reward/success": 1.0,
  "reward/extra_guesses": 1.0,
  "reward/shaped": 0.9
}
```

and its `misc` carries `judge_critique` and the judge's call under `judge_calls`.

### 5. Check it on CPU

The example's test runs `platoon rollout` on the config with a scripted policy (one wasted guess, then a binary
search) in place of the Tinker model:

```bash
uv run pytest tests/examples/test_program_reward_shaping_example.py -q
```

```text
..                                                                       [100%]
2 passed in ...s
```

### 6. Train

```bash
uv run platoon train examples/program-reward-shaping/efficient_search.yaml
```

The example's environment also names a reward processor, `efficient_search:reward_parts`. It returns the
trajectory's (shaped) reward with the last step's `reward/*` parts as components, which training reports as
metrics:

```python
def reward_parts(trajectory: dict[str, Any]) -> tuple[float, dict[str, float]]:
    steps = trajectory.get("steps") or []
    parts = ((steps[-1].get("misc") or {}).get("reward_misc") or {}) if steps else {}
    components = {key: float(value) for key, value in parts.items() if key.startswith("reward/")}
    return float(trajectory["reward"]), components
```

A reward processor decides the training reward, so it must return the shaped `trajectory["reward"]`; one that
returns `reward/success` would undo the shaping.

## Reference

- **The root runs once.** A second `ep.run()` raises. Programs run function harnesses (`@pt.harness`); an
  environment whose harness is a TOML spec runs it directly, without a program.
- **Failures.** A program that raises gives an `error` outcome (failure kind `unexpected:<Type>`); a finalizer that
  raises gives one with failure kind `finalizer:<Type>`, and the rollout's trajectories are rebuilt untrainable. A
  rollout parses the program's options again before the program runs, and options it does not take, which only a
  route built without the config's check can carry, give an `error` outcome with failure kind `program_options`.
- **Deadlines.** When the root hits the rollout's `timeout`, `run()` returns `AgentResult("timeout")`: subagents
  that already finished keep their grades, and those still running are cancelled.
- **Built-in programs.** `single` (the default) and `gym`, which scores a `pt.gym.loop` harness by the return it
  recorded.
- **Registration.** `@pt.program(name)` registers when its module is imported, so name that module in the
  environment's `package:`. Rollout worker processes import it too. Since 0.2.0, a program has a
  version, `1` unless `@pt.program(name, version=...)` says otherwise, which a config may pin with
  `program: name@version`; and an installed package can declare a `platoon.programs` entry point named after the
  program instead, which platoon imports when a config names it ([Publish an extension
  package](../extending/publish-a-package.md#step-5-declare-the-entry-points)).
- **Options and checks.** `@pt.program(name, options=Schema)` declares the program's options: a dataclass (each
  unknown key refused with a did-you-mean, each wrong type with its path, an enum or a path field built as its class,
  and a `ValueError` its `__post_init__` raises reported under the field it names), a pydantic model (an unknown key
  refused with a did-you-mean too, unless the model keeps such keys with `extra="allow"`), or a function that takes
  the options' mapping, returns them parsed and raises `ValueError`. `platoon program show` prints each option's
  description from a pydantic field's `description`, or a dataclass field's `metadata["description"]`
  (`field(default=0.1, metadata={"description": "..."})`). A program without a schema takes no options, and an entry
  that sets some for it is refused, as is one that sets them on an entry without a harness, which runs no program.
  An entry's `eval_overrides.program_options` merge over its `program_options`, key by key, for its evaluation
  rollouts (every split but `train`). `check=` takes a plain function `check(options, ctx)` (not an `async def`),
  which refuses an entry the program cannot serve by raising `ValueError`: it runs when a training or an evaluation
  config loads, once for each phase the command runs (`train`, and `eval` when `platoon train` evaluates), with that
  phase's options and a `pt.CheckContext` (`environment`, the entry; `phase`; `rollout`, the phase's rollout config
  for the entry), and its message becomes a configuration error that names the entry, the program and the phase.
  `platoon train`, `eval` and `rollout` run it before their `--prepare` prepares anything; `platoon prepare`, which
  loads only a config's environments, and `platoon config validate` of a file of environments alone, check the
  options but run no check, since the run's rollout settings are not there to read. A check reads the config only,
  and must be quick. A named environment that sets its `program` may set its `program_options`, and its
  `eval_overrides.program_options`, too, which apply beneath the entry's own while the entry that selects it keeps
  that program.
- **The root's role** (experimental). `@pt.program(name, root_role="generator")` gives the root agent of the
  program's episodes (`ep.root`, which `ep.run()` runs) that role, and so its policy and its tools, for its whole
  life, unless the entry names its own (`agents.root_role`). The config binds the role to a policy
  (`agents.roles.<role>.policy`), and without a binding the root samples `agents.policy`; without either setting the
  root's role is `policy`. Any role will do but `verifier` and `judge`, the roles of the agents platoon opens to judge
  others ([Give an entry's root agent a role](../how-to/train-and-eval.md#give-an-entrys-root-agent-a-role-experimental)).

## Next steps

- [Credit assignment](credit-assignment.md): how shaped rewards become advantages.
- [Subagents](subagents.md): programs can also set `ep.behavior_judge` for verifier grading of subagents.
- [Reward processors and finalizers](../extending/rewards.md) and [Judges and
  verifiers](../extending/judges-and-verifiers.md): the reward rules and judges a program can use.
