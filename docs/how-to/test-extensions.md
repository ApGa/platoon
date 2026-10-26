# Test an environment, a harness or a program on CPU

`platoon.testing` runs your extension the way training runs it, on any machine: no GPU, no Tinker server and no
model download. A scripted policy writes what the model would write; everything else is platoon's own code: the
model gateway renders, samples, parses and records every call, the harness runs, the environment answers and
grades, the episode program runs with its finalizers, and the agent tree becomes trajectories with their exact
tokens. Tests built on it run wherever pytest does, your CI included.

**Goal:** tests for the [quickstart](../../examples/quickstart/README.md)'s environment, harness and config, plus an
episode program, that pass on a laptop.

**Prerequisites:**

- platoon installed with the `tinker` and `ors` extras (see [Install](../getting-started/install.md)), and pytest.
- The quickstart example: [`examples/quickstart/`](../../examples/quickstart/). Its tests are
  [`test_guess.py`](../../examples/quickstart/test_guess.py), which this guide walks through.

## What `platoon.testing` gives you

| Name | What it is |
|---|---|
| `ScriptedPolicy(script)` | Stands in for the Tinker sampler. `script` is a list of completions, used in order, or a function of the decoded prompt that returns the next completion. `calls` records every call |
| `Completion(text, ...)` | A completion with options: `stop_reason="length"` (cut off, so the gateway emits no tool call), `routing_id`, `delay`, `logprob` |
| `tool_call(name, **arguments)` | A tool call as a Qwen3 model writes it, which the gateway parses into an OpenAI tool call |
| `rules({prefix: completion}, default=...)` | A script function that answers by the start of the last user message |
| `last_user_message(prompt)` | The last user turn of a decoded prompt, tool results included |
| `run_episode(task, service=, harness=, policy=, ...)` | Runs one task through `harness_rollout`, the path training and evaluation take, and returns an `EpisodeRun` |
| `serve_policy(policy)` | Serves the policy to `platoon rollout` and `platoon eval` run in your test process, in place of Tinker |
| `make_gateway(policy, ...)` | The real model gateway over a scripted policy, for lower-level tests |
| `toy_tokenizer()`, `toy_renderer(name)` | The tokenizer and renderer those use (see [The toy tokenizer](#the-toy-tokenizer)) |
| `pytest -p platoon.testing` | A pytest plugin with the fixtures `serve_policy`, `toy_tokenizer` and `toy_renderer`; it loads only when you ask for it |

The [reference](#reference) lists every argument.

## Steps

### 1. Script the model

A test decides what the model writes. `ScriptedPolicy([...])` answers the calls in order, which suits one agent:

```python
policy = ScriptedPolicy([tool_call("guess", number=50), tool_call("guess", number=42)])
```

`ScriptedPolicy(fn)` answers each call with `fn(prompt)`, where `prompt` is the rendered prompt, decoded. That suits
policies that react to what the environment said, and agents whose calls interleave (subagents, concurrent tools),
where call order is not fixed. The quickstart's tests use a bisection policy that reads the environment's answers
from the prompt:

```python
import asyncio
import json
import re
from pathlib import Path

import guesser  # noqa: F401  (importing it registers the example-guesser harness)

import platoon as pt
from platoon.cli import main as platoon_cli
from platoon.testing import ScriptedPolicy, run_episode, tool_call

GUESS_ENV = "python://guess:GuessTheNumber"
TRAIN_CONFIG = Path(__file__).with_name("train.yaml")


def bisect(prompt: str) -> str:
    """A policy: the middle of the range that the environment's answers so far leave."""

    low, high = 1, 100
    guesses = [int(number) for number in re.findall(r'"number": (\d+)\}', prompt)]
    answers = re.findall(r"<tool_response>\s*(higher|lower)\s*</tool_response>", prompt)
    for guess, answer in zip(guesses, answers):
        low, high = (guess + 1, high) if answer == "higher" else (low, guess - 1)
    return tool_call("guess", number=(low + high) // 2)
```

The prompts are in the Qwen3 chat format: each turn is `<|im_start|>role\n...<|im_end|>`, and a tool result is a user
turn wrapped in `<tool_response>` tags. A list script that runs out fails the call with `ScriptExhaustedError`,
which names the prompt it could not answer.

### 2. Test an environment

Pair your environment with a built-in harness and a script of tool calls. `run_episode` takes the keys of one
`environments` entry of a config (`service`, `harness`, `program`, `agents`, `harness_options`, `service_kwargs`) and
a task: a `Task`, or a task id the environment resolves, as `platoon rollout --task` does:

```python
def test_the_environment_grades_a_correct_guess():
    # Task train:41 hides 42. The built-in platoon-react harness makes the tool calls the script writes, in order.
    policy = ScriptedPolicy([tool_call("guess", number=50), tool_call("guess", number=42)])
    run = asyncio.run(run_episode("train:41", service=GUESS_ENV, harness="platoon-react", policy=policy))

    assert run.reward == 1.0 and run.outcome["status"] == "scored"
    assert "lower" in policy.calls[1].last_user_message  # what the environment answered to 50
```

`run` is an `EpisodeRun`: `run.root` and `run.children` are the trajectories, `run.outcome` is the graded outcome,
`run.reward` the root's reward, and `run.interactions` the exact prompt and sampled tokens of every call, which is
what trains. A grader that raises, or a harness that crashes, gives an `error` outcome rather than a reward.

### 3. Test a harness

Run your harness on an environment with a policy that exercises it, then check what it did. The trajectory's
`misc["tool_calls"]` records every tool call: its `name`, its `arguments` and the result's `text`, among others:

```python
def test_the_guesser_finds_the_secret_by_bisection():
    run = asyncio.run(run_episode("train:41", service=GUESS_ENV, harness="example-guesser", policy=bisect))

    guesses = [call["arguments"]["number"] for call in run.root.misc["tool_calls"]]
    assert guesses == [50, 25, 37, 43, 40, 41, 42] and run.reward == 1.0


def test_the_guesser_stops_at_its_step_budget():
    policy = ScriptedPolicy(lambda prompt: tool_call("guess", number=1))
    run = asyncio.run(run_episode("train:41", service=GUESS_ENV, harness="example-guesser", policy=policy, max_steps=3))

    assert len(policy.calls) == 3 and run.reward == 0.0
```

A harness is found by name once its module is imported, which `import guesser` at the top of the file does; an
installed package's components are also found by the names their entry points give. For a harness that spawns
subagents, pass `agents=` as in a config (`{"child_grading": "env", "max_depth": 1}`, say), and script with a
function: `rules({...})` answers each agent by its own last message.

### 4. Test an episode program

Name your program with `program=`. Its finalizers run on the built trajectories before `run_episode` returns, as
they do in training:

```python
@pt.program("example-count-guesses")
async def count_guesses(ep: pt.EpisodeContext) -> pt.Outcome:
    """An episode program: run the root agent, grade it, and record its number of guesses on its trajectory."""

    def record(collection: pt.TrajectoryCollection) -> None:
        root = next(iter(collection.trajectories.values()))
        root.misc["guesses"] = len(root.misc.get("tool_calls", []))

    await ep.run()
    ep.add_finalizer(record)
    return await ep.grade()


def test_the_program_records_the_number_of_guesses():
    run = asyncio.run(
        run_episode(
            "train:41", service=GUESS_ENV, harness="example-guesser", program="example-count-guesses", policy=bisect
        )
    )

    assert run.root.misc["guesses"] == 7 and run.reward == 1.0
```

### 5. Test a config through the command line

`serve_policy` makes `platoon rollout` and `platoon eval`, called in your test process, sample the scripted policy
instead of Tinker. Their gateway takes the config's model name, renderer and `renderer_kwargs`, so the config must
name a Qwen3 renderer such as `qwen3_instruct` (or pass `renderer=` to `serve_policy`). As a fixture, from the
plugin, it ends with the test:

```python
def test_platoon_rollout_runs_the_training_config(serve_policy, tmp_path, capsys):
    # The serve_policy fixture (pytest -p platoon.testing) serves the policy in place of Tinker.
    serve_policy(bisect)
    platoon_cli(["rollout", str(TRAIN_CONFIG), "--task", "train:41", "--output-dir", str(tmp_path), "--json-output"])

    summary = json.loads(capsys.readouterr().out)
    assert summary["reward"] == 1.0 and summary["total_steps"] == 7
```

Outside pytest, `with serve_policy(policy) as scripted:` does the same for the block. It also sets `TINKER_API_KEY`
to a placeholder while none is set, since the commands check for a key before they start. It does not reach an
evaluation that runs its rollouts in worker processes (`use_subprocesses`), and it leaves out the config's
`context_window_length`, whose token counts the toy tokenizer does not match; pass `context_window_length=` to test
one. The config's `sample_timeout_seconds` applies, unless you pass one.

A config that names an installed package's components by the names their entry points give, such as
`env: example-arithmetic@1`, runs the same way once the package is installed: the example package of [Publish an
extension package](../extending/publish-a-package.md#step-8-test-on-cpu) tests its named environment so.

### 6. Run the tests

```bash
cd examples/quickstart
PYTHONPATH=. uv run pytest -p platoon.testing test_guess.py
```

Expected output, after pytest's session header (most of the time goes to the first test, which imports platoon's
rollout stack):

```text
test_guess.py .....                                                      [100%]

============================== 5 passed in ...s ===============================
```

`-p platoon.testing` enables the plugin's fixtures; installing platoon never enables them by itself. To enable them
for a whole project, put `pytest_plugins = ["platoon.testing"]` in its top-level `conftest.py`. Tests that use only
`run_episode`, `ScriptedPolicy` and `make_gateway` need no plugin.

## The toy tokenizer

The scripted policy has to turn text into the tokens a server would return, and the gateway has to render prompts,
so tests need a tokenizer. A real model's tokenizer means a download, so `platoon.testing` ships a 7 KB one:

- **Byte-level.** Token `i` is byte `i` for `i` from 0 to 255, so any text round-trips exactly, and the ids of ASCII
  text are its character codes.
- **Qwen3's chat tokens.** Ids 256 to 264 are `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<tool_call>`,
  `</tool_call>`, `<tool_response>`, `</tool_response>`, `<think>` and `</think>`, with Qwen3's `special` flags.
- **Stock renderers.** tinker-cookbook's `qwen3_instruct` (the default), `qwen3` and `qwen3_disable_thinking` work
  over it unchanged and render the same prompt text as over a Qwen3 tokenizer. Only the token ids and counts
  differ.

So a test of what platoon or your extension does with a model call needs no model. Two kinds of test still need the
real tokenizer: tests of a model's own chat template (platoon's `chat_template` renderer reads the template from the
tokenizer), and tests that compare token ids or counts with a real model's. Renderers of other model families need
their own special tokens, which the toy tokenizer does not have.

`toy_tokenizer()` returns a `transformers` fast tokenizer, loaded once per process. The files are in
`platoon/testing/toy_tokenizer/`, so `AutoTokenizer.from_pretrained(TOY_TOKENIZER_DIR)` loads the same tokenizer,
and tinker-cookbook's `get_tokenizer("platoon/toy")` finds it once `toy_tokenizer()` has run.

## Lower level: the gateway

`make_gateway(policy)` returns the real `ModelGateway`, which `ctx.chat` and every HTTP dialect go through, with the
scripted policy as its sampler. Use it to test code that talks to a gateway directly, such as a custom rollout or
a transport:

```python
import asyncio

from platoon.gateway.core import ChatRequest
from platoon.testing import ScriptedPolicy, make_gateway

policy = ScriptedPolicy(["Hello."])
gateway = make_gateway(policy, max_tokens=64)
agent = gateway.open_agent(rollout_id="r", trajectory_id="t", interactions={})
reply = asyncio.run(gateway.chat(agent, ChatRequest.from_params([{"role": "user", "content": "Hi"}], {})))
assert reply.content == "Hello." and policy.calls[0].sampling_params.max_tokens == 64
```

To run a whole episode on such a gateway (other sampling defaults, a context window), pass it to `run_episode` as
`policy=`.

## Check a task sampler

A task sampler of your own ([Write a task sampler](../extending/task-samplers.md)) is checked against the sampler
contract by `check_task_sampler`, which needs neither a model nor a training run:

```python
from platoon.testing import check_task_sampler


def test_my_sampler_keeps_the_contract():
    check_task_sampler("my_pkg.samplers:MySampler", options={"keep": 0.5}, levels=("entry",))
```

It builds the sampler (a class, a registered name, or `module:Class`) with `options` checked against its schema, at
each of `levels`, over a pool of 40 tasks in two entries with synthetic per-task statistics, each context built as the
dispatcher builds it. It asks an adaptive sampler 200 decisions under each of three eligibility patterns (every task
eligible; one label at its limit, which leaves one entry half its tasks; nothing eligible) and 200 under the three in
turn, and a static one 200 with every task eligible. It raises one `AssertionError` that names every check that
failed: `protocol`, `options`, `pool-task-or-nothing`, `fast` (the median decision on a pool of 4,096 tasks within
`budget_ms`, 20 ms by default; None skips it, as a test that shares its machine may), `deterministic`, `static`,
`state`, `observe` and `metrics`. Given by reference, the sampler's decisions are also replayed in two processes with
other hash seeds, so put the package on `PYTHONPATH` (an installed or editable package is). `check_task_sampler`
checks the experimental sampler API (`platoon.experimental`), and changes with it.

## Check a group admission rule

A group admission rule of your own ([Write a group admission rule](../extending/group-admission.md)) is checked against
its contract by `check_group_admission`, which needs neither a model nor a training run:

```python
from platoon.testing import check_group_admission


def test_my_rule_keeps_the_contract():
    check_group_admission("my_pkg.rules:MyRule", options={"limit": 4})
```

It builds the rule (a class, a registered name, or `module:Class`) with `options` checked against its schema, at each
of `levels` (`("run", "entry")` by default; `("entry",)` for a rule only an entry may set), over `check_task_sampler`'s
pool of 40 tasks in two entries and its synthetic per-task statistics, and asks it about 400 settled groups of every
verdict and shape: with and without valid members, without members, without labels or an entry. It raises one
`AssertionError` that names every check that failed: `protocol` (it builds at each level, with entries and without,
and its `check` returns or raises `ValueError`), `options`, `answers` (an `Admission` or a `bool`, a reason of at most
64 characters, nothing raised), `pure` (the rule keeps no state: an instance asked twice and another in reverse order
answer alike; asked ten times over, the first answers as an instance built for each group does, and over another
view of the statistics as a new instance does) and `fast` (the median decision on the statistics of 4,096 tasks within
`budget_ms`, 2 ms by default; None skips it). Given by reference, the rule's answers are also replayed in two processes
with other hash seeds, so put the package on `PYTHONPATH`. No check proves a rule pure: state that shows only after
more calls, or on other inputs, passes. `check_group_admission` checks the experimental admission API
(`platoon.experimental`), and changes with it.

## Check a trainer

A trainer class of your own (`platoon.experimental.TrainerBase`, or any class of the `Trainer` protocol) is checked
against the lifecycle every trainer goes through by `check_trainer`, on CPU, with neither a model nor a Tinker service:

```python
from platoon.testing import check_trainer


def test_my_trainer_keeps_the_lifecycle():
    check_trainer(MyTrainer)
```

It runs the trainer through RFC 0021's lifecycle (built from its setup, entered with `async with`, run; `platoon train`
itself still builds the rl trainer with its constructor and calls its `train()`, which runs the same lifecycle), on a
fake Tinker service whose saves the run's `checkpoints.jsonl` records as a real service's are, and, for a trainer that
rolls out, over 16 tasks whose every group is one datum with a nonzero advantage. It raises one `AssertionError` that names every
check that failed: `trains` (a run of two steps saves its launch checkpoint at step 0 and its final one at step 2, each
row with the run's counters), `interrupts` (a stop request in the third optimizer step of a run of four ends it at a
step boundary, with `<step>_interrupted` saved there) and `resumes` (the next start resumes from that row and trains to
the run's end). Call it outside a running event loop; it puts back the signal handlers it found. `check_trainer`
checks the experimental trainer API (`platoon.experimental`), and changes with it.

## Reference

| Name | Signature and behavior |
|---|---|
| `ScriptedPolicy` | `ScriptedPolicy(script=(), *, tokenizer=None, stop_token=None, logprob=-0.25)`. `script`: a sequence of `str` or `Completion`, or a function `(prompt: str) -> str \| Completion`. `tokenizer`: set by `make_gateway` when None. `stop_token`: appended to a completion that stops (default: the tokenizer's end-of-sequence token, `<|im_end|>`). Attributes: `script`, `calls` (a list of `PolicyCall`), `prompts` (the decoded prompts), `remaining` (the unused part of a list script; None for a function) |
| `PolicyCall` | One answered call: `prompt` (the `tinker.ModelInput`), `text` (decoded), `sampling_params`, `completion`, `tokens`, and the property `last_user_message` |
| `Completion` | `Completion(text, stop_reason="stop", routing_id=None, delay=0.0, logprob=None)` |
| `ScriptExhaustedError` | Raised, as an `AssertionError`, by a call that a list script has no completion for |
| `tool_call` | `tool_call(name, arguments=None, /, **kwargs) -> str` |
| `rules` | `rules(table, *, default=None)`: the reply of the first key that the last user message starts with; without a match, `default`, or an `AssertionError` when there is none |
| `last_user_message` | `last_user_message(prompt) -> str`: `""` when the prompt has no user turn |
| `sample_response` | `sample_response(tokens, *, stop_reason="stop", routing_id=None, logprob=-0.25)`: a `tinker` `SampleResponse`, for samplers of your own |
| `make_gateway` | `make_gateway(policy=(), *, tokenizer=None, renderer=None, renderer_kwargs=None, model_name="platoon/toy", **settings) -> ModelGateway`. `renderer`: a renderer or its name (default `qwen3_instruct`); `settings`: `ModelGateway` arguments such as `max_tokens`, `temperature` or `context_window_length` |
| `serve_policy` | `serve_policy(policy=(), *, tokenizer=None, renderer=None, **settings)`: a context manager that yields the `ScriptedPolicy` |
| `run_episode` | `async run_episode(task, *, service, harness, policy=(), service_kwargs=None, harness_options=None, program="single", agents=None, max_steps=None, rollout=None, output_dir=None) -> EpisodeRun`. `rollout`: a `RolloutConfig` or a mapping of its fields; `output_dir`: default a new temporary directory |
| `EpisodeRun` | `collection`, `interactions`, `gateway`, `output_dir`, and the properties `trajectories` (the root's first), `root`, `children`, `outcome`, `reward` and `policy` |
| `toy_tokenizer`, `toy_renderer` | `toy_tokenizer()`; `toy_renderer(name="qwen3_instruct")` |
| `TOY_MODEL_NAME`, `TOY_RENDERER_NAME`, `TOY_TOKENIZER_DIR` | `"platoon/toy"`, `"qwen3_instruct"`, and the directory of the toy tokenizer's files |
| `check_task_sampler` | `check_task_sampler(sampler, *, options=None, levels=("entry", "run"), budget_ms=20.0)`: raises `AssertionError` naming each check a task sampler fails ([Check a task sampler](#check-a-task-sampler)) |
| `check_group_admission` | `check_group_admission(rule, *, options=None, levels=("run", "entry"), budget_ms=2.0)`: raises `AssertionError` naming each check a group admission rule fails ([Check a group admission rule](#check-a-group-admission-rule)) |
| `check_trainer` | `check_trainer(trainer, *, options=None)`: raises `AssertionError` naming each check a trainer fails ([Check a trainer](#check-a-trainer)) |

The names `platoon.testing` exports are a stable API, but for `check_task_sampler`, `check_group_admission` and
`check_trainer`, which check the experimental sampler, admission and trainer APIs and change with them; its submodules
are not.
