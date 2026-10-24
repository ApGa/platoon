# The training loop's golden trace

`test_training_trace.py` trains a few steps in each scenario below through platoon's own training code, on fakes of
what lies outside it, and compares what the trainer did with the scenario's golden files, byte for byte. It is the
safety net for changes that must not change behavior, such as moving or splitting the training code under
`platoon/train`, and it shows what a change of behavior changes.

## When it fails

The failure says which of these happened, with what to do. To run one scenario:
`uv run pytest "tests/train/parity/test_training_trace.py::test_training_trace_matches_its_golden[<scenario>]"`.

- **The trace differs from its golden files.** The run did something else. The failure lists what differs in each
  start: the first Tinker call, rollout request or log message that differs, and the metrics added, removed or
  changed. Then it shows the first lines of the diff and the `diff -u` command that shows all of it.
  - A change in `tinker` (every datum, update and save) or `requests` changes what trains. One in `metrics`,
    `messages` or `events` alone changes what the run reports or records.
  - If the change is meant, record the golden files again (below) and say in the pull request why each change is.
  - A pull request that only moves or splits code changes no golden file. It may change these files' imports when
    it moves a module they import, and nothing else here.
- **Two runs of a scenario differ.** The trace depends on more than the code: the wall clock (`time.time()`), a
  thread, or state left in the process. The diff shows where. A new metric computed from the wall clock or the
  machine belongs in `trace_format.masked_value`.
- **The scenario no longer exercises what it is there for.** Its check failed (the `_check_...` functions of
  `trace_scenarios.py`, and `check_trains`). Change its config or the fakes until it does, rather than recording a
  golden that tests less.
- **A module still holds a fake after its start.** It was first imported during a run, after the fakes went in
  place. Import it at the top of `training_trace.py`.

To compare two trees beyond the goldens, such as a branch and its base, write each tree's traces to a directory and
`diff -r` them:

```bash
uv run python tests/train/parity/record_golden.py --out /tmp/traces-base    # in the base tree
uv run python tests/train/parity/record_golden.py --out /tmp/traces-branch  # in the branch
```

## What runs

A scenario whose config has `environments` runs through `platoon train`'s entry point
(`platoon.train.tinker.train.arun_tinker_training`): the config is loaded and checked, the entries' tasks are listed,
and the trainer and its training and evaluation `GroupRolloutWorkflow`s are built as for a real run. Any other
scenario runs the trainer as a train script does, on plain task lists. Either way the real trainer, dispatcher, task
samplers, group admission rules, per-task statistics, and checkpoint, resume and evaluation code run. A scenario of
several starts runs each start on the same run directory, as a restart does. Only what lies outside the training
code is faked (`trace_fakes.py`):

| Faked | How |
|---|---|
| The Tinker service | `tinker.ServiceClient`. Its training clients record every call, and each call takes virtual time, as a round trip does. Checkpoints go through tinker-cookbook's own saving, so `checkpoints.jsonl`, and a resume from it, are a real run's |
| The policy | `register_tinker_llm` returns a policy that keeps the version as `TinkerLLM` does, without a tokenizer or renderer |
| The rollouts | At the worker-request boundary, `run_rollout_request`. Each request is answered after a virtual time with datums, rewards and outcomes drawn from a hash of the task, the rollout number and the policy version |
| An environment's tasks | The entries' `dataset_loader`, `rollout` and `task_loader` name functions of `trace_fakes.py` |
| Printing the metrics | `StatsLogger.log` records each logged step instead |

`tinker.ServiceClient`, `register_tinker_llm`, `run_rollout_request` and the clocks are replaced in every platoon
module that holds them, so a change that moves the code calling them needs no change here.

## What a golden holds

Each scenario has two golden files. `golden/<scenario>.json` holds, for each start of its run:

- `tinker`: every call of the Tinker service, in order:
  - the training client opened;
  - each forward_backward, with each loss input's type and every datum as sent (its tokens and every loss input);
  - each optimizer step;
  - each save, and each sampling client and policy version published, the rollout workers' binding included;
- `requests`: every rollout request, training's and evaluation's apart, as
  `[task id, rollout number, policy version, settings, updates applied]`. `settings` indexes the golden's
  `request_settings`, which hold everything else a request carries, the form of its id among them. "Updates
  applied" counts the optimizer steps the fake service had applied when the request came, which a full fine-tuning
  service would already serve. A scenario whose rollouts follow the published weights adds the version its datums
  carry;
- `saves`: each row the start appended to `checkpoints.jsonl`, with its loop state. The loop state names the per-task
  statistics' and the dispatcher's files by SHA-256; the dispatcher's state is hashed with the groups it kept read
  uncompressed, since their gzip bytes differ between Python builds;
- `events`: each line the start appended to `tasks/events.jsonl`: every draw, settle and fate, in order;
- `messages`: the INFO and higher messages of platoon's loggers, in order. Reported durations, the run's temporary
  directory, the import name of `trace_fakes` and the random part of request ids are masked;
- `files`: the files the run directory holds when the start ends;
- `outcome`: `completed`, or how a start that was planned to crash or stop ended.

`golden/<scenario>.metrics.json` holds each start's logged metrics, training's and evaluation's apart, one per line
under its parent path. A metric with the same value at every step holds that value; any other holds one value per
step, null where it was not logged. A metric of the wall clock or the machine holds `"timing"`, and the importance
weight statistics hold `"inexact"`. A `__count` that is 1 at every step is left out.

## Why two runs agree

A trace is a function of the code alone, whatever the machine and its load.

- The run's event loop never waits. When nothing is ready, its clock jumps to the next timer, so a rollout takes its
  virtual seconds at no cost and groups complete in the order of their durations. Each duration has a 32-bit random
  fraction, so no two rollouts end at once, and every call of the fake service takes virtual time, so a publish
  never coincides with a group's end. Events at one instant happen in the order the code schedules them.
- The clocks the training code reads (`time.perf_counter`, `time.monotonic`) follow the virtual clock while the loop
  runs, and the loop runs `run_in_executor`'s work inline, so no thread races the run.
- Rewards are 0 or 1, log-probabilities multiples of 1/16 and a trajectory's steps 1, 2 or 4, and the fake service's
  training log-probabilities are the sampling ones plus 2 on some tokens. So the sums the trainer takes are exact,
  and a value it scales (an advantage over a token count) is rounded once: no value depends on the order of a
  reduction or on the CPU's vector width. The importance weights, e ** 2, would, so the trace keeps their clip
  fractions and masks their mean and extremes.
- The stats trackers, rollout client monitors, rollout worker telemetry and sandbox labels are reset before each
  start, and the signal handlers, logging and labels the run changes are put back after it.
- The real `tinker.ServiceClient` refuses to start, `TINKER_API_KEY` is a placeholder and `TINKER_BASE_URL` an
  address where nothing listens, so no run reaches a Tinker service.

The test runs each scenario twice, and the two runs must agree before either is compared with the golden files.
Recording runs every scenario once more in a fresh process with another hash seed.

## Limits

The golden trace does not cover:

- the rollout worker processes (`rollout_isolation: process`): the worker entry point, its supervisor and the fork
  server, which the worker tests cover;
- the body of `run_rollout_request` and the converter it calls (`get_train_data_for_trajectory_collection`), which
  the fakes replace;
- the policy proxy, the renderers and the Tinker client;
- real numbers: the service's losses and gradients, and timings;
- DEBUG messages.

## The scenarios

| Scenario | What it runs |
|---|---|
| `single-loo` | One environment, no environments list: leave-one-out baselines and a minimum group size |
| `single-entry` | One named environment entry with settings of its own (`rollout_config.max_steps`), in training and evaluation |
| `single-micro` | Two minibatches of two microbatches, two groups at once, the zero-advantage filter off |
| `straggler-default` | A run-level straggler timeout with the default quorum, in groups of three |
| `depth-zero-adv` | Depth-level weighting of subagent datums, then the zero-advantage filter, in microbatches of two groups |
| `mix-default` | A mixture of two entries that set nothing of their own |
| `mix-sizes` | A mixture whose entries set their own group sizes |
| `mix-capped` | The mixture with a limit of one running group on its slow entry (`rollout_group_limits`) |
| `capped-k1` | A limit on the slow entry at `max_staleness: 1`, where draws hold their label's slots |
| `mix-label-by` | The mixture labelled by another column (`train.label_by`), with a straggler policy for one label |
| `stale-follow` | Groups dropped as stale, the slow entry by its own bound, with datums stamped by the policy they sampled |
| `rule-informative` | Group admission by the `informative` rule, whose rejected places are refilled, under the scoring `pass_rate` sampler |
| `rule-k1`, `rule-k1-held`, `rule-k1-stale` | The rule at `max_staleness: 1`: short steps, rejected groups without datums that hold their slots, stale groups |
| `pass-rate` | The `pass_rate` task sampler on the run's one entry, under the `weighted` run sampler |
| `eval-at-start` | Evaluation every second step beside training, and of the starting policy (`eval.at_start`) |
| `eval-long` | An evaluation of the starting policy that outlasts the first step |
| `cadence` | Permanent and recovery checkpoints, both due at once |
| `resume-kept` | A crash after a recovery checkpoint that kept settled groups, then a resume from it |
| `eval-resume` | A crash after a permanent checkpoint, then a resume that skips the evaluation an earlier start did |
| `stop-mid-step`, `stop-before-update` | A stop request (SIGUSR2) after a step's first update, and before it, then a resume |

## Adding a scenario

1. Add an entry to `SCENARIOS` in `trace_scenarios.py`: a description, the config sections it changes (laid over
   `BASE_CONFIG`), how its starts end (`Start`), how its rollouts end (`Rollouts`), and a check of what it is there
   for. A new setting gets a scenario for each of its values that changes what trains.
2. Keep it small: a few steps, a batch of 2 to 4 and groups of 2 to 4, so its golden files stay readable.
3. Record its golden files (below), and add it to the table above.

## Recording the golden files again

Only after a change of behavior that is meant, and in the environment the tests run in (`uv sync --dev --extra
tinker --extra ors`):

```bash
uv run python tests/train/parity/record_golden.py --write single-loo mix-capped  # the scenarios that changed
uv run python tests/train/parity/record_golden.py --write                       # every scenario
uv run python tests/train/parity/record_golden.py                               # compare only, write nothing
```

It writes nothing for a scenario whose runs differ, whose check fails, or whose trace in another process differs.
Review the golden files' diff as code.
