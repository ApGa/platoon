# Platoon documentation

This page is the table of contents. New to platoon? Read the [overview](index.md), then
[install](getting-started/install.md) it and do [your first run](getting-started/first-run.md).

## Getting started

- [Overview](index.md): what platoon is and is not, how the pieces fit, and where to go next.
- [Install](getting-started/install.md): from PyPI or from source, the extras, the TextCraft extension and a
  development install.
- [Changelog](../CHANGELOG.md): what each release changes, and what a version below 1.0 may break.
- [Concepts](getting-started/concepts.md): the vocabulary, from harness to credit assignment.
- [Your first run](getting-started/first-run.md): one episode, a short training run and the trajectory viewer.
- [Train on a Harbor dataset](../README.md#train-on-a-harbor-dataset): a site profile, one config, `platoon prepare`
  and `platoon train`; [`examples/harbor-dataset/`](../examples/harbor-dataset/README.md) is the project to copy.

## Tutorials

Each tutorial has a goal, prerequisites, numbered steps and the output to expect. Most come with a runnable
example under [`examples/`](../examples/), and every example has a CPU test in
[`tests/examples/`](../tests/examples/).

- [Write a function harness](tutorials/function-harness.md): an agent loop as a Python function (`@pt.harness`).
- [Wrap a command-line agent](tutorials/cli-harness.md): an existing agent described by a TOML spec.
- [Built-in harnesses](tutorials/builtin-harnesses.md): `platoon-codeact`, `platoon-react`, `platoon-bash` and
  `mini-swe-agent`, and their options.
- [Write a custom environment](tutorials/custom-environment.md): tasks, tools and a grader.
- [Run environments on env servers](tutorials/remote-environments.md): the same environment as a service.
- [Gym adapters and task tables](tutorials/gym-and-task-tables.md): step environments and task files.
- [Subagents: agents that delegate](tutorials/subagents.md): `ctx.agents.spawn`, budgets and verifiers.
- [Credit assignment: from rewards to advantages](tutorials/credit-assignment.md): baselines and depth weighting.
- [Episode programs: compose an episode and shape its rewards](tutorials/programs.md): `@pt.program` and
  `EpisodeContext`.

## How-to guides

- [Write and override configs](how-to/configs.md): `extends:`, command-line overrides and validation.
- [Train and evaluate](how-to/train-and-eval.md): `platoon train`, `platoon eval` and their outputs.
- [Evaluate a hosted model](how-to/evaluate-hosted-models.md): `platoon eval` on a base URL that speaks your
  harness's dialect (OpenAI, Anthropic, vLLM, SGLang, OpenRouter, a LiteLLM proxy), and `inference.judges`, which
  turns judges off.
- [Models and tool calling](how-to/models-and-tool-calling.md): which tool-call formats `chat_template` reads, how
  its parser is chosen, and what happens to a call that does not parse.
- [Sandboxes and Harbor tasks](how-to/sandboxes-and-harbor.md): `process`, Docker, Singularity/Apptainer and
  enroot sandboxes, and Harbor datasets: fetched by name, held out, prepared once per site.
- [How Harbor tasks are graded](how-to/harbor-grading.md): the steps of a grade, what keeps an agent from forging its
  reward, and which reported reward trains.
- [Run on a cluster](how-to/clusters.md): PBS and Slurm job patterns.
- [Test an environment, a harness or a program on CPU](how-to/test-extensions.md): `platoon.testing`'s scripted
  policy, `run_episode`, the toy tokenizer and the pytest plugin.
- [Troubleshooting](how-to/troubleshooting.md): common errors and what to do about them.

## Extending platoon

Your own harnesses, environments, programs, reward rules, judges, sandbox providers and renderers, in your own
package.

- [Extending platoon](extending/index.md): every extension point, how platoon finds your code, and the rules every
  extension follows.
- [Publish an extension package](extending/publish-a-package.md): a harness, an episode program, a reward processor
  and tasks in one installable package, with entry points and CPU tests.
- [Write an environment service](extending/environment-services.md): tasks from a source that is not an ORS
  environment, a Harbor dataset or a task table.
- [Reward processors and finalizers](extending/rewards.md): change what each trajectory trains on.
- [Judges and verifiers](extending/judges-and-verifiers.md): verifier agents, behavior and parent-usage judges, and
  judge endpoints.
- [Write a sandbox provider](extending/sandbox-providers.md): task sandboxes on infrastructure of your own.
- [Renderers and tool-call parsers](extending/renderers.md): prompt and parse a model's chat format, and read its
  tool calls.
- [Hooks, telemetry and metrics](extending/hooks-and-metrics.md): counters, error detection and where metrics go.
- [Write a task sampler](extending/task-samplers.md) (experimental): choose which task each training group runs, by a
  registered name or `module:Class`, and check it with `check_task_sampler`.
- [Write a group admission rule](extending/group-admission.md) (experimental): decide which settled training groups
  count toward the batch, and check it with `check_group_admission`.

## Reference

Generated from the code by `scripts/gen_reference.py`; `tests/docs/test_reference_up_to_date.py` keeps it current.

- [Command line](reference/cli.md): every `platoon` command and option.
- [Config](reference/config.md): every config section and key, with its type and default.
- [Python API](reference/api.md): the names `platoon` exports.
- [Experimental API](reference/experimental.md): the names `platoon.experimental` exports, which a minor release may
  change: the task sampler API, the per-task statistics it reads, and the group admission API.

## Architecture

- [Architecture](architecture.md): how environments, harnesses, the model gateway, episode programs and the
  trainer fit together, and why.

## RFCs

- [RFC index and process](rfcs/README.md): design proposals for what comes next, their status, and
  [how to propose a change](rfcs/README.md#how-to-propose-a-change).
- [RFC template](rfcs/0000-template.md): the sections every new RFC fills in.

## Contributing

[CONTRIBUTING.md](../CONTRIBUTING.md) covers the development setup, the tests, the commit and pull request
conventions, and how to add an environment, a harness, a renderer or an example. Changes to a public contract or
to what trains, new subsystems, backends or required dependencies, renames or removals of public names, and work
that needs several pull requests or spans repositories start as an RFC: see
[when a change needs an RFC](../CONTRIBUTING.md#when-a-change-needs-an-rfc) and the
[RFC process](rfcs/README.md#how-to-propose-a-change).
