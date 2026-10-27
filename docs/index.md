# Platoon

Platoon trains LLM agents with reinforcement learning: single agents, multi-agent systems, and recursive agents
that delegate to copies of themselves. You bring an environment (tasks, tools and a grader) and an agent loop;
platoon runs episodes, records every model call exactly, assigns credit across the whole agent tree, and trains
through a [Tinker](https://thinkingmachines.ai/tinker/)-compatible training backend: hosted Tinker, or a
self-hosted server such as [miles-tinker](https://github.com/neulab/miles-tinker).

```python
import platoon as pt


@pt.harness("example-guesser")
async def guesser(ctx: pt.AgentContext) -> pt.AgentResult:
    reply = await ctx.chat([{"role": "user", "content": ctx.instruction}])  # sampled and recorded for training
    return pt.AgentResult("completed", final_message=reply.text)
```

```bash
platoon train my_run.yaml
```

## What platoon is

- **An RL trainer for agents.** `platoon train` runs groups of episodes per task, turns each episode's agent tree
  into training data, and centers rewards within the group (GRPO-style, optionally leave-one-out).
- **A recording model gateway.** Every model call, from Python (`ctx.chat`) or from any OpenAI-compatible client
  over HTTP, is rendered, sampled, parsed and recorded with its exact tokens, so training uses what the model
  actually sampled.
- **A home for any agent loop.** A harness is a Python function, or an existing command-line agent described in a
  TOML file. Built-ins cover CodeAct (`platoon-codeact`), tool calling (`platoon-react`) and shell agents
  (`platoon-bash`, `mini-swe-agent`).
- **Native subagents.** An agent spawns subagents with `ctx.agents.spawn`, under step budgets and depth limits;
  each subagent gets its own trajectory and credit, and its own grade with `child_grading: env` or `verifier`.
- **Environment-agnostic.** Environments are OpenReward (ORS) classes, run in process or on environment servers,
  Harbor datasets graded by their own tests, step environments through `pt.gym`, or your own service.
- **One command line.** `platoon train`, `eval`, `rollout`, `prepare`, `config`, `harness`, `env` and `viz`.

## What platoon is not

- **Not a model server or a GPU trainer.** Platoon samples and trains through a service that implements the
  Tinker API. It runs environments and harnesses on CPU and needs no GPU of its own.
- **Not an agent framework you must adopt.** Your harness owns its loop; platoon owns recording, budgets,
  identity and outcomes.
- **Not an environment library.** It ships one example environment, number search; TextCraft is a first-party
  extension in the repository. Real tasks come from your environments or from existing benchmark collections.
- **Not a sandbox for model-written code.** `platoon-codeact` runs the model's Python on the rollout host.
  Harnesses that execute model code declare it (`executes_model_code`), and the flag is recorded with every
  rollout. Use task sandboxes (Docker, Singularity/Apptainer, enroot) for untrusted shell work.

## How the pieces fit

```text
┌─ one episode ──────────────────────────────────────────────────────┐
│                                                                    │
│  environment service ─► session: instruction, tools, grade, fork   │
│        ▲                                                           │
│        │ tool calls: ctx.call_tool, or MCP through the gateway     │
│        │                                                           │
│  harness: the root agent and its subagents                         │
│        │                                                           │
│        │ model calls: ctx.chat, or HTTP with the agent's key       │
│        ▼                                                           │
│  model gateway: render ─► sample ─► parse ─► record                │
│                              │                  │                  │
└──────────────────────────────┼──────────────────┼──────────────────┘
                               ▼                  ▼
                       Tinker-compatible          trainer: trajectories ─► datums,
                       training backend   ◄──     group baselines and advantages,
                       (samples, trains)          forward_backward and optim_step
```

- The **environment service** opens a session per episode: the task's instruction, its tools and its grader.
- The **harness** is the agent code. It runs one agent, which can spawn subagents.
- The **model gateway** sits between every agent and the model. It identifies each agent by its agent key, samples
  through the training backend and records the exact tokens. Its **MCP tool plane** serves each agent the
  environment's tools.
- The **trainer** runs episodes in groups, builds training data from the recorded tokens and the outcomes, and
  sends batches to the training backend, whose updated weights the next episodes sample.

[Concepts](getting-started/concepts.md) defines each term. [Architecture](architecture.md) explains the design.

## Where to go next

**Get started**

- [Install](getting-started/install.md): from PyPI or from source, the extras, the TextCraft extension.
- [Concepts](getting-started/concepts.md): the vocabulary, from harness to credit assignment.
- [Your first run](getting-started/first-run.md): one episode, a short training run, the trajectory viewer.
- [Train on a Harbor dataset](../README.md#train-on-a-harbor-dataset): a site profile, one config, `platoon prepare`
  and `platoon train`, with [a project to copy](../examples/harbor-dataset/README.md).

**Tutorials** (most with a runnable example under [`examples/`](../examples/); every example has a CPU test)

- Harnesses: [write a function harness](tutorials/function-harness.md),
  [wrap a command-line agent](tutorials/cli-harness.md), [the built-in harnesses](tutorials/builtin-harnesses.md).
- Environments: [build an environment](tutorials/custom-environment.md),
  [run it on env servers](tutorials/remote-environments.md),
  [gym adapters and task tables](tutorials/gym-and-task-tables.md).
- Episodes: [train agents with subagents](tutorials/subagents.md),
  [credit assignment](tutorials/credit-assignment.md),
  [episode programs and reward shaping](tutorials/programs.md).

**How-to guides**: [configs](how-to/configs.md), [training and evaluation](how-to/train-and-eval.md),
[evaluating a hosted model](how-to/evaluate-hosted-models.md),
[models and tool calling](how-to/models-and-tool-calling.md),
[sandboxes and Harbor tasks](how-to/sandboxes-and-harbor.md), [how Harbor tasks are graded](how-to/harbor-grading.md),
[clusters](how-to/clusters.md),
[testing an environment, a harness or a program on CPU](how-to/test-extensions.md),
[troubleshooting](how-to/troubleshooting.md).

**Extending platoon**: [the extension points](extending/index.md) and how platoon finds your code,
[publishing an extension package](extending/publish-a-package.md), and guides to
[environment services](extending/environment-services.md), [reward processors and finalizers](extending/rewards.md),
[judges and verifiers](extending/judges-and-verifiers.md), [sandbox providers](extending/sandbox-providers.md),
[renderers and tool-call parsers](extending/renderers.md),
[hooks, telemetry and metrics](extending/hooks-and-metrics.md) and, experimental,
[task samplers](extending/task-samplers.md), [group admission rules](extending/group-admission.md) and
[rollout leases](extending/rollout-leases.md).

**Reference**: [CLI](reference/cli.md), [config](reference/config.md), [Python API](reference/api.md),
[experimental API](reference/experimental.md).

**Background**: [Architecture](architecture.md) explains the design; the [RFCs](rfcs/README.md) propose what
comes next.

## Coming from the original platoon?

This release replaces the original platoon: its AReaL-based trainer, its standalone inference workflows
(the plugins' `run_inference` scripts, now `platoon eval`) and its plugin layout are gone, and there is no
step-by-step upgrade guide; start from [Install](getting-started/install.md) and
[Your first run](getting-started/first-run.md). The code that the
[Recursive Agent Optimization (RAO)](https://arxiv.org/abs/2605.06639) paper's experiments ran on is on the
[`apga/rao-snapshot`](https://github.com/ApGa/platoon/tree/apga/rao-snapshot) branch of
[ApGa/platoon](https://github.com/ApGa/platoon), and the original code is at the tag
[`archive/v0.1`](https://github.com/ApGa/platoon/tree/archive/v0.1).

## Contributing

Bug reports and pull requests are welcome. [CONTRIBUTING.md](../CONTRIBUTING.md) covers the development setup,
the tests and the conventions. Changes to a public contract or to what trains, new subsystems, backends or
required dependencies, renames or removals of public names, and work that needs several pull requests or spans
repositories start as an RFC: see [when a change needs an RFC](../CONTRIBUTING.md#when-a-change-needs-an-rfc) and
the [RFC process](rfcs/README.md#how-to-propose-a-change).
