# Subagents example: a delegating agent on Vault tasks

A function harness, `example-delegator`, that can hand parts of its task to subagents with `ctx.agents.spawn`, and an
environment made for it. The tutorial [docs/tutorials/subagents.md](../../docs/tutorials/subagents.md) walks
through it.

| File | What it is |
|---|---|
| `vault_env.py` | `VaultEnvironment` (ORS): add up the numbers stored under some keys with `read(key)` and the terminal `answer(total)`; `platoon_fork` gives each subagent its own keys on the same vault |
| `delegator.py` | `@pt.harness("example-delegator")`: a tool-calling loop plus a `delegate(goal, max_steps)` tool that spawns a subagent; several `delegate` calls in one reply run in parallel |
| `subagents.yaml` | The environment entry (`agents:` settings for depth, budgets, access, grading and forks) and a training section |

## Run it

From the repository root, with platoon installed (`uv sync --extra tinker --extra ors`) and this directory on
`PYTHONPATH`, so `vault_env` and `delegator` import:

```bash
export PYTHONPATH=$PWD/examples/subagents
uv run platoon config validate examples/subagents/subagents.yaml
uv run platoon rollout examples/subagents/subagents.yaml --task train:0    # one task, needs TINKER_API_KEY
uv run platoon train examples/subagents/subagents.yaml                     # train, needs TINKER_API_KEY
```

`platoon rollout` prints the agent tree: the root and every subagent, each with its steps, reward and outcome.
Override any setting on the command line, for example `environments.0.agents.child_grading=verifier
environments.0.agents.verifier.max_steps=4` to have verifier agents judge the subagents.

## Test it

`tests/examples/test_subagents_example.py` runs the config through `platoon rollout` with a scripted policy instead
of a Tinker model (CPU only, with no model download; see
[Test an environment, a harness or a program on CPU](../../docs/how-to/test-extensions.md)):

```bash
uv run pytest tests/examples/test_subagents_example.py
```
