# Command-line harness example: `tiny-agent`

A command-line agent wrapped by a TOML harness spec, so platoon can install it, launch it against the model
gateway, give it the environment's tools over MCP and read its answer, without changing the agent. It runs here
on number search, the guessing game that ships with platoon. The walkthrough is
[Wrap a command-line agent](../../docs/tutorials/cli-harness.md).

| File | What it is |
|---|---|
| `tiny_agent.py` | the agent: a standard-library Chat Completions loop with a built-in `shell` tool and an MCP client; it knows nothing about platoon |
| `tiny-agent.toml` | the harness spec |
| `config.yaml` | an evaluation config that pairs number search with the spec |

## Run it

From the repository root, after `uv sync --extra tinker --extra ors`, with `TINKER_API_KEY` set:

```bash
uv run platoon harness show examples/cli-harness/tiny-agent.toml               # does the spec load?
uv run platoon rollout examples/cli-harness/config.yaml --task val:0           # one task, printed as a tree
uv run platoon eval examples/cli-harness/config.yaml                           # 20 validation tasks
```

The config names the spec by a path relative to the repository root, so run these from there.

## Test

`tests/examples/test_cli_harness_example.py` runs the agent against a scripted model server and the gateway's
real MCP tool plane, locally and in a process sandbox, and runs the config through `platoon rollout`, all on the
CPU:

```bash
uv run pytest tests/examples/test_cli_harness_example.py
```

The tests need no model download: the gateway renders with `platoon.testing`'s toy tokenizer
([Test an environment, a harness or a program on CPU](../../docs/how-to/test-extensions.md)).
