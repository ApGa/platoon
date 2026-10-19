# Function harness example: `example-tool-loop`

A minimal function harness: an `async` Python function that plays any environment through its tools. It runs
here on number search, the guessing game that ships with platoon. The walkthrough is
[Write a function harness](../../docs/tutorials/function-harness.md).

| File | What it is |
|---|---|
| `tool_loop.py` | the harness, registered as `example-tool-loop` with `@pt.harness` |
| `config.yaml` | an evaluation config that pairs number search with `example-tool-loop` |
| `train.yaml` | a training config: a LoRA, five optimizer steps |
| `pyproject.toml` | packages the harness and registers it through the `platoon.harnesses` entry point |

## Run it

From the repository root, after `uv sync --extra tinker --extra ors`, with `TINKER_API_KEY` set:

```bash
export PYTHONPATH=examples/function-harness
uv run platoon harness show tool_loop:tool_loop                                   # is it registered?
uv run platoon rollout examples/function-harness/config.yaml --task val:0         # one task, printed as a tree
uv run platoon eval examples/function-harness/config.yaml                         # 20 validation tasks
uv run platoon train examples/function-harness/train.yaml                         # five training steps
```

Installed as a package (`uv pip install -e examples/function-harness`), the harness needs neither `PYTHONPATH` nor
the config's `package` line.

## Test

`tests/examples/test_function_harness_example.py` runs the config through `platoon rollout` with a scripted policy
behind the real gateway, on the CPU, and checks the tutorial's commands and outputs:

```bash
uv run pytest tests/examples/test_function_harness_example.py
```

The tests need no model download: the gateway renders with `platoon.testing`'s toy tokenizer
([Test an environment, a harness or a program on CPU](../../docs/how-to/test-extensions.md)).
