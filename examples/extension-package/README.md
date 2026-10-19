# Extension package example: `example-extension`

An installable package that adds components to platoon: a harness, an episode program, a reward processor, a task
table and a named environment that bundles them, declared through entry points so that any platoon install finds
them by name. The walkthrough is [Publish an extension package](../../docs/extending/publish-a-package.md).

| File | What it is |
|---|---|
| `pyproject.toml` | the distribution, its requirement on platoon, and the entry points that name its components |
| `src/example_extension/harness.py` | `example-self-check`: answers, then checks its answer before giving it |
| `src/example_extension/program.py` | `example-exact-answer`: grades the root agent's final answer against the task's `answer` |
| `src/example_extension/rewards.py` | `example-partial-credit`: a reward processor that gives near misses part of the reward |
| `src/example_extension/tasks.py`, `src/example_extension/data/arithmetic.jsonl` | the arithmetic task table, shipped in the package |
| `src/example_extension/entries.py` | `example-arithmetic@1`: the named environment that bundles the task table with the components that play and grade it |
| `eval.yaml`, `train.yaml` | an evaluation and a training config that name the components |
| `named.yaml` | `eval.yaml` through the named environment |
| `tests/test_example_extension.py` | CPU tests, with a scripted policy from `platoon.testing` |

## Run it

From the repository root, after `uv sync --extra tinker --extra ors`:

```bash
uv pip install -e examples/extension-package                                  # registers the entry points
uv run platoon harness show example-self-check                                # is it found by name?
uv run platoon rollout examples/extension-package/eval.yaml --task test-0     # one task; needs TINKER_API_KEY
uv run platoon rollout examples/extension-package/named.yaml --task test-0    # the same, through the named environment
uv run platoon eval examples/extension-package/eval.yaml                      # the 8 test tasks
uv run platoon train examples/extension-package/train.yaml                    # five training steps
```

## Test

The package's own tests need no model, GPU or Tinker key. They play episodes with a scripted policy from
`platoon.testing`; the test of the named environment runs `platoon rollout` on `named.yaml`, which finds every
component through the entry points, so it skips until the package is installed:

```bash
uv run pytest examples/extension-package/tests -q
```

`tests/examples/test_extension_package_example.py` in the repository runs them, and checks that the tutorial quotes
these files as they are.
