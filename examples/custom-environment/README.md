# Custom environment: order packing

An ORS (OpenReward) environment written for platoon, with configs that train on it. The agent moves items from a
shelf into a box until the box holds exactly the order. The walkthrough is
[docs/tutorials/custom-environment.md](../../docs/tutorials/custom-environment.md).

| File | What it is |
|---|---|
| `order_packing.py` | `OrderPackingEnv`: splits, prompt, `pack` (typed parameters), read-only `look`, terminal `done`, and `platoon_fork` for subagents. |
| `check_env.py` | Drives the environment through platoon's session API, without a model. |
| `packing_team.py` | `example-packing-team`, a function harness whose root can `delegate` items to subagents. |
| `serve.py` | Serves the environment on an ORS env server, for `ors+http://`. |
| `train.yaml` | Train with the built-in `platoon-react` harness. |
| `train_team.yaml` | Train with `example-packing-team`: children pack on `platoon_fork` sessions and are graded by them. |
| `remote.yaml` | Train against env servers (`ors+http:///OrderPackingEnv`). |

## Run it

Install platoon from the repository root (`uv sync --extra tinker --extra ors`), then work from this directory
(the test commands below run from the repository root).
`PYTHONPATH=.` lets platoon import the example's modules, here and in rollout worker processes.

```bash
cd examples/custom-environment

# No model needed:
PYTHONPATH=. uv run python check_env.py
PYTHONPATH=. uv run platoon config validate train.yaml train_team.yaml remote.yaml

# Needs TINKER_API_KEY:
PYTHONPATH=. uv run platoon rollout train.yaml --task train:0
PYTHONPATH=. uv run platoon train train.yaml
PYTHONPATH=. uv run platoon train train_team.yaml

# On an env server:
PYTHONPATH=. uv run python serve.py --port 8080 &
OPENREWARD_SESSION_URLS_ORDERPACKINGENV=http://127.0.0.1:8080 PYTHONPATH=. uv run platoon train remote.yaml
```

`check_env.py` prints:

```text
splits: ['train', 'test']
test tasks: ['test:0', 'test:1']
instruction: Pack this order into the box: 2 apple, 1 bread.
tool done(summary): terminal
tool look(): read_only
tool pack(item, quantity): -
pack 0 apple: error input_validation
child: Packed 2 apple. The box holds 2 apple.
child grade: scored reward=1.0 reason=None
root: Packed 1 bread. The box holds 2 apple, 1 bread.
root: Shelf: 4 apple, 1 bread, 3 cheese, 12 egg, 2 milk. Box: 2 apple, 1 bread.
root grade: scored reward=1.0 reason=None
```

## Tests

`tests/examples/test_custom_environment_example.py` runs `check_env.py`, validates the configs and rolls out
`train.yaml` and `train_team.yaml` with a fake sampler; `tests/examples/test_remote_environment_example.py` runs
`remote.yaml` against a local env server. CPU only, from the repository root:

```bash
uv run pytest tests/examples
```

The tests need no model download: the gateway renders with `platoon.testing`'s toy tokenizer
([Test an environment, a harness or a program on CPU](../../docs/how-to/test-extensions.md)).
