# Gym adapters and task tables: hi-lo and questions

A reset/step game trained two ways, and a question table graded by an episode program. The walkthrough is
[docs/tutorials/gym-and-task-tables.md](../../docs/tutorials/gym-and-task-tables.md).

| File | What it is |
|---|---|
| `guess_game.py` | `HiLoGame` (a step environment), the `example-hilo-loop` harness (`pt.gym.loop`) and `HiLoEnvironment` (`pt.gym.as_environment`). |
| `tasks.jsonl` | The hi-lo task table: `id`, `goal`, `target`, `split`. |
| `exact_match.py` | The `example-exact-match` episode program: grades a row by its `answer` column. |
| `questions.jsonl` | The question table: `id`, `goal`, `answer`, `split`. |
| `train_loop.yaml` | `example-hilo-loop` on the task table, scored by the `gym` program. |
| `train_tool.yaml` | `HiLoEnvironment` played by `platoon-react` through its `act` tool. |
| `train_qa.yaml` | The question table answered by `platoon-react` and graded by `example-exact-match`. |

## Run it

Install platoon from the repository root (`uv sync --extra tinker --extra ors`), then work from this directory
(the test commands below run from the repository root).
`PYTHONPATH=.` lets platoon import the example's modules, and the tables' paths are relative to it.

```bash
cd examples/gym-env

# No model needed:
PYTHONPATH=. uv run platoon config validate train_loop.yaml train_tool.yaml train_qa.yaml

# Needs TINKER_API_KEY:
PYTHONPATH=. uv run platoon rollout train_loop.yaml --task train-0   # task-table ids are row ids
PYTHONPATH=. uv run platoon rollout train_tool.yaml --task train:0   # ORS ids are <split>:<index>
PYTHONPATH=. uv run platoon rollout train_qa.yaml --task q0
PYTHONPATH=. uv run platoon train train_loop.yaml
```

## Tests

`tests/examples/test_gym_env_example.py` validates the configs and rolls out each one with a fake sampler, on CPU.
From the repository root:

```bash
uv run pytest tests/examples/test_gym_env_example.py
```

The tests need no model download: the gateway renders with `platoon.testing`'s toy tokenizer
([Test an environment, a harness or a program on CPU](../../docs/how-to/test-extensions.md)).
