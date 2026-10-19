# Program example: shaping number-search rewards

An episode program, `example-efficient-search`, that runs and grades the root agent and then shapes its reward with a
finalizer: a correct search loses `guess_penalty` for every guess beyond what a binary search needs. The finalizer
also asks a judge (`ep.judge_endpoint()`) for a one-line critique, kept for audit and never trained. The tutorial
[docs/tutorials/programs.md](../../docs/tutorials/programs.md) walks through it.

| File | What it is |
|---|---|
| `efficient_search.py` | `@pt.program("example-efficient-search")`, the `EfficiencyShaping` finalizer, and `reward_parts`, the reward processor that trains on the shaped reward and reports its parts |
| `efficient_search.yaml` | Number search (the environment that ships with platoon) with the `platoon-react` harness, this program, and a training section; `rollout_config.extra.guess_penalty` sets the penalty |

## Run it

From the repository root, with platoon installed (`uv sync --extra tinker --extra ors`) and this directory on
`PYTHONPATH`, so `efficient_search` imports:

```bash
export PYTHONPATH=$PWD/examples/program-reward-shaping
uv run platoon rollout examples/program-reward-shaping/efficient_search.yaml --task train:0   # needs TINKER_API_KEY
uv run platoon train examples/program-reward-shaping/efficient_search.yaml                    # needs TINKER_API_KEY
```

The printed tree shows the root's shaped reward next to its graded outcome, for example
`reward 0.9  outcome scored reward=1.0` for a search that took one guess more than binary search needs.

## Test it

`tests/examples/test_program_reward_shaping_example.py` runs the config through `platoon rollout` with a scripted
policy instead of a Tinker model (CPU only, with no model download; see
[Test an environment, a harness or a program on CPU](../../docs/how-to/test-extensions.md)):

```bash
uv run pytest tests/examples/test_program_reward_shaping_example.py
```
