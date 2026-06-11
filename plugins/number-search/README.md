# Number Search

Number Search is a small binary-search-style environment for quick Platoon training and rollout smoke tests. The agent guesses a hidden number and receives "too low", "too high", or "correct" feedback.

## Install

```bash
cd plugins/number-search
uv sync --extra areal
```

Use `--extra tinker` instead of `--extra areal` for Tinker experiments.

## Train

Tinker:

```bash
uv run python -m platoon.number_search.train_tinker \
  --config platoon/number_search/number_search_tinker.yaml
```

AReaL:

```bash
uv run python3 platoon/number_search/train.py \
  --config platoon/number_search/nv_number_search_cispo_areal.yaml
```

Useful configs:

- `platoon/number_search/number_search_tinker.yaml`
- `platoon/number_search/number_search_areal.yaml`
- `platoon/number_search/nv_number_search_areal.yaml`
- `platoon/number_search/nv_number_search_cispo_areal.yaml`

## Environment

Each task contains a range and target number. The only action is `guess(n: int)`. Rewards are `1.0` for the correct guess and `0.0` otherwise.

Datasets live next to the package:

- `platoon/number_search/number_search_train.jsonl`
- `platoon/number_search/number_search_val.jsonl`
