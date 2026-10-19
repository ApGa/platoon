# Number search: a first run

Train a LoRA on number search with the `platoon-codeact` harness: the model writes a Python cell that calls
`guess(number)`, and the environment answers "too low", "too high" or "correct". The walkthrough is
[docs/getting-started/first-run.md](../../docs/getting-started/first-run.md).

| File | What it is |
|---|---|
| `train.yaml` | Training config: `Qwen/Qwen3-4B-Instruct-2507`, LoRA rank 32, 8 tasks × 4 episodes per step, 5 steps. |
| `train_react.yaml` | The same run with the `platoon-react` harness: the model calls `guess` as a native tool instead of writing cells. |
| `train_nemotron.yaml` | Another model family: an `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` LoRA on the first 1,000 training tasks. |
| `eval.yaml` | Evaluation config: 100 validation tasks, the base model or a checkpoint (`inference.model_path`). |

Number search ships with platoon. The environment (`platoon.envs.number_search.env:NumberSearchEnvironment`, an
[OpenReward](https://openreward.ai) environment served in process) has one tool, `guess(number)`: a correct guess
ends the episode with reward 1, and an episode that never gets there is graded 0. Its tasks are generated from
seed 42: the `train` (50,000) and `val` (1,000) splits, with task ids `<split>:<index>`. The CodeAct prompts are
`platoon.envs.number_search.prompts:NumberSearchPromptBuilder` (no `<thought>`, `guess()` in a `<python>` cell).

The configs spell the environment entry out. The named environment `platoon-number-search@1` stands for the same
entry, the options `platoon-codeact` and `platoon-react` need on it and the `val` split for evaluation, so a config
can select it in one line (`platoon env show platoon-number-search` prints it):

```yaml
environments:
  - env: platoon-number-search@1
    harness: platoon-codeact
    dataset_kwargs: {limit: 1000}
```

## Run it

From the repository root, with platoon installed with the `tinker` and `ors` extras and a Tinker-compatible backend
configured (`TINKER_API_KEY`, and `TINKER_BASE_URL` for a self-hosted server such as
[miles-tinker](https://github.com/neulab/miles-tinker)):

```bash
platoon rollout examples/number-search/train.yaml --task val:0 --output-dir runs/first-rollout
platoon viz show-dump runs/first-rollout/trajectory_collection.json

platoon train examples/number-search/train.yaml
platoon viz tail --rdir runs/number-search/first-run/rollouts

platoon eval examples/number-search/eval.yaml inference.model_path=tinker://...   # a sampler_path from checkpoints.jsonl
```

## Test

`tests/examples/test_number_search.py` runs these commands on CPU with a scripted policy that bisects, in place of
the training backend, and checks that every config here loads:

```bash
uv run pytest tests/examples/test_number_search.py
```
