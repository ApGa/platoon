# Supervised fine-tuning on LLaMA-Factory datasets

`trainer: sft` fine-tunes a policy on chat datasets in [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)'s
conventions: `data/dataset_info.json` names two small datasets, one in each of its formats, and each config trains on
one of them. The guide is [Fine-tune on a dataset](../../docs/how-to/train-and-eval.md#fine-tune-on-a-dataset).

| File | What it is |
|---|---|
| `data/dataset_info.json` | The registry: `alpaca_demo` and `sharegpt_demo`, each with its file, format and columns |
| `data/alpaca_demo.json` | 12 alpaca rows: `instruction`, `input` and `output`, and in two of them a `system` prompt, in one a `history` |
| `data/sharegpt_demo.jsonl` | 8 sharegpt conversations (`from`/`value` messages): multi-turn chats, a system prompt as a column and as a first message, and tool calls (`function_call`) with their results (`observation`) and the `tools` they call |
| `alpaca.yaml` | Two epochs over `alpaca_demo`, 4 conversations a step, a permanent checkpoint every 3 steps |
| `sharegpt.yaml` | Three epochs over `sharegpt_demo`, each step in two microbatches, a recovery checkpoint after every step of which the newest 2 are kept |

## Run it

Install platoon from the repository root (`uv sync --extra tinker`) and run from there:

```bash
uv run platoon config validate examples/sft/alpaca.yaml examples/sft/sharegpt.yaml   # no model needed
uv run platoon config validate --tasks examples/sft/alpaca.yaml                       # loads the dataset too

# Needs TINKER_API_KEY (and TINKER_BASE_URL for a self-hosted service):
uv run platoon train examples/sft/alpaca.yaml
uv run platoon train examples/sft/sharegpt.yaml
```

Each run writes to `runs/sft/<alpaca|sharegpt>`. `train.model_name` and `train.renderer_name` choose the policy; the
loss is on the assistant's replies and tool calls, rendered by `qwen3_instruct` as the policy sees them when it
samples. Evaluate a checkpoint with `platoon eval`, an evaluation config whose `inference.model_path` names its
`sampler_path` in the run's `checkpoints.jsonl`.

To train on a dataset of your own, add its entry to `dataset_info.json` (`file_name`, or `hf_hub_url` for one on the
Hugging Face Hub, then `formatting`, `columns` and, for sharegpt, `tags`) and name it in `trainer_options.dataset`.

## Tests

`tests/examples/test_sft_example.py` validates both configs and runs `platoon train` on each, on CPU, on the fake
Tinker service of `platoon.testing` (`fake_tinker_service`), with the toy tokenizer under the `qwen3_instruct`
renderer. From the repository root:

```bash
uv run pytest tests/examples/test_sft_example.py
```
