# RFC 0011: Checkpoint export with a matching chat template

Status: draft. Builds on the `chat_template` renderer, which renders with the model's own chat template ([Architecture](../architecture.md#renderers)). Runs trained before this lands stay exportable: every run's `config.resolved.yaml` records its renderer and settings.

## Summary

Turn a training checkpoint into a Hugging Face model directory that prompts the model the way training did. Each checkpoint records its renderer, settings, base snapshot and template hash. `platoon export RUN --to hf` writes the tokenizer, a chat template with the run's settings built in as defaults, `generation_config.json` and a manifest, and refuses unless the exported template reproduces the training tokens. Converting the Megatron weights is a separate change in miles.

## Motivation

- **No run produces a Hugging Face model.** Full fine-tuning of Qwen3.6 and Qwen3.8 models on a self-hosted Tinker-compatible server built on miles, such as [miles-tinker](https://github.com/neulab/miles-tinker), writes Megatron `torch_dist` checkpoints with optimizer state (about 350 GB for a 27B model), and `save_weights_for_sampler` only pushes weights to SGLang. The gateway has no download route for the cookbook's `build_hf_model`. miles' offline converter cannot read these checkpoints (no `common.pt`, different parameter names). No gateway route calls the live `actor.export_hf`, whose bridge path can silently write zero weights for parameters it cannot map.
- **The template is copied, not chosen.** Every exporter (miles' `copy_assets` and `hf_export`, the cookbook's `save_tokenizer_and_processor`) copies the base snapshot's `chat_template.jinja` unchanged. That is right only when training used the template's defaults: Qwen3.6 on a preserve-thinking base, and Qwen3.8 at `reasoning_effort: xhigh`. It is wrong for Qwen3.8 trained at `medium` or `low`: the template defaults to xhigh and writes its xhigh preamble into the system turn, so prompts diverge at token 3 (medium) or 9 (low).
- **No checkpoint records its renderer.** `checkpoints.jsonl`, the checkpoint's `tinker_meta.json` and the Tinker `user_metadata` hold no renderer name, settings or template hash, and the base revision is recorded only when `model_name` is a snapshot path. Run-level records have the settings: `train.renderer_name` and `train.renderer_kwargs` in `config.resolved.yaml`, and for runs that predate run records the WandB config, named by each row's `wandb_run_id`.

## Proposal

**1. Record the renderer with each checkpoint.** The four `save_checkpoint_async` calls in `platoon/train/tinker/rl.py` add a block to `loop_state`, which the cookbook copies into the `checkpoints.jsonl` row:

```python
"renderer": {
    "name": "qwen3_8_preserve_thinking",                                # as configured
    "kwargs": {"preserve_thinking": True, "reasoning_effort": "medium"},  # effective settings
    "base_model": "<org>/<base model>",
    "base_revision": "<snapshot commit>",
    "template_sha256": "<hash of the template string training rendered with>",
}
```

The training client also gets `user_metadata={"renderer_name": ...}`, the cookbook's convention. The gateway keeps that in memory only, so `checkpoints.jsonl` is the lasting record.

**2. `platoon export RUN --to hf`** (RFC 0008's `export`) writes, starting from the run's own base snapshot:

- the tokenizer and processor files;
- `chat_template.jinja`: the base template with one default line per renderer setting, for a run trained with `preserve_thinking: true` at `reasoning_effort: medium`:
  ```jinja
  {%- if preserve_thinking is not defined %}{%- set preserve_thinking = true %}{%- endif %}
  {%- if reasoning_effort is not defined %}{%- set reasoning_effort = "medium" %}{%- endif %}
  ```
  with `tokenizer_config.json` carrying the same template or none;
- `generation_config.json`, whose `eos_token_id` holds the renderer's stop tokens;
- `platoon_export.json`: checkpoint, base and revision, renderer settings, template hashes, the check's result, and the platoon and transformers versions.

Settings come from the checkpoint row, else `config.resolved.yaml`, else flags. The export refuses a base whose template hash differs from the recorded one, and **refuses unless it reproduces the training tokens**: the exported tokenizer's `apply_chat_template(conv, add_generation_prompt=True)`, called with no kwargs, must give the training renderer's token ids on a fixed set of conversations, with and without the per-request `enable_thinking=False` and `preserve_thinking=False`. A prototype of this check matched all 7 test conversations for runs trained on Qwen3.6 and on Qwen3.8 at each reasoning effort; the unchanged base template matched none for the runs at `medium` and `low`.

**3. Weights come from the training server.** For a server built on miles, the lowest-risk route is a gateway export command that streams the `bridge.export_hf_weights` output that weight sync already pushes to SGLang, writes safetensors, and checks the key set and shapes against the base's index, refusing all-zero tensors. It needs a GPU gateway with the checkpoint loaded. `platoon export` calls it, or takes `--weights DIR`, and writes its own files beside the weights.

## What the chat-template renderer changes

The chat-template renderer replaces the Python copies of the Qwen templates with the base snapshot's own `apply_chat_template`, called with the run's settings. So:

- **One template.** The export is the training template plus default lines. The check renders that template once with explicit settings and once with the built-in defaults, so it tests just the default lines. It runs in seconds on CPU, on the token fixtures saved for the renderer switch.
- **A meaningful hash.** `template_sha256` hashes the exact template training rendered with, so a changed snapshot is caught at export.
- **Complete settings.** Export relies on the renderer passing every setting explicitly, aliases included, and reading its stop tokens from `generation_config.json`. The recorded settings and that file are then all the export needs.

## Serving caveats

- **Reasoning history.** Clients must send `reasoning_content` back. Some clients send it back only for model names they recognize, so an export needs a matching name or a configurable check in the client.
- **Tool definitions.** SGLang reorders their keys and adds `"strict": false` (0.5.22.dev39 also adds `"defer_loading": null`), so prompts with tools differ from token 29. Normalize tools on the server, or render training tools in the server's shape.
- **Per-request `reasoning_effort`.** SGLang forwards it into the template, which for Qwen3.8 raises on anything but xhigh, medium or low. A client that sends `high` gets an error.
- **Server-side defaults** (SGLang's `--default-chat-template-kwargs`, absent in 0.5.10.post1) can stand in for the built-in template, but they do not travel with the weights.
- **transformers version.** Rendering depends on it (checked on 5.3.0 and 5.5.4), so the manifest records it.

## Rollout plan

1. **Record.** Exit: a smoke run's `checkpoints.jsonl` rows carry the block, and resume still reads them.
2. **Template export**, without weights. Exit: exports of runs on Qwen3.6 and on Qwen3.8 at `medium` and `low` pass the check, and an unchanged base template is refused for the `medium` run.
3. **Weights**, in the training server. Exit: a full fine-tuning checkpoint exports with the base's keys and shapes, and SGLang serving it scores a recorded training completion within tolerance of its training-time logprobs.

## Open questions

- Gateway export command (GPU) first, or a fixed offline converter (CPU: Megatron args from the launcher, the `language_model.` prefix stripped, `vision_model.*` mapped)?
- Name exports so that clients recognize them, or ask clients to make their check a setting?
- For runs that predate run records, read the settings from WandB automatically, or require flags?
