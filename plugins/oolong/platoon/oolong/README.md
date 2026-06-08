# Oolong

This plugin adds the [Oolong benchmark](https://github.com/abertsch72/oolong) to Platoon. Oolong evaluates long-context aggregation over synthetic tasks and real D&D campaign transcripts.

## Install

```bash
cd plugins/oolong
uv sync --extra areal --extra wandb
```

Use `--extra tinker` for Tinker experiments.

## Train

```bash
uv run python platoon/oolong/train_scripts/areal/train_areal.py \
  --config platoon/oolong/configs/train/areal/oolong_linear_areal.yaml
```

## References

- [Oolong paper](https://arxiv.org/abs/2511.02817)
- [Oolong GitHub](https://github.com/abertsch72/oolong)
- [HuggingFace datasets](https://huggingface.co/oolongbench)
