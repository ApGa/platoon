# Oolong Plugin for Platoon

This plugin provides support for the [Oolong benchmark](https://github.com/abertsch72/oolong), a challenging long-context aggregation benchmark for language models.

## Overview

Oolong evaluates a model's ability to reason and aggregate information across large text contexts. It includes two datasets:

- **oolong-synth**: Synthetic aggregation tasks with controlled settings
- **oolong-real**: Real-world tasks over D&D campaign transcripts

## Installation

```bash
cd plugins/oolong
uv sync
```

For training with Tinker backend:
```bash
uv sync --extra tinker
```

For training with AReaL backend:
```bash
uv sync --extra areal
```


## References

- [Oolong Benchmark Paper](https://arxiv.org/abs/2511.02817)
- [Oolong GitHub](https://github.com/abertsch72/oolong)
- [HuggingFace Datasets](https://huggingface.co/oolongbench)
