# Recipes

A recipe carries one published experiment over to platoon: configs that run the experiment's setting, and a README
that says what the publication reports, how the configs differ from its runs, and which versions of platoon and of
the extensions the configs were checked with. A recipe is not an extension: it adds no code, only configs that name
the components platoon and the extensions provide.

| Recipe | Experiment |
|---|---|
| [`rao/`](rao/README.md) | [Recursive Agent Optimization](https://arxiv.org/abs/2605.06639) on TextCraft-Synth: the paper's single CodeAct agent against its recursive agent, and its ablation of the subagents' rewards |

CI checks every recipe on CPU, with every first-party extension installed: each config loads, resolves every
component it names and validates (`uv run --no-sync pytest recipes/ -v`, after
`uv sync --all-packages --dev --extra tinker --extra ors --extra platoon`).
