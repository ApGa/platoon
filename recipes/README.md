# Recipes

A recipe carries one published experiment over to platoon, or runs one of platoon's features on a published
experiment's setting: configs that run the setting, and a README that says what the publication reports, how the
configs differ from its runs, and which versions of platoon and of the extensions the configs were checked with. A
recipe is not an extension: it adds no code to platoon, only configs that name the components platoon and the
extensions provide, and the files those configs hand to what they run (a dataset's overlays, a node service's
scripts).

| Recipe | Experiment |
|---|---|
| [`rao/`](rao/README.md) | [Recursive Agent Optimization](https://arxiv.org/abs/2605.06639) on TextCraft-Synth: the paper's single CodeAct agent against its recursive agent, and its ablation of the subagents' rewards |
| [`frontiersmith/`](frontiersmith/README.md) | [FrontierSmith](https://github.com/FrontierCS/FrontierSmith)'s 200 synthetic competitive-programming problems as agentic Harbor tasks, graded by one Frontier-CS judge per node (a node service): an evaluation, and a small training check |
| [`teams/`](teams/README.md) | No publication: teams of agents that platoon-harness runs, trained through the `platoon-harness` harness on the RAO recipe's TextCraft-Synth setting, as a recursive tree and as a mesh of peers |

CI checks every recipe on CPU, with every first-party extension installed: each config loads, resolves every
component it names and validates (`uv run --no-sync pytest recipes/ -v`, after
`uv sync --all-packages --dev --extra tinker --extra ors --extra platoon-harness --extra platoon`).
