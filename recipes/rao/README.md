# RAO on TextCraft-Synth

This recipe carries the TextCraft-Synth experiments of [Recursive Agent Optimization](https://arxiv.org/abs/2605.06639)
(RAO) over to platoon: the paper's single agent against its recursive agent, which delegates crafting subgoals to
subagents, and the paper's ablation of the subagents' rewards. Every arm trains `Qwen/Qwen3-4B-Instruct-2507` on the
852 medium training tasks of TextCraft-Synth.

## The arms

Each arm is one config in `textcraft-synth/`. `linear.yaml` and `recursive.yaml` extend `common.yaml`, the setting
that every arm shares, and `ablation_sparse.yaml` extends `recursive.yaml`:

| Arm | Config | The paper's result | Agents | A subagent's reward |
|---|---|---|---|---|
| linear | `linear.yaml` | Table 1(b), "Single": a success rate of 0.73 over all difficulties | one agent with 200 steps and no subagents (`variant: linear`, `max_depth: 0`) | — |
| recursive | `recursive.yaml` | Table 1(b), "Recursive": 0.96; Fig. 8, "Dense, Weighted": 0.89 after 100 steps | `launch_subagent(targets)` (`variant: depth_aware`, `subagent_budget: fixed`); 25 steps for every agent, root or subagent (`budget: depth_aware`, `child_max_steps: 25`), at most 6 levels deep (`max_depth: 6`) | whether it crafted its own targets (Dense): `child_grading: env` |
| Sparse ablation | `ablation_sparse.yaml` | Fig. 8, "Sparse, Weighted": 0.79 after 100 steps | as `recursive` | the root's success (Sparse): `propagate_root_success: true` |

Table 1(b) is the paper's setting with a 40k context in training and a 256k context in evaluation. The recipe does not
carry Table 1(a), whose agents train and evaluate with an 8k context.

Every arm names its environment as `env: textcraft-synth@1`, the TextCraft extension's named environment at version
1: the TextCraft-Synth environment, the TextCraft prompts for `platoon-codeact`, the `val` split for evaluation and
the `platoon-delegation` reward processor, whose delegation bonus is 0 (λ = 0, as in the paper). Subagents share
their parent's inventory.

The recursive arms center every trajectory of a rollout, root or subagent, on the mean of the other rollouts' root
rewards (`leave_one_out_baseline`, the paper's Eq. 3) and weight depth levels inversely to their trajectory counts in
the batch (`depth_level_weighting`, Eq. 4). The linear arm centers rewards on the mean of its group, as the paper's
linear runs did.

`ablation_sparse.yaml` differs from `recursive.yaml` only in the subagents' reward, its length (100 steps, as the
paper's ablation) and its run's name. Fig. 8's "Dense, Weighted" arm is the `recursive.yaml` run at step 100, where it
keeps a checkpoint. Its two unweighted arms are the two configs with `train.workflow_config.depth_level_weighting=false`
and a `stats.trial_name` of their own, and the Dense one with `train.max_training_steps=100`.

**Not one of the paper's arms:** a recursive agent that sizes each child's budget itself,
`launch_subagent(targets, num_steps)`, from its own 200 steps. The paper's recursive agents had 25 steps each, root
and subagents (A.5), so the recipe has no such arm, although the paper's appendix A.7 prints that agent's action
space. platoon still offers it: the TextCraft prompts' `variant: recursive` with `subagent_budget: argument` and
`budget: linear`.

## Settings and their sources

The arms take the settings of the paper's runs, from:
- **P**: the paper, [arXiv 2605.06639](https://arxiv.org/abs/2605.06639) (v2);
- **R** and **L**: the AReaL configs of the paper's recursive and linear runs,
  `textcraft_synth_ctx40000_depth_aware_medium_areal.yaml` and `textcraft_synth_ctx40000_linear_medium_areal.yaml` in
  [ApGa/platoon at `a495b08`](https://github.com/ApGa/platoon/tree/a495b08859c28971fbaf017cc59ac92b212e171b/plugins/textcraft/platoon/textcraft/configs/areal),
  the last commit before `4f752b7` added root-success propagation;
- **C**: the code at the same commit, and the AReaL version it pins (`35b57e2`). The runs started from earlier
  commits; the table cites the code as `a495b08` has it.

| Arm | Setting | Value | Source |
|---|---|---|---|
| all | Model | `Qwen/Qwen3-4B-Instruct-2507` | P §3.1; R, L `actor.path` |
| all | Training tasks | the 852 medium tasks (`train-medium`), 10 epochs | P §3.1; R, L `train_difficulties`, `total_train_epochs` |
| all | Batch | 16 tasks, 8 rollouts each | P A.5; R, L `train_dataset.batch_size`, `group_size` |
| all | Staleness | at most 3 steps | P A.5; R, L `max_head_offpolicyness` |
| all | Loss | CISPO, the importance ratio clipped to [0, 5] | P A.5; R, L `loss_fn_config` |
| all | Context | 40,000 tokens | P §3.1; R, L `sglang.context_length` |
| all | Sampling | temperature 1, 512-token completions | C `platoon/config_defs.py:17,19`, the agents' sampling defaults, which neither config changes. AReaL's client built each request's sampling from them, so R and L's `gconfig.max_new_tokens` (8,192) never reached a request |
| all | Optimizer | Adam (β 0.9 and 0.95, ε 1e-8), weight decay 0.017, the gradient norm clipped at 1, a constant learning rate | R, L `actor.optimizer`, whose warmup (0.1% of the steps) rounds down to none |
| all | Learning rate | 3e-5, for a rank-32 LoRA | P A.5: the paper's Tinker runs (LoRA rank 32, 3e-5). Its TextCraft-Synth runs trained every weight at 3e-6 (P A.5; R, L) |
| all | Microbatches | 1 per batch | P Eq. 4 and C `platoon/train/areal/rl.py` weight depth levels over the whole batch, and AReaL normalizes the loss over it; platoon does both per microbatch |
| all | Time limits | 7,200 s for a rollout and for a cell | R, L `timeout`, `step_timeout` |
| all | Delegation bonus | none (λ = 0) | P §3.1; C `train_areal_synth.py:35` |
| all | Evaluation during training | every 5 steps, on the first 100 validation tasks of every difficulty, one rollout each, sampled as in training | L `evaluator.freq_steps`; C `train_areal_synth.py:111,150`. R ran no evaluation during training |
| all | Checkpoints | every 25 steps, and a recovery checkpoint every 5 | R, L `saver.freq_steps`, `recover.freq_steps` |
| linear | Steps | 200 | P A.5; L `max_steps` |
| linear | Baseline | the mean of the group | L sets no `leave_one_out_baseline`, whose default is false (C `platoon/train/areal/config_defs.py:19`) |
| linear | Rollout groups at once | 16 | L `rollout.max_concurrent_rollouts` |
| recursive | Steps | 25 for the root and 25 for each subagent | P A.5; R `max_steps`; C `synth_rollout.py:88` |
| recursive | Depth | at most 6 | P §3.1; C `synth_rollout.py:82` |
| recursive | A subagent's reward | whether it crafted its own targets (Dense) | P §2.2.1, §5.0.1; C `env.py`, which grades each subagent on the targets of its goal. R propagates no root success: `4f752b7` added that |
| recursive | Baseline | leave-one-out over the group's root rewards, for every trajectory of a rollout | P Eq. 3; R `leave_one_out_baseline` |
| recursive | Depth weighting | inverse to each depth level's trajectory count | P Eq. 4; R `depth_level_weighting` |
| recursive | Rollout groups at once | 8 | R `rollout.max_concurrent_rollouts` |
| Sparse ablation | A subagent's reward | the root's success (Sparse) | P §5.0.1; R as `4f752b7` changed it (`propogate_root_success: true`) |
| Sparse ablation | Length | 100 steps | P §5.0.1 |

Not carried over: AReaL's seed (343), the layout and microbatch sizes of its engines, and PPO's clip range, which CISPO
does not use. platoon shuffles the training tasks with seed 42.

## Requirements

- platoon 0.4, with the `tinker` and `ors` extras, and the TextCraft extension `platoon-textcraft` 0.3. From a
  checkout: `uv sync --package platoon-textcraft --extra platoon`.
- A Tinker-compatible training service that serves `Qwen/Qwen3-4B-Instruct-2507`: hosted Tinker
  (`TINKER_API_KEY`), or a self-hosted server such as [miles-tinker](https://github.com/neulab/miles-tinker)
  (`TINKER_BASE_URL` too).
- `WANDB_API_KEY`: the arms log to the W&B project `rao-textcraft-synth`. `stats.wandb.mode=disabled` turns that
  off.

## Run

```bash
for arm in linear recursive ablation_sparse; do
  uv run platoon train recipes/rao/textcraft-synth/$arm.yaml log_path=/path/to/runs
done
```

## How these runs differ from the paper's

- The paper fine-tuned every weight with AReaL. Tinker trains a rank-32 LoRA instead, so the learning rate is 3e-5,
  ten times the paper's 3e-6.
- The paper's trainer dropped a group whose trajectories all had the same advantage, and drew another task in its
  place, so that each step trained on 16 groups with a learning signal (C `step_wise.py`). platoon trains the groups
  it draws: such a group adds no gradient, but its trajectories still count in the depth weights and its tokens in
  the loss's normalization.
- The paper's numbers come from evaluations after training. Table 1's evaluation sampled at temperature 0, on 100
  validation tasks drawn with seed 42 from the easy, medium and hard ones
  (C `configs/inference/textcraft_synth_inference.yaml`), with a 256k context and up to 12 levels of delegation
  (P §3.1). The arms' evaluations during training, at temperature 1 on the first 100 validation tasks with the
  training context and depth, follow training; they are not that evaluation. The TextCraft extension's
  `configs/textcraft_synth_eval.yaml` draws the same sample for the linear agent.
- The recipe carries the paper's setting over to platoon; it does not rerun the paper's code. That code is on the
  [`apga/rao-snapshot`](https://github.com/ApGa/platoon/tree/apga/rao-snapshot) branch of
  [ApGa/platoon](https://github.com/ApGa/platoon).

## Published results

The numbers above are the paper's ([arXiv 2605.06639](https://arxiv.org/abs/2605.06639)). This README gives no
numbers of its own: results of these configs on Tinker are not published.

## The tasks

TextCraft-Synth is published as a Harbor dataset,
[`apurvaga/textcraft-synth`](https://huggingface.co/datasets/apurvaga/textcraft-synth): 3,154 tasks, 2,522 for
training and 632 for validation. The arms train on the 852 medium training tasks (`train-medium`) and evaluate on
the first 100 validation tasks. The task files ship in the TextCraft extension
(`extensions/textcraft/src/platoon_textcraft/data/`):

| File | Tasks | SHA-256 |
|---|---:|---|
| `textcraft_synth_train.jsonl` | 2,522 | `685823fbca90aa89e061e61fef58d0122def4221eed14cf2c8c730341363a00f` |
| `textcraft_synth_val.jsonl` | 632 | `84a123ee46e29e65f4d7b95f4943aa56907fc3fd5992268faa749602579dfc0c` |

## Check

```bash
uv sync --all-packages --dev --extra tinker --extra ors --extra platoon
uv run --no-sync pytest recipes/rao -v
```

On CPU, without a training service, `test_rao_recipes.py` checks that `textcraft-synth/` holds exactly these arms;
that every arm loads, resolves every component it names (the named environment, the harness, the program, the
reward processor, the service and the prompt builder) and validates; that `env: textcraft-synth@1` stands for the
environment entry each arm would otherwise spell out; that the arms share the paper's setting; that each selects its
prompts, agent tree, step budget, rollout concurrency, baseline and credit (the group's mean for `linear`,
leave-one-out and Dense rewards for `recursive`, Sparse rewards for `ablation_sparse`); that the arms differ in
nothing else but their run names and tags, and `ablation_sparse.yaml` from `recursive.yaml` only in its credit, its
length and its name; and that they train on the 852 medium tasks and evaluate on the first 100 validation tasks.
The TextCraft extension's tests run both recursive configs through a scripted rollout
(`extensions/textcraft/tests/test_textcraft_rollouts.py`): a subagent that fails its own targets scores 0 under
`recursive.yaml` and the root's success under `ablation_sparse.yaml`.

## Citation

```bibtex
@article{gandhi2026rao,
  title   = {Recursive Agent Optimization},
  author  = {Gandhi, Apurva and Chakraborty, Satyaki and Wang, Xiangjun and Kumar, Aviral and Neubig, Graham},
  journal = {arXiv preprint arXiv:2605.06639},
  year    = {2026}
}

@misc{gandhi2025platoon,
  author       = {Gandhi, Apurva},
  title        = {{Platoon}: Build and Train Systems of Agents},
  howpublished = {\url{https://github.com/ApGa/platoon}},
  year         = {2025}
}
```
