"""The RAO recipe's arms (``textcraft-synth/``): the paper's single agent (``linear``), its recursive agent
(``recursive``) and its Sparse ablation (``ablation_sparse``).  Each arm loads, resolves every component it names and
validates, names the environment entry it would otherwise spell out and shares the paper's setting with the others.
The recursive arm differs from the linear arm only in its prompts, agent tree, step budget, rollout concurrency,
baseline, credit and name, and the ablation from the recursive arm only in its credit, its length and its name.  CPU
only; the TextCraft extension must be installed (``platoon-textcraft``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from platoon_textcraft.prompts import TextCraftPromptBuilder

from platoon import cli
from platoon.envs.named import expand_environment
from platoon.harness.builtin.codeact import CodeActOptions
from platoon.registry import import_from_string
from platoon.rewards import DELEGATION_REWARD_CAP
from platoon.train.auto import AutoDataset, AutoEnvironment, AutoRewardProcessor, AutoRollout, AutoTaskLoader
from platoon.train.components import HARNESS_ROLLOUT
from platoon.train.tinker.batch_transforms import DepthLevelWeightingTransform, build_default_batch_transforms
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig, concurrent_groups
from platoon.utils.config import load_config, load_yaml_config

ARMS_DIR = Path(__file__).resolve().parent / "textcraft-synth"
RAO_ARMS = ("linear", "recursive", "ablation_sparse")
RECURSIVE_ARMS = ("recursive", "ablation_sparse")
# Each arm's prompts (TextCraftPromptBuilder's variant) and its own platoon-codeact options besides the cell timeout.
RAO_VARIANTS = {"linear": "linear", "recursive": "depth_aware", "ablation_sparse": "depth_aware"}
RAO_OPTIONS = {
    "linear": {"subagents": False},
    "recursive": {"subagent_budget": "fixed"},
    "ablation_sparse": {"subagent_budget": "fixed"},
}
# The paper's recursive agent (A.5, §3.1): 25 steps for every agent, root or subagent, at most 6 levels deep, and
# each subagent graded on its own targets.
DEPTH_AWARE_AGENTS = {"max_depth": 6, "budget": "depth_aware", "child_max_steps": 25, "child_grading": "env"}
RAO_AGENTS = {"linear": {"max_depth": 0}, "recursive": DEPTH_AWARE_AGENTS, "ablation_sparse": DEPTH_AWARE_AGENTS}
# Each arm's credit: (leave-one-out baseline, depth-level weighting, root-success propagation).  The linear arm
# centers rewards on the group's mean, as the paper's linear runs did; the recursive arm trains the paper's Dense
# rewards, each subagent graded on its own targets, and the ablation its Sparse ones, the root's success for every
# subagent (§5.0.1).
RAO_CREDIT = {
    "linear": (False, False, False),
    "recursive": (True, True, False),
    "ablation_sparse": (True, True, True),
}


def _load(path: Path, *overrides: str) -> tuple[Any, dict[str, Any]]:
    config, data = load_config(["--config", str(path), *overrides], PlatoonTinkerRLTrainerConfig)
    AutoEnvironment.load(config)
    return config, data


def _rao_path(arm: str) -> Path:
    return ARMS_DIR / f"{arm}.yaml"


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        value = {str(index): item for index, item in enumerate(value)}
    if isinstance(value, dict):
        return {key: item for name, child in value.items() for key, item in _flatten(child, f"{prefix}{name}.").items()}
    return {prefix.rstrip("."): value}


def _differing(first: str, second: str) -> set[str]:
    """The dotted keys whose values differ between two arms' loaded configs."""

    one, other = (_flatten(_load(_rao_path(arm))[1]) for arm in (first, second))
    return {key for key in one.keys() | other.keys() if one.get(key) != other.get(key)}


def test_the_directory_holds_exactly_the_arms():
    # The linear-budget recursive arm, none of the paper's arms, is gone, and the depth-aware arm is recursive.yaml.
    assert sorted(path.stem for path in ARMS_DIR.glob("*.yaml")) == sorted(("common", *RAO_ARMS))


@pytest.mark.parametrize("arm", RAO_ARMS)
def test_every_arm_resolves_and_validates(arm, capsys):
    """As a run does when it starts: the named environment, the harness, the program, the reward processor and the
    service resolve (``platoon config validate --resolve``), and no key is deprecated (``--strict``)."""

    assert cli.config_validate(_rao_path(arm), resolve=True, strict=True) == 0, capsys.readouterr().out


@pytest.mark.parametrize("arm", RAO_ARMS)
def test_every_arm_names_the_environment_entry_it_would_spell_out(arm):
    """``env: textcraft-synth@1`` and the arm's own keys stand for the entry the arm would otherwise spell out: the
    extension's modules, the default program and platoon's ``platoon-delegation``."""

    (entry,) = load_yaml_config(_rao_path(arm))["environments"]
    spelled_out = {
        "service": "python://platoon_textcraft.env:TextCraftSynthEnvironment",
        "harness": "platoon-codeact",
        "harness_options": {
            "prompt_builder": "platoon_textcraft.prompts:TextCraftPromptBuilder",
            "prompt_builder_options": {"variant": RAO_VARIANTS[arm], "synth": True},
            **RAO_OPTIONS[arm],
            "cell_timeout_s": 7200,
        },
        "agents": RAO_AGENTS[arm],
        "reward_processor": "platoon-delegation",
        "dataset_kwargs": {"split_name": "train-medium"},
        "eval_dataset_kwargs": {"split_name": "val", "limit": 100},
    }

    expanded = expand_environment(entry)

    assert expanded.pop("env") == entry["env"] == "textcraft-synth@1"
    assert expanded == spelled_out


@pytest.mark.parametrize("arm", RAO_ARMS)
def test_rao_arms_share_the_paper_setting(arm):
    config, _ = _load(_rao_path(arm))
    train, workflow = config.train, config.train.workflow_config
    rollout, optimizer = workflow.rollout_config, train.optimizer
    inference = rollout.inference_params
    (environment,) = config.environments

    assert (train.model_name, train.renderer_name, train.lora_rank) == (
        "Qwen/Qwen3-4B-Instruct-2507",
        "qwen3_instruct",
        32,
    )
    assert (train.batch_size, workflow.group_size, train.max_staleness, train.num_epochs) == (16, 8, 3, 10)
    assert (train.context_window_length, train.loss_fn) == (40000, "cispo")
    assert train.loss_fn_config == {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0}
    # Adam as in the paper's runs, at the learning rate of a rank-32 LoRA: weight decay 0.017, the gradient norm
    # clipped at 1.
    assert (optimizer.learning_rate, optimizer.beta1, optimizer.beta2, optimizer.eps) == (3e-5, 0.9, 0.95, 1e-8)
    assert (optimizer.weight_decay, optimizer.grad_clip_norm) == (0.017, 1.0)
    # One minibatch of one microbatch: depth weights and the loss's normalization span the whole batch, as in the
    # paper's trainer.
    assert (train.num_minibatches, train.num_microbatches) == (1, 1)
    # A harness rollout: platoon-codeact runs every agent of the tree.
    assert environment.harness == "platoon-codeact"
    assert AutoRollout.from_config(config) is import_from_string(HARNESS_ROLLOUT)
    # The rollout's inference_params are every agent's default sampling; no harness option overrides them.  The
    # configs leave top_p to the gateway (1.0), and set the 512-token completions explicitly: an unset budget would
    # leave the gateway's default (4096).
    assert (inference.temperature, inference.top_p, inference.max_completion_tokens) == (1.0, None, 512)
    assert "sampling" not in environment.harness_options
    assert (rollout.timeout, rollout.step_timeout, environment.harness_options["cell_timeout_s"]) == (7200, 7200, 7200)
    # Eval every 5 steps rolls out exactly like training, one rollout per task, with the same sampling.
    eval_rollout = config.eval.workflow_config.rollout_config
    assert (config.eval.every, config.eval.workflow_config.group_size) == (5, 1)
    assert (eval_rollout.max_steps, eval_rollout.propagate_root_success, eval_rollout.timeout) == (
        rollout.max_steps,
        rollout.propagate_root_success,
        rollout.timeout,
    )
    assert eval_rollout.inference_params == inference
    assert (config.checkpoint.every, config.checkpoint.recovery_every) == (25, 5)
    # No delegation bonus (λ = 0, §3.1).
    assert environment.reward_processor == "platoon-delegation"
    assert AutoRewardProcessor.from_config(config) is import_from_string("platoon.rewards:delegation_capped")
    assert DELEGATION_REWARD_CAP == 0.0


@pytest.mark.parametrize("arm", RAO_ARMS)
def test_rao_arm_selects_its_prompts_agent_tree_and_step_budget(arm):
    config, _ = _load(_rao_path(arm))
    (environment,) = config.environments
    recursive = arm in RECURSIVE_ARMS

    assert environment.program == "single"
    assert environment.harness_options["prompt_builder_options"] == {"variant": RAO_VARIANTS[arm], "synth": True}
    builder = CodeActOptions.from_mapping(environment.harness_options).make_prompt_builder()
    assert isinstance(builder, TextCraftPromptBuilder)
    assert (builder.variant, builder.synth) == (RAO_VARIANTS[arm], True)
    assert environment.harness_options.get("subagents", True) is recursive
    agents = environment.agents
    assert {key: getattr(agents, key) for key in RAO_AGENTS[arm]} == RAO_AGENTS[arm]
    # The root's budget (A.5): 200 steps for the single agent, 25 for a recursive agent's root as for its subagents.
    for rollout in (config.train.workflow_config.rollout_config, config.eval.workflow_config.rollout_config):
        assert rollout.max_steps == (25 if recursive else 200)
    # Rollout groups at once, as the paper's runs: 16 for the linear run, 8 for the recursive one, of 8 rollouts each.
    assert concurrent_groups(config.train) == (8 if recursive else 16)
    assert config.train.max_concurrent_rollouts == (64 if recursive else 128)
    # The ablation trains for its 100 steps (§5.0.1); the arms for 10 epochs.
    assert config.train.max_training_steps == (100 if arm == "ablation_sparse" else None)


@pytest.mark.parametrize("arm", RAO_ARMS)
def test_rao_arm_credit_is_the_papers(arm):
    """Dense for the recursive arm, Sparse for the ablation, and leave-one-out baselines and depth weighting for both
    (§2.2, §5.0.1); the group's mean for the linear arm, whose paper runs had no leave-one-out baseline."""

    config, _ = _load(_rao_path(arm))
    workflow = config.train.workflow_config
    leave_one_out, depth_weighting, propagate = RAO_CREDIT[arm]

    assert (workflow.leave_one_out_baseline, workflow.depth_level_weighting) == (leave_one_out, depth_weighting)
    assert workflow.depth_level_discount_gamma is None
    for rollout in (workflow.rollout_config, config.eval.workflow_config.rollout_config):
        # Every subagent graded on its own targets keeps its grade (Dense), or takes the root's success (Sparse).
        assert rollout.propagate_root_success is propagate
        assert rollout.skip_subagent_reward_computation is False
    if arm in RECURSIVE_ARMS:
        assert config.environments[0].agents.child_grading == "env"
    expected_transforms = [DepthLevelWeightingTransform] if depth_weighting else []
    assert [type(transform) for transform in build_default_batch_transforms(config)] == expected_transforms


def test_the_recursive_arm_differs_from_the_linear_arm_only_in_agents_budget_concurrency_and_credit():
    allowed_prefixes = (
        "environments.0.harness_options.prompt_builder_options.variant",
        "environments.0.harness_options.subagents",
        "environments.0.harness_options.subagent_budget",
        "environments.0.agents.",
        "train.max_concurrent_rollouts",
        "train.workflow_config.leave_one_out_baseline",
        "train.workflow_config.depth_level_weighting",
        "train.workflow_config.rollout_config.max_steps",
        "train.workflow_config.rollout_config.propagate_root_success",
        "stats.trial_name",
        "stats.wandb.tags",
    )

    differing = _differing("linear", "recursive")

    assert "train.workflow_config.leave_one_out_baseline" in differing
    assert not [key for key in differing if not key.startswith(allowed_prefixes)]


def test_the_sparse_ablation_differs_from_the_recursive_arm_only_in_credit_length_and_name():
    assert _differing("recursive", "ablation_sparse") == {
        "train.max_training_steps",
        "train.workflow_config.rollout_config.propagate_root_success",
        "stats.trial_name",
        "stats.wandb.tags",
    }


def test_the_arms_train_and_log_runs_of_their_own():
    # A run resumes from the checkpoints in its directory (log_path/experiment_name/trial_name), so no two arms share
    # one.
    names = {_load(_rao_path(arm))[0].stats.trial_name for arm in RAO_ARMS}
    assert len(names) == len(RAO_ARMS)


def test_an_unweighted_variant_of_fig_8_is_an_override_away():
    config, _ = _load(_rao_path("ablation_sparse"), "train.workflow_config.depth_level_weighting=false")

    assert not config.train.workflow_config.depth_level_weighting
    assert build_default_batch_transforms(config) == []


@pytest.mark.parametrize("arm", RAO_ARMS)
def test_rao_datasets_are_the_medium_train_tasks_and_the_first_val_tasks(arm):
    config, _ = _load(_rao_path(arm))
    train = AutoDataset.from_config(config, "train")
    evaluation = AutoDataset.from_config(config, "eval")
    get_task = AutoTaskLoader.from_config(config)

    assert len(train) == 852 and train["task_id"][0] == "train-medium:0"
    assert evaluation["task_id"] == [f"val:{index}" for index in range(100)]
    assert get_task(train["task_id"][0]).id == "train-medium:0"
