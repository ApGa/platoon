from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"
CONFIG_NAME = (
    "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_"
    "qwen36_moe_cp2_c24_ta20_v19"
)
WRAPPER = REPO_ROOT / "pbs-scripts/openreward-ta-swe-32node-recursive-qwen36-cp2-h200-v10-abci.pbs"
BACKEND = "megatron:(attn:d10p2t4c2|ffn:d10p2t1e8)"


def _merge(base, overlay):
    if isinstance(base, dict) and isinstance(overlay, dict):
        result = dict(base)
        for key, value in overlay.items():
            result[key] = _merge(result[key], value) if key in result else value
        return result
    return overlay


def _compose(name: str, ancestry: tuple[str, ...] = ()) -> dict:
    yaml = pytest.importorskip("yaml")
    if name in ancestry:
        raise AssertionError(f"recursive defaults: {name}")
    leaf = yaml.safe_load((CONFIG_ROOT / f"{name}.yaml").read_text())
    defaults = leaf.pop("defaults", [])
    config: dict = {}
    for default in defaults:
        if default == "_self_":
            config = _merge(config, leaf)
        else:
            config = _merge(config, _compose(default, (*ancestry, name)))
    if "_self_" not in defaults:
        config = _merge(config, leaf)
    return config


def test_qwen36_cp2_leaf_composes_exact_user_contract() -> None:
    config = _compose(CONFIG_NAME)
    workflow = config["workflow_config"]
    openreward = config["openreward"]
    actor = config["actor"]

    assert config["experiment_name"] == "abci-or-ta-swe32-q36-cp2-v19"
    assert config["trial_name"] == "abci-ta-swe32-rec-q36m-cp2-c24-d1-v19-t0"
    assert config["dynamic_bs"] is False
    assert workflow["filter_zero_advantage_datums"] is False
    assert workflow["filter_zero_variance_groups"] is False
    assert workflow["dynamic_batch_recollection_max_rounds"] == 1
    assert workflow["min_successful_group_size"] == 5
    assert workflow["rollout_config"]["max_steps"] == 150
    assert openreward["subagent_default_max_steps"] == 100
    assert openreward["subagent_reward_judge_max_steps"] == 100
    assert openreward["subagent_max_depth"] == 1

    assert actor["backend"] == BACKEND
    assert config["ref"]["backend"] == "${actor.backend}"
    assert config["rollout"]["backend"] == "sglang:d12p1t8"
    assert 10 * 2 * 4 * 2 == 10 * 2 * 1 * 8 == 160
    assert 160 + 12 * 1 * 8 == config["cluster"]["n_nodes"] * 8 == 256
    assert actor["enable_router_replay"] is True
    assert actor["router_replay_num_layers"] == 40
    assert actor["router_replay_topk"] == 8
    assert actor["router_replay_num_experts"] == 256
    assert config["rollout"]["return_routed_experts"] is True
    assert actor["megatron"]["use_deterministic_algorithms"] is True

    for engine in (actor, config["ref"]):
        assert engine["mb_spec"] == {
            "n_mbs": 1,
            "granularity": 1,
            "max_tokens_per_mb": 65_536,
            "n_mbs_divisor": 1,
        }
    assert config["sglang"]["context_length"] == 65_537
    assert config["sglang"]["max_prefill_tokens"] == 65_536
    assert openreward["condenser_max_tokens"] == (4 * 65_536) // 5 == 52_428

    engine_divisor = math.lcm(2, actor["mb_spec"]["n_mbs_divisor"])
    engine_min = math.ceil(max(actor["mb_spec"]["n_mbs"], 2 * 2) / engine_divisor) * engine_divisor
    divisor = math.lcm(10 * 1 * 1 * engine_divisor, config["rollout"]["ensure_batch_divisible_by"])
    minimum = 10 * 1 * 1 * engine_min
    assert (engine_divisor, engine_min, divisor, minimum) == (2, 4, 20, 40)
    assert config["train_dataset"]["batch_size"] * workflow["min_successful_group_size"] == minimum
    assert config["rollout"]["consumer_batch_size"] == config["train_dataset"]["batch_size"] == 8
    assert config["recover"]["mode"] == "auto"
    assert config["recover"]["freq_steps"] == 1


def test_pinned_areal_transformer_proves_hybrid_cp2_ep8_mapping() -> None:
    candidates = (
        REPO_ROOT / "external/AReaL",
        Path("/home/aci18912it/rasp/releases/platoon-v8r17-376d1b9-18e62c0/external/AReaL"),
    )
    areal_root = next((path for path in candidates if path.is_dir()), None)
    if areal_root is None:
        pytest.skip("materialized pinned AReaL checkout is unavailable")
    alloc_mode = areal_root / "areal/api/alloc_mode.py"
    source = alloc_mode.read_text(encoding="utf-8")
    assert 'ATTN_DIM_TYPE: "c" | "d" | "t" | "p"' in source
    assert 'FFN_DIM_TYPE: "d" | "e" | "t" | "p"' in source
    assert "Context parallelism is only effective for attention modules" in source
    assert "Expert parallelism is only effective for MoE (Mixture of Experts) modules" in source
    # Execute the pinned parser's hybrid transformer while stubbing only its
    # unrelated CLI/logging imports. This avoids a mutable controller venv.
    code = f"""
import dataclasses, enum, importlib.util, json, logging, sys, types
areal = types.ModuleType("areal")
api = types.ModuleType("areal.api")
utils = types.ModuleType("areal.utils")
cli = types.ModuleType("areal.api.cli_args")
lark = types.ModuleType("lark")
class Transformer:
    pass
class Lark:
    def __init__(self, *args, **kwargs):
        pass
lark.Transformer = Transformer
lark.Lark = Lark
class SchedulingStrategyType(enum.Enum):
    separation = "separation"
    colocation = "colocation"
@dataclasses.dataclass
class SchedulingStrategy:
    type: object
    target: object
cli.SchedulingStrategy = SchedulingStrategy
cli.SchedulingStrategyType = SchedulingStrategyType
utils.logging = logging
sys.modules.update({{
    "areal": areal,
    "areal.api": api,
    "areal.api.cli_args": cli,
    "areal.utils": utils,
    "lark": lark,
}})
spec = importlib.util.spec_from_file_location("areal.api.alloc_mode", {str(alloc_mode)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
transformer = module._ParallelStrategyTransformer()
attn = [transformer.attn_dim([name, value]) for name, value in (("d", 10), ("p", 2), ("t", 4), ("c", 2))]
ffn = [transformer.ffn_dim([name, value]) for name, value in (("d", 10), ("p", 2), ("t", 1), ("e", 8))]
p = transformer.hybrid_moe_syntax([transformer.attn_section(attn), transformer.ffn_section(ffn)])
print(json.dumps([p.dp_size, p.pp_size, p.tp_size, p.cp_size, p.ep_size, p.etp_size, p.edp_size, p.world_size]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert json.loads(completed.stdout) == [10, 2, 4, 2, 8, 1, 10, 160]


def test_filter_flags_map_to_the_two_executable_zero_signal_paths() -> None:
    workflow_source = (
        REPO_ROOT / "platoon/train/areal/workflows/group_rollout_workflow.py"
    ).read_text()
    trainer_source = (REPO_ROOT / "platoon/train/areal/rl.py").read_text()

    assert "if self.config.filter_zero_variance_groups:" in workflow_source
    assert 'getattr(self.config, "filter_zero_advantage_datums", False)' in workflow_source
    assert '"filter_zero_advantage_datums",' in trainer_source
    assert "_filter_zero_centered_reward_batch(" in trainer_source


def test_fresh_wrapper_is_fail_closed_and_no_resume_path() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    source = WRAPPER.read_text()

    assert "#PBS -l select=32:mpiprocs=8" in source
    assert "#PBS -N rasp-h200-32q36-v19" in source
    assert "platoon-v8r18-376d1b9-18e62c0" in source
    assert "refs/tags/abci-v8r18-376d1b9-18e62c0" in source
    assert "refusing stale success artifact" in source
    assert "refusing stale checkpoint" in source
    assert "OPENREWARD_EXPECTED_MODEL_SHARDS=26" in source
    assert "OPENREWARD_EXPECTED_RESUME_GLOBAL_STEP" in source
    assert "unset OPENREWARD_REQUIRE_EXISTING_RECOVERY" in source
    assert "RESUME_VERIFY=" not in source
    assert "SOURCE_RELEASE_ROOT=" not in source
    assert "qsub" not in source.lower()


def test_rollout_uses_typed_condenser_trigger_instead_of_32k_literal() -> None:
    source = (REPO_ROOT / "plugins/openreward/platoon/openreward/rollout.py").read_text()
    assert "max_tokens=openreward_config.condenser_max_tokens" in source
    assert "max_tokens=int(0.8 * 32768)" not in source
