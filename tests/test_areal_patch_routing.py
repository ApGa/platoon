from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCHES_PATH = REPO_ROOT / "platoon" / "train" / "areal" / "patches.py"
ACTOR_PATH = REPO_ROOT / "platoon" / "train" / "areal" / "actor.py"
ACTOR_CONTROLLER_PATH = REPO_ROOT / "platoon" / "train" / "areal" / "actor_controller.py"
PACKAGE_PATH = REPO_ROOT / "platoon" / "train" / "areal" / "__init__.py"
RL_PATH = REPO_ROOT / "platoon" / "train" / "areal" / "rl.py"

CONTROLLER_PATCHES = (
    "_patch_areal_local_scalar_stats_export",
    "_patch_hf_tokenizer_download_race",
    "_patch_model_response_custom_stop_sequences",
    "_patch_areal_openai_routed_experts_transport",
    "_patch_areal_sglang_routed_experts_launcher",
    "_patch_areal_routed_experts_unpadding",
    "_patch_triton_cache_for_qwen35_gdn_cp",
    "_patch_areal_qwen35_gdn_cp_guards",
    "_patch_batch_task_dispatcher_idle_submit",
    "_patch_rollout_controller_least_loaded_workers",
    "_patch_rollout_controller_pause_rpc_policy",
    "_patch_local_scheduler_fork_ready_timeout",
    "_patch_slurm_scheduler_engine_call_error_retry",
    "_patch_remote_inf_engine_routed_expert_stitching",
    "_patch_remote_inf_engine_asyncio_teardown_race",
    "_patch_remote_inf_engine_proxy_resolution",
    "_patch_areal_proxy_rollout_fork_command",
    "_patch_areal_fork_alias_cleanup",
    "_patch_rollout_controller_cleanup_error_propagation",
    "_patch_areal_openai_message_content_flatten",
    "_patch_areal_openai_reasoning_chat_template",
    "_patch_areal_recover_checkpoint_rotation",
    "_install_process_stall_watchdog",
)
MEGATRON_PATCHES = (
    "_patch_megatron_bridge_attention_backend",
    "_patch_megatron_bridge_qwen35_tp_validation",
    "_patch_megatron_bridge_qwen35_cp_constructor_loss_mode",
    "_patch_megatron_checkpoint_optimizer_bucket_shapes",
    "_patch_areal_megatron_memory_compatibility",
    "_patch_megatron_bridge_qwen3vl_already_cp_local_packed_input",
    "_patch_megatron_core_qwen35_already_cp_local_thd_rope",
    "_patch_megatron_core_gdn_context_parallel_config_validation",
    "_patch_megatron_core_gated_delta_net_context_parallel",
)
ROLLOUT_PATCHES = ("_patch_remote_inf_engine_server_child_exit",)


def _load_patches(name: str):
    spec = importlib.util.spec_from_file_location(name, PATCHES_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record_patch_calls(module, names: tuple[str, ...]) -> list[str]:
    calls: list[str] = []
    for name in names:
        setattr(module, name, lambda name=name: calls.append(name))
    return calls


class AReaLPatchRoutingTests(unittest.TestCase):
    def test_controller_and_worker_patch_sets_are_complete_ordered_and_idempotent(self) -> None:
        module = _load_patches("platoon_areal_patch_routing_test")
        calls = _record_patch_calls(
            module,
            CONTROLLER_PATCHES + ROLLOUT_PATCHES + MEGATRON_PATCHES,
        )

        module.apply_controller_safe_patches()
        module.apply_controller_safe_patches()
        self.assertEqual(calls, list(CONTROLLER_PATCHES))

        module.apply_rollout_worker_patches()
        module.apply_rollout_worker_patches()
        self.assertEqual(calls, [*CONTROLLER_PATCHES, *ROLLOUT_PATCHES])

        module.apply_megatron_worker_patches()
        module.apply_megatron_worker_patches()
        self.assertEqual(
            calls,
            [*CONTROLLER_PATCHES, *ROLLOUT_PATCHES, *MEGATRON_PATCHES],
        )

    def test_apply_all_patches_remains_backward_compatible(self) -> None:
        module = _load_patches("platoon_areal_all_patches_test")
        calls = _record_patch_calls(
            module,
            CONTROLLER_PATCHES + ROLLOUT_PATCHES + MEGATRON_PATCHES,
        )

        module.apply_all_patches()
        module.apply_all_patches()
        self.assertEqual(
            calls,
            [*CONTROLLER_PATCHES, *ROLLOUT_PATCHES, *MEGATRON_PATCHES],
        )

    def test_package_and_actor_route_megatron_patches_only_to_worker(self) -> None:
        package_source = PACKAGE_PATH.read_text()
        self.assertIn("apply_controller_safe_patches()", package_source)
        self.assertNotIn("apply_all_patches()", package_source)
        self.assertNotIn("apply_megatron_worker_patches()", package_source)

        actor_source = ACTOR_PATH.read_text()
        call_then_import = "    apply_megatron_worker_patches()\n    from areal.engine import MegatronPPOActor\n"
        self.assertIn(call_then_import, actor_source)

        controller_source = ACTOR_CONTROLLER_PATH.read_text()
        self.assertNotIn("areal.engine", controller_source)
        rl_source = RL_PATH.read_text()
        self.assertNotIn(
            "from platoon.train.areal.actor import PlatoonPPOActor,",
            rl_source,
        )
        self.assertIn(
            '"PlatoonPPOActor",\n    (),\n    {"__module__": "platoon.train.areal.actor"}',
            rl_source,
        )


if __name__ == "__main__":
    unittest.main()
