#!/usr/bin/env python3
"""Fail-closed verification for the immutable ABCI v8r18/v9 checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path

from verify_openreward_v8_swe_env import verify as verify_swe_runtime

RELEASE_ROOT = Path("/home/aci18912it/rasp/releases/platoon-v8r18-376d1b9-18e62c0")
RELEASE_REF = "refs/tags/abci-v8r18-376d1b9-18e62c0"
MANIFEST_RELATIVE = Path("pbs-scripts/openreward-v9-release.json")
MANIFEST_SHA256 = "22f20679acf73d8a9599a8fe2c9e34f1c33308cd4f24536b5c67b4ee452d7bf1"
UPSTREAM_COMMIT = "376d1b9885ea40464766fb3fea74815c6a698fbf"
TOOLATHLON_BUILD_COMMIT = "69d04f210097870a984d26024016a705a278c1d2"
TOOLATHLON_IMAGE_PATH = Path(
    "/groups/gai51740/aci18912it/rasp/images/openreward-toolathlon-18e62c0d-pbs-53541066e97b.sif"
)
TOOLATHLON_IMAGE_SHA256 = "11d158f7c1a76486f1338f8cf90ce8a0280962792c0e5c95559f19cc391cda14"
TOOLATHLON_RECEIPT_PATH = Path(
    "/groups/gai51740/aci18912it/rasp/images/build-receipts/openreward-toolathlon-18e62c0d-pbs-53541066e97b.sif.json"
)
TOOLATHLON_RECEIPT_SHA256 = "37286eeb005f83e3b9133ceb8204b12d93220ac02cc033118e98c5f314908598"
TOOLATHLON_BASE_PATH = Path("/groups/gai51740/aci18912it/rasp/images/openreward-toolathlon-18e62c0d-c1e60b36.sif")
TOOLATHLON_BASE_SHA256 = "e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267"
TOOLATHLON_BASE_OCI = "sha256:c1e60b3673c848b758bda1b32ec39898feb5d6ff4c7e55ead8661adb04b327bd"
TOOLATHLON_PATCHED_SERVER_SHA256 = "26f58c49d42975616c5bef19b654cc4483e665b2107533000861cecbcd74c8a5"
PATCH_ASSETS = (
    "toolathlon-pbs-portable.def",
    "patch_toolathlon_pbs_portable.py",
    "repair_toolathlon_snowflake_catalog.py",
    "launch_toolathlon_python_mcp.py",
    "probe_toolathlon_python_mcp_launchers.py",
    "verify_toolathlon_pbs_portability.sh",
)
RECEIPT_KEYS = {
    "authorization_scope",
    "base_image",
    "build_tooling",
    "completed_at_epoch",
    "derived_image",
    "embedded",
    "gate",
    "labels",
    "patch_assets",
    "pbs_accounting_attested",
    "portability_patch_sha256",
    "schema_version",
    "status",
    "toolathlon_source_revision",
}
EMBEDDED_RECEIPT_KEYS = {
    "assets",
    "base_server_sha256",
    "base_tool_schemas_sha256",
    "manifest_path",
    "manifest_sha256",
    "patched_server_sha256",
    "playwright",
    "python_mcp",
    "repaired_tool_schemas_sha256",
    "verify",
}
COMPONENTS = {
    "platoon_upstream": UPSTREAM_COMMIT,
    "areal": "d99124ec15102ca2fcd4960cc8beaef3950c2672",
    "openhands": "b1acf52ee362884f092b9eaf5e701a1bb5d1b335",
    "swe_rebench": "538b5c5c9c529d3c78d5ef7fd3df7e54e7902ad7",
    "toolathlon": "18e62c0d0419da88209af911528f2f8a0b44a5ed",
}
MODEL_ARTIFACTS = {
    "nonrecursive_16node": {
        "architecture": "dense_transformer",
        "repository": "apurvaga/Qwen3.8-27B-preserve-thinking",
        "revision": "97c725615b8aa025b800da4ceed38811358b30f1",
        "release_tag": "preserve-thinking-v1",
        "upstream_source": "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "thinking_policy": "preserved",
        "safetensors_shard_count": 18,
        "snapshot_path": (
            "/groups/gai51740/aci18912it/rasp/cache/huggingface/hub/"
            "models--apurvaga--Qwen3.8-27B-preserve-thinking/snapshots/"
            "97c725615b8aa025b800da4ceed38811358b30f1"
        ),
    },
    "recursive_32node": {
        "architecture": "dense_transformer",
        "repository": "apurvaga/Qwen3.8-27B-preserve-thinking",
        "revision": "97c725615b8aa025b800da4ceed38811358b30f1",
        "release_tag": "preserve-thinking-v1",
        "upstream_source": "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "thinking_policy": "preserved",
        "safetensors_shard_count": 18,
        "snapshot_path": (
            "/groups/gai51740/aci18912it/rasp/cache/huggingface/hub/"
            "models--apurvaga--Qwen3.8-27B-preserve-thinking/snapshots/"
            "97c725615b8aa025b800da4ceed38811358b30f1"
        ),
    },
    "fresh_recursive_32node_qwen36_cp2": {
        "architecture": "qwen3_5_moe",
        "repository": "apurvaga/Qwen3.6-35B-A3B-preserve-thinking",
        "revision": "ce815485d640a33461c25e3d5c505b5346e8c857",
        "upstream_source": "Qwen/Qwen3.6-35B-A3B",
        "thinking_policy": "preserved",
        "weights_policy": "unchanged_from_upstream",
        "safetensors_shard_count": 26,
        "snapshot_path": (
            "/groups/gai51740/aci18912it/rasp/cache/huggingface/hub/"
            "models--apurvaga--Qwen3.6-35B-A3B-preserve-thinking/snapshots/"
            "ce815485d640a33461c25e3d5c505b5346e8c857"
        ),
    },
}
EXTERNAL_PATHS = {
    "areal": "external/AReaL",
    "openhands": "external/openhands-agent-sdk",
    "swe_rebench": "external/swe-rebench-v2-openrewardenv",
    "toolathlon": "external/toolathlon-gym",
}
TRAINING_RUNTIME_ASSETS = {
    "platoon/train/areal/rl.py": "9ce579fa76b43d61223cd9a6c9f5bebec45b7801c5407e2258825a86f71c765b",
    "plugins/openreward/platoon/openreward/config_defs.py": (
        "f8acd3bbbe9738211c8103f38c23b6fe9f65c4b11e00b505240e969b6588c51e"
    ),
    "plugins/openreward/platoon/openreward/rollout.py": (
        "653f501c985bd84b3bc5553849d224755fc5c119ae461c4ba5ea8e753b8dfd44"
    ),
}
TRAINABLE_DATUMS_MISSING_MASK_NORMALIZATION = "missing_masks_treated_as_all_trainable_before_filter_and_concat"
TRAINING_SEMANTICS = {
    "full_horizon": True,
    "total_train_epochs": 10,
    "total_train_steps": None,
    "qwen35_gdn_cp_size": 1,
    "qwen35_gdn_packed": False,
    "qwen35_gdn_layout_selection": "model_architecture_and_context_parallel_topology",
    "qwen35_gdn_cp1_layout": "padded_bshd_upstream_gdn",
    "qwen35_gdn_cp_gt1_layout": "packed_thd_platoon_custom_gdn_cp",
    "r3_padded_routes_layout": "sequence_major",
    "dynamic_bs": True,
    "filter_zero_advantage_datums": True,
    "filter_zero_variance_groups": True,
    "trainable_datums_missing_mask_normalization": TRAINABLE_DATUMS_MISSING_MASK_NORMALIZATION,
    "rollout_max_head_offpolicyness_16node": 1,
    "rollout_max_head_offpolicyness_32node": 1,
    "max_concurrent_rollouts_16node": 12,
    "max_concurrent_rollouts_32node": 24,
    "actor_safe_dynamic_batch": {
        "minimum_policy": "topology_derived",
        "divisor_policy": (
            "lcm(actor_dp_times_ppo_minibatches_times_microbatch_granularity_"
            "times_engine_microbatch_divisor,"
            "rollout_ensure_batch_divisible_by)"
        ),
        "engine_microbatch_divisor_policy": ("lcm(pipeline_parallel_size,configured_n_mbs_divisor)"),
        "engine_min_microbatches_policy": ("engine_divisor_aligned_max_of_configured_n_mbs_and_2pp_when_pp_gt_1"),
        "under_minimum_policy": "bounded_recollection_then_skip_optimizer_step",
        "leaf_contracts": {
            "16node_nonrecursive": {
                "actor_dp_size": 10,
                "actor_pipeline_parallel_size": 2,
                "actor_engine_min_microbatches": 8,
                "actor_batch_divisor": 20,
                "actor_safe_minimum_datums": 80,
                "dynamic_batch_recollection_max_rounds": 1,
                "rollout_consumer_batch_size": 8,
            },
            "32node_recursive": {
                "actor_dp_size": 20,
                "actor_pipeline_parallel_size": 2,
                "actor_engine_min_microbatches": 8,
                "actor_batch_divisor": 40,
                "actor_safe_minimum_datums": 160,
                "dynamic_batch_recollection_max_rounds": 2,
                "rollout_consumer_batch_size": 16,
            },
        },
    },
    "balance_accepted_batches": False,
    "resume_training_start_guard": "require_loaded_recover_info_global_step_and_next_step_before_loop",
    "success_candidate_requires_actor_update_and_weight_sync": True,
    "fresh_16node_model_only_lineage": False,
    "fresh_16node_lineage": "v18_qwen38_dense_resume_global_step0_source_job_2155389",
    "nonrecursive_16node_dense_contract": {
        "actor_backend": "megatron:d10p2t4c1",
        "actor_topology": {
            "nodes": 10,
            "gpus": 80,
            "data_parallel_size": 10,
            "pipeline_parallel_size": 2,
            "tensor_parallel_size": 4,
            "context_parallel_size": 1,
            "expert_parallelism": False,
        },
        "rollout_backend": "sglang:d6p1t8",
        "rollout_topology": {
            "nodes": 6,
            "gpus": 48,
            "replicas": 6,
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 8,
        },
        "r3_enabled": False,
        "return_routed_experts": False,
        "actor_router_replay_enabled": False,
        "actor_megatron_use_deterministic_algorithms": False,
        "actor_mtp_enabled": False,
        "expected_model_shards": 18,
        "torch_nccl_heartbeat_timeout_sec": 3600,
    },
    "nonrecursive_16node_resume_contract": {
        "source_pbs_job_id": "2155389.pbs1",
        "experiment_name": "abci-or-ta-swe16-q38-v18",
        "trial_name": "abci-ta-swe16-nr-q38d-c12-v18-t0",
        "recover_mode": "on",
        "allow_preupdate_global_step_zero": True,
        "preupdate_binding_id": "qwen38-16-v18-job2155389",
        "recover_global_step": 0,
        "next_global_step": 1,
        "steps_per_epoch": 846,
        "checkpoint_root": (
            "/groups/gai51740/aci18912it/rasp/experiments/areal/checkpoints/aci18912it/"
            "abci-or-ta-swe16-q38-v18/abci-ta-swe16-nr-q38d-c12-v18-t0"
        ),
        "recover_checkpoint_shards": 80,
        "recover_checkpoint_total_bytes": 164220069555,
        "checkpoint_layout_sha256": (
            "25af0d65a233ee77165489dad0c4bbe43a0a56b6745aa9785eddb23f5b720d0c"
        ),
        "checkpoint_sample_manifest_sha256": (
            "3ab2c47cbe8b1d95f1c13ce09d026c57da8752db14749e0ce4011a762dd9505e"
        ),
        "checkpoint_fixed_content_sha256": (
            "bb5ec86330f5acfdae93b04e6101d88a7244e702fcda812ccc0a97adcc6bf6da"
        ),
        "recover_info_content_sha256": (
            "bfb4e310dc3f28946bd86b437cd8d0267875fd07a627346b151811ab83fd2bdb"
        ),
        "runtime_semantic_verification": "pinned_training_sif_before_controller",
        "dcp_state_keys": 1456,
        "dcp_storage_entries": 9169,
        "dcp_storage_coverage": "80_exact_paths_gap_free_and_in_bounds",
        "lr_scheduler_num_steps": 0,
        "dataloader_samples_yielded": 72,
        "dataloader_iterator_yielded": 9,
        "source_release_root": "/home/aci18912it/rasp/releases/platoon-v8r15-376d1b9-18e62c0",
        "source_release_ref": "refs/tags/abci-v8r15-376d1b9-18e62c0",
        "config_sha256": "0ff5430c6be22ff8419af655243ed005757880bcafa61e40381175fa3bc6ab39",
        "source_log": "/groups/gai51740/aci18912it/rasp/logs/2155389.pbs1.OU",
        "source_log_sha256": (
            "2497762da279fd93486521b9bf8dad78c02d0fc0bec8f39e7172d188cbcdd552"
        ),
        "source_candidate_absent": True,
        "source_success_receipt_absent": True,
        "target_receipt": (
            "/groups/gai51740/aci18912it/rasp/job-state/gates/"
            "abci-ta-swe16-nr-q38d-c12-v18-t0.resume-gs0-v8r18.success.json"
        ),
        "success_receipt_isolated": True,
        "pbs_walltime_hours": 72,
        "checkpoint_artifact_policy": "preserve_in_place_no_rename_move_delete",
    },
    "recursive_32node_lineage": "v16_qwen38_dense_resume_global_step5_source_job_2154492",
    "recursive_32node_dense_contract": {
        "actor_backend": "megatron:d20p2t4c1",
        "actor_topology": {
            "nodes": 20,
            "gpus": 160,
            "data_parallel_size": 20,
            "pipeline_parallel_size": 2,
            "tensor_parallel_size": 4,
            "context_parallel_size": 1,
            "expert_parallelism": False,
        },
        "rollout_backend": "sglang:d12p1t8",
        "rollout_topology": {
            "nodes": 12,
            "gpus": 96,
            "replicas": 12,
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 8,
        },
        "r3_enabled": False,
        "return_routed_experts": False,
        "actor_router_replay_enabled": False,
        "actor_megatron_use_deterministic_algorithms": False,
        "actor_mtp_enabled": False,
        "expected_model_shards": 18,
    },
    "recursive_32node_resume_contract": {
        "source_pbs_job_id": "2154492.pbs1",
        "experiment_name": "abci-or-ta-swe32-q38-v16",
        "trial_name": "abci-ta-swe32-rec-q38d-c24-v16-t0",
        "recover_mode": "on",
        "recover_global_step": 5,
        "next_global_step": 6,
        "steps_per_epoch": 846,
        "checkpoint_root": (
            "/groups/gai51740/aci18912it/rasp/experiments/areal/checkpoints/aci18912it/"
            "abci-or-ta-swe32-q38-v16/abci-ta-swe32-rec-q38d-c24-v16-t0"
        ),
        "recover_checkpoint_shards": 160,
        "recover_checkpoint_total_bytes": 383216730907,
        "checkpoint_layout_sha256": (
            "6699027815f1628a6522c0520d3e30b17df124a41e3583b6ff4a3e4937b556f8"
        ),
        "checkpoint_sample_manifest_sha256": (
            "23aea8bf8cc932936d250dd1646d191efe664047169445da1ce2ef67d7168e11"
        ),
        "checkpoint_fixed_content_sha256": (
            "ff7dccef8552800b23f1888cbf646c9b8f8c9019f5e83831264cd75ad6a15e6d"
        ),
        "recover_info_content_sha256": (
            "86e7b2726bcbc86328deb65dae0544c30afec2664d548b349aa87224626cfd3c"
        ),
        "runtime_semantic_verification": "pinned_training_sif_before_controller",
        "dcp_state_keys": 1472,
        "dcp_storage_entries": 18297,
        "dcp_storage_coverage": "160_exact_paths_gap_free_and_in_bounds",
        "lr_scheduler_num_steps": 6,
        "dataloader_samples_yielded": 208,
        "dataloader_iterator_yielded": 26,
        "source_release_root": "/home/aci18912it/rasp/releases/platoon-v8r13-376d1b9-18e62c0",
        "source_release_ref": "refs/tags/abci-v8r13-376d1b9-18e62c0",
        "config_sha256": "a3f8c04b2a3e59edd708f4f3a39eb38e1a22fa72b63a1f962afb54ac80183ffa",
        "source_candidate": (
            "/groups/gai51740/aci18912it/rasp/job-state/gates/"
            "abci-ta-swe32-rec-q38d-c24-v16-t0.success.candidate.json"
        ),
        "source_candidate_sha256": (
            "b432e3a12ff78921e7aac6cff3914f1c4c5aeffd9f08e3c9918ad5daa7714a22"
        ),
        "target_receipt": (
            "/groups/gai51740/aci18912it/rasp/job-state/gates/"
            "abci-ta-swe32-rec-q38d-c24-v16-t0.resume-gs5-v8r16.success.json"
        ),
        "success_receipt_isolated": True,
        "checkpoint_artifact_policy": "preserve_in_place_no_rename_move_delete",
    },
    "recursive_32node_qwen38_v20_successor_contract": {
        "binding_status": "frozen_gs6_checkpoint",
        "launchable": True,
        "fail_closed_before_services": True,
        "successor_binding_id": "qwen38-32-v20-post-step6",
        "config_relative": (
            "plugins/openreward/platoon/openreward/configs/areal/"
            "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_"
            "qwen38_dense_c24_ta20_v20.yaml"
        ),
        "config_sha256": "f3ffb459d1ed6e9427124e8e7d6daafebb2709f965f86c5629a99e508449fe5c",
        "wrapper_relative": (
            "pbs-scripts/openreward-ta-swe-32node-recursive-qwen38-resume-h200-v11-abci.pbs"
        ),
        "wrapper_sha256": "9aa09ba8c1d97485b1f2730e1687158904c3461d4269e943b2dccc46f582a29e",
        "experiment_name": "abci-or-ta-swe32-q38-v16",
        "trial_name": "abci-ta-swe32-rec-q38d-c24-v16-t0",
        "pbs_job_name": "rasp-h200-32q38-rsp6-v20",
        "checkpoint_root": (
            "/groups/gai51740/aci18912it/rasp/experiments/areal/checkpoints/aci18912it/"
            "abci-or-ta-swe32-q38-v16/abci-ta-swe32-rec-q38d-c24-v16-t0"
        ),
        "target_receipt": (
            "/groups/gai51740/aci18912it/rasp/job-state/gates/"
            "abci-ta-swe32-rec-q38d-c24-v16-t0.resume-post-step6-v8r18.success.json"
        ),
        "source_config_relative": (
            "plugins/openreward/platoon/openreward/configs/areal/"
            "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_"
            "qwen38_dense_c24_ta20_v16.yaml"
        ),
        "source_config_sha256": (
            "a3f8c04b2a3e59edd708f4f3a39eb38e1a22fa72b63a1f962afb54ac80183ffa"
        ),
        "source_candidate": (
            "/groups/gai51740/aci18912it/rasp/job-state/gates/"
            "abci-ta-swe32-rec-q38d-c24-v16-t0.resume-gs5-v8r16.success.candidate.json"
        ),
        "authorized_ancestor_candidate": (
            "/groups/gai51740/aci18912it/rasp/job-state/gates/"
            "abci-ta-swe32-rec-q38d-c24-v16-t0.success.candidate.json"
        ),
        "authorized_ancestor_candidate_sha256": (
            "b432e3a12ff78921e7aac6cff3914f1c4c5aeffd9f08e3c9918ad5daa7714a22"
        ),
        "source_pbs_job_id": "2156202.pbs1",
        "source_release_root": "/home/aci18912it/rasp/releases/platoon-v8r16-376d1b9-18e62c0",
        "source_release_ref": "refs/tags/abci-v8r16-376d1b9-18e62c0",
        "source_candidate_sha256": (
            "12a5a385826dbb2efb4a6dbffcdc057dce7063def8ade0c0d131dae1f802cd94"
        ),
        "source_log": "/groups/gai51740/aci18912it/rasp/logs/2156202.pbs1.OU",
        "source_log_sha256": (
            "08a2837d33589671fae43e796a981ad41e16ca335ab696a1748cd660f1a1b991"
        ),
        "source_terminal_accounting": (
            "live_qstat_F_substate91_exit271_run1_obit_"
            "2026-08-17T15:56:52+09:00_authorized_qdel"
        ),
        "recover_global_step": 6,
        "recover_checkpoint_total_bytes": 383216730907,
        "checkpoint_layout_sha256": (
            "6699027815f1628a6522c0520d3e30b17df124a41e3583b6ff4a3e4937b556f8"
        ),
        "checkpoint_sample_manifest_sha256": (
            "57410a3d50b2a3bf45991568abc687f334f4d02179a7d207da9c32e1872735c4"
        ),
        "checkpoint_fixed_content_sha256": (
            "9417c54953c5651d5c9a73649657ca50da74995847d77258e0a256e193674251"
        ),
        "recover_info_content_sha256": (
            "00c8d7ccdd27f49bd248369ffa45416d89b53dcb23a45ec9d5ea523aeeb7bddb"
        ),
        "dcp_state_keys": 1472,
        "dcp_storage_entries": 18297,
        "lr_scheduler_num_steps": 7,
        "dataloader_samples_yielded": 336,
        "dataloader_iterator_yielded": 42,
        "required_recover_global_step": 6,
        "recover_stats_logger_last_commit_step": 4,
        "candidate_stats_logger_last_commit_step": 6,
        "stats_recovery_order": "load_gs5_cursor4_save_gs6_before_commit_then_candidate_commit6_next_resume_logs7",
        "root_max_steps": 150,
        "subagent_max_steps": 100,
        "subagent_reward_judge_max_steps": 100,
        "workflow_timeout_seconds": 7200,
        "agent_act_and_env_step_timeout_seconds": 5400,
        "subprocess_init_budget_seconds": 120,
        "subprocess_cleanup_grace_seconds": 60,
        "subprocess_sigalrm_seconds": 7380,
        "parent_future_grace_seconds": 30,
        "parent_future_timeout_seconds": 7410,
        "rollout_request_timeout_seconds": 16200,
        "session_inactivity_timeout_seconds": 9000,
        "usable_context_window_tokens": 32768,
        "actor_backend": "megatron:d20p2t4c1",
        "rollout_backend": "sglang:d12p1t8",
        "r3_enabled": False,
        "dynamic_bs": True,
        "filter_zero_advantage_datums": True,
        "filter_zero_variance_groups": True,
        "dynamic_batch_recollection_max_rounds": 2,
        "checkpoint_artifact_policy": (
            "frozen_gs6_preserve_in_place_no_manual_rename_move_delete"
        ),
        "freeze_requirement": "satisfied_quiescent_census_host_pass_pinned_sif_runtime_pass",
    },
    "fresh_recursive_32node_qwen36_cp2_contract": {
        "fresh_start": True,
        "config_relative": (
            "plugins/openreward/platoon/openreward/configs/areal/"
            "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_"
            "qwen36_moe_cp2_c24_ta20_v19.yaml"
        ),
        "config_sha256": "b4f8594c5451bfed2d0539ddc266ca6546fd0eade2b9b5b7f5506f092853e201",
        "wrapper_relative": (
            "pbs-scripts/openreward-ta-swe-32node-recursive-qwen36-cp2-h200-v10-abci.pbs"
        ),
        "wrapper_sha256": "31fb65adfd8f20acc138418abfb47f421a6449564c877db56c84bdb55b3e02a3",
        "experiment_name": "abci-or-ta-swe32-q36-cp2-v19",
        "trial_name": "abci-ta-swe32-rec-q36m-cp2-c24-d1-v19-t0",
        "pbs_job_name": "rasp-h200-32q36-v19",
        "checkpoint_root": (
            "/groups/gai51740/aci18912it/rasp/experiments/areal/checkpoints/aci18912it/"
            "abci-or-ta-swe32-q36-cp2-v19/abci-ta-swe32-rec-q36m-cp2-c24-d1-v19-t0"
        ),
        "target_receipt": (
            "/groups/gai51740/aci18912it/rasp/job-state/gates/"
            "abci-ta-swe32-rec-q36m-cp2-c24-d1-v19-t0.success.json"
        ),
        "stale_checkpoint_and_receipt_policy": "reject_before_controller",
        "recover_mode": "auto_with_preexisting_root_forbidden",
        "actor_backend": "megatron:(attn:d10p2t4c2|ffn:d10p2t1e8)",
        "actor_topology": {
            "nodes": 20,
            "gpus": 160,
            "data_parallel_size": 10,
            "pipeline_parallel_size": 2,
            "attention_tensor_parallel_size": 4,
            "context_parallel_size": 2,
            "expert_tensor_parallel_size": 1,
            "expert_parallel_size": 8,
            "expert_data_parallel_size": 10,
        },
        "rollout_backend": "sglang:d12p1t8",
        "rollout_gpus": 96,
        "r3_enabled": True,
        "router_replay_num_layers": 40,
        "router_replay_topk": 8,
        "router_replay_num_experts": 256,
        "root_max_steps": 150,
        "subagent_max_steps": 100,
        "subagent_reward_judge_max_steps": 100,
        "subagent_max_depth": 1,
        "usable_context_window_tokens": 65536,
        "sglang_exclusive_context_length": 65537,
        "sglang_max_prefill_tokens": 65536,
        "condenser_trigger_ratio": "4/5",
        "condenser_max_tokens": 52428,
        "condenser_first_trigger_token": 52429,
        "dynamic_bs": False,
        "filter_zero_advantage_datums": False,
        "filter_zero_variance_groups": False,
        "dynamic_batch_recollection_max_rounds": 1,
        "actor_n_mbs": 1,
        "actor_microbatch_granularity": 1,
        "actor_n_mbs_divisor": 1,
        "actor_engine_microbatch_divisor": 2,
        "actor_engine_min_microbatches": 4,
        "actor_batch_divisor": 20,
        "actor_safe_minimum_datums": 40,
        "train_dataset_batch_size": 8,
        "minimum_successful_group_size": 5,
        "rollout_consumer_batch_size": 8,
        "expected_model_shards": 26,
    },
    "environment_health_monitor": {
        "toolathlon_endpoint": "/health",
        "swe_rebench_endpoint": "/health",
        "toolathlon_worker_count": 8,
        "toolathlon_worker_base_port": 28100,
        "toolathlon_direct_worker_endpoint": "/health",
        "toolathlon_direct_worker_quorum": 5,
        "toolathlon_affinity_header": "X-Session-ID",
        "toolathlon_affinity_policy": "nginx_consistent_hash",
        "toolathlon_public_probe_count": 4,
        "toolathlon_public_quorum": 1,
        "all_hosts_and_endpoints_concurrent": True,
        "per_probe_max_time_seconds": 15,
        "expected_sweep_bound_seconds": 20,
        "connect_timeout_seconds": 3,
        "request_timeout_seconds": 15,
        "failure_threshold": 10,
        "check_interval_seconds": 30,
        "minimum_consecutive_failure_window_seconds": 270,
        "failure_diagnostic_fields": ["host", "port", "url", "rc", "http", "elapsed"],
    },
    "nonrecursive_16node_ptc": False,
    "recursive_32node_ptc": True,
    "recursive_32node_ptc_mode": "orchestration_only",
    "recursive_32node_ptc_max_tool_calls_per_execution": 1024,
    "recursive_32node_subagent_datum_keep_probability": 0.25,
}
HOST_ACCEPTANCE_ASSETS = [
    "pbs-scripts/openreward-immutable-runtime-gate2-v9.pbs",
    "pbs-scripts/publish_openreward_v9_gate2_result.py",
    "pbs-scripts/verify_openreward_v9_gate2_result.py",
    "pbs-scripts/finalize_openreward_v9_acceptance.py",
    "pbs-scripts/openreward-toolathlon-all-mcp-health-v9.pbs",
    "pbs-scripts/toolathlon_all_mcp_health.py",
    "pbs-scripts/publish_toolathlon_all_mcp_health_result.py",
    "pbs-scripts/verify_toolathlon_all_mcp_health_result.py",
    "pbs-scripts/verify_openreward_resume_checkpoint.py",
]
PLAYWRIGHT_IDENTITY = {
    "browsers_path": "/opt/ms-playwright",
    "base_browser_tree_sha256": "84d979d1a301e24d7ee04009ff85b806cda195cce9d6bd1a8a686f06887805e5",
    "portable_browser_tree_sha256": "84d979d1a301e24d7ee04009ff85b806cda195cce9d6bd1a8a686f06887805e5",
    "browser_file_count": 156,
    "browser_total_bytes": 959511402,
    "browser_symlink_count": 0,
    "browser_directory_count": 13,
    "chromium_revision": "1182",
    "chromium_executable": "/opt/ms-playwright/chromium-1182/chrome-linux/chrome",
    "chromium_sha256": "b93030bc068a3850d4d668e69816fe4a5f04ec93db9b2f63f32f58a39f207db8",
    "headless_shell_executable": "/opt/ms-playwright/chromium_headless_shell-1182/chrome-linux/headless_shell",
    "headless_shell_sha256": "71ebe8760eb72a99b265470d4030fef296f119e25b9aab5d3efbb6ac821483a5",
    "headless_shell_bytes": 305228784,
    "base_config_sha256": "66f5bc92c182bf365e499d8237308452a016d8ca9c5e878d1628393647c82d1a",
    "patched_config_sha256": "47841318775a9afc122097e769a3fcc02d3af7089d372b60fc11da1ea44f0de9",
    "chromium_sandbox": "disabled_for_unprivileged_singularity",
    "rootless_launch": "passed",
}


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_owned_regular(path: Path, *, mode: int) -> os.stat_result:
    if not path.is_absolute() or path.resolve() != path:
        raise RuntimeError(f"release artifact is not direct and canonical: {path}")
    value = path.lstat()
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.getuid()
        or stat.S_IMODE(value.st_mode) != mode
        or value.st_nlink != 1
    ):
        raise RuntimeError(f"release artifact identity/mode mismatch: {path}")
    return value


def _require_model_artifacts() -> None:
    dense = MODEL_ARTIFACTS["recursive_32node"]
    if MODEL_ARTIFACTS["nonrecursive_16node"] != dense:
        raise RuntimeError("release dense model identities differ between node-count contracts")
    for contract, artifact in MODEL_ARTIFACTS.items():
        snapshot = Path(artifact["snapshot_path"])
        if (
            not snapshot.is_absolute()
            or snapshot.resolve() != snapshot
            or snapshot.is_symlink()
            or not snapshot.is_dir()
        ):
            raise RuntimeError(f"release model snapshot is unavailable or indirect: {contract}")
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
            if not (snapshot / name).is_file():
                raise RuntimeError(f"release model snapshot is incomplete ({contract}): {name}")
        try:
            index = json.loads((snapshot / "model.safetensors.index.json").read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"release model shard index is unreadable: {contract}") from exc
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"release model shard index has no weight map: {contract}")
        count = artifact["safetensors_shard_count"]
        expected = {f"model-{ordinal:05d}-of-{count:05d}.safetensors" for ordinal in range(1, count + 1)}
        declared = set(weight_map.values())
        on_disk = {path.name for path in snapshot.glob("model-*-of-*.safetensors") if path.is_file()}
        if declared != expected or on_disk != expected:
            raise RuntimeError(
                f"release model shard set does not match its pinned {count}-shard contract: {contract}"
            )


def _require_ancestor(root: Path, older: str, newer: str) -> None:
    completed = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", older, newer],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"release ancestry check failed: {older} is not an ancestor of {newer}")


def _require_python_mcp_identity(value: object, assets: dict[str, object]) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("v9 Toolathlon Python MCP provenance is missing")
    expected = {
        "audited_python_server_count": 13,
        "rewritten_uv_config_count": 12,
        "policy": "baked_venv_no_uv",
        "base_config_set_sha256": "ed8f303ce1bebf763f45bc838e58e68477bb4cded54bcb455471782705dc9511",
        "patched_config_set_sha256": "0be2dc5984be8280f5ba5e6d781dad56e765a62f72fb9ebd791f51dcc80b1d60",
        "launcher_mapping_sha256": "23497715f4e3388420e1842558e4017cad3aec62fdf8662e294d4480b19412cc",
        "wrapper_path": "/usr/local/libexec/openreward/launch_toolathlon_python_mcp.py",
        "wrapper_sha256": assets.get("launch_toolathlon_python_mcp.py"),
        "probe_path": "/usr/local/libexec/openreward/probe_toolathlon_python_mcp_launchers.py",
        "probe_sha256": assets.get("probe_toolathlon_python_mcp_launchers.py"),
        "rootless_probe_policy": (
            "cleanenv_no_home_exact_launch_initialize_live_name_set_persistence_clean_teardown_no_mutation"
        ),
        "unprivileged_exact_launch_initialize_live_name_set_persistence_clean_teardown_no_mutation": "passed",
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError("v9 Toolathlon Python MCP launch provenance mismatch")
    snowflake = value.get("snowflake_mcp_runtime")
    project = snowflake.get("project_metadata") if isinstance(snowflake, dict) else None
    if (
        not isinstance(snowflake, dict)
        or snowflake.get("policy") != "exact_in_place_record_repaired_ten_tree_replacement"
        or snowflake.get("installed_mcp_version") != "1.26.0"
        or snowflake.get("donor_closure_sha256") != "e7b1bf8895eec6e95d738b68a53cd4f0eaad9672284c9cdbe5ec3613399c2eec"
        or snowflake.get("installed_closure_sha256")
        != "b618e4359067acb69893820bae6a970c4d8a3714d77550c972346f6120497bc5"
        or not isinstance(project, dict)
        or project.get("patched_pyproject_sha256") != "8d8ed2e26ffe2621403a640fff0d598a35a78dbb06da2867e0bcf6ae80df2743"
        or project.get("patched_lock_sha256") != "2945a5f989fd610b8a31017cd29c41cd3a662e5267f1f17d4871afb7f1ace81c"
        or project.get("offline_uv_lock_check") != "passed"
    ):
        raise RuntimeError("v9 Toolathlon Snowflake MCP runtime provenance mismatch")


def _require_final_image_identity(root: Path, manifest: dict[str, object]) -> None:
    image = manifest.get("toolathlon_image")
    derivation = manifest.get("toolathlon_image_derivation")
    if not isinstance(image, dict) or not isinstance(derivation, dict):
        raise RuntimeError("release lacks the derived Toolathlon image contract")
    path = image.get("path")
    digest = image.get("sha256")
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or "PENDING" in path
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or derivation.get("status") != "ready"
        or any("PENDING" in str(value) for value in derivation.values())
    ):
        raise RuntimeError("v9 Toolathlon image identity is not final and cannot launch")
    if image != {
        "path": str(TOOLATHLON_IMAGE_PATH),
        "sha256": TOOLATHLON_IMAGE_SHA256,
        "oci_index_digest": "sha256:494af57e8a0f158d668982bf985b5937e73ba58eb468715f884de03550bd677c",
        "oci_amd64_digest": TOOLATHLON_BASE_OCI,
    }:
        raise RuntimeError("v9 Toolathlon image manifest identity mismatch")
    if derivation.get("playwright") != PLAYWRIGHT_IDENTITY:
        raise RuntimeError("v9 Toolathlon Playwright provenance mismatch")
    if derivation.get("build_commit") != TOOLATHLON_BUILD_COMMIT:
        raise RuntimeError("v9 Toolathlon build commit provenance mismatch")

    receipt_path = derivation.get("build_receipt_path")
    receipt_sha256 = derivation.get("build_receipt_sha256")
    provenance = derivation.get("build_receipt_provenance")
    if (
        receipt_path != str(TOOLATHLON_RECEIPT_PATH)
        or receipt_sha256 != TOOLATHLON_RECEIPT_SHA256
        or not isinstance(provenance, dict)
    ):
        raise RuntimeError("v9 Toolathlon build receipt reference mismatch")

    image_stat = _require_owned_regular(TOOLATHLON_IMAGE_PATH, mode=0o550)
    _require_owned_regular(TOOLATHLON_RECEIPT_PATH, mode=0o400)
    if _sha256(TOOLATHLON_IMAGE_PATH) != TOOLATHLON_IMAGE_SHA256:
        raise RuntimeError("v9 Toolathlon SIF stream digest mismatch")
    if _sha256(TOOLATHLON_RECEIPT_PATH) != TOOLATHLON_RECEIPT_SHA256:
        raise RuntimeError("v9 Toolathlon build receipt stream digest mismatch")
    try:
        receipt = json.loads(TOOLATHLON_RECEIPT_PATH.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("v9 Toolathlon build receipt is unreadable") from exc
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_KEYS:
        raise RuntimeError("v9 Toolathlon build receipt schema mismatch")
    embedded = receipt.get("embedded")
    if not isinstance(embedded, dict) or set(embedded) != EMBEDDED_RECEIPT_KEYS:
        raise RuntimeError("v9 Toolathlon embedded receipt schema mismatch")

    receipt_provenance = {
        "schema_version": receipt.get("schema_version"),
        "gate": receipt.get("gate"),
        "status": receipt.get("status"),
        "authorization_scope": receipt.get("authorization_scope"),
        "pbs_accounting_attested": receipt.get("pbs_accounting_attested"),
        "completed_at_epoch": receipt.get("completed_at_epoch"),
        "derived_image": receipt.get("derived_image"),
        "patch_assets": receipt.get("patch_assets"),
        "build_tooling": receipt.get("build_tooling"),
        "embedded": {
            "manifest_path": embedded.get("manifest_path"),
            "manifest_sha256": embedded.get("manifest_sha256"),
            "assets": embedded.get("assets"),
            "verify": embedded.get("verify"),
        },
        "labels": receipt.get("labels"),
    }
    if receipt_provenance != provenance:
        raise RuntimeError("v9 Toolathlon build receipt provenance mismatch")

    expected_derived = {
        "path": str(TOOLATHLON_IMAGE_PATH),
        "sha256": TOOLATHLON_IMAGE_SHA256,
        "size": image_stat.st_size,
        "device": image_stat.st_dev,
        "inode": image_stat.st_ino,
        "mode": 0o550,
        "uid": os.getuid(),
    }
    if receipt.get("derived_image") != expected_derived:
        raise RuntimeError("v9 Toolathlon build receipt SIF identity mismatch")
    if receipt.get("base_image") != {
        "path": str(TOOLATHLON_BASE_PATH),
        "sha256": TOOLATHLON_BASE_SHA256,
        "oci_digest": TOOLATHLON_BASE_OCI,
    }:
        raise RuntimeError("v9 Toolathlon build receipt base identity mismatch")
    if (
        receipt.get("toolathlon_source_revision") != COMPONENTS["toolathlon"]
        or receipt.get("portability_patch_sha256") != derivation.get("pbs_portability_patch_sha256")
        or embedded.get("base_server_sha256") != derivation.get("base_server_sha256")
        or embedded.get("patched_server_sha256") != TOOLATHLON_PATCHED_SERVER_SHA256
        or derivation.get("patched_server_sha256") != TOOLATHLON_PATCHED_SERVER_SHA256
        or embedded.get("base_tool_schemas_sha256") != derivation.get("base_tool_schemas_sha256")
        or embedded.get("repaired_tool_schemas_sha256") != derivation.get("repaired_tool_schemas_sha256")
        or embedded.get("playwright") != PLAYWRIGHT_IDENTITY
        or embedded.get("python_mcp") != derivation.get("python_mcp")
    ):
        raise RuntimeError("v9 Toolathlon build receipt derivation mismatch")

    assets = receipt.get("patch_assets")
    if not isinstance(assets, dict) or set(assets) != set(PATCH_ASSETS):
        raise RuntimeError("v9 Toolathlon patch asset set mismatch")
    patch_input = "".join(f"{assets[name]}  {name}\n" for name in PATCH_ASSETS).encode()
    if hashlib.sha256(patch_input).hexdigest() != derivation.get("pbs_portability_patch_sha256"):
        raise RuntimeError("v9 Toolathlon portability identity does not match its assets")
    embedded_assets = embedded.get("assets")
    if not isinstance(embedded_assets, dict) or embedded_assets != {
        name: assets[name] for name in PATCH_ASSETS if name != "toolathlon-pbs-portable.def"
    }:
        raise RuntimeError("v9 Toolathlon embedded asset identities mismatch")
    _require_python_mcp_identity(embedded.get("python_mcp"), assets)

    tooling = receipt.get("build_tooling")
    expected_tooling_paths = {
        "builder": root / "containers/build_toolathlon_pbs_portable_sif.sh",
        "publisher": root / "containers/publish_toolathlon_pbs_build_receipt.py",
        "image_publisher": root / "containers/publish_toolathlon_pbs_image.py",
    }
    if not isinstance(tooling, dict):
        raise RuntimeError("v9 Toolathlon build tooling contract is missing")
    for name, tooling_path in expected_tooling_paths.items():
        reference = tooling.get(name)
        if (
            not isinstance(reference, dict)
            or reference.get("name") != tooling_path.name
            or reference.get("sha256") != _sha256(tooling_path)
        ):
            raise RuntimeError(f"v9 Toolathlon build tooling identity mismatch: {name}")
    if tooling.get("singularity") != {
        "path": "/usr/bin/singularity",
        "sha256": "085a757f172dcabda66b44d086913314d6aa574055a8fea90523be82b770cec6",
        "version": "4.4.1-1.el9",
    }:
        raise RuntimeError("v9 Toolathlon build Singularity identity mismatch")

    labels = receipt.get("labels")
    required_labels = {
        "org.opencontainers.image.source": "https://github.com/ApGa/toolathlon-gym",
        "org.opencontainers.image.revision": COMPONENTS["toolathlon"],
        "org.openreward.toolathlon.base-sif-sha256": TOOLATHLON_BASE_SHA256,
        "org.openreward.toolathlon.base-oci-digest": TOOLATHLON_BASE_OCI,
        "org.openreward.toolathlon.pbs-portability": derivation.get("pbs_portability_patch_sha256"),
        "org.openreward.toolathlon.playwright-browsers-sha256": PLAYWRIGHT_IDENTITY["portable_browser_tree_sha256"],
    }
    if not isinstance(labels, dict) or any(labels.get(key) != value for key, value in required_labels.items()):
        raise RuntimeError("v9 Toolathlon derived SIF labels mismatch")


def verify(root: Path) -> None:
    if not root.is_absolute() or root.resolve() != root or root != RELEASE_ROOT:
        raise RuntimeError(f"release root is not the exact canonical path: {root}")
    manifest_path = root / MANIFEST_RELATIVE
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("v9 release manifest is missing or indirect")
    if _sha256(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("v9 release manifest content digest mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("release manifest schema is not 1")
    if manifest.get("release_root") != str(RELEASE_ROOT):
        raise RuntimeError("release manifest root mismatch")
    if manifest.get("release_ref") != RELEASE_REF:
        raise RuntimeError("release manifest ref mismatch")
    if manifest.get("components") != COMPONENTS:
        raise RuntimeError("release component revisions mismatch")
    if manifest.get("model_artifacts") != MODEL_ARTIFACTS:
        raise RuntimeError("release model artifact identity mismatch")
    if manifest.get("training_semantics") != TRAINING_SEMANTICS:
        raise RuntimeError("release training semantics mismatch")
    for contract_name in (
        "fresh_recursive_32node_qwen36_cp2_contract",
        "recursive_32node_qwen38_v20_successor_contract",
    ):
        contract = TRAINING_SEMANTICS[contract_name]
        for kind in ("config", "wrapper"):
            relative = contract[f"{kind}_relative"]
            asset = root / relative
            if (
                not asset.is_file()
                or asset.is_symlink()
                or _sha256(asset) != contract[f"{kind}_sha256"]
            ):
                raise RuntimeError(
                    f"release {contract_name} {kind} content mismatch: {relative}"
                )
    if manifest.get("training_runtime_assets") != TRAINING_RUNTIME_ASSETS:
        raise RuntimeError("release training runtime asset identity mismatch")
    for relative, expected_sha256 in TRAINING_RUNTIME_ASSETS.items():
        asset = root / relative
        if not asset.is_file() or asset.is_symlink() or _sha256(asset) != expected_sha256:
            raise RuntimeError(f"release training runtime asset content mismatch: {relative}")
    if manifest.get("host_acceptance_assets") != HOST_ACCEPTANCE_ASSETS:
        raise RuntimeError("release host acceptance asset set mismatch")
    for relative in HOST_ACCEPTANCE_ASSETS:
        asset = root / relative
        if not asset.is_file() or asset.is_symlink() or not asset.stat().st_mode & 0o111:
            raise RuntimeError(f"release host acceptance asset is unavailable: {relative}")
    _require_final_image_identity(root, manifest)
    _require_model_artifacts()

    head = _git(root, "rev-parse", "--verify", "HEAD")
    release_commit = _git(root, "rev-parse", "--verify", f"{RELEASE_REF}^{{commit}}")
    if head != release_commit:
        raise RuntimeError("release HEAD does not match its pinned tag")
    _require_ancestor(root, UPSTREAM_COMMIT, head)
    _require_ancestor(root, TOOLATHLON_BUILD_COMMIT, head)
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("release checkout has tracked or ordinary untracked changes")
    tracked = _git(root, "ls-files").splitlines()
    if any(
        path.endswith(".orig")
        or path.endswith("cleanup_openreward_2132698_generation.py")
        or path.endswith("test_offline_cleanup_2132698.py")
        for path in tracked
    ):
        raise RuntimeError("release tracks retired migration debris")

    for name, relative in EXTERNAL_PATHS.items():
        external = root / relative
        if external.resolve() != external or not (external / ".git").exists():
            raise RuntimeError(f"materialized external checkout is invalid: {name}")
        if _git(external, "rev-parse", "--verify", "HEAD") != COMPONENTS[name]:
            raise RuntimeError(f"materialized external revision mismatch: {name}")
        if _git(external, "status", "--porcelain=v1", "--untracked-files=all"):
            raise RuntimeError(f"materialized external checkout is dirty: {name}")

    # The SWE dependency closure did not change between v8r2 and v8r18.  The
    # existing runtime verifier reads the retained v8 manifest, so require both
    # manifests to carry byte-for-byte equal SWE contracts before delegating.
    v8_manifest = json.loads((root / "pbs-scripts/openreward-v8-release.json").read_text())
    if v8_manifest.get("swe_runtime") != manifest.get("swe_runtime"):
        raise RuntimeError("v9 and retained v8 SWE runtime contracts differ")
    verify_swe_runtime(root)

    print(f"openreward_v9_release=PASS release_commit={head}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    args = parser.parse_args()
    verify(args.release_root)


if __name__ == "__main__":
    main()
