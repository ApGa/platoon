from __future__ import annotations

import getpass
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PBS_SCRIPTS = REPO_ROOT / "pbs-scripts"
CONFIG_ROOT = REPO_ROOT / "plugins" / "openreward" / "platoon" / "openreward" / "configs" / "areal"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _merge(base, overlay):
    if isinstance(base, dict) and isinstance(overlay, dict):
        result = dict(base)
        for key, value in overlay.items():
            result[key] = _merge(result[key], value) if key in result else value
        return result
    return overlay


def _compose_config(name: str, ancestry: tuple[str, ...] = ()) -> dict:
    if name in ancestry:
        raise AssertionError(f"config default cycle: {ancestry + (name,)}")
    value = yaml.safe_load((CONFIG_ROOT / f"{name}.yaml").read_text()) or {}
    defaults = value.pop("defaults", [])
    result: dict = {}
    for item in defaults:
        if item == "_self_":
            result = _merge(result, value)
        elif isinstance(item, str):
            result = _merge(result, _compose_config(item, ancestry + (name,)))
        else:
            raise AssertionError(f"unsupported config default in {name}: {item!r}")
    if "_self_" not in defaults:
        result = _merge(result, value)
    return result


class ABCILaunchAssetTests(unittest.TestCase):
    def test_leaf_overlays_are_cluster_local(self) -> None:
        cases = (
            ("abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_ta20.yaml", 16, False),
            ("abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_ta20.yaml", 32, True),
        )
        for filename, nodes, recursive in cases:
            text = (CONFIG_ROOT / filename).read_text()
            self.assertIn("type: pbs_prealloc", text)
            self.assertIn(f"n_nodes: {nodes}", text)
            self.assertIn("fileroot: ${oc.env:PLATOON_EXPERIMENT_ROOT}", text)
            self.assertIn("nfs_record_root: ${oc.env:PLATOON_NAME_RESOLVE_ROOT}", text)
            self.assertIn("path: ${oc.env:PLATOON_MODEL_SNAPSHOT}", text)
            self.assertIn(f"enable_recursive_subagents: {str(recursive).lower()}", text)
            self.assertNotIn("/lustre/", text)
            self.assertNotIn("slurm_prealloc", text)

    def test_composed_configs_preserve_parity_topologies_and_curriculum(self) -> None:
        cases = (
            (
                "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_ta20",
                16,
                "sglang:d6p1t8",
                "megatron:(attn:d5p2t4c2|ffn:d5p2t1e8)",
                False,
            ),
            (
                "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_ta20",
                32,
                "sglang:d12p1t8",
                "megatron:(attn:d10p2t4c2|ffn:d10p2t1e8)",
                True,
            ),
        )
        for name, nodes, rollout_backend, actor_backend, recursive in cases:
            config = _compose_config(name)
            self.assertEqual(config["scheduler"]["type"], "pbs_prealloc")
            self.assertEqual(config["cluster"]["n_nodes"], nodes)
            self.assertEqual(config["cluster"]["n_gpus_per_node"], 8)
            self.assertEqual(config["rollout"]["backend"], rollout_backend)
            self.assertEqual(config["actor"]["backend"], actor_backend)
            self.assertEqual(config["train_dataset"]["batch_size"], 8)
            self.assertEqual(config["workflow_config"]["group_size"], 8)
            self.assertIs(config["openreward"]["enable_recursive_subagents"], recursive)
            self.assertFalse(config["openreward"]["balance_accepted_batches"])
            self.assertEqual(
                [
                    (environment["label"], environment["sampling_start_step"])
                    for environment in config["openreward"]["environments"]
                ],
                [("toolathlon", 0), ("swe_rebench", 20)],
            )
            self.assertIsNone(config["valid_dataset"])

    def test_production_wrappers_have_fixed_topology_and_no_continuation(self) -> None:
        sixteen = (PBS_SCRIPTS / "openreward-ta-swe-16node-nonrecursive-abci.pbs").read_text()
        recursive = (PBS_SCRIPTS / "openreward-ta-swe-32node-recursive-abci.pbs").read_text()
        common = (PBS_SCRIPTS / "openreward-ta-swe-prealloc-common.sh").read_text()
        self.assertIn("#PBS -l select=16:mpiprocs=8", sixteen)
        self.assertIn("#PBS -l select=32:mpiprocs=8", recursive)
        self.assertIn("openreward_update_gate.py", recursive)
        self.assertIn("verify", recursive)
        self.assertIn("pbs_prealloc", common)
        self.assertIn("awk '!seen[$1]++", common)
        self.assertIn('source "${ABCI_ENV}"', common)
        self.assertIn('source "${SECRETS_ENV}"', common)
        self.assertIn("prepare_openreward_controller_env.sh", common)
        self.assertIn(
            "PLATOON_AREAL_PBS_HOST_PYTHON=${OPENREWARD_CONTROLLER_JOB_PYTHON}",
            common,
        )
        self.assertIn(
            "PLATOON_AREAL_PBS_WORKER_PYTHON=${OPENREWARD_WORKER_JOB_PYTHON}",
            common,
        )
        self.assertIn(
            '"${OPENREWARD_CONTROLLER_JOB_PYTHON}" -m "${TRAIN_MODULE}"',
            common,
        )
        self.assertIn(
            "export PLATOON_AREAL_PBS_GPU_VISIBILITY_MODE=nvidia-container-cli",
            common,
        )
        self.assertIn(
            "export PLATOON_AREAL_PBS_NVIDIA_SMI_BIN=/usr/bin/nvidia-smi",
            common,
        )
        self.assertNotIn(
            "PLATOON_AREAL_PBS_HOST_PYTHON=${OPENREWARD_JOB_PYTHON}",
            common,
        )
        self.assertIn("/etc/resolv.conf:/etc/resolv.conf:ro", common)
        self.assertIn("/etc/hosts:/etc/hosts:ro", common)
        forward_line = next(
            line for line in common.splitlines() if line.startswith("export PLATOON_AREAL_PBS_FORWARD_ENV=")
        )
        controller_only = (
            "WANDB_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "LITELLM_API_KEY",
            "LITELLM_BASE_URL",
            "RUBRIC_DEFAULT_LLM",
            "HF_TOKEN",
        )
        for secret in controller_only:
            self.assertNotIn(secret, forward_line)
            self.assertIn(f"-u {secret}", common)
        for helper_name in (
            "openreward-node-services.sh",
            "openreward-swe-singularity-server.sh",
        ):
            helper = (PBS_SCRIPTS / helper_name).read_text()
            for secret in controller_only:
                self.assertIn(secret, helper)
        for forbidden in ("sbatch", "srun", "pyxis", "enroot", "gpu_keepalive.py", "qsub"):
            self.assertNotIn(forbidden, common.lower())

    def test_node_services_use_one_absolute_mkfs_for_both_overlay_consumers(self) -> None:
        helper = (PBS_SCRIPTS / "openreward-node-services.sh").read_text()
        validation = '[[ -x "${MKFS_EXT3}" ]]'
        staging = 'actual_toolathlon_sha=$(sha256sum "${OPENREWARD_TOOLATHLON_SIF}"'

        self.assertIn(
            "MKFS_EXT3=${PLATOON_MKFS_EXT3_BIN:-/usr/sbin/mkfs.ext3}",
            helper,
        )
        self.assertIn('[[ "${MKFS_EXT3}" == /* ]]', helper)
        self.assertIn(validation, helper)
        self.assertLess(helper.index(validation), helper.index(staging))
        self.assertIn(
            '"${MKFS_EXT3}" -q -F -d "${overlay_layout}" "${TOOLATHLON_OVERLAY}"',
            helper,
        )
        self.assertIn("export SWE_APPTAINER_MKFS_BIN=${MKFS_EXT3}", helper)
        self.assertNotIn("\nmkfs.ext3 ", helper)

    def test_swe_server_requires_the_committed_clean_backend_revision(self) -> None:
        helper = (PBS_SCRIPTS / "openreward-swe-singularity-server.sh").read_text()
        revision = "538b5c5c9c529d3c78d5ef7fd3df7e54e7902ad7"

        self.assertIn(
            f"EXPECTED_REVISION=${{SWE_REBENCH_SOURCE_REVISION:-{revision}}}",
            helper,
        )
        self.assertIn('git -C "${ENV_ROOT}" rev-parse HEAD', helper)
        self.assertIn(
            'git -C "${ENV_ROOT}" status --porcelain --untracked-files=all',
            helper,
        )
        self.assertIn('die "SWE source worktree is dirty at ${EXPECTED_REVISION}"', helper)
        self.assertIn(
            "VERIFY_PINNED_RUNTIME=${OPENREWARD_VERIFY_PINNED_SWE_RUNTIME:-0}",
            helper,
        )
        self.assertIn('[[ "${VERIFY_PINNED_RUNTIME}" == 0 || "${VERIFY_PINNED_RUNTIME}" == 1 ]]', helper)

    def test_node_service_pbsdsh_smoke_uses_exact_production_helper(self) -> None:
        smoke = (PBS_SCRIPTS / "openreward-node-services-pbsdsh-smoke.pbs").read_text()

        self.assertIn("#PBS -l select=1:mpiprocs=8", smoke)
        self.assertIn("#PBS -l walltime=00:15:00", smoke)
        self.assertIn('${PBSDSH}" -n 0 -- /usr/bin/env', smoke)
        self.assertIn('"PLATOON_ABCI_ENV_FILE=${ABCI_ENV}"', smoke)
        self.assertIn('"PLATOON_NODE_SERVICES_CLEANUP_RECEIPT=${SERVICE_RECEIPT}"', smoke)
        self.assertIn('"PLATOON_NODE_SERVICES_EXPECTED_HOST=${HOST}"', smoke)
        self.assertIn('"PLATOON_NODE_SERVICES_EXPECTED_JOB_ID=${PBS_JOBID}"', smoke)
        self.assertIn('"${NODE_SERVICES}"', smoke)
        self.assertIn("OPENREWARD_NODE_SERVICES_GATE2_RESULT", smoke)
        self.assertIn('[[ "$(result_value "${GATE2_RESULT}" gate2)" == passed ]]', smoke)
        self.assertIn("prepare_openreward_env.sh", smoke)
        self.assertIn('"${current_source_sha}" == "${source_sha}"', smoke)
        self.assertIn("_v8.yaml", smoke)
        self.assertIn("SWE_APPTAINER_SESSION_ROOT=${SMOKE_ROOT}/swe-sessions", smoke)
        self.assertIn("SWE_SINGULARITY_SESSION_ROOT=${SWE_APPTAINER_SESSION_ROOT}", smoke)
        self.assertIn("SINGULARITY_CACHEDIR=${SMOKE_ROOT}/singularity-cache", smoke)
        self.assertNotIn("2132783", smoke)
        self.assertNotIn("_v4.yaml", smoke)
        self.assertIn("http://${HOST}:${port}/", smoke)
        self.assertIn("both production endpoints are healthy", smoke)
        self.assertIn("controller_pool_discovery=PASS", smoke)
        self.assertIn("OPENREWARD_SESSION_URLS_TOOLATHLON=http://${HOST}:${OPENREWARD_PORT}", smoke)
        self.assertIn(
            "OPENREWARD_SESSION_URLS_SWE_REBENCH=http://${HOST}:${OPENREWARD_SWE_REBENCH_PORT}",
            smoke,
        )
        self.assertIn("get_task_records(config.openreward)", smoke)
        self.assertIn('"${CONTROLLER_PYTHON}" -P -', smoke)
        self.assertIn("task discovery did not import the verified controller venv", smoke)
        self.assertIn("allow_legacy_global_pool=False", smoke)
        self.assertIn("swe_singularity_sessions_cleaned=0", smoke)
        self.assertIn('"${SERVICE_RECEIPT_HELPER}" assert-empty-swe', smoke)
        self.assertIn("SWE session entries remain after helper teardown", smoke)
        self.assertIn("an environment endpoint remained reachable after helper teardown", smoke)
        self.assertIn("endpoints_became_healthy=true", smoke)
        self.assertIn("endpoints_stopped=true", smoke)
        self.assertNotIn("qsub", smoke)
        for secret in (
            "WANDB_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "LITELLM_API_KEY",
            "LITELLM_BASE_URL",
            "RUBRIC_DEFAULT_LLM",
            "HF_TOKEN",
        ):
            self.assertIn(f"-u {secret}", smoke)

    def test_update_gate_requires_current_recovery_evidence(self) -> None:
        gate = _load_module("openreward_update_gate_test", PBS_SCRIPTS / "openreward_update_gate.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = "abci-test"
            trial = "abci-test-trial0"
            checkpoint_root = root / "checkpoints" / getpass.getuser() / experiment / trial
            recover_info = checkpoint_root / "recover_info"
            checkpoint = checkpoint_root / "default" / "recover_checkpoint"
            recover_info.mkdir(parents=True)
            checkpoint.mkdir(parents=True)
            step_path = recover_info / "step_info.json"
            step_path.write_text(
                json.dumps(
                    {
                        "epoch": 0,
                        "epoch_step": 0,
                        "global_step": 0,
                        "steps_per_epoch": 1,
                    }
                )
            )
            (checkpoint / "payload").write_bytes(b"checkpoint")
            training_sif = root / "training.sif"
            training_sif.write_bytes(b"sif")
            config = root / "config.yaml"
            config.write_text("trial_name: abci-test-trial0\n")
            marker = root / "gate" / "update.json"
            digest = "a" * 64

            monitor_status = gate.main(
                [
                    "monitor",
                    "--marker",
                    str(marker),
                    "--fileroot",
                    str(root),
                    "--experiment",
                    experiment,
                    "--trial",
                    trial,
                    "--training-sif",
                    str(training_sif),
                    "--training-sif-sha256",
                    digest,
                    "--config",
                    str(config),
                    "--pbs-job-id",
                    "123.pbs1",
                    "--not-before-epoch",
                    "0",
                    "--timeout-seconds",
                    "1",
                    "--poll-seconds",
                    "0.01",
                ]
            )
            self.assertEqual(monitor_status, 0)
            self.assertEqual(os.stat(marker).st_mode & 0o777, 0o600)
            self.assertEqual(
                gate.main(
                    [
                        "verify",
                        "--marker",
                        str(marker),
                        "--fileroot",
                        str(root),
                        "--experiment",
                        experiment,
                        "--trial",
                        trial,
                        "--training-sif",
                        str(training_sif),
                        "--training-sif-sha256",
                        digest,
                    ]
                ),
                0,
            )

            step_path.write_text(
                json.dumps(
                    {
                        "epoch": -1,
                        "epoch_step": -1,
                        "global_step": -1,
                        "steps_per_epoch": 1,
                    }
                )
            )
            self.assertEqual(
                gate.main(
                    [
                        "verify",
                        "--marker",
                        str(marker),
                        "--fileroot",
                        str(root),
                        "--experiment",
                        experiment,
                        "--trial",
                        trial,
                        "--training-sif",
                        str(training_sif),
                        "--training-sif-sha256",
                        digest,
                    ]
                ),
                1,
            )

    def test_swe_cleanup_scrubs_legacy_runtime_config_and_removes_stale_session(self) -> None:
        cleanup = _load_module(
            "cleanup_swe_singularity_sessions_test",
            PBS_SCRIPTS / "cleanup_swe_singularity_sessions.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.chmod(0o700)
            name = "swe-rebench-0123456789abcdef"
            session = root / f"{name}-state"
            config_dir = session / "runtime-config"
            config_dir.mkdir(parents=True)
            session.chmod(0o700)
            config_dir.chmod(0o700)
            pid_file = session / "instance.pid"
            owner = session / ".openreward-session-owner.json"
            owner.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "instance_name": name,
                        "owner_pid": 2**30,
                        "owner_start_time": "dead-owner",
                        "session_directory": str(session),
                        "hostname": __import__("socket").gethostname(),
                        "instance_pid_file": str(pid_file),
                    }
                )
            )
            owner.chmod(0o600)
            # Current backends create the private lease before publishing the
            # owner record.  The janitor may take this unlocked lease in the
            # crash-recovery fixture; freshly created legacy directories are
            # intentionally protected by a separate minimum-age policy.
            lease = session / ".openreward-session.lock"
            lease.touch(mode=0o600)
            lease.chmod(0o600)

            runtime = root / "fake-singularity"
            runtime.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys

if "SINGULARITY_CONFIGDIR" in os.environ or "APPTAINER_CONFIGDIR" in os.environ:
    raise SystemExit(3)
arguments = sys.argv[1:]
if arguments[:3] == ["instance", "list", "--json"]:
    print(json.dumps({"instances": []}))
    raise SystemExit(0)
raise SystemExit(2)
"""
            )
            runtime.chmod(0o755)

            self.assertEqual(cleanup.cleanup_sessions(root, runtime, 1), 1)
            self.assertFalse(session.exists())


if __name__ == "__main__":
    unittest.main()
