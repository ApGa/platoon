from __future__ import annotations

import ast
import getpass
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from platoon import training_success_receipt as receipt

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "plugins" / "openreward" / "platoon" / "openreward" / "configs" / "areal"
GATE_CONFIG = "abci_toolathlon_openhands_areal_gate5_one_update"
PRODUCTION_CONFIG = "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_ta20"
GATE_WRAPPER = REPO_ROOT / "pbs-scripts/openreward-gate5-one-update-16node-abci.pbs"
COMMON_WRAPPER = REPO_ROOT / "pbs-scripts/openreward-ta-swe-prealloc-common.sh"
TASK_SMOKE = REPO_ROOT / "pbs-scripts/toolathlon_session_smoke.py"
TRAINER_SOURCE = REPO_ROOT / "platoon/train/areal/rl.py"
TRAIN_ENTRYPOINT = REPO_ROOT / "plugins/openreward/platoon/openreward/train_scripts/areal/train_areal.py"
UPSTREAM_TRAINER = REPO_ROOT / "external/AReaL/areal/trainer/rl_trainer.py"
UPSTREAM_STATS_LOGGER = REPO_ROOT / "external/AReaL/areal/utils/stats_logger.py"
EXPECTED_RESUME_STEP_ENV = "OPENREWARD_EXPECTED_RESUME_GLOBAL_STEP"
PREUPDATE_NO_LOAD_OPTIM_ENV = "OPENREWARD_PREUPDATE_NO_LOAD_OPTIM"
PREUPDATE_NO_LOAD_OPTIM_BINDING = {
    "OPENREWARD_ALLOW_PREUPDATE_RESUME": "1",
    "OPENREWARD_PREUPDATE_RESUME_BINDING": "qwen38-16-v18-job2155389",
    "OPENREWARD_PREUPDATE_LOAD_POLICY": "qwen38-16-v20-gs0-no-load-optim",
    "OPENREWARD_EXPECTED_RESUME_GLOBAL_STEP": "0",
    "OPENREWARD_EXPECTED_START_GLOBAL_STEP": "1",
    "OPENREWARD_EXPECTED_RESUME_OPTIMIZER_MOMENT_KEYS": "0",
    "OPENREWARD_EXPECTED_RESUME_LR_NUM_STEPS": "0",
    "OPENREWARD_EXPECTED_EXPERIMENT_NAME": "abci-or-ta-swe16-q38-v18",
    "OPENREWARD_EXPECTED_TRIAL_NAME": "abci-ta-swe16-nr-q38d-c12-v18-t0",
}


def _load_isolated_trainer_function(name: str, **injected_globals):
    """Execute one pure trainer helper without importing CUDA/AReaL."""

    tree = ast.parse(TRAINER_SOURCE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"Any": object, "os": os, **injected_globals}
    isolated = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(isolated), str(TRAINER_SOURCE), "exec"), namespace)
    return namespace[name]


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
        raise AssertionError(f"config default cycle: {ancestry + (name,)}")
    value = yaml.safe_load((CONFIG_ROOT / f"{name}.yaml").read_text()) or {}
    defaults = value.pop("defaults", [])
    result: dict = {}
    for item in defaults:
        if item == "_self_":
            result = _merge(result, value)
        elif isinstance(item, str):
            result = _merge(result, _compose(item, ancestry + (name,)))
        else:
            raise AssertionError(f"unsupported config default in {name}: {item!r}")
    if "_self_" not in defaults:
        result = _merge(result, value)
    return result


def _attention_allocation_size(backend: str) -> int:
    match = re.search(r"attn:d(\d+)p(\d+)t(\d+)(?:c(\d+))?", backend)
    assert match is not None, backend
    data, pipeline, tensor, context = (int(value or 1) for value in match.groups())
    return data * pipeline * tensor * context


def _sglang_allocation_size(backend: str) -> int:
    match = re.fullmatch(r"sglang:d(\d+)p(\d+)t(\d+)", backend)
    assert match is not None, backend
    return int(match[1]) * int(match[2]) * int(match[3])


def test_gate5_config_is_one_real_group_on_the_full_production_topology() -> None:
    config = _compose(GATE_CONFIG)
    production = _compose(PRODUCTION_CONFIG)

    assert config["experiment_name"] == "abci-openreward-gate5"
    assert config["trial_name"] == "abci-gate5-ta-one-update-v1-trial0"
    assert config["total_train_steps"] == 1
    assert config["cluster"]["n_nodes"] == 16
    assert config["cluster"]["n_gpus_per_node"] == 8
    assert config["scheduler"]["type"] == "pbs_prealloc"

    # The smoke follows production topology changes (including planned H200
    # CP/concurrency tuning) rather than silently exercising a smaller model.
    assert config["actor"]["backend"] == production["actor"]["backend"]
    assert config["rollout"]["backend"] == production["rollout"]["backend"]
    assert _attention_allocation_size(config["actor"]["backend"]) == 80
    assert _sglang_allocation_size(config["rollout"]["backend"]) == 48

    openreward = config["openreward"]
    assert openreward["environments"] is None
    assert openreward["env_name"] == "toolathlongym"
    assert openreward["task_names"] == ["canvas-announcement-summary"]
    assert openreward["task_indices"] is None
    assert openreward["enable_recursive_subagents"] is False
    assert config["train_dataset"]["batch_size"] == 1
    assert config["workflow_config"]["group_size"] == 8
    assert config["workflow_config"]["min_successful_group_size"] == 5
    assert config["workflow_config"]["filter_zero_variance_groups"] is False
    assert config["workflow_config"]["filter_zero_advantage_datums"] is False
    assert config["workflow_config"]["rollout_config"]["max_steps"] == 20
    assert config["rollout"]["consumer_batch_size"] == "${train_dataset.batch_size}"
    assert config["rollout"]["max_concurrent_rollouts"] == 1
    assert config["recover"]["freq_steps"] == 1
    assert config["actor"]["megatron"]["async_save"] is False
    assert config["valid_dataset"] is None
    assert config["evaluator"]["eval_before_train"] is False
    assert config["stats_logger"]["wandb"]["mode"] == "online"


def test_gate5_wrapper_waits_for_common_cleanup_before_verifying_receipt() -> None:
    subprocess.run(["bash", "-n", str(GATE_WRAPPER)], check=True)
    wrapper = GATE_WRAPPER.read_text()
    common = COMMON_WRAPPER.read_text()
    assert "#PBS -l select=16:mpiprocs=8" in wrapper
    assert "#PBS -l walltime=08:00:00" in wrapper
    assert '"${COMMON}" "${CONFIG}" || status=$?' in wrapper
    assert 'exec "${COMMON}"' not in wrapper
    assert wrapper.index('"${COMMON}" "${CONFIG}"') < wrapper.index("platoon.training_success_receipt verify")
    assert '--pbs-job-id "${PBS_JOBID}"' in wrapper
    assert "cpython-3.12.13-linux-x86_64-gnu/bin/python3.12" in wrapper
    assert "PLATOON_TRAINING_CONFIG_PATH=${CONFIG}" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_NOT_BEFORE_EPOCH=$(date +%s)" in wrapper
    assert "refusing stale Gate 5 success artifact" in wrapper
    assert "refusing stale Gate 5 checkpoint" in wrapper
    assert '[[ ! -e "${prior}" && ! -L "${prior}" ]]' in wrapper
    assert '[[ ! -e "${CHECKPOINT_ROOT}" && ! -L "${CHECKPOINT_ROOT}" ]]' in wrapper
    assert "unset OPENREWARD_GATE_MARKER_TO_PUBLISH" in wrapper
    assert "OPENREWARD_GATE5_ONE_UPDATE_OK" in wrapper

    # The common lifecycle's EXIT trap stops controller/monitor, publishes
    # exact requests on natural success, and requires the original pbsdsh tasks
    # plus exact remote cleanup receipts before it can return to Gate 5.
    assert "trap cleanup EXIT" in common
    assert 'terminate_pid "${controller_pid}" 75' in common
    assert 'terminate_pid "${health_pid}" 5' in common
    assert "request_all_service_shutdowns" in common
    assert "publish_all_service_shutdown_requests" in common
    assert 'for index in "${!service_pids[@]}"' in common
    assert "service_shutdown_deadline=$((SECONDS + 180))" in common
    assert "expected_pbsdsh_status=0" in common
    assert "expected_primary_status=0" in common
    assert "expected_primary_status=143" in common
    assert "service_cleanup_failed" in common


def test_fixed_task_matches_the_task_proven_by_the_pinned_toolathlon_smoke() -> None:
    config = _compose(GATE_CONFIG)
    smoke = TASK_SMOKE.read_text()
    task_match = re.search(r'^TASK_NAME = "([^"]+)"$', smoke, flags=re.MULTILINE)
    assert task_match is not None
    assert config["openreward"]["task_names"] == [task_match.group(1)]
    assert 'environment.list_tasks(split="train")' in smoke
    assert "Toolathlon smoke task not found" in smoke

    migration_evidence = (REPO_ROOT / "docs/abci-migration.md").read_text()
    assert "2130903.pbs1" in migration_evidence
    assert "Exit_status=0" in migration_evidence


def test_trainer_hook_is_after_update_sync_recovery_and_stats_commit() -> None:
    source = TRAINER_SOURCE.read_text()
    train_start = source.index("    def train(")
    train_end = source.index("\n    def _evaluate_fn", train_start)
    train = source[train_start:train_end]

    reset = train.index("actor_update_successful = False")
    collect = train.index("self._collect_rollout_batch(")
    update = train.index("self.actor.ppo_update(optimizer_batch)")
    sync = train.index("self.actor.update_weights(versioned_meta)")
    advance = train.index("self._advance_logical_versions(new_version)", sync)
    synced = train.index("weight_sync_succeeded = True", advance)
    promotion_guard = train.index("_require_preupdate_checkpoint_promotion(", synced)
    recover = train.index("self._save_recover_checkpoint(")
    commit = train.index("self._export_and_commit_stats(")
    candidate_guard = train.index("_training_success_candidate_ready(")
    candidate = train.index("maybe_publish_training_success_candidate(", candidate_guard)
    resume = train.index("self.rollout.resume()", candidate)

    assert (
        reset
        < collect
        < update
        < sync
        < advance
        < synced
        < promotion_guard
        < recover
        < commit
        < candidate_guard
        < candidate
        < resume
    )
    assert "optimizer_batch_datums = int(" in train[collect:update]
    assert "optimizer_action_tokens = int(" in train[collect:update]
    assert "PLATOON_TRAINING_SUCCESS_NOT_BEFORE_EPOCH" in train[candidate:resume]

    tree = ast.parse(source)
    guarded_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if ast.unparse(node.test) != (
            "_training_success_candidate_ready(global_step, "
            "actor_update_successful=actor_update_successful, "
            "weight_sync_succeeded=weight_sync_succeeded)"
        ):
            continue
        guarded_calls.extend(
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and ast.unparse(child.func) == "maybe_publish_training_success_candidate"
        )
    assert len(guarded_calls) == 1


def test_no_update_defers_candidate_but_complete_update_reaches_publisher() -> None:
    enabled_steps: list[int] = []
    ready = _load_isolated_trainer_function(
        "_training_success_candidate_ready",
        candidate_publication_enabled=lambda step: enabled_steps.append(step) or True,
    )
    published: list[int] = []

    def run_hook(*, actor_update_successful: bool, weight_sync_succeeded: bool) -> None:
        if ready(
            6,
            actor_update_successful=actor_update_successful,
            weight_sync_succeeded=weight_sync_succeeded,
        ):
            published.append(6)

    run_hook(actor_update_successful=False, weight_sync_succeeded=False)
    run_hook(actor_update_successful=True, weight_sync_succeeded=False)
    run_hook(actor_update_successful=False, weight_sync_succeeded=True)
    assert published == []
    # The receipt predicate is not evaluated until direct update/sync evidence
    # exists, so an empty or non-finite step cannot publish or raise there.
    assert enabled_steps == []

    run_hook(actor_update_successful=True, weight_sync_succeeded=True)
    assert enabled_steps == [6]
    assert published == [6]


def _preupdate_runtime_helpers():
    active = _load_isolated_trainer_function(
        "_preupdate_no_load_optim_resume_active",
        _PREUPDATE_NO_LOAD_OPTIM_ENV=PREUPDATE_NO_LOAD_OPTIM_ENV,
        _PREUPDATE_NO_LOAD_OPTIM_BINDING=PREUPDATE_NO_LOAD_OPTIM_BINDING,
    )
    guard = _load_isolated_trainer_function("_require_preupdate_checkpoint_promotion")
    ready = _load_isolated_trainer_function(
        "_training_success_candidate_ready",
        candidate_publication_enabled=lambda _step: True,
    )
    return active, guard, ready


def _enable_exact_preupdate_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PREUPDATE_NO_LOAD_OPTIM_ENV, "1")
    for name, value in PREUPDATE_NO_LOAD_OPTIM_BINDING.items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    ("optimizer_batch_datums", "actor_update_successful", "weight_sync_succeeded"),
    (
        (0, False, False),
        (8, False, False),
        (8, True, False),
    ),
    ids=("empty-batch", "nonfinite-update", "missing-weight-sync"),
)
def test_preupdate_nooptim_first_step_failure_preserves_gs0_without_candidate(
    monkeypatch: pytest.MonkeyPatch,
    optimizer_batch_datums: int,
    actor_update_successful: bool,
    weight_sync_succeeded: bool,
) -> None:
    active_fn, guard, ready = _preupdate_runtime_helpers()
    _enable_exact_preupdate_binding(monkeypatch)
    active = active_fn(
        1,
        recover_no_load_optim=True,
        recover_no_save_optim=False,
    )
    saves: list[int] = []
    candidates: list[int] = []

    with pytest.raises(RuntimeError, match="refusing to promote"):
        guard(
            active=active,
            global_step=1,
            optimizer_batch_datums=optimizer_batch_datums,
            actor_update_successful=actor_update_successful,
            weight_sync_succeeded=weight_sync_succeeded,
        )
        saves.append(1)
    if ready(
        1,
        actor_update_successful=actor_update_successful,
        weight_sync_succeeded=weight_sync_succeeded,
    ):
        candidates.append(1)
    assert saves == []
    assert candidates == []


def test_preupdate_nooptim_success_aligns_lr2_then_saves_and_publishes_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_fn, guard, ready = _preupdate_runtime_helpers()
    _enable_exact_preupdate_binding(monkeypatch)
    active = active_fn(
        1,
        recover_no_load_optim=True,
        recover_no_save_optim=False,
    )

    # no_load_optim leaves the fresh scheduler at LR step 0. The exact-bound
    # pre-update fast-forward moves it to 1, and the successful gs1 update moves
    # it to 2 before the ordinary optimizer-bearing checkpoint is saved.
    lr_num_steps = 0
    if active:
        lr_num_steps += 1
    actor_update_successful = True
    weight_sync_succeeded = True
    optimizer_batch_datums = 8
    if actor_update_successful:
        lr_num_steps += 1
    guard(
        active=active,
        global_step=1,
        optimizer_batch_datums=optimizer_batch_datums,
        actor_update_successful=actor_update_successful,
        weight_sync_succeeded=weight_sync_succeeded,
    )
    saves = [(1, lr_num_steps)]
    candidates = [1] if ready(
        1,
        actor_update_successful=actor_update_successful,
        weight_sync_succeeded=weight_sync_succeeded,
    ) else []

    assert saves == [(1, 2)]
    assert candidates == [1]


def test_preupdate_nooptim_binding_is_exact_and_generic_empty_save_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_fn, guard, _ready = _preupdate_runtime_helpers()
    monkeypatch.delenv(PREUPDATE_NO_LOAD_OPTIM_ENV, raising=False)
    active = active_fn(
        1,
        recover_no_load_optim=False,
        recover_no_save_optim=False,
    )
    assert active is False
    generic_saves: list[int] = []
    guard(
        active=active,
        global_step=1,
        optimizer_batch_datums=0,
        actor_update_successful=False,
        weight_sync_succeeded=False,
    )
    generic_saves.append(1)
    assert generic_saves == [1]

    _enable_exact_preupdate_binding(monkeypatch)
    monkeypatch.setenv("OPENREWARD_PREUPDATE_LOAD_POLICY", "wrong-policy")
    with pytest.raises(RuntimeError, match="exact gs0/LR0/start1 binding"):
        active_fn(
            1,
            recover_no_load_optim=True,
            recover_no_save_optim=False,
        )


def test_resume_expectation_is_checked_before_training_side_effects() -> None:
    source = TRAINER_SOURCE.read_text()
    train_start = source.index("    def train(")
    train_end = source.index("\n    def _evaluate_fn", train_start)
    train = source[train_start:train_end]

    resolve = train.index("start_step = _training_start_step(self.recover_info)")
    bind = train.index("_preupdate_no_load_optim_resume_active(", resolve)
    deadline = train.index("StepDeadlineGuard.from_environment()")
    normalize = train.index("normalize_remote_workflow(")
    loop = train.index("for global_step in range(")
    update = train.index("self.actor.ppo_update(optimizer_batch)", loop)
    fast_forward = train.rindex("self.actor.step_lr_scheduler()", loop, update)
    update_scheduler = train.index("self.actor.step_lr_scheduler()", update)
    promotion_guard = train.index("_require_preupdate_checkpoint_promotion(", update_scheduler)
    recover = train.index("self._save_recover_checkpoint(", promotion_guard)
    assert resolve < bind < deadline < normalize < loop
    assert loop < fast_forward < update < update_scheduler < promotion_guard < recover


class _FakeStepInfo:
    def __init__(self, global_step: int, next_global_step: int) -> None:
        self.global_step = global_step
        self._next_global_step = next_global_step

    def next(self) -> SimpleNamespace:
        return SimpleNamespace(global_step=self._next_global_step)


def _fake_recover_info(global_step: int, next_global_step: int) -> SimpleNamespace:
    return SimpleNamespace(last_step_info=_FakeStepInfo(global_step, next_global_step))


def test_resume_expectation_rejects_missing_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_step = _load_isolated_trainer_function(
        "_training_start_step",
        _EXPECTED_RESUME_GLOBAL_STEP_ENV=EXPECTED_RESUME_STEP_ENV,
    )
    monkeypatch.setenv(EXPECTED_RESUME_STEP_ENV, "5")

    with pytest.raises(RuntimeError, match="requires an exact recovery checkpoint"):
        start_step(None)


def test_resume_expectation_rejects_wrong_recovery_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_step = _load_isolated_trainer_function(
        "_training_start_step",
        _EXPECTED_RESUME_GLOBAL_STEP_ENV=EXPECTED_RESUME_STEP_ENV,
    )
    monkeypatch.setenv(EXPECTED_RESUME_STEP_ENV, "5")

    with pytest.raises(RuntimeError, match=r"expected=5, recovered=4, start=5"):
        start_step(_fake_recover_info(4, 5))


def test_resume_expectation_accepts_exact_recovery_and_next_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_step = _load_isolated_trainer_function(
        "_training_start_step",
        _EXPECTED_RESUME_GLOBAL_STEP_ENV=EXPECTED_RESUME_STEP_ENV,
    )
    monkeypatch.setenv(EXPECTED_RESUME_STEP_ENV, "5")

    assert start_step(_fake_recover_info(5, 6)) == 6


@pytest.mark.parametrize("value", ("", "-1", "+5", "5.0", " 5"))
def test_resume_expectation_rejects_noncanonical_integer(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    start_step = _load_isolated_trainer_function(
        "_training_start_step",
        _EXPECTED_RESUME_GLOBAL_STEP_ENV=EXPECTED_RESUME_STEP_ENV,
    )
    monkeypatch.setenv(EXPECTED_RESUME_STEP_ENV, value)

    with pytest.raises(RuntimeError, match="must be a nonnegative integer"):
        start_step(_fake_recover_info(5, 6))


def test_context_closes_wandb_before_atexit_remote_finalization() -> None:
    entrypoint = TRAIN_ENTRYPOINT.read_text()
    trainer = UPSTREAM_TRAINER.read_text()
    stats_logger = UPSTREAM_STATS_LOGGER.read_text()
    helper = (REPO_ROOT / "platoon/training_success_receipt.py").read_text()

    assert "with OpenRewardArealRLTrainer(" in entrypoint
    assert "def __exit__(self, exc_type, exc_value, traceback):" in trainer
    assert "self.close()" in trainer
    assert "self.stats_logger.close()" in trainer
    assert "def close(self):" in stats_logger
    assert "wandb.finish()" in stats_logger
    assert "atexit.register(_finalize_at_exit" in helper


def _write_recovery_tree(root: Path, experiment: str, trial: str, global_step: int) -> None:
    checkpoint_root = root / "checkpoints" / getpass.getuser() / experiment / trial
    recover_info = checkpoint_root / "recover_info"
    checkpoint = checkpoint_root / "default" / "recover_checkpoint"
    recover_info.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    (recover_info / "step_info.json").write_text(
        json.dumps(
            {
                "global_step": global_step,
                "epoch": 0,
                "epoch_step": 0,
                "steps_per_epoch": 1,
            }
        )
    )
    for name in receipt.RECOVER_INFO_FILES:
        path = recover_info / name
        if path.exists():
            continue
        path.write_bytes(b"{}" if path.suffix == ".json" else b"pickle-state")
    (checkpoint / ".metadata").write_bytes(b"dcp-metadata")
    (checkpoint / "__0_0.distcp").write_bytes(b"optimizer-and-model-state")


class _FakeSummary(dict):
    def __init__(self) -> None:
        super().__init__()
        self.updates = 0

    def update(self, *args, **kwargs) -> None:
        self.updates += 1
        super().update(*args, **kwargs)


class _FakeRun:
    entity = "test-entity"
    project = "test-project"
    id = "test-run"
    url = "https://wandb.example/test-run"

    def __init__(self) -> None:
        self.summary = _FakeSummary()

    def log(self, *_args, **_kwargs) -> None:
        raise AssertionError("success receipt must not mutate W&B training history")


class _FakeStatsLogger:
    def state_dict(self) -> dict[str, int]:
        return {"last_commit_step": 0}


def test_receipt_hook_is_a_noop_when_not_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(receipt.RECEIPT_ENV, raising=False)

    class _MustNotBeRead:
        def state_dict(self):
            raise AssertionError("env-absent receipt hook inspected runtime state")

    result = receipt.maybe_publish_training_success_candidate(
        fileroot=tmp_path / "missing",
        experiment="",
        trial="",
        config_path="",
        pbs_job_id="",
        global_step=-1,
        optimizer_update_successful=False,
        optimizer_batch_datums=0,
        optimizer_action_tokens=0,
        weight_sync_succeeded=False,
        expected_version=0,
        actor_version=0,
        rollout_version=0,
        stats_logger=_MustNotBeRead(),
        wandb_mode="disabled",
        training_sif_path="",
        training_sif_sha256="",
        not_before_epoch="",
    )
    assert result is None


def test_candidate_publication_predicate_avoids_repeat_argument_rpcs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "success.json"
    monkeypatch.delenv(receipt.RECEIPT_ENV, raising=False)
    assert receipt.candidate_publication_enabled() is False
    monkeypatch.setenv(receipt.RECEIPT_ENV, str(final))
    assert receipt.candidate_publication_enabled() is True

    candidate, published_final, _run, _experiment, _trial = _publish_valid_candidate(
        tmp_path,
        monkeypatch,
    )
    assert candidate.is_file()
    monkeypatch.setenv(receipt.RECEIPT_ENV, str(published_final))
    assert receipt.candidate_publication_enabled() is False


def test_candidate_publication_minimum_step_is_strict_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(receipt.RECEIPT_ENV, str(tmp_path / "success.json"))
    monkeypatch.setenv(receipt.MIN_GLOBAL_STEP_ENV, "27")
    assert all(receipt.candidate_publication_enabled(step) is False for step in range(27))
    assert receipt.candidate_publication_enabled(27) is True
    monkeypatch.setenv(receipt.MIN_GLOBAL_STEP_ENV, "-1")
    with pytest.raises(RuntimeError, match=receipt.MIN_GLOBAL_STEP_ENV):
        receipt.candidate_publication_enabled(27)


def test_finalize_candidate_cli_is_explicit_and_post_cleanup_capable() -> None:
    source = (REPO_ROOT / "platoon/training_success_receipt.py").read_text()
    assert 'subparsers.add_parser("finalize-candidate")' in source
    assert 'finalize_parser.add_argument("--candidate"' in source
    assert 'finalize_parser.add_argument("--receipt"' in source
    assert 'finalize_parser.add_argument("--remote-timeout-seconds"' in source


@pytest.mark.parametrize(
    ("value", "registered"),
    (("0", False), ("false", False), ("1", True), ("true", True)),
)
def test_finalization_env_is_strict_and_controls_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    registered: bool,
) -> None:
    final = tmp_path / "gate" / "success.json"
    monkeypatch.setenv(receipt.FINALIZE_ENV, value)
    monkeypatch.setattr(receipt.atexit, "register", lambda *_args: None)
    _candidate, published_final, _run, _experiment, _trial = _publish_valid_candidate(
        tmp_path,
        monkeypatch,
    )
    # The common fixture disables registration explicitly; make a fresh
    # publication with default registration to exercise the environment.
    receipt._PUBLISHED_RECEIPTS.discard(published_final)
    candidate = receipt._candidate_path(published_final)
    candidate.unlink()
    receipt.maybe_publish_training_success_candidate(
        fileroot=tmp_path,
        experiment="abci-openreward-gate5-test",
        trial="abci-gate5-test-trial0",
        config_path=tmp_path / "config.yaml",
        pbs_job_id="12345.pbs1",
        global_step=0,
        optimizer_update_successful=True,
        optimizer_batch_datums=5,
        optimizer_action_tokens=37,
        weight_sync_succeeded=True,
        expected_version=1,
        actor_version=1,
        rollout_version=1,
        stats_logger=_FakeStatsLogger(),
        wandb_mode="online",
        training_sif_path=tmp_path / "training.sif",
        training_sif_sha256="a" * 64,
        not_before_epoch=1,
        receipt_path=final,
    )
    assert (published_final in receipt._REGISTERED_RECEIPTS) is registered


def test_finalization_env_rejects_ambiguous_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(receipt.FINALIZE_ENV, "yes")
    monkeypatch.setenv(receipt.RECEIPT_ENV, str(tmp_path / "success.json"))
    with pytest.raises(RuntimeError, match=receipt.FINALIZE_ENV):
        receipt.maybe_publish_training_success_candidate(
            fileroot="not-read",
            experiment="not-read",
            trial="not-read",
            config_path="not-read",
            pbs_job_id="not-read",
            global_step=0,
            optimizer_update_successful=True,
            optimizer_batch_datums=1,
            optimizer_action_tokens=1,
            weight_sync_succeeded=True,
            expected_version=1,
            actor_version=1,
            rollout_version=1,
            stats_logger=None,
            wandb_mode="online",
            training_sif_path="not-read",
            training_sif_sha256="a" * 64,
            not_before_epoch=1,
        )


def _publish_valid_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    experiment = "abci-openreward-gate5-test"
    trial = "abci-gate5-test-trial0"
    _write_recovery_tree(tmp_path, experiment, trial, 0)
    config = tmp_path / "config.yaml"
    config.write_text("total_train_steps: 1\n")
    training_sif = tmp_path / "training.sif"
    training_sif.write_bytes(b"sif")
    fake_run = _FakeRun()
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=fake_run))
    final = tmp_path / "gate" / "success.json"
    candidate = receipt.maybe_publish_training_success_candidate(
        fileroot=tmp_path,
        experiment=experiment,
        trial=trial,
        config_path=config,
        pbs_job_id="12345.pbs1",
        global_step=0,
        optimizer_update_successful=True,
        optimizer_batch_datums=5,
        optimizer_action_tokens=37,
        weight_sync_succeeded=True,
        expected_version=1,
        actor_version=1,
        rollout_version=1,
        stats_logger=_FakeStatsLogger(),
        wandb_mode="online",
        training_sif_path=training_sif,
        training_sif_sha256="a" * 64,
        not_before_epoch=1,
        receipt_path=final,
        register_finalizer=False,
    )
    assert candidate is not None
    return candidate, final, fake_run, experiment, trial


def test_success_receipt_binds_update_sync_checkpoint_and_remote_wandb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path, final, fake_run, experiment, trial = _publish_valid_candidate(
        tmp_path,
        monkeypatch,
    )
    assert os.stat(candidate_path).st_mode & 0o777 == 0o600
    assert fake_run.summary.updates == 1
    summary_metrics = dict(fake_run.summary)
    assert summary_metrics[f"{receipt.METRIC_PREFIX}/optimizer_update_succeeded"] == 1
    assert summary_metrics[f"{receipt.METRIC_PREFIX}/weight_sync_succeeded"] == 1
    assert summary_metrics[f"{receipt.METRIC_PREFIX}/recover_checkpoint_durable"] == 1

    def observe(candidate, _timeout):
        return {
            "path": candidate["wandb"]["path"],
            "url": candidate["wandb"]["url"],
            "state": "finished",
            "summary_metrics": candidate["wandb"]["summary_metrics"],
            "observed_at_utc": "2026-08-13T00:00:00+00:00",
        }

    receipt.finalize_candidate(
        candidate_path,
        final,
        timeout_seconds=1,
        remote_observer=observe,
    )
    verified = receipt.verify_receipt(
        final,
        pbs_job_id="12345.pbs1",
        experiment=experiment,
        trial=trial,
        global_step=0,
    )
    assert os.stat(final).st_mode & 0o777 == 0o600
    assert verified["optimizer_batch_datums"] == 5
    assert verified["optimizer_action_tokens"] == 37
    assert verified["actor_version"] == verified["rollout_version"] == 1
    assert verified["wandb_remote"]["summary_metrics"] == summary_metrics


def test_candidate_is_first_success_only_and_has_a_live_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path, _final, fake_run, experiment, trial = _publish_valid_candidate(
        tmp_path,
        monkeypatch,
    )
    original = candidate_path.read_bytes()

    # A later training step in the same controller must leave the first
    # successful update evidence stable, with no second W&B proof row.
    repeated = receipt.maybe_publish_training_success_candidate(
        fileroot="not-read",
        experiment="not-read",
        trial="not-read",
        config_path="not-read",
        pbs_job_id="not-read",
        global_step=99,
        optimizer_update_successful=False,
        optimizer_batch_datums=0,
        optimizer_action_tokens=0,
        weight_sync_succeeded=False,
        expected_version=0,
        actor_version=0,
        rollout_version=0,
        stats_logger=None,
        wandb_mode="disabled",
        training_sif_path="not-read",
        training_sif_sha256="not-read",
        not_before_epoch="not-read",
        receipt_path=_final,
        register_finalizer=False,
    )
    assert repeated == candidate_path
    assert candidate_path.read_bytes() == original
    assert fake_run.summary.updates == 1

    verified = receipt.verify_candidate(
        candidate_path,
        pbs_job_id="12345.pbs1",
        experiment=experiment,
        trial=trial,
        global_step=0,
    )
    assert verified["kind"] == "candidate"


def test_live_candidate_verifier_accepts_a_visible_running_wandb_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path, _final, _fake_run, _experiment, _trial = _publish_valid_candidate(
        tmp_path,
        monkeypatch,
    )

    def observe(candidate, _timeout, *, allowed_states):
        assert allowed_states == {"running", "finished"}
        return {
            "path": candidate["wandb"]["path"],
            "state": "running",
            "summary_metrics": candidate["wandb"]["summary_metrics"],
        }

    monkeypatch.setattr(receipt, "_observe_wandb_run", observe)
    verified = receipt.verify_candidate(
        candidate_path,
        require_remote_wandb=True,
        remote_timeout_seconds=1,
    )
    assert verified["wandb_remote_observed"]["state"] == "running"


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"optimizer_update_successful": False}, "finite optimizer update"),
        ({"optimizer_batch_datums": 0}, "optimizer_batch_datums"),
        ({"optimizer_action_tokens": 0}, "optimizer_action_tokens"),
        ({"weight_sync_succeeded": False}, "real weight-sync branch"),
        ({"actor_version": 2}, "actor and rollout versions"),
        ({"rollout_version": 2}, "actor and rollout versions"),
        ({"wandb_mode": "offline"}, "W&B online mode"),
    ),
)
def test_candidate_refuses_false_positive_update_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: dict,
    message: str,
) -> None:
    experiment = "abci-openreward-gate5-test"
    trial = "abci-gate5-test-trial0"
    _write_recovery_tree(tmp_path, experiment, trial, 0)
    config = tmp_path / "config.yaml"
    config.write_text("total_train_steps: 1\n")
    training_sif = tmp_path / "training.sif"
    training_sif.write_bytes(b"sif")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=_FakeRun()))
    arguments = {
        "fileroot": tmp_path,
        "experiment": experiment,
        "trial": trial,
        "config_path": config,
        "pbs_job_id": "12345.pbs1",
        "global_step": 0,
        "optimizer_update_successful": True,
        "optimizer_batch_datums": 5,
        "optimizer_action_tokens": 37,
        "weight_sync_succeeded": True,
        "expected_version": 1,
        "actor_version": 1,
        "rollout_version": 1,
        "stats_logger": _FakeStatsLogger(),
        "wandb_mode": "online",
        "training_sif_path": training_sif,
        "training_sif_sha256": "a" * 64,
        "not_before_epoch": 1,
        "receipt_path": tmp_path / "success.json",
        "register_finalizer": False,
    }
    arguments.update(override)
    with pytest.raises((RuntimeError, ValueError), match=message):
        receipt.maybe_publish_training_success_candidate(**arguments)


def test_final_receipt_rejects_changed_checkpoint_after_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate, final, _run, _experiment, _trial = _publish_valid_candidate(
        tmp_path,
        monkeypatch,
    )
    checkpoint_payload = next(path for path in tmp_path.rglob("*.distcp") if path.is_file())
    checkpoint_payload.write_bytes(b"changed-after-candidate")

    with pytest.raises(ValueError, match="recovery evidence changed"):
        receipt.finalize_candidate(
            candidate,
            final,
            timeout_seconds=1,
            remote_observer=lambda *_args: {},
        )


@pytest.mark.parametrize("stale_kind", ("final", "candidate", "error"))
def test_candidate_refuses_stale_artifacts_before_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale_kind: str,
) -> None:
    experiment = "abci-openreward-gate5-test"
    trial = "abci-gate5-test-trial0"
    _write_recovery_tree(tmp_path, experiment, trial, 0)
    config = tmp_path / "config.yaml"
    config.write_text("total_train_steps: 1\n")
    training_sif = tmp_path / "training.sif"
    training_sif.write_bytes(b"sif")
    fake_run = _FakeRun()
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=fake_run))
    final = tmp_path / "gate" / "success.json"
    stale_paths = {
        "final": final,
        "candidate": receipt._candidate_path(final),
        "error": receipt._error_path(final),
    }
    stale = stale_paths[stale_kind]
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("stale")

    with pytest.raises(RuntimeError, match="refusing stale training success evidence"):
        receipt.maybe_publish_training_success_candidate(
            fileroot=tmp_path,
            experiment=experiment,
            trial=trial,
            config_path=config,
            pbs_job_id="12345.pbs1",
            global_step=0,
            optimizer_update_successful=True,
            optimizer_batch_datums=5,
            optimizer_action_tokens=37,
            weight_sync_succeeded=True,
            expected_version=1,
            actor_version=1,
            rollout_version=1,
            stats_logger=_FakeStatsLogger(),
            wandb_mode="online",
            training_sif_path=training_sif,
            training_sif_sha256="a" * 64,
            not_before_epoch=1,
            receipt_path=final,
            register_finalizer=False,
        )
    assert fake_run.summary.updates == 0
