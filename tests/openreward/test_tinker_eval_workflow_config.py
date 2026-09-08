"""The eval workflow config must not silently drop fields.

The OpenReward tinker train script rebuilt the eval ``WorkflowConfig`` from an
explicit field subset whenever eval ``group_size != 1``, so **every** field the
subset did not name was discarded -- including every rollout-isolation key.
Production escapes today only because the shipped config sets
``eval.strategy: none`` and eval ``group_size: 1``, but an unisolated eval
rollout reintroduces exactly the two hazards process isolation removes: a hung
programmatic-tool-calling cell holding a trainer executor thread forever, and a
segfault costing the trainer's ~490 GB checkpoint reload.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import platoon.train.tinker  # noqa: F401
from platoon.config_defs import RolloutConfig
from platoon.train.tinker.config_defs import WorkflowConfig

TRAIN_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "plugins/openreward/platoon/openreward/train_scripts/tinker/train_tinker.py"
)

# The fields eval is allowed to force, and nothing else.  ``straggler_quorum``
# and ``min_successful_group_size`` are in the set only because ``group_size``
# is forced to 1 and both are validated against it, so leaving a configured
# value in place would make the trainer raise at startup.  The old field-subset
# construction hid that by dropping those keys along with everything else it
# did not name.
EVAL_FORCED_FIELDS = {
    "group_size",
    "filter_errors",
    "filter_zero_advantage_datums",
    "straggler_quorum",
    "min_successful_group_size",
}


def _rebuild(config: WorkflowConfig) -> WorkflowConfig:
    """The script's own transformation, applied here."""

    return dataclasses.replace(
        config,
        group_size=1,
        filter_errors=False,
        filter_zero_advantage_datums=False,
        straggler_quorum=None,
        min_successful_group_size=1,
    )


def test_isolation_keys_survive_the_eval_rebuild():
    original = WorkflowConfig(
        group_size=4,
        rollout_config=RolloutConfig(output_dir="eval-out", timeout=1800),
        filter_errors=True,
        filter_zero_advantage_datums=True,
        rollout_isolation="process",
        worker_startup_budget_seconds=222.0,
        worker_rss_soft_limit_bytes=1234,
        worker_launch_concurrent_groups=3,
        worker_python_env_isolation=True,
        min_successful_group_size=2,
        straggler_timeout_seconds=900.0,
        straggler_quorum=3,
    )
    rebuilt = _rebuild(original)

    assert rebuilt.group_size == 1
    assert rebuilt.filter_errors is False
    assert rebuilt.filter_zero_advantage_datums is False
    assert rebuilt.straggler_quorum is None
    assert rebuilt.min_successful_group_size == 1
    assert rebuilt.straggler_timeout_seconds == 900.0
    # The reason this test exists.
    assert rebuilt.rollout_isolation == "process"
    assert rebuilt.worker_startup_budget_seconds == 222.0
    assert rebuilt.worker_rss_soft_limit_bytes == 1234
    assert rebuilt.worker_launch_concurrent_groups == 3
    assert rebuilt.worker_python_env_isolation is True
    assert rebuilt.rollout_config is original.rollout_config


def test_the_rebuild_changes_exactly_the_three_intended_fields():
    """Property-style, so a future field addition is covered without an edit."""

    original = WorkflowConfig(
        group_size=4,
        filter_errors=True,
        filter_zero_advantage_datums=True,
        rollout_isolation="process",
        worker_startup_budget_seconds=111.0,
        min_successful_group_size=2,
        straggler_timeout_seconds=60.0,
        straggler_quorum=2,
        depth_level_weighting=True,
        leave_one_out_baseline=True,
        subagent_datum_keep_probability=0.5,
        subagent_datum_sampling_seed=17,
    )
    rebuilt = _rebuild(original)

    changed = {
        field.name
        for field in dataclasses.fields(WorkflowConfig)
        if getattr(original, field.name) != getattr(rebuilt, field.name)
    }
    assert changed == EVAL_FORCED_FIELDS


def test_the_train_script_uses_replace_and_not_a_field_subset():
    """Guards the actual production code, not just an equivalent here."""

    source = TRAIN_SCRIPT.read_text()
    tree = ast.parse(source)
    replace_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "replace"
    ]
    forced = set()
    for call in replace_calls:
        forced |= {keyword.arg for keyword in call.keywords if keyword.arg}
    assert forced == EVAL_FORCED_FIELDS, forced
    # A bare WorkflowConfig(...) construction is what silently dropped fields.
    assert "WorkflowConfig(" not in source
