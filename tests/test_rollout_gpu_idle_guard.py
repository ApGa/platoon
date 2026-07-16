from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "slurm-scripts" / "rollout_gpu_idle_guard.py"


@pytest.fixture()
def guard_module():
    spec = importlib.util.spec_from_file_location("rollout_gpu_idle_guard", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SequenceProbe:
    def __init__(self, module, utilizations):
        self._module = module
        self._utilizations = iter(utilizations)

    def read(self):
        return self._module.GpuUtilization(
            index=3,
            uuid="GPU-test",
            utilization=next(self._utilizations),
        )


class RecordingBurster:
    low_priority = 0

    def __init__(self):
        self.calls = []

    def burst(self, seconds):
        self.calls.append(seconds)
        return seconds + 0.01, 64


def test_defaults_use_bounded_idle_duty_and_sample_utilization_twice(guard_module):
    config = guard_module.GuardConfig()

    config.validate()

    assert config.interval_seconds == 10
    assert config.interval_jitter_seconds == 2
    assert config.sample_count == 2
    assert config.utilization_threshold == 10
    assert config.burst_seconds == 2
    assert config.matrix_dim == 1024
    assert config.sample_interval_seconds == 2
    assert config.burst_seconds / config.interval_seconds == 0.2
    assert config.burst_seconds / config.interval_seconds <= 0.25


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"matrix_dim": 4096}, "matrix_dim"),
        ({"interval_seconds": 10, "burst_seconds": 2.6}, "duty cycle"),
        ({"interval_seconds": 9}, "interval_seconds"),
        ({"interval_jitter_seconds": -0.1}, "interval_jitter_seconds"),
        ({"interval_jitter_seconds": 10.1}, "interval_jitter_seconds"),
        ({"sample_count": 1}, "sample_count"),
        ({"burst_seconds": 2.1}, "burst_seconds"),
        ({"expected_devices": 8}, "one guard process"),
    ],
)
def test_safety_validation_rejects_high_overhead_configs(guard_module, overrides, match):
    values = guard_module.GuardConfig().__dict__ | overrides

    with pytest.raises(ValueError, match=match):
        guard_module.GuardConfig(**values).validate()


def test_nvidia_smi_rows_are_parsed_strictly(guard_module):
    rows = guard_module.parse_nvidia_smi_output("0, GPU-aaaa, 0\n3, GPU-bbbb, 17\n")

    assert [(row.index, row.uuid, row.utilization) for row in rows] == [
        (0, "GPU-aaaa", 0),
        (3, "GPU-bbbb", 17),
    ]

    with pytest.raises(ValueError, match="invalid nvidia-smi row"):
        guard_module.parse_nvidia_smi_output("0, GPU-aaaa, N/A\n")


def test_visible_gpu_selection_uses_physical_index_or_uuid(guard_module):
    rows = guard_module.parse_nvidia_smi_output("0, GPU-aaaa, 0\n3, GPU-bbbb, 17\n")

    assert guard_module.select_visible_gpu(rows, ("3",)).uuid == "GPU-bbbb"
    assert guard_module.select_visible_gpu(rows, ("GPU-aaaa",)).index == 0

    with pytest.raises(RuntimeError, match="exactly one"):
        guard_module.select_visible_gpu(rows, ())


def test_single_visible_nvidia_smi_row_handles_container_index_remap(guard_module):
    rows = guard_module.parse_nvidia_smi_output("7, GPU-remapped, 0\n")

    selected = guard_module.select_visible_gpu(rows, ("0",))

    assert selected.index == 7


def test_recent_active_sample_skips_burst(guard_module):
    config = guard_module.GuardConfig()
    burster = RecordingBurster()
    waits = []

    result = guard_module.run_cycle(
        SequenceProbe(guard_module, [0, 10]),
        burster,
        config,
        waits.append,
    )

    assert result.action == "skip-active"
    assert result.samples == (0, 10)
    assert burster.calls == []
    assert waits == [2]


def test_two_idle_samples_trigger_one_bounded_burst(guard_module):
    config = guard_module.GuardConfig()
    burster = RecordingBurster()

    result = guard_module.run_cycle(
        SequenceProbe(guard_module, [0, 3]),
        burster,
        config,
        lambda _seconds: None,
    )

    assert result.action == "burst"
    assert burster.calls == [2]
    assert result.operations == 64
    assert result.burst_elapsed_seconds == pytest.approx(2.01)


def test_interval_jitter_is_deterministic_and_bounded_by_slurm_identity(guard_module):
    identity = {
        "SLURM_JOB_ID": "14078417",
        "SLURM_STEP_ID": "6",
        "SLURM_PROCID": "47",
    }
    seed = guard_module.deterministic_jitter_seed(identity)

    assert seed == guard_module.deterministic_jitter_seed(dict(identity))
    for name in identity:
        changed = identity | {name: f"{identity[name]}-different"}
        assert guard_module.deterministic_jitter_seed(changed) != seed

    config = guard_module.GuardConfig()
    first_rng = guard_module.random.Random(seed)
    second_rng = guard_module.random.Random(seed)
    first = [guard_module.cycle_interval_jitter(config, first_rng) for _ in range(8)]
    second = [guard_module.cycle_interval_jitter(config, second_rng) for _ in range(8)]

    assert first == second
    assert all(0 <= jitter <= 2 for jitter in first)
    assert len(set(first)) > 1
    assert (
        guard_module.cycle_interval_jitter(
            guard_module.GuardConfig(interval_jitter_seconds=0),
            guard_module.random.Random(seed),
        )
        == 0
    )


def test_burster_uses_public_default_stream_priority_without_range_query(
    guard_module, monkeypatch
):
    class FakeCuda:
        def __init__(self):
            self.stream_args = None

        def is_available(self):
            return True

        def device_count(self):
            return 1

        def set_device(self, _device):
            return None

        def Stream(self, *, device, priority):
            self.stream_args = (device, priority)
            return object()

    cuda = FakeCuda()
    fake_torch = type("FakeTorch", (), {"cuda": cuda})
    assert not hasattr(cuda, "get_stream_priority_range")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    burster = guard_module.TorchBf16Burster(matrix_dim=1024, operations_per_sync=32)

    assert burster.low_priority == 0
    assert cuda.stream_args == (0, 0)


def test_ready_marker_is_atomic_and_identifies_exact_gpu(guard_module, tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "47")
    gpu = guard_module.GpuUtilization(index=7, uuid="GPU-last", utilization=0)

    marker = guard_module.publish_ready(
        tmp_path,
        gpu,
        guard_module.GuardConfig(),
        low_priority=0,
        jitter_seed=123456,
    )

    assert marker == tmp_path / "47.ready"
    assert marker is not None
    contents = marker.read_text(encoding="utf-8")
    assert "gpu_index=7" in contents
    assert "gpu_uuid=GPU-last" in contents
    assert "stream_priority=0" in contents
    assert "interval_jitter_seconds=2" in contents
    assert "jitter_seed=123456" in contents
    assert list(tmp_path.glob("*.tmp")) == []


def test_environment_defaults_can_be_overridden(guard_module, monkeypatch):
    monkeypatch.setenv("ROLLOUT_IDLE_GUARD_UTILIZATION_THRESHOLD", "15")
    monkeypatch.setenv("ROLLOUT_IDLE_GUARD_INTERVAL_SECONDS", "40")
    monkeypatch.setenv("ROLLOUT_IDLE_GUARD_INTERVAL_JITTER_SECONDS", "1.5")

    config = guard_module.config_from_args(guard_module.parse_args([]))

    assert config.utilization_threshold == 15
    assert config.interval_seconds == 40
    assert config.interval_jitter_seconds == 1.5


def test_script_is_executable_after_install():
    assert SCRIPT_PATH.exists()
    # The source tree mode is part of the launcher contract.
    assert os.access(SCRIPT_PATH, os.X_OK)


def test_openreward_launcher_wires_role_specific_runtime_guards():
    launcher = (REPO_ROOT / "slurm-scripts" / "openreward-toolathlon-prealloc-base.sh").read_text(encoding="utf-8")

    assert "ROLLOUT_IDLE_GUARD_READY_DIR=${OPENREWARD_JOB_STATE_DIR}/gpu-idle-guard-ready" in launcher
    assert "${ROLLOUT_IDLE_GUARD_READY_DIR}/actor" in launcher
    assert "${ROLLOUT_IDLE_GUARD_READY_DIR}/rollout" in launcher
    assert "PLATOON_AREAL_ROLLOUT_IDLE_GUARD_SCRIPT" in launcher
    assert "PLATOON_AREAL_ROLLOUT_IDLE_GUARD_PYTHON" in launcher
    assert "gpu-idle-guard-${RUN_ID}-${JOB_INSTANCE_ID}" in launcher
    assert "ROLLOUT_IDLE_GUARD_INTERVAL_SECONDS:-10" in launcher
    assert "ROLLOUT_IDLE_GUARD_INTERVAL_JITTER_SECONDS:-2" in launcher
    assert "ROLLOUT_IDLE_GUARD_BURST_SECONDS:-2" in launcher
    assert "ROLLOUT_IDLE_GUARD_MATRIX_DIM:-1024" in launcher
