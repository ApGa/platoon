"""AReaL-style checkpoint cadence: permanent, recovery, and weights-only saves.

``_save_checkpoint_and_get_sampling_client`` picks one of three behaviours per
step. A permanent save (``steps_completed % save_every == 0``) writes
``<batch>``; otherwise a ``recovery_every`` hit writes ``recovery-<batch>`` with
the same weights+optimizer payload; otherwise the weights-only update runs. Both
full saves land in ``checkpoints.jsonl`` with a ``state_path``, so
``get_last_checkpoint(required_key="state_path")`` resumes from whichever is
newest regardless of its name.

    HF_HUB_OFFLINE=1 plugins/openreward/.venv/bin/python -m pytest tests/train/tinker -q
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from platoon.train.tinker import rl as rl_module  # noqa: E402
from platoon.train.tinker.config_defs import CheckpointConfig  # noqa: E402
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer  # noqa: E402
from tinker_cookbook import checkpoint_utils  # noqa: E402


class _FakeTrainingClient:
    """Records the two save paths the trainer can take."""

    def __init__(self) -> None:
        self.weights_only_calls = 0
        self.sampling_clients: list[str] = []

    async def save_weights_and_get_sampling_client_async(self):
        self.weights_only_calls += 1
        return f"weights-only-sampler-{self.weights_only_calls}"

    def create_sampling_client(self, sampler_path: str):
        self.sampling_clients.append(sampler_path)
        return f"sampler-from:{sampler_path}"


def _trainer(tmp_path) -> PlatoonTinkerRLTrainer:
    trainer = object.__new__(PlatoonTinkerRLTrainer)
    trainer.config = SimpleNamespace(
        log_path=str(tmp_path),
        stats=SimpleNamespace(experiment_name="exp", trial_name="trial"),
    )
    trainer._stats_logger = SimpleNamespace(wandb_run_id="run-abc")
    os.makedirs(trainer.run_log_path, exist_ok=True)
    return trainer


def _record_saves(monkeypatch) -> list[tuple[str, str, dict]]:
    """Patch save_checkpoint_async to record (name, kind, loop_state)."""
    saves: list[tuple[str, str, dict]] = []

    async def fake_save_checkpoint_async(*, training_client, name, log_path, loop_state, kind, **kwargs):
        saves.append((name, kind, dict(loop_state)))
        return {
            "state_path": f"tinker://model-a/weights/{name}",
            "sampler_path": f"tinker://model-a/sampler_weights/{name}",
        }

    monkeypatch.setattr(rl_module.checkpoint_utils, "save_checkpoint_async", fake_save_checkpoint_async)
    return saves


def _run_steps(trainer, client, *, num_steps: int, save_every: int, recovery_every: int | None):
    async def run():
        for i_batch in range(num_steps):
            await trainer._save_checkpoint_and_get_sampling_client(
                training_client=client,
                i_batch=i_batch,
                save_every=save_every,
                start_batch=0,
                watchdog=None,
                recovery_every=recovery_every,
            )

    asyncio.run(run())


def test_recovery_cadence_saves_a_full_checkpoint_every_step(monkeypatch, tmp_path):
    saves = _record_saves(monkeypatch)
    trainer = _trainer(tmp_path)
    client = _FakeTrainingClient()

    _run_steps(trainer, client, num_steps=6, save_every=3, recovery_every=1)

    assert [name for name, _kind, _state in saves] == [
        "recovery-000001",
        "recovery-000002",
        "000003",
        "recovery-000004",
        "recovery-000005",
        "000006",
    ]
    assert {kind for _name, kind, _state in saves} == {"both"}
    # Every step wrote a resumable checkpoint, so nothing took the weights-only path.
    assert client.weights_only_calls == 0
    # The sampling client always comes from the checkpoint that was just written.
    assert client.sampling_clients == [f"tinker://model-a/sampler_weights/{name}" for name, _k, _s in saves]
    # Recovery saves carry the same loop_state as permanent ones (batch + wandb run id).
    assert [state["batch"] for _n, _k, state in saves] == [1, 2, 3, 4, 5, 6]
    assert {state["wandb_run_id"] for _n, _k, state in saves} == {"run-abc"}


def test_without_recovery_every_the_cadence_is_unchanged(monkeypatch, tmp_path):
    saves = _record_saves(monkeypatch)
    trainer = _trainer(tmp_path)
    client = _FakeTrainingClient()

    _run_steps(trainer, client, num_steps=6, save_every=3, recovery_every=None)

    assert [name for name, _kind, _state in saves] == ["000003", "000006"]
    assert {kind for _name, kind, _state in saves} == {"both"}
    assert client.weights_only_calls == 4  # batches 0, 1, 3, 4


def test_permanent_cadence_wins_when_both_trigger(monkeypatch, tmp_path):
    saves = _record_saves(monkeypatch)
    trainer = _trainer(tmp_path)
    client = _FakeTrainingClient()

    _run_steps(trainer, client, num_steps=4, save_every=2, recovery_every=2)

    assert [name for name, _kind, _state in saves] == ["000002", "000004"]
    assert client.weights_only_calls == 2


def test_recovery_cadence_counts_from_start_batch_after_a_resume(monkeypatch, tmp_path):
    saves = _record_saves(monkeypatch)
    trainer = _trainer(tmp_path)
    client = _FakeTrainingClient()

    async def run():
        for i_batch in range(10, 14):
            await trainer._save_checkpoint_and_get_sampling_client(
                training_client=client,
                i_batch=i_batch,
                save_every=3,
                start_batch=10,
                watchdog=None,
                recovery_every=2,
            )

    asyncio.run(run())

    # steps_completed 1..4 -> recovery at 2 and 4, permanent at 3.
    assert [name for name, _kind, _state in saves] == ["recovery-000012", "000013", "recovery-000014"]
    assert client.weights_only_calls == 1  # steps_completed == 1


def test_resume_picks_the_newest_state_record_even_when_it_is_a_recovery(tmp_path):
    """get_last_checkpoint(required_key="state_path") ignores the name."""
    log_dir = tmp_path / "exp" / "trial"
    log_dir.mkdir(parents=True)
    records = [
        {"name": "recovery-000007", "batch": 7, "state_path": "tinker://m/weights/recovery-000007",
         "sampler_path": "tinker://m/sampler_weights/recovery-000007"},
        {"name": "000005", "batch": 5, "state_path": "tinker://m/weights/000005",
         "sampler_path": "tinker://m/sampler_weights/000005"},
        {"name": "recovery-000008", "batch": 8, "state_path": "tinker://m/weights/recovery-000008",
         "sampler_path": "tinker://m/sampler_weights/recovery-000008"},
    ]
    (log_dir / "checkpoints.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))

    resume = checkpoint_utils.get_last_checkpoint(str(log_dir), required_key="state_path")

    assert resume is not None
    assert resume.name == "recovery-000008"
    assert resume.batch == 8
    assert resume.state_path == "tinker://m/weights/recovery-000008"


def test_recovery_every_must_be_positive_or_null():
    assert CheckpointConfig().recovery_every is None
    assert CheckpointConfig(strategy="step", every=20, recovery_every=1).recovery_every == 1
    for bad in (0, -1):
        try:
            CheckpointConfig(recovery_every=bad)
        except ValueError as exc:
            assert "recovery_every" in str(exc)
        else:
            raise AssertionError(f"recovery_every={bad} should be rejected")
