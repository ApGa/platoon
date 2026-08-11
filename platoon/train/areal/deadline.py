"""Step-boundary draining for wall-time-limited training allocations."""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

DEADLINE_EPOCH_ENV = "PLATOON_TRAINING_DEADLINE_EPOCH"
DRAIN_FILE_ENV = "PLATOON_TRAINING_DRAIN_FILE"
INITIAL_STEP_SECONDS_ENV = "PLATOON_DEADLINE_INITIAL_STEP_SECONDS"
SAFETY_SECONDS_ENV = "PLATOON_DEADLINE_SAFETY_SECONDS"
HISTORY_SIZE_ENV = "PLATOON_DEADLINE_HISTORY_SIZE"
HISTORY_MULTIPLIER_ENV = "PLATOON_DEADLINE_HISTORY_MULTIPLIER"


def _finite_float(
    environment: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
) -> float:
    raw = environment.get(name)
    value = default if raw is None or raw == "" else float(raw)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}, got {raw!r}")
    return value


def _positive_int(
    environment: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    raw = environment.get(name)
    value = default if raw is None or raw == "" else int(raw)
    if value < 1:
        raise ValueError(f"{name} must be positive, got {raw!r}")
    return value


@dataclass(frozen=True)
class DeadlineDecision:
    """Snapshot used to explain a step-start or drain decision."""

    should_drain: bool
    now_epoch: float
    deadline_epoch: float
    remaining_seconds: float
    estimated_step_seconds: float
    safety_seconds: float
    required_seconds: float
    completed_steps: int


@dataclass
class StepDeadlineGuard:
    """Estimate complete-step time and stop only at a checkpoint boundary.

    The guard is deliberately conservative: the configured estimate is a
    permanent floor, while the maximum recent steady-state complete-step
    duration is multiplied by a headroom factor. The first completed step is
    excluded from timing history because cold asynchronous rollout startup is
    not representative of later steps in the same allocation. A decision is
    made before rollout starts, so a drain never leaves an optimizer update or
    recovery checkpoint half written.
    """

    deadline_epoch: float
    drain_file: Path
    initial_step_seconds: float = 1800.0
    safety_seconds: float = 300.0
    history_size: int = 8
    history_multiplier: float = 1.15
    _durations: deque[float] = field(init=False, repr=False)
    _completed_steps: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        numeric = {
            "deadline_epoch": self.deadline_epoch,
            "initial_step_seconds": self.initial_step_seconds,
            "safety_seconds": self.safety_seconds,
            "history_multiplier": self.history_multiplier,
        }
        if any(not math.isfinite(float(value)) for value in numeric.values()):
            raise ValueError(f"Deadline settings must be finite: {numeric}")
        if self.deadline_epoch <= 0:
            raise ValueError("deadline_epoch must be positive")
        if self.initial_step_seconds < 0 or self.safety_seconds < 0:
            raise ValueError("Deadline duration settings must be non-negative")
        if self.history_size < 1:
            raise ValueError("history_size must be positive")
        if self.history_multiplier < 1:
            raise ValueError("history_multiplier must be >= 1")
        self._durations = deque(maxlen=self.history_size)

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> StepDeadlineGuard | None:
        """Build a guard from launcher-provided environment variables."""

        values = os.environ if environment is None else environment
        raw_deadline = values.get(DEADLINE_EPOCH_ENV)
        if raw_deadline is None or raw_deadline == "":
            return None
        raw_drain_file = values.get(DRAIN_FILE_ENV)
        if raw_drain_file is None or raw_drain_file == "":
            raise ValueError(f"{DRAIN_FILE_ENV} is required when {DEADLINE_EPOCH_ENV} is set")
        return cls(
            deadline_epoch=_finite_float(
                values,
                DEADLINE_EPOCH_ENV,
                0.0,
                minimum=1.0,
            ),
            drain_file=Path(raw_drain_file),
            initial_step_seconds=_finite_float(
                values,
                INITIAL_STEP_SECONDS_ENV,
                1800.0,
                minimum=0.0,
            ),
            safety_seconds=_finite_float(
                values,
                SAFETY_SECONDS_ENV,
                300.0,
                minimum=0.0,
            ),
            history_size=_positive_int(values, HISTORY_SIZE_ENV, 8),
            history_multiplier=_finite_float(
                values,
                HISTORY_MULTIPLIER_ENV,
                1.15,
                minimum=1.0,
            ),
        )

    @property
    def completed_steps(self) -> int:
        return self._completed_steps

    @property
    def estimated_step_seconds(self) -> float:
        recent_estimate = max(self._durations) * self.history_multiplier if self._durations else 0.0
        return max(self.initial_step_seconds, recent_estimate)

    def decide(self, *, now_epoch: float | None = None) -> DeadlineDecision:
        now = time.time() if now_epoch is None else float(now_epoch)
        remaining = self.deadline_epoch - now
        estimate = self.estimated_step_seconds
        required = estimate + self.safety_seconds
        return DeadlineDecision(
            should_drain=remaining < required,
            now_epoch=now,
            deadline_epoch=self.deadline_epoch,
            remaining_seconds=remaining,
            estimated_step_seconds=estimate,
            safety_seconds=self.safety_seconds,
            required_seconds=required,
            completed_steps=self._completed_steps,
        )

    def record_completed_step(self, elapsed_seconds: float) -> None:
        elapsed = float(elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"Completed-step duration must be finite and non-negative, got {elapsed_seconds!r}")
        # The first update in each allocation starts from an empty async rollout
        # buffer and includes cold-start latency. Keep it out of the recent
        # steady-state timing window while still counting it as completed.
        if self._completed_steps > 0:
            self._durations.append(elapsed)
        self._completed_steps += 1

    def write_drain_marker(
        self,
        decision: DeadlineDecision,
        *,
        global_step: int,
    ) -> None:
        """Atomically publish why the launcher should start a successor."""

        payload = {
            "reason": "insufficient_time_for_complete_training_step",
            "global_step": int(global_step),
            "now_epoch": decision.now_epoch,
            "deadline_epoch": decision.deadline_epoch,
            "remaining_seconds": decision.remaining_seconds,
            "estimated_step_seconds": decision.estimated_step_seconds,
            "safety_seconds": decision.safety_seconds,
            "required_seconds": decision.required_seconds,
            "completed_steps_in_allocation": decision.completed_steps,
        }
        self.drain_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.drain_file.with_name(f".{self.drain_file.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.drain_file)
