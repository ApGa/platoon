"""End-to-end wall-clock telemetry between applied policy updates."""

from __future__ import annotations

import math
from dataclasses import dataclass

END_TO_END_UPDATE_INTERVAL = "timing/end_to_end_update_interval"
END_TO_END_UPDATE_INTERVAL_AVAILABLE = f"{END_TO_END_UPDATE_INTERVAL}_available"
END_TO_END_UPDATE_INTERVAL_FIRST_COMPLETION = f"{END_TO_END_UPDATE_INTERVAL}_first_completion"
APPLIED_UPDATE_COMPLETED = "timing/applied_update_completed"


@dataclass
class InterAppliedUpdateWallClock:
    """Track monotonic completion-to-completion policy-update latency.

    A completion is recorded only after the new actor weights have been
    successfully published to rollout workers. Logical steps that skip the
    optimizer or weight synchronization do not advance the boundary, so the
    next duration includes all intervening controller work and rollout waits.

    The first completion in a ``train()`` invocation has no invocation-local monotonic
    predecessor. Its duration is therefore omitted instead of being reported
    as a misleading zero; the companion availability and first-completion
    metrics make that case explicit. A later ``train()`` call (including after
    recovery) starts a fresh series; scheduler/process downtime is deliberately
    excluded from steady-state training throughput.
    """

    previous_completion: float | None = None

    def metrics_for_step(self, completed_at: float | None) -> dict[str, float]:
        """Return step metrics and advance the boundary for a real completion."""

        metrics = {
            APPLIED_UPDATE_COMPLETED: float(completed_at is not None),
            END_TO_END_UPDATE_INTERVAL_AVAILABLE: 0.0,
            END_TO_END_UPDATE_INTERVAL_FIRST_COMPLETION: 0.0,
        }
        if completed_at is None:
            return metrics

        completed_at = float(completed_at)
        if not math.isfinite(completed_at):
            raise ValueError(
                "Applied-update completion timestamp must be finite, "
                f"got {completed_at!r}"
            )

        previous = self.previous_completion
        if previous is None:
            self.previous_completion = completed_at
            metrics[END_TO_END_UPDATE_INTERVAL_FIRST_COMPLETION] = 1.0
            return metrics

        elapsed = completed_at - previous
        if elapsed < 0:
            raise ValueError(
                "Applied-update completion timestamps must be monotonic: "
                f"previous={previous!r}, current={completed_at!r}"
            )

        self.previous_completion = completed_at
        metrics[END_TO_END_UPDATE_INTERVAL] = elapsed
        metrics[END_TO_END_UPDATE_INTERVAL_AVAILABLE] = 1.0
        return metrics
