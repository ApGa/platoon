"""Deterministic datum-level sampling for recursive-agent training."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol


class SubagentDatumSampler(Protocol):
    """Backend-neutral interface for sampling post-merge trajectory datums."""

    def sample_mask(
        self,
        *,
        task_id: str,
        trajectory_id: str,
        depth: int,
        num_datums: int,
    ) -> list[bool]: ...


@dataclass(frozen=True)
class DeterministicSubagentDatumSampler:
    """Independently retain non-root datums using stable SHA-256 draws.

    A hash-based draw makes the decision independent of worker scheduling,
    global RNG state, and iteration order.  Root datums are mandatory; every
    non-root datum receives its own Bernoulli draw and a trajectory may
    therefore contribute zero retained datums.
    """

    keep_probability: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        probability = float(self.keep_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("subagent datum keep probability must be in [0, 1]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("subagent datum sampling seed must be an integer")

    def sample_mask(
        self,
        *,
        task_id: str,
        trajectory_id: str,
        depth: int,
        num_datums: int,
    ) -> list[bool]:
        if depth < 0:
            raise ValueError("trajectory depth must be non-negative")
        if num_datums < 0:
            raise ValueError("number of datums must be non-negative")
        if depth == 0 or self.keep_probability >= 1.0:
            return [True] * num_datums
        if self.keep_probability <= 0.0:
            return [False] * num_datums

        # Comparing integer draws avoids platform-dependent float conversion at
        # the selection boundary.  SHA-256 gives each datum a stable, independent
        # 64-bit draw for practical purposes.
        cutoff = int(self.keep_probability * (1 << 64))
        prefix = f"{self.seed}\0{task_id}\0{trajectory_id}\0{depth}\0".encode()
        retained: list[bool] = []
        for datum_index in range(num_datums):
            digest = hashlib.sha256(prefix + str(datum_index).encode()).digest()
            draw = int.from_bytes(digest[:8], byteorder="big", signed=False)
            retained.append(draw < cutoff)
        return retained
