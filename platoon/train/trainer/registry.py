"""Which trainer a run trains with: the config's ``trainer:`` (:data:`CORE_TRAINERS`), checked when the config loads
(:func:`trainer_problems`, exit 78 before anything runs).

This release has one trainer, ``rl`` (the default): platoon train builds it from the config, as it always did.  Core's
other names are reserved for the trainers platoon adds; a trainer that a package writes (RFC 0021) is selectable once
platoon resolves those names.  Imports no Tinker.
"""

from typing import Any

# Core's trainers, by the name a config gives them: rl is :class:`~platoon.train.trainer.rl.PlatoonTinkerRLTrainer`.
CORE_TRAINERS = ("rl",)
# Core's names of the trainers platoon adds: reserved, so that no config or package takes them meanwhile.
RESERVED_TRAINERS = ("sft", "interleaved")


def trainer_problems(config: Any) -> list[str]:
    """What the config's ``trainer:`` refuses, one entry each: a name that is not one of :data:`CORE_TRAINERS`."""

    name = getattr(config, "trainer", "rl")
    if name in CORE_TRAINERS:
        return []
    if name in RESERVED_TRAINERS:
        return [
            f"trainer: {name!r} is reserved for a trainer platoon will add; this release trains with rl (the default)"
        ]
    return [f"trainer: {name!r} is not a trainer: this release has one, rl (the default)"]
