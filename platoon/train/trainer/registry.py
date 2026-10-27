"""Which trainer a run trains with: the config's ``trainer:`` (:data:`CORE_TRAINERS`), with its ``trainer_options``,
checked when the config loads (:func:`trainer_problems`, exit 78 before anything runs).

This release has one trainer, ``rl`` (the default), which platoon train builds from the config.  Its settings are
``train:``'s; its one option is the step hooks it attaches to each of its steps (:class:`RLOptions`).  Core's other
names are reserved for the trainers platoon adds; a trainer that a package writes (RFC 0021) is selectable once platoon
resolves those names.  Imports no Tinker.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from platoon.options import schema_options
from platoon.train.trainer.hook_config import HookSpec, hook_specs

# Core's trainers, by the name a config gives them: rl is :class:`~platoon.train.trainer.rl.PlatoonTinkerRLTrainer`.
CORE_TRAINERS = ("rl",)
# Core's names of the trainers platoon adds: reserved, so that no config or package takes them meanwhile.
RESERVED_TRAINERS = ("sft", "interleaved")


@dataclass(frozen=True)
class RLOptions:
    """The rl trainer's options (``trainer_options``): the step hooks it attaches to each of its steps (``hooks``, RFC
    0021 section 9: :mod:`platoon.train.trainer.hook_config`).  Its other settings are ``train:``'s."""

    hooks: tuple[HookSpec, ...] = ()

    @classmethod
    def parse(cls, values: Mapping[str, Any]) -> "RLOptions":
        """The options of ``values``, ``trainer_options`` as the config gives them; a ValueError names what is
        wrong."""

        if unknown := sorted(set(values) - {"hooks"}):
            raise ValueError(f"the rl trainer takes hooks, not {', '.join(unknown)}: its settings are train:'s")
        return cls(hooks=hook_specs(values.get("hooks") or []))


def trainer_problems(config: Any) -> list[str]:
    """What the config's ``trainer:`` and ``trainer_options`` refuse, one entry each: a name that is not one of
    :data:`CORE_TRAINERS`, and options the trainer does not take."""

    name = getattr(config, "trainer", "rl")
    if name in CORE_TRAINERS:
        return _rl_options().option_problems(getattr(config, "trainer_options", None), "trainer_options")
    if name in RESERVED_TRAINERS:
        return [
            f"trainer: {name!r} is reserved for a trainer platoon will add; this release trains with rl (the default)"
        ]
    return [f"trainer: {name!r} is not a trainer: this release has one, rl (the default)"]


def trainer_options(values: Mapping[str, Any] | None) -> RLOptions:
    """``values``, the rl trainer's ``trainer_options`` as a config gives them, parsed (a ``ConfigError`` with each
    problem)."""

    return _rl_options().parse(values, "trainer_options")


def _rl_options() -> Any:
    return schema_options(RLOptions.parse, name="trainer: rl")
