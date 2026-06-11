from __future__ import annotations

from dataclasses import dataclass, field

from platoon.openreward.config_defs import OpenRewardConfig
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig


@dataclass
class OpenRewardTinkerTrainerConfig(PlatoonTinkerRLTrainerConfig):
    openreward: OpenRewardConfig = field(default_factory=OpenRewardConfig)
    seed: int = 42

    def __post_init__(self):
        if isinstance(self.openreward, dict):
            self.openreward = OpenRewardConfig(**self.openreward)
