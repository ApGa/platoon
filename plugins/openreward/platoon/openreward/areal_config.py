from __future__ import annotations

from dataclasses import dataclass, field

from platoon.openreward.config_defs import OpenRewardConfig
from platoon.train.areal import PlatoonArealRLTrainerConfig


@dataclass
class OpenRewardArealTrainerConfig(PlatoonArealRLTrainerConfig):
    openreward: OpenRewardConfig = field(default_factory=OpenRewardConfig)

    def __post_init__(self):
        if isinstance(self.openreward, dict):
            self.openreward = OpenRewardConfig(**self.openreward)
        super().__post_init__()
