from __future__ import annotations

from dataclasses import dataclass, field

from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig

from platoon.openreward.config_defs import OpenRewardConfig


@dataclass
class OpenRewardTinkerTrainerConfig(PlatoonTinkerRLTrainerConfig):
    openreward: OpenRewardConfig = field(default_factory=OpenRewardConfig)
    seed: int = 42

    def __post_init__(self):
        if isinstance(self.openreward, dict):
            self.openreward = OpenRewardConfig(**self.openreward)
        if any(
            environment.sampling_start_step > 0
            for environment in self.openreward.resolved_environments()
        ):
            raise ValueError(
                "Staged OpenReward environment admission is currently supported "
                "only by the AReaL trainer"
            )
