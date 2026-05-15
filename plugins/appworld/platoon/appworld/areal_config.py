from dataclasses import dataclass

from platoon.train.areal import PlatoonArealRLTrainerConfig


@dataclass
class AppWorldArealTrainerConfig(PlatoonArealRLTrainerConfig):
    recursive: bool = False
    depth_aware: bool = False
