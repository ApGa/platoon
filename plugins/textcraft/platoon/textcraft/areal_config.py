from dataclasses import dataclass

from platoon.train.areal import PlatoonArealRLTrainerConfig


@dataclass
class TextCraftSynthArealTrainerConfig(PlatoonArealRLTrainerConfig):
    train_difficulties: list[str] | None = None
    eval_difficulties: list[str] | None = None
    recursive: bool = False
    depth_aware: bool = False
