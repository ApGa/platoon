from dataclasses import dataclass

from platoon.train.areal import PlatoonArealRLTrainerConfig


@dataclass
class OolongArealTrainerConfig(PlatoonArealRLTrainerConfig):
    recursive: bool = False
    seed: int = 42
    oolong_dataset: str = "synth"
    task_group: str | None = None
    answer_type: str | None = None
    min_context_len: int | None = None
    max_context_len: int | None = None
