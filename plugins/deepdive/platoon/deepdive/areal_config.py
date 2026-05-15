from dataclasses import dataclass

from platoon.train.areal import PlatoonArealRLTrainerConfig


@dataclass
class DeepDiveArealTrainerConfig(PlatoonArealRLTrainerConfig):
    recursive: bool = True
    train_split: str = "qa_rl"
    eval_split: str = "qa_sft"
    train_num_tasks: int | None = None
    eval_num_tasks: int = 100
    seed: int = 42
