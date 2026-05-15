from dataclasses import dataclass

from platoon.train.areal import PlatoonArealRLTrainerConfig


@dataclass
class EmailSearchArealTrainerConfig(PlatoonArealRLTrainerConfig):
    recursive: bool = True
    train_split: str = "train"
    eval_split: str = "test"
    train_num_tasks: int | None = None
    eval_num_tasks: int | None = 100
    max_messages: int | None = 1
    exclude_known_bad_queries: bool = True
    seed: int = 42
