from dataclasses import dataclass, field

import torch
from areal.api.cli_args import InferenceEngineConfig


@dataclass
class VariableBatchInferenceEngineConfig(InferenceEngineConfig):
    shuffle_cross_task: bool = field(default=False)
    ensure_batch_divisible_by: int = field(default=1)


def set_expandable_segments(enable: bool) -> None:
    """Enable or disable expandable segments for cuda.
    Args:
        enable (bool): Whether to enable expandable segments. Used to avoid OOM.
    """
    torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")
