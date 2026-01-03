"""Training backends for Platoon.

Platoon supports two training backends:
- `platoon.train.tinker`: Tinker-based training (service-based architecture)
- `platoon.train.areal`: AReaL-based training (distributed FSDP)

Each backend provides:
- Trainer class (e.g., PlatoonTinkerRLTrainer, PlatoonArealRLTrainer)
- Configuration classes
- Workflow implementations
"""

# Re-export common configurations for convenience
from platoon.config_defs import RolloutConfig

__all__ = ['RolloutConfig']
