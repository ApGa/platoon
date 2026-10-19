"""Training for Platoon.

`platoon.train.tinker` trains against a Tinker-API service (hosted Tinker or a
self-hosted server that implements the Tinker API, such as miles-tinker): the
trainer, its configuration and the rollout workflows live there.
"""

# Re-export common configurations for convenience
from platoon.config_defs import RolloutConfig

__all__ = ["RolloutConfig"]
