"""Controller-only PPO actor RPC surface.

This module deliberately avoids importing :mod:`platoon.train.areal.actor` or
any AReaL engine implementation. A PBS single controller needs only the RPC
controller plus a lightweight class reference whose module/name is resolved by
the worker inside its training container.
"""

from typing import Any

from areal.trainer.ppo.actor import PPOActorController

from platoon.train.areal.numerical_stability import (
    aggregate_optimizer_update_results,
)


class PlatoonPPOActorController(PPOActorController):
    """Actor controller exposing Platoon's worker-side memory hygiene RPC."""

    def ppo_update(self, *args: Any, **kwargs: Any) -> bool:
        results = self._custom_function_call(
            "ppo_update",
            *args,
            rpc_meta={"broadcast": True},
            **kwargs,
        )
        return aggregate_optimizer_update_results(results)

    def clear_device_cache(self) -> None:
        self._custom_function_call("clear_device_cache")
