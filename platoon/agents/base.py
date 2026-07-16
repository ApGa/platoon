from __future__ import annotations

from typing import Protocol, runtime_checkable

from platoon.envs.base import Action, Observation, Task


@runtime_checkable
class Agent(Protocol):
    async def act(self, obs: Observation) -> Action: ...

    async def reset(self) -> None: ...

    async def close(self) -> None: ...


@runtime_checkable
class ForkableAgent(Agent, Protocol):
    async def fork(self, task: Task) -> ForkableAgent:
        """Return an independently closeable child agent.

        Implementations that allocate resources before returning must clean up
        partial allocations if the fork raises, including on cancellation.
        """

        ...
