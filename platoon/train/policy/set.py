"""The run's policies (:class:`PolicySet`), built where the trainer is, each with its model gateway, and opened as each
start begins."""

import asyncio
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING

from platoon.train.policy.policy import Policy
from platoon.train.policy.spec import PolicyId, PolicySpec
from platoon.train.tinker.proxy import register_tinker_llm

if TYPE_CHECKING:
    from platoon.train.checkpoints.resume import StartPoint


class PolicySet(Mapping[PolicyId, Policy]):
    """The run's policies, by id, in the config's order.  The leader (``leader``) is the first: its rows of the run's
    ``checkpoints.jsonl`` are the run's checkpoints."""

    def __init__(self, policies: Iterable[Policy]) -> None:
        self._policies = {policy.id: policy for policy in policies}
        if not self._policies:
            raise ValueError("a run has at least one policy")
        self.leader: PolicyId = next(iter(self._policies))

    @classmethod
    def build(cls, specs: Mapping[PolicyId, PolicySpec]) -> "PolicySet":
        """The run's one policy (``main``, the one ``specs`` holds), sampled through a model gateway on its service,
        which :func:`~platoon.train.tinker.proxy.register_tinker_llm` makes this process's default: the gateway that
        training's and evaluation's rollouts sample.

        More than one policy is refused: registering each would make each the default in turn, the last one, not the
        leader.  A run that samples several policies registers its agents' default policy only and adds the others to
        its gateway."""

        if len(specs) != 1:
            raise ValueError(f"a run has one policy, main, not {len(specs)}: {', '.join(specs) or 'none'}")
        (spec,) = specs.values()
        info = register_tinker_llm(
            spec.model_name,
            spec.renderer_name,
            context_window_length=spec.context_window_length,
            renderer_kwargs=spec.renderer_kwargs,
            base_url=spec.tinker_base_url,
            # The bound on each sampling call, resolved here once (the key, else TINKER_SAMPLE_TIMEOUT_SECONDS), which
            # rollout workers take from the model binding.
            sample_timeout_seconds=spec.sample_timeout_seconds,
        )
        return cls([Policy(spec, info)])

    def versions(self) -> dict[PolicyId, int]:
        """Each policy's version, by id (:attr:`Policy.version`): ``{main: v}`` for a run's one policy, what the
        dispatcher's task samplers are told the policies are at."""

        return {policy.id: policy.version for policy in self._policies.values()}

    def connect(self) -> None:
        """Open each policy's Tinker service client (:meth:`Policy.connect`)."""

        for policy in self._policies.values():
            policy.connect()

    async def open(self, start: "StartPoint", *, stop: asyncio.Event | None = None) -> bool:
        """Open each policy's training client at its start point (``start.policies``, :meth:`Policy.open`), one after
        another; False when ``stop`` was set before the service accepted one.  A run has one policy until several
        train, which will open concurrently then."""

        for policy in self._policies.values():
            if not await policy.open(start.policies[policy.id], stop=stop):
                return False
        return True

    def __getitem__(self, policy: PolicyId) -> Policy:
        return self._policies[policy]

    def __iter__(self) -> Iterator[PolicyId]:
        return iter(self._policies)

    def __len__(self) -> int:
        return len(self._policies)
