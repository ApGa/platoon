"""The run's policies (:class:`PolicySet`), built where the trainer is, each with its model gateway, and opened as each
start begins.

The agents' default policy (``agents.policy``, ``main`` without it) is registered as this process's default model
gateway, as a run's one policy always was; every other policy gets a gateway of its own, on a Tinker session of its own
(one session per policy, also when policies share a service: a fatal client error ends the session it happened on,
and a service may then reclaim every model of that session), which the default gateway serves the agents bound to that
policy with (``ModelGateway.serve_policies``).
"""

import asyncio
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING

from platoon.train.policy.policy import Policy
from platoon.train.policy.spec import AgentBindings, PolicyId, PolicySpec
from platoon.train.tinker.proxy import ModelInfo, open_tinker_llm, register_tinker_llm
from platoon.utils.config import ConfigError

if TYPE_CHECKING:
    from platoon.train.checkpoints.resume import StartPoint


class PolicySet(Mapping[PolicyId, Policy]):
    """The run's policies, by id, in the config's order, and which policy each agent samples (``agents``).  The
    leader (``leader``) is the first policy the run trains: its rows of the run's ``checkpoints.jsonl`` are the run's
    checkpoints."""

    def __init__(self, policies: Iterable[Policy], agents: AgentBindings | None = None) -> None:
        self._policies = {policy.id: policy for policy in policies}
        if not self._policies:
            raise ValueError("a run has at least one policy")
        first = next(iter(self._policies))
        self.leader: PolicyId = next((policy.id for policy in self.trainable()), first)
        self.agents = agents if agents is not None else AgentBindings(default=first)

    @classmethod
    def build(cls, specs: Mapping[PolicyId, PolicySpec], agents: AgentBindings | None = None) -> "PolicySet":
        """The run's policies, each sampled through a model gateway on its service: the agents' default policy's
        (``agents.default``; the first policy without ``agents``) registered by
        :func:`~platoon.train.tinker.proxy.register_tinker_llm` as this process's default, the gateway that training's
        and evaluation's rollouts sample, and every other policy's opened in a Tinker session of its own and served by
        that gateway to the agents bound to it (a policy that only scores, a teacher, is not).  A frozen policy samples
        the sampler its ``load_checkpoint_path`` names.

        A run of one policy registers it with today's call, and nothing more.  More than one policy to train is
        refused: loading refuses it first, since this release trains one."""

        if not specs:
            raise ValueError("a run has at least one policy")
        if len(trained := [spec.id for spec in specs.values() if spec.trainable]) > 1:
            raise ValueError(f"a run trains one policy, not {len(trained)}: {', '.join(trained)}")
        agents = agents if agents is not None else AgentBindings(default=next(iter(specs)))
        infos = {agents.default: _registered(specs[agents.default])}
        for spec in specs.values():
            if spec.id != agents.default:
                infos[spec.id] = _opened(spec)
        if len(sampled := agents.sampled()) > 1:  # agents of some role sample another policy than the default
            infos[agents.default].llm.gateway.serve_policies(
                agents.default,
                runtimes={policy: infos[policy].llm.gateway for policy in sampled[1:]},
                roles=agents.roles,
            )
        return cls([Policy(spec, infos[spec.id]) for spec in specs.values()], agents)

    def trainable(self) -> list[Policy]:
        """The policies the run trains, in order."""

        return [policy for policy in self._policies.values() if policy.trainable]

    def sampled(self) -> list[Policy]:
        """The policies some agent samples (:meth:`AgentBindings.sampled`), the agents' default first: those whose
        samplers rollout workers follow."""

        return [self._policies[policy] for policy in self.agents.sampled()]

    def versions(self) -> dict[PolicyId, int]:
        """Each policy's version, by id (:attr:`Policy.version`): ``{main: v}`` for a run's one policy, what the
        dispatcher's task samplers are told the policies are at."""

        return {policy.id: policy.version for policy in self._policies.values()}

    def connect(self) -> None:
        """Open the Tinker service client of each policy the run trains (:meth:`Policy.connect`); a frozen policy's
        session opened as it was built."""

        for policy in self.trainable():
            policy.connect()

    async def open(self, start: "StartPoint", *, stop: asyncio.Event | None = None) -> bool:
        """Open the training client of each policy the run trains at its start point (``start.policies``,
        :meth:`Policy.open`), one after another; False when ``stop`` was set before the service accepted one.  A run
        trains one policy until several train, which will open concurrently then."""

        for policy in self.trainable():
            if not await policy.open(start.policies[policy.id], stop=stop):
                return False
        return True

    def __getitem__(self, policy: PolicyId) -> Policy:
        return self._policies[policy]

    def __iter__(self) -> Iterator[PolicyId]:
        return iter(self._policies)

    def __len__(self) -> int:
        return len(self._policies)


def _registered(spec: PolicySpec) -> ModelInfo:
    """The agents' default policy's model info, its gateway this process's default: a policy the run trains samples its
    base model until the launch save publishes its weights, and a frozen one the sampler it serves.  It is opened
    first, with today's call and today's errors (a run of one policy fails as it always did)."""

    # A trained default passes no model_path, so that its call stays today's, argument for argument.
    frozen = {} if spec.trainable else {"model_path": spec.load_checkpoint_path}
    return register_tinker_llm(
        spec.model_name,
        spec.renderer_name,
        context_window_length=spec.context_window_length,
        renderer_kwargs=spec.renderer_kwargs,
        base_url=spec.tinker_base_url,
        # The bound on each sampling call, resolved here once (the key, else TINKER_SAMPLE_TIMEOUT_SECONDS), which
        # rollout workers take from the model binding.
        sample_timeout_seconds=spec.sample_timeout_seconds,
        **frozen,
    )


def _opened(spec: PolicySpec) -> ModelInfo:
    """Another policy's model info, on a Tinker session of its own on its service (its gateway is not the process's
    default): a frozen policy samples the sampler it serves, one the run trains its base model until the launch.  An
    error opening it names the policy and its service (:func:`_not_opened`), so that a run of several services says
    which one failed."""

    try:
        return open_tinker_llm(
            spec.model_name,
            spec.renderer_name,
            context_window_length=spec.context_window_length,
            renderer_kwargs=spec.renderer_kwargs,
            model_path=None if spec.trainable else spec.load_checkpoint_path,
            base_url=spec.tinker_base_url,
            sample_timeout_seconds=spec.sample_timeout_seconds,
        )
    except ConfigError:
        raise
    except Exception as exc:
        raise _not_opened(spec, exc) from exc


def _not_opened(spec: PolicySpec, error: Exception) -> RuntimeError:
    """The error of a policy whose sampling client did not open (``error``), naming the policy, what it samples and
    its service.  A key the service refuses stays the configuration error it is (exit 78), which names the service
    already."""

    service = spec.tinker_base_url or "the service TINKER_BASE_URL names, else hosted Tinker"
    model = spec.load_checkpoint_path if not spec.trainable and spec.load_checkpoint_path else spec.model_name
    return RuntimeError(
        f"policy {spec.id}: could not open a sampling client of {model} on {service}: {type(error).__name__}: {error}"
    )
