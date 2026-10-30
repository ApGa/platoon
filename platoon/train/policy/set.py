"""The run's policies (:class:`PolicySet`), built where the trainer is, each with its model gateway, and opened as each
start begins; and the work each of several policies does at once, on its own service (:meth:`PolicySet.each`).

The agents' default policy (``agents.policy``, ``main`` without it) is registered as this process's default model
gateway, as a run's one policy always was; every other policy gets a gateway of its own, on a Tinker session of its own
(one session per policy, also when policies share a service: a fatal client error ends the session it happened on,
and a service may then reclaim every model of that session), which the default gateway serves the agents bound to that
policy with (``ModelGateway.serve_policies``).
"""

import asyncio
import copy
from collections.abc import Awaitable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, TypeVar

from platoon.train.policy.policy import Policy
from platoon.train.policy.spec import AgentBindings, PolicyId, PolicySpec, failure_note, service_name
from platoon.train.tinker.proxy import ModelInfo, open_tinker_llm, register_tinker_llm
from platoon.utils.config import ConfigError

if TYPE_CHECKING:
    from platoon.train.checkpoints.resume import StartPoint

T = TypeVar("T")


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

        A run of one policy registers it with today's call, and nothing more."""

        if not specs:
            raise ValueError("a run has at least one policy")
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

    def led_by(self, policy: PolicyId) -> "PolicySet":
        """The same policies, with ``policy`` as their leader: what the steps of one policy that steps on its own clock
        read their policy through (its view of the loops' shared state, :mod:`platoon.train.step.independent`), since
        the lockstep step reads the policy it trains first as the leader.  The run's leader is unchanged."""

        view = copy.copy(self)
        view.leader = policy
        return view

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
        :meth:`Policy.open`), all at once, each on its own service (:meth:`each`); False when ``stop`` was set before
        a service accepted one."""

        opened = await self.each(
            {policy.id: policy.open(start.policies[policy.id], stop=stop) for policy in self.trainable()},
            doing="opening its training client",
        )
        return all(opened.values())

    async def each(self, work: Mapping[PolicyId, Awaitable[T]], *, doing: str) -> dict[PolicyId, T]:
        """Await each policy's ``work`` at the same time, each on its own service: what each one's came to, by policy,
        in the order of ``work``.  Once one fails the others' work is cancelled, and the error of the first in
        ``work``'s order that failed (not the first to fail in time) is raised as it is, in a run of several trained
        policies with a note that names its policy, its service and what it was ``doing``
        (:func:`~platoon.train.policy.spec.failure_note`): training stops when a policy's service fails, as it does for
        a run's one policy.  One policy's work is awaited alone, as it always was."""

        if len(work) == 1:
            ((policy, awaitable),) = work.items()
            try:
                return {policy: await awaitable}
            except Exception as error:
                if len(self.trainable()) > 1:  # one of several policies' work (another's save): say whose failed
                    error.add_note(failure_note(self[policy].spec, doing))
                raise
        tasks = {policy: asyncio.ensure_future(awaitable) for policy, awaitable in work.items()}
        try:
            await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_EXCEPTION)
        finally:
            for task in tasks.values():  # after a failure, or when the caller is cancelled: the others' work stops
                task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)
        for policy, task in tasks.items():
            if not task.cancelled() and (error := task.exception()) is not None:
                error.add_note(failure_note(self[policy].spec, doing))
                raise error
        return {policy: task.result() for policy, task in tasks.items()}

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

    service = service_name(spec)
    model = spec.load_checkpoint_path if not spec.trainable and spec.load_checkpoint_path else spec.model_name
    return RuntimeError(
        f"policy {spec.id}: could not open a sampling client of {model} on {service}: {type(error).__name__}: {error}"
    )
