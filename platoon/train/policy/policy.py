"""One policy in the trainer process (:class:`Policy`): its spec, its state as training moves it on, and what each start
does with it: open its training client at the start point, save it whole (a checkpoint) or for its sampler, and take a
snapshot of its published sampler for inference outside the rollouts.

Rollout workers never hold one: they sample the sampling client that the trainer publishes for them
(:mod:`platoon.train.policy.binding`).
"""

import asyncio
from typing import TYPE_CHECKING, Any

import tinker
from tinker_cookbook import checkpoint_utils

from platoon.train.policy import service
from platoon.train.policy.checkpoints import Waiter, save_full_checkpoint, save_weights_only
from platoon.train.policy.snapshot import PolicySnapshot
from platoon.train.policy.spec import PolicyId, PolicySpec
from platoon.train.tinker.key_refusal import key_refusals
from platoon.train.tinker.proxy import ModelInfo

if TYPE_CHECKING:
    from platoon.train.checkpoints.resume import PolicyStart
    from platoon.train.config_defs import AdamParams


class Policy:
    """One policy: its spec, its model info (the renderer, the tokenizer and the model gateway's ``TinkerLLM``, which
    samples it and keeps its version), and what each start opens and moves on.  The training loops read and write the
    latter through their shared state, whose leader the policy is
    (:class:`~platoon.train.trainer.state.TrainLoopSharedState`), and a step updates it through its learner
    (:class:`~platoon.train.policy.learner.Learner`)."""

    # Set as a start opens the policy: its Tinker service client (connect), the training client on that service, the
    # sampling client of its newest weights (the launch save's, then each step's) and its optimizer's parameters.
    service_client: tinker.ServiceClient
    training_client: tinker.TrainingClient
    sampling_client: tinker.SamplingClient
    optimizer_params: tinker.AdamParams

    def __init__(self, spec: PolicySpec, info: ModelInfo) -> None:
        self.spec = spec
        self.info = info
        # The optimizer steps the policy took over the run, its earlier starts' among them.
        self.optimizer_steps = 0

    @property
    def id(self) -> PolicyId:
        return self.spec.id

    @property
    def version(self) -> int:
        """The policy's version: how many samplers it published over the run (its model gateway counts them), which a
        rollout's datums carry and the staleness of its groups is counted in."""

        return self.info.llm.version

    @property
    def renderer(self) -> Any:
        """The renderer of the policy's model (a ``tinker_cookbook`` renderer), which a step hook renders its own
        datums with."""

        return self.info.llm.renderer

    @property
    def tokenizer(self) -> Any:
        """The tokenizer of the policy's model."""

        return self.info.llm.tokenizer

    def connect(self) -> None:
        """Open the policy's Tinker service client, a session on its service (:meth:`new_service_client`).  A key the
        service refuses (401, 403) ends training with one line that names the key and the service."""

        with key_refusals():
            self.service_client = self.new_service_client()

    def new_service_client(self) -> tinker.ServiceClient:
        """A new session on the policy's Tinker service (``spec.tinker_base_url``): the one :meth:`connect` opens, and
        each one the run's recovery pruning deletes checkpoints in.  A test's policy opens them on a fake service."""

        return tinker.ServiceClient(base_url=self.spec.tinker_base_url)

    async def open(self, start: "PolicyStart", *, stop: asyncio.Event | None = None) -> bool:
        """Open the policy's training client on its service, at its start point: the one its checkpoint resumes
        (``start.record``: its weights and its optimizer's state), else a new run's (:meth:`create_training_client`),
        asked again after each refusal that asking again may change
        (:func:`~platoon.train.policy.service.open_training_client`); a policy that trains then has its optimizer's
        parameters (``optimizer_params``, from its spec's ``optimizer``).  False when ``stop`` was set before the
        service accepted it.  A key the service refuses (401, 403) ends training with one line that names the key and
        the service."""

        with key_refusals(asked="a training client"):
            training_client = await service.open_training_client(
                lambda: self.create_training_client(self.service_client, start.record), stop=stop
            )
        if training_client is None:
            return False
        self.training_client = training_client
        if self.spec.optimizer is not None:  # a policy that trains: its optimizer's parameters, from its spec
            self.optimizer_params = adam_params(self.spec.optimizer)
        return True

    async def create_training_client(
        self, service_client: tinker.ServiceClient, record: checkpoint_utils.CheckpointRecord | None
    ) -> tinker.TrainingClient:
        """Ask ``service_client`` once for the policy's training client: the one ``record`` resumes, else a new run's
        (:func:`~platoon.train.policy.service.create_training_client`)."""

        return await service.create_training_client(service_client, record, self.spec)

    async def save(self, name: str, *, log_path: str, loop_state: dict[str, Any], wait: Waiter) -> dict[str, str]:
        """Save full checkpoint ``name`` of the policy (its weights and its optimizer's state), which the
        ``checkpoints.jsonl`` in ``log_path`` records with ``loop_state``, awaited under ``wait``
        (:func:`~platoon.train.policy.checkpoints.save_full_checkpoint`).  The paths the save wrote."""

        return await save_full_checkpoint(
            self.training_client, name, log_path=log_path, loop_state=loop_state, wait=wait
        )

    async def save_for_sampler(self) -> tinker.SamplingClient:
        """The sampling client of the policy's weights as they are, from a weights-only save
        (:func:`~platoon.train.policy.checkpoints.save_weights_only`)."""

        return await save_weights_only(self.training_client)

    def snapshot(self) -> PolicySnapshot:
        """The policy's published sampler as it is now, at its version, with its renderer, tokenizer, model name,
        context window and sample timeout (:class:`~platoon.train.policy.snapshot.PolicySnapshot`), whose samples
        are refused once the policy publishes again."""

        llm = self.info.llm
        return PolicySnapshot(
            policy=self.id,
            version=self.version,
            sampling_client=self.sampling_client,
            renderer=self.renderer,
            tokenizer=self.tokenizer,
            model_name=llm.model_name,
            context_window_length=llm.context_window_length,
            sample_timeout_seconds=llm.sample_timeout_seconds,
            version_now=lambda: self.version,
        )


def adam_params(optimizer: "AdamParams") -> tinker.AdamParams:
    """The parameters a policy's optimizer steps with, as the Tinker service takes them, from its Adam settings (a
    spec's ``optimizer``: ``train.optimizer``)."""

    return tinker.AdamParams(
        learning_rate=optimizer.learning_rate,
        beta1=optimizer.beta1,
        beta2=optimizer.beta2,
        eps=optimizer.eps,
        weight_decay=optimizer.weight_decay,
        grad_clip_norm=optimizer.grad_clip_norm,
    )
