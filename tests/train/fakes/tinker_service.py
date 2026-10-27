"""A fake Tinker service for checkpoint tests: sessions, training clients that save checkpoints by path, and REST
clients that delete them, answering as a real service can.

``service.service_client()`` opens a session, as ``tinker.ServiceClient`` does, and each ``training_client()`` of it
is a new training run, whose checkpoints are ``tinker://run-<n>/weights/<name>`` and
``tinker://run-<n>/sampler_weights/<name>``.  The Tinker SDK reports a failed delete as the end of the session the
delete ran in, and a self-hosted service then releases that session's models: here a failed delete ends its session,
and a training client whose session ended fails its next training call.  :func:`make_service_trainer` builds the
trainer with its constructor, over a policy whose sessions are this service's (``fakes.trainer.FakePolicy``), which
saves through tinker-cookbook's own checkpoint saving, so the run directory's ``checkpoints.jsonl`` is the one a real
run writes; the runs that save, prune and resume checkpoints use it.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx
import tinker

from fakes.trainer import FakePolicy, Future, StepLog, enter
from platoon.train.components import task_ids_to_dataset
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.policy.set import PolicySet
from platoon.train.trainer.rl import PlatoonTinkerRLTrainer


def status_error(status: int, message: str) -> tinker.APIStatusError:
    """The error the Tinker SDK raises for an HTTP ``status`` answer to a checkpoint delete."""

    request = httpx.Request("DELETE", "http://tinker.invalid/api/v1/training_runs/run/checkpoints/weights/x")
    response = httpx.Response(status, request=request)
    error_class = {404: tinker.NotFoundError, 500: tinker.InternalServerError}.get(status, tinker.APIStatusError)
    return error_class(message, response=response, body=None)


@dataclass
class FakeSession:
    id: int
    ended: bool = False


@dataclass
class FakeTinkerService:
    """Checkpoints by path, the sessions opened on it and every delete asked of it.

    ``tolerant``: deleting a checkpoint it does not have succeeds (miles-tinker); otherwise it answers 404, as hosted
    Tinker does.  ``failure(path)`` returns an error to answer a delete with instead, or None.  ``rest_deletes``:
    False gives REST clients without a delete method.  A delete waits for ``delete_gate`` when one is set.  A training
    client's ``forward_backward`` first awaits ``on_forward_backward(datums)``, its ``optim_step``
    ``on_optim_step(client)`` (before the step counts) and its ``save_state`` ``on_save_state(name)``, when the test
    sets them: what it records, or the fault it raises.
    """

    tolerant: bool = False
    failure: Callable[[str], BaseException | None] | None = None
    rest_deletes: bool = True
    delete_gate: asyncio.Event | None = None
    stored: set[str] = field(default_factory=set)
    delete_calls: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    sessions: list[FakeSession] = field(default_factory=list)
    runs: int = 0
    on_forward_backward: Callable[[list], Awaitable[None]] | None = None
    on_optim_step: Callable[[FakeTrainingClient], Awaitable[None]] | None = None
    on_save_state: Callable[[str], Awaitable[None]] | None = None

    def service_client(self, **kwargs) -> FakeServiceClient:
        session = FakeSession(len(self.sessions))
        self.sessions.append(session)
        return FakeServiceClient(self, session)

    def paths(self, run: int, *names: str) -> set[str]:
        """The state and sampler paths of checkpoints ``names`` of training run ``run``."""

        return {f"tinker://run-{run}/{kind}/{name}" for name in names for kind in ("weights", "sampler_weights")}


@dataclass
class FakeServiceClient:
    service: FakeTinkerService
    session: FakeSession

    def create_rest_client(self):
        if not self.service.rest_deletes:
            return SimpleNamespace(list_checkpoints_async=None)
        return FakeRestClient(self.service, self.session)

    def training_client(self) -> FakeTrainingClient:
        self.service.runs += 1
        return FakeTrainingClient(self.service, self.session, f"run-{self.service.runs}")

    # A start's training client, new or from a checkpoint's state: each one a new training run.

    async def create_lora_training_client_async(self, base_model: str, rank: int = 32, **kwargs) -> FakeTrainingClient:
        return self.training_client()

    async def create_training_client_from_state_with_optimizer_async(self, path: str, **kwargs) -> FakeTrainingClient:
        return self.training_client()

    async def create_training_client_from_state_async(self, path: str, **kwargs) -> FakeTrainingClient:
        return self.training_client()


@dataclass
class FakeRestClient:
    service: FakeTinkerService
    session: FakeSession

    async def delete_checkpoint_from_tinker_path_async(self, tinker_path: str) -> None:
        service = self.service
        service.delete_calls.append(tinker_path)
        if service.delete_gate is not None:
            await service.delete_gate.wait()
        error = service.failure(tinker_path) if service.failure is not None else None
        if error is None and tinker_path not in service.stored and not service.tolerant:
            error = status_error(404, f"checkpoint {tinker_path} not found")
        if error is not None:
            self.session.ended = True  # what the SDK's fatal telemetry has a self-hosted service do
            raise error
        service.stored.discard(tinker_path)
        service.deleted.append(tinker_path)


@dataclass
class FakeTrainingClient:
    """Saves checkpoints into the service; every update needs its session alive."""

    service: FakeTinkerService
    session: FakeSession
    run_id: str
    optim_steps: int = 0

    def _alive(self) -> None:
        if self.session.ended:
            raise RuntimeError(f"session {self.session.id} ended: the service released its models")

    async def save_state_async(self, name, ttl_seconds=None):
        if self.service.on_save_state is not None:
            await self.service.on_save_state(name)
        return self._save("weights", name)

    async def save_weights_for_sampler_async(self, name, ttl_seconds=None):
        return self._save("sampler_weights", name)

    def _save(self, kind: str, name: str) -> Future:
        self._alive()
        path = f"tinker://{self.run_id}/{kind}/{name}"
        self.service.stored.add(path)
        return Future(SimpleNamespace(path=path))

    def create_sampling_client(self, path):
        return f"sampler:{path}"

    async def forward_backward_async(self, datums, loss_fn, loss_fn_config):
        if self.service.on_forward_backward is not None:
            await self.service.on_forward_backward(datums)
        self._alive()
        outputs = [{"logprobs": item.loss_fn_inputs["target_tokens"]} for item in datums]
        return Future(SimpleNamespace(loss_fn_outputs=outputs))

    async def optim_step_async(self, params):
        if self.service.on_optim_step is not None:
            await self.service.on_optim_step(self)
        self._alive()
        self.optim_steps += 1
        return Future(SimpleNamespace(metrics={}))

    async def save_weights_and_get_sampling_client_async(self):
        self._alive()
        return "weights-only-sampler"


def make_service_trainer(
    config: PlatoonTinkerRLTrainerConfig, service: FakeTinkerService, *, tasks: int = 8, **trainer_options: Any
) -> tuple[PlatoonTinkerRLTrainer, StepLog]:
    """``(trainer, logged)`` for one start of a run on ``service``: each session the trainer opens is a session on it,
    the start trains a new training run there and saves its checkpoints there through tinker-cookbook, which writes the
    run directory's ``checkpoints.jsonl`` as a real run's saves do; ``logged`` records the steps it logs
    (``fakes.trainer.StepLog``).  ``trainer_options`` are the constructor's other arguments (``train_dataset`` in place
    of the ``tasks``, ...)."""

    logged = StepLog()
    options = {"train_dataset": task_ids_to_dataset([f"train:{index}" for index in range(tasks)]), **trainer_options}
    policy = FakePolicy(config, service.service_client)
    trainer = PlatoonTinkerRLTrainer(config, policies=PolicySet([policy]), **options)
    enter(trainer, logged)
    os.makedirs(trainer.run_log_path, exist_ok=True)
    return trainer, logged
