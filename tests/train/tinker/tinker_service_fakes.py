"""A fake Tinker service for checkpoint tests: sessions, training clients that save checkpoints by path, and REST
clients that delete them, answering as a real service can.

``service.service_client()`` opens a session, as ``tinker.ServiceClient`` does, and each ``training_client()`` of it
is a new training run, whose checkpoints are ``tinker://run-<n>/weights/<name>`` and
``tinker://run-<n>/sampler_weights/<name>``.  The Tinker SDK reports a failed delete as the end of the session the
delete ran in, and a self-hosted service then releases that session's models: here a failed delete ends its session,
and a training client whose session ended fails its next training call.  ``make_pruning_trainer`` drives
``PlatoonTinkerRLTrainer.train`` on this service with tinker-cookbook's own checkpoint saving, so the run directory's
``checkpoints.jsonl`` is the one a real run writes.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import httpx
import tinker

from platoon.train.components import task_ids_to_dataset
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.trainer import rl as rl_module
from platoon.utils.stats_tracker import get as get_tracker


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


class _Future:
    def __init__(self, value):
        self.value = value

    async def result_async(self):
        return self.value


@dataclass
class FakeTinkerService:
    """Checkpoints by path, the sessions opened on it and every delete asked of it.

    ``tolerant``: deleting a checkpoint it does not have succeeds (miles-tinker); otherwise it answers 404, as hosted
    Tinker does.  ``failure(path)`` returns an error to answer a delete with instead, or None.  ``rest_deletes``:
    False gives REST clients without a delete method.  A delete waits for ``delete_gate`` when one is set.
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
        return self._save("weights", name)

    async def save_weights_for_sampler_async(self, name, ttl_seconds=None):
        return self._save("sampler_weights", name)

    def _save(self, kind: str, name: str) -> _Future:
        self._alive()
        path = f"tinker://{self.run_id}/{kind}/{name}"
        self.service.stored.add(path)
        return _Future(SimpleNamespace(path=path))

    def create_sampling_client(self, path):
        return f"sampler:{path}"

    async def forward_backward_async(self, datums, loss_fn, loss_fn_config):
        self._alive()
        outputs = [{"logprobs": item.loss_fn_inputs["target_tokens"]} for item in datums]
        return _Future(SimpleNamespace(loss_fn_outputs=outputs))

    async def optim_step_async(self, params):
        self._alive()
        self.optim_steps += 1
        return _Future(SimpleNamespace(metrics={}))

    async def save_weights_and_get_sampling_client_async(self):
        self._alive()
        return "weights-only-sampler"


class PolicyVersions:
    """The trainer's policy version, as ``ModelInfo.llm`` keeps it."""

    version = 0

    def set_version(self, version):
        self.version = version

    def update_sampling_client(self, client, increment_version=True):
        if increment_version:
            self.version += 1


def make_pruning_trainer(
    monkeypatch, config: PlatoonTinkerRLTrainerConfig, service: FakeTinkerService, *, tasks: int = 8
) -> tuple[Any, list[tuple[int, dict]]]:
    """``(trainer, logged)`` for one start of a run on ``service``: ``tinker.ServiceClient`` opens a session on it,
    the start trains a new training run, and ``logged`` collects each logged step with its stats."""

    logged: list[tuple[int, dict]] = []
    monkeypatch.setattr(rl_module.tinker, "ServiceClient", service.service_client)
    trainer: Any = object.__new__(rl_module.PlatoonTinkerRLTrainer)  # its fakes are not the attributes' types
    trainer.config = config
    trainer.train_dataset = task_ids_to_dataset([f"train:{index}" for index in range(tasks)])
    trainer.eval_dataset = None
    trainer.batch_transforms = []
    trainer._model_info = SimpleNamespace(llm=PolicyVersions())
    trainer._stats_logger = SimpleNamespace(
        wandb_run_id=None, log=lambda step, stats, force=False: logged.append((step, dict(stats)))
    )
    trainer._train_tracker = get_tracker("train")

    async def create_training_client(service_client, resume_info):
        return service_client.training_client()

    trainer._create_training_client = create_training_client
    os.makedirs(trainer.run_log_path, exist_ok=True)
    return trainer, logged
