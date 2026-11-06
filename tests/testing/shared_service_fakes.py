"""A Tinker service on CPU whose LoRA adapters learn, for the tests of ``platoon.testing.check_shared_service``
(``test_shared_service_check.py``): ``ToyService``, a session of which each ``tinker.ServiceClient`` opens inside
``toy_service(service)``, and the faults (``FAULTS``), each of which breaks what one of RFC 0021's U1 to U4 asks.

A model's weights are an offset per token id, 0 for the base model, and the logprob it gives a token is
``BASE_LOGPROB + GAIN * offset``, whatever comes before the token.  ``forward_backward`` adds ``-GAIN * weight`` to the
gradient of each target token (``cross_entropy``'s weighted negative log-likelihood); ``optim_step`` moves the offset of
every token with a gradient or an optimizer state by AdamW, as a LoRA's parameters, which all take part in every step,
move, and clears the gradients.  A save keeps the offsets (a state with its optimizer's), and a sampler serves the
offsets it was made of.  A session ends at its session-end event (its telemetry's ``log_fatal_exception``), as a
self-hosted service ends one, and releases its models then: a released model refuses every call, and so does a sampler
of an ended session.  A call that fails sends its session's end event first, as each call of the Tinker SDK's clients
does (``capture_exceptions(fatal=True)``), unless the SDK's telemetry is off (``ToyService(telemetry=False)``, as with
``TINKER_TELEMETRY=0``).
"""

from __future__ import annotations

import copy
import functools
import math
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest import mock

import tinker

BASE_LOGPROB = -4.0
# A step at Tinker's default learning rate (1e-4) raises a trained token's logprob by 0.2.
GAIN = 2000.0

FAULTS = {
    "one adapter": "the service holds one model of the base at a time (U1)",
    "one model": "a second LoRA model of the base is the first one (U1)",
    "shared step": "an optim_step steps every model that holds gradients, with its own parameters (U1)",
    "dropped gradients": "an optim_step clears the gradients that every model holds (U1)",
    "one gradient buffer": "a forward_backward fails while another model holds gradients (U1)",
    "sampler paths by name": "a sampler save's path names the save, not its model (U2)",
    "published base": "a published sampler serves the base model (U2)",
    "end releases all": "a session's end releases every model of the service (U3)",
    "ignores session end": "a session's end event leaves it open, until its lease expires (U3)",
    "no optimizer on reopen": "a state reopened with its optimizer gets a new optimizer (U4)",
    "no step count on reopen": "a state reopened with its optimizer keeps its moments, not its step count (U4)",
    "reopen into another model": "a state reopens into the model that trained last, which it overwrites (U4)",
    "NaN on reopen": "a state reopened with its optimizer reads NaN (U4)",
}


def _fatal(call: Callable[..., Any]) -> Callable[..., Any]:
    """A call of a session, a model or a sampler that sends its session's end event when it fails, as the Tinker SDK's
    calls do."""

    @functools.wraps(call)
    async def failing_fatally(owner: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await call(owner, *args, **kwargs)
        except Exception as exc:
            session = owner if isinstance(owner, ToySession) else owner.session
            if (telemetry := session.get_telemetry()) is not None:
                await telemetry.log_fatal_exception(exc)
            raise

    return failing_fatally


class _Done:
    """A Tinker call's future, already done."""

    def __init__(self, value: Any) -> None:
        self.value = value

    async def result_async(self) -> Any:
        return self.value


@dataclass
class Optimizer:
    """A model's AdamW state: its step count and each token's moments."""

    step: int = 0
    m: dict[int, float] = field(default_factory=dict)
    v: dict[int, float] = field(default_factory=dict)


class ToyService:
    """The service: its sessions in the order they opened, its models, its saves by path (states with their optimizer,
    and sampler weights), the LoRA models asked of it (``created``: base model and rank) and every optimizer step it
    took (``steps``: the model and its learning rate).  With ``telemetry=False`` its sessions have no telemetry, as
    with ``TINKER_TELEMETRY=0``."""

    def __init__(self, *faults: str, telemetry: bool = True) -> None:
        assert set(faults) <= set(FAULTS), f"unknown faults {sorted(set(faults) - set(FAULTS))}"
        self.faults = frozenset(faults)
        self.telemetry = telemetry
        self.sessions: list[ToySession] = []
        self.models: list[ToyModel] = []
        self.states: dict[str, tuple[dict[int, float], Optimizer]] = {}
        self.sampler_weights: dict[str, dict[int, float]] = {}
        self.created: list[tuple[str, int]] = []
        self.steps: list[tuple[str, float]] = []
        self.trained_last: ToyModel | None = None

    def service_client(self, **kwargs: Any) -> ToySession:
        """``tinker.ServiceClient(base_url=...)``: a new session."""

        session = ToySession(self, len(self.sessions) + 1, kwargs.get("base_url"))
        self.sessions.append(session)
        return session

    def open(self, session: ToySession, offsets: dict[int, float], optimizer: Optimizer) -> ToyModel:
        model = ToyModel(self, session, f"model-{len(self.models) + 1}", dict(offsets), optimizer)
        self.models.append(model)
        return model

    def end(self, session: ToySession) -> None:
        session.ended = True
        for model in self.models:
            if model.session is session or "end releases all" in self.faults:
                model.released = True


class ToySession:
    def __init__(self, service: ToyService, number: int, base_url: str | None) -> None:
        self.service, self.number, self.base_url = service, number, base_url
        self.ended = self.closed = False
        self.holder = SimpleNamespace(close=self._close)

    def _close(self) -> None:
        self.closed = True

    def get_telemetry(self) -> ToyTelemetry | None:
        return ToyTelemetry(self) if self.service.telemetry else None

    @_fatal
    async def create_lora_training_client_async(self, base_model: str, rank: int = 32, **kwargs: Any) -> ToyModel:
        service = self.service
        service.created.append((base_model, rank))
        live = [model for model in service.models if not model.released]
        if live and "one adapter" in service.faults:
            raise RuntimeError("no free slot: the service holds one model")
        if live and "one model" in service.faults:
            return live[0]
        return service.open(self, {}, Optimizer())

    @_fatal
    async def create_training_client_from_state_with_optimizer_async(self, path: str, **kwargs: Any) -> ToyModel:
        service = self.service
        offsets, saved = service.states[path]
        optimizer = Optimizer() if "no optimizer on reopen" in service.faults else copy.deepcopy(saved)
        if "no step count on reopen" in service.faults:
            optimizer.step = 0
        if "NaN on reopen" in service.faults:
            offsets = dict.fromkeys(offsets, math.nan)
        if "reopen into another model" in service.faults and service.trained_last is not None:
            model = service.trained_last
            model.offsets, model.optimizer = dict(offsets), optimizer
            return model
        return service.open(self, offsets, optimizer)

    @_fatal
    async def create_sampling_client_async(self, model_path: str | None = None, base_model: str | None = None) -> Any:
        return ToySampler(self, {} if model_path is None else self.service.sampler_weights[model_path])


class ToyTelemetry:
    """A session's telemetry, whose fatal exception ends the session, as the Tinker SDK's session-end event does."""

    def __init__(self, session: ToySession) -> None:
        self.session = session

    async def log_fatal_exception(self, exception: BaseException, severity: str = "ERROR") -> bool:
        if "ignores session end" not in self.session.service.faults:
            self.session.service.end(self.session)
        return True


class ToyModel:
    """A LoRA model of the service: a training client of a session, until the session that opened it ends."""

    def __init__(
        self, service: ToyService, session: ToySession, model_id: str, offsets: dict[int, float], optimizer: Optimizer
    ) -> None:
        self.service, self.session, self.model_id = service, session, model_id
        self.offsets, self.optimizer = offsets, optimizer
        self.gradients: dict[int, float] = {}
        self.released = False

    def _alive(self) -> None:
        if self.released:
            raise RuntimeError(f"unknown model {self.model_id}: the service released it")

    def _outputs(self, data: list[Any]) -> Any:
        outputs = []
        for datum in data:
            logprobs = [_logprob(self.offsets, token) for token in datum.loss_fn_inputs["target_tokens"].data]
            outputs.append({"logprobs": tinker.TensorData(data=logprobs, dtype="float32", shape=[len(logprobs)])})
        return SimpleNamespace(loss_fn_outputs=outputs, metrics={})

    @_fatal
    async def forward_async(self, data: list[Any], loss_fn: str, loss_fn_config: Any = None) -> _Done:
        self._alive()
        return _Done(self._outputs(data))

    @_fatal
    async def forward_backward_async(self, data: list[Any], loss_fn: str, loss_fn_config: Any = None) -> _Done:
        self._alive()
        holding = [model for model in self.service.models if model is not self and model.gradients]
        if holding and "one gradient buffer" in self.service.faults:
            raise RuntimeError(f"{holding[0].model_id} holds gradients: step it first")
        outputs = self._outputs(data)  # the logprobs before the step
        for datum in data:
            inputs = datum.loss_fn_inputs
            for token, weight in zip(inputs["target_tokens"].data, inputs["weights"].data, strict=True):
                self.gradients[token] = self.gradients.get(token, 0.0) - GAIN * weight
        self.service.trained_last = self
        return _Done(outputs)

    @_fatal
    async def optim_step_async(self, params: Any) -> _Done:
        self._alive()
        shared = "shared step" in self.service.faults
        for model in [model for model in self.service.models if model.gradients] if shared else [self]:
            model._adamw(params)
        if "dropped gradients" in self.service.faults:
            for model in self.service.models:
                model.gradients = {}
        return _Done(SimpleNamespace(metrics={}))

    def _adamw(self, params: Any) -> None:
        self.service.steps.append((self.model_id, params.learning_rate))
        optimizer = self.optimizer
        optimizer.step += 1
        beta1, beta2 = params.beta1, params.beta2
        for token in set(optimizer.m) | set(self.gradients):
            gradient = self.gradients.get(token, 0.0)
            m = optimizer.m[token] = beta1 * optimizer.m.get(token, 0.0) + (1 - beta1) * gradient
            v = optimizer.v[token] = beta2 * optimizer.v.get(token, 0.0) + (1 - beta2) * gradient * gradient
            m_hat, v_hat = m / (1 - beta1**optimizer.step), v / (1 - beta2**optimizer.step)
            offset = self.offsets.get(token, 0.0)
            update = m_hat / (math.sqrt(v_hat) + params.eps) + params.weight_decay * offset
            self.offsets[token] = offset - params.learning_rate * update
        self.gradients = {}

    @_fatal
    async def save_state_async(self, name: str, ttl_seconds: int | None = None) -> _Done:
        self._alive()
        path = f"tinker://{self.model_id}/weights/{name}"
        self.service.states[path] = (dict(self.offsets), copy.deepcopy(self.optimizer))
        return _Done(SimpleNamespace(path=path))

    @_fatal
    async def save_weights_for_sampler_async(self, name: str, ttl_seconds: int | None = None) -> _Done:
        self._alive()
        by_name = "sampler paths by name" in self.service.faults
        path = f"tinker://sampler_weights/{name}" if by_name else f"tinker://{self.model_id}/sampler_weights/{name}"
        self.service.sampler_weights[path] = dict(self.offsets)
        return _Done(SimpleNamespace(path=path))

    @_fatal
    async def save_weights_and_get_sampling_client_async(self, name: str | None = None, **kwargs: Any) -> ToySampler:
        self._alive()
        return ToySampler(self.session, {} if "published base" in self.service.faults else self.offsets)


class ToySampler:
    """A sampling client of a session, serving the offsets it was made of."""

    def __init__(self, session: ToySession, offsets: dict[int, float]) -> None:
        self.session, self.offsets = session, dict(offsets)

    @_fatal
    async def compute_logprobs_async(self, prompt: Any) -> list[float | None]:
        if self.session.ended:
            raise RuntimeError(f"unknown sampling session: session {self.session.number} ended")
        tokens = prompt.to_ints()
        return [None] + [_logprob(self.offsets, token) for token in tokens[1:]]


def _logprob(offsets: dict[int, float], token: int) -> float:
    return BASE_LOGPROB + GAIN * offsets.get(token, 0.0)


@contextmanager
def toy_service(service: ToyService) -> Generator[ToyService]:
    """``service`` in place of the Tinker service for the ``with`` block: each ``tinker.ServiceClient`` is a session of
    it."""

    with mock.patch.object(tinker, "ServiceClient", service.service_client):
        yield service
