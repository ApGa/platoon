"""A start asks the training service for its training client until the service accepts it, for up to 20 minutes,
logging each refusal: a service may refuse a new start while it holds a dead start's session, or answer nothing while
it restarts.  A refusal that asking again cannot change is raised at once.  And a deadline on a call to the service
fires on time, not at the next heartbeat."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx
import pytest
import tinker
from fakes.config import make_config
from fakes.trainer import fake_policies

from platoon.train.policy import service as service_module
from platoon.train.policy.spec import MAIN
from platoon.train.trainer.rl import PlatoonTinkerRLTrainer


def _request() -> httpx.Request:
    return httpx.Request("POST", "http://tinker.invalid/api/v1/create_model")


def _status(code: int) -> tinker.APIStatusError:
    return tinker.APIStatusError(f"status {code}", response=httpx.Response(code, request=_request()), body=None)


def _unretrievable(code: int) -> ValueError:
    """What the Tinker SDK raises for a request whose result the service refused to give (the answer is its cause)."""

    try:
        raise ValueError(f"Error retrieving result: status code {code}") from _status(code)
    except ValueError as error:
        return error


class Opener:
    """A training service that refuses a new run's training client with ``errors`` in turn, then opens one; and a
    trainer on it (``trainer``), whose policy's session (``session``) it is."""

    def __init__(self, *errors: BaseException) -> None:
        self.errors = list(errors)
        self.asks = 0
        config = make_config()
        self.trainer = PlatoonTinkerRLTrainer(config, [], policies=fake_policies(config, lambda: self))
        self.session = self.trainer.policies[MAIN].new_service_client()

    async def create_lora_training_client_async(self, base_model, rank=32, **kwargs):
        self.asks += 1
        if self.errors:
            raise self.errors.pop(0)
        return "training-client"

    def open(self):
        return asyncio.run(self.trainer._open_training_client(self.session, None))


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(service_module, "TRAINING_CLIENT_FIRST_WAIT_S", 0.01)
    monkeypatch.setattr(service_module, "TRAINING_CLIENT_LONGEST_WAIT_S", 0.02)


def test_refusals_are_asked_again_and_logged_until_the_service_accepts(caplog):
    caplog.set_level(logging.WARNING)
    opener = Opener(
        tinker.BadRequestError(
            "this gateway serves one model at a time", response=httpx.Response(400, request=_request()), body=None
        ),
        tinker.APIConnectionError(request=_request()),
        _status(503),
        _unretrievable(409),
        tinker.RequestFailedError("model released before initialization", request_id="r", category="user"),
        TimeoutError("no answer"),
    )

    assert opener.open() == "training-client"

    assert opener.asks == 7
    refusals = [record.getMessage() for record in caplog.records]
    assert len(refusals) == 6
    assert refusals[0].startswith(
        "The training service refused the training client (attempt 1, 0s after the first): BadRequestError: this "
        "gateway serves one model at a time. Asking again in 0s, for up to"
    )
    assert "(attempt 2," in refusals[1] and "APIConnectionError: Connection error." in refusals[1]


@pytest.mark.parametrize("error", [_status(401), _status(403), _status(404), _status(422), _unretrievable(404)])
def test_an_answer_that_asking_again_cannot_change_is_raised_at_once(error):
    opener = Opener(error)

    with pytest.raises(type(error)):
        opener.open()

    assert opener.asks == 1


@pytest.mark.parametrize("error", [RuntimeError("a bug"), AssertionError("weights are not LoRA"), ValueError("bad")])
def test_an_error_that_is_not_the_services_answer_is_raised_at_once(error):
    opener = Opener(error)

    with pytest.raises(type(error)):
        opener.open()

    assert opener.asks == 1


def test_a_start_gives_up_after_the_time_it_asks_for(monkeypatch):
    monkeypatch.setattr(service_module, "TRAINING_CLIENT_OPEN_TIMEOUT_S", 0.2)
    opener = Opener(*[tinker.APIConnectionError(request=_request()) for _ in range(1000)])

    began = time.monotonic()
    with pytest.raises(tinker.APIConnectionError) as raised:
        opener.open()

    assert time.monotonic() - began < 5
    assert 2 <= opener.asks < 1000
    assert any("a start asks for 0s" in note for note in raised.value.__notes__)


def test_a_stop_while_waiting_ends_the_asking_without_a_client(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(service_module, "TRAINING_CLIENT_FIRST_WAIT_S", 30.0)  # a wait that the stop cuts short
    opener = Opener(*[tinker.APIConnectionError(request=_request()) for _ in range(10)])

    async def open_and_stop():
        stop = asyncio.Event()
        asyncio.get_running_loop().call_later(0.2, stop.set)
        return await opener.trainer._open_training_client(opener.session, None, stop=stop)

    began = time.monotonic()
    assert asyncio.run(open_and_stop()) is None
    assert time.monotonic() - began < 5
    assert opener.asks == 1
    assert "Stopped asking the training service for a training client: training was stopped" in caplog.text


def test_a_deadline_fires_on_time_and_not_at_the_next_heartbeat():
    trainer = Opener().trainer

    async def wait() -> None:
        await trainer._await_with_heartbeat(asyncio.Event().wait(), name="stalled call", total_timeout=0.3)

    began = time.monotonic()
    with pytest.raises(TimeoutError, match="stalled call timed out after 0s"):
        asyncio.run(wait())
    assert time.monotonic() - began < 5  # the heartbeat interval is 60 s
