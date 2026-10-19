"""The Tinker service's refusal of ``TINKER_API_KEY`` where platoon opens the service.

A 401 or 403 answer, as the Tinker SDK raises it (or wraps it, for a request whose result it could not retrieve), is
the refusal of the key: the policy's sampler (``register_tinker_llm``, which ``platoon eval`` and ``platoon rollout``
serve and the trainer samples through) and the trainer's session and training client end with
``TinkerKeyRefusedError``, a configuration error whose one line names the key and the service.  Every other error of
the service stays what it was, so that a start still asks again after a refusal that may change.
"""

from __future__ import annotations

import asyncio
import signal
from types import SimpleNamespace

import httpx
import pytest
import tinker
from trainer_fakes import Workflow, make_trainer, trainer_config

from platoon.eval.config_defs import TinkerKeyRefusedError
from platoon.train.tinker import proxy, rl
from platoon.train.tinker.key_refusal import key_refusal, key_refusals

KEY = "tml-mine-0123456789"
SESSION = "https://tinker.example/services/tinker-prod/api/v1/create_session"
ADVICE = "set TINKER_API_KEY to a key that service accepts"


def _answer(status: int, body=None, error_class: type[tinker.APIStatusError] = tinker.APIStatusError):
    """The error the Tinker SDK raises for an HTTP ``status`` answer with ``body`` to a session request."""

    response = httpx.Response(status, request=httpx.Request("POST", SESSION))
    return error_class(f"Error code: {status} - {body}", response=response, body=body)


def _unretrievable(error: BaseException) -> ValueError:
    """What the Tinker SDK raises for a request whose result the service refused to give (the answer is its cause)."""

    try:
        raise ValueError("Error retrieving result") from error
    except ValueError as wrapped:
        return wrapped


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", KEY)


def test_a_401_or_a_403_refuses_the_key_and_says_which_service_and_why():
    refusal = key_refusal(_answer(401, {"detail": "Invalid API key"}, tinker.AuthenticationError))

    assert isinstance(refusal, TinkerKeyRefusedError) and refusal.status == 401
    assert str(refusal) == (
        f"the Tinker service at https://tinker.example refused TINKER_API_KEY (HTTP 401: Invalid API key): {ADVICE}"
    )

    answer = {"error": {"message": f"checkpoint tinker://run-1/weights/000004 belongs to another key than {KEY}"}}
    refusal = key_refusal(_unretrievable(_answer(403, answer, tinker.PermissionDeniedError)), asked="a training client")

    assert refusal is not None and refusal.status == 403
    assert str(refusal) == (
        "the Tinker service at https://tinker.example refused TINKER_API_KEY when asked for a training client (HTTP "
        "403: checkpoint tinker://run-1/weights/000004 belongs to another key than TINKER_API_KEY): " + ADVICE
    )
    assert str(key_refusal(_answer(401, "Unauthorized"))).endswith("(HTTP 401: Unauthorized): " + ADVICE)
    assert str(key_refusal(_answer(403))).endswith("(HTTP 403): " + ADVICE)


@pytest.mark.parametrize(
    "error",
    [
        _answer(400),
        _answer(404),
        _answer(429),
        _answer(500),
        _unretrievable(_answer(409)),
        tinker.APIConnectionError(request=httpx.Request("POST", SESSION)),
        ValueError("bad"),
        RuntimeError("a bug"),
    ],
)
def test_every_other_error_stays_what_it_was(error):
    assert key_refusal(error) is None
    with pytest.raises(type(error)) as raised:
        with key_refusals():
            raise error
    assert raised.value is error


def test_the_policy_sampler_ends_a_refused_session_or_sampler_with_the_refusal(monkeypatch):
    refused = _answer(401, {"detail": "Invalid API key"}, tinker.AuthenticationError)

    def refuse_session(**kwargs):
        raise refused

    monkeypatch.setattr(proxy.tinker, "ServiceClient", refuse_session)
    with pytest.raises(TinkerKeyRefusedError, match=r"refused TINKER_API_KEY \(HTTP 401: Invalid API key\)") as raised:
        proxy.register_tinker_llm("m", "qwen3_instruct", base_url="https://tinker.example")
    assert raised.value.__cause__ is refused

    def refuse_sampler(base_model, model_path=None):
        raise _unretrievable(_answer(403, {"detail": "not yours"}))

    monkeypatch.setattr(
        proxy.tinker, "ServiceClient", lambda **kwargs: SimpleNamespace(create_sampling_client=refuse_sampler)
    )
    with pytest.raises(TinkerKeyRefusedError) as raised:
        proxy.register_tinker_llm("m", "qwen3_instruct", model_path="tinker://run-1/sampler_weights/000040")
    assert "when asked for a sampling client of tinker://run-1/sampler_weights/000040 (HTTP 403: not yours)" in str(
        raised.value
    )


@pytest.fixture
def trainer_signals():
    """``train()`` installs handlers for these signals; put the test process's back."""

    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2)
    saved = {sig: signal.getsignal(sig) for sig in signals}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def test_the_trainer_ends_a_refused_session_or_training_client_with_the_refusal(tmp_path, monkeypatch, trainer_signals):
    trainer, saves, _logged = make_trainer(monkeypatch, trainer_config(tmp_path))

    async def refuse_training_client(service_client, resume_info):
        raise _unretrievable(_answer(403, {"detail": "checkpoint of another key"}))

    trainer._create_training_client = refuse_training_client
    with pytest.raises(TinkerKeyRefusedError, match=r"when asked for a training client \(HTTP 403: checkpoint of"):
        asyncio.run(trainer.train(train_workflow=Workflow()))

    def refuse_session(**kwargs):
        raise _answer(401, {"detail": "Invalid API key"}, tinker.AuthenticationError)

    monkeypatch.setattr(rl.tinker, "ServiceClient", refuse_session)
    with pytest.raises(TinkerKeyRefusedError, match=r"refused TINKER_API_KEY \(HTTP 401: Invalid API key\)"):
        asyncio.run(trainer.train(train_workflow=Workflow()))
    assert saves == []  # refused before training began
