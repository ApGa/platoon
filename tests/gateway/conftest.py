"""Fixtures for the gateway tests: ``platoon.testing``'s toy tokenizer under the ``qwen3_instruct`` renderer, and a
fake sampler.

CPU only, no network and no model download: the toy tokenizer renders and parses the Qwen3 chat format, so the
prompts and parses are a Qwen3 model's; only the token ids differ.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # nothing here downloads; make sure nothing tries
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from gateway_fakes import MODEL_NAME, ScriptedSampler  # noqa: E402

from platoon.gateway.core import ModelGateway  # noqa: E402
from platoon.testing import toy_renderer, toy_tokenizer  # noqa: E402


@pytest.fixture(scope="session")
def tokenizer():
    return toy_tokenizer()


@pytest.fixture(scope="session")
def renderer(tokenizer):
    return toy_renderer("qwen3_instruct")


@pytest.fixture
def sampler(tokenizer) -> ScriptedSampler:
    return ScriptedSampler(tokenizer)


@pytest.fixture
def make_gateway(renderer, tokenizer, sampler) -> Callable[..., ModelGateway]:
    def make(**kwargs: Any) -> ModelGateway:
        return ModelGateway(sampler=sampler, renderer=renderer, tokenizer=tokenizer, model_name=MODEL_NAME, **kwargs)

    return make


@pytest.fixture
def gateway(make_gateway) -> ModelGateway:
    return make_gateway()
