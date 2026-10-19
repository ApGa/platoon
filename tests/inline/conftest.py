"""Fixtures for inline function-harness tests: the gateway tests' tokenizer and renderer (``platoon.testing``'s toy
tokenizer under ``qwen3_instruct``), and a sampler whose replies are a function of the prompt, so concurrent agents
get deterministic answers."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gateway"))

from gateway_fakes import MODEL_NAME  # noqa: E402
from inline_fakes import PolicySampler  # noqa: E402

from platoon.gateway.core import ModelGateway  # noqa: E402
from platoon.testing import toy_renderer, toy_tokenizer  # noqa: E402


@pytest.fixture(scope="session")
def tokenizer():
    return toy_tokenizer()


@pytest.fixture(scope="session")
def renderer(tokenizer):
    return toy_renderer("qwen3_instruct")


@pytest.fixture
def make_gateway(renderer, tokenizer) -> Callable[..., tuple[ModelGateway, PolicySampler]]:
    def make(policy: Callable[[str], str], **kwargs: Any) -> tuple[ModelGateway, PolicySampler]:
        sampler = PolicySampler(tokenizer, policy)
        gateway = ModelGateway(sampler=sampler, renderer=renderer, tokenizer=tokenizer, model_name=MODEL_NAME, **kwargs)
        return gateway, sampler

    return make
