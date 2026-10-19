"""Fixtures for the built-in harness tests: ``platoon.testing``'s toy tokenizer under the ``qwen3_instruct``
renderer, and a scripted policy behind a gateway.  ``conftest.py`` here star-imports them, and so does the TextCraft
extension's (``extensions/textcraft/tests``), whose rollouts run the built-in harnesses.

CPU only, no network and no model download: the toy tokenizer renders and parses the Qwen3 chat format, so the
harnesses see a Qwen3 model's prompts.  The golden fixtures (``test_codeact_golden.py``) hold the real Qwen3
tokenizer's token ids, and that module overrides ``tokenizer`` and ``renderer`` with it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENREWARD_DISABLE_UPDATE_CHECK", "1")

import tinker  # noqa: E402

from platoon.testing import TOY_MODEL_NAME, toy_renderer, toy_tokenizer  # noqa: E402

# openreward logs every tool call at INFO through structlog.
logging.getLogger("openreward").setLevel(logging.WARNING)


@dataclass
class PolicySampler:
    """A gateway sampler whose replies come from ``policy(prompt_text) -> reply_text``.

    The policy sees the decoded prompt, so concurrent agents get replies by what they are asked, not by
    call order. ``calls`` records each decoded prompt and reply.
    """

    tokenizer: Any
    policy: Callable[[str], str]
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def sample_async(self, *, prompt, sampling_params, num_samples):
        text = self.tokenizer.decode(prompt.to_ints())
        reply = self.policy(text)
        self.calls.append((text, reply))
        tokens = self.tokenizer.encode(reply, add_special_tokens=False)
        tokens += self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        sequence = {"stop_reason": "stop", "tokens": tokens, "logprobs": [-0.5] * len(tokens)}
        return tinker.types.SampleResponse.model_validate({"sequences": [sequence], "type": "sample"})

    def sample(self, *, prompt, sampling_params, num_samples):
        raise AssertionError("the gateway samples asynchronously")


@pytest.fixture(scope="session")
def model_name():
    return TOY_MODEL_NAME


@pytest.fixture(scope="session")
def tokenizer():
    return toy_tokenizer()


@pytest.fixture(scope="session")
def renderer(tokenizer):
    return toy_renderer("qwen3_instruct")


@pytest.fixture
def make_gateway(renderer, tokenizer, model_name):
    from platoon.gateway.core import ModelGateway

    def make(policy: Callable[[str], str], **kwargs: Any) -> tuple[ModelGateway, PolicySampler]:
        sampler = PolicySampler(tokenizer, policy)
        gateway = ModelGateway(sampler=sampler, renderer=renderer, tokenizer=tokenizer, model_name=model_name, **kwargs)
        return gateway, sampler

    return make


__all__ = ["PolicySampler", "make_gateway", "model_name", "renderer", "tokenizer"]
