"""Fixtures for the example tests: each example runs as its README runs it, end to end on CPU, with no network and
no Tinker.

The gateway is the real one (``platoon.testing``'s toy tokenizer under the ``qwen3_instruct`` renderer, which the
example configs name).  Only sampling is scripted: a policy function reads the prompt and returns the completion
text, so concurrent agents get answers by what they are asked, not by call order.  Two flavours:

- ``policy_gateway(policy)`` installs a gateway over the fake sampler of ``tests/inline/inline_fakes.py``, whose
  policy reads the last user message; ``example(name)`` enters ``examples/<name>``.
- ``serve_policy(policy)`` makes ``platoon rollout`` and ``platoon eval`` sample a policy of the whole decoded
  prompt instead of a Tinker model (``platoon.testing.serve_policy``); ``platoon_cli(*argv)`` runs the command line
  in process.
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENREWARD_DISABLE_UPDATE_CHECK", "1")

TESTS = Path(__file__).resolve().parent.parent
REPO = TESTS.parent
EXAMPLES = REPO / "examples"
for helpers in ("gateway", "inline"):
    if str(TESTS / helpers) not in sys.path:
        sys.path.insert(0, str(TESTS / helpers))
# Each example's modules import by their own names, as its README runs them with the example on PYTHONPATH.
for example in ("subagents", "program-reward-shaping"):
    sys.path.insert(0, str(EXAMPLES / example))

from gateway_fakes import MODEL_NAME  # noqa: E402
from inline_fakes import PolicySampler  # noqa: E402

from platoon import cli, testing  # noqa: E402
from platoon.gateway.core import ModelGateway, set_default_gateway  # noqa: E402

Policy = Callable[[str], str]


@pytest.fixture(autouse=True)
def _repository_root(monkeypatch):
    """Run from the repository root, where the examples' configs and the commands in their READMEs start."""

    monkeypatch.chdir(REPO)


@pytest.fixture(autouse=True)
def _tinker_api_key(monkeypatch):
    """A key, as the examples' READMEs ask for: the commands that serve a policy refuse to start without one that
    the Tinker client would take, before anything runs.  The scripted policies here never send it anywhere."""

    monkeypatch.setenv("TINKER_API_KEY", "tml-test")


@pytest.fixture(scope="session")
def tokenizer():
    return testing.toy_tokenizer()


@pytest.fixture(scope="session")
def renderer(tokenizer):
    return testing.toy_renderer("qwen3_instruct")


@pytest.fixture
def policy_gateway(renderer, tokenizer) -> Iterator[Callable[[Callable[[str], str]], PolicySampler]]:
    """``install(policy)`` makes a gateway over a fake sampler this process's policy gateway, as ``platoon
    rollout`` installs the Tinker policy, and returns the sampler."""

    def install(policy: Callable[[str], str], **kwargs: Any) -> PolicySampler:
        sampler = PolicySampler(tokenizer, policy)
        set_default_gateway(
            ModelGateway(sampler=sampler, renderer=renderer, tokenizer=tokenizer, model_name=MODEL_NAME, **kwargs)
        )
        return sampler

    yield install
    set_default_gateway(None)


@pytest.fixture
def example(monkeypatch) -> Callable[[str], Path]:
    """``example(name)`` enters ``examples/<name>``: the working directory and the front of ``sys.path``."""

    def enter(name: str) -> Path:
        directory = EXAMPLES / name
        monkeypatch.chdir(directory)
        monkeypatch.syspath_prepend(str(directory))
        return directory

    return enter


@pytest.fixture
def serve_policy() -> Iterator[Callable[[Policy], testing.ScriptedPolicy]]:
    """``serve_policy(policy)`` makes ``platoon rollout`` and ``platoon eval`` sample ``policy`` instead of a Tinker
    model, until the test ends, and returns it as a ``platoon.testing.ScriptedPolicy``."""

    with contextlib.ExitStack() as stack:
        yield lambda policy: stack.enter_context(testing.serve_policy(policy))


@pytest.fixture
def platoon_cli() -> Callable[..., None]:
    """``platoon_cli(*argv)`` runs ``platoon ARGV`` in process, failing the test on a non-zero exit."""

    def run(*argv: str) -> None:
        try:
            cli.main(list(argv))
        except SystemExit as exit_info:
            if exit_info.code:
                raise AssertionError(f"platoon {' '.join(argv)} exited with {exit_info.code}") from None

    return run
