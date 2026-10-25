"""Helpers for the harness example tests: the shared fakes on the path, a policy that plays number search, and a
stand-in for the Tinker policy that ``platoon rollout`` would serve."""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any

import pytest

TESTS = Path(__file__).resolve().parents[1]
REPO = TESTS.parent
EXAMPLES = REPO / "examples"
for _directory in ("gateway", "inline", "harness"):
    if str(TESTS / _directory) not in sys.path:
        sys.path.insert(0, str(TESTS / _directory))

from gateway_fakes import MODEL_NAME  # noqa: E402
from inline_fakes import PolicySampler  # noqa: E402

from platoon.gateway import core as gateway_core  # noqa: E402
from platoon.harness import function as function_harnesses  # noqa: E402
from platoon.harness import registry as harness_registry  # noqa: E402
from platoon.testing import toy_tokenizer  # noqa: E402

# A number-search tool call, as Qwen3-4B-Instruct-2507 writes it.
GUESS = '<tool_call>\n{{"name": "guess", "arguments": {{"number": {}}}}}\n</tool_call>'
# What bisect() guesses on number-search task val:0 (a number between 119 and 965; it is 447).
VAL_0_GUESSES = [542, 330, 436, 489, 462, 449, 442, 445, 447]


def bisect():
    """A policy that plays number search well: it halves the range the instruction and the replies leave.

    It sees the last user message (``inline_fakes.PolicySampler``): the instruction, then each guess's reply.
    """

    state: dict[str, int] = {}

    def policy(message: str) -> str:
        if bounds := re.search(r"between (\d+) and (\d+)", message):
            state.update(low=int(bounds[1]), high=int(bounds[2]))
        elif "Too low" in message:
            state["low"] = state["guess"] + 1
        elif "Too high" in message:
            state["high"] = state["guess"] - 1
        state["guess"] = (state["low"] + state["high"]) // 2
        return GUESS.format(state["guess"])

    return policy


def load_tokenizer() -> Any:
    """The test model's tokenizer: ``platoon.testing``'s toy tokenizer, which speaks the Qwen3 chat format."""

    return toy_tokenizer()


def serve_scripted_policy(monkeypatch: pytest.MonkeyPatch, tokenizer: Any, policy: Any) -> PolicySampler:
    """Make ``platoon rollout`` serve ``policy`` behind the real gateway instead of a Tinker sampler.

    ``EvalPolicy.install`` would register a Tinker sampler as the process's gateway; the stand-in registers a
    gateway over a scripted sampler. Returns the sampler, which records what each call was asked.
    """

    from tinker_cookbook.renderers import get_renderer

    sampler = PolicySampler(tokenizer, policy)
    gateway = gateway_core.ModelGateway(
        sampler=sampler, renderer=get_renderer("qwen3_instruct", tokenizer), tokenizer=tokenizer, model_name=MODEL_NAME
    )

    class ScriptedPolicy:
        def __init__(self, **settings: Any) -> None:
            self.settings = settings

        def install(self) -> None:
            gateway_core.set_default_gateway(gateway)

    monkeypatch.setattr(gateway_core, "_default_gateway", None)
    monkeypatch.setattr(importlib.import_module("platoon.eval"), "EvalPolicy", ScriptedPolicy)
    return sampler


def isolate_example(monkeypatch: pytest.MonkeyPatch, directory: Path, modules: tuple[str, ...] = ()) -> None:
    """Put an example directory first on the path, import its ``modules`` afresh, and leave the harness registry
    as it was when the test ends."""

    monkeypatch.syspath_prepend(str(directory))
    for module in modules:
        monkeypatch.delitem(sys.modules, module, raising=False)
    # Core's built-ins register on the process's first harness lookup: make it here, before the copy, or a first
    # lookup inside the example registers them in the copy alone and the tests after it no longer find them.
    harness_registry.available_harnesses()
    monkeypatch.setattr(function_harnesses, "_function_harnesses", dict(function_harnesses._function_harnesses))
