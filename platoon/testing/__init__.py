"""platoon.testing: test environments, harnesses and episode programs on CPU, with no GPU, Tinker server or model
download.

A scripted policy writes what the model would; everything else is platoon's own code, so a test runs an episode
the way training runs it::

    from platoon.testing import ScriptedPolicy, run_episode, tool_call

    policy = ScriptedPolicy([tool_call("guess", number=50), tool_call("guess", number=42)])  # used in order
    run = await run_episode("train:41", service="python://guess:GuessTheNumber", harness="platoon-react", policy=policy)
    assert run.reward == 1.0 and len(policy.calls) == 2

The pieces:

* :class:`ScriptedPolicy`: stands in for the Tinker sampler; its completions come from a list or from a function of
  the decoded prompt (:class:`Completion`, :func:`tool_call` and :func:`rules` help write them), and it records every
  call (:class:`PolicyCall`);
* :func:`run_episode`: one task through ``harness_rollout``, with an environment, a harness and an episode program;
  it returns an :class:`EpisodeRun`;
* :func:`serve_policy`: the scripted policy as the process's policy, for ``platoon rollout`` and ``platoon eval`` run
  in process;
* :func:`make_gateway`: the real model gateway over a scripted policy, for lower-level tests;
* :func:`toy_tokenizer` and :func:`toy_renderer`: a 7 KB byte-level tokenizer with Qwen3's chat tokens, and
  tinker-cookbook's ``qwen3_instruct`` renderer over it, so prompts render and parse offline as they do for a Qwen3
  model, apart from the token ids;
* a pytest plugin with the fixtures ``serve_policy``, ``toy_tokenizer`` and ``toy_renderer``, enabled with
  ``pytest -p platoon.testing`` and never automatically.

The names this module exports are a stable API; its submodules are not.  It needs the ``tinker`` extra (and the
``ors`` extra for ORS environments), and pytest only for the plugin.  ``docs/how-to/test-extensions.md`` is the
guide.
"""

from typing import TYPE_CHECKING, Any

# ``pytest -p platoon.testing`` registers this module as a plugin, and pytest then loads the fixtures from here.
pytest_plugins = ["platoon.testing.plugin"]

# Public name -> the submodule that defines it, imported on first use so that loading the plugin stays cheap.
_EXPORTS = {
    **dict.fromkeys(
        ("TOY_MODEL_NAME", "TOY_RENDERER_NAME", "TOY_TOKENIZER_DIR", "toy_renderer", "toy_tokenizer"),
        "platoon.testing.tokenizer",
    ),
    **dict.fromkeys(
        (
            "Completion",
            "PolicyCall",
            "ScriptExhaustedError",
            "ScriptedPolicy",
            "last_user_message",
            "rules",
            "sample_response",
            "tool_call",
        ),
        "platoon.testing.policy",
    ),
    **dict.fromkeys(("make_gateway", "serve_policy"), "platoon.testing.gateway"),
    **dict.fromkeys(("EpisodeRun", "run_episode"), "platoon.testing.episode"),
}

__all__ = [
    "TOY_MODEL_NAME",
    "TOY_RENDERER_NAME",
    "TOY_TOKENIZER_DIR",
    "Completion",
    "EpisodeRun",
    "PolicyCall",
    "ScriptExhaustedError",
    "ScriptedPolicy",
    "last_user_message",
    "make_gateway",
    "rules",
    "run_episode",
    "sample_response",
    "serve_policy",
    "tool_call",
    "toy_renderer",
    "toy_tokenizer",
]


def __getattr__(name: str) -> Any:
    import importlib

    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from platoon.testing.episode import EpisodeRun, run_episode
    from platoon.testing.gateway import make_gateway, serve_policy
    from platoon.testing.policy import (
        Completion,
        PolicyCall,
        ScriptedPolicy,
        ScriptExhaustedError,
        last_user_message,
        rules,
        sample_response,
        tool_call,
    )
    from platoon.testing.tokenizer import (
        TOY_MODEL_NAME,
        TOY_RENDERER_NAME,
        TOY_TOKENIZER_DIR,
        toy_renderer,
        toy_tokenizer,
    )
