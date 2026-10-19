"""A real model gateway over a scripted policy, and serving one as the process's policy.

:func:`make_gateway` builds a ``platoon.gateway.ModelGateway`` (the one class that renders, samples, parses and records
every model call in platoon) whose sampler is a :class:`~platoon.testing.ScriptedPolicy`, with the toy tokenizer and
the ``qwen3_instruct`` renderer unless told otherwise.  :func:`serve_policy` makes such a gateway the policy that
``harness_rollout``, ``platoon rollout`` and ``platoon eval`` sample in this process, in place of Tinker.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Generator, Mapping
from typing import TYPE_CHECKING, Any

from platoon.testing.policy import Script, ScriptedPolicy, as_policy
from platoon.testing.tokenizer import TOY_MODEL_NAME, TOY_RENDERER_NAME, toy_tokenizer

if TYPE_CHECKING:
    from tinker_cookbook.renderers import Renderer
    from tinker_cookbook.tokenizer_utils import Tokenizer

    from platoon.gateway import ModelGateway

# The TINKER_API_KEY serve_policy sets when none is set: commands that serve a policy refuse to start without a key
# the Tinker client would take, and a scripted policy never sends it anywhere.
PLACEHOLDER_TINKER_API_KEY = "tml-platoon-testing"


def make_gateway(
    policy: ScriptedPolicy | Script = (),
    *,
    tokenizer: Tokenizer | None = None,
    renderer: Renderer | str | None = None,
    renderer_kwargs: Mapping[str, Any] | None = None,
    model_name: str = TOY_MODEL_NAME,
    **settings: Any,
) -> ModelGateway:
    """A ``ModelGateway`` that samples ``policy`` (a :class:`ScriptedPolicy`, or a script to wrap in one).

    Args:
        policy: What the model writes; keep a reference to a ``ScriptedPolicy`` to read its ``calls`` afterwards.
        tokenizer: The tokenizer the gateway renders with (default: the policy's, else the toy tokenizer).  The
            policy samples with the same one.
        renderer: A tinker-cookbook ``Renderer``, or the name of one to build over ``tokenizer`` (platoon's
            renderer names included); default ``qwen3_instruct``.
        renderer_kwargs: Attributes set on a renderer built by name, as a config's ``renderer_kwargs`` are.
        model_name: The model name the gateway reports.
        **settings: Other ``ModelGateway`` arguments: ``max_tokens``, ``temperature``, ``top_k``, ``top_p``,
            ``seed``, ``context_window_length``, ``sample_timeout_seconds``.
    """

    from platoon.gateway import ModelGateway

    policy = as_policy(policy)
    tokenizer = tokenizer or policy.tokenizer or toy_tokenizer()
    if policy.tokenizer is None:
        policy.tokenizer = tokenizer
    elif policy.tokenizer is not tokenizer:
        raise ValueError("The scripted policy samples with another tokenizer than the gateway renders with")
    if renderer is None or isinstance(renderer, str):
        renderer = build_renderer(
            renderer or TOY_RENDERER_NAME, tokenizer, model_name=model_name, **(renderer_kwargs or {})
        )
    elif renderer_kwargs:
        raise ValueError("renderer_kwargs apply only to a renderer named by a string")
    return ModelGateway(sampler=policy, renderer=renderer, tokenizer=tokenizer, model_name=model_name, **settings)


def build_renderer(name: str, tokenizer: Tokenizer, *, model_name: str = TOY_MODEL_NAME, **attributes: Any) -> Renderer:
    """The renderer ``name`` over ``tokenizer``, built as a policy builds it: platoon's renderer names resolve, and
    ``attributes`` (a config's ``renderer_kwargs``) are set on it."""

    from tinker_cookbook.renderers import get_renderer

    from platoon.train.tinker.renderers import register as register_platoon_renderers

    register_platoon_renderers()
    renderer = get_renderer(name, tokenizer, model_name=model_name)
    for key, value in attributes.items():
        if not hasattr(renderer, key):
            raise ValueError(f"Renderer {name!r} has no attribute {key!r} to set")
        setattr(renderer, key, value)
    return renderer


@contextlib.contextmanager
def serve_policy(
    policy: ScriptedPolicy | Script = (),
    *,
    tokenizer: Tokenizer | None = None,
    renderer: Renderer | str | None = None,
    **settings: Any,
) -> Generator[ScriptedPolicy]:
    """Serve ``policy`` as this process's policy while the block runs, and yield it as a ``ScriptedPolicy``.

    Inside the block:

    * ``harness_rollout``, which samples the process's policy gateway, samples ``policy`` through a gateway from
      :func:`make_gateway`;
    * ``platoon rollout`` and ``platoon eval``, run in this process (``platoon.cli.main([...])``), sample
      ``policy`` instead of Tinker.  Their gateway takes the config's model name, renderer and ``renderer_kwargs``
      (over ``tokenizer``, the toy tokenizer by default, so the config should name a Qwen3 renderer such as
      ``qwen3_instruct``) unless ``renderer`` says otherwise, and its ``sample_timeout_seconds`` unless ``settings``
      give one.  It leaves out the config's context window, since the toy tokenizer's token counts are not the
      model's; pass ``context_window_length`` to test one.  An evaluation that runs its rollouts in worker processes
      (``use_subprocesses``) does not see the scripted policy;
    * ``TINKER_API_KEY`` is set to a placeholder when it is unset, since those commands check for one first.

    Everything is restored when the block ends, the previous policy gateway included.
    """

    from platoon.eval.config_defs import EvalPolicy
    from platoon.gateway import core

    scripted = as_policy(policy)

    def install(self: EvalPolicy) -> None:
        if renderer is None:
            chosen, renderer_kwargs = self.renderer_name, self.renderer_kwargs
        else:
            chosen, renderer_kwargs = renderer, None
        gateway = make_gateway(
            scripted,
            tokenizer=tokenizer,
            renderer=chosen,
            renderer_kwargs=renderer_kwargs,
            model_name=self.model_name,
            **{"sample_timeout_seconds": self.sample_timeout_seconds, **settings},
        )
        core.set_default_gateway(gateway)

    with contextlib.ExitStack() as restore:
        api_key = os.environ.get("TINKER_API_KEY")
        if not api_key:
            os.environ["TINKER_API_KEY"] = PLACEHOLDER_TINKER_API_KEY
            restore.callback(_restore_variable, "TINKER_API_KEY", api_key)
        restore.callback(core.set_default_gateway, core._default_gateway)
        restore.callback(setattr, EvalPolicy, "install", EvalPolicy.__dict__["install"])
        setattr(EvalPolicy, "install", install)
        core.set_default_gateway(make_gateway(scripted, tokenizer=tokenizer, renderer=renderer, **settings))
        yield scripted


def _restore_variable(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
