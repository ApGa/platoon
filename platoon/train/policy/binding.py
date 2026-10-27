"""What a rollout worker needs to sample a policy (:func:`model_binding`): the binding file it follows for the
policy's sampling client, and the model, renderer and sampling settings it builds its own model gateway with."""

from typing import Any

from platoon.train.workers.protocol import ModelBinding


def model_binding(
    model_info: Any, *, binding_path: str, follow_poll_seconds: float, checkpoint_version: int
) -> ModelBinding:
    """The binding of the policy whose model info is ``model_info``, for a rollout worker that follows its sampling
    client at ``binding_path`` (every ``follow_poll_seconds``; empty: no worker follows one) and samples it at
    ``checkpoint_version``.

    Never pickles the ``Renderer``: ``Renderer.__reduce__`` carries only
    ``(renderer_name, model_name, has_image_processor)`` and drops
    ``renderer_kwargs``, which ``register_tinker_llm`` applies by
    ``setattr``.  Today's values happen to equal the class defaults, so
    shipping the object is a latent silent-drift bug that fires the day
    someone sets ``preserve_thinking: false``.
    """

    llm = getattr(model_info, "llm", None)
    bare_model_name = getattr(llm, "model_name", None) or str(getattr(model_info, "model_name", "") or "")
    return ModelBinding(
        binding_path=binding_path,
        bootstrap_blob=b"",
        follow_poll_seconds=follow_poll_seconds,
        model_name=bare_model_name,
        renderer_name=getattr(model_info, "renderer_name", "") or "",
        renderer_kwargs=dict(getattr(model_info, "renderer_kwargs", {}) or {}),
        context_window_length=getattr(llm, "context_window_length", None),
        sample_timeout_seconds=getattr(llm, "sample_timeout_seconds", None),
        max_tokens=int(getattr(llm, "max_tokens", 4096)),
        temperature=float(getattr(llm, "temperature", 1.0)),
        top_k=int(getattr(llm, "top_k", -1)),
        top_p=float(getattr(llm, "top_p", 1.0)),
        seed=getattr(llm, "seed", None),
        checkpoint_version=checkpoint_version,
    )
