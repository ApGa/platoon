"""What a rollout worker needs to sample a policy (:func:`model_binding`), which each member's worker request carries
(:mod:`platoon.train.workflow.requests`): the binding file it follows for the policy's sampling client, and the model,
renderer and sampling settings it builds its own model gateway with; the bindings of the policies the agents sample
besides their default, which the trainer publishes before the rollouts that may sample them
(:func:`publish_other_bindings`); and the publication of each new sampling client to a workflow's rollout workers
(:func:`_publish_worker_binding`), at the first step, after each step that updates the policy, and before an
evaluation whose rollouts run in workers."""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from platoon.train.workers.protocol import ModelBinding

if TYPE_CHECKING:
    from platoon.train.policy.set import PolicySet

logger = logging.getLogger(__name__)


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


def publish_other_bindings(
    policies: "PolicySet", workflow: Any, publish: Callable[..., None], *, required: bool = False
) -> None:
    """Publish, for ``workflow``'s rollout workers, the binding of each policy the agents sample besides their default
    (``policies.agents.default``, whose binding the caller publishes), at its sampler and version now: a frozen one's
    never changes, and a trained one's is published again by each step that updates it.  ``publish`` publishes one
    binding (:func:`_publish_worker_binding`).  Called before training's first group (``required``, as the default's
    first binding is) and before each evaluation; nothing in a run whose agents sample one policy."""

    for policy in policies.sampled():
        if policy.id != policies.agents.default:
            publish(workflow, policy.sampling_client, policy.version, required=required, policy=policy.id)


def _publish_worker_binding(
    workflow, sampling_client, version: int, *, required: bool = False, policy: str | None = None
) -> None:
    """Republish the sampling client for rollout workers: the agents' default policy's, or ``policy``'s, another
    policy they sample, to a workflow that binds several (``binds_policies``).

    A no-op for ``inprocess`` mode and for custom workflows that do not
    run workers.  A *re*-publication failure is logged and swallowed: every
    worker keeps the client it already holds, and under the full-finetune
    gateway that client samples the newest weights anyway.

    The FIRST publication is different, and only under
    ``rollout_isolation: process``.  Each worker builds its sampling client
    by reading this file at startup and dies with ``FileNotFoundError``
    when it is missing, so swallowing that failure would leave every
    rollout of the run crashing and the trainer producing no training data
    at all -- for as long as the job is allowed to run.  That one is fatal,
    and fatal here, before the first rollout is dispatched.
    """

    publish = getattr(workflow, "publish_model_binding", None)
    if publish is None:
        return
    try:
        if policy is None:
            publish(sampling_client, version)
        else:
            publish(sampling_client, version, policy=policy)
    except Exception:
        logger.exception("could not publish the rollout worker sampling-client binding")
        if required and getattr(workflow, "rollout_isolation", "inprocess") == "process":
            raise
