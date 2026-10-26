"""A policy as the config declares it (:class:`PolicySpec`), and the run's policies as its config declares them
(:func:`policy_specs`).  Imports no Tinker.

A run trains one policy today, ``main``, which ``train:`` declares: each field of its spec keeps its ``train.*`` name,
so that one vocabulary names a policy wherever it is declared.
"""

from dataclasses import dataclass, field
from typing import Any, Final

from platoon.train import task_stats

PolicyId = str
#: The policy that ``train:`` declares.  The same object as ``task_stats.MAIN``, by whose versions the per-task
#: statistics age their evidence, so that their snapshots keep their format.
MAIN: Final[PolicyId] = task_stats.MAIN


@dataclass(frozen=True)
class PolicySpec:
    """One policy, as the config declares it: the model, how its prompts are rendered and sampled, its Tinker service,
    and how a new run trains it (its adapter's rank, the weights it starts from)."""

    id: PolicyId
    model_name: str
    renderer_name: str
    # The config's, as the renderer takes them.  A dict does not hash, so the spec's hash leaves it out.
    renderer_kwargs: dict[str, Any] = field(default_factory=dict, hash=False)
    context_window_length: int | None = None
    sample_timeout_seconds: float | None = None
    # Its Tinker service; None: TINKER_BASE_URL, else hosted Tinker.
    tinker_base_url: str | None = None
    lora_rank: int = 32
    # The weights a new run starts from (not its optimizer's state); None: the base model's.
    load_checkpoint_path: str | None = None


def policy_specs(config: Any) -> dict[PolicyId, PolicySpec]:
    """The run's policies, in order: ``main``, as ``train:`` declares it (``train.model_name``, ``renderer_name``,
    ``renderer_kwargs``, ``context_window_length``, ``sample_timeout_seconds`` and ``lora_rank``), with the run's
    ``tinker_base_url`` and the weights ``checkpoint.load_checkpoint_path`` starts it from."""

    train = config.train
    main = PolicySpec(
        id=MAIN,
        model_name=train.model_name,
        renderer_name=train.renderer_name,
        renderer_kwargs=train.renderer_kwargs,
        context_window_length=train.context_window_length,
        sample_timeout_seconds=train.sample_timeout_seconds,
        tinker_base_url=config.tinker_base_url,
        lora_rank=train.lora_rank,
        load_checkpoint_path=config.checkpoint.load_checkpoint_path,
    )
    return {MAIN: main}
