"""A policy as the config declares it (:class:`PolicySpec`), and the run's policies as its config declares them
(:func:`policy_specs`).  Imports no Tinker.

A run trains one policy today, ``main``, which ``train:`` declares: each field of its spec keeps its ``train.*`` name,
so that one vocabulary names a policy wherever it is declared.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from platoon.train import task_stats

if TYPE_CHECKING:
    from platoon.train.config_defs import AdamParams

PolicyId = str
# The policy that ``train:`` declares.  The same object as ``task_stats.MAIN``, by whose versions the per-task
# statistics age their evidence, so that their snapshots keep their format.
MAIN: Final[PolicyId] = task_stats.MAIN


@dataclass(frozen=True)
class PolicySpec:
    """One policy, as the config declares it: the model, how its prompts are rendered and sampled, its Tinker service,
    how a new run trains it (its adapter's rank, the weights it starts from), and how each step updates it (its
    optimizer, and the loss its rows train under)."""

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
    # Its optimizer's settings, the loss its rows train under (its objective's, platoon.train.objectives) and the
    # loss's keyword arguments; each None: the run's train.optimizer, train.loss_fn or train.loss_fn_config, which
    # policy_specs puts in its place, so that the trainer reads them from the spec alone.
    optimizer: "AdamParams | None" = field(default=None, hash=False)
    loss_fn: str | None = None
    loss_fn_config: dict[str, Any] | None = field(default=None, hash=False)


def policy_specs(config: Any) -> dict[PolicyId, PolicySpec]:
    """The run's policies, in order: ``main``, as ``train:`` declares it (``train.model_name``, ``renderer_name``,
    ``renderer_kwargs``, ``context_window_length``, ``sample_timeout_seconds``, ``lora_rank``, ``optimizer``,
    ``loss_fn`` and ``loss_fn_config``), with the run's ``tinker_base_url`` and the weights
    ``checkpoint.load_checkpoint_path`` starts it from."""

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
        optimizer=train.optimizer,
        loss_fn=train.loss_fn,
        loss_fn_config=train.loss_fn_config,
    )
    return {MAIN: main}
