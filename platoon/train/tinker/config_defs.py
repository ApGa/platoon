import copy
import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from platoon.config_defs import RolloutConfig
from platoon.deprecations import DeprecatedKey, deprecated_keys, removed_hint
from platoon.train import entry_settings
from platoon.train.admission_api import DEFAULT_MAX_GROUPS_PER_STEP, max_groups_problem
from platoon.train.components import (
    STRAGGLER_POLICY_KEY_REMOVED,
    EnvironmentConfig,
    _admission_checks,
    _sampler_checks,
    normalize_environment_configs,
)
from platoon.train.tinker.checkpoint_pruning import config_pruning_problems
from platoon.utils.config import ConfigError, value_hint
from platoon.utils.stats_logger import StatsLoggerConfig, WandBConfig


@dataclass
class AdamParams:
    """The Adam optimizer's settings, applied by the Tinker service at each optimizer step."""

    learning_rate: float = 3e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    # Gradient-norm clipping threshold; 0 disables clipping. When it is above 0 the service reports the gradient
    # norm, so a very large value (say 1e12) logs the norm without clipping.
    grad_clip_norm: float = 0.0


@dataclass
class StragglerPolicy:
    """A complete straggler policy for the groups of one label (``WorkflowConfig.straggler_policies``)."""

    # Seconds to wait for the rest of a group once ``quorum`` members have settled (None: no cutoff).
    timeout_seconds: float | None = None
    # Settled members that start the timeout (None: ``group_size - 1``). Requires ``timeout_seconds``.
    quorum: int | None = None

    def validate(self, group_size: int) -> None:
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("straggler timeout_seconds must be positive or null")
        if self.quorum is not None and not 1 <= self.quorum <= group_size:
            raise ValueError("straggler quorum must be in [1, group_size] or null")
        if self.timeout_seconds is None and self.quorum is not None:
            raise ValueError("straggler quorum requires timeout_seconds")


@dataclass
class WorkflowConfig:
    """How rollouts are grouped, credited and filtered into training data, and how rollout workers run.

    Each task is rolled out ``group_size`` times; a rollout's advantage is its reward minus the group's mean
    (or, with ``leave_one_out_baseline``, the mean of the other members).
    """

    # Keys a release removed, with what replaces them (platoon.deprecations).
    UNKNOWN_KEY_HINTS: ClassVar[dict[str, str]] = {"straggler_policy_key": STRAGGLER_POLICY_KEY_REMOVED}

    # Rollouts per task; advantages are computed within each group.
    group_size: int = 8
    # How each rollout runs: step and time limits, output directory, default sampling.
    rollout_config: RolloutConfig = field(default_factory=RolloutConfig)
    # Center each rollout's reward on the mean of the other group members rather than the whole group.
    leave_one_out_baseline: bool = False
    # Weight trajectories inversely by how many trajectories share their depth in the agent tree.
    depth_level_weighting: bool = False
    # Alternative depth strategy: scale advantages by gamma**depth, renormalized per microbatch. Mutually
    # exclusive with depth_level_weighting.
    depth_level_discount_gamma: float | None = None
    # Keep every root datum and keep each subagent datum (after prefix merging) with this probability; 1.0 keeps
    # every datum.
    subagent_datum_keep_probability: float = 1.0
    # Seed of the subagent datum sampling.
    subagent_datum_sampling_seed: int = 0
    # Defer typed action errors until group centering, then suppress only error tokens that would otherwise receive
    # positive policy credit. An error is a model call whose step, or the step misc its harness annotated, reports
    # one: a CodeAct cell with an ``error`` (refused, timed out, stderr) or an ``output`` with a traceback, a tool
    # observation with ``is_error`` (``observation_events``), or what a ``trajectory.error_completion_ids`` hook
    # reports; harnesses that annotate none of these are unaffected.
    filter_errors: bool = True
    # Retain agent messages as context, with no policy loss for either sign.
    filter_message_actions: bool = False
    # Zero-advantage datums have no policy-gradient contribution but still incur
    # a model forward/backward. Baselines and rollout metrics are computed before
    # this filter is applied. It only drops such datums from microbatches that
    # also hold nonzero ones: with or without it, a microbatch whose every
    # advantage is exactly zero has a zero objective (cispo, ppo,
    # importance_sampling) and is never sent, so a step whose objective is zero
    # (say every group's rollouts scored alike) takes no optimizer update and
    # keeps its policy version (batch/skipped_optimizer is 1).
    filter_zero_advantage_datums: bool = True
    # Straggler policy. Once ``straggler_quorum`` group members have settled (completed, failed,
    # or been interrupted), wait at most this many additional seconds for the
    # rest, then cancel them. None keeps every member until its own rollout
    # timeout.
    straggler_timeout_seconds: float | None = None
    # Number of settled members that starts the tail-grace clock. None means
    # ``group_size - 1`` (the classic single-straggler policy). Requires
    # ``straggler_timeout_seconds``.
    straggler_quorum: int | None = None
    # Complete straggler policies by label: a task's value of the run's label column (``train.label_by``; by
    # default the mixture entry it belongs to). Unlisted labels keep the global policy above; a null timeout disables
    # the cutoff for a listed label. This keeps slow graders independent of the policy used by other environments in
    # the same training mixture.
    straggler_policies: dict[str, StragglerPolicy] = field(default_factory=dict)
    # Fewest trainable rollouts a group needs to train: rollouts whose root completed with a scored outcome. A rollout
    # whose root ended invalid, error, timeout or cancelled, or that returned nothing (a setup failure, no model call,
    # a cancelled straggler, a worker that died), does not count. A group below it produces no training datums (its
    # rollout workload is still accounted for).
    min_successful_group_size: int = 1

    # --- rollout process isolation ---------------------------------------
    # "inprocess": rollouts are asyncio tasks in the trainer.
    # "process":   one worker process per rollout, killable as a group (see worker_start_method).
    # A process is the only unit of execution Python can actually kill, and it
    # is the only way two rollouts can stop sharing the interpreter's globals
    # (freezegun's freeze stack, IPython's singleton shell, sys.stdout).
    # A worker process sees no GPU (CUDA_VISIBLE_DEVICES=): it samples through the training service, and a sandbox
    # gets its GPUs from its provider (gpu: nv); rollout code that needs a GPU in its own process runs inprocess.
    rollout_isolation: Literal["inprocess", "process"] = "inprocess"
    # How a rollout worker process starts. "forkserver": forked from a process that imported the worker's stack
    # (tinker, torch, transformers, the environments' packages) once, so a worker starts in seconds rather than the
    # 17-24 s an import takes (about a minute when 64 start at once); a forked worker is the process an exec'd one
    # would be, and a worker the fork server cannot fork starts by exec. "exec": a new interpreter per worker.
    worker_start_method: Literal["forkserver", "exec"] = "forkserver"
    # Seconds a worker may take to import and register its components; generous,
    # because many workers may start at once from a shared filesystem.
    worker_startup_budget_seconds: float = 300.0
    # Extra seconds beyond the rollout's time limit (rollout_config.timeout, else
    # its harness's own) and what its environment needs beyond it, for datum
    # conversion and writing the (1-6 MB) response file.
    worker_finalize_grace_seconds: float = 120.0
    # Parent wall deadline = the child's budget + this, so the child's own
    # alarm always owns its tree first and can leave a diagnostic marker.
    worker_parent_grace_seconds: float = 60.0
    # Seconds between SIGTERM and SIGKILL when a worker is stopped; keep it short
    # so that shutdown stays fast.  0.0 means SIGKILL immediately.
    worker_termination_grace_seconds: float = 10.0
    # Groups -- not rollouts -- allowed to be starting up at once.  A group is
    # admitted atomically so its members never skew against the straggler
    # quorum clock, which has no notion of when each member began.
    worker_launch_concurrent_groups: int = 2
    # Per-worker resident-memory (RSS) soft limit in bytes.  Above it the worker
    # writes a diagnostic marker and exits with a distinct code, so an
    # over-allocating rollout is reported as out of memory.  It limits resident
    # memory rather than address space, which the imports alone inflate.
    worker_rss_soft_limit_bytes: int = 24 * 1024**3
    # Refuse to launch a worker below this MemAvailable; 0 disables.
    worker_mem_available_floor_bytes: int = 0
    # How often a worker re-checks the published sampling-client binding.
    worker_binding_poll_seconds: float = 5.0
    # Child budget of a rollout that no time limit bounds: rollout_config.timeout
    # is null, and its harness has no limit of its own (neither its spec's
    # timeout_s nor the task's own agent time limit), or the rollout is a
    # function, or a function harness's episode program, of the environment's
    # own, whose limits platoon cannot know, or its environment service does
    # not say what it needs beyond the harness (rollout_overhead_s).
    worker_unbounded_rollout_budget_seconds: float = 24 * 3600.0
    # Wrap the rollout in isolated_rollout_python_environment().
    worker_python_env_isolation: bool = False

    def __post_init__(self) -> None:
        gamma = self.depth_level_discount_gamma
        if gamma is not None:
            if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not 0 < gamma < math.inf:
                raise ValueError("depth_level_discount_gamma must be a positive finite number or null")
            if self.depth_level_weighting:
                raise ValueError("depth_level_weighting and depth_level_discount_gamma are alternatives; set one")
        if not 0.0 <= self.subagent_datum_keep_probability <= 1.0:
            raise ValueError("subagent_datum_keep_probability must be in [0, 1]")
        if isinstance(self.subagent_datum_sampling_seed, bool) or not isinstance(
            self.subagent_datum_sampling_seed, int
        ):
            raise ValueError("subagent_datum_sampling_seed must be an integer")
        if self.group_size < 1:
            raise ValueError("workflow group_size must be positive")
        StragglerPolicy(self.straggler_timeout_seconds, self.straggler_quorum).validate(self.group_size)
        self.straggler_policies = {
            label: StragglerPolicy(**policy) if isinstance(policy, dict) else policy
            for label, policy in self.straggler_policies.items()
        }
        for label, policy in self.straggler_policies.items():
            if not isinstance(label, str) or not label or not isinstance(policy, StragglerPolicy):
                raise ValueError("straggler_policies requires nonempty labels and StragglerPolicy values")
            policy.validate(self.group_size)
        if not 1 <= self.min_successful_group_size <= self.group_size:
            raise ValueError("min_successful_group_size must be in [1, group_size]")
        if self.rollout_isolation not in ("inprocess", "process"):
            raise ValueError("rollout_isolation must be 'inprocess' or 'process'")
        if self.worker_start_method not in ("forkserver", "exec"):
            raise ValueError("worker_start_method must be 'forkserver' or 'exec'")
        for name in (
            "worker_startup_budget_seconds",
            "worker_finalize_grace_seconds",
            "worker_parent_grace_seconds",
            "worker_termination_grace_seconds",
            "worker_binding_poll_seconds",
            "worker_unbounded_rollout_budget_seconds",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.worker_launch_concurrent_groups < 1:
            raise ValueError("worker_launch_concurrent_groups must be >= 1")
        if self.worker_rss_soft_limit_bytes < 0:
            raise ValueError("worker_rss_soft_limit_bytes must be non-negative")
        if self.worker_mem_available_floor_bytes < 0:
            raise ValueError("worker_mem_available_floor_bytes must be non-negative")
        if self.worker_python_env_isolation and self.rollout_isolation != "process":
            # The overlay mutates sys.path, sys.executable and os.environ
            # process-globally and restores them per call, so with concurrent
            # in-process rollouts the first to finish would tear it down under
            # the others -- breaking their imports and every in-flight MCP
            # bridge launch, which uses sys.executable.
            raise ValueError("worker_python_env_isolation requires rollout_isolation='process'")


# What an evaluation during training takes from train.workflow_config, unless its own eval.workflow_config sets it
# (RFC 0018, section 2): how each rollout runs (rollout_config) and how its worker process does.
EVAL_INHERITED_KEYS = (
    "rollout_config",
    "rollout_isolation",
    "worker_start_method",
    "worker_startup_budget_seconds",
    "worker_finalize_grace_seconds",
    "worker_parent_grace_seconds",
    "worker_termination_grace_seconds",
    "worker_launch_concurrent_groups",
    "worker_rss_soft_limit_bytes",
    "worker_mem_available_floor_bytes",
    "worker_binding_poll_seconds",
    "worker_unbounded_rollout_budget_seconds",
    "worker_python_env_isolation",
)
# The keys that shape groups, advantages and filters, which keep evaluation's own defaults unless it sets them.
EVAL_OWN_KEYS = (
    "group_size",
    "leave_one_out_baseline",
    "depth_level_weighting",
    "depth_level_discount_gamma",
    "subagent_datum_keep_probability",
    "subagent_datum_sampling_seed",
    "filter_errors",
    "filter_message_actions",
    "filter_zero_advantage_datums",
    "straggler_timeout_seconds",
    "straggler_quorum",
    "straggler_policies",
    "min_successful_group_size",
)


def eval_workflow_base(sections: Mapping[str, Any], default: Any) -> Any:
    """What a training config's ``eval.workflow_config`` is laid over (``LAID_OVER``): evaluation's ``default`` with
    the built ``train.workflow_config``'s values of :data:`EVAL_INHERITED_KEYS`, each a copy.  The deprecated names of
    renamed rollout keys are left unset there, as their values live on in the keys they were renamed to."""

    train = getattr(sections.get("train"), "workflow_config", None)
    if not isinstance(train, WorkflowConfig) or not isinstance(default, WorkflowConfig):
        return default
    values = {name: copy.deepcopy(getattr(train, name)) for name in EVAL_INHERITED_KEYS}
    rollout = values["rollout_config"]
    renamed = [name for name, entry in deprecated_keys(type(rollout)).items() if entry.renamed_to is not None]
    if renamed:  # a deprecated name is unset when it is None, as each one's default is
        values["rollout_config"] = dataclasses.replace(rollout, **dict.fromkeys(renamed))
    return dataclasses.replace(default, **values)


# Rollouts that run at once during an evaluation when its config sets no max_concurrent_rollouts.
EVAL_ROLLOUTS_AT_ONCE = 256


# The default label column, each task's mixture entry (``platoon.train.components.ENVIRONMENT_COLUMN``, which
# ``tests/train/tinker/test_label_by.py`` checks it against).
ENVIRONMENT_LABEL = "_platoon_environment"


def rollouts_at_once(section: str, config: "TrainConfig | EvalConfig", *, default: int) -> int:
    """The ``max_concurrent_rollouts`` of the ``train`` or ``eval`` section ``config``, with ``default`` for none.  A
    count that holds no whole group of ``workflow_config.group_size`` is a ``ValueError``."""

    group_size = config.workflow_config.group_size
    rollouts = default if config.max_concurrent_rollouts is None else config.max_concurrent_rollouts
    if isinstance(rollouts, bool) or not isinstance(rollouts, int) or rollouts < group_size:
        raise ValueError(
            f"{section}.max_concurrent_rollouts must hold one group's rollouts at least ({section}.workflow_config."
            f"group_size: {group_size}), got {rollouts!r}"
        )
    return rollouts


def concurrent_groups(section: "TrainConfig | EvalConfig") -> int:
    """Rollout groups that run at once in a ``train`` or ``eval`` section: as many whole groups of its
    ``workflow_config.group_size`` as its ``max_concurrent_rollouts`` holds, which building the section sets."""

    rollouts = section.max_concurrent_rollouts
    if rollouts is None:
        raise ValueError("max_concurrent_rollouts is set when the section is built, and this one was not")
    return max(1, rollouts // section.workflow_config.group_size)


# Why a training config names no hosted model (the unknown-key hint of ``hosted``, ``backend`` and ``inference``).
HOSTED_MODELS_EVALUATE_ONLY = (
    "hosted models (inference.hosted) serve evaluation only, since their calls hold no token ids or logprobs to "
    "train on; training samples through a Tinker-compatible service (tinker_base_url)"
)
_HOSTED_HINTS = {"hosted": HOSTED_MODELS_EVALUATE_ONLY, "backend": HOSTED_MODELS_EVALUATE_ONLY}


@dataclass
class RefillConfig:
    """``train.refill``: how far a step under a group admission rule (``train.group_admission``) refills its batch."""

    UNKNOWN_KEY_HINTS: ClassVar[dict[str, str]] = {
        "when_short": "not in this release: a short step trains the groups it admitted"
    }
    # Its cap's checks, when the config loads beside its other problems (platoon.train.admissions).
    VALUE_PROBLEMS: ClassVar[dict[str, Callable[[Any, Mapping[str, Any], Mapping[str, Any], str], list[str]]]] = {
        "max_groups_per_step": lambda value, refill, root, path: _admission_checks().max_groups_problems(
            value, refill, root, path
        ),
    }

    # The most groups a step under a group admission rule counts, as a multiple of train.batch_size (each group it
    # dequeues, and each marker of a group rejected while it was current): it trains the first batch_size admitted
    # groups, and once it has counted ceil(max_groups_per_step × batch_size) it closes and trains what it admitted. A
    # finite number of at least 1. No null: without a cap, a stretch in which every group is rejected (a pool the policy
    # has solved) would never close a step, so set a large number instead; a step that keeps waiting warns each time its
    # count passes another 10 batches. It bounds what a step counts, not the groups that start while the trainer
    # trains, which the horizon and train.max_concurrent_rollouts bound. No effect when every rule is all (it is still
    # checked when the config loads).
    max_groups_per_step: float = DEFAULT_MAX_GROUPS_PER_STEP

    def __post_init__(self) -> None:
        if (problem := max_groups_problem(self.max_groups_per_step)) is not None:
            raise ValueError(f"train.refill.max_groups_per_step: {problem}")


@dataclass
class TrainConfig:
    """The ``train:`` section: the model, the batch, the loss, the optimizer and the rollout workflow."""

    UNKNOWN_KEY_HINTS: ClassVar[dict[str, str]] = {
        **_HOSTED_HINTS,
        # Keys a release removed, with what replaces them (platoon.deprecations).
        "rollout_group_key": removed_hint("0.3.0", "train.label_by"),
        "num_concurrent_rollout_workflow_workers": removed_hint(
            "0.3.0",
            "train.max_concurrent_rollouts, which counts rollouts, not groups: this value × "
            "train.workflow_config.group_size",
        ),
        "on_stale": "not in this release: a stale group is dropped, and under a group admission rule the step takes "
        "another in its place",
    }
    # Keys that still load, with a warning, until the release named removes them (platoon.deprecations).
    DEPRECATED_KEYS: ClassVar[dict[str, DeprecatedKey]] = {
        "task_order": DeprecatedKey(removed_in="0.4.0", renamed_to="task_sampler", former_default="auto"),
    }
    # The task sampler's name and options, checked when the config loads beside its other problems: a name that no
    # package could register, options for a built-in sampler that takes none, and pass_rate's options
    # (platoon.train.samplers).
    VALUE_PROBLEMS: ClassVar[dict[str, Callable[[Any, Mapping[str, Any], Mapping[str, Any], str], list[str]]]] = {
        "task_sampler": lambda value, train, root, path: _sampler_checks().run_sampler_problems(
            value, train, root, path
        ),
        "task_sampler_options": lambda value, train, root, path: _sampler_checks().run_options_problems(
            value, train, root, path
        ),
        # The group admission rule's name and options likewise (platoon.train.admissions).
        "group_admission": lambda value, train, root, path: _admission_checks().run_rule_problems(
            value, train, root, path
        ),
        "group_admission_options": lambda value, train, root, path: _admission_checks().run_options_problems(
            value, train, root, path
        ),
    }

    # Hugging Face model name of the base model, which the Tinker service trains.
    model_name: str
    # Renderer that turns chat messages into the model's tokens: a tinker-cookbook renderer name, or
    # ``chat_template`` for the model's own chat template.
    renderer_name: str
    # Keyword overrides for the renderer, set as its attributes. ``chat_template`` takes ``enable_thinking``,
    # ``preserve_thinking``, ``reasoning_effort``, ``chat_template_kwargs``, ``tool_call_parser`` (``auto``, a
    # parser's name, ``package.module:attr`` or ``none``) and ``reasoning_tags``; see
    # docs/how-to/models-and-tool-calling.md.
    renderer_kwargs: dict = field(default_factory=dict)
    # Seconds one sampling call of the policy may take before it fails, in training's rollouts and in its evaluations
    # (None: TINKER_SAMPLE_TIMEOUT_SECONDS, else 600). The key wins over the variable, which a gateway job may export
    # for every client it runs.
    sample_timeout_seconds: float | None = None
    # The model's context window in tokens; a prompt that does not fit is refused (None: no check).
    context_window_length: int | None = None
    # Tasks per training batch; each task is rolled out ``workflow_config.group_size`` times. ``platoon train``
    # refuses a batch larger than the training set it would draw whole batches from (one environment under
    # ``task_sampler: auto``); ``task_sampler: weighted`` repeats tasks into whole batches.
    batch_size: int = 32
    # Passes over the training tasks. With max_training_steps as well, training runs the shorter of the two;
    # ``platoon train`` refuses a config that sets neither.
    num_epochs: int | None = None
    # Training batches to run. With num_epochs as well, training runs the shorter of the two.
    max_training_steps: int | None = None
    # Number of minibatches per batch. Batch must be divisible by num_minibatches.
    # Results in num_minibatches weight updates per batch.
    num_minibatches: int = 1
    # Number of microbatches per minibatch. Minibatch must be divisible by num_microbatches.
    # Used for gradient accumulation --> A single weight update per microbatch.
    # With tinker, gradient accumulation is less important, but this may still be useful for streaming
    # to overlap rollout sampling with forward backward computation within the same batch.
    num_microbatches: int = 1
    # How many training steps the rollouts of a batch may run ahead of the trained policy, and how many policy
    # versions old a group may be and still train (None: no limit): at most ``max_staleness + 1`` batches of groups are
    # started and not yet trained. An environment entry may raise its own groups' bound
    # (``environments[].max_staleness``); the run's stays the bound on how far rollouts run ahead.
    max_staleness: int | None = None
    optimizer: AdamParams = field(default_factory=AdamParams)
    # Tinker loss function, for example ``cispo`` or ``importance_sampling``.
    loss_fn: str = "cispo"
    # Keyword arguments of the loss function. A config's loss_fn_config replaces this default whole, so it names
    # every key its loss takes (cispo: both thresholds); the Tinker service fills a missing one with its own default.
    loss_fn_config: dict = field(default_factory=lambda: {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0})
    # Rank of the LoRA adapter that is trained.
    lora_rank: int = 32
    workflow_config: WorkflowConfig = field(default_factory=WorkflowConfig)
    # Rollouts that run at once (None: batch_size × workflow_config.group_size, every rollout of a batch). A group's
    # rollouts start together, so as many whole groups run at once as it holds, and it must hold one.
    max_concurrent_rollouts: int | None = None
    # The dataset column that labels each task (None: ``_platoon_environment``, the mixture entry it belongs to,
    # ``platoon.train.components.ENVIRONMENT_COLUMN``; a single environment's tasks have none), such as a Harbor index
    # field. A task's label is its value of the column as text, ``true`` or ``false`` for a boolean.
    # ``rollout_group_limits`` and the workflow configs' ``straggler_policies`` key on labels, and each step's metrics
    # and each evaluation's break down by them (``train/batch/<column>/<label>/...``, ``eval/<column>/<label>/...``;
    # the default column logs as ``environment``), for the 64 most frequent labels; see
    # docs/how-to/train-and-eval.md.
    label_by: str | None = None
    # Most rollout groups that may run at once for each label, for example to cap a slow environment of a mixture. A
    # draw of a label at its limit waits in the label's backlog, which never holds another label's groups back; past
    # four batches of waiting draws, the label's new draws are skipped (see docs/how-to/train-and-eval.md).
    rollout_group_limits: dict[str, int] = field(default_factory=dict)
    # How a run draws its training tasks (platoon.train.auto.AutoDataset; None: ``auto``).  ``auto``: a mixture's
    # tasks in the weighted fair order of its environments (each shuffled with its ``seed``), in epochs of its tasks
    # filled to whole batches, each epoch going on with the order where the one before it stopped; a single
    # environment's tasks in seeded shuffled passes by the dataloader (seed 42), each epoch the whole batches they
    # fill, its last partial batch starting the next epoch, unless the entry sets a ``task_sampler`` of its own (then
    # as ``weighted``).  ``weighted``: each slot of the order goes to an entry by weight, and the entry's own
    # ``task_sampler`` (``uniform`` by default: its tasks shuffled with its ``seed``) chooses the task that fills it; a
    # single environment's tasks too take this order (last batch filled).  ``uniform``: every training task of every
    # entry in seeded shuffled passes (seed 42; a single environment's in the batches ``auto`` draws), with no entry's
    # share, in epochs of its tasks filled to whole batches; it takes no entry's numeric ``weight``.  ``pass_rate``
    # (experimental): each group's task chosen among every training task of the run by its pass rate in the per-task
    # statistics, when the group can start; it takes no numeric ``weight``, and setting ``pass_rate`` on the entries
    # under ``weighted`` keeps each entry's share instead.  Also a name an entry's ``package`` registers with
    # ``@task_sampler``, or ``module:Class``.  The run's ``uniform`` and ``pass_rate`` ask no entry's sampler.  See
    # docs/how-to/train-and-eval.md.
    task_sampler: str | None = None
    # Options of task_sampler, checked against its options when the config loads (pass_rate's: explore,
    # max_draws_per_epoch, max_outstanding_per_task, candidates, seed); ``auto``, ``weighted`` and ``uniform`` take
    # none.
    task_sampler_options: dict[str, Any] = field(default_factory=dict)
    # The former name of ``task_sampler``, read as that key is, with the values it had (``uniform`` came after it).
    # Beside another ``task_sampler``, its former default ``auto`` (as a dump of platoon 0.2.0's ``config show
    # --defaults`` spells it) is ignored, with a warning, and ``weighted`` is an error.
    task_order: Literal["auto", "weighted"] | None = None
    # Which settled training groups count toward the batch (experimental): ``all`` admits every group, as before;
    # ``informative`` rejects the groups whose workflow's verdict says they train nothing (their rollouts scored alike,
    # or no datum) and, with ``band``, those whose pass rate is outside it. A rejected group's place is refilled (at
    # once, unless it carried no datum), so a step trains the first batch_size admitted groups it dequeues that are not
    # stale, within train.refill's cap, and a short step trains what it admitted. Also a name an entry's ``package``
    # registers with ``@group_admission``, or ``module:Class``. An environment entry may set its own
    # (``environments[].group_admission``). Under a rule other than ``all``, num_minibatches must be 1. See
    # docs/how-to/train-and-eval.md.
    group_admission: str = "all"
    # Options of group_admission, checked against its options when the config loads (informative's: band, two numbers
    # 0 <= lo < hi <= 1); ``all`` takes none.
    group_admission_options: dict[str, Any] = field(default_factory=dict)
    # How far a step under a group admission rule refills its batch.
    refill: RefillConfig = field(default_factory=RefillConfig)

    # Client-side deadline in seconds for a ``forward_backward`` result.  The client submits every microbatch up
    # front and the Tinker service runs them in order, so the first wait covers the whole optimizer step's
    # training compute, not one microbatch: size it for a full step of the largest batches you train.
    forward_backward_timeout_seconds: float = 5400.0
    # Deadline in seconds for the optimizer step itself, which is fast; it only guards against a stalled service.
    optim_step_timeout_seconds: float = 300.0

    def __post_init__(self):
        rollouts = self.batch_size * self.workflow_config.group_size  # every rollout of a batch
        self.max_concurrent_rollouts = rollouts_at_once("train", self, default=rollouts)
        seconds = self.sample_timeout_seconds
        if seconds is not None and (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError(
                f"train.sample_timeout_seconds must be a positive number of seconds or null, got {seconds!r}"
            )
        sampler, legacy = self.task_sampler, self.task_order
        if legacy is not None and legacy not in ("auto", "weighted"):  # it takes the values it had before the rename
            raise ValueError(
                f"train.task_order, the deprecated name of train.task_sampler, takes 'auto' or 'weighted', got "
                f"{legacy!r}: set train.task_sampler instead"
            )
        # task_order at its default before the rename (auto, as a dump of 0.2.0's config show --defaults spells it)
        # yields to task_sampler, and the loader's deprecation says that it was ignored; another sampler conflicts.
        if (
            legacy is not None
            and sampler is not None
            and legacy != sampler
            and not TrainConfig.DEPRECATED_KEYS["task_order"].ignored_beside(legacy, sampler)
        ):
            raise ValueError(
                f"train.task_sampler is {sampler!r} and train.task_order, its deprecated name, is {legacy!r}; set "
                "task_sampler alone"
            )
        # The run's one sampler: the deprecated name's value until it is resolved here, then task_sampler's alone.
        sampler = sampler if sampler is not None else legacy if legacy is not None else "auto"
        if not isinstance(sampler, str) or not sampler:  # which names a sampler is the trainer config's to say
            raise ValueError(f"train.task_sampler must name a task sampler, got {sampler!r}")
        self.task_sampler, self.task_order = sampler, None
        if not isinstance(self.task_sampler_options, Mapping):
            raise ValueError(f"train.task_sampler_options must be a mapping, got {self.task_sampler_options!r}")
        self.task_sampler_options = copy.deepcopy(dict(self.task_sampler_options))
        if not isinstance(self.group_admission, str) or not self.group_admission:
            raise ValueError(f"train.group_admission must name a group admission rule, got {self.group_admission!r}")
        if not isinstance(self.group_admission_options, Mapping):
            raise ValueError(f"train.group_admission_options must be a mapping, got {self.group_admission_options!r}")
        self.group_admission_options = copy.deepcopy(dict(self.group_admission_options))
        if isinstance(self.refill, Mapping):  # a section built in Python from a mapping
            self.refill = RefillConfig(**self.refill)
        elif not isinstance(self.refill, RefillConfig):
            raise ValueError(f"train.refill must be a mapping, got {self.refill!r}")
        label_by = self.label_by if self.label_by is not None else ENVIRONMENT_LABEL
        if not isinstance(label_by, str) or not label_by:
            raise ValueError(f"train.label_by must name a dataset column, got {label_by!r}")
        self.label_by = label_by
        if any(
            not isinstance(key, str) or not key or isinstance(value, bool) or not isinstance(value, int) or value < 1
            for key, value in self.rollout_group_limits.items()
        ):
            raise ValueError("rollout_group_limits requires nonempty labels and positive integer capacities")


# The units ``every`` counts in (``TrainEventTriggerConfig.strategy``): training steps, or passes over the training
# tasks.
STRATEGIES = ("step", "epoch")


@dataclass
class TrainEventTriggerConfig:
    """When a periodic training event (a checkpoint, an evaluation) happens: every ``every`` units of ``strategy``,
    training steps unless it says ``epoch``, or never with ``every: 0``, the default.  A config that sets ``strategy``
    and leaves ``every`` out triggers the event every 1 of its units (:func:`default_every`)."""

    # Units of ``strategy`` between events, training steps unless it says epoch; 0 never triggers the event (None: 0,
    # or 1 when the config sets strategy).
    every: int | None = None
    # The unit of ``every``: ``step``, a training step, or ``epoch``, a pass over the training tasks (None: step). A
    # config that sets it and leaves ``every`` out triggers the event every 1 of its units.
    strategy: Literal["step", "epoch"] | None = None

    def _check_every(self, section: str) -> None:
        """Refuse a ``strategy`` that names no unit (``none``, which platoon 0.3.0 removed, with what replaces it),
        resolve an ``every`` the config leaves out (:func:`default_every`), then refuse one that is not a number of its
        units."""

        if self.strategy is not None and self.strategy not in STRATEGIES:
            choices = ", ".join(map(repr, STRATEGIES))
            hint = value_hint(type(self), "strategy", self.strategy)
            raise ValueError(
                f"{section}.strategy: {self.strategy!r} is not one of {choices}" + (f"; {hint}" if hint else "")
            )
        if self.every is None:
            self.every = default_every(self.strategy)
        if isinstance(self.every, bool) or not isinstance(self.every, int) or self.every < 0:
            unit = "passes over the training tasks (strategy: epoch)" if self.strategy == "epoch" else "training steps"
            raise ValueError(f"{section}.every must be a number of {unit}, or 0 for never, got {self.every!r}")


def default_every(strategy: str | None) -> int:
    """``every`` when a config leaves it out: 1 when the config sets ``strategy`` (``step`` or ``epoch``), so that the
    event comes every step or every epoch, as when ``every`` counted in ``strategy``'s units by default; else 0, which
    never triggers the event.  So ``eval: {strategy: step}`` evaluates every step, and ``eval: {}`` never."""

    return 1 if strategy in STRATEGIES else 0


def event_interval(config: TrainEventTriggerConfig, steps_per_epoch: int) -> int | None:
    """Training steps between the events ``config`` triggers, or None when it triggers none: ``every`` steps, or
    ``every`` passes of ``steps_per_epoch`` steps under ``strategy: epoch``."""

    strategy = getattr(config, "strategy", None)
    every = default_every(strategy) if config.every is None else config.every
    if every < 1:
        return None
    return every * steps_per_epoch if strategy == "epoch" else every


@dataclass
class CheckpointConfig(TrainEventTriggerConfig):
    """The ``checkpoint:`` section: when to save checkpoints, how many recovery checkpoints to keep, and which
    checkpoint to start from.

    ``every`` saves a permanent checkpoint every that many training steps (epochs with ``strategy: epoch``); 0, the
    default, saves none beside the launch, recovery and final checkpoints. A run that finds checkpoints in its log
    directory resumes from the newest, weights and optimizer state.
    """

    # Values a release removed, with what replaces them (platoon.utils.config).
    UNKNOWN_VALUE_HINTS: ClassVar[dict[str, dict[Any, str]]] = {
        "strategy": {"none": removed_hint("0.3.0", "checkpoint.every: 0, which saves no permanent checkpoint")}
    }

    # A Tinker checkpoint (``tinker://...``) whose weights a new run starts from, with a fresh optimizer.
    load_checkpoint_path: str | None = None
    # Recovery cadence. ``every`` selects the *permanent* checkpoints (named
    # ``<batch>``, kept forever); ``every: 0`` writes none, leaving the launch,
    # recovery and final checkpoints. When ``recovery_every`` is set, every
    # ``recovery_every`` steps that is not already a permanent save writes a
    # full weights+optimizer checkpoint named ``recovery-<batch>`` instead of
    # the weights-only update, so a crash loses at most that many steps.
    # Recovery checkpoints are disposable: ``keep_recovery`` deletes all but
    # the newest few. ``None`` disables the recovery cadence.
    recovery_every: int | None = None
    # Recovery checkpoints to keep (None: every one). After each recovery save, platoon deletes the run's recovery
    # checkpoints beyond the newest ``keep_recovery`` through the training service, beside training: never a
    # permanent checkpoint, the one a resume starts from, or one this start saved within the run's largest staleness
    # bound (``train.max_staleness``, or an environment's ``max_staleness`` above it) of policy versions, which
    # rollouts may still sample from. It needs ``train.max_staleness``; with ``recovery_every``,
    # ``keep_recovery × recovery_every`` must exceed that largest bound.
    keep_recovery: int | None = None

    def __post_init__(self) -> None:
        if self.recovery_every is not None and self.recovery_every < 1:
            raise ValueError("checkpoint recovery_every must be >= 1 or null")
        self._check_every("checkpoint")
        if self.keep_recovery is not None and (isinstance(self.keep_recovery, bool) or self.keep_recovery < 1):
            raise ValueError("checkpoint.keep_recovery must be at least 1, or null to keep every recovery checkpoint")


@dataclass
class EvalConfig(TrainEventTriggerConfig):
    """The ``eval:`` section of a training config: evaluation on the eval split during training, every ``every``
    training steps (epochs with ``strategy: epoch``); ``every: 0``, the default, turns it off.  ``at_start: true`` also
    evaluates the starting policy, at step 0.  Each evaluation logs the step it started at as ``eval/step``, and the
    run directory's ``evaluations.jsonl`` records each one that completed, so that a start that resumes does not run it
    again."""

    UNKNOWN_KEY_HINTS: ClassVar[dict[str, str]] = {
        **_HOSTED_HINTS,
        "sample_timeout_seconds": "evaluation during training samples the trained policy through training's model "
        "gateway, so train.sample_timeout_seconds bounds its sampling calls too",
        # Keys a release removed, with what replaces them (platoon.deprecations).
        "num_concurrent_rollout_workflow_workers": removed_hint(
            "0.3.0",
            "eval.max_concurrent_rollouts, which counts rollouts, not groups: this value × "
            "eval.workflow_config.group_size, 1 unless set",
        ),
    }
    # Values a release removed, with what replaces them (platoon.utils.config).
    UNKNOWN_VALUE_HINTS: ClassVar[dict[str, dict[Any, str]]] = {
        "strategy": {"none": removed_hint("0.3.0", "eval.every: 0, which never evaluates")}
    }

    # Also evaluate the starting policy, at step 0, and apply the first update only once that evaluation has ended, so
    # that the held-out curve starts at the base model (or at the weights ``checkpoint.load_checkpoint_path`` loads).
    # False: the first evaluation is at step ``every``. It concerns step 0 alone: a start that resumes there runs the
    # evaluation again only when no earlier start completed it. Ignored, with a warning, when ``every`` is 0.
    at_start: bool = False
    # Rollouts that run at once during an evaluation (None: 256). A group's rollouts start together, so as many whole
    # groups of workflow_config.group_size run at once as it holds, and it must hold one.
    max_concurrent_rollouts: int | None = None
    # How evaluation rollouts run. A training config that loads lays the section over ``train.workflow_config``: the
    # keys it leaves out keep training's rollout settings (``rollout_config``, whose ``inference_params`` merge key by
    # key) and rollout-worker settings (``rollout_isolation``, ``worker_start_method`` and the other ``worker_*``
    # keys), and this default's values for the keys that shape groups, advantages and filters (``group_size: 1``,
    # ``filter_errors: false``, ``filter_zero_advantage_datums: false``; the straggler keys, ``leave_one_out_baseline``
    # and the rest as their class has them). A key the section sets replaces training's, ``null`` included: an
    # ``inference_params`` key set to ``null`` leaves it to the gateway's default, and an ``extra`` replaces training's
    # whole. One that sets ``min_successful_group_size`` or a straggler quorum above 1 sets ``group_size`` too. An
    # environment's ``eval_workflow_kwargs`` overrides ``filter_errors``, which only shapes evaluation datums; nothing
    # trains on them.
    workflow_config: WorkflowConfig = field(
        default_factory=lambda: WorkflowConfig(
            group_size=1,
            filter_errors=False,
            filter_zero_advantage_datums=False,
        )
    )

    def __post_init__(self) -> None:
        self._check_every("eval")
        if not isinstance(self.at_start, bool):
            raise ValueError(f"eval.at_start must be true or false, got {self.at_start!r}")
        self.max_concurrent_rollouts = rollouts_at_once("eval", self, default=EVAL_ROLLOUTS_AT_ONCE)


@dataclass
class StatsConfig:
    """The ``stats:`` section: run names and Weights & Biases logging."""

    # The run directory is ``log_path/experiment_name/trial_name``.
    experiment_name: str = "platoon_tinker"
    # The run's name within its experiment.
    trial_name: str = "run"
    wandb: WandBConfig = field(default_factory=WandBConfig)

    def to_stats_logger_config(self, log_dir: str) -> StatsLoggerConfig:
        """Create a StatsLoggerConfig from this stats config."""
        return StatsLoggerConfig(
            experiment_name=self.experiment_name,
            trial_name=self.trial_name,
            log_dir=log_dir,
            wandb=self.wandb,
        )


@dataclass
class WatchdogConfig:
    """The ``watchdog:`` section: a timer that ends a training process that stops making progress, and how many times
    ``platoon train`` starts training again after it fails."""

    enabled: bool = True
    # Seconds without progress after which the watchdog ends the process.
    timeout_seconds: float = 600
    # Exit code of a process the watchdog ends.
    exit_code: int = 2
    # Restarts after a failure (0: none). Above 0, ``platoon train`` trains in a child process and starts it again
    # after any exit but success (0), a configuration error (78) and an interrupt, at most this many times, each start
    # resuming from the newest checkpoint once the run's leftover rollout workers and sandboxes are reaped. It stops
    # after three starts in a row that saved no new checkpoint.
    max_restarts: int = 0

    def __post_init__(self) -> None:
        restarts = self.max_restarts
        if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts < 0:
            raise ValueError(f"watchdog.max_restarts must be a whole number, 0 or more, got {restarts!r}")
        if restarts and self.enabled:
            from platoon.train.supervise import EXIT_CONFIG, INTERRUPTED

            if isinstance(self.exit_code, int) and not 0 <= self.exit_code <= 255:
                raise ValueError(
                    f"watchdog.exit_code: {self.exit_code} is no exit code, which is 0 to 255: a process that exits "
                    f"with it ends with {self.exit_code % 256}, which watchdog.max_restarts takes for that code. Pick "
                    "one such as 2"
                )
            if self.exit_code == 0 or self.exit_code == EXIT_CONFIG or self.exit_code in INTERRUPTED:
                raise ValueError(
                    f"watchdog.exit_code: {self.exit_code} is the exit code of a success, a configuration error or an "
                    "interrupt, after which watchdog.max_restarts restarts nothing: pick another code, such as 2"
                )


def evaluates(config: "PlatoonTinkerRLTrainerConfig") -> bool:
    """Whether training evaluates on the eval split: ``eval.every`` is at least 1 (a config that sets
    ``eval.strategy`` and leaves ``every`` out evaluates every step or epoch)."""

    return event_interval(config.eval, 1) is not None


def training_problems(config: "PlatoonTinkerRLTrainerConfig", *, num_train_tasks: int | None = None) -> list[str]:
    """What stops ``platoon train`` from running a config that loads, one entry each; checked before anything
    runs.  Not part of loading: ``platoon rollout`` and ``platoon config`` take training configs without a
    duration.  With ``num_train_tasks`` (the listed train split), also a batch larger than the training set,
    which the dataloader cannot fill."""

    problems = []
    if config.train.num_epochs is None and config.train.max_training_steps is None:
        problems.append("train: set num_epochs or max_training_steps (with both, training runs the shorter)")
    for section in ("train", "eval"):
        if not getattr(getattr(config, section).workflow_config.rollout_config, "judges", True):
            problems.append(
                f"{section}.workflow_config.rollout_config.judges: false is an evaluation's setting (platoon eval sets "
                "it from inference.judges), and training runs its judges: switch verifiers off with "
                "enable_subagent_verifiers: false or an environment's agents.verifier.enabled: false instead"
            )
    if num_train_tasks is not None and num_train_tasks < config.train.batch_size:
        problems.append(
            f"train.batch_size: {config.train.batch_size} is more than the {num_train_tasks} training tasks, so no "
            "full batch exists; lower train.batch_size, or set train.task_sampler: weighted to repeat tasks into "
            "whole batches"
        )
    # Loading checks these too, but a config class whose __post_init__ skips this class's loads without them.
    problems.extend(entry_settings.problems(config))
    problems.extend(config_pruning_problems(config))
    problems.extend(task_sampler_problems(config))
    problems.extend(admission_problems(config))
    return problems


def task_sampler_problems(config: Any) -> list[str]:
    """What the config's task samplers refuse, one entry each (platoon.train.samplers): what ``train.task_sampler``
    would leave unread (under ``uniform`` or ``pass_rate``, an entry's numeric ``weight``, since they draw among
    every training task with no entry's share, and an entry's own ``task_sampler`` or ``task_sampler_options``, since
    they hand no slots to the entries), and for a config built in code, which never passes the loader's checks, the
    names and options the loader refuses."""

    from platoon.train.samplers import config_sampler_problems

    return config_sampler_problems(config)


def admission_problems(config: Any) -> list[str]:
    """What the config's group admission keys refuse, one entry each (platoon.train.admissions): ``num_minibatches``
    above 1 beside a rule other than ``all``, and for a config built in code, which never passes the loader's checks,
    the names and options the loader refuses."""

    return _admission_checks().config_admission_problems(config)


@dataclass
class PlatoonTinkerRLTrainerConfig:
    """A training config (``platoon train``): what to train on, how, and where the run writes."""

    UNKNOWN_KEY_HINTS: ClassVar[dict[str, str]] = {
        "inference": "inference is an evaluation config's section (platoon eval), not a training config's; "
        + HOSTED_MODELS_EVALUATE_ONLY
    }
    REQUIRED_KEY_HINTS: ClassVar[dict[str, str]] = {
        "log_path": "set it, or runs in the site profile (~/.config/platoon/site.yaml, or the file $PLATOON_SITE "
        "names), which gives a config without log_path its run directory"
    }
    # An evaluation during training rolls out as training does: ``eval.workflow_config`` is laid over
    # ``train.workflow_config``'s rollout and worker settings (``platoon.utils.config.build_config``).
    LAID_OVER: ClassVar[dict[str, Any]] = {"eval.workflow_config": eval_workflow_base}

    train: TrainConfig
    # Evaluation during training; a config without the section does not evaluate (``eval.every`` is 0).
    eval: EvalConfig = field(default_factory=EvalConfig, kw_only=True)
    # Root of the run directories; a run writes to ``log_path/<stats.experiment_name>/<stats.trial_name>``. A config
    # without one takes the site profile's ``runs`` (platoon.site), and then ``stats.experiment_name`` defaults to the
    # config file's name. A relative one is relative to the working directory.
    log_path: str
    # URL of a self-hosted Tinker-compatible service, such as miles-tinker (https://github.com/neulab/miles-tinker);
    # None: ``TINKER_BASE_URL``, else hosted Tinker.
    tinker_base_url: str | None = None
    # The environments to train on: one, or a mixture whose entries each have a ``name``.
    environments: list[EnvironmentConfig] = field(default_factory=lambda: [EnvironmentConfig()])
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)

    def __post_init__(self):
        self.environments = normalize_environment_configs(self.environments)
        # Each entry's own training settings, laid over the run's (platoon.train.entry_settings), and recovery
        # pruning, which keeps the checkpoints within the largest staleness bound of the run and its entries.
        if problems := [
            *entry_settings.problems(self),
            *config_pruning_problems(self),
            *task_sampler_problems(self),
            *admission_problems(self),
        ]:
            raise ConfigError(problems)
