"""The training config's schema: :class:`PlatoonTinkerRLTrainerConfig` and its sections (``train`` and its
``workflow_config``, ``eval``, ``checkpoint``, ``stats``, ``watchdog``), which ``platoon train`` loads and checks, and
which ``docs/reference/config.md`` is generated from (``scripts/gen_reference.py``).  An ``environments`` entry is a
:class:`~platoon.train.components.EnvironmentConfig`.
"""

import copy
import dataclasses
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, get_args

from platoon.config_defs import RolloutConfig
from platoon.deprecations import deprecated_keys, removed_hint
from platoon.train import entry_settings
from platoon.train.admission_api import DEFAULT_MAX_GROUPS_PER_STEP, max_groups_problem
from platoon.train.checkpoints.pruning import config_pruning_problems
from platoon.train.components import (
    STRAGGLER_POLICY_KEY_REMOVED,
    EnvironmentConfig,
    _admission_checks,
    _sampler_checks,
    normalize_environment_configs,
)
from platoon.train.policy.spec import Stepping, policy_problems
from platoon.train.trainer.registry import rolls_out, trainer_problems
from platoon.train.workflow.protocol import StaleFilterSetting
from platoon.utils.config import ConfigError, value_hint
from platoon.utils.stats_logger import StatsLoggerConfig, WandBConfig

# A dotted module name, as worker_preload lists them.
_MODULE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")


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

    UNKNOWN_KEY_HINTS: ClassVar[dict[str, str]] = {
        "max_starting_rollouts": "a key of train, not of a workflow config: train.max_starting_rollouts",
        "stale_filter": "a key of train, not of a workflow config: train.stale_filter",
        # Keys a release removed, with what replaces them (platoon.deprecations).
        "straggler_policy_key": STRAGGLER_POLICY_KEY_REMOVED,
    }

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
    # timeout. A member that started late (train.permit_release: rollout starts
    # a group's members as permits free) gets its start delay on top. It is a
    # deadline against runaway members, not a way to start groups sooner: a
    # cancelled member is censored (neither a pass nor a failure), and slow
    # members fail more often, so a tight cut biases what trains.
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
    # Modules the fork server imports before it forks the rollout workers, beyond those platoon derives from the
    # environments (each service's and environment class's module, each function harness's and program's, and what
    # harness_rollout imports): what an environment or harness imports only when a rollout opens it, which each worker
    # would otherwise import again. A module that does not import, or whose import starts a thread or a process,
    # leaves a file or socket open or seeds a numpy generator from the OS, is left out with a warning, and the workers
    # import it themselves; one that keeps a value of its own process (an id, a temporary directory) does not belong
    # here. Only with rollout_isolation: process and worker_start_method: forkserver, and not under
    # worker_python_env_isolation; elsewhere it has no effect, and a warning says so.
    worker_preload: list[str] = field(default_factory=list)
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
    # Groups -- not rollouts -- allowed to be starting up at once.  A group holds
    # its slot until every member is up or has exited; when its members start
    # as permits free (train.permit_release: rollout), until its last member has
    # a permit and is up.
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
        if not isinstance(self.worker_preload, (list, tuple)) or not all(
            isinstance(name, str) and _MODULE_NAME.fullmatch(name) for name in self.worker_preload
        ):
            raise ValueError(f"worker_preload must be a list of module names (a.b.c), got {self.worker_preload!r}")
        self.worker_preload = list(self.worker_preload)
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
    "worker_preload",
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
# ``tests/train/test_label_by.py`` checks it against).
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


# The ``train:`` keys that only the rl trainer reads (RFC 0021): its batch, its duration, its loss, its stepping and its
# rollouts.  The others declare the policy every trainer trains (``main``: its model, renderer, adapter and optimizer)
# and the deadlines of its updates.  Under another trainer, setting one of these is a load error
# (platoon.train.trainer.registry).
RL_ONLY_TRAIN_KEYS = (
    "batch_size",
    "num_epochs",
    "max_training_steps",
    "num_minibatches",
    "num_microbatches",
    "max_staleness",
    "stale_filter",
    "stepping",
    "loss_fn",
    "loss_fn_config",
    "workflow_config",
    "max_concurrent_rollouts",
    "permit_release",
    "max_starting_rollouts",
    "label_by",
    "rollout_group_limits",
    "task_sampler",
    "task_sampler_options",
    "group_admission",
    "group_admission_options",
    "refill",
)


@dataclass
class TrainConfig:
    """The ``train:`` section: the model, the batch, the loss, the optimizer and the rollout workflow.  Under a trainer
    other than ``rl``, it declares the policy that trainer trains, and the keys only ``rl`` reads stay unset."""

    UNKNOWN_KEY_HINTS: ClassVar[dict[str, str]] = {
        **_HOSTED_HINTS,
        # Keys a release removed, with what replaces them (platoon.deprecations).
        "rollout_group_key": removed_hint("0.3.0", "train.label_by"),
        "num_concurrent_rollout_workflow_workers": removed_hint(
            "0.3.0",
            "train.max_concurrent_rollouts, which counts rollouts, not groups: this value × "
            "train.workflow_config.group_size",
        ),
        "task_order": removed_hint("0.4.0", "train.task_sampler"),
        "on_stale": "not in this release: a stale group is dropped, and under a group admission rule the step takes "
        "another in its place",
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
    # How the stale filter judges a group against its bound (experimental): ``group`` (the default) by its oldest
    # datum, dropping the whole group when that one is too old; ``datum`` each datum by the oldest policy version its
    # own model calls sampled, dropping only the datums that are too old, the others keeping the advantages their
    # group's credit gave them, and the whole group only when every datum is. A rollout that spans a published policy
    # then keeps the datums it sampled since. Under ``datum`` a workflow that does not stamp each datum with its own
    # version (``GroupRolloutWorkflow`` does) has each datum judged by its rollout's (see
    # docs/how-to/train-and-eval.md).
    stale_filter: StaleFilterSetting = "group"
    # How the run's trained policies step, when it trains several (experimental): ``lockstep`` (the default), every
    # policy in each step, on its rows of the step's batch_size groups, each checkpoint saving them all; or
    # ``independent``, each policy on its own clock, with its own batch (``policies.<id>.batch_size``), staleness bound
    # and checkpoint cadence (``policies.<id>.checkpoint``), no policy waiting for another's training or saves, and
    # every full save of any policy a point a resume can start from. With one trained policy both are the same. A
    # resume keeps the run's stepping. Only the rl trainer reads it: trainer: sft, which trains one policy, refuses it.
    # See docs/how-to/train-and-eval.md.
    stepping: Stepping = "lockstep"
    optimizer: AdamParams = field(default_factory=AdamParams)
    # Tinker loss function, for example ``cispo`` or ``importance_sampling``.
    loss_fn: str = "cispo"
    # Keyword arguments of the loss function. A config's loss_fn_config replaces this default whole, so it names
    # every key its loss takes (cispo: both thresholds); the Tinker service fills a missing one with its own default.
    loss_fn_config: dict = field(default_factory=lambda: {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0})
    # Rank of the LoRA adapter that is trained.
    lora_rank: int = 32
    workflow_config: WorkflowConfig = field(default_factory=WorkflowConfig)
    # Rollouts that run at once (None: batch_size × workflow_config.group_size, every rollout of a batch): the rollout
    # permits the groups share, first come first served, each group reserving its group size of them, and it must hold
    # one group. Under permit_release: rollout, the default, each permit comes back as its rollout ends, so it counts
    # the rollouts running; under group a group holds its permits until its slowest rollout ends, so as many whole
    # groups run at once as it holds.
    max_concurrent_rollouts: int | None = None
    # When a group's rollout permits come back: ``rollout`` (the default since 0.4.0), each as its rollout ends, so that
    # the next group starts on the permits that come back; or ``group``, when its workflow returns, so a finished
    # rollout's permit waits for the group's slowest, as before 0.4.0. Under ``rollout``, with a workflow that runs its
    # rollouts on the lease (``GroupRolloutWorkflow`` does), a group starts on its first free permit and its rollouts
    # start as the rest free, each given its start delay on its straggler deadline. More groups are then open at once,
    # so more long groups can go stale: a warning names the fix (see docs/how-to/train-and-eval.md).
    permit_release: Literal["group", "rollout"] = "rollout"
    # Rollouts that may be starting at once (experimental; None: no limit), training's and evaluation's together,
    # first come, first served. A GroupRolloutWorkflow rollout is starting from its start slot, taken after its permit,
    # until its first model call, its end or workflow_config.worker_startup_budget_seconds + 300 s, whichever is first.
    # For start-ups that contend for a shared capacity, such as sandboxes that each extract a root filesystem; it cannot
    # shorten one-time work such as copying a harness toolchain to each node (see docs/how-to/train-and-eval.md).
    max_starting_rollouts: int | None = None
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
        _check_sample_timeout("train.sample_timeout_seconds", self.sample_timeout_seconds)
        sampler = self.task_sampler if self.task_sampler is not None else "auto"
        if not isinstance(sampler, str) or not sampler:  # which names a sampler is the trainer config's to say
            raise ValueError(f"train.task_sampler must name a task sampler, got {sampler!r}")
        self.task_sampler = sampler
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
        if self.permit_release not in ("group", "rollout"):
            raise ValueError(f"train.permit_release must be 'group' or 'rollout', got {self.permit_release!r}")
        if self.stale_filter not in get_args(StaleFilterSetting):
            raise ValueError(f"train.stale_filter must be 'group' or 'datum', got {self.stale_filter!r}")
        if self.stepping not in get_args(Stepping):
            raise ValueError(f"train.stepping must be 'lockstep' or 'independent', got {self.stepping!r}")
        limit = self.max_starting_rollouts
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError(
                f"train.max_starting_rollouts must be a whole number, 1 or more, or null for no limit, got {limit!r}"
            )


def _check_sample_timeout(key: str, seconds: Any) -> None:
    """Refuse a ``sample_timeout_seconds`` (``key``, as the message names it) that is neither null nor a positive
    number of seconds."""

    if seconds is not None and (
        isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0
    ):
        raise ValueError(f"{key} must be a positive number of seconds or null, got {seconds!r}")


@dataclass
class PolicyCheckpointConfig:
    """``policies.<id>.checkpoint``: when a policy the run trains saves full checkpoints, under ``train.stepping:
    independent``, counted in its own steps.  A key left out is the run's (``checkpoint.every``, in steps, and
    ``checkpoint.recovery_every``)."""

    # Its own steps between permanent checkpoints; 0 saves none (None: checkpoint.every, in steps).
    every: int | None = None
    # Its own steps between recovery checkpoints (None: checkpoint.recovery_every). With checkpoint.keep_recovery,
    # keep_recovery × this must exceed the policy's bound (its max_staleness, else the run's largest), which its own
    # pruner keeps.
    recovery_every: int | None = None

    def __post_init__(self) -> None:
        every, recovery = self.every, self.recovery_every
        if every is not None and (isinstance(every, bool) or not isinstance(every, int) or every < 0):
            raise ValueError(f"checkpoint.every must be a number of the policy's steps, or 0 for never, got {every!r}")
        if recovery is not None and (isinstance(recovery, bool) or not isinstance(recovery, int) or recovery < 1):
            raise ValueError(f"checkpoint.recovery_every must be 1 or more, or null, got {recovery!r}")


@dataclass
class PolicyConfig:
    """A ``policies:`` entry: a model the run trains, samples or scores with besides ``main``, under ``train:``'s names
    for the same settings (experimental).  A policy the run trains (``trainable``, the default) steps with main, in
    lockstep (``train.stepping``, the default: each step trains every trained policy on its rows of the step's groups,
    and each checkpoint saves them all) or on its own clock (``independent``: its own batch and checkpoint cadence),
    each on its own Tinker service with its own optimizer and loss.  A frozen one (``trainable: false``) is sampled by
    the agents ``agents:`` binds to it or scores for a credit estimator (a teacher), and is never stepped, saved or
    pruned."""

    # Hugging Face model name of the base model.
    model_name: str
    # Renderer that turns chat messages into the model's tokens, as train.renderer_name.
    renderer_name: str
    # Renderer attribute overrides, as train.renderer_kwargs.
    renderer_kwargs: dict = field(default_factory=dict)
    # Its context window, as train.context_window_length.
    context_window_length: int | None = None
    # The bound on one of its sampling calls, as train.sample_timeout_seconds.
    sample_timeout_seconds: float | None = None
    # Its Tinker service (None: the run's tinker_base_url, else TINKER_BASE_URL, else hosted Tinker).  It opens a
    # session of its own there, also on the run's service.
    tinker_base_url: str | None = None
    # Whether the run trains it (false: a frozen policy, which only samples or scores).
    trainable: bool = True
    # Rank of the LoRA adapter the run trains, as train.lora_rank (None: train.lora_rank).
    lora_rank: int | None = None
    # A trained policy's: the weights its new run starts from (a ``tinker://`` state path; None: its base model's). A
    # frozen policy's: the sampler it serves (a ``tinker://`` sampler path; None: its base model).
    load_checkpoint_path: str | None = None
    # With load_checkpoint_path, a trained policy's new run takes that checkpoint's optimizer state too (Adam's
    # moments): a new trial that goes on from it. Otherwise its optimizer starts fresh.
    load_optimizer_state: bool = False
    # A trained policy's optimizer, as train.optimizer (None: train.optimizer).
    optimizer: AdamParams | None = None
    # The loss a trained policy's rows train under, as train.loss_fn (None: train.loss_fn), and its keyword arguments
    # (None: train.loss_fn_config when the loss is train.loss_fn's, else none, the loss's own defaults).
    loss_fn: str | None = None
    loss_fn_config: dict | None = None
    # How many of the trained policy's own versions old its rows may be and still train (None: their entry's bound,
    # else train.max_staleness). The dispatcher's horizon counts the smallest bound of the trained policies that agents
    # sample, train.max_staleness included when main's is unset.
    max_staleness: int | None = None
    # Under train.stepping: independent, how many of the policy's shares (its rows of a group) each of its steps trains
    # (None: train.batch_size); a smaller batch steps more often.
    batch_size: int | None = None
    # Under train.stepping: independent, when the policy saves full checkpoints, counted in its own steps (None: the
    # run's checkpoint.every and checkpoint.recovery_every, in its steps).
    checkpoint: PolicyCheckpointConfig | None = None

    def __post_init__(self) -> None:
        # As train:'s own; the loader checks the keys' types, as it does train:'s, and names the entry in front of a
        # refusal (policies.<id>: ...), whose id the entry does not know.
        _check_sample_timeout("sample_timeout_seconds", self.sample_timeout_seconds)
        _check_training_keys(self)


def _check_training_keys(policy: PolicyConfig) -> None:
    """Refuse a ``policies:`` entry's training keys that no run could take, as ``train:``'s own are refused (the loader
    checks their types): an adapter rank below 1, a negative staleness bound, an empty loss name and a batch below 1,
    each named by its key; the loader puts the entry's path in front (``policies.worker: lora_rank ...``), since the
    entry does not know its id."""

    rank = policy.lora_rank
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int) or rank < 1):
        raise ValueError(f"lora_rank must be a whole number, 1 or more, or null, got {rank!r}")
    bound = policy.max_staleness
    if bound is not None and (isinstance(bound, bool) or not isinstance(bound, int) or bound < 0):
        raise ValueError(f"max_staleness must be a whole number of versions, 0 or more, or null, got {bound!r}")
    if policy.loss_fn is not None and (not isinstance(policy.loss_fn, str) or not policy.loss_fn):
        raise ValueError(f"loss_fn must name a Tinker loss, or be null, got {policy.loss_fn!r}")
    batch = policy.batch_size
    if batch is not None and (isinstance(batch, bool) or not isinstance(batch, int) or batch < 1):
        raise ValueError(f"batch_size must be a whole number, 1 or more, or null, got {batch!r}")
    if isinstance(policy.checkpoint, Mapping):  # a section built in Python from a mapping
        if unknown := sorted(set(policy.checkpoint) - {"every", "recovery_every"}):
            raise ValueError(f"checkpoint takes every and recovery_every, not {', '.join(map(str, unknown))}")
        policy.checkpoint = PolicyCheckpointConfig(**policy.checkpoint)


@dataclass
class AgentRoleConfig:
    """``agents.roles.<role>``: the policy the agents of that role sample, and the tools they may see and call."""

    # A policy of the run: main, or a policies: entry (None: agents.policy's).
    policy: str | None = None
    # The tools they, and the agents under them, may see and call (a verifier, which runs under the child it judges,
    # takes the lists of that child's spawner instead): names or fnmatch patterns (``read_*``), as agents see them (a
    # task MCP server's as ``<server>__<tool>``); an empty list allows none (None: every tool).
    tools: list[str] | None = None


@dataclass
class AgentsConfig:
    """``agents:``: which policy each agent samples, for its whole life, and which tools it may call (experimental).
    The root agent samples ``policy``; an agent its harness or program opens with a role samples that role's policy,
    and ``policy`` when ``roles`` names none for it.  An agent of a role that lists ``tools`` (the root agent's role is
    ``policy``) sees and calls only those, and so does every agent under it but a verifier, which runs under the child
    it judges and is limited by the lists of that child's spawner and its own role's (``verifier``) instead."""

    # The root agent's policy, and that of every role roles leaves out (None: main).
    policy: str | None = None
    # The policy and the tools of the agents of each role, by role (the role a harness mints a subagent with, such as
    # ``judge``).
    roles: dict[str, AgentRoleConfig] = field(default_factory=dict)


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
    # With load_checkpoint_path: the new run takes the checkpoint's optimizer state too (Adam's moments), a new trial
    # that goes on from it. Its counters still start at 0, and nothing else of the checkpoint's run is taken up.
    load_optimizer_state: bool = False
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
        if self.load_optimizer_state and not self.load_checkpoint_path:
            raise ValueError(
                "checkpoint.load_optimizer_state takes the optimizer state of checkpoint.load_checkpoint_path: set it"
            )


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
        "max_starting_rollouts": "train.max_starting_rollouts limits evaluation's start-ups too: one limit for the "
        "run's rollouts",
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
    if rolls_out(config) and config.train.num_epochs is None and config.train.max_training_steps is None:
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
    problems.extend(trainer_problems(config))
    problems.extend(policy_problems(config))
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
    # The trainer: rl, GRPO-style training on the rollouts of the training tasks, with ``train:``'s settings; or sft,
    # supervised fine-tuning on chat datasets in LLaMA-Factory's conventions, with ``trainer_options`` as its settings
    # (interleaved is reserved for a trainer platoon adds). See docs/how-to/train-and-eval.md.
    trainer: str = field(default="rl", kw_only=True)
    # The trainer's options, checked against the trainer's when the config loads: rl's are the step hooks it attaches
    # to each of its steps, ``hooks: [pkg.mod:Hook, {hook: pkg.mod:Other, options: {...}}]``
    # (docs/extending/step-hooks.md); sft's (dataset, dataset_dir, batch_size, num_epochs, ...) are listed under
    # Trainers in the config reference.
    trainer_options: dict[str, Any] = field(default_factory=dict, kw_only=True)
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
    # The run's other policies, by id (a lowercase name such as ``teacher``), besides main, which train: declares: ones
    # it trains in lockstep with main, and frozen ones its agents sample or its credit estimators score with
    # (experimental).
    policies: dict[str, PolicyConfig] = field(default_factory=dict)
    # Which policy each agent samples, and which tools it may call, by role (experimental); without it, every agent
    # samples main and may call every tool.
    agents: AgentsConfig = field(default_factory=AgentsConfig)

    def __post_init__(self):
        self.environments = normalize_environment_configs(self.environments)
        # Each entry's own training settings, laid over the run's (platoon.train.entry_settings), and recovery
        # pruning, which keeps the checkpoints within the largest staleness bound of the run and its entries.
        if problems := [
            *trainer_problems(self),
            *policy_problems(self),
            *entry_settings.problems(self),
            *config_pruning_problems(self),
            *task_sampler_problems(self),
            *admission_problems(self),
        ]:
            raise ConfigError(problems)
