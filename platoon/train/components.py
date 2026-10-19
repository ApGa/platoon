"""Shared component protocols and environment config for Auto factories."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import math
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, ClassVar, Generic, Literal, Protocol, TypeVar, cast, get_args, runtime_checkable

from platoon.config_defs import InferenceParams
from platoon.task import Task

logger = logging.getLogger(__name__)

T = TypeVar("T")

Placement = Literal["auto", "task_sandbox", "local"]
BuiltinsMode = Literal["default", "keep", "disable"]
ContextOverflowMode = Literal["grade", "exclude"]
# ``environments[].on_harness_exit``: what a harness that ran out of time, or crashed, counts as.
HarnessTimeoutRule = Literal["grade", "exclude"]
HarnessCrashRule = Literal["grade", "grade_if_gateway_clean", "exclude"]
BudgetMode = Literal["linear", "depth_aware"]
ChildGrading = Literal["env", "verifier", "none"]
ToolAccess = Literal["full", "no_terminal", "read_only"]
ForkMode = Literal["shared", "copy", "fresh"]
# ``train.task_sampler`` (:class:`platoon.train.auto.AutoDataset`), how a run draws its training tasks: ``auto``
# orders a mixture's training tasks by weight and leaves a single environment's to the dataloader's shuffle;
# ``weighted`` orders both by weight.
TaskSampler = Literal["auto", "weighted"]
TASK_SAMPLERS: tuple[str, ...] = get_args(TaskSampler)
# ``environments[].weight: tasks``: the entry's share of a mixture is its number of training tasks, so that every task
# of the mixture is drawn equally often.
TASKS_WEIGHT = "tasks"

# In a mixture every task id is ``<environment name>::<task id>``, and each dataset row also names its
# environment in ENVIRONMENT_COLUMN, the column that labels tasks by default (``train.label_by``).
TASK_ID_SEPARATOR = "::"
ENVIRONMENT_COLUMN = "_platoon_environment"
# Columns platoon writes itself are ``task_id`` and those whose names start with this (ENVIRONMENT_COLUMN, the
# training order's marker): an environment's own columns, such as a Harbor task's index fields, never take them.
PLATOON_COLUMN_PREFIX = "_platoon"
# Task.misc key holding the task's EnvironmentRoute; forked child tasks inherit it with the rest of misc.
ROUTE_MISC_KEY = "platoon_environment"
# The rollout an environment with a harness runs unless it names another.
HARNESS_ROLLOUT = "platoon.harness.rollout:harness_rollout"
# Run-level settings: one value for the whole run, which any one mixture entry may set.
RUN_LEVEL_FIELDS = ("trainer_config", "workflow", "workflow_kwargs", "eval_workflow_kwargs")


@runtime_checkable
class DatasetLoader(Protocol):
    """Build a backend dataset or list of task ids for a train/eval split."""

    def __call__(self, config: Any, split: str, **kwargs: Any) -> Any: ...


@runtime_checkable
class TaskLoader(Protocol):
    """Resolve a task id into a Platoon task."""

    def __call__(self, task_id: str) -> Task: ...


@runtime_checkable
class RolloutFn(Protocol):
    """Run one rollout for a task and rollout config."""

    def __call__(self, task: Task, config: Any) -> Any: ...


RewardProcessor = Callable[[dict[str, Any]], tuple[float, dict[str, Any]]]
WorkflowFactory = Callable[..., Any]
TrainerConfigClass = type[Any]
LossFn = Callable[..., Any]


@dataclass
class VerifierConfig:
    """Verifier agents, which grade children when ``agents.child_grading`` is ``verifier``.

    A verifier runs one level below the child it judges, also below ``agents.max_depth``, and may launch one level
    of helpers (``platoon.harness.budget``).
    """

    # Run verifiers; false switches them off for this environment (inference.judges: false switches off every judge
    # of an evaluation).
    enabled: bool = True
    # Step budget of each verifier agent.
    max_steps: int = 20
    # Tool access of each verifier agent.
    access: ToolAccess = "read_only"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"agents.verifier.enabled must be a boolean, got {self.enabled!r}")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 0:
            raise ValueError(f"agents.verifier.max_steps must be a non-negative integer, got {self.max_steps!r}")
        if self.access not in get_args(ToolAccess):
            raise ValueError(f"agents.verifier.access must be one of {get_args(ToolAccess)}, got {self.access!r}")


@dataclass(frozen=True)
class HarnessExitPolicy:
    """What a harness that ran out of time or crashed counts as (``environments[].on_harness_exit``).

    ``exclude`` keeps the episode out of training (a ``timeout`` or ``error`` outcome, platoon's default); ``grade``
    grades the env as it was left, as Harbor does after an agent timeout or a non-zero exit, so a policy that
    loops until its deadline or kills its own harness is scored on what it did.  ``grade_if_gateway_clean`` grades
    a crash unless the model gateway failed one of the agent's calls (a sampling error or timeout, or another 5xx
    answer).  It sees only those: a crash that other infrastructure caused (a harness client that gave up on a
    slow call and disconnected, a broken route to the gateway, a memory watchdog that killed the harness) is
    graded like one the policy caused, so give harness clients a request timeout longer than the gateway's
    sampling timeout (``train.sample_timeout_seconds`` or ``inference.sample_timeout_seconds``, else
    ``TINKER_SAMPLE_TIMEOUT_SECONDS``, else 600 s).  A crash after a context overflow keeps ``on_context_overflow``.
    """

    # A harness that ran past its time limit: ``grade`` or ``exclude``.
    timeout: HarnessTimeoutRule = "exclude"
    # A harness that exited with an error: ``grade``, ``grade_if_gateway_clean`` or ``exclude``.
    crash: HarnessCrashRule = "exclude"

    def __post_init__(self) -> None:
        if self.timeout not in get_args(HarnessTimeoutRule):
            raise ValueError(
                f"on_harness_exit.timeout must be one of {get_args(HarnessTimeoutRule)}, got {self.timeout!r}"
            )
        if self.crash not in get_args(HarnessCrashRule):
            raise ValueError(f"on_harness_exit.crash must be one of {get_args(HarnessCrashRule)}, got {self.crash!r}")

    @classmethod
    def coerce(cls, value: HarnessExitPolicy | Mapping[str, Any] | str | None) -> HarnessExitPolicy:
        """A policy from config: ``{timeout, crash}``, or ``grade``/``exclude`` for both."""

        if value is None:
            return cls()
        if isinstance(value, HarnessExitPolicy):
            return value
        if isinstance(value, str):
            if value not in get_args(HarnessTimeoutRule):
                raise ValueError(
                    f"on_harness_exit must be grade, exclude or a {{timeout, crash}} mapping, got {value!r}"
                )
            rule = cast(HarnessTimeoutRule, value)  # grade or exclude, as checked above: a rule of both kinds
            return cls(timeout=rule, crash=rule)
        if isinstance(value, Mapping):
            unknown = sorted(set(value) - {"timeout", "crash"})
            if unknown:
                raise ValueError(f"on_harness_exit takes timeout and crash, not {unknown}")
            return cls(**value)
        raise ValueError(f"on_harness_exit must be a mapping, got {value!r}")


@dataclass
class AgentTreeConfig:
    """How function harnesses' subagents (``ctx.agents.spawn``) run in one environment."""

    # Deepest level of the policy tree, the root being 0 (None: unlimited). Verifiers do not count against it: the
    # verifier of a child at ``max_depth`` runs one level below it and may delegate once more, so with
    # ``child_grading: verifier`` the tree reaches ``max_depth + 2`` (``platoon.harness.budget``).
    max_depth: int | None = None
    # ``linear``: children draw on their parent's step budget, which reserves ``max_steps + 1`` per spawn;
    # ``depth_aware``: each agent has its own budget.
    budget: BudgetMode = "linear"
    # Step budget of a child whose spawn names none.
    child_max_steps: int = 15
    # ``env``: grade the child's forked session, whose task carries the child's goal; ``verifier``: a verifier
    # agent judges the child; ``none``: the child shares the episode's outcome.
    child_grading: ChildGrading = "none"
    # Tool access of children, never more than their parent's.
    child_access: ToolAccess = "full"
    # Verifier agents, for ``child_grading: verifier``.
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    # Session fork mode of children (None: the environment's default).
    fork: ForkMode | None = None
    # Seconds each child may run (None: only the rollout's timeout applies).
    child_timeout_s: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.verifier, Mapping):
            self.verifier = VerifierConfig(**self.verifier)
        elif not isinstance(self.verifier, VerifierConfig):
            raise ValueError(f"agents.verifier must be a mapping, got {self.verifier!r}")
        if self.max_depth is not None and (
            isinstance(self.max_depth, bool) or not isinstance(self.max_depth, int) or self.max_depth < 0
        ):
            raise ValueError(f"agents.max_depth must be a non-negative integer or null, got {self.max_depth!r}")
        if self.budget not in get_args(BudgetMode):
            raise ValueError(f"agents.budget must be one of {get_args(BudgetMode)}, got {self.budget!r}")
        if (
            isinstance(self.child_max_steps, bool)
            or not isinstance(self.child_max_steps, int)
            or self.child_max_steps < 1
        ):
            raise ValueError(f"agents.child_max_steps must be a positive integer, got {self.child_max_steps!r}")
        if self.child_grading not in get_args(ChildGrading):
            raise ValueError(
                f"agents.child_grading must be one of {get_args(ChildGrading)}, got {self.child_grading!r}"
            )
        if self.child_access not in get_args(ToolAccess):
            raise ValueError(f"agents.child_access must be one of {get_args(ToolAccess)}, got {self.child_access!r}")
        if self.fork is not None and self.fork not in get_args(ForkMode):
            raise ValueError(f"agents.fork must be one of {get_args(ForkMode)} or null, got {self.fork!r}")
        if self.child_timeout_s is not None and (
            isinstance(self.child_timeout_s, bool)
            or not isinstance(self.child_timeout_s, (int, float))
            or not _positive(self.child_timeout_s)
        ):
            raise ValueError(f"agents.child_timeout_s must be a positive number or null, got {self.child_timeout_s!r}")

    @classmethod
    def coerce(cls, value: AgentTreeConfig | Mapping[str, Any] | None) -> AgentTreeConfig:
        if value is None:
            return cls()
        if isinstance(value, AgentTreeConfig):
            return value
        if isinstance(value, Mapping):
            return cls(**value)
        raise ValueError(f"agents must be a mapping, got {value!r}")


class _Inherited:
    """The default of every key of an entry's ``workflow_config``: the value of the settings it is laid over."""

    def __repr__(self) -> str:
        return "inherited"


# A key an entry's ``workflow_config`` leaves out keeps the value of the settings it is laid over.
INHERITED: Any = _Inherited()

# Why an entry's ``workflow_config`` may not set the rest of a workflow config, and where it goes instead.
_RUN_SETTING = "set it in train.workflow_config, or eval.workflow_config for evaluation"
_ENTRY_REFUSED_KEYS = {
    "depth_level_weighting": "an entry does not set it: depth weights are normalized across the tasks of a "
    f"microbatch, which mixes entries' groups; {_RUN_SETTING}",
    "depth_level_discount_gamma": "an entry does not set it: depth weights are normalized across the tasks of a "
    f"microbatch, which mixes entries' groups; {_RUN_SETTING}",
    "filter_zero_advantage_datums": "an entry does not set it: it filters microbatches, which mix entries' groups; "
    f"{_RUN_SETTING}",
    "straggler_policies": "an entry does not set it: straggler policies by label are the run's, in "
    "train.workflow_config.straggler_policies; an entry sets straggler_timeout_seconds and straggler_quorum for its "
    "own groups",
    "straggler_policy_key": "an entry does not set it: straggler policies by label are the run's (train.label_by)",
    **{
        name: f"an entry does not set it: one pool of rollout workers runs every entry's rollouts; {_RUN_SETTING}"
        for name in (
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
    },
}
_TRAINER_SETS = "an entry does not set it: platoon sets it for every rollout"
_ENTRY_REFUSED_ROLLOUT_KEYS = {
    "judges": "an entry does not set it: training runs its judges, and platoon eval sets it from inference.judges; "
    "switch an entry's verifiers off with its agents.verifier.enabled: false",
    "model_name": _TRAINER_SETS,
    "train": _TRAINER_SETS,
    "output_dir": _TRAINER_SETS,
    "return_dict": _TRAINER_SETS,
    "verbose": "an entry does not set it: it is the run's logging, in train.workflow_config.rollout_config",
    "propogate_root_success": "use propagate_root_success",
}


@dataclass
class EntryRolloutSettings:
    """The rollout settings an environment entry may set for its own rollouts: its ``workflow_config.rollout_config``,
    and that of its ``eval_overrides.workflow_config``.  Each key acts as ``RolloutConfig``'s key of the same name, and
    a key the entry leaves out keeps the value of the rollout config it is laid over.  ``inference_params`` merges
    key by key, and ``extra`` replaces the value it is laid over whole; ``null`` sets a key to null, which unsets the
    value it is laid over (``max_steps: null`` takes the task's own step budget)."""

    UNKNOWN_KEY_HINTS: ClassVar[dict[str, str]] = _ENTRY_REFUSED_ROLLOUT_KEYS

    # Step budget of the root agent of each of the entry's rollouts (null: the task's own, else the harness's).
    max_steps: int | None = INHERITED
    # Time limit of the root agent of each of the entry's rollouts, in seconds (null: none of the rollout's own).
    timeout: int | None = INHERITED
    # Per-step deadline in seconds, for harnesses that read one from the rollout config.
    step_timeout: int = INHERITED
    # Default sampling of the entry's model calls, merged key by key over the sampling it is laid over.
    inference_params: InferenceParams = INHERITED
    # Which wins when a policy call sets its own sampling parameters: ``request`` or ``config``.
    sampling_precedence: Literal["request", "config"] = INHERITED
    # Free-form settings for custom rollouts and harnesses; replaces the mapping it is laid over whole.
    extra: dict[str, Any] = INHERITED
    # Give every trajectory of a rollout the root's success.
    propagate_root_success: bool | None = INHERITED
    # Score every subagent 0.
    skip_subagent_reward_computation: bool = INHERITED
    # Run verifier subagents and behavior judges on completed children.
    enable_subagent_verifiers: bool = INHERITED


@dataclass
class EntryWorkflowSettings:
    """The training settings an environment entry may set for its own groups (``environments[].workflow_config``),
    laid over ``train.workflow_config`` for its training groups, and for its evaluation rollouts what its
    ``eval_overrides.workflow_config`` sets, laid over the evaluation's settings for the entry.  Each key acts as
    ``WorkflowConfig``'s key of the same name, on the entry's groups only, and a key the entry leaves out keeps the
    value of the settings it is laid over.  They are the keys that act on one group or one rollout: the rest of a
    workflow config acts on microbatches, which mix entries' groups, or on the run's one pool of rollout workers, and
    stays the run's."""

    UNKNOWN_KEY_HINTS: ClassVar[dict[str, str]] = _ENTRY_REFUSED_KEYS

    # Rollouts per task, for each of the entry's groups.
    group_size: int = INHERITED
    # Center each rollout's reward on the mean of the other members of its group rather than the whole group.
    leave_one_out_baseline: bool = INHERITED
    # Fewest trainable rollouts one of the entry's groups needs to train.
    min_successful_group_size: int = INHERITED
    # The entry's straggler policy: once ``straggler_quorum`` members of a group have settled, wait at most this many
    # more seconds for the rest (null: no cutoff). A label's ``train.workflow_config.straggler_policies`` entry wins.
    straggler_timeout_seconds: float | None = INHERITED
    # Settled members that start the straggler timeout (null: group_size - 1).
    straggler_quorum: int | None = INHERITED
    # Suppress error tokens that would get positive credit, in the entry's rollouts.
    filter_errors: bool = INHERITED
    # Retain agent messages as context, with no policy loss for either sign.
    filter_message_actions: bool = INHERITED
    # Keep each subagent datum of the entry's rollouts with this probability.
    subagent_datum_keep_probability: float = INHERITED
    # Seed of the subagent datum sampling.
    subagent_datum_sampling_seed: int = INHERITED
    # The entry's rollout settings, laid over the rollout config its settings are laid over, key by key.
    rollout_config: EntryRolloutSettings = INHERITED


# The keys an entry's ``workflow_config`` may set, and those of its ``rollout_config``.
ENTRY_WORKFLOW_KEYS: tuple[str, ...] = tuple(item.name for item in fields(EntryWorkflowSettings))
ENTRY_ROLLOUT_KEYS: tuple[str, ...] = tuple(item.name for item in fields(EntryRolloutSettings))


def entry_workflow_problems(value: Any, path: str) -> list[str]:
    """What is wrong with ``value`` as an entry's ``workflow_config`` (or its ``eval_overrides.workflow_config``) at
    the dotted ``path``, one problem each: a key it may not set, with why and where it goes, and a value of the wrong
    type."""

    from platoon.utils.config import validate_config

    if not isinstance(value, Mapping):
        return [f"{path}: expected a mapping (EntryWorkflowSettings), got {value!r}"]
    return validate_config(dict(value), EntryWorkflowSettings, path=path)


@dataclass
class EvalOverrides:
    """What an ``environments`` entry changes for its evaluation rollouts (``eval_overrides``): those of every split
    but ``train``, that is the rollouts of ``platoon eval`` and those ``platoon train`` runs when it evaluates.  Each
    key merges over the entry's own key of the same name, key by key, and is checked as the entry's own is."""

    # Keys people reach for here that are not options, with where they go instead (``platoon.utils.config``).
    UNKNOWN_KEY_HINTS: ClassVar[dict[str, str]] = {
        "max_steps": "an entry's step limit for its evaluation rollouts is eval_overrides.workflow_config."
        "rollout_config.max_steps; every entry's is eval.workflow_config.rollout_config.max_steps (platoon train) or "
        "inference.workflow.rollout_config.max_steps (platoon eval)",
        "inference_params": "an entry's sampling for its evaluation rollouts is eval_overrides.workflow_config."
        "rollout_config.inference_params; every entry's is eval.workflow_config.rollout_config.inference_params "
        "(platoon train) or inference.workflow.rollout_config.inference_params (platoon eval)",
        "group_size": "an entry's rollouts per task in evaluation are eval_overrides.workflow_config.group_size",
        "dataset_kwargs": "the evaluation tasks come from the entry's eval_dataset_kwargs",
    }
    # ``workflow_config`` is a mapping as given, checked as the settings an entry may set (platoon.utils.config).
    FIELD_SCHEMAS: ClassVar[dict[str, Any]] = {"workflow_config": EntryWorkflowSettings}

    # Harness options of evaluation rollouts, merged over the entry's ``harness_options``.
    harness_options: dict[str, Any] = field(default_factory=dict)
    # The entry's settings for its evaluation rollouts, laid over the evaluation's settings for the entry: the keys of
    # its own workflow_config (EntryWorkflowSettings), such as ``group_size`` and ``rollout_config.max_steps``. They
    # win over the phase's section (eval.workflow_config; platoon eval applies only their rollout_config, over
    # inference.workflow.rollout_config).
    workflow_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.harness_options, Mapping):
            raise ValueError(f"eval_overrides.harness_options must be a mapping, got {self.harness_options!r}")
        self.harness_options = dict(self.harness_options)
        if problems := entry_workflow_problems(self.workflow_config, "eval_overrides.workflow_config"):
            raise ValueError("; ".join(problems))
        self.workflow_config = copy.deepcopy(dict(self.workflow_config))

    @classmethod
    def coerce(cls, value: EvalOverrides | Mapping[str, Any] | None) -> EvalOverrides:
        if value is None:
            return cls()
        if isinstance(value, EvalOverrides):
            return value
        if isinstance(value, Mapping):
            unknown = sorted(set(value) - {item.name for item in fields(cls)})
            if unknown:
                raise ValueError(f"eval_overrides takes {[item.name for item in fields(cls)]}, not {unknown}")
            return cls(**value)
        raise ValueError(f"eval_overrides must be a mapping, got {value!r}")


@dataclass
class EnvironmentConfig:
    """One entry of ``environments``: an environment service, the harness that plays its tasks, and the
    components that load, roll out and score them.

    A run trains on one environment or on a weighted mixture of several; in a mixture each entry needs a unique
    ``name``. An entry with a harness rolls out with ``harness_rollout`` unless it names another rollout, and an
    entry with a service lists and loads its tasks through the service unless it names loaders. Components
    (loaders, rollouts, reward processors, trainer configs) are named as ``module:attr`` or by a name that the
    entry's ``package`` registers.
    """

    # The service options that are going away, as the service the entry names lists them (its class's
    # DEPRECATED_KEYS), which the config loader reports as it reports deprecated keys (platoon.deprecations).
    DEPRECATED_OPTIONS: ClassVar[dict[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]]] = {
        "service_kwargs": lambda entry: _service_deprecated_options(entry.get("service"))
    }
    # ``workflow_config`` is a mapping as given, checked as the settings an entry may set (platoon.utils.config).
    FIELD_SCHEMAS: ClassVar[dict[str, Any]] = {"workflow_config": EntryWorkflowSettings}

    # A named environment, ``name[@version]`` (or ``module:attr``): a partial entry that an installed extension
    # registers in the ``platoon.environments`` entry-point group, laid under this entry's own keys when the
    # config loads (``platoon.envs.named``); the loaded config pins the version it resolved to.
    env: str | None = None
    # Module imported before the entry's components resolve, so its registrations (``@pt.harness``,
    # ``@pt.program``, components) take effect; also imported in every rollout worker process.  An installed
    # extension's components need none: they load by name from their entry points.
    package: str | None = None
    # Training-config class that adds this plugin's config sections, a subclass of the training config, as
    # ``module:Class`` or a name that one of the environments' ``package`` modules registers.  It is read from the
    # config files, before the overrides; one that does not resolve is a configuration error.
    trainer_config: str | None = None
    # Loader of the training tasks (None: the service's tasks of the split).
    dataset_loader: str | None = None
    # Loader of the evaluation tasks (None: dataset_loader).
    eval_dataset_loader: str | None = None
    # Loader that turns a task id into a ``Task`` (None: the service's).
    task_loader: str | None = None
    # Rollout function of training tasks (None: ``harness_rollout`` when the entry has a harness).
    rollout: str | None = None
    # Rollout function of evaluation tasks (None: rollout).
    eval_rollout: str | None = None
    # Function that turns a finished trajectory into its reward and reward components (None: the trajectory's
    # own reward).
    reward_processor: str | None = None
    # Training workflow: how rollout groups become training data.
    workflow: str = "group_rollout"
    # Keyword arguments of dataset_loader.
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    # Keyword arguments of eval_dataset_loader.
    eval_dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    # Tasks to leave out of the entry's training and evaluation tasks, each a task id (an ORS environment's
    # ``<split>:<index>``, a Harbor task's directory name, a task table's row id) or an ORS task's name. They are
    # dropped from the listed tasks before ``limit`` takes its share, and the other tasks keep their ids. An item
    # that names no task of the environment is a configuration error; the run's ``lock.json`` records the ids
    # left out. A named environment (``env``) cannot set it; the entry that selects one does.
    exclude_tasks: list[str] = field(default_factory=list)
    # Keyword arguments of the training workflow.
    workflow_kwargs: dict[str, Any] = field(default_factory=dict)
    # Keyword arguments of the evaluation workflow.
    eval_workflow_kwargs: dict[str, Any] = field(default_factory=dict)
    # The entry's name, required and unique in a mixture; a mixture's task ids are ``<name>::<task id>``.
    name: str | None = None
    # The entry's share of a mixture's training tasks: ``tasks``, its number of training tasks (once selection,
    # ``exclude_tasks`` and a hold-out have taken theirs), which draws every task of the mixture equally often; or a
    # positive number, the entries' shares in proportion to their numbers. A mixture's weights are all numbers or
    # all ``tasks``.
    weight: float | Literal["tasks"] = "tasks"
    # The entry's own training settings, laid over ``train.workflow_config`` for its training groups: the keys that act
    # on one group or one rollout (EntryWorkflowSettings), such as ``group_size``, ``min_successful_group_size``,
    # ``leave_one_out_baseline`` and ``rollout_config.max_steps``. A key it leaves out keeps the run's; ``null`` unsets
    # it. Its rollout settings are also its evaluation's defaults, below the phase's section (eval.workflow_config,
    # inference.workflow), and ``eval_overrides.workflow_config`` sets what its evaluation rollouts change. ``platoon
    # config show`` prints each entry's effective settings and where each came from.
    workflow_config: dict[str, Any] = field(default_factory=dict)
    # How many policy versions old the entry's training groups may be and still train (null: the run's,
    # ``train.max_staleness``): the stale filter's bound for its groups. At least the run's, which stays the bound on
    # how far rollouts run ahead of training: an entry whose groups take many steps raises its own. Recovery pruning
    # keeps the checkpoints within the largest bound of the run and its entries.
    max_staleness: int | None = None
    # The lowest reward that counts a training rollout of the entry as a pass, in the per-task statistics and their
    # metrics (platoon.train.task_stats), judged as platoon eval judges success: on the reward the entry's environment
    # scored, before any reward processor. A partial-credit environment sets its own.
    success_threshold: float = 1.0
    # The environment service: ``harbor``, ``table``, ``python://pkg.module:EnvClass`` (an ORS environment run
    # in process), ``ors+http://host:port/EnvName`` (ORS environment servers) or ``module:Class``.
    service: str | None = None
    # Keyword arguments of the service.
    service_kwargs: dict[str, Any] = field(default_factory=dict)
    # The harness that plays each task, by reference (``name[@version]``, a ``.toml`` spec path or
    # ``module:function``), or a ``{harness: weight}`` mapping drawn per task with ``seed``.
    harness: str | dict[str, float] | None = None
    # Where a harness spec runs: ``task_sandbox``, ``local``, or ``auto`` (the task sandbox when the environment
    # offers one).
    placement: Placement = "auto"
    # A harness spec's built-in tools: ``keep``, ``disable``, or ``default`` (keep them in a task sandbox and
    # disable them elsewhere, where the harness allows).
    builtins: BuiltinsMode = "default"
    # What a harness that fails after the gateway refused an overflowing prompt counts as: ``grade`` scores it
    # as a policy failure (reason ``context_overflow``), ``exclude`` keeps it out of training.
    on_context_overflow: ContextOverflowMode = "grade"
    # What a harness that ran out of time or crashed counts as: ``{timeout: grade | exclude, crash: grade |
    # grade_if_gateway_clean | exclude}``, or ``grade``/``exclude`` for both (default: exclude both).
    on_harness_exit: HarnessExitPolicy = field(default_factory=HarnessExitPolicy)
    # Seed of the per-task harness draw and of the entry's task order in a mixture.
    seed: int = 0
    # The episode program (``@pt.program`` name or ``module:attr``).
    program: str = "single"
    # The harness's options: a function harness reads them as ``ctx.options``; a TOML spec checks them against its
    # ``[options]`` table and renders them into its launch (``platoon harness show NAME`` lists them).
    harness_options: dict[str, Any] = field(default_factory=dict)
    # What evaluation rollouts change (every split but ``train``): ``{harness_options: {...}}``, merged over the
    # entry's own, key by key.
    eval_overrides: EvalOverrides = field(default_factory=EvalOverrides)
    # How the function harness's subagents run.
    agents: AgentTreeConfig = field(default_factory=AgentTreeConfig)

    def __post_init__(self) -> None:
        where = f"environment {self.name!r}" if self.name else "environment"
        if not isinstance(self.program, str) or not self.program.strip():
            raise ValueError(f"{where}: program must be a non-empty program name, got {self.program!r}")
        if not isinstance(self.harness_options, Mapping):
            raise ValueError(f"{where}: harness_options must be a mapping, got {self.harness_options!r}")
        self.harness_options = dict(self.harness_options)
        try:
            self.eval_overrides = EvalOverrides.coerce(self.eval_overrides)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: {exc}") from exc
        if problems := entry_workflow_problems(self.workflow_config, "workflow_config"):
            raise ValueError(f"{where}: " + "; ".join(problems))
        self.workflow_config = copy.deepcopy(dict(self.workflow_config))
        threshold = self.success_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError(f"{where}: success_threshold must be a finite number, got {threshold!r}")
        self.success_threshold = float(threshold)
        staleness = self.max_staleness
        if staleness is not None and (isinstance(staleness, bool) or not isinstance(staleness, int) or staleness < 0):
            raise ValueError(
                f"{where}: max_staleness must be a whole number of policy versions, 0 or more, or null for the run's "
                f"(train.max_staleness), got {staleness!r}"
            )
        try:
            self.agents = AgentTreeConfig.coerce(self.agents)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: {exc}") from exc
        if self.name is not None and (not self.name.strip() or TASK_ID_SEPARATOR in self.name):
            raise ValueError(f"{where}: name must be non-empty and must not contain {TASK_ID_SEPARATOR!r}")
        if self.weight != TASKS_WEIGHT and (
            isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)) or not _positive(self.weight)
        ):
            raise ValueError(f"{where}: weight must be a finite positive number or {TASKS_WEIGHT}, got {self.weight!r}")
        if self.placement not in get_args(Placement):
            raise ValueError(f"{where}: placement must be one of {get_args(Placement)}, got {self.placement!r}")
        if self.builtins not in get_args(BuiltinsMode):
            raise ValueError(f"{where}: builtins must be one of {get_args(BuiltinsMode)}, got {self.builtins!r}")
        if self.on_context_overflow not in get_args(ContextOverflowMode):
            raise ValueError(
                f"{where}: on_context_overflow must be one of {get_args(ContextOverflowMode)}, "
                f"got {self.on_context_overflow!r}"
            )
        try:
            self.on_harness_exit = HarnessExitPolicy.coerce(self.on_harness_exit)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: {exc}") from exc
        if isinstance(self.harness, Mapping):
            if not self.harness or not all(
                isinstance(weight, (int, float)) and not isinstance(weight, bool) and _positive(weight)
                for weight in self.harness.values()
            ):
                raise ValueError(f"{where}: a harness mapping needs at least one spec, each with a positive weight")
            self.harness = dict(self.harness)
        elif self.harness is not None and (not isinstance(self.harness, str) or not self.harness):
            raise ValueError(f"{where}: harness must be a spec reference or a {{spec: weight}} mapping")
        if self.harness is not None and self.service is None:
            raise ValueError(f"{where}: a harness needs an environment `service` to open sessions with")
        if isinstance(self.exclude_tasks, (str, Mapping)) or not isinstance(self.exclude_tasks, Sequence):
            raise ValueError(f"{where}: exclude_tasks must be a list of task ids or names, got {self.exclude_tasks!r}")
        if bad := [item for item in self.exclude_tasks if not _names_a_task(item)]:
            raise ValueError(
                f"{where}: exclude_tasks items must be task ids or names, non-empty strings without surrounding "
                f"spaces; got {bad}"
            )
        self.exclude_tasks = list(dict.fromkeys(self.exclude_tasks))  # in order, each once

    @property
    def label(self) -> str:
        """The environment's name in task ids, dataset rows and routes."""

        return self.name or "default"

    def harness_refs(self) -> list[str]:
        """Every harness spec this environment may run."""

        if self.harness is None:
            return []
        return [self.harness] if isinstance(self.harness, str) else sorted(self.harness)


def _service_deprecated_options(service: Any) -> Mapping[str, Any]:
    """The deprecated options of the service an entry names (:func:`platoon.envs.service.service_deprecated_options`);
    none for an entry without one."""

    if not isinstance(service, str) or not service:
        return {}
    from platoon.envs.service import service_deprecated_options

    return service_deprecated_options(service)


def _positive(value: float) -> bool:
    return math.isfinite(value) and value > 0


def _names_a_task(item: Any) -> bool:
    """Whether ``item`` can be an ``exclude_tasks`` item: a non-empty string without surrounding spaces."""

    return isinstance(item, str) and bool(item) and item == item.strip()


def normalize_environment_configs(environments: Any) -> list[EnvironmentConfig]:
    """Normalize the public `environments` config list and validate it as a mixture."""

    if environments is None:
        return []
    if isinstance(environments, EnvironmentConfig):
        raise ValueError("`environments` must be a list; use `environments: [{...}]` for a single environment")
    if isinstance(environments, dict):
        raise ValueError("`environments` must be a list; use `environments: - ...` for a single environment")
    if not isinstance(environments, (list, tuple)):
        raise TypeError("`environments` must be a list of environment configs")

    normalized: list[EnvironmentConfig] = []
    for environment in environments:
        if isinstance(environment, EnvironmentConfig):
            normalized.append(environment)
        elif isinstance(environment, dict):
            if environment.get("env") is not None:  # built from plain data, not loaded from a file
                from platoon.envs.named import expand_environment

                environment = expand_environment(environment)
            normalized.append(EnvironmentConfig(**environment))
        else:
            raise TypeError("Each `environments` entry must be an EnvironmentConfig or dict")
    validate_environment_mixture(normalized)
    return normalized


@dataclass
class EnvironmentsConfig:
    """A config of ``environments`` alone: a file that training and evaluation configs extend, or the environments of
    one of them, read without its other sections (``platoon prepare``).  Its entries are checked as either kind of
    config checks them, for both phases, training and evaluation."""

    # The environments, as a training or evaluation config lists them.
    environments: list[EnvironmentConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.environments = normalize_environment_configs(self.environments)


def validate_environment_mixture(environments: Sequence[EnvironmentConfig]) -> None:
    """A mixture needs unique names, weights that are all numbers or all ``tasks``, and at most one value for each
    run-level setting."""

    if len(environments) < 2:
        return
    if any(environment.name is None for environment in environments):
        raise ValueError("Every entry of a multi-environment `environments` list needs a `name`")
    labels = [environment.label for environment in environments]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"Environment names must be unique; repeated: {duplicates}")
    by_tasks = [environment.label for environment in environments if environment.weight == TASKS_WEIGHT]
    if by_tasks and len(by_tasks) < len(environments):
        by_numbers = [environment.label for environment in environments if environment.weight != TASKS_WEIGHT]
        raise ValueError(
            f"Mixture entries {by_tasks} weigh by their tasks (weight: {TASKS_WEIGHT}, the weight of an entry that "
            f"sets none) and {by_numbers} by numbers: a mixture's weights are all numbers, which share its slots in "
            f"proportion, or all {TASKS_WEIGHT}, which draws every task equally often; give every entry a number (1 "
            "each for equal shares), or no entry one"
        )
    for name in RUN_LEVEL_FIELDS:
        values = _set_values(environments, name)
        if len(values) > 1:
            raise ValueError(f"Mixture entries set different `{name}` values {values}; set it on one entry")


def merge_run_level_settings(environments: Sequence[EnvironmentConfig]) -> EnvironmentConfig:
    """The first entry, carrying the run-level settings that any entry of the mixture sets."""

    first = environments[0]
    overrides = {
        name: values[0]
        for name in RUN_LEVEL_FIELDS
        if (values := _set_values(environments, name)) and values[0] != getattr(first, name)
    }
    return replace(first, **overrides) if overrides else first


def _set_values(environments: Sequence[EnvironmentConfig], name: str) -> list[Any]:
    default = _FIELD_DEFAULTS[name]
    values: list[Any] = []
    for environment in environments:
        value = getattr(environment, name)
        if value != default and value not in values:
            values.append(value)
    return values


_FIELD_DEFAULTS = {
    item.name: item.default_factory() if callable(item.default_factory) else item.default
    for item in fields(EnvironmentConfig)
}


def row_entry(row: Any, labels: Sequence[str]) -> str | None:
    """The label of the environment entry whose task a dataset row holds: its :data:`ENVIRONMENT_COLUMN`, which every
    row of a mixture has, else the one entry of a run of a single environment (``labels``, the run's entries); None
    for a row of no entry of the run."""

    label = row.get(ENVIRONMENT_COLUMN) if isinstance(row, Mapping) else None
    if isinstance(label, str) and label in labels:
        return label
    return labels[0] if len(labels) == 1 and label is None else None


def platoon_column(name: str) -> bool:
    """Whether a dataset column is one platoon writes itself (``task_id``, ``_platoon*``), which an environment's
    own columns may not take."""

    return name == "task_id" or name.startswith(PLATOON_COLUMN_PREFIX)


def labeled_task_id(label: str, task_id: str) -> str:
    return f"{label}{TASK_ID_SEPARATOR}{task_id}"


def split_task_id(task_id: str) -> tuple[str, str]:
    """``(environment name, task id)`` of a mixture task id."""

    label, separator, raw_id = task_id.partition(TASK_ID_SEPARATOR)
    if not separator:
        raise ValueError(f"Mixture task id {task_id!r} does not start with '<environment>{TASK_ID_SEPARATOR}'")
    return label, raw_id


def choose_harness(harness: str | Mapping[str, float] | None, *, task_id: str, seed: int) -> str | None:
    """The harness spec for one task: the only one, or a weighted draw keyed by ``(seed, task_id)``.

    Every rollout of a task (the whole group, every epoch) gets the same harness.
    """

    if harness is None or isinstance(harness, str):
        return harness
    names = sorted(harness)
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
    point = int.from_bytes(digest[:8], "big") / 2**64 * sum(harness[name] for name in names)
    for name in names:
        point -= harness[name]
        if point < 0:
            return name
    return names[-1]


def mixture_weights(environments: Sequence[EnvironmentConfig], tasks: Mapping[str, int]) -> dict[str, float]:
    """Each entry's weight as a number, by label: its own, or for ``weight: tasks`` its number of training tasks
    (``tasks``, by label)."""

    return {
        environment.label: float(tasks.get(environment.label, 0))
        if environment.weight == TASKS_WEIGHT
        else float(environment.weight)
        for environment in environments
    }


class WeightedOrder(Generic[T]):
    """An endless weighted fair order of items across groups, read one epoch of ``epoch_size`` items at a time.

    Each slot goes to the group with the most credit: every group gains its share of the weights per slot and the
    chosen one pays 1, ties rotating from the group after the last one chosen, so the groups interleave evenly and
    keep their shares over any stretch of the order.  Each group's items are drawn in seeded shuffled passes, every
    item once per pass, so a group that gets more slots than it has items repeats them, and one that gets fewer
    draws every item before any repeats.  An epoch is the next ``epoch_size`` items: it takes up the credits and
    each group's pass where the epoch before it stopped, so a subsampled group draws its other items in the next
    epochs instead of the same ones again.

    The shuffles follow a fixed rule: pass ``p`` of the ``i``-th non-empty group is
    ``random.Random(seeds[label] + i * 10_007 + p * 101).shuffle``, taken from the front, so the same groups,
    weights, seeds and epoch size give the same order, epoch for epoch.  Reading epochs in sequence costs each one
    its own items; reading an earlier epoch than the last one read draws the order again from its start.
    """

    # The epoch read last, with its index.
    _last: tuple[int, list[T]] | None

    def __init__(
        self,
        groups: Mapping[str, Sequence[T]],
        weights: Mapping[str, float],
        *,
        seeds: Mapping[str, int],
        epoch_size: int,
    ) -> None:
        labels = [label for label in groups if groups[label]]
        if not labels:
            raise ValueError("Cannot interleave a mixture whose environments have no tasks")
        if isinstance(epoch_size, bool) or not isinstance(epoch_size, int) or epoch_size < 1:
            raise ValueError(f"epoch_size must be a positive integer, got {epoch_size!r}")
        total = sum(weights[label] for label in labels)
        self._shares = [weights[label] / total for label in labels]
        self._groups = [list(groups[label]) for label in labels]
        self._seeds = [_seed(seeds[label], label) + index * _GROUP_SEED_STRIDE for index, label in enumerate(labels)]
        self._epoch_size = epoch_size
        self._restart()

    @property
    def epoch_size(self) -> int:
        """Items per epoch."""

        return self._epoch_size

    @property
    def items(self) -> list[T]:
        """Every item the order draws from, once each, group by group: those an epoch leaves out included."""

        return [item for group in self._groups for item in group]

    def epoch(self, index: int) -> list[T]:
        """The items of epoch ``index`` (0 is the first), in order."""

        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"An epoch index is a whole number, 0 or more, got {index!r}")
        if self._last is not None and self._last[0] == index:
            return list(self._last[1])
        if index < self._next_epoch:
            self._restart()
        while True:
            items = list(itertools.islice(self._stream, self._epoch_size))
            self._last = (self._next_epoch, items)
            self._next_epoch += 1
            if self._last[0] == index:
                return list(items)

    def _restart(self) -> None:
        self._stream = self._items()
        self._next_epoch = 0
        self._last = None

    def _items(self) -> Iterator[T]:
        count = len(self._groups)
        pools = [_CyclingPool(items, seed=seed) for items, seed in zip(self._groups, self._seeds)]
        credits = [0.0] * count
        cursor = 0
        while True:
            credits = [credit + share for credit, share in zip(credits, self._shares)]
            best = max(credits)
            chosen = next(
                index
                for offset in range(count)
                if math.isclose(credits[index := (cursor + offset) % count], best, rel_tol=0.0, abs_tol=1e-12)
            )
            credits[chosen] -= 1.0
            cursor = (chosen + 1) % count
            yield pools[chosen].next()


def interleave_by_weight(
    groups: Mapping[str, Sequence[T]], weights: Mapping[str, float], *, size: int, seeds: Mapping[str, int]
) -> list[T]:
    """The first ``size`` items of the :class:`WeightedOrder` of ``groups``: the first epoch of a training order
    whose epochs hold ``size`` items."""

    return WeightedOrder(groups, weights, seeds=seeds, epoch_size=size).epoch(0)


# The shuffles' seed strides: per group (its index among the non-empty groups) and per pass through a group.
_GROUP_SEED_STRIDE = 10_007
_PASS_SEED_STRIDE = 101


def _seed(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"environment {label!r}: seed must be an integer, got {value!r}")
    return value


class _CyclingPool(Generic[T]):
    """A group's items in seeded shuffled passes: pass ``p`` is ``random.Random(seed + p * 101)``'s shuffle."""

    def __init__(self, items: Sequence[T], *, seed: int) -> None:
        self._items = list(items)
        self._seed = seed
        self._passes = 0
        self._order: list[T] = []
        self._position = 0

    def next(self) -> T:
        if self._position == len(self._order):
            self._order = list(self._items)
            random.Random(self._seed + self._passes * _PASS_SEED_STRIDE).shuffle(self._order)
            self._passes += 1
            self._position = 0
        item = self._order[self._position]
        self._position += 1
        return item


@dataclass(frozen=True)
class EnvironmentRoute:
    """How one task runs, carried in ``Task.misc`` to rollout workers and into every trajectory.

    Component references are ``module:qualname`` strings, so workers import them
    without the trainer's registries; ``harness`` is the spec drawn for this task.
    ``resolve_task`` marks a task id the service turns into a task in the worker.
    """

    name: str
    rollout: str | None = None
    eval_rollout: str | None = None
    reward_processor: str | None = None
    service: str | None = None
    service_kwargs: dict[str, Any] = field(default_factory=dict)
    harness: str | None = None
    placement: Placement = "auto"
    builtins: BuiltinsMode = "default"
    on_context_overflow: ContextOverflowMode = "grade"
    on_harness_exit: HarnessExitPolicy = field(default_factory=HarnessExitPolicy)
    resolve_task: bool = False
    program: str = "single"
    harness_options: dict[str, Any] = field(default_factory=dict)
    agents: AgentTreeConfig = field(default_factory=AgentTreeConfig)

    def __post_init__(self) -> None:
        # Routes travel as plain dicts in Task.misc (asdict); rebuild the nested configs.
        object.__setattr__(self, "agents", AgentTreeConfig.coerce(self.agents))
        object.__setattr__(self, "on_harness_exit", HarnessExitPolicy.coerce(self.on_harness_exit))
        object.__setattr__(self, "harness_options", dict(self.harness_options or {}))

    def attach(self, task: Task) -> Task:
        """A shallow copy of ``task`` whose misc carries this route (the original is left untouched)."""

        routed = copy.copy(task)
        routed.misc = {**task.misc, ROUTE_MISC_KEY: asdict(self)}
        return routed

    @classmethod
    def of_task(cls, task: Task) -> EnvironmentRoute | None:
        payload = task.misc.get(ROUTE_MISC_KEY)
        return None if payload is None else cls(**payload)

    @classmethod
    def of_trajectory(cls, trajectory: Mapping[str, Any]) -> EnvironmentRoute | None:
        """The route of a serialized trajectory (``TrajectoryCollection.to_dict`` form)."""

        task = trajectory.get("task")
        misc = task.get("misc") if isinstance(task, Mapping) else None
        payload = misc.get(ROUTE_MISC_KEY) if isinstance(misc, Mapping) else None
        return None if payload is None else cls(**payload)


# The rollouts of a mixture of environments (platoon.train.auto), which run each task with a rollout its route names:
# by reference, the route field that names it.
MIXTURE_ROLLOUTS: dict[str, str] = {
    "platoon.train.auto:mixture_rollout": "rollout",
    "platoon.train.auto:mixture_eval_rollout": "eval_rollout",
}

# The episode programs platoon ships (platoon.programs' single, platoon.gym's gym), as routes name them.  After the
# root agent, whose time limit bounds its subagents too, they only grade the session and run the rollout config's
# finalizers.  Any other program is the environment's own code, which may run more after the root, such as judges.
PLATOON_PROGRAMS: tuple[str, ...] = ("single", "gym")

_DEADLINE_SERVICES: dict[str, Any] = {}  # a route's service and options (as JSON) -> the service, for deadlines
_DEADLINE_FAILURES: set[tuple[str, str]] = set()  # (method, service key) that could not answer, warned about once
_SPEC_LIMITS: dict[str, tuple[bool, float | None]] = {}  # a route's harness -> (a TOML spec?, its timeout_s)


def _ask_service(task: Task, method: str, cannot_say: str) -> float | None:
    """The number the environment service of ``task``'s route answers through its optional ``method(task_id)``, for
    deadlines on whole rollouts.  None for a task without a service, a service without the method or one that says
    none, and a service that cannot say, which ``cannot_say`` (a format of the entry's name, the error's type and the
    error) logs once: a deadline is never a reason not to run a rollout."""

    route = EnvironmentRoute.of_task(task)
    if route is None or route.service is None or task.id is None:
        return None
    key = json.dumps([route.service, route.service_kwargs], sort_keys=True, default=str)
    try:
        service = _DEADLINE_SERVICES.get(key)
        if service is None:
            from platoon.envs.service import get_service

            service = _DEADLINE_SERVICES[key] = get_service(route.service, **route.service_kwargs)
        answer = getattr(service, method, None)
        value = answer(str(task.id)) if callable(answer) else None
        return None if value is None else float(value)
    except Exception as exc:
        if (method, key) not in _DEADLINE_FAILURES:
            _DEADLINE_FAILURES.add((method, key))
            logger.warning(cannot_say, route.name, type(exc).__name__, exc)
        return None


def known_rollout_overhead_s(task: Task) -> float | None:
    """The seconds a rollout of ``task`` may take beyond its harness's time limit, as the environment service of the
    task's route says through its optional ``rollout_overhead_s(task_id)`` method (a Harbor service counts its sandbox
    starts, the verifier's timeout and the close); None for a task without a service, a service without the method,
    and one that cannot say, which is logged once.  A deadline counts a harness's own limit in place of a missing
    ``rollout_config.timeout`` only where this is known: a deadline that knew only the harness's time could end the
    rollout while its environment still grades it."""

    overhead = _ask_service(
        task,
        "rollout_overhead_s",
        "Environment %s cannot say how long its rollouts take beyond the harness (%s: %s); their deadlines count a "
        "rollout timeout alone, and keep their fallback without one",
    )
    return None if overhead is None else max(0.0, overhead)


def rollout_overhead_s(task: Task) -> float:
    """The seconds a rollout of ``task`` may take beyond its harness's time limit (``rollout_config.timeout``, else
    :func:`harness_time_limit_s`), which deadlines on whole rollouts add: :func:`known_rollout_overhead_s`, or 0 for a
    task without a service, a service without the method, or one that cannot say, which is logged once: a deadline is
    never a reason not to run a rollout."""

    overhead = known_rollout_overhead_s(task)
    return 0.0 if overhead is None else overhead


def harness_time_limit_s(task: Task, rollout: str | None) -> float | None:
    """How long the harness of a rollout of ``task`` runs at most when ``rollout_config.timeout`` is unset: the
    earliest of its TOML spec's ``timeout_s`` and the task's own time limit for its agent, which the environment
    service of the task's route gives through its optional ``agent_timeout_s(task_id)`` method (a Harbor service:
    the task's ``[agent].timeout_sec`` times ``agent_timeout_multiplier``).  Deadlines on whole rollouts (a rollout
    worker's alarm, an evaluation slot's) count it where they would count that timeout, when the task's environment
    says what a rollout needs beyond the harness (:func:`known_rollout_overhead_s`).

    ``rollout`` is the reference of the rollout function that runs the task; a mixture's runs it with the rollout
    its route names (:data:`MIXTURE_ROLLOUTS`).  Only platoon's harness rollout (:data:`HARNESS_ROLLOUT`) caps a
    harness with these limits, so a rollout of the environment's own, which runs its harness as it likes, gets None,
    as does a harness that neither limits.  So does a function harness that an episode program of the environment's
    own runs (any but :data:`PLATOON_PROGRAMS`), which may run more after the root agent, such as judges, that no
    limit bounds.  A spec or service that cannot say is logged once and limits nothing: then the other limit counts,
    or none, and the deadline is never shorter than the harness's time."""

    if rollout is None or not isinstance(getattr(task, "misc", None), Mapping):  # nothing to go by
        return None
    route = EnvironmentRoute.of_task(task)
    if route is None or route.harness is None:
        return None
    named_by = MIXTURE_ROLLOUTS.get(rollout)
    if (getattr(route, named_by) if named_by is not None else rollout) != HARNESS_ROLLOUT:
        return None
    is_spec, spec_timeout = _spec_limit(route.harness)
    if not is_spec and route.program not in PLATOON_PROGRAMS:  # a TOML spec runs no episode program
        return None
    agent = _ask_service(
        task,
        "agent_timeout_s",
        "Environment %s cannot say its tasks' own time limits for their agents (%s: %s); with no rollout timeout, "
        "their deadlines count the harness spec's time limit only, or none",
    )
    limits = [limit for limit in (spec_timeout, agent) if limit is not None and math.isfinite(limit)]
    return max(0.0, min(limits)) if limits else None


def _spec_limit(ref: str) -> tuple[bool, float | None]:
    """Whether ``ref`` names a TOML harness spec, and its ``timeout_s``, read once per process: ``(False, None)`` for
    a function harness, and for a harness that does not resolve here, which is logged."""

    if ref in _SPEC_LIMITS:
        return _SPEC_LIMITS[ref]
    from platoon.harness.registry import get_harness
    from platoon.harness.spec import HarnessSpec

    try:
        harness = get_harness(ref)
    except Exception as exc:
        logger.warning(
            "Harness %s does not resolve for its time limit (%s: %s); with no rollout timeout, its rollouts' deadlines "
            "count the task's own time limit only, or none",
            ref,
            type(exc).__name__,
            exc,
        )
        harness = None
    limit = (True, harness.timeout_s) if isinstance(harness, HarnessSpec) else (False, None)
    _SPEC_LIMITS[ref] = limit
    return limit


def task_ids_to_dataset(task_ids: Sequence[str]) -> Any:
    """Convert task ids to a Hugging Face Dataset lazily to keep core imports light."""

    from datasets import Dataset

    return Dataset.from_list([{"task_id": task_id} for task_id in task_ids])
