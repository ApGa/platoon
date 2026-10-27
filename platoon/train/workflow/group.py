"""The run's rollout workflow, ``GroupRolloutWorkflow``: each group the dispatcher starts runs through its
``arun_episode``, whose output settles into the queue the training steps dequeue from (the evaluation's groups run
through it too).

A group is ``group_size`` rollouts of one task.  The workflow converts each
rollout's trajectory tree into ``tinker.Datum`` rows and turns trajectory
rewards into per-token advantages. It is the one place where credit is
computed and broadcast over a task group, in the order
:mod:`platoon.train.credit.group` describes.  Its parts live beside it:
its members (``members``), their worker requests (``requests``), the
group's output (``output``) and the workers' telemetry (``telemetry``); each
method whose code lives there stays a method of the class under its name.
They read the settings a partial config may lack through ``settings``.

What a group's workflow measures about the group -- its completion statistics
(``rollout_completion/*``), its baselines and advantages (``credit/baseline/*``,
``credit/root_advantage/*``), its workload distributions (``workload/rollout/*``,
``workload/task/*``), its error-token and subagent-sampling counts -- travels with
the group's output (``group_stats``, a :class:`~platoon.utils.stats_tracker.DeferredStats`)
and is recorded by the trainer in the step that dequeues the group. Recorded when
the group completed, it would land in whichever step was running then: with
prefetching, groups of the next step complete during this one. Per-member events
(worker dispositions, straggler cancellations) are recorded as they happen.

Each group runs with the settings of its environment entry: an entry's own
``workflow_config`` (``platoon.train.entry_settings``) laid over the run's,
which the trainer hands the workflow as ``entry_configs``. Inside a group,
``self.config`` is the group's entry's settings, and ``current_entry()``
(:mod:`platoon.train.workflow.context`) names that entry; outside one,
``self.config`` is the run's. Member tasks copy the group's context, so they
see its settings too; code that reads ``self.config`` from a thread of
``loop.run_in_executor`` gets the run's.

Extension point: to change how advantages are estimated, subclass
``GroupRolloutWorkflow`` and override ``compute_baselines`` (a different
baseline) or ``assign_advantages`` (for example to also rescale), then select
the subclass with ``environments[0].workflow: "package.module.MyWorkflow"`` or
construct it in a train script. Advantages are always computed in the trainer
process, so an override never crosses the rollout-worker boundary. An override
reads its group's settings through ``self.config``.
"""

import asyncio
import logging
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from types import MappingProxyType
from typing import Callable, ClassVar, Self

import tinker
import torch
from tinker import TensorData

from platoon.task import Task
from platoon.train.components import ENVIRONMENT_COLUMN, row_entry
from platoon.train.config_defs import RolloutConfig, WorkflowConfig
from platoon.train.credit.group import (
    # Re-exported: tests read it from this module.  filter_error_tokens looks it up in credit.group, so patch it there.
    _filter_positive_centered_error_tokens,  # noqa: F401
    filter_error_tokens,
    group_baselines,
    group_verdict,
)
from platoon.train.labels import task_label
from platoon.train.startup import WorkerPreload
from platoon.train.tinker.proxy import ModelInfo
from platoon.train.tinker.rollout_metrics import ClientThroughputMonitor
from platoon.train.tinker.rollout_metrics import monitor as client_monitor
from platoon.train.workers.protocol import validated_ref
from platoon.train.workers.run import DEFAULT_REWARD_PROCESSOR
from platoon.train.workers.supervisor import RolloutWorkerSupervisor
from platoon.train.workers.workload import add_datum_funnel, workload_from_raw_rollout
from platoon.train.workflow import members, requests, telemetry
from platoon.train.workflow import output as group_output
from platoon.train.workflow.context import _group_config, _group_entry, _group_stats
from platoon.train.workflow.output import _member, _SingleRolloutOutcome, _TaskRolloutOutput, _trained_keys
from platoon.train.workflow.protocol import StaleFilterSetting, WorkflowSetup, _built_by_keywords
from platoon.train.workflow.settings import GroupSettings
from platoon.train.workflow.start_limit import StartLimit
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.stats_tracker import DeferredStats, StatsTracker
from platoon.utils.stats_tracker import get as get_tracker
from platoon.utils.subagent_sampling import DeterministicSubagentDatumSampler
from platoon.utils.tinker_data_processing import SubagentDatumSamplingStats, TrajectoryCollectionResult

logger = logging.getLogger(__name__)


# The methods through which a ``GroupRolloutWorkflow`` runs its members on the lease (``runs_rollouts_on_lease``).
_LEASE_METHODS = ("arun_episode", "_run_group_rollouts", "_run_group_member")


# The workload helpers of ``workers.workload`` (which the rollout worker calls too) under the names that
# ``platoon.train.tinker.workflows.group_rollout_workflow`` and tests import from this module.
# ``as_member_outcomes`` calls ``_add_datum_funnel`` by this name, so that a patch of it reaches the call.
_workload_from_raw_rollout = workload_from_raw_rollout
_add_datum_funnel = add_datum_funnel


def as_member_outcomes(raw_outcomes: Sequence) -> list[_SingleRolloutOutcome]:
    """What ``_run_group_rollouts`` returned, as one outcome per member.  ``arun_episode_single`` always returns an
    outcome; an override of it (a custom workflow's, a test's) may return a ``TrajectoryCollectionResult`` or None
    instead, which becomes an outcome with its datum funnel, observed unless None."""

    return [
        outcome
        if isinstance(outcome, _SingleRolloutOutcome)
        else _SingleRolloutOutcome(
            result=outcome,
            workload=_add_datum_funnel(RolloutWorkload(), outcome),
            observed=outcome is not None,
        )
        for outcome in raw_outcomes
    ]


class GroupRolloutWorkflow:
    """Workflow that runs multiple rollouts per task and computes group-centered advantages.

    1. Runs `group_size` rollouts for each task in parallel
    2. Collects training data from each rollout
    3. Centers advantages on a per-rollout group baseline (``compute_baselines``,
       broadcast by ``assign_advantages``; see the module docstring)
    4. Returns training data in tinker.Datum format

    Each group runs with its environment entry's settings, ``entry_configs``
    (see the module docstring).

    Its class flags below are its capabilities, which the trainer reads through
    :func:`~platoon.train.workflow.protocol.capabilities_of`.
    """

    # It applies an environment entry's own settings to the entry's groups (``entry_configs``); loading refuses an
    # entry's settings beside a workflow class that does not declare this (``platoon.train.entry_settings``).
    applies_entry_settings: ClassVar[bool] = True
    # Its groups' outputs carry a verdict (``verdict``: trained, uniform, or why the group trains nothing), which the
    # group admission rule ``informative`` reads; resolution refuses it beside a workflow class that does not declare
    # this (``platoon.train.admissions``).
    gives_verdicts: ClassVar[bool] = True
    # It runs each member in a block of its draw's rollout lease (``_run_group_member``), around the member's whole
    # rollout, so under ``train.permit_release: rollout`` its groups start on their first free permit and their members
    # as the rest free. A subclass that overrides ``arun_episode``, ``_run_group_rollouts`` or ``_run_group_member``
    # (``_LEASE_METHODS``) may start members some other way: it reads as False, its groups starting once their whole
    # size of permits is free, unless its own class body sets the flag. A flag set in a class body covers the overrides
    # of that class and its bases, not those of a mixin or of another branch of a diamond (``__init_subclass__``).
    runs_rollouts_on_lease: ClassVar[bool] = True
    # It takes the run's limit on rollouts starting at once as its ``start_limit`` (below), which the trainer sets.
    takes_start_limit: ClassVar[bool] = True
    # It takes how the run's stale filter judges its groups as its ``stale_filter`` (below), which the trainer sets. A
    # subclass that overrides ``arun_episode`` or ``_run_group`` without calling this class's stamps no datum with its
    # own version (``_stamp_own_versions``), so its datums are judged by their rollout's, unwarned: such a subclass sets
    # the flag False in its body, so that the run warns, or stamps them itself.
    takes_stale_filter: ClassVar[bool] = True
    # Its rollouts can run in rollout workers (``rollout_isolation: process``): ``platoon train`` gives it the
    # environments' ``packages`` and the run's ``label_by`` as arguments, and its fork server's ``worker_preload``.
    runs_rollout_workers: ClassVar[bool] = True
    # The settings of each environment entry's groups, by label: the run's with the entry's own laid over them, which
    # ``platoon train`` sets after construction. A group of an entry it does not list runs with ``config``, the run's.
    entry_configs: Mapping[str, WorkflowConfig] = MappingProxyType({})
    # The run's limit on rollouts starting at once (platoon.train.workflow.start_limit), which the trainer sets on its
    # training and evaluation workflows as training starts; None: no limit. _run_group_member takes the slots.
    start_limit: StartLimit | None = None
    # How the run's stale filter judges this workflow's groups (train.stale_filter), which the trainer sets on its
    # training workflow as training starts: under ``datum`` each datum is stamped with its own policy version, the
    # oldest its model calls sampled, in place of its rollout's (_stamp_own_versions).
    stale_filter: StaleFilterSetting = "group"
    # Whether this workflow, one start's, warned of a member's result whose own versions do not match its datums.
    _warned_unstamped: bool = False
    # The node-local directory of the rollout workers' files, made at its first use (requests._ipc_dir).
    _ipc_dir_cache: str | None = None
    # What __init__ sets, at the defaults it sets them to: a custom workflow, or a test, that builds the class with
    # __new__ and sets only the attributes it uses reads the others here.
    stats_scope: str = "train"
    rollout_isolation: str = "inprocess"
    label_by: str = ENVIRONMENT_COLUMN
    packages: tuple[str, ...] = ()
    rollout_fn_ref: str | None = None
    subagent_datum_sampler: DeterministicSubagentDatumSampler | None = None
    # The rollout worker supervisor (``supervisor``), and the semaphore of the startup gate with its limit
    # (members._admission_semaphore), each made at its first use.
    _supervisor: RolloutWorkerSupervisor | None = None
    _admission: asyncio.Semaphore | None = None
    _admission_limit: int = 0

    def __init_subclass__(cls, **kwargs: object) -> None:
        """The flag of a subclass whose own body does not set it: that of the class that sets it nearest in its method
        resolution order, unless one of :data:`_LEASE_METHODS` comes from a class other than this one that the
        declaring class does not derive from (the subclass itself, a mixin, another branch of a diamond): False."""

        super().__init_subclass__(**kwargs)
        if "runs_rollouts_on_lease" in cls.__dict__:
            return
        declared = next(klass for klass in cls.__mro__ if "runs_rollouts_on_lease" in klass.__dict__)
        for name in _LEASE_METHODS:
            owner = next(klass for klass in cls.__mro__ if name in klass.__dict__)
            if owner is not GroupRolloutWorkflow and not issubclass(declared, owner):
                cls.runs_rollouts_on_lease = False
                return

    def __init__(
        self,
        rollout_fn: Callable[[Task, RolloutConfig], dict],
        get_task_fn: Callable[[str], Task],
        config: WorkflowConfig,
        model_info: ModelInfo,
        log_path: str | None = None,
        stats_scope: str = "train",
        filter_errors: bool = False,
        reward_processor: Callable[[dict], tuple[float, dict]] = DEFAULT_REWARD_PROCESSOR,
        packages: Sequence[str] = (),
        label_by: str = ENVIRONMENT_COLUMN,
    ):
        """Initialize the workflow.

        Args:
            rollout_fn: Async function that runs a rollout given a task and RolloutConfig.
            get_task_fn: Function that returns a Task given a task_id.
            config: Workflow configuration (contains group_size and rollout_config).
            model_info: Model information for the tinker LLM.
            log_path: Base log directory for the training run. If provided, rollout results
                      will be stored at {log_path}/rollouts/{stats_scope}/.
            stats_scope: Name for the stats tracker scope (e.g., "train" or "eval").
            filter_errors: Whether to suppress erroneous action tokens only when
                their final group-centered advantage is positive.
            reward_processor: Function to process trajectory rewards.
            packages: The environments' ``package`` modules (``AutoEnvironment.packages``),
                which a rollout worker (``rollout_isolation: process``) imports before its
                rollout, so it knows the harnesses and programs they register.
            label_by: The column that labels the run's tasks (``train.label_by``), whose
                labels ``config.straggler_policies`` key on (:mod:`platoon.train.labels`);
                each group's output carries its task's label (``label``).
        """
        self.rollout_fn = rollout_fn
        self.get_task_fn = get_task_fn
        self.config = config  # the run's; inside a group, ``config`` is the group's entry's
        self.model_info = model_info
        self.log_path = log_path
        self.stats_scope = stats_scope
        self.filter_errors = filter_errors
        self.reward_processor = reward_processor
        self.tracker = get_tracker(stats_scope)
        self.subagent_datum_sampler = (
            DeterministicSubagentDatumSampler(
                keep_probability=config.subagent_datum_keep_probability,
                seed=config.subagent_datum_sampling_seed,
            )
            if config.subagent_datum_keep_probability < 1.0
            else None
        )
        self.rollout_isolation = config.rollout_isolation
        self.packages = tuple(packages)
        self.label_by = label_by
        # Callable references are built ONCE, at construction, and proven to
        # round-trip.  A closure or a functools.partial cannot cross a process
        # boundary by name, and a ``ref or default`` fallback would silently
        # substitute the default reward processor -- changing task_reward,
        # root_rewards_dict and every advantage with no error and entirely
        # plausible metrics.  In ``inprocess`` mode the callables are handed to
        # the shared rollout body directly, so an unnameable callable is only
        # logged, not rejected: an in-process rollout never needs its name.
        self._reward_processor_is_default = reward_processor is DEFAULT_REWARD_PROCESSOR
        self.rollout_fn_ref = self._build_ref(rollout_fn, what="rollout_fn")
        self.reward_processor_ref = (
            None if self._reward_processor_is_default else self._build_ref(reward_processor, what="reward_processor")
        )

    @classmethod
    def from_setup(cls, setup: WorkflowSetup) -> Self:
        """The workflow of one phase of a run, as ``platoon train`` builds it: with the keyword arguments the setup
        stands for, and the environment entries' own settings (``entry_configs``) when they set any."""

        return _built_by_keywords(cls, setup)

    @property
    def config(self) -> WorkflowConfig:
        """The workflow config: inside a group (``arun_episode``), the settings of the group's environment entry
        (``entry_configs``); outside one, the run's, as constructed or assigned."""

        group = _group_config.get()
        return group if group is not None else self._run_config

    @config.setter
    def config(self, value: WorkflowConfig) -> None:
        self._run_config = value

    def _filter_errors(self) -> bool:
        """Whether this group's rollouts suppress positive-credit error tokens: its entry's ``filter_errors`` when the
        entry sets one of its own, else the workflow's (``filter_errors``, which a run-level ``workflow_kwargs`` may
        set; loading refuses both at once)."""

        group = _group_config.get()
        run = getattr(self, "_run_config", None)
        if group is not None and run is not None and group.filter_errors != run.filter_errors:
            return group.filter_errors
        return self.filter_errors

    def _subagent_datum_sampler(self) -> DeterministicSubagentDatumSampler | None:
        """The subagent datum sampler of this group's settings: the workflow's own (``subagent_datum_sampler``, as
        built or as a subclass replaced it) for the run's keep probability and seed, else one per pair an entry sets,
        built once."""

        key = GroupSettings(self.config).subagent_sampling
        run = getattr(self, "_run_config", None)  # None: a workflow built with __new__ that was given no config
        if run is None or key == GroupSettings(run).subagent_sampling:
            return self.subagent_datum_sampler
        keep = key[0]
        if keep >= 1.0:
            return None
        samplers = self.__dict__.setdefault("_entry_subagent_samplers", {})
        if key not in samplers:
            samplers[key] = DeterministicSubagentDatumSampler(keep_probability=key[0], seed=key[1])
        return samplers[key]

    def _build_ref(self, obj, *, what: str) -> str | None:
        """Name a callable for a worker, strictly when a worker will need it."""

        if obj is None:
            if self.rollout_isolation == "process":
                raise ValueError(f"rollout_isolation='process' requires a real {what}")
            return None
        try:
            return validated_ref(obj, what=what)
        except Exception:
            if self.rollout_isolation == "process":
                raise
            logger.debug("%s is not nameable across processes; inprocess mode only", what, exc_info=True)
            return None

    def _get_rollout_config(self) -> RolloutConfig:
        """Get a copy of the rollout config with model info populated."""
        config = deepcopy(self.config.rollout_config)
        config.model_name = self.model_info.model_name
        config.return_dict = True
        config.train = True

        # If log_path is provided, use it as the base for rollout output
        if self.log_path is not None:
            config.output_dir = os.path.join(self.log_path, "rollouts", self.stats_scope)

        return config

    # Methods whose code lives in this package's modules, as module functions that take the workflow first.  Each stays
    # a method under its name, so that replacing it on the class or on an instance (a subclass, a test) replaces what
    # the workflow calls; replacing the module function does not.  Here: the group's output (``output``).
    _make_task_output = group_output._make_task_output

    def _group_stats(self) -> DeferredStats | StatsTracker:
        """Where the statistics of the group being run go.  Inside :meth:`arun_episode`, they go to the group's
        :class:`DeferredStats`, which the trainer records in the step that consumes the group.  Outside it (a direct
        call, say from a subclass or a test), they go straight into the tracker."""

        group_stats = _group_stats.get()
        return group_stats if group_stats is not None else self.tracker

    async def arun_episode(self, data: dict) -> list[tinker.Datum]:
        """Run multiple rollouts for a task and return training data.

        Args:
            data: Dictionary containing 'task_id' and optionally other task data.

        Returns:
            A list-compatible datum result. An empty result still carries exact
            task generation workload for trainer-side batch accounting, and
            ``group_stats``, the group's own statistics, for the consumer to
            record in the step that consumes it (see the module docstring).
        """
        entries = self.entry_configs or {}
        if entries:
            entry = row_entry(data, tuple(entries))
        else:
            entry = data.get(ENVIRONMENT_COLUMN) if isinstance(data, Mapping) else None
        token = _group_stats.set(DeferredStats())
        config_token = _group_config.set(entries.get(entry) if entry is not None else None)
        entry_token = _group_entry.set(entry)
        try:
            output = await self._run_group(data)
        finally:
            _group_entry.reset(entry_token)
            _group_config.reset(config_token)
            _group_stats.reset(token)
        if isinstance(output, _TaskRolloutOutput):
            output.label = task_label(data, self.label_by)
            output.entry = entry
        return output

    async def _run_group(self, data: dict) -> list[tinker.Datum]:
        """Run the group's members (``_run_group_rollouts``), then compute its credit, in the order
        :mod:`platoon.train.credit.group` describes: collect the members' datums (under the per-datum stale filter each
        stamped with its own policy version, :attr:`stale_filter`); stop at the first reason the group trains nothing
        before any credit (``_untrainable_verdict``); center each member's datums on its baseline (``_assign_credit``,
        through ``assign_advantages``); filter the error tokens; and say what the group trains.
        Every return is the group's output (``_make_task_output``), with its verdict and what each member returned."""

        with self._monitor().group_in_flight():
            raw_outcomes = await self._run_group_rollouts(data)
        outcomes = as_member_outcomes(raw_outcomes)
        results = [outcome.result for outcome in outcomes if outcome.result is not None]
        # What each member returned, read before any datum is filtered or stamped (the per-task statistics).
        members = tuple(_member(outcome) for outcome in outcomes)
        if self.stale_filter == "datum":  # the per-datum stale filter judges each datum by its own version
            self._stamp_own_versions(data, results)

        def output(datums: list[tinker.Datum], verdict: str, trainable_rollouts: int = 0) -> _TaskRolloutOutput:
            return self._make_task_output(
                datums, outcomes=outcomes, trainable_rollouts=trainable_rollouts, verdict=verdict, members=members
            )

        datums = self._collect_datums(results)
        if (verdict := self._untrainable_verdict(data, outcomes, results, datums)) is not None:
            return output([], verdict)

        # The completed roots' rewards as the members returned them, for group_verdict's DEBUG line: an override of
        # assign_advantages may change them.
        completed_rewards = [result.task_reward for result in results if result.task_reward_valid]
        self._assign_credit(data, results)
        datums, error_filter_totals, _ = filter_error_tokens(results)
        if error_filter_totals:
            self._group_stats().scalar(**error_filter_totals)
        verdict, trainable_rollouts = group_verdict(datums, results, completed_rewards, self.config, data)
        if verdict == "no_datums":  # no datum the filter kept has a policy action token
            return output([], verdict)
        return output(datums, verdict, trainable_rollouts)

    def _stamp_own_versions(self, data: dict, results: list[TrajectoryCollectionResult]) -> None:
        """Stamp each member's datums with their own policy versions, under the per-datum stale filter.  A result whose
        own versions do not match its datums (a subclass that changed ``result.datums`` after the conversion, or a
        conversion that gives none) keeps its rollout's, warned once per start."""

        unstamped = sum(not result.stamp_own_versions() for result in results)
        if unstamped and not self._warned_unstamped:
            self._warned_unstamped = True
            logger.warning(
                "train.stale_filter is datum, but the datums of %s of the %s members of task %s do not match their "
                "own versions (TrajectoryCollectionResult.own_versions) in number, as when a workflow drops datums "
                "after the conversion: they keep their rollout's version, by which the stale filter judges them. Keep "
                "a result's own_versions in step with its datums (warned once per start)",
                unstamped,
                len(results),
                data.get("task_id"),
            )

    def _collect_datums(self, results: list[TrajectoryCollectionResult]) -> list[tinker.Datum]:
        """The members' datums, in member order.  Records the group's subagent sampling counts when its settings sample
        subagent datums (``subagent_datum_keep_probability``)."""

        all_data: list[tinker.Datum] = []
        subagent_sampling_stats = SubagentDatumSamplingStats()
        for result in results:
            all_data.extend(result.datums)
            subagent_sampling_stats.merge(result.subagent_sampling_stats)
        if self._subagent_datum_sampler() is not None:
            self._group_stats().scalar(**subagent_sampling_stats.to_metrics())
        return all_data

    def _untrainable_verdict(
        self,
        data: dict,
        outcomes: list[_SingleRolloutOutcome],
        results: list[TrajectoryCollectionResult],
        datums: list[tinker.Datum],
    ) -> str | None:
        """Why the group trains nothing, decided before any credit, with a warning, in this order: its members trained
        different reward keys (``reward_key_mismatch``), none returned a result (``no_results``), too few are trainable
        (``below_min``, counted as ``group_size_rejected``), no root completed (``no_completed_root``), or they left no
        datum (``no_datums``).  None: the group goes on to its credit."""

        if len(keys := _trained_keys(outcomes)) > 1:
            # Its rewards are not on one scale: the group's members trained different names of the reward contract's
            # list, which their tests' reward files decided (counted as reward_key_mismatch).
            logger.warning(
                "Rejecting task %s group: its rollouts trained different reward keys (%s), so none of them trains",
                data["task_id"],
                ", ".join(sorted(keys)),
            )
            return "reward_key_mismatch"

        if not results:
            logger.warning(f"No results found for task {data['task_id']}")
            return "no_results"

        min_successful = GroupSettings(self.config).min_successful_group_size
        trainable_members = sum(1 for result in results if result.task_reward_valid)
        if min_successful > 1 and trainable_members < min_successful:
            # Too few trainable members for a meaningful within-task baseline: only a rollout whose root completed
            # with a scored outcome counts, so an untrainable member (an invalid, error, timeout or cancelled root)
            # is missing like one that returned nothing (a setup failure, a cancelled straggler, a dead worker). The
            # generation work is still accounted for through the outcomes. With the default of 1, a group without a
            # completed root takes the branch below.
            logger.warning(
                "Rejecting task %s group with %s trainable rollout(s) of %s (%s returned a result); "
                "min_successful_group_size is %s",
                data["task_id"],
                trainable_members,
                len(outcomes),
                len(results),
                min_successful,
            )
            self._group_stats().scalar(group_size_rejected=1.0)
            return "below_min"

        # Interrupted roots carry partial task rewards. They stay in rollout
        # statistics but never enter a baseline, so a group trains only when at
        # least one of its roots completed.
        if not trainable_members:
            logger.warning(
                "No completed root rewards available for task %s; retaining rollout stats but skipping training",
                data["task_id"],
            )
            return "no_completed_root"

        if not datums:
            logger.warning(f"No train data found for task {data['task_id']}")
            return "no_datums"
        return None

    def _assign_credit(self, data: dict, valid_results: list[TrajectoryCollectionResult]) -> None:
        """The group's credit, over the members that returned a result (``valid_results``) of the group of task row
        ``data``: each member's datums centered on its baseline (:meth:`assign_advantages`)."""

        self.assign_advantages(valid_results)

    def compute_baselines(self, results: list[TrajectoryCollectionResult]) -> list[float]:
        """Return one baseline per usable rollout of a task group.

        The default is the completed-root group mean, or its leave-one-out
        variant with ``leave_one_out_baseline`` (see :func:`group_baselines`).
        """

        return group_baselines(
            [result.task_reward for result in results],
            [result.task_reward_valid for result in results],
            leave_one_out=self.config.leave_one_out_baseline,
        )

    def assign_advantages(self, results: list[TrajectoryCollectionResult]) -> None:
        """Center every datum of each rollout on that rollout's baseline.

        The converter loaded each action token with its trajectory's reward,
        so this yields ``reward - baseline`` for the root and every subagent
        datum of a rollout. Non-action positions are left untouched.
        """

        baselines = self.compute_baselines(results)
        for result, baseline in zip(results, baselines, strict=True):
            for datum in result.datums:
                advantages = datum.loss_fn_inputs["advantages"].to_torch()
                mask = datum.loss_fn_inputs["mask"].to_torch()
                datum.loss_fn_inputs["advantages"] = TensorData.from_torch(
                    torch.where(mask > 0, advantages - baseline, advantages)
                )
        # Completed roots only, as in the baselines. The spread is 0 for group-mean baselines and
        # positive for leave-one-out ones whenever the group's rewards differ.
        completed = [
            (result.task_reward, baseline)
            for result, baseline in zip(results, baselines, strict=True)
            if result.task_reward_valid
        ]
        if completed:
            self._group_stats().scalar(
                **{
                    "credit/baseline/avg": sum(baseline for _, baseline in completed) / len(completed),
                    "credit/baseline/spread": max(b for _, b in completed) - min(b for _, b in completed),
                    "credit/root_advantage/abs_avg": sum(abs(reward - baseline) for reward, baseline in completed)
                    / len(completed),
                }
            )

    # Moved methods, as above: running the members (``members``) and their worker requests (``requests``).
    _admission_semaphore = members._admission_semaphore
    _admit_group = members._admit_group
    _run_group_member = members._run_group_member
    _member_start_slot = members._member_start_slot
    _run_group_rollouts = members._run_group_rollouts
    _cancel_late_members = members._cancel_late_members
    _slug = requests._slug
    _model_binding = requests._model_binding
    _build_worker_request = requests._build_worker_request
    _rollout_budget_seconds = requests._rollout_budget_seconds
    _ipc_dir = requests._ipc_dir

    def _monitor(self) -> ClientThroughputMonitor:
        """The client throughput monitor of this workflow's stats scope."""

        return client_monitor(self.stats_scope)

    @property
    def _isolation(self) -> str:
        """Where the members' rollouts run: ``rollout_isolation``.  The members, their requests and the rollout
        workers' start read it under this one name, so that a subclass, or a test's stand-in for the workflow, that
        sets ``_isolation`` switches them all."""

        return self.rollout_isolation

    # What the rollout workers' fork server imports and builds ahead of them, each when it is fork-safe: what the run's
    # environments and harnesses import and build when a rollout opens them, and the run's workflow_config
    # .worker_preload (platoon.train.startup).  platoon train sets it after construction, before the
    # supervisor is built.
    worker_preload: WorkerPreload = WorkerPreload()

    @property
    def supervisor(self) -> RolloutWorkerSupervisor:
        """Lazily built so ``inprocess`` runs never touch node-local scratch."""

        if self._supervisor is None:
            settings = GroupSettings(self.config)
            self._supervisor = RolloutWorkerSupervisor(
                scratch_dir=self._ipc_dir(),
                termination_grace_seconds=settings.worker_termination_grace_seconds,
                parent_grace_seconds=settings.worker_parent_grace_seconds,
                mem_available_floor_bytes=settings.worker_mem_available_floor_bytes,
                start_method=settings.worker_start_method,
                preload=self.worker_preload,
            )
        return self._supervisor

    async def warm_up(self) -> None:
        """Start what this workflow's rollouts need before the first one: with ``rollout_isolation: process`` and the
        fork server, the fork server, which imports the worker's stack (and :attr:`worker_preload`) while the trainer
        gets ready to train, instead of when the first rollouts ask for it.  Nothing for in-process rollouts or exec'd
        workers."""

        if self._isolation != "process" or GroupSettings(self.config).worker_start_method != "forkserver":
            return
        await self.supervisor.warm(self._warm_up_request())

    _warm_up_request = requests._warm_up_request  # moved, as above (``requests``)

    def publish_model_binding(self, sampling_client, version: int) -> None:
        """Republish the trainer's sampling client for its rollout workers."""

        if self._isolation != "process":
            return
        self.supervisor.publish_binding(sampling_client, version)

    # Moved methods, as above: a worker's files (``requests``), a member's outcome (``output``), the workers'
    # telemetry (``telemetry``).
    _worker_log_dir = requests._worker_log_dir
    _binding_path = property(requests._binding_path)
    _outcome_from_response = group_output._outcome_from_response
    _warn_of_missing_routing_ids = telemetry._warn_of_missing_routing_ids
    _record_worker_telemetry = telemetry._record_worker_telemetry
    # One member's rollout, in process or in a rollout worker (``members``): the override point of a workflow that
    # runs its rollouts its own way, which ``_run_group_member`` calls on the instance.
    arun_episode_single = members.arun_episode_single
