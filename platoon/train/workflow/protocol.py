"""The workflow contract: what the trainer needs of a rollout workflow, and how it reads what a workflow returns.

A workflow (:class:`RolloutWorkflow`) turns one draw's task row into its group's output with ``arun_episode``.  What a
workflow class can do beyond that it declares (:class:`WorkflowCapabilities`), and the trainer reads it through
:func:`capabilities_of`; ``platoon train`` builds a workflow class from a :class:`WorkflowSetup`.  What a group's output
gives the trainer besides its rows is its report (:class:`OutputReport`), which the trainer reads through
:func:`report_of`; its rows, by the policy each trains, the trainer reads through :func:`rows_by_policy`.  Each reader
also takes what was written before the contract was: a class's flags, an output's attributes (:func:`carried_report`),
a plain list of datums, each read as the trainer read them then.

:class:`~platoon.train.workflow.group.GroupRolloutWorkflow` is platoon's workflow, and
:class:`~platoon.train.workflow.output.GroupOutput` its output: a list of datums with its report.

This module loads neither ``tinker`` nor ``torch``.
"""

import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    import tinker

    from platoon.train.task_stats import MemberResult
    from platoon.utils.rollout_stats import RolloutStats
    from platoon.utils.rollout_workload import RolloutWorkload
    from platoon.utils.stats_tracker import DeferredStats


class RolloutWorkflow(Protocol):
    """What the trainer needs of a rollout workflow: one draw's task row in, its group's output out (None: the group
    failed, and trains nothing).  The output is a :class:`RolloutOutput`, or a sequence of datums, all of the run's one
    trained policy (a ``GroupOutput``, or a plain list).

    The optional members, which the trainer uses when a workflow has them:

    - ``publish_model_binding(sampling_client, version)``: hand the sampling client of a new policy version to the
      workflow's rollout workers (training does after each version it publishes, an evaluation before it starts);
    - ``warm_up()``: start, as training starts, what its rollouts need before the first one (awaited);
    - ``rollout_isolation``: ``"process"`` when its rollouts run in rollout workers, which read the published client;
    - ``start_limit``: the run's limit on rollouts starting at once, which the trainer sets on a workflow that takes it
      (``takes_start_limit``), and whose metrics an evaluation reports while it is set;
    - ``entry_configs``: each environment entry's settings, which ``platoon train`` sets on a workflow that applies
      them (``applies_entry_settings``);
    - its capabilities: a ``capabilities`` class attribute, or the class flags of :class:`WorkflowCapabilities`;
    - ``from_setup(setup)``, a class method: ``platoon train`` builds the run's workflow class with it, from a
      :class:`WorkflowSetup`, and builds a class without one with ``GroupRolloutWorkflow``'s keyword arguments.
    """

    async def arun_episode(self, data: dict) -> "RolloutOutput | Sequence[tinker.Datum] | None": ...


@dataclass(frozen=True)
class WorkflowCapabilities:
    """What a workflow class can do beyond ``arun_episode``, which the trainer reads (:func:`capabilities_of`) instead
    of testing whether the class is a ``GroupRolloutWorkflow``.  A class declares them in a ``capabilities`` class
    attribute, or as class flags of the same names; ``GroupRolloutWorkflow`` sets five of them as flags."""

    # It applies an environment entry's own settings to the entry's groups (``entry_configs``): loading refuses an
    # entry's settings beside a workflow that does not (platoon.train.entry_settings).
    applies_entry_settings: bool = False
    # Its groups' outputs carry a verdict, which the group admission rule ``informative`` reads: resolution refuses the
    # rule beside a workflow that gives none (platoon.train.admissions).
    gives_verdicts: bool = False
    # It runs each rollout in a block of its draw's rollout lease, so that its groups start on their first free permit
    # under ``train.permit_release: rollout`` (platoon.train.leases).
    runs_rollouts_on_lease: bool = False
    # It takes the run's limit on rollouts starting at once as its ``start_limit`` (``train.max_starting_rollouts``).
    takes_start_limit: bool = False
    # Its rollouts can run in rollout workers: ``platoon train`` gives it the environments' ``packages`` and the run's
    # ``label_by`` as arguments, and the ``worker_preload`` of its workers' fork server (platoon.train.launch).
    runs_rollout_workers: bool = False
    # Its ``publish_model_binding`` takes ``policy=``, which a run that samples more than one policy needs.
    binds_policies: bool = False


def capabilities_of(workflow: object) -> WorkflowCapabilities:
    """What ``workflow`` (an instance or a class) can do: its ``capabilities`` when it declares them, else its class
    flags, a flag it does not set False.  Each flag is read as the trainer read it before the contract was written down:
    ``applies_entry_settings`` and ``runs_rollouts_on_lease`` by their truth, the others only when they are True.

    Declared capabilities, a class's own or inherited, are all six, and win over every flag: over a subclass's own
    flags, and over ``GroupRolloutWorkflow``'s rule that a subclass overriding how a member runs does not run its
    rollouts on the lease (``runs_rollouts_on_lease``), which sets the flag only."""

    declared = getattr(workflow, "capabilities", None)
    if isinstance(declared, WorkflowCapabilities):
        return declared
    return WorkflowCapabilities(
        applies_entry_settings=bool(getattr(workflow, "applies_entry_settings", False)),
        gives_verdicts=getattr(workflow, "gives_verdicts", False) is True,
        runs_rollouts_on_lease=bool(getattr(workflow, "runs_rollouts_on_lease", False)),
        takes_start_limit=getattr(workflow, "takes_start_limit", False) is True,
        runs_rollout_workers=getattr(workflow, "runs_rollout_workers", False) is True,
        binds_policies=getattr(workflow, "binds_policies", False) is True,
    )


@dataclass(frozen=True)
class OutputReport:
    """What the trainer reads of a group's output besides its rows, in one place: the per-task statistics and the group
    admission rules read its members and verdict, the step that consumes the group its statistics and its workload,
    and the label metrics its label.  What an output does not give is None (``members`` empty, ``verdict`` unknown).
    ``workload`` and its four counts come together: all set, or all None."""

    # The task's label (its value of the run's label column, ``train.label_by``), by which the metrics break down, and
    # the label of the environment entry whose settings the group ran with.
    label: str | None = None
    entry: str | None = None
    # What generating the group took, its rollouts (requested, observed, trainable) and the datums it kept.
    workload: "RolloutWorkload | None" = None
    requested_rollouts: int | None = None
    observed_rollouts: int | None = None
    trainable_rollouts: int | None = None
    task_retained_datums: int | None = None
    # The group's rollout statistics, and what its workflow measured about it, which the step that consumes the group
    # records (a group's statistics belong to that step, not to the one running when it completed).
    rollout_stats: "RolloutStats | None" = None
    group_stats: "DeferredStats | None" = None
    # What each member returned, and what the group trains or why it trains nothing (platoon.train.task_stats).
    members: tuple["MemberResult", ...] = ()
    verdict: str = "unknown"


@dataclass(frozen=True)
class WorkflowSetup:
    """What ``platoon train`` builds one phase's workflow with (a workflow class's ``from_setup``), in place of
    ``GroupRolloutWorkflow``'s keyword arguments, whose ``model_info`` cannot carry several policies: the phase's
    settings and the run's components, and the policies its rollouts sample."""

    phase: Literal["train", "eval"]
    rollout_fn: Callable[..., Any]
    get_task_fn: Callable[[str], Any]
    # The phase's workflow settings (train.workflow_config or eval.workflow_config), and each environment entry's own
    # laid over them, by entry (None: no entry sets its own; platoon.train.entry_settings).
    config: Any
    entry_configs: Mapping[str, Any] | None
    # Each sampled policy's model info (its renderer, its tokenizer and the model gateway that samples it and keeps its
    # version), by policy id, the agents' default policy first: with one policy, main's.
    policies: Mapping[str, Any]
    log_path: str
    stats_scope: str
    filter_errors: bool
    reward_processor: Callable[..., Any]
    # The environments' packages and the run's label column, for a workflow that runs rollout workers
    # (WorkflowCapabilities.runs_rollout_workers; None for one that does not).
    packages: Sequence[str] | None = None
    label_by: str | None = None
    # The environment's other arguments of the phase's workflow (workflow_kwargs, eval_workflow_kwargs), as given.
    options: Mapping[str, Any] = field(default_factory=dict)


def _built_by_keywords(workflow_cls: Any, setup: WorkflowSetup) -> Any:
    """A workflow of ``workflow_cls`` built as ``platoon train`` always built one: with the keyword arguments of
    ``GroupRolloutWorkflow``'s constructor that ``setup`` stands for (:func:`_keywords`), and each environment entry's
    own settings as its ``entry_configs`` when an entry sets any.  ``GroupRolloutWorkflow.from_setup`` builds it so, as
    ``platoon train`` builds a class without ``from_setup``."""

    workflow = workflow_cls(**_keywords(setup))
    if setup.entry_configs is not None:
        workflow.entry_configs = dict(setup.entry_configs)
    return workflow


def _keywords(setup: WorkflowSetup) -> dict[str, Any]:
    """The keyword arguments of ``GroupRolloutWorkflow``'s constructor that ``setup`` stands for, in its order: the
    first policy's model info as ``model_info``, ``packages`` and ``label_by`` when given, then the environment's
    ``options``, which win over those two.  ``TypeError`` for options that repeat another."""

    keywords = {
        "rollout_fn": setup.rollout_fn,
        "get_task_fn": setup.get_task_fn,
        "config": setup.config,
        "model_info": next(iter(setup.policies.values())),
        "log_path": setup.log_path,
        "stats_scope": setup.stats_scope,
        "filter_errors": setup.filter_errors,
        "reward_processor": setup.reward_processor,
    }
    if repeated := sorted(set(keywords) & set(setup.options)):
        raise TypeError(f"got multiple values for keyword argument {repeated[0]!r} (the environment's options)")
    if setup.packages is not None:
        keywords["packages"] = setup.packages
    if setup.label_by is not None:
        keywords["label_by"] = setup.label_by
    return {**keywords, **setup.options}


class RolloutOutput(Protocol):
    """A group's output written to the contract: its rows, by the policy each trains, and its report.  The other output
    a workflow may return is a sequence of datums (``GroupOutput``, a plain list)."""

    def rows(self) -> Mapping[str, Sequence["tinker.Datum"]]: ...

    @property
    def report(self) -> OutputReport: ...


def report_of(output: object) -> OutputReport:
    """The report of a group's output: its ``report`` when it gives one (a :class:`RolloutOutput`, a ``GroupOutput``),
    else the one its attributes give (:func:`carried_report`: an output written before the contract was, such as a
    workflow's own list subclass), which for a plain list of datums, or None for a group that failed, is empty."""

    report = _attribute(output, "report")
    if isinstance(report, OutputReport):
        return report
    return carried_report(output)


def rows_by_policy(output: Any, trained: str) -> Mapping[str, Sequence["tinker.Datum"]]:
    """A group's rows by the policy each trains.  A list of datums (a ``GroupOutput``, or a workflow's plain list), or
    any other output without ``rows()`` (a tuple of datums, say), is all ``trained``'s, the run's one trained policy:
    ``{trained: output}``, the output itself.  Any other output answers ``rows()``.  The step reads every output's rows
    through this one function."""

    if isinstance(output, list):
        return {trained: output}
    rows = _attribute(output, "rows")
    return rows() if callable(rows) else {trained: output}


def carried_report(output: object) -> OutputReport:
    """The report an output's attributes give, each read as the trainer read it before the contract was written down
    and left out when it is malformed or cannot be read (its property raises): the labels as they are; the workload
    only with its four counts, each a whole number of 0 or more (:func:`malformed_counts`); the statistics only of
    their types; the members as a tuple, none when they are no sequence; the verdict only as text.  Never raises."""

    workload_fields: dict[str, Any] = {}
    if _is_a(workload := _attribute(output, "workload"), _ROLLOUT_WORKLOAD):
        counts = _counts(output)
        if not _malformed(counts):
            workload_fields = {"workload": workload, **counts}
    rollout_stats = _attribute(output, "rollout_stats")
    group_stats = _attribute(output, "group_stats")
    verdict = _attribute(output, "verdict")
    try:
        members = tuple(_attribute(output, "members") or ())
    except Exception:  # no sequence
        members = ()
    return OutputReport(
        label=_attribute(output, "label"),
        entry=_attribute(output, "entry"),
        **workload_fields,
        rollout_stats=rollout_stats if _is_a(rollout_stats, _ROLLOUT_STATS) else None,
        group_stats=group_stats if _is_a(group_stats, _DEFERRED_STATS) else None,
        members=members,
        verdict=verdict if isinstance(verdict, str) else "unknown",
    )


def malformed_counts(output: object) -> dict[str, Any] | None:
    """The rollout counts an output carries beside its workload when one of them is malformed (no whole number of 0 or
    more), which its report leaves out with its workload (:func:`carried_report`), and of which the step that consumes
    the group warns; None when it carries no workload, or well-formed counts."""

    if not _is_a(_attribute(output, "workload"), _ROLLOUT_WORKLOAD):
        return None
    counts = _counts(output)
    return counts if _malformed(counts) else None


# The rollout counts that come with a workload, in the order the step's warning names them.
_ROLLOUT_COUNTS = ("requested_rollouts", "observed_rollouts", "trainable_rollouts", "task_retained_datums")
# The types of the workload and the statistics an output carries, by module and name.
_ROLLOUT_WORKLOAD = ("platoon.utils.rollout_workload", "RolloutWorkload")
_ROLLOUT_STATS = ("platoon.utils.rollout_stats", "RolloutStats")
_DEFERRED_STATS = ("platoon.utils.stats_tracker", "DeferredStats")


def _attribute(output: object, name: str) -> Any:
    """``output``'s attribute ``name``; None when it has none, or reading it raises."""

    try:
        return getattr(output, name, None)
    except Exception:
        return None


def _is_a(value: object, kind: tuple[str, str]) -> bool:
    """Whether ``value`` is of the type ``kind`` names, ``(module, name)``, without importing its module, which loads
    torch: a value of the type has loaded it."""

    module, name = kind
    cls = getattr(sys.modules.get(module), name, None)
    return isinstance(cls, type) and isinstance(value, cls)


def _counts(output: object) -> dict[str, Any]:
    return {name: _attribute(output, name) for name in _ROLLOUT_COUNTS}


def _malformed(counts: Mapping[str, Any]) -> bool:
    return any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values())
