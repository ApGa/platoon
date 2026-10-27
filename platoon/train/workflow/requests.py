"""What a member's rollout runs with: its worker request (the task, the rollout config, the model binding, the
conversion policy, the budgets), and where a worker's files go.  The request is built as the member starts
(``arun_episode_single``), for a rollout in process or in a rollout worker.

The functions here that take the workflow first are its methods still: ``GroupRolloutWorkflow`` keeps each under its
name, so to replace one, replace the method, on the class or an instance, not the function.
"""

import os
import uuid
from dataclasses import asdict
from typing import TYPE_CHECKING

from platoon.programs import DRAW_KEY
from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.task import Task
from platoon.train.components import harness_time_limit_s, known_rollout_overhead_s, rollout_overhead_s
from platoon.train.config_defs import RolloutConfig
from platoon.train.policy.binding import model_binding
from platoon.train.workers.protocol import (
    PROTOCOL_VERSION,
    ConversionPolicy,
    ModelBinding,
    RolloutWorkerRequest,
    ipc_dir,
)
from platoon.train.workflow.settings import GroupSettings
from platoon.train.workflow.start_limit import held_slot

if TYPE_CHECKING:
    from platoon.train.workflow.group import GroupRolloutWorkflow


def _slug(workflow: "GroupRolloutWorkflow", value: str) -> str:
    """``value`` (a task id) as the start of a request id, which names a worker's files: its characters other than
    letters, digits and ``._-`` replaced by ``-``."""

    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in str(value)).strip("-") or "task"


def _model_binding(workflow: "GroupRolloutWorkflow", checkpoint_version: int) -> ModelBinding:
    """Describe the sampling model for a rollout host: the binding of the workflow's policy
    (:func:`~platoon.train.policy.binding.model_binding`), whose file only a worker follows, so only ``process`` mode
    resolves (and creates) the directory that holds it, for the same reason as the scratch paths in
    ``_build_worker_request``."""

    return model_binding(
        workflow.model_info,
        binding_path=workflow._binding_path if workflow._isolation == "process" else "",
        follow_poll_seconds=GroupSettings(workflow.config).worker_binding_poll_seconds,
        checkpoint_version=checkpoint_version,
    )


def _build_worker_request(workflow: "GroupRolloutWorkflow", data: dict, rollout_number: int) -> RolloutWorkerRequest:
    """Everything the parent must decide before a rollout can run, in process or in a worker.  It resolves the task
    parent-side, which also removes the shared-mutable-``Task`` aliasing some plugin datasets have.  It applies
    ``max_steps``, snapshots the checkpoint version and joins ``output_dir`` with it.  A training draw's provenance in
    ``data`` (under ``DRAW_KEY``, when the trainer asks for it) goes into the rollout config's ``extra``, where the
    episode reads it (``ctx.draw``)."""

    task_id = data["task_id"]
    task = workflow.get_task_fn(task_id)
    rollout_config = workflow._get_rollout_config()
    if rollout_config.max_steps is not None:
        task.max_steps = rollout_config.max_steps
    if (draw := data.get(DRAW_KEY)) is not None:  # the trainer asked for draw provenance: the episode's ctx.draw
        rollout_config.extra = {**rollout_config.extra, DRAW_KEY: asdict(draw)}

    # Get checkpoint version from the LLM
    checkpoint_version = workflow.model_info.llm.version
    rollout_config.output_dir = os.path.join(rollout_config.output_dir, str(checkpoint_version))

    request_id = f"{workflow._slug(task_id)}-r{rollout_number}-{uuid.uuid4().hex[:8]}"
    # Only a worker needs node-local scratch, and only a worker ever writes
    # these files.  Resolving them in ``inprocess`` mode would make the
    # fallback -- the mode a run rolls BACK to -- create and depend on a
    # directory it never uses, so an unwritable $TMPDIR would fail every
    # rollout in the mode that is supposed to be unaffected by any of this.
    if workflow._isolation == "process":
        scratch = workflow._ipc_dir()
        response_path = os.path.join(scratch, f"{request_id}.res")
        ready_path = os.path.join(scratch, f"{request_id}.ready")
    else:
        response_path = ready_path = ""
    settings = GroupSettings(workflow.config)
    depth_level_weighting = workflow.config.depth_level_weighting
    depth_discount = settings.depth_level_discount_gamma is not None
    return RolloutWorkerRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id=request_id,
        task_id=task_id,
        rollout_number=rollout_number,
        task=task,
        rollout_config=asdict(rollout_config),
        rollout_fn_ref=workflow.rollout_fn_ref or "",
        model=workflow._model_binding(checkpoint_version),
        conversion=ConversionPolicy(
            filter_errors=workflow._filter_errors(),
            filter_message_actions=settings.filter_message_actions,
            include_traj_depth=depth_level_weighting or depth_discount,
            include_traj_start=depth_level_weighting,
            reward_processor_is_default=workflow._reward_processor_is_default,
            reward_processor_ref=workflow.reward_processor_ref,
            subagent_datum_sampler=workflow._subagent_datum_sampler(),
        ),
        response_path=response_path,
        ready_path=ready_path,
        startup_budget_seconds=settings.worker_startup_budget_seconds,
        rollout_budget_seconds=workflow._rollout_budget_seconds(rollout_config, task),
        rss_soft_limit_bytes=settings.worker_rss_soft_limit_bytes,
        python_env_isolation=settings.worker_python_env_isolation,
        packages=tuple(workflow.packages),
        sandbox_labels=sandbox_cleanup.labels(),
        signal_first_call=held_slot() is not None,
    )


def _rollout_budget_seconds(
    workflow: "GroupRolloutWorkflow", rollout_config: RolloutConfig, task: Task | None = None
) -> float:
    """The child's own deadline: the rollout's time limit, what the task's environment needs beyond it (a Harbor
    task's sandbox starts and verifier, :func:`platoon.train.components.rollout_overhead_s`), plus finalize grace.
    The time limit is ``rollout_config.timeout``, else the harness's own
    (:func:`platoon.train.components.harness_time_limit_s`: the earliest of its spec's ``timeout_s`` and the
    task's own agent time limit) where the task's environment says what it needs beyond the harness
    (:func:`platoon.train.components.known_rollout_overhead_s`); any other rollout gets
    ``worker_unbounded_rollout_budget_seconds``.

    The child also converts datums and writes a multi-MB response after the
    rollout proper, so its alarm must outlast the rollout's time limit.
    """

    settings = GroupSettings(workflow.config)
    grace = settings.worker_finalize_grace_seconds
    unbounded = settings.worker_unbounded_rollout_budget_seconds
    if rollout_config.timeout is not None:
        overhead = rollout_overhead_s(task) if task is not None else 0.0
        return float(rollout_config.timeout) + overhead + grace
    limit = harness_time_limit_s(task, getattr(workflow, "rollout_fn_ref", None)) if task is not None else None
    # Without what the environment needs beyond the harness, a deadline that counted the harness alone could end
    # the rollout while it is graded: such a rollout keeps the unbounded budget, as one that nothing limits.
    overhead = known_rollout_overhead_s(task) if task is not None and limit is not None else None
    if limit is None or overhead is None:
        return unbounded
    return limit + overhead + grace


def _ipc_dir(workflow: "GroupRolloutWorkflow") -> str:
    """The node-local directory of the workflow's rollout workers' files, made at the first call."""

    cached = workflow._ipc_dir_cache
    if cached is None:
        cached = ipc_dir()
        workflow._ipc_dir_cache = cached
    return cached


def _warm_up_request(workflow: "GroupRolloutWorkflow") -> RolloutWorkerRequest:
    """A request like each of this workflow's rollouts', for what plans a fork server: the environments' packages,
    the model the workers sample (whose renderer the server builds) and the startup budget.  It runs nowhere."""

    return RolloutWorkerRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id=f"{workflow.stats_scope}-warm-up",
        task_id="",
        rollout_number=0,
        task=Task(id="warm-up"),
        rollout_config={},
        rollout_fn_ref=workflow.rollout_fn_ref or "",
        model=workflow._model_binding(0),
        conversion=ConversionPolicy(),
        response_path="",
        ready_path="",
        startup_budget_seconds=GroupSettings(workflow.config).worker_startup_budget_seconds,
        packages=tuple(workflow.packages),
    )


def _worker_log_dir(workflow: "GroupRolloutWorkflow", checkpoint_version: int) -> str | None:
    """Next to the rollout artifacts, so both can be searched together."""

    if workflow.log_path is None:
        return None
    return os.path.join(workflow.log_path, "rollouts", workflow.stats_scope, str(checkpoint_version), "workers")


def _binding_path(workflow: "GroupRolloutWorkflow") -> str:
    """The file of the published sampling client, which the workflow's rollout workers follow."""

    return os.path.join(workflow._ipc_dir(), "model_binding.pkl")
