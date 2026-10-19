"""Evaluation rollouts: run each task's rollout, record its outcome and write its events.

``DefaultEvalGroupWorkflow`` runs the rollout function the environments
select (``AutoRollout``: ``harness_rollout`` for routed environments) on tasks
from their task loader, in this process or, with ``use_subprocesses``, in worker
processes: each of up to ``subprocess_max_workers`` slots has a process of its
own, which runs one rollout at a time and is reused until a rollout in it hangs
past the hard timeout or dies, when that slot alone gets a new process. Each
finished rollout also gets an event file for the trajectory viewer under
``<rollout dir>/events/``.

Killing a slot's process does not stop the sandboxes its rollout started (a
Docker container, a Singularity instance, a command in a session of its own),
so each slot's process labels its sandboxes and lists how to reap them, and the
sandboxes a retired process left are reaped by that label
(:mod:`platoon.sandbox.cleanup`).
"""

from __future__ import annotations

import asyncio
import contextlib
import glob
import logging
import multiprocessing as mp
import os
import shutil
import signal
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

from platoon.config_defs import RolloutConfig
from platoon.eval.accounting import rollout_accounting
from platoon.eval.config_defs import EvalPolicy, EvalWorkflowConfig, TinkerKeyRefusedError, run_provenance
from platoon.gateway.trace import OUTCOME_MISC_KEY
from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.task import Task
from platoon.train.components import EnvironmentRoute
from platoon.utils import scratch
from platoon.utils.judges import is_judge_trajectory
from platoon.utils.trajectory_status import outcome_status, trained_reward_key
from platoon.visualization.event_sinks import emit_rollout_events, rollout_events_path

logger = logging.getLogger(__name__)

#: How long reaping the sandboxes that slot processes left may take, each time: when a retired process is gone, and
#: when the slots close.
SANDBOX_REAP_SECONDS = 120.0
#: How long a retired slot's process gets to exit before its sandboxes are reaped: the providers leave alone the
#: sandboxes of an owner that still exists, a killed process included until it is waited for.
PROCESS_EXIT_SECONDS = 10.0
# The root trajectory's harness record (``platoon.harness.rollout.HARNESS_MISC_KEY``).
HARNESS_MISC_KEY = "harness"

RewardProcessor = Callable[[dict[str, Any]], tuple[float, dict[str, float]]]
SuccessFn = Callable[[dict[str, Any], float, dict[str, float]], bool]
WorkflowMetricsFn = Callable[[dict[str, Any]], dict[str, float]]


@dataclass
class EvalRolloutRecord:
    task_id: str
    rollout_index: int
    success: bool
    reward: float
    reward_components: dict[str, float]
    num_steps_total: int
    num_steps_root: int
    num_steps_subtrajectories: int
    num_subtrajectories: int
    subtrajectory_depth_counts: dict[str, int] = field(default_factory=dict)
    subtrajectory_depth_steps: dict[str, int] = field(default_factory=dict)
    wall_time_seconds: float | None = None
    workflow_metrics: dict[str, float] = field(default_factory=dict)
    source_path: str | None = None
    error: str | None = None
    trajectory_collection: dict[str, Any] | None = None
    # The root outcome's status (``scored``, ``error``, ``invalid``, ``timeout``, ``cancelled``); None when the
    # rollout recorded none (a custom rollout, or one that raised).
    outcome_status: str | None = None
    # The reward key the root's scored outcome trained: the name its grader reported the reward under
    # (``platoon.utils.trajectory_status.trained_reward_key``); None for one platoon scored itself, or not scored.
    reward_key: str | None = None
    # The rollout's model calls and judges, counted from its trajectories (``platoon.eval.accounting``): ``usage`` by
    # kind of call, ``cost``, ``backend`` (attempts, rate limits, failed calls, refusals, calls whose provider reported
    # no usage, served models) and ``judges``.
    usage: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    backend: dict[str, Any] = field(default_factory=dict)
    judges: dict[str, Any] = field(default_factory=dict)
    # Why no further rollout of the evaluation should start, when this one shows a problem every rollout would have
    # (a judge refused because inference.judges is false, a backend that refuses the run's credentials or model).
    stop_run: str | None = None


class EvalWorkflow(Protocol):
    config: EvalWorkflowConfig

    async def arun_rollout(
        self,
        data: dict[str, Any],
        rollout_index: int,
        output_dir: str,
    ) -> EvalRolloutRecord: ...

    def record_from_collection(
        self,
        task_id: str,
        rollout_index: int,
        trajectory_collection: dict[str, Any],
        wall_time_seconds: float | None = None,
        source_path: str | None = None,
    ) -> EvalRolloutRecord: ...


def _get_root_trajectory(collection: dict[str, Any]) -> dict[str, Any] | None:
    trajectories = collection.get("trajectories")
    if not isinstance(trajectories, dict) or not trajectories:
        return None
    root = next(iter(trajectories.values()))
    return root if isinstance(root, dict) else None


def _count_steps(traj: dict[str, Any]) -> int:
    steps = traj.get("steps")
    return len(steps) if isinstance(steps, list) else 0


def _default_success_fn_factory(success_threshold: float) -> SuccessFn:
    """Success by ``success_threshold``.

    A traced rollout (one whose root records its outcome, as every harness rollout's does) succeeds when its outcome
    was scored and its scored reward, which the trace writes on the last step as ``reward/success``, reaches the
    threshold; the reward processor's output does not change that.  A legacy rollout's own success flag
    (``reward/success`` or ``success``) counts at 1, and without one its processed root reward meets the threshold.
    """

    def _default_success_fn(
        collection: dict[str, Any],
        root_reward: float,
        reward_components: dict[str, float],
    ) -> bool:
        del reward_components  # intentionally unused in default logic
        root = _get_root_trajectory(collection)
        misc = root.get("misc") if root is not None else None
        outcome = misc.get(OUTCOME_MISC_KEY) if isinstance(misc, dict) else None
        # Recorded by the trace: an outcome with a status, as outcome_status reads it.
        traced = isinstance(outcome, dict) and isinstance(outcome.get("status"), str)
        if traced and outcome["status"] != "scored":
            return False  # nothing was measured, e.g. a timeout counted as a failure
        if root is not None:
            steps = root.get("steps")
            if isinstance(steps, list) and steps:
                last_step = steps[-1]
                if isinstance(last_step, dict):
                    reward_misc = last_step.get("misc", {}).get("reward_misc", {})
                    if isinstance(reward_misc, dict):
                        for key in ("reward/success", "success"):
                            if key in reward_misc:
                                try:
                                    return float(reward_misc[key]) >= (success_threshold if traced else 1.0)
                                except Exception:
                                    pass
        return root_reward >= success_threshold

    return _default_success_fn


def _compute_depth_stats(collection: dict[str, Any]) -> tuple[int, int, int, dict[str, int], dict[str, int]]:
    """Steps and subtrajectories of the policy's agents, by depth: verifier and judge agents, which judge the policy's
    agents (:func:`platoon.utils.judges.is_judge_trajectory`), are counted apart (``EvalRolloutRecord.judges``), so
    that reports with judges on and off compare directly."""

    trajectories = collection.get("trajectories")
    if not isinstance(trajectories, dict) or not trajectories:
        return 0, 0, 0, {}, {}
    root_id = next(iter(trajectories))
    trajectories = {
        traj_id: traj
        for traj_id, traj in trajectories.items()
        if traj_id == root_id or not (isinstance(traj, dict) and is_judge_trajectory(traj))
    }

    ids = list(trajectories.keys())
    root_id = ids[0]
    parents: dict[str, str | None] = {}
    steps_per_traj: dict[str, int] = {}
    for traj_id, traj in trajectories.items():
        if not isinstance(traj, dict):
            continue
        parent_info = traj.get("parent_info")
        parent_id = parent_info.get("id") if isinstance(parent_info, dict) else None
        parents[traj_id] = parent_id
        steps_per_traj[traj_id] = _count_steps(traj)

    depth_cache: dict[str, int] = {}

    def _depth_for(traj_id: str) -> int:
        if traj_id in depth_cache:
            return depth_cache[traj_id]
        parent_id = parents.get(traj_id)
        if traj_id == root_id or parent_id is None or parent_id not in parents:
            depth_cache[traj_id] = 0
            return 0
        depth = _depth_for(parent_id) + 1
        depth_cache[traj_id] = depth
        return depth

    depth_counts: dict[str, int] = {}
    depth_steps: dict[str, int] = {}
    num_steps_total = 0
    num_steps_root = 0
    num_steps_sub = 0
    num_subtrajectories = 0

    for traj_id, step_count in steps_per_traj.items():
        depth = _depth_for(traj_id)
        depth_key = str(depth)
        num_steps_total += step_count
        if depth == 0:
            num_steps_root += step_count
        else:
            num_subtrajectories += 1
            num_steps_sub += step_count
            depth_counts[depth_key] = depth_counts.get(depth_key, 0) + 1
            depth_steps[depth_key] = depth_steps.get(depth_key, 0) + step_count

    return num_steps_total, num_steps_root, num_steps_sub, depth_counts, depth_steps


def _run_wide_problem(root: dict[str, Any], accounting: dict[str, Any]) -> str | None:
    """What in a finished rollout every other rollout of the evaluation would meet too, so that the evaluation stops:
    a judge refused while inference.judges is false, or a backend that refused the run's key or model."""

    judges = accounting["judges"]
    if judges["refused"]:
        route = EnvironmentRoute.of_trajectory(root)
        harness = (root.get("misc") or {}).get(HARNESS_MISC_KEY) or {}
        who = ", ".join(
            f"{name} {value!r}"
            for name, value in (
                ("program", route.program if route is not None else None),
                ("harness", harness.get("spec") or (route.harness if route is not None else None)),
            )
            if value
        )
        what = str(judges["refused_first"] or "A judge").lower()
        return (
            f"judges_off: {what} was refused because inference.judges is false; the rollout's {who or 'code'} "
            "judges agents: evaluate it with inference.judges on, or have the program do without judges when "
            "EpisodeContext.judges is false"
        )
    fatal = accounting["backend"]["fatal"]
    if fatal:
        return f"backend_refused: {fatal}"
    return None


def _write_rollout_events(output_dir: str, task_id: str, collection: dict[str, Any]) -> None:
    """Write a finished rollout's event file for the trajectory viewer; never fails the rollout."""
    if not collection.get("trajectories"):
        return
    try:
        emit_rollout_events(collection, rollout_events_path(output_dir, task_id, str(collection.get("id"))))
    except Exception:
        logger.warning("Could not write the event file of a rollout of task %s", task_id, exc_info=True)


def _start_slot_process(
    worker: str,
    sandboxes_path: str,
    labels: dict[str, str],
    initializer: Callable[..., Any] | None,
    initargs: tuple[Any, ...],
) -> None:
    """Set up a slot's process: label every sandbox it starts with ``labels`` (the evaluation's, its run's
    ``platoon.run``) and ``platoon.worker: <worker>``, and list how to reap them in ``sandboxes_path``
    (:mod:`platoon.sandbox.cleanup`), then run the slots' own initializer."""

    sandbox_cleanup.set_labels({**labels, sandbox_cleanup.WORKER_LABEL: worker})
    sandbox_cleanup.mirror_to(sandboxes_path)
    if initializer is not None:
        initializer(*initargs)


class _ProcessSlots:
    """Worker processes for subprocess rollouts: at most ``size`` at once, each a single-process executor.

    A slot's process runs one rollout at a time and is reused (it keeps the policy it installed). A rollout that
    hangs past its hard timeout, dies (an OOM kill, a segfault, ``os._exit``) or is cancelled retires only its
    own slot's process: in one shared pool, a dead process fails every future in flight and every later one.

    Each process labels its sandboxes with an id of its own and lists how to reap them in a file of a private
    directory (:func:`_start_slot_process`). Once a retired process is gone, the sandboxes its file still lists are
    reaped by that label, and :meth:`aclose` reaps what the processes left when the slots close.

    ``admit``, when given, runs as each rollout takes a slot, and refuses it by raising: a rollout that waited for a
    slot does not start once the evaluation has a reason to start none.
    """

    def __init__(
        self,
        size: int,
        *,
        initializer: Callable[..., Any] | None = None,
        initargs: tuple[Any, ...] = (),
        admit: Callable[[], None] | None = None,
    ) -> None:
        self._free = asyncio.Semaphore(size)
        self._admit = admit
        self._idle: list[ProcessPoolExecutor] = []
        self._executors: dict[ProcessPoolExecutor, str] = {}  # -> the file that lists its process's sandboxes
        self._initializer = initializer  # runs first in every process a slot starts
        self._initargs = initargs
        self._directory: str | None = None  # private (0700): those files
        self._reaping: set[asyncio.Task[None]] = set()

    async def run(self, timeout_s: float, fn: Callable[..., Any], *args: Any, grace_s: float = 0.0) -> Any:
        """``fn(*args)`` in a slot's process, within ``timeout_s`` (``TimeoutError`` after it).

        Past ``timeout_s`` the process gets ``grace_s`` more to end itself (a rollout past its hard timeout stops
        its sandboxes before it kills its process group) before it is killed; a process that dies after
        ``timeout_s`` timed out too.
        """

        async with self._free:
            if self._admit is not None:
                self._admit()
            executor = self._idle.pop() if self._idle else self._start()
            started = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(executor, fn, *args), timeout=timeout_s + grace_s
                )
            except asyncio.TimeoutError:
                await self._retire(executor, kill=True)
                raise TimeoutError(f"Subprocess hard timeout after {timeout_s}s") from None
            except BrokenProcessPool:
                await self._retire(executor)  # its process died
                if time.monotonic() - started >= timeout_s:  # its own hard timeout ended it
                    raise TimeoutError(f"Subprocess hard timeout after {timeout_s}s") from None
                raise
            except Exception:
                self._idle.append(executor)  # the rollout failed; its process is fine
                raise
            except BaseException:
                # Cancelled: the rollout still runs in the process. Its sandboxes are reaped all the same.
                with contextlib.suppress(asyncio.CancelledError):
                    await self._retire(executor, kill=True)
                raise
            self._idle.append(executor)
            return result

    def _start(self) -> ProcessPoolExecutor:
        if self._directory is None:
            self._directory = scratch.mkdtemp("platoon-eval-sandboxes-")
        worker = f"eval-{uuid.uuid4().hex[:12]}"
        sandboxes_path = os.path.join(self._directory, f"{worker}.sandboxes")
        executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn"),
            initializer=_start_slot_process,
            # This process's labels (the evaluation's run label) as they are when the slot starts.
            initargs=(worker, sandboxes_path, sandbox_cleanup.labels(), self._initializer, self._initargs),
        )
        self._executors[executor] = sandboxes_path
        return executor

    async def _retire(self, executor: ProcessPoolExecutor, *, kill: bool = False) -> None:
        """Stop using ``executor``, killing its process first with ``kill``, then reap the sandboxes its process
        left.  A failed reap is logged, not raised; the reap goes on when the caller is cancelled meanwhile, and
        :meth:`aclose` waits for it."""

        sandboxes_path = self._executors.pop(executor, None)
        processes = list((getattr(executor, "_processes", None) or {}).values())
        if kill:
            # The rollout made its process a group leader (os.setpgrp), so this also stops what it started.
            for process in processes:
                _kill_process_group(process.pid)
        executor.shutdown(wait=False, cancel_futures=False)
        if sandboxes_path is None:
            return
        reaping = asyncio.ensure_future(_reap_sandboxes_left([sandboxes_path], processes))
        self._reaping.add(reaping)
        reaping.add_done_callback(self._reaping.discard)
        await asyncio.shield(reaping)

    async def aclose(self) -> None:
        """Stop every slot's process, then reap the sandboxes the processes left and remove their files."""

        executors, self._executors = list(self._executors), {}
        self._idle.clear()
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)
        if self._reaping:
            await asyncio.wait(set(self._reaping))
        directory, self._directory = self._directory, None
        if directory is not None:
            # The files of these processes, and of retired ones whose reap did not end.
            await _reap_sandboxes_left(sorted(glob.glob(os.path.join(directory, "*.sandboxes"))), [])
            shutil.rmtree(directory, ignore_errors=True)


async def _reap_sandboxes_left(sandboxes_paths: list[str], processes: list[Any]) -> None:
    """Reap the sandboxes that slot processes left, as their files list them, then delete the files.

    Waits first, within :data:`PROCESS_EXIT_SECONDS`, for ``processes`` to exit: the providers leave alone the
    sandboxes of an owner that still exists.  Never raises: a failure is logged.
    """

    try:
        if processes:
            await asyncio.to_thread(_join_processes, processes, PROCESS_EXIT_SECONDS)
        records = [record for path in sandboxes_paths for record in sandbox_cleanup.read_mirror(path)]
        if records:
            logger.warning(
                "An evaluation rollout's process left sandboxes running; reaping them (%s)",
                ", ".join(f"{record.provider} {record.labels}" for record in records),
            )
            reaped = await sandbox_cleanup.reap_records(records, SANDBOX_REAP_SECONDS)
            logger.warning("Reaped %d sandbox(es) of an evaluation rollout's process", reaped)
        for path in sandboxes_paths:
            with contextlib.suppress(OSError):
                os.unlink(path)
    except Exception:
        logger.warning("Cannot reap the sandboxes listed in %s", ", ".join(sandboxes_paths), exc_info=True)


def _join_processes(processes: list[Any], timeout_s: float) -> None:
    """Wait, within ``timeout_s`` in all, for ``processes`` (``multiprocessing`` processes) to exit."""

    deadline = time.monotonic() + timeout_s
    for process in processes:
        with contextlib.suppress(Exception):
            process.join(max(0.0, deadline - time.monotonic()))


def _kill_process_group(pid: int | None) -> None:
    if not pid:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        if os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)


class DefaultEvalGroupWorkflow:
    """Default rollout workflow for evaluation."""

    def __init__(
        self,
        rollout_fn: Callable[[Task, RolloutConfig], Any],
        get_task_fn: Callable[[str], Task],
        config: EvalWorkflowConfig,
        model_name: str,
        policy: EvalPolicy | None = None,
        reward_processor: RewardProcessor = lambda traj: (float(traj.get("reward", 0.0)), {}),
        success_fn: SuccessFn | None = None,
        workflow_metrics_fn: WorkflowMetricsFn | None = None,
        packages: Sequence[str] = (),
    ):
        """``policy``, when given, is installed as the default gateway of every process that runs rollouts.

        ``packages`` are the environments' ``package`` modules (``AutoEnvironment.packages``): subprocess rollouts
        import them first, so the harnesses and programs they register exist there as in the process that loaded
        the config.
        """
        self.rollout_fn = rollout_fn
        self.get_task_fn = get_task_fn
        self.config = config
        self.model_name = model_name
        self.policy = policy
        self.reward_processor = reward_processor
        self.success_fn = success_fn or _default_success_fn_factory(config.success_threshold)
        self.workflow_metrics_fn = workflow_metrics_fn
        self.packages = tuple(packages)
        self._slots: _ProcessSlots | None = None
        self._policy_installed = False
        self._key_refused: TinkerKeyRefusedError | None = None  # the Tinker service's refusal of the key, once one came

    @property
    def provenance(self) -> dict[str, Any]:
        """What each rollout's ``metadata.json`` records about the run that made it (:func:`run_provenance`):
        whether its judges ran, and its policy's identity, or ``backend: None`` without a policy.  ``EvalRunner``
        refuses to add rollouts to an output directory whose rollouts record another."""

        return run_provenance(self.policy, judges=bool(getattr(self.config.rollout_config, "judges", True)))

    def _get_rollout_config(self, output_dir: str) -> RolloutConfig:
        rollout_config = deepcopy(self.config.rollout_config)
        rollout_config.model_name = self.model_name
        rollout_config.return_dict = True
        rollout_config.train = False
        rollout_config.output_dir = output_dir
        return rollout_config

    def _install_policy(self) -> None:
        """Serve the policy from this process, once, for in-process rollouts."""
        if self.policy is not None and not self._policy_installed:
            self.policy.install()
            self._policy_installed = True

    def record_from_collection(
        self,
        task_id: str,
        rollout_index: int,
        trajectory_collection: dict[str, Any],
        wall_time_seconds: float | None = None,
        source_path: str | None = None,
    ) -> EvalRolloutRecord:
        root = _get_root_trajectory(trajectory_collection)
        if root is None:
            raise ValueError("No root trajectory found in trajectory collection")

        status = outcome_status(root)
        error = None
        if status not in (None, "scored") and not (status == "timeout" and self.config.count_timeout_as_failure):
            # An unmeasured rollout (a crash, a grader or env failure, an invalid grade, a timeout) is an
            # errored one, never a valid failure with reward 0.
            error = root.get("error_message") or f"Outcome {status}"
        accounting = rollout_accounting(trajectory_collection)
        stop_run = _run_wide_problem(root, accounting)
        if stop_run is not None:
            error = stop_run
        elif accounting["backend"]["failed_calls"]:
            # An outage must not be scored as the policy's failure, whatever its harness did next: the rollout is
            # errored, and a resumed evaluation runs it again (retry_errored).
            failed = accounting["backend"]["failed_calls"]
            error = f"backend_unavailable: {failed} model call(s) failed after retries"
        reward, reward_components = self.reward_processor(root)
        success = self.success_fn(trajectory_collection, reward, reward_components)
        num_steps_total, num_steps_root, num_steps_sub, depth_counts, depth_steps = _compute_depth_stats(
            trajectory_collection
        )

        workflow_metrics = {}
        if self.workflow_metrics_fn is not None:
            try:
                workflow_metrics = self.workflow_metrics_fn(trajectory_collection)
            except Exception:
                logger.exception("workflow_metrics_fn failed for task %s rollout %d", task_id, rollout_index)
                workflow_metrics = {}

        return EvalRolloutRecord(
            task_id=task_id,
            rollout_index=rollout_index,
            success=success,
            reward=reward,
            reward_components=reward_components,
            num_steps_total=num_steps_total,
            num_steps_root=num_steps_root,
            num_steps_subtrajectories=num_steps_sub,
            num_subtrajectories=sum(depth_counts.values()),
            subtrajectory_depth_counts=depth_counts,
            subtrajectory_depth_steps=depth_steps,
            wall_time_seconds=wall_time_seconds,
            workflow_metrics=workflow_metrics,
            source_path=source_path,
            error=error,
            trajectory_collection=trajectory_collection,
            outcome_status=status,
            reward_key=trained_reward_key(root),
            usage=accounting["usage"],
            cost=accounting["cost"],
            backend=accounting["backend"],
            judges=accounting["judges"],
            stop_run=stop_run,
        )

    def _process_slots(self) -> _ProcessSlots:
        if self._slots is None:
            from platoon.train.auto import import_environment_packages

            self._slots = _ProcessSlots(
                self.config.subprocess_max_workers or self.config.num_concurrent_workers,
                initializer=import_environment_packages,
                initargs=(self.packages,),
                admit=self._admit_rollout,
            )
        return self._slots

    def _admit_rollout(self) -> None:
        """Before a rollout takes a worker process: once the Tinker service refused the key, raise that refusal, so
        that no rollout that waited for a worker process asks the service again."""

        if self._key_refused is not None:
            raise self._key_refused

    @staticmethod
    def _subprocess_hard_timeout_seconds(
        rollout_config: RolloutConfig, task: Task | None = None, rollout: str | None = None
    ) -> int:
        """The slot's hard deadline, as the slot itself counts it (``subprocess_worker._hard_timeout_seconds``), for a
        rollout of ``task`` by the rollout function ``rollout`` (a reference)."""

        from platoon.eval.subprocess_worker import _hard_timeout_seconds

        return _hard_timeout_seconds({"timeout": rollout_config.timeout}, task, rollout)

    async def arun_rollout(self, data: dict[str, Any], rollout_index: int, output_dir: str) -> EvalRolloutRecord:
        task_id = str(data["task_id"])
        start = time.perf_counter()
        try:
            rollout_config = self._get_rollout_config(output_dir=output_dir)
            task = self.get_task_fn(task_id)
            if rollout_config.max_steps is not None:
                task.max_steps = rollout_config.max_steps
            if self.config.use_subprocesses:
                from platoon.eval.subprocess_worker import PARENT_GRACE_SECONDS, run_rollout_subprocess
                from platoon.train.tinker.workers.protocol import validated_ref

                rollout_ref = validated_ref(self.rollout_fn, what="evaluation rollout function")
                result = await self._process_slots().run(
                    self._subprocess_hard_timeout_seconds(rollout_config, task, rollout_ref),
                    run_rollout_subprocess,
                    rollout_ref,
                    task,
                    asdict(rollout_config),
                    self.policy,
                    # Past its hard timeout the process stops its sandboxes, then ends itself.
                    grace_s=PARENT_GRACE_SECONDS,
                )
            else:
                self._install_policy()
                result = await asyncio.create_task(self.rollout_fn(task, rollout_config))
                if hasattr(result, "to_dict"):
                    result = result.to_dict()

            if not isinstance(result, dict):
                raise TypeError(
                    f"Expected the rollout to return a trajectory collection dict, got {type(result).__name__}"
                )
            _write_rollout_events(output_dir, task_id, result)

            return self.record_from_collection(
                task_id=task_id,
                rollout_index=rollout_index,
                trajectory_collection=result,
                wall_time_seconds=time.perf_counter() - start,
            )
        except TinkerKeyRefusedError as refusal:
            # Every rollout would be refused alike: the evaluation ends with it (EvalRunner.run_rollout_stage).
            self._key_refused = self._key_refused or refusal
            raise
        except Exception as e:
            logger.exception("Evaluation rollout failed for task %s rollout %d: %s", task_id, rollout_index, e)
            return EvalRolloutRecord(
                task_id=task_id,
                rollout_index=rollout_index,
                success=False,
                reward=0.0,
                reward_components={},
                num_steps_total=0,
                num_steps_root=0,
                num_steps_subtrajectories=0,
                num_subtrajectories=0,
                wall_time_seconds=time.perf_counter() - start,
                error=str(e),
            )

    async def aclose(self) -> None:
        if self._slots is not None:
            slots, self._slots = self._slots, None
            await slots.aclose()
