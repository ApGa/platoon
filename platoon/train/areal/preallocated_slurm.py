"""AReaL scheduler that launches workers inside a preallocated Slurm job."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from areal.api import Job, Worker
from areal.api.cli_args import SchedulingSpec, SchedulingStrategyType
from areal.infra.scheduler.exceptions import (
    WorkerCreationError,
    WorkerFailedError,
    WorkerNotFoundError,
)
from areal.infra.scheduler.slurm import SlurmScheduler, SlurmWorkerInfo
from areal.infra.utils.proc import build_streaming_log_cmd
from areal.utils import logging

logger = logging.getLogger("PreallocatedSlurmScheduler")


class PreallocatedSlurmScheduler(SlurmScheduler):
    """Run AReaL Slurm workers as job steps in a preallocated Slurm job.

    Upstream ``SlurmScheduler`` writes a child sbatch script for each separated
    worker role. This variant keeps the same RPC/name-resolve worker model, but
    starts each role with ``srun`` directly so the outer user-owned sbatch script
    remains responsible for account, partition, time limit, and allocation size.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._role_processes: dict[str, subprocess.Popen] = {}
        self._role_commands: dict[str, str] = {}
        self._role_idle_guard_processes: dict[str, subprocess.Popen] = {}
        self._role_idle_guard_commands: dict[str, str] = {}
        self._stopping_roles: set[str] = set()
        # Round-robin cursor used to pin each separated role to distinct nodes of
        # the allocation. Without this, every single-node srun step launches with
        # `--overlap` and no `--nodelist`, so Slurm stacks them all on the first
        # node (e.g. actor + sglang sharing GPUs) while the rest of the allocation
        # idles. See _allocation_nodes / create_workers.
        self._separated_node_cursor = 0
        self._alloc_nodes_cache: list[str] | None = None

        # AReaL configures workers one at a time.  For Megatron workers each
        # /configure request performs expensive model setup, so the serial loop
        # costs tens of minutes on a multi-node job.  Configure different hosts
        # concurrently while retaining a single stream per host (which avoids
        # eight workers on one node competing for CPU, disk, and host memory).
        configure_concurrency = os.environ.get(
            "PLATOON_AREAL_PREALLOC_CONFIGURE_CONCURRENCY", "16"
        )
        try:
            self._configure_concurrency = int(configure_concurrency)
        except ValueError as exc:
            raise ValueError(
                "PLATOON_AREAL_PREALLOC_CONFIGURE_CONCURRENCY must be a positive integer, "
                f"got {configure_concurrency!r}"
            ) from exc
        if self._configure_concurrency <= 0:
            raise ValueError(
                "PLATOON_AREAL_PREALLOC_CONFIGURE_CONCURRENCY must be a positive integer, "
                f"got {configure_concurrency!r}"
            )
        self._configure_lock = threading.Lock()
        self._configured_worker_generations: dict[str, tuple[int, ...]] = {}

    @staticmethod
    def _worker_host_key(worker_info: SlurmWorkerInfo) -> str:
        """Return a stable host key after worker network discovery."""

        return worker_info.node or worker_info.worker.ip or worker_info.worker.id

    def _configure_worker(self, worker_info: SlurmWorkerInfo, worker_rank: int) -> None:
        """Configure one role in bounded host-parallel streams.

        Upstream calls this method from a serial loop once per worker.  The
        first call configures the complete current generation for that role;
        subsequent loop calls are no-ops.  Calling the upstream implementation
        directly inside the pool preserves its readiness checks, RPC payload,
        timeout, and exception translation.
        """

        role_workers = self._workers.get(worker_info.role)
        if not role_workers or not any(w is worker_info for w in role_workers):
            # Defensive fallback for an upstream lifecycle we do not own.
            SlurmScheduler._configure_worker(self, worker_info, worker_rank)
            return

        generation = tuple(id(w) for w in role_workers)
        with self._configure_lock:
            if self._configured_worker_generations.get(worker_info.role) == generation:
                return

            workers_by_host: dict[
                str, list[tuple[int, SlurmWorkerInfo]]
            ] = defaultdict(list)
            for rank, role_worker in enumerate(role_workers):
                workers_by_host[self._worker_host_key(role_worker)].append(
                    (rank, role_worker)
                )

            host_count = len(workers_by_host)
            max_workers = min(self._configure_concurrency, host_count)
            logger.info(
                "Configuring %s workers for role '%s' across %s hosts "
                "(%s concurrent host streams)",
                len(role_workers),
                worker_info.role,
                host_count,
                max_workers,
            )
            started = time.monotonic()

            def configure_host(
                host_workers: list[tuple[int, SlurmWorkerInfo]],
            ) -> tuple[int, Exception] | None:
                for rank, host_worker in host_workers:
                    try:
                        SlurmScheduler._configure_worker(self, host_worker, rank)
                    except Exception as exc:  # preserve the original upstream exception
                        return rank, exc
                return None

            failures: list[tuple[int, Exception]] = []
            with ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix=f"areal-configure-{worker_info.role}",
            ) as executor:
                futures = {
                    executor.submit(configure_host, host_workers): host
                    for host, host_workers in workers_by_host.items()
                }
                for future in as_completed(futures):
                    try:
                        failure = future.result()
                    except Exception as exc:
                        # This only covers failures in our wrapper itself; normal
                        # upstream configuration errors are returned with rank.
                        first_rank = workers_by_host[futures[future]][0][0]
                        failure = (first_rank, exc)
                    if failure is not None:
                        failures.append(failure)

            if failures:
                failures.sort(key=lambda item: item[0])
                failed_ranks = [rank for rank, _ in failures]
                logger.error(
                    "Worker configuration failed for role '%s' on rank(s) %s",
                    worker_info.role,
                    failed_ranks,
                )
                # A failed generation is deliberately not cached, so a caller
                # can retry after fixing a transient worker/RPC issue.
                raise failures[0][1]

            self._configured_worker_generations[worker_info.role] = generation
            logger.info(
                "Configured %s workers for role '%s' in %.1fs",
                len(role_workers),
                worker_info.role,
                time.monotonic() - started,
            )

    @staticmethod
    def _split_env_list(name: str) -> list[str]:
        value = os.environ.get(name, "")
        return shlex.split(value) if value else []

    def _srun_base_args(self) -> list[str]:
        return self._split_env_list("PLATOON_AREAL_PREALLOC_SRUN_ARGS")

    def _srun_binary(self) -> str:
        return os.environ.get("PLATOON_AREAL_PREALLOC_SRUN_BIN", "srun")

    @staticmethod
    def _has_overlap_arg(args: list[str]) -> bool:
        return any(arg == "--overlap" or arg.startswith("--overlap=") for arg in args)

    @staticmethod
    def _truthy_env(value: str | None) -> bool:
        return value is not None and value.lower() not in {"0", "false", "no", "off"}

    def _container_srun_args(self) -> list[str]:
        if os.environ.get("PLATOON_AREAL_PREALLOC_USE_PYXIS", "1") == "0":
            return []

        args: list[str] = []
        image = os.environ.get("PLATOON_AREAL_PREALLOC_CONTAINER_IMAGE")
        mounts = os.environ.get("PLATOON_AREAL_PREALLOC_CONTAINER_MOUNTS")
        workdir = os.environ.get("PLATOON_AREAL_PREALLOC_CONTAINER_WORKDIR")
        if image:
            args.append(f"--container-image={image}")
        if mounts:
            args.append(f"--container-mounts={mounts}")
        if workdir:
            args.append(f"--container-workdir={workdir}")
        return args

    def _worker_preamble(self) -> list[str]:
        preamble = os.environ.get("PLATOON_AREAL_PREALLOC_WORKER_PREAMBLE")
        if not preamble:
            return []
        return [preamble]

    def _allocation_nodes(self) -> list[str]:
        """Expand the current Slurm allocation into an ordered list of hostnames.

        Used to pin separated roles to distinct nodes. Best-effort: returns an
        empty list when run outside a Slurm allocation or when ``scontrol`` is
        unavailable, in which case node spreading is silently skipped.
        """

        if self._alloc_nodes_cache is not None:
            return self._alloc_nodes_cache

        nodes: list[str] = []
        nodelist = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get(
            "SLURM_NODELIST"
        )
        if nodelist:
            try:
                out = subprocess.check_output(
                    ["scontrol", "show", "hostnames", nodelist],
                    text=True,
                )
                nodes = [n for n in out.split() if n]
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning(
                    "Could not expand SLURM node list %r for node spreading: %s",
                    nodelist,
                    exc,
                )
                nodes = []
        self._alloc_nodes_cache = nodes
        return nodes

    def _assign_separated_nodelist(self, role: str, nodes: int) -> str | None:
        """Pick the next ``nodes`` allocation hostnames for a separated role.

        Returns a comma-separated ``--nodelist`` value, or ``None`` when spreading
        is disabled or the allocation can't be resolved (falls back to Slurm's
        default placement).
        """

        if not self._truthy_env(
            os.environ.get("PLATOON_AREAL_PREALLOC_SPREAD_NODES", "1")
        ):
            return None

        alloc_nodes = self._allocation_nodes()
        if not alloc_nodes or nodes <= 0 or nodes > len(alloc_nodes):
            return None

        start = self._separated_node_cursor
        if start + nodes > len(alloc_nodes):
            # Not enough distinct nodes left before the end; wrap to the front.
            start = 0
        assigned = alloc_nodes[start : start + nodes]
        self._separated_node_cursor = (start + nodes) % len(alloc_nodes)
        nodelist = ",".join(assigned)
        logger.info("Pinning separated role '%s' to node(s): %s", role, nodelist)
        return nodelist

    def _build_role_srun_command(
        self,
        role: str,
        replicas: int,
        nodes: int,
        total_gpus: int,
        cpus_per_task: int,
        mem_per_task: int,
        schedulings: list[SchedulingSpec],
        nodelist: str | None,
        exclude: str | None,
    ) -> str:
        """Build the worker role command normally embedded in AReaL's sbatch."""

        spec = schedulings[0]
        ntasks_per_node = replicas // nodes if nodes > 0 else replicas
        if total_gpus % self.n_gpus_per_node != 0:
            raise ValueError(
                "Preallocated Slurm only supports allocating entire nodes. "
                f"Requesting {total_gpus} GPUs but each node has {self.n_gpus_per_node}."
            )

        mem_per_cpu = (
            mem_per_task // cpus_per_task if cpus_per_task > 0 else mem_per_task
        )

        rpc_cmd = spec.cmd or "python -m areal.infra.rpc.rpc_server"
        rpc_cmd_flags = [
            "--experiment-name",
            self.experiment_name,
            "--trial-name",
            self.trial_name,
            "--role",
            role,
            "--name-resolve-type",
            self.name_resolve_config.type,
            "--nfs-record-root",
            self.name_resolve_config.nfs_record_root,
            "--etcd3-addr",
            self.name_resolve_config.etcd3_addr,
        ]
        if self.fileroot:
            rpc_cmd_flags.extend(["--fileroot", str(self.fileroot)])
        rpc_cmd = " ".join([rpc_cmd] + [shlex.quote(str(flag)) for flag in rpc_cmd_flags])

        bash_cmds = (spec.additional_bash_cmds or []).copy()
        if total_gpus > 0:
            gpus_per_task = spec.gpu
            if gpus_per_task == 1:
                cuda_setup_cmd = (
                    f"export CUDA_VISIBLE_DEVICES=$((SLURM_LOCALID * {gpus_per_task}))"
                )
            else:
                cuda_setup_cmd = (
                    f"export CUDA_VISIBLE_DEVICES=$(seq -s, $((SLURM_LOCALID * {gpus_per_task})) "
                    f"$((SLURM_LOCALID * {gpus_per_task} + {gpus_per_task} - 1)))"
                )
            bash_cmds.insert(0, cuda_setup_cmd)
            bash_cmds.insert(1, "export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")

        bash_cmds.extend(self._worker_preamble())
        bash_cmds.append(rpc_cmd)
        worker_script = ";\n".join(cmd.strip() for cmd in bash_cmds if cmd.strip())
        worker_cmd = ["/bin/bash", "-lc", worker_script]

        base_srun_args = self._srun_base_args()
        exclusive_env = os.environ.get("PLATOON_AREAL_PREALLOC_WORKER_EXCLUSIVE")
        use_exclusive = (
            self._truthy_env(exclusive_env)
            if exclusive_env is not None
            else not self._has_overlap_arg(base_srun_args)
        )

        srun_args = [
            self._srun_binary(),
            *base_srun_args,
            f"--nodes={nodes}",
            f"--ntasks={replicas}",
            f"--ntasks-per-node={ntasks_per_node}",
            f"--cpus-per-task={cpus_per_task}",
            f"--mem-per-cpu={mem_per_cpu}M",
        ]
        if use_exclusive:
            srun_args.append("--exclusive")
        if total_gpus > 0:
            gpu_flag_template = os.environ.get(
                "PLATOON_AREAL_PREALLOC_GPU_FLAG",
                "--gpus-per-node={gpus}",
            )
            srun_args.append(gpu_flag_template.format(gpus=self.n_gpus_per_node))
        if nodelist:
            srun_args.append(f"--nodelist={nodelist}")
        if exclude:
            srun_args.append(f"--exclude={exclude}")
        srun_args.extend(self._container_srun_args())
        srun_args.extend(worker_cmd)

        role_log = self._log_path_of(role)
        merged_log = self._merged_log_path()
        return build_streaming_log_cmd(srun_args, role_log, merged_log, role)

    def _launch_role_process(self, role: str, command: str) -> subprocess.Popen:
        logger.info("Launching role '%s' inside preallocated Slurm job", role)
        logger.info("Preallocated Slurm srun command for role '%s': %s", role, command)
        return subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            start_new_session=True,
        )

    @staticmethod
    def _idle_guard_enabled(role: str) -> bool:
        return role in {"actor", "rollout"} and PreallocatedSlurmScheduler._truthy_env(
            os.environ.get("PLATOON_AREAL_ROLLOUT_IDLE_GUARD", "0")
        )

    def _build_idle_guard_command(
        self, role: str, nodes: int, nodelist: str | None
    ) -> str:
        """Build a sibling srun pinned to an exact guarded-role topology."""

        if not nodelist:
            raise WorkerCreationError(
                role,
                "Idle guard requires an exact role nodelist",
            )
        script = os.environ.get("PLATOON_AREAL_ROLLOUT_IDLE_GUARD_SCRIPT")
        python = os.environ.get("PLATOON_AREAL_ROLLOUT_IDLE_GUARD_PYTHON")
        ready_root = os.environ.get("PLATOON_AREAL_ROLLOUT_IDLE_GUARD_READY_DIR")
        log_prefix_root = os.environ.get("PLATOON_AREAL_ROLLOUT_IDLE_GUARD_LOG_PREFIX")
        missing = [
            name
            for name, value in (
                ("PLATOON_AREAL_ROLLOUT_IDLE_GUARD_SCRIPT", script),
                ("PLATOON_AREAL_ROLLOUT_IDLE_GUARD_PYTHON", python),
                ("PLATOON_AREAL_ROLLOUT_IDLE_GUARD_READY_DIR", ready_root),
                ("PLATOON_AREAL_ROLLOUT_IDLE_GUARD_LOG_PREFIX", log_prefix_root),
            )
            if not value
        ]
        if missing:
            raise WorkerCreationError(
                role,
                f"Idle guard configuration is missing: {', '.join(missing)}",
            )
        ready_dir = f"{ready_root}/{role}"
        log_prefix = f"{log_prefix_root}-{role}"

        guard_cmd = " ".join(
            [
                "exec",
                shlex.quote(python),
                "-u",
                shlex.quote(script),
                "--expected-devices",
                "1",
                "--ready-dir",
                shlex.quote(ready_dir),
            ]
        )
        bash_cmds = [*self._worker_preamble(), guard_cmd]
        guard_script = ";\n".join(command.strip() for command in bash_cmds if command.strip())
        task_count = nodes * self.n_gpus_per_node
        srun_args = [
            self._srun_binary(),
            "--overlap",
            "--exact",
            "--unbuffered",
            "--kill-on-bad-exit=1",
            f"--job-name={role}-idle-guard",
            f"--nodes={nodes}",
            f"--ntasks={task_count}",
            f"--ntasks-per-node={self.n_gpus_per_node}",
            "--cpu-bind=none",
            "--cpus-per-task=1",
            "--mem-per-cpu=1024M",
            "--gpus-per-task=1",
            "--gpu-bind=single:1",
            f"--nodelist={nodelist}",
            f"--output={log_prefix}-%N-%t.log",
            f"--error={log_prefix}-%N-%t.log",
            *self._container_srun_args(),
            "/bin/bash",
            "-lc",
            guard_script,
        ]
        return " ".join(shlex.quote(str(arg)) for arg in srun_args)

    def _launch_idle_guard_process(self, role: str, command: str) -> subprocess.Popen:
        logger.info("Launching GPU idle guard for role '%s'", role)
        logger.info("Preallocated Slurm GPU idle guard command for '%s': %s", role, command)
        return subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            start_new_session=True,
        )

    @staticmethod
    def _prepare_idle_guard_ready_dir(role: str) -> Path:
        ready_value = os.environ["PLATOON_AREAL_ROLLOUT_IDLE_GUARD_READY_DIR"]
        ready_dir = Path(ready_value) / role
        ready_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("*.ready", ".*.tmp"):
            for marker in ready_dir.glob(pattern):
                marker.unlink()
        return ready_dir

    def _wait_for_idle_guard(
        self,
        role: str,
        proc: subprocess.Popen,
        ready_dir: Path,
        expected_tasks: int,
    ) -> None:
        timeout = float(os.environ.get("PLATOON_AREAL_ROLLOUT_IDLE_GUARD_READY_TIMEOUT", "120"))
        if timeout <= 0:
            raise ValueError("PLATOON_AREAL_ROLLOUT_IDLE_GUARD_READY_TIMEOUT must be positive")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = proc.poll()
            if status is not None:
                raise RuntimeError(f"GPU idle guard exited before readiness with status {status}")
            if all((ready_dir / f"{rank}.ready").is_file() for rank in range(expected_tasks)):
                logger.info(
                    "GPU idle guard for role '%s' is ready on %s GPU task(s)",
                    role,
                    expected_tasks,
                )
                return
            time.sleep(0.25)
        ready_count = sum((ready_dir / f"{rank}.ready").is_file() for rank in range(expected_tasks))
        raise TimeoutError(
            f"GPU idle guard for {role} timed out: {ready_count}/{expected_tasks} tasks"
        )

    def _start_idle_guard(
        self,
        role: str,
        nodes: int,
        nodelist: str | None,
    ) -> None:
        command = self._build_idle_guard_command(role, nodes, nodelist)
        ready_dir = self._prepare_idle_guard_ready_dir(role)
        proc = self._launch_idle_guard_process(role, command)
        self._role_idle_guard_processes[role] = proc
        self._role_idle_guard_commands[role] = command
        try:
            self._wait_for_idle_guard(
                role,
                proc,
                ready_dir,
                nodes * self.n_gpus_per_node,
            )
        except Exception as exc:
            self._terminate_idle_guard_process(role)
            raise WorkerCreationError(
                role,
                "GPU idle guard failed to become ready",
                str(exc),
            ) from exc

    def _terminate_idle_guard_process(self, role: str) -> None:
        proc = self._role_idle_guard_processes.pop(role, None)
        self._role_idle_guard_commands.pop(role, None)
        if proc is None:
            return
        try:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Force killing GPU idle guard for role '%s'", role)
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
        except ProcessLookupError:
            pass

    def _check_job_status(self, role: str) -> None:
        """Check the local srun processes instead of querying child Slurm jobs."""

        if role in self._colocated_roles:
            return self._check_job_status(self._colocated_roles[role])

        guard_proc = self._role_idle_guard_processes.get(role)
        if guard_proc is not None and role not in self._stopping_roles:
            guard_status = guard_proc.poll()
            if guard_status is not None:
                guard_pid = guard_proc.pid
                self._role_idle_guard_processes.pop(role, None)
                self._role_idle_guard_commands.pop(role, None)
                self._terminate_role_process(role)
                raise WorkerFailedError(
                    f"{role}-idle-guard/*",
                    guard_pid,
                    f"GPU idle guard exited unexpectedly with status {guard_status}",
                )

        proc = self._role_processes.get(role)
        if proc is None:
            if role in self._workers:
                return
            raise WorkerNotFoundError(f"Role '{role}' not found")

        status = proc.poll()
        if status is None:
            return
        if role in self._stopping_roles:
            return

        self._terminate_idle_guard_process(role)
        logs = self._read_log_tail(role)
        raise WorkerFailedError(
            f"{role}/*",
            proc.pid,
            f"srun process exited with status {status}. Logs:\n{logs}",
        )

    def _terminate_role_process(self, role: str) -> None:
        proc = self._role_processes.pop(role, None)
        if proc is None:
            return

        self._stopping_roles.add(role)
        try:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Force killing preallocated Slurm role '%s'", role)
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
        except ProcessLookupError:
            pass
        finally:
            self._stopping_roles.discard(role)

    def create_workers(self, job: Job, *args, **kwargs) -> list[str]:
        role = job.role
        if ":" in role:
            raise ValueError("Invalid worker name.")
        num_workers = job.replicas

        if role in self._workers:
            raise WorkerCreationError(role, f"Role '{role}' already exists")
        if num_workers <= 0:
            raise WorkerCreationError(
                role, "Invalid configuration", "replicas must be greater than 0"
            )

        schedulings = self._prepare_worker_specs(role, num_workers, job.tasks)
        strategy = job.scheduling_strategy
        strategy_type = SchedulingStrategyType(strategy.type)
        colocate_role = strategy.target

        logger.info(
            "Creating %s workers for role '%s' (strategy: %s, colocate_with: %s)",
            num_workers,
            role,
            strategy_type,
            colocate_role,
        )

        if strategy_type == SchedulingStrategyType.colocation:
            if not colocate_role:
                raise WorkerCreationError(
                    role,
                    "Invalid strategy",
                    "Colocation strategy requires target role to be specified",
                )
            if colocate_role not in self._workers:
                raise WorkerNotFoundError(
                    f"Cannot colocate with role '{colocate_role}' - role not found"
                )

            target_workers = self._workers[colocate_role]
            if num_workers != len(target_workers):
                raise WorkerCreationError(
                    role,
                    "Replica count mismatch",
                    f"Colocated role must have same replica count as target "
                    f"({num_workers} != {len(target_workers)})",
                )

            if strategy.fork:
                return self.fork_workers(role, colocate_role)

            worker_ids = [w.worker.id for w in target_workers]
            self._colocated_roles[role] = colocate_role
            return worker_ids

        if strategy_type != SchedulingStrategyType.separation:
            raise ValueError(f"Unknown scheduling strategy type: {strategy_type}")

        spec = schedulings[0]
        total_gpus = spec.gpu * num_workers
        nodes = max(1, (total_gpus + self.n_gpus_per_node - 1) // self.n_gpus_per_node)
        cpus_per_task = spec.cpu
        mem_per_task = spec.mem * 1024
        nodelist = spec.nodelist
        if not nodelist:
            # AReaL leaves nodelist unset, so every single-node `--overlap` step
            # otherwise lands on the first node. Pin separated roles to distinct
            # nodes so e.g. the trainer and colocated sglang don't share GPUs.
            nodelist = self._assign_separated_nodelist(role, nodes)

        command = self._build_role_srun_command(
            role=role,
            replicas=num_workers,
            nodes=nodes,
            total_gpus=total_gpus,
            cpus_per_task=cpus_per_task,
            mem_per_task=mem_per_task,
            schedulings=schedulings,
            nodelist=nodelist,
            exclude=spec.exclude,
        )
        guard_started = False
        if self._idle_guard_enabled(role):
            self._start_idle_guard(role, nodes, nodelist)
            guard_started = True
        try:
            proc = self._launch_role_process(role, command)
        except Exception:
            if guard_started:
                self._terminate_idle_guard_process(role)
            raise

        # Track the live srun before constructing/registering worker metadata so
        # any later exception can terminate both siblings without leaking a step.
        self._role_processes[role] = proc
        self._role_commands[role] = command
        try:
            workers: list[SlurmWorkerInfo] = []
            worker_ids: list[str] = []
            for idx in range(num_workers):
                worker_id = f"{role}/{idx}"
                worker = Worker(id=worker_id, ip="", worker_ports=[], engine_ports=[])
                worker_spec = (
                    schedulings[idx]
                    if len(schedulings) == num_workers
                    else schedulings[0]
                )
                workers.append(
                    SlurmWorkerInfo(
                        worker=worker,
                        role=role,
                        slurm_job_id=proc.pid,
                        task_index=idx,
                        discovered=False,
                        spec=worker_spec,
                    )
                )
                worker_ids.append(worker_id)

            self._workers[role] = workers
            self._jobs[role] = proc.pid
        except Exception:
            self._workers.pop(role, None)
            self._jobs.pop(role, None)
            self._terminate_idle_guard_process(role)
            self._terminate_role_process(role)
            self._role_commands.pop(role, None)
            raise

        return worker_ids

    def delete_workers(self, role: str | None = None):
        if role is None:
            for r in list(self._colocated_roles.keys()):
                self.delete_workers(r)
            for r in list(self._workers.keys()):
                self.delete_workers(r)
            return

        if role in self._colocated_roles:
            if role in self._workers:
                logger.info("Removing forked role '%s' (managed by parent worker)", role)
                del self._workers[role]
            self._configured_worker_generations.pop(role, None)
            del self._colocated_roles[role]
            return

        if role not in self._workers:
            logger.warning("Role '%s' not found, skipping deletion", role)
            return

        logger.info("Deleting preallocated Slurm workers for role '%s'", role)
        self._stopping_roles.add(role)
        try:
            self._terminate_idle_guard_process(role)
            self._terminate_role_process(role)
        finally:
            self._stopping_roles.discard(role)
        del self._workers[role]
        self._configured_worker_generations.pop(role, None)
        self._jobs.pop(role, None)
        self._role_commands.pop(role, None)

    @classmethod
    def from_scheduler_config(cls, config):
        """Construct the scheduler from an AReaL experiment config."""

        return cls(exp_config=config)


def scheduling_spec_asdict(spec: SchedulingSpec) -> dict:
    """Return a serializable SchedulingSpec dict for tests/debugging."""

    return asdict(spec)
