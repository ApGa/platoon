"""AReaL scheduler for workers inside an existing PBS Pro allocation.

ABCI allocates whole eight-GPU H200 nodes and writes each physical hostname
eight times to ``PBS_NODEFILE`` when the job requests ``select=N:mpiprocs=8``.
This scheduler preserves first-seen allocation order, reserves disjoint node
ranges for separated roles, and starts one ``pbsdsh`` supervisor per physical
host.  The supervisors assign explicit worker/local ranks and GPU slices.
"""

from __future__ import annotations

import fcntl
import getpass
import hashlib
import inspect
import json
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

from areal.api import Job, Worker
from areal.api.cli_args import SchedulingSpec, SchedulingStrategyType
from areal.infra.scheduler.exceptions import (
    WorkerCreationError,
    WorkerFailedError,
    WorkerNotFoundError,
)
from areal.infra.scheduler.slurm import SlurmScheduler, SlurmWorkerInfo
from areal.utils import logging

from platoon.train.areal.fork_lifecycle import (
    cleanup_dependent_aliases_before_owner,
    cleanup_forked_alias_workers,
)
from platoon.train.areal.preallocated_slurm import PreallocatedSlurmScheduler

logger = logging.getLogger("PreallocatedPBSScheduler")

_MANIFEST_VERSION = 1
_LOCK_METADATA_VERSION = 1
_GPU_VISIBILITY_MODES = {"legacy", "nvidia-container-cli"}
_DEFAULT_NVIDIA_SMI_BINARY = "/usr/bin/nvidia-smi"
_FORK_WORKER_PYTHON_ENV = "PLATOON_AREAL_PBS_WORKER_PYTHON"
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_HOST_ENV_EXACT = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LANGUAGE",
    "TZ",
    "TMPDIR",
    "TMP",
    "TEMP",
    "PWD",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "PKG_CONFIG_PATH",
    "MANPATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_ROOT",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
}
_HOST_ENV_PREFIXES = (
    "PBS_",
    "LC_",
    "CUDA_",
    "NVIDIA_",
    "NCCL_",
    "UCX_",
    "UCC_",
    "OMPI_",
    "OPAL_",
    "PMIX_",
    "PMI_",
    "FI_",
    "HPCX_",
    "HCOLL_",
    "SHARP_",
    "NVSHMEM_",
    "I_MPI_",
    "MODULE",
    "LMFILES_",
    "PE_",
)
_HOST_SECRET_PREFIXES = (
    "WANDB_",
    "OPENAI_",
    "LITELLM_",
    "HF_",
    "HUGGINGFACE_",
    "HUGGING_FACE_",
    "AWS_",
    "AZURE_",
    "GCP_",
    "GOOGLE_",
    "ANTHROPIC_",
    "COHERE_",
    "GROQ_",
    "MISTRAL_",
    "TOGETHER_",
    "FIREWORKS_",
    "SERPER_",
    "JINA_",
    "APPTAINER_",
    "SINGULARITY_",
)


def _validated_fork_worker_python(value: str) -> str:
    """Validate a worker-container interpreter without resolving its target.

    The production worker venv uses an absolute symlink whose target exists in
    the training SIF, not on the controller host.  ``resolve()``/``exists()``
    would therefore reject the valid immutable environment.  We instead
    validate the shared-filesystem directory entry itself and require either an
    absolute symlink target or an executable regular file.
    """

    if not value or "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{_FORK_WORKER_PYTHON_ENV} must be a non-empty single-line path")
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(value) != value:
        raise ValueError(f"{_FORK_WORKER_PYTHON_ENV} must be an absolute normalized path")
    try:
        entry = path.lstat()
    except OSError as exc:
        raise ValueError(f"{_FORK_WORKER_PYTHON_ENV} does not name an existing directory entry") from exc
    if stat.S_ISLNK(entry.st_mode):
        target = os.readlink(path)
        if not os.path.isabs(target):
            raise ValueError(f"{_FORK_WORKER_PYTHON_ENV} symlink target must be absolute")
    elif stat.S_ISREG(entry.st_mode):
        if not os.access(path, os.X_OK):
            raise ValueError(f"{_FORK_WORKER_PYTHON_ENV} regular file must be executable")
    else:
        raise ValueError(f"{_FORK_WORKER_PYTHON_ENV} must be a symlink or executable regular file")
    return value


class _ForkWorkerPythonSession:
    """Instance-local adapter for AReaL's otherwise fixed fork interpreter."""

    def __init__(self, session: Any, *, worker_python: str, controller_python: str):
        self._session = session
        self._worker_python = worker_python
        self._controller_python = controller_python

    def post(self, url: str, *args, **kwargs):
        if url.endswith("/fork"):
            payload = kwargs.get("json")
            if not isinstance(payload, dict):
                raise RuntimeError("AReaL fork request is missing its JSON payload")
            raw_cmd = payload.get("raw_cmd")
            if not isinstance(raw_cmd, list) or not raw_cmd:
                raise RuntimeError("AReaL fork request is missing raw_cmd")
            if raw_cmd[0] != self._controller_python:
                raise RuntimeError("AReaL fork command interpreter changed unexpectedly")
            rewritten_payload = dict(payload)
            rewritten_cmd = list(raw_cmd)
            rewritten_cmd[0] = self._worker_python
            rewritten_payload["raw_cmd"] = rewritten_cmd
            kwargs = dict(kwargs)
            kwargs["json"] = rewritten_payload
        return self._session.post(url, *args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._session, name)


_HOST_SECRET_SUFFIXES = (
    "_TOKEN",
    "_API_KEY",
    "_SECRET",
    "_PASSWORD",
    "_CREDENTIALS",
    "_ACCESS_KEY",
)
_HOST_SECRET_EXACT = {
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "DOCKER_AUTH_CONFIG",
    "KUBECONFIG",
    "LD_PRELOAD",
    "PYTHONPATH",
    "PYTHONHOME",
    "BASH_ENV",
    "ENV",
}
_HOST_ENVIRONMENT_KEYWORDS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "APIKEY",
    "API_KEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
)


class PBSNode(NamedTuple):
    """One physical allocation host and its first PBS vnode index."""

    hostname: str
    vnode_index: int


def parse_pbs_nodefile_lines(lines: list[str]) -> list[PBSNode]:
    """Return ordered unique hosts with their first zero-based vnode index."""

    nodes: list[PBSNode] = []
    seen: set[str] = set()
    for vnode_index, raw_line in enumerate(lines):
        hostname = raw_line.strip()
        if not hostname or hostname in seen:
            continue
        seen.add(hostname)
        nodes.append(PBSNode(hostname=hostname, vnode_index=vnode_index))
    return nodes


def parse_pbs_nodefile(path: str | os.PathLike[str]) -> list[PBSNode]:
    with Path(path).open() as stream:
        return parse_pbs_nodefile_lines(stream.readlines())


def _private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except PermissionError:
        pass
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _canonical_discovery_identity(
    record_root: str | os.PathLike[str],
    registry_user: str,
    experiment_name: str,
    trial_name: str,
) -> dict[str, Any]:
    root = Path(record_root)
    if not root.is_absolute():
        raise ValueError(f"NFS record root must be absolute: {root}")
    if not registry_user or registry_user in {".", ".."} or Path(registry_user).name != registry_user:
        raise ValueError(f"NFS registry user must be one safe path component: {registry_user!r}")
    return {
        "version": _LOCK_METADATA_VERSION,
        "record_root": str(root.resolve(strict=False)),
        "registry_user": registry_user,
        "experiment_name": experiment_name,
        "trial_name": trial_name,
    }


def _identity_digest(identity: dict[str, Any], role: str | None = None) -> str:
    value = dict(identity)
    if role is not None:
        value["role"] = role
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _external_lock_paths(
    identity: dict[str, Any],
    role: str | None = None,
) -> tuple[Path, Path | None]:
    # Keep locks outside the experiment/trial subtree cleared by upstream
    # SlurmScheduler, but inside the resolver's per-user namespace.  A private
    # directory directly below the shared resolver root would prevent other
    # registry users from creating their own locks.
    lock_root = Path(str(identity["record_root"])) / str(identity["registry_user"]) / ".pbs-prealloc-locks"
    trial_path = lock_root / f"trial-{_identity_digest(identity)}.lock"
    role_path = None
    if role is not None:
        role_path = lock_root / f"role-{_identity_digest(identity, role)}.lock"
    return trial_path, role_path


def _external_constructor_lock_path(identity: dict[str, Any]) -> Path:
    trial_path, _role_path = _external_lock_paths(identity)
    return trial_path.with_name(f"constructor-{_identity_digest(identity)}.lock")


def _open_private_lock(path: Path):
    try:
        path.parent.mkdir(mode=0o700, parents=True)
    except FileExistsError:
        pass
    parent_stat = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != os.getuid()
        or stat.S_IMODE(parent_stat.st_mode) & 0o077
    ):
        raise PermissionError(f"Refusing unsafe PBS lifecycle lock directory: {path.parent}")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    stream = os.fdopen(descriptor, "r+")
    lock_stat = os.fstat(stream.fileno())
    if (
        not stat.S_ISREG(lock_stat.st_mode)
        or lock_stat.st_uid != os.getuid()
        or stat.S_IMODE(lock_stat.st_mode) & 0o077
    ):
        stream.close()
        raise PermissionError(f"Refusing non-private PBS lifecycle lock: {path}")
    return stream


def _acquire_lock(stream, operation: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(stream.fileno(), operation | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out acquiring PBS trial lifecycle ownership")
            time.sleep(0.1)


def _write_locked_metadata(stream, value: dict[str, Any]) -> None:
    stream.seek(0)
    stream.truncate()
    json.dump(value, stream, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())


def _constructor_discovery_identity(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    values = dict(kwargs)
    if args:
        try:
            bound = inspect.signature(SlurmScheduler.__init__).bind_partial(
                None,
                *args,
                **kwargs,
            )
        except (TypeError, ValueError):
            bound = None
        if bound is not None:
            values.update({name: value for name, value in bound.arguments.items() if name != "self"})
    exp_config = values.get("exp_config")
    if exp_config is not None:
        experiment_name = getattr(exp_config, "experiment_name", None)
        trial_name = getattr(exp_config, "trial_name", None)
        cluster = getattr(exp_config, "cluster", None)
        name_resolve = getattr(cluster, "name_resolve", None)
        record_root = getattr(name_resolve, "nfs_record_root", None)
        name_resolve_type = getattr(name_resolve, "type", None)
    else:
        experiment_name = values.get("experiment_name")
        trial_name = values.get("trial_name")
        record_root = values.get("nfs_record_root")
        name_resolve_type = values.get("name_resolve_type", "nfs")
    if str(name_resolve_type) != "nfs":
        return None
    if not experiment_name or not trial_name or not record_root:
        return None
    return _canonical_discovery_identity(
        record_root,
        getpass.getuser(),
        str(experiment_name),
        str(trial_name),
    )


class PreallocatedPBSScheduler(PreallocatedSlurmScheduler):
    """Launch AReaL roles transactionally in a preallocated PBS job."""

    def __init__(self, *args, **kwargs):
        controller_python = sys.executable
        fork_worker_python = None
        if _FORK_WORKER_PYTHON_ENV in os.environ:
            fork_worker_python = _validated_fork_worker_python(os.environ[_FORK_WORKER_PYTHON_ENV])
        trial_identity = _constructor_discovery_identity(args, kwargs)
        constructor_lock = None
        trial_lock = None
        if trial_identity is None and os.environ.get("PBS_JOBID"):
            raise RuntimeError(
                "pbs_prealloc must resolve its NFS experiment/trial identity "
                "before AReaL clears name-resolution records"
            )
        if trial_identity is not None:
            trial_lock_path, _role_lock_path = _external_lock_paths(trial_identity)
            constructor_lock_path = _external_constructor_lock_path(trial_identity)
            constructor_lock = _open_private_lock(constructor_lock_path)
            try:
                timeout = float(os.environ.get("PLATOON_AREAL_PBS_TRIAL_LOCK_TIMEOUT", "300"))
                # This mutex keeps the EX -> SH activity-lock conversion free
                # of competing constructors. flock conversion itself is not
                # guaranteed to be atomic on every supported filesystem.
                _acquire_lock(constructor_lock, fcntl.LOCK_EX, timeout)
                _write_locked_metadata(
                    constructor_lock,
                    {
                        "version": _LOCK_METADATA_VERSION,
                        "kind": "pbs-constructor-lifecycle",
                        "identity": trial_identity,
                    },
                )
                trial_lock = _open_private_lock(trial_lock_path)
                # Upstream SlurmScheduler clears the complete trial subtree in
                # its constructor. Only a new controller with exclusive trial
                # ownership may perform that clear.
                _acquire_lock(trial_lock, fcntl.LOCK_EX, timeout)
                _write_locked_metadata(
                    trial_lock,
                    {
                        "version": _LOCK_METADATA_VERSION,
                        "kind": "pbs-trial-lifecycle",
                        "identity": trial_identity,
                    },
                )
                super().__init__(*args, **kwargs)
                # The controller and all of its role launchers share ownership.
                # A replacement constructor needs LOCK_EX, so it cannot clear
                # until controller death and every launcher cleanup complete.
                fcntl.flock(trial_lock.fileno(), fcntl.LOCK_SH)
            except BaseException:
                if trial_lock is not None:
                    trial_lock.close()
                raise
            finally:
                constructor_lock.close()
        else:
            super().__init__(*args, **kwargs)
        self._pbs_trial_lifecycle_lock = trial_lock
        self._pbs_trial_lock_identity = trial_identity
        self._pbs_nodes_cache: list[PBSNode] | None = None
        self._allocation_lock = threading.RLock()
        self._node_owners: dict[str, str] = {}
        self._role_nodes: dict[str, list[PBSNode]] = {}
        self._role_manifests: dict[str, Path] = {}
        self._role_status_paths: dict[str, Path] = {}
        self._role_termination_paths: dict[str, Path] = {}
        self._role_generations: dict[str, str] = {}
        self._role_log_threads: dict[str, threading.Thread] = {}
        self._role_log_errors: dict[str, str] = {}
        self._container_image_digests: dict[tuple[str, int, int], str] = {}
        # Capture once per scheduler instance.  Forks may be concurrent, so the
        # command must not depend on later environment or sys.executable edits.
        self._fork_controller_python = controller_python
        self._fork_worker_python = fork_worker_python

    async def _fork_single_worker(
        self,
        session: Any,
        role: str,
        idx: int,
        target_wi: SlurmWorkerInfo,
        target_role: str,
        command: str | None = None,
    ) -> SlurmWorkerInfo:
        """Fork with the immutable worker-container Python, not controller Python."""

        if self._fork_worker_python is None:
            raise WorkerCreationError(
                role,
                "PBS fork worker interpreter is not configured",
                f"Set {_FORK_WORKER_PYTHON_ENV} to the immutable worker Python path",
            )
        session_with_worker_python = _ForkWorkerPythonSession(
            session,
            worker_python=self._fork_worker_python,
            controller_python=self._fork_controller_python,
        )
        return await super()._fork_single_worker(
            session_with_worker_python,
            role,
            idx,
            target_wi,
            target_role,
            command,
        )

    _fork_single_worker.__platoon_worker_python_fork_patch__ = True

    def _release_trial_lifecycle_lock(self) -> None:
        lock = getattr(self, "_pbs_trial_lifecycle_lock", None)
        if lock is not None:
            self._pbs_trial_lifecycle_lock = None
            lock.close()

    def __del__(self):
        try:
            self._release_trial_lifecycle_lock()
        except BaseException:
            pass

    @staticmethod
    def _safe_component(value: str) -> str:
        result = _SAFE_PATH_COMPONENT.sub("_", value).strip("._")
        return result or "unnamed"

    @staticmethod
    def _split_names(value: str | None) -> list[str]:
        if not value:
            return []
        return [part for part in re.split(r"[\s,]+", value) if part]

    @staticmethod
    def _split_shell_list(value: str | None) -> list[str]:
        return shlex.split(value) if value else []

    def _pbs_allocation_nodes(self) -> list[PBSNode]:
        if self._pbs_nodes_cache is not None:
            return self._pbs_nodes_cache
        nodefile = os.environ.get("PBS_NODEFILE")
        if not nodefile:
            raise RuntimeError("pbs_prealloc requires PBS_NODEFILE and must run inside a PBS allocation")
        path = Path(nodefile)
        with path.open() as stream:
            all_lines = stream.readlines()
        raw_lines = [line.strip() for line in all_lines if line.strip()]
        nodes = parse_pbs_nodefile_lines(all_lines)
        if not nodes:
            raise RuntimeError(f"PBS nodefile is empty: {path}")

        validate_slots = self._truthy_env(os.environ.get("PLATOON_AREAL_PBS_VALIDATE_NODEFILE_SLOTS", "1"))
        if validate_slots:
            counts: dict[str, int] = {}
            for hostname in raw_lines:
                counts[hostname] = counts.get(hostname, 0) + 1
            invalid = {hostname: count for hostname, count in counts.items() if count != self.n_gpus_per_node}
            if invalid:
                raise RuntimeError(
                    f"PBS nodefile does not match select=N:mpiprocs={self.n_gpus_per_node}; slot counts: {invalid}"
                )
        self._pbs_nodes_cache = nodes
        return nodes

    @staticmethod
    def _spec_signature(spec: SchedulingSpec) -> Any:
        try:
            return asdict(spec)
        except (TypeError, ValueError):
            try:
                return vars(spec)
            except TypeError:
                return repr(spec)

    def _validate_homogeneous_specs(
        self,
        role: str,
        schedulings: list[SchedulingSpec],
    ) -> SchedulingSpec:
        if not schedulings:
            raise WorkerCreationError(role, "Invalid configuration", "No worker specs")
        signature = self._spec_signature(schedulings[0])
        if any(self._spec_signature(spec) != signature for spec in schedulings[1:]):
            raise WorkerCreationError(
                role,
                "Unsupported heterogeneous scheduling",
                "pbs_prealloc requires every replica in a role to use one identical spec",
            )
        return schedulings[0]

    def _role_topology(self, role: str, replicas: int, spec: SchedulingSpec) -> tuple[int, int]:
        try:
            gpus_per_worker = int(spec.gpu)
        except (TypeError, ValueError) as error:
            raise WorkerCreationError(role, "Invalid GPU topology", f"gpu must be an integer: {spec.gpu!r}") from error
        if gpus_per_worker <= 0 or gpus_per_worker > self.n_gpus_per_node:
            raise WorkerCreationError(
                role,
                "Invalid GPU topology",
                f"gpu per worker must be in [1, {self.n_gpus_per_node}]",
            )
        if self.n_gpus_per_node % gpus_per_worker:
            raise WorkerCreationError(
                role,
                "Non-divisible GPU topology",
                f"{gpus_per_worker} GPUs per worker does not divide {self.n_gpus_per_node} GPUs per node",
            )
        workers_per_node = self.n_gpus_per_node // gpus_per_worker
        if replicas % workers_per_node:
            raise WorkerCreationError(
                role,
                "Partial-node GPU topology",
                f"{replicas} replicas x {gpus_per_worker} GPUs does not fill whole {self.n_gpus_per_node}-GPU nodes",
            )
        return replicas // workers_per_node, workers_per_node

    def _reserve_nodes(
        self,
        role: str,
        count: int,
        nodelist: str | None,
        exclude: str | None,
    ) -> list[PBSNode]:
        with self._allocation_lock:
            if role in self._role_nodes:
                raise WorkerCreationError(role, f"Role '{role}' already has PBS nodes reserved")
            allocation = self._pbs_allocation_nodes()
            by_name = {node.hostname: node for node in allocation}
            excluded = set(self._split_names(exclude))
            requested_names = self._split_names(nodelist)
            if requested_names:
                if len(requested_names) != count or len(set(requested_names)) != count:
                    raise WorkerCreationError(
                        role,
                        "Invalid nodelist",
                        f"Expected exactly {count} distinct PBS hosts",
                    )
                unknown = [name for name in requested_names if name not in by_name]
                if unknown:
                    raise WorkerCreationError(role, "Invalid nodelist", f"Hosts not in this allocation: {unknown}")
                candidates = [by_name[name] for name in requested_names]
            else:
                candidates = [
                    node
                    for node in allocation
                    if node.hostname not in excluded and node.hostname not in self._node_owners
                ][:count]

            unavailable = [
                node.hostname for node in candidates if node.hostname in excluded or node.hostname in self._node_owners
            ]
            if unavailable:
                raise WorkerCreationError(
                    role,
                    "PBS hosts already reserved or excluded",
                    f"Unavailable hosts: {unavailable}",
                )
            if len(candidates) != count:
                free_count = sum(
                    node.hostname not in excluded and node.hostname not in self._node_owners for node in allocation
                )
                raise WorkerCreationError(
                    role,
                    "Insufficient PBS allocation",
                    f"Role needs {count} whole nodes but only {free_count} are free",
                )

            for node in candidates:
                self._node_owners[node.hostname] = role
            self._role_nodes[role] = list(candidates)
            logger.info(
                "Reserved PBS node(s) for role '%s': %s",
                role,
                ",".join(node.hostname for node in candidates),
            )
            return list(candidates)

    def _release_nodes(self, role: str) -> None:
        with self._allocation_lock:
            nodes = self._role_nodes.pop(role, [])
            for node in nodes:
                if self._node_owners.get(node.hostname) == role:
                    del self._node_owners[node.hostname]

    def _role_state_dir(self) -> Path:
        configured = os.environ.get("PLATOON_AREAL_PBS_STATE_DIR")
        if configured:
            root = Path(configured)
        elif self.fileroot:
            root = Path(str(self.fileroot)) / ".pbs-prealloc"
        else:
            raise RuntimeError("pbs_prealloc needs a shared fileroot or PLATOON_AREAL_PBS_STATE_DIR")
        job_id = self._safe_component(os.environ.get("PBS_JOBID", "unknown-job"))
        trial = self._safe_component(str(self.trial_name))
        state_dir = root / job_id / trial
        state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(state_dir, 0o700)
        except PermissionError:
            pass
        return state_dir

    def _worker_environment(self, role: str, spec: SchedulingSpec) -> dict[str, str]:
        result: dict[str, str] = {}
        for name, value in (spec.env_vars or {}).items():
            if not _ENVIRONMENT_NAME.fullmatch(name):
                raise WorkerCreationError(
                    role,
                    "Invalid worker environment",
                    f"Invalid variable name {name!r}",
                )
            result[name] = str(value)
        for name in self._split_names(os.environ.get("PLATOON_AREAL_PBS_FORWARD_ENV")):
            if not _ENVIRONMENT_NAME.fullmatch(name):
                raise ValueError(f"Invalid forwarded environment variable name: {name!r}")
            if name in os.environ:
                result[name] = os.environ[name]
        return result

    @staticmethod
    def _unsafe_host_environment_name(name: str) -> bool:
        upper = name.upper()
        return (
            upper in _HOST_SECRET_EXACT
            or upper.startswith(_HOST_SECRET_PREFIXES)
            or upper.endswith(_HOST_SECRET_SUFFIXES)
            or any(keyword in upper for keyword in _HOST_ENVIRONMENT_KEYWORDS)
            or upper.startswith(("APPTAINERENV_", "SINGULARITYENV_"))
        )

    def _host_launcher_environment(self, role: str) -> dict[str, str]:
        """Return the small non-secret environment needed by PBS/HPC-X.

        Explicit worker credentials already live in the mode-0600 manifest.
        They must not also be inherited by the controller-side launcher or
        intermediate pbsdsh supervisors.
        """

        result = {
            name: value
            for name, value in os.environ.items()
            if not self._unsafe_host_environment_name(name)
            and (name in _HOST_ENV_EXACT or name.startswith(_HOST_ENV_PREFIXES))
        }
        for name in self._split_names(os.environ.get("PLATOON_AREAL_PBS_LAUNCHER_FORWARD_ENV")):
            if not _ENVIRONMENT_NAME.fullmatch(name):
                raise WorkerCreationError(
                    role,
                    "Invalid host launcher environment",
                    f"Invalid variable name {name!r}",
                )
            if self._unsafe_host_environment_name(name):
                raise WorkerCreationError(
                    role,
                    "Unsafe host launcher environment",
                    f"Refusing to expose credential/runtime-control variable {name!r}",
                )
            if name in os.environ:
                result[name] = os.environ[name]
        return result

    def _worker_pbs_preamble(self) -> list[str]:
        preamble = os.environ.get("PLATOON_AREAL_PBS_WORKER_PREAMBLE")
        return [preamble] if preamble else []

    def _gpu_visibility(self, role: str) -> dict[str, str]:
        """Return the explicit host/container GPU mapping contract.

        ``legacy`` preserves the original numeric CUDA masking used on sites
        where Singularity binds the complete NVIDIA device tree.  ABCI opts in
        to ``nvidia-container-cli`` because its site configuration requires a
        host-side ``NVIDIA_VISIBLE_DEVICES`` value when ``--contain`` is used.
        The supervisor resolves numeric PBS assignments to physical UUIDs on
        each compute host; the controller must not assume identical UUIDs.
        """

        mode = os.environ.get("PLATOON_AREAL_PBS_GPU_VISIBILITY_MODE", "legacy").strip()
        if mode not in _GPU_VISIBILITY_MODES:
            raise WorkerCreationError(
                role,
                "Invalid GPU visibility mode",
                f"Expected one of {sorted(_GPU_VISIBILITY_MODES)}, got {mode!r}",
            )
        nvidia_smi_binary = os.environ.get(
            "PLATOON_AREAL_PBS_NVIDIA_SMI_BIN",
            _DEFAULT_NVIDIA_SMI_BINARY,
        )
        if mode == "nvidia-container-cli" and not Path(nvidia_smi_binary).is_absolute():
            raise WorkerCreationError(
                role,
                "Unsafe NVIDIA discovery binary",
                "PLATOON_AREAL_PBS_NVIDIA_SMI_BIN must be an absolute path",
            )
        return {
            "mode": mode,
            "nvidia_smi_binary": nvidia_smi_binary,
        }

    def _rpc_worker_script(self, role: str, spec: SchedulingSpec) -> str:
        rpc_command = spec.cmd or "python -m areal.infra.rpc.rpc_server"
        if role in {"rollout", "eval-rollout"} and rpc_command.strip() == "python -m areal.infra.rpc.rpc_server":
            # The lightweight Platoon entrypoint installs inference-worker
            # patches before delegating to AReaL's parser/server unchanged.
            # Actor workers keep the upstream entrypoint and explicit custom
            # commands are never rewritten.
            rpc_command = "python -m platoon.areal_rpc_server"
        flags: list[str] = [
            "--experiment-name",
            str(self.experiment_name),
            "--trial-name",
            str(self.trial_name),
            "--role",
            role,
            "--name-resolve-type",
            str(self.name_resolve_config.type),
            "--nfs-record-root",
            str(self.name_resolve_config.nfs_record_root),
            "--etcd3-addr",
            str(self.name_resolve_config.etcd3_addr),
        ]
        if self.fileroot:
            flags.extend(["--fileroot", str(self.fileroot)])
        quoted_flags = " ".join(shlex.quote(value) for value in flags)
        rpc_command = (
            f'{rpc_command} {quoted_flags} --worker-index "${{PLATOON_WORKER_INDEX:?PLATOON_WORKER_INDEX is unset}}"'
        )
        commands = list(spec.additional_bash_cmds or [])
        commands.extend(self._worker_pbs_preamble())
        commands.append(
            "printf 'Platoon PBS worker role=%s host=%s worker_index=%s "
            "local_rank=%s CUDA_VISIBLE_DEVICES=%s assigned_gpu_ids=%s "
            "assigned_gpu_uuids=%s container=%s digest=%s\\n' "
            f'{shlex.quote(role)} "$(hostname)" "${{PLATOON_WORKER_INDEX}}" '
            '"${PLATOON_LOCAL_RANK}" "${CUDA_VISIBLE_DEVICES}" '
            '"${PLATOON_ASSIGNED_GPU_IDS}" "${PLATOON_ASSIGNED_GPU_UUIDS:-}" '
            '"${PLATOON_CONTAINER_IMAGE}" "${PLATOON_CONTAINER_IMAGE_DIGEST}" >&2'
        )
        commands.append(f"exec {rpc_command}")
        return ";\n".join(command.strip() for command in commands if command.strip())

    def _worker_command(self, role: str, spec: SchedulingSpec) -> list[str]:
        script = self._rpc_worker_script(role, spec)
        runtime = os.environ.get("PLATOON_AREAL_PBS_CONTAINER_RUNTIME", "singularity")
        if runtime.lower() in {"", "none", "host"}:
            return ["/bin/bash", "-c", script]

        image = os.environ.get("PLATOON_AREAL_PBS_CONTAINER_IMAGE") or os.environ.get("PLATOON_TRAINING_SIF")
        if not image:
            raise WorkerCreationError(
                role,
                "Missing training container",
                "Set PLATOON_AREAL_PBS_CONTAINER_IMAGE or PLATOON_TRAINING_SIF",
            )
        command = [
            runtime,
            "exec",
            "--nv",
            "--cleanenv",
            "--no-home",
            "--no-mount",
            "home,cwd,bind-paths",
        ]
        for bind in self._split_shell_list(os.environ.get("PLATOON_AREAL_PBS_CONTAINER_BINDS")):
            command.extend(["--bind", bind])
        workdir = os.environ.get("PLATOON_AREAL_PBS_CONTAINER_WORKDIR")
        if workdir:
            command.extend(["--pwd", workdir])
        command.extend(self._split_shell_list(os.environ.get("PLATOON_AREAL_PBS_CONTAINER_ARGS")))
        command.extend([image, "/bin/bash", "-c", script])
        return command

    def _container_image_digest(self, role: str) -> str:
        runtime = os.environ.get("PLATOON_AREAL_PBS_CONTAINER_RUNTIME", "singularity")
        if runtime.lower() in {"", "none", "host"}:
            return "host-runtime"
        image_value = os.environ.get("PLATOON_AREAL_PBS_CONTAINER_IMAGE") or os.environ.get("PLATOON_TRAINING_SIF")
        if not image_value:
            raise WorkerCreationError(role, "Missing training container")
        image = Path(image_value)
        try:
            image_stat = image.stat()
        except OSError as error:
            raise WorkerCreationError(
                role,
                "Training container is not readable",
                f"Could not stat {image}: {error}",
            ) from error
        if not image.is_file():
            raise WorkerCreationError(role, "Training container is not a file", str(image))
        cache_key = (str(image.resolve()), image_stat.st_size, image_stat.st_mtime_ns)
        digest = self._container_image_digests.get(cache_key)
        if digest is None:
            hasher = hashlib.sha256()
            with image.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            self._container_image_digests[cache_key] = digest
        return f"sha256:{digest}"

    def _role_manifest(
        self,
        role: str,
        replicas: int,
        spec: SchedulingSpec,
        nodes: list[PBSNode],
        workers_per_node: int,
    ) -> tuple[Path, dict[int, str], Path, Path, str]:
        state_dir = self._role_state_dir()
        role_component = self._safe_component(role)
        generation = uuid4().hex
        manifest_path = state_dir / f"{role_component}.{generation}.manifest.json"
        status_path = state_dir / f"{role_component}.{generation}.launcher-status.json"
        termination_path = state_dir / f"{role_component}.{generation}.termination-request.json"

        if str(self.name_resolve_config.type) != "nfs":
            raise WorkerCreationError(
                role,
                "Unsupported name resolution",
                "pbs_prealloc requires NFS name resolution for crash-safe discovery cleanup",
            )
        record_root = Path(str(self.name_resolve_config.nfs_record_root))
        if not record_root.is_absolute():
            raise WorkerCreationError(
                role,
                "Unsafe name-resolution root",
                f"NFS record root must be absolute: {record_root}",
            )
        registry_user = getpass.getuser()
        discovery_components = [
            registry_user,
            str(self.experiment_name),
            str(self.trial_name),
            role,
        ]
        invalid_components = [value for value in discovery_components if self._safe_component(value) != value]
        if invalid_components:
            raise WorkerCreationError(
                role,
                "Unsafe name-resolution path",
                f"Discovery path components must be simple names: {invalid_components!r}",
            )
        discovery_role_root = (
            record_root / registry_user / str(self.experiment_name) / str(self.trial_name) / "workers" / role
        )
        lock_identity = _canonical_discovery_identity(
            record_root,
            registry_user,
            str(self.experiment_name),
            str(self.trial_name),
        )
        if self._pbs_trial_lock_identity != lock_identity:
            raise WorkerCreationError(
                role,
                "Name-resolution identity changed after scheduler construction",
                "Refusing a worker launch that is not protected by the controller's trial lock",
            )
        trial_lock_path, lifecycle_lock_path = _external_lock_paths(
            lock_identity,
            role,
        )
        constructor_lock_path = _external_constructor_lock_path(lock_identity)
        assert lifecycle_lock_path is not None
        discovery_entries = [str(discovery_role_root / str(worker_index) / "ENTRY") for worker_index in range(replicas)]
        gpus_per_worker = int(spec.gpu)
        worker_hosts: dict[int, str] = {}
        hosts: list[dict[str, Any]] = []
        next_worker = 0
        for node in nodes:
            assignments = []
            for local_rank in range(workers_per_node):
                worker_index = next_worker
                next_worker += 1
                gpu_start = local_rank * gpus_per_worker
                gpu_ids = list(range(gpu_start, gpu_start + gpus_per_worker))
                assignments.append(
                    {
                        "worker_index": worker_index,
                        "local_rank": local_rank,
                        "gpu_ids": gpu_ids,
                    }
                )
                worker_hosts[worker_index] = node.hostname
            host_component = self._safe_component(node.hostname)
            hosts.append(
                {
                    "hostname": node.hostname,
                    "vnode_index": node.vnode_index,
                    "control_path": str(
                        state_dir / (f"{role_component}.{generation}.{host_component}.{node.vnode_index}.control.json")
                    ),
                    "workers": assignments,
                }
            )
        if next_worker != replicas:
            raise AssertionError(f"Internal PBS topology error: {next_worker} != {replicas}")

        manifest: dict[str, Any] = {
            "version": _MANIFEST_VERSION,
            "generation": generation,
            "role": role,
            "replicas": replicas,
            "pbs_job_id": os.environ.get("PBS_JOBID", ""),
            "pbsdsh_binary": os.environ.get("PLATOON_AREAL_PBS_PBSDSH_BIN", "pbsdsh"),
            "python_binary": os.environ.get("PLATOON_AREAL_PBS_HOST_PYTHON", sys.executable),
            "supervisor_script": str(Path(__file__).with_name("pbs_role_supervisor.py")),
            "launcher_status_path": str(status_path),
            "termination_request_path": str(termination_path),
            "discovery": {
                "type": "nfs",
                "record_root": str(record_root),
                "role_root": str(discovery_role_root),
                "entry_paths": discovery_entries,
                "lock_identity": lock_identity,
                "constructor_lifecycle_lock_path": str(constructor_lock_path),
                "trial_lifecycle_lock_path": str(trial_lock_path),
                "lifecycle_lock_path": str(lifecycle_lock_path),
            },
            "container_image": os.environ.get("PLATOON_AREAL_PBS_CONTAINER_IMAGE")
            or os.environ.get("PLATOON_TRAINING_SIF", ""),
            "container_image_digest": self._container_image_digest(role),
            "termination_grace_seconds": float(os.environ.get("PLATOON_AREAL_PBS_TERMINATION_GRACE_SECONDS", "10")),
            "termination_reverse_order_default": True,
            "gpu_visibility": self._gpu_visibility(role),
            "worker_command": self._worker_command(role, spec),
            "worker_environment": self._worker_environment(role, spec),
            "hosts": hosts,
        }
        try:
            _private_json(manifest_path, manifest)
        except Exception:
            manifest_path.unlink(missing_ok=True)
            raise
        return manifest_path, worker_hosts, status_path, termination_path, generation

    def _build_role_launcher_command(self, role: str, manifest_path: Path) -> list[str]:
        del role
        return [
            os.environ.get("PLATOON_AREAL_PBS_HOST_PYTHON", sys.executable),
            "-u",
            str(Path(__file__).with_name("pbs_role_launcher.py")),
            "--manifest",
            str(manifest_path),
            "--parent-pid",
            str(os.getpid()),
        ]

    def _stream_role_logs(
        self,
        role: str,
        process: subprocess.Popen[str],
        role_log,
        merged_log,
    ) -> None:
        prefix = f"[{role}]".ljust(10)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except (AttributeError, OSError, ValueError):
                    pass
                role_log.write(line)
                role_log.flush()
                merged_log.write(f"{prefix}{line}")
                merged_log.flush()
        except BaseException as error:
            self._role_log_errors[role] = f"{type(error).__name__}: {error}"
            logger.error("PBS log streaming failed for role '%s': %s", role, error)
            # Continue draining when the failure came from a destination file;
            # otherwise the actual launcher could block on its stdout pipe and
            # never reach remote cleanup.
            if process.stdout is not None:
                try:
                    for _line in process.stdout:
                        pass
                except (OSError, ValueError):
                    pass
        finally:
            if process.stdout is not None:
                process.stdout.close()
            role_log.close()
            merged_log.close()

    def _launch_role_process(self, role: str, command: list[str]) -> subprocess.Popen[str]:
        logger.info("Launching role '%s' inside preallocated PBS job", role)
        logger.info("Preallocated PBS launcher command for role '%s': %s", role, shlex.join(command))
        role_log_path = Path(self._log_path_of(role))
        merged_log_path = Path(self._merged_log_path())
        role_log_path.parent.mkdir(parents=True, exist_ok=True)
        merged_log_path.parent.mkdir(parents=True, exist_ok=True)
        role_log = role_log_path.open("a", encoding="utf-8", errors="replace", buffering=1)
        try:
            merged_log = merged_log_path.open("a", encoding="utf-8", errors="replace", buffering=1)
        except BaseException:
            role_log.close()
            raise
        try:
            # This Popen is the launcher itself, not a shell/tee pipeline.  Its
            # exit therefore means its finally block and redundant pbsdsh
            # cleanup have both completed.
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
                env=self._host_launcher_environment(role),
            )
        except BaseException:
            role_log.close()
            merged_log.close()
            raise
        thread = threading.Thread(
            target=self._stream_role_logs,
            args=(role, process, role_log, merged_log),
            name=f"pbs-{self._safe_component(role)}-logs",
            daemon=True,
        )
        # Publish launcher ownership before starting the auxiliary log thread.
        # If Thread.start itself fails, create_workers' rollback can still
        # signal the real launcher and require its remote-cleanup status.
        self._role_processes[role] = process
        self._role_log_threads[role] = thread
        try:
            thread.start()
        except BaseException:
            self._role_log_threads.pop(role, None)
            if process.stdout is not None:
                process.stdout.close()
            role_log.close()
            merged_log.close()
            raise
        return process

    def _join_role_log_thread(self, role: str) -> None:
        thread = self._role_log_threads.get(role)
        if thread is None:
            return
        thread.join(timeout=10)
        if thread.is_alive():
            message = "log streaming thread did not finish after launcher exit"
            self._role_log_errors[role] = message
            logger.error("PBS %s for role '%s'", message, role)
            return
        self._role_log_threads.pop(role, None)

    def _confirmed_launcher_status(self, role: str) -> dict[str, Any]:
        status_path = self._role_status_paths.get(role)
        if status_path is None:
            raise RuntimeError(f"Role '{role}' has no PBS launcher status path")
        try:
            file_stat = status_path.stat()
            if file_stat.st_uid != os.getuid() or file_stat.st_mode & 0o077:
                raise PermissionError(f"Refusing non-private PBS launcher status: {status_path}")
            with status_path.open() as stream:
                status = json.load(stream)
        except FileNotFoundError as error:
            raise RuntimeError(
                f"PBS launcher for role '{role}' exited without confirming remote cleanup; "
                f"nodes remain reserved ({status_path})"
            ) from error
        if not isinstance(status, dict) or status.get("version") != _MANIFEST_VERSION or not status.get("complete"):
            raise RuntimeError(f"Invalid or incomplete PBS launcher status for role '{role}': {status_path}")
        failures = status.get("cleanup_failures")
        if not isinstance(failures, list):
            raise RuntimeError(f"PBS launcher status has invalid cleanup_failures for role '{role}'")
        if failures:
            raise RuntimeError(
                f"PBS remote cleanup failed for role '{role}'; nodes remain reserved: "
                f"{json.dumps(failures, sort_keys=True)}"
            )
        expected_generation = self._role_generations.get(role)
        if not expected_generation or status.get("generation") != expected_generation:
            raise RuntimeError(
                f"PBS launcher cleanup status generation does not match role '{role}'; nodes remain reserved"
            )
        return status

    def _check_job_status(self, role: str) -> None:
        if role in self._colocated_roles:
            return self._check_job_status(self._colocated_roles[role])
        process = self._role_processes.get(role)
        if process is None:
            if role in self._workers:
                return
            raise WorkerNotFoundError(f"Role '{role}' not found")
        status = process.poll()
        if status is None or role in self._stopping_roles:
            return
        self._join_role_log_thread(role)
        raise WorkerFailedError(
            f"{role}/*",
            process.pid,
            f"PBS role launcher exited with status {status}. Logs:\n{self._read_log_tail(role)}",
        )

    def _terminate_role_process(self, role: str, *, reverse_order: bool = False) -> None:
        process = self._role_processes.get(role)
        if process is None:
            return
        self._stopping_roles.add(role)
        try:
            if process.poll() is None:
                termination_path = self._role_termination_paths.get(role)
                if termination_path is None:
                    raise RuntimeError(f"Role '{role}' has no PBS termination request path")
                _private_json(
                    termination_path,
                    {
                        "version": _MANIFEST_VERSION,
                        "role": role,
                        "generation": self._role_generations.get(role),
                        "reverse_order": bool(reverse_order),
                    },
                )
                # Signal only the launcher. It owns the pbsdsh connections and
                # must be allowed to run its orderly and redundant cleanup.
                process.terminate()
                timeout = float(os.environ.get("PLATOON_AREAL_PBS_LAUNCHER_STOP_TIMEOUT", "60"))
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    logger.warning("Force killing preallocated PBS role '%s'", role)
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            self._join_role_log_thread(role)
            self._confirmed_launcher_status(role)
            self._role_processes.pop(role, None)
        except ProcessLookupError:
            self._join_role_log_thread(role)
            self._confirmed_launcher_status(role)
            self._role_processes.pop(role, None)
        finally:
            self._stopping_roles.discard(role)

    def create_workers(self, job: Job, *args, **kwargs) -> list[str]:
        role = job.role
        if ":" in role or self._safe_component(role) != role:
            raise ValueError("Invalid worker name.")
        replicas = job.replicas
        if role in self._workers or role in self._colocated_roles:
            raise WorkerCreationError(role, f"Role '{role}' already exists")
        if replicas <= 0:
            raise WorkerCreationError(role, "Invalid configuration", "replicas must be greater than 0")

        schedulings = self._prepare_worker_specs(role, replicas, job.tasks)
        spec = self._validate_homogeneous_specs(role, schedulings)
        strategy = job.scheduling_strategy
        strategy_type = SchedulingStrategyType(strategy.type)
        colocate_role = strategy.target

        if strategy_type == SchedulingStrategyType.colocation:
            if not colocate_role:
                raise WorkerCreationError(
                    role,
                    "Invalid strategy",
                    "Colocation strategy requires target role to be specified",
                )
            if colocate_role not in self._workers:
                raise WorkerNotFoundError(f"Cannot colocate with role '{colocate_role}' - role not found")
            target_workers = self._workers[colocate_role]
            if replicas != len(target_workers):
                raise WorkerCreationError(
                    role,
                    "Replica count mismatch",
                    f"Colocated role must have same replica count as target ({replicas} != {len(target_workers)})",
                )
            if strategy.fork:
                return self.fork_workers(role, colocate_role)
            self._colocated_roles[role] = colocate_role
            return [worker.worker.id for worker in target_workers]

        if strategy_type != SchedulingStrategyType.separation:
            raise ValueError(f"Unknown scheduling strategy type: {strategy_type}")

        node_count, workers_per_node = self._role_topology(role, replicas, spec)
        reserved = self._reserve_nodes(
            role,
            node_count,
            getattr(spec, "nodelist", None),
            getattr(spec, "exclude", None),
        )
        manifest_path: Path | None = None
        try:
            (
                manifest_path,
                worker_hosts,
                status_path,
                termination_path,
                generation,
            ) = self._role_manifest(role, replicas, spec, reserved, workers_per_node)
            command = self._build_role_launcher_command(role, manifest_path)
            self._role_status_paths[role] = status_path
            self._role_termination_paths[role] = termination_path
            self._role_generations[role] = generation
            process = self._launch_role_process(role, command)
            self._role_processes[role] = process
            self._role_commands[role] = shlex.join(command)
            self._role_manifests[role] = manifest_path

            workers: list[SlurmWorkerInfo] = []
            worker_ids: list[str] = []
            for worker_index in range(replicas):
                worker_id = f"{role}/{worker_index}"
                worker = Worker(id=worker_id, ip="", worker_ports=[], engine_ports=[])
                worker_info = SlurmWorkerInfo(
                    worker=worker,
                    role=role,
                    slurm_job_id=process.pid,
                    task_index=worker_index,
                    discovered=False,
                    spec=spec,
                )
                # Current AReaL's SlurmWorkerInfo exposes node; retain compatibility
                # with older revisions where discovery only populated worker.ip.
                try:
                    worker_info.node = worker_hosts[worker_index]
                except (AttributeError, TypeError):
                    pass
                workers.append(worker_info)
                worker_ids.append(worker_id)
            self._workers[role] = workers
            self._jobs[role] = process.pid
            return worker_ids
        except Exception as creation_error:
            try:
                self._terminate_role_process(role)
            except Exception as cleanup_error:
                # Preserve every ownership/control record. Releasing these
                # nodes, or allowing this role name to launch again, would let
                # a new worker overlap a remotely live process.
                self._workers.setdefault(role, [])
                logger.error(
                    "PBS role '%s' creation failed and cleanup was not confirmed: %s",
                    role,
                    cleanup_error,
                )
                raise cleanup_error from creation_error
            self._workers.pop(role, None)
            self._jobs.pop(role, None)
            self._role_commands.pop(role, None)
            self._role_manifests.pop(role, None)
            self._role_status_paths.pop(role, None)
            self._role_termination_paths.pop(role, None)
            self._role_generations.pop(role, None)
            self._role_log_errors.pop(role, None)
            if manifest_path is not None:
                manifest_path.unlink(missing_ok=True)
            self._release_nodes(role)
            raise

    def delete_workers(
        self,
        role: str | None = None,
        reverse_order: bool = False,
    ) -> None:
        if role is None:
            aliases = list(self._colocated_roles.keys())
            owners = [name for name in self._workers if name not in self._colocated_roles]
            if reverse_order:
                aliases.reverse()
                owners.reverse()
            for name in aliases:
                self.delete_workers(name, reverse_order=reverse_order)
            for name in owners:
                self.delete_workers(name, reverse_order=reverse_order)
            return

        cleanup_dependent_aliases_before_owner(
            self,
            role,
            reverse_order=reverse_order,
        )
        if role in self._colocated_roles:
            cleanup_forked_alias_workers(
                self,
                role,
                reverse_order=reverse_order,
            )
            self._workers.pop(role, None)
            self._configured_worker_generations.pop(role, None)
            del self._colocated_roles[role]
            return
        if role not in self._workers:
            logger.warning("Role '%s' not found, skipping deletion", role)
            return

        logger.info("Deleting preallocated PBS workers for role '%s'", role)
        if reverse_order:
            self._terminate_role_process(role, reverse_order=True)
        else:
            self._terminate_role_process(role)
        self._workers.pop(role, None)
        self._configured_worker_generations.pop(role, None)
        self._jobs.pop(role, None)
        self._role_commands.pop(role, None)
        manifest_path = self._role_manifests.pop(role, None)
        if manifest_path is not None:
            manifest_path.unlink(missing_ok=True)
        self._role_status_paths.pop(role, None)
        termination_path = self._role_termination_paths.pop(role, None)
        if termination_path is not None:
            termination_path.unlink(missing_ok=True)
        self._role_generations.pop(role, None)
        self._role_log_errors.pop(role, None)
        self._release_nodes(role)

    @classmethod
    def from_scheduler_config(cls, config):
        return cls(exp_config=config)
