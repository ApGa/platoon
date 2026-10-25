"""The trainer<->rollout-worker contract.

Transport is **pickle protocol 5 through node-local files**, not a pipe:

* a multi-MB response on a 64 KB pipe deadlocks a child that exits before the
  parent drains it;
* a worker needs ~1.9 s of interpreter teardown *after* it writes its response,
  so the parent must be able to read a result that arrived before a kill --
  files make that trivially true and are what makes classification
  response-first (see :mod:`platoon.train.tinker.workers.supervisor`);
* ``tinker.Datum``/``TensorData`` are binary, so JSON would be both larger and
  lossy on floats.

Files live under ``$TMPDIR/platoon-rollout-ipc``, which must be node-local
storage (a job's local scratch), never a shared filesystem: response files are
1-6 MB each, and a shared filesystem degrades badly.  The **request file carries
the gateway API key** inside the pickled ``SamplingClient``'s constructor
kwargs, so it is written 0600, never logged, and unlinked by the child the
moment it has been loaded.
"""

from __future__ import annotations

import errno
import functools
import importlib
import logging
import os
import pickle
import stat
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from platoon.task import Task
from platoon.train.tinker.rollout_metrics import RolloutTimingSummary
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.scratch import node_scratch
from platoon.utils.subagent_sampling import DeterministicSubagentDatumSampler
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 2

#: The directory in node-local scratch that holds request/response/ready/diag files, one per user:
#: ``platoon-rollout-ipc-<uid>`` (:func:`ipc_dir`).
IPC_DIRNAME = "platoon-rollout-ipc"

#: Files older than this are swept at supervisor construction.
IPC_SWEEP_AGE_SECONDS = 24 * 60 * 60

#: The diagnostic marker a dying worker leaves behind is written with a single
#: ``os.write`` from a signal handler, so it is pre-formatted and bounded.
DIAG_MAX_BYTES = 4096

#: Exit code the RSS watchdog uses, so an over-allocating rollout is classified
#: deterministically instead of being inferred from an anonymous ``rc=-9``.
EXIT_RSS_LIMIT = 87

#: Exit code used when the alarm handler cannot signal its own group.
EXIT_HARD_TIMEOUT = 88

WorkerStatus = Literal["completed", "no_trajectories", "conversion_empty", "error"]


class ProtocolVersionError(ValueError):
    """A request or response was written by an incompatible build."""


class RefResolutionError(ValueError):
    """A ``module:qualname`` reference does not round-trip.

    Raised at workflow construction, never at dispatch: a closure or a
    ``functools.partial`` cannot cross a process boundary by name, and
    silently substituting a default reward processor would change every
    advantage in the run with no error and plausible-looking metrics.
    """


# --------------------------------------------------------------------------
# Payloads
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelBinding:
    """Everything a worker needs to reach the gateway without a ServiceClient.

    Workers **follow** the trainer's sampling client, they do not pin it:
    ``binding_path`` is re-``stat``ed at most once per ``follow_poll_seconds``
    and reloaded on mtime change.  Pinning would be a no-op under a full
    fine-tuning gateway (which always serves the latest pushed weights) and a
    hard mid-rollout failure under a LoRA gateway, which retires
    sampler versions after ``sampler_versions_to_keep`` optim steps.

    ``renderer_name`` + ``renderer_kwargs`` cross instead of a pickled
    ``Renderer``: ``Renderer.__reduce__`` drops ``renderer_kwargs``, which
    ``register_tinker_llm`` applies by ``setattr``, so shipping the object
    would silently drop a non-default ``preserve_thinking``.
    """

    binding_path: str
    bootstrap_blob: bytes  # pickle.dumps(sampling_client); b"" means "no model" (tests)
    follow_poll_seconds: float
    model_name: str  # bare HF id
    renderer_name: str
    renderer_kwargs: dict[str, Any] = field(default_factory=dict)
    context_window_length: int | None = None
    # The trainer's bound on one sampling call, as its gateway resolved it (``train.sample_timeout_seconds``, else the
    # trainer's ``TINKER_SAMPLE_TIMEOUT_SECONDS``), so a worker samples under the trainer's bound whatever its own
    # environment holds.  None leaves it to the worker's gateway.
    sample_timeout_seconds: float | None = None
    max_tokens: int = 4096
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    seed: int | None = None
    checkpoint_version: int = 0


@dataclass(frozen=True)
class ConversionPolicy:
    """Datum-conversion policy, carried explicitly so it cannot drift.

    ``reward_processor_is_default`` is an explicit boolean and is never
    inferred from ``reward_processor_ref is None``: see
    :class:`RefResolutionError`.
    """

    filter_errors: bool = False
    filter_message_actions: bool = False
    include_traj_depth: bool = False
    include_traj_start: bool = False
    reward_processor_is_default: bool = True
    reward_processor_ref: str | None = None
    subagent_datum_sampler: DeterministicSubagentDatumSampler | None = None


@dataclass(frozen=True)
class RolloutWorkerRequest:
    """One rollout, fully specified.  Built by the parent, read by the child."""

    protocol_version: int
    request_id: str
    task_id: str
    rollout_number: int
    task: Task
    rollout_config: dict[str, Any]  # asdict(RolloutConfig); output_dir already versioned
    rollout_fn_ref: str
    model: ModelBinding
    conversion: ConversionPolicy
    response_path: str
    ready_path: str
    startup_budget_seconds: float = 300.0
    rollout_budget_seconds: float = 3720.0
    rss_soft_limit_bytes: int = 24 * 1024**3
    python_env_isolation: bool = False
    # The environments' ``package`` modules: they register the harnesses, programs and loaders the rollout looks
    # up by name, so the worker imports them before it starts, as the trainer did
    # (``platoon.train.auto.import_environment_packages``).
    packages: tuple[str, ...] = ()
    # The labels the trainer puts on its sandboxes (its run's ``platoon.run``), which the worker puts on its own
    # beside its ``platoon.worker`` label (:mod:`platoon.sandbox.cleanup`).
    sandbox_labels: dict[str, str] = field(default_factory=dict)
    # Whether the worker marks its rollout's first model call (``first_call_path``), for the trainer's start limit
    # (platoon.train.tinker.start_limit).
    signal_first_call: bool = False

    @property
    def diag_path(self) -> str:
        return self.response_path + ".diag"

    @property
    def first_call_path(self) -> str:
        """Where the worker marks its rollout's first model call (``signal_first_call``), next to its readiness file."""

        return self.response_path + ".first_call"

    @property
    def sessions_path(self) -> str:
        """Where the worker lists the remote env sessions it holds (0600: it carries their API keys), which the
        supervisor deletes when the worker could not (:mod:`platoon.envs.ors.cleanup`)."""

        return self.response_path + ".sessions"

    @property
    def sandboxes_path(self) -> str:
        """Where the worker lists how to reap the sandboxes it holds (their providers and its labels, ``platoon.worker``
        among them), which the supervisor reaps once the worker is gone (:mod:`platoon.sandbox.cleanup`)."""

        return self.response_path + ".sandboxes"


@dataclass(frozen=True)
class RolloutWorkerResponse:
    """One rollout's outcome.  Everything after ``observed`` is diagnostic, but for ``reward_key``, which decides
    whether the rollout's group trains."""

    protocol_version: int
    request_id: str
    status: WorkerStatus
    result: TrajectoryCollectionResult | None
    workload: RolloutWorkload  # funnel already applied when result is not None
    observed: bool
    # ---- diagnostics (never load-bearing for training) ----
    error_type: str | None = None
    error_repr: str | None = None
    startup_seconds: float = 0.0
    rollout_seconds: float = 0.0
    conversion_seconds: float = 0.0
    model_calls: int = 0
    routing_id_present: int = 0  # interactions whose routing_id is not None  [R3 guard]
    extra_allow_verified: bool = False
    telemetry_disabled: bool = False
    torch_threads: int = 0
    peak_rss_bytes: int = 0
    cell_guard: dict[str, int] = field(default_factory=dict)  # CUMULATIVE counters only
    worker_pid: int = 0
    worker_pgid: int = 0
    timing: RolloutTimingSummary | None = None
    # The status of the outcome the rollout's root recorded (``platoon.utils.trajectory_status.outcome_status``),
    # also when conversion left no result; ``error`` for a rollout whose environment could not open its session
    # (``EnvSetupError``), which records none; else None.  Counted per step, never trained on.
    outcome_status: str | None = None
    # The reward key the root's scored outcome trained (``platoon.utils.trajectory_status.trained_reward_key``), also
    # when conversion left no result; else None.  Counted per step; and unlike the diagnostics, training reads it: a
    # group whose members trained different keys does not train.
    reward_key: str | None = None
    # The reward the root's environment scored, before any reward processor
    # (``platoon.utils.trajectory_status.scored_success``), else None.  The per-task statistics count a pass on it, as
    # platoon eval judges success; never trained on.
    success_reward: float | None = None


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def ipc_dir() -> str:
    """The node-local directory holding IPC files: ``platoon-rollout-ipc-<uid>`` in node-local scratch
    (:func:`platoon.utils.scratch.node_scratch`), created 0700.

    Scratch can be shared by every user of a node (``/tmp``, or a site's ``$PLATOON_NODE_SCRATCH``), so each user has a
    directory of their own, and one that is not theirs alone is refused (``RuntimeError``): the files hold the
    gateway's key, and the trainer unpickles the responses.  Another user's directory, or a symlink, is refused; this
    user's, if others may enter it, is made private.
    """

    path = os.path.join(str(node_scratch()), f"{IPC_DIRNAME}-{os.getuid()}")
    os.makedirs(path, mode=0o700, exist_ok=True)
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError(
            f"{path} is not a directory of this user's, so the rollout workers' files, which hold the gateway's key, "
            "may not go there: remove it, or set PLATOON_NODE_SCRATCH to a directory of your own"
        )
    if stat.S_IMODE(info.st_mode) & 0o077:
        os.chmod(path, 0o700)
    return path


def _atomic_write(path: str, payload: bytes, *, mode: int) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    descriptor = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        unlink_quietly(tmp_path)
        raise
    os.chmod(tmp_path, mode)
    os.replace(tmp_path, path)


def unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            logger.debug("could not unlink %s: %s", path, exc)


def dump_request(request: RolloutWorkerRequest, path: str) -> None:
    """Write a request 0600 and atomically (the blob holds the gateway key)."""

    _atomic_write(path, pickle.dumps(request, protocol=5), mode=0o600)


def load_request(path: str) -> RolloutWorkerRequest:
    with open(path, "rb") as handle:
        request = pickle.load(handle)
    if not isinstance(request, RolloutWorkerRequest):
        raise ProtocolVersionError(f"{path} does not contain a RolloutWorkerRequest")
    if request.protocol_version != PROTOCOL_VERSION:
        raise ProtocolVersionError(f"rollout worker request protocol {request.protocol_version} != {PROTOCOL_VERSION}")
    return request


def assert_datum_funnel(response: RolloutWorkerResponse) -> None:
    """Fail in the child when a response would silently lose its datums.

    The parent validates ``0 <= retained <= post_sampling <= policy_eligible <=
    postmerge`` and raises ``ValueError``; the trainer catches that, records
    ``failed_rollouts`` and enqueues ``None``.  Asserting here turns that
    anonymous, silent whole-task data loss into an attributable traceback in
    the process that produced the datums.
    """

    result = response.result
    if result is None or not result.datums:
        return
    workload = response.workload
    retained = len(result.datums)
    if not (
        0 < retained <= workload.post_sampling_datums <= workload.policy_eligible_datums <= workload.postmerge_datums
    ):
        raise ValueError(
            "rollout worker response lost its datum funnel (add_datum_funnel was not applied): "
            f"datums={retained}, post_sampling={workload.post_sampling_datums}, "
            f"policy_eligible={workload.policy_eligible_datums}, "
            f"postmerge={workload.postmerge_datums}"
        )


def dump_response(response: RolloutWorkerResponse, path: str) -> None:
    assert_datum_funnel(response)
    _atomic_write(path, pickle.dumps(response, protocol=5), mode=0o600)


def load_response(path: str) -> RolloutWorkerResponse:
    with open(path, "rb") as handle:
        response = pickle.load(handle)
    if not isinstance(response, RolloutWorkerResponse):
        raise ProtocolVersionError(f"{path} does not contain a RolloutWorkerResponse")
    if response.protocol_version != PROTOCOL_VERSION:
        raise ProtocolVersionError(
            f"rollout worker response protocol {response.protocol_version} != {PROTOCOL_VERSION}"
        )
    return response


# --------------------------------------------------------------------------
# The .diag marker
# --------------------------------------------------------------------------


def encode_diag(fields: dict[str, Any]) -> bytes:
    """Pre-format a diagnostic marker for a single ``os.write``.

    A worker that is about to ``killpg`` itself for a hard timeout or an RSS
    overrun writes this first.  Without it, ``hard_timeout``, an OOM kill and
    an external sweep are three failures with the identical observable
    (``rc=-9``, no response) and three opposite operator responses -- and the
    cell-guard counters of the very rollout that hung would be lost, so
    ``ptc_cells_stuck`` would read 0 precisely when a cell was killed for
    hanging.
    """

    parts = []
    for key, value in fields.items():
        text = str(value).replace("\n", " ").replace("\t", " ").replace("=", "-")
        parts.append(f"{key}={text}")
    payload = ("\t".join(parts) + "\n").encode("utf-8", "replace")
    return payload[:DIAG_MAX_BYTES]


def decode_diag(payload: bytes | str) -> dict[str, str]:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", "replace")
    fields: dict[str, str] = {}
    for chunk in payload.strip().split("\t"):
        if "=" in chunk:
            key, _, value = chunk.partition("=")
            fields[key.strip()] = value.strip()
    return fields


def read_diag(path: str) -> dict[str, str] | None:
    """Read a ``.diag`` marker, or ``None`` when there is none."""

    try:
        with open(path, "rb") as handle:
            payload = handle.read(DIAG_MAX_BYTES)
    except OSError:
        return None
    if not payload.strip():
        return None
    return decode_diag(payload)


# --------------------------------------------------------------------------
# Callable references
# --------------------------------------------------------------------------


def resolve_ref(ref: str) -> Callable:
    """Resolve ``"package.module:qualname"`` to the object it names."""

    if not isinstance(ref, str) or ":" not in ref:
        raise RefResolutionError(f"callable reference must be 'module:qualname', got {ref!r}")
    module_name, _, qualname = ref.partition(":")
    if not module_name or not qualname:
        raise RefResolutionError(f"callable reference must be 'module:qualname', got {ref!r}")
    module = importlib.import_module(module_name)
    try:
        return functools.reduce(getattr, qualname.split("."), module)
    except AttributeError as exc:
        raise RefResolutionError(f"{module_name} has no attribute path {qualname!r}") from exc


def callable_ref(obj: Any) -> str:
    """Build ``"module:qualname"`` for a callable that can cross a process.

    A function defined at module scope in a ``python -m pkg.script`` entry point
    has ``__module__ == "__main__"``, while ``sys.modules["__main__"].__spec__.name``
    holds its real dotted name.  Train scripts may define ``reward_processor``
    in the script itself.
    """

    module_name = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if not module_name or not qualname:
        raise RefResolutionError(
            f"{obj!r} has no importable name; move it to a module so a rollout worker can import it"
        )
    if "<locals>" in qualname or "<lambda>" in qualname:
        raise RefResolutionError(
            f"{obj!r} is a closure or lambda and cannot be named across a process boundary; "
            "move it to module scope in an importable module"
        )
    if module_name == "__main__":
        spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        spec_name = getattr(spec, "name", None)
        if not spec_name:
            raise RefResolutionError(
                f"{obj!r} is defined in __main__ with no module spec; run the trainer with "
                "'python -m package.module' or move the callable into an importable module"
            )
        module_name = spec_name
    return f"{module_name}:{qualname}"


def validated_ref(obj: Any, *, what: str) -> str:
    """Build a ref and prove it round-trips, raising a named error otherwise."""

    ref = callable_ref(obj)
    try:
        resolved = resolve_ref(ref)
    except Exception as exc:
        raise RefResolutionError(
            f"{what} {obj!r} maps to {ref!r} which does not import in this process: {exc}"
        ) from exc
    if resolved is obj:
        return ref
    # Identity is the strongest check but it cannot always hold, and requiring
    # it would reject exactly the case the ``__main__`` mapping exists for: a
    # callable defined in a ``python -m pkg.script`` entry point lives in
    # ``__main__``, and resolving ``pkg.script:name`` imports that file a second
    # time, producing an equivalent but distinct object.  Fall back to matching
    # the qualified name, which still catches a ref that lands on some other
    # callable entirely.
    if not callable(resolved) or getattr(resolved, "__qualname__", None) != getattr(obj, "__qualname__", None):
        raise RefResolutionError(
            f"{what} {obj!r} maps to {ref!r} which resolves to a different object {resolved!r}; "
            "a rollout worker would silently run the wrong callable"
        )
    logger.debug("%s %r resolves to a re-imported copy via %r", what, obj, ref)
    return ref
