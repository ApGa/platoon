"""Publishing and following the trainer's sampling client.

Workers **follow**, they do not pin
=================================

The obvious design -- give each worker the ``SamplingClient`` its rollout
started with -- is wrong twice over:

* Under the production full-finetune gateway it is a **no-op**.  The gateway
  resolves a non-LoRA record to "the engines' own weights (FFT serves only the
  latest pushed version)", so a pinned old client still samples from the newest
  weights.  Believing otherwise would send someone hunting a kl/entropy change
  that cannot occur.
* Under a LoRA gateway it is a **new hard failure**.  Sampler versions are
  retired after ``sampler_versions_to_keep`` optim steps, protecting only
  versions with a sample in flight, so a rollout outliving three steps would
  get ``UserInputError: sampler version ... was retired`` on every remaining
  call -- root agent, subagents, judges and verifier alike.  Today the shared
  client is always rolled forward, so this is impossible.

So the parent republishes a small binding file on every
``update_sampling_client`` and the worker reloads it.  Net semantic change:
none for model sampling. Staleness is anchored to the oldest binding actually
used for generation, so waiting for an environment before the first sample
does not age otherwise fresh training data.

The telemetry foot-gun
======================

Unpickling a ``SamplingClient`` builds a **shadow** client holder that reuses
the trainer's ``session_id`` and, by default, its telemetry.  Any fatal
exception on that shadow path emits ``SESSION_END``, which the gateway treats
as the client exiting: it reclaims that session's models and slots and the
training state dies with the lease.  Every worker therefore sets
``TINKER_TELEMETRY=0`` before any tinker import and reports that it did; this
module asserts the resulting holder carries no telemetry when it can see one.
"""

from __future__ import annotations

import logging
import os
import pickle
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["FollowingSamplingClient", "load_binding_blob", "publish_binding", "read_binding"]

_BINDING_MAGIC = "platoon-tinker-model-binding"


def publish_binding(sampling_client: Any, version: int, path: str) -> None:
    """Atomically republish the trainer's sampling client for its workers.

    Written 0600: the pickled client carries the gateway API key in its
    constructor kwargs.
    """

    payload = pickle.dumps(
        {"magic": _BINDING_MAGIC, "version": int(version), "client": sampling_client},
        protocol=5,
    )
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, mode=0o700, exist_ok=True)
    tmp_path = f"{path}.{os.getpid()}.tmp"
    descriptor = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, path)


def read_binding(path: str) -> tuple[Any, int]:
    """Load ``(sampling_client, version)`` from a published binding file."""

    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("magic") != _BINDING_MAGIC:
        raise ValueError(f"{path} is not a platoon tinker model binding")
    return payload["client"], int(payload.get("version", 0))


def load_binding_blob(blob: bytes) -> Any:
    """Unpickle a bootstrap sampling client shipped inside the request."""

    return pickle.loads(blob)


class FollowingSamplingClient:
    """A ``SamplingClient`` that follows a republished binding file.

    ``TinkerLLM`` reads ``self.sampling_client`` at exactly two call sites, so
    this is a drop-in.  The binding file is ``stat``ed at most once per
    ``poll_seconds`` and reloaded only on an inode/mtime/size change; a corrupt or
    unreadable binding leaves the previously loaded client in service and logs
    a warning rather than failing the rollout.
    """

    def __init__(
        self,
        binding_path: str,
        bootstrap_blob: bytes = b"",
        poll_seconds: float = 5.0,
        bootstrap_version: int = 0,
    ) -> None:
        self._binding_path = binding_path
        self._poll_seconds = max(0.0, float(poll_seconds))
        self._lock = threading.Lock()
        self._client: Any = load_binding_blob(bootstrap_blob) if bootstrap_blob else None
        self._version = bootstrap_version
        self._signature: tuple[int, int, int] | None = None
        self._last_poll = 0.0
        self._reloads = 0
        self._oldest_sampled_version: int | None = None
        if self._client is None:
            # No bootstrap: the binding file is the only source, so read it now
            # and fail loudly here rather than at the first model call.
            self._reload(force=True)

    # -- following -------------------------------------------------------
    @property
    def binding_path(self) -> str:
        return self._binding_path

    @property
    def followed_version(self) -> int:
        return self._version

    @property
    def reload_count(self) -> int:
        return self._reloads

    @property
    def oldest_sampled_version(self) -> int | None:
        """Minimum policy version used by this rollout, excluding setup waits."""

        return self._oldest_sampled_version

    def _signature_of(self) -> tuple[int, int, int] | None:
        try:
            info = os.stat(self._binding_path)
        except OSError:
            return None
        # Atomic publications replace the inode. Timestamp granularity can be
        # too coarse to distinguish consecutive equal-sized bindings.
        return (info.st_ino, info.st_mtime_ns, info.st_size)

    def _reload(self, *, force: bool = False) -> None:
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._last_poll) < self._poll_seconds:
                return
            self._last_poll = now
            signature = self._signature_of()
            if signature is None:
                if force and self._client is None:
                    raise FileNotFoundError(f"no sampling-client binding at {self._binding_path}")
                return
            if signature == self._signature:
                return
            try:
                client, version = read_binding(self._binding_path)
            except Exception:
                # A half-written or corrupt binding must never end a rollout:
                # the previously loaded client still samples the same weights.
                logger.warning(
                    "rollout worker: could not load sampling-client binding %s; keeping the "
                    "previous client",
                    self._binding_path,
                    exc_info=True,
                )
                if self._client is None and force:
                    raise
                return
            self._signature = signature
            self._client = client
            self._version = version
            self._reloads += 1
            logger.info(
                "rollout worker: followed sampling-client binding version %s (reload #%s)",
                version,
                self._reloads,
            )

    def current(self) -> Any:
        """The client to use for the next call, reloading at most once per interval."""

        if self._poll_seconds >= 0.0:
            try:
                self._reload()
            except Exception:  # pragma: no cover - _reload already degrades
                logger.debug("rollout worker: binding poll failed", exc_info=True)
        client = self._client
        if client is None:
            raise RuntimeError(f"no sampling client available from {self._binding_path}")
        return client

    # -- SamplingClient surface -----------------------------------------
    def _for_sampling(self) -> Any:
        self._reload()
        with self._lock:
            if self._client is None:
                raise RuntimeError(f"no sampling client available from {self._binding_path}")
            version = self._version
            if self._oldest_sampled_version is None or version < self._oldest_sampled_version:
                self._oldest_sampled_version = version
            return self._client

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        return self._for_sampling().sample(*args, **kwargs)

    async def sample_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._for_sampling().sample_async(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.current(), name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"FollowingSamplingClient(binding_path={self._binding_path!r}, "
            f"version={self._version}, reloads={self._reloads})"
        )
