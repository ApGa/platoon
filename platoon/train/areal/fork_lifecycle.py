"""Fail-closed lifecycle handling for AReaL forked scheduler aliases."""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any


class ForkedAliasCleanupError(RuntimeError):
    """Raised when a forked alias cannot be safely detached from its owner."""


@dataclass(frozen=True)
class _ForkedAliasWorker:
    index: int
    owner_endpoint: str
    fork_endpoint: str
    discovery_key: str


def _positive_timeout(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise ForkedAliasCleanupError(f"{name} must be a positive finite number, got {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ForkedAliasCleanupError(f"{name} must be a positive finite number, got {raw!r}")
    return value


def _areal_name_resolve_components():
    # Keep imports lazy so this controller-only helper does not broaden the
    # import surface of scheduler modules used in lightweight environments.
    from areal.utils import name_resolve, names  # pyright: ignore[reportMissingImports]
    from areal.utils.network import format_hostport  # pyright: ignore[reportMissingImports]

    return name_resolve, names, format_hostport


def _requests_module():
    # AReaL's real scheduler environment owns this dependency. Keep it lazy so
    # lightweight config/unit imports do not require the full runtime stack.
    import requests  # pyright: ignore[reportMissingImports]

    return requests


def _worker_endpoint(worker_info: Any, *, label: str, format_hostport) -> str:
    worker = getattr(worker_info, "worker", None)
    ip = getattr(worker, "ip", None)
    ports = getattr(worker, "worker_ports", None)
    if not ip or not ports:
        raise ForkedAliasCleanupError(f"{label} has no discovered RPC endpoint")
    try:
        port = int(ports[0])
    except (TypeError, ValueError, IndexError) as exc:
        raise ForkedAliasCleanupError(f"{label} has an invalid primary RPC port: {ports!r}") from exc
    return format_hostport(str(ip), port)


def _worker_host(worker_info: Any, *, label: str) -> str:
    worker = getattr(worker_info, "worker", None)
    ip = getattr(worker, "ip", None)
    if not ip:
        raise ForkedAliasCleanupError(f"{label} has no discovered RPC host")
    return str(ip)


def _fork_index(role: str, worker_info: Any) -> int:
    worker_id = str(getattr(getattr(worker_info, "worker", None), "id", ""))
    worker_role, separator, raw_index = worker_id.rpartition("/")
    if not separator or worker_role != role:
        raise ForkedAliasCleanupError(f"Forked role {role!r} contains unexpected worker id {worker_id!r}")
    try:
        index = int(raw_index)
    except ValueError as exc:
        raise ForkedAliasCleanupError(f"Forked worker id {worker_id!r} has a non-integer index") from exc
    if index < 0:
        raise ForkedAliasCleanupError(f"Forked worker id {worker_id!r} has a negative index")
    return index


def _owner_worker_by_index(target_role: str, owner_workers: list[Any]) -> dict[int, Any]:
    indexed: dict[int, Any] = {}
    for owner_worker in owner_workers:
        index = _fork_index(target_role, owner_worker)
        if index in indexed:
            raise ForkedAliasCleanupError(f"Owner role {target_role!r} contains duplicate worker index {index}")
        indexed[index] = owner_worker
    return indexed


def _kill_response_attested(response: Any, role: str, index: int) -> None:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ForkedAliasCleanupError(
            f"Owner Guard returned non-JSON while killing {role}/{index}: HTTP {response.status_code}"
        ) from exc

    if response.status_code == 200 and isinstance(payload, dict) and payload.get("status") == "success":
        return

    # A retry after a partially completed teardown can observe that the live
    # owner Guard no longer tracks the child. This is safe only with the pinned
    # Guard's exact not-found attestation and the independent endpoint-death
    # check below.
    expected_not_found = f"Forked worker {role}/{index} not found"
    if response.status_code == 404 and isinstance(payload, dict) and payload.get("error") == expected_not_found:
        return

    raise ForkedAliasCleanupError(
        f"Owner Guard refused to kill {role}/{index}: HTTP {response.status_code}, payload={payload!r}"
    )


def _wait_for_endpoint_death(
    endpoint: str,
    owner_endpoint: str,
    timeout: float,
    requests_module: Any,
) -> None:
    health_url = f"http://{endpoint}/health"
    deadline = time.monotonic() + timeout
    last_observation = "endpoint remained reachable"
    consecutive_refusals = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            response = requests_module.get(health_url, timeout=min(1.0, remaining))
        except requests_module.exceptions.ConnectionError:
            consecutive_refusals += 1
            # A single reset/refusal can be transient. The owner Guard request
            # already proved the node reachable; require a stable sequence on
            # the child's exact port before treating it as a death attestation.
            if consecutive_refusals >= 3:
                try:
                    owner_health = requests_module.get(
                        f"http://{owner_endpoint}/health",
                        timeout=min(1.0, max(0.001, deadline - time.monotonic())),
                    )
                except requests_module.exceptions.RequestException as exc:
                    consecutive_refusals = 0
                    last_observation = (
                        f"child refused connections, but its owner Guard could not attest node reachability: {exc}"
                    )
                else:
                    if owner_health.status_code == 200:
                        return
                    consecutive_refusals = 0
                    last_observation = (
                        "child refused connections, but its owner Guard health "
                        f"answered HTTP {owner_health.status_code}"
                    )
            last_observation = f"child endpoint refused {consecutive_refusals}/3 consecutive probes"
        except requests_module.exceptions.Timeout:
            consecutive_refusals = 0
            last_observation = "health request timed out"
        except requests_module.exceptions.RequestException as exc:
            consecutive_refusals = 0
            last_observation = f"health request failed without a death attestation: {exc}"
        else:
            consecutive_refusals = 0
            last_observation = f"health endpoint still answered HTTP {response.status_code}"
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    raise ForkedAliasCleanupError(
        f"Forked worker endpoint {endpoint} did not attest process death within {timeout:g}s: {last_observation}"
    )


def _kill_and_attest_worker(
    worker: _ForkedAliasWorker,
    *,
    role: str,
    request_timeout: float,
    death_timeout: float,
) -> None:
    requests_module = _requests_module()
    try:
        response = requests_module.post(
            f"http://{worker.owner_endpoint}/kill_forked_worker",
            json={"role": role, "worker_index": worker.index},
            timeout=request_timeout,
        )
    except requests_module.exceptions.RequestException as exc:
        raise ForkedAliasCleanupError(f"Owner Guard request failed while killing {role}/{worker.index}: {exc}") from exc
    _kill_response_attested(response, role, worker.index)
    _wait_for_endpoint_death(
        worker.fork_endpoint,
        worker.owner_endpoint,
        death_timeout,
        requests_module,
    )


def _reconcile_exact_discovery_entries(
    workers: list[_ForkedAliasWorker],
    *,
    role: str,
    name_resolve: Any,
) -> None:
    mismatches: list[str] = []
    matching: list[_ForkedAliasWorker] = []
    for worker in workers:
        try:
            current_value = name_resolve.get(worker.discovery_key)
        except name_resolve.NameEntryNotFoundError:
            continue
        if current_value != worker.fork_endpoint:
            mismatches.append(
                f"{role}/{worker.index} key {worker.discovery_key!r} points to "
                f"{current_value!r}, expected {worker.fork_endpoint!r}"
            )
            continue
        matching.append(worker)
    if mismatches:
        raise ForkedAliasCleanupError("Refusing to delete replaced fork discovery entries: " + "; ".join(mismatches))

    # Validation is deliberately complete before the first mutation, so one
    # replaced record cannot leave earlier records partially deleted while
    # scheduler metadata remains for a retry. Recheck the complete set once
    # immediately before mutation. The generic AReaL resolver has no CAS API;
    # scheduler metadata prevents same-controller role reuse during this small
    # compare/delete interval.
    still_matching: list[_ForkedAliasWorker] = []
    for worker in matching:
        try:
            current_value = name_resolve.get(worker.discovery_key)
        except name_resolve.NameEntryNotFoundError:
            continue
        if current_value != worker.fork_endpoint:
            raise ForkedAliasCleanupError(
                f"Refusing to delete replaced fork discovery entry for {role}/{worker.index}: "
                f"key {worker.discovery_key!r} now points to {current_value!r}, "
                f"expected {worker.fork_endpoint!r}"
            )
        still_matching.append(worker)

    for worker in still_matching:
        try:
            name_resolve.delete(worker.discovery_key)
        except name_resolve.NameEntryNotFoundError:
            # The forked process may have completed its own finally cleanup
            # between our compare and exact-key delete.
            pass


def cleanup_forked_alias_workers(
    scheduler: Any,
    role: str,
    *,
    reverse_order: bool = False,
) -> bool:
    """Kill and attest a forked alias before scheduler metadata is removed.

    Returns ``False`` for a plain colocated alias, whose role mapping reuses its
    owner's workers and therefore has no child processes to kill. For a true
    forked alias, every child is killed through its still-live owner Guard,
    every child endpoint is confirmed unreachable, and only its exact matching
    discovery key is deleted. Any failure raises before caller-owned role
    metadata is dropped.
    """

    target_role = getattr(scheduler, "_colocated_roles", {}).get(role)
    workers_by_role = getattr(scheduler, "_workers", {})
    if target_role is None or role not in workers_by_role:
        return False

    forked_workers = list(workers_by_role[role])
    owner_workers = workers_by_role.get(target_role)
    if owner_workers is None:
        raise ForkedAliasCleanupError(
            f"Cannot delete forked role {role!r}: owner role {target_role!r} is no longer present"
        )
    owner_workers_by_index = _owner_worker_by_index(target_role, list(owner_workers))

    name_resolve, names, format_hostport = _areal_name_resolve_components()
    seen_indices: set[int] = set()
    workers: list[_ForkedAliasWorker] = []
    for forked_worker in forked_workers:
        index = _fork_index(role, forked_worker)
        if index in seen_indices:
            raise ForkedAliasCleanupError(f"Forked role {role!r} contains duplicate worker index {index}")
        seen_indices.add(index)
        if index not in owner_workers_by_index:
            raise ForkedAliasCleanupError(f"Forked worker {role}/{index} has no owner worker {target_role}/{index}")
        owner_worker = owner_workers_by_index[index]
        owner_host = _worker_host(
            owner_worker,
            label=f"owner worker {target_role}/{index}",
        )
        fork_host = _worker_host(
            forked_worker,
            label=f"forked worker {role}/{index}",
        )
        if fork_host != owner_host:
            raise ForkedAliasCleanupError(
                f"Forked worker {role}/{index} host {fork_host!r} does not match "
                f"owner worker {target_role}/{index} host {owner_host!r}"
            )
        workers.append(
            _ForkedAliasWorker(
                index=index,
                owner_endpoint=_worker_endpoint(
                    owner_worker,
                    label=f"owner worker {target_role}/{index}",
                    format_hostport=format_hostport,
                ),
                fork_endpoint=_worker_endpoint(
                    forked_worker,
                    label=f"forked worker {role}/{index}",
                    format_hostport=format_hostport,
                ),
                discovery_key=names.worker_discovery(
                    scheduler.experiment_name,
                    scheduler.trial_name,
                    role,
                    index,
                ),
            )
        )

    workers.sort(key=lambda worker: worker.index, reverse=reverse_order)

    request_timeout = _positive_timeout("PLATOON_AREAL_FORK_KILL_REQUEST_TIMEOUT", 15.0)
    death_timeout = _positive_timeout("PLATOON_AREAL_FORK_DEATH_TIMEOUT", 15.0)
    failures: list[str] = []

    def kill_concurrently(worker_batch: list[_ForkedAliasWorker]) -> None:
        if not worker_batch:
            return
        with ThreadPoolExecutor(max_workers=min(32, len(worker_batch))) as executor:
            pending = {
                executor.submit(
                    _kill_and_attest_worker,
                    worker,
                    role=role,
                    request_timeout=request_timeout,
                    death_timeout=death_timeout,
                ): worker
                for worker in worker_batch
            }
            for future in as_completed(pending):
                worker = pending[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append(f"{role}/{worker.index}: {exc}")

    if workers and reverse_order:
        # Preserve the scheduler's rank-zero-last contract without multiplying
        # teardown time by the number of proxy workers: all nonzero ranks run
        # concurrently and complete before rank zero is submitted.
        rank_zero = next((worker for worker in workers if worker.index == 0), None)
        kill_concurrently([worker for worker in workers if worker.index != 0])
        if rank_zero is not None:
            try:
                _kill_and_attest_worker(
                    rank_zero,
                    role=role,
                    request_timeout=request_timeout,
                    death_timeout=death_timeout,
                )
            except Exception as exc:
                failures.append(f"{role}/{rank_zero.index}: {exc}")
    else:
        kill_concurrently(workers)
    if failures:
        raise ForkedAliasCleanupError(
            f"Forked role {role!r} cleanup failed; scheduler metadata retained: " + "; ".join(sorted(failures))
        )

    _reconcile_exact_discovery_entries(
        workers,
        role=role,
        name_resolve=name_resolve,
    )
    return True


def cleanup_dependent_aliases_before_owner(
    scheduler: Any,
    role: str,
    *,
    reverse_order: bool = False,
) -> tuple[str, ...]:
    """Delete every alias targeting ``role`` before that role can disappear.

    Some AReaL controllers delete a concrete worker role directly rather than
    calling ``delete_workers(role=None)``. Resolve dependents at the scheduler
    boundary so every caller gets the same alias-before-owner invariant. The
    recursive public calls also handle nested forks leaf-first while retaining
    each scheduler's normal alias bookkeeping. Any failure propagates before
    the requested role's deletion path is entered.
    """

    colocated_roles = getattr(scheduler, "_colocated_roles", {})

    # Validate the complete functional graph before the first recursive
    # deletion. AReaL permits any existing worker role (including a forked
    # alias) as a fork target, so nested aliases are valid but cycles are not.
    for start_role in colocated_roles:
        path: list[str] = []
        positions: dict[str, int] = {}
        current_role = start_role
        while current_role in colocated_roles:
            if current_role in positions:
                cycle = path[positions[current_role] :] + [current_role]
                raise ForkedAliasCleanupError(
                    "Cannot delete workers with a cyclic colocation graph: " + " -> ".join(cycle)
                )
            positions[current_role] = len(path)
            path.append(current_role)
            current_role = colocated_roles[current_role]

    dependent_roles = [alias_role for alias_role, target_role in colocated_roles.items() if target_role == role]
    if reverse_order:
        dependent_roles.reverse()
    for alias_role in dependent_roles:
        scheduler.delete_workers(
            role=alias_role,
            reverse_order=reverse_order,
        )
    return tuple(dependent_roles)
