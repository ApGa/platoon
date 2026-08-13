"""Focused tests for fail-closed AReaL fork-alias teardown."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "platoon/train/areal/fork_lifecycle.py"
PATCHES_PATH = REPO_ROOT / "platoon/train/areal/patches.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeNameEntryNotFoundError(Exception):
    pass


class FakeNameResolve:
    NameEntryNotFoundError = FakeNameEntryNotFoundError

    def __init__(self, entries: dict[str, str]):
        self.entries = dict(entries)
        self.deleted: list[str] = []

    def get(self, key: str) -> str:
        try:
            return self.entries[key]
        except KeyError as exc:
            raise self.NameEntryNotFoundError(key) from exc

    def delete(self, key: str) -> None:
        if key not in self.entries:
            raise self.NameEntryNotFoundError(key)
        del self.entries[key]
        self.deleted.append(key)


class FakeNames:
    @staticmethod
    def worker_discovery(experiment: str, trial: str, role: str, index: int) -> str:
        return f"user/{experiment}/{trial}/workers/{role}/{index}"


class FakeRequestException(Exception):
    pass


class FakeConnectionError(FakeRequestException):
    pass


class FakeTimeout(FakeRequestException):
    pass


class FakeResponse:
    def __init__(self, status_code: int, payload: object):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeRequests:
    exceptions = types.SimpleNamespace(
        RequestException=FakeRequestException,
        ConnectionError=FakeConnectionError,
        Timeout=FakeTimeout,
    )

    def __init__(self, post_response: FakeResponse, *, health_response: object | None = None):
        self.post_response = post_response
        self.health_response = health_response or FakeConnectionError("connection refused")
        self.posts: list[tuple[str, dict, float]] = []
        self.gets: list[tuple[str, float]] = []

    def post(self, url: str, *, json: dict, timeout: float):
        self.posts.append((url, json, timeout))
        if isinstance(self.post_response, Exception):
            raise self.post_response
        return self.post_response

    def get(self, url: str, *, timeout: float):
        self.gets.append((url, timeout))
        if ":11" in url:
            return FakeResponse(200, {"status": "ok"})
        if isinstance(self.health_response, Exception):
            raise self.health_response
        return self.health_response


def _format_hostport(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _worker(role: str, index: int, host: str, port: int):
    return types.SimpleNamespace(
        worker=types.SimpleNamespace(
            id=f"{role}/{index}",
            ip=host,
            worker_ports=[str(port)],
        )
    )


def _scheduler(*, discovery_value: str | None = None, worker_count: int = 1):
    owners = [_worker("rollout", index, f"owner-{index}", 11000 + index) for index in range(worker_count)]
    forked_workers = [_worker("proxy", index, f"fork-{index}", 22000 + index) for index in range(worker_count)]
    # Preserve the single-worker endpoint spellings used by the concise tests.
    if worker_count == 1:
        owners[0].worker.ip = "owner-host"
        forked_workers[0].worker.ip = "owner-host"
    if discovery_value is None:
        discovery_value = "owner-host:22000" if worker_count == 1 else "fork-0:22000"
    entries = {
        f"user/exp/trial/workers/proxy/{index}": (discovery_value if index == 0 else f"fork-{index}:{22000 + index}")
        for index in range(worker_count)
    }
    scheduler = types.SimpleNamespace(
        experiment_name="exp",
        trial_name="trial",
        _colocated_roles={"proxy": "rollout"},
        _workers={"rollout": owners, "proxy": forked_workers},
    )
    return scheduler, FakeNameResolve(entries), "user/exp/trial/workers/proxy/0"


def _wire_runtime(monkeypatch, module, name_resolve, requests_module) -> None:
    monkeypatch.setattr(
        module,
        "_areal_name_resolve_components",
        lambda: (name_resolve, FakeNames, _format_hostport),
    )
    monkeypatch.setattr(module, "_requests_module", lambda: requests_module)


def test_fork_cleanup_kills_via_live_owner_and_deletes_only_exact_key(monkeypatch):
    module = _load_module("platoon_fork_lifecycle_success_test", HELPER_PATH)
    scheduler, name_resolve, key = _scheduler()
    requests_module = FakeRequests(FakeResponse(200, {"status": "success"}))
    _wire_runtime(monkeypatch, module, name_resolve, requests_module)

    assert module.cleanup_forked_alias_workers(scheduler, "proxy") is True

    assert requests_module.posts == [
        (
            "http://owner-host:11000/kill_forked_worker",
            {"role": "proxy", "worker_index": 0},
            15.0,
        )
    ]
    assert requests_module.gets[0][0] == "http://owner-host:22000/health"
    assert name_resolve.deleted == [key]
    assert "rollout" in scheduler._workers
    assert "proxy" in scheduler._workers
    assert scheduler._colocated_roles == {"proxy": "rollout"}


def test_fork_cleanup_retry_accepts_exact_guard_not_found_only_after_death(monkeypatch):
    module = _load_module("platoon_fork_lifecycle_retry_test", HELPER_PATH)
    scheduler, name_resolve, key = _scheduler()
    requests_module = FakeRequests(FakeResponse(404, {"error": "Forked worker proxy/0 not found"}))
    _wire_runtime(monkeypatch, module, name_resolve, requests_module)

    assert module.cleanup_forked_alias_workers(scheduler, "proxy") is True
    assert name_resolve.deleted == [key]

    # A second retry is idempotent once the exact key is already absent.
    assert module.cleanup_forked_alias_workers(scheduler, "proxy") is True
    assert name_resolve.deleted == [key]
    assert len(requests_module.gets) == 8


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(500, {"error": "kill failed"}),
        FakeResponse(200, {"status": "unexpected"}),
        FakeResponse(404, {"error": "some other child not found"}),
        FakeResponse(200, ValueError("not json")),
        FakeRequestException("owner unreachable"),
        FakeTimeout("owner request timed out"),
    ],
)
def test_fork_cleanup_response_failures_preserve_metadata_and_record(monkeypatch, response):
    module = _load_module(f"platoon_fork_lifecycle_response_failure_{id(response)}", HELPER_PATH)
    scheduler, name_resolve, key = _scheduler()
    requests_module = FakeRequests(response)
    _wire_runtime(monkeypatch, module, name_resolve, requests_module)

    with pytest.raises(module.ForkedAliasCleanupError, match="metadata retained"):
        module.cleanup_forked_alias_workers(scheduler, "proxy")

    assert scheduler._colocated_roles == {"proxy": "rollout"}
    assert set(scheduler._workers) == {"rollout", "proxy"}
    assert name_resolve.entries == {key: "owner-host:22000"}
    assert name_resolve.deleted == []


def test_fork_cleanup_health_failure_preserves_metadata_and_record(monkeypatch):
    module = _load_module("platoon_fork_lifecycle_health_failure_test", HELPER_PATH)
    scheduler, name_resolve, key = _scheduler()
    requests_module = FakeRequests(
        FakeResponse(200, {"status": "success"}),
        health_response=FakeResponse(200, {"status": "ok"}),
    )
    _wire_runtime(monkeypatch, module, name_resolve, requests_module)
    monkeypatch.setenv("PLATOON_AREAL_FORK_DEATH_TIMEOUT", "0.01")

    with pytest.raises(module.ForkedAliasCleanupError, match="metadata retained"):
        module.cleanup_forked_alias_workers(scheduler, "proxy")

    assert scheduler._colocated_roles == {"proxy": "rollout"}
    assert set(scheduler._workers) == {"rollout", "proxy"}
    assert name_resolve.entries == {key: "owner-host:22000"}


def test_single_transient_connection_refusal_is_not_death_attestation(monkeypatch):
    module = _load_module("platoon_fork_lifecycle_transient_refusal_test", HELPER_PATH)
    scheduler, name_resolve, key = _scheduler()

    class TransientRequests(FakeRequests):
        def __init__(self):
            super().__init__(FakeResponse(200, {"status": "success"}))
            self.health_calls = 0

        def get(self, url: str, *, timeout: float):
            self.gets.append((url, timeout))
            self.health_calls += 1
            if self.health_calls == 1:
                raise FakeConnectionError("transient reset")
            return FakeResponse(200, {"status": "ok"})

    requests_module = TransientRequests()
    _wire_runtime(monkeypatch, module, name_resolve, requests_module)
    monkeypatch.setenv("PLATOON_AREAL_FORK_DEATH_TIMEOUT", "0.15")

    with pytest.raises(module.ForkedAliasCleanupError, match="metadata retained"):
        module.cleanup_forked_alias_workers(scheduler, "proxy")

    assert requests_module.health_calls >= 2
    assert name_resolve.entries == {key: "owner-host:22000"}
    assert scheduler._colocated_roles == {"proxy": "rollout"}


def test_child_refusals_without_live_owner_do_not_attest_death(monkeypatch):
    module = _load_module("platoon_fork_lifecycle_owner_health_test", HELPER_PATH)
    scheduler, name_resolve, key = _scheduler()

    class UnreachableOwnerRequests(FakeRequests):
        def get(self, url: str, *, timeout: float):
            self.gets.append((url, timeout))
            raise FakeConnectionError("unreachable")

    requests_module = UnreachableOwnerRequests(FakeResponse(404, {"error": "Forked worker proxy/0 not found"}))
    _wire_runtime(monkeypatch, module, name_resolve, requests_module)
    monkeypatch.setenv("PLATOON_AREAL_FORK_DEATH_TIMEOUT", "0.35")

    with pytest.raises(module.ForkedAliasCleanupError, match="metadata retained"):
        module.cleanup_forked_alias_workers(scheduler, "proxy")

    assert any("owner-host" in url for url, _timeout in requests_module.gets)
    assert name_resolve.entries == {key: "owner-host:22000"}
    assert scheduler._colocated_roles == {"proxy": "rollout"}


def test_fork_cleanup_discovery_mismatch_fails_closed_without_deleting(monkeypatch):
    module = _load_module("platoon_fork_lifecycle_mismatch_test", HELPER_PATH)
    scheduler, name_resolve, key = _scheduler(discovery_value="replacement-host:33000")
    requests_module = FakeRequests(FakeResponse(200, {"status": "success"}))
    _wire_runtime(monkeypatch, module, name_resolve, requests_module)

    with pytest.raises(module.ForkedAliasCleanupError, match="Refusing to delete replaced"):
        module.cleanup_forked_alias_workers(scheduler, "proxy")

    assert scheduler._colocated_roles == {"proxy": "rollout"}
    assert set(scheduler._workers) == {"rollout", "proxy"}
    assert name_resolve.entries == {key: "replacement-host:33000"}
    assert name_resolve.deleted == []


def test_fork_cleanup_rejects_child_host_different_from_owner_before_kill(monkeypatch):
    module = _load_module("platoon_fork_lifecycle_host_mismatch_test", HELPER_PATH)
    scheduler, name_resolve, key = _scheduler()
    scheduler._workers["proxy"][0].worker.ip = "stale-other-host"
    requests_module = FakeRequests(FakeResponse(200, {"status": "success"}))
    _wire_runtime(monkeypatch, module, name_resolve, requests_module)

    with pytest.raises(module.ForkedAliasCleanupError, match="does not match owner"):
        module.cleanup_forked_alias_workers(scheduler, "proxy")

    assert requests_module.posts == []
    assert requests_module.gets == []
    assert name_resolve.entries == {key: "owner-host:22000"}
    assert name_resolve.deleted == []
    assert scheduler._colocated_roles == {"proxy": "rollout"}
    assert set(scheduler._workers) == {"rollout", "proxy"}


def test_multiworker_cleanup_maps_ids_to_owners_and_validates_all_before_delete(monkeypatch):
    module = _load_module("platoon_fork_lifecycle_multiworker_test", HELPER_PATH)
    scheduler, name_resolve, _key = _scheduler(worker_count=3)
    for index, owner_worker in enumerate(scheduler._workers["rollout"]):
        owner_worker.worker.ip = f"fork-{index}"
    # List order is intentionally unrelated to worker rank; identity must come
    # from each owner Worker.id, not a positional assumption.
    scheduler._workers["rollout"] = list(reversed(scheduler._workers["rollout"]))
    last_key = "user/exp/trial/workers/proxy/2"
    name_resolve.entries[last_key] = "replacement-host:33002"
    requests_module = FakeRequests(FakeResponse(200, {"status": "success"}))
    _wire_runtime(monkeypatch, module, name_resolve, requests_module)

    with pytest.raises(module.ForkedAliasCleanupError, match="Refusing to delete replaced"):
        module.cleanup_forked_alias_workers(scheduler, "proxy")

    assert {(url, payload["worker_index"]) for url, payload, _timeout in requests_module.posts} == {
        ("http://fork-0:11000/kill_forked_worker", 0),
        ("http://fork-1:11001/kill_forked_worker", 1),
        ("http://fork-2:11002/kill_forked_worker", 2),
    }
    assert name_resolve.deleted == []
    assert len(name_resolve.entries) == 3
    assert scheduler._colocated_roles == {"proxy": "rollout"}
    assert set(scheduler._workers) == {"rollout", "proxy"}


def test_reverse_order_submits_rank_zero_only_after_higher_ranks_complete(monkeypatch):
    module = _load_module("platoon_fork_lifecycle_reverse_order_test", HELPER_PATH)
    scheduler, name_resolve, _key = _scheduler(worker_count=3)
    for index, owner_worker in enumerate(scheduler._workers["rollout"]):
        owner_worker.worker.ip = f"fork-{index}"
    observed: list[tuple[str, int]] = []
    completed: set[int] = set()
    monkeypatch.setattr(
        module,
        "_areal_name_resolve_components",
        lambda: (name_resolve, FakeNames, _format_hostport),
    )

    def kill(worker, *, role, request_timeout, death_timeout):
        del role, request_timeout, death_timeout
        observed.append(("start", worker.index))
        if worker.index == 0:
            assert completed == {1, 2}
        completed.add(worker.index)
        observed.append(("done", worker.index))

    monkeypatch.setattr(module, "_kill_and_attest_worker", kill)

    assert (
        module.cleanup_forked_alias_workers(
            scheduler,
            "proxy",
            reverse_order=True,
        )
        is True
    )
    assert completed == {0, 1, 2}
    assert observed[-2:] == [("start", 0), ("done", 0)]
    assert {event for event in observed[:-2]} == {
        ("start", 1),
        ("done", 1),
        ("start", 2),
        ("done", 2),
    }


@pytest.mark.parametrize(
    "name,value",
    [
        ("PLATOON_AREAL_FORK_KILL_REQUEST_TIMEOUT", "not-a-number"),
        ("PLATOON_AREAL_FORK_KILL_REQUEST_TIMEOUT", "0"),
        ("PLATOON_AREAL_FORK_DEATH_TIMEOUT", "nan"),
        ("PLATOON_AREAL_FORK_DEATH_TIMEOUT", "-1"),
    ],
)
def test_invalid_timeout_configuration_is_normalized_and_preserves_state(
    monkeypatch,
    name,
    value,
):
    module = _load_module(f"platoon_fork_lifecycle_invalid_timeout_{name}_{value}", HELPER_PATH)
    scheduler, name_resolve, key = _scheduler()
    requests_module = FakeRequests(FakeResponse(200, {"status": "success"}))
    _wire_runtime(monkeypatch, module, name_resolve, requests_module)
    monkeypatch.setenv(name, value)

    with pytest.raises(module.ForkedAliasCleanupError, match=name):
        module.cleanup_forked_alias_workers(scheduler, "proxy")

    assert requests_module.posts == []
    assert name_resolve.entries == {key: "owner-host:22000"}
    assert scheduler._colocated_roles == {"proxy": "rollout"}
    assert set(scheduler._workers) == {"rollout", "proxy"}


def test_plain_colocated_alias_remains_metadata_only(monkeypatch):
    module = _load_module("platoon_fork_lifecycle_plain_alias_test", HELPER_PATH)
    owner = _worker("actor", 0, "owner-host", 11000)
    scheduler = types.SimpleNamespace(
        experiment_name="exp",
        trial_name="trial",
        _colocated_roles={"ref": "actor"},
        _workers={"actor": [owner]},
    )
    monkeypatch.setattr(
        module,
        "_requests_module",
        lambda: (_ for _ in ()).throw(AssertionError("plain alias must not use Guard")),
    )

    assert module.cleanup_forked_alias_workers(scheduler, "ref") is False
    assert scheduler._colocated_roles == {"ref": "actor"}
    assert scheduler._workers == {"actor": [owner]}


def test_dependent_alias_helper_deletes_all_aliases_before_owner_in_requested_order():
    module = _load_module("platoon_fork_lifecycle_dependents_test", HELPER_PATH)
    events = []

    class Scheduler:
        def __init__(self):
            self._colocated_roles = {
                "proxy-a": "rollout",
                "unrelated": "actor",
                "proxy-b": "rollout",
            }

        def delete_workers(self, role=None, reverse_order=False):
            events.append((role, reverse_order))
            self._colocated_roles.pop(role)

    scheduler = Scheduler()
    deleted = module.cleanup_dependent_aliases_before_owner(
        scheduler,
        "rollout",
        reverse_order=True,
    )

    assert deleted == ("proxy-b", "proxy-a")
    assert events == [("proxy-b", True), ("proxy-a", True)]
    assert scheduler._colocated_roles == {"unrelated": "actor"}


def test_dependent_alias_helper_recurses_nested_forks_leaf_first():
    module = _load_module("platoon_fork_lifecycle_nested_dependents_test", HELPER_PATH)
    events = []

    class Scheduler:
        def __init__(self):
            self._colocated_roles = {
                "proxy-parent": "rollout",
                "proxy-leaf": "proxy-parent",
            }
            self._workers = {
                "rollout": [object()],
                "proxy-parent": [object()],
                "proxy-leaf": [object()],
            }

        def delete_workers(self, role=None, reverse_order=False):
            module.cleanup_dependent_aliases_before_owner(
                self,
                role,
                reverse_order=reverse_order,
            )
            events.append((role, reverse_order))
            self._colocated_roles.pop(role, None)
            self._workers.pop(role)

    scheduler = Scheduler()
    scheduler.delete_workers("rollout", reverse_order=True)

    assert events == [
        ("proxy-leaf", True),
        ("proxy-parent", True),
        ("rollout", True),
    ]
    assert scheduler._colocated_roles == {}
    assert scheduler._workers == {}


def test_dependent_alias_cycle_fails_before_any_metadata_mutation():
    module = _load_module("platoon_fork_lifecycle_dependent_cycle_test", HELPER_PATH)
    events = []

    class Scheduler:
        def __init__(self):
            self._colocated_roles = {"proxy-a": "proxy-b", "proxy-b": "proxy-a"}
            self._workers = {"proxy-a": [object()], "proxy-b": [object()]}

        def delete_workers(self, role=None, reverse_order=False):
            module.cleanup_dependent_aliases_before_owner(
                self,
                role,
                reverse_order=reverse_order,
            )
            events.append(role)
            self._colocated_roles.pop(role)
            self._workers.pop(role)

    scheduler = Scheduler()
    original_mappings = dict(scheduler._colocated_roles)
    original_workers = dict(scheduler._workers)

    with pytest.raises(module.ForkedAliasCleanupError, match="cyclic colocation graph"):
        scheduler.delete_workers("proxy-a")

    assert events == []
    assert scheduler._colocated_roles == original_mappings
    assert scheduler._workers == original_workers


class _PatchedSchedulerBase:
    def __init__(self, *, forked: bool):
        self._colocated_roles = {"alias": "owner"}
        self._workers = {"owner": [object()]}
        if forked:
            self._workers["alias"] = [object()]
        self.original_calls = 0

    def delete_workers(self, role: str | None = None, reverse_order: bool = False):
        del reverse_order
        self.original_calls += 1
        if role in self._colocated_roles:
            self._workers.pop(role, None)
            self._colocated_roles.pop(role)


def test_local_and_slurm_delete_patches_are_idempotent_and_gate_metadata(monkeypatch):
    patches = _load_module("platoon_fork_lifecycle_scheduler_patch_test", PATCHES_PATH)
    cleanup_calls: list[tuple[object, str, bool]] = []
    fail_cleanup = False

    def cleanup(scheduler, role, *, reverse_order=False):
        cleanup_calls.append((scheduler, role, reverse_order))
        if fail_cleanup:
            raise RuntimeError("attestation failed")
        return role in scheduler._workers

    helper_module = types.ModuleType("platoon.train.areal.fork_lifecycle")
    helper_module.cleanup_dependent_aliases_before_owner = lambda scheduler, role, *, reverse_order=False: ()
    helper_module.cleanup_forked_alias_workers = cleanup
    monkeypatch.setitem(sys.modules, "platoon.train.areal.fork_lifecycle", helper_module)

    class LocalScheduler(_PatchedSchedulerBase):
        pass

    class SlurmScheduler(_PatchedSchedulerBase):
        pass

    assert patches._patch_areal_fork_alias_cleanup(LocalScheduler, SlurmScheduler) is True
    assert patches._patch_areal_fork_alias_cleanup(LocalScheduler, SlurmScheduler) is False

    for scheduler_cls in (LocalScheduler, SlurmScheduler):
        forked = scheduler_cls(forked=True)
        forked.delete_workers("alias", reverse_order=True)
        assert forked.original_calls == 1
        assert forked._colocated_roles == {}
        assert set(forked._workers) == {"owner"}

        plain = scheduler_cls(forked=False)
        plain.delete_workers("alias")
        assert plain.original_calls == 1
        assert plain._colocated_roles == {}
        assert set(plain._workers) == {"owner"}

    fail_cleanup = True
    for scheduler_cls in (LocalScheduler, SlurmScheduler):
        scheduler = scheduler_cls(forked=True)
        with pytest.raises(RuntimeError, match="attestation failed"):
            scheduler.delete_workers("alias")
        assert scheduler.original_calls == 0
        assert scheduler._colocated_roles == {"alias": "owner"}
        assert set(scheduler._workers) == {"owner", "alias"}

    assert len(cleanup_calls) == 6


def test_local_and_slurm_delete_all_recursion_cleans_alias_before_owner(monkeypatch):
    patches = _load_module("platoon_fork_lifecycle_delete_all_patch_test", PATCHES_PATH)
    events: list[str] = []

    def cleanup(scheduler, role, *, reverse_order=False):
        if role in scheduler._colocated_roles and role in scheduler._workers:
            events.append(f"cleanup:{role}:reverse={reverse_order}")
            return True
        return False

    helper_module = types.ModuleType("platoon.train.areal.fork_lifecycle")
    helper_module.cleanup_dependent_aliases_before_owner = lambda scheduler, role, *, reverse_order=False: ()
    helper_module.cleanup_forked_alias_workers = cleanup
    monkeypatch.setitem(sys.modules, "platoon.train.areal.fork_lifecycle", helper_module)

    class DeleteAllScheduler:
        def __init__(self):
            self._colocated_roles = {"alias": "owner"}
            self._workers = {"owner": [object()], "alias": [object()]}

        def delete_workers(self, role: str | None = None, reverse_order: bool = False):
            del reverse_order
            if role is None:
                for alias in list(self._colocated_roles):
                    self.delete_workers(alias)
                for owner in list(self._workers):
                    self.delete_workers(owner)
                return
            if role in self._colocated_roles:
                events.append("drop:alias")
                self._workers.pop(role)
                self._colocated_roles.pop(role)
                return
            events.append(f"drop:{role}")
            self._workers.pop(role)

    class LocalScheduler(DeleteAllScheduler):
        pass

    class SlurmScheduler(DeleteAllScheduler):
        pass

    patches._patch_areal_fork_alias_cleanup(LocalScheduler, SlurmScheduler)
    for scheduler_cls in (LocalScheduler, SlurmScheduler):
        events.clear()
        scheduler = scheduler_cls()
        scheduler.delete_workers(reverse_order=True)
        assert events == [
            "cleanup:alias:reverse=True",
            "drop:alias",
            "drop:owner",
        ]
        assert scheduler._workers == {}
        assert scheduler._colocated_roles == {}


def test_local_and_slurm_direct_owner_delete_cleans_all_aliases_first(monkeypatch):
    patches = _load_module("platoon_fork_lifecycle_direct_owner_patch_test", PATCHES_PATH)

    def cleanup_alias(scheduler, role, *, reverse_order=False):
        del reverse_order
        if role in scheduler._colocated_roles and role in scheduler._workers:
            events.append(f"kill:{role}")
            return True
        return False

    def cleanup_dependents(scheduler, role, *, reverse_order=False):
        if role in scheduler._colocated_roles:
            return ()
        aliases = [alias for alias, target in scheduler._colocated_roles.items() if target == role]
        if reverse_order:
            aliases.reverse()
        for alias in aliases:
            scheduler.delete_workers(alias, reverse_order=reverse_order)
        return tuple(aliases)

    helper_module = types.ModuleType("platoon.train.areal.fork_lifecycle")
    helper_module.cleanup_dependent_aliases_before_owner = cleanup_dependents
    helper_module.cleanup_forked_alias_workers = cleanup_alias
    monkeypatch.setitem(sys.modules, "platoon.train.areal.fork_lifecycle", helper_module)

    class SchedulerBase:
        def __init__(self):
            self._colocated_roles = {"proxy-a": "rollout", "proxy-b": "rollout"}
            self._workers = {
                "rollout": [object()],
                "proxy-a": [object()],
                "proxy-b": [object()],
            }

        def delete_workers(self, role=None, reverse_order=False):
            del reverse_order
            events.append(f"drop:{role}")
            if role in self._colocated_roles:
                self._colocated_roles.pop(role)
            self._workers.pop(role)

    class LocalScheduler(SchedulerBase):
        pass

    class SlurmScheduler(SchedulerBase):
        pass

    patches._patch_areal_fork_alias_cleanup(LocalScheduler, SlurmScheduler)
    for scheduler_cls in (LocalScheduler, SlurmScheduler):
        events: list[str] = []
        scheduler = scheduler_cls()
        scheduler.delete_workers("rollout", reverse_order=True)
        assert events == [
            "kill:proxy-b",
            "drop:proxy-b",
            "kill:proxy-a",
            "drop:proxy-a",
            "drop:rollout",
        ]
        assert scheduler._workers == {}
        assert scheduler._colocated_roles == {}


def test_local_and_slurm_direct_owner_delete_is_fail_closed(monkeypatch):
    patches = _load_module("platoon_fork_lifecycle_direct_owner_failure_test", PATCHES_PATH)

    def cleanup_alias(scheduler, role, *, reverse_order=False):
        del scheduler, role, reverse_order
        raise RuntimeError("alias attestation failed")

    def cleanup_dependents(scheduler, role, *, reverse_order=False):
        for alias, target in list(scheduler._colocated_roles.items()):
            if target == role:
                scheduler.delete_workers(alias, reverse_order=reverse_order)
        return ()

    helper_module = types.ModuleType("platoon.train.areal.fork_lifecycle")
    helper_module.cleanup_dependent_aliases_before_owner = cleanup_dependents
    helper_module.cleanup_forked_alias_workers = cleanup_alias
    monkeypatch.setitem(sys.modules, "platoon.train.areal.fork_lifecycle", helper_module)

    class SchedulerBase(_PatchedSchedulerBase):
        def delete_workers(self, role=None, reverse_order=False):
            del reverse_order
            self.original_calls += 1
            self._workers.pop(role)

    class LocalScheduler(SchedulerBase):
        pass

    class SlurmScheduler(SchedulerBase):
        pass

    patches._patch_areal_fork_alias_cleanup(LocalScheduler, SlurmScheduler)
    for scheduler_cls in (LocalScheduler, SlurmScheduler):
        scheduler = scheduler_cls(forked=True)
        with pytest.raises(RuntimeError, match="alias attestation failed"):
            scheduler.delete_workers("owner")
        assert scheduler.original_calls == 0
        assert set(scheduler._workers) == {"owner", "alias"}
        assert scheduler._colocated_roles == {"alias": "owner"}


def test_rollout_controller_cleanup_errors_become_terminal_after_cleanup(monkeypatch):
    patches = _load_module("platoon_rollout_cleanup_error_patch_test", PATCHES_PATH)

    class Scheduler:
        def __init__(self, *, fail_owner=False):
            self.fail_owner = fail_owner
            self.events: list[str] = []

        def delete_workers(self, role=None, reverse_order=False):
            del reverse_order
            if role == "rollout":
                self.events.extend(["delete:proxy", "delete:rollout"])
                if self.fail_owner:
                    raise RuntimeError("fork alias cleanup failed")
                return
            self.events.append(f"delete:{role}:retry")

    class RolloutController:
        def __init__(self, scheduler):
            self.scheduler = scheduler
            self.events: list[str] = []

        def destroy(self):
            try:
                self.scheduler.delete_workers(role="rollout")
            except Exception:
                self.events.append("caught:rollout")
            self.events.append("remaining-cleanup")
            try:
                self.scheduler.delete_workers(role="proxy")
            except Exception:
                self.events.append("caught:proxy")
            self.events.append("destroy-complete")
            return "destroyed"

    original_signature = __import__("inspect").signature(RolloutController.destroy)
    assert patches._patch_rollout_controller_cleanup_error_propagation(RolloutController) is True
    assert patches._patch_rollout_controller_cleanup_error_propagation(RolloutController) is False
    assert __import__("inspect").signature(RolloutController.destroy) == original_signature
    assert getattr(
        RolloutController.destroy,
        "__platoon_cleanup_error_propagation_patch__",
        False,
    )

    successful_scheduler = Scheduler()
    successful = RolloutController(successful_scheduler)
    assert successful.destroy() == "destroyed"
    assert successful.scheduler is successful_scheduler
    assert successful_scheduler.events == [
        "delete:proxy",
        "delete:rollout",
        "delete:proxy:retry",
    ]

    failed_scheduler = Scheduler(fail_owner=True)
    failed = RolloutController(failed_scheduler)
    with pytest.raises(RuntimeError, match="could not be confirmed") as exc_info:
        failed.destroy()
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "fork alias cleanup failed"
    assert failed.scheduler is failed_scheduler
    assert failed.events == [
        "caught:rollout",
        "remaining-cleanup",
        "destroy-complete",
    ]
    assert failed_scheduler.events == [
        "delete:proxy",
        "delete:rollout",
        "delete:proxy:retry",
    ]


def test_rollout_controller_patch_restores_scheduler_on_base_exception():
    patches = _load_module("platoon_rollout_cleanup_restore_patch_test", PATCHES_PATH)

    class RolloutController:
        def __init__(self):
            self.scheduler = object()

        def destroy(self):
            raise KeyboardInterrupt("stop")

    patches._patch_rollout_controller_cleanup_error_propagation(RolloutController)
    controller = RolloutController()
    scheduler = controller.scheduler
    with pytest.raises(KeyboardInterrupt, match="stop"):
        controller.destroy()
    assert controller.scheduler is scheduler
