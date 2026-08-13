"""Launch and monitor one PBS role supervisor per physical host."""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_MANIFEST_VERSION = 1
_LOCK_METADATA_VERSION = 1
_CLEANUP_FAILURE_EXIT_STATUS = 70
_STATUS_WRITE_FAILURE_EXIT_STATUS = 71
_PR_SET_PDEATHSIG = 1


def _load_manifest(path: Path) -> dict[str, Any]:
    file_stat = path.stat()
    if file_stat.st_uid != os.getuid():
        raise PermissionError(f"Refusing manifest not owned by this user: {path}")
    if stat.S_IMODE(file_stat.st_mode) & 0o077:
        raise PermissionError(f"Refusing non-private manifest: {path}")
    with path.open() as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("version") != _MANIFEST_VERSION:
        raise ValueError(f"Unsupported PBS role manifest: {path}")
    expected_job_id = str(manifest.get("pbs_job_id", ""))
    actual_job_id = os.environ.get("PBS_JOBID", "")
    if expected_job_id and actual_job_id and expected_job_id != actual_job_id:
        raise RuntimeError(f"PBS manifest belongs to job {expected_job_id!r}, not {actual_job_id!r}")
    return manifest


def _atomic_private_json(path: Path, value: dict[str, Any]) -> None:
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
        temporary.unlink(missing_ok=True)


def _identity_digest(identity: dict[str, Any], role: str | None = None) -> str:
    value = dict(identity)
    if role is not None:
        value["role"] = role
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _open_private_lock(path: Path, *, create: bool):
    if create:
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
    flags = os.O_RDWR | os.O_NOFOLLOW
    if create:
        flags |= os.O_CREAT
    descriptor = os.open(path, flags, 0o600)
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


def _acquire_flock(stream, operation: int) -> None:
    while True:
        try:
            fcntl.flock(stream.fileno(), operation | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            # A stop signal does not bypass ownership: this launcher must queue
            # behind the predecessor before it can spawn *or* confirm cleanup.
            time.sleep(0.1)


def _locked_json(stream) -> dict[str, Any]:
    stream.seek(0)
    value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("PBS lifecycle lock metadata is not a JSON object")
    return value


def _write_locked_json(stream, value: dict[str, Any]) -> None:
    stream.seek(0)
    stream.truncate()
    json.dump(value, stream, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())


def _acquire_trial_lifecycle_lock(manifest: dict[str, Any]):
    """Hold shared trial ownership from before first spawn through cleanup."""

    _role_root, _entries, _role_lock_path, trial_lock_path, identity = _validated_discovery_paths(manifest)
    stream = _open_private_lock(trial_lock_path, create=False)
    try:
        _acquire_flock(stream, fcntl.LOCK_SH)
        expected = {
            "version": _LOCK_METADATA_VERSION,
            "kind": "pbs-trial-lifecycle",
            "identity": identity,
        }
        if _locked_json(stream) != expected:
            raise RuntimeError(f"PBS trial lifecycle metadata does not match manifest: {trial_lock_path}")
        return stream
    except BaseException:
        stream.close()
        raise


def _acquire_lifecycle_lock(manifest: dict[str, Any]):
    """Serialize every generation of one role through discovery cleanup.

    Generation-specific manifests/status files prevent record collisions.  A
    stable role lock ensures an older parent-death cleanup finishes before a
    replacement generation can clear stale records and spawn workers.
    """

    _role_root, _entries, lock_path, _trial_lock_path, identity = _validated_discovery_paths(manifest)
    stream = _open_private_lock(lock_path, create=True)
    try:
        _acquire_flock(stream, fcntl.LOCK_EX)
        _write_locked_json(
            stream,
            {
                "version": _LOCK_METADATA_VERSION,
                "kind": "pbs-role-lifecycle",
                "identity": identity,
                "role": str(manifest.get("role", "")),
                "generation": str(manifest.get("generation", "")),
            },
        )
        return stream
    except BaseException:
        stream.close()
        raise


def _validated_discovery_paths(
    manifest: dict[str, Any],
) -> tuple[Path, list[Path], Path, Path, dict[str, Any]]:
    discovery = manifest.get("discovery")
    if not isinstance(discovery, dict) or discovery.get("type") != "nfs":
        raise ValueError("PBS manifest requires NFS discovery cleanup metadata")
    record_root = Path(str(discovery.get("record_root", "")))
    role_root = Path(str(discovery.get("role_root", "")))
    if not record_root.is_absolute() or not role_root.is_absolute():
        raise ValueError("PBS discovery paths must be absolute")
    try:
        role_root.relative_to(record_root)
    except ValueError as error:
        raise ValueError(f"PBS discovery role root escapes record root: {role_root}") from error
    if role_root.name != str(manifest.get("role", "")):
        raise ValueError(f"PBS discovery role root does not match manifest role: {role_root}")
    raw_entries = discovery.get("entry_paths")
    replicas = int(manifest.get("replicas", -1))
    if not isinstance(raw_entries, list) or len(raw_entries) != replicas:
        raise ValueError("PBS discovery entry count does not match replicas")
    entries = [Path(str(value)) for value in raw_entries]
    expected = [role_root / str(index) / "ENTRY" for index in range(replicas)]
    if entries != expected:
        raise ValueError("PBS discovery entries are not the exact worker-index paths")
    identity = discovery.get("lock_identity")
    if not isinstance(identity, dict):
        raise ValueError("PBS manifest has no discovery lock identity")
    expected_identity = {
        "version": _LOCK_METADATA_VERSION,
        "record_root": str(record_root.resolve(strict=False)),
        "registry_user": role_root.parents[3].name,
        "experiment_name": role_root.parents[2].name,
        "trial_name": role_root.parents[1].name,
    }
    if identity != expected_identity:
        raise ValueError(f"PBS discovery lock identity is not canonical: {identity}")
    expected_role_root = (
        record_root
        / str(identity["registry_user"])
        / str(identity["experiment_name"])
        / str(identity["trial_name"])
        / "workers"
        / str(manifest.get("role", ""))
    )
    if role_root != expected_role_root:
        raise ValueError(f"PBS discovery role root is not the exact canonical namespace: {role_root}")
    registry_user = str(identity["registry_user"])
    if not registry_user or registry_user in {".", ".."} or Path(registry_user).name != registry_user:
        raise ValueError(f"PBS registry user is not one safe path component: {registry_user!r}")
    lock_root = Path(str(identity["record_root"])) / registry_user / ".pbs-prealloc-locks"
    constructor_lock_value = discovery.get("constructor_lifecycle_lock_path")
    if not constructor_lock_value:
        raise ValueError("PBS manifest has no constructor lifecycle lock path")
    constructor_lock_path = Path(str(constructor_lock_value))
    expected_constructor_lock = lock_root / f"constructor-{_identity_digest(identity)}.lock"
    if constructor_lock_path != expected_constructor_lock:
        raise ValueError(f"PBS constructor lifecycle lock is not the exact stable path: {constructor_lock_path}")
    lock_value = discovery.get("lifecycle_lock_path")
    if not lock_value:
        raise ValueError("PBS manifest has no discovery lifecycle lock path")
    lock_path = Path(str(lock_value))
    expected_lock = lock_root / f"role-{_identity_digest(identity, role_root.name)}.lock"
    if lock_path != expected_lock:
        raise ValueError(f"PBS discovery lifecycle lock is not the exact stable role path: {lock_path}")
    trial_lock_value = discovery.get("trial_lifecycle_lock_path")
    if not trial_lock_value:
        raise ValueError("PBS manifest has no trial lifecycle lock path")
    trial_lock_path = Path(str(trial_lock_value))
    expected_trial_lock = lock_root / f"trial-{_identity_digest(identity)}.lock"
    if trial_lock_path != expected_trial_lock:
        raise ValueError(f"PBS trial lifecycle lock is not the exact stable path: {trial_lock_path}")
    return role_root, entries, lock_path, trial_lock_path, identity


def _assert_lifecycle_generation(
    lifecycle_lock,
    manifest: dict[str, Any],
) -> None:
    """Require cleanup ownership to still name this exact generation."""

    try:
        lock_stat = os.fstat(lifecycle_lock.fileno())
        metadata = _locked_json(lifecycle_lock)
    except (AttributeError, OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("PBS discovery lifecycle lock is no longer held") from error
    expected_generation = str(manifest.get("generation", ""))
    if (
        not expected_generation
        or lock_stat.st_uid != os.getuid()
        or stat.S_IMODE(lock_stat.st_mode) & 0o077
        or metadata.get("kind") != "pbs-role-lifecycle"
        or metadata.get("generation") != expected_generation
        or metadata.get("role") != manifest.get("role")
        or metadata.get("identity") != manifest.get("discovery", {}).get("lock_identity")
    ):
        raise RuntimeError(
            "PBS discovery lifecycle generation changed before record cleanup: "
            f"expected {expected_generation!r}, found {metadata!r}"
        )


def _cleanup_discovery_records(manifest: dict[str, Any], lifecycle_lock) -> int:
    """Remove only this locked generation's exact worker discovery entries."""

    role_root, entries, _lock_path, _trial_lock_path, _identity = _validated_discovery_paths(manifest)
    _assert_lifecycle_generation(lifecycle_lock, manifest)

    def owned_directory(path: Path) -> bool:
        try:
            directory_stat = path.lstat()
        except FileNotFoundError:
            return False
        if not stat.S_ISDIR(directory_stat.st_mode) or directory_stat.st_uid != os.getuid():
            raise PermissionError(f"Refusing unsafe PBS discovery directory: {path}")
        return True

    if not owned_directory(role_root):
        return 0
    removed = 0
    for entry in entries:
        # Re-check for every record.  Under the normal flock contract another
        # generation cannot acquire the lock, while this extra nonce check also
        # fails closed if a stale cleanup callback is ever invoked after a
        # replacement generation has taken ownership.
        _assert_lifecycle_generation(lifecycle_lock, manifest)
        if not owned_directory(entry.parent):
            continue
        try:
            entry_stat = entry.lstat()
        except FileNotFoundError:
            # A worker may have self-removed its own ENTRY during orderly exit.
            # Remove that exact now-empty worker-index directory as part of the
            # same generation cleanup so the exact role subtree can disappear.
            try:
                entry.parent.rmdir()
            except OSError:
                pass
            continue
        if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_uid != os.getuid():
            raise PermissionError(f"Refusing unsafe PBS discovery entry: {entry}")
        entry.unlink()
        removed += 1
        try:
            entry.parent.rmdir()
        except OSError:
            pass
    # Remove only the now-empty exact role directory. Never recurse into or
    # remove the shared workers/trial tree, which may contain other live roles.
    if owned_directory(role_root):
        try:
            role_root.rmdir()
        except OSError:
            pass
    return removed


def supervisor_command(
    manifest_path: Path,
    host_index: int,
    manifest: dict[str, Any],
) -> list[str]:
    host = manifest["hosts"][host_index]
    supervisor_script = manifest.get("supervisor_script") or str(Path(__file__).with_name("pbs_role_supervisor.py"))
    return [
        str(manifest.get("pbsdsh_binary", "pbsdsh")),
        "-n",
        str(int(host["vnode_index"])),
        "--",
        str(manifest.get("python_binary", sys.executable)),
        str(supervisor_script),
        "--manifest",
        str(manifest_path),
        "--host-index",
        str(host_index),
    ]


def cleanup_command(
    host: dict[str, Any],
    manifest: dict[str, Any],
    *,
    reverse_order: bool = False,
) -> list[str]:
    supervisor_script = manifest.get("supervisor_script") or str(Path(__file__).with_name("pbs_role_supervisor.py"))
    command = [
        str(manifest.get("pbsdsh_binary", "pbsdsh")),
        "-n",
        str(int(host["vnode_index"])),
        "--",
        str(manifest.get("python_binary", sys.executable)),
        str(supervisor_script),
        "--cleanup-host",
        str(host["control_path"]),
        "--grace-seconds",
        str(float(manifest.get("termination_grace_seconds", 10.0))),
        "--generation",
        str(manifest["generation"]),
    ]
    if reverse_order:
        command.append("--reverse-order")
    return command


def _arm_parent_death_signal(expected_parent_pid: int) -> bool:
    """Ask Linux to TERM this launcher when its controller exits.

    ``PR_SET_PDEATHSIG`` alone has a fork/prctl race. Comparing PPID after the
    prctl closes it: a mismatch makes the normal launcher cleanup path run
    before any supervisors are spawned.
    """

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return os.getppid() == expected_parent_pid


def _stop_processes(
    processes: list[subprocess.Popen[Any]],
    grace_seconds: float,
    *,
    reverse_order: bool = False,
) -> None:
    if reverse_order:
        # PBS host order follows global worker order. Stop all non-rank-zero
        # hosts as one bounded batch, wait for them, then stop host zero. This
        # preserves the checkpointing contract without multiplying the grace
        # timeout by the number of allocated nodes.
        _stop_processes(processes[1:], grace_seconds)
        _stop_processes(processes[:1], grace_seconds)
        return
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + max(0.0, grace_seconds)
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        if process.poll() is None:
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass


def _requested_reverse_order(manifest: dict[str, Any]) -> bool:
    default = bool(manifest.get("termination_reverse_order_default", False))
    request_value = manifest.get("termination_request_path")
    if not request_value:
        return default
    request_path = Path(str(request_value))
    try:
        request_stat = request_path.stat()
        if request_stat.st_uid != os.getuid() or stat.S_IMODE(request_stat.st_mode) & 0o077:
            raise PermissionError(f"Refusing non-private termination request: {request_path}")
        with request_path.open() as stream:
            request = json.load(stream)
    except FileNotFoundError:
        return default
    if (
        not isinstance(request, dict)
        or request.get("version") != _MANIFEST_VERSION
        or request.get("role") != manifest.get("role")
        or request.get("generation") != manifest.get("generation")
    ):
        raise ValueError(f"Invalid PBS termination request: {request_path}")
    return bool(request.get("reverse_order", False))


def launch_role(
    manifest_path: Path,
    *,
    expected_parent_pid: int | None = None,
) -> int:
    manifest = _load_manifest(manifest_path)
    hosts = manifest.get("hosts")
    if not isinstance(hosts, list) or not hosts:
        raise ValueError("PBS role manifest must contain at least one host")

    grace_seconds = float(manifest.get("termination_grace_seconds", 10.0))
    processes: list[subprocess.Popen[Any]] = []
    stopping = False
    primary_status = 0
    launcher_exception: str | None = None
    cleanup_failures: list[dict[str, str]] = []
    status_write_failed = False
    lifecycle_lock = None
    trial_lifecycle_lock = None
    owns_discovery_generation = False
    stale_discovery_records_removed = 0
    discovery_records_removed = 0

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        try:
            if expected_parent_pid is not None and not _arm_parent_death_signal(expected_parent_pid):
                stopping = True
                print(
                    "PBS controller exited before its role launcher armed PDEATHSIG; cleaning up",
                    file=sys.stderr,
                    flush=True,
                )
            trial_lifecycle_lock = _acquire_trial_lifecycle_lock(manifest)
            lifecycle_lock = _acquire_lifecycle_lock(manifest)
            if not stopping:
                try:
                    stale_discovery_records_removed = _cleanup_discovery_records(
                        manifest,
                        lifecycle_lock,
                    )
                except BaseException as error:
                    cleanup_failures.append(
                        {
                            "hostname": "<name-resolve-start>",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    raise
                owns_discovery_generation = True

            # Spawn one persistent pbsdsh connection per *physical* host.
            # Starting serially keeps startup failures transactional.
            for host_index in range(len(hosts)):
                if stopping:
                    break
                process = subprocess.Popen(supervisor_command(manifest_path, host_index, manifest))
                processes.append(process)

            while not stopping:
                failed = next(
                    (process for process in processes if process.poll() is not None),
                    None,
                )
                if failed is not None:
                    print(
                        f"PBS host supervisor exited with status {failed.returncode}",
                        file=sys.stderr,
                        flush=True,
                    )
                    primary_status = failed.returncode or 1
                    break
                time.sleep(0.2)
        except BaseException as error:
            launcher_exception = f"{type(error).__name__}: {error}"
            raise
        finally:
            try:
                reverse_order = _requested_reverse_order(manifest)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                reverse_order = False
                cleanup_failures.append(
                    {
                        "hostname": "<termination-request>",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                print(
                    f"PBS termination request invalid; using safe default order: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            _stop_processes(
                processes,
                grace_seconds,
                reverse_order=reverse_order,
            )

            # pbsdsh normally propagates TERM. A second, identity-checked remote
            # cleanup closes the gap where the connection itself died first.
            def cleanup_one(host: dict[str, Any]) -> dict[str, str] | None:
                hostname = str(host.get("hostname"))
                try:
                    completed = subprocess.run(
                        cleanup_command(
                            host,
                            manifest,
                            reverse_order=reverse_order,
                        ),
                        check=False,
                        timeout=max(5.0, grace_seconds + 2.0),
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    return {
                        "hostname": hostname,
                        "error": f"{type(error).__name__}: {error}",
                    }
                returncode = int(getattr(completed, "returncode", 0))
                if returncode != 0:
                    return {
                        "hostname": hostname,
                        "error": f"cleanup command exited with status {returncode}",
                    }
                return None

            def cleanup_parallel(selected_hosts: list[dict[str, Any]]):
                if not selected_hosts:
                    return []
                executor = ThreadPoolExecutor(
                    max_workers=min(32, len(selected_hosts)),
                    thread_name_prefix="pbs-role-cleanup",
                )
                try:
                    futures = [executor.submit(cleanup_one, host) for host in selected_hosts]
                    return [future.result() for future in as_completed(futures)]
                finally:
                    executor.shutdown(wait=True)

            if reverse_order:
                cleanup_results = cleanup_parallel(hosts[1:])
                cleanup_results.extend(cleanup_parallel(hosts[:1]))
            else:
                cleanup_results = cleanup_parallel(hosts)
            for failure in cleanup_results:
                if failure is None:
                    continue
                cleanup_failures.append(failure)
                print(
                    f"PBS cleanup failed for {failure['hostname']}: {failure['error']}",
                    file=sys.stderr,
                    flush=True,
                )

            # Only a launcher holding the stable role lock may remove records,
            # and only after every remote cleanup command confirms no survivor.
            if not cleanup_failures and owns_discovery_generation:
                try:
                    discovery_records_removed = _cleanup_discovery_records(
                        manifest,
                        lifecycle_lock,
                    )
                except BaseException as error:
                    cleanup_failures.append(
                        {
                            "hostname": "<name-resolve-finish>",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    print(
                        f"PBS discovery cleanup failed: {error}",
                        file=sys.stderr,
                        flush=True,
                    )

            cleanup_failures.sort(key=lambda item: item["hostname"])
            status_path_value = manifest.get("launcher_status_path")
            if status_path_value:
                status_path = Path(str(status_path_value))
                try:
                    _atomic_private_json(
                        status_path,
                        {
                            "version": _MANIFEST_VERSION,
                            "complete": True,
                            "role": str(manifest.get("role", "")),
                            "generation": str(manifest.get("generation", "")),
                            "primary_status": primary_status,
                            "launcher_exception": launcher_exception,
                            "cleanup_host_count": len(hosts),
                            "cleanup_failures": cleanup_failures,
                            "stale_discovery_records_removed": (stale_discovery_records_removed),
                            "discovery_records_removed": discovery_records_removed,
                        },
                    )
                except OSError as error:
                    status_write_failed = True
                    print(
                        f"PBS launcher could not record cleanup completion: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
            else:
                status_write_failed = True
                print(
                    "PBS manifest has no launcher_status_path; cleanup cannot be confirmed",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if lifecycle_lock is not None:
            lifecycle_lock.close()
        if trial_lifecycle_lock is not None:
            trial_lifecycle_lock.close()

    if status_write_failed:
        return _STATUS_WRITE_FAILURE_EXIT_STATUS
    if cleanup_failures:
        return _CLEANUP_FAILURE_EXIT_STATUS
    return primary_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args(argv)
    return launch_role(args.manifest, expected_parent_pid=args.parent_pid)


if __name__ == "__main__":
    raise SystemExit(main())
