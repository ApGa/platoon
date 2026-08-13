"""Contracts for fail-closed PBS cleanup of exact SWE instances."""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "pbs-scripts" / "cleanup_swe_singularity_sessions.py"
SPEC = importlib.util.spec_from_file_location("swe_session_cleanup", HELPER)
assert SPEC is not None and SPEC.loader is not None
cleanup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cleanup)


class SWESingularityCleanupTests(unittest.TestCase):
    def _session(self, root: Path, *, with_lock: bool = True) -> Path:
        root.mkdir(mode=0o700)
        session = root / "swe-rebench-0123456789abcdef-private"
        session.mkdir(mode=0o700)
        if with_lock:
            (session / cleanup.SESSION_LOCK_FILE).touch(mode=0o600)
        return session

    def _bind_instance(self, session: Path, pid: int = 424242) -> str:
        name = "swe-rebench-0123456789abcdef"
        pid_path = session / cleanup.INSTANCE_PID_FILE
        pid_path.write_text(f"{pid}\n")
        pid_path.chmod(0o600)
        (session / cleanup.OWNER_FILE).write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "instance_name": name,
                    "session_directory": str(session),
                    "hostname": socket.gethostname(),
                    "instance_pid_file": str(pid_path),
                    "instance_pid": pid,
                    "instance_start_time": "17",
                    "owner_pid": 999999,
                    "owner_start_time": "3",
                }
            )
            + "\n"
        )
        (session / cleanup.OWNER_FILE).chmod(0o600)
        return name

    def test_missing_owner_record_is_recoverable_under_free_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sessions"
            session = self._session(root)
            with mock.patch.object(cleanup, "_instance_record", return_value=None):
                count = cleanup.cleanup_sessions(root, Path("/bin/true"), 0)
            self.assertEqual(count, 1)
            self.assertFalse(session.exists())

    def test_held_session_lease_prevents_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sessions"
            session = self._session(root)
            with (session / cleanup.SESSION_LOCK_FILE).open("a+") as lease:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
                with self.assertRaisesRegex(RuntimeError, "lease is still held"):
                    cleanup.cleanup_sessions(root, Path("/bin/true"), 0)
            self.assertTrue(session.is_dir())

    def test_young_legacy_directory_without_lease_is_not_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sessions"
            session = self._session(root, with_lock=False)
            with mock.patch.dict(
                os.environ,
                {"SWE_APPTAINER_LEGACY_SESSION_MIN_AGE_SECONDS": "300"},
            ):
                with self.assertRaisesRegex(RuntimeError, "has no lease"):
                    cleanup.cleanup_sessions(root, Path("/bin/true"), 0)
            self.assertTrue(session.is_dir())

    def test_exact_bound_pid_identity_is_stopped_without_wildcards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sessions"
            session = self._session(root)
            name = self._bind_instance(session)
            stopped = subprocess.CompletedProcess([], 0, "stopped")
            with mock.patch.object(
                cleanup,
                "_instance_record",
                side_effect=[{"instance": name, "pid": 424242}, None],
            ), mock.patch.object(
                cleanup,
                "_owned_process_start_time",
                return_value="17",
            ), mock.patch.object(
                cleanup,
                "_process_identity_matches",
                return_value=False,
            ), mock.patch.object(cleanup, "_run", return_value=stopped) as run:
                count = cleanup.cleanup_sessions(root, Path("/mock/singularity"), 7)

            self.assertEqual(count, 1)
            self.assertFalse(session.exists())
            run.assert_called_once_with(
                Path("/mock/singularity"),
                ["instance", "stop", "--timeout", "7", name],
                60,
            )
            self.assertNotIn("--all", run.call_args.args[1])
            self.assertFalse(any("*" in value for value in run.call_args.args[1]))

    def test_pid_mismatch_fails_closed_without_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sessions"
            session = self._session(root)
            name = self._bind_instance(session)
            with mock.patch.object(
                cleanup,
                "_instance_record",
                return_value={"instance": name, "pid": 424243},
            ), mock.patch.object(
                cleanup,
                "_owned_process_start_time",
                return_value="17",
            ), mock.patch.object(
                cleanup,
                "_process_identity_matches",
                return_value=False,
            ), mock.patch.object(cleanup, "_run") as run:
                with self.assertRaisesRegex(RuntimeError, "PID identity is ambiguous"):
                    cleanup.cleanup_sessions(root, Path("/mock/singularity"), 7)

            run.assert_not_called()
            self.assertTrue(session.is_dir())


if __name__ == "__main__":
    unittest.main()
