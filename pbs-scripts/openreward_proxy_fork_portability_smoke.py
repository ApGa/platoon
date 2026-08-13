#!/usr/bin/env python3
"""One-node live gate for PBS fork-child Python portability.

This intentionally starts only an AReaL guard, a forked Platoon proxy, and a
tiny HTTP health target.  It never loads model weights or starts SGLang.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import signal
import stat
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import urlopen

REQUESTED_PROXY_MODULE = "areal.experimental.openai.proxy.proxy_rollout_server"
EFFECTIVE_PROXY_MODULE = "platoon.areal_proxy_rollout"
PROXY_ENGINE = "areal.engine.sglang_remote.RemoteSGLangEngine"
INLINE_AGENT_CLASS = "areal.experimental.openai.proxy.online_agent._OnlineAgent"
TARGET_ROLE = "rollout"
PROXY_ROLE = "proxy-rollout"
CONTROLLER_CREDENTIAL_CANARY = "proxy-fork-smoke-controller-only-sentinel"
_OPENAI_PLACEHOLDERS = {
    "OPENAI_API_KEY": b"none",
    "OPENAI_BASE_URL": b"none",
}
_FORBIDDEN_CREDENTIAL_NAMES = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "LITELLM_API_KEY",
    "LITELLM_BASE_URL",
    "RUBRIC_DEFAULT_LLM",
    "WANDB_API_KEY",
}


class SmokeInterrupted(RuntimeError):
    pass


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if self.path != "/health":
            self.send_error(404)
            return
        payload = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def _private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
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


def _proc_cmdline(path: Path) -> list[str]:
    try:
        raw = path.read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [part.decode(errors="replace") for part in raw.rstrip(b"\0").split(b"\0") if part]


def _proc_environment(path: Path) -> tuple[dict[str, bytes], bytes]:
    try:
        raw = path.read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError) as error:
        raise RuntimeError(f"cannot attest fork-child environment from {path}") from error
    result: dict[str, bytes] = {}
    for part in raw.rstrip(b"\0").split(b"\0"):
        if b"=" not in part:
            continue
        raw_name, value = part.split(b"=", 1)
        name = raw_name.decode(errors="replace")
        if name in result:
            raise RuntimeError(f"fork-child environment repeats variable {name!r}")
        result[name] = value
    return result, raw


def proxy_processes(
    *,
    experiment: str,
    trial: str,
    worker_python: str,
    proc_root: Path = Path("/proc"),
    own_pid: int | None = None,
) -> list[dict[str, Any]]:
    """Return exact proxy children and fail on the wrong raw_cmd interpreter."""

    matches: list[dict[str, Any]] = []
    for child in proc_root.iterdir():
        if not child.name.isdigit() or int(child.name) == (own_pid or os.getpid()):
            continue
        command = _proc_cmdline(child / "cmdline")
        if not command:
            continue
        if not (
            EFFECTIVE_PROXY_MODULE in command and experiment in command and trial in command and PROXY_ROLE in command
        ):
            continue
        if command[0] != worker_python:
            raise RuntimeError(
                f"forked proxy raw_cmd used the wrong Python: expected {worker_python!r}, observed {command[0]!r}"
            )
        if REQUESTED_PROXY_MODULE in command:
            raise RuntimeError("fork-child argv retained the unpatched upstream proxy module")
        environment, raw_environment = _proc_environment(child / "environ")
        environment_names = set(environment)
        missing_safe = sorted({"HF_HOME", "PATH"} - environment_names)
        if missing_safe:
            raise RuntimeError(f"fork-child environment attestation lacks safe names: {missing_safe}")
        if CONTROLLER_CREDENTIAL_CANARY.encode() in raw_environment:
            raise RuntimeError("controller credential canary leaked to proxy child")
        if any(environment.get(name) != placeholder for name, placeholder in _OPENAI_PLACEHOLDERS.items()):
            # AReaL's OpenAI client deliberately installs these non-secret
            # placeholders in the owner Guard before /fork.  Require their
            # exact values without ever including an observed value in logs.
            raise RuntimeError("fork-child OpenAI placeholder environment is not exact")
        leaked = sorted(environment_names & _FORBIDDEN_CREDENTIAL_NAMES)
        if leaked:
            raise RuntimeError(f"controller credential names leaked to proxy child: {leaked}")
        matches.append(
            {
                "pid": int(child.name),
                "cmdline": command,
                "credential_names_present": leaked,
                "openai_placeholders": {name: "none" for name in sorted(_OPENAI_PLACEHOLDERS)},
                "controller_canary_absent": True,
            }
        )
    return sorted(matches, key=lambda item: int(item["pid"]))


def trial_processes(*, experiment: str, trial: str, proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for child in proc_root.iterdir():
        if not child.name.isdigit() or int(child.name) == os.getpid():
            continue
        command = _proc_cmdline(child / "cmdline")
        if experiment in command and trial in command:
            matches.append({"pid": int(child.name), "cmdline": command})
    return sorted(matches, key=lambda item: int(item["pid"]))


def wait_for_exact_proxy_process(*, experiment: str, trial: str, worker_python: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        matches = proxy_processes(
            experiment=experiment,
            trial=trial,
            worker_python=worker_python,
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"expected one forked proxy process, found {matches}")
        time.sleep(0.2)
    raise TimeoutError("forked proxy process never became visible in /proc")


def wait_for_no_trial_processes(*, experiment: str, trial: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        last = trial_processes(experiment=experiment, trial=trial)
        if not last:
            return
        time.sleep(0.2)
    raise TimeoutError(f"PBS smoke processes survived cleanup: {last}")


def _assert_private_regular(path: Path) -> None:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid() or stat.S_IMODE(value.st_mode) & 0o077:
        raise PermissionError(f"unsafe smoke artifact: {path}")


def _health_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - private cluster endpoint
        if response.status != 200:
            raise RuntimeError(f"health request returned HTTP {response.status}: {url}")
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise TypeError(f"health response is not an object: {value!r}")
    return value


def prune_empty_trial_tree(trial_root: Path) -> None:
    """Remove only an attested empty, job-owned name-resolution trial tree."""

    try:
        root_stat = trial_root.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(root_stat.st_mode) or root_stat.st_uid != os.getuid():
        raise RuntimeError(f"unsafe name-resolution trial root survived: {trial_root}")

    directories = [trial_root]
    for raw_root, dirnames, filenames in os.walk(trial_root, topdown=True, followlinks=False):
        current = Path(raw_root)
        for filename in filenames:
            child = current / filename
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode) or not stat.S_ISREG(child_stat.st_mode):
                raise RuntimeError(f"unsafe name-resolution entry survived cleanup: {child}")
            raise RuntimeError(f"name-resolution records survived cleanup under: {trial_root}")
        for dirname in dirnames:
            child = current / dirname
            child_stat = child.lstat()
            if not stat.S_ISDIR(child_stat.st_mode) or child_stat.st_uid != os.getuid():
                raise RuntimeError(f"unsafe name-resolution entry survived cleanup: {child}")
            directories.append(child)

    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        directory.rmdir()
    try:
        trial_root.lstat()
    except FileNotFoundError:
        return
    raise RuntimeError(f"name-resolution trial subtree survived: {trial_root}")


def _engine_config(args: argparse.Namespace):
    from areal.api.cli_args import AgentConfig, InferenceEngineConfig

    return InferenceEngineConfig(
        backend="sglang:d1",
        consumer_batch_size=1,
        max_concurrent_rollouts=1,
        queue_size=1,
        setup_timeout=20.0,
        tokenizer_path=str(args.model_snapshot),
        experiment_name=args.experiment,
        trial_name=args.trial,
        fileroot=str(args.fileroot),
        agent=AgentConfig(
            admin_api_key=secrets.token_urlsafe(32),
            agent_cls_path=INLINE_AGENT_CLASS,
            mode="inline",
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    from areal.api import Job
    from areal.api.cli_args import SchedulingSpec
    from areal.utils.network import gethostip

    from platoon.train.areal.pbs_cleanup_barrier import assert_cleanup_complete
    from platoon.train.areal.preallocated_pbs import PreallocatedPBSScheduler

    controller_python = str(Path(sys.executable).absolute())
    worker_python = str(args.worker_python)
    actual_driver_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if actual_driver_sha256 != args.driver_sha256:
        raise RuntimeError("executed driver snapshot digest does not match its launch receipt")
    if controller_python != str(args.controller_python):
        raise RuntimeError(f"smoke is not running in the exact controller venv: {controller_python}")
    if worker_python == controller_python:
        raise RuntimeError("controller and worker interpreters must be distinct")
    if os.environ.get("PLATOON_AREAL_PBS_WORKER_PYTHON") != worker_python:
        raise RuntimeError("scheduler worker-Python contract is not exported exactly")
    if os.environ.get("OPENAI_API_KEY") != CONTROLLER_CREDENTIAL_CANARY:
        raise RuntimeError("synthetic controller-only credential canary is absent")

    server = ThreadingHTTPServer((gethostip(), 0), _HealthHandler)
    target_host, target_port = server.server_address[:2]
    target_url = f"http://{target_host}:{target_port}"
    target_thread = threading.Thread(target=server.serve_forever, daemon=True)
    target_thread.start()

    scheduler: Any | None = None
    proxy_worker = None
    engine_created = False
    provenance: dict[str, Any] | None = None
    cleanup_error: BaseException | None = None
    primary_error: BaseException | None = None
    try:
        if _health_json(f"{target_url}/health").get("status") != "ok":
            raise RuntimeError("lightweight inference target is unhealthy")
        scheduler = PreallocatedPBSScheduler(
            n_gpus_per_node=8,
            experiment_name=args.experiment,
            trial_name=args.trial,
            fileroot=str(args.fileroot),
            startup_timeout=args.startup_timeout,
            health_check_interval=0.2,
            name_resolve_type="nfs",
            nfs_record_root=str(args.name_resolve_root),
        )
        target_ids = scheduler.create_workers(
            Job(
                role=TARGET_ROLE,
                replicas=1,
                tasks=[
                    SchedulingSpec(
                        cpu=1,
                        gpu=8,
                        mem=1,
                        port_count=2,
                        cmd="python -m platoon.areal_rpc_server",
                    )
                ],
            )
        )
        if target_ids != [f"{TARGET_ROLE}/0"]:
            raise RuntimeError(f"unexpected target worker IDs: {target_ids}")
        target_workers = scheduler.get_workers(TARGET_ROLE, timeout=args.startup_timeout)
        if len(target_workers) != 1:
            raise RuntimeError(f"expected one target guard, found {target_workers}")

        proxy_ids = scheduler.fork_workers(
            PROXY_ROLE,
            TARGET_ROLE,
            command=REQUESTED_PROXY_MODULE,
        )
        if proxy_ids != [f"{PROXY_ROLE}/0"]:
            raise RuntimeError(f"unexpected proxy worker IDs: {proxy_ids}")
        proxy_workers = scheduler.get_workers(PROXY_ROLE, timeout=30.0)
        if len(proxy_workers) != 1:
            raise RuntimeError(f"expected one proxy, found {proxy_workers}")
        proxy_worker = proxy_workers[0]
        provenance = wait_for_exact_proxy_process(
            experiment=args.experiment,
            trial=args.trial,
            worker_python=worker_python,
            timeout=20.0,
        )

        asyncio.run(
            scheduler.create_engine(
                proxy_worker.id,
                PROXY_ENGINE,
                engine_name="proxy-smoke-engine",
                config=_engine_config(args),
            )
        )
        engine_created = True
        asyncio.run(
            scheduler.async_call_engine(
                proxy_worker.id,
                "initialize",
                engine_name="proxy-smoke-engine",
                addr=f"{target_host}:{target_port}",
                train_data_parallel_size=1,
                http_timeout=60.0,
                max_retries=1,
            )
        )
        proxy_url = f"http://{proxy_worker.ip}:{proxy_worker.worker_ports[0]}"
        proxy_health = _health_json(f"{proxy_url}/health")
        if proxy_health != {"status": "ok", "initialized": True}:
            raise RuntimeError(f"proxy engine did not initialize: {proxy_health}")
        return {
            "version": 1,
            "status": "passed",
            "pbs_job_id": args.pbs_job_id,
            "experiment": args.experiment,
            "trial": args.trial,
            "controller_python": controller_python,
            "worker_python": worker_python,
            "requested_proxy_module": REQUESTED_PROXY_MODULE,
            "effective_proxy_module": EFFECTIVE_PROXY_MODULE,
            "proxy_engine": PROXY_ENGINE,
            "proxy_address": proxy_url,
            "proxy_health": proxy_health,
            "proxy_process": provenance,
            "source_sha256": args.source_sha256,
            "training_sif_sha256": args.training_sif_sha256,
            "worker_environment_key": args.worker_environment_key,
            "controller_environment_key": args.controller_environment_key,
            "driver_sha256": actual_driver_sha256,
        }
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if scheduler is not None:
            cleanup_errors: list[BaseException] = []
            try:
                if proxy_worker is not None and engine_created:
                    try:
                        asyncio.run(
                            scheduler.async_call_engine(
                                proxy_worker.id,
                                "destroy",
                                engine_name="proxy-smoke-engine",
                                http_timeout=30.0,
                                max_retries=1,
                            )
                        )
                    except BaseException as error:
                        cleanup_errors.append(error)
                # Production cleanup removes aliases first, then terminates
                # their owner guard.  The guard owns every /fork child and
                # kills it before publishing the durable launcher status.
                try:
                    scheduler.delete_workers(role=None, reverse_order=True)
                except BaseException as error:
                    cleanup_errors.append(error)
                if cleanup_errors:
                    cleanup_error = cleanup_errors[0]
            except BaseException as error:
                cleanup_error = error
        server.shutdown()
        server.server_close()
        target_thread.join(timeout=5.0)
        if scheduler is not None:
            state_dir = args.scheduler_state_root / args.pbs_job_id.replace("/", "_") / args.trial
            try:
                assert_cleanup_complete(
                    state_dir,
                    pbs_job_id=args.pbs_job_id,
                    experiment=args.experiment,
                    trial=args.trial,
                )
                wait_for_no_trial_processes(
                    experiment=args.experiment,
                    trial=args.trial,
                    timeout=30.0,
                )
                trial_root = args.name_resolve_root / os.environ["USER"] / args.experiment / args.trial
                prune_empty_trial_tree(trial_root)
            except BaseException as error:
                cleanup_error = cleanup_error or error
        if cleanup_error is not None:
            if primary_error is not None:
                primary_error.add_note(
                    f"Proxy smoke cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}"
                )
            else:
                raise cleanup_error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--scheduler-state-root", type=Path, required=True)
    parser.add_argument("--fileroot", type=Path, required=True)
    parser.add_argument("--name-resolve-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--controller-python", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--pbs-job-id", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--training-sif-sha256", required=True)
    parser.add_argument("--worker-environment-key", required=True)
    parser.add_argument("--controller-environment-key", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (
        args.state_root,
        args.scheduler_state_root,
        args.fileroot,
        args.name_resolve_root,
    ):
        if not path.is_absolute():
            raise ValueError(f"smoke paths must be absolute: {path}")
    if not args.model_snapshot.is_dir():
        raise FileNotFoundError(args.model_snapshot)

    def interrupted(signum: int, _frame: Any) -> None:
        raise SmokeInterrupted(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = run(args)
    result_path = args.state_root / "RESULT.json"
    _private_json(result_path, result)
    _assert_private_regular(result_path)
    print(f"proxy_fork_portability_smoke=passed result={result_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
