#!/usr/bin/env python3
"""Live acceptance test for the ABCI SWE Singularity sandbox backend."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

from dataset_store import TaskDataset
from preflight import EXECUTION_COLUMNS, execute_task
from sandbox_backends import SingularitySandbox, _singularity_pull_uri


TASK_INDEX = 11
EDIT_MARKER = "OPENREWARD_ABCI_LIVE_EDIT_7f52f5e8"
STATE_FILE = ".openreward-abci-live-state"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


async def checked(
    sandbox: SingularitySandbox,
    command: str,
    *,
    timeout: float | None = None,
) -> str:
    result = await sandbox.run(command, timeout=timeout)
    if result.return_code != 0:
        raise RuntimeError(
            f"sandbox command failed with {result.return_code}: {command}\n"
            f"{result.output}"
        )
    return result.output.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_instance_record(value: object, name: str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if value.get("instance", value.get("name")) == name:
            return value
        for child in value.values():
            found = _find_instance_record(child, name)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_instance_record(child, name)
            if found is not None:
                return found
    return None


async def instance_pid(sandbox: SingularitySandbox) -> int:
    completed = await asyncio.to_thread(
        subprocess.run,
        [
            sandbox._resolve_runtime_bin(),
            "instance",
            "list",
            "--json",
            sandbox.name,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        env=sandbox._instance_environment(),
    )
    require(
        completed.returncode == 0,
        f"could not inspect instance {sandbox.name}: {completed.stdout}",
    )
    payload = json.loads(completed.stdout or "{}")
    record = _find_instance_record(payload, sandbox.name)
    require(record is not None, f"instance {sandbox.name} missing from JSON list")
    for key in ("pid", "PID", "Pid"):
        if key in record:
            return int(record[key])
    raise RuntimeError(f"instance record has no PID: {record}")


async def lower_image_ownership(
    sandbox: SingularitySandbox,
    relative_path: str,
) -> str:
    require(sandbox.image_path is not None, "adapted SIF path is absent")
    require(sandbox.container_workdir is not None, "container WORKDIR is absent")
    completed = await asyncio.to_thread(
        subprocess.run,
        [
            sandbox._resolve_runtime_bin(),
            "exec",
            "--cleanenv",
            "--containall",
            "--no-home",
            "--no-mount",
            "home,cwd,bind-paths",
            "--cwd",
            sandbox.container_workdir,
            str(sandbox.image_path),
            "/usr/bin/stat",
            "-c",
            "%u:%g:%a",
            relative_path,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        env=sandbox._runtime_environment(),
    )
    require(
        completed.returncode == 0,
        f"could not inspect lower-image ownership: {completed.stdout}",
    )
    return completed.stdout.strip()


def leaked_instance_processes() -> list[str]:
    leaks: list[str] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except OSError:
            continue
        if "swe-rebench-" in command:
            leaks.append(f"pid={entry.name} cmd={command[:500]}")
    return leaks


async def stop_all(sandboxes: list[SingularitySandbox]) -> None:
    results = await asyncio.gather(
        *(sandbox.stop() for sandbox in sandboxes),
        return_exceptions=True,
    )
    errors = [result for result in results if isinstance(result, BaseException)]
    if errors:
        raise RuntimeError(f"sandbox teardown failed: {errors}")


async def exercise_lifecycle(
    row: dict[str, Any],
) -> tuple[dict[str, Any], Path, str, Path, str]:
    sandboxes = [SingularitySandbox(row["image_name"]) for _ in range(2)]
    cache_path = sandboxes[0]._cached_image_path()
    cache_existed_before = cache_path.exists()
    print(
        "[gate3] concurrent starts exercise lazy pull/publication lock; "
        f"cache_existed_before={cache_existed_before}",
        flush=True,
    )
    start_results = await asyncio.gather(
        *(sandbox.start() for sandbox in sandboxes),
        return_exceptions=True,
    )
    try:
        start_errors = [
            result for result in start_results if isinstance(result, BaseException)
        ]
        require(not start_errors, f"concurrent sandbox startup failed: {start_errors}")
        first, second = sandboxes
        require(first.raw_image_path == cache_path, "first raw cache path changed")
        require(second.raw_image_path == cache_path, "second raw cache path changed")
        adapted_path = first.image_path
        require(adapted_path is not None, "service-adapted SIF path is absent")
        require(
            second.image_path == adapted_path,
            "concurrent sessions did not converge on one adapted SIF",
        )
        require(await first._sif_is_valid(cache_path), "published cache is not a SIF")
        require(
            await first._sif_is_valid(adapted_path),
            "published service-adapted cache is not a SIF",
        )
        require(
            stat.S_IMODE(cache_path.stat().st_mode) & 0o222 == 0,
            "published SIF cache entry is writable",
        )
        image_sha256 = await asyncio.to_thread(sha256_file, cache_path)
        adapted_sha256 = await asyncio.to_thread(sha256_file, adapted_path)
        sif_listing_result = await asyncio.to_thread(
            subprocess.run,
            [first._resolve_runtime_bin(), "sif", "list", str(cache_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
            env=first._runtime_environment(),
        )
        require(
            sif_listing_result.returncode == 0,
            f"Singularity could not list SIF metadata: {sif_listing_result.stdout}",
        )
        sif_listing = sif_listing_result.stdout.strip()
        print(
            f"[gate3] raw_cache_path={cache_path} "
            f"raw_size_bytes={cache_path.stat().st_size} "
            f"raw_sif_sha256={image_sha256}\n"
            f"[gate3] adapted_cache_path={adapted_path} "
            f"adapted_size_bytes={adapted_path.stat().st_size} "
            f"adapted_sif_sha256={adapted_sha256} "
            f"adapter_key={first.service_adapter_key}\n"
            f"[gate3] raw_sif_metadata:\n{sif_listing}",
            flush=True,
        )

        tracked = await checked(
            first,
            "paths=$(git ls-files '*.py') || exit $?; "
            "printf '%s\\n' \"$paths\" | while IFS= read -r path; do "
            "[ -f \"$path\" ] || continue; "
            "printf '%s\\n' \"$path\"; break; done",
        )
        tracked = tracked.splitlines()[0] if tracked else ""
        require(tracked, "task checkout contains no root-owned tracked Python file")
        quoted_tracked = shlex.quote(tracked)
        ownership = await checked(first, f"stat -c '%u:%g:%a' {quoted_tracked}")
        lower_ownership = await lower_image_ownership(first, tracked)
        require(
            lower_ownership.split(":", 1)[0] == "0",
            f"immutable lower-layer file is not root-owned: {lower_ownership}",
        )
        require(
            ownership.split(":", 1)[0] == str(os.getuid()),
            f"writable upper-layer file is not owned by the submitting uid: {ownership}",
        )

        # Append to an existing root-owned lower-layer file, not merely a new
        # file in a writable directory. This is the operation task agents need.
        await checked(
            first,
            f"printf '\\n# {EDIT_MARKER}\\n' >> {quoted_tracked} && "
            f"grep -Fq {shlex.quote(EDIT_MARKER)} {quoted_tracked} && "
            f"printf 'foreground\\n' > {STATE_FILE}",
        )
        await checked(
            first,
            "nohup /bin/sh -c "
            + shlex.quote(
                f"sleep 1; printf 'background\\n' >> {STATE_FILE}"
            )
            + " </dev/null >/tmp/openreward-live-background.log 2>&1 &",
        )
        for _ in range(50):
            background_state = await first.run(
                f"grep -Fq background {STATE_FILE}", timeout=5
            )
            if background_state.return_code == 0:
                break
            await asyncio.sleep(0.1)
        else:
            raise RuntimeError("background process did not update persistent checkout")
        persisted = await checked(
            first,
            f"grep -F {shlex.quote(EDIT_MARKER)} {quoted_tracked}; cat {STATE_FILE}",
        )
        require("background" in persisted, "background edit was not persistent")

        await checked(
            second,
            f"test ! -e {STATE_FILE} && "
            f"! grep -Fq {shlex.quote(EDIT_MARKER)} {quoted_tracked}",
        )
        await checked(second, f"printf 'second-session\\n' > {STATE_FILE}")
        print(
            f"[gate3] root_owned_edit=ok owner_mode={ownership} "
            f"lower_owner_mode={lower_ownership} "
            "background_persistence=ok concurrent_isolation=ok",
            flush=True,
        )

        first_pid_before = await instance_pid(first)
        timeout_result = await first.run("exec sleep 30", timeout=0.5)
        require(timeout_result.return_code == 124, "forced timeout did not return 124")
        first_pid_after = await instance_pid(first)
        require(
            first_pid_after != first_pid_before,
            "timeout recovery did not restart the persistent instance",
        )
        await checked(
            first,
            f"grep -Fq {shlex.quote(EDIT_MARKER)} {quoted_tracked} && "
            f"grep -Fq background {STATE_FILE}",
        )

        second_pid_before = await instance_pid(second)
        cancelled_run = asyncio.create_task(
            second.run("printf entered > /tmp/openreward-cancel-entered; exec sleep 30")
        )
        for _ in range(50):
            entered = await second.run(
                "test -e /tmp/openreward-cancel-entered", timeout=5
            )
            if entered.return_code == 0:
                break
            await asyncio.sleep(0.1)
        else:
            cancelled_run.cancel()
            await asyncio.gather(cancelled_run, return_exceptions=True)
            raise RuntimeError("cancellation target command never entered")
        cancelled_run.cancel()
        try:
            await cancelled_run
        except asyncio.CancelledError:
            pass
        else:
            raise RuntimeError("sandbox run did not preserve asyncio cancellation")
        second_pid_after = await instance_pid(second)
        require(
            second_pid_after != second_pid_before,
            "cancellation recovery did not restart the persistent instance",
        )
        await checked(second, f"grep -Fq second-session {STATE_FILE}")
        await checked(
            second,
            f"! grep -Fq {shlex.quote(EDIT_MARKER)} {quoted_tracked}",
        )
        print(
            "[gate3] timeout_restart=ok "
            f"pid={first_pid_before}->{first_pid_after}; "
            "cancellation_restart=ok "
            f"pid={second_pid_before}->{second_pid_after}",
            flush=True,
        )

        evidence = {
            "cache_existed_before": cache_existed_before,
            "raw_cache_path": str(cache_path),
            "raw_cache_size_bytes": cache_path.stat().st_size,
            "raw_sif_sha256": image_sha256,
            "raw_sif_metadata": sif_listing,
            "adapted_cache_path": str(adapted_path),
            "adapted_cache_size_bytes": adapted_path.stat().st_size,
            "adapted_sif_sha256": adapted_sha256,
            "adapter_key": first.service_adapter_key,
            "adapter_version": first._SERVICE_ADAPTER_VERSION,
            "adapter_provenance_path": str(
                first._service_provenance_path(adapted_path)
            ),
            "tracked_file": tracked,
            "tracked_file_owner_mode": ownership,
            "tracked_file_lower_owner_mode": lower_ownership,
            "timeout_instance_pids": [first_pid_before, first_pid_after],
            "cancellation_instance_pids": [second_pid_before, second_pid_after],
        }
        return evidence, cache_path, image_sha256, adapted_path, adapted_sha256
    finally:
        await stop_all(sandboxes)


async def main_async(args: argparse.Namespace) -> None:
    started = time.monotonic()
    dataset = TaskDataset(args.data_dir)
    row = await asyncio.to_thread(
        dataset.get_row,
        TASK_INDEX,
        columns=EXECUTION_COLUMNS,
    )
    require(
        row["instance_id"] == "pytest-dev__pyfakefs-916",
        f"verified index {TASK_INDEX} changed identity: {row['instance_id']}",
    )
    print(
        f"[gate3] task_index={TASK_INDEX} instance_id={row['instance_id']} "
        f"image={row['image_name']} source={_singularity_pull_uri(row['image_name'])}",
        flush=True,
    )

    (
        lifecycle,
        cache_path,
        initial_sha256,
        adapted_path,
        initial_adapted_sha256,
    ) = await exercise_lifecycle(row)
    session_dirs = list(args.session_root.glob("swe-rebench-*"))
    require(not session_dirs, f"session directories survived lifecycle test: {session_dirs}")
    require(not leaked_instance_processes(), "instance process survived lifecycle teardown")

    print("[gate3] running exact base and gold preflight verifier phases", flush=True)
    verifier = await execute_task(
        dataset,
        TASK_INDEX,
        test_timeout=args.test_timeout,
        diagnostic_chars=12000,
        delete_image=False,
    )
    require(verifier.get("valid") is True, f"preflight verifier failed: {verifier}")
    base = verifier["phases"]["base"]
    gold = verifier["phases"]["gold"]
    require(base.get("valid") is True, f"base preflight invalid: {base}")
    require(
        base.get("observed_fail_to_pass_failure_count", 0) > 0,
        f"base checkout did not demonstrate reward-equivalent zero: {base}",
    )
    require(gold.get("valid") is True, f"gold preflight invalid: {gold}")
    require(
        not gold.get("wrong_fail_to_pass") and not gold.get("wrong_pass_to_pass"),
        f"gold checkout did not demonstrate reward-equivalent one: {gold}",
    )

    final_sha256 = await asyncio.to_thread(sha256_file, cache_path)
    require(
        final_sha256 == initial_sha256,
        "immutable SIF content changed while writable sessions ran",
    )
    final_adapted_sha256 = await asyncio.to_thread(sha256_file, adapted_path)
    require(
        final_adapted_sha256 == initial_adapted_sha256,
        "immutable service-adapted SIF changed while writable sessions ran",
    )
    session_dirs = list(args.session_root.glob("swe-rebench-*"))
    require(not session_dirs, f"session directories survived verifier: {session_dirs}")
    leaks = leaked_instance_processes()
    require(not leaks, f"instance processes survived verifier teardown: {leaks}")

    evidence = {
        "schema_version": 1,
        "gate": "swe-singularity-live",
        "job_id": os.getenv("PBS_JOBID"),
        "task_index": TASK_INDEX,
        "instance_id": row["instance_id"],
        "image": row["image_name"],
        "source_uri": _singularity_pull_uri(row["image_name"]),
        "lifecycle": lifecycle,
        "verifier": {
            "base_valid": base["valid"],
            "base_reward_equivalent": 0,
            "base_observed_f2p_failures": base[
                "observed_fail_to_pass_failure_count"
            ],
            "gold_valid": gold["valid"],
            "gold_reward_equivalent": 1,
            "gold_wrong_f2p": gold["wrong_fail_to_pass_count"],
            "gold_wrong_p2p": gold["wrong_pass_to_pass_count"],
        },
        "raw_cache_sha256_unchanged": True,
        "adapted_cache_sha256_unchanged": True,
        "session_directories_remaining": 0,
        "instance_processes_remaining": 0,
        "duration_seconds": time.monotonic() - started,
        "status": "passed",
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.result.with_name(f".{args.result.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.result)
    print(
        "[gate3] base_reward_equivalent=0 gold_reward_equivalent=1 "
        "sessions_remaining=0 instances_remaining=0",
        flush=True,
    )
    print(f"swe_singularity_live_smoke=ok evidence={args.result}", flush=True)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--data-dir", type=Path, required=True)
    argument_parser.add_argument("--session-root", type=Path, required=True)
    argument_parser.add_argument("--result", type=Path, required=True)
    argument_parser.add_argument("--test-timeout", type=float, default=900)
    return argument_parser


if __name__ == "__main__":
    asyncio.run(main_async(parser().parse_args()))
