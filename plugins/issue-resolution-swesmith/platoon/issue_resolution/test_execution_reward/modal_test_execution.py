from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
from tabnanny import verbose
import time
import traceback

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uuid
import modal
import tenacity
from unidiff import PatchSet
from datasets import load_dataset
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
    RUN_EVALUATION_LOG_DIR,
    LOG_INSTANCE,
    DOCKER_WORKDIR,
    DOCKER_PATCH,
    TESTS_TIMEOUT,
)
from swesmith.profiles import registry
from typing import Any

SANDBOX_ENTRYPOINT = "run_evaluation_modal_entrypoint"
LOCAL_MODAL_TEST_EXECUTION_PATH = str(Path(__file__).resolve())
LOCAL_SANDBOX_ENTRYPOINT_PATH = str(
    (Path(__file__).resolve().parent / f"{SANDBOX_ENTRYPOINT}.py").resolve()
)
REMOTE_SANDBOX_ENTRYPOINT_PATH = f"/root/{SANDBOX_ENTRYPOINT}.py"
REMOTE_EVAL_SCRIPT_PATH = "/root/eval.sh"
DEFAULT_MODEL_NAME = "modal"
DEFAULT_TIMEOUT = 60 * 15  # 15 minutes
REMOTE_LOG_ROOT = Path("/tmp/swesmith-modal-eval")
DOCKER_TEST_OUTPUT = "/tmp/test_output.txt"
app = modal.App("swesmith-evaluation")

swesmith_image = (
    modal.Image.debian_slim()
    .pip_install("swesmith", "tenacity", "unidiff", "datasets", "swebench")
    .run_commands(
        "mkdir -p /root/platoon/issue_resolution && touch /root/platoon/__init__.py && touch /root/platoon/issue_resolution/__init__.py"
    )
    .env({"PYTHONPATH": "/root"})
)

@dataclass
class SandboxExecutionResult:
    instance_id: str
    prediction: dict[str, Any]
    timeout: int
    test_output: str
    run_instance_log: str
    patch_diff: str
    errored: bool
    timed_out: bool


@dataclass
class ModalEvaluationResult:
    instance_id: str
    status: str
    resolved: bool
    report: dict[str, Any]
    log_dir: str | None
    timeout: int
    errored: bool
    timed_out: bool

class ModalSandboxRuntime:
    def __init__(
        self,
        image_name: str, 
        verbose: bool = True,
        build_image_from_scratch: bool = False,
        profile = None,
        timeout: int = 60 * 15,
    ):
        if build_image_from_scratch:
            raise NotImplementedError("Building images from scratch is not yet implemented for ModalSandboxRuntime.")

        self.verbose = verbose
        self.image = modal.Image.from_registry(image_name).add_local_file(
            LOCAL_SANDBOX_ENTRYPOINT_PATH,
            REMOTE_SANDBOX_ENTRYPOINT_PATH,
        )
        self.sandbox = self._get_sandbox(timeout=timeout)
        self._stream_tasks = []

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(7),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    )
    def _get_sandbox(self, timeout: int):
        # Sometimes network flakiness causes the image build to fail,
        # so we retry a few times.
        return modal.Sandbox.create(
            image=self.image,
            timeout=timeout, 
            cpu=4
        )
    async def _read_stream(
        self,
        stream: modal.io_streams.StreamReader,
        output_list: list[str],
        merged_output: list[str] | None = None,
    ):
        try:
            async for line in stream:
                output_list.append(line)
                if merged_output is not None:
                    merged_output.append(line)
                if self.verbose:
                    print(line)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error reading stream: {e}")

    async def _read_output(
        self,
        p: modal.container_process.ContainerProcess,
        stdout: list[str],
        stderr: list[str],
        merged_output: list[str],
    ):
        self._stream_tasks = [
            asyncio.create_task(self._read_stream(p.stdout, stdout, merged_output)),
            asyncio.create_task(self._read_stream(p.stderr, stderr, merged_output)),
        ]
        try:
            await asyncio.gather(*self._stream_tasks)
        except asyncio.CancelledError:
            pass    

    def exec(self, command: str) -> tuple[str, int]:
        p = self.sandbox.exec("python", "-m", SANDBOX_ENTRYPOINT, command)
        stdout = []
        stderr = []
        merged_output = []
        try:
            # We separate stdout/stderr because some tests rely on them being separate.
            # We still read stdout/stderr simultaneously to continuously
            # flush both streams and avoid blocking.
            asyncio.run(self._read_output(p, stdout, stderr, merged_output))
        except Exception as e:
            print(f"Error during command execution: {e}")
        p.wait()
        if merged_output:
            return "".join(merged_output), p.returncode
        return "".join(stdout + stderr), p.returncode
        
    def write_file(self, file_path: str, content: str):
        self.sandbox.open(file_path, "w").write(content)

    def close(self):
        if self._stream_tasks:
            try:
                # Forcefully kill remaining streams
                for task in self._stream_tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            asyncio.wait_for(task, timeout=0.1)
                        except asyncio.TimeoutError:
                            pass
                        except Exception:
                            pass

                self.sandbox.terminate()
            except Exception:
                pass
            finally:
                self._stream_tasks = []    
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def read_file(self, file_path: str) -> str:
        return self.sandbox.open(file_path, "r").read()

@app.function(
    image=(
        swesmith_image
        .add_local_file(
            LOCAL_MODAL_TEST_EXECUTION_PATH,
            "/root/modal_test_execution.py",
        )
        .add_local_file(
            LOCAL_MODAL_TEST_EXECUTION_PATH,
            "/root/platoon/issue_resolution/modal_test_execution.py",
        )
        .add_local_file(
            LOCAL_SANDBOX_ENTRYPOINT_PATH,
            REMOTE_SANDBOX_ENTRYPOINT_PATH,
        )
    ),
    timeout=120 * 60,  # Much larger than default timeout to account for image build time
    include_source=False,
)
def run_instance_modal(
    prediction: dict[str, Any] | str,
    instance: dict[str, Any],
    run_id: str,
    f2p_only: bool = False,
    is_gold: bool = False,
    timeout: int | None = None,
    verbose: bool = False,
    build_image_from_scratch: bool = False,
) -> ModalEvaluationResult:
    runner: ModalSandboxRuntime | None = None
    log_lines: list[str] = []
    instance_id = instance[KEY_INSTANCE_ID]
    prediction = _normalize_prediction(prediction, instance_id)
    profile = registry.get_from_inst(instance)
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    commit = prediction[KEY_INSTANCE_ID]
    patch_diff = prediction[KEY_PREDICTION] or ""
    changed_files = ""
    if patch_diff.strip():
        changed_files = " ".join(chunk.path for chunk in PatchSet(patch_diff))
    image_name: str = profile.image_name
    f2p_files, p2p_files = profile.get_test_files(instance)
    test_files = " ".join(f2p_files + p2p_files)
    test_command, _ = profile.get_test_cmd(instance, f2p_only=f2p_only)
    eval_script = _build_eval_script(test_command)
    apply_patch_commands = _build_apply_patch_commands(is_gold)

    def log(message: str):
        log_lines.append(f"{message}\n")
        if verbose:
            print(message)

    def finalize(
        *,
        test_output: str = "",
        errored: bool = False,
        timed_out: bool = False,
    ) -> ModalEvaluationResult:
        raw_result = SandboxExecutionResult(
            instance_id=instance_id,
            prediction=prediction,
            timeout=timeout,
            test_output=test_output,
            run_instance_log="".join(log_lines),
            patch_diff=patch_diff,
            errored=errored,
            timed_out=timed_out,
        )
        return _write_result_artifacts(
            instance=instance,
            run_id=run_id,
            raw_result=raw_result,
            f2p_only=f2p_only,
        )

    try:
        runner = ModalSandboxRuntime(
            image_name=image_name,
            verbose=verbose,
            build_image_from_scratch=build_image_from_scratch,
            profile=profile,
            timeout=timeout,
        )

        fetch_output, fetch_code = runner.exec(f"cd {DOCKER_WORKDIR} && git fetch")
        # assert fetch_code == 0, f"Git fetch failed: {fetch_output}"

        checkout_output, checkout_code = runner.exec(f"cd {DOCKER_WORKDIR} && git checkout {commit}")
        if checkout_code != 0:
            log(f"git checkout {commit} exit={checkout_code}\n{checkout_output}")
            return finalize(errored=True)
    
        
        bug_checkout_output, bug_checkout_code = runner.exec(
            f"cd {DOCKER_WORKDIR} && git checkout HEAD~1"
        )
        if bug_checkout_code != 0:
            log(f"git checkout HEAD~1 exit={bug_checkout_code}\n{bug_checkout_output}")
            return finalize(errored=True)

        runner.write_file(DOCKER_PATCH, patch_diff)
        if patch_diff.strip():
            if changed_files:
                reset_output, reset_code = runner.exec(f"cd {DOCKER_WORKDIR} && git checkout -- {changed_files}")

            apply_succeeded = False
            for command in apply_patch_commands:
                apply_output, apply_code = runner.exec(f"cd {DOCKER_WORKDIR} && {command}")
                if apply_code == 0:
                    apply_succeeded = True
                    log(f"Patch applied successfully with command: '{command}'")
                    break
                else:
                    log(f"Patch apply command '{command}' failed with exit code {apply_code}:\n{apply_output}")
            if not apply_succeeded:
                log(f"Failed to apply patch with commands: {apply_patch_commands}\nLast output: {apply_output}")
                return finalize(errored=True)

            if test_files:
                revert_output, revert_code = runner.exec(
                    f"cd {DOCKER_WORKDIR} && git checkout -- {test_files}"
                )
                # assert revert_code == 0, f"Git checkout for test files failed: {revert_output}"
            
        runner.write_file(REMOTE_EVAL_SCRIPT_PATH, eval_script)
        run_command: str = f"cd {DOCKER_WORKDIR} && python3 -c 'import sys; sys.setrecursionlimit(10000)'"
        run_command += f" && timeout {timeout}s /bin/bash {REMOTE_EVAL_SCRIPT_PATH}"
        _, returncode = runner.exec(
            f"timeout {timeout}s /bin/bash {REMOTE_EVAL_SCRIPT_PATH} > {DOCKER_TEST_OUTPUT} 2>&1"
        )
        test_output = runner.read_file(DOCKER_TEST_OUTPUT)
        log(f"Test command output: {test_output}")
        timed_out = returncode == 124
        return finalize(test_output=test_output, timed_out=timed_out)
    except modal.exception.SandboxTimeoutError:
        log(f"Evaluation timed out after {timeout} seconds")
        return finalize(timed_out=True)
    except Exception:
        log(traceback.format_exc())
        return finalize(errored=True)
    finally:
        if runner is not None:
            runner.close()

def _normalize_prediction(prediction: dict[str, Any] | str, instance_id: str) -> dict[str, Any]:
    if isinstance(prediction, str):
        prediction = {
            KEY_INSTANCE_ID: instance_id,
            KEY_PREDICTION: prediction,
            KEY_MODEL: DEFAULT_MODEL_NAME,
        }
    else:
        prediction = dict(prediction)

    prediction.setdefault(KEY_INSTANCE_ID, instance_id)
    prediction.setdefault(KEY_PREDICTION, "")
    prediction.setdefault(KEY_MODEL, DEFAULT_MODEL_NAME)
    return prediction

def _build_eval_script(test_command: str) -> str:
    from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START
    return (
        "#!/bin/bash\n"
        "set -uxo pipefail\n"
        f"cd {DOCKER_WORKDIR}\n"
        f": '{TEST_OUTPUT_START}'\n"
        f"{test_command}\n"
        f": '{TEST_OUTPUT_END}'\n"
    )

def _build_apply_patch_commands(is_gold: bool) -> list[str]:
    from swesmith.constants import GIT_APPLY_CMDS

    patch_arg = f"--reverse {DOCKER_PATCH}" if is_gold else DOCKER_PATCH
    return [f"{command} {patch_arg}" for command in GIT_APPLY_CMDS]

def _write_result_artifacts(
    instance: dict[str, Any],
    run_id: str,
    raw_result: SandboxExecutionResult,
    f2p_only: bool = False,
) -> ModalEvaluationResult:
    from swesmith.constants import KEY_TIMED_OUT
    from swesmith.harness.grading import get_eval_report

    log_dir = REMOTE_LOG_ROOT / run_id / raw_result.instance_id
    log_dir.mkdir(parents=True, exist_ok=True)

    (log_dir / LOG_INSTANCE).write_text(raw_result.run_instance_log)
    (log_dir / "patch.diff").write_text(raw_result.patch_diff)

    if raw_result.test_output or raw_result.timed_out:
        test_output = raw_result.test_output
        if raw_result.timed_out:
            timeout_error = f"{TESTS_TIMEOUT}: {raw_result.timeout} seconds exceeded"
            test_output = (
                f"{test_output}\n\n{timeout_error}"
            )
        (log_dir / LOG_TEST_OUTPUT).write_text(test_output)
    
    report_path = log_dir / LOG_REPORT
    model_name = raw_result.prediction.get(KEY_MODEL, DEFAULT_MODEL_NAME)
    patch_exists = raw_result.prediction.get(KEY_PREDICTION) is not None

    if raw_result.timed_out:
        report = {KEY_TIMED_OUT: True, "timeout": raw_result.timeout}
        report = {
            KEY_TIMED_OUT: True,
            "timeout": raw_result.timeout,
            "patch_exists": patch_exists,
            "resolved": False,
            KEY_MODEL: model_name,
        }
        report_path.write_text(json.dumps(report, indent=4))
        return ModalEvaluationResult(
            instance_id=raw_result.instance_id,
            status="timeout",
            resolved=False,
            report=report,
            log_dir=str(log_dir),
            timeout=raw_result.timeout,
            errored=False,
            timed_out=True,
        )
    
    test_log_path = log_dir / LOG_TEST_OUTPUT
    if raw_result.errored or not test_log_path.exists():
        report = {
            "patch_exists": patch_exists,
            "resolved": False,
            "errored": True,
            KEY_MODEL: model_name,
        }
        report_path.write_text(json.dumps(report, indent=4))
        return ModalEvaluationResult(
            instance_id=raw_result.instance_id,
            status="error",
            resolved=False,
            report=report,
            log_dir=str(log_dir),
            timeout=raw_result.timeout,
            errored=True,
            timed_out=False,
        )

    report = get_eval_report(raw_result.prediction, instance, test_log_path, f2p_only=f2p_only)
    report[KEY_MODEL] = model_name
    report_path.write_text(json.dumps(report, indent=4))
    return ModalEvaluationResult(
        instance_id=raw_result.instance_id,
        status=f"completed: {raw_result.test_output}",
        resolved=bool(report.get("resolved", False)),
        report=report,
        log_dir=str(log_dir),
        timeout=raw_result.timeout,
        errored=False,
        timed_out=False,
    )

def validate_modal_credentials():
    has_env_credentials = bool(
        os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")
    )
    has_modal_config = (Path.home() / ".modal.toml").exists()
    if has_env_credentials or has_modal_config:
        return
    raise RuntimeError(
        "Modal credentials not found. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET, "
        "or run `modal setup`."
    )

def sample_testing_code():
    # validate_modal_credentials()
    dataset = load_dataset("SWE-bench/SWE-smith-py", split="train")
    repo_seen = set()
    instances = []
    repo_list = [
        "swesmith/jaraco__inflect.c079a96a",
        "swesmith/paramiko__paramiko.23f92003",
        "swesmith/lepture__mistune.bf54ef67",
        "swesmith/python__mypy.e93f06ce",
        "swesmith/HIPS__autograd.ac044f0d",
        "swesmith/gruns__furl.da386f68",
        "swesmith/datamade__usaddress.a42a8f0c",
        "swesmith/life4__textdistance.c3aca916",
        "swesmith/marshmallow-code__marshmallow.9716fc62",
        "swesmith/pyca__pyopenssl.04766a49",
        "swesmith/gweis__isodate.17cb25eb",
        "swesmith/pydantic__pydantic.acb0f10f"
        "swesmith/Project-MONAI__MONAI.a09c1f08",
        "swesmith/mozillazg__python-pinyin.e42dede5"
    ]
    tmp_list = []
    for instance in dataset:
        tmp_list.append(instance)
    import random
    random.shuffle(tmp_list)
        
    for instance in tmp_list:
        if len(instance["problem_statement"].strip()) > 0:
            if instance["repo"] not in repo_seen: #and instance["repo"] in repo_list:
                repo_seen.add(instance["repo"])
                instances.append(instance)
    # instances = instances[-2:]
    start_time = time.time()
    cnt = 0
    # print(len(repo_seen), len(instances))
    # exit()
    print(len(instances))
    # exit()
    with modal.enable_output():
        with app.run():
            results = run_instance_modal.starmap(
                [
                    (
                        {
                            KEY_INSTANCE_ID: instance[KEY_INSTANCE_ID],
                            KEY_PREDICTION: instance["patch"],
                            KEY_MODEL: "test_model",
                        },
                        instance,
                        "temp",
                        False,
                        True,
                        5*60,
                        True,
                        False
                    )
                    for instance in instances
                ],
                return_exceptions=True,
            )
            err = 0
            fail = 0
            problematic_repos = []
            for res in results:
                try:
                    if not res.resolved:
                        print(f"Test failed for instance_id: {res.instance_id}")
                        fail += 1
                        with open(f"error_log_{fail}.json", "w") as f:
                            from dataclasses import asdict
                            repo_name = "".join(res.instance_id.split(".")[:-1])
                            problematic_repos.append(repo_name)
                            json.dump(asdict(res), f, indent=4)
                    else:
                        print(f"Test passed for instance_id: {res.instance_id}")
                except Exception as e:
                    print(f"Error during evaluation: {res.instance_id}, error: {e}")
                    err += 1
            with open("problematic_repos6.txt", "w") as f:
                for repo in problematic_repos:
                    f.write(f"{repo}\n")
    print(f"Total time for testing: {time.time() - start_time} seconds")