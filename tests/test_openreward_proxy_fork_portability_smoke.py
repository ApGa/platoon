"""Focused checks for the one-node proxy fork portability gate."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "pbs-scripts" / "openreward_proxy_fork_portability_smoke.py"
PBS = REPO_ROOT / "pbs-scripts" / "openreward-proxy-fork-portability-smoke.pbs"


def _load_driver():
    spec = importlib.util.spec_from_file_location("proxy_fork_portability_smoke_test", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_process(
    root: Path,
    pid: int,
    command: list[str],
    environment: dict[str, str] | None = None,
) -> None:
    process = root / str(pid)
    process.mkdir()
    (process / "cmdline").write_bytes(b"\0".join(item.encode() for item in command) + b"\0")
    values = environment or {}
    (process / "environ").write_bytes(b"\0".join(f"{key}={value}".encode() for key, value in values.items()) + b"\0")


def _safe_proxy_environment() -> dict[str, str]:
    return {
        "HF_HOME": "/cache",
        "OPENAI_API_KEY": "none",
        "OPENAI_BASE_URL": "none",
        "PATH": "/worker/bin",
    }


def test_process_probe_attests_effective_module_worker_python_and_no_credentials(tmp_path):
    driver = _load_driver()
    worker_python = "/groups/cache/worker/bin/python"
    command = [
        worker_python,
        "-m",
        driver.EFFECTIVE_PROXY_MODULE,
        "--experiment-name",
        "proxy-fork-portability",
        "--trial-name",
        "job-42",
        "--role",
        driver.PROXY_ROLE,
    ]
    _fake_process(tmp_path, 101, command, _safe_proxy_environment())

    assert driver.proxy_processes(
        experiment="proxy-fork-portability",
        trial="job-42",
        worker_python=worker_python,
        proc_root=tmp_path,
        own_pid=999,
    ) == [
        {
            "pid": 101,
            "cmdline": command,
            "credential_names_present": [],
            "openai_placeholders": {
                "OPENAI_API_KEY": "none",
                "OPENAI_BASE_URL": "none",
            },
            "controller_canary_absent": True,
        }
    ]


def test_process_probe_rejects_controller_python_or_upstream_module_in_final_argv(tmp_path):
    driver = _load_driver()
    base = [
        "/controller/python",
        "-m",
        driver.EFFECTIVE_PROXY_MODULE,
        driver.REQUESTED_PROXY_MODULE,
        "proxy-fork-portability",
        "job-42",
        driver.PROXY_ROLE,
    ]
    _fake_process(tmp_path, 102, base)

    with pytest.raises(RuntimeError, match="wrong Python"):
        driver.proxy_processes(
            experiment="proxy-fork-portability",
            trial="job-42",
            worker_python="/worker/python",
            proc_root=tmp_path,
            own_pid=999,
        )

    base[0] = "/worker/python"
    (tmp_path / "102" / "cmdline").write_bytes(b"\0".join(item.encode() for item in base) + b"\0")
    with pytest.raises(RuntimeError, match="unpatched upstream"):
        driver.proxy_processes(
            experiment="proxy-fork-portability",
            trial="job-42",
            worker_python="/worker/python",
            proc_root=tmp_path,
            own_pid=999,
        )


@pytest.mark.parametrize(
    "environment_update",
    [
        {"OPENAI_API_KEY": "proxy-fork-smoke-controller-only-sentinel"},
        {"UNRELATED": "prefix-proxy-fork-smoke-controller-only-sentinel-suffix"},
    ],
)
def test_process_probe_rejects_controller_canary_bytes_anywhere(tmp_path, environment_update):
    driver = _load_driver()
    command = [
        "/worker/python",
        "-m",
        driver.EFFECTIVE_PROXY_MODULE,
        "proxy-fork-portability",
        "job-42",
        driver.PROXY_ROLE,
    ]
    environment = _safe_proxy_environment()
    environment.update(environment_update)
    _fake_process(tmp_path, 103, command, environment)

    with pytest.raises(RuntimeError, match="credential canary leaked"):
        driver.proxy_processes(
            experiment="proxy-fork-portability",
            trial="job-42",
            worker_python="/worker/python",
            proc_root=tmp_path,
            own_pid=999,
        )


@pytest.mark.parametrize(
    ("environment_update", "message"),
    [
        ({"OPENAI_API_KEY": "not-none"}, "placeholder environment is not exact"),
        ({"OPENAI_BASE_URL": "not-none"}, "placeholder environment is not exact"),
        ({"WANDB_API_KEY": "not-a-real-secret"}, "credential names leaked"),
        ({"HF_TOKEN": "not-a-real-secret"}, "credential names leaked"),
        ({"AWS_ACCESS_KEY_ID": "not-a-real-secret"}, "credential names leaked"),
    ],
)
def test_process_probe_rejects_wrong_placeholders_or_forbidden_names(tmp_path, environment_update, message):
    driver = _load_driver()
    command = [
        "/worker/python",
        "-m",
        driver.EFFECTIVE_PROXY_MODULE,
        "proxy-fork-portability",
        "job-42",
        driver.PROXY_ROLE,
    ]
    environment = _safe_proxy_environment()
    environment.update(environment_update)
    _fake_process(tmp_path, 105, command, environment)

    with pytest.raises(RuntimeError, match=message) as caught:
        driver.proxy_processes(
            experiment="proxy-fork-portability",
            trial="job-42",
            worker_python="/worker/python",
            proc_root=tmp_path,
            own_pid=999,
        )
    assert "not-a-real-secret" not in str(caught.value)
    assert "not-none" not in str(caught.value)


@pytest.mark.parametrize("environment", [None, {"PATH": "/worker/bin"}])
def test_process_probe_fails_closed_on_unattestable_or_incomplete_environment(tmp_path, environment):
    driver = _load_driver()
    command = [
        "/worker/python",
        "-m",
        driver.EFFECTIVE_PROXY_MODULE,
        "proxy-fork-portability",
        "job-42",
        driver.PROXY_ROLE,
    ]
    _fake_process(tmp_path, 104, command, environment or {})
    if environment is None:
        (tmp_path / "104" / "environ").unlink()
        match = "cannot attest fork-child environment"
    else:
        match = "lacks safe names"

    with pytest.raises(RuntimeError, match=match):
        driver.proxy_processes(
            experiment="proxy-fork-portability",
            trial="job-42",
            worker_python="/worker/python",
            proc_root=tmp_path,
            own_pid=999,
        )


def test_driver_exercises_public_requested_to_effective_contract_and_orderly_cleanup():
    source = DRIVER.read_text()

    assert 'REQUESTED_PROXY_MODULE = "areal.experimental.openai.proxy.proxy_rollout_server"' in source
    assert 'EFFECTIVE_PROXY_MODULE = "platoon.areal_proxy_rollout"' in source
    assert 'INLINE_AGENT_CLASS = "areal.experimental.openai.proxy.online_agent._OnlineAgent"' in source
    assert "agent_cls_path=INLINE_AGENT_CLASS" in source
    assert "command=REQUESTED_PROXY_MODULE" in source
    assert 'os.environ.get("PLATOON_AREAL_PBS_WORKER_PYTHON")' in source
    assert "scheduler.delete_workers(role=None, reverse_order=True)" in source
    assert "engine_created = False" in source
    assert "engine_created = True" in source
    assert "proxy_worker is not None and engine_created" in source
    assert "primary_error.add_note(" in source
    assert "assert_cleanup_complete(" in source
    assert "wait_for_no_trial_processes(" in source
    assert '"requested_proxy_module": REQUESTED_PROXY_MODULE' in source
    assert '"effective_proxy_module": EFFECTIVE_PROXY_MODULE' in source
    assert "hashlib.sha256(Path(__file__).read_bytes()).hexdigest()" in source
    assert '"driver_sha256": actual_driver_sha256' in source
    assert 'parser.add_argument("--driver-sha256", required=True)' in source


def test_engine_config_uses_pinned_nonempty_inline_agent_class(tmp_path, monkeypatch):
    driver = _load_driver()

    class FakeAgentConfig(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class FakeInferenceEngineConfig(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    cli_args = ModuleType("areal.api.cli_args")
    cli_args.AgentConfig = FakeAgentConfig
    cli_args.InferenceEngineConfig = FakeInferenceEngineConfig
    api = ModuleType("areal.api")
    api.cli_args = cli_args
    areal = ModuleType("areal")
    areal.api = api
    monkeypatch.setitem(sys.modules, "areal", areal)
    monkeypatch.setitem(sys.modules, "areal.api", api)
    monkeypatch.setitem(sys.modules, "areal.api.cli_args", cli_args)

    config = driver._engine_config(
        SimpleNamespace(
            model_snapshot=tmp_path,
            experiment="exp",
            trial="trial",
            fileroot=tmp_path,
        )
    )

    assert config.agent.mode == "inline"
    assert config.agent.agent_cls_path == driver.INLINE_AGENT_CLASS
    assert config.agent.admin_api_key


def test_empty_trial_tree_prunes_only_directories_bottom_up(tmp_path):
    driver = _load_driver()
    trial_root = tmp_path / "user" / "experiment" / "trial"
    (trial_root / "workers" / "proxy" / "0").mkdir(parents=True)

    driver.prune_empty_trial_tree(trial_root)

    assert not trial_root.exists()
    assert trial_root.parent.is_dir()


@pytest.mark.parametrize("unsafe_kind", ("file", "symlink"))
def test_trial_tree_pruning_rejects_records_and_symlinks(tmp_path, unsafe_kind):
    driver = _load_driver()
    trial_root = tmp_path / "trial"
    workers = trial_root / "workers"
    workers.mkdir(parents=True)
    unsafe = workers / "record"
    if unsafe_kind == "file":
        unsafe.write_text("value")
        match = "records survived"
    else:
        unsafe.symlink_to(tmp_path / "missing")
        match = "unsafe name-resolution entry"

    with pytest.raises(RuntimeError, match=match):
        driver.prune_empty_trial_tree(trial_root)

    assert trial_root.is_dir()
    assert unsafe.exists() or unsafe.is_symlink()


def test_pbs_wrapper_is_one_node_bounded_cache_only_and_fail_closed():
    source = PBS.read_text()

    assert "#PBS -l select=1:mpiprocs=8" in source
    assert "#PBS -l walltime=00:20:00" in source
    assert "gate2-immutable-runtime" in source
    assert 'source "${ABCI_ENV}"' not in source
    assert "secrets.env" in source
    assert '"${PREPARE_ENV}" resolve' in source
    assert '"${PREPARE_ENV}" build' not in source
    assert "no passed Gate 2 environment matches the exact current source/SIF" in source
    assert "export PLATOON_AREAL_PBS_WORKER_PYTHON=${worker_python}" in source
    assert "export PLATOON_AREAL_PBS_HOST_PYTHON=${controller_python}" in source
    assert "host-only controller Python unexpectedly executed inside the contained SIF" in source
    assert 'PLATOON_AREAL_PBS_CONTAINER_ARGS="--containall --no-home --no-mount home,cwd,bind-paths' in source
    assert "refusing stale smoke state" in source
    assert "refusing stale name-resolution trial" in source
    assert "proxy-fork-smoke-controller-only-sentinel" in source
    assert '--driver-sha256 "${driver_sha}"' in source
    assert '"driver_sha256": driver_sha' in source
    assert 'PLATOON_AREAL_PBS_FORWARD_ENV="HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE"' in source
    assert "pbs_cleanup_barrier.py" in source
    assert 'assert value.get("proxy_health") == {"status": "ok", "initialized": True}' in source
    assert '"OPENAI_API_KEY": "none"' in source
    assert '"OPENAI_BASE_URL": "none"' in source
    assert 'get("controller_canary_absent") is True' in source
    for forbidden in ("qsub", "qdel", "qsig", "wandb login"):
        assert forbidden not in source


def test_pbs_wrapper_shell_syntax_and_private_asset_modes():
    result = subprocess.run(
        ["bash", "-n", str(PBS)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert PBS.stat().st_mode & 0o002 == 0
    assert DRIVER.stat().st_mode & 0o002 == 0
