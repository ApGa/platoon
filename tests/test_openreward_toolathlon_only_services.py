from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "pbs-scripts/openreward-ta-swe-prealloc-common.sh"
NODE_SERVICES = ROOT / "pbs-scripts/openreward-node-services.sh"

TOOLATHLON_ONLY_BINDINGS = (
    (
        "qwen36-32-v27-rootprop64",
        "32",
        "qwen36",
        "abci-or-ta32-q36-cp2-rootprop-v27",
        "abci-ta32-rec-q36m-cp2-c12-d2-rootprop-v27-t0",
    ),
    (
        "qwen36-32-v27-behavior64",
        "32",
        "qwen36",
        "abci-or-ta32-q36-cp2-behavior-gate-v27",
        "abci-ta32-rec-q36m-cp2-c12-d2-behavior-gate-v27-t0",
    ),
    (
        "qwen36-32-v33-behavior-usage64",
        "32",
        "qwen36",
        "abci-or-ta32-q36-cp2-behavior-usage-gate-v33",
        "abci-ta32-rec-q36m-cp2-c12-d2-behavior-usage-gate-v33-t0",
    ),
    (
        "qwen35-16-v27-behavior64",
        "16",
        "qwen35",
        "abci-or-ta16-q35-9b-dp-behavior-v27",
        "abci-ta16-rec-q359b-dp-c12-d2-behavior-v27-t0",
    ),
    (
        "qwen35-16-v28-cp2-behavior64",
        "16",
        "qwen35",
        "abci-or-ta16-q35-9b-cp2-behavior-v28",
        "abci-ta16-rec-q359b-cp2-c12-d2-behavior-v28-t0",
    ),
    (
        "qwen35-16-v30-tp2cp2-behavior64",
        "16",
        "qwen35",
        "abci-or-ta16-q35-9b-tp2cp2-behavior-v30",
        "abci-ta16-rec-q359b-tp2cp2-c12-d2-behavior-v30-t0",
    ),
)

BEHAVIOR_RESUME_BINDING = (
    "qwen36-32-v31-behavior-gs13",
    "32",
    "qwen36",
    "abci-or-ta32-q36-cp2-behavior-gate-v27",
    "abci-ta32-rec-q36m-cp2-c12-d2-behavior-gate-v27-t0",
)

ROOTPROP_RESUME_BINDING = (
    "qwen36-32-v32-rootprop-gs13",
    "32",
    "qwen36",
    "abci-or-ta32-q36-cp2-rootprop-v27",
    "abci-ta32-rec-q36m-cp2-c12-d2-rootprop-v27-t0",
)


def _marked(source: str, begin: str, end: str) -> str:
    start = source.index(begin) + len(begin)
    return source[start : source.index(end, start)]


def _run_common_authorization(
    binding: str,
    nodes: str,
    model_family: str,
    experiment: str,
    trial: str,
    *,
    environment_set: str = "toolathlon_only",
    incoming_private_binding: str = "",
) -> subprocess.CompletedProcess[str]:
    source = COMMON.read_text(encoding="utf-8")
    authorization = _marked(
        source,
        "# BEGIN OPENREWARD_ENVIRONMENT_SET_AUTHORIZATION",
        "# END OPENREWARD_ENVIRONMENT_SET_AUTHORIZATION",
    )
    harness = f"""set -euo pipefail
die() {{ printf '%s\n' "$*" >&2; exit 2; }}
ENVIRONMENT_SET=$1
FRESH_RUN_BINDING=$2
RESUME_SUCCESSOR_BINDING=
EXPECTED_NNODES=$3
OPENREWARD_MODEL_FAMILY=$4
EXPECTED_EXPERIMENT=$5
EXPECTED_TRIAL=$6
OPENREWARD_ENVIRONMENT_SET_BINDING=$7
EXPECTED_RECURSIVE=1
EXPECTED_PTC=1
REQUIRE_EXISTING_RECOVERY=0
ENVIRONMENT_SET_BINDING=
{authorization}
printf 'environment_set=%s\nbinding=%s\n' \
    "${{OPENREWARD_ENVIRONMENT_SET}}" "${{OPENREWARD_ENVIRONMENT_SET_BINDING}}"
"""
    return subprocess.run(
        [
            "bash",
            "-c",
            harness,
            "authorization-test",
            environment_set,
            binding,
            nodes,
            model_family,
            experiment,
            trial,
            incoming_private_binding,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_common_resume_authorization(
    binding: str,
    nodes: str,
    model_family: str,
    experiment: str,
    trial: str,
) -> subprocess.CompletedProcess[str]:
    source = COMMON.read_text(encoding="utf-8")
    authorization = _marked(
        source,
        "# BEGIN OPENREWARD_ENVIRONMENT_SET_AUTHORIZATION",
        "# END OPENREWARD_ENVIRONMENT_SET_AUTHORIZATION",
    )
    harness = f"""set -euo pipefail
die() {{ printf '%s\n' "$*" >&2; exit 2; }}
ENVIRONMENT_SET=toolathlon_only
FRESH_RUN_BINDING=
RESUME_SUCCESSOR_BINDING=$1
EXPECTED_NNODES=$2
OPENREWARD_MODEL_FAMILY=$3
EXPECTED_EXPERIMENT=$4
EXPECTED_TRIAL=$5
OPENREWARD_ENVIRONMENT_SET_BINDING=
EXPECTED_RECURSIVE=1
EXPECTED_PTC=1
REQUIRE_EXISTING_RECOVERY=1
ENVIRONMENT_SET_BINDING=
{authorization}
printf 'environment_set=%s\nbinding=%s\n' \
    "${{OPENREWARD_ENVIRONMENT_SET}}" "${{OPENREWARD_ENVIRONMENT_SET_BINDING}}"
"""
    return subprocess.run(
        [
            "bash",
            "-c",
            harness,
            "resume-authorization-test",
            binding,
            nodes,
            model_family,
            experiment,
            trial,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_supervisor_authorization(
    binding: str,
    nodes: str,
    model_family: str,
    experiment: str,
    trial: str,
    *,
    environment_set: str = "toolathlon_only",
) -> subprocess.CompletedProcess[str]:
    source = NODE_SERVICES.read_text(encoding="utf-8")
    authorization = _marked(
        source,
        "# BEGIN OPENREWARD_ENVIRONMENT_SET_SUPERVISOR_AUTHORIZATION",
        "# END OPENREWARD_ENVIRONMENT_SET_SUPERVISOR_AUTHORIZATION",
    )
    harness = f"""set -euo pipefail
die() {{ printf '%s\n' "$*" >&2; exit 2; }}
ENVIRONMENT_SET=$1
ENVIRONMENT_SET_BINDING=$2
EXPECTED_NNODES=$3
OPENREWARD_MODEL_FAMILY=$4
EXPECTED_EXPERIMENT=$5
EXPECTED_TRIAL=$6
EXPECTED_RECURSIVE=1
EXPECTED_PTC=1
OPENREWARD_SESSION_URLS_SWE_REBENCH=http://must-be-unset:8084
OPENREWARD_SWE_REBENCH_PORT=8084
{authorization}
[[ ! -v OPENREWARD_SESSION_URLS_SWE_REBENCH ]]
[[ ! -v OPENREWARD_SWE_REBENCH_PORT ]]
printf 'supervisor_environment_set=%s\n' "${{ENVIRONMENT_SET}}"
"""
    return subprocess.run(
        [
            "bash",
            "-c",
            harness,
            "supervisor-authorization-test",
            environment_set,
            binding,
            nodes,
            model_family,
            experiment,
            trial,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_common_guard_authorization(
    binding: str,
    *,
    environment_binding: str | None = None,
) -> subprocess.CompletedProcess[str]:
    source = COMMON.read_text(encoding="utf-8")
    authorization_end = source.index("# END OPENREWARD_ENVIRONMENT_SET_AUTHORIZATION")
    start = source.index(
        'if [[ "${NODE_MEMORY_GUARD_ENABLED}" == 0 ]]; then',
        authorization_end,
    )
    end = source.index(
        '\nexport \\\n    OPENREWARD_NODE_MEMORY_GUARD_ENABLED',
        start,
    )
    authorization = source[start:end]
    bound_environment = binding if environment_binding is None else environment_binding
    harness = f"""set -euo pipefail
die() {{ printf '%s\n' "$*" >&2; exit 2; }}
NODE_MEMORY_GUARD_ENABLED=0
RESUME_SUCCESSOR_BINDING=
FRESH_RUN_BINDING=$1
ENVIRONMENT_SET=toolathlon_only
ENVIRONMENT_SET_BINDING=$2
NODE_MEMORY_GUARD_DISABLED_BINDING=
EXPECTED_NNODES=16
EXPECTED_RECURSIVE=1
EXPECTED_PTC=1
EXPECTED_EXPERIMENT=unused
EXPECTED_TRIAL=unused
{authorization}
printf 'disabled_binding=%s\n' "${{NODE_MEMORY_GUARD_DISABLED_BINDING}}"
"""
    return subprocess.run(
        [
            "bash",
            "-c",
            harness,
            "common-guard-authorization-test",
            binding,
            bound_environment,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_supervisor_guard_authorization(
    binding: str,
    *,
    environment_binding: str | None = None,
) -> subprocess.CompletedProcess[str]:
    source = NODE_SERVICES.read_text(encoding="utf-8")
    authorization_end = source.index(
        "# END OPENREWARD_ENVIRONMENT_SET_SUPERVISOR_AUTHORIZATION"
    )
    start = source.index(
        'if [[ "${NODE_MEMORY_GUARD_ENABLED}" == 0 ]]; then',
        authorization_end,
    )
    end = source.index('\n[[ "${MKFS_EXT3}" == /* ]]', start)
    authorization = source[start:end]
    bound_environment = binding if environment_binding is None else environment_binding
    harness = f"""set -euo pipefail
die() {{ printf '%s\n' "$*" >&2; exit 2; }}
NODE_MEMORY_GUARD_ENABLED=0
NODE_MEMORY_GUARD_DISABLED_BINDING=$1
ENVIRONMENT_SET=toolathlon_only
ENVIRONMENT_SET_BINDING=$2
{authorization}
printf 'supervisor_disabled_binding=%s\n' "${{NODE_MEMORY_GUARD_DISABLED_BINDING}}"
"""
    return subprocess.run(
        [
            "bash",
            "-c",
            harness,
            "supervisor-guard-authorization-test",
            binding,
            bound_environment,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_common_resume_guard_authorization(
    binding: str,
    *,
    environment_binding: str | None = None,
) -> subprocess.CompletedProcess[str]:
    source = COMMON.read_text(encoding="utf-8")
    authorization_end = source.index("# END OPENREWARD_ENVIRONMENT_SET_AUTHORIZATION")
    start = source.index(
        'if [[ "${NODE_MEMORY_GUARD_ENABLED}" == 0 ]]; then',
        authorization_end,
    )
    end = source.index(
        '\nexport \\\n    OPENREWARD_NODE_MEMORY_GUARD_ENABLED',
        start,
    )
    authorization = source[start:end]
    bound_environment = binding if environment_binding is None else environment_binding
    harness = f"""set -euo pipefail
die() {{ printf '%s\n' "$*" >&2; exit 2; }}
NODE_MEMORY_GUARD_ENABLED=0
RESUME_SUCCESSOR_BINDING=$1
FRESH_RUN_BINDING=
ENVIRONMENT_SET=toolathlon_only
ENVIRONMENT_SET_BINDING=$2
NODE_MEMORY_GUARD_DISABLED_BINDING=
EXPECTED_NNODES=32
EXPECTED_RECURSIVE=1
EXPECTED_PTC=1
EXPECTED_EXPERIMENT=abci-or-ta32-q36-cp2-behavior-gate-v27
EXPECTED_TRIAL=abci-ta32-rec-q36m-cp2-c12-d2-behavior-gate-v27-t0
{authorization}
printf 'disabled_binding=%s\n' "${{NODE_MEMORY_GUARD_DISABLED_BINDING}}"
"""
    return subprocess.run(
        [
            "bash",
            "-c",
            harness,
            "common-resume-guard-authorization-test",
            binding,
            bound_environment,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("binding", "nodes", "model_family", "experiment", "trial"),
    TOOLATHLON_ONLY_BINDINGS,
)
def test_toolathlon_only_mode_requires_the_exact_fresh_tuple_on_both_hosts(
    binding: str,
    nodes: str,
    model_family: str,
    experiment: str,
    trial: str,
) -> None:
    common = _run_common_authorization(
        binding, nodes, model_family, experiment, trial
    )
    assert common.returncode == 0, common.stderr
    assert f"binding={binding}" in common.stdout

    supervisor = _run_supervisor_authorization(
        binding, nodes, model_family, experiment, trial
    )
    assert supervisor.returncode == 0, supervisor.stderr

    wrong_trial = _run_common_authorization(
        binding, nodes, model_family, experiment, trial + "-wrong"
    )
    assert wrong_trial.returncode == 2
    wrong_supervisor = _run_supervisor_authorization(
        binding, nodes, model_family, experiment, trial + "-wrong"
    )
    assert wrong_supervisor.returncode == 2


@pytest.mark.parametrize(
    "binding", [binding[0] for binding in TOOLATHLON_ONLY_BINDINGS]
)
def test_toolathlon_only_fresh_bindings_authorize_exact_guard_disablement(
    binding: str,
) -> None:
    common = _run_common_guard_authorization(binding)
    assert common.returncode == 0, common.stderr
    assert f"disabled_binding={binding}" in common.stdout

    supervisor = _run_supervisor_guard_authorization(binding)
    assert supervisor.returncode == 0, supervisor.stderr
    assert f"supervisor_disabled_binding={binding}" in supervisor.stdout

    mismatched_common = _run_common_guard_authorization(
        binding, environment_binding="wrong-binding"
    )
    assert mismatched_common.returncode == 2
    mismatched_supervisor = _run_supervisor_guard_authorization(
        binding, environment_binding="wrong-binding"
    )
    assert mismatched_supervisor.returncode == 2


@pytest.mark.parametrize("resume_binding", (BEHAVIOR_RESUME_BINDING, ROOTPROP_RESUME_BINDING))
def test_toolathlon_only_resume_authorizes_exact_tuple_and_guard(resume_binding) -> None:
    binding, nodes, model_family, experiment, trial = resume_binding
    common = _run_common_resume_authorization(
        binding, nodes, model_family, experiment, trial
    )
    assert common.returncode == 0, common.stderr
    assert f"binding={binding}" in common.stdout

    supervisor = _run_supervisor_authorization(
        binding, nodes, model_family, experiment, trial
    )
    assert supervisor.returncode == 0, supervisor.stderr

    wrong_trial = _run_common_resume_authorization(
        binding, nodes, model_family, experiment, trial + "-wrong"
    )
    assert wrong_trial.returncode == 2

    common_guard = _run_common_resume_guard_authorization(binding)
    assert common_guard.returncode == 0, common_guard.stderr
    assert f"disabled_binding={binding}" in common_guard.stdout
    supervisor_guard = _run_supervisor_guard_authorization(binding)
    assert supervisor_guard.returncode == 0, supervisor_guard.stderr

    mismatched_common_guard = _run_common_resume_guard_authorization(
        binding, environment_binding="wrong-binding"
    )
    assert mismatched_common_guard.returncode == 2
    mismatched_supervisor_guard = _run_supervisor_guard_authorization(
        binding, environment_binding="wrong-binding"
    )
    assert mismatched_supervisor_guard.returncode == 2


def test_environment_set_is_fail_closed_and_mixed_mode_remains_the_default() -> None:
    binding, nodes, model_family, experiment, trial = TOOLATHLON_ONLY_BINDINGS[0]
    unknown = _run_common_authorization(
        binding,
        nodes,
        model_family,
        experiment,
        trial,
        environment_set="toolathlon_and_something_else",
    )
    assert unknown.returncode == 2

    spoofed = _run_common_authorization(
        binding,
        nodes,
        model_family,
        experiment,
        trial,
        incoming_private_binding=binding,
    )
    assert spoofed.returncode == 2

    mixed = _run_common_authorization(
        "", "32", "qwen36", "legacy", "legacy", environment_set="toolathlon_swe"
    )
    assert mixed.returncode == 0, mixed.stderr
    assert "environment_set=toolathlon_swe" in mixed.stdout
    assert "binding=\n" in mixed.stdout


def test_toolathlon_only_health_sweep_never_constructs_a_swe_probe(
    tmp_path: Path,
) -> None:
    source = COMMON.read_text(encoding="utf-8")
    health_functions = _marked(
        source,
        "# BEGIN OPENREWARD_HEALTH_MONITOR_FUNCTIONS",
        "# END OPENREWARD_HEALTH_MONITOR_FUNCTIONS",
    )
    url_log = tmp_path / "urls.log"
    harness = tmp_path / "toolathlon-only-health.sh"
    harness.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
ENVIRONMENT_SET=toolathlon_only
HEALTH_CONNECT_TIMEOUT_SECONDS=3
HEALTH_MAX_TIME_SECONDS=15
TOOLATHLON_WORKER_BASE_PORT=28100
TOOLATHLON_WORKERS=8
TOOLATHLON_WORKER_QUORUM=5
TOOLATHLON_PUBLIC_PROBES=4
TOOLATHLON_PUBLIC_QUORUM=1
OPENREWARD_PORT=8082
JOB_ID_SAFE=toolathlon-only-test
service_pids=($$)
service_hosts=(node001)
URL_LOG=$1
curl() {
    local url=
    while (($#)); do
        case $1 in
            http://*) url=$1 ;;
        esac
        shift
    done
    printf '%s\n' "${url}" >>"${URL_LOG}"
    printf '200\t0.001'
}
"""
        + health_functions
        + "\nservice_pools_healthy 0\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(harness), str(url_log)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    urls = url_log.read_text(encoding="utf-8").splitlines()
    assert len(urls) == 12
    assert all(":8084/" not in url for url in urls)
    assert {url for url in urls if ":2810" in url} == {
        f"http://node001:{port}/health" for port in range(28100, 28108)
    }


def test_toolathlon_only_omits_swe_urls_and_supervisor_work() -> None:
    common = COMMON.read_text(encoding="utf-8")
    services = NODE_SERVICES.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(COMMON)], check=True)
    subprocess.run(["bash", "-n", str(NODE_SERVICES)], check=True)

    assert 'if [[ "${ENVIRONMENT_SET}" == toolathlon_swe ]]; then\n    [[ -f "${SWE_REBENCH_DATA_DIR}/task_index.json" ]]' in common
    assert 'unset OPENREWARD_SWE_REBENCH_PORT OPENREWARD_SESSION_URLS_SWE_REBENCH' in common
    assert '"OPENREWARD_ENVIRONMENT_SET=${ENVIRONMENT_SET}"' in common
    assert '"OPENREWARD_ENVIRONMENT_SET_BINDING=${ENVIRONMENT_SET_BINDING}"' in common

    forward_start = common.index(
        'if [[ "${ENVIRONMENT_SET}" == toolathlon_only ]]; then',
        common.index("export PLATOON_AREAL_PBS_FORWARD_ENV="),
    )
    forward_end = common.index("\nfi", forward_start)
    toolathlon_only_forward = common[forward_start:forward_end]
    assert "OPENREWARD_SESSION_URLS_TOOLATHLON" in toolathlon_only_forward
    assert "OPENREWARD_SESSION_URLS_SWE_REBENCH" not in toolathlon_only_forward

    assert 'unset OPENREWARD_SESSION_URLS_SWE_REBENCH OPENREWARD_SWE_REBENCH_PORT' in services
    assert 'if [[ "${ENVIRONMENT_SET}" == toolathlon_swe ]]; then\n    echo "[node-services] starting SWE-rebench' in services
    assert 'if [[ "${ENVIRONMENT_SET}" == toolathlon_swe ]]; then\n        janitor_log=${LOCAL_ROOT}/node-services-final-swe-cleanup.log' in services
    assert 'local -a child_names=(toolathlon swe-rebench)' in services
    assert 'child_names=(toolathlon)' in services


def test_partial_reward_mode_reaches_toolathlon_before_cleanenv_service_start() -> None:
    common = COMMON.read_text(encoding="utf-8")
    services = NODE_SERVICES.read_text(encoding="utf-8")

    mode_export = "export OPENREWARD_REWARD_MODE=partial"
    service_start = 'echo "[pbs-openreward] starting one Toolathlon'
    assert common.count(mode_export) == 1
    assert common.index(mode_export) < common.index(service_start)

    controller_launch = common[
        common.index(service_start) : common.index(
            "\nHEALTH_CONNECT_TIMEOUT_SECONDS=", common.index(service_start)
        )
    ]
    assert '"OPENREWARD_REWARD_MODE=${OPENREWARD_REWARD_MODE}"' in controller_launch

    assert "REWARD_MODE=${OPENREWARD_REWARD_MODE:-}" in services
    assert '[[ "${REWARD_MODE}" == partial ]]' in services
    container_start = services.index('"${RUNTIME}" exec --cleanenv')
    container_end = services.index(
        "/app/openreward-toolathlon-resilient-entrypoint.sh &", container_start
    )
    container_launch = services[container_start:container_end]
    assert 'OPENREWARD_REWARD_MODE="${REWARD_MODE}"' in container_launch

    for forward_line in re.findall(
        r'^.*PLATOON_AREAL_PBS_FORWARD_ENV="[^"]+"$', common, flags=re.MULTILINE
    ):
        assert "OPENREWARD_REWARD_MODE" in forward_line


@pytest.mark.parametrize(("mode", "expected_status"), (("partial", 0), ("binary", 2), ("", 2)))
def test_toolathlon_service_rejects_nonpartial_reward_mode(
    mode: str, expected_status: int
) -> None:
    services = NODE_SERVICES.read_text(encoding="utf-8")
    start = services.index('[[ "${REWARD_MODE}" == partial ]]')
    end = services.index('\n[[ -x "${TOOLATHLON_ENTRYPOINT}" ]]', start)
    validation = services[start:end]
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'die() {{ printf "%s\\n" "$*" >&2; exit 2; }}\n{validation}',
            "reward-mode-test",
        ],
        env={"PATH": os.environ["PATH"], "REWARD_MODE": mode},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected_status, result.stderr


def _snapshot_validator() -> str:
    source = COMMON.read_text(encoding="utf-8")
    opening = 'if ! "${CONTROLLER_PYTHON_BASE}" -I -c \'\n'
    closing = (
        '\n\' "${PLATOON_MODEL_SNAPSHOT}" "${EXPECTED_MODEL_SHARDS}" '
        '"${OPENREWARD_MODEL_FAMILY}"; then'
    )
    start = source.index(opening) + len(opening)
    return source[start : source.index(closing, start)]


def _write_snapshot(snapshot: Path, prefix: str, count: int) -> None:
    snapshot.mkdir()
    names = [
        f"{prefix}{index:05d}-of-{count:05d}.safetensors"
        for index in range(1, count + 1)
    ]
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    f"tensor.{index}": name for index, name in enumerate(names)
                }
            }
        ),
        encoding="utf-8",
    )
    for name in names:
        (snapshot / name).write_bytes(b"test shard")


def _validate_snapshot(
    snapshot: Path, count: int, family: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-c", _snapshot_validator(), str(snapshot), str(count), family],
        check=False,
        capture_output=True,
        text=True,
    )


def test_snapshot_shard_prefix_is_exactly_conditioned_on_model_family(tmp_path: Path) -> None:
    qwen35 = tmp_path / "qwen35"
    qwen36 = tmp_path / "qwen36"
    arbitrary = tmp_path / "arbitrary"
    _write_snapshot(qwen35, "model.safetensors-", 4)
    _write_snapshot(qwen36, "model-", 3)
    _write_snapshot(arbitrary, "weights-", 2)

    assert _validate_snapshot(qwen35, 4, "qwen35").returncode == 0
    assert _validate_snapshot(qwen36, 3, "qwen36").returncode == 0
    assert _validate_snapshot(qwen36, 3, "qwen38").returncode == 0

    assert _validate_snapshot(qwen35, 4, "qwen36").returncode != 0
    assert _validate_snapshot(qwen36, 3, "qwen35").returncode != 0
    assert _validate_snapshot(arbitrary, 2, "qwen35").returncode != 0
    unsupported = _validate_snapshot(qwen36, 3, "qwen39")
    assert unsupported.returncode != 0
    assert "unsupported model family" in unsupported.stderr
