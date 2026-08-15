from __future__ import annotations

import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE = REPO_ROOT / "pbs-scripts" / "qwen35-routed-experts-sglang-smoke.pbs"


def test_smoke_uses_exact_immutable_runtime_and_production_server_args() -> None:
    subprocess.run(["bash", "-n", str(SMOKE)], check=True)
    text = SMOKE.read_text()

    assert SMOKE.stat().st_mode & 0o111 == 0o111
    assert "#PBS -l select=1:mpiprocs=8" in text
    assert "#PBS -l walltime=01:00:00" in text
    assert ("WORKER_ENV_KEY=18c5656107b2172786057c189f5c928859f3569abce0eaa30adb52b3e98f2445") in text
    assert "SOURCE_SHA256=4038f0718a9f0f88142c1d688708b62dcd9abd7a97ebeae8fd196c506f4efaf4" in text
    assert "SIF_SHA256=53c770558ad86e4a8ec4d2532cd3ba1c60a10bd75fff6a51049dfb411b002b12" in text
    assert "CONFIG_16=${WORKER_ENV}/lib/python3.12/site-packages/" in text
    assert "16node_nonrecursive_h200_cp1_c12_ta20_v5.yaml" in text
    assert "CONFIG_32=${WORKER_ENV}/lib/python3.12/site-packages/" in text
    assert "32node_recursive_h200_cp1_c24_ta20_v5.yaml" in text
    assert 'raw_config, _ = parse_cli_args(["--config", config_path])' in text
    assert "raw_config.sglang = to_structured_cfg(raw_config.sglang, SGLangConfig)" in text
    assert "sglang_config = OmegaConf.to_object(raw_config.sglang)" in text
    assert "load_expr_config" not in text
    assert "save_config" not in text
    assert "name_resolve.reconfigure" not in text
    assert "apply_controller_safe_patches" not in text
    assert "patches_module._patch_areal_sglang_routed_experts_launcher(SGLangConfig)" in text
    assert "command = SGLangConfig.build_cmd(" in text
    assert "tp_size=8" in text
    assert 'command[1:3] != ["-m", "platoon.sglang_server"]' in text
    assert '"--enable-return-routed-experts" not in command' in text
    assert "float(sglang_config.mem_fraction_static) != float(expected_mem)" in text
    assert 'sglang_config.log_level = "info"' in text
    assert 'NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}"' in text
    assert "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" in text
    assert '[[ "${#GPU_UUIDS[@]}" -eq 8 ]]' in text
    assert "--cleanenv --containall --no-home" in text
    assert "--no-mount home,cwd,bind-paths" in text
    assert "PLATOON_QWEN35_GDN_CP=" not in text
    assert "PLATOON_QWEN35_GDN_PACKED=" not in text
    assert "PLATOON_QWEN35_GDN_CP_CONV_BACKEND=fla" in text


def test_smoke_prefers_allocator_fix_before_disabling_custom_all_reduce() -> None:
    text = SMOKE.read_text()
    first_call = 'run_variant allocator-false-custom-ar-mem08 "${CONFIG_16}" 0.8 "${PORT_A_08}" 0'
    second_allocator_call = 'run_variant allocator-false-custom-ar-mem07 "${CONFIG_32}" 0.7 "${PORT_A_07}" 0'
    fallback_call = 'run_variant allocator-false-no-custom-ar-mem08 "${CONFIG_16}" 0.8 "${PORT_B_08}" 1'
    second_fallback_call = 'run_variant allocator-false-no-custom-ar-mem07 "${CONFIG_32}" 0.7 "${PORT_B_07}" 1'

    assert text.index(first_call) < text.index(second_allocator_call) < text.index(fallback_call)
    assert text.index(fallback_call) < text.index(second_fallback_call)
    assert "readonly ALLOCATOR_CONF=expandable_segments:False" in text
    assert "PYTORCH_CUDA_ALLOC_CONF=${ALLOCATOR_CONF}" in text
    assert "-u PYTORCH_CUDA_ALLOC_CONF" in text
    assert 'sglang_config.disable_custom_all_reduce = disable_custom_ar == "1"' in text
    assert '("--disable-custom-all-reduce" in command) != (disable_custom_ar == "1")' in text
    assert "TORCH_NCCL_TRACE_BUFFER_SIZE=4096" in text
    assert "TORCH_NCCL_DUMP_ON_TIMEOUT=1" in text
    assert "TORCH_NCCL_DESYNC_DEBUG=1" in text

    variant_body = re.search(r"^run_variant\(\) \{(?P<body>.*?)^\}", text, re.MULTILINE | re.DOTALL)
    assert variant_body is not None
    assert 'kill -0 "${server_pid}"' in variant_body.group("body")
    assert "PLATOON_SMOKE_CUSTOM_AR_REGISTERED" in variant_body.group("body")
    assert "ranks != set(range(8))" in variant_body.group("body")
    assert "CustomAllreduce" in variant_body.group("body")
    assert "deadline=$((SECONDS + 900))" in variant_body.group("body")
    assert "SECONDS + 1200" not in text

    fallback_gate = re.search(r"^exact_ipc_failure\(\) \{(?P<body>.*?)^\}", text, re.MULTILINE | re.DOTALL)
    assert fallback_gate is not None
    for signature in (
        "get_graph_buffer_ipc_meta",
        "RuntimeError: invalid argument",
        "/sgl-kernel/csrc/allreduce/custom_all_reduce.cuh:350",
    ):
        assert signature in fallback_gate.group("body")
    assert text.index('if ! exact_ipc_failure "${a_failure_log}"') < text.index(fallback_call)
    assert "fallback forbidden" in text
    assert 'if has_known_ipc_marker "${server_log}"' in text


def test_smoke_requires_real_routed_expert_response_and_durable_results() -> None:
    text = SMOKE.read_text()

    assert '"input_ids": [9707, 825, 2805]' in text
    assert '"return_routed_experts": True' in text
    assert 'endpoint": "/generate"' in text
    assert "base64.b64decode(encoded, validate=True)" in text
    assert 'struct.iter_unpack("=i", raw)' in text
    assert "route_rows = prompt_tokens + completion_tokens - 1" in text
    assert "route_width = 320" in text
    assert "len(route_ids) != expected_int32_count" in text
    assert '"routed_experts_rows": route_rows' in text
    assert '"routed_experts_width": route_width' in text
    assert 'summary["prompt_tokens"] <= 0 or summary["completion_tokens"] <= 0' in text
    assert "grep -Fq 'Expanded SGLang routed-expert capture capacity'" in text
    assert '"http://${HOST_IP}:${port}/v1/models"' in text
    assert '"http://${HOST_IP}:${port}/health"' in text
    assert "readonly RESULT=${RESULT_ROOT}/RESULT.json" in text
    assert "${mode}.server.log" in text
    assert "${mode}.result.json" in text
    assert "${mode}.request.json" in text
    assert "${mode}.command.txt" in text
    assert "${mode}.command.json" in text
    assert "printf '%q ' \"${launch_command[@]}\"" in text
    assert "os.chmod(temporary, 0o600)" in text
    assert "temporary.replace(path)" in text
    publish_index = text.index('publish_json "${RESULT}" PASS "${mode}"')
    pass_index = text.index('echo "qwen35_routed_experts_sglang_smoke=PASS selected_mode=${mode}"')
    assert publish_index < pass_index
    assert "finalize_pass allocator-false-custom-ar 0" in text
    assert "qsub" not in text.lower()


def test_smoke_scrubs_credentials_and_checks_teardown_between_variants() -> None:
    text = SMOKE.read_text()

    for secret in (
        "WANDB_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "LITELLM_API_KEY",
        "LITELLM_BASE_URL",
        "RUBRIC_DEFAULT_LLM",
        "HF_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        assert secret in text
    assert 'unset "${secret_name}" "SINGULARITYENV_${secret_name}"' in text
    assert text.count("assert_no_gpu_processes") >= 5
    assert "preexisting_gpu_processes" in text
    assert '"preexisting_gpu_processes": preexisting.splitlines()' in text
    assert '[[ ! -e "${RESULT_ROOT}" && ! -L "${RESULT_ROOT}" ]]' in text
    assert 'mkdir -p "${LOCAL_ROOT}/tmp/home"' in text
    assert '"${LOCAL_ROOT}/tmp/home" "${RESULT_ROOT}"' in text
    assert "server_pgid=${server_pid}" in text
    assert 'kill -TERM -- "-${server_pgid}"' in text
    assert 'kill -KILL -- "-${server_pgid}"' in text
    assert 'kill -0 -- "-${server_pgid}"' in text
    assert "server process group survived" in text
    assert 'gpu_processes=$("${NVIDIA_SMI}"' in text
    assert 'gpu_processes=$("${NVIDIA_SMI}"' in text
    assert "2>/dev/null) || return 1" in text
    assert "trap '' INT TERM" in text
    assert "trap - EXIT" in text


def test_smoke_records_runtime_versions_allocator_and_command_provenance() -> None:
    text = SMOKE.read_text()

    for distribution in (
        "areal",
        "platoon",
        "platoon-openreward",
        "sglang",
        "sglang-kernel",
        "torch",
    ):
        assert f'"{distribution}"' in text
    assert '"torch_cuda": torch.version.cuda' in text
    assert '"python": platform.python_version()' in text
    assert '"allocator_environment": {"PYTORCH_CUDA_ALLOC_CONF":' in text
    assert '"PYTORCH_CUDA_ALLOC_CONF"] != "expandable_segments:False"' in text
    assert '"nccl_environment": {' in text
    assert '"command": command' in text
    assert 'record["command_record"] = command_record' in text
    assert '"custom_ar_registration_count"' in text
    assert '"custom_ar_registration_ranks"' in text
    assert '"known_ipc_markers"' in text
    assert "CUDA_DEVICE_MAX_CONNECTIONS=1" in text
    assert "PYTHONDONTWRITEBYTECODE=1" in text
    assert "expected_packages = {" in text
    assert "sglang_config = OmegaConf.to_object(raw_config.sglang)" in text
    assert '"sglang": "0.5.10.post1"' in text
    assert '"sglang-kernel": "0.4.1"' in text
    assert '"torch": "2.9.1+cu129"' in text


def test_smoke_attests_real_custom_ar_registration_after_return_in_every_rank() -> None:
    text = SMOKE.read_text()

    assert "readonly SMOKE_HOOK_ROOT=${LOCAL_ROOT}/tmp/smoke-hook" in text
    assert "PLATOON_SGLANG_SMOKE_ATTEST_CUSTOM_AR=1" in text
    assert "PYTHONPATH=/tmp/smoke-hook" in text
    assert "_original_register_graph_buffers = CustomAllreduce.register_graph_buffers" in text
    assert "result = _original_register_graph_buffers(self)" in text
    assert "CustomAllreduce.register_graph_buffers = _attested_register_graph_buffers" in text
    assert 'f"pid={os.getpid()} rank={self.rank} world_size={self.world_size}' in text
    assert "os.O_EXCL | os.O_NOFOLLOW" in text
    hook = re.search(r"source = r'''(?P<program>.*?)'''", text, re.DOTALL)
    assert hook is not None
    hook_text = hook.group("program")
    assert hook_text.index("result = _original_register_graph_buffers(self)") < hook_text.index(
        '"PLATOON_SMOKE_CUSTOM_AR_REGISTERED "'
    )


def test_smoke_custom_ar_hook_emits_only_after_original_returns(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    text = SMOKE.read_text()
    hook = re.search(r"source = r'''(?P<program>.*?)'''", text, re.DOTALL)
    assert hook is not None

    module_names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.distributed",
        "sglang.srt.distributed.device_communicators",
    )
    for module_name in module_names:
        module = types.ModuleType(module_name)
        module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, module)

    target_name = "sglang.srt.distributed.device_communicators.custom_all_reduce"
    target = types.ModuleType(target_name)

    class FakeCustomAllreduce:
        rank = 3
        world_size = 8
        fail = False

        def register_graph_buffers(self) -> str:
            if self.fail:
                raise RuntimeError("registration failed")
            return "registered"

    target.CustomAllreduce = FakeCustomAllreduce  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, target_name, target)
    monkeypatch.setenv("PLATOON_SGLANG_SMOKE_ATTEST_CUSTOM_AR", "1")
    exec(compile(hook.group("program"), f"{SMOKE}:sitecustomize", "exec"), {})

    instance = FakeCustomAllreduce()
    assert instance.register_graph_buffers() == "registered"
    assert re.fullmatch(
        r"PLATOON_SMOKE_CUSTOM_AR_REGISTERED pid=[1-9][0-9]* rank=3 world_size=8\n",
        capfd.readouterr().err,
    )
    instance.fail = True
    with pytest.raises(RuntimeError, match="registration failed"):
        instance.register_graph_buffers()
    assert capfd.readouterr().err == ""


def test_smoke_requires_exact_h200_uuid_mapping_inside_and_outside_container() -> None:
    text = SMOKE.read_text()

    assert "readonly EXPECTED_GPU_NAME='NVIDIA H200'" in text
    assert '[[ "${#GPU_NAMES[@]}" -eq 8 ]]' in text
    assert 'sort -u | wc -l)" -eq 8' in text
    assert "readonly NVIDIA_VISIBLE_DEVICES=$(IFS=,;" in text
    assert "inner_gpu_inventory=${RESULT_ROOT}/inner-gpus.csv" in text
    assert "/usr/bin/nvidia-smi --query-gpu=name,uuid" in text
    assert "set(uuids) != set(expected)" in text
    assert "len(set(uuids)) != 8" in text
    assert "strict=True" not in text
    assert "malformed inner GPU inventory rows" in text
    assert '--bind "${PROJECT_ROOT}:${PROJECT_ROOT}:ro"' in text
    assert '--bind "${RESULT_ROOT}:${RESULT_ROOT}:rw"' in text


def test_all_embedded_python_programs_compile() -> None:
    text = SMOKE.read_text()
    programs = re.findall(r"<<'PY'\n(?P<program>.*?)\nPY$", text, re.MULTILINE | re.DOTALL)

    assert len(programs) == 9
    for program in programs:
        compile(program, str(SMOKE), "exec")

    hook = re.search(r"source = r'''(?P<program>.*?)'''", text, re.DOTALL)
    assert hook is not None
    compile(hook.group("program"), f"{SMOKE}:sitecustomize", "exec")
