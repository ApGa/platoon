import logging
import sys
import uuid
from pathlib import Path
import modal

from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
logger = logging.getLogger(__name__)
from dataclasses import asdict

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    # Modal deserializes the remote return value by importing the defining module
    # by name, which in this deployment is `modal_test_execution`.
    sys.path.insert(0, _THIS_DIR)

def extract_tests_status(report: object) -> dict | None:
    if not isinstance(report, dict):
        return None
    value = report.get("tests_status")
    if isinstance(value, dict):
        return value
    return None

def compute_composite_reward(binary_reward: float, f2p_pass_fraction: float, p2p_fail_fraction: float) -> float:
    return (0.5 * binary_reward) + (0.5 * ((0.8 * f2p_pass_fraction + 0.2 * (1.0 - p2p_fail_fraction))))

def count_items(value: object) -> int:
    if isinstance(value, list):
        return len(value)
    return 0

# Test execution reward: run evaluation of this patch on Modal
async def compute_test_execution_reward(model_patch: str, instance: dict):
    # empty model patch is guaranteed to fail, so we can skip evaluation on Modal
    binary_reward = 0.0
    tests_status = None
    f2p_pass_fraction = 0.0
    p2p_fail_fraction = 0.0

    if len(model_patch.strip()) == 0:
        composite_reward = compute_composite_reward(binary_reward, f2p_pass_fraction, p2p_fail_fraction)
        return binary_reward, {"error": "Empty model patch ==> guaranteed to not resolve issues.", "binary_reward": binary_reward, "f2p_pass_fraction": f2p_pass_fraction, "p2p_fail_fraction": p2p_fail_fraction, "composite_reward": composite_reward}

    # Run tests on modal
    try:
        run_id = f"rl-{uuid.uuid4().hex}"
        with modal.enable_output():
            modal_fn = modal.Function.from_name("swesmith-evaluation", "run_instance_modal")
            res = await modal_fn.remote.aio(
                prediction={
                    KEY_INSTANCE_ID: instance[KEY_INSTANCE_ID],
                    KEY_PREDICTION: model_patch,
                    KEY_MODEL: "test_model",
                },
                instance=instance,
                run_id=run_id,
                f2p_only=False,
                is_gold=False,
                timeout=5*60,
                verbose=False,
                build_image_from_scratch=False,
            )
        info = {"model_patch": model_patch, "evaluation_logs": asdict(res)}
        try:
            binary_reward = 1.0 if res.resolved else 0.0
            info = {"model_patch": model_patch, "evaluation_logs": asdict(res)}
        except Exception as e:
            binary_reward = 0.0
            info = {"model_patch": model_patch, "evaluation_logs": str(e)}
    except Exception as e:
        binary_reward = 0.0
        info = {"model_patch": model_patch, "evaluation_logs": str(e)}
    except:
        binary_reward = 0.0
        info = {"model_patch": model_patch, "evaluation_logs": "Failed to evaluate patch on Modal due to an unknown error."}

    try:
        tests_status = extract_tests_status(res.report)
    except Exception as e:
        tests_status = None

    if tests_status is not None:
        fail_to_pass = tests_status.get("FAIL_TO_PASS") or {}
        f2p_success = count_items(fail_to_pass.get("success"))
        f2p_failure = count_items(fail_to_pass.get("failure"))
        f2p_total = f2p_success + f2p_failure
        f2p_pass_fraction = f2p_success / f2p_total if f2p_total > 0 else 0.0

        pass_to_pass = tests_status.get("PASS_TO_PASS") or {}
        p2p_success = count_items(pass_to_pass.get("success"))
        p2p_failure = count_items(pass_to_pass.get("failure"))
        p2p_total = p2p_success + p2p_failure
        p2p_fail_fraction = p2p_failure / p2p_total if p2p_total > 0 else 0.0
    composite_reward = compute_composite_reward(binary_reward, f2p_pass_fraction, p2p_fail_fraction)
    info.update({
        "binary_reward": binary_reward,
        "f2p_pass_fraction": f2p_pass_fraction,
        "p2p_fail_fraction": p2p_fail_fraction,
        "composite_reward": composite_reward,
    })
    return binary_reward, info