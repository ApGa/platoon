from platoon.config_defs import RolloutConfig
from platoon.inference.subprocess_worker import _hard_timeout_seconds
from platoon.inference.workflow import DefaultInferenceGroupWorkflow


def test_inference_subprocess_hard_timeout_includes_cleanup_grace():
    assert _hard_timeout_seconds({"timeout": 10}) == 190
    assert _hard_timeout_seconds({}) == 1080
    assert DefaultInferenceGroupWorkflow._subprocess_hard_timeout_seconds(RolloutConfig(timeout=10)) == 190
