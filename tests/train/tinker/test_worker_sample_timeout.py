"""A rollout worker samples under the trainer's sampling timeout, whatever its own environment holds.

The trainer resolves the timeout once (``train.sample_timeout_seconds``, else its ``TINKER_SAMPLE_TIMEOUT_SECONDS``,
else 600) and each worker request's model binding carries it.  This runs the real worker entrypoint with a model
bound, the toy tokenizer, so the worker builds its own policy gateway as in training; the rollout samples nothing and
reports that gateway's bound.  It is slow, as every real worker is: the worker imports the tinker/torch stack.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)
from platoon.config_defs import RolloutConfig
from platoon.task import Task
from platoon.testing import TOY_TOKENIZER_DIR
from platoon.train.tinker.workers.binding import publish_binding
from platoon.train.tinker.workers.protocol import (
    PROTOCOL_VERSION,
    ConversionPolicy,
    ModelBinding,
    RolloutWorkerRequest,
    dump_request,
    load_response,
)

STUB_DIR = os.path.dirname(os.path.abspath(__file__))


def test_a_worker_samples_under_the_bound_its_binding_carries_not_its_own_variable(tmp_path):
    binding_path = tmp_path / "model_binding.pkl"
    publish_binding("a sampling client the rollout never calls", 0, str(binding_path))
    request = RolloutWorkerRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="sample-timeout-0",
        task_id="task",
        rollout_number=0,
        task=Task(goal="g", id="task"),
        rollout_config=asdict(RolloutConfig(output_dir=str(tmp_path))),
        rollout_fn_ref="worker_stub_rollouts:report_policy_gateway",
        model=ModelBinding(
            binding_path=str(binding_path),
            bootstrap_blob=b"",
            follow_poll_seconds=5.0,
            model_name=str(TOY_TOKENIZER_DIR),  # a tokenizer directory: loads offline
            renderer_name="qwen3_instruct",
            sample_timeout_seconds=1800.0,  # the trainer's bound
        ),
        conversion=ConversionPolicy(),
        response_path=str(tmp_path / "sample-timeout-0.res"),
        ready_path=str(tmp_path / "sample-timeout-0.ready"),
        startup_budget_seconds=600.0,
        rollout_budget_seconds=600.0,
    )
    request_path = str(tmp_path / "sample-timeout-0.req")
    dump_request(request, request_path)
    facts_path = tmp_path / "facts.json"
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join([STUB_DIR, env.get("PYTHONPATH", "")]),
            "STUB_FACTS_PATH": str(facts_path),
            "HF_HUB_OFFLINE": "1",
            "OPENHANDS_SUPPRESS_BANNER": "1",
            "TINKER_SAMPLE_TIMEOUT_SECONDS": "300",  # what the worker's own environment says
            "PLATOON_ROLLOUT_WORKER_STARTUP_BUDGET": repr(float(request.startup_budget_seconds)),
            "PLATOON_ROLLOUT_WORKER_DIAG": request.diag_path,
        }
    )

    completed = subprocess.run(
        [sys.executable, "-m", "platoon.train.tinker.workers.entrypoint", request_path],
        capture_output=True,
        text=True,
        env=env,
        start_new_session=True,
        timeout=600,
    )

    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
    assert load_response(request.response_path).status == "no_trajectories"
    assert json.loads(facts_path.read_text()) == {
        "model_name": str(TOY_TOKENIZER_DIR),
        "sample_timeout_seconds": 1800.0,
        "variable": "300",
    }
