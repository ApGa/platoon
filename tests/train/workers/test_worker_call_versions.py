"""A rollout worker's model calls each record the policy version that sampled them (RFC 0020, P5), and its response
gives each datum's own version beside the rollout's.

This runs the real worker entrypoint with a model bound, the toy tokenizer, and a rollout during which the trainer
republishes the binding twice: each call's version is that of the client that answered it, every datum still carries
the rollout's version (the oldest any call sampled), and the datums' own versions cross the response to the parent,
which stamps them under the per-datum stale filter.  It is slow, as every real worker is: the worker imports the
tinker/torch stack.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict

from fakes import WORKER_STUBS

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)
from platoon.config_defs import RolloutConfig
from platoon.task import Task
from platoon.testing import TOY_TOKENIZER_DIR
from platoon.train.step.staleness import StaleFilter, datum_version
from platoon.train.workers.binding import publish_binding
from platoon.train.workers.protocol import (
    PROTOCOL_VERSION,
    ConversionPolicy,
    ModelBinding,
    RolloutWorkerRequest,
    dump_request,
    load_response,
)

STUB_DIR = str(WORKER_STUBS)


def test_a_workers_calls_record_the_versions_that_sampled_them_and_its_datums_carry_their_own(tmp_path, monkeypatch):
    # The client the binding pickles must be the stub module's, which the worker imports by name.
    monkeypatch.syspath_prepend(STUB_DIR)
    from worker_stub_rollouts import VersionedClient

    binding_path = str(tmp_path / "model_binding.pkl")
    publish_binding(VersionedClient(3), 3, binding_path)
    request = RolloutWorkerRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="call-versions-0",
        task_id="task",
        rollout_number=0,
        task=Task(goal="g", id="task", misc={"binding": binding_path}),
        rollout_config=asdict(RolloutConfig(output_dir=str(tmp_path))),
        rollout_fn_ref="worker_stub_rollouts:span_published_versions",
        model=ModelBinding(
            binding_path=binding_path,
            bootstrap_blob=b"",
            follow_poll_seconds=0.0,  # reload the binding at each call
            model_name=str(TOY_TOKENIZER_DIR),  # a tokenizer directory: loads offline
            renderer_name="qwen3_instruct",
            checkpoint_version=3,
        ),
        conversion=ConversionPolicy(),
        response_path=str(tmp_path / "call-versions-0.res"),
        ready_path=str(tmp_path / "call-versions-0.ready"),
        startup_budget_seconds=600.0,
        rollout_budget_seconds=600.0,
    )
    request_path = str(tmp_path / "call-versions-0.req")
    dump_request(request, request_path)
    facts_path = tmp_path / "facts.json"
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join([STUB_DIR, env.get("PYTHONPATH", "")]),
            "STUB_FACTS_PATH": str(facts_path),
            "HF_HUB_OFFLINE": "1",
            "OPENHANDS_SUPPRESS_BANNER": "1",
            "PLATOON_ROLLOUT_WORKER_STARTUP_BUDGET": repr(float(request.startup_budget_seconds)),
            "PLATOON_ROLLOUT_WORKER_DIAG": request.diag_path,
        }
    )

    completed = subprocess.run(
        [sys.executable, "-m", "platoon.train.workers.entrypoint", request_path],
        capture_output=True,
        text=True,
        env=env,
        start_new_session=True,
        timeout=600,
    )

    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
    calls = json.loads(facts_path.read_text())
    assert [(version, tokens) for _call, version, tokens in calls] == [
        (3, [1003]),
        (4, [1004]),
        (4, [1004]),
        (6, [1006]),
    ]
    response = load_response(request.response_path)
    assert response.status == "completed" and response.result is not None
    result = response.result
    assert [datum_version(datum) for datum in result.datums] == [3, 3, 3, 3]  # the rollout's, as before
    assert result.own_versions == (3, 4, 4, 6)

    assert result.stamp_own_versions()
    assert [datum_version(datum) for datum in result.datums] == [3, 4, 4, 6]
    verdict = StaleFilter(1, {}, per_datum=True).check(None, result.datums, 6)
    assert not verdict.stale and verdict.masked == (0, 1, 2)
