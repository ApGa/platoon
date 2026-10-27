"""A rollout worker's request as the worker tests give it to a worker, a fork server or the supervisor.

:func:`worker_request` is the request of task ``task``'s first rollout, run by a function of the worker stubs
(``fakes/worker_stubs``, which a test puts on the worker's path) with no model bound, so the worker opens no gateway
and loads no tokenizer; a test gives the fields its case turns on (budgets, labels, ...) in place of the defaults.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from platoon.config_defs import RolloutConfig
from platoon.task import Task
from platoon.train.workers.protocol import PROTOCOL_VERSION, ConversionPolicy, ModelBinding, RolloutWorkerRequest


def worker_request(
    directory: Path | str,
    request_id: str = "task-r0-aaaaaaaa",
    *,
    rollout: str = "worker_stub_entrypoint:main",
    output_dir: Path | str | None = None,
    **fields: Any,
) -> RolloutWorkerRequest:
    """The request ``request_id`` of task ``task``'s first rollout, run by ``rollout`` (``module:function``) with no
    model bound: its model binding, response and ready marker in ``directory``, its rollout's output in
    ``output_dir`` (``directory`` when None), and ``fields`` in place of the other fields' defaults."""

    directory = Path(directory)
    request: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "task_id": "task",
        "rollout_number": 0,
        "task": Task(goal="g", id="task"),
        "rollout_config": asdict(RolloutConfig(output_dir=str(output_dir if output_dir is not None else directory))),
        "rollout_fn_ref": rollout,
        "model": ModelBinding(
            binding_path=str(directory / "model_binding.pkl"),
            bootstrap_blob=b"",
            follow_poll_seconds=5.0,
            model_name="",  # no model bound: no gateway, no tokenizer
            renderer_name="",
        ),
        "conversion": ConversionPolicy(),
        "response_path": str(directory / f"{request_id}.res"),
        "ready_path": str(directory / f"{request_id}.ready"),
    }
    return RolloutWorkerRequest(**{**request, **fields})
