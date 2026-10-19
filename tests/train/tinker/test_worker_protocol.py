"""The trainer<->rollout-worker IPC contract.

Covers the payload round trip (including that the R3 ``routing_handle``
survives as int64[1]), the 0600/atomic file discipline that protects the
gateway API key inside a pickled sampling client, callable references that
must resolve in a *different* interpreter, and the regression gate for the
datum funnel: a response that carries datums without the funnel would be
discarded by the trainer as an anonymous ``failed_rollouts``, so it must fail
loudly in the child instead.
"""

from __future__ import annotations

import os
import pickle
import stat
from dataclasses import asdict, replace

import pytest
import tinker
import torch
from tinker import TensorData

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)
from platoon.config_defs import InferenceParams, RolloutConfig
from platoon.task import Task
from platoon.train.tinker.workers.protocol import (
    PROTOCOL_VERSION,
    ConversionPolicy,
    ModelBinding,
    ProtocolVersionError,
    RefResolutionError,
    RolloutWorkerRequest,
    RolloutWorkerResponse,
    callable_ref,
    decode_diag,
    dump_request,
    dump_response,
    encode_diag,
    load_request,
    load_response,
    resolve_ref,
    validated_ref,
)
from platoon.train.tinker.workers.workload import add_datum_funnel
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.subagent_sampling import DeterministicSubagentDatumSampler
from platoon.utils.tinker_data_processing import (
    ROUTING_HANDLE_KEY,
    TrajectoryCollectionResult,
    TrajectoryStats,
)


def _datum(reward: float, routing_id: int) -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor([2, 3, 4])),
            "logprobs": TensorData.from_torch(torch.tensor([0.0, -0.1, -0.2])),
            "advantages": TensorData.from_torch(torch.tensor([0.0, reward, reward])),
            "mask": TensorData.from_torch(torch.tensor([0.0, 1.0, 1.0])),
            "checkpoint_version": TensorData.from_torch(torch.tensor([7], dtype=torch.int64)),
            ROUTING_HANDLE_KEY: TensorData.from_torch(torch.tensor([routing_id], dtype=torch.int64)),
        },
    )


def _result(num_datums: int = 2) -> TrajectoryCollectionResult:
    datums = [_datum(1.0, 4242 + index) for index in range(num_datums)]
    return TrajectoryCollectionResult(
        datums=datums,
        task_reward=1.0,
        trajectory_stats=[
            TrajectoryStats(
                trajectory_id="root",
                reward=1.0,
                num_steps=1,
                num_input_tokens=3,
                num_output_tokens=2,
                num_datums=num_datums,
                is_root=True,
            )
        ],
        root_rewards_dict={"task": 1.0},
    )


def _request(tmp_path, **overrides) -> RolloutWorkerRequest:
    rollout_config = RolloutConfig(
        model_name="Qwen/Qwen3",
        output_dir=str(tmp_path / "out" / "3"),
        timeout=3600,
        inference_params=InferenceParams(temperature=0.7, top_p=0.9, max_completion_tokens=1024),
        extra={"environment": {"label": "number_search"}},
    )
    base = dict(
        protocol_version=PROTOCOL_VERSION,
        request_id="task-r0-deadbeef",
        task_id="task",
        rollout_number=0,
        task=Task(goal="do it", id="task", max_steps=42, misc={"index": 3, "name": "t"}),
        rollout_config=asdict(rollout_config),
        rollout_fn_ref="platoon.harness.rollout:harness_rollout",
        model=ModelBinding(
            binding_path=str(tmp_path / "model_binding.pkl"),
            bootstrap_blob=b"",
            follow_poll_seconds=5.0,
            model_name="Qwen/Qwen3",
            renderer_name="qwen3_6_preserve_thinking",
            renderer_kwargs={"preserve_thinking": True, "enable_thinking": True},
            context_window_length=65536,
            checkpoint_version=7,
        ),
        conversion=ConversionPolicy(
            filter_errors=True,
            filter_message_actions=True,
            include_traj_depth=True,
            include_traj_start=True,
            reward_processor_is_default=False,
            reward_processor_ref="platoon.train.tinker.workers.run:default_reward_processor",
            subagent_datum_sampler=DeterministicSubagentDatumSampler(keep_probability=0.25, seed=11),
        ),
        response_path=str(tmp_path / "task-r0-deadbeef.res"),
        ready_path=str(tmp_path / "task-r0-deadbeef.ready"),
    )
    base.update(overrides)
    return RolloutWorkerRequest(**base)


def test_request_round_trips_task_config_and_sampler(tmp_path):
    request = _request(tmp_path)
    path = str(tmp_path / "req.pkl")
    dump_request(request, path)
    loaded = load_request(path)

    assert loaded == request
    assert loaded.conversion.filter_message_actions is True
    assert loaded.task.misc == {"index": 3, "name": "t"}
    # RolloutConfig must reconstruct exactly, including the nested dataclass.
    rebuilt = RolloutConfig(**loaded.rollout_config)
    assert rebuilt.inference_params == InferenceParams(temperature=0.7, top_p=0.9, max_completion_tokens=1024)
    assert rebuilt.output_dir.endswith(os.path.join("out", "3"))
    assert loaded.conversion.subagent_datum_sampler == DeterministicSubagentDatumSampler(keep_probability=0.25, seed=11)


def test_response_round_trips_datums_including_routing_handle(tmp_path):
    result = _result()
    workload = add_datum_funnel(
        RolloutWorkload(environment_steps=1, model_calls=2, input_tokens=6, output_tokens=4, trajectories=1),
        result,
    )
    response = RolloutWorkerResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id="task-r0-deadbeef",
        status="completed",
        result=result,
        workload=workload,
        observed=True,
        model_calls=2,
        routing_id_present=2,
    )
    path = str(tmp_path / "res.pkl")
    dump_response(response, path)
    loaded = load_response(path)

    assert loaded.status == "completed"
    assert loaded.workload == workload
    assert len(loaded.result.datums) == 2
    for original, restored in zip(result.datums, loaded.result.datums):
        assert set(restored.loss_fn_inputs) == set(original.loss_fn_inputs)
        for key, value in original.loss_fn_inputs.items():
            restored_tensor = restored.loss_fn_inputs[key].to_torch()
            assert restored_tensor.dtype == value.to_torch().dtype, key
            assert torch.equal(restored_tensor, value.to_torch()), key
        handle = restored.loss_fn_inputs[ROUTING_HANDLE_KEY].to_torch()
        assert handle.dtype == torch.int64 and handle.shape == (1,)
        assert restored.model_input.to_ints() == original.model_input.to_ints()


def test_request_file_is_private_and_atomic(tmp_path):
    request = _request(tmp_path)
    path = str(tmp_path / "req.pkl")
    dump_request(request, path)

    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, oct(mode)
    # The request carries the gateway API key inside the pickled sampling
    # client, so no group/other bits may ever be set.
    assert not mode & (stat.S_IRWXG | stat.S_IRWXO)
    assert not os.path.exists(path + ".tmp")


def test_dump_is_atomic_under_a_failing_pickle(tmp_path):
    path = str(tmp_path / "res.pkl")
    dump_request(_request(tmp_path), path)
    first = open(path, "rb").read()

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        dump_request(_request(tmp_path, task=Unpicklable()), path)  # type: ignore[arg-type]
    # The previous complete file is untouched and no partial file remains.
    assert open(path, "rb").read() == first
    assert not os.path.exists(path + ".tmp")


def test_resolve_ref_and_callable_ref():
    from platoon.train.tinker.workers.run import default_reward_processor

    assert resolve_ref("platoon.train.tinker.workers.run:default_reward_processor") is default_reward_processor
    assert callable_ref(default_reward_processor) == "platoon.train.tinker.workers.run:default_reward_processor"
    assert validated_ref(default_reward_processor, what="reward_processor") == (
        "platoon.train.tinker.workers.run:default_reward_processor"
    )

    with pytest.raises(RefResolutionError):
        resolve_ref("platoon.train.tinker.workers.run")
    with pytest.raises(RefResolutionError):
        resolve_ref("platoon.train.tinker.workers.run:does_not_exist")
    with pytest.raises(ModuleNotFoundError):
        resolve_ref("platoon.no.such.module:thing")


def test_callable_ref_rejects_closures_and_partials():
    import functools

    from platoon.train.tinker.workers.run import default_reward_processor

    def outer():
        def inner(traj):
            return traj["reward"], {}

        return inner

    with pytest.raises(RefResolutionError):
        callable_ref(outer())
    with pytest.raises(RefResolutionError):
        callable_ref(lambda traj: (traj["reward"], {}))
    with pytest.raises(RefResolutionError):
        callable_ref(functools.partial(default_reward_processor))


def test_callable_ref_maps_main_through_the_module_spec(monkeypatch):
    """A reward processor defined in a ``python -m pkg.script`` entry point.

    Several train scripts define ``reward_processor`` in the script itself, so
    its ``__module__`` is ``"__main__"`` while ``__spec__.name`` holds the real
    dotted name.  Without this mapping the worker's ``__main__`` is the
    entrypoint and the ref cannot resolve.
    """

    import sys
    import types

    from platoon.train.tinker.workers.run import default_reward_processor

    shim = types.ModuleType("__main__")
    shim.__spec__ = importlib_spec("platoon.train.tinker.workers.run")
    monkeypatch.setitem(sys.modules, "__main__", shim)

    main_scoped = types.FunctionType(
        default_reward_processor.__code__, default_reward_processor.__globals__, "default_reward_processor"
    )
    main_scoped.__module__ = "__main__"
    main_scoped.__qualname__ = "default_reward_processor"
    assert callable_ref(main_scoped) == "platoon.train.tinker.workers.run:default_reward_processor"


def importlib_spec(name):
    import importlib.util

    return importlib.util.find_spec(name)


def test_protocol_version_mismatch_raises(tmp_path):
    request = _request(tmp_path)
    path = str(tmp_path / "req.pkl")
    with open(path, "wb") as handle:
        pickle.dump(replace(request, protocol_version=PROTOCOL_VERSION + 1), handle, protocol=5)
    with pytest.raises(ProtocolVersionError):
        load_request(path)

    response = RolloutWorkerResponse(
        protocol_version=PROTOCOL_VERSION + 1,
        request_id="x",
        status="completed",
        result=None,
        workload=RolloutWorkload(),
        observed=True,
    )
    response_path = str(tmp_path / "res.pkl")
    with open(response_path, "wb") as handle:
        pickle.dump(response, handle, protocol=5)
    with pytest.raises(ProtocolVersionError):
        load_response(response_path)

    with open(response_path, "wb") as handle:
        handle.write(b"not a pickle at all")
    with pytest.raises(Exception):
        load_response(response_path)


def test_dump_response_rejects_datums_without_the_funnel(tmp_path):
    """REGRESSION GATE: datums must never travel without their funnel.

    ``_make_task_output`` validates
    ``retained <= post_sampling <= policy_eligible <= postmerge`` and raises;
    the trainer catches that, records ``failed_rollouts`` and enqueues
    ``None``.  A worker that skipped ``add_datum_funnel`` would therefore lose
    every task that produced training data, silently, while the run kept
    burning its allocation.
    """

    result = _result()
    unfunneled = RolloutWorkerResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id="task-r0-deadbeef",
        status="completed",
        result=result,
        workload=RolloutWorkload(trajectories=1),  # postmerge_datums == 0
        observed=True,
    )
    path = str(tmp_path / "res.pkl")
    with pytest.raises(ValueError, match="datum funnel"):
        dump_response(unfunneled, path)
    assert not os.path.exists(path)

    funneled = replace(unfunneled, workload=add_datum_funnel(RolloutWorkload(trajectories=1), result))
    dump_response(funneled, path)
    assert load_response(path).workload.postmerge_datums == 2

    # An empty result is legitimate (Bernoulli sampling / policy exclusion may
    # remove every datum) and must not trip the gate.
    empty = replace(unfunneled, result=_result(num_datums=0), workload=RolloutWorkload())
    dump_response(empty, str(tmp_path / "empty.pkl"))


def test_diag_marker_round_trips():
    payload = encode_diag(
        {"phase": "rollout", "elapsed": 3720.5, "ptc_cells_stuck_at_exit": 1, "source": "while True: pass"}
    )
    assert payload.endswith(b"\n")
    fields = decode_diag(payload)
    assert fields["phase"] == "rollout"
    assert fields["ptc_cells_stuck_at_exit"] == "1"
    assert "while True" in fields["source"]
