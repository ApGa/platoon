from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "platoon/train/areal/router_replay.py"
SPEC = importlib.util.spec_from_file_location("platoon_areal_router_replay_tests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
router_replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = router_replay
SPEC.loader.exec_module(router_replay)

ROUTED_EXPERTS_FIELD = router_replay.ROUTED_EXPERTS_FIELD
ROUTED_EXPERTS_VALID_FIELD = router_replay.ROUTED_EXPERTS_VALID_FIELD
RouterBinding = router_replay.RouterBinding
RouterReplayBatch = router_replay.RouterReplayBatch
RouterReplayEngineState = router_replay.RouterReplayEngineState
RouterReplayError = router_replay.RouterReplayError
RouterReplayMicrobatch = router_replay.RouterReplayMicrobatch
_collect_router_bindings = router_replay._collect_router_bindings
_collect_chunk_transformer_layers = router_replay._collect_chunk_transformer_layers
_collective_raise_on_error = router_replay._collective_raise_on_error
_layer_coverage_errors = router_replay._layer_coverage_errors
_patched_native_get_replay_topk = router_replay._patched_native_get_replay_topk
_prepare_engine_microbatches = router_replay._prepare_engine_microbatches
_set_binding_forward_data = router_replay._set_binding_forward_data
_split_engine_microbatches = router_replay._split_engine_microbatches
_validate_and_record_cycle = router_replay._validate_and_record_cycle
build_local_router_replay_microbatch = router_replay.build_local_router_replay_microbatch
configure_router_replay_engine = router_replay.configure_router_replay_engine
install_areal_router_replay_hooks = router_replay.install_areal_router_replay_hooks
pop_and_split_actor_router_replay = router_replay.pop_and_split_actor_router_replay
split_packed_for_context_parallel = router_replay.split_packed_for_context_parallel


def _valid_except_terminal(lengths: list[int], sequence_length: int) -> torch.Tensor:
    valid = torch.zeros(len(lengths), sequence_length, dtype=torch.bool)
    for row, length in enumerate(lengths):
        valid[row, : length - 1] = True
    return valid


def test_actor_payload_is_strictly_validated_reordered_and_split():
    lengths = [5, 4, 3]
    attention = torch.arange(5).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    routes = torch.arange(3 * 5 * 2 * 2, dtype=torch.int16).reshape(3, 5, 2, 2) % 8
    routes[0, 0] = torch.tensor([0, 1])  # expert zero is data, not an invalid sentinel
    valid = _valid_except_terminal(lengths, 5)
    data = {
        "attention_mask": attention,
        ROUTED_EXPERTS_FIELD: routes,
        ROUTED_EXPERTS_VALID_FIELD: valid,
    }
    mb_list = SimpleNamespace(
        forward_indices=[2, 0, 1],
        mbs=[
            {"attention_mask": attention[[2, 0]]},
            {"attention_mask": attention[[1]]},
        ],
    )
    config = SimpleNamespace(
        enable_router_replay=True,
        router_replay_num_layers=2,
        router_replay_topk=2,
        router_replay_num_experts=8,
    )

    split = pop_and_split_actor_router_replay(data, mb_list, config)

    assert ROUTED_EXPERTS_FIELD not in data
    assert ROUTED_EXPERTS_VALID_FIELD not in data
    torch.testing.assert_close(split[0].routes, routes[[2, 0]])
    torch.testing.assert_close(split[0].valid, valid[[2, 0]])
    torch.testing.assert_close(split[1].routes, routes[[1]])


def test_actor_payload_fails_closed_on_missing_interior_or_valid_terminal_route():
    attention = torch.ones(1, 4, dtype=torch.bool)
    routes = torch.zeros(1, 4, 2, 2, dtype=torch.uint8)
    config = SimpleNamespace(
        enable_router_replay=True,
        router_replay_num_layers=2,
        router_replay_topk=2,
        router_replay_num_experts=8,
    )
    mb_list = SimpleNamespace(
        forward_indices=[0],
        mbs=[{"attention_mask": attention}],
    )

    for invalid_mask in (
        torch.tensor([[True, False, True, False]]),
        torch.tensor([[True, True, True, True]]),
    ):
        data = {
            "attention_mask": attention,
            ROUTED_EXPERTS_FIELD: routes,
            ROUTED_EXPERTS_VALID_FIELD: invalid_mask,
        }
        with pytest.raises(RouterReplayError, match="coverage is incomplete"):
            pop_and_split_actor_router_replay(data, mb_list, config)


def test_actor_payload_rejects_duplicate_topk_experts():
    attention = torch.ones(1, 3, dtype=torch.bool)
    routes = torch.tensor([[[[0, 1]], [[2, 2]], [[0, 0]]]], dtype=torch.uint8)
    valid = torch.tensor([[True, True, False]])
    data = {
        "attention_mask": attention,
        ROUTED_EXPERTS_FIELD: routes,
        ROUTED_EXPERTS_VALID_FIELD: valid,
    }
    config = SimpleNamespace(
        enable_router_replay=True,
        router_replay_num_layers=1,
        router_replay_topk=2,
        router_replay_num_experts=8,
    )
    mb_list = SimpleNamespace(forward_indices=[0], mbs=[{"attention_mask": attention}])

    with pytest.raises(RouterReplayError, match="must be unique"):
        pop_and_split_actor_router_replay(data, mb_list, config)


def test_cp_zigzag_split_preserves_trailing_layer_and_topk_dimensions():
    base = torch.arange(16).reshape(16, 1, 1)
    tensor = torch.cat([base, base + 100], dim=2)
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)

    rank0 = split_packed_for_context_parallel(tensor, cu_seqlens, cp_size=2, cp_rank=0)
    rank1 = split_packed_for_context_parallel(tensor, cu_seqlens, cp_size=2, cp_rank=1)

    torch.testing.assert_close(rank0[:, 0, 0], torch.tensor([0, 1, 6, 7, 8, 9, 14, 15]))
    torch.testing.assert_close(rank1[:, 0, 0], torch.tensor([2, 3, 4, 5, 10, 11, 12, 13]))
    torch.testing.assert_close(rank0[:, 0, 1], rank0[:, 0, 0] + 100)


def test_microbatch_alignment_applies_cp_before_tp_and_keeps_invalid_rows_live():
    routes = torch.zeros(2, 4, 2, 1, dtype=torch.int16)
    routes[0, :, 0, 0] = torch.tensor([10, 11, 12, 13])
    routes[1, :, 0, 0] = torch.tensor([20, 21, 22, 23])
    routes[:, :, 1, 0] = routes[:, :, 0, 0] + 100
    batch = RouterReplayBatch(
        routes=routes,
        valid=torch.tensor([[True, True, False, False], [True, False, False, False]]),
    )

    # Real lengths are 3 and 2. Each is padded to four, then an extra four-token
    # batch-padding sequence is appended. CP0 selects [first,last] from each
    # sequence; TP0 then keeps the first half of that local sequence.
    result = build_local_router_replay_microbatch(
        batch,
        old_cu_seqlens=torch.tensor([0, 3, 5], dtype=torch.int32),
        padded_cu_seqlens=torch.tensor([0, 4, 8, 12], dtype=torch.int32),
        expected_layers=frozenset({0, 1}),
        use_padded_seq=False,
        cp_size=2,
        cp_rank=0,
        sequence_parallel=True,
        tp_size=2,
        scatter_fn=lambda value: value[: value.shape[0] // 2].contiguous(),
    )

    assert result.routes.shape == (3, 2, 1)
    torch.testing.assert_close(result.routes[:, 0, 0], torch.tensor([10, 0, 20], dtype=torch.int16))
    torch.testing.assert_close(result.routes[:, 1, 0], torch.tensor([110, 0, 120], dtype=torch.int16))
    torch.testing.assert_close(result.valid, torch.tensor([True, False, True]))


class _FakeReplay:
    def __init__(self):
        self.target_topk_idx = None
        self.recorded_topk_idx = None
        self.router_replay_action = None
        self.replay_backward_list = []
        self._platoon_target_valid = None
        self._platoon_backward_valid = []

    def set_target_indices(self, indices):
        self.target_topk_idx = indices
        self.replay_backward_list.append(indices)

    def set_router_replay_action(self, action):
        self.router_replay_action = action

    def clear_router_replay_action(self):
        self.router_replay_action = None

    def clear_indices(self):
        self.target_topk_idx = None
        self.replay_backward_list = []


class _FakeRouter(torch.nn.Module):
    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number
        self.router_replay = _FakeReplay()


class _AttachableRouter(torch.nn.Module):
    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number
        self.topk = 2
        self.router_replay = None
        self.config = SimpleNamespace(
            moe_enable_routing_replay=False,
            moe_aux_loss_coeff=0.01,
            moe_z_loss_coeff=0.001,
        )


class _NativeFakeReplay(_FakeReplay):
    global_router_replay_instances = []

    def __init__(self):
        super().__init__()
        self.global_router_replay_instances.append(self)


def test_global_layer_numbers_select_pp_vp_local_router_slabs():
    model = torch.nn.Module()
    model.first = _FakeRouter(layer_number=1)
    model.third = _FakeRouter(layer_number=3)
    bindings = _collect_router_bindings(model)
    assert [binding.layer_index for binding in bindings] == [0, 2]

    routes = torch.arange(5 * 4 * 2, dtype=torch.uint8).reshape(5, 4, 2)
    microbatch = RouterReplayMicrobatch(
        routes=routes,
        valid=torch.tensor([True, True, True, True, False]),
        expected_layers=frozenset({0, 2}),
    )
    action = object()
    for binding in bindings:
        _set_binding_forward_data(binding, microbatch, action)

    torch.testing.assert_close(bindings[0].replay.target_topk_idx, routes[:, 0])
    torch.testing.assert_close(bindings[1].replay.target_topk_idx, routes[:, 2])
    assert bindings[0].replay.target_topk_idx.dtype == torch.uint8
    assert microbatch.seen_layers == {0, 2}


def test_post_construction_attachment_is_per_router_and_preserves_loss_coefficients():
    _NativeFakeReplay.global_router_replay_instances.clear()
    model = torch.nn.Module()
    model.first = _AttachableRouter(layer_number=1)
    model.third = _AttachableRouter(layer_number=3)
    coefficients = [
        (module.config.moe_aux_loss_coeff, module.config.moe_z_loss_coeff) for module in (model.first, model.third)
    ]

    bindings = _collect_router_bindings(
        model,
        attach_missing=True,
        replay_cls=_NativeFakeReplay,
    )

    assert [binding.layer_index for binding in bindings] == [0, 2]
    assert bindings[0].replay is not bindings[1].replay
    assert model.first.config.moe_enable_routing_replay
    assert model.third.config.moe_enable_routing_replay
    assert _NativeFakeReplay.global_router_replay_instances == []
    assert coefficients == [
        (module.config.moe_aux_loss_coeff, module.config.moe_z_loss_coeff) for module in (model.first, model.third)
    ]


def test_engine_microbatch_split_reorders_routes_and_validity_together():
    routes = torch.arange(3 * 4 * 2 * 1).reshape(3, 4, 2, 1)
    valid = torch.tensor(
        [
            [True, True, False, False],
            [True, False, False, False],
            [True, True, True, False],
        ]
    )
    mb_list = SimpleNamespace(
        forward_indices=[2, 0, 1],
        mbs=[
            {"cu_seqlens": torch.tensor([0, 3, 6], dtype=torch.int32)},
            {"cu_seqlens": torch.tensor([0, 2], dtype=torch.int32)},
        ],
    )

    split = _split_engine_microbatches(RouterReplayBatch(routes, valid), mb_list)

    torch.testing.assert_close(split[0].routes, routes[[2, 0]])
    torch.testing.assert_close(split[0].valid, valid[[2, 0]])
    torch.testing.assert_close(split[1].routes, routes[[1]])
    torch.testing.assert_close(split[1].valid, valid[[1]])


def test_configuration_rejects_fused_routing_before_model_use():
    engine = SimpleNamespace(
        config=SimpleNamespace(enable_router_replay=True),
        tf_config=SimpleNamespace(
            moe_enable_routing_replay=False,
            moe_router_fusion=True,
        ),
    )

    with pytest.raises(RouterReplayError, match="moe_router_fusion"):
        configure_router_replay_engine(engine)


@pytest.mark.parametrize(
    ("recompute_granularity", "recompute_method"),
    [("selective", "uniform"), ("full", "block")],
)
def test_configuration_rejects_partial_or_selective_checkpoint_recompute(
    recompute_granularity,
    recompute_method,
):
    engine = SimpleNamespace(
        config=SimpleNamespace(enable_router_replay=True, gradient_checkpointing=True),
        mcore_config=SimpleNamespace(
            recompute_granularity=recompute_granularity,
            recompute_method=recompute_method,
        ),
        tf_config=SimpleNamespace(
            moe_enable_routing_replay=False,
            moe_router_fusion=False,
            moe_router_load_balancing_type="aux_loss",
            # megatron-bridge exposes this stale pre-provider snapshot.
            recompute_granularity=None,
            recompute_method=None,
        ),
    )

    with pytest.raises(RouterReplayError, match="full/uniform recompute"):
        configure_router_replay_engine(engine)


def _make_stale_bridge_config_engine(*, runtime_recompute=("full", "uniform")):
    router = SimpleNamespace(
        config=SimpleNamespace(
            moe_router_fusion=False,
            recompute_granularity=runtime_recompute[0],
            recompute_method=runtime_recompute[1],
            sequence_parallel=True,
        ),
        topk=2,
    )
    binding = RouterBinding(0, router, _FakeReplay())
    engine = SimpleNamespace(
        config=SimpleNamespace(
            enable_router_replay=True,
            gradient_checkpointing=True,
            router_replay_num_layers=1,
            router_replay_topk=2,
            router_replay_num_experts=8,
        ),
        mcore_config=SimpleNamespace(
            recompute_granularity="full",
            recompute_method="uniform",
        ),
        tf_config=SimpleNamespace(
            moe_enable_routing_replay=False,
            moe_router_fusion=False,
            moe_router_load_balancing_type="aux_loss",
            num_layers=1,
            moe_router_topk=2,
            num_moe_experts=8,
            # These deliberately model bridge.transformer_config before AReaL
            # applies overrides to the provider that constructs the model.
            recompute_granularity=None,
            recompute_method=None,
            sequence_parallel=False,
        ),
        model=[SimpleNamespace(module=None)],
    )
    return engine, binding


def test_configuration_uses_constructed_router_config_not_stale_bridge_snapshot(monkeypatch):
    engine, binding = _make_stale_bridge_config_engine()
    monkeypatch.setattr(router_replay, "_collect_router_bindings", lambda _model, **_kwargs: (binding,))

    configure_router_replay_engine(engine)

    state = engine._platoon_router_replay_state
    assert state.full_recompute is True
    assert state.sequence_parallel is True


def test_configuration_rejects_runtime_router_recompute_mismatch(monkeypatch):
    engine, binding = _make_stale_bridge_config_engine(runtime_recompute=("selective", "uniform"))
    monkeypatch.setattr(router_replay, "_collect_router_bindings", lambda _model, **_kwargs: (binding,))

    with pytest.raises(RouterReplayError, match="runtime config must use full/uniform recompute"):
        configure_router_replay_engine(engine)


def test_microbatch_preparation_uses_runtime_sequence_parallel_state(monkeypatch):
    megatron = ModuleType("megatron")
    megatron.__path__ = []
    megatron_core = ModuleType("megatron.core")
    megatron_core.parallel_state = SimpleNamespace(
        get_context_parallel_world_size=lambda: 2,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 4,
    )
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)

    captured = {}

    def fake_build(_batch, **kwargs):
        captured.update(kwargs)
        return "prepared"

    monkeypatch.setattr(router_replay, "build_local_router_replay_microbatch", fake_build)
    state = RouterReplayEngineState(
        bindings=(),
        expected_layers=frozenset({0}),
        num_layers=1,
        topk=2,
        num_experts=8,
        full_recompute=True,
        sequence_parallel=True,
    )
    engine = SimpleNamespace(
        use_padded_seq=False,
        # Deliberately stale: preparation must not use this value.
        tf_config=SimpleNamespace(sequence_parallel=False),
        _platoon_router_replay_state=state,
    )
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    mb_list = SimpleNamespace(
        forward_indices=None,
        mbs=[{"cu_seqlens": cu_seqlens}],
        padded_mbs=[{"cu_seqlens": cu_seqlens}],
        old_cu_seqlens_list=None,
    )
    batch = RouterReplayBatch(
        routes=torch.zeros(1, 4, 1, 2, dtype=torch.uint8),
        valid=torch.tensor([[True, True, True, False]]),
    )

    assert _prepare_engine_microbatches(engine, mb_list, batch) == ["prepared"]
    assert captured["sequence_parallel"] is True
    assert captured["tp_size"] == 4


def test_configuration_rejects_auxiliary_or_mtp_router_layer(monkeypatch):
    router = SimpleNamespace(
        config=SimpleNamespace(moe_router_fusion=False),
        topk=2,
    )
    replay = _FakeReplay()
    monkeypatch.setattr(
        router_replay,
        "_collect_router_bindings",
        lambda _model, **_kwargs: (RouterBinding(4, router, replay),),
    )
    engine = SimpleNamespace(
        config=SimpleNamespace(
            enable_router_replay=True,
            router_replay_num_layers=4,
            router_replay_topk=2,
            router_replay_num_experts=8,
        ),
        tf_config=SimpleNamespace(
            moe_enable_routing_replay=False,
            moe_router_fusion=False,
            moe_router_load_balancing_type="aux_loss",
            num_layers=4,
            moe_router_topk=2,
            num_moe_experts=8,
        ),
        model=[SimpleNamespace(module=None)],
    )

    with pytest.raises(RouterReplayError, match="MTP or another auxiliary router"):
        configure_router_replay_engine(engine)


def test_layer_coverage_detects_missing_router_and_replica_disagreement():
    payloads = [
        {"rank": 0, "pp_rank": 0, "layers": (0,), "errors": ()},
        {"rank": 1, "pp_rank": 0, "layers": (), "errors": ()},
        {"rank": 2, "pp_rank": 1, "layers": (2, 3), "errors": ()},
    ]

    errors = _layer_coverage_errors(payloads, num_layers=4)

    assert any("missing=[1]" in error for error in errors)
    assert any("inconsistent router layers" in error for error in errors)


def test_layer_coverage_rejects_duplicate_ownership_across_pipeline_ranks():
    payloads = [
        {"rank": 0, "pp_rank": 0, "layers": (0, 1, 2), "errors": ()},
        {"rank": 1, "pp_rank": 1, "layers": (2, 3), "errors": ()},
    ]

    errors = _layer_coverage_errors(payloads, num_layers=4)

    assert any("multiple PP ranks" in error and "2" in error for error in errors)


def test_chunk_layer_discovery_exposes_missing_router_binding():
    class Layer(torch.nn.Module):
        def __init__(self, layer_number, with_router):
            super().__init__()
            self.layer_number = layer_number
            if with_router:
                self.router = _FakeRouter(layer_number)

    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList([Layer(1, True), Layer(2, False)])

    expected = _collect_chunk_transformer_layers(model)
    actual = frozenset(binding.layer_index for binding in _collect_router_bindings(model))

    assert expected == frozenset({0, 1})
    assert actual == frozenset({0})


def test_chunk_layer_discovery_ignores_vision_decoder_layers():
    class Layer(torch.nn.Module):
        def __init__(self, layer_number):
            super().__init__()
            self.layer_number = layer_number

    model = torch.nn.Module()
    model.language_model = torch.nn.Module()
    model.language_model.decoder = torch.nn.Module()
    model.language_model.decoder.layers = torch.nn.ModuleList([Layer(1), Layer(2)])
    model.vision_model = torch.nn.Module()
    model.vision_model.decoder = torch.nn.Module()
    model.vision_model.decoder.layers = torch.nn.ModuleList([Layer(10), Layer(11), Layer(12)])

    assert _collect_chunk_transformer_layers(model) == frozenset({0, 1})


def test_packed_forward_rejects_replay_microbatch_on_unbound_chunk(monkeypatch):
    areal_module = ModuleType("areal")
    areal_module.__path__ = []
    engine_package = ModuleType("areal.engine")
    engine_package.__path__ = []
    megatron_engine = ModuleType("areal.engine.megatron_engine")
    megatron_engine.packed_context_parallel_forward = lambda *_args, **_kwargs: "live"
    monkeypatch.setitem(sys.modules, "areal", areal_module)
    monkeypatch.setitem(sys.modules, "areal.engine", engine_package)
    monkeypatch.setitem(sys.modules, "areal.engine.megatron_engine", megatron_engine)

    install_areal_router_replay_hooks()

    with pytest.raises(RouterReplayError, match="no router binding metadata"):
        megatron_engine.packed_context_parallel_forward(
            torch.nn.Module(),
            {router_replay._MICROBATCH_FIELD: object()},
        )


def test_collective_error_consensus_propagates_remote_rank_failure():
    class FakeDist:
        ReduceOp = SimpleNamespace(MAX="max")

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_rank():
            return 0

        @staticmethod
        def get_world_size():
            return 2

        @staticmethod
        def all_reduce(flag, op):
            assert op == "max"
            flag.fill_(1)

        @staticmethod
        def all_gather_object(output, local):
            output[0] = local
            output[1] = {"rank": 1, "error": "checkpoint queue was not empty"}

    with pytest.raises(RouterReplayError, match="rank 1: checkpoint queue was not empty"):
        _collective_raise_on_error(None, phase="lifecycle", dist_module=FakeDist, device="cpu")


def test_collective_error_consensus_normal_path_uses_only_scalar_all_reduce():
    calls = []

    class FakeDist:
        ReduceOp = SimpleNamespace(MAX="max")

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_rank():
            return 0

        @staticmethod
        def get_world_size():
            return 2

        @staticmethod
        def all_reduce(flag, op):
            calls.append(("all_reduce", int(flag.item()), op))

        @staticmethod
        def all_gather_object(_output, _local):
            raise AssertionError("normal lifecycle path must not serialize Python objects")

    _collective_raise_on_error(None, phase="lifecycle", dist_module=FakeDist, device="cpu")

    assert calls == [("all_reduce", 0, "max")]


def test_native_replay_splices_live_topk_only_for_explicitly_invalid_rows():
    replay_forward, replay_backward = object(), object()
    replay = _FakeReplay()
    target = torch.tensor([[0, 1], [0, 1], [2, 3]])
    valid = torch.tensor([True, False, True])
    replay.set_target_indices(target)
    replay._platoon_target_valid = valid
    replay._platoon_backward_valid.append(valid)
    scores = torch.tensor([[9.0, 8.0, 1.0, 0.0], [1.0, 2.0, 8.0, 9.0], [0.0, 1.0, 8.0, 9.0]])

    def live_topk(value, topk, **_kwargs):
        return torch.topk(value, topk, dim=1)

    def original(*_args, **_kwargs):
        raise AssertionError("native fallback should be handled by the guarded helper")

    replay.router_replay_action = replay_forward
    _, forward_indices = _patched_native_get_replay_topk(
        replay,
        scores,
        2,
        None,
        None,
        live_topk,
        replay_forward_action=replay_forward,
        replay_backward_action=replay_backward,
        original=original,
    )
    torch.testing.assert_close(forward_indices[0], target[0])
    torch.testing.assert_close(forward_indices[1], torch.tensor([3, 2]))
    torch.testing.assert_close(forward_indices[2], target[2])
    # Rows 0 and 2 have the same live/replayed expert sets. Row 2 deliberately
    # uses the opposite order to verify that this diagnostic is order-insensitive.
    assert replay._platoon_device_counters.tolist() == [1, 3, 2, 0]

    replay.router_replay_action = replay_backward
    _, backward_indices = _patched_native_get_replay_topk(
        replay,
        scores,
        2,
        None,
        None,
        live_topk,
        replay_forward_action=replay_forward,
        replay_backward_action=replay_backward,
        original=original,
    )
    torch.testing.assert_close(backward_indices, forward_indices)
    assert replay.replay_backward_list == []
    assert replay._platoon_backward_valid == []
    assert replay._platoon_forward_calls == replay._platoon_backward_calls == 1


def test_pipeline_style_recompute_consumes_targets_and_masks_fifo():
    replay_forward, replay_backward = object(), object()
    replay = _FakeReplay()
    binding = RouterBinding(layer_index=0, router=None, replay=replay)
    scores_a = torch.tensor([[9.0, 8.0, 1.0, 0.0], [1.0, 2.0, 8.0, 9.0]])
    scores_b = torch.tensor([[1.0, 8.0, 9.0, 0.0], [9.0, 8.0, 1.0, 0.0]])
    routes_a = torch.tensor([[[0, 1]], [[0, 1]]], dtype=torch.uint8)
    routes_b = torch.tensor([[[2, 1]], [[2, 3]]], dtype=torch.uint8)
    valid_a = torch.tensor([True, False])
    valid_b = torch.tensor([True, False])

    def live_topk(value, topk, **_kwargs):
        return torch.topk(value, topk, dim=1)

    def original(*_args, **_kwargs):
        raise AssertionError("replay action should use the guarded helper")

    forward_results = []
    for routes, valid, scores in (
        (routes_a, valid_a, scores_a),
        (routes_b, valid_b, scores_b),
    ):
        microbatch = RouterReplayMicrobatch(routes, valid, frozenset({0}))
        _set_binding_forward_data(binding, microbatch, replay_forward)
        _, indices = _patched_native_get_replay_topk(
            replay,
            scores,
            2,
            None,
            None,
            live_topk,
            replay_forward_action=replay_forward,
            replay_backward_action=replay_backward,
            original=original,
        )
        forward_results.append(indices)

    assert [target.dtype for target in replay.replay_backward_list] == [torch.uint8, torch.uint8]
    replay.router_replay_action = replay_backward
    backward_results = []
    for scores in (scores_a, scores_b):
        _, indices = _patched_native_get_replay_topk(
            replay,
            scores,
            2,
            None,
            None,
            live_topk,
            replay_forward_action=replay_forward,
            replay_backward_action=replay_backward,
            original=original,
        )
        backward_results.append(indices)

    torch.testing.assert_close(backward_results[0], forward_results[0])
    torch.testing.assert_close(backward_results[1], forward_results[1])
    assert replay.replay_backward_list == []
    assert replay._platoon_backward_valid == []


def test_full_recompute_lifecycle_fails_closed_on_missing_backward_replay(monkeypatch):
    monkeypatch.setattr(router_replay, "_record_metrics", lambda **_values: None)
    bindings = (
        RouterBinding(0, None, _FakeReplay()),
        RouterBinding(2, None, _FakeReplay()),
    )
    for binding in bindings:
        binding.replay._platoon_forward_calls = 2
        binding.replay._platoon_backward_calls = 2
        binding.replay._platoon_device_counters = torch.tensor([1, 8, 7, 2])
    engine = SimpleNamespace(
        config=SimpleNamespace(gradient_checkpointing=True),
        # Deliberately stale: lifecycle checks must use the verified state.
        tf_config=SimpleNamespace(recompute_granularity=None),
        _platoon_router_replay_state=RouterReplayEngineState(
            bindings=bindings,
            expected_layers=frozenset({0, 2}),
            num_layers=4,
            topk=2,
            num_experts=8,
            full_recompute=True,
            sequence_parallel=True,
        ),
    )
    microbatches = [
        RouterReplayMicrobatch(torch.empty(0), torch.empty(0), frozenset({0, 2}), {0, 2}),
        RouterReplayMicrobatch(torch.empty(0), torch.empty(0), frozenset({0, 2}), {0, 2}),
    ]

    _validate_and_record_cycle(engine, microbatches)

    bindings[0].replay._platoon_backward_calls = 1
    with pytest.raises(RouterReplayError, match="checkpoint replay calls"):
        _validate_and_record_cycle(engine, microbatches)
