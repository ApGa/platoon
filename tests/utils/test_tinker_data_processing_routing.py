"""Routing handle (miles-tinker R3, contract C2) in Tinker datums.

``Datum.loss_fn_inputs["routing_handle"]`` is an int64 ``[1]`` tensor holding
the gateway ``routing_id`` of the LAST sample call merged into the datum; it is
absent when that interaction carried no id, and the trainer's strip tuple in
``platoon/train/tinker/rl.py`` never removes it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import tinker
import torch

import platoon.train.tinker  # noqa: F401  (resolves the package's import cycle first)
from platoon.train.tinker.batch_transforms import LOSS_NORMALIZATION_TOKENS_KEY
from platoon.utils import tinker_data_processing as tdp

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("advantage", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("filter_message_actions", [False, True])
def test_message_action_policy_mask_preserves_context_and_routing(advantage, filter_message_actions):
    message = {**_step("message"), "action_events": [{"kind": "MessageEvent", "source": "agent"}]}
    tool = {**_step("tool"), "action_events": [{"kind": "ActionEvent", "source": "agent"}]}
    result = tdp.trajectory_to_data(
        trajectory={"steps": [message, tool]},
        interactions={
            "message": _interaction([1, 2], [3, 4], 11),
            "tool": _interaction([1, 2, 3, 4, 5], [6, 7], 12),
        },
        task_id="task", trajectory_id="root", trajectory_advantage=advantage,
        checkpoint_version=0, filter_message_actions=filter_message_actions,
    )
    assert len(result.datums) == 1
    datum = result.datums[0]
    assert datum.model_input.to_ints() == [1, 2, 3, 4, 5, 6]
    message_mask = 0 if filter_message_actions else 1
    message_advantage = 0.0 if filter_message_actions else advantage
    assert datum.loss_fn_inputs["mask"].to_torch().tolist() == [0, message_mask, message_mask, 0, 1, 1]
    assert datum.loss_fn_inputs["advantages"].to_torch().tolist() == [
        0, message_advantage, message_advantage, 0, advantage, advantage,
    ]
    assert _handle(datum).item() == 12


def test_message_only_datum_keeps_baseline_membership_and_next_tool_starts_trajectory():
    result = tdp.trajectory_to_data(
        trajectory={"steps": [
            {**_step("message"), "action_events": {"action_events": [{"kind": "MessageEvent", "source": "agent"}]}},
            _step("tool"),
        ]},
        interactions={"message": _interaction([1, 2], [3], 11), "tool": _interaction([8, 9], [10], 12)},
        task_id="task", trajectory_id="child", trajectory_advantage=-1.0,
        checkpoint_version=0, traj_depth=1, filter_message_actions=True,
    )
    assert len(result.datums) == 2
    assert result.datums[0].loss_fn_inputs["mask"].to_torch().sum().item() == 0
    assert result.datums[0].loss_fn_inputs["traj_start"].to_torch().item() == 0
    assert result.datums[1].model_input.to_ints() == [8, 9]
    assert result.datums[1].loss_fn_inputs["traj_start"].to_torch().item() == 1
    assert result.num_output_tokens == 2


def test_message_detection_excludes_user_messages_and_preserves_mixed_tool_completions():
    from platoon.utils.trajectory_error_filtering import detected_message_completion_ids

    steps = [
        {**_step("message"), "action_events": [SimpleNamespace(kind="MessageEvent", source="agent")]},
        {**_step("user"), "action_events": [{"kind": "MessageEvent", "source": "user"}]},
        {**_step("mixed"), "action_events": [{"kind": "MessageEvent", "source": "agent"}]},
        {**_step("mixed"), "action_events": [{"kind": "ActionEvent", "action": {"message": "text"}}]},
    ]
    assert detected_message_completion_ids(steps) == {"message"}


def _interaction(prompt: list[int], completion: list[int], routing_id: int | None):
    """Stand-in for ``TinkerLLMInteraction`` (+ the ``routing_id`` field of 8.1)."""

    return SimpleNamespace(
        obs=tinker.ModelInput.from_ints(prompt),
        action=SimpleNamespace(tokens=list(completion), logprobs=[-0.5] * len(completion)),
        routing_id=routing_id,
    )


def _step(completion_id: str) -> dict:
    return {"misc": {"action_misc": {"completion_id": completion_id}}}


def _convert(interactions: dict) -> list[tinker.Datum]:
    trajectory = {"steps": [_step(cid) for cid in interactions]}
    result = tdp.trajectory_to_data(
        trajectory=trajectory,
        interactions=interactions,
        task_id="task",
        trajectory_id="traj",
        trajectory_advantage=1.0,
        checkpoint_version=3,
    )
    return result.datums


def _handle(datum: tinker.Datum) -> torch.Tensor:
    return datum.loss_fn_inputs[tdp.ROUTING_HANDLE_KEY].to_torch()


def test_merged_steps_carry_the_last_interactions_routing_id():
    turn1 = _interaction([1, 2, 3], [10, 11], routing_id=101)
    # Turn 2 extends turn 1 (prompt = previous prompt + completion + new obs).
    turn2 = _interaction([1, 2, 3, 10, 11, 4, 5], [12, 13, 14], routing_id=202)
    datums = _convert({"c1": turn1, "c2": turn2})

    assert len(datums) == 1
    handle = _handle(datums[0])
    assert handle.dtype == torch.int64
    assert handle.shape == (1,)
    assert handle.item() == 202
    # The datum's tokens equal the last call's prompt + generated tokens.
    full_tokens = datums[0].model_input.to_ints() + datums[0].loss_fn_inputs["target_tokens"].to_torch()[-1:].tolist()
    assert full_tokens == [1, 2, 3, 10, 11, 4, 5, 12, 13, 14]


def test_non_extending_steps_produce_datums_with_their_own_ids():
    turn1 = _interaction([1, 2, 3], [10, 11], routing_id=101)
    turn2 = _interaction([7, 8, 9], [12], routing_id=202)  # does not extend turn1
    datums = _convert({"c1": turn1, "c2": turn2})

    assert len(datums) == 2
    assert _handle(datums[0]).item() == 101
    assert _handle(datums[1]).item() == 202


def test_interactions_without_routing_ids_emit_no_handle():
    turn1 = _interaction([1, 2, 3], [10, 11], routing_id=None)
    turn2 = _interaction([1, 2, 3, 10, 11, 4], [12], routing_id=None)
    datums = _convert({"c1": turn1, "c2": turn2})

    assert len(datums) == 1
    assert tdp.ROUTING_HANDLE_KEY not in datums[0].loss_fn_inputs

    # Legacy interaction objects without the attribute at all.
    legacy = SimpleNamespace(obs=turn1.obs, action=turn1.action)
    datums = _convert({"c1": legacy})
    assert tdp.ROUTING_HANDLE_KEY not in datums[0].loss_fn_inputs


def test_handle_follows_the_last_merged_call_even_when_it_has_no_id():
    # Coverage is all-or-nothing per datum: a final call without an id means
    # the datum's full tokens are not covered by any cached routing.
    turn1 = _interaction([1, 2, 3], [10, 11], routing_id=101)
    turn2 = _interaction([1, 2, 3, 10, 11, 4], [12], routing_id=None)
    datums = _convert({"c1": turn1, "c2": turn2})
    assert len(datums) == 1
    assert tdp.ROUTING_HANDLE_KEY not in datums[0].loss_fn_inputs

    # A flush-and-restart resets the accumulator's id.
    acc = tdp.SequenceAccumulator(last_routing_id=5)
    acc.clear()
    assert acc.last_routing_id is None


def _rl_strip_tuple() -> tuple[str, ...]:
    """Extract the exact strip tuple from rl.py's filtered-datum rebuild."""

    source = (REPO_ROOT / "platoon/train/tinker/rl.py").read_text()
    match = re.search(r"if k\s*\n?\s*not in \(([^)]*)\)", source)
    assert match is not None, "strip tuple not found in rl.py"
    names = []
    for element in match.group(1).split(","):
        element = element.strip()
        if not element:
            continue
        try:
            names.append(ast.literal_eval(element))
        except ValueError:
            # Symbolic constant, e.g. LOSS_NORMALIZATION_TOKENS_KEY.
            names.append({"LOSS_NORMALIZATION_TOKENS_KEY": LOSS_NORMALIZATION_TOKENS_KEY}[element])
    return tuple(names)


def test_rl_rebuild_keeps_the_routing_handle():
    strip = _rl_strip_tuple()
    assert set(strip) == {"mask", "checkpoint_version", "traj_depth", "traj_start", LOSS_NORMALIZATION_TOKENS_KEY}
    assert tdp.ROUTING_HANDLE_KEY not in strip

    turn = _interaction([1, 2, 3], [10, 11], routing_id=4242)
    (datum,) = _convert({"c1": turn})
    rebuilt = tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k not in strip},
    )
    assert "mask" not in rebuilt.loss_fn_inputs
    assert rebuilt.loss_fn_inputs[tdp.ROUTING_HANDLE_KEY].to_torch().item() == 4242
