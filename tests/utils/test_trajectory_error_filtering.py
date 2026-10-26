"""``filter_errors`` on gateway-built trajectories.

A gateway step keeps its harness's annotations of the model call in its misc (``ctx.annotate_step``), where
legacy steps carried them as step fields: CodeAct's ``error`` and ``output``, an OpenHands harness's
``observation_events``.  The detector reads both places, so the error tokens of a gateway rollout are marked and,
once centered, suppressed where their advantage is positive.
"""

from __future__ import annotations

import pytest
import tinker
import torch

from platoon.envs.service import Outcome
from platoon.gateway.core import AgentSession, ChatResponse
from platoon.gateway.trace import build_trajectory_collection
from platoon.task import Task
from platoon.train.credit import group as credit_group_module
from platoon.utils.tinker_data_processing import get_train_data_for_trajectory_collection
from platoon.utils.trajectory_error_filtering import (
    ERROR_ACTION_MASK_KEY,
    detected_error_completion_ids,
    step_reports_error,
)

TRACEBACK = "Traceback (most recent call last):\n  File \"<cell>\", line 1\nNameError: name 'x' is not defined"


def _call(call_id: str, prompt: list[int], tokens: list[int]) -> ChatResponse:
    return ChatResponse(
        call_id=call_id,
        model="m",
        prompt=tinker.ModelInput.from_ints(prompt),
        tokens=tokens,
        logprobs=[-0.5] * len(tokens),
        stop_reason="stop",
        parse_success=True,
        role="assistant",
        content="",
        reasoning_content=None,
        tool_calls=None,
        parsed_tool_calls=0,
        routing_id=None,
    )


def _rollout(annotations: list[dict]) -> tuple[dict, dict]:
    """A root agent whose calls extend each other, each annotated as a harness annotates it; reward 1."""

    agent = AgentSession(agent_id="root", rollout_id="r", trajectory_id="root", secret="s", interactions={})
    prompt: list[int] = [1, 2]
    for index, misc in enumerate(annotations):
        call = _call(f"call-{index}", list(prompt), [10 + index, 20 + index])
        agent.calls.append(call)
        agent.step_misc[call.call_id] = dict(misc)
        prompt += call.tokens + [3]  # the next prompt extends this one and its completion: one datum
    collection = build_trajectory_collection(
        [agent], task=Task(id="t", goal="g"), outcome=Outcome(status="scored", rewards={"reward": 1.0})
    )
    return collection.to_dict(), {call.call_id: call.interaction() for call in agent.calls}


def _codeact(error: str | None = None, output: str | None = "ok\n") -> dict:
    """The step misc CodeAct annotates on the call that wrote a cell (``CodeActStep.annotation``)."""

    return {"thought": None, "code": "print(x)", "output": output, "error": error}


@pytest.mark.parametrize(
    "misc",
    [
        _codeact(error="NameError: name 'x' is not defined"),
        _codeact(output=TRACEBACK),
        {"observation_events": [{"kind": "ObservationEvent", "observation": {"is_error": True}}]},
    ],
    ids=["codeact-error", "codeact-traceback-output", "observation-event"],
)
def test_a_gateway_steps_annotated_error_is_detected(misc):
    trajectory, _ = _rollout([_codeact(), misc, _codeact()])
    steps = next(iter(trajectory["trajectories"].values()))["steps"]

    assert [step_reports_error(step) for step in steps] == [False, True, False]
    assert detected_error_completion_ids(steps) == {"call-1"}


def test_a_clean_gateway_step_reports_no_error():
    trajectory, _ = _rollout([_codeact(), {"observation_events": [{"kind": "ObservationEvent", "observation": {}}]}])
    steps = next(iter(trajectory["trajectories"].values()))["steps"]

    assert detected_error_completion_ids(steps) == set()


def test_legacy_step_fields_are_still_read():
    steps = [{"misc": {"action_misc": {"completion_id": "c1"}}, "error": "boom"}]

    assert detected_error_completion_ids(steps) == {"c1"}


def test_the_error_tokens_of_a_gateway_rollout_are_suppressed_where_their_advantage_is_positive():
    trajectory, interactions = _rollout([_codeact(), _codeact(error="NameError: name 'x' is not defined")])
    result = get_train_data_for_trajectory_collection(
        trajectory_collection=trajectory,
        interactions=interactions,
        task_id="t",
        checkpoint_version=0,
        filter_errors=True,
    )
    (datum,) = result.datums
    error_mask = datum.loss_fn_inputs[ERROR_ACTION_MASK_KEY].to_torch().bool()
    targets = datum.loss_fn_inputs["target_tokens"].to_torch()
    assert targets[error_mask].tolist() == [11, 21]  # exactly the erroneous call's sampled tokens

    # Centered on a baseline below the reward, every action token has a positive advantage: the error's are dropped.
    advantages = datum.loss_fn_inputs["advantages"].to_torch()
    mask = datum.loss_fn_inputs["mask"].to_torch()
    centered = torch.where(mask > 0, advantages - 0.5, advantages)
    datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(centered)
    keep, metrics = credit_group_module._filter_positive_centered_error_tokens(datum)

    assert keep and metrics["suppressed_positive_action_tokens"] == 2.0
    trained = datum.loss_fn_inputs["mask"].to_torch() > 0
    assert targets[trained].tolist() == [10, 20]
