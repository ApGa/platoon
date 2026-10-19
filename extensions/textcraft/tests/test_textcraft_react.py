"""The ``platoon-react`` harness on TextCraft-Synth: it hands a final answer to the environment's terminal tool,
through the real gateway and env service (as ``tests/harness/builtin/test_react.py`` runs number search)."""

from __future__ import annotations

import pytest
from inline_runs import open_service_task, run_tree
from platoon.harness.agent import AgentResult
from test_textcraft_rollouts import TEXTCRAFT_SYNTH


@pytest.mark.asyncio
async def test_react_hands_a_final_answer_to_the_terminal_tool(make_gateway):
    gateway, sampler = make_gateway(lambda prompt: "Nothing to craft; I am done.")
    task, session = await open_service_task(TEXTCRAFT_SYNTH, "train-medium:0")

    run = await run_tree(gateway, "platoon-react", session, task, max_steps=5)

    assert run.result == AgentResult("completed", final_message="Nothing to craft; I am done.")
    # TextCraft's finish(message) is terminal: kept out of the model's tools, it gets the answer as its message.
    assert '"name": "finish"' not in sampler.calls[0][0] and '"name": "craft"' in sampler.calls[0][0]
    (call,) = run.root.misc["tool_calls"]
    assert (call["name"], call["arguments"]) == ("finish", {"message": "Nothing to craft; I am done."})
