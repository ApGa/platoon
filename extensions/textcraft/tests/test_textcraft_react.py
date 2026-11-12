"""The ``platoon-react`` harness on TextCraft-Synth, through the real gateway and env service (as
``tests/harness/builtin/test_react.py`` runs number search): it hands a final answer to the environment's terminal
tool, and the game gets the lists and dicts of a model whose tool calls write every value as text."""

from __future__ import annotations

import json
from typing import Any

import pytest
from inline_runs import open_service_task, run_tree
from platoon import testing
from platoon.harness.agent import AgentResult
from test_textcraft_rollouts import TEXTCRAFT_SYNTH
from transformers import PreTrainedTokenizerFast

from platoon_textcraft.env import TextCraftSynthEnvironment

# A chat template for the toy tokenizer: each message's role and text, which is all a scripted model reads.
CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def xml_call(name: str, **arguments: Any) -> str:
    """A tool call in Qwen3.5's XML, which writes each value as text: a string as it is, a list or dict as JSON."""

    parameters = "".join(
        f"<parameter={key}>\n{value if isinstance(value, str) else json.dumps(value)}\n</parameter>\n"
        for key, value in arguments.items()
    )
    return f"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>"


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


@pytest.mark.asyncio
async def test_react_hands_the_game_the_lists_and_dicts_of_xml_tool_calls(tmp_path):
    """Qwen3.5's XML tool calls write every argument value as text, and the chat-template renderer's ``qwen3_coder``
    parser types each value by its tool's schema: ``get_info`` gets a list of items, and ``craft`` a dict of
    ingredients and an item with its count, so the gold trajectory wins the game."""

    spec = TextCraftSynthEnvironment.list_tasks("train-medium")[0]
    ((item, _),) = spec["misc"]["target_items"].items()
    crafts = [
        {"ingredients": step["ingredients"], "target": [step["target"][0], step["result_count"]]}
        for step in spec["misc"]["gold_trajectory"]
    ]
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(testing.TOY_TOKENIZER_DIR), chat_template=CHAT_TEMPLATE)
    completion = "\n".join([xml_call("get_info", items=[item]), *(xml_call("craft", **craft) for craft in crafts)])
    policy = testing.ScriptedPolicy([completion, "Crafted."], tokenizer=tokenizer)
    gateway = testing.make_gateway(
        policy, renderer="chat_template", renderer_kwargs={"tool_call_parser": "qwen3_coder"}
    )

    run = await testing.run_episode(
        "train-medium:0", service=TEXTCRAFT_SYNTH, harness="platoon-react", policy=gateway, output_dir=tmp_path
    )

    calls = run.root.misc["tool_calls"]
    assert [call["arguments"] for call in calls] == [{"items": [item]}, *crafts, {"message": "Crafted."}]
    assert [info["item"] for info in json.loads(calls[0]["text"])] == [item]
    assert run.reward == 1.0
