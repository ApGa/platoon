"""The new renderer must be token-exact against the supplied HF template."""

from pathlib import Path

import pytest

from platoon.train.tinker.renderers.qwen3_8_preserve_thinking import Qwen3_8PreserveThinkingRenderer

SNAPSHOT = Path(
    "/groups/gai51740/aci18912it/rasp/cache/huggingface/hub/"
    "models--apurvaga--Qwen3.8-27B-preserve-thinking/snapshots/"
    "97c725615b8aa025b800da4ceed38811358b30f1"
)
pytestmark = pytest.mark.skipif(not SNAPSHOT.exists(), reason="Qwen3.8 snapshot is unavailable")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(SNAPSHOT))


@pytest.mark.parametrize("effort", ["xhigh", "medium", "low"])
@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("system", [None, "", "Follow the task carefully."])
@pytest.mark.parametrize(
    "tools", [None, [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]]
)
def test_hf_template_parity(tokenizer, effort, thinking, system, tools):
    messages = [] if system is None else [{"role": "system", "content": system}]
    messages += [
        {"role": "user", "content": "Look up the result."},
        {"role": "assistant", "content": "<think>Keep this historical reasoning.</think>Found it."},
        {"role": "user", "content": "Check again."},
    ]
    renderer = Qwen3_8PreserveThinkingRenderer(tokenizer, reasoning_effort=effort)
    expected = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        enable_thinking=thinking,
        reasoning_effort=effort,
        preserve_thinking=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    assert renderer.build_generation_prompt(messages, tools=tools, enable_thinking=thinking).to_ints() == expected
    assert (
        tokenizer.encode(
            renderer.render_conversation_text(messages, tools=tools, enable_thinking=thinking), add_special_tokens=False
        )
        == expected
    )
