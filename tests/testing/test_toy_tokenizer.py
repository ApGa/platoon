"""platoon.testing's toy tokenizer: byte-level ids, Qwen3's chat tokens, and the Qwen3 renderers over it."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from huggingface_hub.constants import HF_HUB_CACHE

from platoon.testing import TOY_MODEL_NAME, TOY_TOKENIZER_DIR, tool_call, toy_renderer, toy_tokenizer
from platoon.testing.tokenizer import SPECIAL_TOKENS, build_toy_tokenizer, byte_characters

# The Qwen3 tokenizer the toy one stands in for, when it is on disk.
QWEN = Path(
    os.environ.get(
        "QWEN3_4B_INSTRUCT_SNAPSHOT",
        Path(HF_HUB_CACHE) / "models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554",
    )
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "guess",
            "description": "Guess the number.",
            "parameters": {"type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]},
        },
    }
]
CONVERSATION = [
    {"role": "system", "content": "You are a careful guesser."},
    {"role": "user", "content": "Find the secret number from 1 to 100. Café ☕ — 数字?"},
]


def test_token_ids_are_bytes_then_qwens_chat_tokens():
    tokenizer = toy_tokenizer()

    assert tokenizer.encode("Hi\n", add_special_tokens=False) == [72, 105, 10]
    assert tokenizer.encode("é", add_special_tokens=False) == list("é".encode())
    for token_id, (content, _special) in enumerate(SPECIAL_TOKENS, start=256):
        assert tokenizer.encode(content, add_special_tokens=False) == [token_id]
    assert len(tokenizer) == 256 + len(SPECIAL_TOKENS)
    assert (tokenizer.eos_token, tokenizer.pad_token, tokenizer.bos_token) == ("<|im_end|>", "<|endoftext|>", None)


def test_special_flags_follow_qwen3():
    tokenizer = toy_tokenizer()
    text = "<|im_start|>user\n<think>a</think><tool_call>b</tool_call><|im_end|>"

    # Like Qwen3's, only the chat markers are special: the tool-call and thinking tags survive skip_special_tokens.
    assert (
        tokenizer.decode(tokenizer.encode(text), skip_special_tokens=True)
        == "user\n<think>a</think><tool_call>b</tool_call>"
    )


@pytest.mark.parametrize(
    "text",
    ["", "plain ascii", "unicode: café ☕ 数字 🎲", "controls \x00\x01\x7f\t\r\n", "<|im_start|>x<|im_end|><think>"],
)
def test_any_text_round_trips(text):
    tokenizer = toy_tokenizer()

    assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == text


def test_the_shipped_files_are_the_builders_and_about_7_kb(tmp_path):
    shipped = json.loads((TOY_TOKENIZER_DIR / "tokenizer.json").read_text())
    built = json.loads((build_toy_tokenizer(tmp_path) / "tokenizer.json").read_text())

    for part in ("model", "added_tokens", "pre_tokenizer", "decoder", "normalizer"):
        assert shipped[part] == built[part], part
    assert shipped["model"]["vocab"] == {character: byte for byte, character in byte_characters().items()}
    assert (TOY_TOKENIZER_DIR / "tokenizer_config.json").read_text() == (tmp_path / "tokenizer_config.json").read_text()
    assert sum(path.stat().st_size for path in TOY_TOKENIZER_DIR.iterdir()) < 8 * 1024


def test_auto_tokenizer_and_tinker_cookbook_load_it():
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from transformers import AutoTokenizer

    loaded = AutoTokenizer.from_pretrained(str(TOY_TOKENIZER_DIR))

    tokenizer = toy_tokenizer()  # registers it with tinker-cookbook, in a process whose tests had not loaded it yet
    assert get_tokenizer(TOY_MODEL_NAME) is tokenizer
    assert loaded.encode("<|im_start|>hi", add_special_tokens=False) == [257, 104, 105]


def test_qwen3_instruct_renders_the_qwen3_chat_format_and_parses_tool_calls():
    tokenizer, renderer = toy_tokenizer(), toy_renderer()

    prompt = tokenizer.decode(renderer.build_generation_prompt(CONVERSATION).to_ints())
    message, parsed = renderer.parse_response(
        tokenizer.encode(tool_call("guess", number=50) + "<|im_end|>", add_special_tokens=False)
    )

    assert prompt == (
        "<|im_start|>system\nYou are a careful guesser.<|im_end|>\n"
        "<|im_start|>user\nFind the secret number from 1 to 100. Café ☕ — 数字?<|im_end|>\n<|im_start|>assistant\n"
    )
    assert parsed and message["tool_calls"][0].function.name == "guess"
    assert json.loads(message["tool_calls"][0].function.arguments) == {"number": 50}


def test_qwen3_parses_thinking():
    tokenizer, renderer = toy_tokenizer(), toy_renderer("qwen3")

    message, parsed = renderer.parse_response(
        tokenizer.encode("<think>low or high?</think>Fifty.<|im_end|>", add_special_tokens=False)
    )

    assert parsed and message["content"] == [
        {"type": "thinking", "thinking": "low or high?"},
        {"type": "text", "text": "Fifty."},
    ]


@pytest.mark.skipif(
    not (QWEN / "tokenizer.json").exists(), reason="the Qwen3-4B-Instruct-2507 tokenizer is not on disk"
)
@pytest.mark.parametrize("name", ["qwen3_instruct", "qwen3", "qwen3_disable_thinking"])
def test_the_toy_renders_the_text_qwen3s_tokenizer_renders(name):
    from tinker_cookbook.renderers import get_renderer
    from transformers import AutoTokenizer

    from platoon.gateway.core import to_cookbook_tool_specs

    qwen = AutoTokenizer.from_pretrained(str(QWEN))
    history = [
        *CONVERSATION,
        {"role": "assistant", "content": tool_call("guess", number=50)},
        {"role": "tool", "content": "lower"},
    ]

    def rendered(tokenizer, renderer):
        prefix = renderer.create_conversation_prefix_with_tools(to_cookbook_tool_specs(TOOLS), system_prompt="Guess.")
        return tokenizer.decode(renderer.build_generation_prompt([*prefix, *history[1:]]).to_ints())

    assert rendered(toy_tokenizer(), toy_renderer(name)) == rendered(qwen, get_renderer(name, qwen))
