"""Fakes and constants shared by the gateway tests: a scripted sampler and the test model.

The test model is ``platoon.testing``'s toy tokenizer (byte-level, with Qwen3's chat tokens) under tinker-cookbook's
``qwen3_instruct`` renderer: the prompts and parses are a Qwen3 model's, so no model download is needed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from platoon.testing import TOY_MODEL_NAME, TOY_TOKENIZER_DIR, Completion, sample_response

# The test model's tokenizer files: tokenizer.json and tokenizer_config.json, which AutoTokenizer loads.
SNAPSHOT = TOY_TOKENIZER_DIR
MODEL_NAME = TOY_MODEL_NAME

TOOLS: list[Any] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather forecast.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]
WEATHER_CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'

__all__ = ["MODEL_NAME", "SNAPSHOT", "TOOLS", "WEATHER_CALL", "Completion", "ScriptedSampler", "sample_response"]


@dataclass
class ScriptedSampler:
    """Answers each ``sample_async`` with the next scripted completion and records what it was asked."""

    tokenizer: Any
    script: list[Completion] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def tokens(self, completion: Completion) -> list[int]:
        tokens = self.tokenizer.encode(completion.text, add_special_tokens=False)
        if completion.stop_reason == "stop":
            tokens += self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        return tokens

    async def sample_async(self, *, prompt, sampling_params, num_samples):
        self.calls.append({"prompt": prompt, "sampling_params": sampling_params, "num_samples": num_samples})
        completion = self.script.pop(0)
        if completion.delay:
            await asyncio.sleep(completion.delay)
        return sample_response(
            self.tokens(completion), stop_reason=completion.stop_reason, routing_id=completion.routing_id
        )

    def sample(self, *, prompt, sampling_params, num_samples):
        raise AssertionError("the gateway samples asynchronously")
