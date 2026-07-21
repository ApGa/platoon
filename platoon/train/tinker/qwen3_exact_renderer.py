"""Repo-local Qwen3 renderers with exact tool-call round-tripping."""

from __future__ import annotations

import json
import re

import tinker
from tinker_cookbook.renderers import register_renderer
from tinker_cookbook.renderers.base import (
    ContentPart,
    RenderContext,
    RenderedMessage,
    TextPart,
    ToolCall,
    parse_response_for_stop_token,
    remove_thinking,
)
from tinker_cookbook.renderers.qwen3 import Qwen3InstructRenderer


_THINK_START = "<think>"
_THINK_END = "</think>"

#FIXME: is this really aligned with what qwen3 reasoning models do? Doesn't matter for qwen3-4B-instruct
def _split_qwen_reasoning_content(content: str) -> tuple[list[ContentPart], str]:
    if _THINK_END not in content:
        return [], content
    thinking_prefix, remaining = content.split(_THINK_END, 1)
    thinking = thinking_prefix.rstrip("\n").split(_THINK_START)[-1].lstrip("\n")
    remaining = remaining.lstrip("\n")
    return ([{"type": "thinking", "thinking": thinking}] if thinking else []), remaining


class Qwen3InstructExactRenderer(Qwen3InstructRenderer):
    """Qwen3 instruct renderer that preserves HF tool-call whitespace exactly."""

    def _format_tool_call(self, tool_call: ToolCall) -> str:
        payload = {
            "name": tool_call.function.name,
            "arguments": json.loads(tool_call.function.arguments),
        }
        return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"

    # Only change in this function as compared to upstream render_message is to not forcefully add a newline between content and tool-call.
    def render_message(self, message, ctx: RenderContext) -> RenderedMessage:
        maybe_newline = "\n" if ctx.idx > 0 else ""

        role = self._get_qwen_role_for_message(message)
        header_str = f"{maybe_newline}<|im_start|>{role}\n"

        content = message["content"]

        if isinstance(content, list):
            parts = content
            if self.strip_thinking_from_history and message["role"] == "assistant" and not ctx.is_last:
                parts = remove_thinking(parts)
            rendered_parts = []
            for part in parts:
                if part["type"] == "thinking":
                    rendered_parts.append(f"<think>{part['thinking']}</think>")
                elif part["type"] == "text":
                    rendered_parts.append(part["text"])
            output_content = "".join(rendered_parts)
        else:
            output_content = content

        if message["role"] == "tool":
            output_content = self._wrap_qwen_tool_response(output_content)

        if message.get("tool_calls"):
            # NOTE: The below block is mimicking Qwen3 4B Instruct chat template but causes prefix merging to fail, so removing this to speed up trainer. This is causing a lot of bottlenecks in our forward_backward with >25-30% rollouts have atleast one such instance of extra newline causing prefix merge to fail.
            # if output_content:
            #     output_content += "\n"
            output_content += "\n".join(
                [
                    self._format_tool_call(tool_call)
                    for tool_call in message["tool_calls"]
                ]
            )

        output_content += "<|im_end|>"
        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_content, add_special_tokens=False)
            )
        ]
        return RenderedMessage(header=header, output=output)

    # Differences with upstream build_generation_prompt: (1) It has a single user turn for responses to parallel tool calls whereas upstream adds one <|im_start|>user turn for each tool call. (2) For header of tool response, it does not proactively add newline at end of \n<|im_start|>user but adds before each tool response open tag. This is DIFFERENT from upstream as it only has one tool response per turn
    def build_generation_prompt(self, messages, role: str = "assistant", prefill: str | None = None):
        chunks: list[tinker.types.ModelInputChunk] = []
        if self._bos_tokens:
            chunks.append(tinker.types.EncodedTextChunk(tokens=self._bos_tokens))

        last_user_idx = max(
            (idx for idx, message in enumerate(messages) if message["role"] == "user"),
            default=-1,
        )

        for idx, message in enumerate(messages):
            ctx = RenderContext(
                idx=idx,
                is_last=(idx == len(messages) - 1),
                prev_message=messages[idx - 1] if idx > 0 else None,
                last_user_index=last_user_idx,
            )

            if message["role"] == "tool":
                prev_is_tool = idx > 0 and messages[idx - 1]["role"] == "tool"
                next_is_tool = (idx + 1) < len(messages) and messages[idx + 1]["role"] == "tool"

                if not prev_is_tool:
                    maybe_newline = "\n" if idx > 0 else ""
                    chunks.append(
                        tinker.types.EncodedTextChunk(
                            tokens=self.tokenizer.encode(
                                f"{maybe_newline}<|im_start|>user",
                                add_special_tokens=False,
                            )
                        )
                    )

                content = message["content"]
                if not isinstance(content, str):
                    content = "".join(part["text"] for part in content if part["type"] == "text")
                output_content = f"\n<tool_response>\n{content}\n</tool_response>"
                if not next_is_tool:
                    output_content += "<|im_end|>"
                chunks.append(
                    tinker.types.EncodedTextChunk(
                        tokens=self.tokenizer.encode(output_content, add_special_tokens=False)
                    )
                )
                continue

            rendered_message = self.render_message(message, ctx)
            if rendered_message.header:
                chunks.append(rendered_message.header)
            chunks.extend(
                [
                    chunk
                    for chunk in rendered_message.output
                    if not isinstance(chunk, tinker.EncodedTextChunk) or chunk.tokens
                ]
            )

        suffix_ctx = RenderContext(
            idx=len(messages),
            is_last=True,
            prev_message=messages[-1] if messages else None,
            last_user_index=last_user_idx,
        )
        suffix_tokens = self._get_generation_suffix(role, suffix_ctx)
        if suffix_tokens:
            chunks.append(tinker.types.EncodedTextChunk(tokens=suffix_tokens))

        if prefill:
            chunks.append(
                tinker.types.EncodedTextChunk(
                    tokens=self.tokenizer.encode(prefill, add_special_tokens=False)
                )
            )
        return tinker.ModelInput(chunks=chunks)

    def parse_response(self, response: list[int]):
        parsed_message, parse_success = super().parse_response(response)
        # We do not touch upstream behaviour for two cases: (1) If the model did not terminate with <|im_end|> due to exceeding max output length, (2) if the tool call parsing failed due to chat template errors. 
        if not parse_success or not parsed_message.get("unparsed_tool_calls"):
            return parsed_message, parse_success
        
        # NOTE: Below block can be simplified to just have line 1177, 178, and then just return assistant_message, parse_success.

        response = self._normalize_response_tokens(response)
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        assert isinstance(assistant_message["content"], str)
        thinking_parts, content = _split_qwen_reasoning_content(assistant_message["content"])
        if thinking_parts:
            assistant_message["content"] = thinking_parts
            if content:
                assistant_message["content"].append(TextPart(type="text", text=content))
        else:
            assistant_message["content"] = content
        return assistant_message, True


def register_qwen3_exact_renderer() -> None:
    register_renderer(
        "qwen3_instruct_exact",
        lambda tokenizer, image_processor=None: Qwen3InstructExactRenderer(tokenizer),
    )