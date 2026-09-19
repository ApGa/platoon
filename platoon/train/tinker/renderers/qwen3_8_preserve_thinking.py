"""Qwen3.8's preserve-thinking template, sharing Qwen3.6 tool parsing."""

from platoon.train.tinker.renderers.qwen3_6_preserve_thinking import (
    Qwen3_6PreserveThinkingRenderer,
    RendererError,
)


class Qwen3_8PreserveThinkingRenderer(Qwen3_6PreserveThinkingRenderer):
    """Retain historical reasoning and the snapshot's reasoning-effort preamble."""

    def __init__(self, tokenizer, image_processor=None, *, reasoning_effort="xhigh", **kwargs):
        super().__init__(tokenizer, image_processor, **kwargs)
        self.reasoning_effort = reasoning_effort

    def _contexts(self, messages, *, preserve_thinking, enable_thinking):
        enable = self.enable_thinking if enable_thinking is None else enable_thinking
        instructions = ""
        if enable:
            if self.reasoning_effort == "xhigh":
                instructions = (
                    "Reasoning effort is set to xhigh. Please think carefully through the task, "
                    "validate key assumptions, consider plausible alternatives, and prioritize "
                    "correctness, consistency, and clarity in the final answer."
                )
            elif self.reasoning_effort == "low":
                instructions = (
                    "Reasoning effort is set to low. Keep your thinking brief and focused, "
                    "moving directly to the conclusion without unnecessary elaboration."
                )
            elif self.reasoning_effort != "medium":
                raise RendererError("Qwen3.8 reasoning_effort must be xhigh, medium, or low")
        # _prepare_messages has already built the tools block. The new template
        # puts reasoning instructions before that block and the system prompt.
        if messages[0].get("role") == "system":
            content = self._render_content(messages[0].get("content"), is_system=True).strip()
            content = "\n\n".join(part for part in (instructions, content) if part)
            if content:
                messages[0] = {**messages[0], "content": content}
            else:
                messages.pop(0)
        elif instructions:
            messages.insert(0, {"role": "system", "content": instructions})
        return super()._contexts(messages, preserve_thinking=preserve_thinking, enable_thinking=enable_thinking)


class Qwen3_8PreserveThinkingDisableThinkingRenderer(Qwen3_8PreserveThinkingRenderer):
    def __init__(self, tokenizer, image_processor=None, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        super().__init__(tokenizer, image_processor, **kwargs)
