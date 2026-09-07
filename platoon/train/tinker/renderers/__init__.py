"""Platoon-side tinker_cookbook renderers.

Call :func:`register` before ``tinker_cookbook.renderers.get_renderer`` so that
``train.renderer_name`` can name ``qwen3_6_preserve_thinking`` or
``qwen3_6_preserve_thinking_disable_thinking``.
"""

from platoon.train.tinker.renderers.qwen3_6_preserve_thinking import (
    RENDERER_FACTORIES,
    RENDERER_NAME,
    RENDERER_NAME_DISABLE_THINKING,
    Qwen3_6PreserveThinkingDisableThinkingRenderer,
    Qwen3_6PreserveThinkingRenderer,
    Qwen3_6RenderContext,
    register,
)

__all__ = [
    "RENDERER_FACTORIES",
    "RENDERER_NAME",
    "RENDERER_NAME_DISABLE_THINKING",
    "Qwen3_6PreserveThinkingDisableThinkingRenderer",
    "Qwen3_6PreserveThinkingRenderer",
    "Qwen3_6RenderContext",
    "register",
]
