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
)
from platoon.train.tinker.renderers.qwen3_6_preserve_thinking import (
    register as _register_qwen36,
)
from platoon.train.tinker.renderers.qwen3_8_preserve_thinking import (
    Qwen3_8PreserveThinkingDisableThinkingRenderer,
    Qwen3_8PreserveThinkingRenderer,
)


def register(force: bool = False) -> list[str]:
    RENDERER_FACTORIES.update(
        {
            "qwen3_8_preserve_thinking": Qwen3_8PreserveThinkingRenderer,
            "qwen3_8_preserve_thinking_disable_thinking": Qwen3_8PreserveThinkingDisableThinkingRenderer,
        }
    )
    return _register_qwen36(force=force)


__all__ = [
    "RENDERER_FACTORIES",
    "RENDERER_NAME",
    "RENDERER_NAME_DISABLE_THINKING",
    "Qwen3_6PreserveThinkingDisableThinkingRenderer",
    "Qwen3_6PreserveThinkingRenderer",
    "Qwen3_6RenderContext",
    "Qwen3_8PreserveThinkingRenderer",
    "Qwen3_8PreserveThinkingDisableThinkingRenderer",
    "register",
]
