"""Tokenizers and renderers built ahead (``proxy.prepare_renderer``, what a rollout worker fork server builds once
for the workers it forks): taken once, by the first build asking for the same."""

from __future__ import annotations

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)
from platoon.train.tinker import proxy


def test_a_renderer_built_ahead_is_taken_once_by_a_build_of_the_same(monkeypatch):
    built = []

    def build(model_name, renderer_name, renderer_kwargs):
        built.append((model_name, renderer_name, renderer_kwargs))
        return object(), object()

    monkeypatch.setattr(proxy, "_build_renderer", build)
    monkeypatch.setattr(proxy, "_prepared", {})
    proxy.prepare_renderer("org/model", "chat_template", {"preserve_thinking": False})
    assert len(built) == 1
    ahead = proxy._prepared[("org/model", "chat_template", '{"preserve_thinking": false}')]

    assert proxy.build_renderer("org/model", "chat_template", {"preserve_thinking": False}) == ahead
    assert len(built) == 1  # taken, not built
    assert proxy.build_renderer("org/model", "chat_template", {"preserve_thinking": False}) != ahead
    assert len(built) == 2  # once only: the next build builds

    proxy.prepare_renderer("org/model", "chat_template", {"preserve_thinking": False})
    proxy.build_renderer("org/model", "chat_template", {"preserve_thinking": True})  # other arguments: built
    proxy.build_renderer("org/other", "chat_template", {"preserve_thinking": False})
    assert len(built) == 5 and len(proxy._prepared) == 1
