from __future__ import annotations

from functools import wraps
from typing import Any


def _patch_remote_inf_engine_proxy_resolution() -> None:
    """Let custom RolloutWorkflow instances receive worker-local proxy URLs.

    Upstream AReaL already threads a per-worker ``proxy_addr`` through
    ``RemoteInfEngine.submit()``, but only injects it when wrapping agent-like
    workflows in ``OpenAIProxyWorkflow``. Platoon's custom workflows are already
    ``RolloutWorkflow`` instances, so inline mode needs a small patch to bind the
    same worker-local proxy URL onto those workflow objects before execution.
    """

    from areal.api import RolloutWorkflow
    from areal.infra.remote_inf_engine import RemoteInfEngine

    original = RemoteInfEngine._resolve_workflow
    if getattr(original, "__platoon_proxy_patch__", False):
        return

    def _inject_proxy_addr(workflow: RolloutWorkflow, proxy_addr: str) -> None:
        setter = getattr(workflow, "set_proxy_base_url", None)
        if callable(setter):
            setter(proxy_addr)
            return
        if hasattr(workflow, "proxy_base_url"):
            setattr(workflow, "proxy_base_url", proxy_addr)

    @wraps(original)
    def _resolve_workflow_with_proxy_addr(
        self,
        workflow: Any,
        workflow_kwargs: dict[str, Any] | None,
        group_size: int = 1,
        proxy_addr: str | None = None,
    ) -> RolloutWorkflow:
        resolved = original(
            self,
            workflow,
            workflow_kwargs,
            group_size=group_size,
            proxy_addr=proxy_addr,
        )
        if proxy_addr is not None and isinstance(resolved, RolloutWorkflow):
            _inject_proxy_addr(resolved, proxy_addr)
        return resolved

    _resolve_workflow_with_proxy_addr.__platoon_proxy_patch__ = True
    RemoteInfEngine._resolve_workflow = _resolve_workflow_with_proxy_addr


def apply_all_patches() -> None:
    """Apply Platoon compatibility patches for the current AReaL release."""

    _patch_remote_inf_engine_proxy_resolution()
