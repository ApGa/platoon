"""Helpers for workflows that need explicit remote reconstruction."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from areal.api import WorkflowLike


@runtime_checkable
class RemoteWorkflowSerializable(Protocol):
    """Workflow objects that can describe how workers should reconstruct them."""

    def to_remote_workflow(self) -> tuple[WorkflowLike | None, dict[str, Any] | None]:
        """Return the workflow reference and kwargs used on remote workers."""


def normalize_remote_workflow(
    workflow: WorkflowLike | None,
    workflow_kwargs: dict[str, Any] | None,
) -> tuple[WorkflowLike | None, dict[str, Any] | None]:
    """Convert opt-in workflow instances into a remotely reconstructible form."""

    if isinstance(workflow, RemoteWorkflowSerializable):
        remote_workflow, remote_kwargs = workflow.to_remote_workflow()
        if workflow_kwargs:
            merged_kwargs = dict(remote_kwargs or {})
            merged_kwargs.update(workflow_kwargs)
            return remote_workflow, merged_kwargs
        return remote_workflow, remote_kwargs
    return workflow, workflow_kwargs


def callable_import_path(fn: Callable) -> str | None:
    """Return an import path for a callable, including script-run functions.

    Training scripts are often executed as ``__main__``. AReaL workers cannot
    import ``__main__.foo``, so recover a package/module path from ``__file__``
    when possible.
    """

    name = getattr(fn, "__name__", "")
    module = getattr(fn, "__module__", "")
    if not name or name == "<lambda>":
        return None
    if module and module != "__main__":
        return f"{module}.{name}"

    file_name = fn.__globals__.get("__file__")
    if not file_name:
        return None

    path = Path(file_name).resolve()
    candidates: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            rel_path = path.relative_to(Path(entry).resolve())
        except ValueError:
            continue
        if rel_path.suffix != ".py":
            continue
        parts = list(rel_path.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            candidates.append(".".join(parts))

    # Prefer package-qualified paths over script-directory aliases. Do not
    # import candidates here: training modules can have expensive AReaL imports,
    # and workers will validate the path when reconstructing the workflow.
    candidates.sort(key=lambda candidate: (not candidate.startswith("platoon."), len(candidate)))
    if candidates:
        return f"{candidates[0]}.{name}"
    return None
