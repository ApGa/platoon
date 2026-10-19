"""Harness layer: agent loops run against the model gateway, as data (TOML specs) or as code (functions).

``spec`` loads and renders ``HarnessSpec`` TOML files and ``runner`` launches one inside a task sandbox or as a
local subprocess.  ``agent`` defines the agent contract function harnesses code against, ``function`` the
``@harness`` decorator, and ``inline`` runs a function harness in process.  ``agents``, ``budget`` and
``verify`` provide subagents, step budgets and verifier children.  ``registry`` resolves a harness name to
either kind, and ``rollout`` provides ``harness_rollout``, the ``RolloutFn`` that ties an env session, the
gateway and a harness run into a trajectory collection.  ``builtin`` holds the built-in harnesses:
``platoon-codeact``, ``platoon-react``, ``platoon-bash`` and the ``mini-swe-agent`` spec.

``import platoon as pt; @pt.harness("name")`` is the decorator: ``platoon.harness`` is this package, and
calling the package calls :func:`platoon.harness.function.harness`.  A plain function attribute on ``platoon``
could not work, because importing any ``platoon.harness.*`` module rebinds ``platoon.harness`` to the package.
"""

import sys
import types
from typing import Any

from platoon.harness.registry import get_harness, get_harness_spec
from platoon.harness.spec import HarnessSpec, HarnessSpecError, load_spec


class _HarnessPackage(types.ModuleType):
    def __call__(self, name: str, **kwargs: Any) -> Any:
        from platoon.harness.function import harness

        return harness(name, **kwargs)


sys.modules[__name__].__class__ = _HarnessPackage

__all__ = ["HarnessSpec", "HarnessSpecError", "get_harness", "get_harness_spec", "load_spec"]
