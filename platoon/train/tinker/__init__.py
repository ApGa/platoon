"""The policy client (``proxy``), its renderers, the Tinker service's refusal of a key (``key_refusal``) and the
rollout timing (``rollout_metrics``).

Before platoon 0.5.0 this package held the trainer, its configuration and the rollout workflows; 0.5.0 moved them into
the stage packages of :mod:`platoon.train` (``trainer``, ``step``, ``dispatch``, ``workflow``, ``credit``, ``workers``,
``checkpoints``, ``evaluation``), and 0.6.0 removed their old paths (:data:`REMOVED`).  Importing an old module, or
reading a name that this package or an old package exported, raises ``ImportError`` with what replaces it, and running
an old module with ``python -m`` prints the same and exits 78, a configuration error.  An old package still imports,
empty."""

import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from collections.abc import Sequence

from platoon.deprecations import removed_hint

# The names this package exported before platoon 0.6.0.
_NAMES = (
    "PlatoonTinkerRLTrainer",
    "Watchdog",
    "BatchTransform",
    "BatchTransformContext",
    "DepthLevelWeightingTransform",
    "build_default_batch_transforms",
    "PlatoonTinkerRLTrainerConfig",
    "TrainConfig",
    "EvalConfig",
    "CheckpointConfig",
    "StatsConfig",
    "WatchdogConfig",
    "AdamParams",
    "WorkflowConfig",
)
#: Each module path and name that platoon 0.6.0 removed from this package, relative to it, with what replaces it.
REMOVED = {
    **dict.fromkeys(_NAMES, "platoon.train"),
    "batch_transforms": "platoon.train.step.batch_transforms",
    "censoring": "platoon.train.step.metrics",
    "checkpoint_pruning": "platoon.train.checkpoints.pruning",
    "config_defs": "platoon.train.config_defs",
    "dataset_order": "platoon.train.dataset_order",
    "dispatcher": "platoon.train.dispatch.dispatcher, and platoon.train.checkpoints.dispatch_state for its state file",
    "evaluation_record": "platoon.train.evaluation.record",
    "kept_groups": "platoon.train.checkpoints.kept_groups",
    "permits": "platoon.train.dispatch.permits",
    "rl": (
        "platoon.train for PlatoonTinkerRLTrainer and Watchdog, and the modules of platoon.train.trainer, .policy, "
        ".step, .dispatch, .checkpoints and .evaluation, which docs/training-code.md maps"
    ),
    "staleness": "platoon.train.step.staleness",
    "start_limit": "platoon.train.workflow.start_limit",
    "task_records": "platoon.train.task_records",
    "train": "platoon.train.launch (python -m platoon.train.launch)",
    "worker_env": "platoon.train.workers.env",
    "workers": "platoon.train.workers",
    "workers.binding": "platoon.train.workers.binding",
    "workers.entrypoint": "platoon.train.workers.entrypoint (python -m platoon.train.workers.entrypoint)",
    "workers.protocol": "platoon.train.workers.protocol",
    "workers.registry": "platoon.train.workers.registry",
    "workers.run": "platoon.train.workers.run",
    "workers.supervisor": "platoon.train.workers.supervisor",
    "workers.telemetry": "platoon.train.workers.telemetry",
    "workers.workload": "platoon.train.workers.workload",
    "workflows": "platoon.train.workflow",
    "workflows.base": "platoon.train.workflow.protocol",
    "workflows.group_rollout_workflow": "platoon.train.workflow",
}
# The old packages, which still import so that their old modules can say what replaces them.
_PACKAGES = frozenset({"workers", "workflows"})


def removed_message(path: str) -> str:
    """What platoon says of ``platoon.train.tinker.<path>``, a removed path or a name in an old package."""

    found = path
    while found not in REMOVED:
        found = found.rpartition(".")[0]
    return f"{__name__}.{path}: {removed_hint('0.6.0', REMOVED[found])}"


def _read(package: str, name: str) -> object:
    """Read ``name`` of ``platoon.train.tinker`` (``package`` "") or of an old package: a removed name raises
    ``ImportError``."""

    path = f"{package}.{name}" if package else name
    if name.startswith("__") or not (path in REMOVED or package in _PACKAGES):
        module = f"{__name__}.{package}" if package else __name__
        raise AttributeError(f"module {module!r} has no attribute {name!r}")
    raise ImportError(removed_message(path), name=f"{__name__}.{path}")


def __getattr__(name: str) -> object:
    return _read("", name)


class _RemovedPaths(importlib.abc.MetaPathFinder, importlib.abc.InspectLoader):
    """The finder and loader of the removed module paths, which no file holds."""

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: types.ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        old = fullname.removeprefix(f"{__name__}.")
        if old == fullname or old not in REMOVED or old in _NAMES:
            return None
        return importlib.util.spec_from_loader(fullname, self, is_package=old in _PACKAGES)

    def is_package(self, fullname: str) -> bool:
        return fullname.removeprefix(f"{__name__}.") in _PACKAGES

    def get_source(self, fullname: str) -> str:
        """An old module's code: it raises ``ImportError``, or, run with ``python -m``, prints the same and exits 78."""

        message = removed_message(fullname.removeprefix(f"{__name__}."))
        return (
            "import sys\n"
            "if __name__ == '__main__':\n"
            f"    print({'platoon: python -m ' + message!r}, file=sys.stderr)\n"
            "    sys.exit(78)\n"
            f"raise ImportError({message!r}, name={fullname!r})\n"
        )

    def exec_module(self, module: types.ModuleType) -> None:
        package = module.__name__.removeprefix(f"{__name__}.")
        if package in _PACKAGES:
            module.__dict__["__getattr__"] = lambda name: _read(package, name)
        else:
            super().exec_module(module)


if not any(isinstance(finder, _RemovedPaths) for finder in sys.meta_path):
    sys.meta_path.append(_RemovedPaths())
