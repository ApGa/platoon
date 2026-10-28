"""The step hooks a config attaches to the rl trainer (``trainer_options.hooks``): what each item names
(:class:`HookSpec`, read by :func:`hook_specs` when the config loads), and the hooks built from them and registered
with the run (:func:`attach_hooks`, as the trainer checks, before any service opens).

Each item is a reference to a hook class, ``pkg.mod:Hook`` or a name a package registered (kind ``step_hook``), or
``{hook: <ref>, options: {...}}``.  The class builds the hook with ``from_run(run, options)``: ``StepHookBase``'s
takes no options.  Imports no Tinker.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from platoon.train.trainer.run import TrainerRun


@dataclass(frozen=True)
class HookSpec:
    """A step hook that the config attaches (an item of ``trainer_options.hooks``): its class's reference, and the
    options it is built with."""

    ref: str
    options: Mapping[str, Any] = field(default_factory=dict)


def hook_specs(items: Any) -> tuple[HookSpec, ...]:
    """The hooks that ``trainer_options.hooks`` names: each item a reference to a hook class, or ``{hook: <ref>,
    options: {...}}``; a ValueError names the first item that is neither."""

    if not isinstance(items, (list, tuple)):
        raise ValueError(f"hooks: expected a list of step hooks, got {items!r}")
    specs = []
    for index, item in enumerate(items):
        if isinstance(item, str) and item:
            specs.append(HookSpec(item))
        elif isinstance(item, Mapping) and _names_a_hook(item):
            specs.append(HookSpec(item["hook"], dict(item.get("options") or {})))
        else:
            raise ValueError(
                f"hooks[{index}]: expected a step hook (pkg.mod:Hook) or {{hook: pkg.mod:Hook, options: {{...}}}}, "
                f"got {item!r}"
            )
    return tuple(specs)


def _names_a_hook(item: Mapping[str, Any]) -> bool:
    """Whether ``item`` is ``{hook: <ref>, options: {...}}``, its options left out or a mapping."""

    ref, options = item.get("hook"), item.get("options")
    known = set(item) <= {"hook", "options"}
    return known and isinstance(ref, str) and bool(ref) and (options is None or isinstance(options, Mapping))


def attach_hooks(specs: Sequence[HookSpec], run: "TrainerRun") -> None:
    """Build each hook that ``specs`` name, its class resolved as the config's other components are (a name a package
    registered, else an import path: :func:`platoon.registry.resolve_component`), with ``from_run(run, options)``, and
    register it with the run (``run.hooks.register``, which makes one with keys a participant of its checkpoints).  A
    ``ConfigError`` (exit 78) names each hook that does not resolve, build or register, by its item."""

    from platoon.utils.config import ConfigError

    problems = []
    for index, spec in enumerate(specs):
        try:
            run.hooks.register(_built(spec, run))
        except ConfigError as exc:  # a key the hook may not hold (run.checkpoints.register): its problems, by item
            problems.extend(f"trainer_options.hooks[{index}]: {problem}" for problem in exc.problems)
        except ValueError as exc:
            problems.append(f"trainer_options.hooks[{index}]: {exc}")
    if problems:
        raise ConfigError(problems)


def _built(spec: HookSpec, run: "TrainerRun") -> Any:
    """The hook ``spec`` names, built with its class's ``from_run``; a ValueError says which part failed, whatever the
    error of the import or of ``from_run``, which it chains."""

    from platoon.registry import resolve_component

    try:
        cls = resolve_component("step_hook", spec.ref)
    except Exception as exc:  # any error of the import names the item (exit 78)
        raise ValueError(f"{spec.ref!r} names no step hook ({type(exc).__name__}: {exc})") from exc
    if getattr(cls, "from_run", None) is None:
        raise ValueError(f"{spec.ref} has no from_run(run, options): subclass platoon.experimental.StepHookBase")
    try:
        return cls.from_run(run, spec.options)
    except Exception as exc:  # any error of from_run (a missing option: KeyError) names the item (exit 78)
        raise ValueError(f"{spec.ref} cannot be built ({type(exc).__name__}: {exc})") from exc
