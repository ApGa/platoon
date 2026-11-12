"""Function harnesses: an async Python function that runs one agent, registered with ``@pt.harness``.

```python
import platoon as pt

@pt.harness("my-agent", version="2")
async def my_agent(ctx: pt.AgentContext) -> pt.AgentResult:
    reply = await ctx.chat([{"role": "user", "content": ctx.instruction}])
    return pt.AgentResult("completed", final_message=reply.text)
```

The decorator returns the function unchanged and registers a :class:`FunctionHarness` under ``name@version``,
so environment configs name it exactly like a TOML ``HarnessSpec`` (``harness: my-agent``) and
``platoon.harness.registry.get_harness`` resolves either kind.  ``runtime="inline"`` (the only runtime today)
runs the function in the rollout worker's event loop, next to the gateway.  ``executes_model_code`` declares
that the harness runs code the model wrote (CodeAct, OpenHands); it is recorded with every rollout so a site
policy can refuse such harnesses outside a sandbox later.  ``budget_enforcement="gateway"`` has the model gateway,
rather than the function, stop each of its agents at its step budget.
"""

from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, get_args

from platoon import extensions
from platoon.harness.agent import AgentFn
from platoon.harness.spec import BudgetEnforcement

logger = logging.getLogger(__name__)

Runtime = Literal["inline"]
RUNTIMES: tuple[Runtime, ...] = ("inline",)
# The attribute the decorator sets on the function it registers.
HARNESS_ATTRIBUTE = "__platoon_harness__"


@dataclass(frozen=True)
class FunctionHarness:
    """A harness implemented by an async Python function (:data:`platoon.harness.agent.AgentFn`)."""

    name: str
    version: str
    # Where the function runs; ``inline`` runs it in the rollout worker's event loop.
    runtime: Runtime
    # The harness runs code the model wrote.
    executes_model_code: bool
    # The decorated function.
    fn: AgentFn = field(repr=False, compare=False)
    # Who stops each of its agents at its step budget: ``harness``, the function itself (``ctx.budget``,
    # ``out_of_steps``), or ``gateway``, the model gateway (RFC 0009, option A).
    budget_enforcement: BudgetEnforcement = "harness"
    # The options that hold paths of files the harness reads, which load beside the config file that gives them.
    path_options: tuple[str, ...] = ()

    @property
    def ref(self) -> str:
        """``name@version``, the key environment configs and trajectories use."""
        return f"{self.name}@{self.version}"

    @property
    def source(self) -> str:
        """``module:qualname`` of the function."""
        return f"{self.fn.__module__}:{self.fn.__qualname__}"


_lock = threading.Lock()
_function_harnesses: dict[str, FunctionHarness] = {}


def harness(
    name: str,
    *,
    version: str = "1",
    runtime: Runtime = "inline",
    executes_model_code: bool = False,
    budget_enforcement: BudgetEnforcement = "harness",
    path_options: Sequence[str] = (),
) -> Callable[[AgentFn], AgentFn]:
    """Register the decorated ``async def fn(ctx: AgentContext) -> AgentResult`` as harness ``name@version``.

    Environment configs then name it like a TOML harness spec (``harness: name``).  The name follows platoon's
    naming rule (:mod:`platoon.extensions`): lower case, ``<owner>-<name>``, with ``platoon-*`` reserved for core.
    ``runtime="inline"``, the only runtime, runs the function in the rollout worker's event loop.
    ``executes_model_code`` declares that the harness runs code the model wrote; it is recorded with every
    rollout.  ``budget_enforcement="gateway"`` has the model gateway hold each of the harness's agents to its step
    budget, for a harness that cannot honour ``ctx.budget`` itself, such as one that runs another agent program over
    HTTP: once ``out_of_steps`` holds for the agent, counting the ``policy`` calls of the agents it minted over HTTP
    with its own, a ``policy`` call gets the budget stop, never sampled, and a run that then completes is graded
    ``limit``, with reason ``budget`` (RFC 0009, option A).  ``aux`` and ``judge`` calls stay outside the budget.  A
    rollout whose root has no step budget (``rollout_config.max_steps`` or the task's) is refused, with a
    ``harness_config`` error, rather than run unbounded.  ``path_options``
    names the options that hold paths of files the harness reads: a relative one that a config file gives is made
    absolute against the file's directory when the config loads, as a service's ``path_options`` are (RFC 0016,
    section 1).  The decorator returns the function unchanged::

        @pt.harness("my-agent", version="2")
        async def my_agent(ctx: pt.AgentContext) -> pt.AgentResult:
            reply = await ctx.chat([{"role": "user", "content": ctx.instruction}])
            return pt.AgentResult("completed", final_message=reply.text)
    """

    extensions.check_name(name, what="harness")
    version = extensions.check_version(version, what=f"harness {name!r}")
    if runtime not in RUNTIMES:
        raise ValueError(f"Harness {name!r}: runtime must be one of {RUNTIMES}, got {runtime!r}")
    if budget_enforcement not in get_args(BudgetEnforcement):
        raise ValueError(
            f"Harness {name!r}: budget_enforcement must be one of {get_args(BudgetEnforcement)}, "
            f"got {budget_enforcement!r}"
        )
    if isinstance(path_options, str) or not all(isinstance(option, str) and option for option in path_options):
        raise ValueError(f"Harness {name!r}: path_options must be a sequence of option names, got {path_options!r}")

    def decorate(fn: AgentFn) -> AgentFn:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@harness({name!r}) needs an async function, got {fn!r}")
        extensions.check_registration("harness", name, module=getattr(fn, "__module__", None))
        register_function_harness(
            FunctionHarness(
                name=name,
                version=version,
                runtime=runtime,
                executes_model_code=bool(executes_model_code),
                fn=fn,
                budget_enforcement=budget_enforcement,
                path_options=tuple(path_options),
            )
        )
        setattr(fn, HARNESS_ATTRIBUTE, _function_harnesses[f"{name}@{version}"])
        return fn

    return decorate


def register_function_harness(record: FunctionHarness) -> None:
    """Add ``record`` to the process registry.

    Registering the same function again (a re-imported module) replaces the record; a different function under
    an existing ``name@version`` is an error.
    """

    with _lock:
        existing = _function_harnesses.get(record.ref)
        if existing is not None and existing.source != record.source:
            raise ValueError(f"Harness {record.ref} is defined twice: {existing.source} and {record.source}")
        _function_harnesses[record.ref] = record


def function_harnesses() -> list[FunctionHarness]:
    """Every registered function harness, by reference."""

    with _lock:
        return [_function_harnesses[ref] for ref in sorted(_function_harnesses)]


def harness_of(fn: object) -> FunctionHarness | None:
    """The :class:`FunctionHarness` a decorated function was registered as, or None."""

    record = getattr(fn, HARNESS_ATTRIBUTE, None)
    return record if isinstance(record, FunctionHarness) else None
