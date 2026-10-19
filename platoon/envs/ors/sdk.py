"""Version-tolerant access to the openreward SDK, from 0.1.125 to 0.1.163 and later.

The two versions platoon runs against differ where it matters here:

* 0.1.125 has ``tool(fn=None, *, shared=True)`` only. An episode ends when a
  tool returns ``ToolOutput(finished=True)``; ``RunToolError`` has no
  ``reason``; a tool that raises reaches a remote client as a plain
  ``RuntimeError``.
* 0.1.163 adds ``@terminal`` (the submit tool, hidden from ``list_tools`` and
  reported as ``ListToolsOutput.terminal_tool``), ``@tool(concurrent=...,
  exclusive=...)``, a per-instance finished latch, ``get_messages`` and group
  scoring. A tool that raises reaches a remote client as ``ToolFailed`` and
  kills the session.

Either way an env server reports a tool that raised by its text alone, which
:func:`infra_error` reads as an in-process exception's class is read.

Nothing here compares version numbers: every difference is feature-detected.
Importing this module does not import the SDK.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import dataclasses
import importlib
import inspect
import logging
import os
import pickle
import re
import sys
import threading
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

TOOL_ROUTING_META_KEY = "openhands.dev/tool-routing"
TOOL_ROUTING_SCHEMA_KEY = "x-openhands-tool-routing"  # the routing for SDKs whose ToolSpec carries no metadata
TOOL_ERROR_META_KEY = "openhands.dev/tool-error"  # an env-declared tool error in ToolOutput.metadata
TERMINAL_TOOL_META_KEY = "openreward.dev/terminal-tool"  # {"arg": name | None} for an SDK @terminal tool
READ_ONLY_META_KEYS = ("platoon.dev/read-only", "readOnlyHint")
# Where a remote listing may carry a tool's meta: keys of a provider-format tool dict, or attributes of the client's
# ToolSpec.  openreward's env server sends none (its ToolSpec forbids extra keys, up to 0.1.163 at least).
LISTED_META_KEYS = ("_meta", "meta", "metadata")
# The SDK modules an in-process env runs on.  Each has a module-level structlog ``logger`` that prints to stdout,
# and each imports the SDK's ``run_user_callable`` by name.
IN_PROCESS_RUNTIME_MODULES = ("openreward.environments.environment", "openreward.environments.session")
# Set explicitly, the SDK's own log level wins over quiet_in_process_logs.
SDK_LOG_LEVEL_VAR = "OPENREWARD_LOG_LEVEL"
# The env methods platoon calls on the live env, so only in process: a shared child's view, a copy, a grade from state.
HOOK_ATTRIBUTES = ("platoon_fork", "platoon_copy", "platoon_grade")

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def import_sdk() -> ModuleType:
    """``openreward.environments``, with the SDK's update check off (it would phone home from every worker) and
    its sync env calls on platoon's threads (:func:`run_user_callable`)."""

    os.environ.setdefault("OPENREWARD_DISABLE_UPDATE_CHECK", "1")
    try:
        import openreward.environments as environments
    except ImportError as exc:
        raise ImportError(
            "ORS environments need the openreward SDK: install the ors extra, e.g. pip install 'platoon[ors]', or in a "
            "platoon checkout uv sync --extra tinker --extra ors (uv sync removes the extras it is not given)"
        ) from exc
    _run_sdk_calls_on_platoon_threads()
    return environments


def _run_sdk_calls_on_platoon_threads() -> None:
    """Point the SDK's own ``run_user_callable`` (sync tools, task tools and the class task cache all go through
    it) at platoon's, so its sync env code also runs on daemon threads.  Both supported SDKs import the helper by
    name into these modules."""

    for name in IN_PROCESS_RUNTIME_MODULES:
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "run_user_callable"):
            module.run_user_callable = run_user_callable


def quiet_in_process_logs(level: int = logging.WARNING) -> None:
    """Drop the SDK's log lines below ``level`` in this process, for envs that run in it (``python://``).

    The SDK's loggers print to stdout, and its env runtime logs ``tool_call_completed`` at INFO after every tool
    call: server telemetry.  In process, that line lands wherever the harness reads stdout; a CodeAct cell captures
    stdout while its tool calls run, so the line reached the policy's observation.  Remote envs log in their
    server's process and are unaffected.  An explicit ``OPENREWARD_LOG_LEVEL`` wins.  Idempotent.
    """

    if os.environ.get(SDK_LOG_LEVEL_VAR):
        return
    import_sdk()
    for name in IN_PROCESS_RUNTIME_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:  # pragma: no cover - both supported SDKs have both modules
            continue
        sdk_logger = getattr(module, "logger", None)
        quieted = _filtering_at(sdk_logger, level)
        if quieted is not sdk_logger:
            module.logger = quieted  # type: ignore[attr-defined]


def _filtering_at(sdk_logger: Any, level: int) -> Any:
    """``sdk_logger`` (a structlog filtering logger, as the SDK builds them) rebuilt to drop lines below ``level``,
    printing where it printed; ``sdk_logger`` itself when it already filters them, or is no such logger."""

    effective_level = getattr(sdk_logger, "get_effective_level", None)
    if not callable(effective_level) or effective_level() >= level:
        return sdk_logger
    import structlog

    try:
        wrapped, processors, context = sdk_logger._logger, sdk_logger._processors, sdk_logger._context
    except AttributeError:  # pragma: no cover - every structlog the SDKs pin has them
        return sdk_logger
    return structlog.make_filtering_bound_logger(level)(wrapped, processors, context)


def import_client() -> ModuleType:
    """The top-level ``openreward`` package, for the HTTP client."""

    import_sdk()
    import openreward

    return openreward


async def run_user_callable(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call env code the way the SDK's server does: coroutines on the loop, sync code in a thread.

    Sync code runs on a daemon thread of its own (:func:`run_in_thread`), not on the loop's default executor: a
    call that hangs and is abandoned (its caller timed out) holds only its own thread, so it can neither fill
    the executor that ``to_thread`` and DNS lookups (``loop.getaddrinfo``) share nor keep the process from
    exiting.
    """

    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    result = await run_in_thread(fn, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


_abandoned_lock = threading.Lock()
_abandoned_calls = 0


class _ThreadCall:
    """One :func:`run_in_thread` call: whether its code returned and whether its caller gave up on it first."""

    __slots__ = ("finished", "abandoned")

    def __init__(self) -> None:
        self.finished = False
        self.abandoned = False

    def abandon(self) -> int | None:
        """Count the call as abandoned unless it already returned; the number now abandoned, or None."""

        global _abandoned_calls
        with _abandoned_lock:
            if self.finished:
                return None
            self.abandoned = True
            _abandoned_calls += 1
            return _abandoned_calls

    def end(self) -> None:
        """The code returned (in its thread): uncount it if it was abandoned."""

        global _abandoned_calls
        with _abandoned_lock:
            self.finished = True
            if self.abandoned:
                _abandoned_calls -= 1


def abandoned_calls() -> int:
    """How many env calls are still running in this process after their callers gave up on them."""

    with _abandoned_lock:
        return _abandoned_calls


def _count_abandoned(delta: int) -> int:
    global _abandoned_calls
    with _abandoned_lock:
        _abandoned_calls += delta
        return _abandoned_calls


def _warn_abandoned(what: str, running: int) -> None:
    logger.warning("Gave up waiting for env code %s; %d abandoned env calls are still running", what, running)


def abandon(work: asyncio.Future[Any], what: str) -> None:
    """Count ``work``, env code its caller gave up on while it runs on, as abandoned until it ends."""

    if work.done():
        return
    _warn_abandoned(what, _count_abandoned(1))
    work.add_done_callback(lambda _: _count_abandoned(-1))


async def run_in_thread(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run sync ``fn`` on a daemon thread of its own, in a copy of the caller's context, and await its result.

    Cancelling the caller abandons the call: the thread runs on (Python cannot stop it), counted by
    :func:`abandoned_calls` until it returns, and its result is dropped.
    """

    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()
    context = contextvars.copy_context()
    call = _ThreadCall()

    def deliver(value: Any, error: BaseException | None) -> None:
        if future.done():  # the caller gave up
            return
        if error is None:
            future.set_result(value)
        else:
            future.set_exception(error)

    def work() -> None:
        value, error = None, None
        try:
            value = context.run(fn, *args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - delivered to the caller, as to_thread delivers it
            error = exc
        call.end()
        try:
            loop.call_soon_threadsafe(deliver, value, error)
        except RuntimeError:  # the loop closed while the call ran
            pass

    threading.Thread(target=work, name=f"platoon-env:{callable_name(fn)}", daemon=True).start()
    try:
        return await future
    except asyncio.CancelledError:
        running = call.abandon()
        if running is not None:
            _warn_abandoned(callable_name(fn), running)
        raise


def deepcopy_env(env: Any) -> Any:
    """A deep copy of an ``Environment`` instance, as a ``copy`` fork takes it.

    The SDK's per-instance tool locks and gate (0.1.163, created lazily) start afresh in the copy, and asyncio
    primitives that were ever contended keep their binding to the running loop.  Raises ``TypeError`` when the
    env holds something that cannot be copied.
    """

    memo: dict[int, Any] = {
        id(obj): None
        for obj in (getattr(env, "_tool_locks", None), getattr(env, "_exclusive_gate", None))
        if obj is not None
    }
    try:
        loop = asyncio.get_running_loop()
        memo[id(loop)] = loop
    except RuntimeError:
        pass
    try:
        return copy.deepcopy(env, memo)
    except (TypeError, copy.Error, pickle.PicklingError) as exc:
        raise TypeError(f"A copy fork deep-copies the env, and {type(env).__name__} cannot be: {exc}") from exc


def overrides_teardown(env: Any) -> bool:
    """Whether ``env``'s class defines its own ``teardown`` (the SDK's does nothing)."""

    base = getattr(import_sdk().Environment, "teardown", None)
    return getattr(type(env), "teardown", None) is not base


def callable_name(fn: Any) -> str:
    """A name for env code in logs and thread names."""

    return str(getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or type(fn).__name__)


@dataclass
class RawTool:
    """A tool as an env (in process or remote) lists it, before platoon adds its hints."""

    name: str
    description: str
    input_schema: Mapping[str, Any] | None
    meta: Mapping[str, Any] | None = None
    sdk_terminal: bool = False  # SDK @terminal, or @tool(exclusive=True), which the SDK reserves for submit tools
    # Only an SDK @terminal tool takes the final assistant message (as its ``terminal_arg`` field, or no
    # argument): an exclusive submit tool is terminal for access filtering but is called like any other tool.
    routes_message: bool = False
    terminal_arg: str | None = None
    # @tool(concurrent=True) (0.1.163). The flag is not on the wire: a remote tool has it only from the class its
    # service names (with_class_descriptors).
    sdk_concurrent: bool = False


def is_tool(fn: Any) -> bool:
    return callable(fn) and bool(getattr(fn, "_env_tool", False))


def is_sdk_terminal(fn: Any) -> bool:
    return takes_final_message(fn) or bool(getattr(fn, "_env_tool_exclusive", False))


def takes_final_message(fn: Any) -> bool:
    """Whether ``fn`` is an SDK ``@terminal`` tool, the one a final assistant message is routed into."""

    return bool(getattr(fn, "_env_tool_terminal", False))


def is_sdk_concurrent(fn: Any) -> bool:
    """``@tool(concurrent=True)``, which the SDK ignores on a terminal tool."""
    return bool(getattr(fn, "_env_tool_concurrent", False)) and not is_sdk_terminal(fn)


def tool_function(env: Any, name: str) -> Any | None:
    """The ``@tool`` method ``name`` of an env (an instance, or its class) or of one of its declared toolsets."""

    fn = getattr(env, name, None)
    if is_tool(fn):
        return fn
    for toolset in getattr(env if isinstance(env, type) else type(env), "toolsets", ()) or ():
        fn = getattr(toolset, name, None)
        if is_tool(fn):
            return fn
    return None


def parameter_model(fn: Any) -> type | None:
    """The pydantic model of a tool's single parameter, or None for a tool without parameters."""

    real = inspect.unwrap(fn)
    params = [p for p in inspect.signature(real).parameters.values() if p.name != "self"]
    if not params:
        return None
    hints = typing.get_type_hints(real, include_extras=True)
    return hints.get(params[0].name)


def model_schema(model: type | None) -> dict[str, Any] | None:
    if model is None:
        return None
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()  # type: ignore[attr-defined]  # pydantic v1


def terminal_arg(fn: Any) -> str | None:
    """The single field a terminal tool takes the final message as (the SDK's ``TerminalToolSpec.arg``)."""

    fields = list(getattr(parameter_model(fn), "model_fields", {}) or {})
    return fields[0] if len(fields) == 1 else None


async def local_tools(env: Any) -> list[RawTool]:
    """Every tool an in-process env serves to a session, including a 0.1.163 ``@terminal`` tool.

    Uses the SDK's own session listing (class tools, task tools, toolsets), so
    names and schemas match what its server sends.
    """

    try:
        from openreward.environments.session import list_session_tools
    except ImportError:  # pragma: no cover - both supported SDKs have it
        listing = type(env).list_tools()
        task_tools = await run_user_callable(env.list_task_tools)
        specs = [*listing.tools, *task_tools.tools]
        terminal = getattr(listing, "terminal_tool", None)
    else:
        listing = await list_session_tools(env, None)
        specs, terminal = list(listing.tools), getattr(listing, "terminal_tool", None)
    tools = []
    for spec in specs:
        fn = tool_function(env, spec.name)
        tools.append(
            RawTool(
                name=spec.name,
                description=spec.description,
                input_schema=spec.input_schema,
                meta=getattr(fn, "_meta", None),
                sdk_terminal=is_sdk_terminal(fn),
                routes_message=takes_final_message(fn),
                terminal_arg=terminal_arg(fn) if takes_final_message(fn) else None,
                sdk_concurrent=is_sdk_concurrent(fn),
            )
        )
    if terminal is not None and all(tool.name != terminal.name for tool in tools):
        # 0.1.163 hides the @terminal tool from the model's list; platoon lists it, marked terminal.
        fn = tool_function(env, terminal.name)
        tools.append(
            RawTool(
                name=terminal.name,
                description=terminal.description,
                input_schema=model_schema(parameter_model(fn)) if fn is not None else _terminal_schema(terminal.arg),
                meta=getattr(fn, "_meta", None),
                sdk_terminal=True,
                routes_message=True,
                terminal_arg=terminal.arg,
            )
        )
    return tools


async def remote_tools(session: Any) -> list[RawTool]:
    """Every tool a remote session serves, including a 0.1.163 ``@terminal`` tool (rebuilt from its descriptor).

    A tool keeps whatever meta the listing carries (:data:`LISTED_META_KEYS`).  openreward's env server sends
    none, so over ``ors+http://`` a tool's meta, ``readOnlyHint`` included, comes from the service's
    ``tool_meta`` and ``read_only_tools`` options until the server sends it.
    """

    try:
        listed = await session.list_tools()
    except TypeError:
        # A server that sends keys the SDK's ToolSpec lacks (``_meta``): the provider formats keep them.  That
        # listing also records the terminal tool on the session (0.1.163; 0.1.125 has neither); terminal_tool()
        # would list again in the strict format and fail the same way.
        raw = [dict(tool) for tool in await session.list_tools(format="anthropic")]
        terminal = getattr(session, "_terminal_tool", None)
    else:
        raw = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
                **{key: getattr(t, key) for key in LISTED_META_KEYS if isinstance(getattr(t, key, None), Mapping)},
            }
            for t in listed
        ]
        terminal = await session.terminal_tool() if hasattr(session, "terminal_tool") else None
    tools = []
    for tool in raw:
        meta: dict[str, Any] = {}
        for key in LISTED_META_KEYS:
            if isinstance(tool.get(key), Mapping):
                meta.update(tool[key])
        tools.append(
            RawTool(
                name=str(tool["name"]),
                description=str(tool.get("description") or ""),
                input_schema=tool.get("input_schema") or tool.get("parameters"),
                meta=meta or None,
            )
        )
    if terminal is not None and all(tool.name != terminal.name for tool in tools):
        tools.append(
            RawTool(
                name=terminal.name,
                description=terminal.description,
                input_schema=_terminal_schema(terminal.arg),
                sdk_terminal=True,
                routes_message=True,
                terminal_arg=terminal.arg,
            )
        )
    return tools


def with_class_descriptors(tools: Sequence[RawTool], cls: type) -> list[RawTool]:
    """A remote session's tools as the class its env server serves describes them in process (:func:`local_tools`).

    An env server sends each tool's name, description and schema, and 0.1.163 the ``@terminal`` tool's name,
    argument and description: nothing else.  A listed tool that ``cls`` defines takes from its function what
    :func:`local_tools` takes from it: its ``_meta``, the SDK's terminal (``@terminal``, ``@tool(exclusive=True)``)
    and concurrent flags, and for a ``@terminal`` tool the schema of its parameter model.  A tool the class does not
    define (a task tool) stays as listed.
    """

    described = []
    for tool in tools:
        fn = tool_function(cls, tool.name)
        if fn is None:
            described.append(tool)
            continue
        routes = takes_final_message(fn)
        described.append(
            dataclasses.replace(
                tool,
                input_schema=model_schema(parameter_model(fn)) if routes else tool.input_schema,
                meta=getattr(fn, "_meta", None) or tool.meta,
                sdk_terminal=is_sdk_terminal(fn),
                routes_message=routes,
                terminal_arg=terminal_arg(fn) if routes else None,
                sdk_concurrent=is_sdk_concurrent(fn),
            )
        )
    return described


def class_listing_differences(cls: type, tools: Sequence[Any], terminal: Any) -> list[str]:
    """How an env server's listing of its class's tools (its ``/tools``: the tools, and 0.1.163's ``@terminal``
    descriptor) differs from what ``cls.list_tools()`` lists here, one line each; empty when they agree.

    The names must match, the terminal tool's included (a server on another SDK version may list it as an ordinary
    tool), and a tool that both list as an ordinary tool must have the same description and schema (titles aside).
    """

    def entries(specs: Sequence[Any], terminal_tool: Any) -> dict[str, tuple[str, dict[str, Any]] | None]:
        listed: dict[str, tuple[str, dict[str, Any]] | None] = {
            str(spec.name): (str(spec.description or "").strip(), normalize_schema(spec.input_schema)) for spec in specs
        }
        if terminal_tool is not None:
            listed.setdefault(str(terminal_tool.name), None)  # all both sides describe of it is its name
        return listed

    listing = cls.list_tools()
    here = entries(listing.tools, getattr(listing, "terminal_tool", None))
    there = entries(tools, terminal)
    differences = []
    if missing := sorted(set(here) - set(there)):
        differences.append(f"the server lists no {', '.join(missing)}")
    if extra := sorted(set(there) - set(here)):
        differences.append(f"the class has no {', '.join(extra)}")
    for name in sorted(set(here) & set(there)):
        mine, theirs = here[name], there[name]
        if mine is None or theirs is None:
            continue
        if mine[0] != theirs[0]:
            differences.append(f"{name} has another description")
        if mine[1] != theirs[1]:
            differences.append(f"{name} takes other parameters")
    return differences


def openai_tools(tools: Sequence[RawTool]) -> list[dict[str, Any]]:
    """``tools`` as the SDK's ``list_tools(format="openai")`` lists them, which hides ``@terminal`` tools: the
    catalog that the ``get_task`` bridge tool shows the model."""

    from openreward.api.environments.client import sanitize_tool_schema

    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": sanitize_tool_schema(tool.input_schema, "openai") if tool.input_schema else None,
        }
        for tool in tools
        if not tool.routes_message
    ]


def _terminal_schema(arg: str | None) -> dict[str, Any]:
    if arg is None:
        return dict(_EMPTY_SCHEMA)
    return {"type": "object", "properties": {arg: {"type": "string"}}, "required": [arg]}


def normalize_schema(schema: Mapping[str, Any] | None) -> dict[str, Any]:
    """An MCP-ready object schema: pydantic's ``title`` keywords dropped, everything else (``x-`` keys) kept."""

    if not schema:
        return dict(_EMPTY_SCHEMA)
    return _strip_titles(schema)


# The JSON Schema keywords whose values are subschemas: a map of them, a list of them, or one (``items`` may also
# be a list in older drafts).  Every other keyword's value is data (``default``, ``examples``, ``const``, ``enum``,
# a ``discriminator``, an ``x-`` extension), which may hold a ``title`` key of its own.
_SUBSCHEMA_MAPS = frozenset({"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"})
_SUBSCHEMA_LISTS = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})
_SUBSCHEMAS = frozenset(
    {
        "items",
        "additionalItems",
        "additionalProperties",
        "unevaluatedItems",
        "unevaluatedProperties",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
    }
)


def _strip_titles(schema: Any) -> Any:
    """``schema`` without the ``title`` keyword of any (sub)schema; only schema positions are walked.

    Unlike the SDK's stripper, which drops every ``title`` key it meets, a property named ``title`` and data
    with a ``title`` key (a dict default, a model default, a routing extension) come through unchanged.
    """

    if not isinstance(schema, Mapping):
        return copy.deepcopy(schema)  # a boolean schema, or something malformed: left as it is
    stripped: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title":
            continue
        if key in _SUBSCHEMA_MAPS and isinstance(value, Mapping):
            stripped[key] = {name: _strip_titles(subschema) for name, subschema in value.items()}
        elif (key in _SUBSCHEMA_LISTS or key in _SUBSCHEMAS) and isinstance(value, list):
            stripped[key] = [_strip_titles(subschema) for subschema in value]
        elif key in _SUBSCHEMAS:
            stripped[key] = _strip_titles(value)
        else:
            stripped[key] = copy.deepcopy(value)
    return stripped


def tool_meta(
    provider_meta: Mapping[str, Any] | None,
    schema: Mapping[str, Any],
    *,
    extra: Mapping[str, Any] | None = None,
    routing_override: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """A tool's MCP ``_meta``: the provider's ``_meta`` plus routing from the schema extension.

    The schema's root-level ``x-openhands-tool-routing`` becomes
    ``openhands.dev/tool-routing`` unless the provider set it; an override wins.
    """

    meta = {**(provider_meta or {}), **(extra or {})}
    routing = schema.get(TOOL_ROUTING_SCHEMA_KEY)
    if TOOL_ROUTING_META_KEY not in meta and isinstance(routing, Mapping):
        meta[TOOL_ROUTING_META_KEY] = dict(routing)
    if routing_override is not None:
        meta[TOOL_ROUTING_META_KEY] = dict(routing_override)
    return meta or None


def meta_says_read_only(meta: Mapping[str, Any] | None) -> bool:
    if not meta:
        return False
    annotations = meta.get("annotations")
    return any(meta.get(key) is True for key in READ_ONLY_META_KEYS) or (
        isinstance(annotations, Mapping) and annotations.get("readOnlyHint") is True
    )


def content_blocks(blocks: Sequence[Any]) -> list[dict[str, Any]]:
    """SDK text and image blocks as MCP content blocks."""

    content = []
    for block in blocks:
        kind = getattr(block, "type", None)
        if kind == "text":
            content.append({"type": "text", "text": block.text})
        elif kind == "image":
            content.append({"type": "image", "data": block.data, "mimeType": block.mimeType})
        else:
            content.append({"type": "text", "text": str(block)})
    return content


def prompt_text(blocks: Sequence[Any]) -> str:
    """A prompt's text; images become placeholders (the blocks stay available for multimodal harnesses)."""

    parts = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif getattr(block, "type", None) == "image":
            parts.append(f"[image: {block.mimeType}]")
        else:
            text = getattr(block, "text", None)
            parts.append(text if isinstance(text, str) else str(block))
    return "\n".join(parts)


def task_name(spec: Any, index: int) -> str:
    """A task's display name: the first of ``task_name``, ``id``, ``task_id`` and ``name`` the spec sets, else
    ``task-N``."""

    if isinstance(spec, Mapping):
        for key in ("task_name", "id", "task_id", "name"):
            if spec.get(key):
                return str(spec[key])
    return f"task-{index}"


def split_names(splits: Sequence[Any]) -> list[str]:
    return [str(getattr(split, "name", split)) for split in splits]


# The openreward SDK's own infrastructure errors, by class name (both supported SDKs name them so, in modules of their
# own): a session that is gone, and the transport errors of its HTTP client.  An env tool that raises one failed because
# a session or a service the env itself uses failed (a sandbox it opened, an API it calls), not because of the
# policy's call.  Only the SDK's classes count: another library's class of the same name (httpx's TransportError, the
# base of its ConnectError and ReadTimeout) reaches an env server's client as text that none of _INFRA_ERROR_TEXTS
# matches, so in process it must be the tool's own failure too.
INFRA_ERROR_CLASSES = frozenset(
    {"SessionTerminatedError", "TransportError", "MaxRetriesError", "HeartbeatTimeoutError"}
)
_SDK_PACKAGE = "openreward"
# The texts the SDK gives those errors, which is all of them an env server sends when a tool raised one:
# SessionTerminatedError's "<kind> session terminated (sid=...): ..." and MaxRetriesError's list of the errors it met
# (its lines lose their indentation on the way: the server sends them as SSE data lines).
_INFRA_ERROR_TEXTS = (
    re.compile(r"^(?:environment|sandbox) session terminated \(sid="),
    re.compile(r"\n\s*Encountered \d+ error\(s\):"),
)


def infra_error(exc: BaseException, *, remote: bool) -> bool:
    """Whether a tool that raised ``exc`` failed for infrastructure: the SDK's own session or transport error.

    In process, ``exc`` is what the env's tool raised, and its class decides: one of the SDK's classes in
    :data:`INFRA_ERROR_CLASSES`, or a subclass of one (a class of the same name from another library is not).  An env
    server reports a raised tool by its text alone (0.1.163's ``ToolFailed``, 0.1.125's ``RuntimeError``), so
    remotely the text decides: the texts the SDK gives a ``SessionTerminatedError`` and a ``MaxRetriesError``.  A
    ``HeartbeatTimeoutError`` (no text) or a ``TransportError`` raised with a text of its own therefore reads as the
    tool's own failure on an env server; an exception the env raised from one (``raise ... from``) reads as the
    tool's own failure in both modes.
    """

    if not remote:
        return any(
            cls.__name__ in INFRA_ERROR_CLASSES and cls.__module__.partition(".")[0] == _SDK_PACKAGE
            for cls in type(exc).__mro__
        )
    text = _session_message(exc)
    return any(pattern.search(text) for pattern in _INFRA_ERROR_TEXTS)


def error_kind(exc: BaseException) -> str:
    """Classify a remote call failure by exception class name, which differs across SDK versions.

    ``invalid_call``: the env refused the call (unknown tool, bad arguments,
    episode over) and the session lives on. ``dead``: the session is gone,
    which the server says with a 404 or 410 (0.1.125 wraps that response in a
    ``RuntimeError``). ``tool_raised``: the env's tool raised, whatever its
    message (0.1.163 also kills the session). ``transport``: anything else
    (connection, timeouts, retries exhausted, another HTTP status, an SSE
    stream that lost its task).
    """

    names = {cls.__name__ for cls in type(exc).__mro__}
    if "ToolCallError" in names:
        return "invalid_call"
    if "SessionTerminatedError" in names:
        return "dead"
    status = response_status(exc)
    if status is not None:
        return "dead" if status in (404, 410) else "transport"
    if _session_message(exc) == _SSE_LOST_TASK:
        return "transport"
    if "ToolFailed" in names or "_RemoteSSEError" in names or type(exc) is RuntimeError:
        return "tool_raised"
    return "transport"


# The one SSE ``error`` event that reports the stream, not the tool: a resumed call whose task the server no longer
# knows.  The server's only other error event carries what the call raised, so its "Unknown SSE error" (the
# client's text for an empty message) is env code that raised ``TimeoutError()``, a bare ``assert`` and the like.
_SSE_LOST_TASK = "unknown task_id"
# The prefix the SDK puts on a session call's error message.
_SESSION_PREFIX = re.compile(r"^\((?:environment|sandbox) session sid=[^)]*\)\s*")


def response_status(exc: BaseException) -> int | None:
    """The HTTP status of an aiohttp ``ClientResponseError`` that ``exc`` is or (0.1.125, which re-raises it as
    a ``RuntimeError``) was raised from; None for any other failure."""

    for candidate in (exc, exc.__context__, exc.__cause__):
        if candidate is None or "ClientResponseError" not in {cls.__name__ for cls in type(candidate).__mro__}:
            continue
        status = getattr(candidate, "status", None)
        if isinstance(status, int):
            return status
    return None


def _session_message(exc: BaseException) -> str:
    return _SESSION_PREFIX.sub("", str(exc)).strip()
