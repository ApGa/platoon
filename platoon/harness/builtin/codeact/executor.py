"""Where cells run: an agent's IPython shell and cell loop (the harness declares ``executes_model_code``).

:mod:`~platoon.harness.builtin.codeact.harness` explains how the cell thread and the agent's loop work together.
Each agent gets one :class:`CellExecutor`, used as ``async with``: an embedded IPython shell whose namespace holds a
stub per cell function and a restricted ``asyncio``, and the agent's cell loop, an event loop in a thread of its own,
where cells run with ``run_cell_async`` (so top-level ``await`` works).

A cell first goes through the checks of the original ``CodeActAgent`` (``checks.check_cell``); a refused cell never
runs. Exceptions in a cell show up as its traceback text, and a cell that outlives its timeout is cancelled and
reported as timed out; the calls it makes after that are refused, even while synchronous code in it runs on.
Output beyond ``max_output_chars`` is cut (the original agent never cut it, so the default keeps everything). What a
cell did comes back as a :class:`CellResult`, a plain value, so an executor in another process could return one.

A stub (:func:`cell_stub`) is a real Python function with its cell function's parameters, so Python binds every call
in the cell; the :class:`LoopBridge` then runs the function's ``invoke`` coroutine on the agent's loop, in a fresh
context, so nothing a call prints or logs reaches the cell's output. Leaving the executor closes the bridge: calls still
running are cancelled and later ones refused, so a cell left running can no longer reach the env.

A cell's output reaches the model's prompt, so it names no path platoon or Python is installed at. The futures and
tasks of the cell loop (a gather, a task, ``asyncio.current_task()``) print as ``<_GatheringFuture pending>`` or
``<Task pending name='Task-3' coro=<launch_subagent()>>``, since asyncio's own repr names the source file and line of
every callback and coroutine (:class:`_CellLoop`). The installed files that tracebacks, warnings and module reprs
name are shown by their paths below their import roots, such as ``File json/decoder.py:338``
(``ipython_shell.shorten_install_paths``); a cell's own files keep their paths.
"""

from __future__ import annotations

import asyncio
import atexit
import builtins
import concurrent.futures
import contextvars
import inspect
import reprlib
import sys
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from IPython.terminal.embed import InteractiveShellEmbed
from traitlets.config.loader import Config

from platoon.harness.builtin.codeact.checks import Checks, check_cell
from platoon.harness.builtin.codeact.ipython_shell import (
    ShellCapture,
    drop_out_prompts,
    run_cell_async,
    shorten_install_paths,
    strip_ansi_escape_sequences,
)

if TYPE_CHECKING:
    from platoon.harness.builtin.codeact.namespace import CellFunction, Parameter

Invoke = Callable[[dict[str, Any]], Awaitable[Any]]

LOOP_SHUTDOWN_TIMEOUT_S = 5.0
SETTLE_ITERATIONS = 3  # loop iterations a stopping loop gives the callbacks still scheduled (_stop_cells, __aexit__)
AGENT_ENDED = "The agent has ended: its cell functions can no longer be called"
# The cell a call comes from: set in the context its code runs in, which the tasks it starts inherit, so a cell that
# timed out can be told apart from the next one (LoopBridge.retire).
_CELL: contextvars.ContextVar[object | None] = contextvars.ContextVar("codeact_cell", default=None)


@dataclass(frozen=True)
class CellResult:
    """What running one cell did.

    Attributes:
        code: The cell as run (stripped).
        output: What it printed (None when it did not run). A traceback is part of it: IPython prints tracebacks
            to stdout.
        error: The refusal, what the cell wrote to stderr, or the timeout notice.
        timed_out: The cell outlived its timeout and was cancelled.
    """

    code: str
    output: str | None = None
    error: str | None = None
    timed_out: bool = False


def _gather(*aws: Any, return_exceptions: bool = False) -> asyncio.Future[list[Any]]:
    """``asyncio.gather`` whose outcome is always retrieved, and whose repr names no file.

    A cell may gather calls it never awaits (``pending = asyncio.gather(launch_subagent(...), ...)``); leaving the
    executor then cancels them, which sets the gather's exception.  Awaiting the gather still raises it, but asyncio
    no longer logs it as never retrieved.  The callback that retrieves it would show in the gather's repr with this
    module's absolute path, so the gather gets the cell loop's short repr (:class:`_CellLoop`).
    """

    future = asyncio.gather(*aws, return_exceptions=return_exceptions)
    future.add_done_callback(_retrieve)
    if _GatheringFuture is not None and type(future) is _GatheringFuture:
        future.__class__ = _CellGatheringFuture
    return future


def _retrieve(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


# --------------------------------------------------------------------------- the cell loop's futures and tasks


def _outcome_repr(future: asyncio.Future[Any]) -> list[str]:
    """asyncio's state and outcome fields of a repr, without its callbacks and source locations."""

    state = future._state.lower()  # type: ignore[attr-defined]  # what asyncio's own repr reads
    if state != "finished":
        return [state]
    exception = future._exception  # type: ignore[attr-defined]  # unlike exception(), marks nothing retrieved
    if exception is not None:
        return [state, f"exception={exception!r}"]
    return [state, f"result={reprlib.repr(future._result)}"]  # type: ignore[attr-defined]


class _CellFuture(asyncio.Future):
    """A future of the cell loop (``loop.create_future()``: sleeps, shields, a cell's own futures)."""

    def __repr__(self) -> str:
        return f"<Future {' '.join(_outcome_repr(self))}>"


class _CellTask(asyncio.Task):
    """A task of the cell loop: ``<Task pending name='Task-3' coro=<launch_subagent()>>``."""

    def set_name(self, value: object) -> None:
        # Python 3.13's loop.create_task names every task a task factory makes, None when the cell gave it no name,
        # which would show as name='None'; Python 3.12 and 3.14 keep the default name, as this does.
        if value is not None:
            super().set_name(value)

    def __repr__(self) -> str:
        info = _outcome_repr(self)
        if info[0] == "pending" and self.cancelling():
            info[0] = "cancelling"
        coro = self.get_coro()
        name = getattr(coro, "__qualname__", None) or type(coro).__name__
        info[1:1] = [f"name={self.get_name()!r}", f"coro=<{name}()>"]
        return f"<Task {' '.join(info)}>"


_GatheringFuture: type[asyncio.Future[Any]] | None = getattr(asyncio.tasks, "_GatheringFuture", None)

if _GatheringFuture is not None:

    class _CellGatheringFuture(_GatheringFuture):  # type: ignore[misc,valid-type]
        """A gather made in a cell (:func:`_gather` swaps its class): ``<_GatheringFuture pending>``, as the original
        agent printed it."""

        def __repr__(self) -> str:
            return f"<_GatheringFuture {' '.join(_outcome_repr(self))}>"


def _cell_task(loop: asyncio.AbstractEventLoop, coro: Any, **kwargs: Any) -> asyncio.Task[Any]:
    return _CellTask(coro, loop=loop, **kwargs)


class _CellLoop(asyncio.SelectorEventLoop):
    """The cell loop: its futures and tasks print what they are, never where their code lives.

    asyncio's repr of a future or task lists its callbacks and its coroutine with their source file and line, such
    as ``cb=[_retrieve() at /abs/path/platoon/harness/builtin/codeact/executor.py:85]`` or ``wait_for=<Future
    pending cb=[_chain_future.<locals>._call_check_cancel() at /abs/path/lib/python3.12/asyncio/futures.py:389]>``.
    A cell that shows one (its last expression, ``print``) would put the install paths into the model's prompt.
    """

    def __init__(self) -> None:
        super().__init__()
        self.set_task_factory(_cell_task)

    def create_future(self) -> asyncio.Future[Any]:
        return _CellFuture(loop=self)


class SafeAsyncio:
    """A restricted asyncio wrapper that only exposes safe operations.

    Blocks dangerous operations like set_event_loop(), new_event_loop(), run(),
    and set_event_loop_policy() that can cause deadlocks or interfere with
    the host event loop.
    """

    __name__ = "asyncio"  # So it's injected with the correct name in shell namespace

    # Safe operations for concurrent execution
    gather = staticmethod(_gather)
    sleep = staticmethod(asyncio.sleep)
    create_task = staticmethod(asyncio.create_task)
    wait = staticmethod(asyncio.wait)
    wait_for = staticmethod(asyncio.wait_for)
    as_completed = staticmethod(asyncio.as_completed)
    shield = staticmethod(asyncio.shield)

    # Safe synchronization primitives
    Event = asyncio.Event
    Lock = asyncio.Lock
    Semaphore = asyncio.Semaphore
    BoundedSemaphore = asyncio.BoundedSemaphore
    Condition = asyncio.Condition
    Queue = asyncio.Queue
    LifoQueue = asyncio.LifoQueue
    PriorityQueue = asyncio.PriorityQueue

    # Safe introspection (read-only)
    get_running_loop = staticmethod(asyncio.get_running_loop)
    current_task = staticmethod(asyncio.current_task)
    all_tasks = staticmethod(asyncio.all_tasks)
    iscoroutine = staticmethod(asyncio.iscoroutine)
    # asyncio's own warns on Python 3.14 (it is deprecated there), and the warning would reach the cell's error.
    iscoroutinefunction = staticmethod(inspect.iscoroutinefunction)

    # Timeout context managers
    timeout = staticmethod(asyncio.timeout)
    timeout_at = staticmethod(asyncio.timeout_at)

    # Expose TimeoutError for catching
    TimeoutError = asyncio.TimeoutError

    # Block everything else - especially dangerous operations like:
    # set_event_loop, new_event_loop, run, set_event_loop_policy, get_event_loop
    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            f"asyncio.{name} is disabled in sandbox. "
            f"Only safe operations like gather, create_task, sleep, wait, wait_for, "
            f"and synchronization primitives (Lock, Event, Queue, etc.) are allowed."
        )


# Singleton instance to be used as the "asyncio" module in the sandbox
safe_asyncio = SafeAsyncio()


def _sandboxed_import(name, globals=None, locals=None, fromlist=(), level=0):
    """The cells' ``__import__``: ``import asyncio`` and its submodules give :data:`safe_asyncio`."""

    if name == "asyncio" or name.startswith("asyncio."):
        return safe_asyncio
    return builtins.__import__(name, globals, locals, fromlist, level)


def truncate_output(text: str, limit: int | None) -> str:
    """``text`` cut to ``limit`` characters with a marker saying how much was dropped."""

    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + f"\n...[output truncated: {len(text) - limit} more characters]..."


@dataclass(eq=False)
class _Call:
    """One cell-function call through the bridge: its ``invoke`` coroutine, its future once submitted, and whether the
    agent's loop has started it."""

    coroutine: Any
    future: concurrent.futures.Future[Any] | None = None
    started: bool = False


class LoopBridge:
    """The cell thread's way back to the agent's loop: runs a cell function's ``invoke`` there, by name.

    The bridge keeps each call until it ends. :meth:`close` (the agent has ended) cancels the calls still running,
    refuses later ones and drops the functions, so neither a cell left running nor an IPython shell that outlives
    its executor can reach the env or keep the agent's context alive.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, invokes: Mapping[str, Invoke]) -> None:
        self.loop = loop
        self._loop_thread = threading.get_ident()
        self._invokes = dict(invokes)
        self._in_flight: dict[concurrent.futures.Future[Any], _Call] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._retired: set[object] = set()  # cells that timed out or were cancelled: their calls are refused

    def submit(self, name: str, arguments: dict[str, Any]) -> concurrent.futures.Future[Any]:
        """Start function ``name``'s ``invoke(arguments)`` on the agent's loop, in a fresh context.

        The call sees none of the cell's context variables, ``ShellCapture``'s in particular: what the env prints or
        logs during a tool call (openreward's ``tool_call_completed`` line at INFO, say) goes where it would outside
        any cell instead of into the cell's output, and a subagent the call launches does not inherit the cell's
        capture. A call reaches the cell only through what it returns or raises.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError(AGENT_ENDED)
            if _CELL.get() in self._retired:
                # A cell that outlived its timeout runs on until it awaits (a synchronous one, until it ends): what
                # it calls after the timeout would act on the env behind the agent's back.
                raise concurrent.futures.CancelledError()
            call = _Call(self._invokes[name](arguments))
            starting = self._start(call)
            try:
                # run_coroutine_threadsafe starts the task in a copy of the context it is called in: an empty one.
                future = contextvars.Context().run(asyncio.run_coroutine_threadsafe, starting, self.loop)
            except BaseException:
                starting.close()
                call.coroutine.close()
                raise
            call.future = future
            self._in_flight[future] = call
        future.add_done_callback(self._forget)
        return future

    async def _start(self, call: _Call) -> Any:
        """On the agent's loop: run ``call``, unless it was cancelled before the loop got to it.

        Cancelling a call's future from another thread cancels its task only through the loop's queue, behind the
        task's first step: a call that a timed-out cell made just before its agent ended would otherwise still start,
        and reach the env. When the task is cancelled before that step instead (Python 3.13 and later cancel it at
        once when the future was cancelled first, or is cancelled on the agent's loop), this never runs, and
        :meth:`_forget` closes the call's coroutine.
        """

        __tracebackhide__ = True  # IPython leaves this frame out of a cell's traceback
        with self._lock:  # submit() has stored the call's future by the time it lets the lock go
            call.started = not (call.future is not None and call.future.cancelled())
        if not call.started:
            call.coroutine.close()
            raise asyncio.CancelledError
        return await call.coroutine

    def _forget(self, future: concurrent.futures.Future[Any]) -> None:
        with self._lock:
            call = self._in_flight.pop(future, None)
            if call is not None and not call.started:
                # The call ended before the agent's loop started it, so its invoke never ran: closed, it does not warn
                # that it was never awaited when it is collected (and _start, if it runs later, does not start it).
                call.coroutine.close()

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Run function ``name`` on the agent's loop and block this (cell) thread until it returns.

        A cancelled call (the cell timed out, the agent ended) raises ``CancelledError`` here at once.
        """

        __tracebackhide__ = True  # IPython leaves this frame out of a cell's traceback
        if threading.get_ident() == self._loop_thread:
            raise RuntimeError("A synchronous cell function cannot block the agent's loop; run cells on a cell loop")
        try:
            future = self.submit(name, arguments)
            error = future.exception()  # waits for the call; raises when it was cancelled
        except concurrent.futures.CancelledError:
            raise asyncio.CancelledError from None
        if error is None:
            return future.result()
        try:
            raise error  # its traceback goes on in invoke, without concurrent.futures' frames
        finally:
            error = future = None  # no reference cycle through this frame

    async def call_async(self, name: str, arguments: dict[str, Any]) -> Any:
        """Await function ``name`` on the agent's loop from a coroutine on the cell loop; cancelling one cancels the
        other."""

        __tracebackhide__ = True  # IPython leaves this frame out of a cell's traceback
        try:
            future = self.submit(name, arguments)
        except concurrent.futures.CancelledError:
            raise asyncio.CancelledError from None
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            future.cancel()
            raise

    def retire(self, cell: object) -> None:
        """Refuse every later call of ``cell`` (and of the tasks it started), which timed out or was cancelled; its
        call in flight, if any, :meth:`cancel_in_flight` cancels."""

        with self._lock:
            self._retired.add(cell)

    def cancel_in_flight(self) -> None:
        """Cancel the calls still running on the agent's loop; a cell blocked in one gets ``CancelledError`` at once.

        An env call cancelled mid-flight may already have been applied by the env.
        """

        with self._lock:
            in_flight = list(self._in_flight)
        for future in in_flight:
            future.cancel()

    def close(self) -> None:
        """Refuse later calls, drop the functions and cancel the calls still running: the agent has ended."""

        with self._lock:
            self._closed = True
            self._invokes.clear()
        self.cancel_in_flight()


class _NotPassed:
    """The default of an optional stub parameter the call left out (the function applies its own default)."""

    def __repr__(self) -> str:
        return "<not passed>"


NOT_PASSED: Any = _NotPassed()


def cell_stub(function: CellFunction, bridge: LoopBridge) -> Callable[..., Any]:
    """The Python function cells call for ``function``: a real function with its parameters.

    Python binds each call, so a call that does not fit raises CPython's ``TypeError`` where it is made, word for
    word, and nothing runs: ``asyncio.gather(launch_subagent(bad), launch_subagent(good))`` launches nothing. The
    arguments the call passed, by parameter name, then go to ``function.invoke`` on the agent's loop through
    ``bridge``: a synchronous function blocks the cell thread until it returns, an async one is a coroutine for the
    cell to await. The stub carries the function's name, docstring and signature for ``help()``.

    The stub is compiled with stand-in parameter names (``p0``, ``p1``, ...) that then take the real ones, so any
    name binds as given: ``from``, ``file-path``, or ``ﬁle``, which Python would read as ``file`` in source.
    """

    parameters = function.parameters
    names = tuple(parameter.name for parameter in parameters)
    if any(later.required and not earlier.required for earlier, later in zip(parameters, parameters[1:])):
        raise ValueError(f"{function.name!r}: a required parameter follows an optional one, which Python cannot bind")

    def passed(values: tuple[Any, ...]) -> dict[str, Any]:
        return {name: value for name, value in zip(names, values) if value is not NOT_PASSED}

    stand_ins = [f"p{index}" for index in range(len(names))]
    signature = ", ".join(
        stand_in if parameter.required else f"{stand_in}=__not_passed"
        for stand_in, parameter in zip(stand_ins, parameters)
    )
    values = "".join(f"{stand_in}, " for stand_in in stand_ins)
    define, wait = ("async def", "await ") if function.is_async else ("def", "")
    source = (
        f"{define} stub({signature}):\n"
        "    __tracebackhide__ = True\n"  # IPython leaves this frame out of a cell's traceback
        f"    return {wait}__call(__name, __passed(({values})))\n"
    )
    scope: dict[str, Any] = {
        "__name__": __name__,
        "__call": bridge.call_async if function.is_async else bridge.call,
        "__name": function.name,
        "__passed": passed,
        "__not_passed": NOT_PASSED,
    }
    exec(compile(source, "<cell function>", "exec"), scope)
    stub = scope["stub"]
    code = stub.__code__
    stub.__code__ = code.replace(
        co_varnames=names + code.co_varnames[len(names) :], co_name=function.name, co_qualname=function.name
    )
    stub.__name__ = stub.__qualname__ = function.name  # also the name of an async stub's coroutines
    stub.__doc__ = function.description
    stub.__signature__ = _signature(parameters)
    return stub


def _signature(parameters: Sequence[Parameter]) -> inspect.Signature:
    """What ``help()`` shows for a stub: its parameters, or ``(*args, **kwargs)`` for names Python cannot show."""

    try:
        return inspect.Signature(
            [
                inspect.Parameter(
                    parameter.name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=inspect.Parameter.empty if parameter.required else parameter.default,
                )
                for parameter in parameters
            ]
        )
    except (ValueError, IndexError):  # a name such as ``from``, ``file-path`` or ``""``
        return inspect.Signature(
            [
                inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL),
                inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
            ]
        )


def _create_shell(namespace: Mapping[str, Any]) -> InteractiveShellEmbed:
    """A quiet embedded IPython shell whose cells see ``namespace`` and import only the safe ``asyncio``."""

    original_excepthook = sys.excepthook
    config = Config()
    # history keeps files open preventing making > ~50 envs
    config.HistoryManager.enabled = False
    shell = InteractiveShellEmbed(config=config)
    sys.excepthook = original_excepthook  # prevents it from changing traceback format globally
    _drop_atexit_handlers(shell)
    shell.user_ns.update(namespace)
    # Intercept `import asyncio` so cells cannot bypass SafeAsyncio. The builtins are copied: patching the
    # global ones would make the original import point at the sandboxed one (infinite recursion).
    shell.user_ns["__builtins__"] = {**vars(builtins), "__import__": _sandboxed_import}
    return shell


def _drop_atexit_handlers(shell: InteractiveShellEmbed) -> None:
    """Unregister the atexit handlers IPython gives every shell.

    They pin the shell and its namespace until the process exits, and ``atexit_operations`` then runs a full
    ``gc.collect()`` per shell. Shells are made per agent, so a process running rollouts in-process would keep every
    one. With history disabled there is nothing to flush at exit. ``ScriptMagics`` (created with the shell before
    IPython 9.17, at its first use after) registers ``kill_bg_processes``, which its ``__del__`` also runs.
    """

    atexit.unregister(shell.atexit_operations)
    script_magics = shell.magics_manager.registry.get("ScriptMagics")
    if script_magics is not None:
        atexit.unregister(script_magics.kill_bg_processes)


class CellExecutor:
    """One agent's IPython shell and cell loop (see the module docstring); use it as ``async with``.

    Entering it, on the agent's loop, creates the shell and binds the cell functions to that loop.

    Args:
        functions: What cells can call; each becomes a function of the cell namespace (:func:`cell_stub`).
            ``asyncio`` is always the restricted :data:`safe_asyncio`.
        checks: Which of the refusals apply (``checks.check_cell``); the unawaited-call check covers the
            async functions.
        max_output_chars: Cut each of a cell's output and error text to this length (None keeps everything).
    """

    def __init__(
        self,
        functions: Sequence[CellFunction],
        *,
        checks: Checks = Checks(),
        max_output_chars: int | None = None,
    ) -> None:
        self.functions = list(functions)
        self.async_names = frozenset(function.name for function in self.functions if function.is_async)
        self.checks = checks
        self.max_output_chars = max_output_chars
        self.shell: InteractiveShellEmbed | None = None
        self._bridge: LoopBridge | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stopping: threading.Event | None = None  # set when the cell loop must stop

    async def __aenter__(self) -> CellExecutor:
        invokes = {function.name: function.invoke for function in self.functions}
        self._bridge = LoopBridge(asyncio.get_running_loop(), invokes)
        namespace = {function.name: cell_stub(function, self._bridge) for function in self.functions}
        self.shell = _create_shell({**namespace, "asyncio": safe_asyncio})
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Stop the cells: close the bridge, then let the cell loop cancel what the cells left running and stop.

        A cell blocked in a call stops waiting at once, and any later call raises in the cell; a call the exit cancels
        has seen its cancellation when the exit returns. The cell loop stops itself (:func:`_stop_cells`), so a
        cancelled exit (a parent cancelling its children again) cannot leave it running. A cell stuck in synchronous
        code cannot be interrupted: the exit gives up on it after 2 x ``LOOP_SHUTDOWN_TIMEOUT_S`` and leaves its daemon
        thread behind, still running the cell but unable to reach the env.
        """

        bridge, shell, loop, thread, stopping = self._bridge, self.shell, self._loop, self._thread, self._stopping
        self._bridge = self.shell = self._loop = self._thread = self._stopping = None  # run() raises from now on
        self.functions = []
        if bridge is not None:
            bridge.close()
        if shell is not None:
            _drop_atexit_handlers(shell)  # ScriptMagics may have come with a cell since the shell was made
        if loop is None or thread is None or stopping is None:
            return
        stop = _stop_cells(stopping)
        try:
            asyncio.run_coroutine_threadsafe(stop, loop)
        except RuntimeError:  # the loop is closed: an unexpected error ended its thread
            stop.close()
            return
        await asyncio.to_thread(thread.join, 2 * LOOP_SHUTDOWN_TIMEOUT_S)
        # The cancelled calls' tasks run on this loop. On Python 3.13 and later the await above can return without
        # yielding to it (asyncio completes the wrapped future at once when the thread had ended by then), which would
        # leave their cancellations for after the agent went on: give the loop its turns.
        for _ in range(SETTLE_ITERATIONS):
            await asyncio.sleep(0)

    async def run(self, code: str, *, timeout_s: float | None = None) -> CellResult:
        """Run one cell: the refusal when it must not run, else what it printed; cancelled after ``timeout_s``."""

        if self.shell is None or self._bridge is None:
            raise RuntimeError("Run cells inside `async with CellExecutor(...)`")
        code = code.strip()
        refusal = check_cell(code, self.checks, self.async_names)
        if refusal is not None:
            return CellResult(code, error=refusal)

        cell = object()
        future = asyncio.run_coroutine_threadsafe(self._run_captured(self.shell, code, cell), self._cell_loop())
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.wrap_future(future), timeout_s)
        except asyncio.TimeoutError:
            self._bridge.retire(cell)  # first, so the cell cannot call again once its blocked call is cancelled
            future.cancel()
            self._bridge.cancel_in_flight()  # a cell blocked in a synchronous call stops waiting
            return CellResult(code, error=f"Cell did not finish within {timeout_s:g} seconds", timed_out=True)
        except asyncio.CancelledError:
            self._bridge.retire(cell)
            future.cancel()
            self._bridge.cancel_in_flight()
            raise

        stdout = drop_out_prompts(strip_ansi_escape_sequences(stdout))  # a value's line, without IPython's Out[n]:
        stdout = shorten_install_paths(stdout)  # "File json/decoder.py:338", not where Python is installed
        stderr = shorten_install_paths(strip_ansi_escape_sequences(stderr))
        return CellResult(
            code,
            output=truncate_output(stdout, self.max_output_chars),
            error=truncate_output(stderr, self.max_output_chars),
        )

    @staticmethod
    async def _run_captured(shell: InteractiveShellEmbed, code: str, cell: object) -> tuple[str, str]:
        """On the cell loop: run the cell and return what it printed to stdout and stderr."""

        _CELL.set(cell)  # in this task's own context, which the tasks the cell starts copy
        with ShellCapture() as capture:
            await run_cell_async(shell, code)
        return capture.pop_stdout(), capture.pop_stderr()

    def _cell_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            loop = _CellLoop()
            started, stopping = threading.Event(), threading.Event()  # locals: the thread must not keep the executor

            def serve() -> None:
                asyncio.set_event_loop(loop)
                loop.call_soon(started.set)
                try:
                    while not stopping.is_set():  # only _stop_cells ends the loop: a cell may call its stop() too
                        try:
                            loop.run_forever()
                        except (SystemExit, KeyboardInterrupt):
                            pass  # re-raised by asyncio from a task a cell started; the task's awaiter still gets it
                finally:
                    loop.close()

            self._thread = threading.Thread(target=serve, name="codeact-cells", daemon=True)
            self._thread.start()
            started.wait()
            self._loop, self._stopping = loop, stopping
        return self._loop


async def _stop_cells(stopping: threading.Event) -> None:
    """On the cell loop: cancel what the cells left running, give it ``LOOP_SHUTDOWN_TIMEOUT_S`` to end, then stop
    the loop (its thread closes it)."""

    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    if tasks:
        done, _ = await asyncio.wait(tasks, timeout=LOOP_SHUTDOWN_TIMEOUT_S)
        for task in done:
            if not task.cancelled():
                task.exception()  # retrieved, so asyncio does not log it
    # Let the callbacks these endings scheduled run before the loop stops. The calls closing the bridge cancelled
    # may have ended before this coroutine started, and a gather of them that no cell awaited learns of that one
    # callback later, then retrieves its outcome one callback after that (SafeAsyncio.gather).
    for _ in range(SETTLE_ITERATIONS):
        await asyncio.sleep(0)
    stopping.set()
    asyncio.get_running_loop().stop()


__all__ = ["CellExecutor", "CellResult", "LoopBridge", "SafeAsyncio", "cell_stub", "safe_asyncio", "truncate_output"]
