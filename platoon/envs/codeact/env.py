from __future__ import annotations

import ast
import asyncio
import builtins
import sys
from copy import deepcopy
from typing import Callable, Protocol, Sequence, runtime_checkable

from IPython.terminal.embed import InteractiveShellEmbed
from traitlets.config.loader import Config

from platoon.agents.actions.common import finish
from platoon.envs.base import Task
from platoon.episode.context import (
    current_trajectory,
    current_trajectory_collection,
    error_message,
    finish_message,
)
from platoon.utils.ipython_shell import ShellCapture, strip_ansi_escape_sequences

from .types import (
    CodeActAction,
    CodeActObservation,
    CodeActStep,
    CodeExecutor,
    ForkableCodeExecutor,
)


@runtime_checkable
class CodeActEnv(Protocol):
    # TODO: Come back to think about how we might want to add a factory + config to be string-based configurable.
    def __init__(
        self,
        task: Task,
        code_executor: CodeExecutor,
        return_obs_copy: bool = True,
        parent_state: CodeActObservation | None = None,
        **kwargs,
    ):
        self._code_executor = code_executor
        self._task = task
        self._parent_state = parent_state
        self._state = CodeActObservation(task=task, history=[])
        if parent_state is not None:
            self._state.misc["parent_state"] = parent_state
        self._return_obs_copy = return_obs_copy
        self._init_kwargs = kwargs

    @property
    def task(self) -> Task:
        return self._task

    @property
    def code_executor(self) -> CodeExecutor:
        return self._code_executor

    async def reset(self) -> CodeActObservation:
        self._code_executor = await self._code_executor.reset()
        self._state = CodeActObservation(task=self._state.task, history=[])
        if self._parent_state is not None:
            self._state.misc["parent_state"] = self._parent_state
        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj_collection.set_trajectory_task(traj.id, self._state.task)
        traj.reward = 0.0
        # This is not needed here, but we keep it for demo in case env wants to add steps/obs during reset.
        for step in self._state.history:
            traj_collection.add_trajectory_step(traj.id, step)
        return await self.observe()

    async def evaluate(self) -> tuple[float, dict]:
        return 0.0, {}

    async def step(self, action: CodeActAction) -> CodeActObservation:
        step = await self._code_executor.run(action.parsed_code)

        if finish_message.get(None) is not None or error_message.get(None) is not None:
            self._state.finished = True
            self._state.misc["finish_message"] = finish_message.get()

        step.thought = action.parsed_thought
        step.reward, reward_info = await self.evaluate()
        step.misc["action_misc"] = action.misc
        step.misc["reward_misc"] = reward_info
        self._state.reward += step.reward
        self._state.history.append(step)

        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj_collection.add_trajectory_step(traj.id, self._state.history[-1])
        if self._state.finished:
            traj.reward = self._state.reward

        return await self.observe()

    async def close(self) -> None:
        self._state = CodeActObservation(task=self._state.task, history=[])
        await self._code_executor.close()

    async def observe(self, return_copy: bool | None = None) -> CodeActObservation:
        if return_copy is None:
            return_copy = self._return_obs_copy

        if return_copy:
            return deepcopy(self._state)
        else:
            return self._state

    async def fork(self, task: Task) -> CodeActEnv:
        if isinstance(self._code_executor, ForkableCodeExecutor):
            return type(self)(
                task=task,
                code_executor=await self._code_executor.fork(task=task),
                return_obs_copy=self._return_obs_copy,
                parent_state=await self.observe(),
                **deepcopy(self._init_kwargs),
            )
        else:
            raise ValueError(
                "CodeExecutor is not forkable. "
                "Either implement fork() for your CodeExecutor or implement a new ForkableEnv for this task."
            )


class SafeAsyncio:
    """A restricted asyncio wrapper that only exposes safe operations.

    Blocks dangerous operations like set_event_loop(), new_event_loop(), run(),
    and set_event_loop_policy() that can cause deadlocks or interfere with
    the host event loop.
    """

    __name__ = "asyncio"  # So it's injected with the correct name in shell namespace

    # Safe operations for concurrent execution
    gather = staticmethod(asyncio.gather)
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
    iscoroutinefunction = staticmethod(asyncio.iscoroutinefunction)

    # Timeout context manager (Python 3.11+)
    if hasattr(asyncio, "timeout"):
        timeout = staticmethod(asyncio.timeout)
    if hasattr(asyncio, "timeout_at"):
        timeout_at = staticmethod(asyncio.timeout_at)

    # Expose TimeoutError for catching
    TimeoutError = asyncio.TimeoutError

    # Block everything else - especially dangerous operations like:
    # set_event_loop, new_event_loop, run, set_event_loop_policy, get_event_loop
    def __getattr__(self, name):
        raise RuntimeError(
            f"asyncio.{name} is disabled in sandbox. "
            f"Only safe operations like gather, create_task, sleep, wait, wait_for, "
            f"and synchronization primitives (Lock, Event, Queue, etc.) are allowed."
        )


# Singleton instance to be used as the "asyncio" module in the sandbox
safe_asyncio = SafeAsyncio()


def _make_sandboxed_import(safe_asyncio_instance: SafeAsyncio):
    """Create a sandboxed import function that intercepts asyncio imports.

    This ensures that even if an agent does `import asyncio` or
    `from asyncio import something`, they get our safe wrapper instead.
    """
    _original_import = builtins.__import__

    def _sandboxed_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Intercept asyncio and all its submodules
        if name == "asyncio" or name.startswith("asyncio."):
            # For `from asyncio import X` or `import asyncio`, return safe wrapper
            # Python will then fetch attributes from it
            return safe_asyncio_instance
        return _original_import(name, globals, locals, fromlist, level)

    return _sandboxed_import


class UnawaitedAsyncCallDetector(ast.NodeVisitor):
    """AST visitor to detect calls to async functions without 'await'.

    This catches common mistakes like:
        launch_subagent(...)  # Missing await
        x = launch_subagent(...)  # Missing await

    But allows:
        await launch_subagent(...)
        result = await launch_subagent(...)
        asyncio.gather(launch_subagent(...), ...)  # Wrapped in gather
    """

    # Functions that must be awaited
    ASYNC_FUNCTIONS = {"launch_subagent"}

    def __init__(self):
        self.errors: list[str] = []
        self._in_await = False
        self._in_gather_or_similar = False

    def visit_Await(self, node: ast.Await) -> None:
        # Mark that we're inside an await expression
        old_in_await = self._in_await
        self._in_await = True
        self.generic_visit(node)
        self._in_await = old_in_await

    def visit_Call(self, node: ast.Call) -> None:
        # Check if this is a call to asyncio.gather or similar
        func = node.func
        is_gather_like = False
        if isinstance(func, ast.Attribute):
            if func.attr in ("gather", "wait", "wait_for", "create_task", "as_completed"):
                is_gather_like = True

        if is_gather_like:
            # Inside gather-like calls, coroutines are allowed without direct await
            old_in_gather = self._in_gather_or_similar
            self._in_gather_or_similar = True
            self.generic_visit(node)
            self._in_gather_or_similar = old_in_gather
        else:
            # Check if this call is to an async function that needs await
            func_name = None
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr

            if func_name in self.ASYNC_FUNCTIONS:
                if not self._in_await and not self._in_gather_or_similar:
                    self.errors.append(
                        f"ERROR: {func_name}() requires 'await' to run. "
                        f"Use: `await {func_name}(...)` or `await asyncio.gather({func_name}(...), ...)`"
                    )

            self.generic_visit(node)


class IPythonCodeExecutor(CodeExecutor):
    # TODO: Separate actions and modules? Use this info to build action space description?
    def __init__(
        self,
        task: Task,
        actions: tuple[Callable[..., object], ...] | Sequence[Callable[..., object]] = (finish, safe_asyncio),
    ):
        self.task = task
        self.actions = actions
        self.shell = self._create_shell()
        # self.timeout_seconds = timeout_seconds

    def _create_shell(self) -> InteractiveShellEmbed:
        original_excepthook = sys.excepthook
        config = Config()
        # history keeps files open preventing making > ~50 envs
        config.HistoryManager.enabled = False
        shell = InteractiveShellEmbed(config=config)
        sys.excepthook = original_excepthook  # prevents it from changing traceback format globally
        for action in self.actions:
            shell.user_ns[action.__name__] = action

        # Install sandboxed import to intercept `import asyncio` statements
        # This ensures agents can't bypass SafeAsyncio by importing directly
        # IMPORTANT: We must create a COPY of __builtins__ to avoid modifying the global
        # builtins which would cause infinite recursion (since _original_import would
        # then point to our sandboxed version)
        sandboxed_import = _make_sandboxed_import(safe_asyncio)
        existing_builtins = shell.user_ns.get("__builtins__")
        if isinstance(existing_builtins, dict):
            # Create a copy of the builtins dict
            shell.user_ns["__builtins__"] = {**existing_builtins, "__import__": sandboxed_import}
        else:
            # __builtins__ is a module - convert to dict with our custom import
            shell.user_ns["__builtins__"] = {**vars(existing_builtins), "__import__": sandboxed_import}

        return shell

    # TODO: Can make this more robust and have better error handling + sandboxing + timeouts.
    async def run(self, code: str) -> CodeActStep:
        code = code.strip()

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            message = (
                "Execution failed. Traceback:\n"
                + "Syntax error in line:\n"
                + (e.text or "").rstrip()
                + "\n"
                + "Message: "
                + e.msg
            )
            return CodeActStep(code=code, error=message)

        if not code:
            return CodeActStep(code=code, error="No code available to execute.")

        # Check for unawaited async function calls before execution
        detector = UnawaitedAsyncCallDetector()
        detector.visit(tree)
        if detector.errors:
            return CodeActStep(code=code, error="\n".join(detector.errors))

        with ShellCapture() as capture:
            await self.shell.run_cell_async(code)

        cap_stdout = strip_ansi_escape_sequences(capture.pop_stdout())
        cap_stderr = strip_ansi_escape_sequences(capture.pop_stderr())

        # TODO: This might cause unexpected filtering of outputs.
        # Guard against empty stdout before indexing first line
        first_line = cap_stdout.splitlines()[0] if cap_stdout.splitlines() else ""
        if cap_stdout.startswith("Out[") or ("[?7hOut[1]:" in first_line):
            cap_stdout = "".join(cap_stdout.split(":")[1:])

        return CodeActStep(
            code=code,
            output=cap_stdout,
            error=cap_stderr,
        )

    async def describe_action_space(self) -> str:
        raise NotImplementedError(
            "IPythonCodeExecutor does not yet implement describe_action_space. Please implement it in your subclass."
        )

    async def reset(self) -> CodeExecutor:
        return self
