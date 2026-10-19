"""CodeAct cells: IPython's cell transform on the agent's cell loop, the stubs that hand cell-function calls to the
agent's loop, the original agent's refusals and output handling, and what leaving the executor stops."""

from __future__ import annotations

import asyncio
import gc
import inspect
import json
import logging
import os
import re
import threading
import time
import warnings
import weakref
from pathlib import Path
from typing import Any

import IPython
import pytest
from test_codeact_namespace import FakeContext

import platoon
from platoon.harness.builtin.codeact import CellFunction, Checks, CodeActPromptBuilder, Parameter, ipython_shell
from platoon.harness.builtin.codeact.executor import (
    CellExecutor,
    CellResult,
    LoopBridge,
    _CellLoop,
    safe_asyncio,
    truncate_output,
)
from platoon.harness.builtin.codeact.ipython_shell import drop_out_prompts
from platoon.harness.builtin.codeact.namespace import Ending, finish_function, launch_subagent_function


def _function(name: str, invoke: Any, *parameters: Parameter, is_async: bool = False) -> CellFunction:
    return CellFunction(name, list(parameters), f"The {name} function.", invoke, is_async=is_async)


async def _run(code: str, functions=(), *, timeout_s: float | None = None, **kwargs) -> CellResult:
    async with CellExecutor(list(functions), **kwargs) as cells:
        return await cells.run(code, timeout_s=timeout_s)


def _last_line(cell: CellResult) -> str:
    return cell.output.strip().splitlines()[-1]


class Recorder:
    """Cell functions that record their calls: ``echo(text, suffix="")`` and ``await fetch(url)``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    def functions(self) -> list[CellFunction]:
        async def echo(arguments: dict[str, Any]) -> str:
            self.calls.append(("echo", arguments, threading.current_thread().name))
            await asyncio.sleep(0.01)
            return arguments["text"] + arguments.get("suffix", "")

        async def fetch(arguments: dict[str, Any]) -> str:
            self.calls.append(("fetch", arguments, threading.current_thread().name))
            await asyncio.sleep(0.01)
            return f"page {arguments['url']}"

        return [
            _function("echo", echo, Parameter("text", "str"), Parameter("suffix", "str", required=False, default="")),
            _function("fetch", fetch, Parameter("url", "str"), is_async=True),
        ]


class SlowCall:
    """A cell function whose ``invoke`` sleeps ``delay`` seconds on the agent's loop; records starts and cancels."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.started: list[float] = []
        self.cancelled: list[float] = []

    def function(self, name: str = "slow", *, is_async: bool = False) -> CellFunction:
        async def invoke(arguments: dict[str, Any]) -> str:
            self.started.append(time.monotonic())
            try:
                await asyncio.sleep(self.delay)
            except asyncio.CancelledError:
                self.cancelled.append(time.monotonic())
                raise
            return "done"

        return _function(name, invoke, is_async=is_async)


# ---------------------------------------------------------------------------- cells


@pytest.mark.asyncio
async def test_cell_output_is_captured():
    assert await _run("x = 6 * 7\nprint(x)") == CellResult("x = 6 * 7\nprint(x)", output="42\n", error="")


@pytest.mark.asyncio
async def test_cell_can_await_at_top_level():
    # Recursive agents launch subagents with a top-level ``await`` in a cell.
    cell = await _run("await asyncio.sleep(0)\nprint('awaited')")

    assert (cell.output, cell.error) == ("awaited\n", "")


@pytest.mark.asyncio
async def test_a_cell_exception_is_reported_as_its_output():
    # IPython prints tracebacks to stdout: they are part of the output, and the error stays empty.
    cell = await _run("1 / 0")

    assert _last_line(cell) == "ZeroDivisionError: division by zero" and cell.error == "" and not cell.timed_out


@pytest.mark.asyncio
async def test_refused_cells_never_run():
    cell = await _run("  print('ran')\nwhile True: pass  ", checks=Checks(while_loops=True))

    assert cell.code == "print('ran')\nwhile True: pass"  # stripped
    assert cell.output is None and cell.error.startswith("ERROR: `while` loops are not allowed")


@pytest.mark.asyncio
async def test_the_unawaited_check_covers_the_async_functions():
    functions = Recorder().functions()

    assert (await _run("fetch('a')", functions)).error.startswith("ERROR: fetch() requires 'await' to run.")
    assert (await _run("print(await asyncio.gather(fetch('a'), fetch('b')))", functions)).output == (
        "['page a', 'page b']\n"
    )
    assert (await _run("echo('a')", functions)).error == ""  # synchronous
    # Only the agent's own async functions count: the original agent also refused its archived names (search_web,
    # read_email, ...), which would refuse calls to an env's synchronous tools of the same name.
    assert "NameError: name 'search_web' is not defined" in (await _run("search_web('a')")).output
    unchecked = await _run("fetch('a').close()", functions, checks=Checks(unawaited_async_calls=False))
    assert (unchecked.output, unchecked.error) == ("", "")


def test_a_gather_a_cell_never_awaits_leaves_no_unretrieved_exception(caplog):
    # A cell that gathers launches without awaiting them, then ends: leaving the executor cancels the gathered
    # calls, which sets the gather's exception. Nobody awaits it, and asyncio must not log it as never retrieved.
    async def launch(arguments: dict[str, Any]) -> None:
        await asyncio.sleep(30)

    launch_subagent = _function("launch_subagent", launch, Parameter("goal", "str"), is_async=True)

    async def scenario() -> None:
        async with CellExecutor([launch_subagent]) as cells:
            cell = await cells.run("asyncio.gather(launch_subagent('a'), launch_subagent('b'))\nprint('launched')")
            assert cell.output == "launched\n"

    with caplog.at_level(logging.ERROR, logger="asyncio"):
        asyncio.run(scenario())
        gc.collect()

    assert "never retrieved" not in caplog.text


@pytest.mark.asyncio
async def test_futures_and_tasks_in_a_cells_output_name_no_paths():
    # A pending gather used to print as "<_GatheringFuture pending cb=[_retrieve() at
    # /abs/path/.../executor.py:85]>", and a task's repr names asyncio's own file, so the cell's output put the
    # absolute install paths into the model's prompt.
    functions = [SlowCall(0.3).function("launch", is_async=True)]
    async with CellExecutor(functions) as cells:
        gathered = await cells.run("g = asyncio.gather(launch(), launch())\ng")
        printed = await cells.run("print(g)")
        task = await cells.run("t = asyncio.create_task(launch())\nawait asyncio.sleep(0.05)\nt")
        listed = await cells.run("print([t], asyncio.current_task(), asyncio.all_tasks())")
        future = await cells.run("asyncio.get_running_loop().create_future()")
        own = await cells.run("async def own():\n    await asyncio.sleep(1)\nasyncio.create_task(own())")
        done = await cells.run("await asyncio.sleep(0.5)\nprint(g, t)")

    for cell in (gathered, printed, task, listed, future, own, done):
        assert cell.error == "" and "/" not in cell.output and ".py" not in cell.output, cell
    assert gathered.output == " <_GatheringFuture pending>\n"  # what the original agent's cells printed
    assert printed.output == "<_GatheringFuture pending>\n"
    assert re.fullmatch(r" <Task pending name='Task-\d+' coro=<launch\(\)>>\n", task.output)
    assert "coro=<CellExecutor._run_captured()>" in listed.output
    assert future.output == " <Future pending>\n"
    assert re.fullmatch(r" <Task pending name='Task-\d+' coro=<own\(\)>>\n", own.output)
    assert re.fullmatch(
        r"<_GatheringFuture finished result=\['done', 'done'\]> "
        r"<Task finished name='Task-\d+' coro=<launch\(\)> result='done'>\n",
        done.output,
    )


def test_a_cell_loop_task_named_none_keeps_its_default_name():
    # Python 3.13's loop.create_task calls set_name(None) on every task a task factory makes, so a cell's task
    # printed as name='None' there; on any Python, a task of the cell loop keeps its name when it is named None.
    loop = _CellLoop()

    async def nothing() -> None:
        pass

    try:
        task = loop.create_task(nothing())
        default = task.get_name()
        task.set_name(None)
        assert task.get_name() == default and re.fullmatch(r"Task-\d+", default)
        task.set_name("named")
        assert task.get_name() == "named"
        loop.run_until_complete(task)
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_cells_get_the_restricted_asyncio_even_when_they_import_it():
    cell = await _run("import asyncio\nasyncio.run(None)")

    assert "asyncio.run is disabled in sandbox" in cell.output + cell.error


@pytest.mark.asyncio
async def test_last_expression_prompt_is_dropped_as_the_original_agent_dropped_it():
    # IPython prints a cell's value after its "Out[n]:" prompt; the line keeps the space after it, as it did there.
    cell = await _run("'Too low, try again.'")

    assert cell.output == " 'Too low, try again.'\n"


@pytest.mark.asyncio
async def test_out_prompts_are_dropped_line_by_line_as_the_original_agent_meant():
    # The original agent cut an output that starts with the prompt at its first colon and dropped every other colon,
    # and it kept a prompt that followed printed lines.
    assert (await _run("{'a': 1}")).output == " {'a': 1}\n"
    assert (await _run("print('first: line')\n7")).output == "first: line\n 7\n"
    assert (await _run("print('Out of steps: no')")).output == "Out of steps: no\n"


def test_drop_out_prompts_keeps_the_rest_of_each_line():
    assert drop_out_prompts("Out[0]: 'a: b'\n") == " 'a: b'\n"
    assert drop_out_prompts("x\nOut[12]: \n{'a': 1,\n 'b': 2}\n") == "x\n \n{'a': 1,\n 'b': 2}\n"
    assert drop_out_prompts("\x1b[?7hOut[1]: 5\n") == " 5\n"  # a terminal's escape before the prompt goes too
    assert drop_out_prompts("see Out[1]: here\n") == "see Out[1]: here\n"  # only at the start of a line


@pytest.mark.asyncio
async def test_a_cell_that_outlives_its_timeout_is_cancelled_and_reported():
    async with CellExecutor([]) as cells:
        cell = await cells.run("await asyncio.sleep(30)", timeout_s=0.2)
        after = await cells.run("print('next')")

    assert cell == CellResult("await asyncio.sleep(30)", error="Cell did not finish within 0.2 seconds", timed_out=True)
    assert after.output == "next\n"  # the cell loop is free again


@pytest.mark.asyncio
async def test_output_is_cut_at_max_output_chars():
    cell = await _run("print('x' * 50)", max_output_chars=10)

    assert cell.output == "x" * 10 + "\n...[output truncated: 41 more characters]..."
    assert truncate_output("short", None) == "short"


@pytest.mark.asyncio
async def test_concurrent_cells_keep_their_own_output():
    async with CellExecutor([]) as first, CellExecutor([]) as second:
        cells = await asyncio.gather(
            first.run("for i in range(3):\n    print('first', i)\n    await asyncio.sleep(0.01)"),
            second.run("for i in range(3):\n    print('second', i)\n    await asyncio.sleep(0.01)"),
        )

    assert cells[0].output == "first 0\nfirst 1\nfirst 2\n"
    assert cells[1].output == "second 0\nsecond 1\nsecond 2\n"


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:To exit:UserWarning")  # IPython's hint after a SystemExit
async def test_system_exit_in_a_task_a_cell_started_does_not_stop_the_cell_loop():
    # asyncio re-raises SystemExit out of the loop running the task; the cell still gets it, and the loop goes on.
    code = "async def f():\n    import sys\n    sys.exit(1)\nawait asyncio.gather(f())"
    async with CellExecutor([]) as cells:
        cell = await cells.run(code, timeout_s=5)
        after = await cells.run("print('next cell')", timeout_s=5)

    assert not cell.timed_out and "SystemExit: 1" in cell.output
    assert after.output == "next cell\n"


@pytest.mark.asyncio
async def test_a_cell_that_stops_its_loop_does_not_end_it():
    # SafeAsyncio hands cells the running loop; stopping it must neither lose the cell nor end the cell loop.
    async with CellExecutor([]) as cells:
        cell = await cells.run("asyncio.get_running_loop().stop()\nprint('stopped')", timeout_s=5)
        after = await cells.run("print('next cell')", timeout_s=5)
        thread = cells._thread

    assert (cell.output, after.output) == ("stopped\n", "next cell\n")
    assert not thread.is_alive()


# ---------------------------------------------------------------------------- leaving the executor


@pytest.mark.asyncio
async def test_leaving_the_executor_cancels_what_cells_left_running():
    async with CellExecutor([]) as cells:
        await cells.run("import asyncio\nleft = asyncio.create_task(asyncio.sleep(30))")
        left = cells.shell.user_ns["left"]
        thread = cells._thread

    assert left.cancelled() and not thread.is_alive()


@pytest.mark.asyncio
async def test_cells_run_only_inside_the_executor():
    def cell_threads() -> list[threading.Thread]:
        return [thread for thread in threading.enumerate() if thread.name == "codeact-cells"]

    cells = CellExecutor([])
    with pytest.raises(RuntimeError, match="async with CellExecutor"):
        await cells.run("print(1)")
    async with cells:
        await cells.run("print(1)")
    before = cell_threads()
    with pytest.raises(RuntimeError, match="async with CellExecutor"):
        await cells.run("print(1)")  # after the exit too, and it starts no cell thread
    assert cell_threads() == before


@pytest.mark.asyncio
async def test_a_timeout_cancels_the_call_a_cell_is_blocked_in_and_the_cell_calls_nothing_after():
    # The cell catches what its calls raise and keeps calling: none of those calls may reach the agent's loop.
    slow = SlowCall(delay=2.0)
    code = "for i in range(4):\n    try:\n        print(slow())\n    except BaseException as error:\n        pass"
    async with CellExecutor([slow.function()]) as cells:
        cell = await cells.run(code, timeout_s=0.3)
        timed_out_at = time.monotonic()
        thread = cells._thread
    left_at = time.monotonic()
    await asyncio.sleep(0.3)

    assert cell.timed_out
    assert len(slow.started) == 1 and len(slow.cancelled) == 1  # the call it was blocked in, cancelled at once
    assert left_at - timed_out_at < 2 and not thread.is_alive()  # the exit did not wait for the tool


@pytest.mark.asyncio
async def test_a_call_cancelled_before_the_agent_s_loop_starts_it_never_starts():
    # A timed-out cell that calls again before its agent ends: the call waits in the agent's loop's queue when leaving
    # the executor closes the bridge, which cancels it. Under load the cell can win that race (the test above).
    slow = SlowCall(delay=2.0)
    invoke = slow.function().invoke
    coroutines = []

    def kept(arguments: dict[str, Any]) -> Any:
        coroutines.append(invoke(arguments))
        return coroutines[-1]

    bridge = LoopBridge(asyncio.get_running_loop(), {"slow": kept})
    submitted = []
    cell = threading.Thread(target=lambda: submitted.append(bridge.submit("slow", {})))
    cell.start()
    cell.join()  # holds up the agent's loop, so the call is queued on it and has not started
    bridge.close()
    await asyncio.sleep(0.1)

    assert submitted[0].cancelled() and slow.started == []
    # Its coroutine is closed, so it never warns that it was never awaited: Python 3.13 and later cancel the call's
    # task before the task's first step, the step that closes it on 3.12.
    assert inspect.getcoroutinestate(coroutines[0]) == inspect.CORO_CLOSED


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_a_timed_out_cell_cannot_call_again_while_the_executor_stays_open(is_async):
    # The cell swallows the cancellation of its blocked call and calls again at once; with the executor still open, a
    # cell that timed out must not reach the env again (a synchronous one runs on until it ends).
    slow = SlowCall(delay=2.0)
    call = "await slow()" if is_async else "slow()"
    code = f"for i in range(50):\n    try:\n        {call}\n    except BaseException:\n        pass"
    async with CellExecutor([slow.function(is_async=is_async)]) as cells:
        cell = await cells.run(code, timeout_s=0.3)
        await asyncio.sleep(0.5)
        started = len(slow.started)
        after = await cells.run("print(slow())" if not is_async else "print(await slow())", timeout_s=5)

    assert cell.timed_out and started == 1
    assert after.output == "done\n"  # the next cell calls as before


@pytest.mark.asyncio
async def test_a_timed_out_await_cancels_its_call():
    slow = SlowCall(delay=30)
    cell = await _run("print(await slow())", [slow.function(is_async=True)], timeout_s=0.3)

    assert cell.timed_out and len(slow.started) == len(slow.cancelled) == 1


@pytest.mark.asyncio
async def test_a_call_the_exit_cancels_has_seen_its_cancellation_when_the_exit_returns(monkeypatch):
    # The exit waits for the cell thread with asyncio.to_thread, which on Python 3.13 and later can return without
    # yielding to the agent's loop (when the thread had already ended), so the test above failed under load there:
    # the call's task saw its cancellation only after the exit. Here that wait never yields, on every Python.
    async def without_yielding(function: Any, /, *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", without_yielding)
    slow = SlowCall(delay=30)
    cell = await _run("print(await slow())", [slow.function(is_async=True)], timeout_s=0.3)

    assert cell.timed_out and len(slow.started) == len(slow.cancelled) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_cancelling_a_cell_cancels_its_call(is_async):
    slow = SlowCall(delay=30)
    code = "print(await slow())" if is_async else "print(slow())"
    async with CellExecutor([slow.function(is_async=is_async)]) as cells:
        running = asyncio.create_task(cells.run(code))
        while not slow.started:
            await asyncio.sleep(0.01)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        while not slow.cancelled:
            await asyncio.sleep(0.01)

    assert len(slow.started) == len(slow.cancelled) == 1


@pytest.mark.asyncio
async def test_a_second_cancellation_while_leaving_still_stops_the_cell_loop():
    # A parent cancels a child agent again (cancel_children) while the child is still leaving its executor.
    holder: dict[str, Any] = {}

    async def agent() -> None:
        async with CellExecutor([]) as cells:
            running = asyncio.create_task(cells.run("import time\ntime.sleep(1)"))
            await asyncio.sleep(0.1)
            holder["thread"] = cells._thread
            await running

    task = asyncio.create_task(agent())
    await asyncio.sleep(0.2)
    task.cancel()  # the cell sleeps in synchronous code, so leaving waits for it
    await asyncio.sleep(0.2)
    task.cancel()  # again, while the exit waits
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.to_thread(holder["thread"].join, 5)

    assert not holder["thread"].is_alive()


@pytest.mark.asyncio
async def test_executors_do_not_pin_their_ipython_shells():
    # IPython pins every shell in atexit handlers (atexit_operations, and before 9.17 ScriptMagics'), and the first
    # runs a full gc.collect() per shell at exit. Rollouts create a shell per agent, so a shell must die with its
    # executor; only IPython's singleton, the first shell the process made, stays.
    async with CellExecutor([]) as cells:
        assert (await cells.run("x = 1\nprint(x)")).output == "1\n"
        shell = weakref.ref(cells.shell)

    del cells
    gc.collect()
    assert shell() is None or shell() is type(shell())._instance


@pytest.mark.asyncio
async def test_agents_running_together_release_their_contexts():
    # Cells on several threads swap IPython's process-wide hooks (sys.displayhook, builtins.get_ipython) out of
    # order, which can keep shells alive; the context the cell functions close over must still be released.
    class Context:
        pass

    async def agent() -> weakref.ref:
        context = Context()

        async def tool(arguments: dict[str, Any], context: Context = context) -> str:
            return "ok"

        async with CellExecutor([_function("tool", tool)]) as cells:
            await cells.run("await asyncio.sleep(0.01)\nprint(tool())\n1")
        return weakref.ref(context)

    contexts = await asyncio.gather(*(agent() for _ in range(12)))
    gc.collect()
    assert [context() for context in contexts] == [None] * 12


# ---------------------------------------------------------------------------- cell functions


@pytest.mark.asyncio
async def test_a_call_waits_in_the_cell_while_the_function_runs_on_the_agents_loop():
    recorder = Recorder()

    cell = await _run("print(echo('hi', suffix='!'))\nprint(await fetch('a.com'))", recorder.functions())

    assert cell.output == "hi!\npage a.com\n"
    main = threading.main_thread().name
    assert recorder.calls == [("echo", {"text": "hi", "suffix": "!"}, main), ("fetch", {"url": "a.com"}, main)]


@pytest.mark.asyncio
async def test_a_function_gets_only_the_arguments_the_call_passed_in_parameter_order():
    recorder = Recorder()

    await _run("echo('a')\necho(suffix='', text='b')", recorder.functions())

    assert [list(arguments.items()) for _, arguments, _ in recorder.calls] == [
        [("text", "a")],
        [("text", "b"), ("suffix", "")],  # parameter order, whatever the call's order
    ]


@pytest.mark.asyncio
async def test_a_function_runs_in_a_fresh_context_so_what_it_prints_stays_out_of_the_cell():
    # A regression test: a call used to run in a copy of the cell's context, where ShellCapture redirects stdout,
    # so what the env printed or logged during a tool call (openreward's tool_call_completed line) reached the
    # cell's output, and a subagent inherited its parent cell's capture.
    captures: list[Any] = []

    async def log(arguments: dict[str, Any]) -> str:
        captures.append(ipython_shell._current_capture_out.get())
        print("[info] tool_call_completed tool=log")
        return "logged"

    functions = [_function("log", log), _function("log_async", log, is_async=True)]
    cell = await _run("print(log())\nprint(await log_async())\nprint(await asyncio.gather(log_async()))", functions)

    assert cell.output == "logged\nlogged\n['logged']\n"
    assert captures == [None, None, None]  # no capture: the line went where it goes outside any cell


@pytest.mark.asyncio
async def test_stubs_carry_the_functions_name_docstring_and_signature():
    cell = await _run(
        "import inspect\n"
        "print(echo.__name__, inspect.signature(echo), echo.__doc__)\n"
        "print(inspect.signature(fetch), asyncio.iscoroutinefunction(fetch), asyncio.iscoroutinefunction(echo))\n"
        "print(echo.__module__, echo.__defaults__)",
        Recorder().functions(),
    )

    assert cell.output == (
        "echo (text, suffix='') The echo function.\n(url) True False\n"
        # the stubs are made in the executor; a left-out argument is not passed, and the function applies its default
        "platoon.harness.builtin.codeact.executor (<not passed>,)\n"
    )
    assert cell.error == ""


def test_the_cells_asyncio_iscoroutinefunction_warns_on_no_python():
    # asyncio.iscoroutinefunction is deprecated on Python 3.14, where a cell that called it got the warning in its
    # error (outside pytest, which keeps warnings from reaching a cell).
    async def coroutine_function() -> None:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert safe_asyncio.iscoroutinefunction(coroutine_function) is True
        assert safe_asyncio.iscoroutinefunction(print) is False


@pytest.mark.asyncio
async def test_bad_calls_raise_cpythons_errors_where_they_are_made():
    recorder = Recorder()
    functions = recorder.functions()

    cells = [
        await _run(code, functions)
        for code in (
            "echo('a', 'b', 'c')",
            "echo(suffix='!')",
            "echo('a', text='b')",
            "await fetch(page='a.com')",
        )
    ]

    assert [_last_line(cell) for cell in cells] == [
        "TypeError: echo() takes from 1 to 2 positional arguments but 3 were given",
        "TypeError: echo() missing 1 required positional argument: 'text'",
        "TypeError: echo() got multiple values for argument 'text'",
        "TypeError: fetch() got an unexpected keyword argument 'page'",
    ]
    # Python binds the call, as for a plain function: the traceback shows the cell and nothing of platoon.
    assert not any("File " in cell.output.split("----> 1", 1)[1] for cell in cells)
    assert recorder.calls == []


@pytest.mark.asyncio
async def test_the_builtins_raise_the_original_arity_errors():
    # finish and launch_subagent were real Python functions in the original agent, so these are its texts word for
    # word.
    ctx = FakeContext()
    functions = [
        finish_function(ctx, None, Ending()),
        launch_subagent_function(ctx, CodeActPromptBuilder(), fixed_budget=False),
    ]

    cells = [await _run(code, functions) for code in ("finish('a', 'b')", "print(await launch_subagent())")]

    assert [_last_line(cell) for cell in cells] == [
        "TypeError: finish() takes from 0 to 1 positional arguments but 2 were given",
        "TypeError: launch_subagent() missing 2 required positional arguments: 'targets' and 'num_steps'",
    ]
    assert ctx.tool_calls == [] and ctx.spawned == []


@pytest.mark.asyncio
async def test_a_gather_with_a_bad_call_runs_nothing():
    # The bad call raises while gather's arguments are evaluated, before any coroutine is scheduled.
    recorder = Recorder()

    cell = await _run("await asyncio.gather(fetch(), fetch('b.com'))", recorder.functions())

    assert "TypeError: fetch() missing 1 required positional argument: 'url'" in cell.output
    assert recorder.calls == []


@pytest.mark.asyncio
async def test_any_parameter_name_binds_with_cpythons_words():
    calls: list[dict[str, Any]] = []

    async def send(arguments: dict[str, Any]) -> str:
        calls.append(arguments)
        return "sent"

    mail = [Parameter("to"), Parameter("from"), Parameter("subject"), Parameter("cc", required=False)]
    functions = [
        _function("send", send, *mail),
        _function("ﬁle", send, Parameter("ﬁle")),
        _function("up\x00load", send, Parameter("file-path")),
    ]
    bad = [
        await _run(code, functions)
        for code in (
            "send()",
            "send('a')",
            "send('a', 'b')",
            "send('a', 'b', 'c', 'd', 'e')",
            "send('a', to='b')",
            "send(**{'file-path': 'x'})",
            "globals()['ﬁle'](file='x')",  # Python reads the identifier ﬁle (with the ligature) as file
        )
    ]
    good = await _run(
        "print(send('ann', **{'from': 'bob', 'subject': 'hi'}))\n"
        "print(globals()['ﬁle'](**{'ﬁle': 'x'}))\n"
        "print(globals()['up\\x00load']('x'))\n"
        "import inspect\nprint(inspect.signature(send))",
        functions,
    )

    assert [_last_line(cell) for cell in bad] == [
        "TypeError: send() missing 3 required positional arguments: 'to', 'from', and 'subject'",
        "TypeError: send() missing 2 required positional arguments: 'from' and 'subject'",
        "TypeError: send() missing 1 required positional argument: 'subject'",
        "TypeError: send() takes from 3 to 4 positional arguments but 5 were given",
        "TypeError: send() got multiple values for argument 'to'",
        "TypeError: send() got an unexpected keyword argument 'file-path'",
        "TypeError: ﬁle() got an unexpected keyword argument 'file'",
    ]
    assert good.output == "sent\nsent\nsent\n(*args, **kwargs)\n"  # Python cannot show `from` in a signature
    assert calls == [{"to": "ann", "from": "bob", "subject": "hi"}, {"ﬁle": "x"}, {"file-path": "x"}]


@pytest.mark.asyncio
async def test_a_function_that_raises_shows_the_cell_and_the_function_not_the_plumbing():
    async def fail(arguments: dict[str, Any]) -> str:
        raise ValueError("the env says no")

    cells = [
        await _run(code, [_function("craft", fail, Parameter("item")), _function("launch", fail, is_async=True)])
        for code in ("craft('a')", "await launch()")
    ]

    for cell in cells:
        assert _last_line(cell) == "ValueError: the env says no"
        assert "[... skipping hidden 3 frame]" in cell.output and ".fail(arguments)" in cell.output
        plumbing = ("codeact/executor.py", "concurrent/futures", "in stub", "Could not get source")
        assert not any(text in cell.output for text in plumbing)


@pytest.mark.asyncio
async def test_tracebacks_name_installed_files_by_their_module_paths():
    # An error raised in installed code under a cell function, shown by IPython or by traceback.print_exc() in the
    # cell, named the absolute paths platoon and Python are installed at: 'File "/abs/path/platoon/harness/builtin/
    # codeact/executor.py", line 314, in call_async' and "File ~/abs/path/lib/python3.12/json/decoder.py:338".
    async def parse(arguments: dict[str, Any]) -> Any:
        return json.loads(arguments["text"])

    functions = [
        _function("parse", parse, Parameter("text")),
        _function("parse_later", parse, Parameter("text"), is_async=True),
    ]
    async with CellExecutor(functions) as cells:
        shown = [await cells.run(code) for code in ("parse('{')", "await parse_later('{')")]
        printed = await cells.run(
            "import traceback\ntry:\n    await parse_later('{')\nexcept ValueError:\n    traceback.print_exc()"
        )
        stack = await cells.run("import traceback\ntraceback.print_stack()")
        own = await cells.run("exec(compile('1 / 0', '/elsewhere/script.py', 'exec'))")

    home = os.path.expanduser("~")
    for directory in {str(Path(module.__file__).parents[1]) for module in (platoon, json, IPython)}:
        forms = [directory + "/"] + (["~" + directory[len(home) :] + "/"] if directory.startswith(home + "/") else [])
        for cell in (*shown, printed, stack):
            assert not any(form in cell.output + cell.error for form in forms), (directory, cell)
    for cell in shown:
        assert "File json/decoder.py:" in cell.output and _last_line(cell).startswith("JSONDecodeError: ")
    assert 'File "platoon/harness/builtin/codeact/executor.py", line' in printed.error
    assert 'File "json/decoder.py", line' in printed.error and printed.output == ""
    assert 'File "IPython/core/interactiveshell.py", line' in stack.error
    assert "File /elsewhere/script.py:1" in own.output  # a cell's own file keeps its path


@pytest.mark.asyncio
async def test_a_synchronous_function_cannot_block_the_agents_loop():
    recorder = Recorder()
    async with CellExecutor(recorder.functions()) as cells:
        with pytest.raises(RuntimeError, match="cannot block the agent's loop"):
            cells.shell.user_ns["echo"]("called on the agent's loop")

    assert recorder.calls == []


@pytest.mark.asyncio
async def test_a_required_parameter_after_an_optional_one_is_refused():
    function = _function("odd", lambda arguments: None, Parameter("a", required=False), Parameter("b"))

    with pytest.raises(ValueError, match="a required parameter follows an optional one"):
        async with CellExecutor([function]):
            pass
