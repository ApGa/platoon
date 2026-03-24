from __future__ import annotations

from appworld import AppWorld
from appworld.common.utils import get_stack_trace_from_exception
from IPython.core.interactiveshell import ExecutionResult
from IPython.terminal.embed import InteractiveShellEmbed
from pathlib import Path
from rubric.core.checklist import RubricChecklistFast 
from textwrap import dedent
from traitlets.config.loader import Config
import ast
import sys
import asyncio
import re
import uuid

from platoon.utils.timeout import async_timeout_call
from platoon.agents.actions.common import finish
from platoon.agents.actions.subagent import launch_subagent
from platoon.envs.base import Task, SubTask
from platoon.envs.codeact import CodeExecutor, CodeActEnv, safe_asyncio
from platoon.envs.codeact.env import _make_sandboxed_import
from platoon.utils.ipython_shell import ShellCapture, strip_ansi_escape_sequences
from platoon.episode.context import finish_message, error_message
from platoon.envs.codeact.types import CodeActStep, CodeActObservation
from platoon.utils.prompt_retriever import PromptRetriever
from platoon.appworld.agent import AppWorldCodeActPromptBuilder


def _patch_freezegun_idempotent_stop() -> None:
    """Make freeze_time.stop() idempotent to handle stale Requester entries.

    When a task finishes and AppWorld.close() is called, time freezers are
    stopped. However, class-level Requester state may still hold references to
    those stopped freezers. When the next task calls AppWorld.close_all() during
    initialize(), it attempts to stop already-stopped freezers, causing
    freeze_factories.pop() to raise IndexError on an empty list. Making stop()
    silently ignore that case fixes sequential task execution in subprocess
    workers without modifying the appworld library.
    """
    from appworld.common.time import freeze_time as _AppWorldFreezeTime

    original_stop = _AppWorldFreezeTime.stop

    def _safe_stop(self: _AppWorldFreezeTime) -> None:
        try:
            original_stop(self)
        except IndexError:
            pass  # Freezer was already stopped; treat as a no-op.

    _AppWorldFreezeTime.stop = _safe_stop  # type: ignore[method-assign]


_patch_freezegun_idempotent_stop()


DEFAULT_APPWORLD_TIMEOUT_SECONDS = 1800


class AppWorldAsync(AppWorld):
    
    def __init__(
        self,
        shell_id: str,
        *args,
        allow_silent_success: bool = True,
        **kwargs,
    ):
        try:
            super().__init__(*args, **kwargs)
        except IndexError as e:
            # Handle freezegun race condition when multiple AppWorld instances
            # are initialized concurrently and try to clean up shared state
            if "pop from empty list" in str(e):
                # Retry initialization once - the close_all() should have completed by now
                super().__init__(*args, **kwargs)
            else:
                raise

        self.allow_silent_success = allow_silent_success
        self.shells = {}
        self.register_shell(shell_id)
        if self.remote_environment_url:
            raise ValueError("Remote environment is not supported for AppWorldAsync")
        
    def get_shell(self, shell_id: str) -> InteractiveShellEmbed:
        return self.shells[shell_id]
    
    def register_shell(self, shell_id: str) -> None:
        self.shells[shell_id] = self._create_shell()

    def unregister_shell(self, shell_id: str | None) -> None:
        if shell_id is None:
            return
        shell = self.shells.pop(shell_id, None)
        if shell is None:
            return
        # Best-effort cleanup of shell state for forked subagent contexts.
        shell.user_ns.clear()
    
    def _run_shell_preamble(self, shell: "InteractiveShellEmbed") -> None:
        """Run the lightweight import/print/input preamble in a new shell.

        Replicates the shell.run_cell(preamble) part of _execute_preamble()
        but skips ApiCollection.load().  This avoids expensive set_local_dbs()
        + freeze creation on every shell (primary and all subagent forks).
        All AppWorldAsync shells share the primary apis/requester that was
        set up by AppWorld.initialize().
        """
        import os as _os

        from appworld.environment import TRUE_AVAILABLE_IMPORTS

        preamble = TRUE_AVAILABLE_IMPORTS + "\n\n"
        if self.import_utils:
            from appworld.common.utils import read_file

            file_path = _os.path.join("generate", "tasks", "task_generators", "imports.py")
            utils_import_code = read_file(file_path).strip()
            preamble += utils_import_code
        preamble += dedent(
            """
    def print(*args, **kwargs):
        if not kwargs and len(args) == 1 and isinstance(args[0], (list, tuple, dict)):
            indent = 1 if len(json.dumps(args[0])) >= 100 else None
            builtins.print(json.dumps(args[0], indent=indent))
        else:
            builtins.print(*args, **kwargs)

    def input(*args, **kwargs):
        raise Exception("input(..) is not allowed. All decisions must be made autonomously.")
    """
        )
        shell.run_cell(preamble)

    def _create_shell(self) -> InteractiveShellEmbed:
        original_excepthook = sys.excepthook
        config = Config()
        config.HistoryManager.enabled = (
            False  # history keeps files open preventing making > ~50 envs.
        )
        shell = InteractiveShellEmbed(config=config)
        # Run only the lightweight preamble (imports + print/input overrides).
        # All AppWorldAsync shells share the primary apis/requester set up by
        # AppWorld.initialize().  Calling _execute_preamble() here would run
        # the expensive ApiCollection.load() + set_local_dbs() and start a new
        # time freeze for every shell (primary shell + every subagent fork),
        # all of which would be immediately discarded.
        self._run_shell_preamble(shell)
        shell.user_ns["apis"] = self.apis
        shell.user_ns["requester"] = self.requester
        if self.include_direct_functions:
            separator = self.direct_function_separator
            sub_codes = [
                f"{app_name}{separator}{api_name} = apis.{app_name}.{api_name}"
                for app_name, info in self.apis.items()
                for api_name in info.keys()
            ]
            shell.run_cell("\n".join(sub_codes))
        sys.excepthook = (
            original_excepthook  # prevents it from changing traceback format globally
        )
        shell.user_ns["finish"] = finish
        shell.user_ns["launch_subagent"] = launch_subagent
        shell.user_ns["asyncio"] = safe_asyncio

        sandboxed_import = _make_sandboxed_import(safe_asyncio)
        existing_builtins = shell.user_ns.get("__builtins__")
        if isinstance(existing_builtins, dict):
            shell.user_ns["__builtins__"] = {**existing_builtins, "__import__": sandboxed_import}
        else:
            shell.user_ns["__builtins__"] = {**vars(existing_builtins), "__import__": sandboxed_import}

        return shell
    
    async def _shell_run_cell(self, shell_id: str, code: str) -> ExecutionResult | None:
        self._maybe_raise_remote_environment_error("_shell_run_cell")
        shell = self.get_shell(shell_id)
        if self.timeout_seconds is None:
            return await shell.run_cell_async(code)
        try:
            return await async_timeout_call(
                shell.run_cell_async, timeout_seconds=self.timeout_seconds, raw_cell=code
            )
        except asyncio.TimeoutError:
            return None
        
    async def execute(self, shell_id: str, code: str) -> CodeActStep:
        
        code = code.replace("import asyncio\n", "")
        
        if self.raise_on_unsafe_syntax:
            is_syntax_safe, safety_message = self.safety_guard.is_syntax_safe(code)
            if not is_syntax_safe:
                message = "Execution failed. Traceback:\n" + safety_message
                self.environment_io.append({
                    "number": self.num_interactions + 1,
                    "input": code,
                    "output": message
                })
                return CodeActStep(
                    code=code,
                    error=message,
                )
        
        self.requester.reset_request_count()
        if self.num_interactions >= self.max_interactions:
            # for proper error message.
            code = f'raise Exception(f"Maximum number of executions ({self.max_interactions}) reached.")'

        if self.null_patch_unsafe_execution:
            self.safety_guard.enable()

        code = code.strip()

        try:
            ast.parse(code)
        except SyntaxError as e:
            message = (
                "Execution failed. Traceback:\n"
                + "Syntax error in line:\n"
                + (e.text or "").rstrip()
                + "\n"
                + "Message: "
                + e.msg
            )
            self.environment_io.append({
                "number": self.num_interactions + 1,
                "input": code,
                "output": message
            })
            if self.null_patch_unsafe_execution:
                self.safety_guard.disable()
            return CodeActStep(
                code=code,
                error=message,
            )

        if not code:
            message = "No code available to execute."
            self.environment_io.append({
                "number": self.num_interactions + 1,
                "input": code,
                "output": message
            })
            if self.null_patch_unsafe_execution:
                self.safety_guard.disable()
            return CodeActStep(
                code=code,
                error=message,
            )

        with ShellCapture() as capture:
            result = await self._shell_run_cell(shell_id, code)

        cap_stdout = strip_ansi_escape_sequences(capture.pop_stdout())
        cap_stderr = strip_ansi_escape_sequences(capture.pop_stderr())

        # TODO: This might cause unexpected filtering of outputs.
        # Guard against empty stdout before indexing first line
        first_line = cap_stdout.splitlines()[0] if cap_stdout.splitlines() else ""
        if cap_stdout.startswith("Out[") or ("[?7hOut[1]:" in first_line):
            cap_stdout = "".join(cap_stdout.split(":")[1:]) 

        if result is None:
            assert self.timeout_seconds is not None
            message = f"Execution failed. Traceback:\nExecution timed out after {self.timeout_seconds} seconds."
        elif result.success:
            message = cap_stdout
            if not message.strip() and not self.allow_silent_success:
                message = "Execution successful."
            cap_stdout = message
        else:
            try:
                result.raise_error()
                message = ""  # to make mypy happy.
            except asyncio.CancelledError:
                # Cancellation typically means the outer episode task timed out
                # or was otherwise interrupted while this step was running.
                message_ = (
                    "asyncio.exceptions.CancelledError\n"
                    "Step execution was cancelled by an outer timeout/cancellation "
                    "(e.g., episode step timeout)."
                )
                message = "Execution failed. Traceback:\n" + message_
                cap_stderr = message_
                cap_stdout = cap_stdout.replace(", use %tb to see the full traceback.", ".")
            except Exception as exception:
                stack_trace = get_stack_trace_from_exception(
                    exception, only_ipython=True, add_http_exception_message=True
                )
                lines = stack_trace.splitlines()
                if "ipython-input" not in stack_trace:
                    # happens for syntax errors.
                    message_ = stack_trace
                else:
                    # happens for runtime errors.
                    index = next(
                        index for index, line in enumerate(lines) if "ipython-input" in line
                    )
                    message_ = "\n".join(lines[index:])
                    message_ = re.sub(
                        r'File "<ipython-input-\d+-\w+>"',
                        r'File "<python-input>"',
                        message_,
                    )
                message_ = message_.replace("appworld.requester.NumRequestsLimitError", "Exception")
                message_ = message_.replace(
                    "appworld.common.utils.TimeoutError: Function run_cell execution",
                    "Exception: Execution",
                )
                message_ = dedent(message_)
                message = "Execution failed. Traceback:\n" + message_
                cap_stderr = message_
                cap_stdout = cap_stdout.replace(", use %tb to see the full traceback.", ".")
        
        self.environment_io.append({
            "number": self.num_interactions + 1,  # AppWorld expects a 'number' field for logging
            "input": code,
            "output": message.rstrip()
        })
        if self.null_patch_unsafe_execution:
            self.safety_guard.disable()
        self.num_interactions += 1

        self._save_state(self.output_db_home_path_on_disk)
        self.save_logs()

        # This needs to happen both at the start and end of the execution because one can
        # call gym.apis from outside of execute as well for building the prompt.
        self.requester.reset_request_count()

        return CodeActStep(
            code=code,
            output=cap_stdout,
            error=cap_stderr,
        )

    async def fork(self, shell_id: str) -> AppWorldAsync:
        self.register_shell(shell_id)
        return self
    

class AppWorldCodeExecutor(CodeExecutor):
    
    def __init__(
        self,
        task: Task,
        world: AppWorldAsync | None = None,
        shell_id: str | None = None,
        experiment_name: str | None = None,
        owns_world: bool | None = None,
        timeout_seconds: int | None = DEFAULT_APPWORLD_TIMEOUT_SECONDS,
    ):
        self.task = task
        self.shell_id = shell_id

        if world is None:
            # Generate unique experiment name for parallel rollouts
            if experiment_name is None:
                experiment_name = f"platoon-appworld-{task.id}-{uuid.uuid4().hex}"

            self.world = AppWorldAsync(
                remote_environment_url=None,
                task_id=task.id,
                experiment_name=experiment_name,
                allow_silent_success=True,
                shell_id=self.shell_id,
                timeout_seconds=timeout_seconds,
                # This needs to be disabled to allow for writing logs to file when we launch subagents
                null_patch_unsafe_execution=False,
            )
            self.owns_world = True if owns_world is None else owns_world
        else:
            self.world = world
            self.owns_world = False if owns_world is None else owns_world
        
        self.task.goal = task.goal or self.world.task.instruction
        self.prompt_retriever = PromptRetriever(prompts_dir=Path(__file__).parent / "prompts")
        
    async def run(self, code: str) -> CodeActStep:
        return await self.world.execute(self.shell_id, code)
    
    async def describe_action_space(self) -> str:
        return self.prompt_retriever.get_prompt("user-action-space-description", supervisor=self.world.task.supervisor)
    
    async def reset(self) -> AppWorldCodeExecutor:
        return type(self)(
            self.task,
            world=self.world,
            shell_id=self.shell_id,
            owns_world=self.owns_world,
        )
    
    async def close(self) -> None:
        if self.owns_world:
            # AppWorld.close() is synchronous and can block indefinitely (e.g.
            # database teardown or freezegun cleanup after a cancelled rollout).
            # Run it in a thread so asyncio.wait_for can impose a deadline.
            # If it times out, the subprocess will be killed by SIGALRM anyway.
            loop = asyncio.get_running_loop()
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, self.world.close),
                    timeout=30.0,
                )
            except BaseException:
                pass  # Best-effort; SIGALRM will clean up the subprocess
        else:
            self.world.unregister_shell(self.shell_id)
        
    async def fork(self, task: Task) -> AppWorldCodeExecutor:
        shell_id = str(uuid.uuid4())
        return type(
            self
        )(
            task,
            world=await self.world.fork(shell_id),
            shell_id=shell_id,
            owns_world=False,
        )
    

class AppWorldRecursiveCodeExecutor(AppWorldCodeExecutor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.task, SubTask) and self.task.parent_tasks:
            self.current_task_is_subtask = True
        else:
            self.current_task_is_subtask = False
        self._launched_subagent_ids_this_step: set[str] = set()
        self._subagent_success_by_child_this_step: dict[str, float] = {}

    def reset_subagent_stats(self) -> None:
        """Reset subagent tracking for a new step."""
        self._launched_subagent_ids_this_step.clear()
        self._subagent_success_by_child_this_step.clear()

    def get_subagent_stats(self) -> tuple[int, float]:
        """Get (unique launched children, summed child success score) for current step."""
        return len(self._launched_subagent_ids_this_step), sum(self._subagent_success_by_child_this_step.values())

    async def run(self, code: str) -> CodeActStep:
        """Run code and track subagent launches."""
        from platoon.episode.context import current_trajectory, current_trajectory_collection

        # Get current trajectory to identify children
        traj_collection = current_trajectory_collection.get()
        current_traj = current_trajectory.get()

        # Track trajectories before execution to find new ones
        traj_ids_before = set(traj_collection.trajectories.keys())

        # Execute the code
        result = await super().run(code)

        # Check if any new child trajectories were created (subagents launched)
        for traj_id, traj in traj_collection.trajectories.items():
            if traj_id in traj_ids_before:
                continue
            if traj_id in self._launched_subagent_ids_this_step:
                continue
            if not traj.parent_info or traj.parent_info.id != current_traj.id:
                continue

            self._launched_subagent_ids_this_step.add(traj_id)
            success_reward = 0.0
            if traj.steps:
                final_step = traj.steps[-1]
                reward_misc = final_step.misc.get("reward_misc", {})
                success_reward = float(reward_misc.get("reward/success", 0.0))
            self._subagent_success_by_child_this_step[traj_id] = success_reward

        return result

    async def describe_action_space(self) -> str:
        return self.prompt_retriever.get_prompt(
            "user-recursive-action-space-description",
            supervisor=self.world.task.supervisor,
            current_task_is_subtask=self.current_task_is_subtask,
        )

class AppWorldEnv(CodeActEnv):

    def __init__(
        self,
        task: Task,
        code_executor: AppWorldCodeExecutor | None = None,
        timeout_seconds: int | None = DEFAULT_APPWORLD_TIMEOUT_SECONDS,
        skip_subagent_reward_computation: bool = False,
        **kwargs,
    ):
        if code_executor is None:
            code_executor = AppWorldCodeExecutor(task, timeout_seconds=timeout_seconds)
        self._skip_subagent_reward_computation = skip_subagent_reward_computation

        super().__init__(task, code_executor, **kwargs)

    @property
    def code_executor(self) -> AppWorldCodeExecutor:
        return self._code_executor

    async def reset(self) -> CodeActObservation:
        await super().reset()
        self._state.action_space = await self.code_executor.describe_action_space()
        return await self.observe()

    async def evaluate(self) -> tuple[float, dict]:
        score, reward_misc = 0., {}
        is_subagent_task = isinstance(self._task, SubTask) and bool(self._task.parent_tasks)
        if self._skip_subagent_reward_computation and is_subagent_task:
            reward_misc["reason"] = "Skipped subagent reward computation"
            reward_misc["reward/success"] = 0.0
            return 0.0, reward_misc

        if self._state.finished:
            if isinstance(self._task, SubTask) and self._task.parent_tasks:
                try:
                    rubric_checklist = RubricChecklistFast(self._task.goal)
                    prompt_builder = AppWorldCodeActPromptBuilder()
                    action_history = prompt_builder.build_action_history_description(await self.observe())
                    # Pull messages from episode-level context vars first; fall back to last step if available
                    final_message = finish_message.get() or (self._state.history[-1].misc.get("finish_message") if self._state.history else None)
                    err_message = error_message.get() or (self._state.history[-1].misc.get("error_message") if self._state.history else None)

                    rubric_context = f"We need to judge the performance of an agent on the task.\n\n# Agent Trajectory Info\n## Action History\n{action_history}\n\n## Final Message\n{final_message}\n\n## Error Message\n{err_message}"
                    score, reason = await rubric_checklist.aevaluate(include_reason=True, context=rubric_context)

                    reward_misc["reason"] = reason
                    reward_misc["rubric_dict"] = rubric_checklist.to_dict()

                except Exception as e:
                    reward_misc["reason"] = f"Failed rubric-based evaluation: {e}"
                    score = 0.
            else:
                try:
                    score = float(self.code_executor.world.evaluate(suppress_errors=False).to_dict()["success"])
                    reward_misc["reason"] = "Trajectory reward provided by AppWorld environment."
                except Exception as e:
                    reward_misc["reason"] = f"Failed to evaluate task: {e}"

        reward_misc["reward/success"] = score
        return score, reward_misc

    async def fork(self, task: Task) -> AppWorldEnv:
        code_executor = await self.code_executor.fork(task)
        return type(self)(
            task,
            code_executor=code_executor,
            skip_subagent_reward_computation=self._skip_subagent_reward_computation,
        )
    

class AppWorldRecursiveEnv(AppWorldEnv):
    """Environment for AppWorld tasks with recursive agent spawning and delegation rewards."""

    def __init__(
        self,
        task: Task,
        code_executor: AppWorldRecursiveCodeExecutor | None = None,
        timeout_seconds: int | None = DEFAULT_APPWORLD_TIMEOUT_SECONDS,
        skip_subagent_reward_computation: bool = False,
        **kwargs,
    ):
        if code_executor is None:
            code_executor = AppWorldRecursiveCodeExecutor(task, timeout_seconds=timeout_seconds)
        super().__init__(
            task,
            code_executor,
            timeout_seconds=timeout_seconds,
            skip_subagent_reward_computation=skip_subagent_reward_computation,
            **kwargs,
        )

    @property
    def code_executor(self) -> AppWorldRecursiveCodeExecutor:
        return self._code_executor

    def _get_subagent_stats_and_reset(self) -> tuple[int, float]:
        """Get per-step unique launched children and summed child success score."""
        stats = self.code_executor.get_subagent_stats()
        self.code_executor.reset_subagent_stats()
        return stats

    async def evaluate(self) -> tuple[float, dict]:
        score, reward_misc = await super().evaluate()

        # Get subagent stats for this step (also resets for next step)
        launched, success_total = self._get_subagent_stats_and_reset()

        reward_misc["reward/subagent_launched"] = launched
        reward_misc["reward/subagent_succeeded"] = success_total

        return score, reward_misc


class AppWorldDepthAwareCodeExecutor(AppWorldRecursiveCodeExecutor):
    """Code executor for depth-aware budget tracking.

    Replaces the shell's ``launch_subagent`` with a wrapper that uses a
    fixed ``max_steps`` default — the agent does not need to specify
    ``max_steps`` when calling ``launch_subagent``.
    """

    def __init__(self, *args, subagent_max_steps: int = 25, **kwargs):
        super().__init__(*args, **kwargs)
        self._subagent_max_steps = subagent_max_steps
        self._inject_depth_aware_launch_subagent()

    def _inject_depth_aware_launch_subagent(self) -> None:
        """Replace launch_subagent in the shell with a version using fixed max_steps."""
        max_steps = self._subagent_max_steps

        async def depth_aware_launch_subagent(goal: str) -> str:
            """Launch a subagent to solve a task.

            Args:
                goal: The goal of the subagent.

            Returns:
                Returns the answer or finish message for the goal.
            """
            return await launch_subagent(goal=goal, max_steps=max_steps)

        shell = self.world.get_shell(self.shell_id)
        shell.user_ns["launch_subagent"] = depth_aware_launch_subagent

    async def describe_action_space(self) -> str:
        return self.prompt_retriever.get_prompt(
            "user-depth-aware-action-space-description",
            supervisor=self.world.task.supervisor,
            current_task_is_subtask=self.current_task_is_subtask,
        )

    async def fork(self, task: Task) -> "AppWorldDepthAwareCodeExecutor":
        shell_id = str(uuid.uuid4())
        return AppWorldDepthAwareCodeExecutor(
            task,
            world=await self.world.fork(shell_id),
            shell_id=shell_id,
            owns_world=False,
            subagent_max_steps=self._subagent_max_steps,
        )


class AppWorldDepthAwareEnv(AppWorldRecursiveEnv):
    """Environment for depth-aware recursive AppWorld training.

    Uses ``AppWorldDepthAwareCodeExecutor`` so agents do not specify
    ``max_steps`` when delegating.  Paired with
    ``DepthAwareStepBudgetTracker`` in the rollout function.
    """

    def __init__(
        self,
        task: Task,
        code_executor: AppWorldDepthAwareCodeExecutor | None = None,
        subagent_max_steps: int = 25,
        timeout_seconds: int | None = DEFAULT_APPWORLD_TIMEOUT_SECONDS,
        skip_subagent_reward_computation: bool = False,
        **kwargs,
    ):
        self._subagent_max_steps = subagent_max_steps
        if code_executor is None:
            code_executor = AppWorldDepthAwareCodeExecutor(
                task,
                subagent_max_steps=subagent_max_steps,
                timeout_seconds=timeout_seconds,
            )
        super().__init__(
            task,
            code_executor,
            timeout_seconds=timeout_seconds,
            skip_subagent_reward_computation=skip_subagent_reward_computation,
            **kwargs,
        )

    async def fork(self, task: Task) -> "AppWorldDepthAwareEnv":
        code_executor = await self.code_executor.fork(task)
        return AppWorldDepthAwareEnv(
            task,
            code_executor=code_executor,
            subagent_max_steps=self._subagent_max_steps,
            skip_subagent_reward_computation=self._skip_subagent_reward_computation,
        )
    