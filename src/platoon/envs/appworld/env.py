from __future__ import annotations

import asyncio
import ast
from pathlib import Path
import sys
from typing import Literal
import re
import uuid
from copy import deepcopy
from textwrap import dedent

from appworld import AppWorld
from appworld.common.utils import get_stack_trace_from_exception
from IPython.core.interactiveshell import ExecutionResult
#from IPython.utils.io import capture_output
from IPython.terminal.embed import InteractiveShellEmbed
from rubric.core.checklist import RubricChecklistFast
from traitlets.config.loader import Config

from platoon.agents.actions.common import finish, finish_message
from platoon.episode.context import error_message
from platoon.agents.actions.subagent import launch_subagent
from platoon.agents.appworld.codeact import AppWorldCodeActPromptBuilder
from platoon.envs.base import SubTask, Task
from platoon.envs.codeact import CodeExecutor, CodeActStep, CodeActEnv, CodeActAction, CodeActObservation
from platoon.envs.appworld.server_pool import SERVER_POOL
from platoon.utils.ipython_shell import ShellCapture, strip_ansi_escape_sequences
from platoon.utils.timeout import async_timeout_call
from platoon.utils.prompt_retriever import PromptRetriever
from platoon.envs.codeact.rubrics import generate_rubric_tree

class AppWorldAsync(AppWorld):

    def __init__(self, *args, allow_silent_success: bool = True, is_server: bool = False, init_id: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.shell_id = init_id or str(uuid.uuid4())
        self.shells = {}
        self.is_server = is_server

        if self.shell and not self.is_server:
            self.shell = self._create_shell()
            self.shells = {self.shell_id: self.shell}
        
        self.allow_silent_success = allow_silent_success
        
        if self.remote_environment_url:
            self._remote_environment_call("register_shell_id", shell_id=self.shell_id)
            self._remote_environment_call("set_attributes", attributes={"allow_silent_success": allow_silent_success})
    
    async def register_shell_id(self, shell_id: str):
        self.shells[shell_id] = self._create_shell()
            
    async def _shell_run_cell(self, id, code: str) -> ExecutionResult:
        self._maybe_raise_remote_environment_error("_shell_run_cell")
        shell = self.shells[id]
        if self.timeout_seconds is None:
            return await shell.run_cell_async(code)
        try:
            return await async_timeout_call(
                shell.run_cell_async, timeout_seconds=self.timeout_seconds, raw_cell=code
            )
        except asyncio.TimeoutError:
            return None

    async def execute(self, code: str, shell_id: str = "") -> CodeActStep:
        if not shell_id:
            shell_id = self.shell_id

        if self.remote_environment_url:
            message = self._remote_environment_call("execute", code=code, shell_id=shell_id)
            if "(sqlite3.ProgrammingError) Cannot operate on a closed database." in message:
                raise Exception("Looks like you are operating on a closed task world object.")
            self.environment_io.append({"input": code, "output": message})
            step = CodeActStep(**message)
            if step.misc.get("finish_message"):
                finish_message.set(step.misc["finish_message"])
            return step

        if self.raise_on_unsafe_syntax:
            is_syntax_safe, safety_message = self.safety_guard.is_syntax_safe(code)
            if not is_syntax_safe:
                message = "Execution failed. Traceback:\n" + safety_message
                self.environment_io.append({"input": code, "output": message})
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
            self.environment_io.append({"input": code, "output": message})
            if self.null_patch_unsafe_execution:
                self.safety_guard.disable()
            return CodeActStep(
                code=code,
                error=message,
            )

        if not code:
            message = "No code available to execute."
            self.environment_io.append({"input": code, "output": message})
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
        # with capture_output() as capture:
        #     result = await self._shell_run_cell(shell_id, code)
        # cap_stdout = capture.stdout
        # cap_stderr = capture.stderr

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

        self.environment_io.append({"input": code, "output": message.rstrip()})
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
    
    @property
    def execution_mode(self) -> Literal["client", "local", "server"]:
        if self.is_server:
            return "server"

        if not self.shells:
            return "client"
        
        return "local"

    def _create_shell(self) -> InteractiveShellEmbed:
        original_excepthook = sys.excepthook
        config = Config()
        config.HistoryManager.enabled = (
            False  # history keeps files open preventing making > ~50 envs.
        )
        shell = InteractiveShellEmbed(config=config)
        old_shell = self.shell
        self.shell = shell
        self._execute_preamble()
        self.shell = old_shell
        sys.excepthook = (
            original_excepthook  # prevents it from changing traceback format globally
        )
        shell.user_ns["finish"] = finish
        shell.user_ns["launch_subagent"] = launch_subagent
        shell.user_ns["asyncio"] = asyncio
        return shell


    async def fork(self, shell_id: str) -> AppWorldAsync:
        if self.execution_mode == "local":
            # forked_world = deepcopy(self)
            # forked_world.shell_id = str(uuid.uuid4())
            # shell = self._create_shell()
            # forked_world.shells = {forked_world.shell_id: shell}
            # return forked_world
            await self.register_shell_id(shell_id)
            
        else:
            # forked_world = deepcopy(self)
            # forked_world.shell_id = str(uuid.uuid4())
            # self._remote_environment_call("register_shell_id", shell_id=shell_id)
            raise ValueError("Forking is not supported when using AppWorld with remote execution.")
        
        return self

class AppWorldCodeExecutor(CodeExecutor):
    
    def __init__(self, task: Task, local_mode: bool = True, world: AppWorldAsync | None = None, shell_id: str | None = None):
        self.task = task
        self.local_mode = local_mode
        self.shell_id = shell_id
        if self.local_mode:
            self.server_url = None
        else:
            self.server_url = SERVER_POOL.request_server(add_new_if_unavailable=True)
        if world is None:
            self.world = AppWorldAsync(
                remote_environment_url=self.server_url,
                task_id=task.id,
                allow_silent_success=True,
                init_id=self.shell_id,
                # This needs to be disabled to allow for writing logs to file when we launch subagents
                null_patch_unsafe_execution=False, 
            )
        else:
            self.world = world
        
        self.task.goal = task.goal or self.world.task.instruction
        self.prompt_retriever = PromptRetriever(prompts_dir=Path(__file__).parent / "prompts")

    async def run(self, code: str) -> CodeActStep:
        return await self.world.execute(code, shell_id=self.shell_id)
    
    async def describe_action_space(self) -> str:
        return self.prompt_retriever.get_prompt("user-action-space-description", supervisor=self.world.task.supervisor)
    
    async def reset(self) -> AppWorldCodeExecutor:
        return type(self)(self.task, local_mode=self.local_mode, world=self.world, shell_id=self.shell_id)
    
    async def close(self) -> None:
        self.world.close()
        if not self.local_mode:
            SERVER_POOL.release_server(self.server_url)

    async def fork(self, task: Task) -> AppWorldCodeExecutor: # TODO: Maybe we should move this to the recursive executor
        shell_id = str(uuid.uuid4())
        return type(self)(task, local_mode=self.local_mode, world=await self.world.fork(shell_id), shell_id=shell_id)

class AppWorldRecursiveCodeExecutor(AppWorldCodeExecutor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.task, SubTask) and self.task.parent_tasks:
            self.current_task_is_subtask = True
        else:
            self.current_task_is_subtask = False
            

    async def describe_action_space(self) -> str:
        return self.prompt_retriever.get_prompt(
            "user-recursive-action-space-description",
            supervisor=self.world.task.supervisor,
            current_task_is_subtask=self.current_task_is_subtask,
        )

class AppWorldEnv(CodeActEnv):

    @property
    def code_executor(self) -> AppWorldCodeExecutor:
        return self._code_executor

    async def reset(self) -> CodeActObservation:
        await super().reset()
        self._state.action_space = await self.code_executor.describe_action_space()
        return await self.observe()

    # TODO: Need to hook finish message with app world's finish action
    async def evaluate(self) -> tuple[float, dict]:
        score, reward_misc = 0., {}
        if self._state.finished:
            if isinstance(self._task, SubTask) and self._task.parent_tasks:
                try:
                    # rubric_tree = generate_rubric_tree(self._task.goal)
                    # prompt_builder = AppWorldCodeActPromptBuilder()
                    # score, reason = rubric_tree.evaluate(
                    #     include_reason=True,
                    #     action_history=prompt_builder.build_action_history_description(await self.observe()),
                    #     final_message=self._state.history[-1].misc.get("finish_message"),
                    #     error_message=self._state.history[-1].misc.get("error_message"),
                    # )

                    rubric_checklist = RubricChecklistFast(self._task.goal)
                    prompt_builder = AppWorldCodeActPromptBuilder()
                    action_history = prompt_builder.build_action_history_description(await self.observe())
                    # Pull messages from episode-level context vars first; fall back to last step if available
                    final_message = finish_message.get() or (self._state.history[-1].misc.get("finish_message") if self._state.history else None)
                    err_message = error_message.get() or (self._state.history[-1].misc.get("error_message") if self._state.history else None)
                    
                    rubric_context = f"We need to judge the performance of an agent on the task.\n\n# Agent Trajectory Info\n## Action History\n{action_history}\n\n## Final Message\n{final_message}\n\n## Error Message\n{err_message}"
                    score, reason = rubric_checklist.evaluate(include_reason=True, context=rubric_context)

                    reward_misc["reason"] = reason
                    reward_misc["rubric_dict"] = rubric_checklist.to_dict()
                    return score, reward_misc
                
                except Exception as e:
                    reward_misc["reason"] = f"Failed rubric-based evaluation: {e}"
                    return 0., reward_misc
            else:
                try:
                    score = float(self.code_executor.world.evaluate(suppress_errors=False).to_dict()["success"])
                    reward_misc["reason"] = "Trajectory reward provided by AppWorld environment."
                except Exception as e:
                    reward_misc["reason"] = f"Failed to evaluate task: {e}"
        return score, reward_misc
