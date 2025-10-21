from __future__ import annotations

from copy import deepcopy
from typing import Protocol, runtime_checkable

from platoon.agents.actions.common import finish
from platoon.envs.base import Task
from IPython.terminal.embed import InteractiveShellEmbed
from traitlets.config.loader import Config
import sys
import ast
import asyncio
from typing import Callable, Sequence

from .types import CodeExecutor, ForkableCodeExecutor, CodeActObservation, CodeActAction, CodeActStep
from platoon.episode.context import finish_message
from platoon.episode.context import current_trajectory_collection, current_trajectory, error_message
from platoon.utils.ipython_shell import ShellCapture, strip_ansi_escape_sequences


@runtime_checkable
class CodeActEnv(Protocol):

    # TODO: Come back to think about how we might want to add a factory + config to be string-based configurable.
    def __init__(self, task: Task, code_executor: CodeExecutor, return_obs_copy: bool = True, parent_state: CodeActObservation | None = None, **kwargs):
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
        # This is not needed here, but we keep it here for demonstration in case an env wants to add steps/obs during reset.
        for step in self._state.history:
            traj_collection.add_trajectory_step(traj.id, step)
        return await self.observe()
    
    async def evaluate(self) -> tuple[float, dict]:
        return 0., {}
    
    async def step(self, action: CodeActAction) -> CodeActObservation:
        step = await self._code_executor.run(action.parsed_code)
        
        if finish_message.get(None) is not None or error_message.get(None) is not None:
            self._state.finished = True
            self._state.misc["finish_message"] = finish_message.get()

        step.thought = action.parsed_thought
        step.reward, reward_info = await self.evaluate()
        step.misc['action_misc'] = action.misc
        step.misc['reward_misc'] = reward_info
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
                **deepcopy(self._init_kwargs)
            )
        else:
            raise ValueError("CodeExecutor is not forkable. "
            "Either implement fork() for your CodeExecutor or implement a new ForkableEnv for this task."
            )


class IPythonCodeExecutor(CodeExecutor):
    # TODO: Separate actions and modules? Use this info to build action space description?
    def __init__(self, task: Task, actions: Sequence[Callable] = (finish, asyncio)):
        self.task = task
        self.actions = actions
        self.shell = self._create_shell()
        #self.timeout_seconds = timeout_seconds
        
    def _create_shell(self) -> InteractiveShellEmbed:
        original_excepthook = sys.excepthook
        config = Config()
        # history keeps files open preventing making > ~50 envs
        config.HistoryManager.enabled = False 
        shell = InteractiveShellEmbed(config=config)
        sys.excepthook = original_excepthook # prevents it from changing traceback format globally
        for action in self.actions:
            shell.user_ns[action.__name__] = action
        return shell

    # TODO: Can make this more robust and have better error handling + sandboxing + timeouts.
    async def run(self, code: str) -> CodeActStep:
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
            return CodeActStep(code=code, error=message)

        if not code:
            return CodeActStep(code=code, error="No code available to execute.")

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
        raise NotImplementedError("IPythonCodeExecutor does not yet implement describe_action_space. Please implement it in your subclass.")

    async def reset(self) -> CodeExecutor:
        return self
