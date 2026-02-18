from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import json

from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder, PromptMode
from platoon.envs.base import SubTask, Task
from platoon.envs.codeact import CodeActObservation, CodeActStep, CodeActAction
from platoon.utils.llm_client import ConversationWithMetadata
from platoon.utils.prompt_retriever import PromptRetriever, get_prompt


@dataclass
class Supervisor:
    first_name: str
    last_name: str
    email: str
    phone_number: str
    
class AppWorldCodeActPromptBuilder(CodeActPromptBuilder):
    def __init__(
        self, 
        prompt_mode: PromptMode = "sequence_extension", 
        include_reasoning: bool = True, 
        prompts_dir: str | Path | None = None,
    ):
        super().__init__(prompt_mode=prompt_mode, include_reasoning=include_reasoning, prompts_dir=prompts_dir)
        self.appworld_prompt_retriever = PromptRetriever(prompts_dir=Path(__file__).parent / "prompts")
    
    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        if "env_specific_system_context" not in context:
            context["env_specific_system_context"] = self.appworld_prompt_retriever.get_prompt("system-env-specific-system-context")
        return super().build_system_prompt(obs, **context)
    
    def build_messages_from_traj_dump(self, traj_collection_dump: dict, reward_threshold: float) -> list[ConversationWithMetadata]:
        messages: list[ConversationWithMetadata] = []
        
        for traj in traj_collection_dump["trajectories"].values():
            if traj["reward"] < reward_threshold:
                    continue
            task_id = traj["task"]["id"]
            task_specs_path = Path(os.environ["APPWORLD_ROOT"]) / "data" / "tasks" / task_id / "specs.json"
            with open(task_specs_path, "r") as f:
                task_specs = json.load(f)
            supervisor = Supervisor(**task_specs["supervisor"])
            appworld_env_prompts_path = Path(__file__).parent / "prompts"
            action_space_description = get_prompt("user-action-space-description", prompts_dir=appworld_env_prompts_path, supervisor=supervisor)
            task = None
            if traj["task"] is not None:
                if "parent_tasks" in traj["task"]:
                    task = SubTask.from_dict(traj["task"])
                else:
                    task = Task.from_dict(traj["task"])
            reward = traj["reward"]
            misc = traj["misc"]
            history = []
            
            obs = CodeActObservation(
                task=task,
                reward=reward,
                misc=misc,
                history=history,
                action_space=action_space_description
            )
            
            traj_misc = {
                "id": traj["id"],
                "task": traj["task"],
                "parent_info": traj["parent_info"],
                "reward": traj["reward"],
                **traj["misc"],
            }
            
            for i, step_dict in enumerate(traj["steps"]):
                step = CodeActStep(**step_dict)
                step.misc["traj_misc"] = traj_misc
                step.misc["step_num"] = len(obs.history)
                agent_action = CodeActAction(
                    parsed_code=step.code,
                    parsed_thought=step.thought,
                )
                if self.prompt_mode == "sequence_extension" and i < len(traj["steps"]) - 1:
                    obs.history.append(step)
                    continue
                messages.append(ConversationWithMetadata(
                    messages=self.build_messages(obs, agent_action),
                    misc=step.misc
                ))
            
        return messages

class AppWorldRecursiveCodeActPromptBuilder(AppWorldCodeActPromptBuilder):
    def __init__(self, 
        prompt_mode: PromptMode = "sequence_extension", 
        include_reasoning: bool = True, 
        prompts_dir: str | Path | None = None,
    ):
        super().__init__(prompt_mode=prompt_mode, include_reasoning=include_reasoning, prompts_dir=prompts_dir)
        self.appworld_prompt_retriever = PromptRetriever(prompts_dir=Path(__file__).parent / "prompts")
        
    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        if "env_specific_system_context" not in context:
            context["env_specific_system_context"] = self.appworld_prompt_retriever.get_prompt("system-recursive-env-specific-system-context")
        return super().build_system_prompt(obs, **context)
    
    def build_messages_from_traj_dump(self, traj_collection_dump: dict, reward_threshold: float) -> list[ConversationWithMetadata]:
        messages: list[ConversationWithMetadata] = []
        
        supervisor = None

        for i, traj in enumerate(traj_collection_dump["trajectories"].values()):
            if i == 0:
                task_id = traj["task"]["id"]
                task_specs_path = Path(os.environ["APPWORLD_ROOT"]) / "data" / "tasks" / task_id / "specs.json"
                with open(task_specs_path, "r") as f:
                    task_specs = json.load(f)
                supervisor = Supervisor(**task_specs["supervisor"])
            
            if traj["reward"] < reward_threshold:
                    continue
            
            task = None
            current_task_is_subtask = False
            if traj["task"] is not None:
                if "parent_tasks" in traj["task"] and len(traj["task"]["parent_tasks"]) > 0:
                    task = SubTask.from_dict(traj["task"])
                    current_task_is_subtask = True
                else:
                    task = Task.from_dict(traj["task"])
            
            appworld_env_prompts_path = Path(__file__).parent / "prompts"
            action_space_description = get_prompt(
                "user-recursive-action-space-description",
                prompts_dir=appworld_env_prompts_path,
                supervisor=supervisor,
                current_task_is_subtask=current_task_is_subtask
            )
            reward = traj["reward"]
            misc = traj["misc"]
            history = []
            
            obs = CodeActObservation(
                task=task,
                reward=reward,
                misc=misc,
                history=history,
                action_space=action_space_description
            )
            
            traj_misc = {
                "id": traj["id"],
                "task": traj["task"],
                "parent_info": traj["parent_info"],
                "reward": traj["reward"],
                **traj["misc"],
            }

            for i, step_dict in enumerate(traj["steps"]):
                step = CodeActStep(**step_dict)
                step.misc["traj_misc"] = traj_misc
                step.misc["step_num"] = len(obs.history)
                agent_action = CodeActAction(
                    parsed_code=step.code,
                    parsed_thought=step.thought,
                )
                if traj["reward"] >= reward_threshold:
                    if self.prompt_mode == "sequence_extension" and i < len(traj["steps"]) - 1:
                        obs.history.append(step)
                        continue
                    messages.append(ConversationWithMetadata(
                        messages=self.build_messages(obs, agent_action),
                        misc=step.misc
                    ))
            
        return messages

class AppWorldAgent(CodeActAgent):
    def __init__(self,
        prompt_mode: PromptMode = "sequence_extension", 
        include_reasoning: bool = True, 
        prompts_dir: str | Path | None = None,
        prompt_builder: AppWorldCodeActPromptBuilder | None = None,
        **kwargs
    ):
        if prompt_builder is None:
            prompt_builder = AppWorldCodeActPromptBuilder(prompt_mode=prompt_mode, include_reasoning=include_reasoning, prompts_dir=prompts_dir)
        super().__init__(
            prompt_builder=prompt_builder,
            prompt_mode=prompt_mode,
            include_reasoning=include_reasoning,
            **kwargs,
        )
        
class AppWorldRecursiveAgent(AppWorldAgent):

    def __init__(self,
        prompt_mode: PromptMode = "sequence_extension", 
        include_reasoning: bool = True, 
        prompts_dir: str | Path | None = None,
        prompt_builder: AppWorldRecursiveCodeActPromptBuilder | None = None,
        **kwargs,
    ):
        if prompt_builder is None:
            prompt_builder = AppWorldRecursiveCodeActPromptBuilder(prompt_mode=prompt_mode, include_reasoning=include_reasoning, prompts_dir=prompts_dir)
        super().__init__(
            prompt_builder=prompt_builder,
            prompt_mode=prompt_mode,
            include_reasoning=include_reasoning,
            **kwargs,
        )

    async def fork(self, task: Task) -> AppWorldRecursiveAgent:
        return AppWorldRecursiveAgent(
            prompt_builder=self.prompt_builder,
            prompt_mode=self.prompt_builder.prompt_mode,
            include_reasoning=self.include_reasoning,
            llm_client=self.llm_client,
            inference_params=self.inference_params,
            stuck_in_loop_threshold=self.stuck_in_loop_threshold,
            stuck_in_loop_window=self.stuck_in_loop_window,
        )
