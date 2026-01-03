from pathlib import Path
from typing import Literal

from platoon.envs.codeact import CodeActObservation, CodeActAction
from platoon.utils.llm_client import ChatMessage, Conversation, ConversationWithMetadata
from platoon.utils.prompt_retriever import PromptRetriever


PromptMode = Literal["sequence_extension", "no_sequence_extension"]


class CodeActPromptBuilder:
    """Builds prompts for CodeAct agents.
    
    Supports two prompt modes:
    - "sequence_extension" (default): Uses a multi-turn conversation format where 
      each step appends to the conversation. This enables sequence extension for 
      efficient training - consecutive observations are prefixes of each other,
      allowing tinker to merge consecutive steps into fewer Datums.
    - "no_sequence_extension": Uses a single user message with the full action 
      history embedded. This is the legacy format that rebuilds the entire prompt 
      each step.
    
    Args:
        prompts_dir: Directory containing prompt templates.
        prompt_mode: The prompt format to use.
        include_reasoning: If True, prompts instruct the agent to include <thought> tags.
            If False, only <python> tags are expected. Default is True.
    """

    def __init__(
        self, 
        prompts_dir: str | Path | None = None,
        prompt_mode: PromptMode = "sequence_extension",
        include_reasoning: bool = True,
    ):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent / "prompts"
        self.retriever = PromptRetriever(prompts_dir=prompts_dir)
        self.prompt_mode = prompt_mode
        self.include_reasoning = include_reasoning
        
    # TODO: We need to refactor this to be more general.
    def build_messages_from_traj_dump(self, traj_collection_dump: dict, reward_threshold: float) -> list[ConversationWithMetadata]:
        raise NotImplementedError("Subclasses must implement this method")

            
    def build_messages(self, obs: CodeActObservation, agent_action: CodeActAction | None = None) -> Conversation:
        """Build the conversation messages for the agent.
        
        Args:
            obs: The current observation
            agent_action: Optional action to append (for training data generation)
            
        Returns:
            List of ChatMessage objects forming the conversation
        """
        if self.prompt_mode == "sequence_extension":
            return self._build_messages_sequence_extension(obs, agent_action)
        else:
            return self._build_messages_no_sequence_extension(obs, agent_action)
    
    def _build_messages_sequence_extension(self, obs: CodeActObservation, agent_action: CodeActAction | None = None) -> Conversation:
        """Build messages in sequence extension format.
        
        The conversation grows by appending turns:
        - [System] Initial instructions
        - [User] Task description + action space + instruction to start
        - [Assistant] Action 0
        - [User] Output 0
        - [Assistant] Action 1
        - [User] Output 1
        - ...
        
        This format ensures each step's observation is a prefix of the next step's
        observation, enabling efficient sequence merging during training.
        """
        messages: Conversation = []
        
        # System message
        messages.append(ChatMessage(
            role="system", 
            content=self.build_system_prompt(obs)
        ))
        
        # Initial user message with task and action space
        initial_user_content = self._build_initial_user_message(obs)
        messages.append(ChatMessage(role="user", content=initial_user_content))
        
        # Add history as alternating assistant/user turns
        for i, step in enumerate(obs.history):
            # Assistant turn: the action taken
            action_str = self._format_action_for_history(step)
            messages.append(ChatMessage(role="assistant", content=action_str))
            
            # User turn: the observation/output from that action
            output_str = self._format_observation_for_history(step, i)
            messages.append(ChatMessage(role="user", content=output_str))
        
        # If we have an agent_action (for training), append it
        if agent_action:
            messages.append(ChatMessage(role="assistant", content=str(agent_action)))
        
        return messages
    
    def _build_messages_no_sequence_extension(self, obs: CodeActObservation, agent_action: CodeActAction | None = None) -> Conversation:
        """Build messages without sequence extension (legacy format).
        
        Uses a single user message with the full action history embedded.
        Each step rebuilds the entire user message, so observations are not prefixes.
        """
        system_prompt = self.build_system_prompt(obs)
        user_prompt = self.build_user_prompt(
            obs,
            task=str(obs.task),
            action_space_description=str(obs.action_space),
            action_history_description=self.build_action_history_description(obs),
            next_action_str=self.build_next_action_str(obs),
        )

        messages: Conversation = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]

        if agent_action:
            messages.append(ChatMessage(role="assistant", content=str(agent_action)))
        
        return messages
    
    def _build_initial_user_message(self, obs: CodeActObservation) -> str:
        """Build the initial user message for sequence_extension mode.
        
        Uses the user-initial.jinja template if available, otherwise builds programmatically.
        """
        try:
            return self.retriever.get_prompt(
                "user-initial",
                task=str(obs.task),
                action_space_description=str(obs.action_space) if obs.action_space else "",
                next_action_str=self.build_next_action_str(obs),
            )
        except Exception:
            # Fallback to programmatic construction if template not available
            parts = []
            parts.append(f"# Task\n\n{obs.task}")
            if obs.action_space:
                parts.append(f"# Action Space\n\n{obs.action_space}")
            parts.append(self.build_next_action_str(obs))
            return "\n\n".join(parts)
    
    def _format_action_for_history(self, step) -> str:
        """Format a historical action for the assistant turn.
        
        Reconstructs the action in the standard format. If include_reasoning is True,
        uses <thought>...</thought><python>...</python> format. Otherwise, just <python>...</python>.
        """
        parts = []
        if self.include_reasoning and hasattr(step, 'thought') and step.thought:
            parts.append(f"<thought>{step.thought}</thought>")
        if hasattr(step, 'code') and step.code:
            parts.append(f"<python>{step.code}</python>")
        return "\n".join(parts) if parts else ""
    
    def _format_observation_for_history(self, step, step_index: int) -> str:
        """Format a historical observation/output for the user turn.
        
        Uses the user-observation.jinja template if available, otherwise builds programmatically.
        """
        output = getattr(step, 'output', None) or ""
        error = getattr(step, 'error', None) or ""
        
        try:
            return self.retriever.get_prompt(
                "user-observation",
                step_index=step_index,
                output=output,
                error=error,
            )
        except Exception:
            # Fallback to programmatic construction if template not available
            parts = [f"[Cell {step_index} Output]"]
            has_content = False
            if output:
                parts.append(output)
                has_content = True
            if error:
                parts.append(f"Error: {error}")
                has_content = True
            if not has_content:
                parts.append("(No output)")
            parts.append("\nProvide your next action.")
            return "\n".join(parts)
    
    # TODO: Support history truncation in the future.
    def build_action_history_description(self, obs: CodeActObservation) -> str:
        """Build action history for single-turn mode."""
        if not obs.history:
            return "No actions taken yet."

        processed_cells = [f"Cell {i}:\n" + str(cell) for i, cell in enumerate(obs.history)]
        return "\n".join(processed_cells)
    
    # TODO: Revisit whether we want to use obs inside or pass from outside.
    def build_next_action_str(self, obs: CodeActObservation, **context) -> str:
        return self.retriever.get_prompt(
            "user-next-action-str",
            include_reasoning=self.include_reasoning,
            **context
        )

    def build_system_prompt(self, obs: CodeActObservation, **context) -> str:
        return self.retriever.get_prompt(
            "system",
            include_reasoning=self.include_reasoning,
            **context
        )
    
    def build_user_prompt(self, obs: CodeActObservation, **context) -> str:
        return self.retriever.get_prompt("user", **context)
