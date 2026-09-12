from dataclasses import dataclass, field
from typing import Any

from platoon.envs.base import Observation
from platoon.episode.trajectory import TrajectoryStep

from openhands.sdk.conversation.base import ConversationStateProtocol
from openhands.sdk.event import EventID
from openhands.sdk.event.base import Event
from openhands.sdk.event.llm_convertible.action import ActionEvent


@dataclass
class OpenHandsObservation(Observation):
    conversation_state: ConversationStateProtocol | None = None
    last_step_action_id: EventID | None = None
    last_step_observation_id: EventID | None = None
    # The first terminal event bounds the exported trajectory even if the SDK
    # has already buffered more events before Platoon next polls it.
    terminal_event_id: EventID | None = None


@dataclass
class OpenHandsAction:
    action_events: list[Event] | None = None
    misc: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenHandsTrajectoryStep(TrajectoryStep):
    action_events: list[ActionEvent] | None = None
    observation_events: list[Event] | None = None
    reward: float = 0.0
