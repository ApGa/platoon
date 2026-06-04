from __future__ import annotations

import json
from typing import Any

from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event.base import Event
from platoon.openhands.env import OpenHandsEnv


def _claim_done_payload(event: Event) -> dict[str, Any] | None:
    observation = getattr(event, "observation", None)
    if getattr(observation, "tool_name", None) != "claim_done":
        return None

    for block in getattr(observation, "content", []) or []:
        text = getattr(block, "text", None)
        if not isinstance(text, str):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("finished") is True:
            return payload
    return None


class OpenRewardOpenHandsEnv(OpenHandsEnv):
    def __init__(self, *args, **kwargs):
        self._final_payload: dict[str, Any] | None = None
        self._final_reward_consumed = False
        callbacks = list(kwargs.pop("callbacks", []) or [])
        callbacks.append(self._stop_on_claim_done)
        super().__init__(*args, callbacks=callbacks, **kwargs)

    def _stop_on_claim_done(self, event: Event) -> None:
        payload = _claim_done_payload(event)
        if payload is None:
            return
        self._final_payload = payload
        if self._conversation is not None:
            self._conversation.state.execution_status = ConversationExecutionStatus.FINISHED

    async def evaluate(self) -> tuple[float, dict]:
        if self._final_payload is None or self._final_reward_consumed:
            return 0.0, {}

        self._final_reward_consumed = True
        reward = self._final_payload.get("reward", 0.0)
        if not isinstance(reward, (int, float)):
            reward = 0.0
        return float(reward), {
            "reward/success": float(reward),
            "reward/openreward": float(reward),
            "openreward/final_payload": self._final_payload,
        }
