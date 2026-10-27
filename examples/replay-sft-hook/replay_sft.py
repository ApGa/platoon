"""A step hook that replays supervised conversations during RL training (``ReplaySFT``).

At the end of every ``every``-th RL step, after the step's own policy-gradient updates and before the step publishes,
it takes the next ``batch_size`` conversations of a JSONL file and runs one cross-entropy update on the policy's
assistant turns: the same training client and Adam state as the RL updates, here with a learning rate of its own. The
step then publishes once, its RL and replay updates together, as one new policy version, so the replay spends none of
the run's staleness budget. Its cursor is part of every full checkpoint, so a resumed run goes on with the next
conversations.

The rl trainer attaches it from the config (``trainer_options.hooks``); ``train.yaml`` shows how.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import platoon.experimental as px

# The conversations the example replays, beside this file.
CONVERSATIONS = Path(__file__).resolve().parent / "conversations.jsonl"


@dataclasses.dataclass(frozen=True)
class ReplayOptions:
    """What ``trainer_options.hooks[].options`` sets."""

    # A JSONL file of {"messages": [{"role": ..., "content": ...}, ...]} conversations (None: this example's).
    data: str | None = None
    # Replay at the end of every this many steps.
    every: int = 1
    # Conversations per replay update.
    batch_size: int = 4
    # The replay updates' learning rate (None: the policy's, train.optimizer.learning_rate).
    learning_rate: float | None = None


class CrossEntropy:
    """Tinker's built-in next-token cross entropy, on each datum's weighted tokens; its metric is the mean negative
    log-likelihood of the trained tokens.  The example carries its own (an objective is any class of
    ``platoon.experimental.Objective``); platoon's, the sft trainer's (``platoon.train.objectives.cross_entropy``),
    reports the sum and the weight of that likelihood instead."""

    loss_fn = "cross_entropy"
    loss_fn_config: dict[str, Any] = {}
    side_channels: frozenset[str] = frozenset()

    def metrics(self, originals: list[Any], result: Any) -> dict[str, float]:
        weighted = trained = 0.0
        for datum, output in zip(originals, result.loss_fn_outputs, strict=True):
            weights = datum.loss_fn_inputs["weights"].to_torch()
            weighted -= float((output["logprobs"].to_torch() * weights).sum())
            trained += float(weights.sum())
        return {"nll": weighted / trained} if trained else {}


class ReplaySFT(px.StepHookBase):
    """Replays supervised conversations at the end of every ``every``-th step: one cross-entropy update of the
    policy the step trains, published with the step's own."""

    name = "replay_sft"
    # Its entry of every full checkpoint's loop_state: where it stands in the conversations.
    keys = ("replay_sft",)

    def __init__(self, options: ReplayOptions, conversations: list[list[dict[str, Any]]], policy: Any) -> None:
        self.options = options
        self.conversations = conversations
        self.policy = policy
        self.position = 0

    @classmethod
    def from_run(cls, run: Any, options: Mapping[str, Any]) -> ReplaySFT:
        """The hook as ``trainer_options.hooks`` names it, with its options, for the policy the run trains
        (``run.policies.leader``)."""

        parsed = ReplayOptions(**options)
        lines = Path(parsed.data or CONVERSATIONS).read_text(encoding="utf-8").splitlines()
        conversations = [json.loads(line)["messages"] for line in lines if line.strip()]
        return cls(parsed, conversations, run.policies[run.policies.leader])

    # --- a participant of the run's checkpoints: its cursor ------------------------------------------------------

    def resume(self, start: Any) -> None:
        """Take up the cursor the checkpoint saved (none on a fresh run), before the training client opens."""

        self.position = start.loop_state.get("replay_sft", {}).get("position", 0)

    async def checkpoint_state(self, name: str) -> dict[str, Any]:
        return {"replay_sft": {"position": self.position}}

    # --- the step hook -----------------------------------------------------------------------------------------------

    async def update(self, end: px.StepEnd) -> None:
        """At the end of every ``every``-th step, one cross-entropy update on the next ``batch_size``
        conversations."""

        if (end.step + 1) % self.options.every:
            return
        count = self.options.batch_size
        batch = [self.conversations[(self.position + index) % len(self.conversations)] for index in range(count)]
        await end.train(self.policy.id, self._datums(batch), objective=CrossEntropy(), adam_params=self._adam_params())
        self.position = (self.position + count) % len(self.conversations)
        end.tracker.scalar(**{"replay_sft/conversations": float(count)})

    def _datums(self, batch: list[list[dict[str, Any]]]) -> list[Any]:
        """The conversations rendered with the policy's renderer, the loss on their assistant turns, as a mean over
        the batch's trained tokens (Tinker sums the loss)."""

        from tinker_cookbook.renderers import TrainOnWhat
        from tinker_cookbook.supervised.common import datum_from_model_input_weights

        renderer = self.policy.renderer
        examples = [
            renderer.build_supervised_example(messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
            for messages in batch
        ]
        trained = sum(float(weights.sum()) for _, weights in examples)
        return [datum_from_model_input_weights(model_input, weights / trained) for model_input, weights in examples]

    def _adam_params(self) -> Any:
        """This update's Adam parameters: the policy's (``optimizer_params``, a ``tinker.AdamParams``), with the
        replay's learning rate when it has one."""

        if self.options.learning_rate is None:
            return None
        return self.policy.optimizer_params.model_copy(update={"learning_rate": self.options.learning_rate})
