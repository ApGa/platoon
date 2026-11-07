"""``example-efficient-search``: an episode program that shapes number-search rewards toward fewer guesses.

The program runs the root agent with the environment's harness, grades it as usual, and registers a finalizer.
Once the rollout's trajectories are built, the finalizer

1. shapes the root's reward: a correct search keeps reward 1 within the guesses a binary search needs, and loses
   ``guess_penalty``, the program's option (``program_options`` in the config), for every guess beyond them;
2. asks a judge (``ep.judge_endpoint()``) for a one-line critique of the search and keeps it in the root's misc.
   The judge samples the policy model through the gateway; its calls are recorded for audit and never trained.

Import this module (``package: efficient_search``) to register the program. ``reward_parts`` is the environment's
reward processor: it trains on the shaped reward and reports its parts as metrics.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import openai

import platoon as pt
from platoon.utils.trajectory_status import trajectory_was_interrupted

_RANGE = re.compile(r"between (-?\d+) and (-?\d+)")


@dataclass(frozen=True)
class SearchOptions:
    """The program's options, which an environments entry sets in ``program_options``."""

    guess_penalty: float = 0.1  # reward lost per guess beyond binary search's

    def __post_init__(self) -> None:
        if not 0 <= self.guess_penalty <= 1:
            raise ValueError(f"guess_penalty: must be between 0 and 1, got {self.guess_penalty}")


@pt.program("example-efficient-search", options=SearchOptions)
async def efficient_search(ep: pt.EpisodeContext[SearchOptions]) -> pt.Outcome:
    """Run and grade the root agent; once its trajectory is built, shape its reward (:class:`EfficiencyShaping`)."""

    await ep.run()
    ep.add_finalizer(
        EfficiencyShaping(
            goal=ep.session.instruction,
            judge=ep.judge_endpoint(),
            penalty=ep.options.guess_penalty,  # program_options in the config, checked when it loads
        )
    )
    return await ep.grade()


@dataclass
class EfficiencyShaping:
    """A finalizer: shape the root's reward by its number of guesses, and add a judge's critique."""

    goal: str
    judge: pt.ModelEndpoint
    penalty: float = 0.1

    async def __call__(self, collection: pt.TrajectoryCollection) -> None:
        root = next(iter(collection.trajectories.values()))  # the root's trajectory comes first
        if trajectory_was_interrupted(root):
            return  # an untrainable outcome (timeout, cancellation, infra error) stays out of training as it is
        guesses = [call for call in root.misc.get("tool_calls", []) if call.get("name") == "guess"]
        extra = max(0, len(guesses) - binary_search_guesses(self.goal))
        root.reward = root.reward * max(0.0, 1.0 - self.penalty * extra)
        # The parts go next to reward/success, which the trace set and which evaluation counts as success.
        if root.steps:
            reward_misc = root.steps[-1].misc.setdefault("reward_misc", {})
            reward_misc["reward/extra_guesses"] = float(extra)
            reward_misc["reward/shaped"] = root.reward
        root.misc["judge_critique"] = await self.critique(guesses)

    async def critique(self, guesses: list[dict]) -> str:
        """One sentence from the judge on how the search could have used fewer guesses."""

        moves = "; ".join(f"{call.get('arguments', {}).get('number')}: {call.get('text', '')}" for call in guesses)
        client = openai.AsyncOpenAI(
            base_url=self.judge.base_url, api_key=self.judge.api_key, default_headers=self.judge.headers("judge")
        )
        try:
            reply = await client.chat.completions.create(
                model=self.judge.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Task: {self.goal}\nGuesses and feedback: {moves}\n"
                        "In one sentence, how could this search have used fewer guesses?",
                    }
                ],
                max_tokens=64,
            )
        finally:
            await client.close()
        return reply.choices[0].message.content or ""


def reward_parts(trajectory: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """A reward processor (``reward_processor: efficient_search:reward_parts``): the trajectory's shaped reward, with
    the ``reward/*`` parts of its last step as components, which training reports as metrics."""

    steps = trajectory.get("steps") or []
    parts = ((steps[-1].get("misc") or {}).get("reward_misc") or {}) if steps else {}
    components = {key: float(value) for key, value in parts.items() if key.startswith("reward/")}
    return float(trajectory["reward"]), components


def binary_search_guesses(goal: str | None) -> int:
    """How many guesses a binary search needs, at most, for the range a number-search goal names."""

    match = _RANGE.search(goal or "")
    if match is None:
        return 0
    low, high = (int(value) for value in match.groups())
    return max(1, math.ceil(math.log2(high - low + 2)))
