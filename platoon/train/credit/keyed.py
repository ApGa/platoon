"""``credit: keyed``: each agent's rows centered on the mean reward of the group's agents that share its key.

The key is any of ``task``, ``policy``, ``role`` and ``depth`` (``credit_options.key``; by default ``[task, policy,
role]``, RFC 0003's): under ``[task, role]`` a group's root agents are centered on the roots' mean and its subagents on
the subagents', instead of every agent on its rollout's root baseline as under ``group``.  With ``leave_one_out`` an
agent's baseline is the mean of the others of its key, and an agent alone in its key gets an advantage of 0.  Each
agent trains on its own reward, as the converter loaded it: an agent that was not scored (its reward None) enters no
baseline, and the rows it has, if any, train nothing.  This is the subset of RFC 0001's ``BaselineSpec`` that has a
user.

An agent whose trajectory names spans (``AgentRun.spans``) enters its key's baseline with its own reward all the same,
but trains each call of each of its scored spans on that span's reward less its baseline shared equally by its scored
spans, and every other call on nothing.  So one span that holds all of its credit trains as the agent would, and when
its spans hold all of its credit their advantages sum to its own, however many spans it was cut into (to 0 for an
agent alone in its key under ``leave_one_out``, whose spans then train against each other).  For a group whose scored
agents name spans, ``credit/keyed/unscored_tokens`` counts the action tokens of those agents that lie in no scored span
of theirs, which train nothing.

``credit/token_advantage/abs_avg`` is the mean absolute advantage of the group's action tokens as it sets them, those
that train nothing included: the size of what they train with, which ``credit/agent_advantage/abs_avg``, over the
scored agents' own advantages, no longer says once spans cut an agent's credit.
"""

from collections import defaultdict
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field

import torch

from platoon.train.credit.api import AgentRun, CreditEstimatorBase, CreditGroup, CreditSetup, TrainingSample
from platoon.train.credit.group import group_baselines

# What an agent's key may be made of.
KEY_PARTS = ("task", "policy", "role", "depth")


@dataclass
class KeyedOptions:
    """``credit_options`` under ``credit: keyed``."""

    # What an agent's baseline is keyed by: one or more of task, policy, role and depth, each once. A group holds one
    # task, so task alone keys every agent of the group together.
    key: list[str] = field(default_factory=lambda: ["task", "policy", "role"])
    # Center each agent on the mean of the others of its key, not of all of them (an agent alone in its key gets 0, and
    # spans that hold all of its credit train against each other, summing to 0).
    leave_one_out: bool = False

    def __post_init__(self) -> None:
        if not self.key or any(part not in KEY_PARTS for part in self.key) or len(set(self.key)) != len(self.key):
            raise ValueError(f"key: one or more of {', '.join(KEY_PARTS)}, each once, got {self.key!r}")


class Keyed(CreditEstimatorBase):
    """Baselines keyed by task, policy, role or depth (the module's docstring)."""

    needs_agents = True
    reads_spans = True
    options_schema = KeyedOptions
    options: KeyedOptions

    def __init__(self, setup: CreditSetup, options: KeyedOptions | None = None) -> None:
        super().__init__(setup, options if options is not None else KeyedOptions())

    async def assign(self, group: CreditGroup) -> None:
        by_key: dict[tuple[Hashable, ...], list[tuple[AgentRun, float]]] = defaultdict(list)
        for agent in group.agents():
            if agent.reward is None:
                for sample in agent.samples:  # unscored: trains nothing
                    sample.set_advantage(0.0)
            else:
                by_key[self._key(group, agent)].append((agent, agent.reward))
        advantages = []
        unscored_tokens: int | None = None  # counted once a scored agent names spans
        for scored in by_key.values():
            rewards = [reward for _, reward in scored]
            baselines = group_baselines(rewards, [True] * len(rewards), leave_one_out=self.options.leave_one_out)
            for (agent, reward), baseline in zip(scored, baselines, strict=True):
                advantages.append(reward - baseline)
                if agent.spans is None:
                    for sample in agent.samples:
                        sample.set_advantage(reward - baseline)
                else:
                    unscored_tokens = (unscored_tokens or 0) + _train_spans(agent, baseline)
        if advantages:
            metrics = {
                "credit/keyed/baselines": float(len(by_key)),
                "credit/agent_advantage/abs_avg": sum(map(abs, advantages)) / len(advantages),
            }
            if (size := _token_advantage(group.samples())) is not None:
                metrics["credit/token_advantage/abs_avg"] = size
            if unscored_tokens is not None:
                metrics["credit/keyed/unscored_tokens"] = float(unscored_tokens)
            group.stats.scalar(**metrics)

    def _key(self, group: CreditGroup, agent: AgentRun) -> tuple[Hashable, ...]:
        parts = {"task": group.task_id, "policy": agent.policy, "role": agent.role, "depth": agent.depth}
        return tuple(parts[part] for part in self.options.key)


def _train_spans(agent: AgentRun, baseline: float) -> int:
    """Train each call of each scored span of ``agent``, whose trajectory names spans, on the span's reward less
    ``baseline`` shared equally by its scored spans, and its other calls on nothing (the module's docstring); return how
    many of its action tokens lie in no scored span.  A ``ValueError`` for a row with action tokens that names no calls
    (``TrainingSample.calls``): which span its tokens lie in is unknown."""

    scored: list[tuple[frozenset[str], float]] = []
    for span in agent.spans or ():
        if span.reward is not None:
            scored.append((span.calls, span.reward))
    advantage = {call: reward - baseline / len(scored) for calls, reward in scored for call in calls}
    unscored = 0
    for sample in agent.samples:
        actions = sample.mask > 0
        if not sample.calls and bool(actions.any()):
            raise ValueError(
                f"a row of agent {agent.agent_id}, whose trajectory names spans, names none of its model calls "
                "(TrainingSample.calls), so which span its tokens lie in is unknown"
            )
        values = torch.zeros_like(sample.advantages)
        in_scored = torch.zeros_like(actions)
        for call, start, end in sample.calls:
            if call in advantage:
                values[start:end] = advantage[call]
                in_scored[start:end] = True
        sample.set_advantages(values)
        unscored += int((actions & ~in_scored).sum())
    return unscored


def _token_advantage(samples: Iterable[TrainingSample]) -> float | None:
    """The mean absolute advantage of ``samples``' action tokens; None when they have none."""

    total, tokens = 0.0, 0
    for sample in samples:
        actions = sample.mask > 0
        total += float(sample.advantages[actions].abs().sum())
        tokens += int(actions.sum())
    return total / tokens if tokens else None
