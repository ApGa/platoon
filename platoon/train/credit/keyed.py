"""``credit: keyed``: each agent's rows centered on the mean reward of the group's agents that share its key.

The key is any of ``task``, ``policy``, ``role`` and ``depth`` (``credit_options.key``; by default ``[task, policy,
role]``, RFC 0003's): under ``[task, role]`` a group's root agents are centered on the roots' mean and its subagents on
the subagents', instead of every agent on its rollout's root baseline as under ``group``.  With ``leave_one_out`` an
agent's baseline is the mean of the others of its key, and an agent alone in its key gets an advantage of 0.  Each
agent trains on its own reward, as the converter loaded it: an agent that was not scored (its reward None) enters no
baseline, and the rows it has, if any, train nothing.  This is the subset of RFC 0001's ``BaselineSpec`` that has a
user.
"""

from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass, field

from platoon.train.credit.api import AgentRun, CreditEstimatorBase, CreditGroup, CreditSetup
from platoon.train.credit.group import group_baselines

# What an agent's key may be made of.
KEY_PARTS = ("task", "policy", "role", "depth")


@dataclass
class KeyedOptions:
    """``credit_options`` under ``credit: keyed``."""

    # What an agent's baseline is keyed by: one or more of task, policy, role and depth, each once. A group holds one
    # task, so task alone keys every agent of the group together.
    key: list[str] = field(default_factory=lambda: ["task", "policy", "role"])
    # Center each agent on the mean of the others of its key, not of all of them (an agent alone in its key gets 0).
    leave_one_out: bool = False

    def __post_init__(self) -> None:
        if not self.key or any(part not in KEY_PARTS for part in self.key) or len(set(self.key)) != len(self.key):
            raise ValueError(f"key: one or more of {', '.join(KEY_PARTS)}, each once, got {self.key!r}")


class Keyed(CreditEstimatorBase):
    """Baselines keyed by task, policy, role or depth (the module's docstring)."""

    needs_agents = True
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
        for scored in by_key.values():
            rewards = [reward for _, reward in scored]
            baselines = group_baselines(rewards, [True] * len(rewards), leave_one_out=self.options.leave_one_out)
            for (agent, reward), baseline in zip(scored, baselines, strict=True):
                advantages.append(reward - baseline)
                for sample in agent.samples:
                    sample.set_advantage(reward - baseline)
        if advantages:
            group.stats.scalar(
                **{
                    "credit/keyed/baselines": float(len(by_key)),
                    "credit/agent_advantage/abs_avg": sum(map(abs, advantages)) / len(advantages),
                }
            )

    def _key(self, group: CreditGroup, agent: AgentRun) -> tuple[Hashable, ...]:
        parts = {"task": group.task_id, "policy": agent.policy, "role": agent.role, "depth": agent.depth}
        return tuple(parts[part] for part in self.options.key)
