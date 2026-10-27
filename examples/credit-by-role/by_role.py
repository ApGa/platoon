"""``ByRole``: a credit estimator that gives each role of a multi-agent rollout a baseline of its own.

Under the workflow's own credit, every agent of a rollout is centered on its rollout's root baseline: in a group whose
roots all failed, a subagent that did its part still trains as if it beat the group. ``ByRole`` centers each agent on
the group's agents of its role instead: the root agents (role ``policy``) on the group's roots, the subagents (role
``subagent``) on its subagents. Select it with ``environments[].credit: by_role:ByRole``. It credits as core's
``credit: keyed`` does with its default key: a tour of the credit API, and a start for an estimator of your own.
"""

from collections import defaultdict
from dataclasses import dataclass

import platoon.experimental as px


@dataclass
class ByRoleOptions:
    # Center each agent on the mean reward of the other agents of its role, not of all of them.
    leave_one_out: bool = False


class ByRole(px.CreditEstimatorBase):
    needs_agents = True  # the rollout workers tag each row with its agent, role and policy
    options_schema = ByRoleOptions

    def __init__(self, setup: px.CreditSetup, options: ByRoleOptions | None = None) -> None:
        super().__init__(setup, options or ByRoleOptions())

    async def assign(self, group: px.CreditGroup) -> None:
        by_role: dict[str, list[px.AgentRun]] = defaultdict(list)
        for agent in group.agents():
            if agent.reward is not None:  # an agent that was not scored enters no baseline
                by_role[agent.role].append(agent)
        for role, agents in by_role.items():
            rewards = [agent.reward or 0.0 for agent in agents]
            for agent, reward, baseline in zip(agents, rewards, self.baselines(rewards)):
                for sample in agent.samples:
                    sample.set_advantage(reward - baseline)  # or sample.set_advantages(tensor), per token
            group.stats.scalar(**{f"credit/by_role/{role}/reward": sum(rewards) / len(rewards)})

    def baselines(self, rewards: list[float]) -> list[float]:
        """The group mean of ``rewards``, or with ``leave_one_out`` each one's mean of the others (an agent alone in
        its role is its own baseline, and trains nothing)."""

        total, count = sum(rewards), len(rewards)
        if not self.options.leave_one_out or count == 1:
            return [total / count] * count
        return [(total - reward) / (count - 1) for reward in rewards]
