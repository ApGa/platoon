"""The ``single`` program's subagent reward finalizers, and core's reward processor for recursive agents
(``platoon.rewards``): ``single`` applies ``skip_subagent_reward_computation``, then ``propagate_root_success``, as
the original recursive rollouts did, and ``platoon-delegation`` resolves with nothing installed."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from extension_site import sandbox

from platoon import extensions, rewards
from platoon.config_defs import RolloutConfig
from platoon.envs.service import Outcome
from platoon.programs import single
from platoon.registry import BUILTIN_REWARD_PROCESSORS, resolve_component
from platoon.rewards import delegation_capped, zero_subagent_rewards
from platoon.trajectory import ParentInfo, Trajectory, TrajectoryCollection, TrajectoryStep
from platoon.utils.subagent_rewards import propagate_root_success


class _Episode:
    """Enough of an ``EpisodeContext`` for ``single``: the rollout config, and what the program asked for."""

    def __init__(self, config: RolloutConfig) -> None:
        self.config = config
        self.ran = False
        self.finalizers: list[Any] = []

    async def run(self, harness: str | None = None, **options: Any) -> None:
        self.ran = True

    def add_finalizer(self, fn: Any) -> None:
        self.finalizers.append(fn)

    async def grade(self) -> Outcome:
        return Outcome(status="scored", rewards={"reward": 1.0})


@pytest.mark.parametrize(
    ("skip", "propagate", "finalizers"),
    [
        (False, None, []),
        (False, True, [propagate_root_success]),
        (True, False, [zero_subagent_rewards]),
        (True, True, [zero_subagent_rewards, propagate_root_success]),
    ],
)
def test_single_adds_the_subagent_reward_finalizers_the_rollout_config_asks_for(skip, propagate, finalizers):
    config = RolloutConfig(skip_subagent_reward_computation=skip, propagate_root_success=propagate)
    episode = _Episode(config)

    outcome = asyncio.run(single(episode))

    assert episode.ran and (outcome.status, outcome.reward) == ("scored", 1.0)
    assert episode.finalizers == finalizers


def _tree() -> TrajectoryCollection:
    """A root that launched two children, one of which succeeded: a recursive TextCraft-Synth episode, graded."""

    def child(name: str, reward: float) -> Trajectory:
        step = TrajectoryStep(misc={"reward_misc": {"reward/success": reward}})
        return Trajectory(id=name, parent_info=ParentInfo(id="root", fork_step=1), steps=[step], reward=reward)

    launch = TrajectoryStep(misc={"reward_misc": {"reward/subagent_launched": 2, "reward/subagent_succeeded": 1.0}})
    finish = TrajectoryStep(misc={"reward_misc": {"reward/success": 1.0}})
    root = Trajectory(id="root", steps=[launch, finish], reward=1.0)
    return TrajectoryCollection(trajectories={t.id: t for t in (root, child("solved", 1.0), child("failed", 0.0))})


def test_the_finalizers_in_singles_order_give_every_agent_the_roots_success():
    collection = _tree()
    root = collection.trajectories["root"]

    zero_subagent_rewards(collection)
    assert [trajectory.reward for trajectory in collection.trajectories.values()] == [1.0, 0.0, 0.0]
    assert root.steps[0].misc["reward_misc"]["reward/subagent_succeeded"] == 0.0
    assert collection.trajectories["solved"].steps[-1].misc["reward_misc"]["reward/success"] == 0.0

    propagate_root_success(collection)
    assert [trajectory.reward for trajectory in collection.trajectories.values()] == [1.0, 1.0, 1.0]
    assert root.steps[0].misc["reward_misc"]["reward/subagent_succeeded"] == 2.0


def test_delegation_capped_scores_the_success_and_reports_every_component(monkeypatch):
    launched = {"misc": {"reward_misc": {"reward/subagent_launched": 2, "reward/subagent_succeeded": 1.0}}}
    trajectory = {"reward": 1.0, "steps": [launched, {"misc": {"reward_misc": {"reward/success": 1.0}}}]}
    components = {"reward/success": 1.0, "reward/subagent_launched": 2.0, "reward/subagent_succeeded": 1.0}

    # The bonus is capped at 0, so the score is the success alone, as in the RAO runs.
    assert rewards.DELEGATION_REWARD_CAP == 0.0
    assert delegation_capped(trajectory) == (1.0, components)
    # A trajectory whose steps report nothing scores its reward, and reports 0 for the delegation components.
    assert delegation_capped({"reward": 0.5, "steps": [{"misc": {}}]}) == (
        0.5,
        {"reward/success": 0.5, "reward/subagent_launched": 0.0, "reward/subagent_succeeded": 0.0},
    )
    # With a cap, the share of the launched children that succeeded adds up to it.
    monkeypatch.setattr(rewards, "DELEGATION_REWARD_CAP", 0.5)
    assert delegation_capped(trajectory)[0] == 1.25


def test_platoon_delegation_is_a_built_in_whatever_is_installed(tmp_path, monkeypatch):
    with sandbox(tmp_path, monkeypatch) as site:
        # A distribution of the project's own that declares the name: core's function wins, and its entry point,
        # which names a module that does not exist, is never loaded.
        site.install(
            "platoon-leftover",
            entry_points={"platoon.reward_processors": {"platoon-delegation": "platoon_leftover:delegation_capped"}},
        )

        assert resolve_component("reward_processor", "platoon-delegation") is delegation_capped
        origin = extensions.recorded_origin("reward_processor", "platoon-delegation")
        assert (origin.distribution, origin.entry_point) == ("platoon", None)
        rows = {(info.name, info.distribution): info for info in extensions.list_components("reward_processor")}
        builtin = rows[("platoon-delegation", "platoon")]
        assert builtin.builtin and builtin.value == BUILTIN_REWARD_PROCESSORS["platoon-delegation"]
        assert rows[("platoon-delegation", "platoon-leftover")].status.startswith("ignored: platoon-delegation is")
        assert extensions.load_failures("reward_processor") == {}
