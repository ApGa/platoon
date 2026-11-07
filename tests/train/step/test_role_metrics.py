"""A step's metrics by agent role (``platoon.train.step.roles``), in a run with an entry whose estimator credits each
role of its trees apart (``credit: roles``): whole starts of the rl trainer on CPU, on ``platoon.testing``'s fake Tinker
service, whose training logprobs are all 0, with training workflows whose groups hold rows tagged with roles and rows
without one.  Each step logs each role's share of the advantage mass it sent, over every row (an untagged one in the
whole, under no role), its rows' mean ``|log r|``, the share of their tokens whose ratio is outside ``[0.5, 2]`` and
their sampler's entropy; a run without such an entry logs none of them and sends the same rows.  A row the per-datum
stale filter dropped, or one a finished minibatch let go, lends its role to no other row, in one minibatch or several;
a second policy's lane logs its own under its prefix, in lockstep and on its own clock; a role's share of the mass,
held by ``roles`` as each group is credited and settled as a run does, is its part of what each step sends; the
counting itself; and what turns it on: the class of an entry's estimator, however the config names it."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import tinker
import torch
from fakes.independent import WORKER_URL, Lineages, Services, TaggedGroups, start

import platoon.experimental as px
from platoon import registry as registry_module
from platoon.options import schema_options
from platoon.testing import trainers as checks
from platoon.train.components import task_ids_to_dataset
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig, WorkflowConfig
from platoon.train.credit.api import AgentRun, CreditGroup, CreditSetup, MemberRollout, PolicyView, TrainingSample
from platoon.train.credit.registry import REGISTRY_KIND, resolve_credit_estimator
from platoon.train.credit.roles import Roles, RolesOptions
from platoon.train.credit.runtime import assign_credit, steps_log_by_role
from platoon.train.policy.learner import PrefixedTracker
from platoon.train.policy.spec import MAIN, agent_bindings, policy_specs
from platoon.train.step.roles import RoleMetrics
from platoon.train.step.step import TrainStep
from platoon.train.trainer.api import TaskSources, TrainerSetup, Workflows
from platoon.train.trainer.rl import PlatoonTinkerRLTrainer
from platoon.train.workflow.output import PolicyGroupOutput
from platoon.train.workflow.protocol import OutputReport
from platoon.utils.config import build_config
from platoon.utils.stats_logger import StatsLogger
from platoon.utils.stats_tracker import DeferredStats
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult


def _row(advantage: float, logprob: float, version: int = 0) -> tinker.Datum:
    """A row of two action tokens with ``advantage``, sampled at ``logprob``, at ``version``."""

    inputs = {
        "target_tokens": [2, 3, 4],
        "logprobs": [0.0, logprob, logprob],
        "advantages": [0.0, advantage, advantage],
        "mask": [0.0, 1.0, 1.0],
        "checkpoint_version": [version],
    }
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={key: tinker.TensorData.from_torch(torch.tensor(value)) for key, value in inputs.items()},
    )


class _ByRole:
    """An entry's estimator as the steps see it: one whose class has them log their metrics by role, as ``roles``'s
    does."""

    logs_by_role = True


class Productions:
    """A training workflow whose every group holds a generator's row (advantage 3, sampled at -0.25), two solvers'
    (advantages -1 and 1, at -1) and a row without a role (advantage 2, at -0.5); with ``by_role``, it credits an entry
    with an estimator that has the steps log by role."""

    def __init__(self, policies: Any, *, by_role: bool = True) -> None:
        self.policies = policies
        self.entry_credit = {"generate": _ByRole()} if by_role else {}

    async def arun_episode(self, data: dict) -> PolicyGroupOutput:
        await asyncio.sleep(0)
        version = self.policies[MAIN].version
        rows = [(3.0, -0.25, "generator"), (-1.0, -1.0, "solver"), (1.0, -1.0, "solver"), (2.0, -0.5, None)]
        samples = tuple(
            TrainingSample(_row(advantage, logprob, version), MAIN, 0, role=role, version=version)
            for advantage, logprob, role in rows
        )
        return PolicyGroupOutput(samples, OutputReport())


class Mixed(Productions):
    """Groups of an entry credited by role, whose generator's row is always beyond the staleness bound (sampled 10
    versions back) and whose solvers' rows are fresh, each followed by a group of an entry whose estimator reads no
    agents, a list of six untagged rows, as a curated floor's: under the per-datum stale filter each tagged group's
    generator row is dropped, and its copy freed, before the next group's copies are made."""

    async def arun_episode(self, data: dict) -> Any:
        version = self.policies[MAIN].version
        if int(data["task_id"].split("-")[1]) % 2:
            await asyncio.sleep(0.01)
            return [_row(2.0, -0.5, version) for _ in range(6)]
        rows = [(3.0, version - 10, "generator"), (-1.0, version, "solver"), (1.0, version, "solver")]
        samples = tuple(
            TrainingSample(_row(advantage, -1.0, at), MAIN, 0, role=role, version=at) for advantage, at, role in rows
        )
        return PolicyGroupOutput(samples, OutputReport())


def _train(root: Path, workflow: Any = Productions, *, tasks: int = 8, **train: Any) -> tuple[Any, list[dict]]:
    """A start of the rl trainer on the fake service, two steps of two groups unless ``train`` says otherwise; the
    service, and each step's metrics."""

    config = build_config(
        PlatoonTinkerRLTrainerConfig,
        {
            "train": {"model_name": "m", "renderer_name": "r", "batch_size": 2, "max_training_steps": 2, **train},
            "eval": {"every": 0},
            "watchdog": {"enabled": False},
            "stats": {"wandb": {"mode": "disabled"}},
            "log_path": str(root / "logs"),
        },
    )
    setup = TrainerSetup(
        config=config,
        policy_specs=policy_specs(config),
        agents=agent_bindings(config),
        tasks=TaskSources(train_dataset=task_ids_to_dataset([f"task-{index}" for index in range(tasks)])),
        workflow_builder=lambda policies: Workflows(workflow(policies), None),
    )
    service, logged = checks.FakePolicyService(), []
    with (
        checks._faked(service),
        checks._restored_signals(),
        pytest.MonkeyPatch.context() as patch,
    ):
        patch.setattr(StatsLogger, "log", lambda self, step, stats, *args, **kwargs: logged.append(dict(stats)))
        asyncio.run(checks._run(PlatoonTinkerRLTrainer.from_setup(setup)))
    return service, logged


def test_each_step_logs_each_role_s_share_of_the_mass_it_sent_and_how_far_its_ratios_strayed(tmp_path):
    service, metrics = _train(tmp_path / "roles")

    assert len(metrics) == 2
    for stats in metrics:
        # Of each group's mass of 14 (the generator's 6, the solvers' 4, and 4 of the row without a role).
        assert stats["train/roles/generator/mass_share"] == pytest.approx(6 / 14)
        assert stats["train/roles/solver/mass_share"] == pytest.approx(4 / 14)
        # Trained at logprob 0: |log r| is how far below 0 each was sampled; 1 > log 2 puts r outside [0.5, 2].
        assert stats["train/roles/generator/log_ratio/abs_avg"] == pytest.approx(0.25)
        assert stats["train/roles/solver/log_ratio/abs_avg"] == pytest.approx(1.0)
        assert (stats["train/roles/generator/ratio_outside_frac"], stats["train/roles/solver/ratio_outside_frac"]) == (
            0.0,
            1.0,
        )
        # The sampler's entropy over each role's tokens: the mean negative logprob they were sampled at.
        assert stats["train/roles/generator/entropy"] == pytest.approx(0.25)
        assert stats["train/roles/solver/entropy"] == pytest.approx(1.0)
        assert not [name for name in stats if name.startswith("train/roles/None")]

    plain, without = _train(tmp_path / "plain", lambda policies: Productions(policies, by_role=False))
    assert not [name for stats in without for name in stats if "roles/" in name]
    sent = [[value.tolist() for value in datum.loss_fn_inputs.values()] for datum in service.datums]
    assert sent == [[value.tolist() for value in datum.loss_fn_inputs.values()] for datum in plain.datums]


def test_a_row_the_stale_filter_dropped_lends_its_role_to_no_row_kept_after_it():
    kept: list[tinker.Datum] = []
    lane: Any = SimpleNamespace(policy_id=MAIN, add_rows=lambda fill, datums: kept.extend(datums))
    step: Any = SimpleNamespace(roles=RoleMetrics(), tally=SimpleNamespace(total_datums=0))
    # A group whose generator's row the per-datum stale filter masks, then one whose rows carry no tags, each row made
    # before the step copies any: the step's copies of the second take the addresses its copies of the first let go.
    generated, floor = [_row(5.0, -1.0), _row(1.0, -1.0)], [_row(1.0, -1.0) for _ in range(8)]
    TrainStep._keep_rows(step, lane, generated, SimpleNamespace(masked={0}), None, ["generator", "solver"])
    TrainStep._keep_rows(step, lane, floor, SimpleNamespace(masked=()), None, None)
    step.roles.sent(MAIN, kept, 1.0)
    recorded: dict[str, float] = {}
    step.roles.record(MAIN, SimpleNamespace(scalar=lambda **values: recorded.update(values)))

    assert recorded == {"roles/generator/mass_share": 0.0, "roles/solver/mass_share": pytest.approx(2 / 18)}


@pytest.mark.parametrize("parts", [{}, {"num_minibatches": 2, "num_microbatches": 2}])
def test_rows_dropped_as_stale_beside_untagged_rows_count_under_no_other_role(tmp_path, parts):
    batch = 8 if not parts else 4
    _, metrics = _train(
        tmp_path,
        Mixed,
        tasks=200,
        max_training_steps=6,
        batch_size=batch,
        stale_filter="datum",
        max_staleness=3,
        **parts,
    )

    # Every step that kept a tagged group counts its generator's row, which it never sent, at 0.
    counted = [stats for stats in metrics if "train/roles/generator/mass_share" in stats]
    assert len(metrics) == 6 and len(counted) >= 3
    assert [stats["train/roles/generator/mass_share"] for stats in counted] == [0.0] * len(counted)
    assert not [name for stats in metrics for name in stats if name.startswith("train/roles/generator/log_ratio")]
    assert all(
        0 < stats["train/roles/solver/mass_share"] <= 1 for stats in counted if "train/roles/solver/entropy" in stats
    )


def test_the_counting_by_role_of_a_lane_of_another_policy_goes_under_its_prefix():
    roles, recorded = RoleMetrics(), {}
    rows = [_row(1.0, -2.0), _row(-3.0, -0.5), _row(5.0, -0.5)]
    roles.keep("worker", rows, ["critic", "writer", None])
    roles.keep("worker", [_row(1.0, -1.0)], ["idle"])  # kept, then masked as stale: never sent
    roles.sent("worker", rows, 0.5)
    trained = tinker.TensorData.from_torch(torch.full((3,), -1.5))  # trained at -1.5: log r is -1.5 less the sampled
    roles.observer("worker")(rows, SimpleNamespace(loss_fn_outputs=[{"logprobs": trained}] * 3))
    tracker: Any = SimpleNamespace(scalar=lambda **values: recorded.update(values))
    roles.record("worker", PrefixedTracker(tracker, "policies/worker/"))
    roles.record(MAIN, tracker)  # a lane it counted nothing of

    assert recorded == {
        "policies/worker/roles/critic/mass_share": pytest.approx(2 / 18),
        "policies/worker/roles/critic/log_ratio/abs_avg": pytest.approx(0.5),
        "policies/worker/roles/critic/ratio_outside_frac": 0.0,
        "policies/worker/roles/critic/entropy": pytest.approx(2.0),
        "policies/worker/roles/writer/mass_share": pytest.approx(6 / 18),
        "policies/worker/roles/writer/log_ratio/abs_avg": pytest.approx(1.0),
        "policies/worker/roles/writer/ratio_outside_frac": 1.0,
        "policies/worker/roles/writer/entropy": pytest.approx(0.5),
        "policies/worker/roles/idle/mass_share": 0.0,  # its rows sent nothing
    }


class AsReward(px.CreditEstimatorBase):
    """Each scored agent's rows at its reward: a role's estimator that leaves its raw mass to the test."""

    needs_agents = True

    async def assign(self, group: px.CreditGroup) -> None:
        for agent in group.agents():
            for sample in agent.samples:
                sample.set_advantage(agent.reward or 0.0)


class Credited:
    """A training workflow whose groups are productions, a generator's row and four solvers' (two action tokens each),
    credited by a real ``roles`` as the run's workflow credits its groups, the generator held at a share of 0.1: over
    its first 96 groups the generator's rewards grow and the solvers' shrink, so that the generator's raw part of each
    group's mass rises from about two fifths to two thirds, and stays there.  The run settles each group's credit as it
    files it, as it does a workflow's."""

    def __init__(self, policies: Any) -> None:
        self.policies = policies
        by_role = {
            "generator": {"credit": f"{__name__}:AsReward", "share": 0.1},
            "solver": {"credit": f"{__name__}:AsReward"},
        }
        setup = CreditSetup(
            name="roles",
            entry="generate",
            settings=WorkflowConfig(group_size=1),
            policies=(MAIN,),
            max_staleness={MAIN: 2},
        )
        self.roles = Roles(setup, schema_options(RolesOptions).parse({"by_role": by_role}, "credit_options"))
        self.entry_credit = {"generate": self.roles}
        self.rng, self.raw = random.Random(11), []

    async def arun_episode(self, data: dict) -> PolicyGroupOutput:
        await asyncio.sleep(0)
        version = self.policies[MAIN].version
        progress = min(len(self.raw) / 96, 1.0)
        generator = self.rng.choice([-1, 1]) * (1 + 1.5 * progress) * self.rng.uniform(0.5, 1.5)
        solvers = [self.rng.choice([-1, 1]) * (1 - 0.4 * progress) * self.rng.uniform(0, 1) for _ in range(4)]
        root = AgentRun("generator", "t-g", None, "generator", MAIN, 0, generator)
        agents = [root, *(AgentRun(f"s{i}", f"t-s{i}", "t-g", "solver", MAIN, 1, r) for i, r in enumerate(solvers))]
        for agent in agents:
            row = _row(0.0, -1.0, version)
            agent.samples = [TrainingSample(row, MAIN, 0, agent.agent_id, agent.role, agent.trajectory_id, 1, version)]
        samples = [sample for agent in agents for sample in agent.samples]
        result = TrajectoryCollectionResult(
            datums=[sample.datum for sample in samples],
            task_reward=generator,
            trajectory_stats=[],
            root_rewards_dict={},
            task_reward_valid=True,
        )
        views = {MAIN: PolicyView(MAIN, current_version=lambda: self.policies[MAIN].version)}
        member = MemberRollout(0, samples, agents, generator, True, result)
        group = CreditGroup(
            data["task_id"],
            None,
            "generate",
            WorkflowConfig(group_size=1),
            [member],
            views,
            {MAIN: version},
            DeferredStats(),
        )
        credit = await assign_credit(self.roles, group)
        self.raw.append((2 * abs(generator), 2 * sum(abs(r) for r in solvers)))
        return PolicyGroupOutput(tuple(samples), OutputReport(), credit)


def test_a_role_s_share_held_as_each_group_is_credited_is_its_part_of_what_each_step_sends(tmp_path, monkeypatch):
    built: list[Credited] = []

    def workflow(policies: Any) -> Credited:
        built.append(Credited(policies))
        return built[-1]

    _, metrics = _train(tmp_path, workflow, tasks=400, max_training_steps=64, batch_size=4)

    # From the 40th step of 4 groups on, the generator's raw part of the mass is two thirds; each step sends a tenth.
    raw = built[0].raw[160:]
    assert sum(g for g, _ in raw) / sum(g + s for g, s in raw) > 0.6
    late = [stats["train/roles/generator/mass_share"] for stats in metrics[40:]]
    assert sum(late) / len(late) == pytest.approx(0.1, abs=0.02)
    assert built[0].roles._masses[MAIN].live()  # every group's credit settled as the run filed it


# --- a second policy --------------------------------------------------------------------------------------------------


class TwoPolicies:
    """A training workflow, credited by role, whose every group holds rows of main (a generator's, advantage 3 at
    -0.25, and a solver's, advantage 1 at -1) and of the worker (a critic's, advantage 1 at -0.5, and a writer's,
    advantage 3 at -2)."""

    binds_policies = True

    def __init__(self, policies: Any) -> None:
        self.policies = policies
        self.entry_credit = {"generate": _ByRole()}

    async def arun_episode(self, data: dict) -> PolicyGroupOutput:
        await asyncio.sleep(0)
        rows = [
            (MAIN, "generator", 3.0, -0.25),
            (MAIN, "solver", 1.0, -1.0),
            ("worker", "critic", 1.0, -0.5),
            ("worker", "writer", 3.0, -2.0),
        ]
        samples = tuple(
            TrainingSample(_row(advantage, logprob, self.policies[policy].version), policy, 0, role=role)
            for policy, role, advantage, logprob in rows
        )
        return PolicyGroupOutput(samples, OutputReport())

    def publish_model_binding(self, sampling_client: Any, version: int, *, policy: str | None = None) -> None:
        pass


def test_a_second_policy_s_lane_logs_its_rows_metrics_by_role_under_its_prefix_in_lockstep(tmp_path, monkeypatch):
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(StatsLogger, "log", lambda self, step, stats, *args, **kwargs: logged.append(dict(stats)))
    data = {
        "train": {"model_name": "m", "renderer_name": "r", "batch_size": 2, "max_training_steps": 2},
        "policies": {"worker": {"model_name": "w", "renderer_name": "r", "tinker_base_url": WORKER_URL}},
        "agents": {"roles": {"critic": {"policy": "worker"}, "writer": {"policy": "worker"}}},
        "eval": {"every": 0},
        "watchdog": {"enabled": False},
        "stats": {"wandb": {"mode": "disabled"}},
        "log_path": str(tmp_path / "logs"),
    }
    config = build_config(PlatoonTinkerRLTrainerConfig, copy.deepcopy(data))
    setup = TrainerSetup(
        config=config,
        policy_specs=policy_specs(config),
        agents=agent_bindings(config),
        tasks=TaskSources(train_dataset=task_ids_to_dataset([f"task-{index}" for index in range(8)])),
        workflow_builder=lambda policies: Workflows(TwoPolicies(policies), None),
    )
    services = Services(Lineages())
    with (
        checks._faked(services.main),
        mock.patch.object(tinker, "ServiceClient", services.service_client),
        checks._restored_signals(),
    ):
        asyncio.run(checks._run(PlatoonTinkerRLTrainer.from_setup(setup)))

    assert len(logged) == 2
    for stats in logged:
        # Each lane's shares are of its own policy's mass: main's 8, the worker's 8.
        assert stats["train/roles/generator/mass_share"] == pytest.approx(6 / 8)
        assert stats["train/roles/solver/entropy"] == pytest.approx(1.0)
        assert stats["train/policies/worker/roles/critic/mass_share"] == pytest.approx(2 / 8)
        assert stats["train/policies/worker/roles/writer/mass_share"] == pytest.approx(6 / 8)
        assert stats["train/policies/worker/roles/writer/log_ratio/abs_avg"] == pytest.approx(2.0)
        assert stats["train/policies/worker/roles/critic/entropy"] == pytest.approx(0.5)
        assert (
            "train/roles/critic/mass_share" not in stats
            and "train/policies/worker/roles/generator/mass_share" not in stats
        )


class RoleTagged(TaggedGroups):
    """The independent runs' tagged groups, credited by role: main's row a generator's, the worker's a critic's."""

    def __init__(self, policies: Any) -> None:
        super().__init__(policies)
        self.entry_credit = {"generate": _ByRole()}

    async def arun_episode(self, data: dict) -> PolicyGroupOutput:
        output = await super().arun_episode(data)
        main, worker = output.samples
        roles = (dataclasses.replace(main, role="generator"), dataclasses.replace(worker, role="critic"))
        return PolicyGroupOutput(roles, OutputReport())


def test_each_policy_stepping_on_its_own_clock_logs_its_own_metrics_by_role(tmp_path, monkeypatch):
    found: list[tuple[dict[str, Any], str]] = []
    monkeypatch.setattr(StatsLogger, "log_row", lambda self, stats, *, axis: found.append((dict(stats), axis)))
    start(tmp_path, Lineages(), steps=2, workflow=RoleTagged)

    mains = [stats for stats, axis in found if axis == "run/step"]
    workers = [stats for stats, axis in found if axis == "policies/worker/step"]
    assert len(mains) == 2 and len(workers) == 8
    for stats in mains:
        assert stats["train/roles/generator/mass_share"] == 1.0
        assert stats["train/roles/generator/entropy"] == pytest.approx(0.5)
        assert not [name for name in stats if "roles/critic" in name]
    for stats in workers:
        assert stats["policies/worker/train/roles/critic/mass_share"] == 1.0
        assert stats["policies/worker/train/roles/critic/log_ratio/abs_avg"] == pytest.approx(0.875)
        assert not [name for name in stats if "roles/generator" in name]


# --- what turns it on ------------------------------------------------------------------------------------------------


def test_a_run_logs_by_role_when_an_entry_s_estimator_s_class_says_so_however_the_config_names_it(monkeypatch):
    monkeypatch.setattr(registry_module, "_REGISTRIES", {**registry_module._REGISTRIES})
    registry_module._REGISTRIES.pop(REGISTRY_KIND, None)
    px.credit_estimator("acme-roles")(type("AcmeRoles", (Roles,), {}))  # a package's, derived from core's
    setup = CreditSetup(name="x", entry="generate", settings=WorkflowConfig(), policies=(MAIN,))

    def built(ref: str) -> Any:
        cls = resolve_credit_estimator(ref)
        options = {"by_role": {"solver": {"credit": "keyed"}}} if issubclass(cls, Roles) else {}
        return cls(setup, schema_options(cls.options_schema).parse(options, "credit_options"))

    for ref in ("roles", "platoon.train.credit.roles:Roles", "acme-roles"):
        workflow = SimpleNamespace(entry_credit={"floor": built("keyed"), "generate": built(ref)})
        assert steps_log_by_role(workflow) is True, ref
    assert steps_log_by_role(SimpleNamespace(entry_credit={"generate": built("running")})) is False
    assert steps_log_by_role(SimpleNamespace()) is False  # a workflow that credits no entry with an estimator
