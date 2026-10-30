"""``GroupRolloutWorkflow`` runs each group with its environment entry's settings (``entry_configs``): inside a group
``self.config`` is the entry's and ``current_entry()`` names it; outside one, the run's.  ``DefaultEvalGroupWorkflow``
runs each rollout with its entry's rollout config."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from platoon.config_defs import RolloutConfig
from platoon.eval.config_defs import EvalWorkflowConfig
from platoon.eval.workflow import DefaultEvalGroupWorkflow
from platoon.task import Task
from platoon.train.components import ENVIRONMENT_COLUMN
from platoon.train.config_defs import WorkflowConfig
from platoon.train.workflow import context as context_module
from platoon.train.workflow import group as group_module
from platoon.train.workflow import output as output_module
from platoon.utils.rollout_workload import RolloutWorkload


class _Recording(group_module.GroupRolloutWorkflow):
    """Records what each member sees, and returns no result: the group trains nothing."""

    def __init__(self, config: WorkflowConfig, **kwargs) -> None:
        super().__init__(
            rollout_fn=None,
            get_task_fn=None,
            config=config,
            model_info=SimpleNamespace(model_name="m", llm=None),
            stats_scope="entry-settings",
            **kwargs,
        )
        self.seen: list[tuple] = []

    async def arun_episode_single(self, data: dict, rollout_number: int = 0):
        self.seen.append(
            (
                context_module.current_entry(),
                self.config.group_size,
                self._filter_errors(),
                self._subagent_datum_sampler(),
            )
        )
        await asyncio.sleep(0)
        return output_module.MemberOutcome(result=None, workload=RolloutWorkload(), observed=False)


def test_each_group_reads_its_entry_s_settings_and_outside_a_group_the_run_s():
    run = WorkflowConfig(group_size=2, filter_errors=True)
    workflow = _Recording(run, filter_errors=True)
    workflow.entry_configs = {
        "quick": run,
        "slow": WorkflowConfig(group_size=3, filter_errors=False, subagent_datum_keep_probability=0.5),
    }

    async def run_groups():
        return await asyncio.gather(
            workflow.arun_episode({"task_id": "quick::a", ENVIRONMENT_COLUMN: "quick"}),
            workflow.arun_episode({"task_id": "slow::b", ENVIRONMENT_COLUMN: "slow"}),
        )

    quick, slow = asyncio.run(run_groups())

    seen = {}
    for entry, group_size, filter_errors, sampler in workflow.seen:
        seen.setdefault(entry, set()).add((group_size, filter_errors, sampler is None))
    assert seen == {"quick": {(2, True, True)}, "slow": {(3, False, False)}}
    assert len(workflow.seen) == 5  # two members of a quick group, three of a slow one
    assert (quick.entry, slow.entry) == ("quick", "slow")
    assert workflow.config is run and context_module.current_entry() is None  # outside a group: the run's


def test_an_entry_s_subagent_sampler_is_built_once_and_the_run_s_is_reused():
    run = WorkflowConfig(group_size=1, subagent_datum_keep_probability=0.25, subagent_datum_sampling_seed=7)
    workflow = _Recording(run)
    workflow.entry_configs = {
        "same": run,
        "other": WorkflowConfig(group_size=1, subagent_datum_keep_probability=0.5, subagent_datum_sampling_seed=7),
    }

    async def run_groups():
        for entry in ("same", "other", "other"):
            await workflow.arun_episode({"task_id": f"{entry}::t", ENVIRONMENT_COLUMN: entry})

    asyncio.run(run_groups())

    samplers = [sampler for _, _, _, sampler in workflow.seen]
    assert samplers[0] is workflow.subagent_datum_sampler
    assert samplers[1] is samplers[2] and samplers[1] is not samplers[0]
    assert samplers[1].keep_probability == 0.5


def test_the_run_s_groups_use_the_subagent_sampler_the_workflow_holds_as_a_subclass_replaced_it():
    """A group of the run's own settings takes ``subagent_datum_sampler`` as it is, as before entries had settings:
    one a subclass put there after construction, or None to sample nothing, also with a keep probability of 1."""

    from platoon.utils.subagent_sampling import DeterministicSubagentDatumSampler

    custom = DeterministicSubagentDatumSampler(keep_probability=0.9, seed=1)
    seen = []
    for keep, replacement in ((0.25, custom), (0.25, None), (1.0, custom)):
        workflow = _Recording(WorkflowConfig(group_size=1, subagent_datum_keep_probability=keep))
        workflow.subagent_datum_sampler = replacement
        asyncio.run(workflow.arun_episode({"task_id": "a::t", ENVIRONMENT_COLUMN: "a"}))
        seen.append(workflow.seen[0][3])
    assert seen == [custom, None, custom]


def test_a_workflow_without_entries_runs_every_group_with_its_own_config_and_names_the_row_s_entry():
    run = WorkflowConfig(group_size=1)
    workflow = _Recording(run)

    output = asyncio.run(workflow.arun_episode({"task_id": "a::t", ENVIRONMENT_COLUMN: "a"}))

    assert workflow.seen[0][:2] == ("a", 1) and output.entry == "a"
    assert group_module.GroupRolloutWorkflow.applies_entry_settings is True


SEEN: list[tuple[str, int | None, int | None, int | None]] = []


async def _rollout(task: Task, config: RolloutConfig) -> dict:
    SEEN.append((str(task.id), config.max_steps, config.timeout, task.max_steps))
    return {
        "id": f"collection-{task.id}",
        "trajectories": {
            "root": {
                "id": "root",
                "task": {"id": task.id, "goal": "", "max_steps": task.max_steps},
                "steps": [{"misc": {}}],
                "reward": 1.0,
                "misc": {},
            }
        },
    }


def test_each_evaluation_rollout_runs_with_its_entry_s_rollout_config(tmp_path):
    SEEN.clear()
    workflow = DefaultEvalGroupWorkflow(
        rollout_fn=_rollout,
        get_task_fn=lambda task_id: Task(id=task_id, goal=""),
        config=EvalWorkflowConfig(rollout_config=RolloutConfig(max_steps=5, timeout=60)),
        model_name="m",
    )
    workflow.entry_rollout_configs = {
        "quick": RolloutConfig(max_steps=30, timeout=60),
        "slow": RolloutConfig(max_steps=None, timeout=None),
    }

    async def run():
        for entry in ("quick", "slow", "unknown"):
            row = {"task_id": f"{entry}::t", ENVIRONMENT_COLUMN: entry}
            await workflow.arun_rollout(row, 0, str(tmp_path / entry))

    asyncio.run(run())

    assert SEEN == [("quick::t", 30, 60, 30), ("slow::t", None, None, None), ("unknown::t", 5, 60, 5)]
