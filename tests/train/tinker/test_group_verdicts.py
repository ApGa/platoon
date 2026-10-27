"""Every group's output says what happened, for the per-task statistics: what each member returned
(``members``, with the reward its environment scored) and the workflow's ``verdict`` on what the group trains, or why
it trains nothing, set at each return of ``GroupRolloutWorkflow._run_group``; a workflow that raised settles as
``failed`` in the trainer's records."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import tinker
import torch
from tinker import TensorData

from platoon.config_defs import RolloutConfig
from platoon.task import Task
from platoon.train.config_defs import WorkflowConfig
from platoon.train.dispatch.dispatcher import POSITION_COLUMN, SEQ_COLUMN
from platoon.train.task_records import TaskRecorder
from platoon.train.tinker.rollout_metrics import RolloutTimeline, timeline_scope
from platoon.train.workers import run as worker_run
from platoon.train.workers.protocol import PROTOCOL_VERSION, ConversionPolicy, ModelBinding, RolloutWorkerRequest
from platoon.train.workflow import group as group_module
from platoon.train.workflow import output as output_module
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats


def _datum(reward: float, *, mask: tuple[float, ...] = (0.0, 1.0, 1.0), version: int = 3) -> tinker.Datum:
    """A datum whose action tokens carry the rollout's reward, as the converter loads them before centering."""

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor([2, 3, 4])),
            "logprobs": TensorData.from_torch(torch.tensor([0.0, -0.1, -0.2])),
            "advantages": TensorData.from_torch(torch.tensor([0.0, reward, reward])),
            "mask": TensorData.from_torch(torch.tensor(list(mask))),
            "checkpoint_version": TensorData.from_torch(torch.tensor([version])),
        },
    )


def _outcome(
    reward: float | None,
    *,
    valid: bool = True,
    datums: bool = True,
    mask: tuple[float, ...] = (0.0, 1.0, 1.0),
    key: str | None = None,
    success: float | None = None,
    status: str | None = "scored",
) -> output_module._SingleRolloutOutcome:
    """One member: ``reward`` None for one that returned nothing (a dead worker); ``valid`` False for a root that
    ended untrainable; ``success`` the reward its environment scored, before any reward processor."""

    if reward is None:
        return output_module._SingleRolloutOutcome(
            result=None, workload=RolloutWorkload(), observed=False, disposition="worker_crashed"
        )
    result = TrajectoryCollectionResult(
        datums=[_datum(reward, mask=mask)] if datums and valid else [],
        task_reward=reward,
        trajectory_stats=[
            TrajectoryStats(
                trajectory_id="root",
                reward=reward,
                num_steps=4,
                num_input_tokens=3,
                num_output_tokens=7,
                num_datums=1,
                is_root=True,
            )
        ],
        root_rewards_dict={},
        task_reward_valid=valid,
    )
    return output_module._SingleRolloutOutcome(
        result=result,
        workload=RolloutWorkload(
            trajectories=1,
            postmerge_datums=1,
            policy_eligible_datums=int(datums and valid),
            post_sampling_datums=int(datums and valid),
        ),
        observed=True,
        outcome_status=status if valid else "error",
        reward_key=key,
        success_reward=success,
        seconds=12.5,
    )


class _Workflow(group_module.GroupRolloutWorkflow):
    def __init__(self, outcomes: list[output_module._SingleRolloutOutcome], **config) -> None:
        super().__init__(
            rollout_fn=None,
            get_task_fn=None,
            config=WorkflowConfig(group_size=len(outcomes), filter_zero_advantage_datums=False, **config),
            model_info=SimpleNamespace(model_name="m", base_url="u", api_key="k", llm=None),
            stats_scope="test_group_verdicts",
        )
        self.outcomes = outcomes

    async def arun_episode_single(self, data: dict, rollout_number: int = 0):
        return self.outcomes[rollout_number]


def _run(outcomes: list[output_module._SingleRolloutOutcome], **config) -> output_module._TaskRolloutOutput:
    return asyncio.run(_Workflow(outcomes, **config).arun_episode({"task_id": "t"}))


# Each return of _run_group, with members that take it there.
VERDICT_SITES = {
    "reward_key_mismatch": ([_outcome(1.0, key="reward/a"), _outcome(0.0, key="reward/b")], {}),
    "no_results": ([_outcome(None), _outcome(None)], {}),
    "below_min": (
        [_outcome(1.0), _outcome(0.0), _outcome(None), _outcome(1.0, valid=False)],
        {"min_successful_group_size": 3},
    ),
    "no_completed_root": ([_outcome(1.0, valid=False), _outcome(0.0, valid=False)], {}),
    "no_datums": ([_outcome(1.0, datums=False), _outcome(0.0, datums=False)], {}),
    # The second no_datums return: datums whose action tokens are all masked, which the error filter drops.
    "no_datums_masked": ([_outcome(1.0, mask=(0.0, 0.0, 0.0)), _outcome(0.0, mask=(0.0, 0.0, 0.0))], {}),
    "uniform": ([_outcome(1.0), _outcome(1.0), _outcome(1.0)], {}),
    "trained": ([_outcome(1.0), _outcome(0.0), _outcome(1.0), _outcome(None)], {}),
}


@pytest.mark.parametrize("site", list(VERDICT_SITES))
def test_each_return_of_a_group_gives_its_verdict_and_every_member(site):
    outcomes, config = VERDICT_SITES[site]

    output = _run(outcomes, **config)

    assert output.verdict == site.removesuffix("_masked")
    assert (len(output) > 0) is (site in ("uniform", "trained"))
    assert len(output.members) == len(outcomes)  # every member, a dead worker's included
    for member, outcome in zip(output.members, outcomes, strict=True):
        assert member.valid is bool(outcome.result is not None and outcome.result.task_reward_valid)
        assert member.reward == (outcome.result.task_reward if outcome.result is not None else None)
        assert member.disposition == outcome.disposition


def test_a_workflow_that_raised_settles_as_failed(tmp_path):
    config = SimpleNamespace(
        environments=[SimpleNamespace(label="quick", success_threshold=1.0)],
        train=SimpleNamespace(workflow_config=WorkflowConfig(group_size=4), max_staleness=None, task_sampler=None),
        eval=SimpleNamespace(workflow_config=None),
    )
    recorder = TaskRecorder(run_dir=str(tmp_path), config=config, tasks=[{"task_id": "quick::a"}], start=1)
    row = {"task_id": "quick::a", SEQ_COLUMN: 0, POSITION_COLUMN: 0}  # as the dispatcher stamps the rows it draws
    recorder.draw(row, seq=0, position=0, version=0)

    record = recorder.settle(row, entry="quick", label="quick", output=None)

    assert record is not None and (record.verdict, record.members, record.group_size) == ("failed", (), 4)
    assert (record.start, record.seq, record.position) == (1, 0, 0)  # the draw the dispatcher stamped on the row
    assert recorder.store.task("quick::a").groups == 1
    # A row the recorder did not draw (an evaluation's) records nothing.
    assert recorder.settle({"task_id": "quick::a"}, entry="quick", label="quick", output=None) is None
    recorder.close()


def test_a_member_passes_on_the_reward_its_environment_scored_not_the_one_it_trained():
    """A reward processor that scales every reward by 0.98 trains 0.98 for a correct rollout: the member keeps the
    1.0 its environment scored and passes at the default threshold of 1.0."""

    scaled = output_module._member(_outcome(0.98, success=1.0))
    assert (scaled.reward, scaled.scored, scaled.passes(1.0)) == (0.98, 1.0, True)
    # A rollout that records no scored reward (a legacy one without a success flag) passes on its trained reward.
    assert output_module._member(_outcome(1.0)).scored == 1.0
    # An outcome that was not scored is censored, and so is a member without a result.
    assert output_module._member(_outcome(1.0, valid=False)).scored is None
    crashed = output_module._member(_outcome(None))
    assert (crashed.valid, crashed.scored, crashed.outcome, crashed.passes(0.0)) == (False, None, "none", False)
    # What the member's root and datums record.
    assert (scaled.steps, scaled.output_tokens, scaled.seconds, scaled.versions) == (4, 7, 12.5, {"main": (3, 3)})


def scaled_reward_processor(trajectory: dict) -> tuple[float, dict]:
    """A shaping reward processor: every reward, scaled by 0.98."""

    return 0.98 * trajectory["reward"], {"reward/scaled": 0.98 * trajectory["reward"]}


class _Session:
    """Stands in for the worker's model proxy session: no model calls."""

    def __init__(self) -> None:
        self.interactions: dict = {}
        self._scope = timeline_scope(RolloutTimeline())
        self.timeline = None

    async def __aenter__(self):
        self.timeline = self._scope.__enter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._scope.__exit__(exc_type, exc, tb)


def test_a_worker_s_member_keeps_the_scored_reward_through_a_reward_processor(tmp_path, monkeypatch):
    """The worker trains the reward processor's 0.98 and reports the 1.0 the trace scored, so the member passes."""

    root = {
        "reward": 1.0,
        "steps": [{"misc": {}}, {"misc": {"reward_misc": {"reward/success": 1.0}}}],
        "misc": {"outcome": {"status": "scored"}},
    }

    async def rollout_fn(_task, _config):
        return {"trajectories": {"root": root}}

    def convert(**kwargs):  # the conversion trains the processor's reward
        reward, _misc = kwargs["reward_processor"](root)
        return TrajectoryCollectionResult(
            datums=[_datum(reward)],
            task_reward=reward,
            trajectory_stats=[
                TrajectoryStats(
                    trajectory_id="root",
                    reward=reward,
                    num_steps=2,
                    num_input_tokens=3,
                    num_output_tokens=7,
                    num_datums=1,
                    is_root=True,
                )
            ],
            root_rewards_dict={},
        )

    monkeypatch.setattr(worker_run, "TinkerLLMProxySession", _Session)
    monkeypatch.setattr(worker_run, "get_train_data_for_trajectory_collection", convert)
    request = RolloutWorkerRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="task-r0",
        task_id="task",
        rollout_number=0,
        task=Task(goal="g", id="task"),
        rollout_config=asdict(RolloutConfig(output_dir=str(tmp_path), timeout=60)),
        rollout_fn_ref="platoon.harness.rollout:harness_rollout",
        model=ModelBinding(
            binding_path=str(tmp_path / "binding.pkl"),
            bootstrap_blob=b"",
            follow_poll_seconds=5.0,
            model_name="m",
            renderer_name="r",
            checkpoint_version=3,
        ),
        conversion=ConversionPolicy(
            reward_processor_is_default=False, reward_processor_ref=f"{__name__}:scaled_reward_processor"
        ),
        response_path=str(tmp_path / "response.pkl"),
        ready_path=str(tmp_path / "ready"),
    )

    response = asyncio.run(worker_run.run_rollout_request(request, None, rollout_fn=rollout_fn))

    assert response.status == "completed"
    assert (response.result.task_reward, response.success_reward) == (pytest.approx(0.98), 1.0)
    member = output_module._member(_Workflow([_outcome(0.0)])._outcome_from_response(response, "completed"))
    assert (member.reward, member.scored, member.passes(1.0)) == (pytest.approx(0.98), 1.0, True)


def test_the_scored_reward_is_the_trace_s_reward_success_else_a_legacy_success_flag():
    def read(reward_misc):
        return worker_run._success_reward(
            {"trajectories": {"root": {"steps": [{"misc": {}}, {"misc": {"reward_misc": reward_misc}}]}}}
        )

    assert read({"reward/success": 1.0, "reward/total": 0.98}) == 1.0
    assert read({"success": True}) == 1.0  # a legacy rollout's own flag
    assert read({"reward/total": 0.98}) is None
    assert read({"reward/success": "not a number"}) is None
    assert worker_run._success_reward(None) is None


def _root(*, traced: str | None, reward_misc: dict | None, steps: int = 2) -> dict:
    """A root trajectory: traced (its outcome recorded, with that status) or not, its last step's reward_misc."""

    misc = {"outcome": {"status": traced}} if traced is not None else {}
    trajectory_steps = [{"misc": {}} for _ in range(steps)]
    if reward_misc is not None and trajectory_steps:
        trajectory_steps[-1]["misc"]["reward_misc"] = reward_misc
    return {"reward": 0.0, "steps": trajectory_steps, "misc": misc}


@pytest.mark.parametrize(
    ("root", "processed", "threshold"),
    [
        pytest.param(_root(traced="scored", reward_misc={"reward/success": 1.0}), 0.98, 1.0, id="traced-shaped"),
        pytest.param(_root(traced="scored", reward_misc={"reward/success": 0.6}), 0.6, 0.5, id="traced-partial"),
        pytest.param(_root(traced="scored", reward_misc=None, steps=0), 1.0, 1.0, id="traced-without-steps"),
        pytest.param(_root(traced=None, reward_misc={"success": True}), 0.3, 0.5, id="legacy-boolean-flag"),
        pytest.param(_root(traced=None, reward_misc={}), 1.0, 1.0, id="legacy-without-flag"),
        pytest.param(_root(traced=None, reward_misc={"success": 0.7}), 0.7, 0.5, id="legacy-fractional-flag"),
        pytest.param(_root(traced=None, reward_misc={"reward/success": 1.0}), 1.0, 2.0, id="legacy-flag-above-1"),
    ],
)
def test_training_judges_a_pass_as_platoon_eval_judges_success(root, processed, threshold):
    """A member passes exactly when platoon eval's default success function says the rollout succeeded: traced
    rollouts on their scored reward against the threshold, a legacy rollout's own flag at 1, and a legacy rollout
    without one on its processed reward."""

    from platoon.eval.workflow import _default_success_fn_factory
    from platoon.utils.trajectory_status import outcome_status, scored_success, trajectory_was_interrupted

    evaluated = _default_success_fn_factory(threshold)({"trajectories": {"root": root}}, processed, {})
    result = SimpleNamespace(
        task_reward=processed, task_reward_valid=not trajectory_was_interrupted(root), trajectory_stats=[], datums=[]
    )
    outcome = SimpleNamespace(
        result=result,
        outcome_status=outcome_status(root),
        success_reward=scored_success(root),
        disposition="completed",
        reward_key=None,
        seconds=1.0,
    )
    assert output_module._member(outcome).passes(threshold) == evaluated
