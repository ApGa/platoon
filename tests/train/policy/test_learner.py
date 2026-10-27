"""A policy's learner (``platoon.train.policy.learner``) and the objective it updates the policy under
(``platoon.train.objectives``): a minibatch's update sends each microbatch through the policy's training client under
the objective's loss, steps the policy's optimizer once over them all, and records the objective's metrics of each,
averaged, and the optimizer's.  The training step updates the policy its loops train, under the loss and with the
optimizer that the policy's spec took from the config when the trainer was built: it filters and sends each
microbatch under that loss, without the objective's side channels, awaits each result under the trainer's heartbeat,
and logs that optimizer's learning rate."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest
import tinker
import torch
from fakes.config import make_config
from fakes.loop import loop_trainer
from fakes.trainer import StepLog, make_trainer

from platoon.train.config_defs import AdamParams
from platoon.train.objectives import policy_gradient
from platoon.train.objectives.api import LOSS_NORMALIZATION_TOKENS_KEY
from platoon.train.objectives.policy_gradient import PolicyGradient
from platoon.train.policy.learner import Learner, OptimizerUpdate, PolicyUpdate
from platoon.train.policy.policy import Policy, adam_params
from platoon.train.policy.set import PolicySet
from platoon.train.policy.spec import MAIN, PolicySpec
from platoon.train.step.metrics import RunTotals
from platoon.train.step.microbatch import MicrobatchFill
from platoon.train.trainer import loops
from platoon.train.trainer.run import StepClock
from platoon.train.trainer.state import TrainLoopSharedState
from platoon.utils.stats_tracker import StatsTracker


class _Future:
    def __init__(self, value: Any, error: BaseException | None = None) -> None:
        self.value = value
        self.error = error

    async def result_async(self) -> Any:
        if self.error is not None:
            raise self.error
        return self.value


class _TrainingClient:
    """Records each call, in order, in ``events``; the optimizer step's result raises ``optim_error``, if given."""

    def __init__(self, events: list[Any], optim_error: BaseException | None = None) -> None:
        self.events = events
        self.optim_error = optim_error

    async def forward_backward_async(self, datums, loss_fn, loss_fn_config):
        self.events.append(("forward_backward", datums, loss_fn, loss_fn_config))
        return _Future(SimpleNamespace(loss_fn_outputs=f"result of {datums[0]}"))

    async def optim_step_async(self, params):
        self.events.append(("optim_step", params))
        return _Future(SimpleNamespace(metrics={"grad_norm": 2.0}), self.optim_error)


def _learner(events: list[Any], recorded: dict[str, float], optim_error: BaseException | None = None) -> Learner:
    """A learner of a policy that took 3 optimizer steps, whose calls, waits and metrics go to ``events``, and whose
    recorded metrics go to ``recorded``."""

    info: Any = SimpleNamespace()
    client: Any = _TrainingClient(events, optim_error)
    policy = Policy(PolicySpec(MAIN, model_name="m", renderer_name="r"), info)
    policy.training_client = client
    policy.optimizer_params = adam_params(AdamParams(learning_rate=1e-5))
    policy.optimizer_steps = 3
    loss_fn_config = {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0}

    def training_metrics(**inputs: Any) -> dict[str, float]:
        events.append(("metrics", inputs))
        return {"optim/kl": 1.0 if inputs["datums"] == ["original 0"] else 3.0}

    async def wait(awaitable, *, name, total_timeout):
        events.append(("wait", name, total_timeout))
        return await awaitable

    tracker: Any = SimpleNamespace(scalar=lambda **metrics: recorded.update(metrics))
    return Learner(
        policy,
        objective=PolicyGradient("cispo", loss_fn_config, training_metrics=training_metrics),
        timeouts=SimpleNamespace(forward_backward_timeout_seconds=5.0, optim_step_timeout_seconds=1.0),
        wait=wait,
        tracker=tracker,
    )


async def _send_two_microbatches(learner: Learner) -> PolicyUpdate:
    update = learner.update()
    assert not update.submitted
    for microbatch in (0, 1):
        sent: Any = [f"sent {microbatch}"]
        originals: Any = [f"original {microbatch}"]
        await update.submit(sent, originals, minibatch=0, microbatch=microbatch)
    return update


def test_an_update_sends_each_microbatch_under_the_objective_then_steps_the_policy_s_optimizer_once():
    events: list[Any] = []
    recorded: dict[str, float] = {}
    learner = _learner(events, recorded)
    policy, loss_fn_config = learner.policy, learner.objective.loss_fn_config
    tally = SimpleNamespace(optimizer_steps=0)

    async def run() -> PolicyUpdate:
        update = await _send_two_microbatches(learner)
        await update.apply(tally)
        return update

    update = asyncio.run(run())

    assert update.submitted and OptimizerUpdate is PolicyUpdate
    sent = [event for event in events if event[0] == "forward_backward"]
    assert sent == [("forward_backward", [f"sent {n}"], "cispo", loss_fn_config) for n in (0, 1)]
    assert sent[0][3] is loss_fn_config  # the config's own, as the service is sent it
    # The optimizer step is submitted before any result is awaited, with the policy's optimizer parameters.
    order = ["forward_backward", "forward_backward", "optim_step", "wait", "metrics", "wait", "metrics", "wait"]
    assert [event[0] for event in events] == order and events[2][1] is policy.optimizer_params
    waits = [event[1:] for event in events if event[0] == "wait"]
    deadlines = [("forward_backward (microbatch 0)", 5.0), ("forward_backward (microbatch 1)", 5.0)]
    assert waits == [*deadlines, ("optim_step", 1.0)]
    metrics = [event[1] for event in events if event[0] == "metrics"]
    names = ["datums", "forward_backward_result", "loss_fn", "loss_fn_config"]
    assert [list(inputs) for inputs in metrics] == [names, names]
    assert metrics[1]["datums"] == ["original 1"]
    assert metrics[1]["forward_backward_result"].loss_fn_outputs == "result of sent 1"
    assert recorded == {"optim/kl": 2.0, "optim/grad_norm": 2.0}  # the objective's, averaged, and the optimizer's
    assert tally.optimizer_steps == 1 and policy.optimizer_steps == 4


@pytest.mark.parametrize("error", [RuntimeError("optim_step failed"), asyncio.CancelledError()])
def test_an_optimizer_step_whose_result_never_comes_is_not_counted(error):
    """An optimizer step whose result raises, or is cancelled, is counted neither in the step's tally nor in the
    policy's optimizer steps, and records no optimizer metrics: a checkpoint saved after it records the steps taken."""

    events: list[Any] = []
    recorded: dict[str, float] = {}
    learner = _learner(events, recorded, optim_error=error)
    tally = SimpleNamespace(optimizer_steps=0)

    async def run() -> None:
        update = await _send_two_microbatches(learner)
        with pytest.raises(type(error)):
            await update.apply(tally)

    asyncio.run(run())

    assert [event[0] for event in events][-1] == "wait"  # the optimizer step's result, awaited last
    assert tally.optimizer_steps == 0 and learner.policy.optimizer_steps == 3
    assert recorded == {"optim/kl": 2.0}  # the microbatches' metrics only


class _FailingClient(_TrainingClient):
    """A training client that fails at ``where``: the submission of a forward_backward (``fb submit``) or of the
    optimizer step (``optim submit``), or a forward_backward's result (``fb result``)."""

    def __init__(self, events: list[Any], where: str) -> None:
        super().__init__(events)
        self.where = where

    async def forward_backward_async(self, datums, loss_fn, loss_fn_config):
        if self.where == "fb submit":
            raise RuntimeError("the service is gone")
        future = await super().forward_backward_async(datums, loss_fn, loss_fn_config)
        if self.where == "fb result":
            future.error = RuntimeError("the service is gone")
        return future

    async def optim_step_async(self, params):
        if self.where == "optim submit":
            raise RuntimeError("the service is gone")
        return await super().optim_step_async(params)


@pytest.mark.parametrize(
    ("where", "line"),
    [
        ("fb submit", "forward_backward_async submission failed after"),
        ("optim submit", "optim_step_async submission failed: the service is gone"),
        ("fb result", "forward_backward result_async failed after"),
        ("optim result", "optim_step result_async failed after"),
    ],
)
def test_a_failed_call_to_the_service_is_logged_where_it_failed_and_raised(caplog, where, line):
    events: list[Any] = []
    learner = _learner(events, {}, optim_error=RuntimeError("the service is gone") if where == "optim result" else None)
    if where != "optim result":
        learner.policy.training_client = _FailingClient(events, where)

    async def run() -> None:
        update = await _send_two_microbatches(learner)
        await update.apply(SimpleNamespace(optimizer_steps=0))

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="the service is gone"):
        asyncio.run(run())

    assert [record.getMessage() for record in caplog.records if line in record.getMessage()]


def test_training_metrics_that_fail_are_logged_and_the_update_goes_on(caplog):
    events: list[Any] = []
    recorded: dict[str, float] = {}
    learner = _learner(events, recorded)

    def failing(**inputs: Any) -> dict[str, float]:
        raise ValueError("no logprobs")

    learner.objective = PolicyGradient("cispo", learner.objective.loss_fn_config, training_metrics=failing)
    tally = SimpleNamespace(optimizer_steps=0)

    async def run() -> None:
        update = await _send_two_microbatches(learner)
        await update.apply(tally)

    with caplog.at_level(logging.WARNING):
        asyncio.run(run())

    assert "Failed to compute training metrics for microbatch 0: no logprobs" in caplog.text
    assert tally.optimizer_steps == 1 and recorded == {"optim/grad_norm": 2.0}


def test_the_policy_gradient_objective_leaves_out_the_trainer_s_loss_inputs_and_computes_its_metrics(monkeypatch):
    computed: list[dict[str, Any]] = []
    monkeypatch.setattr(policy_gradient, "compute_training_metrics", lambda **inputs: computed.append(inputs) or {})
    objective = PolicyGradient("ppo", {"clip_low_threshold": 0.8})
    originals: Any = ["datum"]
    result: Any = SimpleNamespace()

    assert objective.metrics(originals, result) == {}
    assert computed == [
        {
            "datums": originals,
            "forward_backward_result": result,
            "loss_fn": "ppo",
            "loss_fn_config": {"clip_low_threshold": 0.8},
        }
    ]
    assert objective.side_channels == {
        "mask",
        "checkpoint_version",
        "traj_depth",
        "traj_start",
        LOSS_NORMALIZATION_TOKENS_KEY,
    }


def _step_context(state: Any, train: dict[str, Any], *, heartbeat: Any = None) -> loops.StepContext:
    """The context of a start's steps, as the training loop builds it for ``state``, by a trainer under ``train:``
    (``fakes.loop``; no batch transforms), whose waits on the service run under ``heartbeat`` when given (in place of
    the trainer's ``_await_with_heartbeat``)."""

    trainer: Any = loop_trainer(make_config(train=train))
    if heartbeat is not None:
        trainer._await_with_heartbeat = heartbeat
    workflow: Any = object()
    return loops.StepContext.build(
        trainer,
        state,
        workflow,
        0,
        1,
        queue=asyncio.Queue(),
        dispatcher=None,
        tasks=[],
        column="environment",
        group_limits={},
    )


def _step_learner(state: Any, train: dict[str, Any]) -> Learner:
    """The learner of a start's steps, as the training loop builds it."""

    return _step_context(state, train).learner


def _step(state: Any, train: dict[str, Any]) -> loops.TrainStep:
    """A start's first step (of one minibatch of one microbatch of one group), as the training loop makes it."""

    return loops.TrainStep(_step_context(state, train), RunTotals())


def _train(filter_zero_advantage_datums: bool = False, **settings: Any) -> dict[str, Any]:
    """``train:`` of one group a step, of one rollout, with ``settings``."""

    workflow = {"group_size": 1, "filter_zero_advantage_datums": filter_zero_advantage_datums}
    return {"batch_size": 1, "num_minibatches": 1, "num_microbatches": 1, "workflow_config": workflow, **settings}


def _state(**given: Any) -> TrainLoopSharedState:
    """What a start's loops share, over the run's policies ``given["policies"]``, at step 0."""

    parts: dict[str, Any] = {
        "config": make_config(),
        "train_dataloader": [],
        "stats_logger": StepLog(),
        "train_tracker": StatsTracker("train"),
    }
    return TrainLoopSharedState(
        shutdown_event=asyncio.Event(),
        eval_dataloader=None,
        sampling_client_step=0,
        clock=StepClock(step=0, updated=asyncio.Event()),
        **{**parts, **given},
    )


def _main(**settings: Any) -> Policy:
    """Main's policy, with a spec as :func:`~platoon.train.policy.spec.policy_specs` makes one: with its optimizer's
    settings, its loss and the loss's config (``settings`` in place of the defaults)."""

    spec_settings: dict[str, Any] = {"optimizer": AdamParams(), "loss_fn": "cispo", "loss_fn_config": {}, **settings}
    info: Any = SimpleNamespace()
    return Policy(PolicySpec(MAIN, model_name="m", renderer_name="r", **spec_settings), info)


def _datum(advantage: float, **inputs: torch.Tensor) -> tinker.Datum:
    """A datum of two action tokens, each with ``advantage``, and the loss inputs ``inputs`` after the usual ones."""

    usual = {
        "target_tokens": torch.tensor([2, 3]),
        "logprobs": torch.tensor([-0.1, -0.2]),
        "advantages": torch.tensor([advantage, advantage]),
        "mask": torch.tensor([1.0, 1.0]),
        "checkpoint_version": torch.tensor([0]),
    }
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={key: tinker.TensorData.from_torch(values) for key, values in {**usual, **inputs}.items()},
    )


def test_the_step_updates_the_leader_policy_under_the_loss_its_spec_took_when_the_trainer_was_built(monkeypatch):
    """Its objective's metrics are ``compute_training_metrics`` as the training loop's module names it at each call,
    where tests patch it."""

    loss_fn_config = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
    policy = _main(loss_fn="ppo", loss_fn_config=loss_fn_config)
    spec = policy.spec

    learner = _step_learner(_state(policies=PolicySet([policy])), _train(loss_fn="cispo", loss_fn_config={}))
    monkeypatch.setattr(loops, "compute_training_metrics", lambda **inputs: {"patched": float(len(inputs))})

    assert learner.policy is policy and policy.spec is spec
    assert learner.objective.loss_fn == "ppo" and learner.objective.loss_fn_config is loss_fn_config
    originals: Any = []
    result: Any = SimpleNamespace()
    assert learner.objective.metrics(originals, result) == {"patched": 4.0}


def test_a_step_on_a_state_a_test_made_as_platoon_0_4_0_did_trains_under_the_run_s_settings():
    """A shared state made with platoon 0.4.0's fields holds a policy of its own, whose spec has neither a loss nor an
    optimizer: it takes the run's.  A state made without any policy (a test's) is updated in its place, under the run's
    loss, ``cispo`` with its default thresholds here, and under ``cispo`` with no config when a test's config names no
    loss (one that is not a run's config)."""

    loss_fn_config = {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0}
    state = TrainLoopSharedState(  # platoon 0.4.0's fields: the policy's state and the loops' step and event
        config=make_config(),
        shutdown_event=asyncio.Event(),
        train_step=0,
        train_dataloader=[],
        eval_dataloader=None,
        sampling_client_step=0,
        sampling_client_updated_event=asyncio.Event(),
        stats_logger=StepLog(),
        train_tracker=StatsTracker("train"),
        model_info=SimpleNamespace(model_name="m", renderer_name="r"),
        optimizer_steps=2,
    )
    train = _train(loss_fn="importance_sampling", loss_fn_config=loss_fn_config, optimizer={"learning_rate": 1e-6})
    context = _step_context(state, train)
    learner, run_train = context.learner, context.trainer.config.train
    assert state.policies is not None and learner.policy is state.policies[MAIN]
    assert learner.objective.loss_fn == "importance_sampling"
    assert learner.objective.loss_fn_config is run_train.loss_fn_config
    assert learner.policy.spec.optimizer is run_train.optimizer

    bare = SimpleNamespace(watchdog=None, train_tracker=SimpleNamespace(), optimizer_steps=0)
    learner = _step_learner(bare, _train())
    assert learner.policy is bare
    assert (learner.objective.loss_fn, learner.objective.loss_fn_config) == ("cispo", loss_fn_config)

    unnamed = SimpleNamespace(watchdog=None, train_tracker=SimpleNamespace(), optimizer_steps=0)
    policy = loops._test_state_policy(unnamed, SimpleNamespace(train=SimpleNamespace()))
    assert (policy.spec.loss_fn, policy.spec.loss_fn_config) == ("cispo", {})


def test_the_trainer_s_optimizer_steps_with_the_settings_its_policy_s_spec_took(tmp_path):
    """As the trainer is built, its policy takes ``train.optimizer``: the optimizer parameters of each start are those,
    also after the config changes."""

    config = make_config(train={"optimizer": {"learning_rate": 1e-5, "grad_clip_norm": 1.0}}, log_path=str(tmp_path))
    trainer, _saves, _logged = make_trainer(config, train_dataset=[])
    config.train.optimizer = AdamParams(learning_rate=0.5)

    async def ready(*, stop):
        return None

    start: Any = SimpleNamespace(batch=0, policy_version=0, resume_info=None)
    startup: Any = SimpleNamespace(ready=ready)
    dataloader: Any = []
    state = asyncio.run(trainer._loop_state(start, asyncio.Event(), dataloader, startup))

    params = state.optimizer_params
    assert isinstance(params, tinker.AdamParams) and (params.learning_rate, params.grad_clip_norm) == (1e-5, 1.0)
    assert params == adam_params(AdamParams(learning_rate=1e-5, grad_clip_norm=1.0))


def test_the_step_s_learner_awaits_each_result_under_the_trainer_s_heartbeat_feeding_the_watchdog():
    """The waiter of the step's learner is the trainer's heartbeat, as the trainer has it at each wait, which logs every
    60 s and feeds the start's watchdog; its deadlines are ``train:``'s and its metrics go to the start's tracker."""

    calls: list[dict[str, Any]] = []

    async def heartbeat(awaitable, **arguments):
        calls.append(arguments)
        return await awaitable

    async def result() -> str:
        return "result"

    async def wait() -> Any:
        return await learner.wait(result(), name="optim_step", total_timeout=2.0)

    watchdog: Any = SimpleNamespace()
    state = _state(policies=PolicySet([_main()]), watchdog=watchdog)
    context = _step_context(state, _train(), heartbeat=heartbeat)
    learner = context.learner

    assert asyncio.run(wait()) == "result"
    assert calls == [{"name": "optim_step", "heartbeat_interval": 60.0, "watchdog": watchdog, "total_timeout": 2.0}]
    assert calls[0]["watchdog"] is watchdog
    assert learner.timeouts is context.trainer.config.train and learner.tracker is state.train_tracker


def test_a_step_filters_and_tests_its_microbatches_under_the_loss_its_policy_trains_under():
    """Zero advantages zero the objective of ``cispo`` with its clip thresholds only, the config's loss here, so that
    its zero-advantage filter would drop a zero datum and its zero test a zero microbatch; the policy's spec adds a
    term (``kl_coef``), under which the step drops neither."""

    state = _state(policies=PolicySet([_main(loss_fn_config={"kl_coef": 0.1})]), train_tracker=StatsTracker("train"))
    step = _step(state, _train(filter_zero_advantage_datums=True, loss_fn="cispo", loss_fn_config={}))

    prepared = step.prepare(MicrobatchFill([_datum(0.0), _datum(1.0)], places=1), 0, 0)
    assert prepared is not None and len(prepared.datums) == 2
    assert step.prepare(MicrobatchFill([_datum(0.0)], places=1), 0, 0) is not None


def test_a_step_sends_its_datums_without_the_objective_s_side_channels_and_with_everything_else():
    """The loss inputs only the trainer reads are left out of the copies sent; the others are sent in their order, the
    routing handle the gateway records among them, and the originals are kept for the objective's metrics."""

    step = _step(_state(policies=PolicySet([_main()])), _train())
    datum = _datum(1.0, routing_handle=torch.tensor([4242], dtype=torch.int64), traj_start=torch.tensor([1.0]))

    prepared = step.prepare(MicrobatchFill([datum], places=1), 0, 0)

    assert prepared is not None and prepared.originals[0] is datum
    (sent,) = prepared.datums
    assert list(sent.loss_fn_inputs) == ["target_tokens", "logprobs", "advantages", "routing_handle"]
    assert sent.loss_fn_inputs["routing_handle"].to_torch().tolist() == [4242]


def test_a_step_logs_the_learning_rate_its_policy_s_optimizer_steps_with():
    """``optim/lr`` is the learning rate of the policy's spec, with which its optimizer steps: ``train.optimizer``'s
    as the trainer was built, also when the config's is replaced after that."""

    policy = _main(optimizer=AdamParams(learning_rate=3e-5))
    state = _state(policies=PolicySet([policy]), train_dataloader=[None])
    step = _step(state, _train(optimizer={"learning_rate": 7e-4}))

    step._log_stats(0.0)

    assert [stats["optim/lr"] for stats in state.stats_logger.stats] == [3e-5]
