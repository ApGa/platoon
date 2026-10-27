"""When the trainer evaluates: every ``eval.every`` training steps, ``0`` for never (``eval.strategy: epoch`` counts it
in passes over the training tasks), the starting policy at step 0 only with ``eval.at_start``, the step a start resumes
at unless an earlier start completed its evaluation (the run's ``evaluations.jsonl``), and an evaluation due at the last
step before the run ends.  ``test_eval_resume.py`` runs whole starts of a run, crashes and restarts among them.

``checkpoint`` takes the same two settings; ``test_checkpoint_cadence.py`` covers what a step saves.

    HF_HUB_OFFLINE=1 python -m pytest tests/train/evaluation -q
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
from collections.abc import Collection
from pathlib import Path

import pytest
import tinker
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from fakes.config import make_config  # noqa: E402
from fakes.loop import loop_state, loop_trainer  # noqa: E402
from fakes.trainer import StepLog, TrainingClient  # noqa: E402

from platoon.train.config_defs import CheckpointConfig, EvalConfig  # noqa: E402
from platoon.train.evaluation.record import completed_evaluation_steps  # noqa: E402
from platoon.train.trainer import signals as signals_module  # noqa: E402
from platoon.utils.jsonl import read_jsonl  # noqa: E402


def _config(eval_config: EvalConfig, checkpoint_config: CheckpointConfig, **train):
    """A run's config with these evaluation and checkpoint sections, and ``train``'s settings."""

    return dataclasses.replace(make_config(train=train), eval=eval_config, checkpoint=checkpoint_config)


def _shared_state(eval_config: EvalConfig, checkpoint_config: CheckpointConfig, *, batches_per_epoch: int = 4):
    """What a start's loops share, under a config with these sections, over an epoch of ``batches_per_epoch``."""

    return loop_state(loop_trainer(_config(eval_config, checkpoint_config)), batches_per_epoch=batches_per_epoch)


@pytest.mark.parametrize(("every", "steps"), [(0, None), (1, 1), (3, 3)])
def test_every_counts_training_steps(every, steps):
    state = _shared_state(EvalConfig(every=every), CheckpointConfig(every=every))

    assert state.eval_every == steps
    assert state.save_every == steps


@pytest.mark.parametrize(
    ("strategy", "every", "steps"), [(None, 3, 3), ("step", 3, 3), ("epoch", 2, 8), ("epoch", 0, None)]
)
def test_a_strategy_sets_the_unit_of_every(strategy, every, steps):
    """``strategy: epoch`` counts ``every`` in passes over the training tasks (4 steps here), and ``step``, as a
    config that leaves it out does, in steps."""

    state = _shared_state(EvalConfig(strategy=strategy, every=every), CheckpointConfig(strategy=strategy, every=every))

    assert state.eval_every == steps
    assert state.save_every == steps


@pytest.mark.parametrize(
    ("eval_config", "evaluates"),
    [
        (EvalConfig(), False),
        (EvalConfig(every=0), False),
        (EvalConfig(every=5), True),
        (EvalConfig(strategy="step"), True),  # every left out: 1
        (EvalConfig(strategy="epoch", every=1), True),
        (EvalConfig(strategy="epoch", every=0), False),
    ],
)
def test_only_an_eval_setting_that_triggers_starts_the_eval_loop(eval_config, evaluates):
    config = _config(eval_config, CheckpointConfig())
    trainer = loop_trainer(config, eval_dataset=object())
    state = loop_state(trainer)

    assert trainer._evaluates_during_training(state) is evaluates
    without = loop_trainer(config)  # no eval split: nothing to evaluate on
    assert without._evaluates_during_training(loop_state(without)) is False


# One group a step, of one rollout, under the cispo loss.
_TRAIN = {
    "batch_size": 1,
    "num_minibatches": 1,
    "num_microbatches": 1,
    "max_concurrent_rollouts": 1,
    "workflow_config": {"group_size": 1, "filter_zero_advantage_datums": False},
    "loss_fn": "cispo",
    "loss_fn_config": {},
    "optimizer": {"learning_rate": 1e-6},
}


def _datum(advantage: float = 0.0) -> tinker.Datum:
    """A datum whose tokens all have ``advantage``: with none, a step consumes it without an optimizer update."""

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            key: tinker.TensorData.from_torch(torch.tensor(values))
            for key, values in {
                "advantages": [advantage, advantage],
                "mask": [1.0, 1.0],
                "target_tokens": [2, 3],
                "logprobs": [-0.1, -0.2],
                "checkpoint_version": [0],
            }.items()
        },
    )


def _feed_training(trainer, feed) -> None:
    """The training steps' groups come from ``feed(shared_state, results)``, in place of the dispatcher's groups."""

    def dispatch(task_group, shared_state, start_batch, end_batch, workflow, results):
        task_group.create_task(feed(shared_state, results))
        return None

    trainer._dispatch = dispatch


class _EvalSplit:
    """An eval dataloader of ``size`` one-task batches."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def get_batch(self, index: int) -> list[dict]:
        return [{"task_id": f"eval:{index}"}]


def _train_and_evaluate(
    *,
    steps: int,
    eval_every: int,
    start: int = 0,
    paced: bool = False,
    at_start: bool = False,
    completed: Collection[int] = (),
    record: Path | None = None,
) -> tuple[list[int], StepLog]:
    """Run the training loop from step ``start`` to ``steps`` with zero-advantage steps beside the evaluation loop, as
    ``train`` runs them, with fake rollouts; return the training steps current when tasks were evaluated and what was
    logged (a ``StepLog``).  ``paced`` holds each step's rollouts back until the evaluation loop has handled the
    step before, so that no evaluation due is overtaken (a real step takes longer than its evaluation's start).
    ``at_start`` is ``eval.at_start``; ``completed`` the steps whose evaluation an earlier start of the run completed,
    as its ``evaluations.jsonl`` lists them, which ``train`` reads when it resumes; ``record`` the file each completed
    evaluation is appended to."""

    config = _config(
        EvalConfig(every=eval_every, at_start=at_start, max_concurrent_rollouts=1), CheckpointConfig(every=0), **_TRAIN
    )
    trainer = loop_trainer(config, eval_dataset=object())
    evaluated: list[int] = []

    async def train_groups(shared_state, results):  # every step's group is ready at once, with no training signal
        for _ in range(steps - start):
            while paced and shared_state.eval_handled_step != shared_state.train_step:
                await asyncio.sleep(0)
            results.put_nowait([_datum()])

    _feed_training(trainer, train_groups)

    async def worker(shared_state, workflow, queue, results, tracker=None, rows=None):
        while True:  # an evaluation worker
            await queue.get()
            evaluated.append(shared_state.train_step)
            results.put_nowait(None)

    trainer._rollout_workflow_worker_loop = worker

    async def run():
        state = loop_state(trainer, start_batch=start, batches_per_epoch=4)
        state.start_step = start
        state.start_step_evaluated = start in completed
        state.evaluations_path = None if record is None else str(record)
        state.eval_dataloader = _EvalSplit(2)
        state.sampling_client_updated_event.set()  # as at launch
        state.eval_loop_running = trainer._evaluates_during_training(state)
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(trainer._train_loop(group, state, start, steps, object()))
                group.create_task(trainer._eval_loop(state, object()))
        except* signals_module.TerminateTrainLoop:
            pass
        return state

    state = asyncio.run(run())
    assert state.train_step == steps
    return evaluated, state.stats_logger


def _evaluations(logged: StepLog) -> list[tuple[int, int]]:
    """``(eval/step, eval/num_tasks)`` of each evaluation logged."""

    return [(stats["eval/step"], stats["eval/num_tasks"]) for stats in logged.stats if "eval/num_tasks" in stats]


def test_an_evaluation_due_at_the_last_step_runs_before_the_loops_stop():
    """Training runs ahead of the evaluation of step 0 here (``eval.at_start``); the one due at the last step still
    runs before the loops stop, on the policy after that step."""

    evaluated, logged = _train_and_evaluate(steps=2, eval_every=2, at_start=True)

    assert _evaluations(logged) == [(0, 2), (2, 2)]
    assert evaluated[-2:] == [2, 2]  # both eval tasks of the last evaluation, after the last step
    assert logged[-1] == 2  # logged at the step training reached


def test_an_evaluation_logs_its_time_its_tasks_and_its_step_last_in_that_order():
    """After the evaluation's own metrics, as the tracker exports them: ``eval/time``, ``eval/num_tasks``, then the step
    whose policy it evaluated (``eval/step``), the order a reader of the logged row sees."""

    _, logged = _train_and_evaluate(steps=2, eval_every=2)

    [evaluation] = [stats for stats in logged.stats if "eval/step" in stats]
    assert list(evaluation)[-3:] == ["eval/time", "eval/num_tasks", "eval/step"]


@pytest.mark.parametrize(("at_start", "steps"), [(False, [2, 4]), (True, [0, 2, 4])])
def test_a_fresh_run_evaluates_every_due_step_and_its_starting_policy_only_with_at_start(at_start, steps):
    evaluated, logged = _train_and_evaluate(steps=4, eval_every=2, paced=True, at_start=at_start)

    assert _evaluations(logged) == [(step, 2) for step in steps]
    assert evaluated == [step for step in steps for _task in range(2)]


@pytest.mark.parametrize("at_start", [False, True])
@pytest.mark.parametrize(
    ("start", "completed", "steps"),
    [
        (3, (), [4]),  # between two multiples: the next one
        (3, (2,), [4]),
        (2, (), [2, 4]),  # at a multiple whose evaluation no earlier start completed: a crash cut it short
        (2, (2,), [4]),  # at one that an earlier start completed, which does not run twice
        (2, (2, 4), [4]),  # a later step's record says nothing of the policy this start trains to that step
    ],
)
def test_a_resumed_run_evaluates_at_the_multiples_from_where_it_resumed(at_start, start, completed, steps):
    """A start that resumes at step 3 evaluates at 4 only.  One that resumes at step 2 evaluates at 2 again only when
    the run's evaluations.jsonl lists no completed evaluation of it, and at 4.  None evaluates step 0, with
    ``eval.at_start`` or without."""

    _, logged = _train_and_evaluate(
        steps=4, eval_every=2, start=start, paced=True, at_start=at_start, completed=completed
    )

    assert [step for step, _ in _evaluations(logged)] == steps


@pytest.mark.parametrize(("at_start", "completed", "steps"), [(True, (), [0, 2]), (True, (0,), [2]), (False, (), [2])])
def test_a_start_that_resumes_at_step_0_evaluates_the_starting_policy_unless_that_completed(at_start, completed, steps):
    """A crash before the first step's checkpoint resumes the run at step 0, from the checkpoint its start saved: with
    ``eval.at_start`` that start evaluates the starting policy again unless an earlier start completed it."""

    _, logged = _train_and_evaluate(steps=2, eval_every=2, paced=True, at_start=at_start, completed=completed)

    assert [step for step, _ in _evaluations(logged)] == steps


@pytest.mark.parametrize(("completed", "steps"), [((), [4]), ((4,), [])])
def test_a_start_that_resumes_at_the_last_step_runs_its_evaluation_only_if_a_crash_cut_it_short(completed, steps):
    """A crash during the evaluation due at the last step, after that step's checkpoint, resumes the run there with no
    step left to train: the start runs that evaluation before it stops, unless an earlier start completed it."""

    _, logged = _train_and_evaluate(steps=4, eval_every=2, start=4, completed=completed)

    assert [step for step, _ in _evaluations(logged)] == steps


@pytest.mark.parametrize(("at_start", "evaluations"), [(False, []), (True, [(0, 2)])])
def test_a_last_step_with_no_evaluation_due_stops_at_once(at_start, evaluations):
    evaluated, logged = _train_and_evaluate(steps=3, eval_every=5, at_start=at_start)

    assert _evaluations(logged) == evaluations  # the starting policy's, with eval.at_start, and no other
    assert len(evaluated) == 2 * len(evaluations)


def test_each_evaluation_that_completes_is_recorded_for_later_starts(tmp_path):
    record = tmp_path / "evaluations.jsonl"

    _, logged = _train_and_evaluate(steps=4, eval_every=2, paced=True, at_start=True, record=record)

    assert completed_evaluation_steps(tmp_path) == {0, 2, 4}
    lines = read_jsonl(record)
    assert [(line["step"], line["tasks"]) for line in lines] == [(0, 2), (2, 2), (4, 2)]
    # Each line says the step training had reached when the evaluation was logged, the step its metrics went to.
    assert [line["logged_at"] for line in lines] == [step for step, stats in logged.rows() if "eval/step" in stats]


def test_a_shutdown_request_ends_the_wait_for_an_evaluation():
    """A first SIGINT or SIGTERM during the last evaluation stops the run as before, instead of waiting for an
    evaluation whose workers no longer take tasks."""

    trainer = loop_trainer(_config(EvalConfig(every=1), CheckpointConfig()))
    state = loop_state(trainer)
    state.eval_loop_running, state.train_step = True, 3  # the evaluation loop has not handled step 3 yet

    async def run():
        waiting = asyncio.create_task(trainer._wait_for_last_step_evaluation(state))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        state.shutdown_event.set()
        await asyncio.wait_for(waiting, timeout=5)

    asyncio.run(run())


class _FirstUpdate(Exception):
    """The fake training client's first optimizer step, where the test stops."""


def _first_update(
    *, at_start: bool, completed: Collection[int] = (), start: int = 0
) -> list[tuple[int | None, list[int]]]:
    """Train from step ``start`` (with no update taken before it) beside a slow evaluation until the first optimizer
    step, where the test stops; return the step the evaluation loop had handled by then and the training steps current
    when tasks were evaluated."""

    evaluated: list[int] = []
    first_update: list[tuple[int | None, list[int]]] = []

    def stop(_count):  # the training client's first optimizer step
        first_update.append((state.eval_handled_step, list(evaluated)))
        raise _FirstUpdate()

    config = _config(
        EvalConfig(every=2, at_start=at_start, max_concurrent_rollouts=1), CheckpointConfig(every=0), **_TRAIN
    )
    trainer = loop_trainer(config, eval_dataset=object(), training_client=TrainingClient(on_optim_step=stop))
    state = loop_state(trainer, start_batch=start, batches_per_epoch=4)
    state.start_step = start

    async def train_groups(shared_state, results):  # the first step's group is ready at once, with a training signal
        results.put_nowait([_datum(advantage=1.0)])

    _feed_training(trainer, train_groups)

    async def worker(shared_state, workflow, queue, results, tracker=None, rows=None):
        while True:  # an evaluation worker, slower than training's first step
            await queue.get()
            await asyncio.sleep(0.02)
            evaluated.append(shared_state.train_step)
            results.put_nowait(None)

    trainer._rollout_workflow_worker_loop = worker

    async def run():
        state.start_step_evaluated = start in completed
        state.eval_dataloader = _EvalSplit(2)
        state.sampling_client_updated_event.set()  # as at launch
        state.eval_loop_running = trainer._evaluates_during_training(state)
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(trainer._train_loop(group, state, start, 4, object()))
                group.create_task(trainer._eval_loop(state, object()))
        except* _FirstUpdate:
            pass

    asyncio.run(run())
    return first_update


def test_the_first_update_waits_for_the_evaluation_of_the_starting_policy():
    """An evaluation's rollouts sample the policy that is current when they run, so with ``eval.at_start`` a fresh run
    applies its first update only once the evaluation of its starting policy has ended: every task of it ran on step
    0's policy, although the first step's rollouts were ready at once.  A start that resumes at step 0, when no earlier
    start completed that evaluation, waits for it as well."""

    first_update = _first_update(at_start=True)

    assert first_update == [(0, [0, 0])]  # step 0's evaluation was over, both its tasks on the starting policy


@pytest.mark.parametrize(("at_start", "completed"), [(False, ()), (True, (0,))])
def test_the_first_update_waits_for_no_evaluation_of_the_starting_policy_that_does_not_run(at_start, completed):
    """Without ``eval.at_start`` step 0 is not evaluated, and a start that resumes at step 0 after an earlier start
    completed the evaluation of the starting policy does not run it again: the first update waits for neither."""

    [(_handled, evaluated)] = _first_update(at_start=at_start, completed=completed)

    assert evaluated == []  # no evaluation task had run, nor was one due


def test_a_resumed_start_s_first_update_waits_for_no_evaluation_even_before_its_first_update():
    """A start that resumes at step 2 having taken no update yet (its steps so far had nothing to train on), with
    ``eval.at_start``: only a fresh run's first update waits for the evaluation of the policy it starts from, so this
    one is applied while the evaluation due where the start resumes still runs."""

    [(_handled, evaluated)] = _first_update(at_start=True, start=2)

    assert evaluated == []


def test_a_shutdown_request_ends_the_wait_for_the_evaluation_of_the_starting_policy():
    """A first SIGINT or SIGTERM while the first update waits drops the batch, which applied no update, as a signal
    during its rollouts does; without an evaluation loop, or without ``eval.at_start``, there is nothing to wait
    for."""

    trainer = loop_trainer(_config(EvalConfig(every=1, at_start=True), CheckpointConfig()))
    state = loop_state(trainer)

    async def run():
        await asyncio.wait_for(trainer._wait_for_starting_policy_evaluation(state), timeout=5)  # no evaluation loop
        state.eval_loop_running = True  # one that has not handled step 0 yet
        state.config = _config(EvalConfig(every=1), CheckpointConfig())  # without at_start
        await asyncio.wait_for(trainer._wait_for_starting_policy_evaluation(state), timeout=5)  # nothing to wait for
        state.config = _config(EvalConfig(every=1, at_start=True), CheckpointConfig())
        waiting = asyncio.create_task(trainer._wait_for_starting_policy_evaluation(state))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        state.shutdown_event.set()
        with pytest.raises(signals_module.TrainInterrupted):
            await asyncio.wait_for(waiting, timeout=5)

    asyncio.run(run())
