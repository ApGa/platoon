"""When the trainer evaluates: every ``eval.every`` training steps, ``0`` for never (the deprecated ``eval.strategy``
keeps its old units), the starting policy at step 0 only with ``eval.at_start``, the step a start resumes at unless an
earlier start completed its evaluation (the run's ``evaluations.jsonl``), and an evaluation due at the last step before
the run ends.  ``test_eval_resume.py`` runs whole starts of a run, crashes and restarts among them.

``checkpoint`` takes the same two settings; ``test_checkpoint_cadence.py`` covers what a step saves.

    HF_HUB_OFFLINE=1 python -m pytest tests/train/tinker -q
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Collection
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import tinker
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from platoon.train.tinker import rl as rl_module  # noqa: E402
from platoon.train.tinker.config_defs import CheckpointConfig, EvalConfig  # noqa: E402
from platoon.train.tinker.evaluation_record import completed_evaluation_steps  # noqa: E402
from platoon.utils.jsonl import read_jsonl  # noqa: E402
from platoon.utils.stats_tracker import StatsTracker  # noqa: E402


def _shared_state(eval_config: EvalConfig, checkpoint_config: CheckpointConfig, *, batches_per_epoch: int = 4):
    return rl_module.TrainLoopSharedState(
        config=SimpleNamespace(eval=eval_config, checkpoint=checkpoint_config),
        shutdown_event=asyncio.Event(),
        train_step=0,
        train_dataloader=[None] * batches_per_epoch,
        eval_dataloader=None,
        optimizer_params=None,
        service_client=None,
        training_client=None,
        sampling_client=None,
        sampling_client_step=0,
        sampling_client_updated_event=asyncio.Event(),
        model_info=None,
        stats_logger=None,
        train_tracker=None,
    )


@pytest.mark.parametrize(("every", "steps"), [(0, None), (1, 1), (3, 3)])
def test_every_counts_training_steps(every, steps):
    state = _shared_state(EvalConfig(every=every), CheckpointConfig(every=every))

    assert state.eval_every == steps
    assert state.save_every == steps


@pytest.mark.parametrize(("strategy", "every", "steps"), [("none", 1, None), ("step", 3, 3), ("epoch", 2, 8)])
def test_a_deprecated_strategy_keeps_its_units(strategy, every, steps):
    """Until platoon 0.3.0 removes ``strategy``, a config that sets it counts ``every`` as it did: ``epoch`` in passes
    over the training tasks (4 steps here), ``step`` in steps, and ``none`` never."""

    state = _shared_state(EvalConfig(strategy=strategy, every=every), CheckpointConfig(strategy=strategy, every=every))

    assert state.eval_every == steps
    assert state.save_every == steps


@pytest.mark.parametrize(
    ("eval_config", "evaluates"),
    [
        (EvalConfig(), False),
        (EvalConfig(every=0), False),
        (EvalConfig(every=5), True),
        (EvalConfig(strategy="none", every=5), False),
        (EvalConfig(strategy="epoch", every=1), True),
    ],
)
def test_only_an_eval_setting_that_triggers_starts_the_eval_loop(eval_config, evaluates):
    trainer = object.__new__(rl_module.PlatoonTinkerRLTrainer)
    state = _shared_state(eval_config, CheckpointConfig())

    trainer.eval_dataset = object()
    assert trainer._evaluates_during_training(state) is evaluates
    trainer.eval_dataset = None  # no eval split: nothing to evaluate on
    assert trainer._evaluates_during_training(state) is False


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


class _EvalSplit:
    """An eval dataloader of ``size`` one-task batches."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def get_batch(self, index: int) -> list[dict]:
        return [{"task_id": f"eval:{index}"}]


def _train_and_evaluate(
    monkeypatch,
    *,
    steps: int,
    eval_every: int,
    start: int = 0,
    paced: bool = False,
    at_start: bool = False,
    completed: Collection[int] = (),
    record: Path | None = None,
):
    """Run the training loop from step ``start`` to ``steps`` with zero-advantage steps beside the evaluation loop, as
    ``train`` runs them, with fake rollouts; return the training steps current when tasks were evaluated and what was
    logged, as ``(step, stats)``.  ``paced`` holds each step's rollouts back until the evaluation loop has handled the
    step before, so that no evaluation due is overtaken (a real step takes longer than its evaluation's start).
    ``at_start`` is ``eval.at_start``; ``completed`` the steps whose evaluation an earlier start of the run completed,
    as its ``evaluations.jsonl`` lists them, which ``train`` reads when it resumes; ``record`` the file each completed
    evaluation is appended to."""

    trainer = object.__new__(rl_module.PlatoonTinkerRLTrainer)
    trainer.config = SimpleNamespace(
        train=SimpleNamespace(
            batch_size=1,
            num_minibatches=1,
            num_microbatches=1,
            max_concurrent_rollouts=1,
            workflow_config=SimpleNamespace(group_size=1, filter_zero_advantage_datums=False),
            max_staleness=None,
            loss_fn="cispo",
            loss_fn_config={},
            optimizer=SimpleNamespace(learning_rate=1e-6),
        ),
        eval=EvalConfig(every=eval_every, at_start=at_start, max_concurrent_rollouts=1),
        checkpoint=CheckpointConfig(every=0),
    )
    trainer.eval_dataset = object()
    trainer.batch_transforms = []
    trainer._train_dataloader_loop = AsyncMock()
    monkeypatch.setattr(rl_module, "client_monitor", lambda scope: Mock())
    monkeypatch.setattr(rl_module, "client_step_metrics", lambda scope: {})
    monkeypatch.setattr(rl_module, "_ptc_cell_guard_stats", lambda scope: {})
    evaluated: list[int] = []
    logged: list[tuple[int, dict]] = []

    async def worker(shared_state, workflow, queue, results, tracker=None):
        if tracker is None:  # a training worker: every step's group is ready at once, with no training signal
            for _ in range(steps - start):
                while paced and shared_state.eval_handled_step != shared_state.train_step:
                    await asyncio.sleep(0)
                results.put_nowait([_datum()])
            return
        while True:  # an evaluation worker
            await queue.get()
            evaluated.append(shared_state.train_step)
            results.put_nowait(None)

    trainer._rollout_workflow_worker_loop = worker

    async def run():
        state = _shared_state(trainer.config.eval, trainer.config.checkpoint)
        state.train_step = state.start_step = start
        state.start_step_evaluated = start in completed
        state.evaluations_path = None if record is None else str(record)
        state.config = trainer.config
        state.eval_dataloader = _EvalSplit(2)
        state.train_tracker = StatsTracker("train")
        state.model_info = SimpleNamespace(llm=SimpleNamespace(version=0))
        state.stats_logger = SimpleNamespace(log=lambda *, step, stats, force=False: logged.append((step, stats)))
        state.sampling_client_updated_event.set()  # as at launch
        state.eval_loop_running = trainer._evaluates_during_training(state)
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(trainer._train_loop(group, state, start, steps, object()))
                group.create_task(trainer._eval_loop(state, object()))
        except* rl_module.TerminateTrainLoop:
            pass
        return state

    state = asyncio.run(run())
    assert state.train_step == steps
    return evaluated, logged


def _evaluations(logged: list[tuple[int, dict]]) -> list[tuple[int, int]]:
    """``(eval/step, eval/num_tasks)`` of each evaluation logged."""

    return [(stats["eval/step"], stats["eval/num_tasks"]) for _, stats in logged if "eval/num_tasks" in stats]


def test_an_evaluation_due_at_the_last_step_runs_before_the_loops_stop(monkeypatch):
    """Training runs ahead of the evaluation of step 0 here (``eval.at_start``); the one due at the last step still
    runs before the loops stop, on the policy after that step."""

    evaluated, logged = _train_and_evaluate(monkeypatch, steps=2, eval_every=2, at_start=True)

    assert _evaluations(logged) == [(0, 2), (2, 2)]
    assert evaluated[-2:] == [2, 2]  # both eval tasks of the last evaluation, after the last step
    assert logged[-1][0] == 2  # logged at the step training reached


@pytest.mark.parametrize(("at_start", "steps"), [(False, [2, 4]), (True, [0, 2, 4])])
def test_a_fresh_run_evaluates_every_due_step_and_its_starting_policy_only_with_at_start(monkeypatch, at_start, steps):
    evaluated, logged = _train_and_evaluate(monkeypatch, steps=4, eval_every=2, paced=True, at_start=at_start)

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
def test_a_resumed_run_evaluates_at_the_multiples_from_where_it_resumed(monkeypatch, at_start, start, completed, steps):
    """A start that resumes at step 3 evaluates at 4 only.  One that resumes at step 2 evaluates at 2 again only when
    the run's evaluations.jsonl lists no completed evaluation of it, and at 4.  None evaluates step 0, with
    ``eval.at_start`` or without."""

    _, logged = _train_and_evaluate(
        monkeypatch, steps=4, eval_every=2, start=start, paced=True, at_start=at_start, completed=completed
    )

    assert [step for step, _ in _evaluations(logged)] == steps


@pytest.mark.parametrize(("at_start", "completed", "steps"), [(True, (), [0, 2]), (True, (0,), [2]), (False, (), [2])])
def test_a_start_that_resumes_at_step_0_evaluates_the_starting_policy_unless_that_completed(
    monkeypatch, at_start, completed, steps
):
    """A crash before the first step's checkpoint resumes the run at step 0, from the checkpoint its start saved: with
    ``eval.at_start`` that start evaluates the starting policy again unless an earlier start completed it."""

    _, logged = _train_and_evaluate(
        monkeypatch, steps=2, eval_every=2, paced=True, at_start=at_start, completed=completed
    )

    assert [step for step, _ in _evaluations(logged)] == steps


@pytest.mark.parametrize(("completed", "steps"), [((), [4]), ((4,), [])])
def test_a_start_that_resumes_at_the_last_step_runs_its_evaluation_only_if_a_crash_cut_it_short(
    monkeypatch, completed, steps
):
    """A crash during the evaluation due at the last step, after that step's checkpoint, resumes the run there with no
    step left to train: the start runs that evaluation before it stops, unless an earlier start completed it."""

    _, logged = _train_and_evaluate(monkeypatch, steps=4, eval_every=2, start=4, completed=completed)

    assert [step for step, _ in _evaluations(logged)] == steps


@pytest.mark.parametrize(("at_start", "evaluations"), [(False, []), (True, [(0, 2)])])
def test_a_last_step_with_no_evaluation_due_stops_at_once(monkeypatch, at_start, evaluations):
    evaluated, logged = _train_and_evaluate(monkeypatch, steps=3, eval_every=5, at_start=at_start)

    assert _evaluations(logged) == evaluations  # the starting policy's, with eval.at_start, and no other
    assert len(evaluated) == 2 * len(evaluations)


def test_each_evaluation_that_completes_is_recorded_for_later_starts(monkeypatch, tmp_path):
    record = tmp_path / "evaluations.jsonl"

    _, logged = _train_and_evaluate(monkeypatch, steps=4, eval_every=2, paced=True, at_start=True, record=record)

    assert completed_evaluation_steps(tmp_path) == {0, 2, 4}
    lines = read_jsonl(record)
    assert [(line["step"], line["tasks"]) for line in lines] == [(0, 2), (2, 2), (4, 2)]
    # Each line says the step training had reached when the evaluation was logged, the step its metrics went to.
    assert [line["logged_at"] for line in lines] == [step for step, stats in logged if "eval/step" in stats]


def test_a_shutdown_request_ends_the_wait_for_an_evaluation():
    """A first SIGINT or SIGTERM during the last evaluation stops the run as before, instead of waiting for an
    evaluation whose workers no longer take tasks."""

    trainer = object.__new__(rl_module.PlatoonTinkerRLTrainer)
    state = _shared_state(EvalConfig(every=1), CheckpointConfig())
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
    monkeypatch, *, at_start: bool, completed: Collection[int] = ()
) -> list[tuple[int | None, list[int]]]:
    """Train from step 0 beside a slow evaluation until the first optimizer step, where the test stops; return the step
    the evaluation loop had handled by then and the training steps current when tasks were evaluated."""

    trainer = object.__new__(rl_module.PlatoonTinkerRLTrainer)
    trainer.config = SimpleNamespace(
        train=SimpleNamespace(
            batch_size=1,
            num_minibatches=1,
            num_microbatches=1,
            max_concurrent_rollouts=1,
            workflow_config=SimpleNamespace(group_size=1, filter_zero_advantage_datums=False),
            max_staleness=None,
            loss_fn="cispo",
            loss_fn_config={},
            optimizer=SimpleNamespace(learning_rate=1e-6),
        ),
        eval=EvalConfig(every=2, at_start=at_start, max_concurrent_rollouts=1),
        checkpoint=CheckpointConfig(every=0),
    )
    trainer.eval_dataset = object()
    trainer.batch_transforms = []
    trainer._train_dataloader_loop = AsyncMock()
    monkeypatch.setattr(rl_module, "client_monitor", lambda scope: Mock())
    monkeypatch.setattr(rl_module, "client_step_metrics", lambda scope: {})
    monkeypatch.setattr(rl_module, "_ptc_cell_guard_stats", lambda scope: {})
    evaluated: list[int] = []
    first_update: list[tuple[int | None, list[int]]] = []

    async def worker(shared_state, workflow, queue, results, tracker=None):
        if tracker is None:  # a training worker: the first step's group is ready at once, with a training signal
            results.put_nowait([_datum(advantage=1.0)])
            return
        while True:  # an evaluation worker, slower than training's first step
            await queue.get()
            await asyncio.sleep(0.02)
            evaluated.append(shared_state.train_step)
            results.put_nowait(None)

    trainer._rollout_workflow_worker_loop = worker

    async def run():
        state = _shared_state(trainer.config.eval, trainer.config.checkpoint)
        state.start_step_evaluated = 0 in completed
        state.config = trainer.config
        state.eval_dataloader = _EvalSplit(2)
        state.train_tracker = StatsTracker("train")
        state.model_info = SimpleNamespace(llm=SimpleNamespace(version=0))
        state.stats_logger = SimpleNamespace(log=lambda *, step, stats, force=False: None)

        async def optim_step_async(params):
            first_update.append((state.eval_handled_step, list(evaluated)))
            raise _FirstUpdate()

        state.training_client = SimpleNamespace(
            forward_backward_async=AsyncMock(return_value=Mock()), optim_step_async=optim_step_async
        )
        state.sampling_client_updated_event.set()  # as at launch
        state.eval_loop_running = trainer._evaluates_during_training(state)
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(trainer._train_loop(group, state, 0, 4, object()))
                group.create_task(trainer._eval_loop(state, object()))
        except* _FirstUpdate:
            pass

    asyncio.run(run())
    return first_update


def test_the_first_update_waits_for_the_evaluation_of_the_starting_policy(monkeypatch):
    """An evaluation's rollouts sample the policy that is current when they run, so with ``eval.at_start`` a fresh run
    applies its first update only once the evaluation of its starting policy has ended: every task of it ran on step
    0's policy, although the first step's rollouts were ready at once.  A start that resumes at step 0, when no earlier
    start completed that evaluation, waits for it as well."""

    first_update = _first_update(monkeypatch, at_start=True)

    assert first_update == [(0, [0, 0])]  # step 0's evaluation was over, both its tasks on the starting policy


@pytest.mark.parametrize(("at_start", "completed"), [(False, ()), (True, (0,))])
def test_the_first_update_waits_for_no_evaluation_of_the_starting_policy_that_does_not_run(
    monkeypatch, at_start, completed
):
    """Without ``eval.at_start`` step 0 is not evaluated, and a start that resumes at step 0 after an earlier start
    completed the evaluation of the starting policy does not run it again: the first update waits for neither."""

    [(_handled, evaluated)] = _first_update(monkeypatch, at_start=at_start, completed=completed)

    assert evaluated == []  # no evaluation task had run, nor was one due


def test_a_shutdown_request_ends_the_wait_for_the_evaluation_of_the_starting_policy():
    """A first SIGINT or SIGTERM while the first update waits drops the batch, which applied no update, as a signal
    during its rollouts does; without an evaluation loop, or without ``eval.at_start``, there is nothing to wait
    for."""

    trainer = object.__new__(rl_module.PlatoonTinkerRLTrainer)
    state = _shared_state(EvalConfig(every=1, at_start=True), CheckpointConfig())

    async def run():
        await asyncio.wait_for(trainer._wait_for_starting_policy_evaluation(state), timeout=5)  # no evaluation loop
        state.eval_loop_running = True  # one that has not handled step 0 yet
        state.config = SimpleNamespace(eval=EvalConfig(every=1), checkpoint=CheckpointConfig())  # without at_start
        await asyncio.wait_for(trainer._wait_for_starting_policy_evaluation(state), timeout=5)  # nothing to wait for
        state.config = SimpleNamespace(eval=EvalConfig(every=1, at_start=True), checkpoint=CheckpointConfig())
        waiting = asyncio.create_task(trainer._wait_for_starting_policy_evaluation(state))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        state.shutdown_event.set()
        with pytest.raises(rl_module.TrainInterrupted):
            await asyncio.wait_for(waiting, timeout=5)

    asyncio.run(run())
