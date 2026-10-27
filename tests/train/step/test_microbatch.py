"""A step's microbatch as it is prepared and sent (``TrainStep.prepare``, ``platoon.train.step.microbatch``): the batch
transforms see every datum the step collected, the zero-advantage filter runs once, after them, and keeps the action
tokens of the datums it drops in the denominator of those it sends."""

from __future__ import annotations

import tinker
import torch
from fakes.config import make_config
from fakes.loop import loop_state, loop_trainer, run_train_loop
from fakes.trainer import TrainingClient, datum


def _advantages(datum: tinker.Datum) -> list[float]:
    return datum.loss_fn_inputs["advantages"].to_torch().tolist()


def test_the_zero_filter_runs_once_after_the_batch_transforms():
    """A transform turns the group's first datum's advantages to zero and its second's from zero to 1: the step sends
    the second alone, with the first's action tokens in its denominator, as one filter run after the transforms keeps
    it.  A filter before the transforms would have dropped the second (zero then) and sent the first; a second run
    after them would count only the datum it kept."""

    seen: list[list[list[float]]] = []

    def swap(datums, context):
        seen.append([_advantages(each) for each in datums])
        for each, advantage in zip(datums, (0.0, 1.0)):
            each.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(torch.tensor([0.0, advantage, advantage]))
        return datums

    client = TrainingClient()
    config = make_config(
        train={
            "batch_size": 1,
            "num_minibatches": 1,
            "num_microbatches": 1,
            "max_concurrent_rollouts": 2,
            "workflow_config": {"group_size": 2, "filter_zero_advantage_datums": True},
        }
    )
    trainer = loop_trainer(config, training_client=client)
    trainer.batch_transforms = [swap]

    (stats,) = run_train_loop(
        trainer, loop_state(trainer), [[datum(advantage=1.0, version=0), datum(advantage=0.0, version=0)]], steps=1
    )

    assert seen == [[[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]  # the transforms saw both datums, as the group gave them
    [[sent]] = client.forward_backwards
    assert _advantages(sent) == [0.0, 0.25, 0.25]  # the one kept, over the 4 action tokens of both
    assert stats["train/zero_advantage_filter/eligible_datums"] == 2
    assert stats["train/zero_advantage_filter/retained_datums"] == 1
