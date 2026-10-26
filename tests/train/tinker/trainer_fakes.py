"""Drive ``PlatoonTinkerRLTrainer.train`` end to end on CPU: Tinker, checkpoints and rollouts are faked.

``make_trainer`` builds a trainer over ``task_ids_to_dataset`` tasks without running ``__init__`` (which would
register a Tinker sampler) and records the checkpoints ``train()`` saves and the steps it logs.  Every rollout
group is one datum with a nonzero advantage, so each step applies an optimizer update.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import tinker
import torch
import yaml

from platoon.train.components import task_ids_to_dataset
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.trainer import rl as rl_module
from platoon.utils.config import resolve_config
from platoon.utils.stats_tracker import get as get_tracker


def trainer_config(tmp_path: Path, *overrides: str) -> PlatoonTinkerRLTrainerConfig:
    """Three steps of batch 2 with evaluation off and no watchdog; ``overrides`` are ``key=value`` strings."""

    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {
                    "model_name": "m",
                    "renderer_name": "r",
                    "batch_size": 2,
                    "max_training_steps": 3,
                    "max_concurrent_rollouts": 16,  # two groups of the default 8
                },
                "eval": {"every": 0},
                "watchdog": {"enabled": False},
                "log_path": str(tmp_path / "logs"),
            }
        )
    )
    return resolve_config(path, overrides, PlatoonTinkerRLTrainerConfig).config


def datum(advantage: float = 1.0) -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            name: tinker.TensorData.from_torch(torch.tensor(values))
            for name, values in {
                "target_tokens": [2, 3, 4],
                "logprobs": [0.0, -0.1, -0.2],
                "advantages": [0.0, advantage, advantage],
                "mask": [0.0, 1.0, 1.0],
            }.items()
        },
    )


class _Future:
    def __init__(self, value):
        self.value = value

    async def result_async(self):
        return self.value


class TrainingClient:
    """Counts optimizer steps; ``on_optim_step(count)`` runs as each one is submitted."""

    def __init__(self, on_optim_step=None):
        self.optim_steps = 0
        self.on_optim_step = on_optim_step

    def create_sampling_client(self, path):
        return f"sampler:{path}"

    async def forward_backward_async(self, datums, loss_fn, loss_fn_config):
        outputs = [{"logprobs": item.loss_fn_inputs["target_tokens"]} for item in datums]
        return _Future(SimpleNamespace(loss_fn_outputs=outputs))

    async def optim_step_async(self, params):
        self.optim_steps += 1
        if self.on_optim_step is not None:
            self.on_optim_step(self.optim_steps)
        return _Future(SimpleNamespace(metrics={}))

    async def save_weights_and_get_sampling_client_async(self):
        return "weights-only-sampler"


class Workflow:
    """Each rollout group takes ``delay`` seconds and returns one datum (none with ``learn=False``);
    ``on_start(count)`` runs as each one starts."""

    def __init__(self, delay: float = 0.05, learn: bool = True, on_start=None):
        self.delay, self.learn, self.on_start = delay, learn, on_start
        self.started = 0

    async def arun_episode(self, data):
        self.started += 1
        if self.on_start is not None:
            self.on_start(self.started)
        await asyncio.sleep(self.delay)
        return [datum()] if self.learn else []


class _LLM:
    version = 0

    def set_version(self, version):
        self.version = version

    def update_sampling_client(self, client, increment_version=True):
        if increment_version:
            self.version += 1


def make_trainer(
    monkeypatch,
    config: PlatoonTinkerRLTrainerConfig,
    *,
    tasks: int = 8,
    eval_tasks: int = 0,
    training_client: TrainingClient | None = None,
):
    """``(trainer, saves, logged)``: the names of the checkpoints saved and the steps logged, in order."""

    saves: list[str] = []
    logged: list[int] = []

    async def save_checkpoint_async(*, training_client, name, log_path, loop_state, kind, **kwargs):
        saves.append(name)
        return {"state_path": f"tinker://m/weights/{name}", "sampler_path": f"tinker://m/sampler_weights/{name}"}

    monkeypatch.setattr(rl_module.checkpoint_utils, "save_checkpoint_async", save_checkpoint_async)
    monkeypatch.setattr(rl_module.tinker, "ServiceClient", lambda **kwargs: SimpleNamespace())
    trainer = object.__new__(rl_module.PlatoonTinkerRLTrainer)
    trainer.config = config
    trainer.train_dataset = task_ids_to_dataset([f"train:{index}" for index in range(tasks)])
    trainer.eval_dataset = task_ids_to_dataset([f"val:{index}" for index in range(eval_tasks)]) if eval_tasks else None
    trainer.batch_transforms = []
    trainer._model_info = SimpleNamespace(llm=_LLM())
    trainer._stats_logger = SimpleNamespace(wandb_run_id=None, log=lambda step, stats, force=False: logged.append(step))
    trainer._train_tracker = get_tracker("train")

    async def create_training_client(service_client, resume_info):
        return training_client or TrainingClient()

    trainer._create_training_client = create_training_client
    return trainer, saves, logged
