"""One start of a supervised training run on a fake Tinker service, for ``test_supervise.py``.

``python supervised_start_stub.py CONFIG`` stands in for ``python -m platoon.train.supervise CONFIG``: it trains with
``PlatoonTinkerRLTrainer.train`` on the config's run directory, as that does, but over eight task ids with a workflow
whose every group is one datum (as ``fakes/trainer.py`` does), through a fake Tinker service whose state is the
JSON file ``$FAKE_TINKER``:

* ``script``: what goes wrong in each start, by start number (``"1"``, ``"2"``, ...): ``{"stall_at": N}``, the start's
  N-th ``forward_backward`` never returns, so the trainer's deadline ends the start with exit 1; ``{"die_at": N}``,
  the service dies at it: the call fails with a connection error.  ``then_refuse`` lists how the service then answers
  the next start's first asks for a training client: a status (400: a service that trains one model at a time holds
  the dead start's session) or ``null`` (no connection: the service restarts).
* Each rollout starts a sandbox of every provider ``$FAKE_SANDBOX_PROVIDERS`` lists (JSON ``[[name, options],
  ...]``; an hpc-sandbox one from the image ``$FAKE_SANDBOX_IMAGE``), runs a long sleep in each ``process`` one, and
  leaves them running, as a killed rollout worker would.

The file records each start under ``starts.<n>``: what each ask for a training client got (``refused <status>`` or
``opened <the state it resumed from>``), the optimizer steps it took, and the sandboxes it found when it began.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import tinker
import torch

from platoon.sandbox import SandboxSpec, get_provider
from platoon.train.components import task_ids_to_dataset
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.policy import service as service_module
from platoon.train.policy.policy import Policy
from platoon.train.policy.set import PolicySet
from platoon.train.policy.spec import MAIN, policy_specs
from platoon.train.trainer import rl as rl_module
from platoon.utils.config import load_config

STATE = Path(os.environ["FAKE_TINKER"])
PROVIDERS = json.loads(os.environ.get("FAKE_SANDBOX_PROVIDERS", "[]"))
IMAGE = os.environ.get("FAKE_SANDBOX_IMAGE", "image.sif")


def load() -> dict:
    return json.loads(STATE.read_text())


def save(state: dict) -> None:
    partial = STATE.with_name(STATE.name + ".partial")
    partial.write_text(json.dumps(state, indent=2))
    partial.replace(STATE)


def note(start: int, key: str, value) -> None:
    state = load()
    record = state.setdefault("starts", {}).setdefault(str(start), {})
    if isinstance(value, int) and key == "steps":
        record[key] = record.get(key, 0) + value
    else:
        record.setdefault(key, []).append(value)
    save(state)


def sandboxes() -> int:
    """The sandboxes of every provider that are on disk now: process sandbox directories, and hpc-sandbox sessions
    (a directory each, or a file each in ``fake_hpc_sandbox``)."""

    found = 0
    for name, options in PROVIDERS:
        if name == "process":
            found += len(list(Path(options["root"]).glob("process-*")))
        else:
            sessions = Path(options["scratch_root"]) / "sessions"
            found += len([path for path in sessions.glob("*") if path.is_dir() or path.suffix == ".json"])
    return found


class _Future:
    def __init__(self, value=None, *, never: bool = False, on_result=None) -> None:
        self.value, self.never, self.on_result = value, never, on_result

    async def result_async(self):
        if self.never:
            await asyncio.Event().wait()
        if self.on_result is not None:
            self.on_result()
        return self.value


def _request() -> httpx.Request:
    return httpx.Request("POST", "http://tinker.invalid/api/v1/create_model")


class TrainingClient:
    def __init__(self, start: int, script: dict) -> None:
        self.start, self.script, self.calls = start, script, 0

    async def save_state_async(self, name, ttl_seconds=None):
        return _Future(SimpleNamespace(path=f"tinker://fake/weights/{name}"))

    async def save_weights_for_sampler_async(self, name, ttl_seconds=None):
        return _Future(SimpleNamespace(path=f"tinker://fake/sampler_weights/{name}"))

    def create_sampling_client(self, path):
        return f"sampler:{path}"

    async def save_weights_and_get_sampling_client_async(self):
        return "weights-only-sampler"

    async def forward_backward_async(self, datums, loss_fn, loss_fn_config):
        self.calls += 1
        if self.calls in (self.script.get("stall_at"), self.script.get("die_at")):
            state = load()
            state["refuse"] = self.script.get("then_refuse", [])
            save(state)
            if self.script.get("stall_at") == self.calls:
                return _Future(never=True)
            raise tinker.APIConnectionError(request=_request())
        outputs = [{"logprobs": item.loss_fn_inputs["target_tokens"]} for item in datums]
        return _Future(SimpleNamespace(loss_fn_outputs=outputs))

    async def optim_step_async(self, params):
        # Counted once the trainer has the result: a step whose forward_backward stalls submits one it never awaits.
        return _Future(SimpleNamespace(metrics={}), on_result=lambda: note(self.start, "steps", 1))


class ServiceClient:
    """Opens training clients as the state file says (``refuse``), for the start it runs in."""

    start = 0

    def __init__(self, **kwargs) -> None:
        pass

    async def create_lora_training_client_async(self, base_model, rank=32, **kwargs):
        return self._open(None)

    async def create_training_client_from_state_with_optimizer_async(self, path, user_metadata=None):
        return self._open(path)

    def _open(self, path):
        state = load()
        refusals = state.get("refuse") or []
        if refusals:
            status = refusals.pop(0)
            save(state)
            note(self.start, "asks", f"refused {status}")
            if status is None:
                raise tinker.APIConnectionError(request=_request())
            response = httpx.Response(status, request=_request())
            raise tinker.BadRequestError("this gateway serves one model at a time", response=response, body=None)
        note(self.start, "asks", f"opened {path}")
        return TrainingClient(self.start, state.get("script", {}).get(str(self.start), {}))


class _Policy(Policy):
    """Main, as the config declares it, whose sessions are on the fake service of this start."""

    def new_service_client(self) -> Any:
        return ServiceClient()


class _LLM:
    version = 0

    def set_version(self, version):
        self.version = version

    def update_sampling_client(self, client, increment_version=True):
        if increment_version:
            self.version += 1


class Workflow:
    """Each group starts a sandbox of every provider, leaves them running, and returns one datum."""

    async def arun_episode(self, data):
        for name, options in PROVIDERS:
            image = "local" if name == "process" else IMAGE
            sandbox = await get_provider(name, **options).start(SandboxSpec(image=image))
            if name == "process":
                await sandbox.exec("sleep 600 > /dev/null 2>&1 &")
        return [_datum()]


def _datum() -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            name: tinker.TensorData.from_torch(torch.tensor(values))
            for name, values in {
                "target_tokens": [2, 3, 4],
                "logprobs": [0.0, -0.1, -0.2],
                "advantages": [0.0, 1.0, 1.0],
                "mask": [0.0, 1.0, 1.0],
            }.items()
        },
    )


def main() -> None:
    state = load()
    state["count"] = start = state.get("count", 0) + 1
    save(state)
    note(start, "sandboxes_at_start", sandboxes())
    ServiceClient.start = start
    service_module.TRAINING_CLIENT_FIRST_WAIT_S = service_module.TRAINING_CLIENT_LONGEST_WAIT_S = 0.05
    loaded, _ = load_config(args=["--config", sys.argv[1]], config_class=PlatoonTinkerRLTrainerConfig)
    config = cast(PlatoonTinkerRLTrainerConfig, loaded)
    trainer = rl_module.PlatoonTinkerRLTrainer(
        config,
        task_ids_to_dataset([f"train:{index}" for index in range(8)]),
        policies=PolicySet([_Policy(policy_specs(config)[MAIN], cast(Any, SimpleNamespace(llm=_LLM())))]),
    )

    async def run() -> None:
        async with trainer:
            await trainer.train(train_workflow=Workflow())

    asyncio.run(run())


if __name__ == "__main__":
    main()
