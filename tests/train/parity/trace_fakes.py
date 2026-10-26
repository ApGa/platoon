"""Fakes of what lies outside the training code, for the golden trace: the Tinker service, the policy, the rollouts at
the worker-request boundary and an environment's tasks.  :class:`FakeRun` records what the trainer asks of them, and
what they answer is drawn from hashes of the question, in numbers whose sums are exact, so that a run is a function
of the code.  They are the trace's own: a change to a fake changes every golden."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import math
import re
import signal
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePath
from types import SimpleNamespace
from typing import Any, cast

import tinker
import torch
from trace_format import Masks, OneLine, to_data

from platoon.task import Task
from platoon.train.tinker import proxy
from platoon.train.workers.protocol import PROTOCOL_VERSION, RolloutWorkerResponse
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats

# Virtual seconds each call of the fake service takes.  A real call is a round trip, during which the dispatcher, the
# rollouts and the evaluation run, so the order of what the trainer does around each await shows in the trace.
CALL_SECONDS = 1 / 64
# Virtual seconds a forward_backward's result takes after its submission, and per microbatch it waited for, an
# optimizer step takes to be applied: rollouts go on meanwhile, as beside a real service.
FORWARD_BACKWARD_SECONDS = 0.375
# The fake rollouts' pass rates, by the number that ends the task id (t3 passes at PASS_RATES[3]): mixed groups, and
# groups that all pass or all fail, whose advantages are zero.
PASS_RATES = (0.5, 1.0, 0.5, 0.25, 0.5, 0.0, 0.75, 0.5)
# What the fake service adds to the training log-probability of each token whose target id is a multiple of 4.  The
# importance weight of such an action token, e ** 2, is above cispo's default clip_high_threshold (5), so the clip
# fractions show the sign of every log ratio; the sums stay exact (log-probabilities are multiples of 1/16).
LOGPROB_SHIFT = 2.0


@dataclass(frozen=True)
class Rollouts:
    """How a scenario's fake rollouts end.

    ``invalid_one_in``: one rollout in that many (by hash) ends with an ``error`` outcome and no datum.
    ``subagents``: about half the rollouts add a subagent trajectory at depth 1.
    ``stragglers``: the last member (rollout 1) of about half the groups of an odd task takes 1000 s, beyond any grace.
    ``slowdown``: multiplies the seconds of the tasks whose id contains a key.
    ``follow``: a rollout's datums carry the policy version current at its first sample, as a client that follows the
    published weights stamps them, so a group can carry several; ``first_share`` bounds the share of a rollout's
    seconds before that sample.  Under a start limit (``train.max_starting_rollouts``) a rollout makes its first model
    call at that sample, which gives its start slot back.
    """

    invalid_one_in: int | None = 8
    subagents: bool = False
    stragglers: bool = False
    slowdown: Mapping[str, float] = field(default_factory=dict)
    follow: bool = False
    first_share: float = 0.25


@dataclass(frozen=True)
class Start:
    """One start of a scenario's run, and how it ends when it does not finish its steps.

    ``crash_after_saves``: the service fails the first forward_backward after that many full saves (the launch save
    first), as a crash would; the next start resumes from the last checkpoint.
    ``stop_at``: ``(call, n)``: at its n-th call of that name (``forward_backward``, ``optim_step``) the service raises
    SIGUSR2, the stop request ``platoon train`` passes on; the start stops at a step boundary and saves
    ``<step>_interrupted``, which the next start resumes from.
    """

    crash_after_saves: int | None = None
    stop_at: tuple[str, int] | None = None


class ServiceGone(RuntimeError):
    """The fake service's failure that ends a start which :class:`Start` crashes."""


class _Future:
    """What a Tinker call returns: its result, after ``seconds`` of virtual time."""

    def __init__(self, value: Any, seconds: float = 0.0) -> None:
        self.value, self.seconds = value, seconds

    async def result_async(self) -> Any:
        if self.seconds:
            await asyncio.sleep(self.seconds)
        return self.value


class FakeServiceClient:
    """``tinker.ServiceClient``: training clients for a new run or from a saved state."""

    def __init__(self, run: FakeRun, **kwargs: Any) -> None:
        self.run = run

    async def create_lora_training_client_async(self, base_model: str, rank: int = 32, **kwargs: Any):
        return await self._open("create_lora_training_client", base_model, rank)

    async def create_training_client_from_state_with_optimizer_async(self, path: str, **kwargs: Any):
        return await self._open("create_training_client_from_state_with_optimizer", path)

    async def create_training_client_from_state_async(self, path: str, **kwargs: Any):
        return await self._open("create_training_client_from_state", path)

    async def _open(self, call: str, *arguments: Any) -> FakeTrainingClient:
        self.run.call(call, *arguments)
        await asyncio.sleep(CALL_SECONDS)
        return self.run.training_client()


class FakeTrainingClient:
    """A Tinker training client that records every call.  Its training log-probabilities are the sampling ones with
    :data:`LOGPROB_SHIFT`, each optimizer step reports the datums it applied as its ``grad_norm``, and an update counts
    as applied (:attr:`FakeRun.updates_applied`) once the service has worked through the microbatches before it."""

    def __init__(self, run: FakeRun, name: str) -> None:
        self.run, self.name = run, name
        self.pending_datums = 0
        self.pending_microbatches = 0
        self.weights = 0

    async def forward_backward_async(self, datums: list[tinker.Datum], loss_fn: str, loss_fn_config: dict):
        run = self.run
        if run.start.crash_after_saves is not None and run.full_saves >= run.start.crash_after_saves:
            raise ServiceGone("the training service went away")
        run.stop_if_due("forward_backward")
        run.call("forward_backward", forward_backward_record(datums, loss_fn, loss_fn_config))
        self.pending_datums += len(datums)
        self.pending_microbatches += 1
        outputs = [{"logprobs": training_logprobs(datum)} for datum in datums]
        await asyncio.sleep(CALL_SECONDS)
        return _Future(SimpleNamespace(loss_fn_outputs=outputs, metrics={}), FORWARD_BACKWARD_SECONDS)

    async def optim_step_async(self, params: tinker.AdamParams):
        run = self.run
        run.stop_if_due("optim_step")
        run.call("optim_step", to_data(params))
        metrics, self.pending_datums = {"grad_norm": float(self.pending_datums)}, 0
        microbatches, self.pending_microbatches = max(1, self.pending_microbatches), 0
        asyncio.get_running_loop().call_later(FORWARD_BACKWARD_SECONDS * microbatches, run.update_applied)
        await asyncio.sleep(CALL_SECONDS)
        return _Future(SimpleNamespace(metrics=metrics))

    async def save_state_async(self, name: str, ttl_seconds: int | None = None):
        self.run.full_saves += 1
        return await self._save("save_state", name, f"tinker://{self.name}/weights/{name}")

    async def save_weights_for_sampler_async(self, name: str, ttl_seconds: int | None = None):
        return await self._save("save_weights_for_sampler", name, f"tinker://{self.name}/sampler_weights/{name}")

    async def _save(self, call: str, name: str, path: str) -> _Future:
        self.run.call(call, name)
        await asyncio.sleep(CALL_SECONDS)
        return _Future(SimpleNamespace(path=path))

    async def save_weights_and_get_sampling_client_async(self):
        self.weights += 1
        self.run.call("save_weights_and_get_sampling_client")
        await asyncio.sleep(CALL_SECONDS)
        return f"sampler:{self.name}/weights-only-{self.weights}"

    def create_sampling_client(self, path: str):
        self.run.call("create_sampling_client", path)
        return f"sampler:{path}"


def forward_backward_record(datums: Sequence[tinker.Datum], loss_fn: str, loss_fn_config: Any) -> dict[str, Any]:
    """A forward_backward as a trace holds it: its loss, the type of each loss input, and every datum as sent."""

    kinds: dict[str, set[str]] = {}
    for datum in datums:
        for key, value in datum.loss_fn_inputs.items():
            kinds.setdefault(key, set()).add(_kind(value))
    return {
        "loss_fn": loss_fn,
        "loss_fn_config": to_data(loss_fn_config),
        "inputs": OneLine(
            {key: found.pop() if len(found) == 1 else sorted(found) for key, found in sorted(kinds.items())}
        ),
        "datums": [_datum_record(datum) for datum in datums],
    }


def _kind(value: tinker.TensorData) -> str:
    """A loss input's dtype, with its shape when it is not a vector."""

    shape = value.shape
    return str(value.dtype) if shape is None or len(shape) == 1 else f"{value.dtype}{list(shape)}"


def _datum_record(datum: tinker.Datum) -> OneLine:
    """A datum as sent: its tokens, and the numbers of every loss input as they go over the wire."""

    inputs = {key: to_data(value.data) for key, value in sorted(datum.loss_fn_inputs.items())}
    return OneLine({"tokens": datum.model_input.to_ints(), **inputs})


def training_logprobs(datum: tinker.Datum) -> tinker.TensorData:
    """The fake service's training log-probabilities of ``datum``: its sampling ones, plus :data:`LOGPROB_SHIFT` where
    the target token's id is a multiple of 4."""

    sampling = datum.loss_fn_inputs["logprobs"].to_torch()
    shifted = datum.loss_fn_inputs["target_tokens"].to_torch() % 4 == 0
    return tinker.TensorData.from_torch(sampling + LOGPROB_SHIFT * shifted.to(sampling.dtype))


class FakePolicy:
    """The policy ``register_tinker_llm`` returns (``ModelInfo.llm``): the version the trainer sets and bumps with
    each sampling client it publishes, as ``TinkerLLM`` keeps it, and the sampling settings a rollout's model binding
    carries."""

    def __init__(self, run: FakeRun, model_name: str, context_window_length: int | None, timeout: float | None):
        self.run = run
        self.model_name = model_name
        self.context_window_length = context_window_length
        self.sample_timeout_seconds = 600.0 if timeout is None else float(timeout)
        self.max_tokens, self.temperature, self.top_k, self.top_p, self.seed = 4096, 1.0, -1, 1.0, None
        self.sampling_client: Any = None
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def set_version(self, version: int) -> None:
        self._version = version
        self.run.call("set_version", version)

    def update_sampling_client(self, sampling_client: Any, increment_version: bool = True) -> None:
        self.sampling_client = sampling_client
        if increment_version:
            self._version += 1
        self.run.call("publish", self._version, sampling_client)


# --------------------------------------------------------------------------- the rollouts


def _hash_draws(*parts: Any) -> Callable[[int, int], int]:
    """``draw(offset, bits)``: ``bits`` bits of a SHA-256 of ``parts``, from byte ``offset`` on."""

    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()

    def draw(offset: int, bits: int) -> int:
        assert 0 <= offset <= len(digest) - 8 and 0 < bits <= 64
        return int.from_bytes(digest[offset : offset + 8], "big") >> (64 - bits)

    return draw


def _tensor(values: Sequence[Any], dtype: torch.dtype | None = None) -> tinker.TensorData:
    return tinker.TensorData.from_torch(torch.tensor(list(values), dtype=dtype))


@dataclass(frozen=True)
class _Plan:
    """How one rollout ends: its reward, whether it is valid, its virtual seconds, the share of them before its first
    sample, its trajectories (name, reward, datums) and each trajectory's steps."""

    reward: float
    valid: bool
    seconds: float
    first_sample: float
    trajectories: tuple[tuple[str, float, int], ...]
    steps: int


def _plan(rollouts: Rollouts, task_id: str, number: int, version: int) -> _Plan:
    """The end of rollout ``number`` of task ``task_id`` on policy ``version``, drawn from a hash of the three: it
    passes at its task's rate in :data:`PASS_RATES`."""

    draw = _hash_draws(task_id, number, version)
    found = re.search(r"(\d+)$", task_id)
    index = int(found.group(1)) if found else 0
    reward = 1.0 if draw(0, 16) / 2**16 < PASS_RATES[index % len(PASS_RATES)] else 0.0
    valid = not (rollouts.invalid_one_in and draw(2, 16) % rollouts.invalid_one_in == 0)
    seconds = 1.0 + 4.0 * draw(4, 32) / 2**32
    seconds *= math.prod(factor for key, factor in rollouts.slowdown.items() if key in task_id)
    if rollouts.stragglers and index % 2 == 1 and number == 1 and draw(8, 1):
        seconds = 1000.0
    trajectories = [("root", reward, 1)]
    if rollouts.subagents and draw(9, 1):
        trajectories.append(("subagent", float(draw(10, 1)), 1 + draw(11, 1)))
    first = rollouts.first_share * _hash_draws("first-sample", task_id, number, version)(0, 8) / 256
    return _Plan(reward, valid, seconds, first, tuple(trajectories), 1 << draw(24, 8) % 3)  # steps 1, 2 or 4


def _rollout_datum(request: Any, version: int, name: str, index: int, reward: float, depth: int | None) -> tinker.Datum:
    """One datum as the converter makes it: an observation token, then action tokens loaded with ``reward``."""

    draw = _hash_draws(request.task_id, request.rollout_number, version, name, index)
    length = 3 + draw(0, 8) % 3
    tokens = [1 + draw(1 + position, 10) % 1000 for position in range(length)]
    actions = length - 2
    inputs = {
        "target_tokens": _tensor(tokens[1:]),
        "logprobs": _tensor([-draw(12 + position, 4) / 16 for position in range(length - 1)], torch.float32),
        "advantages": _tensor([0.0] + [reward] * actions, torch.float32),
        "mask": _tensor([0.0] + [1.0] * actions, torch.float32),
        "checkpoint_version": _tensor([version]),
    }
    if depth is not None:
        inputs["traj_depth"] = _tensor([depth], torch.long)
        inputs["traj_start"] = _tensor([1.0 if index == 0 else 0.0], torch.float32)
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(tokens[:-1]), loss_fn_inputs=inputs)


def _response(request: Any, plan: _Plan, version: int) -> RolloutWorkerResponse:
    """The worker's response to ``request``, as :func:`_plan` ends it."""

    with_depth = request.conversion.include_traj_depth
    datums: list[tinker.Datum] = []
    stats = []
    for depth, (name, reward, count) in enumerate(plan.trajectories):
        made = [
            _rollout_datum(request, version, name, index, reward, depth if with_depth else None)
            for index in range(count)
        ]
        datums += made
        stats.append(
            TrajectoryStats(
                trajectory_id=name,
                reward=reward,
                num_steps=plan.steps,
                num_input_tokens=sum(datum.model_input.length for datum in made),
                num_output_tokens=sum(int(datum.loss_fn_inputs["mask"].to_torch().sum()) for datum in made),
                num_datums=count,
                is_root=depth == 0,
            )
        )
    steps = sum(item.num_steps for item in stats)
    valid = plan.valid
    workload = RolloutWorkload(
        environment_steps=steps,
        model_calls=steps,
        input_tokens=sum(item.num_input_tokens for item in stats),
        output_tokens=sum(item.num_output_tokens for item in stats),
        trajectories=len(stats),
        postmerge_datums=len(datums),
        policy_eligible_datums=len(datums) if valid else 0,
        post_sampling_datums=len(datums) if valid else 0,
    )
    result = TrajectoryCollectionResult(
        datums=datums if valid else [],
        task_reward=plan.reward,
        trajectory_stats=stats,
        root_rewards_dict={},
        task_reward_valid=valid,
    )
    return RolloutWorkerResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id=request.request_id,
        status="completed",
        result=result,
        workload=workload,
        observed=True,
        rollout_seconds=plan.seconds,
        model_calls=steps,
        outcome_status="scored" if valid else "error",
        reward_key="reward" if valid else None,
        success_reward=plan.reward if valid else None,
    )


# --------------------------------------------------------------------------- the run


class FakeRun:
    """The fakes of one run of a scenario, and what they record of the start that runs (:meth:`begin`): its Tinker
    calls, its rollout requests, its logged metrics and its log messages, and under a start limit its rollouts'
    start-ups."""

    def __init__(self, rollouts: Rollouts, masks: Masks) -> None:
        self.rollouts = rollouts
        self.masks = masks
        self.settings: list[str] = []  # each distinct request's settings, across starts
        self.clients = 0
        self.begin(Start())

    def begin(self, start: Start) -> None:
        """Record a new start, which ends as ``start`` says."""

        self.start = start
        self.calls: list[list[Any]] = []
        self.requests: dict[str, list[list[Any]]] = {"train": [], "eval": []}
        self.logged: list[tuple[int, bool, dict[str, Any]]] = []
        self.messages: list[str] = []
        self.counts: Counter[str] = Counter()
        self.full_saves = 0
        self.updates_applied = 0
        self.stopped = False
        # Under a start limit: the rollouts between their request and their first model call now, and the first calls
        # made and the most rollouts starting at once (``start_ups``, empty without a limit).
        self.starting = 0
        self.start_ups: dict[str, int] = {}

    def call(self, name: str, *arguments: Any) -> None:
        self.calls.append([name, *(self.masks.strings(argument) for argument in arguments)])

    def stop_if_due(self, call: str) -> None:
        """Raise SIGUSR2, the stop request ``platoon train`` passes on, at the call the start stops at."""

        self.counts[call] += 1
        if self.start.stop_at == (call, self.counts[call]):
            assert callable(signal.getsignal(signal.SIGUSR2)), "the trainer handles no SIGUSR2: nothing would stop"
            self.stopped = True
            signal.raise_signal(signal.SIGUSR2)

    def update_applied(self) -> None:
        self.updates_applied += 1

    def service_client(self, **kwargs: Any) -> FakeServiceClient:
        return FakeServiceClient(self, **kwargs)

    def training_client(self) -> FakeTrainingClient:
        self.clients += 1
        return FakeTrainingClient(self, f"run-{self.clients}")

    def register_policy(
        self,
        model_name: str,
        renderer_name: str,
        context_window_length: int | None = None,
        renderer_kwargs: dict[str, Any] | None = None,
        *,
        model_path: str | None = None,
        base_url: str | None = None,
        sample_timeout_seconds: float | None = None,
    ) -> proxy.ModelInfo:
        """``register_tinker_llm``: a :class:`FakePolicy`, which needs no tokenizer or renderer."""

        policy = FakePolicy(self, model_name, context_window_length, sample_timeout_seconds)
        return proxy.ModelInfo(
            llm=cast(Any, policy),
            model_name=model_name,
            renderer_name=renderer_name,
            renderer_kwargs=dict(renderer_kwargs or {}),
        )

    def log_stats(self, step: int, stats: Mapping[str, Any], force: bool) -> None:
        self.logged.append((step, force, dict(stats)))

    def binding(self, workflow: Any, sampling_client: Any, version: int) -> None:
        """A publication of the rollout workers' binding (``publish_model_binding``), a no-op in process."""

        self.call("publish_binding", workflow.stats_scope, version, sampling_client)

    async def rollout(
        self,
        request: Any,
        llm: Any,
        *,
        rollout_fn: Any = None,
        reward_processor: Any = None,
        on_first_call: Callable[[], None] | None = None,
    ):
        """``run_rollout_request``: record the request, with the updates the service had applied when it came, then
        answer it after the rollout's virtual seconds.  Under a start limit, which passes ``on_first_call``, the rollout
        is starting until its first model call, at its first sample, and calls ``on_first_call`` then, as the model
        gateway tells the rollout's timeline (:meth:`_start_up`)."""

        # rollout_config.train is true in evaluation too: the directory names the workflow (rollouts/<scope>/...).
        scope = PurePath(request.rollout_config["output_dir"]).parent.name
        version = int(request.model.checkpoint_version)
        row = [request.task_id, request.rollout_number, version, self._settings(request), self.updates_applied]
        self.requests[scope].append(row)
        plan = _plan(self.rollouts, request.task_id, request.rollout_number, version)
        response = _response(request, plan, version)
        if not self.rollouts.follow and on_first_call is None:
            await asyncio.sleep(plan.seconds)
            return response
        first = plan.seconds * plan.first_sample
        if on_first_call is None:
            await asyncio.sleep(first)
        else:
            await self._start_up(first)
            on_first_call()
        if self.rollouts.follow:
            row.append(int(llm.version))  # the version its datums carry
            for datum in response.result.datums if response.result is not None else []:
                datum.loss_fn_inputs["checkpoint_version"] = _tensor([row[-1]])
        await asyncio.sleep(plan.seconds - first)
        return response

    async def _start_up(self, seconds: float) -> None:
        """A rollout's start-up under a start limit: the ``seconds`` from its request to its first model call, counted
        in ``start_ups``: the most rollouts starting at once, and the first call once it comes."""

        self.starting += 1
        self.start_ups["most_at_once"] = max(self.start_ups.get("most_at_once", 0), self.starting)
        try:
            await asyncio.sleep(seconds)
        finally:
            self.starting -= 1
        self.start_ups["first_calls"] = self.start_ups.get("first_calls", 0) + 1

    def _settings(self, request: Any) -> int:
        """The index in ``request_settings`` of what ``request`` carries beside its task, number and version."""

        data = to_data(request, self.masks)
        for key in ("task_id", "rollout_number"):
            data.pop(key)
        # The request id's form, without the task, the number and the random part that vary from request to request.
        request_id = data["request_id"]
        form = re.sub(rf"^.+-r{request.rollout_number}-[0-9a-f]{{8}}$", "<task>-r<number>-<hex>", request_id)
        data["request_id"] = form if form != request_id else re.sub(r"[0-9a-f]{8}$", "<hex>", request_id)
        data["task"].pop("id")
        data["model"].pop("checkpoint_version")
        data["rollout_config"]["output_dir"] = str(PurePath(data["rollout_config"]["output_dir"]).parent)
        text = json.dumps(data, sort_keys=True)
        if text not in self.settings:
            self.settings.append(text)
        return self.settings.index(text)


# --------------------------------------------------------------------------- what the configs name


def task_rows(config: Any, split: str, *, count: int, kinds: Sequence[str] = (), rate: float | None = None) -> list:
    """An environment entry's ``dataset_loader``: ``count`` tasks ``t<i>`` (``v<i>`` outside the train split); with
    ``rate``, only the numbers whose pass rate is ``rate``; with ``kinds``, a ``kind`` column taking them in turn (a
    label column for ``train.label_by``)."""

    prefix = "t" if split == "train" else "v"
    numbers = (number for number in itertools.count() if rate is None or PASS_RATES[number % len(PASS_RATES)] == rate)
    rows = []
    for position, number in enumerate(itertools.islice(numbers, count)):
        row: dict[str, Any] = {"task_id": f"{prefix}{number}"}
        if kinds:
            row["kind"] = kinds[position % len(kinds)]
        rows.append(row)
    return rows


def unused_rollout(task: Task, rollout_config: Any) -> dict:
    """The entries' ``rollout`` and the workflows' ``rollout_fn``: a rollout of the fakes never calls it."""

    raise AssertionError("the fake rollouts answer every request: no rollout function runs")


def get_task(task_id: str) -> Task:
    """The entries' ``task_loader`` and the workflows' ``get_task_fn``: a task of that id and nothing else."""

    return Task(id=task_id)
